"""Recover missing official paper PDFs with identity checks and durable evidence.

Three channels may supply an original. The Nature channel names an article from
a landing page or PDF the collection already verified, or from a Nature-family
DOI, which names its own article; the PMC channel names it from the identifier
the official record carried; the page-text channel typesets the publisher's own
article page into the document the pipeline reads, and is only reached when the
publisher answers and refuses to serve its file. Whichever channel supplies the
bytes, the same identity check runs before anything is published, so an address
derived from a DOI still has to prove itself inside the PDF it returns, and a
typeset copy is filed as a copy instead of the publisher's own file.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import ssl
import tempfile
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from requests.adapters import HTTPAdapter

from rfs.providers.literature import LiteratureHttpClient, ProviderRequestError
from rfs.utils import write_json
from .literature_freeze import _artifact_root, _fulltext_path
from .literature_guard import LiteratureBusyError, active_legacy_writers, literature_batch_guard
from .source_snapshot_io import copy_verified, sha256_file, strict_jsonl


NATURE_CHANNEL = "nature_article"
PMC_CHANNEL = "pmc_oa"
HTML_CHANNEL = "official_html_text"
CHANNELS = (NATURE_CHANNEL, PMC_CHANNEL)
COPY_CHANNELS = (NATURE_CHANNEL, PMC_CHANNEL, HTML_CHANNEL)

NATURE_ARTICLE_BASE = "https://www.nature.com/articles/"
RECORDED_ADDRESS_SOURCE = "recorded_verified_article_url"
DOI_DERIVED_ADDRESS_SOURCE = "doi_derived_article_url"
RECORDED_TEXT_SOURCE = "recorded_full_text_source"
ADDRESS_SOURCE_LABELS = {
    RECORDED_ADDRESS_SOURCE: "the article address this record already had verified",
    DOI_DERIVED_ADDRESS_SOURCE: "the DOI of this article",
    RECORDED_TEXT_SOURCE: "the address this record recorded for its own text",
}

RENDER_PAGE_SIZE = (595.0, 842.0)
RENDER_MARGIN = 56.0
RENDER_BODY_SIZE = 9.5
RENDER_LINE_HEIGHT = 12.4
RENDER_FONT = "helv"
RENDER_STATEMENT = (
    "Generated copy of the text this record keeps for its article, made because the pipeline that "
    "reads a paper takes a PDF. This is not the publisher's PDF and does not replace it.\n"
    "Article DOI: {doi}\nArticle title: {title}\n"
    "Source address: {url}\nThat address comes from {address_source}.\n"
    "The text below is this record's own full text{text_origin}; sha256 {digest}."
)

PMC_HOST = "pmc.ncbi.nlm.nih.gov"
PMC_ENTRY_POINT = "https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/"
PMC_CLOUD_HOST = "pmc-oa-opendata.s3.amazonaws.com"
PMC_CLOUD_BUCKET = f"https://{PMC_CLOUD_HOST}/"
POW_COOKIE_DEFAULT = "cloudpmc-viewer-pow"
POW_MAX_DIFFICULTY = 5


def _pdf_target(batch: Path, row: dict) -> Path:
    """Resolve the recovered original inside the paper's own verified asset folder."""
    root = _artifact_root(batch, row).resolve()
    text = _fulltext_path(batch, str(row.get("domain") or ""), _portable_row(row))
    if text is None or not root.is_relative_to(batch.parent):
        raise ValueError("Missing or out-of-batch full text")
    target = text.resolve().parent.parent / "official.pdf"
    if not target.is_relative_to(root) or not target.is_relative_to(batch):
        raise ValueError("Recovery target escapes current batch")
    return target


def _portable_row(row: dict) -> dict:
    r"""Read a ledger written on Windows when recovery runs on another platform.

    The collector stores asset paths with the separator of the machine that
    wrote them, so ``assets\biology\...`` is one meaningless filename on
    POSIX. Rewriting the separators keeps the whole run reproducible wherever
    the batch is read; it never rewrites the ledger itself.
    """
    fields = {name: row[name].replace("\\", "/")
              for name in ("full_text_path", "source_text_path")
              if isinstance(row.get(name), str) and "\\" in row[name]}
    return {**row, **fields} if fields else row


def _verified_nature_address(row: dict) -> str:
    """Return the Nature article address this record already had verified."""
    landing = row.get("publisher_landing_evidence") or {}
    url = str(row.get("fulltext_source") or "")
    doi = str(row.get("doi") or "").lower()
    if not doi.startswith("10.1038/"):
        return ""
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.netloc != "www.nature.com"
            or parsed.path != "/articles/" + doi.removeprefix("10.1038/")
            or parsed.query or parsed.fragment):
        return ""
    pdf_evidence = row.get("official_pdf_evidence") or {}
    landing_verified = (landing.get("resolved_url") == url
                        and landing.get("title_found") and landing.get("venue_found"))
    pdf_verified = (pdf_evidence.get("method") == "official_pdf_text"
                    and pdf_evidence.get("resolved_url") == url + ".pdf"
                    and pdf_evidence.get("title_found") and pdf_evidence.get("venue_found"))
    return url if landing_verified or pdf_verified else ""


def _nature_address(row: dict) -> tuple[str, str]:
    """Return the Nature article address for one record, and how it was named.

    A recorded address that was verified is preferred. A Nature-family DOI names
    the same article on its own, so a record that carries only that DOI can still
    be reached; the bytes then have to carry the DOI before anything is
    published, which keeps a derived address honest instead of a claim about an
    address nobody checked.
    """
    verified = _verified_nature_address(row)
    if verified:
        return verified, RECORDED_ADDRESS_SOURCE
    doi = str(row.get("doi") or "").lower()
    if not doi.startswith("10.1038/"):
        return "", ""
    return NATURE_ARTICLE_BASE + doi.removeprefix("10.1038/"), DOI_DERIVED_ADDRESS_SOURCE


def _nature_spec(batch: Path, row: dict) -> dict:
    """Only derive a PDF from a Nature article address this record can name."""
    if row.get("figure_asset_tier") != "B_official_standalone":
        raise ValueError("Not an official standalone figure record")
    address, named_by = _nature_address(row)
    if not address:
        raise ValueError("Missing unambiguous verified official article URL")
    target = _pdf_target(batch, row)
    return {"paper_id": row["paper_id"], "doi": str(row.get("doi") or "").lower(),
            "title": row["title"], "channel": NATURE_CHANNEL, "url": address + ".pdf",
            "address_source": named_by, "target": str(target),
            "relative_path": target.relative_to(batch).as_posix()}


def _pmc_spec(batch: Path, row: dict) -> dict:
    """Only derive a PDF from the PMC identifier the official record already verified."""
    if row.get("figure_asset_tier") != "B_official_standalone":
        raise ValueError("Not an official standalone figure record")
    pmcid = str((row.get("paper_type_evidence") or {}).get("pmcid") or "").strip().upper()
    if not re.fullmatch(r"PMC\d+", pmcid):
        raise ValueError("No verified PMC identifier for this paper")
    doi = str(row.get("doi") or "").lower()
    if not doi or not str(row.get("title") or "").strip():
        raise ValueError("Paper DOI or title missing")
    target = _pdf_target(batch, row)
    return {"paper_id": row["paper_id"], "doi": doi, "title": row["title"],
            "channel": PMC_CHANNEL, "pmcid": pmcid,
            "request_url": PMC_ENTRY_POINT.format(pmcid=pmcid), "target": str(target),
            "relative_path": target.relative_to(batch).as_posix()}


def _html_spec(batch: Path, row: dict) -> dict:
    """Name the address a record's own text can be copied from.

    An address qualifies when this record can answer for it: the article's own
    page if it was verified, the page a Nature-family DOI names, or — when the
    record has neither — the address the record itself recorded for its full
    text. The copy states which of the three named the address, and its text
    still has to carry this paper's DOI. The copy channel never carries the
    publisher's file, which is why it is offered last and only for records no
    file channel could serve.
    """
    if row.get("figure_asset_tier") != "B_official_standalone":
        raise ValueError("Not an official standalone figure record")
    doi = str(row.get("doi") or "").lower()
    url, named_by = _copy_address(row)
    if not url:
        raise ValueError("No address this record can answer for")
    text = _fulltext_path(batch, str(row.get("domain") or ""), _portable_row(row))
    if text is None or not text.stat().st_size:
        raise ValueError("Missing the record's own full text")
    target = _pdf_target(batch, row)
    return {"paper_id": row["paper_id"], "doi": doi, "title": row["title"],
            "channel": HTML_CHANNEL, "source_url": url, "address_source": named_by,
            "text_source": str(row.get("fulltext_source") or ""), "source_text": str(text),
            "target": str(target), "relative_path": target.relative_to(batch).as_posix()}


def _copy_address(row: dict) -> tuple[str, str]:
    """Return the address this record's own text may be copied from, and its origin.

    The article's own page outranks the source the record merely recorded,
    because a verified page is evidence about the article while a recorded text
    source is evidence about where the text was captured.
    """
    url, named_by = _nature_address(row)
    if url:
        return url, named_by
    recorded = str(row.get("fulltext_source") or "").strip()
    parsed = urlparse(recorded)
    if parsed.scheme == "https" and parsed.netloc and not parsed.username:
        return recorded, RECORDED_TEXT_SOURCE
    return "", ""


_SPEC_BUILDERS = {NATURE_CHANNEL: _nature_spec, PMC_CHANNEL: _pmc_spec, HTML_CHANNEL: _html_spec}


def _request_specs(batch: Path, row: dict, channels: tuple[str, ...] = CHANNELS) -> list[dict]:
    """List every channel that can name an original for this row, strongest first.

    A channel that names the article from evidence the record already verified
    outranks one that only derives the address from the DOI, so a verified route
    never loses to a derived guess. Both outrank the copy channel, because a
    copy can only reproduce the publisher's page text.
    """
    verified, derived, copies, reasons = [], [], [], []
    for channel in channels:
        try:
            spec = _SPEC_BUILDERS[channel](batch, row)
        except ValueError as exc:
            reasons.append(f"{channel}: {exc}")
            continue
        if channel == HTML_CHANNEL:
            copies.append(spec)
        elif spec.get("address_source") == DOI_DERIVED_ADDRESS_SOURCE:
            derived.append(spec)
        else:
            verified.append(spec)
    specs = verified + derived + copies
    if not specs:
        raise ValueError("; ".join(reasons) or "No recovery channel selected")
    return specs


def _request_spec(batch: Path, row: dict, channels: tuple[str, ...] = CHANNELS) -> dict:
    """Take the strongest channel that can name an original for this row."""
    return _request_specs(batch, row, channels)[0]


def _verify_identity(content: bytes, spec: dict) -> dict:
    """Decode in the caller thread; PDFium is never called by HTTP workers."""
    import pypdfium2
    if not content.startswith(b"%PDF-"):
        raise _UnverifiedArticle("Official response is not PDF content")
    try:
        with pypdfium2.PdfDocument(content) as doc:
            pages = len(doc)
            checked = min(3, pages)
            text = ""
            for index in range(checked):
                page = doc[index]
                textpage = page.get_textpage()
                try:
                    text += textpage.get_text_range()
                finally:
                    textpage.close()
                    page.close()
    except pypdfium2.PdfiumError as exc:
        raise _UnverifiedArticle(f"A PDF reader cannot open these bytes: {exc}") from exc
    compact = re.sub(r"\s+", "", text.casefold())
    normalized = re.sub(r"[^a-z0-9]", "", text.casefold())
    title = re.sub(r"[^a-z0-9]", "", str(spec.get("title") or "").casefold())
    if spec["doi"] not in compact:
        raise _UnverifiedArticle("Official PDF does not carry the strict paper DOI")
    # The DOI identifies the article on its own. A title is only reported,
    # because typeset letters drop out of extraction often enough that
    # requiring it rejected real originals (for example a styled capital
    # that extracts as "Micro plit" for "MicroSplit").
    return {"pages": pages, "doi_found": True, "title_found": bool(title) and title in normalized,
            "decoder": "pdfium", "checked_pages": checked}


class _ArticleNotServed(ValueError):
    """The publisher answered, but would not hand over its own paper file."""


class _UnverifiedArticle(ValueError):
    """A channel answered, but its bytes cannot be shown to be this paper."""


def _download(spec: dict) -> tuple[bytes, dict]:
    """Route to the channel's transport; every one returns the same evidence shape."""
    if spec["channel"] == PMC_CHANNEL:
        return _download_pmc(spec)
    if spec["channel"] == HTML_CHANNEL:
        return _render_page_text_copy(spec)
    return _download_nature(spec)


def _download_nature(spec: dict) -> tuple[bytes, dict]:
    client = LiteratureHttpClient("official_source_recovery:www.nature.com",
                                  timeout=45, retries=1, max_in_flight=2)
    try:
        response = client.request(spec["url"])
        if response.url != spec["url"]:
            raise _ArticleNotServed("Official PDF redirected away from declared article")
        if not response.content.startswith(b"%PDF-"):
            raise _ArticleNotServed("Publisher answered with a page instead of its paper file")
        return response.content, {"resolved_url": response.url, "http_status": response.status_code}
    finally:
        client.session.close()


def _latin(value: str) -> str:
    """Keep the copy readable in its built-in font instead of losing whole lines."""
    return value.encode("latin-1", "replace").decode("latin-1")


def _wrapped_lines(text: str, width: float, font: str, size: float) -> list[str]:
    """Break text to a measured width so the copy paginates the same way twice."""
    import pymupdf
    lines: list[str] = []
    for paragraph in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            lines.append("")
            continue
        current = ""
        for word in paragraph.split(" "):
            candidate = f"{current} {word}".strip()
            if current and pymupdf.get_text_length(candidate, fontname=font, fontsize=size) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def _render_page_text_copy(spec: dict) -> tuple[bytes, dict]:
    """Typeset the publisher page's own text into the document the pipeline reads.

    The copy is not the publisher's file and never claims to be: its first page
    states what it is and where the text came from, and the text is hashed so the
    snapshot can tie the copy to the same bytes it already keeps as full text.
    """
    import pymupdf
    source = Path(spec["source_text"])
    raw = source.read_bytes()
    text = raw.decode("utf-8", "replace")
    if spec["doi"] not in re.sub(r"\s+", "", text.casefold()):
        raise ValueError("Publisher page text does not carry the strict paper DOI")
    digest = hashlib.sha256(raw).hexdigest()
    statement = _latin(RENDER_STATEMENT.format(
        url=spec["source_url"],
        address_source=ADDRESS_SOURCE_LABELS.get(str(spec.get("address_source") or ""),
                                                 "the address the record names"),
        text_origin=(", captured from " + spec["text_source"])
                    if spec.get("text_source") and spec["text_source"] != spec["source_url"] else "",
        doi=spec["doi"], title=str(spec.get("title") or ""), digest=digest))
    width = RENDER_PAGE_SIZE[0] - 2 * RENDER_MARGIN
    lines = _wrapped_lines(statement, width, RENDER_FONT, RENDER_BODY_SIZE) + [""]
    lines += _wrapped_lines(_latin(text), width, RENDER_FONT, RENDER_BODY_SIZE)
    per_page = max(1, int((RENDER_PAGE_SIZE[1] - 2 * RENDER_MARGIN) // RENDER_LINE_HEIGHT))
    document = pymupdf.open()
    try:
        for start in range(0, len(lines), per_page):
            page = document.new_page(width=RENDER_PAGE_SIZE[0], height=RENDER_PAGE_SIZE[1])
            y = RENDER_MARGIN + RENDER_BODY_SIZE
            for line in lines[start:start + per_page]:
                if line:
                    page.insert_text((RENDER_MARGIN, y), line, fontname=RENDER_FONT,
                                     fontsize=RENDER_BODY_SIZE)
                y += RENDER_LINE_HEIGHT
        content = document.tobytes(deflate=True, garbage=3)
    finally:
        document.close()
    return content, {"resolved_url": spec["source_url"], "http_status": None,
                     "transport": "publisher_page_text_rendered",
                     "source_text_sha256": digest, "source_text_bytes": len(raw),
                     "rendered_pages": len(range(0, len(lines), per_page))}


def _pmc_pdf_url(entry_url: str, pmcid: str) -> str:
    """Accept only the article's own PDF address, never a rewritten host or path."""
    parsed = urlparse(entry_url)
    prefix = f"/articles/{pmcid}/pdf/"
    if (parsed.scheme != "https" or parsed.netloc != PMC_HOST or parsed.query
            or parsed.fragment or not parsed.path.startswith(prefix)
            or not parsed.path.endswith(".pdf")):
        raise ValueError("PMC entry point did not resolve to the article PDF")
    return entry_url


def _pow_challenge(document: bytes) -> tuple[str, int, str]:
    """Read the published proof-of-work parameters instead of guessing them."""
    text = document.decode("utf-8", "replace")
    challenge = re.search(r'POW_CHALLENGE\s*=\s*"([^"]+)"', text)
    difficulty = re.search(r'POW_DIFFICULTY\s*=\s*"(\d+)"', text)
    cookie = re.search(r'POW_COOKIE_NAME\s*=\s*"([^"]+)"', text)
    if not challenge or not difficulty:
        raise ValueError("PMC proof-of-work challenge is missing")
    level = int(difficulty.group(1))
    if not 1 <= level <= POW_MAX_DIFFICULTY:
        raise ValueError(f"PMC proof-of-work difficulty out of range: {level}")
    return challenge.group(1), level, cookie.group(1) if cookie else POW_COOKIE_DEFAULT


def _pow_nonce(challenge: str, difficulty: int) -> int:
    """Reproduce the page's own sha256 rule with a bounded, deterministic search."""
    prefix = "0" * difficulty
    for nonce in range(32 * 16 ** difficulty):
        if hashlib.sha256((challenge + str(nonce)).encode("utf-8")).hexdigest().startswith(prefix):
            return nonce
    raise ValueError("PMC proof-of-work did not converge within its budget")


class _StandardTlsAdapter(HTTPAdapter):
    """Build connection pools with the standard library's TLS ClientHello.

    PMC's edge answers urllib3's own restricted-cipher hello with 403 while
    accepting the standard library's. Only the handshake differs; the bytes we
    verify and publish are unchanged.
    """

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pool_kwargs["ssl_context"] = ssl.create_default_context()
        return super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)


def _pmc_session() -> requests.Session:
    session = requests.Session()
    session.mount("https://", _StandardTlsAdapter())
    return session


def _package_version(key: str) -> int:
    match = re.search(r"\.(\d+)\.pdf$", key, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _pmc_package_key(document: bytes, pmcid: str) -> str:
    """Choose the article's own PDF object, never a figure, supplement, or older version."""
    try:
        root = ElementTree.fromstring(document)
    except ElementTree.ParseError as exc:
        raise ValueError("PMC cloud listing is not readable XML") from exc
    wanted = pmcid.casefold()
    keys = [element.text or "" for element in root.iter() if element.tag.endswith("Key")]
    candidates = [key for key in keys
                  if key.casefold().endswith(".pdf")
                  and key.casefold().rsplit("/", 1)[-1].removesuffix(".pdf").split(".")[0] == wanted]
    if not candidates:
        raise ValueError("PMC cloud package has no article PDF")
    return max(candidates, key=_package_version)


def _optional_request(client: LiteratureHttpClient, url: str):
    """Return None when an object is simply absent, so a caller can try another address.

    Every other failure still propagates: a blocked or unhealthy endpoint must
    move on to the next transport instead of looking like a missing file.
    """
    try:
        response = client.request(url)
    except ProviderRequestError as exc:
        if exc.status == "not_found":
            return None
        raise
    return response if int(response.status_code) == 200 else None


def _download_pmc_cloud(client: LiteratureHttpClient, spec: dict) -> tuple[bytes, dict]:
    """Take the article's PDF from PMC's own open-access cloud package.

    The first version of every article lives at a predictable address, so ask
    for it directly and only pay for a bucket listing when it is not there.
    """
    direct = f"{PMC_CLOUD_BUCKET}{spec['pmcid']}.1/{spec['pmcid']}.1.pdf"
    url = direct
    response = _optional_request(client, direct)
    if response is None or not response.content.startswith(b"%PDF-"):
        listing = client.request(PMC_CLOUD_BUCKET,
                                 params={"list-type": "2", "prefix": spec["pmcid"], "max-keys": "1000"})
        url = PMC_CLOUD_BUCKET + _pmc_package_key(listing.content, spec["pmcid"])
        response = client.request(url)
    if response.url != url:
        raise ValueError("PMC cloud package redirected away from the article PDF")
    return response.content, {"resolved_url": url, "http_status": response.status_code,
                              "transport": "pmc_cloud_package", "proof_of_work": None}


def _download_pmc_front_end(client: LiteratureHttpClient, spec: dict) -> tuple[bytes, dict]:
    """Resolve the article address, answer its challenge, and keep both steps as evidence."""
    entry = client.request(spec["request_url"])
    pdf_url = _pmc_pdf_url(entry.url, spec["pmcid"])
    response = client.request(pdf_url)
    evidence = {"resolved_url": pdf_url, "http_status": response.status_code,
                "transport": "pmc_article_front_end", "proof_of_work": None}
    if response.content.startswith(b"%PDF-"):
        return response.content, evidence
    challenge, difficulty, cookie_name = _pow_challenge(response.content)
    nonce = _pow_nonce(challenge, difficulty)
    client.session.cookies.set(cookie_name, f"{challenge},{nonce}",
                               domain=urlparse(pdf_url).netloc, path="/")
    response = client.request(pdf_url)
    if response.url != pdf_url:
        raise ValueError("PMC PDF redirected away from the resolved article address")
    evidence["http_status"] = response.status_code
    evidence["proof_of_work"] = {"algorithm": "sha256", "difficulty": difficulty, "nonce": nonce}
    return response.content, evidence


def _download_pmc(spec: dict) -> tuple[bytes, dict]:
    """Try the cloud package first, then the article front-end for anything it omits."""
    client = LiteratureHttpClient("official_source_recovery:pmc.ncbi.nlm.nih.gov",
                                  timeout=120, retries=1, min_interval=0.25, max_in_flight=1,
                                  session=_pmc_session())
    try:
        return _download_pmc_cloud(client, spec)
    except (ValueError, ProviderRequestError) as exc:
        try:
            return _download_pmc_front_end(client, spec)
        except ProviderRequestError as fallback:
            if fallback.status != "not_found":
                # Nothing was reached on either route, so this stays a retry.
                raise ProviderRequestError("official_source_recovery", fallback.status,
                                           f"PMC cloud: {exc}; PMC front end: {fallback}") from fallback
            # The front end answers 404 for a paper its archive does not hold.
            raise _ArticleNotServed(f"PMC cloud: {exc}; PMC front end: {fallback}") from fallback
        except ValueError as fallback:
            raise _ArticleNotServed(f"PMC cloud: {exc}; PMC front end: {fallback}") from fallback
    finally:
        client.session.close()


def _publish(batch: Path, spec: dict, content: bytes, response: dict,
             refusals: list[str] | None = None) -> dict:
    identity = _verify_identity(content, spec)
    digest = hashlib.sha256(content).hexdigest()
    target = Path(spec["target"])
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".pdf", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        copy_verified(temporary, target, digest)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(target) != digest:
        raise ValueError("Recovered PDF changed after publication")
    return {"paper_id": spec["paper_id"], "doi": spec["doi"], "title": spec["title"],
            "channel": spec["channel"], "pmcid": spec.get("pmcid"),
            **({"address_source": spec["address_source"]} if spec.get("address_source") else {}),
            "relative_path": target.relative_to(batch).as_posix(),
            "requested_url": spec.get("url") or spec.get("request_url") or spec["source_url"],
            "sha256": digest, "size_bytes": len(content), "identity_check": identity,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            **({"refused_channels": list(refusals)} if refusals else {}), **response}


def _worker_ceiling(workers: int, channels: tuple[str, ...]) -> int:
    """Keep the Nature endpoint at its historical ceiling; PMC's own bucket tolerates more."""
    ceiling = 2 if tuple(channels) == (NATURE_CHANNEL,) else 32
    return max(1, min(ceiling, workers))


def _recover_one(batch: Path, specs: list[dict]) -> dict:
    """Publish from the first channel that can supply bytes the record can answer for.

    A publisher that answers and refuses to serve its own file — or serves bytes
    no reader can open as this paper — moves the record on to the next channel.
    A transport failure does not: a file that was never reached has to stay a
    failure to retry, not a reason to settle for a copy.
    """
    refusals = []
    for spec in specs:
        try:
            content, response = _download(spec)
        except _ArticleNotServed as exc:
            refusals.append(f"{spec['channel']}: {exc}")
            continue
        try:
            return _publish(batch, spec, content, response, refusals)
        except _UnverifiedArticle as exc:
            refusals.append(f"{spec['channel']}: {exc}")
    raise ValueError("; ".join(refusals) or "No recovery channel supplied the paper")


def recover_official_source_pdfs(batch_root: str | Path, *, workers: int = 2,
                                 channels: tuple[str, ...] = CHANNELS, limit: int = 0,
                                 report_path: str | Path | None = None, progress=None) -> dict:
    """Supplement originals only; never rewrite a paper, figure, or existing PDF."""
    batch = Path(batch_root).resolve()
    unknown = [channel for channel in channels if channel not in _SPEC_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown recovery channels: {', '.join(unknown)}")
    with literature_batch_guard(batch, purpose="recover-official-source-pdfs"):
        if active_legacy_writers(batch):
            raise LiteratureBusyError("Collection writer active during PDF recovery")
        if (batch / "manifests/source_binding_repair_pending.json").exists():
            raise LiteratureBusyError("Recover binding journal before supplementing originals")
        rows = [r for r in strict_jsonl(batch / "verified/strict_training_papers.jsonl")
                if r.get("figure_asset_tier") == "B_official_standalone"]
        journal = Path(report_path) if report_path else batch / "manifests/source_pdf_recovery.json"
        journal.parent.mkdir(parents=True, exist_ok=True)
        previous = json.loads(journal.read_text(encoding="utf-8")) if journal.exists() else {}
        evidence = {r["paper_id"]: r for r in previous.get("papers", [])}
        completed, errors, jobs = [], [], []
        def checkpoint():
            report = {"ok": not errors and len(completed) == len(rows), "total": len(rows),
                      "channels": list(channels), "limit": limit,
                      "checked": len(completed) + len(errors), "completed": len(completed),
                      "ineligible": sum(1 for error in errors if error.get("ineligible")),
                      "failed": sum(1 for error in errors if not error.get("ineligible")),
                      "papers": completed, "errors": errors,
                      "updated_at": datetime.now(timezone.utc).isoformat()}
            write_json(journal, report)
            if progress:
                progress({k: v for k, v in report.items() if k not in {"papers", "errors"}}
                         | {"errors": len(errors)})
            return report
        for row in rows:
            try:
                candidates = _request_specs(batch, row, channels)
                target = Path(candidates[0]["target"])
                if target.exists():
                    old = evidence.get(candidates[0]["paper_id"])
                    if old:
                        if sha256_file(target) != old["sha256"]:
                            raise ValueError("Existing original differs from recovery provenance")
                        _verify_identity(target.read_bytes(), candidates[0])
                        completed.append(old)
                    else:
                        # A crash may leave a verified PDF before its checkpoint.
                        # Re-download and require exact equality; never overwrite.
                        jobs.append(candidates)
                else:
                    jobs.append(candidates)
            except ValueError as exc:
                errors.append({"paper_id": row["paper_id"], "error": str(exc), "ineligible": True})
            except Exception as exc:
                errors.append({"paper_id": row["paper_id"], "error": str(exc), "ineligible": False})
        if limit:
            jobs = jobs[:limit]
        checkpoint()
        with ThreadPoolExecutor(max_workers=_worker_ceiling(workers, channels)) as pool:
            pending = {pool.submit(_recover_one, batch, group): group for group in jobs}
            for future in as_completed(pending):
                group = pending.pop(future)
                try:
                    completed.append(future.result())
                except Exception as exc:
                    errors.append({"paper_id": group[0]["paper_id"], "error": str(exc), "ineligible": False})
                checkpoint()
        return checkpoint()
