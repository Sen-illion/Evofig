"""Bind official standalone figure files without extracting paper-PDF objects."""

from __future__ import annotations

import hashlib
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
from typing import Any, Iterable
from urllib.parse import quote, urljoin, urlparse, urlunparse
from xml.etree import ElementTree

from rfs.providers.literature import LiteratureHttpClient, ProviderRequestError
from .literature_freeze import ACCEPTED_TYPES, DOMAINS
from .literature_io import atomic_write_text


# Kept in step with the widest sibling list (probe_fresh_arxiv_sources.py) so a
# paper is not admitted or refused merely because of which collector reached it.
FIGURE_TERMS = re.compile(
    r"\b(architecture|framework|pipeline|workflow|overview|schematic|illustration|diagram|"
    r"system diagram|system mechanism|method overview|technical route|experimental workflow|"
    r"experimental procedure|experimental design|study design|mechanism diagram|process flow|"
    r"methodology|proposed method|proposed approach|our approach|model architecture|"
    r"experimental setup|synthesis route|fabrication process|reaction mechanism|"
    r"mechanistic scheme|workflow diagram|"
    # Materials, chemistry and physics papers name the same kind of figure
    # differently; without these the list only recognised CS and biology usage
    # and refused titles like "Concept of the PTD-immobilized photocatalyst".
    r"concept|conceptual design|working principle|operating principle|design principle|"
    r"design strategy|synthesis strategy|fabrication route|device design|"
    r"measurement setup|reaction scheme|"
    # "illustration" alone misses the far commoner verb forms, and a graphical
    # abstract is by definition the paper's overview figure.
    r"sketch|graphical abstract|illustrating|illustrates|"
    r"proposed mechanism|proposed model|schematic model)\b",
    re.I,
)
# These name a *result* figure. They are matched against the figure title only,
# so a bare technique word like "microscopy" must stay qualified: "Schematic of
# the microscopy setup" is a framework figure, "Microscopy images of..." is not.
EXCLUDED_TERMS = re.compile(
    r"\b(roc curve|confusion matrix|ablation study|result comparison|microscopy image|"
    r"histology|dataset sample|qualitative result|f1 score|survival curve|"
    r"kaplan.meier|heatmap|volcano plot|western blot|diffraction pattern)\b",
    re.I,
)
ASSET_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".eps", ".tif", ".tiff", ".webp"}
XLINK = "{http://www.w3.org/1999/xlink}href"
REGISTRY_REDIRECT_HOSTS = {"doi.org", "dx.doi.org", "api.crossref.org", "dblp.org"}
AGGREGATOR_LANDING_HOSTS = {
    "europepmc.org",
    "www.europepmc.org",
    "pubmed.ncbi.nlm.nih.gov",
    "www.semanticscholar.org",
    "openalex.org",
    "api.openalex.org",
}
# One Springer Nature figure object is served at several widths, and the markup a
# route reads advertises the narrowest of them for its own layout: article pages
# embed ``m312``, the PMC JATS asset route names ``lw685`` and magazine assets
# answer as ``lw767``. Storing that spelling verbatim stores a downscaled copy of
# the published figure, so the size token is re-spelled and the wider spellings
# are tried first. The URL a route gave stays last, because it is the spelling
# that is known to answer.
ASSET_SIZE_LADDERS = {
    "media.springernature.com": ("full", "original", "lw1200", "lw1000", "lw685", "m312"),
    "media.nature.com": ("full", "original", "lw1200", "lw1000", "lw767", "w767"),
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _identity(row: dict[str, Any]) -> str:
    for field in ("doi", "paper_id", "official_paper_id", "pmcid", "pmid", "arxiv_id"):
        if value := _clean(row.get(field)):
            return value.casefold()
    return re.sub(r"[^a-z0-9]+", " ", _clean(row.get("title")).casefold()).strip()


def _leading_sentence(text: str) -> str:
    """Approximate a figure title when the publisher ships no structured one."""
    match = re.search(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
    return (text[: match.start()] if match else text)[:300]


def _caption_title(title: str, body: str) -> str:
    """The sentence that names the whole figure, never the per-panel description."""
    value = _clean(title)
    if value:
        # "Fig. 1: <title>" on publisher HTML; the label carries no topical signal.
        return re.sub(r"^\s*(?:Fig(?:ure)?\.?\s*\d+\s*[:.|\-│]\s*)", "", value, flags=re.I)[:300]
    return _leading_sentence(_clean(body))


def _eligible_caption(caption: str) -> bool:
    """Judge the figure title alone.

    A multi-panel caption runs 800+ characters and describes every panel, so
    matching over the whole of it both admits figures whose only mention of a
    schematic points at a *different* figure, and rejects genuine framework
    diagrams because one panel happens to say "micrograph" or "accuracy".
    """
    return bool(FIGURE_TERMS.search(caption)) and not bool(EXCLUDED_TERMS.search(caption))


def _publisher_direct_url(doi: str) -> str:
    """Skip the DOI redirector so the throttle lane is keyed by the real publisher."""
    value = _clean(doi).casefold()
    if value.startswith("10.1145/"):
        return f"https://dl.acm.org/doi/{value}"
    if value.startswith("10.1038/"):
        return f"https://www.nature.com/articles/{value.split('/', 1)[1]}"
    if value.startswith("10.1186/"):
        # BMC runs on the Springer platform, so the shared figure markup applies.
        return f"https://link.springer.com/article/{value}"
    return ""


def _official_probe_url(row: dict[str, Any]) -> str:
    """Prefer an identity-verified publisher landing over a DOI redirector."""
    canonical = _clean(row.get("official_url"))
    landing = row.get("publisher_landing_evidence") or {}
    resolved = _clean(landing.get("resolved_url"))
    parsed = urlparse(resolved)
    if (
        landing.get("title_found")
        and landing.get("venue_found")
        and parsed.scheme in {"http", "https"}
        and parsed.netloc.casefold() not in {"doi.org", "api.crossref.org", "dblp.org"}
    ):
        return resolved
    return _publisher_direct_url(row.get("doi")) or canonical


def _request(client: Any, url: str) -> Any:
    requester = client.for_url(url) if hasattr(client, "for_url") else client
    return requester.request(url)


def _springer_asset_alias(doi: str, href: str) -> list[str]:
    """Rewrite a PMC graphic name onto Springer Nature's image host.

    Springer-platform journals (Nature, Nature Communications, BMC) name their
    JATS graphics ``<journal>_<year>_<id>_Fig<n>_HTML``, which is exactly the
    object name on media.springernature.com. The PMC asset host answers 403 for
    the same figure, so without this the PMC route finds a figure it cannot
    fetch. The stored extension is unreliable; the served object is a PNG.
    """
    stem = Path(href.split("?", 1)[0]).stem
    if "_Fig" not in stem:
        return []
    art = quote(f"art:{_clean(doi)}", safe="")
    return [f"https://media.springernature.com/lw685/springer-static/image/{art}/MediaObjects/{stem}.png"]


def _plos_asset_alias(doi: str, href: str) -> list[str]:
    """Rewrite a PMC graphic name onto the PLOS endpoint, which serves the image.

    PLOS ignores the journal slug in this path, so one spelling covers every
    PLOS title; the PMC asset host answers 403 for the same figure.
    """
    match = re.search(r"(g\d+)", Path(href.split("?", 1)[0]).stem)
    if not match:
        return []
    return [f"https://journals.plos.org/plosone/article/figure/image?size=large&id={doi}.{match.group(1)}"]


def _resolution_ladder(url: str) -> list[str]:
    """Return one asset URL widest first, keeping the given URL as the fallback."""
    parsed = urlparse(url)
    ladder = ASSET_SIZE_LADDERS.get(parsed.netloc.casefold())
    if not ladder:
        return [url]
    segments = parsed.path.split("/")
    for index in range(1, len(segments)):
        if segments[index] not in ladder:
            continue
        wider = ladder[: ladder.index(segments[index])]
        if not wider:
            return [url]
        scheme, netloc, _path, params, query, fragment = parsed
        return [
            urlunparse(
                (scheme, netloc, "/".join([*segments[:index], size, *segments[index + 1 :]]), params, query, fragment)
            )
            for size in wider
        ] + [url]
    return [url]


def _figure_routes(row: dict[str, Any], official_url: str) -> list[dict[str, Any]]:
    """Order the figure sources so a route that binds nothing falls through."""
    pmc: list[dict[str, Any]] = []
    pmcid = _clean(row.get("pmcid") or (row.get("paper_type_evidence") or {}).get("pmcid"))
    doi_value = _clean(row.get("doi"))
    if pmcid:
        pmcid = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
        alias = None
        if doi_value.casefold().startswith("10.1371/"):
            alias = lambda href: _plos_asset_alias(doi_value, href)  # noqa: E731
        elif doi_value.casefold().startswith(("10.1038/", "10.1186/")):
            alias = lambda href: _springer_asset_alias(doi_value, href)  # noqa: E731
        pmc.append(
            {
                "kind": "pmc_jats",
                "url": f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
                "figure_base": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/bin/",
                "asset_alias": alias,
            }
        )
    doi = _clean(row.get("doi"))
    aggregated = urlparse(official_url).netloc.casefold() in AGGREGATOR_LANDING_HOSTS
    # A known publisher is addressed directly so the throttle lane and the
    # circuit breaker are keyed by the host that actually serves the figure.
    publisher = _publisher_direct_url(doi)
    # The DOI redirector resolves to the publisher we may already be addressing.
    # Keeping both fetches one page twice and, because the throttle lane is keyed
    # by the requested host, does the second fetch off the publisher's polite
    # lane — which is what earned a host-wide 403.
    redirector = urlparse(official_url).netloc.casefold() in REGISTRY_REDIRECT_HOSTS
    landings: list[str] = [
        publisher,
        "" if aggregated or (redirector and publisher) else official_url,
        f"https://doi.org/{doi}" if doi and not publisher else "",
        # An aggregator landing is dropped, not kept as a last resort: it is a
        # JavaScript shell that answers 403 to every scripted client, so trying
        # it can only spend a request and mask the real per-route verdict.
    ]
    routes = pmc + [
        {"kind": "official_html_figure", "url": value, "figure_base": None} for value in landings if value
    ]
    return list({route["url"]: route for route in routes}.values())


class _FigureParser(HTMLParser):
    # Publishers split one caption across <figcaption> (the short title) and a
    # sibling description block, so the whole <figure> is read minus its chrome.
    CHROME_TAGS = {"a", "svg", "button", "script", "style"}

    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self.figure_depth = 0
        self.chrome_depth = 0
        self.title_depth = 0
        self.title: list[str] = []
        self.body: list[str] = []
        self.assets: list[str] = []
        self.figures: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value or "" for key, value in attrs}
        lower = tag.casefold()
        if lower == "figure":
            if self.figure_depth == 0:
                self.title, self.body, self.assets = [], [], []
                self.chrome_depth = self.title_depth = 0
            self.figure_depth += 1
        if not self.figure_depth:
            return
        if lower == "figcaption":
            self.title_depth += 1
        if lower in self.CHROME_TAGS:
            self.chrome_depth += 1
        for key in ("src", "data-src", "href"):
            value = values.get(key)
            if value and lower in {"img", "source", "a"}:
                candidate = urljoin(self.base_url, value)
                if Path(candidate.split("?", 1)[0]).suffix.casefold() in ASSET_SUFFIXES:
                    self.assets.append(candidate)
        srcset = values.get("srcset") or values.get("data-srcset")
        if srcset and lower in {"img", "source"}:
            for item in srcset.split(","):
                value = item.strip().split()[0]
                if value:
                    self.assets.append(urljoin(self.base_url, value))

    def handle_endtag(self, tag: str) -> None:
        lower = tag.casefold()
        if self.figure_depth and lower in self.CHROME_TAGS and self.chrome_depth:
            self.chrome_depth -= 1
        if self.figure_depth and lower == "figcaption" and self.title_depth:
            self.title_depth -= 1
        if lower == "figure" and self.figure_depth:
            self.figure_depth -= 1
            if self.figure_depth == 0:
                title = _caption_title(" ".join(self.title), " ".join(self.body))
                if _eligible_caption(title) and self.assets:
                    caption = _clean(" ".join([*self.title, *self.body]))[:2000]
                    self.figures.append(
                        {
                            "caption": caption,
                            "caption_title": title,
                            "asset_source_urls": list(dict.fromkeys(self.assets)),
                        }
                    )

    def handle_data(self, data: str) -> None:
        if not self.figure_depth or self.chrome_depth:
            return
        (self.title if self.title_depth else self.body).append(data)


class _ArticleTextParser(HTMLParser):
    """Extract only explicit article/main content, excluding scripts and navigation."""

    def __init__(self) -> None:
        super().__init__()
        self.content_depth = 0
        self.ignored_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower = tag.casefold()
        if lower in {"article", "main"}:
            self.content_depth += 1
        if self.content_depth and lower in {"script", "style", "nav", "aside", "noscript"}:
            self.ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        lower = tag.casefold()
        if self.content_depth and lower in {"script", "style", "nav", "aside", "noscript"} and self.ignored_depth:
            self.ignored_depth -= 1
        if lower in {"article", "main"} and self.content_depth:
            self.content_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.content_depth and not self.ignored_depth:
            value = _clean(data)
            if value:
                self.parts.append(value)


def parse_html_figures(body: str, base_url: str) -> list[dict[str, Any]]:
    parser = _FigureParser(base_url)
    parser.feed(body)
    return parser.figures


_XML_PATH_IMAGE = re.compile(
    r"<img\b[^>]*?(?:src=\"(?P<src1>[^\"]+)\"[^>]*?data-path-from-xml=\"(?P<name1>[^\"]+)\""
    r"|data-path-from-xml=\"(?P<name2>[^\"]+)\"[^>]*?src=\"(?P<src2>[^\"]+)\")",
    re.I,
)


def parse_html_asset_map(body: str) -> dict[str, str]:
    """Map a publisher page's images back to the graphic names used in JATS.

    Silverchair journals (Oxford University Press) serve figures from a signed
    CDN URL that cannot be constructed, but they tag each ``<img>`` with the
    JATS graphic name in ``data-path-from-xml``. That makes the landing page the
    only place the PMC route's figure can actually be fetched from, since the
    PMC asset host answers 403.
    """
    mapping: dict[str, str] = {}
    for match in _XML_PATH_IMAGE.finditer(body):
        name = _clean(match.group("name1") or match.group("name2"))
        source = unescape(_clean(match.group("src1") or match.group("src2")))
        if name and source.startswith("http"):
            mapping.setdefault(Path(name).stem.casefold(), source)
    return mapping


def parse_html_fulltext(body: str) -> str:
    parser = _ArticleTextParser()
    parser.feed(body)
    return _clean(" ".join(parser.parts))


def _formula_graphics(figure: Any) -> set[int]:
    """Identify graphics that render an equation inside the caption, not the figure."""
    blocked: set[int] = set()
    for holder in ("inline-formula", "disp-formula"):
        for node in figure.findall(f".//{holder}"):
            blocked.update(id(child) for child in node.iter())
    return blocked


def parse_jats_figures(
    body: str,
    base_url: str,
    *,
    asset_alias: Any = None,
) -> tuple[list[dict[str, Any]], str]:
    root = ElementTree.fromstring(body)
    figures: list[dict[str, Any]] = []
    for figure in root.findall(".//fig"):
        caption_node = figure.find("caption")
        if caption_node is None:
            continue
        # JATS separates the figure title from the per-panel description.
        title_node = caption_node.find("title")
        title_text = " ".join(title_node.itertext()) if title_node is not None else ""
        caption = _clean(" ".join(caption_node.itertext()))
        title = _caption_title(title_text, caption)
        if not _eligible_caption(title):
            continue
        blocked = _formula_graphics(figure)
        assets = []
        for node in figure.findall(".//graphic") + figure.findall(".//media"):
            href = node.get(XLINK) or node.get("href")
            if href and id(node) not in blocked:
                if asset_alias:
                    assets.extend(asset_alias(href))
                assets.append(urljoin(base_url, href))
        if assets:
            figures.append(
                {
                    "caption": caption,
                    "caption_title": title,
                    "asset_source_urls": list(dict.fromkeys(assets)),
                }
            )
    fulltext = _clean(" ".join(root.itertext()))
    return figures, fulltext


def _asset_extension(url: str, content_type: str) -> str:
    suffix = Path(url.split("?", 1)[0]).suffix.casefold()
    if suffix in ASSET_SUFFIXES:
        return suffix
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/svg+xml": ".svg",
        "application/pdf": ".pdf",
        "image/tiff": ".tiff",
        "image/webp": ".webp",
    }
    return mapping.get(content_type.split(";", 1)[0].casefold(), "")


def _valid_standalone_asset(content: bytes, extension: str) -> bool:
    if len(content) < 32 or content.lstrip().lower().startswith((b"<html", b"<!doctype html")):
        return False
    lower = content[:512].lower()
    if extension == ".png":
        return content.startswith(b"\x89PNG\r\n\x1a\n")
    if extension in {".jpg", ".jpeg"}:
        return content.startswith(b"\xff\xd8")
    if extension == ".svg":
        return b"<svg" in lower
    if extension == ".pdf":
        if not content.startswith(b"%PDF"):
            return False
        try:
            import pymupdf

            with pymupdf.open(stream=content, filetype="pdf") as document:
                return 1 <= document.page_count <= 2
        except Exception:
            return False
    if extension == ".eps":
        return content.startswith(b"%!PS")
    if extension in {".tif", ".tiff"}:
        return content.startswith((b"II*\x00", b"MM\x00*"))
    if extension == ".webp":
        return content.startswith(b"RIFF") and content[8:12] == b"WEBP"
    return False


def _resolve_figure_assets(
    figures: list[dict[str, Any]], asset_map: dict[str, str]
) -> list[dict[str, Any]]:
    """Point each figure at the landing page's copy of its own graphic."""
    resolved: list[dict[str, Any]] = []
    for figure in figures:
        sources = [
            asset_map[stem]
            for url in figure.get("asset_source_urls") or []
            if (stem := Path(url.split("?", 1)[0]).stem.casefold()) in asset_map
        ]
        if sources:
            resolved.append({**figure, "asset_source_urls": list(dict.fromkeys(sources))})
    return resolved


def _bind_route_assets(
    client: Any,
    batch: Path,
    folder: Path,
    figures: list[dict[str, Any]],
    source_kind: str,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Download the widest valid standalone asset of every eligible figure."""
    candidates: list[dict[str, Any]] = []
    saved: list[str] = []
    failures: list[str] = []
    for index, figure in enumerate(figures, 1):
        bound_figure = False
        for source in figure.get("asset_source_urls") or []:
            for candidate in _resolution_ladder(source):
                try:
                    asset = _request(client, candidate)
                except ProviderRequestError as exc:
                    # A wider spelling that answers 404 never existed; only the
                    # URL the route advertised may decide the failure verdict, or
                    # probing alternates would relabel a figureless paper.
                    if candidate == source:
                        failures.append(exc.status)
                    continue
                content_type = _clean(asset.headers.get("content-type"))
                extension = _asset_extension(asset.url, content_type)
                if not extension or not _valid_standalone_asset(asset.content, extension):
                    continue
                destination = folder / "figures" / f"figure_{index:03d}{extension}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(asset.content)
                relative = str(destination.relative_to(batch))
                saved.append(relative)
                candidates.append(
                    {
                        "caption": figure["caption"],
                        "caption_title": figure.get("caption_title", ""),
                        "asset_path": relative,
                        "asset_source_url": asset.url,
                        "asset_sha256": hashlib.sha256(asset.content).hexdigest(),
                        "binding_status": "verified",
                        "binding_method": source_kind,
                    }
                )
                bound_figure = True
                break
            if bound_figure:
                break
    return candidates, saved, failures


def _pdf_fulltext(content: bytes) -> str:
    try:
        import pymupdf

        with pymupdf.open(stream=content, filetype="pdf") as document:
            return _clean(" ".join(page.get_text("text") for page in document))
    except Exception:
        return ""


def bind_official_asset(
    batch: Path,
    domain: str,
    row: dict[str, Any],
    *,
    client: LiteratureHttpClient | None = None,
) -> dict[str, Any]:
    """Download one HTML/JATS-linked figure asset and retain its provenance."""
    canonical_official_url = _clean(row.get("official_url"))
    official_url = _official_probe_url(row)
    host = urlparse(official_url).netloc.casefold() or "unknown"
    client = client or LiteratureHttpClient(
        f"official_asset:{host}",
        min_interval=0.2,
        retries=1,
        timeout=15,
        max_in_flight=2,
        circuit_breaker_threshold=5,
    )
    record_id = _identity(row)
    result: dict[str, Any] = {
        "record_id": record_id,
        "title": row.get("title"),
        "official_url": official_url,
        "identifiers": {
            field: row.get(field)
            for field in ("doi", "paper_id", "official_paper_id", "pmid", "pmcid", "arxiv_id")
            if row.get(field)
        },
        "figure_asset_tier": None,
        "standalone_asset": False,
    }
    if canonical_official_url and canonical_official_url != official_url:
        result["canonical_official_url"] = canonical_official_url
    if _clean(row.get("official_source")).casefold() == "aaai_ojs":
        result.update(
            {
                "status": "no_eligible_standalone_figure",
                "probe_note": (
                    "AAAI OJS exposes the paper-level PDF rather than independent figure assets; "
                    "strict B policy forbids PDF object extraction or cropping."
                ),
            }
        )
        return result
    folder = batch / "assets" / domain / "official" / re.sub(r"[^A-Za-z0-9_.-]+", "_", record_id)[:160]
    candidates: list[dict[str, Any]] = []
    saved: list[str] = []
    asset_failures: list[str] = []
    probe_errors: list[dict[str, str]] = []
    fulltext = ""
    fulltext_source_url = ""
    fulltext_kind = ""
    fetched_any = False
    asset_map: dict[str, str] = {}
    unbound: list[tuple[str, str, list[dict[str, Any]]]] = []
    for route in _figure_routes(row, official_url):
        try:
            response = _request(client, route["url"])
            if route["kind"] == "pmc_jats":
                figures, route_fulltext = parse_jats_figures(
                    response.text, route["figure_base"], asset_alias=route.get("asset_alias")
                )
            else:
                body = response.text[:8_000_000]
                figures = parse_html_figures(body, response.url)
                route_fulltext = parse_html_fulltext(response.text[:12_000_000])
                asset_map.update(parse_html_asset_map(body))
        except ProviderRequestError as exc:
            probe_errors.append({"route": route["kind"], "status": exc.status, "error": str(exc)[:300]})
            continue
        except (ElementTree.ParseError, ValueError) as exc:
            probe_errors.append({"route": route["kind"], "status": "invalid_document", "error": str(exc)[:300]})
            continue
        fetched_any = True
        if len(route_fulltext) > len(fulltext):
            fulltext, fulltext_source_url, fulltext_kind = route_fulltext, response.url, route["kind"]
        bound, bound_paths, route_failures = _bind_route_assets(client, batch, folder, figures, route["kind"])
        asset_failures.extend(route_failures)
        if not bound and figures:
            unbound.append((route["kind"], route["url"], figures))
        if bound:
            candidates, saved = bound, bound_paths
            result["figure_route"] = route["kind"]
            result["figure_route_url"] = route["url"]
            break
    if not candidates and asset_map:
        # A route found the figure but could not fetch it, and a later landing
        # page exposed where that same graphic really lives. Silverchair (OUP)
        # is the case this recovers: its CDN URLs are signed and so cannot be
        # constructed, while the PMC asset host refuses the file outright.
        for kind, url, figures in unbound:
            resolved = _resolve_figure_assets(figures, asset_map)
            if not resolved:
                continue
            bound, bound_paths, route_failures = _bind_route_assets(client, batch, folder, resolved, kind)
            asset_failures.extend(route_failures)
            if bound:
                candidates, saved = bound, bound_paths
                result["figure_route"] = kind
                result["figure_route_url"] = url
                result["figure_asset_recovered_from_landing"] = True
                break
    if not candidates:
        if asset_failures:
            result["status"] = (
                "retryable_rate_limited"
                if "rate_limited" in asset_failures
                else "retryable_circuit_open"
                if "circuit_open" in asset_failures
                else "retryable_failure"
            )
        elif fetched_any:
            # A route did serve the article; the caption gate, not the network,
            # is what rejected it. Never let an earlier route's 404 mask that.
            result["status"] = "no_eligible_standalone_figure"
        elif probe_errors:
            # Every route failed. Classify by the most recoverable verdict so a
            # single burst 403 cannot terminalize an otherwise reachable paper.
            seen = {error["status"] for error in probe_errors}
            result["status"] = (
                "retryable_rate_limited" if "rate_limited" in seen
                else "retryable_circuit_open" if "circuit_open" in seen
                else "retryable_failure" if "failed" in seen
                else "terminal_blocked" if "blocked" in seen
                else "terminal_not_found" if "not_found" in seen
                else "terminal_invalid_document" if "invalid_document" in seen
                else "retryable_failure"
            )
        else:
            result["status"] = "no_eligible_standalone_figure"
        if probe_errors:
            result["probe_error"] = probe_errors[0]["error"]
            result["probe_errors"] = probe_errors
        return result
    result.update(
        {
            "status": "verified",
            "figure_asset_tier": "B_official_standalone",
            "standalone_asset": True,
            "framework_candidates": candidates,
            "figure_assets": saved,
            "asset_source_url": candidates[0]["asset_source_url"],
        }
    )
    paper_pdf: bytes | None = None
    paper_pdf_url = _clean(row.get("official_pdf_url"))
    if len(fulltext) < 5000 and paper_pdf_url:
        try:
            paper_response = _request(client, paper_pdf_url)
            if paper_response.content.startswith(b"%PDF"):
                extracted = _pdf_fulltext(paper_response.content)
                if len(extracted) >= 5000:
                    fulltext = extracted
                    fulltext_source_url = paper_response.url
                    paper_pdf = paper_response.content
        except ProviderRequestError:
            pass
    if len(fulltext) >= 5000:
        text_path = folder / "text" / "full.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(text_path, fulltext)
        if paper_pdf:
            pdf_path = folder / "paper" / "full.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(paper_pdf)
            result["full_pdf_path"] = str(pdf_path.relative_to(batch))
            result["full_pdf_sha256"] = hashlib.sha256(paper_pdf).hexdigest()
        result.update(
            {
                "fulltext_status": "extracted",
                "full_text_path": str(text_path.relative_to(batch)),
                "fulltext_sha256": hashlib.sha256(fulltext.encode("utf-8")).hexdigest(),
                "fulltext_source_url": fulltext_source_url,
                "fulltext_method": "official_pdf_text_no_figure_extraction" if paper_pdf else fulltext_kind,
            }
        )
    return result


def bind_official_assets(
    batch_root: str | Path,
    *,
    year: int = 2026,
    domains: Iterable[str] = DOMAINS,
    domain_workers: int = 5,
    paper_workers: int = 16,
    limit_per_domain: int = 0,
    coordinate_source_probe: bool = False,
) -> dict[str, Any]:
    """Run the resumable B-tier orchestrator without duplicating parser logic."""
    from .official_asset_orchestration import orchestrate_official_assets

    return orchestrate_official_assets(
        batch_root,
        year=year,
        domains=domains,
        domain_workers=domain_workers,
        paper_workers=paper_workers,
        limit_per_domain=limit_per_domain,
        coordinate_source_probe=coordinate_source_probe,
    )


__all__ = [
    "bind_official_asset",
    "bind_official_assets",
    "parse_html_figures",
    "parse_html_fulltext",
    "parse_jats_figures",
]
