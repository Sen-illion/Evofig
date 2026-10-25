"""Exact arXiv metadata indexing and bounded batch lookup primitives."""

from __future__ import annotations

from collections import defaultdict
import html
import re
import time
from typing import Any, Iterable
from xml.etree import ElementTree

from .literature import (
    ARXIV_NS,
    ArxivProvider,
    ProviderRequestError,
    clean,
    literature_session,
    normalized_doi,
    normalized_title,
)


_ARXIV_ID_PATTERN = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf|e-print)/|10\.48550/arxiv\.)(\d{4}\.\d{4,5})(?:v\d+)?",
    re.IGNORECASE,
)


def extract_arxiv_id(value: Any) -> str:
    """Return one modern arXiv identifier embedded in a DOI or URL."""
    match = _ARXIV_ID_PATTERN.search(str(value or ""))
    return match.group(1) if match else ""


def normalize_arxiv_record(value: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one cached/provider record into the strict source-link shape."""
    arxiv_id = clean(value.get("arxiv_id"))
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id) or extract_arxiv_id(value.get("doi"))
    if not arxiv_id:
        for field in ("arxiv_url", "pdf_url", "source_url", "id", "url"):
            arxiv_id = extract_arxiv_id(value.get(field))
            if arxiv_id:
                break
    title = clean(value.get("title"))
    if not arxiv_id or not title:
        return None
    raw_version = clean(value.get("arxiv_version"))
    version = raw_version if re.fullmatch(r"v\d+", raw_version) else "v1"
    return {
        "arxiv_id": arxiv_id,
        "arxiv_version": version,
        "title": title,
        "title_norm": normalized_title(title),
        "doi": normalized_doi(value.get("doi")) or None,
        "journal_ref": clean(value.get("journal_ref")) or None,
        "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://export.arxiv.org/pdf/{arxiv_id}",
        "source_url": f"https://export.arxiv.org/e-print/{arxiv_id}",
    }


class ArxivMetadataIndex:
    """Immutable exact DOI/title index that rejects ambiguous mappings."""

    def __init__(self, records: Iterable[dict[str, Any]]) -> None:
        by_id: dict[str, dict[str, Any]] = {}
        for raw in records:
            normalized = normalize_arxiv_record(raw)
            if normalized:
                by_id.setdefault(normalized["arxiv_id"], normalized)
        self.records = tuple(by_id.values())
        title_ids: dict[str, set[str]] = defaultdict(set)
        doi_ids: dict[str, set[str]] = defaultdict(set)
        for record in self.records:
            title_ids[record["title_norm"]].add(record["arxiv_id"])
            if record.get("doi"):
                doi_ids[str(record["doi"])].add(record["arxiv_id"])
        self._by_id = by_id
        self._title_ids = dict(title_ids)
        self._doi_ids = dict(doi_ids)

    def match(self, *, title: Any, doi: Any = None) -> tuple[dict[str, Any] | None, str]:
        """Match DOI first and then normalized title, requiring one arXiv ID."""
        doi_key = normalized_doi(doi)
        if doi_key:
            ids = self._doi_ids.get(doi_key, set())
            if len(ids) == 1:
                return self._by_id[next(iter(ids))], "exact_doi_cache"
            arxiv_id = extract_arxiv_id(doi_key)
            if arxiv_id in self._by_id:
                return self._by_id[arxiv_id], "exact_arxiv_doi_cache"
        title_key = normalized_title(title)
        ids = self._title_ids.get(title_key, set())
        if len(ids) == 1:
            return self._by_id[next(iter(ids))], "exact_title_cache"
        return None, "ambiguous_title_cache" if len(ids) > 1 else "no_exact_cache_match"


def parse_arxiv_atom(text: str) -> list[dict[str, Any]]:
    """Parse arXiv Atom entries without relaxing the exact-match policy."""
    root = ElementTree.fromstring(text)
    records: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", ARXIV_NS):
        raw_id = clean(entry.findtext("a:id", default="", namespaces=ARXIV_NS))
        title = clean(entry.findtext("a:title", default="", namespaces=ARXIV_NS))
        arxiv_id = extract_arxiv_id(raw_id)
        if not arxiv_id or not title:
            continue
        version_match = re.search(r"(v\d+)$", raw_id)
        records.append(
            {
                "arxiv_id": arxiv_id,
                "arxiv_version": version_match.group(1) if version_match else "v1",
                "title": title,
                "doi": entry.findtext("arxiv:doi", default="", namespaces=ARXIV_NS),
                "journal_ref": entry.findtext("arxiv:journal_ref", default="", namespaces=ARXIV_NS),
            }
        )
    return records


def batch_match_exact(
    provider: ArxivProvider,
    titles: Iterable[str],
    *,
    batch_size: int = 20,
) -> dict[str, dict[str, Any]]:
    """Resolve a small title tail with one request per bounded title batch."""
    values = [clean(title) for title in titles if clean(title)]
    output: dict[str, dict[str, Any]] = {}
    size = max(1, min(int(batch_size), 20))
    for offset in range(0, len(values), size):
        group = values[offset : offset + size]
        query = " OR ".join(f'ti:"{title.replace(chr(34), "")}"' for title in group)
        response = provider.client.request(
            "https://export.arxiv.org/api/query",
            params={
                "search_query": query,
                "start": 0,
                "max_results": min(100, max(10, len(group) * 5)),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )
        index = ArxivMetadataIndex(parse_arxiv_atom(response.text))
        for title in group:
            record, _method = index.match(title=title)
            if record:
                output[normalized_title(title)] = record
    return output


SEMANTIC_SCHOLAR_MATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search/match"
ARXIV_WEB_SEARCH_URL = "https://arxiv.org/search/"
_SEMANTIC_SCHOLAR_FIELDS = "title,year,externalIds"
_MODERN_ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}")
_ARXIV_RESULT_BLOCK = '<li class="arxiv-result">'
_ARXIV_RESULT_ID = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})")
_ARXIV_RESULT_TITLE = re.compile(r'<p class="title is-5 mathjax">(.*?)</p>', re.DOTALL)


def modern_arxiv_id(value: Any) -> str:
    """Return a bare modern arXiv identifier from a bare id or prefixed form."""
    text = re.sub(r"^arxiv:", "", clean(value), flags=re.IGNORECASE)
    text = re.sub(r"v\d+$", "", text)
    return text if _MODERN_ARXIV_ID.fullmatch(text) else ""


class ArxivTitleResolver:
    """Resolve exact paper titles without the export API's title query.

    On 2026-09-25 export.arxiv.org answered HTTP 406 to every `ti:` and quoted
    `search_query` sent from this host while bare `all:`/`cat:` queries still
    worked, which silently reduced live title matching to nothing. Two public
    channels stand in for it, in order:

    1. Semantic Scholar's exact title match endpoint.
    2. The public arXiv search page.

    A candidate is accepted only when its own title equals the requested title
    after normalization, so a fuzzy upstream hit can never become a binding.
    """

    def __init__(
        self,
        *,
        session: Any | None = None,
        min_interval: float = 2.0,
        web_min_interval: float = 3.0,
        retries: int = 2,
        backoff: float = 2.5,
        timeout: int = 45,
        web_search: bool = True,
    ) -> None:
        self.session = session if session is not None else literature_session()
        self.min_interval = max(0.0, float(min_interval))
        self.web_min_interval = max(0.0, float(web_min_interval))
        self.retries = max(0, int(retries))
        self.backoff = max(0.0, float(backoff))
        self.timeout = max(5, int(timeout))
        self.web_search = bool(web_search)
        self._last_request = 0.0

    def resolve(self, title: Any) -> tuple[dict[str, Any] | None, str]:
        """Return one arXiv record plus the channel that produced it."""
        target = normalized_title(title)
        if not target:
            return None, "empty_title"
        record = self._semantic_scholar(str(title), target)
        if record:
            return record, "exact_title_semantic_scholar"
        if self.web_search:
            record = self._arxiv_search_page(str(title), target)
            if record:
                return record, "exact_title_arxiv_web_search"
        return None, "no_live_title_match"

    def _wait(self, interval: float) -> None:
        if interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request
        if elapsed < interval:
            time.sleep(interval - elapsed)

    def _get(self, url: str, params: dict[str, Any], interval: float) -> Any:
        """One paced GET; None means the channel has nothing for this request."""
        for attempt in range(self.retries + 1):
            self._wait(interval)
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except Exception as exc:  # requests raises several unrelated transports
                self._last_request = time.monotonic()
                if attempt >= self.retries:
                    raise ProviderRequestError("arxiv_title_resolver", "transport", str(exc)[:200]) from exc
                time.sleep(self.backoff * (attempt + 1))
                continue
            self._last_request = time.monotonic()
            if response.status_code == 404:
                return None
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt >= self.retries:
                    raise ProviderRequestError(
                        "arxiv_title_resolver",
                        f"http_{response.status_code}",
                        f"{url} answered HTTP {response.status_code} after {self.retries + 1} attempts",
                    )
                time.sleep(self.backoff * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        return None

    def _semantic_scholar(self, title: str, target: str) -> dict[str, Any] | None:
        response = self._get(
            SEMANTIC_SCHOLAR_MATCH_URL,
            {"query": title, "fields": _SEMANTIC_SCHOLAR_FIELDS},
            self.min_interval,
        )
        if response is None:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        candidates = payload.get("data") if isinstance(payload, dict) else None
        for candidate in candidates or []:
            if not isinstance(candidate, dict) or normalized_title(candidate.get("title")) != target:
                continue
            external = candidate.get("externalIds") or {}
            arxiv_id = modern_arxiv_id(external.get("ArXiv")) or extract_arxiv_id(external.get("DOI"))
            if not arxiv_id:
                return None
            doi = clean(external.get("DOI"))
            if extract_arxiv_id(doi) == arxiv_id:
                doi = ""
            return normalize_arxiv_record(
                {"arxiv_id": arxiv_id, "title": candidate.get("title"), "doi": doi}
            )
        return None

    def _arxiv_search_page(self, title: str, target: str) -> dict[str, Any] | None:
        response = self._get(
            ARXIV_WEB_SEARCH_URL,
            {"searchtype": "all", "query": f'"{title.replace(chr(34), "")}"'},
            self.web_min_interval,
        )
        if response is None:
            return None
        for block in response.text.split(_ARXIV_RESULT_BLOCK)[1:]:
            id_match = _ARXIV_RESULT_ID.search(block)
            title_match = _ARXIV_RESULT_TITLE.search(block)
            if not id_match or not title_match:
                continue
            found = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", title_match.group(1)))).strip()
            if normalized_title(found) != target:
                continue
            return normalize_arxiv_record({"arxiv_id": id_match.group(1), "title": found})
        return None


__all__ = [
    "ArxivMetadataIndex",
    "ArxivTitleResolver",
    "batch_match_exact",
    "extract_arxiv_id",
    "modern_arxiv_id",
    "normalize_arxiv_record",
    "parse_arxiv_atom",
]
