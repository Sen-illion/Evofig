"""Rate-limited literature metadata providers for Judge corpus collection."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
import threading
import time
from typing import Any, Iterable
from xml.etree import ElementTree

import requests

from .rate_limit import serialized_request


PROVIDER_STATUSES = {
    "complete",
    "not_yet_published",
    "rate_limited",
    "blocked",
    "failed",
}
ARXIV_NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(value or ""))).strip()


def normalized_title(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean(value).casefold()).strip()


def normalized_doi(value: Any) -> str:
    return re.sub(r"^https?://doi\.org/", "", clean(value).casefold()).rstrip(".")


@dataclass(slots=True)
class ProviderPage:
    """One resumable provider page with a stable status vocabulary."""

    provider: str
    records: list[dict[str, Any]] = field(default_factory=list)
    status: str = "complete"
    next_cursor: str | None = None
    request: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in PROVIDER_STATUSES:
            raise ValueError(f"unsupported provider status: {self.status}")


class ProviderRequestError(RuntimeError):
    """A classified provider request failure."""

    def __init__(self, provider: str, status: str, message: str) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status


_CIRCUITS: dict[str, dict[str, float | int]] = {}
_CIRCUITS_LOCK = threading.Lock()


LITERATURE_USER_AGENT = "PaperImage-LiteratureCollector/1.0 (mailto:paperimage@example.com)"


def trusts_system_proxy() -> bool:
    """Whether literature traffic should adopt the environment's proxy."""
    return os.environ.get("RFS_LITERATURE_TRUST_ENV", "").strip() == "1"


def literature_session(user_agent: str | None = None) -> requests.Session:
    """Build a session that ignores the system proxy unless opted in.

    Every literature source is a public academic host reachable directly. A
    desktop VPN publishes itself as the Windows system proxy, which requests
    adopts by default; routing through it breaks the TLS handshake for
    api.crossref.org and journals.plos.org while helping no host at all.
    """
    session = requests.Session()
    session.trust_env = trusts_system_proxy()
    # Never leave the requests default in place: publishers 403 it.
    session.headers["User-Agent"] = user_agent or LITERATURE_USER_AGENT
    return session


def reset_literature_circuit_state() -> None:
    """Clear provider circuit state for tests and isolated canaries."""
    with _CIRCUITS_LOCK:
        _CIRCUITS.clear()


class LiteratureHttpClient:
    """HTTP client with one independent Retry-After-aware lane per provider."""

    def __init__(
        self,
        provider: str,
        *,
        min_interval: float = 0.25,
        retries: int = 2,
        timeout: int = 45,
        session: requests.Session | None = None,
        max_in_flight: int = 1,
        circuit_breaker_threshold: int = 0,
        circuit_breaker_cooldown: float = 300.0,
    ) -> None:
        self.provider = provider
        self.min_interval = max(0.0, min_interval)
        self.retries = max(0, retries)
        self.timeout = max(5, timeout)
        self.max_in_flight = max(1, int(max_in_flight))
        self.circuit_breaker_threshold = max(0, int(circuit_breaker_threshold))
        self.circuit_breaker_cooldown = max(0.0, float(circuit_breaker_cooldown))
        self.session = session or requests.Session()
        # See literature_session(): the system proxy is opt-in, not default.
        self.session.trust_env = trusts_system_proxy()
        # Assign, never setdefault: requests.Session() ships a default
        # "python-requests/x.y" User-Agent, so setdefault silently kept it and
        # publishers answered 403 to the whole crawl. A caller that set its own
        # identifying agent keeps it.
        agent = str(self.session.headers.get("User-Agent") or "")
        if not agent or agent.startswith("python-requests"):
            self.session.headers["User-Agent"] = LITERATURE_USER_AGENT

    def _before_request(self) -> None:
        if not self.circuit_breaker_threshold:
            return
        with _CIRCUITS_LOCK:
            state = _CIRCUITS.get(self.provider) or {}
            if float(state.get("open_until") or 0.0) > time.monotonic():
                raise ProviderRequestError(self.provider, "circuit_open", "provider circuit is cooling down")

    def _record_success(self) -> None:
        if self.circuit_breaker_threshold:
            with _CIRCUITS_LOCK:
                _CIRCUITS[self.provider] = {"failures": 0, "open_until": 0.0}

    def _record_failure(self) -> None:
        if not self.circuit_breaker_threshold:
            return
        with _CIRCUITS_LOCK:
            state = _CIRCUITS.setdefault(self.provider, {"failures": 0, "open_until": 0.0})
            failures = int(state.get("failures") or 0) + 1
            state["failures"] = failures
            if failures >= self.circuit_breaker_threshold:
                state["open_until"] = time.monotonic() + self.circuit_breaker_cooldown

    def request(self, url: str, *, params: dict[str, Any] | None = None) -> requests.Response:
        self._before_request()
        try:
            response, _attempts = serialized_request(
                lambda: self.session.get(url, params=params, timeout=self.timeout, allow_redirects=True),
                lane=f"literature:{self.provider}",
                retries=self.retries,
                min_interval=self.min_interval,
                base_delay=1.0,
                max_delay=30.0,
                max_in_flight=self.max_in_flight,
                retry_exceptions=(requests.RequestException,),
            )
        except requests.RequestException as exc:
            self._record_failure()
            raise ProviderRequestError(self.provider, "failed", str(exc)) from exc
        status_code = int(getattr(response, "status_code", 0) or 0)
        if status_code == 429:
            self._record_failure()
            raise ProviderRequestError(self.provider, "rate_limited", "provider returned HTTP 429")
        if status_code in {401, 403, 451}:
            # Access-policy responses are terminal for this URL, but they do
            # not establish that the host is unhealthy. Counting a 403 here
            # used to open the shared host circuit and skip hundreds of
            # unrelated public records without issuing their requests.
            raise ProviderRequestError(self.provider, "blocked", f"provider returned HTTP {status_code}")
        if status_code in {404, 410}:
            raise ProviderRequestError(self.provider, "not_found", f"provider returned HTTP {status_code}")
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            self._record_failure()
            raise ProviderRequestError(self.provider, "failed", str(exc)) from exc
        self._record_success()
        return response

    def json(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.request(url, params=params)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderRequestError(
                self.provider,
                "failed",
                f"provider returned invalid JSON from {response.url}",
            ) from exc
        if not isinstance(payload, dict):
            raise ProviderRequestError(self.provider, "failed", "provider JSON root is not an object")
        return dict(payload)


class CrossrefProvider:
    """Discover DOI-deposited proceedings and journal candidates."""

    name = "crossref"

    def __init__(self, client: LiteratureHttpClient | None = None) -> None:
        self.client = client or LiteratureHttpClient(self.name, min_interval=0.1)

    def list_works(
        self,
        *,
        year: int,
        container: str,
        cursor: str = "*",
        rows: int = 200,
        prefix: str | None = None,
    ) -> ProviderPage:
        filters = [f"from-pub-date:{year}-01-01", f"until-pub-date:{year}-12-31"]
        if normalized_doi(prefix):
            filters.append(f"prefix:{normalized_doi(prefix)}")
        request = {
            "filter": ",".join(filters),
            "query.container-title": container,
            "rows": min(1000, max(1, rows)),
            "cursor": cursor,
            "select": "DOI,title,author,published,container-title,URL,link,type,page",
        }
        payload = self.client.json("https://api.crossref.org/works", params=request)
        message = payload.get("message") or {}
        records: list[dict[str, Any]] = []
        for item in message.get("items") or []:
            title = clean((item.get("title") or [""])[0])
            doi = normalized_doi(item.get("DOI"))
            if not title or not doi:
                continue
            date_parts = ((item.get("published") or {}).get("date-parts") or [[]])[0]
            records.append(
                {
                    "title": title,
                    "title_norm": normalized_title(title),
                    "doi": doi,
                    "official_url": clean(item.get("URL")) or f"https://doi.org/{doi}",
                    "publication_date": "-".join(str(value).zfill(2) for value in date_parts[:3]),
                    "venue": clean((item.get("container-title") or [container])[0]),
                    "record_type": clean(item.get("type")),
                    "page": clean(item.get("page")),
                    "provider": self.name,
                    "official_proof": False,
                    "raw_evidence": {"requested_container": container},
                }
            )
        return ProviderPage(
            provider=self.name,
            records=records,
            next_cursor=clean(message.get("next-cursor")) or None,
            request=request,
            evidence={
                "deposited_by_publishers": True,
                "requires_official_verification": True,
                "raw_item_count": len(message.get("items") or []),
                "total_results": int(message.get("total-results") or 0),
            },
        )

    def iter_works(
        self,
        *,
        year: int,
        container: str,
        rows: int = 1000,
        prefix: str | None = None,
    ) -> Iterable[ProviderPage]:
        """Yield every Crossref cursor page exactly once."""
        cursor = "*"
        seen: set[str] = set()
        fetched_raw = 0
        page_size = min(1000, max(1, rows))
        while cursor not in seen:
            seen.add(cursor)
            page = self.list_works(
                year=year,
                container=container,
                cursor=cursor,
                rows=page_size,
                prefix=prefix,
            )
            yield page
            next_cursor = page.next_cursor
            raw_count = int(page.evidence.get("raw_item_count") or 0)
            fetched_raw += raw_count
            total_results = int(page.evidence.get("total_results") or 0)
            if not next_cursor or next_cursor in seen or raw_count < page_size or (total_results and fetched_raw >= total_results):
                break
            cursor = next_cursor


class EuropePmcProvider:
    """Discover open biology research articles from Europe PMC."""

    name = "europe_pmc"

    def __init__(self, client: LiteratureHttpClient | None = None) -> None:
        self.client = client or LiteratureHttpClient(self.name, min_interval=0.15)

    def list_works(
        self,
        *,
        year: int,
        journals: Iterable[str],
        cursor: str = "*",
        rows: int = 1000,
        window: tuple[str, str] | None = None,
    ) -> ProviderPage:
        journal_query = " OR ".join(f'JOURNAL:"{journal}"' for journal in journals)
        lower, upper = window or (f"{year}-01-01", f"{year}-12-31")
        query = f"OPEN_ACCESS:Y AND FIRST_PDATE:[{lower} TO {upper}] AND ({journal_query})"
        request = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": min(1000, max(1, rows)),
            "cursorMark": cursor,
        }
        payload = self.client.json(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=request
        )
        results = (payload.get("resultList") or {}).get("result") or []
        records = []
        for item in results:
            title = clean(item.get("title"))
            if not title:
                continue
            pmid = clean(item.get("pmid") or item.get("id"))
            pmcid = clean(item.get("pmcid"))
            records.append(
                {
                    "title": title,
                    "title_norm": normalized_title(title),
                    "doi": normalized_doi(item.get("doi")),
                    "pmid": pmid or None,
                    "pmcid": pmcid or None,
                    "official_url": f"https://europepmc.org/article/MED/{pmid}" if pmid else None,
                    "publication_date": clean(item.get("firstPublicationDate") or item.get("firstIndexDate"))[:10] or None,
                    # Europe PMC nests the title; the flat key never existed, so
                    # every harvested biology row carried an empty venue.
                    "venue": clean(
                        ((item.get("journalInfo") or {}).get("journal") or {}).get("title")
                        or item.get("journalTitle")
                    ),
                    "abstract": item.get("abstractText"),
                    "record_type": clean(item.get("pubType")),
                    "provider": self.name,
                    "official_proof": bool(pmid or pmcid),
                }
            )
        return ProviderPage(
            provider=self.name,
            records=records,
            next_cursor=clean(payload.get("nextCursorMark")) or None,
            request=request,
            evidence={
                "open_access_query": True,
                "hit_count": int(payload.get("hitCount") or 0),
                "raw_item_count": len(results),
            },
        )

    def iter_works(
        self,
        *,
        year: int,
        journals: Iterable[str],
        rows: int = 1000,
        max_records: int = 0,
        window: tuple[str, str] | None = None,
    ) -> Iterable[ProviderPage]:
        """Yield every Europe PMC cursor page exactly once.

        Without this driver the caller issues a single request and silently
        keeps one page of a result set that runs to five figures.
        """
        journals = list(journals)
        cursor = "*"
        seen: set[str] = set()
        fetched = 0
        page_size = min(1000, max(1, rows))
        while cursor not in seen:
            seen.add(cursor)
            page = self.list_works(
                year=year, journals=journals, cursor=cursor, rows=page_size, window=window
            )
            yield page
            raw_count = int(page.evidence.get("raw_item_count") or 0)
            fetched += raw_count
            hit_count = int(page.evidence.get("hit_count") or 0)
            next_cursor = page.next_cursor
            if (
                not next_cursor
                or next_cursor in seen
                or raw_count == 0
                or (hit_count and fetched >= hit_count)
                or (max_records and fetched >= max_records)
            ):
                break
            cursor = next_cursor


class DblpProvider:
    """Supplement conference metadata without granting publication proof."""

    name = "dblp"

    def __init__(self, client: LiteratureHttpClient | None = None) -> None:
        # dblp answers 429 to a steady crawl and publishes no quota, so pace it
        # well under one request per second and let the client's own backoff
        # absorb the bursts instead of failing the whole venue.
        self.client = client or LiteratureHttpClient(self.name, min_interval=1.0, retries=4)

    def search(self, query: str, *, offset: int = 0, rows: int = 1000) -> ProviderPage:
        request = {"q": query, "format": "json", "f": offset, "h": min(1000, rows)}
        payload = self.client.json("https://dblp.org/search/publ/api", params=request)
        hits = ((payload.get("result") or {}).get("hits") or {}).get("hit") or []
        records = []
        for hit in hits:
            info = hit.get("info") or {}
            title = clean(info.get("title"))
            if title:
                records.append(
                    {
                        "title": title,
                        "title_norm": normalized_title(title),
                        "doi": normalized_doi(info.get("doi")),
                        "official_url": clean(info.get("url")) or None,
                        "publication_date": clean(info.get("year")),
                        "venue": clean(info.get("venue")),
                        "record_type": clean(info.get("type")),
                        "provider": self.name,
                        "official_proof": False,
                    }
                )
        total = int(clean(((payload.get("result") or {}).get("hits") or {}).get("@total")) or 0)
        next_offset = offset + len(records)
        return ProviderPage(
            provider=self.name,
            records=records,
            next_cursor=str(next_offset) if records and next_offset < total else None,
            request=request,
            evidence={"requires_official_verification": True},
        )


class SemanticScholarProvider:
    """Optionally enrich one exact DOI/title candidate with public links."""

    name = "semantic_scholar"

    def __init__(self, client: LiteratureHttpClient | None = None) -> None:
        self.client = client or LiteratureHttpClient(self.name, min_interval=1.0)
        token = clean(os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
        if token:
            self.client.session.headers["x-api-key"] = token

    def enrich_doi(self, doi: str) -> dict[str, Any]:
        key = normalized_doi(doi)
        payload = self.client.json(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{key}",
            params={"fields": "title,authors,year,venue,externalIds,openAccessPdf,url"},
        )
        payload["provider"] = self.name
        payload["official_proof"] = False
        return payload


class UnpaywallProvider:
    """Locate a legal open copy for a DOI without granting publication proof."""

    name = "unpaywall"

    def __init__(self, client: LiteratureHttpClient | None = None, *, email: str | None = None) -> None:
        self.client = client or LiteratureHttpClient(self.name, min_interval=0.25)
        self.email = clean(email or os.getenv("UNPAYWALL_EMAIL") or "paperimage@example.com")

    def enrich_doi(self, doi: str) -> dict[str, Any]:
        payload = self.client.json(
            f"https://api.unpaywall.org/v2/{normalized_doi(doi)}", params={"email": self.email}
        )
        best = payload.get("best_oa_location") or {}
        return {
            "provider": self.name,
            "official_proof": False,
            "is_oa": bool(payload.get("is_oa")),
            "license": clean(best.get("license")) or None,
            "landing_url": clean(best.get("url_for_landing_page")) or None,
            "pdf_url": clean(best.get("url_for_pdf")) or None,
        }


class ArxivProvider:
    """Resolve one exact arXiv counterpart by DOI or normalized title."""

    name = "arxiv"

    def __init__(self, client: LiteratureHttpClient | None = None) -> None:
        self.client = client or LiteratureHttpClient(self.name, min_interval=3.0)

    def match_exact(self, *, title: str, doi: str | None = None) -> list[dict[str, Any]]:
        query = f'doi:"{normalized_doi(doi)}"' if normalized_doi(doi) else f'ti:"{clean(title)}"'
        response = self.client.request(
            "https://export.arxiv.org/api/query",
            params={"search_query": query, "start": 0, "max_results": 10},
        )
        root = ElementTree.fromstring(response.text)
        records: dict[str, dict[str, Any]] = {}
        target = normalized_title(title)
        for entry in root.findall("a:entry", ARXIV_NS):
            entry_title = clean(entry.findtext("a:title", default="", namespaces=ARXIV_NS))
            if normalized_title(entry_title) != target:
                continue
            raw_id = clean(entry.findtext("a:id", default="", namespaces=ARXIV_NS)).rsplit("/", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", raw_id)
            if arxiv_id:
                records[arxiv_id] = {
                    "arxiv_id": arxiv_id,
                    "title": entry_title,
                    "title_norm": target,
                    "doi": normalized_doi(entry.findtext("arxiv:doi", default="", namespaces=ARXIV_NS)),
                    "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                    "pdf_url": f"https://export.arxiv.org/pdf/{arxiv_id}",
                    "source_url": f"https://export.arxiv.org/e-print/{arxiv_id}",
                    "provider": self.name,
                    "official_proof": False,
                }
        return list(records.values()) if len(records) == 1 else []


__all__ = [
    "ArxivProvider",
    "CrossrefProvider",
    "DblpProvider",
    "EuropePmcProvider",
    "LiteratureHttpClient",
    "PROVIDER_STATUSES",
    "ProviderPage",
    "ProviderRequestError",
    "SemanticScholarProvider",
    "UnpaywallProvider",
    "normalized_doi",
    "normalized_title",
    "reset_literature_circuit_state",
]
