"""Canonical evidence for one paper's formal publisher record.

The strict literature gate decides from the ledger alone whether a discovered
paper has a verifiable formal publication record.  Three kinds of evidence are
accepted: a publisher landing page carrying the title and venue, a publisher
PDF whose text carries the title, and the exact ACM main-proceedings
registration that ACM deposits for its own DOI.

The third kind exists because `dl.acm.org` answers every automated client
with a Cloudflare challenge, so ACM main proceedings cannot be verified from
the landing page or from the publisher PDF.  The rules live here so the
collector that records the evidence and the strict gate that accepts it cannot
disagree: the gate rebuilds the expected evidence from the row's own
identifiers and requires the recorded copy to match it exactly.  The collector
stays responsible for reading that record from Crossref.
"""

from __future__ import annotations

import re
from typing import Any


ACM_EVIDENCE_METHOD = "crossref_acm_main_proceedings"
ACM_MIN_PAGE_SPAN = 7
ACM_PAGE_SPAN_FIELD = "page_span"
LANDING_REGISTRY_MARKERS = ("doi.org/", "api.crossref.org", "dblp.org")
ACM_MAIN_PROCEEDINGS: dict[str, frozenset[str]] = {
    "WWW": frozenset({"Proceedings of the ACM Web Conference 2026"}),
    "SIGIR": frozenset(
        {
            "Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval"
        }
    ),
    "KDD": frozenset(
        {
            "Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1",
            "Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2",
        }
    ),
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _page_span(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def acm_main_proceedings_evidence(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return exact ACM main-proceedings registration evidence for one row.

    ACM landing pages frequently return 403 or terminate TLS connections in
    automated collection.  The candidate collector already preserves the
    Crossref record registered by ACM, including the proceedings container,
    article type, and page span.  This fallback is intentionally exact: it
    excludes companion proceedings, ICTIR, workshops, and short papers.
    """
    venue = _clean(row.get("venue"))
    allowed = ACM_MAIN_PROCEEDINGS.get(venue)
    evidence = row.get("paper_type_evidence") or {}
    doi = _clean(row.get("doi")).casefold()
    official_url = _clean(row.get("official_url")).casefold()
    container = _clean(evidence.get("crossref_container"))
    page_span = _page_span(evidence.get(ACM_PAGE_SPAN_FIELD))
    if not allowed or _clean(row.get("official_source")) != "acm_proceedings":
        return None
    if not doi.startswith("10.1145/") or official_url != f"https://doi.org/{doi}":
        return None
    if _clean(row.get("paper_type")).casefold() != "full":
        return None
    if evidence.get("crossref_type") != "proceedings-article" or container not in allowed:
        return None
    if page_span < ACM_MIN_PAGE_SPAN:
        return None
    return {
        "method": ACM_EVIDENCE_METHOD,
        "doi": doi,
        "container": container,
        ACM_PAGE_SPAN_FIELD: page_span,
    }


def acm_main_proceedings_proof(row: dict[str, Any]) -> bool:
    """Return whether the row records the ACM evidence its own identifiers imply."""
    expected = acm_main_proceedings_evidence(row)
    if expected is None:
        return False
    recorded = row.get("official_record_evidence") or {}
    return (
        _clean(recorded.get("method")) == expected["method"]
        and _clean(recorded.get("doi")).casefold() == expected["doi"]
        and _clean(recorded.get("container")) == expected["container"]
        and _page_span(recorded.get(ACM_PAGE_SPAN_FIELD)) == expected[ACM_PAGE_SPAN_FIELD]
    )


def publisher_proof_ok(row: dict[str, Any]) -> bool:
    """Return whether the row carries accepted evidence of its formal publisher record."""
    landing = row.get("publisher_landing_evidence") or {}
    resolved = _clean(landing.get("resolved_url")).casefold()
    landing_ok = bool(
        landing.get("title_found")
        and landing.get("venue_found")
        and resolved
        and not any(marker in resolved for marker in LANDING_REGISTRY_MARKERS)
    )
    pdf_ok = _clean((row.get("official_pdf_evidence") or {}).get("method")) == "official_pdf_text"
    return landing_ok or pdf_ok or acm_main_proceedings_proof(row)


__all__ = [
    "ACM_EVIDENCE_METHOD",
    "ACM_MAIN_PROCEEDINGS",
    "ACM_MIN_PAGE_SPAN",
    "acm_main_proceedings_evidence",
    "acm_main_proceedings_proof",
    "publisher_proof_ok",
]
