"""Domain strata for EditableFigureBench.

Every gold instance carries one domain label so the released benchmark can report
per-domain results instead of a single pooled number. The label is a deterministic
join on arXiv metadata: the primary arXiv category decides, and for the broad cs.AI
bucket the remaining categories decide. No model is called, so the labels cost
nothing, never drift between reruns, and a reader can regroup them from the raw
`arxiv_primary_category` / `arxiv_categories` fields stored next to every label.

The five strata are deliberately coarse. This corpus was collected with an
LLM/agent style arXiv query, so language, agent, and general machine learning
carry the mass while vision and the cross-domain bucket stay small; that skew is
reported rather than hidden.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DOMAIN_SCHEMA_VERSION = "editable-figure-bench-domain-v001"

DEFAULT_CATEGORIES_PATH = "experiments/literature_review/papers/papers.csv"

DOMAIN_ORDER: tuple[str, ...] = ("language", "agent", "vision", "general", "other")

DOMAIN_LABELS: dict[str, str] = {
    "language": "Language, retrieval and knowledge",
    "agent": "Agents and sequential decision making",
    "vision": "Vision and multimodal",
    "general": "General machine learning",
    "other": "Safety, cross-domain and other fields",
}

PRIMARY_CATEGORY_DOMAINS: dict[str, str] = {
    "cs.CL": "language",
    "cs.IR": "language",
    "cs.MA": "agent",
    "cs.RO": "agent",
    "cs.CV": "vision",
    "eess.IV": "vision",
    "eess.SP": "vision",
    "cs.LG": "general",
}

BROAD_PRIMARY_CATEGORIES: frozenset[str] = frozenset({"cs.AI"})

SECONDARY_CATEGORY_RULES: tuple[tuple[str, frozenset[str]], ...] = (
    ("agent", frozenset({"cs.MA", "cs.RO"})),
    ("language", frozenset({"cs.CL", "cs.IR"})),
    ("vision", frozenset({"cs.CV", "eess.IV", "eess.SP"})),
)

FALLBACK_DOMAIN = "other"


def normalize_categories(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Split a semicolon separated arXiv category field into unique labels."""
    if value is None:
        return ()
    raw = value.split(";") if isinstance(value, str) else list(value)
    seen: list[str] = []
    for item in raw:
        token = str(item).strip()
        if token and token not in seen:
            seen.append(token)
    return tuple(seen)


def classify_domain(primary_category: str | None, categories: str | Sequence[str] | None) -> str:
    """Map one paper to a domain stratum.

    The primary category wins when it is specific. cs.AI is a catch-all bucket, so
    it is refined by the remaining categories and falls back to general machine
    learning. A paper whose primary category is outside these strata (for example
    cs.CR, q-bio.BM, cond-mat) lands in the cross-domain bucket.
    """
    primary = str(primary_category or "").strip()
    listed = normalize_categories(categories)
    if primary and primary not in BROAD_PRIMARY_CATEGORIES:
        return PRIMARY_CATEGORY_DOMAINS.get(primary, FALLBACK_DOMAIN)
    remaining = frozenset(item for item in listed if item != primary)
    for domain, keys in SECONDARY_CATEGORY_RULES:
        if remaining & keys:
            return domain
    if primary in BROAD_PRIMARY_CATEGORIES:
        return "general"
    return FALLBACK_DOMAIN


def load_arxiv_categories(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read arXiv category metadata keyed by paper id.

    A missing or unreadable file returns an empty index: the caller records the
    domain layer as unavailable instead of inventing labels.
    """
    target = Path(path)
    if not target.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            paper_id = str(row.get("arxiv_id") or "").strip()
            if not paper_id:
                continue
            categories = normalize_categories(row.get("categories"))
            primary = str(row.get("primary_category") or "").strip()
            if not primary and categories:
                primary = categories[0]
            records[paper_id] = {
                "arxiv_primary_category": primary,
                "arxiv_categories": list(categories),
                "domain": classify_domain(primary, categories),
            }
    return records


def domain_fields(paper_id: str, index: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Domain fields for one instance, marked unlabeled when the paper is unknown."""
    record = index.get(str(paper_id))
    if record is None:
        return {
            "domain": None,
            "domain_schema_version": DOMAIN_SCHEMA_VERSION,
            "domain_unlabeled": True,
            "arxiv_primary_category": None,
            "arxiv_categories": [],
        }
    return {
        "domain": str(record.get("domain") or FALLBACK_DOMAIN),
        "domain_schema_version": DOMAIN_SCHEMA_VERSION,
        "domain_unlabeled": False,
        "arxiv_primary_category": str(record.get("arxiv_primary_category") or ""),
        "arxiv_categories": list(record.get("arxiv_categories") or ()),
    }


def summarize_domains(values: Iterable[str | None]) -> dict[str, int]:
    """Count strata in the canonical order, plus any unlabeled instances."""
    counts = {name: 0 for name in DOMAIN_ORDER}
    unlabeled = 0
    for value in values:
        key = str(value or "")
        if key in DOMAIN_LABELS:
            counts[key] += 1
        else:
            unlabeled += 1
    summary = {name: counts[name] for name in DOMAIN_ORDER if counts[name]}
    if unlabeled:
        summary["unlabeled"] = unlabeled
    return summary
