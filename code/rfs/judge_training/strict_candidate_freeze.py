"""Promote newly collected A/B candidates into the strict literature view."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any, Iterable

from rfs.utils import write_json

from .literature_freeze import (
    DOMAINS,
    _clean,
    _identity_keys,
    _jsonl,
    _jsonl_text,
    _norm,
    _publish_with_backup,
    build_strict_training_view,
    strict_failures,
)


def _payload(path: Path) -> dict[str, Any]:
    try:
        import json

        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _source_index(batch: Path, domain: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in (
        batch / "manifests" / "source_probe.json",
        batch / "manifests" / f"source_probe_{domain}.json",
    ):
        payload = _payload(path)
        for row in (payload.get("domains", {}).get(domain, {}) or {}).get("rows", []):
            if row.get("arxiv_id"):
                result[str(row["arxiv_id"])] = row
    return result


def _fulltext_index(batch: Path, domain: str) -> dict[str, dict[str, Any]]:
    payload = _payload(batch / "manifests" / "fulltext_probe.json")
    return {
        str(row["arxiv_id"]): row
        for row in (payload.get("domains", {}).get(domain, {}) or {}).get("rows", [])
        if row.get("arxiv_id")
    }


def _public_index(batch: Path, domain: str) -> dict[str, dict[str, Any]]:
    return {
        _clean(row.get("record_id")): row
        for row in _jsonl(batch / "manifests" / f"official_asset_rows_{domain}.jsonl")
        if _clean(row.get("record_id"))
    }


def _ledger_split(row: dict[str, Any]) -> str:
    """Assign the ledger split with the authoritative bucket rule.

    Leaving this unset wrote rows that downstream reporting silently counts as
    training data. The seed and ratios match ``split_ledger`` so the sealed D0
    split, which re-derives from ``paper_id``, agrees with the ledger.
    """
    from .split_ledger import DEFAULT_RATIOS, DEFAULT_SEED, bucket_u64, split_for_bucket, split_thresholds

    identity = _clean(row.get("paper_id")) or _candidate_identity(row)
    return split_for_bucket(bucket_u64(identity, DEFAULT_SEED), split_thresholds(DEFAULT_RATIOS))


def _candidate_identity(row: dict[str, Any]) -> str:
    for field in ("doi", "paper_id", "official_paper_id", "arxiv_id"):
        if value := _clean(row.get(field)):
            return value.casefold()
    return _norm(row.get("title"))


def _source_text_metadata(
    batch: Path,
    domain: str,
    paper: dict[str, Any],
    source_text: dict[str, Any],
    official_asset: dict[str, Any],
) -> dict[str, Any]:
    arxiv_id = _clean(paper.get("arxiv_id"))
    if source_text.get("text_status") == "extracted" and arxiv_id:
        relative = f"assets/{domain}/{arxiv_id}/text/full.txt"
        path = batch / relative
        if path.is_file():
            return {
                "full_text_path": relative,
                "fulltext_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fulltext_source": paper.get("arxiv_url") or f"https://arxiv.org/abs/{arxiv_id}",
            }
    if official_asset.get("fulltext_status") == "extracted" and official_asset.get("full_text_path"):
        return {
            "full_text_path": official_asset["full_text_path"],
            "fulltext_sha256": official_asset.get("fulltext_sha256"),
            "fulltext_source": official_asset.get("fulltext_source_url"),
        }
    if arxiv_id:
        relative = f"assets/{domain}/{arxiv_id}/text/source_text.txt"
        path = batch / relative
        if path.is_file():
            return {
                "source_text_path": relative,
                "fulltext_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fulltext_source": paper.get("arxiv_url") or f"https://arxiv.org/abs/{arxiv_id}",
            }
    return {}


def _new_strict_row(
    batch: Path,
    domain: str,
    paper: dict[str, Any],
    source: dict[str, Any],
    public: dict[str, Any],
    text: dict[str, Any],
) -> dict[str, Any] | None:
    item = dict(paper)
    figure: dict[str, Any] | None = None
    tier: str | None = None
    candidates = [
        candidate
        for candidate in source.get("framework_candidates") or []
        if candidate.get("resolved_files") and candidate.get("pdf_caption_bound")
    ]
    if candidates:
        candidate = candidates[0]
        figure = {
            "caption": candidate.get("caption"),
            "tex_file": candidate.get("tex_file"),
            "source_files": candidate.get("resolved_files"),
            "pdf_page": candidate.get("pdf_page"),
            "image_paths": source.get("extracted_figure_paths") or [],
            "binding_status": "verified",
        }
        tier = "A_source_verified"
    elif public.get("figure_asset_tier") == "B_official_standalone" and public.get("standalone_asset"):
        candidate = next(iter(public.get("framework_candidates") or []), None)
        if candidate:
            figure = {
                **candidate,
                "image_paths": public.get("figure_assets") or [candidate.get("asset_path")],
                "binding_status": "verified",
                "asset_source_url": candidate.get("asset_source_url") or public.get("asset_source_url"),
            }
            tier = "B_official_standalone"
    if not figure or not tier:
        return None
    item.update(
        {
            "figure": figure,
            "figure_asset_tier": tier,
            "standalone_asset": tier == "B_official_standalone",
            "source_code_available": tier == "A_source_verified",
            "record_origin": "strict_multisource_2026",
            "freeze_status": "verified",
            "source_status": "verified",
            "split": _ledger_split(item),
            **_source_text_metadata(batch, domain, paper, text, public),
        }
    )
    return item if not strict_failures(batch, domain, item) else None


def freeze_strict_candidates(
    batch_root: str | Path,
    *,
    year: int = 2026,
    domains: Iterable[str] = DOMAINS,
    target_per_domain: int = 2000,
    domain_workers: int = 5,
    paper_workers: int = 16,
) -> dict[str, Any]:
    """Append only A/B-qualified candidates, then rebuild the strict view."""
    batch = Path(batch_root).resolve()
    selected_domains = [domain for domain in dict.fromkeys(domains) if domain in DOMAINS]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = batch / "manifests" / "backups" / f"strict_append_{stamp}"
    global_keys: set[str] = set()
    for domain in DOMAINS:
        for row in _jsonl(batch / "verified" / domain / "papers.jsonl"):
            global_keys.update(_identity_keys(row))
    added_by_domain: dict[str, int] = {}
    rejected_by_domain: dict[str, int] = {}
    for domain in selected_domains:
        verified_path = batch / "verified" / domain / "papers.jsonl"
        existing = _jsonl(verified_path)
        candidates_path = batch / "candidates" / domain / "papers_arxiv_matched.jsonl"
        candidates = _jsonl(candidates_path) or _jsonl(batch / "candidates" / domain / "papers.jsonl")
        source_index = _source_index(batch, domain)
        public_index = _public_index(batch, domain)
        text_index = _fulltext_index(batch, domain)
        additions: list[dict[str, Any]] = []
        rejected = 0
        for paper in candidates:
            if int(paper.get("year") or 0) != year:
                continue
            keys = _identity_keys(paper)
            if keys and keys & global_keys:
                continue
            arxiv_id = _clean(paper.get("arxiv_id"))
            item = _new_strict_row(
                batch,
                domain,
                paper,
                source_index.get(arxiv_id, {}),
                public_index.get(_candidate_identity(paper), {}),
                text_index.get(arxiv_id, {}),
            )
            if item is None:
                rejected += 1
                continue
            additions.append(item)
            global_keys.update(_identity_keys(item))
        changed = _publish_with_backup(verified_path, _jsonl_text([*existing, *additions]), backup)
        added_by_domain[domain] = len(additions) if changed else 0
        rejected_by_domain[domain] = rejected
    strict_report = build_strict_training_view(
        batch,
        target_per_domain=target_per_domain,
        domain_workers=domain_workers,
        paper_workers=paper_workers,
    )
    report = {
        "version": "strict-candidate-freeze-v001",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "year": year,
        "asset_policy": "strict-ab",
        "added_by_domain": added_by_domain,
        "rejected_by_domain": rejected_by_domain,
        "strict_training": strict_report,
        "backup": str(backup.relative_to(batch)) if backup.exists() else None,
        "ok": True,
    }
    write_json(batch / "manifests" / "strict_candidate_freeze.json", report)
    return report


__all__ = ["freeze_strict_candidates"]
