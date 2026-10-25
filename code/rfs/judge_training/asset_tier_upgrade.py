"""Promote ledger rows that now have a strict A/B asset on disk.

Every freeze path skips a paper that is already in a ledger before it looks at
that paper's assets, so a row frozen as ``C_pdf_object_extracted`` stays at that
tier forever even once the official-asset probe has bound a real publisher
figure for it. This module is the missing consumer: it rewrites only the figure
block and tier of rows whose promotion the current gates already accept, and it
refuses to touch anything else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .literature_freeze import DOMAINS, _publish_with_backup, strict_failures
from .literature_io import atomic_write_text


PROMOTABLE_FROM = {"C_pdf_object_extracted", None, ""}


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def _asset_index(batch: Path, domain: str) -> dict[str, dict[str, Any]]:
    """Index verified B-tier probe results by DOI."""
    index: dict[str, dict[str, Any]] = {}
    for row in _jsonl(batch / "manifests" / f"official_asset_rows_{domain}.jsonl"):
        if row.get("figure_asset_tier") != "B_official_standalone" or not row.get("standalone_asset"):
            continue
        doi = _clean((row.get("identifiers") or {}).get("doi")).casefold()
        if doi:
            index[doi] = row
    return index


def _promoted(row: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any] | None:
    """Return the row with its figure rebound to the official standalone asset."""
    candidate = next(iter(probe.get("framework_candidates") or []), None)
    if not candidate:
        return None
    figure = {
        **candidate,
        "image_paths": probe.get("figure_assets") or [candidate.get("asset_path")],
        "binding_status": "verified",
        "asset_source_url": candidate.get("asset_source_url") or probe.get("asset_source_url"),
    }
    item = dict(row)
    item.update(
        {
            "figure": figure,
            "figure_asset_tier": "B_official_standalone",
            "standalone_asset": True,
            "source_code_available": False,
            "asset_tier_upgrade": {
                "from": row.get("figure_asset_tier"),
                "to": "B_official_standalone",
                "asset_source_url": figure.get("asset_source_url"),
            },
        }
    )
    for field in ("full_text_path", "fulltext_sha256", "fulltext_source_url"):
        if probe.get(field) and not item.get(field):
            item[field] = probe[field]
    return item


def upgrade_asset_tiers(
    batch_root: str | Path,
    *,
    domains: Iterable[str] = DOMAINS,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Promote every ledger row whose promotion the strict gates already accept."""
    batch = Path(batch_root).resolve()
    selected = [value for value in dict.fromkeys(domains) if value in DOMAINS]
    stamp = __import__("time").strftime("%Y%m%dT%H%M%SZ", __import__("time").gmtime())
    backup = batch / "manifests" / "backups" / f"asset_tier_upgrade_{stamp}"
    report: dict[str, Any] = {
        "version": "asset-tier-upgrade-v001",
        "generated_at": stamp,
        "dry_run": dry_run,
        "domains": {},
    }
    changed_any = False
    for domain in selected:
        path = batch / "verified" / domain / "papers.jsonl"
        rows = _jsonl(path)
        index = _asset_index(batch, domain)
        upgraded = 0
        refused: dict[str, int] = {}
        output: list[dict[str, Any]] = []
        for row in rows:
            doi = _clean(row.get("doi")).casefold()
            probe = index.get(doi)
            if not probe or _clean(row.get("figure_asset_tier")) not in PROMOTABLE_FROM:
                output.append(row)
                continue
            item = _promoted(row, probe)
            if item is None:
                refused["no_framework_candidate"] = refused.get("no_framework_candidate", 0) + 1
                output.append(row)
                continue
            failures = strict_failures(batch, domain, item)
            if failures:
                # A promotion that would not survive the gates is not a promotion.
                key = ",".join(failures)
                refused[key] = refused.get(key, 0) + 1
                output.append(row)
                continue
            output.append(item)
            upgraded += 1
        report["domains"][domain] = {
            "rows": len(rows),
            "candidates": sum(1 for row in rows if _clean(row.get("doi")).casefold() in index),
            "upgraded": upgraded,
            "refused": refused,
        }
        if upgraded and not dry_run:
            changed_any = _publish_with_backup(path, _jsonl_text(output), backup) or changed_any
    report["backup"] = str(backup) if changed_any else None
    report["outputs_changed"] = changed_any
    if not dry_run:
        atomic_write_text(
            batch / "manifests" / "asset_tier_upgrade.json",
            json.dumps(report, ensure_ascii=False, indent=2),
        )
    return report


__all__ = ["upgrade_asset_tiers"]
