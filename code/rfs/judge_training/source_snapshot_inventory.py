"""Audit one consistent strict-source generation before any D0 publication."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
from typing import Any, Callable

from .literature_freeze import DOMAINS, _archive_members, canonicalize, strict_failures
from .source_snapshot_assets import paper_asset_specs
from .source_snapshot_io import canonical_hash, sha256_file, stable_bytes


def identity_aliases(row: dict) -> list[str]:
    """Normalize known identifier forms while preserving source values elsewhere."""
    result = []
    for field in ("paper_id", "official_paper_id", "doi", "arxiv_id", "pmid", "pmcid"):
        value = str(row.get(field) or "").strip().casefold()
        if not value:
            continue
        if field == "doi":
            value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)
        elif field == "arxiv_id":
            value = re.sub(r"^(?:https?://(?:export\.)?arxiv\.org/(?:abs|pdf)/|arxiv:\s*)", "", value)
            value = re.sub(r"v\d+$", "", value.removesuffix(".pdf"))
        result.append(f"{field}:{value}")
    return sorted(set(result))


def _input_files(batch: Path) -> list[Path]:
    paths = [batch / "verified" / domain / "papers.jsonl" for domain in DOMAINS]
    paths.extend([batch / "verified/strict_training_papers.jsonl", batch / "manifests/strict_training_dashboard.json"])
    for name in ("multisource_collection_events.jsonl", "strict_candidate_freeze.json", "source_probe.json",
                 "fulltext_extraction.json", "source_pdf_recovery.json"):
        if (batch / "manifests" / name).is_file():
            paths.append(batch / "manifests" / name)
    return paths


def capture_metadata(batch: Path) -> tuple[list[dict], dict[str, bytes]]:
    records, contents = [], {}
    for path in _input_files(batch):
        raw = stable_bytes(path)
        relative = path.relative_to(batch).as_posix()
        contents[relative] = raw
        records.append({"relative_path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)})
    return records, contents


def assert_metadata_unchanged(batch: Path, records: list[dict]) -> None:
    for item in records:
        if sha256_file(batch / item["relative_path"]) != item["sha256"]:
            raise RuntimeError(f"Input generation changed: {item['relative_path']}")


def _rows_from_bytes(name: str, content: bytes) -> list[dict]:
    result = []
    for index, line in enumerate(content.decode("utf-8-sig").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed input {name}:{index}") from exc
        if not isinstance(row, dict) or not str(row.get("paper_id") or "").strip():
            raise ValueError(f"Missing paper identity at {name}:{index}")
        result.append(row)
    return result


def _check_unique_papers(rows: list[dict]) -> None:
    owners = {}
    for row in rows:
        for alias in identity_aliases(row):
            previous = owners.get(alias)
            if previous is not None:
                raise ValueError(f"Canonical identity overlap requires upstream resolution: {alias}: {previous}, {row['paper_id']}")
            owners[alias] = row["paper_id"]


def _merge_provenance(qualified: dict, audit: list[dict]) -> dict:
    aliases, destinations = {}, {}
    for row in canonicalize(qualified, include_provenance=True):
        all_aliases = sorted({alias for source in row["canonical_sources"] for alias in identity_aliases(source["record"])})
        dois = {value for value in all_aliases if value.startswith("doi:")}
        if len(dois) > 1:
            raise ValueError(f"Conflicting DOI identities in canonical paper: {row['paper_id']}")
        aliases[row["paper_id"]] = all_aliases
        for source in row["canonical_sources"]:
            destinations[(source["domain"], canonical_hash(source["record"]))] = {
                "canonical_paper_id": row["paper_id"], "identity_keys": source["identity_keys"]}
    owners = {}
    for paper_id, values in aliases.items():
        for alias in values:
            if alias in owners and owners[alias] != paper_id:
                raise ValueError(f"Canonical alias overlap requires upstream resolution: {alias}")
            owners[alias] = paper_id
    for item in audit:
        item.update(destinations.get((item["domain"], item["record_sha256"]), {"canonical_paper_id": None}))
    return aliases


def inspect_paper_assets(job: tuple[str, dict]) -> tuple[dict, list, list, str | None]:
    """One process owns its MuPDF context; never decode PDFs concurrently in threads."""
    batch, row = job
    try:
        specs, issues = paper_asset_specs(Path(batch), row)
        return row, specs, issues, None
    except Exception as exc:
        return row, [], [], str(exc)


def audit_source_generation(
    batch_root: str | Path, *, workers: int = 4,
    progress: Callable[[dict], None] | None = None,
    tiers: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Audit all strict rows; never silently remove a failed strict record.

    ``tiers`` narrows which audit-clean records this version publishes, never the
    audit itself: every ledger record still gets its own verdict, the ledger-wide
    consistency checks still run against the whole strict view, and the manifest
    states the scope the published set was selected under.
    """
    batch = Path(batch_root).resolve()
    inputs, contents = capture_metadata(batch)
    strict = _rows_from_bytes("strict", contents["verified/strict_training_papers.jsonl"])
    if not strict:
        raise ValueError("Cannot freeze an empty strict view")
    _check_unique_papers(strict)
    wanted = tuple(tiers or ())
    published = [row for row in strict if not wanted or row.get("figure_asset_tier") in wanted]
    if not published:
        raise ValueError("Published scope selects no strict record")
    scope = {"figure_asset_tiers": list(wanted), "in_scope_papers": len(published),
             "out_of_scope_strict_papers": len(strict) - len(published), "strict_view_papers": len(strict)}
    dashboard = json.loads(contents["manifests/strict_training_dashboard.json"])
    historical = {domain: _rows_from_bytes(domain, contents[f"verified/{domain}/papers.jsonl"]) for domain in DOMAINS}
    _archive_members.cache_clear()
    qualified: dict[str, list[dict]] = {domain: [] for domain in DOMAINS}
    audit = []
    reasons = Counter()
    checked_at = datetime.now(timezone.utc).isoformat()
    for domain, rows in historical.items():
        for index, row in enumerate(rows):
            failures = strict_failures(batch, domain, row)
            reasons.update(failures)
            audit.append({"paper_id": row["paper_id"], "domain": domain, "source_line": index + 1,
                          "record_sha256": canonical_hash(row), "eligible": not failures, "reasons": failures,
                          "checked_at": checked_at, "check_method": "deterministic_source_gates",
                          "checker": "rfs.judge_training.literature_freeze.strict_failures"})
            if not failures:
                qualified[domain].append(row)
    expected = canonicalize(qualified)
    expected_hash = canonical_hash(sorted(expected, key=lambda x: x["paper_id"]))
    if expected_hash != canonical_hash(sorted(strict, key=lambda x: x["paper_id"])):
        raise ValueError("Strict view differs from current ledger/gates; rebuild strict before capturing D0")
    aliases = _merge_provenance(qualified, audit)
    if dashboard.get("strict_training_total") != len(strict):
        raise ValueError("Strict dashboard count differs from strict view")
    if dashboard.get("historical_verified_total") != sum(map(len, historical.values())):
        raise ValueError("Historical dashboard count differs from ledgers")
    unique_by_domain = Counter(row["primary_domain"] for row in expected)
    for domain in DOMAINS:
        report = dashboard.get("domains", {}).get(domain, {})
        if (
            report.get("historical") != len(historical[domain])
            or report.get("strict") != unique_by_domain[domain]
            or report.get("strict_membership") != len(qualified[domain])
        ):
            raise ValueError(f"Domain dashboard mismatch: {domain}")
    assets, problems, review = [], [], []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(published))),
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = ((str(batch), row) for row in published)
        for index, (row, specs, issues, error) in enumerate(pool.map(inspect_paper_assets, jobs), 1):
            assets.extend(specs)
            review.append({"paper_id": row["paper_id"], "semantic_review_status": "unreviewed",
                           "issues": ["figure_completeness_and_semantic_binding_not_reviewed", *issues]})
            if error:
                problems.append({"paper_id": row["paper_id"], "error": error})
            if progress and (index % 50 == 0 or index == len(published)):
                progress({"stage": "audit", "checked": index, "total": len(published), "errors": len(problems)})
    assert_metadata_unchanged(batch, inputs)
    primary = Counter(row.get("primary_domain") or row.get("domain") for row in published)
    labels = Counter(domain for row in published for domain in row.get("domain_labels", [row.get("domain")]))
    target = int(dashboard.get("target_per_domain") or 2000)
    report = {
        "unique_papers": len(published), "historical_ledger_records": sum(map(len, historical.values())),
        "primary_domains": dict(primary), "domain_labels": dict(labels),
        "duplicate_domain_memberships": sum(labels.values()) - len(published),
        "target_per_domain": target, "quota_met": len(published) >= target * len(DOMAINS) and all(labels[d] >= target for d in DOMAINS),
        "shortfalls": {d: max(0, target - labels[d]) for d in DOMAINS},
        "scope": scope,
    }
    return {"ok": not problems, "papers": published, "assets": assets, "audit": audit, "paper_aliases": aliases,
            "scope": scope,
            "review_queue": review, "errors": problems, "input_files": inputs,
            "input_contents": contents, "input_set_sha256": canonical_hash(inputs),
            "paper_set_sha256": canonical_hash(sorted(row["paper_id"] for row in published)),
            "domain_report": report, "exclusion_summary": {"failure_reasons": dict(reasons),
                "excluded_ledger_records": sum(not x["eligible"] for x in audit),
                "out_of_scope_records": dict(Counter(row.get("figure_asset_tier") for row in strict
                    if wanted and row.get("figure_asset_tier") not in wanted)),
                "pending_full_text": [{"paper_id": x["paper_id"], "domain": x["domain"],
                    "status": "needs_complete_paper_text"} for x in audit
                    if "fulltext_preview_only" in x["reasons"]],
                "pending_figure_rendering": [{"paper_id": x["paper_id"], "domain": x["domain"],
                    "status": "needs_renderable_source_figure"} for x in audit if "figure_rendering_missing" in x["reasons"]]}}
