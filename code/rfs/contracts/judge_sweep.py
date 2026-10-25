"""Contract for the Memory sweep that retires evidence an older Judge measured.

A trained Judge replaces the bootstrap one that wrote the first Memory rows, and those rows
still carry the score the old Judge gave them. The sweep decides which rows the incoming
Judge still stands behind, quarantines the rest so they stop being injected, and records
what re-validation under the new Judge concluded. This module fixes the shape of the report
so the code that runs the sweep, the code that reads it, and the humans auditing it agree.
"""

from __future__ import annotations

from typing import Any

from .evolution import EXPERIENCE_STATUSES, judge_fingerprint_identity, normalize_judge_fingerprint


MEMORY_SWEEP_CONTRACT = "rfs.memory_sweep_report"
MEMORY_SWEEP_VERSION = 1
SWEEP_SELECTIONS = ("mismatched", "bootstrap", "all")
SWEEP_KINDS = ("experience", "patch")
QUARANTINE_REASONS = (
    "judge_fingerprint_mismatch",
    "bootstrap_judge_memory",
    "source_memory_quarantined",
    "selection_requested",
)
QUARANTINE_STATUS = "retrying"
DEMOTED_STATUS = "overfit_or_failed"
SWEEP_OUTCOMES = ("restored", "demoted", "unverifiable")
SWEEP_SKIP_REASONS = (
    "status_not_retrievable",
    "no_source_memory",
    "source_memory_unavailable",
)
REVALIDATION_PROBLEMS = (
    "no_recorded_validation_report",
    "suite_unreadable",
    "sources_not_restored",
    "unsupported_kind",
    "revalidation_failed",
)
SWEEP_COUNTER_KEYS = ("considered", "quarantined", "kept", "skipped", "pending", "restored", "demoted", "unverifiable")


def _texts(value: Any) -> list[str]:
    return [str(item) for item in value if str(item)] if isinstance(value, list) else []


def normalize_sweep_target(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize one Memory row the sweep decided about."""

    source = raw if isinstance(raw, dict) else {}
    kind = str(source.get("kind") or "experience")
    fingerprint = normalize_judge_fingerprint(source.get("judge_fingerprint"))
    return {
        "id": str(source.get("id") or ""),
        "kind": kind if kind in SWEEP_KINDS else "experience",
        "apply_at": str(source.get("apply_at") or ""),
        "status": str(source.get("status") or ""),
        "reason": str(source.get("reason") or ""),
        "judge_fingerprint": fingerprint,
        "fingerprint_identity": judge_fingerprint_identity(fingerprint),
        "source_ids": _texts(source.get("source_ids")),
    }


def normalize_sweep_revalidation(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize what re-validating one quarantined row under the incoming Judge concluded."""

    source = raw if isinstance(raw, dict) else {}
    outcome = str(source.get("outcome") or "unverifiable")
    return {
        "id": str(source.get("id") or ""),
        "kind": "patch" if str(source.get("kind") or "") == "patch" else "experience",
        "outcome": outcome if outcome in SWEEP_OUTCOMES else "unverifiable",
        "status": str(source.get("status") or ""),
        "suite": str(source.get("suite") or ""),
        "report": str(source.get("report") or ""),
        "detail": str(source.get("detail") or ""),
    }


def normalize_memory_sweep_report(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a Memory sweep report without dropping what it decided about."""

    source = raw if isinstance(raw, dict) else {}
    fingerprint = normalize_judge_fingerprint(source.get("incoming_fingerprint"))
    selection = str(source.get("selection") or "mismatched")
    quarantine = [normalize_sweep_target(item) for item in source.get("quarantine") or [] if isinstance(item, dict)]
    kept = [normalize_sweep_target(item) for item in source.get("kept") or [] if isinstance(item, dict)]
    skipped = [normalize_sweep_target(item) for item in source.get("skipped") or [] if isinstance(item, dict)]
    revalidated = [normalize_sweep_revalidation(item) for item in source.get("revalidated") or [] if isinstance(item, dict)]
    return {
        "contract": MEMORY_SWEEP_CONTRACT,
        "version": MEMORY_SWEEP_VERSION,
        "summary": str(source.get("summary") or ""),
        "ok": bool(source.get("ok", True)),
        "library": str(source.get("library") or ""),
        "out_dir": str(source.get("out_dir") or ""),
        "selection": selection if selection in SWEEP_SELECTIONS else "mismatched",
        "dry_run": bool(source.get("dry_run")),
        "incoming_fingerprint": fingerprint,
        "incoming_identity": judge_fingerprint_identity(fingerprint),
        "counts": sweep_counts(quarantine=quarantine, kept=kept, skipped=skipped, revalidated=revalidated),
        "quarantine": quarantine,
        "kept": kept,
        "skipped": skipped,
        "revalidated": revalidated,
    }


def sweep_counts(*, quarantine: list[dict[str, Any]], kept: list[dict[str, Any]], skipped: list[dict[str, Any]], revalidated: list[dict[str, Any]]) -> dict[str, int]:
    """Derive the counts from the rows, so a report can never disagree with itself."""

    return {
        "considered": len(quarantine) + len(kept) + len(skipped),
        "quarantined": len(quarantine),
        "kept": len(kept),
        "skipped": len(skipped),
        "pending": sum(1 for item in skipped if item["status"] == QUARANTINE_STATUS),
        "restored": sum(1 for item in revalidated if item["outcome"] == "restored"),
        "demoted": sum(1 for item in revalidated if item["outcome"] == "demoted"),
        "unverifiable": sum(1 for item in revalidated if item["outcome"] == "unverifiable"),
    }


def validate_memory_sweep_report(report: dict[str, Any]) -> dict[str, Any]:
    """Check a sweep report against the contract and its own counts."""

    normalized = normalize_memory_sweep_report(report)
    errors: list[str] = []
    for key in ("quarantine", "kept", "skipped"):
        for entry in normalized[key]:
            if not entry["id"]:
                errors.append(f"sweep {key} entry without an id is not auditable")
    for entry in normalized["quarantine"]:
        if entry["reason"] not in QUARANTINE_REASONS:
            errors.append(f"unsupported quarantine reason: {entry['reason']}")
        if entry["status"] and entry["status"] not in EXPERIENCE_STATUSES:
            errors.append(f"quarantine entry {entry['id']} records an unsupported status: {entry['status']}")
    for entry in normalized["skipped"]:
        if entry["reason"] not in SWEEP_SKIP_REASONS:
            errors.append(f"unsupported skip reason: {entry['reason']}")
    for entry in normalized["revalidated"]:
        if not entry["id"]:
            errors.append("a revalidation entry without an id is not auditable")
        if entry["outcome"] == "unverifiable" and not entry["detail"]:
            errors.append(f"unverifiable revalidation {entry['id']} must say why")
    expected = sweep_counts(quarantine=normalized["quarantine"], kept=normalized["kept"], skipped=normalized["skipped"], revalidated=normalized["revalidated"])
    for key, value in expected.items():
        if normalized["counts"][key] != value:
            errors.append(f"sweep count {key} says {normalized['counts'][key]} but its rows say {value}")
    if normalized["dry_run"] and normalized["revalidated"]:
        errors.append("a dry run cannot have revalidated anything")
    return {"ok": not errors, "errors": errors, "report": normalized}


__all__ = [
    "DEMOTED_STATUS",
    "MEMORY_SWEEP_CONTRACT",
    "MEMORY_SWEEP_VERSION",
    "QUARANTINE_REASONS",
    "QUARANTINE_STATUS",
    "SWEEP_COUNTER_KEYS",
    "SWEEP_KINDS",
    "SWEEP_OUTCOMES",
    "SWEEP_SELECTIONS",
    "SWEEP_SKIP_REASONS",
    "REVALIDATION_PROBLEMS",
    "normalize_memory_sweep_report",
    "normalize_sweep_revalidation",
    "normalize_sweep_target",
    "sweep_counts",
    "validate_memory_sweep_report",
]
