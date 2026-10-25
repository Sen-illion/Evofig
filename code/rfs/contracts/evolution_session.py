"""Contract for one wall-clock evolution session.

A session is the whole loop the build plan describes, run against a clock instead of a human
typing the next command: evolve a paper, validate the experience that run produced on other
papers, promote or refute it, then evolve the next paper under the Memory the earlier stations
earned. The interesting part of such a report is not that it finished but which station
stopped, why, and what it had already recorded before it stopped. This module fixes that
vocabulary -- the station kinds, the statuses a station can end in, and the reasons it can
stop for -- so the driver, the report it writes and whoever audits it cannot drift apart.
"""

from __future__ import annotations

from typing import Any


EVOLUTION_SESSION_CONTRACT = "rfs.evolution_session_report"
# Version 2 adds the optional ``dataset_export`` block. A report written before it is still
# read: the block is absent, everything else is unchanged.
EVOLUTION_SESSION_VERSION = 2
SUPPORTED_SESSION_VERSIONS = (1, 2)
# A station either evolves one paper or validates one experience on other papers.
SESSION_STATION_KINDS = ("evolve", "validate")
SESSION_STATION_STATUSES = ("pending", "completed", "failed", "skipped")
# Why a station stopped without finishing. ``budget_exhausted`` and
# ``no_candidate_experience`` are the two a healthy session stops for: a session that runs out
# of wall clock drops the rest of its queue instead of cutting a paper in half, and a session
# with nothing new to validate says so rather than validating nothing quietly.
SESSION_STATION_REASONS = (
    "",
    "budget_exhausted",
    "no_candidate_experience",
    "paper_unreadable",
    "run_failed",
    "validation_failed",
    "rate_limited",
    "validation_disabled",
)
# The reasons that mean the station could not do its job; the rest mean it never started.
SESSION_FAILURE_REASONS = ("run_failed", "validation_failed", "rate_limited", "paper_unreadable")
SESSION_SKIP_REASONS = ("budget_exhausted", "no_candidate_experience", "paper_unreadable", "validation_disabled")
SESSION_ATTEMPT_STATUSES = ("ok", "failed")
# Why the session's training export did not land. ``""`` is a successful export; a failure
# names ``export_failed`` and carries the detail, and never changes the session's own verdict.
DATASET_EXPORT_REASONS = ("", "export_failed")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def session_station_id(kind: str, subject: str) -> str:
    """Name a station by what it is and what it runs, so a rerun keeps the same label."""

    return f"{str(kind)}:{str(subject)}"


def normalize_session_budget(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the wall-clock budget the session ran under."""

    source = raw if isinstance(raw, dict) else {}
    hours = _number(source.get("hours"))
    used = _number(source.get("used_seconds"))
    return {
        "hours": round(hours, 4),
        "used_seconds": round(used, 2),
        "remaining_seconds": round(max(0.0, hours * 3600.0 - used), 2),
        "exhausted": bool(source.get("exhausted")),
    }


def normalize_session_attempt(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize one try at a station; a retry is a new attempt, never a rewrite of the old one."""

    source = raw if isinstance(raw, dict) else {}
    status = str(source.get("status") or "")
    return {
        "attempt": int(_number(source.get("attempt"))),
        "status": status if status in SESSION_ATTEMPT_STATUSES else "failed",
        "reason": str(source.get("reason") or ""),
        "detail": str(source.get("detail") or ""),
        "out_dir": str(source.get("out_dir") or ""),
        "seconds": round(_number(source.get("seconds")), 2),
    }


def normalize_session_station(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize one station of the queue, keeping what it decided and what it recorded."""

    source = raw if isinstance(raw, dict) else {}
    kind = str(source.get("kind") or "")
    subject = str(source.get("subject") or "")
    status = str(source.get("status") or "pending")
    status = status if status in SESSION_STATION_STATUSES else "pending"
    evidence = source.get("evidence") if isinstance(source.get("evidence"), dict) else {}
    return {
        "station_id": str(source.get("station_id") or session_station_id(kind, subject)),
        "kind": kind,
        "subject": subject,
        "status": status,
        "reason": str(source.get("reason") or ""),
        "detail": str(source.get("detail") or ""),
        "out_dir": str(source.get("out_dir") or ""),
        "seconds": round(_number(source.get("seconds")), 2),
        "ok": bool(source.get("ok")) if "ok" in source else status == "completed",
        "attempts": [normalize_session_attempt(item) for item in source.get("attempts") or [] if isinstance(item, dict)],
        "evidence": {str(key): value for key, value in evidence.items()},
    }


def _declared_version(value: Any) -> int:
    """Read the report version a document declares, keeping an older one readable.

    A report that names no version, or one nobody defined, is read as the current version:
    writing is what must never guess, and the validator checks the word the document used.
    """

    try:
        number = int(value)
    except (TypeError, ValueError):
        return EVOLUTION_SESSION_VERSION
    return number if number in SUPPORTED_SESSION_VERSIONS else EVOLUTION_SESSION_VERSION


def normalize_dataset_export(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize the training export a session ran at the end of its budget."""

    source = raw if isinstance(raw, dict) else {}
    reason = str(source.get("reason") or "")
    return {
        "attempted": bool(source.get("attempted")),
        "ok": bool(source.get("ok")),
        "reason": reason if reason in DATASET_EXPORT_REASONS else "export_failed",
        "detail": str(source.get("detail") or ""),
        "dataset_dir": str(source.get("dataset_dir") or ""),
        "review_dir": str(source.get("review_dir") or ""),
        "records": int(_number(source.get("records"))),
        "pairs": int(_number(source.get("pairs"))),
        "review_pairs": int(_number(source.get("review_pairs"))),
        "split": str(source.get("split") or ""),
        "report": str(source.get("report") or ""),
    }


def normalize_evolution_session_report(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a session report without dropping what it decided about."""

    source = raw if isinstance(raw, dict) else {}
    stations = [normalize_session_station(item) for item in source.get("stations") or [] if isinstance(item, dict)]
    version = _declared_version(source.get("version"))
    blockers = [
        {
            "station_id": str(item.get("station_id") or ""),
            "kind": str(item.get("kind") or ""),
            "reason": str(item.get("reason") or ""),
            "detail": str(item.get("detail") or ""),
        }
        for item in source.get("blockers") or []
        if isinstance(item, dict)
    ]
    counts = {name: sum(1 for item in stations if item["status"] == name) for name in SESSION_STATION_STATUSES}
    report = {
        "contract": EVOLUTION_SESSION_CONTRACT,
        "version": version,
        "session_id": str(source.get("session_id") or ""),
        "summary": str(source.get("summary") or ""),
        "ok": bool(source.get("ok", True)),
        "dry_run": bool(source.get("dry_run")),
        "out_dir": str(source.get("out_dir") or ""),
        "config": str(source.get("config") or ""),
        "experience_library": str(source.get("experience_library") or ""),
        "validation_suite": str(source.get("validation_suite") or ""),
        "validate": bool(source.get("validate", True)),
        "budget": normalize_session_budget(source.get("budget")),
        "stations": stations,
        "counts": {
            "stations": len(stations),
            "evolve": sum(1 for item in stations if item["kind"] == "evolve"),
            "validate": sum(1 for item in stations if item["kind"] == "validate"),
            **counts,
        },
        "blockers": blockers,
    }
    export = source.get("dataset_export")
    if isinstance(export, dict) and export:
        report["dataset_export"] = normalize_dataset_export(export)
    return report


def validate_evolution_session_report(report: dict[str, Any]) -> dict[str, Any]:
    """Check a session report against the contract and against its own stations."""

    normalized = normalize_evolution_session_report(report)
    errors: list[str] = []
    stations = normalized["stations"]
    declared = report.get("version") if isinstance(report, dict) else None
    if declared is not None:
        try:
            number = int(declared)
        except (TypeError, ValueError):
            number = -1
        if number not in SUPPORTED_SESSION_VERSIONS:
            errors.append(f"unknown session report version: {declared!r}")
    if not stations:
        errors.append("a session report must carry at least one station")
    # Normalization sends a status it does not know to ``pending``, which is right for reading
    # an old report and wrong for writing a new one: a status nobody defined is a defect, so
    # the word the report actually used is checked before it is softened.
    for item in report.get("stations") or [] if isinstance(report, dict) else []:
        if not isinstance(item, dict):
            continue
        declared = str(item.get("status") or "pending")
        if declared not in SESSION_STATION_STATUSES:
            label = str(item.get("station_id") or item.get("subject") or "unnamed")
            errors.append(f"station {label} declares an unknown status: {declared!r}")
    seen: set[str] = set()
    for item in stations:
        station = item["station_id"]
        if not station:
            errors.append("every station needs a station_id")
        elif station in seen:
            errors.append(f"station {station} appears more than once in the queue")
        seen.add(station)
        if item["kind"] not in SESSION_STATION_KINDS:
            errors.append(f"station {station} is of an unknown kind: {item['kind']!r}")
        if not item["subject"]:
            errors.append(f"station {station} does not name what it runs")
        if item["status"] == "failed":
            if item["reason"] not in SESSION_FAILURE_REASONS:
                errors.append(f"station {station} failed for an unknown reason: {item['reason']!r}")
            if not item["detail"]:
                errors.append(f"station {station} failed without saying what stopped it")
            if not item["attempts"]:
                errors.append(f"station {station} failed without recording the attempt that failed")
        if item["status"] == "skipped":
            if item["reason"] not in SESSION_SKIP_REASONS:
                errors.append(f"station {station} was skipped for an unknown reason: {item['reason']!r}")
            if item["attempts"]:
                errors.append(f"station {station} was skipped after it had already run an attempt")
        if item["status"] == "pending" and item["attempts"]:
            errors.append(f"station {station} is still pending while it already recorded an attempt")
        if item["status"] == "completed":
            if item["reason"]:
                errors.append(f"station {station} completed but still names a reason: {item['reason']!r}")
            if not item["attempts"]:
                errors.append(f"station {station} claims it completed without recording the attempt that ran")
        numbers = [attempt["attempt"] for attempt in item["attempts"]]
        if numbers != list(range(1, len(numbers) + 1)):
            errors.append(f"station {station} numbered its attempts out of order: {numbers}")
        for attempt in item["attempts"]:
            if attempt["status"] == "failed" and attempt["reason"] not in SESSION_FAILURE_REASONS:
                errors.append(f"station {station} attempt {attempt['attempt']} failed for an unknown reason: {attempt['reason']!r}")
    failed = [item["station_id"] for item in stations if item["status"] == "failed"]
    if normalized["ok"] and failed:
        errors.append("the report claims success while these stations failed: " + ", ".join(failed))
    if not normalized["ok"] and not failed:
        errors.append("the report claims failure without naming a failed station")
    if normalized["blockers"] and not failed:
        errors.append("the report lists blockers while every station completed")
    if normalized["dry_run"]:
        covered = [item["station_id"] for item in stations if item["status"] not in ("pending", "skipped")]
        if covered:
            errors.append("a dry run cannot have carried out these stations: " + ", ".join(covered))
    budget = normalized["budget"]
    if budget["hours"] <= 0:
        errors.append("a session needs a positive wall-clock budget in hours")
    if budget["exhausted"] and any(item["status"] == "pending" for item in stations):
        errors.append("a session that ran out of budget must say which stations it dropped, not leave them pending")
    dropped_for_budget = [item["station_id"] for item in stations if item["status"] == "skipped" and item["reason"] == "budget_exhausted"]
    if dropped_for_budget and not budget["exhausted"]:
        errors.append("a station cannot be dropped for a budget the report says was not exhausted")
    export = normalized.get("dataset_export")
    if export:
        if export["ok"] and export["reason"]:
            errors.append(f"the dataset export claims success but still names a reason: {export['reason']!r}")
        if export["ok"] and not export["attempted"]:
            errors.append("the dataset export claims success without having been attempted")
        if not export["ok"]:
            if not export["attempted"]:
                errors.append("the dataset export is neither successful nor attempted")
            if not export["detail"]:
                errors.append("the dataset export failed without saying what stopped it")
    return {"ok": not errors, "errors": errors, "report": normalized}


__all__ = [
    "DATASET_EXPORT_REASONS",
    "EVOLUTION_SESSION_CONTRACT",
    "EVOLUTION_SESSION_VERSION",
    "SESSION_ATTEMPT_STATUSES",
    "SESSION_FAILURE_REASONS",
    "SESSION_SKIP_REASONS",
    "SESSION_STATION_KINDS",
    "SESSION_STATION_REASONS",
    "SESSION_STATION_STATUSES",
    "SUPPORTED_SESSION_VERSIONS",
    "normalize_dataset_export",
    "normalize_evolution_session_report",
    "normalize_session_attempt",
    "normalize_session_budget",
    "normalize_session_station",
    "session_station_id",
    "validate_evolution_session_report",
]
