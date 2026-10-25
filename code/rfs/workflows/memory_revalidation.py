"""Earn back the Memory a newer Judge quarantined, on the harness that promotes Memory.

Quarantining stale Memory is only half of a Judge handover: the rows have to be re-measured
or retired. This module runs the same paired ON/OFF validation that promotes an experience
in the first place -- on the suite that originally measured the row, so the re-measurement
is comparable -- then stamps the rows the incoming Judge still agrees with and retires the
ones it does not. Whether the incoming Judge is the trained one or a bootstrap stand-in is
the caller's decision, expressed as its fingerprint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ..contracts.evolution import normalize_judge_fingerprint
from ..contracts.judge_sweep import DEMOTED_STATUS, REVALIDATION_PROBLEMS, validate_memory_sweep_report
from ..evolution.direct import ExperienceStore, PromptPatchStore
from ..evolution.direct.store import DEFAULT_RETRIEVAL_STATUSES
from ..judge_training.memory_sweep import apply_memory_sweep, collect_sweep_candidates, plan_memory_sweep
from ..utils import ensure_dir, read_json, write_json
from .direct_experience_validation import run_direct_experience_validation


def _recorded_suite(store: ExperienceStore, patch_store: PromptPatchStore, entry: dict[str, Any]) -> tuple[str, str]:
    """Find the suite that measured a row, so its re-measurement runs on the same papers."""

    target = patch_store if str(entry.get("kind") or "") == "patch" else store
    report_path = target.last_validation_report(str(entry.get("id") or ""))
    if not report_path:
        return "", "no_recorded_validation_report"
    if not Path(report_path).exists():
        return "", "suite_unreadable"
    try:
        spec = read_json(report_path)
    except (OSError, ValueError):
        return "", "suite_unreadable"
    suite = str((spec if isinstance(spec, dict) else {}).get("suite") or "")
    if not suite or not Path(suite).exists():
        return "", "suite_unreadable"
    return suite, ""


def revalidate_quarantined(
    report: dict[str, Any],
    *,
    library: str | Path,
    out: str | Path = "",
    runner: Callable[..., dict[str, Any]] | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """Re-measure the quarantined rows and report what the incoming Judge concluded.

    A row comes back at whatever rung the paired harness put it on, and that rung is the
    answer: if the incoming Judge still places it in the injectable set it is restored there,
    and only a row that fell out of that set entirely is retired. Requiring ``promoted`` is
    what this used to do, and it retired every row the new Judge agreed with at
    ``same_domain_validated`` or ``cross_domain_validated`` -- rungs a run may inject -- so
    each handover permanently destroyed Memory that had not regressed at all. The worry
    behind that rule was a row quietly returning at a weaker rung; the rung it returns at is
    recorded on the row's event and in the outcome instead, so the drop is visible rather
    than fatal. A row whose original suite can no longer be read stays quarantined and says
    why.
    """

    root = ensure_dir(out).resolve() if out else None
    store = ExperienceStore(library)
    patch_store = PromptPatchStore(library)
    incoming = normalize_judge_fingerprint(report.get("incoming_fingerprint"))
    outcomes: list[dict[str, Any]] = []
    restored: set[str] = set()
    for entry in report.get("quarantine") or []:
        if limit and len(outcomes) >= int(limit):
            break
        identifier = str(entry.get("id") or "")
        kind = "patch" if str(entry.get("kind") or "") == "patch" else "experience"
        target = patch_store if kind == "patch" else store
        if kind == "patch":
            unresolved = [str(item) for item in entry.get("source_ids") or [] if str(item) not in restored]
            if unresolved:
                outcomes.append({"id": identifier, "kind": kind, "outcome": "unverifiable", "suite": "", "status": "", "report": "", "detail": "sources_not_restored: " + ", ".join(unresolved)})
                continue
        suite, problem = _recorded_suite(store, patch_store, entry)
        if problem:
            outcomes.append({"id": identifier, "kind": kind, "outcome": "unverifiable", "suite": "", "status": "", "report": "", "detail": problem})
            continue
        destination = (root / identifier) if root is not None else out
        options: dict[str, Any] = {"suite": suite, "experience_library": str(library), "out": destination}
        options["prompt_patch_id" if kind == "patch" else "experience_id"] = identifier
        if runner is not None:
            options["runner"] = runner
        try:
            result = run_direct_experience_validation(**options)
        except Exception as exc:  # a suite that cannot run leaves the row quarantined, not condemned
            outcomes.append({"id": identifier, "kind": kind, "outcome": "unverifiable", "suite": suite, "status": "", "report": "", "detail": f"revalidation_failed: {exc}"})
            continue
        status = str((result if isinstance(result, dict) else {}).get("status") or "")
        report_path = str((root / identifier / "cross_paper_validation.json") if root is not None else "")
        previous = normalize_judge_fingerprint(entry.get("judge_fingerprint"))
        if status in DEFAULT_RETRIEVAL_STATUSES:
            # The re-validation already moved the row to this rung, so nothing is transitioned
            # here; the event records which Judge measured it and how the rung moved.
            before = str(entry.get("status") or "")
            target.append_event(identifier, "judge_fingerprint_changed", judge_fingerprint=incoming, previous_fingerprint=previous, validation_report=report_path, sweep="judge_version_changed", status=status, previous_status=before)
            outcomes.append({"id": identifier, "kind": kind, "outcome": "restored", "suite": suite, "status": status, "report": report_path, "detail": f"rung moved from {before} to {status}" if before and before != status else ""})
            restored.add(identifier)
        else:
            target.transition(identifier, DEMOTED_STATUS, reason="failed_revalidation_under_new_judge", sweep="judge_version_changed", incoming_fingerprint=incoming, previous_fingerprint=previous, validation_report=report_path)
            outcomes.append({"id": identifier, "kind": kind, "outcome": "demoted", "suite": suite, "status": status, "report": report_path, "detail": ""})
    return outcomes


def sweep_memory_with_judge(
    *,
    experience_library: str | Path,
    incoming: dict[str, Any],
    selection: str = "mismatched",
    out: str | Path = "",
    dry_run: bool = True,
    revalidate: bool = False,
    runner: Callable[..., dict[str, Any]] | None = None,
    limit: int = 0,
    store: ExperienceStore | None = None,
    patch_store: PromptPatchStore | None = None,
) -> dict[str, Any]:
    """Plan the sweep, apply it, optionally re-measure the quarantined rows, and report."""

    if revalidate and dry_run:
        raise ValueError("re-validating Memory requires applying the quarantine first")
    library = Path(experience_library)
    root = ensure_dir(out).resolve() if out else None
    experience_store = store if store is not None else ExperienceStore(library)
    patch_library = patch_store if patch_store is not None else PromptPatchStore(library)
    plan = plan_memory_sweep(collect_sweep_candidates(store=experience_store, patch_store=patch_library), incoming=incoming, selection=selection)
    report = apply_memory_sweep(plan, store=experience_store, patch_store=patch_library, dry_run=dry_run)
    if revalidate:
        report["revalidated"] = revalidate_quarantined(report, library=library, out=root or "", runner=runner, limit=limit)
    report["library"] = str(library.resolve())
    report["out_dir"] = str(root) if root is not None else ""
    checked = validate_memory_sweep_report(report)
    final = {**checked["report"], "validation": {"ok": bool(checked["ok"]), "errors": list(checked["errors"])}, "ok": bool(checked["ok"])}
    if root is not None:
        write_json(root / "memory_sweep_report.json", final)
    return final


__all__ = ["revalidate_quarantined", "sweep_memory_with_judge"]
