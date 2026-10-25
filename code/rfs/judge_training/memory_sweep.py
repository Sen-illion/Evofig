"""Retire the Memory an older Judge measured once a new Judge takes over.

The first Memory rows were scored by whatever Judge happened to be online, so a trained
Judge that later replaces it inherits evidence it never verified. This module decides which
rows the incoming Judge still stands behind and quarantines the rest, so a decision can no
longer cite a measurement nobody currently believes. Nothing here calls a provider: it
reads the libraries, writes a report, and leaves judging to the paired ON/OFF harness every
other candidate already goes through.

A patch is quarantined with the evidence it was rendered from, because a directive is only
as trustworthy as the measurement behind it; a patch whose source rows are missing from the
library is skipped rather than condemned, since nothing can be concluded about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..contracts.evolution import judge_fingerprint_identity, normalize_judge_fingerprint, same_judge
from ..contracts.judge_sweep import (
    QUARANTINE_STATUS,
    SWEEP_SELECTIONS,
    normalize_memory_sweep_report,
    sweep_counts,
)
from ..contracts.prompt_patch import normalize_prompt_patch
from ..evolution.direct import ExperienceStore, PromptPatchStore
from ..evolution.direct.store import DEFAULT_RETRIEVAL_STATUSES


@dataclass(frozen=True)
class SweepCandidate:
    """One retrievable Memory row and the Judge it was last measured under."""

    identifier: str
    kind: str
    apply_at: str
    status: str
    fingerprint: dict[str, Any]
    source_ids: tuple[str, ...] = ()


def collect_sweep_candidates(
    *,
    store: ExperienceStore | None = None,
    patch_store: PromptPatchStore | None = None,
) -> list[SweepCandidate]:
    """Read both libraries into the rows a sweep can decide about, newest Judge first.

    A row carries the fingerprint it was written under unless a re-validation has since
    re-measured it, in which case the newer fingerprint is the one that counts.
    """

    candidates: list[SweepCandidate] = []
    if store is not None:
        for record in store.records_with_status():
            identifier = str(record.get("experience_id") or "")
            if not identifier:
                continue
            candidates.append(
                SweepCandidate(
                    identifier=identifier,
                    kind="experience",
                    apply_at=str(record.get("apply_at") or ""),
                    status=str(record.get("status") or ""),
                    fingerprint=store.revalidation_fingerprint(identifier) or normalize_judge_fingerprint(record.get("judge_fingerprint")),
                )
            )
    if patch_store is not None:
        for record in patch_store.records_with_status():
            patch = normalize_prompt_patch(record)
            if not patch["patch_id"]:
                continue
            candidates.append(
                SweepCandidate(
                    identifier=str(patch["patch_id"]),
                    kind="patch",
                    apply_at=str(patch["apply_at"]),
                    status=str(record.get("status") or ""),
                    fingerprint=patch_store.revalidation_fingerprint(str(patch["patch_id"])) or {},
                    source_ids=tuple(str(item) for item in patch["source_experience_ids"]),
                )
            )
    return sorted(candidates, key=lambda item: (item.kind, item.identifier))


def _entry(candidate: SweepCandidate, reason: str) -> dict[str, Any]:
    fingerprint = normalize_judge_fingerprint(candidate.fingerprint)
    return {
        "id": candidate.identifier,
        "kind": candidate.kind,
        "apply_at": candidate.apply_at,
        "status": candidate.status,
        "reason": reason,
        "judge_fingerprint": fingerprint,
        "fingerprint_identity": judge_fingerprint_identity(fingerprint),
        "source_ids": list(candidate.source_ids),
    }


def _experience_reason(candidate: SweepCandidate, *, incoming: dict[str, Any], selection: str) -> str:
    """Name why one Memory row leaves the injectable set, or an empty string to keep it."""

    if selection == "all":
        return "selection_requested"
    fingerprint = normalize_judge_fingerprint(candidate.fingerprint)
    if same_judge(fingerprint, incoming):
        return ""
    if selection == "bootstrap" and fingerprint["trained"]:
        return ""
    return "bootstrap_judge_memory" if selection == "bootstrap" else "judge_fingerprint_mismatch"


def plan_memory_sweep(
    candidates: Iterable[SweepCandidate],
    *,
    incoming: dict[str, Any],
    selection: str = "mismatched",
) -> dict[str, Any]:
    """Decide what a new Judge leaves injectable, without touching the libraries."""

    # The verdict only depends on the rows and the two judges, never on the order they arrive
    # in, so the plan is sorted and a library read twice plans the same way twice.
    rows = sorted(candidates, key=lambda item: (item.kind, item.identifier))
    incoming_fingerprint = normalize_judge_fingerprint(incoming)
    chosen = selection if selection in SWEEP_SELECTIONS else "mismatched"
    allowed = set(DEFAULT_RETRIEVAL_STATUSES)
    quarantine: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    verdicts: dict[str, str] = {}
    # Experiences are decided first: a patch is judged by the evidence it was rendered from.
    ordered = [item for item in rows if item.kind != "patch"] + [item for item in rows if item.kind == "patch"]
    for candidate in ordered:
        if candidate.status not in allowed:
            verdicts[candidate.identifier] = "skip"
            skipped.append(_entry(candidate, "status_not_retrievable"))
            continue
        if candidate.kind != "patch":
            reason = _experience_reason(candidate, incoming=incoming_fingerprint, selection=chosen)
            verdicts[candidate.identifier] = "quarantine" if reason else "keep"
            (quarantine if reason else kept).append(_entry(candidate, reason))
            continue
        if not candidate.source_ids:
            verdicts[candidate.identifier] = "skip"
            skipped.append(_entry(candidate, "no_source_memory"))
            continue
        unknown = [item for item in candidate.source_ids if item not in verdicts]
        if unknown:
            verdicts[candidate.identifier] = "skip"
            skipped.append(_entry(candidate, "source_memory_unavailable"))
            continue
        stale = [item for item in candidate.source_ids if verdicts.get(item) == "quarantine"]
        reason = "source_memory_quarantined" if stale else ""
        verdicts[candidate.identifier] = "quarantine" if reason else "keep"
        (quarantine if reason else kept).append(_entry(candidate, reason))
    return {
        "incoming_fingerprint": incoming_fingerprint,
        "selection": chosen,
        "counts": sweep_counts(quarantine=quarantine, kept=kept, skipped=skipped, revalidated=[]),
        "quarantine": quarantine,
        "kept": kept,
        "skipped": skipped,
    }


def apply_memory_sweep(
    plan: dict[str, Any],
    *,
    store: ExperienceStore | None = None,
    patch_store: PromptPatchStore | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Quarantine the rows the plan condemned, unless this is a dry run.

    Quarantine is a lifecycle transition to ``retrying``: the row keeps its evidence and its
    history readable, but it is no longer eligible for retrieval, so no stage can inject it
    while it waits to be re-measured.
    """

    incoming = normalize_judge_fingerprint(plan.get("incoming_fingerprint"))
    if not dry_run:
        for entry in plan.get("quarantine") or []:
            target = store if str(entry.get("kind") or "") != "patch" else patch_store
            if target is None:
                continue
            target.transition(
                str(entry.get("id") or ""),
                QUARANTINE_STATUS,
                reason=str(entry.get("reason") or ""),
                sweep="judge_version_changed",
                previous_fingerprint=normalize_judge_fingerprint(entry.get("judge_fingerprint")),
                incoming_fingerprint=incoming,
            )
    report = normalize_memory_sweep_report({**plan, "dry_run": bool(dry_run)})
    counts = report["counts"]
    if counts["quarantined"]:
        state = "would be quarantined" if dry_run else "are quarantined"
        report["summary"] = f"{counts['quarantined']} of {counts['considered']} retrievable rows were measured by another Judge and {state}; re-measure them with the paired ON/OFF harness."
    elif counts["pending"]:
        report["summary"] = f"nothing new to sweep: {counts['pending']} rows are already quarantined and still await re-measurement."
    elif counts["kept"]:
        report["summary"] = f"nothing to sweep: all {counts['kept']} retrievable rows were measured by the incoming Judge."
    else:
        report["summary"] = f"nothing to sweep: none of the {counts['considered']} rows in this library is in the injectable set."
    return report


__all__ = [
    "SweepCandidate",
    "apply_memory_sweep",
    "collect_sweep_candidates",
    "plan_memory_sweep",
]
