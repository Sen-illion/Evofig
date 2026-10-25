"""Contract for the one-command Judge handover.

A trained Judge arrives as a checkpoint and has to leave as the Judge the pipeline actually
runs under: it gets measured, gated, registered, and the Memory an older Judge wrote is
re-measured against it. Those steps used to be a sequence a human typed, handing the paths of
one step's artifacts to the next by hand. This module fixes the vocabulary of that chain --
the steps, the statuses a step can end in, and the reasons a step refuses to continue -- so
the driver, the report it writes, and the humans auditing it cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from .evolution import normalize_judge_fingerprint


JUDGE_HANDOVER_CONTRACT = "rfs.judge_handover_report"
JUDGE_HANDOVER_VERSION = 1
HANDOVER_STEPS = ("train", "predict", "promote", "register", "sweep")
HANDOVER_STEP_STATUSES = ("pending", "completed", "reused", "skipped", "blocked", "failed")
# The Judge roles this chain can register, weakest first; the fingerprint records the
# strongest one the handover actually earned.
HANDOVER_ROLE_ORDER = ("experimental", "shadow", "online", "frozen")
# Why a step refuses to let the chain continue. A blocked step names one of these, so a
# reader never has to guess whether a missing artifact or a failed gate stopped the run.
HANDOVER_BLOCK_REASONS = (
    "no_suite",
    "no_dataset",
    "no_training_report",
    "no_checkpoint",
    "no_endpoint",
    "no_predictions",
    "no_bootstrap_predictions",
    "no_model_id",
    "no_dataset_version",
    "no_registry",
    "no_git_commit",
    "promotion_not_approved",
)
# Why a step did not apply at all; neither is a failure of the handover itself.
HANDOVER_SKIP_REASONS = ("no_experience_library", "earlier_step_blocked")


def handover_step_statuses(*, dry_run: bool = False) -> tuple[str, ...]:
    """Return the statuses a step may end in, which a dry run narrows."""

    return ("pending", "skipped") if dry_run else HANDOVER_STEP_STATUSES


def normalize_handover_step(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize one step of the chain, keeping what it decided and what it wrote."""

    source = raw if isinstance(raw, dict) else {}
    status = str(source.get("status") or "pending")
    artifacts = source.get("artifacts") if isinstance(source.get("artifacts"), dict) else {}
    would_block = str(source.get("would_block") or "")
    return {
        "step": str(source.get("step") or ""),
        "status": status if status in HANDOVER_STEP_STATUSES else "pending",
        "detail": str(source.get("detail") or ""),
        "reason": str(source.get("reason") or ""),
        "would_block": would_block,
        "artifacts": {str(key): str(value) for key, value in artifacts.items() if str(value)},
    }


def normalize_judge_handover_report(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a handover report without dropping what it decided about."""

    source = raw if isinstance(raw, dict) else {}
    promotion = source.get("promotion") if isinstance(source.get("promotion"), dict) else {}
    counts = source.get("sweep") if isinstance(source.get("sweep"), dict) else {}
    blockers = [
        {"step": str(item.get("step") or ""), "reason": str(item.get("reason") or ""), "detail": str(item.get("detail") or "")}
        for item in source.get("blockers") or []
        if isinstance(item, dict)
    ]
    return {
        "contract": JUDGE_HANDOVER_CONTRACT,
        "version": JUDGE_HANDOVER_VERSION,
        "summary": str(source.get("summary") or ""),
        "ok": bool(source.get("ok", True)),
        "dry_run": bool(source.get("dry_run")),
        "force": bool(source.get("force")),
        "out_dir": str(source.get("out_dir") or ""),
        "model_id": str(source.get("model_id") or ""),
        "dataset_version": str(source.get("dataset_version") or ""),
        "suite_id": str(source.get("suite_id") or ""),
        "suite_manifest": str(source.get("suite_manifest") or ""),
        "checkpoint": str(source.get("checkpoint") or ""),
        "registry": str(source.get("registry") or ""),
        "judge_fingerprint": normalize_judge_fingerprint(source.get("judge_fingerprint")),
        "predictions_source": str(source.get("predictions_source") or ""),
        "shadow_omitted": bool(source.get("shadow_omitted")),
        "registered_as_experimental": bool(source.get("registered_as_experimental")),
        "roles": [str(role) for role in source.get("roles") or []],
        "promotion": {str(key): value for key, value in promotion.items()},
        "sweep": {str(key): int(value or 0) for key, value in counts.items()},
        "steps": [normalize_handover_step(item) for item in source.get("steps") or [] if isinstance(item, dict)],
        "blockers": blockers,
    }


def validate_judge_handover_report(report: dict[str, Any]) -> dict[str, Any]:
    """Check a handover report against the contract and its own steps."""

    normalized = normalize_judge_handover_report(report)
    errors: list[str] = []
    steps = normalized["steps"]
    if tuple(item["step"] for item in steps) != HANDOVER_STEPS:
        errors.append("a handover report must carry its steps exactly once and in order: " + ", ".join(HANDOVER_STEPS))
    for item in steps:
        if item["status"] == "blocked":
            if item["reason"] not in HANDOVER_BLOCK_REASONS:
                errors.append(f"step {item['step']} is blocked for an unknown reason: {item['reason']!r}")
            if not item["detail"]:
                errors.append(f"step {item['step']} is blocked without saying what stopped it")
        if item["status"] == "skipped" and item["reason"] not in HANDOVER_SKIP_REASONS:
            errors.append(f"step {item['step']} was skipped for an unknown reason: {item['reason']!r}")
        if item["would_block"] and item["would_block"] not in HANDOVER_BLOCK_REASONS:
            errors.append(f"step {item['step']} foresees an unknown blocker: {item['would_block']!r}")
    if normalized["dry_run"]:
        covered = [item["step"] for item in steps if item["status"] not in ("pending", "skipped")]
        if covered:
            errors.append("a dry run cannot have carried out these steps: " + ", ".join(covered))
    blocked = [item["step"] for item in steps if item["status"] in ("blocked", "failed")]
    if normalized["ok"] and blocked:
        errors.append("the report claims success while these steps did not complete: " + ", ".join(blocked))
    if not normalized["ok"] and not blocked:
        errors.append("the report claims failure without naming a blocked or failed step")
    if normalized["blockers"] and not blocked:
        errors.append("the report lists blockers while every step completed")
    by_step = {item["step"]: item for item in steps}
    if not normalized["dry_run"] and by_step["register"]["status"] in ("completed", "reused"):
        if not normalized["promotion"]:
            errors.append("a registered handover must report the promotion block it registered under")
        if not by_step["register"]["artifacts"].get("registry_entry"):
            errors.append("a registered handover must name the registry entry it wrote")
        if not normalized["roles"]:
            errors.append("a registered handover must name the roles the Judge was registered with")
    if not normalized["dry_run"] and by_step["sweep"]["status"] in ("completed", "reused") and not normalized["sweep"]:
        errors.append("a handover that re-dressed Memory must report what its sweep did")
    return {"ok": not errors, "errors": errors, "report": normalized}


__all__ = [
    "HANDOVER_BLOCK_REASONS",
    "HANDOVER_ROLE_ORDER",
    "HANDOVER_SKIP_REASONS",
    "HANDOVER_STEPS",
    "HANDOVER_STEP_STATUSES",
    "JUDGE_HANDOVER_CONTRACT",
    "JUDGE_HANDOVER_VERSION",
    "handover_step_statuses",
    "normalize_handover_step",
    "normalize_judge_handover_report",
    "validate_judge_handover_report",
]
