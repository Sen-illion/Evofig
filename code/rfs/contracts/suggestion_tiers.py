"""The two tiers one round's advice lands in: this paper's fix, and the rule for any paper.

A defect the advisory seat locates is answered twice in one answer, and the two halves travel
different distances.

The *local fix* is what this round does to this figure: the stage it re-plans, the requirement
that stage's prompt failed to state, and the numbers the seat read the defect off. It is run
here, measured here, and kept in this round's own record. It never leaves the paper.

The *portable rule* is the same lesson lifted to the rung a prompt patch lives on. It is not run
here. It is a candidate, and it reaches another paper only after the ladder has measured it
there. Until then it is offered back to the paper that wrote it and nowhere else.

Keeping the two in one record is what lets a reader tell "what we changed this round" from
"what we now believe in general". Those are the two halves of a self-evolving loop rather than
one, and a record that keeps only the first is an iteration, not an evolution.

The answer already carries both halves -- a defect is written as the repair this figure needs
and, separately, as the sentence another figure would be drawn by -- so the two tiers are read
out of one answer rather than asked for twice.
"""

from __future__ import annotations

from typing import Any

from .evolution import APPLY_AT_STAGES


SUGGESTION_TIERS_CONTRACT = "rfs.suggestion_tiers"
SUGGESTION_TIERS_VERSION = 1
#: The two tiers, named once so a record, a report and a test cannot disagree about them.
LOCAL_FIX_SCOPE = "this_paper"
PORTABLE_RULE_SCOPE = "any_paper"


def _text(value: Any) -> str:
    return str(value or "").strip()


def normalize_local_fix(raw: Any) -> dict[str, Any]:
    """Read the half of the answer that is about this figure alone.

    ``apply_at`` is the stage the round re-planned, ``requirement`` is the sentence that stage's
    prompt failed to state, and ``directive`` is the seat's own wording of it. ``evidence`` is
    the reading the seat named it from, so a later round can check the attribution rather than
    take it on trust. Whether it ran is recorded by the caller from whether the move actually
    reached the figure, never from the answer.
    """

    source = raw if isinstance(raw, dict) else {}
    return {
        "apply_at": _text(source.get("apply_at")),
        "requirement": _text(source.get("requirement")),
        "directive": _text(source.get("directive")),
        "action": dict(source.get("action")) if isinstance(source.get("action"), dict) else {},
        "evidence": [dict(item) for item in (source.get("evidence") or []) if isinstance(item, dict)],
        "executed": bool(source.get("executed")),
        "reason": _text(source.get("reason")),
    }


def normalize_portable_rule(raw: Any) -> dict[str, Any]:
    """Read the half of the answer that could be true of any paper.

    A rule that never rendered into a patch is still recorded, with the reason it did not, so
    "nothing travelled this round" and "the travelling half broke" read differently.
    """

    source = raw if isinstance(raw, dict) else {}
    return {
        "patch_id": _text(source.get("patch_id")),
        "apply_at": _text(source.get("apply_at")),
        "directive": _text(source.get("directive")),
        "class": _text(source.get("class")),
        "status": _text(source.get("status")),
        "condition": dict(source.get("condition")) if isinstance(source.get("condition"), dict) else {},
        "reason": _text(source.get("reason")),
    }


def suggestion_tiers_record(
    round_index: int,
    *,
    local_fix: Any,
    portable_rule: Any,
    authored_by: str = "",
    transferable_half: bool = False,
) -> dict[str, Any]:
    """Land one round's two tiers together, with what each of them is allowed to do.

    ``runs_here`` and ``travels`` are written down rather than left to a reader: the local fix is
    the one the round executes, and the portable rule is the one that may only be injected after
    a measurement promotes it. ``transferable_half`` says whether the answer carried the second
    half at all, which is what tells an unrendered patch apart from an unasked question.
    """

    local = normalize_local_fix(local_fix)
    portable = normalize_portable_rule(portable_rule)
    return {
        "contract": SUGGESTION_TIERS_CONTRACT,
        "version": SUGGESTION_TIERS_VERSION,
        "round": int(round_index),
        "local_fix": {**local, "runs_here": True, "scope": LOCAL_FIX_SCOPE},
        "portable_rule": {
            **portable,
            "runs_here": False,
            "scope": PORTABLE_RULE_SCOPE,
            "travels": bool(portable.get("patch_id")),
        },
        "authored_by": _text(authored_by),
        "transferable_half": bool(transferable_half),
    }


def validate_suggestion_tiers(record: dict[str, Any] | None) -> list[str]:
    """Check a two-tier record names stages the pipeline runs and keeps the tiers apart.

    The whole point of the record is that the tier which runs here and the tier which travels are
    not confused, so a record that has the local fix claiming it travels, or the portable rule
    claiming it ran, is a contract error rather than a wording choice.
    """

    source = record if isinstance(record, dict) else {}
    errors: list[str] = []
    local = source.get("local_fix") if isinstance(source.get("local_fix"), dict) else {}
    portable = source.get("portable_rule") if isinstance(source.get("portable_rule"), dict) else {}
    if not local:
        errors.append("a two-tier record must carry a local_fix")
    if not portable:
        errors.append("a two-tier record must carry a portable_rule, even when it rendered none")
    for name, tier in (("local_fix", local), ("portable_rule", portable)):
        stage = _text(tier.get("apply_at"))
        if stage and stage not in APPLY_AT_STAGES:
            errors.append(f"{name} names a stage outside the pipeline order: {stage}")
    if local and local.get("runs_here") is False:
        errors.append("a local_fix runs in the round that wrote it")
    if portable and portable.get("runs_here") is True:
        errors.append("a portable_rule may not run in the round that wrote it")
    if portable and str(portable.get("scope") or "") and str(portable.get("scope")) != PORTABLE_RULE_SCOPE:
        errors.append("a portable_rule is scoped to any paper, not to one")
    return errors


__all__ = [
    "LOCAL_FIX_SCOPE",
    "PORTABLE_RULE_SCOPE",
    "SUGGESTION_TIERS_CONTRACT",
    "SUGGESTION_TIERS_VERSION",
    "normalize_local_fix",
    "normalize_portable_rule",
    "suggestion_tiers_record",
    "validate_suggestion_tiers",
]