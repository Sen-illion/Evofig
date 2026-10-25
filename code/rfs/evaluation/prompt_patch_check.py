"""Enforce the prohibitions a promoted prompt patch declares, deterministically.

A prohibition is only worth having if a machine checks it. Every prohibited patch carries
a condition over one measured decomposability metric, and this module decides -- from the
report the same round already measured, with no provider call -- whether the compiled
candidate broke it. A broken prohibition becomes a blocker, so the round leaves through
the accept/rollback path that already exists instead of a new one.

Prohibitions that name no measurable metric are reported as unchecked rather than silently
passing, because a patch nobody can verify is not enforcement.
"""

from __future__ import annotations

import operator
from typing import Any

from ..contracts.prompt_patch import PROHIBITION_OPERATORS, is_enforceable_prohibition, normalize_prompt_patch


PROHIBITED_CHECK_CONTRACT = "rfs.prompt_patch_prohibition_check"
PROHIBITED_CHECK_VERSION = 1
OPERATORS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt, "==": operator.eq}
assert set(OPERATORS) == set(PROHIBITION_OPERATORS), "the enforceable operators must match the patch contract"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prohibition_conditions(patches: Any) -> list[dict[str, Any]]:
    """Read the machine-checkable conditions out of the prohibited patches."""

    conditions: list[dict[str, Any]] = []
    for item in patches if isinstance(patches, (list, tuple)) else []:
        patch = normalize_prompt_patch(item if isinstance(item, dict) else {})
        if patch["class"] != "prohibited":
            continue
        condition = patch["condition"] if isinstance(patch["condition"], dict) else {}
        metric = str(condition.get("metric") or "").strip()
        op = str(condition.get("op") or "").strip()
        bound = _number(condition.get("value"))
        conditions.append(
            {
                "patch_id": patch["patch_id"],
                "directive": patch["directive"],
                "metric": metric,
                "op": op,
                "value": bound,
                "checkable": is_enforceable_prohibition(condition),
            }
        )
    return conditions


def check_prohibited_patches(report: dict[str, Any] | None, patches: Any) -> dict[str, Any]:
    """Return whether the compiled candidate violates any declared prohibition."""

    metrics = (report or {}).get("metrics") if isinstance((report or {}).get("metrics"), dict) else {}
    checked: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    unchecked: list[dict[str, Any]] = []
    for condition in prohibition_conditions(patches):
        if not condition["checkable"]:
            unchecked.append({"patch_id": condition["patch_id"], "reason": "the prohibition names no measurable metric bound", "directive": condition["directive"]})
            continue
        measured = _number(metrics.get(condition["metric"]))
        if measured is None:
            unchecked.append({"patch_id": condition["patch_id"], "reason": f"{condition['metric']} was not measured this round", "directive": condition["directive"]})
            continue
        row = {
            "patch_id": condition["patch_id"],
            "metric": condition["metric"],
            "op": condition["op"],
            "value": condition["value"],
            "measured": round(measured, 4),
        }
        checked.append(row)
        if not OPERATORS[condition["op"]](measured, float(condition["value"])):
            violations.append(
                {
                    **row,
                    "directive": condition["directive"],
                    "reason": f"{condition['metric']}={row['measured']} breaks the prohibition {condition['metric']} {condition['op']} {condition['value']}",
                }
            )
    return {
        "contract": PROHIBITED_CHECK_CONTRACT,
        "version": PROHIBITED_CHECK_VERSION,
        "ok": not violations,
        "checked": checked,
        "violations": violations,
        "unchecked": unchecked,
        "blockers": [f"prohibited prompt patch {item['patch_id']} violated: {item['reason']}" for item in violations],
    }


__all__ = [
    "OPERATORS",
    "PROHIBITED_CHECK_CONTRACT",
    "PROHIBITED_CHECK_VERSION",
    "check_prohibited_patches",
    "prohibition_conditions",
]
