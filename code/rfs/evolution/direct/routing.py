"""Route Memory rows to the stage that runs them and settle the conflicts between them.

Retrieval answers which records could apply to a state; routing answers which record the
next round actually runs. Rows are grouped by the stage they apply at, and the rows that
compete for one stage are ordered by a fixed precedence -- the boundary a record declares,
the median repair gain it measured, then how many cases support it -- so the same library
always produces the same injection, and a conflict can be audited afterwards.

An injected action keeps an object id only when the paper in front of it still has that
object, so a repair learned on another figure contributes its action and its boundary
instead of binding to ids that never existed here.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Iterable

from ...contracts.memory_parameters import INJECTABLE_PARAMETERS, copy_injectable_parameters, injection_blocked_reason
from ...contracts.repair_actions import prescription_cannot_contrast
from .experience import stage_for_action

ROUTING_CONTRACT = "rfs.experience_routing"
ROUTING_VERSION = 1
INJECTED_SOURCE = "memory_retrieval"
SEED_SOURCE = "seed_experience"


def _number(value: Any) -> float | None:
    """Read a measured number, keeping "not measured" distinct from a measured zero."""

    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _texts(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _evidence(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("evidence") if isinstance(row.get("evidence"), dict) else {}


def experience_id(row: dict[str, Any]) -> str:
    """Read the Memory id a retrieval row stands for."""

    return str(_evidence(row).get("experience_id") or row.get("experience_id") or "")


def gains(row: dict[str, Any]) -> list[float]:
    """Collect every repair gain a row carries, so precedence reads measurements only."""

    outcome = row.get("outcome") if isinstance(row.get("outcome"), dict) else {}
    values = [row.get("repair_gain"), _evidence(row).get("repair_gain"), outcome.get("score_delta"), outcome.get("overall_score_delta")]
    return [number for number in (_number(value) for value in values) if number is not None]


def median_gain(row: dict[str, Any]) -> float:
    """Return the median measured gain of a row, or 0.0 when it measured none."""

    measured = gains(row)
    return round(float(median(measured)), 4) if measured else 0.0


def support(row: dict[str, Any]) -> int:
    """Return how many cases back a row."""

    count = _evidence(row).get("support")
    if isinstance(count, int) and not isinstance(count, bool):
        return max(0, count)
    cases = row.get("validated_cases")
    return len(cases) if isinstance(cases, list) else 0


def boundary_tier(row: dict[str, Any], *, required: Iterable[Any] = ()) -> int:
    """Rank the boundary a row declares against the boundary the current state requires.

    ``forbidden_changes`` is the list of edits a repair must not make. A record that
    declares such a boundary is safer to inject than one that declares none, and a record
    that covers everything the state forbids is safer still. It reads as 2 when the
    declared boundary covers what the state requires, 1 when it declares a boundary that
    does not cover it, and 0 when it declares none.
    """

    declared = {item.casefold() for item in _texts(row.get("forbidden_changes"))}
    demanded = {item.casefold() for item in _texts(required)}
    if not declared:
        return 0
    return 2 if demanded <= declared else 1


def precedence_key(row: dict[str, Any], *, required: Iterable[Any] = ()) -> tuple:
    """Return the fixed order two rows competing for one stage are resolved in.

    The trailing id is only there so an exact tie is still resolved deterministically.
    """

    return (-boundary_tier(row, required=required), -median_gain(row), -support(row), experience_id(row))


def _winning_reason(row: dict[str, Any], *, required: Iterable[Any] = ()) -> str:
    boundary = {
        2: "it declares the boundary this state requires",
        1: "it declares a boundary this state does not require",
        0: "it declares no boundary",
    }[boundary_tier(row, required=required)]
    return f"{boundary}, median repair gain {median_gain(row):+}, {support(row)} supporting case(s)"


def route_by_stage(rows: Iterable[dict[str, Any]], *, statuses: Iterable[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Group Memory rows by the stage they apply at, keeping only eligible statuses."""

    allowed = {str(item) for item in statuses} if statuses else None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if allowed is not None and str(row.get("status") or "") not in allowed:
            continue
        grouped.setdefault(str(row.get("apply_at") or ""), []).append(row)
    return {name: grouped[name] for name in sorted(grouped)}


def route_experiences(rows: Iterable[dict[str, Any]], *, stage: str = "", limit: int = 1, required_forbidden: Iterable[Any] = (), statuses: Iterable[str] | None = None) -> dict[str, Any]:
    """Rank the rows that apply at one stage and report why the winner won."""

    considered = [row for row in rows if isinstance(row, dict)]
    grouped = route_by_stage(considered, statuses=statuses)
    demanded = _texts(required_forbidden)
    stage_name = str(stage or "")
    ranked = sorted(grouped.get(stage_name, []), key=lambda row: precedence_key(row, required=demanded))
    selected = ranked[: max(0, int(limit))]
    if not stage_name:
        reason = "no stage was named, so no experience could be routed"
    elif not ranked:
        reason = f"no experience applies at {stage_name}"
    else:
        reason = f"{len(ranked)} experience(s) apply at {stage_name}; {experience_id(selected[0])} won because {_winning_reason(selected[0], required=demanded)}"
    return {
        "rows": selected,
        "report": {
            "contract": ROUTING_CONTRACT,
            "version": ROUTING_VERSION,
            "stage": stage_name,
            "stages": {name: len(items) for name, items in grouped.items()},
            "considered": len(considered),
            "required_forbidden": demanded,
            "conflicts": [
                {
                    "experience_id": experience_id(row),
                    "boundary_tier": boundary_tier(row, required=demanded),
                    "median_repair_gain": median_gain(row),
                    "support": support(row),
                    "selected": position < len(selected),
                }
                for position, row in enumerate(ranked)
            ],
            "selected": [experience_id(row) for row in selected],
            "reason": reason,
        },
    }


def injection_action(row: dict[str, Any], *, known_ids: Iterable[Any] = (), stage: str = "") -> dict[str, Any]:
    """Turn one routed Memory row into the bounded action the round loop executes.

    Object ids survive only when ``known_ids`` still holds them: a repair learned on a
    different figure keeps its action, its bounds and its declared boundary, and lets the
    executor pick a target here rather than claiming objects this paper never had.
    """

    source = row.get("action") if isinstance(row.get("action"), dict) else {}
    strategy = row.get("strategy") if isinstance(row.get("strategy"), dict) else {}
    identifier = experience_id(row)
    action_type = str(source.get("type") or source.get("action") or "").strip()
    if not action_type or action_type == "initial":
        return {"action": None, "experience_id": identifier, "dropped_ids": [], "reason": "the routed experience carries no executable action", "stage": str(stage or row.get("apply_at") or "")}
    known = {str(item) for item in known_ids if str(item)}
    action: dict[str, Any] = {"type": action_type, "scope_type": str(source.get("scope_type") or "global")}
    dropped: list[str] = []
    for key in ("semantic_node_ids", "object_ids", "slot_ids", "edge_ids"):
        values = _texts(source.get(key))
        dropped.extend(item for item in values if item not in known)
        kept = [item for item in values if item in known]
        if kept:
            action[key] = kept
    copy_injectable_parameters(source, action)
    label = str(row.get("applicability") or source.get("applicability") or "").strip().casefold()
    if label in {"common", "domain"}:
        action["applicability"] = label
    action["problem"] = str(source.get("problem") or strategy.get("instruction") or "")
    action["forbidden_changes"] = _texts(row.get("forbidden_changes")) or _texts(source.get("forbidden_changes"))
    action["experience_id"] = identifier
    return {
        "action": action,
        "experience_id": identifier,
        "dropped_ids": sorted(dict.fromkeys(dropped)),
        "reason": "the experience's action was injected",
        "stage": str(stage or row.get("apply_at") or stage_for_action(action_type)),
    }


def inject_experience(rows: Iterable[dict[str, Any]], action: dict[str, Any] | None, *, known_ids: Iterable[Any] = (), limit: int = 1, statuses: Iterable[str] | None = None) -> dict[str, Any]:
    """Return the action to run instead of the proposed one, plus what chose it.

    The proposed action names the stage, so an experience is only injected before the stage
    whose repair it was measured on. The boundary the current state demands comes from that
    same proposed action, because the Judge that selected it is what forbids an edit here.
    Nothing routed means the proposal stands, so an empty library changes nothing.
    """

    proposed = action if isinstance(action, dict) else {}
    stage = stage_for_action(proposed.get("type"))
    routed = route_experiences(rows, stage=stage, limit=limit, required_forbidden=_texts(proposed.get("forbidden_changes")), statuses=statuses)
    if not routed["rows"]:
        return {"action": None, "injection": {"experience_id": "", "stage": stage, "source": "none", "status": "", "reason": routed["report"]["reason"]}, "report": routed["report"]}
    chosen = injection_action(routed["rows"][0], known_ids=known_ids, stage=stage)
    if not chosen["action"]:
        return {"action": None, "injection": {"experience_id": "", "stage": stage, "source": "none", "status": "", "reason": chosen["reason"]}, "report": routed["report"]}
    blocked = prescription_cannot_contrast(chosen["action"]) or injection_blocked_reason(chosen["action"])
    if blocked:
        return {
            "action": None,
            "injection": {
                "experience_id": chosen["experience_id"],
                "stage": stage,
                "source": "none",
                "status": str(routed["rows"][0].get("status") or ""),
                "reason": blocked,
                "dropped_object_ids": chosen["dropped_ids"],
            },
            "report": routed["report"],
        }
    return {
        "action": chosen["action"],
        "injection": {
            "experience_id": chosen["experience_id"],
            "stage": stage,
            "source": INJECTED_SOURCE,
            # The rung is recorded with the injection so a run says which wall its evidence
            # got over, not only which row it used.
            "status": str(routed["rows"][0].get("status") or ""),
            "reason": routed["report"]["reason"],
            "dropped_object_ids": chosen["dropped_ids"],
        },
        "report": routed["report"],
    }


def seed_injection(action: dict[str, Any] | None, experience_id: str, *, status: str = "") -> dict[str, Any]:
    """Describe the injection a run that was handed one validated experience starts from."""

    return {
        "experience_id": str(experience_id or ""),
        "stage": stage_for_action((action or {}).get("type")),
        "source": SEED_SOURCE,
        "status": str(status or ""),
        "reason": "the run was seeded with this experience",
    }


__all__ = [
    "INJECTABLE_PARAMETERS",
    "INJECTED_SOURCE",
    "ROUTING_CONTRACT",
    "ROUTING_VERSION",
    "SEED_SOURCE",
    "boundary_tier",
    "experience_id",
    "gains",
    "inject_experience",
    "injection_action",
    "median_gain",
    "precedence_key",
    "route_by_stage",
    "route_experiences",
    "seed_injection",
    "support",
]
