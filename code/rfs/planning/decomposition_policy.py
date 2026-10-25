"""Turn a measured decomposability report into one bounded replan intent.

The measurement lives in :mod:`rfs.evaluation.decomposability`; this module only
decides what to do about it, and it decides the same way every time:

* a hard break (an unbound scientific node, a slot outside its panel, an ungrouped
  node) cannot be repaired locally, so it asks to re-segment at once;
* a soft break (crowded panels, connectors crossing a boundary) is first handled
  locally, and only asked to re-plan globally once the same soft signature has
  repeated for :data:`SOFT_ESCALATION_STREAK` rounds;
* a soft break that is entirely about connectors is routed, not laid out, because a
  node-level action cannot move any of its targets;
* ``evolution_scope == "block"`` never escalates: a run asked to stay local is
  downgraded to the closest legal local action, or to no action at all.

The intent is a plain bounded action, so it goes through the same
``select_bounded_action`` / ``apply_bounded_action`` whitelist and the same
accept/rollback comparison as an action the Judge proposed.
"""

from __future__ import annotations

from typing import Any, Iterable


DECOMPOSITION_POLICY_CONTRACT = "rfs.decomposition_action"
DECOMPOSITION_POLICY_VERSION = 1
SOFT_ESCALATION_STREAK = 2
GLOBAL_ACTIONS = {"resegment", "replan_structure", "relayout_global"}
CONNECTOR_METRICS = {"cross_panel_connectors"}


def _items(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _target_ids(violations: Iterable[dict[str, Any]], key: str) -> list[str]:
    """Collect one typed target list, because a node action cannot move a connector."""

    seen: list[str] = []
    for violation in violations:
        for value in violation.get(key) or []:
            text = str(value).strip()
            if text and text not in seen:
                seen.append(text)
    return seen


def _object_ids(violations: Iterable[dict[str, Any]]) -> list[str]:
    """Collect the measured objects, which are kept for the record rather than the action."""

    return _target_ids(violations, "object_ids")


def _only_crossed_connectors(violations: list[dict[str, Any]]) -> bool:
    """Report whether connector routing is the whole problem, so a node action cannot help."""

    return bool(violations) and all(str(item.get("metric") or "") in CONNECTOR_METRICS for item in violations)


def plan_decomposition_action(report: dict[str, Any] | None, *, evolution_scope: str = "joint") -> dict[str, Any]:
    """Return the bounded action this report justifies, or nothing when it is clean."""

    source = report if isinstance(report, dict) else {}
    violations = _items(source.get("violations"))
    hard = [item for item in violations if item.get("kind") == "hard"]
    soft = [item for item in violations if item.get("kind") == "soft"]
    repeat = source.get("repeat") if isinstance(source.get("repeat"), dict) else {}
    streak = int(repeat.get("soft_streak") or 0)
    scope = str(evolution_scope or "joint").strip().casefold()

    action, scope_type, escalation, reason = "", "", False, ""
    if hard:
        action, scope_type, escalation = "resegment", "global", True
        names = ", ".join(sorted(str(item.get("metric") or "") for item in hard))
        reason = f"a hard decomposability break cannot be repaired locally: {names}"
    elif soft and streak >= SOFT_ESCALATION_STREAK:
        action, scope_type, escalation = "replan_structure", "global", True
        names = ", ".join(sorted(str(item.get("metric") or "") for item in soft))
        reason = f"the same soft decomposability break repeated {streak} rounds: {names}"
    elif soft and _only_crossed_connectors(soft):
        # `relayout_slots` only ever moves nodes, so a break that is entirely about crossed
        # connectors used to be handed to an action that could not touch any of its targets.
        action, scope_type = "reroute", "connector"
        names = ", ".join(sorted(str(item.get("metric") or "") for item in soft))
        reason = f"the crossed connectors themselves have to be routed: {names}"
    elif soft:
        action, scope_type = "relayout_slots", "slot"
        names = ", ".join(sorted(str(item.get("metric") or "") for item in soft))
        reason = f"a local reflow should clear this soft decomposability break: {names}"

    targeted = [item for item in (hard or soft)]
    object_ids = _object_ids(targeted)
    target_node_ids = _target_ids(targeted, "target_node_ids")
    target_connector_ids = _target_ids(targeted, "target_connector_ids")
    if action and action in GLOBAL_ACTIONS and scope == "block":
        if soft:
            action, scope_type, escalation = "relayout_slots", "slot", False
            reason = f"block-only evolution: keeping the repair local instead of escalating ({reason})"
        else:
            action, scope_type, escalation, reason = "", "", False, f"block-only evolution: no local action can clear a hard break ({reason})"

    return {
        "contract": DECOMPOSITION_POLICY_CONTRACT,
        "version": DECOMPOSITION_POLICY_VERSION,
        "required": bool(action),
        "action": action,
        "scope_type": scope_type,
        "escalation": bool(escalation),
        "reason": reason,
        "evolution_scope": scope,
        "soft_streak": streak,
        "violations": violations,
        "object_ids": object_ids,
        "slot_ids": [value for value in object_ids if value.startswith("slot_")],
        "target_node_ids": target_node_ids,
        "target_connector_ids": target_connector_ids,
        "provenance": {
            "report_status": str(source.get("status") or ""),
            "report_round": source.get("round"),
        },
    }
