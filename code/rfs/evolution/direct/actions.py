"""Auditable bounded action execution over direct-pipeline planning state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ...contracts.evolution import MECHANISM_ACTIONS
from ...contracts.memory_parameters import layout_rules_from, router_rules_from
from .snapshots import state_hash


#: The bounded repair moves the advisory Judge may name. Kept to exactly these ten so
#: ``action_is_repair`` in the Judge-training data still means "the Judge chose a repair".
SUPPORTED_ACTIONS = {
    "reanalyze_semantics",
    "replan_structure",
    "replan_theme",
    "resegment",
    "relayout_global",
    "relayout_slots",
    "resize_nodes",
    "reroute",
    "repair_interfaces",
    "regen_assets",
}

#: What this executor accepts: the ten repairs plus the mechanism-level rules a Memory row may
#: carry. The second group changes no box here -- the rule travels on the action and the compile
#: that follows draws by it -- so they are not repairs and not part of the Judge's menu.
EXECUTABLE_ACTIONS = SUPPORTED_ACTIONS | MECHANISM_ACTIONS


def normalize_mechanism_rule(candidate: Any) -> dict[str, Any]:
    """Read a proposed mechanism-level rule, or ``{}`` when the compiler could not run it.

    A rule names one of the mechanism actions and carries only the parameters that action reads,
    so a reflection cannot smuggle a bounded repair's object ids in through the rule channel. An
    empty parameter block is refused too: a rule that says nothing is not a rule.
    """

    source = candidate if isinstance(candidate, dict) else {}
    action_type = str(source.get("type") or source.get("action") or "").strip()
    if action_type not in MECHANISM_ACTIONS:
        return {}
    if action_type == "router_constraint":
        rules = router_rules_from(source)
        return {"type": action_type, "apply_at": "ppt_compilation", "constraints": rules} if rules else {}
    rules = layout_rules_from(source)
    return {"type": action_type, "apply_at": "ppt_compilation", "params": rules} if rules else {}


def apply_program_action(
    state: dict[str, Any], action: dict[str, Any], *, retry_index: int = 0
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute non-geometric action semantics without changing scientific claims."""

    updated = deepcopy(state)
    action_type = str(action.get("type") or "")
    if action_type not in EXECUTABLE_ACTIONS:
        raise ValueError(f"unsupported direct evolution action: {action_type}")
    layout = updated.setdefault("layout", {})
    targets = {str(item) for item in action.get("edge_ids", []) + action.get("object_ids", [])}
    semantic_targets = {str(item) for item in action.get("semantic_node_ids", [])}
    changes: list[dict[str, Any]] = []

    if action_type == "reanalyze_semantics":
        # The re-extraction itself is `upstream_retry`'s; this pass only records what was
        # asked for. The request used to be appended to an `analysis_requests` list on the
        # state, which no module ever read and which made the state diff report a change,
        # so an escalation the workflow had already declined still read as an applied repair.
        changes.append({"object_type": "analysis_request", "after": {
            "semantic_node_ids": sorted(semantic_targets),
            "problem": str(action.get("problem") or ""),
            "retry_index": retry_index,
            "policy": "recheck_evidence_without_inventing_content",
        }})
    elif action_type in {"reroute", "repair_interfaces"}:
        for edge in layout.get("connectors", []):
            if not isinstance(edge, dict):
                continue
            edge_id = str(edge.get("id") or "")
            endpoints = {str(edge.get("source") or ""), str(edge.get("target") or "")}
            if targets and edge_id not in targets and not endpoints.intersection(semantic_targets | targets):
                continue
            before = deepcopy(edge)
            edge["path_percent"] = []
            edge["routing_revision"] = retry_index + 1
            if action_type == "repair_interfaces":
                edge["interface_repair_requested"] = True
            changes.append({"object_type": "connector", "object_id": edge_id, "before": before, "after": deepcopy(edge)})
    elif action_type in {"resegment", "replan_structure"}:
        # Nodes are re-boxed only after ``upstream_retry`` replaces the layout. This pass
        # records the request so the audit names the escalation; it must not count as a
        # state change when the re-run was declined.
        changes.append({"object_type": "structure_request" if action_type == "replan_structure" else "segmentation", "after": {
            "retry_index": retry_index,
            "problem": str(action.get("problem") or ""),
            "policy": "re_run_the_upstream_stage_that_owns_this_break",
        }})
    elif action_type == "replan_theme":
        # The theme planner is re-run by the workflow, which owns the provider call and the
        # panel styles the new theme produces. This pass records the request, exactly like
        # ``reanalyze_semantics`` does, so the state diff says nothing was recoloured here.
        changes.append({"object_type": "theme_request", "after": {
            "roles": sorted({str(item) for item in action.get("target_roles", []) if str(item)}),
            "problem": str(action.get("problem") or ""),
            "retry_index": retry_index,
            "policy": "replan_the_theme_against_the_measured_color_feedback",
        }})
    elif action_type in MECHANISM_ACTIONS:
        # A mechanism action moves nothing here. It states a rule the compile that follows draws
        # by, and that rule travels on the action itself, so recording the request is the whole
        # execution. Keeping it out of the state is what lets `applied` go on meaning "this pass
        # changed the state it was handed", which is how a no-op is caught.
        changes.append({"object_type": "mechanism_rule", "after": {
            "rule_type": action_type,
            "rule": deepcopy(action.get("constraints") or action.get("params") or {}),
            "retry_index": retry_index,
            "policy": "carry_the_rule_to_the_compiler_that_draws_the_next_figure",
        }})

    # `applied` has to mean "this pass changed the state it was handed". The geometric actions
    # are executed elsewhere and the escalations are executed by `upstream_retry`, so this pass
    # is a no-op for both, and claiming otherwise is what let an action that moved nothing
    # report success. What such an action asked for belongs in `program_changes`, which is the
    # record, not in the state, which is what the round compiles and diffs. Only the input's own
    # keys count, so the empty containers `setdefault` creates do not read as a change.
    comparable = {**{key: updated.get(key) for key in state}, **{key: value for key, value in updated.items() if key not in state and value}}
    original = {key: state.get(key) for key in state}
    return updated, {
        "action_type": action_type,
        "applied": state_hash(comparable) != state_hash(original),
        "program_changes": changes,
    }
