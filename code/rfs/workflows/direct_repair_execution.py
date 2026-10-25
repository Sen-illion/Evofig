"""Cross-stage execution of one bounded direct-evolution repair action."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts.evolution import FIGURE_MOVE_EVIDENCE_CONTRACT, MECHANISM_RULE_REPORT_CONTRACT
from ..evolution.direct.actions import apply_program_action
from ..evolution.direct.controller import apply_bounded_action
from ..evolution.direct.snapshots import create_state_snapshot, diff_state_snapshots
from ..utils import write_json


MECHANISM_RULE_REPORT_FILE = "mechanism_rule_report.json"

#: Where one mechanism rule's compile-time application is written down, and the keys inside that
#: file which name what the compiler could apply and what it could not satisfy. The rule rides on
#: the action and the compile that draws the figure is what honours it, so those two files are
#: the only place the two halves of the mechanism meet.
_MECHANISM_COMPILE_EVIDENCE = {
    "layout_constraint": ("layout_constraint_report.json", "applied", "unmet"),
    "router_constraint": ("arrow_quality_report.json", "router_constraint_applied", "router_constraint_unmet"),
}


def execute_direct_repair(
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    round_index: int,
    retry_index: int = 0,
    domain_profile: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Execute, diff, and audit exactly one supported bounded action."""

    before = create_state_snapshot(state, label="before_action", round_index=round_index)
    layout, icons, geometric = apply_bounded_action(
        state.get("layout") or {},
        state.get("icons") or {},
        action,
        retry_index=retry_index,
        domain_profile=domain_profile,
    )
    intermediate = {**state, "layout": layout, "icons": icons}
    updated, program = apply_program_action(intermediate, action, retry_index=retry_index)
    after = create_state_snapshot(updated, label="after_action", round_index=round_index)
    diff = diff_state_snapshots(before, after)
    scientific_unchanged = before["state"].get("method") == after["state"].get("method")
    applied_budget = geometric.get("applied_budget") if isinstance(geometric.get("applied_budget"), dict) else {}
    executed = dict(action)
    executed.update(applied_budget)
    report = {
        "contract": "rfs.direct_action_execution",
        "version": 1,
        "action": executed,
        "action_type": str(action.get("type") or ""),
        # "applied" is the measured state change, not the intent: a geometric pass that
        # matched no target and a program pass that changed nothing both report False.
        "applied": bool(diff["changed"]),
        "geometric_applied": bool(geometric.get("changed_node_ids") or geometric.get("changed_asset_ids")),
        "program_applied": bool(program.get("applied")),
        "changed_node_ids": geometric.get("changed_node_ids") or [],
        "changed_asset_ids": geometric.get("changed_asset_ids") or [],
        "program_changes": program.get("program_changes") or [],
        "object_diff": diff,
        "preserve_checks": {
            "scientific_contract_unchanged": scientific_unchanged,
            "input_state_unmodified": True,
        },
        "retry_index": retry_index,
        "applied_budget": applied_budget,
    }
    return updated, report, before, after


def mechanism_rule_evidence(
    *,
    round_index: int,
    action: dict[str, Any],
    candidate_dir: str | Path,
    figure_diff: dict[str, Any] | None = None,
    round_cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What the compile did with one mechanism-level rule.

    A mechanism rule moves no box where it is executed: the executor records the request and the
    compile that draws the figure is what honours it. So "this figure was drawn by the rule" is
    invisible in the action's own report, and without this file it is invisible everywhere --
    which is exactly the state both mechanism actions were in while every part of them was
    already written. The two halves are kept apart because they can disagree: a rule the compiler
    applied and a figure that came out identical is a rule with no effect on this drawing, and
    that is worth being able to read off the record rather than infer.

    The record is built here rather than only written, because a reflection asked why a rule
    failed has to see it: the round already holds the answer when the reflection is called, and
    a reflection that was never shown it can only guess at its own reason.
    """

    action_type = str(action.get("type") or "")
    rule = action.get("constraints") if action_type == "router_constraint" else action.get("params")
    cost = round_cost if isinstance(round_cost, dict) else {}
    diff = figure_diff if isinstance(figure_diff, dict) else {}
    filename, applied_key, unmet_key = _MECHANISM_COMPILE_EVIDENCE.get(action_type, ("", "", ""))
    compile_report: dict[str, Any] = {}
    if filename:
        path = Path(candidate_dir) / filename
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                loaded = {}
            compile_report = loaded if isinstance(loaded, dict) else {}
    applied = list(compile_report.get(applied_key) or []) if compile_report else []
    unmet = list(compile_report.get(unmet_key) or []) if compile_report else []
    changed = bool(diff.get("changed"))
    record = {
        "contract": MECHANISM_RULE_REPORT_CONTRACT,
        "version": 1,
        "round": int(round_index),
        "rule_type": action_type,
        "rule": dict(rule) if isinstance(rule, dict) else {},
        "apply_at": str(action.get("apply_at") or "ppt_compilation"),
        "source": str(action.get("source") or ""),
        "compile_file_searched": filename,
        "compile_report_found": bool(compile_report),
        "compile_applied": applied,
        "compile_unmet": unmet,
        "figure_changed": changed,
        "changed_object_ids": [str(item) for item in (diff.get("changed_object_ids") or [])],
        "generation_calls": int(cost.get("figure_generations") or 0),
        "stage_reruns": int(cost.get("stage_reruns") or 0),
        # Both halves, because either alone reads as success: a rule the compiler applied to a
        # figure it left identical did not change this drawing.
        "took_effect": bool(applied) and changed,
    }
    return record


def write_mechanism_rule_report(
    round_dir: str | Path,
    *,
    round_index: int,
    action: dict[str, Any],
    candidate_dir: str | Path,
    figure_diff: dict[str, Any] | None = None,
    round_cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write down what one mechanism rule reached the compile, and what the compile did with it."""

    record = mechanism_rule_evidence(
        round_index=round_index,
        action=action,
        candidate_dir=candidate_dir,
        figure_diff=figure_diff,
        round_cost=round_cost,
    )
    write_json(Path(round_dir) / MECHANISM_RULE_REPORT_FILE, record)
    return record


def figure_move_evidence(
    *,
    round_index: int,
    action: dict[str, Any],
    figure_diff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What one move on the figure itself did to it.

    A compile-time rule leaves an implementation report of its own, because the compile that
    draws the figure is what honours it. A move that edits the figure program leaves none, and a
    reflection asked why such a round failed had nothing to answer against -- so both refusals
    that earn another attempt ("the move was not carried out", "it was carried out and broke
    something else") were rejected as uncheckable, and a losing suggestion was written off on one
    reading. The round already holds the evidence: the diff between the program it was handed and
    the program it produced. Only the halves an attribution is read against are recorded, so the
    shape matches the compile report's where it matters.
    """

    diff = figure_diff if isinstance(figure_diff, dict) else {}
    return {
        "contract": FIGURE_MOVE_EVIDENCE_CONTRACT,
        "version": 1,
        "round": int(round_index),
        "action_type": str(action.get("type") or ""),
        # The two fields an attribution is read against, under the same names the compile report
        # uses: a move on the figure is never "unmet", because there is no second step to miss --
        # only a program that either changed or did not.
        "compile_unmet": [],
        "figure_changed": bool(diff.get("changed")),
        "changed_object_ids": [str(item) for item in (diff.get("changed_object_ids") or [])],
        "source": "figure_program_diff",
    }


def record_unexecuted_repair(
    state: dict[str, Any],
    action: dict[str, Any],
    *,
    round_index: int,
    reason: str = "",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Audit an escalation the workflow declined, without moving boxes.

    ``replan_structure`` / ``resegment`` / ``reanalyze_semantics`` / ``replan_theme`` are
    owned by ``upstream_retry``. Falling through to the geometric executor used to nudge
    every node under the escalation's name when the re-run was refused.
    """

    before = create_state_snapshot(state, label="before_action", round_index=round_index)
    after = create_state_snapshot(state, label="after_action", round_index=round_index)
    diff = diff_state_snapshots(before, after)
    report = {
        "contract": "rfs.direct_action_execution",
        "version": 1,
        "action": dict(action),
        "action_type": str(action.get("type") or ""),
        "applied": False,
        "geometric_applied": False,
        "program_applied": False,
        "changed_node_ids": [],
        "changed_asset_ids": [],
        "program_changes": [],
        "object_diff": diff,
        "preserve_checks": {"scientific_contract_unchanged": True, "input_state_unmodified": True},
        "retry_index": 0,
        "applied_budget": {},
        "declined_reason": str(reason or ""),
    }
    return state, report, before, after
