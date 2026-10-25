"""One round's stage trace: what each stage changed, and the numbers the figure moved.

The advisory role is asked to name the stage a defect entered at, and it can only answer that if
the round tells it what changed where. The trace is assembled from artifacts the run already
wrote -- the per-stage inheritance report, the action that ran, and the figure program before and
after -- so it costs no model call and adds no measurement of its own.
"""

from __future__ import annotations

from typing import Any

from ..composition.arrow_routing import count_connector_crossings
from ..contracts.evolution import STAGE_ORDER, citable_fields, normalize_field_name as _normalise_field
from ..contracts.fault_attribution import requirement_is_on_menu
from ..evaluation.connector_visibility import evaluate_connector_visibility
from ..planning.density_advisory import build_density_advisory

STAGE_TRACE_CONTRACT = "rfs.stage_trace"
STAGE_TRACE_VERSION = 1

STAGE_GATE_CONTRACT = "rfs.stage_gate"
STAGE_GATE_VERSION = 1
STAGE_GATE_FILE = "stage_gate.json"

#: The stages a coach may name, in the order the pipeline runs them.
TRACE_STAGES = STAGE_ORDER

#: Which trace stage a stage the run inherited or replanned belongs to.
REPLAY_STAGE_TO_TRACE_STAGE = {
    "method": "global_planning",
    "scientific_hierarchy": "global_planning",
    "relation_explanations": "global_planning",
    "reference_analysis": "global_planning",
    "design_seed": "global_planning",
    "figure_text_plan": "global_planning",
    "color_theme": "theme_planning",
    "icons": "icon_generation",
}

#: Which trace stage a repair action applies at.
ACTION_TRACE_STAGE = {
    "reanalyze_semantics": "global_planning",
    "replan_structure": "global_planning",
    "resegment": "segmentation",
    "relayout_global": "segmentation",
    "relayout_slots": "segmentation",
    "resize_nodes": "ppt_compilation",
    "repair_interfaces": "ppt_compilation",
    "reroute": "ppt_compilation",
    "regen_assets": "icon_generation",
    "replan_theme": "theme_planning",
    "router_constraint": "ppt_compilation",
    "layout_constraint": "ppt_compilation",
}

#: The numbers the trace measures, all of them read from the figure program.
METRIC_KEYS = (
    "node_count",
    "panel_count",
    "icon_count",
    "connector_count",
    "connector_issue_count",
    "crossing_count",
    "node_occupancy",
    "panel_occupancy",
    "sparse_panel_count",
    "wrapped_label_count",
)


def _wrapped_label_count(nodes: list[Any]) -> int:
    wrapped = 0
    for node in nodes:
        if not isinstance(node, dict):
            continue
        lines = node.get("label_lines")
        if isinstance(lines, list) and len(lines) > 1:
            wrapped += 1
            continue
        if "\n" in str(node.get("label") or ""):
            wrapped += 1
    return wrapped


def figure_metrics(program: dict[str, Any] | None) -> dict[str, Any]:
    """The deterministic numbers one figure has, read from its program alone."""

    source = program if isinstance(program, dict) else {}
    nodes = [item for item in (source.get("semantic_nodes") or source.get("nodes") or []) if isinstance(item, dict)]
    density = build_density_advisory({"panels": source.get("panels") or [], "nodes": nodes})
    visibility = evaluate_connector_visibility(source)
    return {
        "node_count": int(density.get("node_count") or 0),
        "panel_count": int(density.get("panel_count") or 0),
        "icon_count": len([item for item in source.get("slots") or [] if isinstance(item, dict)]),
        "connector_count": int(visibility.get("connector_count") or 0),
        "connector_issue_count": int(visibility.get("issue_count") or 0),
        "crossing_count": int(count_connector_crossings(source)),
        "node_occupancy": round(float(density.get("node_occupancy_estimate") or 0.0), 4),
        "panel_occupancy": round(float(density.get("panel_occupancy_estimate") or 0.0), 4),
        "sparse_panel_count": len(density.get("sparse_panels") or []),
        "wrapped_label_count": _wrapped_label_count(nodes),
    }


def _metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key in METRIC_KEYS:
        old, new = before.get(key), after.get(key)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)) and not isinstance(old, bool) and not isinstance(new, bool):
            delta[key] = round(float(new) - float(old), 4)
    return delta


def action_trace_stage(action: dict[str, Any] | None) -> str:
    """Which trace stage the round's own action applies at."""

    source = action if isinstance(action, dict) else {}
    return ACTION_TRACE_STAGE.get(str(source.get("type") or ""), "")


def _moved(delta: dict[str, float]) -> list[str]:
    return [key for key, value in delta.items() if value]


def build_stage_trace(
    *,
    round_index: int,
    before_program: dict[str, Any] | None,
    after_program: dict[str, Any] | None,
    reuse_report: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Say, stage by stage, whether this round changed anything there and why."""

    metrics_before = figure_metrics(before_program)
    metrics_after = figure_metrics(after_program)
    delta = _metric_delta(metrics_before, metrics_after)
    outcomes = (reuse_report or {}).get("stage_outcomes") if isinstance(reuse_report, dict) else {}
    outcomes = outcomes if isinstance(outcomes, dict) else {}
    reasons: dict[str, list[str]] = {stage: [] for stage in TRACE_STAGES}
    for replay_stage, trace_stage in REPLAY_STAGE_TO_TRACE_STAGE.items():
        outcome = outcomes.get(replay_stage) if isinstance(outcomes.get(replay_stage), dict) else {}
        if str(outcome.get("status") or "") != "replanned":
            continue
        detail = str(outcome.get("reason") or "")
        reasons[trace_stage].append(f"{replay_stage} was replanned this round" + (f": {detail}" if detail else ""))
    action_stage = action_trace_stage(action)
    if action_stage:
        action_type = str((action or {}).get("type") or "")
        reasons[action_stage].append(f"this round's action is {action_type}, which applies at {action_stage}")
    moved = _moved(delta)
    if moved:
        reasons["ppt_compilation"].append("the compiled figure moved: " + ", ".join(f"{key} {delta[key]:+g}" for key in moved))
    stages = [
        {"stage": stage, "changed": bool(reasons[stage]), "reasons": reasons[stage]}
        for stage in TRACE_STAGES
    ]
    return {
        "contract": STAGE_TRACE_CONTRACT,
        "version": STAGE_TRACE_VERSION,
        "round": int(round_index),
        "stages": stages,
        "changed_stages": [item["stage"] for item in stages if item["changed"]],
        "metrics": {"before": metrics_before, "after": metrics_after, "delta": delta},
        "action": {"type": str((action or {}).get("type") or ""), "apply_at": action_stage},
    }


def stage_gate_report(
    *,
    round_index: int,
    trace: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What each stage did this round, beside the reading that stage has.

    The trace already says, stage by stage, whether the round changed anything there. This puts
    the number that stage's own report measured beside it: the paper's declared budget for the
    reading, what the figure delivered for the content plan, and the figure's own metrics for the
    rest, which is all the stages below the planning layers have to report.

    Nothing here gates a round. The round it describes has already been accepted or rolled back,
    and a stage that did nothing is a fact to read -- usually the fact that answers "why did
    nothing happen" -- not a failure to be scored. It is written down because "the content plan
    did nothing this round" cannot be read without also reading how much there was to plan with.
    """

    source = trace if isinstance(trace, dict) else {}
    measured = coverage if isinstance(coverage, dict) else {}
    budget = (measured.get("paper_budget") or {}).get("declared") or {}
    delivered = measured.get("delivered") or {}
    def _reading(values: dict[str, Any]) -> dict[str, Any]:
        """Only the numbers this stage actually read. A round with no report reads nothing."""

        return {key: value for key, value in values.items() if value is not None}

    readings = {
        "paper_reading": _reading(
            {
                "declared_nodes": budget.get("nodes"),
                "declared_edges": budget.get("edges"),
                "declared_modules": budget.get("modules"),
                "declared_relations": budget.get("relations"),
            }
        ),
        "content_planning": _reading(
            {
                "delivered_nodes": delivered.get("nodes"),
                "delivered_edges": delivered.get("edges"),
                "bound_edges": delivered.get("bound_edges"),
            }
        ),
    }
    stages = []
    for item in source.get("stages") or []:
        if not isinstance(item, dict):
            continue
        stage = str(item.get("stage") or "")
        stages.append(
            {
                "stage": stage,
                "changed": bool(item.get("changed")),
                "reasons": [str(reason) for reason in item.get("reasons") or []],
                "reading": dict(readings.get(stage) or {}),
            }
        )
    metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
    return {
        "contract": STAGE_GATE_CONTRACT,
        "version": STAGE_GATE_VERSION,
        "round": int(round_index),
        "stages": stages,
        "moved_stages": [item["stage"] for item in stages if item["changed"]],
        "figure_metrics": {
            "before": dict(metrics.get("before") or {}),
            "after": dict(metrics.get("after") or {}),
            "delta": dict(metrics.get("delta") or {}),
        },
        "content_coverage_status": str(measured.get("status") or "not_measured"),
        "advisories": [str(item.get("code") or "") for item in measured.get("advisories") or [] if isinstance(item, dict)],
        "note": "recorded only: nothing in this file gates the round it describes",
    }


def compact_stage_trace(trace: dict[str, Any] | None) -> dict[str, Any]:
    """The part of a trace a prompt may carry: the stages that changed, and by how much.

    A round that changed nothing returns an empty mapping, so the prompt renders exactly as it
    did before the trace existed. Every prompt-facing key is derived from the trace itself.
    """

    source = trace if isinstance(trace, dict) else {}
    stages = [
        {"stage": str(item.get("stage") or ""), "reasons": [str(reason) for reason in item.get("reasons") or []]}
        for item in source.get("stages") or []
        if isinstance(item, dict) and item.get("changed")
    ]
    if not stages:
        return {}
    metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
    delta = {key: value for key, value in (metrics.get("delta") or {}).items() if value}
    # The measured fields travel with the deltas: the advisory seat has to cite one of them as
    # evidence, and a field that did not move is exactly the evidence for a defect the round did
    # not fix. Without the names in front of it the seat can only guess, and a guess is discarded
    # as unreadable before it is ever looked at.
    measured = {key: metrics["after"][key] for key in METRIC_KEYS if key in (metrics.get("after") or {})}
    return {
        "round": source.get("round"),
        "changed_stages": [item["stage"] for item in stages],
        "known_stages": [str(item.get("stage") or "") for item in source.get("stages") or [] if isinstance(item, dict)],
        "stages": stages,
        "metric_delta": delta,
        "metrics": measured,
        "action": dict(source.get("action") or {}),
    }


def _evidence_is_readable(item: dict[str, Any], stage: str, known_stages: set[str], known_fields: set[str]) -> bool:
    """Whether one cited item names a stage the round ran and a field it showed the seat.

    A measured zero is a reading, not a missing field. ``crossing_count`` is 0 in exactly the
    rounds where the connectors are fine, and reading the value with ``or ""`` turned that zero
    into a blank, so a claim that cited the round's own measurements was thrown away as unreadable
    and the attribution the trace had just accepted never landed. Only a value that is genuinely
    absent counts as missing.
    """

    named = str(item.get("stage") or stage).strip()
    field = _normalise_field(item.get("field"))
    value = item.get("value")
    if named not in known_stages or field not in known_fields:
        return False
    return value is not None and not (isinstance(value, str) and not value.strip())


def _requirement_status(stage: str, requirement: str) -> str:
    """Whether a claim named the requirement the stage failed to write, and whether it is one of the menu's.

    ``unstated`` is the answer to a seat that named a stage and stopped. ``invented`` is a
    sentence the stage was never offered, which is kept because a requirement nobody wrote down
    is still a place to look, and flagged because the ledger has to be able to tell the two
    apart when it scores how often the location was right.
    """

    if not stage:
        return ""
    if not requirement:
        return "unstated"
    return "on_menu" if requirement_is_on_menu(stage, requirement) else "invented"


def attribution_verdict(
    judgement: dict[str, Any] | None,
    trace: dict[str, Any] | None,
    *,
    shown: Any = None,
    shown_fields: Any = None,
    stages: Any = None,
) -> dict[str, Any]:
    """Decide whether an attribution names a stage the trace has and evidence it can be read against.

    A claim that names no stage and offers no evidence is not a rejection -- the coach is allowed
    to say "this figure's defect entered nowhere in particular". A claim that does point somewhere
    must point at a stage the round ran and at a field the round put in front of the seat; anything
    else is discarded rather than kept as if it were evidence.

    ``shown_fields`` are the field names the prompt itself carried, and ``shown`` is the
    evidence block the same round handed the seat -- its colour report, its
    grounding Gate, the advice it offered -- so a claim about one of those is read against what
    the seat could actually see. The readable items survive even when another item of the same
    claim could not be read, which is what stops one unreadable line from discarding a defect the
    coach had located correctly; the dropped items travel beside them so the ledger can show what
    was thrown away.

    ``stages`` names the layers a claim may point at when the question is not about one round's
    trace: the boundary review reads a finished figure and asks which layer the work should go
    back to, so the caller hands the layer order over and the same validator checks the claim
    instead of a second one being written beside it.
    """

    source = judgement if isinstance(judgement, dict) else {}
    stage = str(source.get("fault_stage") or "").strip()
    requirement = str(source.get("fault_requirement") or "").strip()
    raw = source.get("fault_evidence")
    evidence = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    if not stage and not evidence:
        return {"ok": True, "reason": "", "attribution": {}}
    # A claim that names a stage but no requirement is kept rather than refused. The stage and its
    # readable evidence are the expensive half -- they say where the fix goes and what was
    # measured -- and throwing them away over a missing sentence would cost the round its
    # location. What must not happen is the omission reading as a complete answer, so the half
    # that is missing is named on the claim the ledger stores and the metric scores.
    attribution = {"fault_stage": stage, "fault_requirement": requirement, "fault_requirement_status": _requirement_status(stage, requirement), "fault_evidence": evidence}
    known_stages = {str(item.get("stage") or "") for item in (trace or {}).get("stages") or [] if isinstance(item, dict)}
    if stages is not None:
        known_stages = {str(item) for item in stages if str(item)}
    metrics = (trace or {}).get("metrics") if isinstance((trace or {}).get("metrics"), dict) else {}
    metric_fields = {str(key) for key in (metrics.get("after") or {})}
    # The trace is shown to the seat inside ``process_context``, and it cites a metric by the path
    # it read rather than by the bare key: live rounds named
    # ``process_context.stage_trace.metrics.node_occupancy`` and had the claim thrown away for a
    # spelling. A path the round itself wrote to is as checkable as the key at its end, so the
    # containers the trace is shown under are accepted too.
    # A caller that recorded the field names its own prompt carried hands them over
    # directly: the seat knows which blocks it rendered and the workflow does not, and
    # walking a payload again would only be a second opinion about the same question.
    known_fields = metric_fields | (set(shown_fields) if shown_fields else citable_fields(shown))
    known_fields |= {f"{prefix}.{key}" for prefix in ("metrics", "stage_trace.metrics", "process_context.stage_trace.metrics") for key in metric_fields}
    if not known_stages:
        return {"ok": False, "reason": "no_stage_trace", "attribution": attribution}
    if stage not in known_stages:
        return {"ok": False, "reason": "fault_stage_not_in_trace", "attribution": attribution}
    if not evidence:
        return {"ok": False, "reason": "fault_evidence_missing", "attribution": attribution}
    readable = [item for item in evidence if _evidence_is_readable(item, stage, known_stages, known_fields)]
    dropped = [item for item in evidence if item not in readable]
    if not readable:
        return {"ok": False, "reason": "fault_evidence_unreadable", "attribution": {**attribution, "fault_evidence_dropped": dropped}}
    kept = {"fault_stage": stage, "fault_requirement": requirement, "fault_requirement_status": _requirement_status(stage, requirement), "fault_evidence": readable}
    if dropped:
        kept["fault_evidence_dropped"] = dropped
    return {"ok": True, "reason": "", "attribution": kept}


def ledger_attribution(verdict: dict[str, Any] | None) -> dict[str, Any]:
    """The attribution a round ledger records, whether or not the trace could read it.

    A claim the trace refused is dropped from the judgement before it is stored, so the round
    ledger needs a line of its own for it: without one, "the coach placed the fault nowhere" and
    "the coach placed it somewhere the trace could not read" read the same way in
    ``evolution_rounds.json``, and a discarded guess is indistinguishable from silence. The
    claim itself is not repeated here -- the rejection event carries it in full -- only the fact
    and the reason.
    """

    source = verdict if isinstance(verdict, dict) else {}
    kept = bool(source.get("ok"))
    claim = source.get("attribution") if isinstance(source.get("attribution"), dict) else {}
    line = {
        "fault_stage": str(claim.get("fault_stage") or "") if kept else "",
        "fault_requirement": str(claim.get("fault_requirement") or "") if kept else "",
        "fault_requirement_status": str(claim.get("fault_requirement_status") or "") if kept else "",
        "fault_evidence": [dict(item) for item in (claim.get("fault_evidence") or []) if isinstance(item, dict)] if kept else [],
        "attribution_rejected": "" if kept else str(source.get("reason") or ""),
    }
    # A kept claim that also carried an item the gate could not read says so, so "the coach
    # pointed here" and "and this part of its pointing was thrown away" stay separable.
    dropped = [dict(item) for item in (claim.get("fault_evidence_dropped") or []) if isinstance(item, dict)]
    if kept and dropped:
        line["fault_evidence_dropped"] = dropped
    return line


__all__ = [
    "ACTION_TRACE_STAGE",
    "METRIC_KEYS",
    "STAGE_GATE_CONTRACT",
    "STAGE_GATE_FILE",
    "STAGE_GATE_VERSION",
    "REPLAY_STAGE_TO_TRACE_STAGE",
    "STAGE_TRACE_CONTRACT",
    "STAGE_TRACE_VERSION",
    "TRACE_STAGES",
    "action_trace_stage",
    "attribution_verdict",
    "citable_fields",
    "build_stage_trace",
    "compact_stage_trace",
    "figure_metrics",
    "ledger_attribution",
    "stage_gate_report",
]
