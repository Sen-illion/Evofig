"""Bootstrap Judge for direct paper-to-PPTX evolution."""

from __future__ import annotations

from typing import Any
import json

from ..contracts import validate_figure_layout_plan, validate_icon_generation_plan, validate_method_figure_content
from ..contracts.evolution import REPAIR_ACTIONS
from ..contracts.repair_actions import attach_heuristic_budget
from ..providers import call_vlm_json, resolve_vlm_model
from .geometry_quality import evaluate_density_quality, evaluate_geometry_quality


#: Counts that mean the drawing is wrong.
#:
#: ``hierarchy_containment_count`` is deliberately absent: it counts the
#: children drawn inside their own container, which is the layout working as
#: intended, and the geometry gate records it instead of a collision.  Counting
#: it as a violation looked harmless while only the pass/fail gate used it, but
#: it also halves the cleanliness of a correctly nested figure, so a repair that
#: nests a child properly reads as a regression to the evolution target metric.
#:
#: The two pass counts read here are the connector-through-a-box counts. A crossing -- two
#: connector segments crossing each other -- is measured in ``rfs.composition.arrow_routing``
#: and is a different reading; the two are never mixed.
GEOMETRY_VIOLATION_KEYS = (
    "icon_text_overlap_count",
    "node_overlap_count",
    "connector_label_overlap_count",
    "text_pass_count",
    "box_pass_count",
    "node_boundary_violation_count",
)

#: The names the two pass counts carried before the rename. A report written then is read under
#: its own name rather than silently counting zero: the number is the same measurement and only
#: the label changed.
LEGACY_VIOLATION_KEYS = {
    "text_pass_count": "connector_path_text_crossing_count",
    "box_pass_count": "path_crossing_count",
}


def _violation_count(geometry: dict[str, Any], key: str) -> int:
    """One defect count, read under its current name or the one it carried before the rename."""

    value = geometry.get(key)
    if value is None:
        value = geometry.get(LEGACY_VIOLATION_KEYS.get(key, ""))
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _prescribe(action_type: str) -> dict[str, Any]:
    """Name a bounded repair the way the executor will run it, numbers included."""

    return attach_heuristic_budget({"action": action_type})


def _normalize_feedback(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict) and str(item.get("stage") or "").strip()] if isinstance(value, list) else []


def deterministic_dimension_scores(geometry: dict[str, Any], density: dict[str, Any]) -> dict[str, float]:
    """Expose the continuous quality values the pass/fail gate would otherwise discard.

    Every entry is 0..1 and higher-is-better, so a repair that improves geometry or
    density without removing a whole gate failure is still measurable.
    """
    violations = sum(_violation_count(geometry, key) for key in GEOMETRY_VIOLATION_KEYS)
    occupancy = max(0.0, float(density.get("canvas_occupancy") or 0.0))
    minimum_occupancy = float((density.get("targets") or {}).get("minimum_occupancy") or 0.0)
    panel_count = max(1, int(density.get("panel_count") or 0))
    sparse_panels = max(0, int(density.get("sparse_panel_count") or 0))
    return {
        "geometry_cleanliness": round(1.0 / (1.0 + violations), 4),
        "density_fitness": round(min(1.0, occupancy / minimum_occupancy) if minimum_occupancy > 0 else min(1.0, occupancy), 4),
        "panel_occupancy": round(max(0.0, 1.0 - sparse_panels / panel_count), 4),
        "canvas_occupancy": round(min(1.0, occupancy), 4),
        "editable_text_area": round(min(1.0, max(0.0, float(density.get("editable_text_area") or 0.0))), 4),
    }


def judge_direct_run(method: dict[str, Any], layout: dict[str, Any], icons: dict[str, Any], *, program: dict[str, Any] | None = None, mode: str = "heuristic", model: str | None = None, trained: bool = False, frozen: bool = False, prior_feedback: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return stage-attributed feedback while the trained Judge is unavailable."""
    checks = [
        ("semantic", validate_method_figure_content(method)),
        ("structure", validate_figure_layout_plan(layout)),
        ("asset", validate_icon_generation_plan(icons)),
    ]
    feedback = []
    dimension_scores: dict[str, float] = {}
    for stage, result in checks:
        if result.get("ok"):
            continue
        feedback.append({"stage": stage, "verdict": "fault", "severity": "blocker", "defect_key": f"{stage}:contract", "problem": "; ".join(result.get("errors", [])), "repair": [_prescribe("reanalyze_semantics" if stage == "semantic" else "replan_structure" if stage == "structure" else "regen_assets")]})
    if program is not None:
        # Both compile-stage faults below used to ask for `recompile`. Compiling the same state
        # again re-emits the same program, so the gate was naming a real defect and prescribing
        # a repair that could not touch it. A figure with almost no visible objects, and one
        # with no editable text, are both failures of the plan the compile stage was handed:
        # the structure is what decides which objects exist and which of them carry a label.
        visible = sum(len(program.get(key, []) or []) for key in ("panels", "semantic_nodes", "slots"))
        if visible < 3:
            feedback.append({"stage":"compile","verdict":"fault","severity":"blocker","defect_key":"compile:too_few_objects","problem":"final figure program contains too few visible objects","repair":[_prescribe("replan_structure")]})
        expected={str(item.get("asset_id")) for item in icons.get("icons",[]) if item.get("asset_id")}; placed={str(item.get("asset_id")) for item in program.get("slots",[]) if item.get("asset_id")}; missing=sorted(expected-placed)
        if missing:
            feedback.append({"stage":"asset","verdict":"fault","severity":"blocker","defect_key":"asset:unplaced_icons","problem":"unplaced icon assets: "+", ".join(missing),"repair":[_prescribe("repair_interfaces")]})
        text_program = program.get("text_program") if isinstance(program.get("text_program"), dict) else {}
        if not text_program.get("items"):
            feedback.append({"stage":"compile","verdict":"fault","severity":"major","defect_key":"compile:no_editable_text","problem":"final figure program contains no editable text items","repair":[_prescribe("replan_structure")]})
        geometry = evaluate_geometry_quality(program)
        if geometry["status"] != "pass":
            # The defect is named by what was measured, and its severity is the reader's, not
            # the repair's: the same geometry break was answered with reroute, relayout_slots or
            # relayout_global depending on which ids happened to be on hand, and the prescription
            # was the identity, so one defect read as three new ones. A figure whose only finding
            # is an overhang inside the tolerance carries no visible break and reads as advice.
            visible = int(geometry.get("visible_break_count") or 0)
            severity = "blocker" if visible else "major"
            defect_key = "interface:box_pass" if int(geometry.get("box_pass_count") or 0) or int(geometry.get("text_pass_count") or 0) else "interface:icon_text_overlap" if int(geometry.get("icon_text_overlap_count") or 0) else "interface:boundary_violation"
            edge_ids = sorted({str(item.get("connector_id") or "") for item in (geometry.get("box_passes") or []) + (geometry.get("text_passes") or []) if str(item.get("connector_id") or "")})
            slot_ids = sorted({str(item.get("slot_id") or "") for item in (geometry.get("icon_text_overlaps") or []) + (geometry.get("boundary_violations") or []) if str(item.get("slot_id") or "")})
            node_ids = sorted({str(item.get("node_id") or item.get("node_a") or item.get("node_b") or "") for item in (geometry.get("node_overlaps") or []) + (geometry.get("icon_text_overlaps") or []) if str(item.get("node_id") or item.get("node_a") or item.get("node_b") or "")})
            if edge_ids:
                repair = [_prescribe("reroute")]
                feedback.append({"stage": "interface", "verdict": "fault", "severity": severity, "defect_key": defect_key, "problem": f"Icon/text overlap={geometry['icon_text_overlap_count']}, box passes={geometry['box_pass_count']}", "edge_ids": edge_ids, "requested_actions": [{**repair[0], "edge_ids": edge_ids}], "repair": [{**repair[0], "edge_ids": edge_ids}]})
            elif slot_ids or node_ids:
                repair = [_prescribe("relayout_slots")]
                payload = {**repair[0], "slot_ids": slot_ids, "object_ids": node_ids or slot_ids}
                feedback.append({"stage": "interface", "verdict": "fault", "severity": severity, "defect_key": defect_key, "problem": f"Icon/text overlap={geometry['icon_text_overlap_count']}, boundary violations={geometry['node_boundary_violation_count']}", "slot_ids": slot_ids, "object_ids": node_ids or slot_ids, "requested_actions": [payload], "repair": [payload]})
            else:
                feedback.append({"stage": "interface", "verdict": "fault", "severity": severity, "defect_key": defect_key, "problem": f"Icon/text overlap={geometry['icon_text_overlap_count']}, boundary violations={geometry['node_boundary_violation_count']}", "repair": [_prescribe("relayout_global")]})
        density = evaluate_density_quality(program)
        if density["status"] != "pass":
            sparse_panels = {str(item.get("panel_id") or "") for item in density.get("sparse_panels") or [] if str(item.get("panel_id") or "")}
            only_panel = next(iter(sparse_panels)) if len(sparse_panels) == 1 else ""
            node_ids = sorted(
                {
                    str(item.get("id") or "")
                    for item in program.get("semantic_nodes") or []
                    if isinstance(item, dict)
                    and str(item.get("id") or "")
                    and (str(item.get("panel_id") or "") in sparse_panels or (only_panel and not str(item.get("panel_id") or "")))
                }
            )
            slot_ids = sorted(
                {
                    str(item.get("id") or "")
                    for item in program.get("slots") or []
                    if isinstance(item, dict)
                    and str(item.get("id") or "")
                    and (
                        str(item.get("panel_id") or "") in sparse_panels
                        or str(item.get("parent_id") or "") in set(node_ids)
                        or (only_panel and not str(item.get("panel_id") or ""))
                    )
                }
            )
            if only_panel and (slot_ids or node_ids):
                repair = [_prescribe("resize_nodes")]
                payload = {**repair[0], "slot_ids": slot_ids, "object_ids": node_ids or slot_ids}
                feedback.append({"stage": "structure", "verdict": "fault", "severity": "major", "defect_key": "structure:sparse_panels", "problem": f"sparse panels={density['sparse_panel_count']}, empty region ratio={density['empty_region_ratio']}", "slot_ids": slot_ids, "object_ids": node_ids or slot_ids, "requested_actions": [payload], "repair": [payload]})
            else:
                feedback.append({"stage": "structure", "verdict": "fault", "severity": "major", "defect_key": "structure:sparse_panels", "problem": f"sparse panels={density['sparse_panel_count']}, empty region ratio={density['empty_region_ratio']}", "repair": [_prescribe("relayout_global")]})
        dimension_scores = deterministic_dimension_scores(geometry, density)
    score = 1.0 if not feedback else max(0.0, 1.0 - 0.25 * len(feedback))
    if mode == "vlm" and not feedback:
        feedback_rule = "Do not propose repairs; independently score acceptance." if frozen else "Provide preserve items and bounded repair actions."
        prompt = f"""You are a {'Frozen' if frozen else 'Online'} Bootstrap Judge for a direct paper-to-PPTX scientific framework pipeline.
Judge semantic fidelity, structure/readability, icon responsibility, connector clarity, and editable-PPT suitability. Scientific errors are blockers. {feedback_rule}
Output JSON only with overall_score 0..1, status pass|fault, preserve array, and stage_feedback entries. Each fault entry uses stage semantic|structure|asset|interface|compile, severity blocker|major|minor, problem, semantic_node_ids, slot_ids, edge_ids, object_ids, and repair actions chosen from {'|'.join(sorted(REPAIR_ACTIONS))}. The action name must be one of those exactly: an action the controller does not know is dropped. A mechanism-level rule names its parameters instead of object ids: constraints for router_constraint (shared_vertical_channel, forbid_crossing, min_bend, max_segments) and params for layout_constraint (align_grid, gutter_min, panel_occupancy_min); a rule with none of them is dropped.
Method contract:
{json.dumps(method, ensure_ascii=False)}
Layout plan:
{json.dumps(layout, ensure_ascii=False)}
Icon plan:
{json.dumps(icons, ensure_ascii=False)}
Prior feedback:
{json.dumps(prior_feedback or [], ensure_ascii=False)}"""
        raw = call_vlm_json(prompt, [], model=resolve_vlm_model("RFS_FROZEN_JUDGE_MODEL" if frozen else "RFS_ONLINE_JUDGE_MODEL", explicit_model=model), timeout=180, retries=1)
        feedback = _normalize_feedback(raw.get("stage_feedback"))
        try:
            score = max(0.0, min(1.0, float(raw.get("overall_score") or 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        preserve = raw.get("preserve") if isinstance(raw.get("preserve"), list) else []
        status = str(raw.get("status") or ("pass" if not feedback else "fault"))
    else:
        preserve, status = [], "pass" if not feedback else "fault"
    return {"judge_role": "frozen_bootstrap" if frozen else "bootstrap", "trained": bool(trained), "mode": mode, "overall_score": round(score, 4), "dimension_scores": dimension_scores, "stage_feedback": feedback, "preserve": preserve, "repair": [item for entry in feedback for item in entry.get("repair", []) if isinstance(item, dict)], "blocking_issue_count": sum(str(item.get("severity")) == "blocker" for item in feedback), "status": status}
