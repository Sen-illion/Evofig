"""Non-blocking visual density diagnostics for planned figure layouts."""

from __future__ import annotations

from typing import Any


DELIVERY_MIN_PANEL_OCCUPANCY = 0.45


def _area(box: dict[str, Any]) -> float:
    try:
        return max(0.0, float(box.get("w", 0.0))) * max(0.0, float(box.get("h", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def build_density_advisory(plan: dict[str, Any] | None, experience: dict[str, Any] | None = None) -> dict[str, Any]:
    """Diagnose sparse structure plans before expensive asset generation."""
    source = plan if isinstance(plan, dict) else {}
    rules = experience if isinstance(experience, dict) else {}
    geometry = rules.get("global_geometry_rules") if isinstance(rules.get("global_geometry_rules"), dict) else {}
    panel_rules = rules.get("panel_sizing_rules") if isinstance(rules.get("panel_sizing_rules"), dict) else {}
    plan_targets = source.get("density_targets") if isinstance(source.get("density_targets"), dict) else {}
    action = rules.get("action") if isinstance(rules.get("action"), dict) else {}
    memory_targets = rules.get("density_targets") if isinstance(rules.get("density_targets"), dict) else {}
    if isinstance(action.get("density_targets"), dict):
        memory_targets = {**memory_targets, **action["density_targets"]}
    plan_targets = {**plan_targets, **memory_targets}
    min_occupancy = float(geometry.get("canvas_occupancy_target", [0.45, 0.86])[0] if isinstance(geometry.get("canvas_occupancy_target"), list) else geometry.get("minimum_occupancy", plan_targets.get("canvas_occupancy_min", plan_targets.get("minimum_occupancy", 0.45))))
    max_empty = float(geometry.get("empty_region_max_ratio", plan_targets.get("max_empty_region_ratio", 0.55)))
    requested_panel_occupancy = plan_targets.get(
        "panel_occupancy_min",
        geometry.get("panel_occupancy_min", DELIVERY_MIN_PANEL_OCCUPANCY),
    )
    try:
        min_panel_occupancy = min(
            0.75,
            max(DELIVERY_MIN_PANEL_OCCUPANCY, float(requested_panel_occupancy)),
        )
    except (TypeError, ValueError):
        min_panel_occupancy = DELIVERY_MIN_PANEL_OCCUPANCY
    single_max_area = float(panel_rules.get("single_node_max_area", 0.08))
    single_min_ratio = float(panel_rules.get("single_node_min_content_ratio", 0.42))
    nodes = [item for item in source.get("nodes", []) if isinstance(item, dict)]
    panels = [item for item in source.get("panels", []) if isinstance(item, dict)]
    node_areas = [_area(item.get("bbox_percent") or {}) for item in nodes]
    panel_areas = [_area(item.get("bbox_percent") or {}) for item in panels]
    occupancy = min(1.0, sum(node_areas))
    panel_occupancy = min(1.0, sum(panel_areas))
    ys = [float((item.get("bbox_percent") or {}).get("y", 0.0)) for item in nodes if isinstance(item.get("bbox_percent"), dict)]
    warnings: list[str] = []
    sparse_panels = []
    compact_panel_candidates = []
    for panel in panels:
        pbox = panel.get("bbox_percent") or {}
        area = _area(pbox)
        children = [n for n in nodes if str(n.get("panel_id")) == str(panel.get("id"))]
        child_area = sum(_area(n.get("bbox_percent") or {}) for n in children)
        ratio = child_area / max(area, 1e-9)
        if area >= min(single_max_area, 0.08) and ratio < min_panel_occupancy:
            sparse_panels.append({
                "panel_id": str(panel.get("id")),
                "occupancy": round(ratio, 4),
                "minimum_occupancy": round(min_panel_occupancy, 4),
                "occupancy_deficit": round(min_panel_occupancy - ratio, 4),
                "node_count": len(children),
                "reason": "panel area is large relative to its payload",
            })
        if len(children) == 1 and area >= single_max_area and ratio < single_min_ratio:
            compact_panel_candidates.append({"panel_id": str(panel.get("id")), "occupancy": round(ratio, 4), "node_count": 1, "reason": "single low-information payload should use a compact card"})
    if occupancy < 0.30:
        warnings.append("planned node occupancy is sparse; compact reflow is likely needed")
    if nodes and ys and max(ys) - min(ys) < 0.35:
        warnings.append("node distribution is vertically concentrated")
    if len(node_areas) >= 3 and max(node_areas) > 3 * max(min(node_areas), 1e-6):
        warnings.append("node sizes vary substantially; inspect hierarchy emphasis")
    status = "replan_required" if (occupancy < min_occupancy and sparse_panels) or occupancy < 0.25 else "advisory" if warnings or compact_panel_candidates or (1.0 - occupancy) > max_empty else "pass"
    return {
        "contract": "rfs.density_advisory",
        "blocking": False,
        "node_count": len(nodes),
        "panel_count": len(panels),
        "node_occupancy_estimate": round(occupancy, 4),
        "panel_occupancy_estimate": round(panel_occupancy, 4),
        "warnings": warnings,
        "sparse_panels": sparse_panels,
        "compact_panel_candidates": compact_panel_candidates,
        "status": status,
        "confidence": "high" if status == "replan_required" else "medium" if warnings else "high",
        "recommendation": "compact_panels" if compact_panel_candidates else "compact_reflow" if status == "replan_required" else "review" if warnings else "keep_layout",
        "experience_rules_applied": {
            "canvas_occupancy_target": geometry.get("canvas_occupancy_target"),
            "empty_region_max_ratio": geometry.get("empty_region_max_ratio"),
            "panel_occupancy_min": min_panel_occupancy,
            "single_node_max_area": single_max_area,
            "single_node_min_content_ratio": single_min_ratio,
            "minimum_font_size": memory_targets.get("minimum_font_size") or memory_targets.get("minimum_font_size_pt"),
            "node_spacing": memory_targets.get("node_spacing"),
            "occupancy_band": memory_targets.get("occupancy_band"),
        },
    }
