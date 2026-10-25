"""Map content-free exemplar regions onto paper-grounded semantic groups."""

from __future__ import annotations

from typing import Any


def build_template_semantic_mapping(template: dict[str, Any] | None, structure: dict[str, Any], target_ratio: str) -> tuple[dict[str, Any], dict[str, Any]]:
    source = template if isinstance(template, dict) else {}
    panels = [item for item in source.get("panels", []) if isinstance(item, dict)]
    groups = [item for item in structure.get("groups", []) if isinstance(item, dict)]
    mappings = []
    for index, group in enumerate(groups):
        panel = panels[min(index, len(panels) - 1)] if panels else {}
        mappings.append({
            "semantic_group_id": group.get("id"),
            "template_panel_id": panel.get("id") if panel else None,
            "adopt": ["macro_proportion", "hierarchy", "internal_rhythm"] if panel else [],
            "reject": ["scientific_labels", "reference_specific_objects"],
        })
    source_ratio = float(source.get("source_aspect_ratio") or source.get("aspect_ratio") or 16 / 9)
    try:
        left, right = str(target_ratio).split(":", 1)
        destination_ratio = float(left) / float(right)
    except (TypeError, ValueError, ZeroDivisionError):
        destination_ratio = source_ratio
    mapping = {"contract": "rfs.template_semantic_mapping", "mappings": mappings, "template_id": source.get("template_id")}
    reflow = {
        "contract": "rfs.canvas_reflow_report",
        "source_aspect_ratio": round(source_ratio, 4),
        "target_aspect_ratio": round(destination_ratio, 4),
        "reflow_required": abs(source_ratio - destination_ratio) / max(destination_ratio, 0.1) > 0.12,
        "policy": "semantic_group_reflow_no_crop",
        "preserved": ["reading_order", "group_hierarchy", "flow_direction"],
        "not_preserved": ["absolute_panel_coordinates"],
    }
    return mapping, reflow
