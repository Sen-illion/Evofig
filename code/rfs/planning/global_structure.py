"""Deterministic pre-generation global structure and segmentation planning."""

from __future__ import annotations

from typing import Any

from ..contracts import (
    normalize_global_structure_plan,
    normalize_segmentation_plan,
    validate_global_structure_plan,
    validate_segmentation_plan,
)


def _bbox(value: Any, fallback: dict[str, float]) -> dict[str, float]:
    raw = value if isinstance(value, dict) else fallback
    x = max(0.0, min(0.995, float(raw.get("x", fallback["x"]))))
    y = max(0.0, min(0.995, float(raw.get("y", fallback["y"]))))
    w = max(0.01, min(1.0 - x, float(raw.get("w", fallback["w"]))))
    h = max(0.01, min(1.0 - y, float(raw.get("h", fallback["h"]))))
    return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}


def _fallback_box(index: int, count: int) -> dict[str, float]:
    columns = max(1, min(4, count))
    rows = max(1, (count + columns - 1) // columns)
    gap = 0.025
    width = (0.9 - gap * (columns - 1)) / columns
    height = min(0.28, (0.72 - gap * (rows - 1)) / rows)
    column = index % columns
    row = index // columns
    return _bbox(None, {"x": 0.05 + column * (width + gap), "y": 0.16 + row * (height + gap), "w": width, "h": height})


def _entity_group(field: str, role: str) -> str:
    token = f"{field} {role}".casefold()
    if field == "inputs" or "input" in token or "query" in token:
        return "input_group"
    if field == "outputs" or "output" in token or "answer" in token:
        return "output_group"
    if any(word in token for word in ("aggregate", "discussion", "debate", "fusion", "merge")):
        return "aggregation_group"
    return "parallel_module_group"


def _topology_boxes(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    """Lay out semantic groups compactly while preserving a left-to-right flow."""
    zones = {
        "input_group": {"x": 0.035, "y": 0.10, "w": 0.18, "h": 0.80},
        "parallel_module_group": {"x": 0.245, "y": 0.075, "w": 0.45, "h": 0.85},
        "aggregation_group": {"x": 0.72, "y": 0.18, "w": 0.13, "h": 0.64},
        "output_group": {"x": 0.875, "y": 0.22, "w": 0.10, "h": 0.56},
    }
    result: dict[str, dict[str, float]] = {}
    for group_id, items in grouped.items():
        zone = zones[group_id]
        count = max(1, len(items))
        columns = 2 if group_id == "parallel_module_group" and count > 2 else 1
        rows = (count + columns - 1) // columns
        gap_x, gap_y = 0.018, 0.028
        width = (zone["w"] - gap_x * (columns - 1)) / columns
        height = min(0.30, (zone["h"] - gap_y * (rows - 1)) / rows)
        block_h = rows * height + max(0, rows - 1) * gap_y
        y0 = zone["y"] + max(0.0, (zone["h"] - block_h) / 2)
        for index, item in enumerate(items):
            column, row = index % columns, index // columns
            result[str(item["id"])] = _bbox(None, {
                "x": zone["x"] + column * (width + gap_x),
                "y": y0 + row * (height + gap_y),
                "w": width,
                "h": height,
            })
    return result


def build_global_structure_plan(
    figure_specification: dict[str, Any],
    semantic_plan: dict[str, Any] | None = None,
    *,
    source: str = "direct_paper_contract",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build auditable global and segmentation contracts before image generation."""

    specification = figure_specification if isinstance(figure_specification, dict) else {}
    semantic = semantic_plan if isinstance(semantic_plan, dict) else {}
    semantic_nodes = [item for item in semantic.get("nodes", []) if isinstance(item, dict)]
    semantic_groups = [item for item in semantic.get("groups", []) if isinstance(item, dict)]
    semantic_group_by_node = {
        str(node_id): str(group.get("id"))
        for group in semantic_groups if group.get("id")
        for node_id in group.get("node_ids", []) if str(node_id).strip()
    }
    raw_entities: list[dict[str, Any]] = []
    for field in ("inputs", "modules", "outputs", "innovations"):
        for item in specification.get(field, []) if isinstance(specification.get(field), list) else []:
            if isinstance(item, dict) and (item.get("id") or item.get("name") or item.get("label")):
                raw_entities.append({"field": field, **item})
    by_id = {str(item.get("id")): item for item in raw_entities if item.get("id")}
    nodes: list[dict[str, Any]] = []
    slots: list[dict[str, Any]] = []
    semantic_by_source = {
        str(item.get("semantic_source_id") or item.get("id")): item
        for item in semantic_nodes
        if item.get("semantic_source_id") or item.get("id")
    }
    # The paper contract is authoritative. VLM geometry may enrich it, never replace
    # it or silently drop required entities.
    source_nodes: list[dict[str, Any]] = []
    for item in raw_entities:
        entity_id = str(item.get("id") or "")
        source_nodes.append({**item, **semantic_by_source.get(entity_id, {})})
    for item in semantic_nodes:
        entity_id = str(item.get("semantic_source_id") or item.get("id") or "")
        if entity_id and entity_id not in by_id:
            source_nodes.append(item)
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in ("input_group", "parallel_module_group", "aggregation_group", "output_group")}
    for index, item in enumerate(source_nodes):
        item.setdefault("id", item.get("semantic_source_id") or f"node_{index + 1:02d}")
        source_item = by_id.get(str(item.get("semantic_source_id") or item.get("id")), item)
        semantic_id = str(item.get("semantic_source_id") or item.get("id") or "")
        group_id = str(item.get("group_id") or semantic_group_by_node.get(semantic_id) or _entity_group(str(source_item.get("field") or item.get("field") or "modules"), str(item.get("role") or source_item.get("role") or "module")))
        if group_id not in grouped:
            group_id = "parallel_module_group"
        item["group_id"] = group_id
        grouped[group_id].append(item)
    topology_boxes = _topology_boxes(grouped)
    for index, item in enumerate(source_nodes):
        node_id = str(item.get("semantic_source_id") or item.get("id") or item.get("semantic_id") or f"node_{index + 1:02d}")
        source_item = by_id.get(node_id, item)
        label = str(item.get("label") or source_item.get("name") or source_item.get("label") or node_id).strip()
        bbox = _bbox(item.get("bbox_percent"), topology_boxes.get(node_id, _fallback_box(index, len(source_nodes))))
        nodes.append({
            "id": node_id,
            "label": label,
            "role": str(item.get("role") or source_item.get("role") or "module"),
            "field": str(source_item.get("field") or item.get("field") or "modules"),
            "group_id": str(item.get("group_id") or "parallel_module_group"),
            "panel_id": str(item.get("panel_id") or item.get("group_id") or "parallel_module_group"),
            "internal_layout": str(item.get("internal_layout") or ("input_stack" if item.get("group_id") == "input_group" else "output_focus" if item.get("group_id") == "output_group" else "icon_grid" if item.get("group_id") == "parallel_module_group" else "title_top_visual_center")),
            "bbox_percent": bbox,
            "evidence_ids": [str(value) for value in (item.get("evidence_ids") or source_item.get("evidence_ids") or []) if value],
        })
        slots.append({
            "slot_id": f"slot_{node_id}",
            "semantic_node_ids": [node_id],
            "bbox_percent": bbox,
            "visual_asset_scope": [f"{node_id}_visual"],
            "editable_object_ids": [f"label_{node_id}"],
            "input_ports": ["left_center"],
            "output_ports": ["right_center"],
            "asset_policy": "visual_only_no_text_no_arrows",
            "text_policy": "native_ppt",
            "connector_policy": "native_ppt",
        })
    known = {item["id"] for item in nodes}
    relations = []
    for index, relation in enumerate(specification.get("relations", []) if isinstance(specification.get("relations"), list) else []):
        if not isinstance(relation, dict):
            continue
        source_id = str(relation.get("source") or relation.get("source_id") or "")
        target_id = str(relation.get("target") or relation.get("target_id") or "")
        if source_id in known and target_id in known:
            relations.append({
                "id": str(relation.get("id") or f"edge_{index + 1:02d}"),
                "source": source_id,
                "target": target_id,
                "type": str(relation.get("type") or relation.get("relation_type") or "data_flow"),
                "label": str(relation.get("label") or ""),
            })
    topology = str(specification.get("topology") or semantic.get("topology") or "unknown")
    groups = []
    panels = []
    for group_id, items in grouped.items():
        if not items:
            continue
        boxes = [topology_boxes.get(str(item.get("id"))) or _bbox(item.get("bbox_percent"), _fallback_box(0, 1)) for item in items]
        x0, y0 = min(box["x"] for box in boxes), min(box["y"] for box in boxes)
        x1, y1 = max(box["x"] + box["w"] for box in boxes), max(box["y"] + box["h"] for box in boxes)
        group_box = _bbox(None, {"x": max(0.015, x0 - 0.012), "y": max(0.025, y0 - 0.045), "w": min(0.97 - max(0.015, x0 - 0.012), x1 - x0 + 0.024), "h": min(0.95 - max(0.025, y0 - 0.045), y1 - y0 + 0.065)})
        node_ids = [str(item.get("semantic_source_id") or item.get("id")) for item in items]
        groups.append({"id": group_id, "role": group_id.removesuffix("_group"), "node_ids": node_ids, "bbox_percent": group_box})
        panels.append({"id": group_id, "role": group_id.removesuffix("_group"), "node_ids": node_ids, "bbox_percent": group_box})
    required_ids = {str(item.get("id")) for item in raw_entities if item.get("id")}
    planned_ids = {item["id"] for item in nodes}
    required_relations = {
        (str(item.get("source") or item.get("source_id") or ""), str(item.get("target") or item.get("target_id") or ""))
        for item in specification.get("relations", []) if isinstance(item, dict)
    }
    planned_relations = {(item["source"], item["target"]) for item in relations}
    global_plan = normalize_global_structure_plan({
        "summary": "Pre-generation paper-grounded global structure plan.",
        "topology": topology,
        "reading_order": [item["id"] for item in nodes],
        "panels": panels,
        "groups": groups,
        "nodes": nodes,
        "relations": relations,
        "layout_policy": {
            "reserve_native_text": True,
            "reserve_native_connectors": True,
            "contain_no_crop": True,
            "source": source,
            "layout_mode": "vlm" if semantic_nodes else "topology_fallback",
            "fallback_used": not bool(semantic_nodes),
        },
        "decomposability": {
            "required": True,
            "target": "slot_boundaries_and_cross_slot_interfaces_are_recoverable",
        },
    })
    segmentation = normalize_segmentation_plan({
        "plan_id": "segmentation_pre_generation",
        "summary": "Pre-generation semantic slot and interface plan.",
        "slots": slots,
        "rationale": ["One conservative visual slot per semantic node before local Icon generation."],
        "provenance": {"source": source, "global_structure_contract": global_plan["contract"]},
    })
    validation = {
        "global_structure": validate_global_structure_plan(global_plan),
        "segmentation": validate_segmentation_plan(segmentation),
    }
    validation["coverage"] = {
        "ok": required_ids.issubset(planned_ids) and required_relations.issubset(planned_relations),
        "required_node_ids": sorted(required_ids),
        "planned_node_ids": sorted(planned_ids),
        "missing_node_ids": sorted(required_ids - planned_ids),
        "missing_relations": [{"source": source_id, "target": target_id} for source_id, target_id in sorted(required_relations - planned_relations)],
    }
    validation["ok"] = bool(validation["global_structure"]["ok"] and validation["segmentation"]["ok"] and validation["coverage"]["ok"])
    return global_plan, segmentation, validation
