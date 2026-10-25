"""Contract helpers for determining ownership of visible editable text."""

from __future__ import annotations


from .figure_text import fit_panel_title_size, panel_title_lane, panel_title_max_lines


def visible_text_items(program: dict) -> list[dict]:
    text_program = program.get("text_program") if isinstance(program, dict) else None
    items = text_program.get("items", []) if isinstance(text_program, dict) else []
    return [item for item in items if isinstance(item, dict) and item.get("visible", True)]


def node_text_owned_by_program(program: dict, node: dict) -> bool:
    node_id = str(node.get("id") or "")
    if not node_id:
        return False
    for item in visible_text_items(program):
        role = str(item.get("role") or item.get("text_role") or "").casefold()
        if role in {"connector_label", "panel_title", "panel_header", "title"}:
            continue
        targets = {str(item.get(key)) for key in ("owner_id", "owner_node_id", "target_id", "semantic_source_id", "node_id") if item.get(key)}
        if node_id in targets:
            return True
        item_id = str(item.get("id") or "")
        if role in {"node_label", "primary_node", "secondary_node", "node"} and item_id in {node_id, f"text_{node_id}", f"label_{node_id}"}:
            return True
        if item_id.casefold() in {f"node_{node_id}".casefold(), f"text_{node_id}".casefold()}:
            return True
    return False


def canonicalize_node_text_program(program: dict) -> tuple[dict, dict]:
    """Make text_program the single visible owner for every semantic node label."""
    from copy import deepcopy

    updated = deepcopy(program)
    text_program = updated.get("text_program") if isinstance(updated.get("text_program"), dict) else {"items": []}
    items = [item for item in text_program.get("items", []) if isinstance(item, dict)]
    nodes = [node for node in updated.get("semantic_nodes", []) if isinstance(node, dict) and node.get("id")]
    node_ids = {str(node.get("id")): node for node in nodes}
    panels = [panel for panel in updated.get("panels", []) if isinstance(panel, dict) and panel.get("id")]
    panel_ids = {str(panel.get("id")): panel for panel in panels}
    node_by_semantic = {str(node.get("semantic_source_id") or node.get("semantic_node_id") or node.get("id")): node for node in nodes}
    result = []
    seen_nodes: set[str] = set()
    dropped = []
    seen_panel_titles: set[str] = set()
    for item in items:
        role = str(item.get("text_role") or item.get("role") or "").casefold()
        target_id = str(item.get("target_id") or item.get("panel_id") or "")
        item_id = str(item.get("id") or "")
        # Panel titles are rendered by the panel header. Normalize malformed
        # planner output (missing role/text_role) so the text program remains
        # an editable metadata owner without creating a second textbox.
        title_target = ""
        if target_id in panel_ids and (
            role in {"panel_title", "panel_header"}
            or item_id.casefold().endswith("_title")
            or str(item.get("text") or "").strip() == str(panel_ids[target_id].get("title") or "").strip()
        ):
            title_target = target_id
        if not title_target and item_id.casefold().startswith("panel_") and item_id.casefold().endswith("_title"):
            candidate = item_id[:-6]
            if candidate in panel_ids:
                title_target = candidate
        # A planner may also name the heading exactly after its panel
        # (``panel_intrinsic``) or drop the target id entirely. Those forms
        # still have to be rebound, otherwise the heading keeps whatever box
        # the planner returned - typically the whole panel.
        if not title_target and role in {"panel_title", "panel_header"}:
            if item_id in panel_ids:
                title_target = item_id
            else:
                text_norm = str(item.get("text") or item.get("short_text") or "").strip().casefold()
                matches = [
                    panel_id for panel_id, panel in panel_ids.items()
                    if text_norm and text_norm == str(panel.get("title") or "").strip().casefold()
                ]
                if len(matches) == 1:
                    title_target = matches[0]
        if title_target:
            normalized_title = {**item, "role": "panel_title", "text_role": "panel_title", "panel_id": title_target, "target_id": title_target, "visible": True, "layer_ownership": "editable_text_layer", "editable_in": "pptx"}
            normalized_title.update(_panel_title_lane_binding(updated, panel_ids[title_target], normalized_title))
            if title_target in seen_panel_titles:
                dropped.append(str(item.get("id") or "duplicate_panel_title"))
                continue
            result.append(normalized_title)
            seen_panel_titles.add(title_target)
            continue
        owner = str(item.get("owner_node_id") or item.get("owner_id") or item.get("target_id") or item.get("node_id") or item.get("semantic_source_id") or "")
        if not owner:
            for candidate in (item_id, item_id.removeprefix("text_"), item_id.removeprefix("label_"), item_id.removeprefix("node_")):
                if candidate in node_ids or candidate in node_by_semantic:
                    owner = candidate
                    break
        if not owner and role in {"node_label", "primary_node", "secondary_node", "node"}:
            text_norm = str(item.get("text") or item.get("short_text") or "").strip().casefold()
            matches = [node for node in nodes if text_norm and text_norm == str(node.get("label") or "").strip().casefold()]
            if len(matches) == 1:
                owner = str(matches[0].get("id"))
        is_node_label = role in {"node_label", "primary_node", "secondary_node", "node"} or owner in node_ids or owner in node_by_semantic
        if is_node_label:
            node = node_ids.get(owner) or node_by_semantic.get(owner)
            if not node:
                dropped.append(str(item.get("id") or "unknown"))
                continue
            node_id = str(node.get("id"))
            if node_id in seen_nodes:
                dropped.append(str(item.get("id") or "duplicate"))
                continue
            normalized = {**item, "id": f"text_{node_id}", "text_role": "node_label", "role": "node_label", "owner_node_id": node_id, "target_id": node_id, "semantic_source_id": node.get("semantic_source_id") or node_id, "visible": True, "layer_ownership": "editable_text_layer", "editable_in": "pptx"}
            if isinstance(node.get("label_bbox_percent"), dict):
                normalized["bbox_percent"] = dict(node["label_bbox_percent"])
            normalized["text"] = str(normalized.get("short_text") or normalized.get("text") or node.get("label") or node_id)
            result.append(normalized)
            seen_nodes.add(node_id)
            continue
        result.append(item)
    for node in nodes:
        node_id = str(node.get("id"))
        if node_id in seen_nodes or str(node.get("render_policy") or "").casefold() in {"panel_only", "hidden"}:
            continue
        result.append({"id": f"text_{node_id}", "text": str(node.get("label") or node_id), "text_role": "node_label", "role": "node_label", "owner_node_id": node_id, "target_id": node_id, "semantic_source_id": node.get("semantic_source_id") or node_id, "bbox_percent": dict(node.get("label_bbox_percent") or node.get("bbox_percent") or {}), "visible": True, "editable": True, "layer_ownership": "editable_text_layer", "editable_in": "pptx"})
        seen_nodes.add(node_id)
    updated["text_program"] = {**text_program, "items": result, "node_label_owner": "text_program"}
    return updated, {"status": "pass", "node_count": len(nodes), "owned_node_count": len(seen_nodes), "dropped_duplicate_or_orphan_count": len(dropped), "dropped_items": dropped}


def _panel_title_lane_binding(program: dict, panel: dict, title_item: dict) -> dict:
    """Pin a panel title to the header band the compiler actually draws.

    Panel headings are painted by the panel header, not by an arbitrary text
    box. A planner that answers with the whole panel as the title box produces
    an oversized heading splashed across the payload, so the lane - and the
    font size that fits it - are recomputed here from the finished panel box.
    """

    canvas = program.get("canvas") if isinstance(program.get("canvas"), dict) else {}
    canvas_width_in = float(canvas.get("width_in") or 13.333)
    canvas_height_in = float(canvas.get("height_in") or 7.5)
    lane = panel_title_lane(panel.get("bbox_percent"), canvas_height_in=canvas_height_in)
    if not lane:
        return {}
    lane_width_in = float(lane["w"]) * canvas_width_in
    lane_height_in = float(lane["h"]) * canvas_height_in
    text = str(title_item.get("text") or panel.get("title") or "")
    requested = float(title_item.get("font_size_pt") or 16.0)
    size = fit_panel_title_size(text, lane_width_in, requested, lane_height_in)
    return {"bbox_percent": lane, "font_size_pt": round(size, 2), "font_size_role": "panel_title", "min_readable_size_pt": 8.0, "max_lines": panel_title_max_lines(text, lane_width_in, size), "text": text or title_item.get("text")}


__all__ = ["visible_text_items", "node_text_owned_by_program", "canonicalize_node_text_program"]
