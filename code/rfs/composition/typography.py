"""Role- and geometry-aware typography for direct editable scientific figures."""

from __future__ import annotations

from copy import deepcopy
from math import hypot
from typing import Any

from PIL import ImageFont


NODE_RANGES = {"primary_node": (12.0, 14.0), "secondary_node": (10.5, 12.0), "node_label": (11.0, 13.0)}


def _font(size_pt: float, bold: bool = False) -> ImageFont.ImageFont:
    for name in ("arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, max(1, round(size_pt * 96 / 72)))
        except OSError:
            continue
    return ImageFont.load_default()


def _fits(text: str, size: float, width_px: float, height_px: float, max_lines: int, bold: bool) -> bool:
    font = _font(size, bold)
    words = text.split() or [text]
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and font.getbbox(candidate)[2] > width_px:
            lines.append(current); current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    line_height = max(1, font.getbbox("Ag")[3]) * 1.18
    # A line wider than the box is not a fit, however few lines there are.  Counting
    # lines alone called a 14pt "Unstructured" a fit for a lane half its width, and
    # the deck then broke the word in half to make it so.
    widest = max((font.getbbox(line)[2] for line in lines), default=0)
    return len(lines) <= max_lines and len(lines) * line_height <= height_px and widest <= width_px


def optimize_direct_typography(program: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose readable sizes from role hierarchy and actual object/path geometry."""
    updated = deepcopy(program)
    targets = updated.get("density_targets") if isinstance(updated.get("density_targets"), dict) else {}
    try:
        floor = float(targets.get("minimum_font_size") or targets.get("minimum_font_size_pt") or 0.0)
    except (TypeError, ValueError):
        floor = 0.0
    canvas = updated.get("canvas") or {}
    width_px = float(canvas.get("width_in") or 13.5) * 96
    height_px = float(canvas.get("height_in") or 7.5) * 96
    node_records, unresolved, repair_requests = [], [], []
    for node in updated.get("semantic_nodes", []):
        if not isinstance(node, dict):
            continue
        box = node.get("label_bbox_percent") or node.get("bbox_percent")
        if not isinstance(box, dict):
            continue
        role = str(node.get("font_size_role") or "node_label")
        minimum, preferred = NODE_RANGES.get(role, NODE_RANGES["node_label"])
        if floor:
            minimum = max(minimum, floor)
            preferred = max(preferred, minimum)
        label = " ".join(str(node.get("label") or "").split())
        selected = None
        for half_pt in range(round(preferred * 2), round(minimum * 2) - 1, -1):
            candidate = half_pt / 2
            if _fits(label, candidate, float(box["w"]) * width_px * 0.92, float(box["h"]) * height_px * 0.88, 2, bool(node.get("bold", True))):
                selected = candidate; break
        node["font_size_pt"] = selected if selected is not None else minimum
        node["min_readable_size_pt"] = minimum
        node["typography_status"] = "fit" if selected is not None else "requires_relayout"
        record = {"node_id": node.get("id"), "role": role, "selected_font_size_pt": node["font_size_pt"], "minimum_font_size_pt": minimum, "status": node["typography_status"]}
        node_records.append(record)
        if selected is None:
            unresolved.append(record)
            repair_requests.append({
                "action": "resize_node",
                "node_id": node.get("id"),
                "minimum_bbox_percent": {"x": float(box.get("x", 0)), "y": float(box.get("y", 0)), "w": min(0.42, float(box.get("w", 0)) * 1.25), "h": min(0.28, float(box.get("h", 0)) * 1.2)},
                "reason": "label_does_not_fit_readable_range",
            })
    arrows = {str(x.get("id")): x for x in updated.get("arrows", []) if isinstance(x, dict)}
    connector_records = []
    for item in (updated.get("text_program") or {}).get("items", []):
        if not isinstance(item, dict) or str(item.get("role") or item.get("text_role")) != "connector_label":
            continue
        arrow = arrows.get(str(item.get("connector_id") or item.get("target_id")))
        path = arrow.get("path_percent", []) if arrow else []
        lengths = [hypot((b[0] - a[0]) * width_px, (b[1] - a[1]) * height_px) for a, b in zip(path[:-1], path[1:])]
        segment_px = max(lengths, default=0.0)
        text = " ".join(str(item.get("text") or "").split())
        desired = 10.0 if segment_px >= 150 else 9.0
        required_px = _font(desired).getbbox(text or " ")[2]
        fits = required_px <= segment_px * 0.82
        item["font_size_pt"] = desired
        item["min_readable_size_pt"] = 9.0
        item["max_lines"] = 1 if fits else 2
        item["typography_status"] = "fit" if fits else "requires_longer_segment"
        connector_records.append({"text_id": item.get("id"), "connector_id": item.get("connector_id"), "segment_length_px": round(segment_px, 1), "required_width_px": required_px, "font_size_pt": desired, "status": item["typography_status"]})
        if not fits and item.get("connector_id"):
            repair_requests.append({"action": "reserve_connector_lane", "connector_id": item.get("connector_id"), "object_ids": [item.get("connector_id")], "axis": "horizontal", "amount": 0.04, "reason": "connector_label_needs_longer_segment"})
    return updated, {"summary": "Role- and geometry-aware direct typography plan.", "status": "needs_relayout" if unresolved or any(item["status"] != "fit" for item in connector_records) else "pass", "node_count": len(node_records), "unresolved_node_count": len(unresolved), "nodes": node_records, "connector_labels": connector_records, "repair_requests": repair_requests}


__all__ = ["optimize_direct_typography"]
