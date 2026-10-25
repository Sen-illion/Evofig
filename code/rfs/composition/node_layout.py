"""Deterministic internal layout for editable semantic nodes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..contracts.priority import priority_number
from ..generation.asset_metrics import measure_ink_bbox
from .fonts import body_font_family, load_font


#: The smallest icon the horizontal icon-left/label-right split may leave behind.
#: Below it the card is no longer a card with a picture in it, so a card that
#: cannot give both an icon and its longest word room is stacked instead.
_ICON_MIN_W = 0.016

#: The shortest card that can carry the stacked layout: an icon, a gap and one
#: readable label line.
_STACK_MIN_H_IN = 0.60

#: The size a label may still be drawn at.  A lane is only reserved wide enough
#: for the longest word at this size; anything smaller the typography stage
#: would refuse anyway.
_LABEL_FLOOR_PT = 9.0


def _longest_word_width_in(label: str, family: str, *, bold: bool) -> float:
    """Width the label's longest word needs at the smallest size we would draw it.

    The lane is cut on words, never inside one, so a lane narrower than a single
    word is exactly what draws "Unstruc/tured".
    """

    words = [word for word in str(label or "").split() if word]
    if not words:
        return 0.0
    longest = max(words, key=len)
    font = load_font(max(1, round(_LABEL_FLOOR_PT * 96 / 72)), family, bold=bold)
    return float(font.getbbox(longest)[2]) / 96.0


def _inset(box: dict[str, Any], left: float, top: float, right: float, bottom: float) -> dict[str, float]:
    width, height = float(box["w"]), float(box["h"])
    return {
        "x": round(float(box["x"]) + width * left, 4),
        "y": round(float(box["y"]) + height * top, 4),
        "w": round(width * max(0.02, 1.0 - left - right), 4),
        "h": round(height * max(0.02, 1.0 - top - bottom), 4),
    }


def _glyph_aspect(assets: dict[str, dict[str, Any]], asset_id: Any, canvas_ratio: float) -> float:
    """Return the delivered glyph height divided by its width, in slide units.

    Generated icons are inserted with contain semantics, so a slot taller than
    the glyph only adds invisible padding and pushes the label away from the
    mark it belongs to. Measuring the visible bbox lets the slot hug the glyph.
    """

    asset = assets.get(str(asset_id or "")) or {}
    path = str(asset.get("path") or "")
    if not path:
        return canvas_ratio
    measured = measure_ink_bbox(path)
    size = measured.get("size")
    if not isinstance(size, list) or len(size) != 2 or not all(size):
        return canvas_ratio
    width_px = float(measured.get("width_ratio") or 1.0) * float(size[0])
    height_px = float(measured.get("height_ratio") or 1.0) * float(size[1])
    if width_px <= 0 or height_px <= 0:
        return canvas_ratio
    return height_px / width_px * canvas_ratio


def arrange_node_contents(program: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Allocate disjoint Icon and editable-label regions inside each node."""

    updated = deepcopy(program)
    slots = {
        str(item.get("id") or "").removeprefix("slot_"): item
        for item in updated.get("slots", []) or []
        if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)
    }
    assets = {str(item.get("id") or ""): item for item in updated.get("assets", []) or [] if isinstance(item, dict)}
    records: list[dict[str, Any]] = []
    style = updated.get("style") if isinstance(updated.get("style"), dict) else {}
    family = body_font_family(style)
    for node in updated.get("semantic_nodes", []) or []:
        if not isinstance(node, dict) or not isinstance(node.get("bbox_percent"), dict):
            continue
        node_id = str(node.get("id") or "")
        slot = slots.get(node_id)
        if not slot:
            continue
        box = node["bbox_percent"]
        label = str(node.get("label") or node_id)
        compact_label = " ".join(label.split())
        long_label = len(compact_label) > 34
        visual_role = str(node.get("visual_role") or slot.get("visual_role") or "standard_icon")
        complexity = str(node.get("complexity") or slot.get("complexity") or "medium").casefold()
        raw_priority = node.get("visual_priority", node.get("importance", slot.get("visual_priority", 0.5)))
        priority = priority_number(raw_priority)
        # VLM layouts frequently use shallow scientific cards. Splitting those
        # cards vertically leaves a label lane that is physically shorter than
        # one readable text line, so use an icon-left/text-right layout there.
        # Normalized x/y units are converted through the canvas aspect ratio so
        # the icon remains approximately square in physical slide coordinates.
        canvas = updated.get("canvas") if isinstance(updated.get("canvas"), dict) else {}
        width_in = float(canvas.get("width_in") or 13.333)
        height_in = float(canvas.get("height_in") or 7.5)
        physical_height_in = float(box["h"]) * height_in
        large_visual = visual_role in {"large_visual", "hero_visual", "wide_visual"} or priority >= 0.82 or complexity == "high"
        slot["visual_role"], slot["visual_priority"], slot["complexity"] = visual_role, round(priority, 3), complexity
        shallow = physical_height_in < 0.78 and not large_visual
        gap = min(0.008, float(box["w"]) * 0.04)
        # A label lane is cut on words, so it has to hold the longest one: a lane
        # narrower than that is what draws "Unstruc/tured".  When the lane would eat
        # the icon, the card is stacked instead and the label takes the full width.
        bold = bool(node.get("bold", True))
        # The lane also pays the 0.008 inset on both sides of the card.
        lane_needed = _longest_word_width_in(compact_label, family, bold=bold) / width_in + gap + 0.016
        if shallow and float(box["w"]) - lane_needed < _ICON_MIN_W and physical_height_in >= _STACK_MIN_H_IN:
            shallow = False
        if shallow:
            preferred = max(0.24, min(0.48, float(node.get("preferred_area") or slot.get("preferred_area") or 0.34)))
            icon_width = min(float(box["w"]) * preferred, float(box["h"]) * height_in / width_in * 0.82)
            icon_width = max(_ICON_MIN_W, min(icon_width, float(box["w"]) - lane_needed))
            aspect = _glyph_aspect(assets, slot.get("asset_id"), width_in / height_in)
            icon_height = min(float(box["h"]) * 0.80, max(0.012, icon_width) * aspect)
            icon_box = {
                "x": round(float(box["x"]) + 0.008, 4),
                "y": round(float(box["y"]) + max(0.0, (float(box["h"]) - icon_height) / 2.0), 4),
                "w": round(max(0.012, icon_width), 4),
                "h": round(icon_height, 4),
            }
            label_x = icon_box["x"] + icon_box["w"] + gap
            label_box = {
                "x": round(label_x, 4),
                "y": round(float(box["y"]) + float(box["h"]) * 0.08, 4),
                "w": round(max(0.02, float(box["x"]) + float(box["w"]) - label_x - 0.008), 4),
                "h": round(float(box["h"]) * 0.84, 4),
            }
            internal_layout = "icon_left_editable_label_right"
        else:
            label_ratio = 0.32 if large_visual else (0.38 if long_label else 0.30)
            side = 0.08 if large_visual else 0.12
            top_ratio = 0.06 if large_visual else 0.08
            label_box = _inset(box, 0.055, 1.0 - label_ratio, 0.055, 0.055)
            icon_width = float(box["w"]) * max(0.02, 1.0 - 2.0 * side)
            aspect = _glyph_aspect(assets, slot.get("asset_id"), width_in / height_in)
            available = float(box["h"]) * max(0.02, 1.0 - top_ratio - label_ratio - 0.035)
            icon_height = min(available, max(0.012, icon_width) * aspect)
            gap = max(0.004, float(box["h"]) * 0.02)
            label_height = float(label_box["h"])
            max_group = float(box["h"]) * 0.92
            if icon_height + gap + label_height > max_group:
                label_height = max(float(box["h"]) * 0.06, max_group - icon_height - gap)
            group_top = float(box["y"]) + max(0.0, (float(box["h"]) - (icon_height + gap + label_height)) / 2.0)
            icon_box = {
                "x": round(float(box["x"]) + float(box["w"]) * side, 4),
                "y": round(group_top, 4),
                "w": round(icon_width, 4),
                "h": round(icon_height, 4),
            }
            label_box = {**label_box, "y": round(group_top + icon_height + gap, 4), "h": round(label_height, 4)}
            internal_layout = "large_visual_label_bottom" if large_visual else "icon_above_editable_label"
        slot["bbox_percent"] = icon_box
        slot["parent_id"] = node_id
        slot["layout_role"] = "node_icon"
        node["label_bbox_percent"] = label_box
        node["internal_layout"] = internal_layout
        records.append({"node_id": node_id, "asset_id": slot.get("asset_id"), "icon_bbox_percent": icon_box, "label_bbox_percent": label_box, "long_label": long_label, "visual_role": visual_role, "visual_priority": priority, "internal_layout": internal_layout})
    return updated, {"summary": "Deterministic node-internal Icon/text layout.", "status": "pass", "node_count": len(records), "nodes": records}
