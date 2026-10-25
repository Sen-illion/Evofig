"""Contract for editable figure text planning."""

from __future__ import annotations
from typing import Any

from .design_plan import _normalize_bbox, _valid_bbox


# A panel title is drawn inside the panel's own header band.  PowerPoint and
# the preview renderer both size that band as min(0.34in, 17% of the panel),
# so the editable title text must be normalized to the same box: when a
# planner returns the whole panel as the title box, the heading renders as
# oversized text across the payload instead of as a caption.
PANEL_TITLE_BAND_MAX_IN = 0.34
PANEL_TITLE_BAND_HEIGHT_RATIO = 0.17
PANEL_TITLE_MIN_LANE_PERCENT = 0.022


def panel_title_lane(
    panel_bbox: dict[str, Any] | None, *, canvas_height_in: float = 7.5
) -> dict[str, float]:
    """Return the header band a panel title may occupy, in percent coordinates."""

    box = _normalize_bbox(panel_bbox)
    if not isinstance(box, dict) or not _valid_bbox(box):
        return {}
    band = min(
        PANEL_TITLE_BAND_MAX_IN / max(1e-6, float(canvas_height_in)),
        PANEL_TITLE_BAND_HEIGHT_RATIO * float(box["h"]),
    )
    return {
        "x": round(float(box["x"]), 4),
        "y": round(float(box["y"]), 4),
        "w": round(float(box["w"]), 4),
        "h": round(max(PANEL_TITLE_MIN_LANE_PERCENT, band), 4),
    }


def fit_panel_title_size(
    text: str, width_in: float, requested: float, lane_height_in: float | None = None
) -> float:
    """Keep a long heading inside its header band instead of across the panel."""

    chars = max(1, len(" ".join(str(text or "").split())))
    # Approximate Arial bold width at 0.52em per character, leaving margins.
    max_size = (max(0.35, width_in - 0.12) * 72.0) / (chars * 0.52)
    if lane_height_in:
        # A single line still needs leading and descender room in the band.
        max_size = min(max_size, float(lane_height_in) * 72.0 * 0.86)
    return max(8.0, min(float(requested), max_size, 20.0))


def panel_title_max_lines(text: str, width_in: float, font_size_pt: float) -> int:
    """Return one line when the fitted heading spans the lane, otherwise two."""

    chars = max(1, len(" ".join(str(text or "").split())))
    one_line_in = chars * 0.52 * float(font_size_pt) / 72.0
    return 1 if one_line_in <= max(0.35, width_in - 0.12) else 2


def normalize_figure_text_plan(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    src = raw if isinstance(raw, dict) else {}
    items = []
    for value in src.get("items", []):
        if not isinstance(value, dict):
            continue
        item = {**value, "bbox_percent": _normalize_bbox(value.get("bbox_percent"))}
        role = str(item.get("text_role") or item.get("role") or "free_text")
        item["text_role"] = role
        item["role"] = role
        item["target_id"] = str(item.get("target_id") or item.get("semantic_source_id") or item.get("panel_id") or "")
        item["font_size_role"] = str(item.get("font_size_role") or role)
        item["font_weight"] = str(item.get("font_weight") or ("bold" if role in {"panel_title", "node_label"} else "regular"))
        item["max_lines"] = max(1, int(item.get("max_lines") or (2 if role == "node_label" else 1)))
        item["min_readable_size_pt"] = max(8.0, float(item.get("min_readable_size_pt") or 8.0))
        items.append(item)
    return {
        "contract": "rfs.figure_text_plan",
        "version": 1,
        "items": items,
        "text_policy": (
            src.get("text_policy")
            if isinstance(src.get("text_policy"), dict)
            else {"native_ppt_only": True}
        ),
        "provenance": (
            src.get("provenance") if isinstance(src.get("provenance"), dict) else {}
        ),
    }


def validate_figure_text_plan(plan: dict[str, Any]) -> dict[str, Any]:
    p = normalize_figure_text_plan(plan)
    errors = []
    if not p["items"]:
        errors.append("figure text plan requires at least one item")
    for i, x in enumerate(p["items"]):
        if not x.get("id") or not x.get("text_role") or not x.get("text"):
            errors.append(f"item[{i}] requires id/text_role/text")
        if not _valid_bbox(x.get("bbox_percent")):
            errors.append(f"item[{i}] requires a legal bbox_percent")
    return {"ok": not errors, "errors": errors, "item_count": len(p["items"])}
