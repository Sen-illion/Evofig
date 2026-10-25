"""Global canvas safe-area constraints for semantic figure designs."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any


def _read_number(value: Any) -> float | None:
    """Read one rectangle number, or ``None`` when the model wrote no number."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _readable_box(item: dict[str, Any]) -> tuple[dict[str, float] | None, dict[str, float] | None]:
    """Return the object's box as four numbers, or ``None`` when it has no readable box.

    The design model sometimes leaves a box half-drawn -- a rectangle object with
    no height, or a corner list where one corner did not parse. Reading it raised
    ``KeyError`` and the paper lost every remaining attempt to the same crash. A
    box whose numbers are all readable is fitted here; a box that is not is left
    exactly as the model wrote it, counted, and named in the returned report so the
    validator can ask for it instead of the fitter guessing a rectangle.
    """
    raw = item.get("bbox_percent")
    if not isinstance(raw, dict):
        return None, None
    values = {key: _read_number(raw.get(key)) for key in ("x", "y", "w", "h")}
    missing = [key for key, value in values.items() if value is None]
    if missing:
        partial = {key: value for key, value in values.items() if value is not None}
        return None, partial or None
    box = {key: values[key] for key in ("x", "y", "w", "h")}
    return {key: float(box[key]) for key in ("x", "y", "w", "h")}, None


def fit_design_to_vertical_safe_area(
    plan: dict[str, Any],
    *,
    top: float = 0.11,
    bottom: float = 0.96,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit all planned objects into a reserved vertical band without re-layout."""
    updated = deepcopy(plan)
    candidates = [
        item
        for key in ("panels", "nodes", "groups")
        for item in updated.get(key, [])
        if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)
    ]
    objects: list[tuple[dict[str, Any], dict[str, float]]] = []
    unreadable: list[dict[str, Any]] = []
    for item in candidates:
        box, partial = _readable_box(item)
        if box is None:
            unreadable.append({"id": str(item.get("id") or ""), "readable": partial or {}})
            continue
        objects.append((item, box))
    unreadable_report = {"count": len(unreadable), "objects": unreadable} if unreadable else {"count": 0, "objects": []}
    if not objects:
        return updated, {"status": "not_applicable", "top": top, "bottom": bottom, "transformed": False, "object_count": 0, "unreadable_boxes": unreadable_report}
    minimum = min(box["y"] for _, box in objects)
    maximum = max(box["y"] + box["h"] for _, box in objects)
    if minimum >= top and maximum <= bottom:
        return updated, {"status": "pass", "top": top, "bottom": bottom, "transformed": False, "object_count": len(objects), "source_extent": [round(minimum, 4), round(maximum, 4)], "unreadable_boxes": unreadable_report}
    source_height = max(1e-9, maximum - minimum)
    scale = (bottom - top) / source_height

    def transform_y(value: float) -> float:
        return top + (float(value) - minimum) * scale

    for item, _box in objects:
        box = item["bbox_percent"]
        box["y"] = round(transform_y(float(box["y"])), 4)
        box["h"] = round(float(box["h"]) * scale, 4)
    for connector in updated.get("connectors", []):
        if not isinstance(connector, dict) or not isinstance(connector.get("path_percent"), list):
            continue
        connector["path_percent"] = [
            [float(point[0]), round(transform_y(float(point[1])), 4)]
            for point in connector["path_percent"]
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
    return updated, {
        "status": "adjusted",
        "top": top,
        "bottom": bottom,
        "transformed": True,
        "object_count": len(objects),
        "source_extent": [round(minimum, 4), round(maximum, 4)],
        "scale": round(scale, 6),
        "policy": "preserve_relative_geometry_with_vertical_affine_fit",
        "unreadable_boxes": unreadable_report,
    }


__all__ = ["fit_design_to_vertical_safe_area"]
