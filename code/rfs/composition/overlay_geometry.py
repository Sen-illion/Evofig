"""Keep a composition overlay from overwriting resolved figure geometry.

The PPT composition stage is a styling and labelling overlay: it decides how the
figure is drawn, not what shape it has. It is also a language model, so it
happily returns x/y/w/h for every object it looked at - and those boxes silently
replace geometry the structure stage had already resolved. That is how a card
ends up drawn on top of its neighbour while every deterministic gate still
reports a clean layout: the collision arrived with the overlay, after the
geometry pass had finished.

This module lets an overlay restyle and relabel freely, but only keeps an
overlay box while the merged layout still satisfies the geometry contract. A box
that collides, escapes its panel or leaves the canvas is dropped and the
resolved box is kept instead, with the reason recorded.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _valid_bbox(box: Any) -> bool:
    if not isinstance(box, dict):
        return False
    try:
        x = float(box["x"])
        y = float(box["y"])
        width = float(box["w"])
        height = float(box["h"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0 and x + width <= 1.001 and y + height <= 1.001


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        float(left["x"]) < float(right["x"]) + float(right["w"]) - 1e-9
        and float(right["x"]) < float(left["x"]) + float(left["w"]) - 1e-9
        and float(left["y"]) < float(right["y"]) + float(right["h"]) - 1e-9
        and float(right["y"]) < float(left["y"]) + float(left["h"]) - 1e-9
    )


def _covers(outer: dict[str, Any], inner: dict[str, Any]) -> bool:
    return (
        float(inner["x"]) >= float(outer["x"]) - 1e-6
        and float(inner["y"]) >= float(outer["y"]) - 1e-6
        and float(inner["x"]) + float(inner["w"]) <= float(outer["x"]) + float(outer["w"]) + 1e-6
        and float(inner["y"]) + float(inner["h"]) <= float(outer["y"]) + float(outer["h"]) + 1e-6
    )


def _entry(item: dict[str, Any], overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    override = overrides.get(str(item.get("id")))
    merged = {**item, **override} if isinstance(override, dict) else dict(item)
    return {"structure": item, "merged": merged, "override": override if isinstance(override, dict) else None}


def _overrides_geometry(entry: dict[str, Any]) -> bool:
    return isinstance(entry.get("override"), dict) and "bbox_percent" in entry["override"]


def _record(entry: dict[str, Any], kind: str, reason: str) -> dict[str, Any]:
    return {
        "object_type": kind,
        "object_id": str(entry["merged"].get("id") or ""),
        "reason": reason,
        "rejected_bbox_percent": entry["merged"].get("bbox_percent"),
        "kept_bbox_percent": entry["structure"].get("bbox_percent"),
    }


def guard_overlay_geometry(
    layout: dict[str, Any],
    *,
    panel_overrides: dict[str, dict[str, Any]] | None = None,
    node_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge overlay fields, keeping geometry that still satisfies the contract."""

    panel_map = panel_overrides if isinstance(panel_overrides, dict) else {}
    node_map = node_overrides if isinstance(node_overrides, dict) else {}
    source = layout if isinstance(layout, dict) else {}
    panel_entries = [_entry(item, panel_map) for item in source.get("panels") or [] if isinstance(item, dict)]
    node_entries = [_entry(item, node_map) for item in source.get("nodes") or [] if isinstance(item, dict)]
    records: list[dict[str, Any]] = []

    # Panels first: a rejected panel box would invalidate every node box that
    # still refers to it, so the panels have to be final before nodes are judged.
    for entry in panel_entries:
        if not _overrides_geometry(entry):
            continue
        box = entry["merged"].get("bbox_percent")
        peers = [other["merged"].get("bbox_percent") for other in panel_entries if other is not entry]
        peers = [item for item in peers if _valid_bbox(item)]
        panel_id = str(entry["merged"].get("id"))
        escapes = [
            node["merged"].get("bbox_percent")
            for node in node_entries
            if str(node["merged"].get("panel_id") or "") == panel_id
            and _valid_bbox(node["merged"].get("bbox_percent"))
            and _valid_bbox(box)
            and not _covers(box, node["merged"]["bbox_percent"])
        ]
        if _valid_bbox(box) and not any(_overlap(box, item) for item in peers) and not escapes:
            continue
        records.append(_record(entry, "panel", "overlay panel box collides, drops payload outside it, or leaves the canvas"))
        entry["merged"]["bbox_percent"] = entry["structure"].get("bbox_percent")

    panel_boxes = {
        str(entry["merged"].get("id")): entry["merged"]["bbox_percent"]
        for entry in panel_entries
        if _valid_bbox(entry["merged"].get("bbox_percent"))
    }
    for _ in range(3):
        changed = False
        for entry in node_entries:
            if not _overrides_geometry(entry):
                continue
            box = entry["merged"].get("bbox_percent")
            peers = [other["merged"].get("bbox_percent") for other in node_entries if other is not entry]
            peers = [item for item in peers if _valid_bbox(item)]
            owner = panel_boxes.get(str(entry["merged"].get("panel_id") or ""))
            if (
                _valid_bbox(box)
                and not any(_overlap(box, item) for item in peers)
                and (owner is None or _covers(owner, box))
            ):
                continue
            records.append(_record(entry, "node", "overlay node box collides, escapes its panel, or leaves the canvas"))
            entry["merged"]["bbox_percent"] = entry["structure"].get("bbox_percent")
            changed = True
        if not changed:
            break

    guarded = {
        **{key: value for key, value in source.items() if key not in {"panels", "nodes"}},
        "panels": [entry["merged"] for entry in panel_entries],
        "nodes": [entry["merged"] for entry in node_entries],
    }
    return guarded, {
        "summary": "Overlay geometry that broke the resolved layout was rejected.",
        "policy": "an overlay may restyle and relabel an object; it may not move it into a collision or out of its panel",
        "status": "corrected" if records else "pass",
        "rejected_count": len(records),
        "rejected": records,
    }



def keep_panels_below_title_band(
    panels: list[dict[str, Any]],
    title_block: dict[str, Any] | None,
    *,
    gap: float = 0.008,
    minimum_height: float = 0.08,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep every panel below the band the paper title is drawn in.

    The title owns the top of the canvas. The composition overlay may re-pose
    the panels it looked at, and it measures those boxes from the canvas edge
    rather than from the title, so a panel can end up drawn across the title
    text. No other gate sees that overlap: the cards still sit inside their
    panel, and the title is not an object the geometry gates know about. The
    overlap therefore has to be settled here, by measurement, once the panel
    list is final.

    Only the title edge moves. A panel keeps its own left, right and bottom and
    is shortened by however much it had reached above the title lane; a panel
    left with no usable height gets the minimum instead of a negative box.
    """

    band = title_block.get("bbox_percent") if isinstance(title_block, dict) else None
    if not isinstance(band, dict):
        return panels, {"summary": "This canvas draws no title band.", "status": "not_applicable", "adjustment_count": 0, "adjustments": []}
    floor = float(band.get("y") or 0.0) + float(band.get("h") or 0.0) + gap
    updated = deepcopy(panels)
    adjustments: list[dict[str, Any]] = []
    for panel in updated:
        box = panel.get("bbox_percent") if isinstance(panel, dict) else None
        if not isinstance(box, dict):
            continue
        try:
            top, height = float(box["y"]), float(box["h"])
        except (KeyError, TypeError, ValueError):
            continue
        bottom = top + height
        if top >= floor - 1e-9:
            continue
        kept_height = bottom - floor
        if kept_height < minimum_height:
            kept_height = minimum_height
        box["y"] = round(floor, 4)
        box["h"] = round(kept_height, 4)
        adjustments.append({
            "panel_id": str(panel.get("id") or ""),
            "from": {"y": round(top, 4), "h": round(height, 4)},
            "to": {"y": box["y"], "h": box["h"]},
            "reason": "panel_reached_into_the_title_band",
        })
    return updated, {
        "summary": "Panels were kept below the band the paper title is drawn in.",
        "policy": "the title band is reserved; a panel is shortened from the top, never moved sideways or downward",
        "status": "adjusted" if adjustments else "pass",
        "title_band_bottom": round(floor, 4),
        "adjustment_count": len(adjustments),
        "adjustments": adjustments,
    }


__all__ = ["guard_overlay_geometry", "keep_panels_below_title_band"]
