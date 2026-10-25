"""Adapt reference-inspired panels to the current paper's semantic payload.

The stage bands come from a reference template and are usually roomier than the
science a given paper contributes. This module keeps the template geometry and
spreads the paper's own payload inside it, so the delivered slide stays full
without stretching individual cards or stranding the figure on the canvas.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _norm(text: Any) -> str:
    return "".join(ch for ch in str(text or "").casefold() if ch.isalnum())


def _area(box: dict[str, Any]) -> float:
    return float(box.get("w") or 0) * float(box.get("h") or 0)


def _center_inside(outer: dict[str, Any], inner: dict[str, Any]) -> bool:
    center_x = float(inner.get("x") or 0) + float(inner.get("w") or 0) / 2
    center_y = float(inner.get("y") or 0) + float(inner.get("h") or 0) / 2
    return (
        float(outer.get("x") or 0) <= center_x <= float(outer.get("x") or 0) + float(outer.get("w") or 0)
        and float(outer.get("y") or 0) <= center_y <= float(outer.get("y") or 0) + float(outer.get("h") or 0)
    )


def _x_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        float(left["x"]) < float(right["x"]) + float(right["w"]) - 1e-9
        and float(right["x"]) < float(left["x"]) + float(left["w"]) - 1e-9
    )


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return _x_overlap(left, right) and (
        float(left["y"]) < float(right["y"]) + float(right["h"]) - 1e-9
        and float(right["y"]) < float(left["y"]) + float(left["h"]) - 1e-9
    )


def _payload_fits(
    box: dict[str, Any],
    payload: list[dict[str, Any]],
    *,
    inner_top: float,
    inner_bottom: float,
    previously_overlapping: set[tuple[int, int]],
) -> bool:
    """Reject a candidate geometry that overflows the panel body or collides."""

    for node in payload:
        item = node["bbox_percent"]
        if float(item["y"]) < inner_top - 1e-6 or float(item["y"]) + float(item["h"]) > inner_bottom + 1e-6:
            return False
        if float(item["x"]) < float(box["x"]) - 1e-6 or float(item["x"]) + float(item["w"]) > float(box["x"]) + float(box["w"]) + 1e-6:
            return False
    for index, left in enumerate(payload):
        for right in payload[index + 1:]:
            if tuple(sorted((id(left), id(right)))) in previously_overlapping:
                continue
            if _overlap(left["bbox_percent"], right["bbox_percent"]):
                return False
    return True


def _spread_payload_into_panels(
    panels: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    container_ids: set[str],
    *,
    title_lane: float = 0.075,
    side_padding: float = 0.028,
    bottom_padding: float = 0.03,
    min_gap: float = 0.012,
    max_growth: float = 1.6,
) -> list[dict[str, Any]]:
    """Spread a panel's own cards across the band the template already drew.

    A stage band is borrowed from a reference template, so it is usually taller
    and wider than the payload a given paper contributes. That leaves a visible
    stripe of empty panel under the last card, and the two obvious fixes are both
    wrong: shrinking the band strands the figure in the middle of the canvas, and
    re-planning the structure is a lottery. This pass keeps the band exactly
    where the template put it - the slide stays full - and spreads the payload
    inside it instead, growing the cards as one block about their own centre
    until they reach the panel's inner box. Growing the block, rather than each
    card, keeps the cards' relative alignment and therefore the flow line. The
    top of the highest card is never lifted, so the title lane survives, a card
    never leaves its panel, and growth is bounded by ``max_growth`` so a narrow
    card is never stretched into an empty column. A panel whose payload cannot
    grow - because it would collide or overflow - keeps the planner geometry.
    """

    records: list[dict[str, Any]] = []
    for panel in panels:
        box = panel.get("bbox_percent")
        if not isinstance(box, dict):
            continue
        payload = [
            node
            for node in nodes
            if str(node.get("panel_id")) == str(panel.get("id"))
            and str(node.get("id")) not in container_ids
            and isinstance(node.get("bbox_percent"), dict)
        ]
        if len(payload) < 2:
            continue
        inner_left = float(box["x"]) + side_padding
        inner_right = float(box["x"]) + float(box["w"]) - side_padding
        inner_top = float(box["y"]) + title_lane
        inner_bottom = float(box["y"]) + float(box["h"]) - bottom_padding
        if inner_bottom - inner_top <= 0.06 or inner_right - inner_left <= 0.05:
            continue
        boxes = [node["bbox_percent"] for node in payload]
        left = min(float(item["x"]) for item in boxes)
        right = max(float(item["x"]) + float(item["w"]) for item in boxes)
        top = min(float(item["y"]) for item in boxes)
        bottom = max(float(item["y"]) + float(item["h"]) for item in boxes)
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0 or top < inner_top - 1e-6:
            continue
        center_x = (left + right) / 2.0
        # Scaling anchors on the highest card, never above it: the title lane
        # that adapt_design_panels just reserved must survive the spread.
        anchor_top = max(inner_top, top)
        original = {id(node): dict(node["bbox_percent"]) for node in payload}
        previously_overlapping = {
            tuple(sorted((id(first), id(second))))
            for index, first in enumerate(payload)
            for second in payload[index + 1:]
            if _overlap(first["bbox_percent"], second["bbox_percent"])
        }
        for mode in ("both", "vertical"):
            scale_x = 1.0
            if mode == "both":
                headroom_x = min(inner_right - center_x, center_x - inner_left)
                scale_x = min(max_growth, max(1.0, 2.0 * headroom_x / width)) if headroom_x > 0 else 1.0
            # Spreading only ever grows: a payload that is already wider or
            # taller than the inner box is a planning problem, not slack.
            scale_y = min(max_growth, max(1.0, (inner_bottom - anchor_top) / height)) if inner_bottom - bottom > min_gap else 1.0
            if scale_x <= 1.0 + 1e-9 and scale_y <= 1.0 + 1e-9:
                break
            for node in payload:
                item = node["bbox_percent"]
                item["x"] = round(center_x + (float(item["x"]) - center_x) * scale_x, 4)
                item["w"] = round(float(item["w"]) * scale_x, 4)
                item["y"] = round(anchor_top + (float(item["y"]) - top) * scale_y, 4)
                item["h"] = round(float(item["h"]) * scale_y, 4)
            if _payload_fits(box, payload, inner_top=inner_top, inner_bottom=inner_bottom, previously_overlapping=previously_overlapping):
                records.append({
                    "panel_id": panel.get("id"),
                    "reason": "payload_spread_into_panel",
                    "growth_mode": mode,
                    "scale_x": round(scale_x, 4),
                    "scale_y": round(scale_y, 4),
                    "band": {"y": round(float(box["y"]), 4), "h": round(float(box["h"]), 4)},
                    "payload_span": [round(anchor_top, 4), round(anchor_top + height * scale_y, 4)],
                    "card_count": len(payload),
                })
                break
            for node in payload:
                node["bbox_percent"].update(original[id(node)])
    return records


def adapt_design_panels(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Classify container anchors and remove reference zones without payload."""
    updated = deepcopy(plan)
    panels = [item for item in updated.get("panels", []) if isinstance(item, dict)]
    nodes = [item for item in updated.get("nodes", []) if isinstance(item, dict)]
    panel_map = {str(item.get("id")): item for item in panels if item.get("id")}
    node_map = {str(item.get("id")): item for item in nodes if item.get("id")}
    container_ids: set[str] = set()
    hierarchy_containers: list[dict[str, Any]] = []
    for node in nodes:
        panel = panel_map.get(str(node.get("panel_id")))
        if not panel or not isinstance(node.get("bbox_percent"), dict) or not isinstance(panel.get("bbox_percent"), dict):
            continue
        coverage = _area(node["bbox_percent"]) / max(_area(panel["bbox_percent"]), 1e-9)
        if coverage >= 0.68 and (_norm(node.get("label")) == _norm(panel.get("title")) or str(node.get("role")) in {"container", "group"}):
            node.update({"kind": "container", "render_policy": "panel_only", "object_type": "container_anchor"})
            container_ids.add(str(node.get("id")))
        else:
            node.setdefault("kind", "semantic_node")
            node.setdefault("render_policy", "node_with_icon")

    for child in nodes:
        parent_id = str(child.get("parent_id") or "")
        parent = node_map.get(parent_id)
        if (
            not parent
            or not isinstance(parent.get("bbox_percent"), dict)
            or not isinstance(child.get("bbox_percent"), dict)
            or not _center_inside(parent["bbox_percent"], child["bbox_percent"])
        ):
            continue
        parent.update(
            {
                "kind": "container",
                "render_policy": "panel_only",
                "object_type": "container_anchor",
                "visual_role": "semantic_container",
            }
        )
        parent.setdefault("container_child_ids", [])
        if str(child.get("id")) not in parent["container_child_ids"]:
            parent["container_child_ids"].append(str(child.get("id")))
        container_ids.add(parent_id)
        hierarchy_containers.append(
            {"node_id": parent_id, "child_id": str(child.get("id")), "reason": "visible_nested_semantic_child"}
        )

    payload_by_panel = {str(panel.get("id")): [] for panel in panels}
    for node in nodes:
        if str(node.get("id")) not in container_ids:
            payload_by_panel.setdefault(str(node.get("panel_id")), []).append(str(node.get("id")))
    removed, expanded, compacted = [], [], []
    kept = []
    for panel in panels:
        panel_id = str(panel.get("id"))
        payload = payload_by_panel.get(panel_id, [])
        panel["payload_node_ids"] = payload
        panel["visual_role"] = "semantic_panel" if payload else "unsupported_reference_zone"
        if payload:
            kept.append(panel)
        else:
            removed.append({"panel_id": panel_id, "reason": "no_semantic_or_annotation_payload"})
    removed_ids = {item["panel_id"] for item in removed}
    for panel in kept:
        box = panel.get("bbox_percent")
        if not isinstance(box, dict):
            continue
        for removed_panel in panels:
            if str(removed_panel.get("id")) not in removed_ids or not isinstance(removed_panel.get("bbox_percent"), dict):
                continue
            other = removed_panel["bbox_percent"]
            same_row = abs(float(box["y"]) - float(other["y"])) < 0.03 and abs(float(box["h"]) - float(other["h"])) < 0.04
            if same_row and float(other["x"]) > float(box["x"]):
                old = dict(box)
                box["w"] = round(min(0.98 - float(box["x"]), float(other["x"]) + float(other["w"]) - float(box["x"])), 4)
                expanded.append({"panel_id": panel.get("id"), "from": old, "to": dict(box), "absorbed_panel": removed_panel.get("id")})
        header_bottom = float(box["y"]) + min(0.075, float(box["h"]) * 0.24)
        for node in nodes:
            if str(node.get("panel_id")) != str(panel.get("id")) or str(node.get("id")) in container_ids or not isinstance(node.get("bbox_percent"), dict):
                continue
            node_box = node["bbox_percent"]
            if float(node_box["y"]) < header_bottom:
                old = dict(node_box)
                bottom = float(node_box["y"]) + float(node_box["h"])
                node_box["y"] = round(header_bottom + 0.008, 4)
                node_box["h"] = round(max(0.035, bottom - float(node_box["y"])), 4)
                expanded.append({"node_id": node.get("id"), "from": old, "to": dict(node_box), "reason": "reserve_panel_title_lane"})
        # A panel with one small payload is a card, not a macro-container.
        # Resize the container around its content while preserving the node's
        # coordinates and all semantic bindings. This avoids large empty
        # sidebars when a paper contributes only one input/output concept.
        payload_nodes = [node for node in nodes if str(node.get("panel_id")) == str(panel.get("id")) and str(node.get("id")) not in container_ids and isinstance(node.get("bbox_percent"), dict)]
        if len(payload_nodes) == 1 and isinstance(box, dict):
            child = payload_nodes[0]["bbox_percent"]
            panel_area = _area(box)
            child_area = _area(child)
            role = str(panel.get("role") or "").casefold()
            if panel_area > 0.08 or child_area / max(panel_area, 1e-9) < 0.42:
                pad_x = 0.035
                pad_top = 0.075
                pad_bottom = 0.055
                old = dict(box)
                left = max(0.015, float(child["x"]) - pad_x)
                top = max(0.04, float(child["y"]) - pad_top)
                right = min(0.985, float(child["x"]) + float(child["w"]) + pad_x)
                bottom = min(0.96, float(child["y"]) + float(child["h"]) + pad_bottom)
                box.update({"x": round(left, 4), "y": round(top, 4), "w": round(max(0.06, right - left), 4), "h": round(max(0.12, bottom - top), 4)})
                # Reflow the payload into the compact card as well. Merely
                # shrinking the panel leaves the original tiny node stranded
                # in a large amount of whitespace. Keep a title lane and give
                # the semantic payload a readable card-sized footprint.
                child = payload_nodes[0]["bbox_percent"]
                inner_x, inner_y = float(box["x"]) + 0.06 * float(box["w"]), float(box["y"]) + 0.28 * float(box["h"])
                inner_w, inner_h = 0.88 * float(box["w"]), 0.58 * float(box["h"])
                child.update({"x": round(inner_x, 4), "y": round(inner_y, 4), "w": round(max(0.045, inner_w), 4), "h": round(max(0.055, inner_h), 4)})
                compacted.append({"panel_id": panel.get("id"), "from": old, "to": dict(box), "node_id": payload_nodes[0].get("id"), "child_from": {"x": child.get("x"), "y": child.get("y"), "w": child.get("w"), "h": child.get("h")}, "reason": "single_low_information_payload_reflowed"})
    for item in _spread_payload_into_panels(kept, nodes, container_ids):
        compacted.append(item)
    updated["panels"] = kept
    updated["nodes"] = [node for node in nodes if str(node.get("panel_id")) not in removed_ids]
    valid_nodes = {str(node.get("id")) for node in updated["nodes"]}
    updated["connectors"] = [edge for edge in updated.get("connectors", []) if str(edge.get("source")) in valid_nodes and str(edge.get("target")) in valid_nodes]
    return updated, {"summary": "Semantic adaptation of reference-inspired zones.", "status": "adapted", "container_node_ids": sorted(container_ids), "hierarchy_containers": hierarchy_containers, "removed": removed, "expanded": expanded, "compacted": compacted, "kept_panel_ids": [str(item.get("id")) for item in kept]}


__all__ = ["adapt_design_panels"]
