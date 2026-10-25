"""Deterministic joint node/asset geometry pass before connector routing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _area(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(0.0, min(float(a["x"]) + float(a["w"]), float(b["x"]) + float(b["w"])) - max(float(a["x"]), float(b["x"]))) * max(0.0, min(float(a["y"]) + float(a["h"]), float(b["y"]) + float(b["h"])) - max(float(a["y"]), float(b["y"])))


def _shift(box: dict[str, Any], dx: float, dy: float) -> None:
    box["x"] = round(float(box["x"]) + dx, 4)
    box["y"] = round(float(box["y"]) + dy, 4)


def _clamp(box: dict[str, Any], panel: dict[str, Any]) -> tuple[float, float]:
    dx = max(float(panel["x"]) + 0.012 - float(box["x"]), min(float(panel["x"]) + float(panel["w"]) - 0.012 - float(box["x"]) - float(box["w"]), 0.0))
    dy = max(float(panel["y"]) + 0.012 - float(box["y"]), min(float(panel["y"]) + float(panel["h"]) - 0.012 - float(box["y"]) - float(box["h"]), 0.0))
    return dx, dy


def _collision_total(box: dict[str, Any], peers: list[dict[str, Any]]) -> float:
    return sum(_area(box, other) for other in peers)


def _is_container(node: dict[str, Any]) -> bool:
    """A container frame is drawn around its children, so it always overlaps them."""

    return (
        str(node.get("render_policy") or "").casefold() in {"panel_only", "ppt_native_container"}
        or str(node.get("object_type") or "").casefold() == "container_anchor"
        or str(node.get("visual_role") or "").casefold() == "semantic_container"
    )


def _descends_from(cursor: str, target: str, parent_of: dict[str, str]) -> bool:
    seen: set[str] = set()
    while cursor and cursor not in seen:
        if cursor == target:
            return True
        seen.add(cursor)
        cursor = parent_of.get(cursor, "")
    return False


def _displacement_candidates(anchor: dict[str, Any], moving: dict[str, Any]) -> list[tuple[float, float]]:
    """Right, down, left and up - in the order the router has always used."""

    x_right = dict(moving)
    _shift(x_right, float(anchor["x"]) + float(anchor["w"]) + 0.012 - float(moving["x"]), 0.0)
    y_down = dict(moving)
    _shift(y_down, 0.0, float(anchor["y"]) + float(anchor["h"]) + 0.012 - float(moving["y"]))
    x_left = dict(moving)
    _shift(x_left, float(anchor["x"]) - 0.012 - float(moving["x"]) - float(moving["w"]), 0.0)
    y_up = dict(moving)
    _shift(y_up, 0.0, float(anchor["y"]) - 0.012 - float(moving["y"]) - float(moving["h"]))
    return [(x_right["x"] - float(moving["x"]), x_right["y"] - float(moving["y"])), (y_down["x"] - float(moving["x"]), y_down["y"] - float(moving["y"])), (x_left["x"] - float(moving["x"]), x_left["y"] - float(moving["y"])), (y_up["x"] - float(moving["x"]), y_up["y"] - float(moving["y"]))]


def resolve_joint_geometry(program: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate conflicting peer nodes/assets and invalidate stale routes."""
    updated = deepcopy(program)
    nodes = [node for node in updated.get("semantic_nodes", []) if isinstance(node, dict) and isinstance(node.get("bbox_percent"), dict)]
    panels = {str(panel.get("id")): panel for panel in updated.get("panels", []) if isinstance(panel, dict) and isinstance(panel.get("bbox_percent"), dict)}
    slots = {str(slot.get("parent_id") or str(slot.get("id") or "").removeprefix("slot_")): slot for slot in updated.get("slots", []) if isinstance(slot, dict) and isinstance(slot.get("bbox_percent"), dict)}
    parent_of = {str(node.get("id")): str(node.get("parent_id") or "") for node in nodes}
    changes, unresolved = [], []
    for index, left in enumerate(nodes):
        for right in nodes[index + 1:]:
            if str(left.get("panel_id")) != str(right.get("panel_id")):
                continue
            # A container frame drawn around its descendants is not a
            # collision; treating it as one pushes the payload out of the
            # container and, once clamped back, on top of itself.
            if _is_container(left) or _is_container(right):
                continue
            if _descends_from(str(left.get("id")), str(right.get("id")), parent_of) or _descends_from(str(right.get("id")), str(left.get("id")), parent_of):
                continue
            if str(left.get("parent_id") or "") == str(right.get("id") or "") or str(right.get("parent_id") or "") == str(left.get("id") or ""):
                # Parent/child containment is a semantic relationship; only
                # repair it when the child is not actually contained.
                lb, rb = left["bbox_percent"], right["bbox_percent"]
                contained = (float(lb["x"]) <= float(rb["x"]) and float(lb["y"]) <= float(rb["y"]) and float(lb["x"])+float(lb["w"]) >= float(rb["x"])+float(rb["w"]) and float(lb["y"])+float(lb["h"]) >= float(rb["y"])+float(rb["h"])) or (float(rb["x"]) <= float(lb["x"]) and float(rb["y"]) <= float(lb["y"]) and float(rb["x"])+float(rb["w"]) >= float(lb["x"])+float(lb["w"]) and float(rb["y"])+float(rb["h"]) >= float(lb["y"])+float(lb["h"]))
                if contained:
                    continue
            overlap = _area(left["bbox_percent"], right["bbox_percent"])
            if overlap <= 0.0005:
                continue
            panel = panels.get(str(right.get("panel_id")))
            if not panel:
                continue
            rb = right["bbox_percent"]
            lb = left["bbox_percent"]
            peers = [other["bbox_percent"] for other in nodes if other is not right]
            baseline = _collision_total(rb, peers)
            original = dict(rb)
            placed = None
            for offset in _displacement_candidates(lb, rb):
                candidate = dict(rb)
                _shift(candidate, *offset)
                _shift(candidate, *_clamp(candidate, panel["bbox_percent"]))
                # Only accept a move that actually clears this pair without
                # making any other collision worse; otherwise the pass would
                # report repairs while piling cards onto the same spot.
                if _area(candidate, lb) > 0.0005 or _collision_total(candidate, peers) > baseline + 1e-9:
                    continue
                placed = candidate
                break
            if not placed:
                unresolved.append({"node_id": right.get("id"), "blocked_by": left.get("id"), "overlap_area": round(overlap, 6), "reason": "no_separation_candidate_survives_the_panel_clamp"})
                continue
            rb.update(placed)
            total_dx, total_dy = round(float(rb["x"]) - float(original["x"]), 4), round(float(rb["y"]) - float(original["y"]), 4)
            slot = slots.get(str(right.get("id")))
            if slot:
                _shift(slot["bbox_percent"], total_dx, total_dy)
            changes.append({"node_id": right.get("id"), "displaced_by": left.get("id"), "delta": {"x": total_dx, "y": total_dy}, "overlap_area": round(overlap, 6)})
    if changes:
        for arrow in updated.get("arrows", []) or []:
            if isinstance(arrow, dict):
                arrow["path_percent"] = []
    return updated, {"summary": "Joint node/asset geometry resolution before connector routing.", "status": "applied" if changes else "pass", "change_count": len(changes), "changes": changes, "unresolved_count": len(unresolved), "unresolved": unresolved, "routes_invalidated": bool(changes)}


__all__ = ["resolve_joint_geometry"]
