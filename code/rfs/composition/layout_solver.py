"""Deterministic candidate-based global layout and connector solver."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .arrow_routing import style_and_route_arrows
from .layout_repairs import apply_layout_constraints


def _valid(box: object) -> bool:
    try:
        return isinstance(box, dict) and float(box["w"]) > 0 and float(box["h"]) > 0
    except Exception:
        return False


def _overlap(a: dict, b: dict) -> float:
    return max(0.0, min(float(a["x"]) + float(a["w"]), float(b["x"]) + float(b["w"])) - max(float(a["x"]), float(b["x"]))) * max(0.0, min(float(a["y"]) + float(a["h"]), float(b["y"]) + float(b["h"])) - max(float(a["y"]), float(b["y"])))


def _contains(container: dict, inner: dict) -> bool:
    return float(container["x"]) <= float(inner["x"]) and float(container["y"]) <= float(inner["y"]) and float(container["x"]) + float(container["w"]) >= float(inner["x"]) + float(inner["w"]) and float(container["y"]) + float(container["h"]) >= float(inner["y"]) + float(inner["h"])


def _clamp(box: dict) -> None:
    box["x"] = round(min(1.0 - float(box["w"]), max(0.0, float(box["x"]))), 4)
    box["y"] = round(min(1.0 - float(box["h"]), max(0.0, float(box["y"]))), 4)


def _movable_items(program: dict[str, Any]) -> list[dict[str, Any]]:
    items = list((program.get("text_program", {}) or {}).get("items", []) or [])
    for key in ("formulas", "labels", "relation_labels", "ports"):
        items.extend(program.get(key, []) or [])
    result = [item for item in items if isinstance(item, dict) and item.get("visible", True) and _valid(item.get("bbox_percent"))]
    for item in result:
        if not item.get("owner_id") and item.get("target_id"):
            item["owner_id"] = item["target_id"]
    return result


def _fixed_items(program: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for key in ("slots", "semantic_nodes", "cards"):
        result.extend(item for item in program.get(key, []) or [] if isinstance(item, dict) and _valid(item.get("bbox_percent")))
    return result


def _candidate(program: dict[str, Any], axis: str, direction: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate = deepcopy(program)
    movable = _movable_items(candidate)
    fixed = _fixed_items(candidate)
    fixed_by_id = {str(item.get("id") or ""): item for item in fixed}
    changes = []
    for index, item in enumerate(movable):
        box = item["bbox_percent"]
        before = dict(box)
        for _ in range(24):
            owner = str(item.get("owner_id") or item.get("parent_id") or "")
            blockers = [other["bbox_percent"] for other in movable[index + 1:] if _overlap(box, other["bbox_percent"]) > 0]
            owner_geometry = (fixed_by_id.get(owner) or {}).get("bbox_percent") if owner else None
            blockers.extend(other["bbox_percent"] for other in fixed if str(other.get("id") or "") != owner and _overlap(box, other["bbox_percent"]) > 0 and not (owner_geometry and _contains(other["bbox_percent"], owner_geometry)))
            if not blockers:
                break
            delta = max(0.006, float(box.get("h" if axis == "y" else "w", 0.04)) * 0.55) * direction
            box[axis] = float(box.get(axis, 0)) + delta
            _clamp(box)
            owner_box = (fixed_by_id.get(owner) or {}).get("bbox_percent") if owner else None
            if _valid(owner_box):
                box["w"] = round(min(float(box["w"]), float(owner_box["w"]) * 0.94), 4)
                box["h"] = round(min(float(box["h"]), float(owner_box["h"]) * 0.90), 4)
                box["x"] = round(min(max(float(box["x"]), float(owner_box["x"]) + 0.003), float(owner_box["x"]) + float(owner_box["w"]) - float(box["w"]) - 0.003), 4)
                box["y"] = round(min(max(float(box["y"]), float(owner_box["y"]) + 0.003), float(owner_box["y"]) + float(owner_box["h"]) - float(box["h"]) - 0.003), 4)
        if box != before:
            changes.append({"object_id": item.get("id"), "before": before, "after": dict(box)})
    return candidate, changes


def _score(program: dict[str, Any], missing_edges: list[str]) -> dict[str, Any]:
    movable = _movable_items(program)
    fixed = _fixed_items(program)
    peer_overlaps = []
    for index, left in enumerate(movable):
        for right in movable[index + 1:]:
            if _overlap(left["bbox_percent"], right["bbox_percent"]) > 0:
                peer_overlaps.append([left.get("id"), right.get("id")])
    # Fixed-object containment is often intentional. Only conflicts with an
    # explicit different owner are blocking.
    ownership_conflicts = []
    for item in movable:
        owner = str(item.get("owner_id") or item.get("parent_id") or "")
        for fixed_item in fixed:
            fixed_id = str(fixed_item.get("id") or "")
            owner_geometry = next((obj["bbox_percent"] for obj in fixed if str(obj.get("id") or "") == owner), None)
            if owner and fixed_id not in {owner, ""} and _overlap(item["bbox_percent"], fixed_item["bbox_percent"]) > 0 and not (owner_geometry and _contains(fixed_item["bbox_percent"], owner_geometry)):
                ownership_conflicts.append([item.get("id"), fixed_id])
    label_collisions = []
    labels = [item for key in ("labels", "relation_labels") for item in program.get(key, []) or [] if isinstance(item, dict) and _valid(item.get("bbox_percent"))]
    obstacles = fixed + [item for item in movable if item not in labels]
    for label in labels:
        for obstacle in obstacles:
            if str(label.get("owner_id") or label.get("parent_id") or "") == str(obstacle.get("id") or ""):
                continue
            if _overlap(label["bbox_percent"], obstacle["bbox_percent"]) > 0:
                label_collisions.append([label.get("id"), obstacle.get("id")])
    hard = len(peer_overlaps) + len(ownership_conflicts) + len(label_collisions) + len(missing_edges)
    displacement = sum(abs(float(item["bbox_percent"]["x"]) - 0.5) + abs(float(item["bbox_percent"]["y"]) - 0.5) for item in movable)
    return {"hard_fault_count": hard, "peer_overlaps": peer_overlaps, "ownership_conflicts": ownership_conflicts, "label_collisions": label_collisions, "missing_edge_ids": missing_edges, "score": round(hard * 10000 + displacement, 4)}


def solve_layout_constraints(program: dict[str, Any], out_dir: str | Path, *, reroute: bool = True, params: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose the best deterministic layout candidate and reroute connectors.

    ``params`` carries the mechanism-level layout rules a Memory row learned. They are applied
    to the candidate the solver selected, before the connectors are rerouted against it, so a
    rule about the drawing reaches a figure the row never saw.
    """
    program = deepcopy(program)
    endpoint_ids = {str(item.get("id")) for key in ("slots", "semantic_nodes", "cards", "panels") for item in program.get(key, []) or [] if isinstance(item, dict) and item.get("id")}
    missing_edges = []
    for arrow in program.get("arrows", []) or []:
        if not isinstance(arrow, dict):
            continue
        source = str(arrow.get("source_id") or arrow.get("source") or "")
        target = str(arrow.get("target_id") or arrow.get("target") or "")
        if source not in endpoint_ids or target not in endpoint_ids:
            missing_edges.append(str(arrow.get("id") or ""))
        path = arrow.get("path_percent") if isinstance(arrow.get("path_percent"), list) else []
        arrow["auto_reroute_allowed"] = len(path) <= 2 or str(arrow.get("route_policy") or "").lower() == "fallback_reroute_allowed"

    candidates = []
    for candidate_id, axis, direction in (("down", "y", 1.0), ("up", "y", -1.0), ("right", "x", 1.0), ("left", "x", -1.0)):
        candidate, changes = _candidate(program, axis, direction)
        metrics = _score(candidate, missing_edges)
        candidates.append((metrics["score"], candidate_id, candidate, changes, metrics))
    _, selected_id, updated, changes, metrics = min(candidates, key=lambda item: (item[0], item[1]))
    updated, layout_constraint_report = apply_layout_constraints(updated, params)
    if reroute:
        updated = style_and_route_arrows(updated, out_dir, mode="reference", repair_routing=True)
    routed = [item.get("id") for item in updated.get("arrows", []) or [] if item.get("route_generation_status") == "repair_route_selected"]
    unusable = [str(item.get("id") or "") for item in updated.get("arrows", []) or [] if len(item.get("path_percent") or []) < 2]
    missing_edges = list(dict.fromkeys(missing_edges + unusable))
    updated["control_shapes"] = [dict(item) for item in updated.get("arrows", []) or []]
    blocked = bool(metrics["hard_fault_count"] or missing_edges)
    return updated, {
        "summary": "Deterministic candidate-based global layout constraint solve.",
        "status": "blocked" if blocked else "pass",
        "selected_candidate_id": selected_id,
        "candidate_scores": [{"candidate_id": item[1], **item[4]} for item in candidates],
        "object_relocation_count": len(changes),
        "text_relocation_count": len(changes),
        "object_relocations": changes,
        "unresolved_text_pairs": metrics["peer_overlaps"],
        "ownership_conflicts": metrics["ownership_conflicts"],
        "label_collisions": metrics["label_collisions"],
        "rerouted_edge_ids": routed,
        "missing_or_orphan_edge_ids": missing_edges,
        "routing_policy": "all-visible-object-obstacle-aware",
        "semantic_object_policy": "required objects and relations are never deleted by layout repair",
        "layout_constraints": layout_constraint_report["rules"],
        "layout_constraint_report": layout_constraint_report,
        "hard_constraints": {"object_overlap_count": metrics["hard_fault_count"], "missing_connector_path_count": len(missing_edges)},
    }
