"""Final authoritative connector geometry and path-bound label placement."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import hypot
from typing import Any


# Feedback arrows used to be forced onto fixed canvas-edge lanes, which sent
# adjacent boxes on a detour to the slide border and back. Loops now hug the
# two endpoints first; the canvas edge stays available as a last resort.
FEEDBACK_CLEARANCE = 0.05


@dataclass(frozen=True)
class ConnectorRoutePolicy:
    """Tunable weights for connector route selection.

    The router is parameterised rather than hard-coded because the self-evolution
    loop is meant to tune these numbers instead of the routing code. A program
    may carry its own ``route_policy`` mapping to override them for one figure,
    and the values that produced a route are echoed in the routing report.

    ``obstacle_rank`` and ``crossing_rank`` are rank separators, not ink: any
    number of extra turns is still cheaper than slicing another line, and any
    crossing is still cheaper than drawing through a box.
    """

    obstacle_rank: float = 1_000_000.0
    crossing_rank: float = 1_000.0
    bend_penalty: float = 0.035
    slant_penalty: float = 1.4
    # Two boxes only get a straight link when the arrow can leave one face and
    # enter the other on a lane both of them cover. A hairline overlap would
    # graze two corners, so a narrow one falls back to the plain face-mid points.
    shared_lane_min_span: float = 0.015
    # A blocked link steps just past the obstacle instead of running out to the
    # slide border; the wide canvas lanes stay available as the last resort.
    detour_offsets: tuple[float, ...] = (0.02, 0.035, 0.06, 0.10)


DEFAULT_ROUTE_POLICY = ConnectorRoutePolicy()


def _route_policy_for(program: dict[str, Any], override: ConnectorRoutePolicy | None) -> ConnectorRoutePolicy:
    """Routing weights for one program: explicit argument, its own mapping, default."""

    if isinstance(override, ConnectorRoutePolicy):
        return override
    raw = program.get("route_policy") if isinstance(program, dict) else None
    if not isinstance(raw, dict):
        return DEFAULT_ROUTE_POLICY
    allowed = {
        field: float(raw[field]) for field in DEFAULT_ROUTE_POLICY.__dataclass_fields__
        if isinstance(raw.get(field), (int, float)) and not isinstance(raw.get(field), bool)
    }
    offsets = raw.get("detour_offsets")
    if isinstance(offsets, (list, tuple)) and offsets:
        numbers = [float(value) for value in offsets if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if len(numbers) == len(offsets):
            allowed["detour_offsets"] = tuple(numbers)
    return replace(DEFAULT_ROUTE_POLICY, **allowed) if allowed else DEFAULT_ROUTE_POLICY

STYLE_BY_RELATION = {
    "data_flow": {"route_style": "rounded_elbow", "stroke_color": "#496A7C", "stroke_width_pt": 1.8, "arrowhead_size": "med", "line_pattern": "solid"},
    "prediction": {"route_style": "soft_curve", "stroke_color": "#2A63A1", "stroke_width_pt": 2.2, "arrowhead_size": "med", "line_pattern": "solid"},
    "feedback": {"route_style": "dashed_loop", "stroke_color": "#7251A1", "stroke_width_pt": 2.0, "arrowhead_size": "med", "line_pattern": "dash"},
}


def _center(box: dict[str, Any]) -> tuple[float, float]:
    return float(box["x"]) + float(box["w"]) / 2, float(box["y"]) + float(box["h"]) / 2


def _shared_lane(lo_a: float, hi_a: float, lo_b: float, hi_b: float, min_span: float) -> float | None:
    """Midpoint of the span two facing edges share, or None when they miss."""

    lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
    if hi - lo < min_span:
        return None
    return round((lo + hi) / 2, 4)


def _anchors(source: dict[str, Any], target: dict[str, Any], policy: ConnectorRoutePolicy = DEFAULT_ROUTE_POLICY) -> tuple[list[float], list[float]]:
    """Exit and entry points, placed on a shared lane so the link can be straight.

    Reference figures line their boxes up so a connector is one clean stroke.
    When the two facing edges overlap, both endpoints sit on that overlap, which
    turns the link into a plain horizontal/vertical line instead of an elbow
    that only exists because the two centres disagree.
    """

    sb, tb = source["bbox_percent"], target["bbox_percent"]
    sx, sy = _center(sb); tx, ty = _center(tb)
    horizontal = abs(tx - sx) >= abs(ty - sy)
    lane = _shared_lane(
        float(sb["y"]) if horizontal else float(sb["x"]),
        float(sb["y"]) + float(sb["h"]) if horizontal else float(sb["x"]) + float(sb["w"]),
        float(tb["y"]) if horizontal else float(tb["x"]),
        float(tb["y"]) + float(tb["h"]) if horizontal else float(tb["x"]) + float(tb["w"]),
        policy.shared_lane_min_span,
    )
    if lane is not None:
        if horizontal and tx >= sx and float(sb["x"]) + float(sb["w"]) <= float(tb["x"]):
            return [round(float(sb["x"]) + float(sb["w"]), 4), lane], [round(float(tb["x"]), 4), lane]
        if horizontal and tx < sx and float(tb["x"]) + float(tb["w"]) <= float(sb["x"]):
            return [round(float(sb["x"]), 4), lane], [round(float(tb["x"]) + float(tb["w"]), 4), lane]
        if not horizontal and ty >= sy and float(sb["y"]) + float(sb["h"]) <= float(tb["y"]):
            return [lane, round(float(sb["y"]) + float(sb["h"]), 4)], [lane, round(float(tb["y"]), 4)]
        if not horizontal and ty < sy and float(tb["y"]) + float(tb["h"]) <= float(sb["y"]):
            return [lane, round(float(sb["y"]), 4)], [lane, round(float(tb["y"]) + float(tb["h"]), 4)]
    if horizontal:
        start = [float(sb["x"]) + (float(sb["w"]) if tx >= sx else 0.0), sy]
        end = [float(tb["x"]) + (0.0 if tx >= sx else float(tb["w"])), ty]
    else:
        start = [sx, float(sb["y"]) + (float(sb["h"]) if ty >= sy else 0.0)]
        end = [tx, float(tb["y"]) + (0.0 if ty >= sy else float(tb["h"]))]
    return start, end


def _path(source: dict[str, Any], target: dict[str, Any], relation_type: str, policy: ConnectorRoutePolicy = DEFAULT_ROUTE_POLICY) -> list[list[float]]:
    start, end = _anchors(source, target, policy)
    if relation_type == "feedback":
        lane = min(0.96, max(start[1], end[1]) + 0.08)
        return [start, [start[0], lane], [end[0], lane], end]
    if relation_type == "prediction" and abs(end[0] - start[0]) >= 0.035 and abs(end[1] - start[1]) >= 0.035:
        dx, dy = end[0] - start[0], end[1] - start[1]
        bend = max(0.045, min(0.12, hypot(dx, dy) * 0.18))
        control = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2 - bend]
        return [[round((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0], 4), round((1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1], 4)] for t in [i / 8 for i in range(9)]]
    if abs(end[0] - start[0]) < 0.035 or abs(end[1] - start[1]) < 0.035:
        return [start, end]
    middle = (start[0] + end[0]) / 2
    return [start, [middle, start[1]], [middle, end[1]], end]


def _intersects(a: dict[str, float], b: dict[str, Any]) -> bool:
    return not (a["x"] + a["w"] <= float(b["x"]) or float(b["x"]) + float(b["w"]) <= a["x"] or a["y"] + a["h"] <= float(b["y"]) or float(b["y"]) + float(b["h"]) <= a["y"])


def _point_in_box(point: list[float], box: dict[str, Any], margin: float = 0.006) -> bool:
    return float(box["x"]) - margin <= point[0] <= float(box["x"]) + float(box["w"]) + margin and float(box["y"]) - margin <= point[1] <= float(box["y"]) + float(box["h"]) + margin


def _segment_hits_box(a: list[float], b: list[float], box: dict[str, Any]) -> bool:
    return any(_point_in_box([a[0] + (b[0] - a[0]) * index / 20, a[1] + (b[1] - a[1]) * index / 20], box) for index in range(21))


def _path_hits(path: list[list[float]], obstacles: list[dict[str, Any]]) -> int:
    return sum(any(_segment_hits_box(a, b, box) for a, b in zip(path[:-1], path[1:])) for box in obstacles)


def _orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))


def _segments_cross(a: list[float], b: list[float], c: list[float], d: list[float]) -> bool:
    # Proper crossings only; touching at a shared endpoint is intentional.
    values = (_orientation(a, b, c), _orientation(a, b, d), _orientation(c, d, a), _orientation(c, d, b))
    return (values[0] > 0) != (values[1] > 0) and (values[2] > 0) != (values[3] > 0)


def _path_crossings(path: list[list[float]], existing_paths: list[list[list[float]]]) -> int:
    return sum(
        1
        for other in existing_paths
        for a, b in zip(path[:-1], path[1:])
        for c, d in zip(other[:-1], other[1:])
        if _segments_cross(a, b, c, d)
    )


def _path_length(path: list[list[float]]) -> float:
    return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:]))


def _slant_extent(path: list[list[float]]) -> float:
    """How far a two-point link runs off axis, for the proportional slant charge.

    The reference figures draw a near neighbour as one short slanted stroke and
    it reads as deliberate; what they never draw is a long lazy diagonal. The
    charge therefore grows with the off-axis extent instead of vetoing every
    non-axis-aligned stroke outright.
    """

    if len(path) != 2:
        return 0.0
    dx = abs(float(path[1][0]) - float(path[0][0]))
    dy = abs(float(path[1][1]) - float(path[0][1]))
    if dx <= 0.012 or dy <= 0.012:
        return 0.0
    return min(dx, dy)


def _faces_are_adjacent(source: dict[str, Any], target: dict[str, Any], gap: float = 0.08) -> bool:
    """True when two boxes face each other across a narrow gap."""

    sb, tb = source["bbox_percent"], target["bbox_percent"]
    s_right, t_right = float(sb["x"]) + float(sb["w"]), float(tb["x"]) + float(tb["w"])
    s_bottom, t_bottom = float(sb["y"]) + float(sb["h"]), float(tb["y"]) + float(tb["h"])
    x_overlap = min(s_right, t_right) - max(float(sb["x"]), float(tb["x"]))
    y_overlap = min(s_bottom, t_bottom) - max(float(sb["y"]), float(tb["y"]))
    x_gap = max(0.0, max(float(sb["x"]), float(tb["x"])) - min(s_right, t_right))
    y_gap = max(0.0, max(float(sb["y"]), float(tb["y"])) - min(s_bottom, t_bottom))
    return (x_overlap > 0 and y_gap <= gap) or (y_overlap > 0 and x_gap <= gap)


def _add_lane(candidates: list[dict[str, Any]], path_id: str, path: list[list[float]]) -> None:
    """Keep a lane only when it really steps around the boxes.

    A lane on the axis the link already runs along collapses onto the link
    itself; keeping it would offer the scorer a pointless spike past the box and
    back. Only a lane that still has two distinct corners is a route.
    """

    deduped = [point for index, point in enumerate(path) if index == 0 or point != path[index - 1]]
    if len(deduped) == len(path):
        candidates.append({"id": path_id, "path": deduped})


def _local_loop_candidates(source: dict[str, Any], target: dict[str, Any], start: list[float], end: list[float], offsets: tuple[float, ...] | None = None, prefix: str = "local_loop") -> list[dict[str, Any]]:
    """Lanes that stay just outside the two endpoint boxes.

    Feedback arrows have used these lanes since the canvas-edge detour was
    fixed. Every relation type gets them now, because a card parked between two
    boxes is cleared by nudging a couple of percent past it, not by running out
    to the slide border and back.
    """

    sb, tb = source["bbox_percent"], target["bbox_percent"]

    def clamp(value: float) -> float:
        return round(min(0.985, max(0.015, value)), 4)

    if offsets is None:
        offsets = tuple(FEEDBACK_CLEARANCE * step for step in (1, 2, 3, 5))
    left = min(float(sb["x"]), float(tb["x"]))
    right = max(float(sb["x"]) + float(sb["w"]), float(tb["x"]) + float(tb["w"]))
    top = min(float(sb["y"]), float(tb["y"]))
    bottom = max(float(sb["y"]) + float(sb["h"]), float(tb["y"]) + float(tb["h"]))
    candidates: list[dict[str, Any]] = []
    # Try ever wider lanes around the two boxes. A card parked between them
    # blocks the tight lane, so the search has to be able to step out until the
    # loop clears every obstacle; the scorer keeps the shortest clean one.
    for index, offset in enumerate(offsets, start=1):
        lanes = {
            "above": clamp(top - offset),
            "below": clamp(bottom + offset),
        }
        for side, lane in lanes.items():
            _add_lane(candidates, f"{prefix}_{side}_{index}", [start, [start[0], lane], [end[0], lane], end])
        columns = {"left": clamp(left - offset), "right": clamp(right + offset)}
        for side, column in columns.items():
            _add_lane(candidates, f"{prefix}_{side}_{index}", [start, [column, start[1]], [column, end[1]], end])
    return candidates


def _route_candidates(source: dict[str, Any], target: dict[str, Any], relation_type: str, policy: ConnectorRoutePolicy = DEFAULT_ROUTE_POLICY) -> list[dict[str, Any]]:
    start, end = _anchors(source, target, policy)
    middle_x, middle_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    candidates = [
        {"id": "direct", "path": [start, end]},
        {"id": "orthogonal_x", "path": [start, [middle_x, start[1]], [middle_x, end[1]], end]},
        {"id": "orthogonal_y", "path": [start, [start[0], middle_y], [end[0], middle_y], end]},
    ]
    if relation_type == "prediction" and abs(end[0] - start[0]) >= 0.035 and abs(end[1] - start[1]) >= 0.035:
        candidates.append({"id": "soft_curve", "path": _path(source, target, relation_type, policy)})
    for lane in (0.025, 0.975):
        candidates.append({"id": "side_lane", "path": [start, [lane, start[1]], [lane, end[1]], end]})
    if relation_type == "feedback":
        candidates.extend(_local_loop_candidates(source, target, start, end))
    else:
        candidates.extend(_local_loop_candidates(source, target, start, end, offsets=policy.detour_offsets, prefix="detour"))
    for lane in (0.04, 0.96):
        candidates.append({"id": "outer_lane", "path": [start, [start[0], lane], [end[0], lane], end]})
    return candidates


def _select_route(source: dict[str, Any], target: dict[str, Any], relation_type: str, obstacles: list[dict[str, Any]], existing_path: list[list[float]], existing_paths: list[list[list[float]]] | None = None, policy: ConnectorRoutePolicy = DEFAULT_ROUTE_POLICY) -> tuple[list[list[float]], dict[str, Any]]:
    existing_paths = existing_paths or []
    candidates = []
    if len(existing_path) >= 2:
        candidates.append({"id": "planned", "path": existing_path})
    candidates.extend(_route_candidates(source, target, relation_type, policy))
    scored = []
    for candidate in candidates:
        path = candidate["path"]
        hits = _path_hits(path, obstacles)
        crossings = _path_crossings(path, existing_paths)
        length = _path_length(path)
        turns = max(0, len(path) - 2)
        diagonal = any(abs(float(b[0]) - float(a[0])) > 0.012 and abs(float(b[1]) - float(a[1])) > 0.012 for a, b in zip(path[:-1], path[1:]))
        slant = _slant_extent(path)
        slant_charge = slant * policy.slant_penalty
        score = hits * policy.obstacle_rank + crossings * policy.crossing_rank + length + turns * policy.bend_penalty + slant_charge
        scored.append({"candidate_id": candidate["id"], "path": path, "obstacle_hits": hits, "crossings": crossings, "score": round(score, 4), "length": round(length, 4), "turns": turns, "axis_aligned": not diagonal, "slant_extent": round(slant, 4), "slant_penalty": round(slant_charge, 4)})
    selected = min(scored, key=lambda item: item["score"])
    if relation_type == "feedback" and not _faces_are_adjacent(source, target):
        loops = [item for item in scored if (item["candidate_id"].startswith("local_loop") or item["candidate_id"] == "outer_lane") and item["obstacle_hits"] == 0]
        if loops:
            selected = min(loops, key=lambda item: item["score"])
    planned = next((item for item in scored if item["candidate_id"] == "planned"), None)
    if planned and planned["obstacle_hits"] == 0 and relation_type != "feedback":
        selected = planned
    return selected["path"], {"selected_candidate": selected["candidate_id"], "candidates": scored, "obstacle_hits": selected["obstacle_hits"]}


def apply_connector_route_selections(program: dict[str, Any], selections: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply VLM choices only from prevalidated deterministic route candidates."""
    updated = deepcopy(program)
    selected_by_id = {str(item.get("arrow_id")): str(item.get("candidate_id")) for item in selections if isinstance(item, dict)}
    applied = []
    person_kept = []
    for arrow in updated.get("arrows", []) or []:
        candidate_id = selected_by_id.get(str(arrow.get("id") or ""))
        # The candidate list below was scored before the person's sentence arrived, so picking
        # again from it can only put the detour they had just objected to back on the figure. A
        # person's sentence settles the drawing after everything else has had its say.
        if candidate_id and str(arrow.get("route_generation_status") or "") == "person_rule_direct":
            person_kept.append(str(arrow.get("id") or ""))
            continue
        candidates = (arrow.get("route_selection") or {}).get("candidates") or []
        candidate = next((item for item in candidates if str(item.get("candidate_id")) == candidate_id and int(item.get("obstacle_hits") or 0) == 0), None)
        if not candidate or not isinstance(candidate.get("path"), list):
            continue
        arrow["path_percent"] = deepcopy(candidate["path"])
        arrow["route_selection"] = {**(arrow.get("route_selection") or {}), "selected_candidate": candidate_id, "selected_by": "vlm_visual_preference"}
        arrow["route_generation_status"] = "vlm_selected_deterministic_candidate"
        applied.append(str(arrow.get("id")))
    return updated, {"status": "applied" if applied else "unchanged", "applied_arrow_ids": applied, "person_rule_kept": person_kept, "preserved_fields": ["source", "target", "relation_type", "label", "evidence_ids"]}


def apply_memory_routing_constraints(program: dict[str, Any], constraints: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay a stored route decision by id, then by relation, then by copied path."""

    source = constraints if isinstance(constraints, dict) else {}
    selections = source.get("selections") if isinstance(source.get("selections"), list) else []
    updated, by_candidate = apply_connector_route_selections(program, selections)
    applied = list(by_candidate.get("applied_arrow_ids") or [])
    arrows = {str(item.get("id") or ""): item for item in updated.get("arrows", []) or [] if isinstance(item, dict)}
    for item in selections:
        if not isinstance(item, dict):
            continue
        path = item.get("path_percent") if isinstance(item.get("path_percent"), list) else item.get("path")
        if not isinstance(path, list) or len(path) < 2:
            continue
        arrow = arrows.get(str(item.get("arrow_id") or item.get("connector_id") or ""))
        if arrow is None:
            relation = str(item.get("relation_type") or "")
            source_id, target_id = str(item.get("source") or ""), str(item.get("target") or "")
            arrow = next(
                (
                    candidate
                    for candidate in updated.get("arrows", []) or []
                    if isinstance(candidate, dict)
                    and (
                        (relation and str(candidate.get("relation_type") or "") == relation)
                        or (source_id and target_id and str(candidate.get("source") or "") == source_id and str(candidate.get("target") or "") == target_id)
                    )
                ),
                None,
            )
        if arrow is None:
            continue
        arrow_id = str(arrow.get("id") or "")
        if arrow_id in applied:
            continue
        arrow["path_percent"] = deepcopy(path)
        applied.append(arrow_id)
    return updated, {"status": "applied" if applied else "unchanged", "applied_arrow_ids": applied, "source": "memory"}


def _label_box(path: list[list[float]], text: str, obstacles: list[dict[str, Any]], font_size_pt: float = 9.0, avoid_paths: list[list[list[float]]] | None = None) -> tuple[dict[str, float], dict[str, Any]]:
    """Place a connector label beside its own line and off every other line."""

    avoid_paths = [item for item in (avoid_paths or []) if len(item) >= 2]
    segments = [(a, b, hypot(b[0] - a[0], b[1] - a[1])) for a, b in zip(path[:-1], path[1:])]
    index, (a, b, _length) = max(enumerate(segments), key=lambda value: value[1][2])
    width = min(0.30, max(0.07, len(text) * 0.0062 * max(0.85, float(font_size_pt) / 9.0)))
    height = 0.042 if len(text) <= 24 else 0.058
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = max(hypot(dx, dy), 1e-6)
    nx, ny = -dy / length, dx / length
    for offset in (-0.035, 0.035, -0.065, 0.065, 0.0):
        cx, cy = (a[0] + b[0]) / 2 + nx * offset, (a[1] + b[1]) / 2 + ny * offset
        box = {"x": round(max(0.005, min(0.995 - width, cx - width / 2)), 4), "y": round(max(0.005, min(0.995 - height, cy - height / 2)), 4), "w": round(width, 4), "h": round(height, 4)}
        if not any(_intersects(box, obstacle) for obstacle in obstacles) and not any(_path_hits(other, [box]) for other in avoid_paths):
            return box, {"placement": "near_segment", "segment_index": index, "along": 0.5, "normal_offset": offset}
    return box, {"placement": "near_segment", "segment_index": index, "along": 0.5, "normal_offset": offset, "collision_unresolved": True}


def resolve_direct_connectors(program: dict[str, Any], *, policy: ConnectorRoutePolicy | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make final node geometry authoritative for arrows and relation labels."""

    updated = deepcopy(program)
    route_policy = _route_policy_for(updated, policy)
    nodes = {str(item.get("id")): item for item in updated.get("semantic_nodes", []) if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)}
    labels = [item for item in updated.get("labels", []) if isinstance(item, dict)]
    label_by_target = {str(item.get("target_id")): item for item in labels if item.get("target_id")}
    text_program = updated.get("text_program") if isinstance(updated.get("text_program"), dict) else {"items": []}
    existing_text = [item for item in text_program.get("items", []) if isinstance(item, dict) and str(item.get("text_role") or item.get("role") or "") != "connector_label"]
    slots = [item for item in updated.get("slots", []) if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)]
    obstacles = [item["bbox_percent"] for item in nodes.values()]
    panel_title_obstacles = [
        item.get("bbox_percent") for item in text_program.get("items", [])
        if isinstance(item, dict)
        and str(item.get("role") or item.get("text_role") or "").casefold() == "panel_title"
        and isinstance(item.get("bbox_percent"), dict)
    ]
    # The paper heading is editable text too: a route that grazes it draws a
    # line straight through the title, so it is an obstacle like any other.
    title_block = updated.get("title_block") if isinstance(updated.get("title_block"), dict) else {}
    title_block_obstacle = title_block.get("bbox_percent") if isinstance(title_block.get("bbox_percent"), dict) else None
    records, connector_text, routed_paths = [], [], []
    dense_graph = len(updated.get("arrows", [])) > 12
    for arrow in updated.get("arrows", []):
        if not isinstance(arrow, dict):
            continue
        arrow_id = str(arrow.get("id") or "")
        source = nodes.get(str(arrow.get("source") or arrow.get("source_id")))
        target = nodes.get(str(arrow.get("target") or arrow.get("target_id")))
        relation_type = str(arrow.get("relation_type") or arrow.get("type") or "data_flow").casefold()
        relation_type = relation_type if relation_type in STYLE_BY_RELATION else "data_flow"
        existing_path = arrow.get("path_percent") if isinstance(arrow.get("path_percent"), list) else []
        route_selection = {"selected_candidate": "unresolved", "candidates": [], "obstacle_hits": 0}
        if source and target:
            excluded = {str(arrow.get("source") or arrow.get("source_id")), str(arrow.get("target") or arrow.get("target_id"))}
            route_obstacles = [node["bbox_percent"] for node_id, node in nodes.items() if node_id not in excluded]
            route_obstacles.extend(
                node.get("label_bbox_percent") for node_id, node in nodes.items()
                if node_id not in excluded and isinstance(node.get("label_bbox_percent"), dict)
            )
            route_obstacles.extend(panel_title_obstacles)
            if title_block_obstacle:
                route_obstacles.append(title_block_obstacle)
            # Labels already bound to earlier connectors are placed text; a
            # later route must not be drawn across them.
            route_obstacles.extend(item["bbox_percent"] for item in connector_text)
            route_obstacles.extend(item["bbox_percent"] for item in slots if str(item.get("parent_id") or "") not in excluded)
            selected_path, route_selection = _select_route(source, target, relation_type, route_obstacles, existing_path, routed_paths, route_policy)
            arrow["path_percent"] = selected_path
            if relation_type == "prediction" and route_selection["selected_candidate"] in {"direct", "soft_curve"} and abs(selected_path[-1][0] - selected_path[0][0]) >= 0.035 and abs(selected_path[-1][1] - selected_path[0][1]) >= 0.035:
                arrow["path_percent"] = _path(source, target, relation_type, route_policy)
                arrow["path_kind"] = "bezier_polyline"
                arrow["curve_sample_count"] = len(arrow["path_percent"])
            offset = arrow.get("route_lane_offset_percent") if isinstance(arrow.get("route_lane_offset_percent"), dict) else {}
            if offset:
                arrow["path_percent"] = [[round(float(point[0]) + float(offset.get("x") or 0.0), 4), round(float(point[1]) + float(offset.get("y") or 0.0), 4)] for point in arrow["path_percent"]]
            routed_paths.append(arrow["path_percent"])
        arrow["relation_type"] = relation_type
        arrow["route_generation_status"] = "preserved_planned_path" if route_selection["selected_candidate"] == "planned" else "deterministic_reroute"
        arrow["routing_algorithm"] = "connector-candidate-router-v1"
        arrow["route_selection"] = route_selection
        arrow["arrowhead"] = "triangle"
        label_source = label_by_target.get(arrow_id, {})
        text = " ".join(str(arrow.get("label") or label_source.get("text") or "").split())
        # Preserve scientific relation labels. Dense graphs use shorter labels
        # only when an explicit low-priority hint is present; never erase text
        # solely because the graph is dense.
        if dense_graph and len(text) > 42 and str(arrow.get("label_priority") or "normal") == "low":
            text = " ".join(text.split()[:8]) + "…"
        placement = None
        if text and len(arrow.get("path_percent") or []) >= 2:
            label_font_size = float(label_source.get("font_size_pt") or arrow.get("label_font_size_pt") or 9.0)
            box, placement = _label_box(arrow["path_percent"], text, obstacles + [item["bbox_percent"] for item in connector_text], label_font_size, routed_paths[:-1])
            connector_text.append({"id": f"connector_text_{arrow_id}", "text": text, "text_role": "connector_label", "role": "connector_label", "target_id": arrow_id, "connector_id": arrow_id, "bbox_percent": box, "font_size_role": "connector_label", "font_size_pt": label_font_size, "min_readable_size_pt": 8.0, "max_lines": 2, "color_hex": arrow.get("stroke_color") or "#496A7C", "editable": True, "visible": not bool(placement.get("collision_unresolved")), "layer_ownership": "editable_text_layer", "editable_in": "pptx", "path_binding": placement, "evidence_ids": arrow.get("label_evidence_ids") or arrow.get("evidence_ids") or []})
        records.append({"connector_id": arrow_id, "relation_type": relation_type, "point_count": len(arrow.get("path_percent") or []), "arrowhead": "triangle", "label": text, "label_bound_to_path": bool(placement), "path_binding": placement, "route_selection": route_selection, "path_kind": arrow.get("path_kind") or "polyline"})
    # Only a label that actually reached the editable text program may be
    # dropped here; a connector whose label could not be placed keeps its
    # planned entry instead of losing the relation text silently.
    placed_label_ids = {str(item.get("connector_id") or "") for item in connector_text}
    updated["labels"] = [item for item in labels if str(item.get("target_id") or "") not in placed_label_ids]
    updated["text_program"] = {**text_program, "items": existing_text + connector_text}
    rerouted = [record["connector_id"] for record in records if (record.get("route_selection") or {}).get("selected_candidate") != "planned"]
    preserved = [record["connector_id"] for record in records if (record.get("route_selection") or {}).get("selected_candidate") == "planned"]
    unresolved = [record["connector_id"] for record in records if record["point_count"] < 2 or int((record.get("route_selection") or {}).get("obstacle_hits") or 0) > 0]
    report = {"summary": "Final authoritative connector geometry report.", "status": "pass" if all(record["point_count"] >= 2 and record["arrowhead"] == "triangle" for record in records) else "blocked", "connector_count": len(records), "arrowhead_count": sum(record["arrowhead"] == "triangle" for record in records), "labeled_connector_count": sum(bool(record["label"]) for record in records), "path_bound_label_count": sum(record["label_bound_to_path"] for record in records), "unresolved_label_collision_count": sum(bool((record.get("path_binding") or {}).get("collision_unresolved")) for record in records), "rerouted_connector_ids": rerouted, "preserved_connector_ids": preserved, "unresolved_connector_ids": unresolved, "route_policy": asdict(route_policy), "connectors": records}
    return updated, report
