"""Deterministic editable connector routing for the direct pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils import write_json

# PowerPoint (and LibreOffice Impress) scale the triangle stub with the stroke
# width.  The factors below were measured from a rendered probe: a 2pt and a
# 3pt connector draw the sm/med/lg head at 1.85x / 2.8x / 4.9x the stroke, and
# thin strokes are clamped to a minimum stroke for the head maths.
POWERPOINT_ARROWHEAD_FACTORS = {"sm": 1.85, "med": 2.8, "lg": 4.9}
POWERPOINT_MIN_HEAD_SCALING_STROKE_PT = 1.9
PREVIEW_POINTS_PER_CANVAS_HEIGHT = 277.0

ROUTER_VERSION = "reference-constrained-orthogonal-v1"
AESTHETIC_ROUTER_VERSION = "reference-tunnel-aesthetic-v2"
#: Names the rule set a route was drawn under, so a figure records how it was drawn and not
#: only which edges it has. A row learned on one paper carries the rule, not the paper's edges.
ROUTER_CONSTRAINT_VERSION = "router-constraint-v1"
#: The mechanism-level rules a router can honour. They are about the drawing, not the figure.
ROUTER_CONSTRAINT_KEYS = (
    "shared_vertical_channel",
    "forbid_crossing",
    "min_bend",
    "max_segments",
    # Two rules a person states when they look at a figure: two boxes that sit close together are
    # joined by one direct connector, and a connector between two boxes of one panel does not leave
    # that panel. They were not expressible before, so a person asking for either had nowhere to
    # put the request and the router went on drawing the detour they had just objected to.
    "direct_when_close",
    "stay_inside_parent",
)
#: How close two boxes have to be, in canvas percent, for ``direct_when_close`` to join them
#: directly. It is the rule's own number, so a person who means "only when they touch" says so.
#:
#: The default has to cover the boxes a person reads as neighbours. The first sentence a person
#: wrote named two boxes 0.107 apart -- stacked neighbours inside one group -- and at 0.08 the rule
#: matched no connector at all, so the detour the person had objected to survived a whole round
#: while that round's own account said the sentence had reached the stage that draws. A default
#: that quietly draws nothing is the failure the rule exists to fix, so it sits above the case
#: that named it.
DIRECT_WHEN_CLOSE_GAP = 0.12
#: How far a path may sit outside its panel before ``stay_inside_parent`` counts it as leaving.
PARENT_BOX_PAD = 0.002
DEFAULT_REFERENCE_TUNNEL_PERCENT = 0.024


def _bbox_center(bbox: dict[str, Any]) -> list[float]:
    return [
        round(float(bbox["x"]) + float(bbox["w"]) / 2, 4),
        round(float(bbox["y"]) + float(bbox["h"]) / 2, 4),
    ]


def _anchor_point(obj: dict | None, anchor: str) -> list[float] | None:
    if not obj or not isinstance(obj.get("bbox_percent"), dict):
        return None
    bbox = obj["bbox_percent"]
    x = float(bbox["x"])
    y = float(bbox["y"])
    w = float(bbox["w"])
    h = float(bbox["h"])
    points = {
        "left_mid": [x, y + h / 2],
        "right_mid": [x + w, y + h / 2],
        "top_mid": [x + w / 2, y],
        "bottom_mid": [x + w / 2, y + h],
        "center": [x + w / 2, y + h / 2],
    }
    point = points.get(anchor) or points["center"]
    return [round(point[0], 4), round(point[1], 4)]


def _edge_path(source: dict | None, target: dict | None) -> list[list[float]]:
    if not source or not target:
        return []
    sbox = source.get("bbox_percent") if isinstance(source.get("bbox_percent"), dict) else None
    tbox = target.get("bbox_percent") if isinstance(target.get("bbox_percent"), dict) else None
    if not sbox or not tbox:
        return []
    sx, sy = _bbox_center(sbox)
    tx, ty = _bbox_center(tbox)
    dx = tx - sx
    dy = ty - sy
    if abs(dx) >= abs(dy):
        s_anchor = "right_mid" if dx >= 0 else "left_mid"
        t_anchor = "left_mid" if dx >= 0 else "right_mid"
    else:
        s_anchor = "bottom_mid" if dy >= 0 else "top_mid"
        t_anchor = "top_mid" if dy >= 0 else "bottom_mid"
    s = _anchor_point(source, s_anchor)
    t = _anchor_point(target, t_anchor)
    return [s, t] if s and t else []


def _clean_path(points: list[list[float] | None]) -> list[list[float]]:
    cleaned: list[list[float]] = []
    for point in points:
        if not point or len(point) < 2:
            continue
        x = min(1.0, max(0.0, float(point[0])))
        y = min(1.0, max(0.0, float(point[1])))
        rounded = [round(x, 4), round(y, 4)]
        if not cleaned or abs(cleaned[-1][0] - rounded[0]) > 0.001 or abs(cleaned[-1][1] - rounded[1]) > 0.001:
            cleaned.append(rounded)
    if len(cleaned) <= 2:
        return cleaned
    simplified = [cleaned[0]]
    for index in range(1, len(cleaned) - 1):
        prev = simplified[-1]
        cur = cleaned[index]
        nxt = cleaned[index + 1]
        same_x = abs(prev[0] - cur[0]) < 0.001 and abs(cur[0] - nxt[0]) < 0.001
        same_y = abs(prev[1] - cur[1]) < 0.001 and abs(cur[1] - nxt[1]) < 0.001
        if not (same_x or same_y):
            simplified.append(cur)
    simplified.append(cleaned[-1])
    return simplified


def _segments(points: list[list[float]]) -> list[tuple[list[float], list[float]]]:
    return [(points[i], points[i + 1]) for i in range(max(0, len(points) - 1))]


def _path_length(points: list[list[float]]) -> float:
    total = 0.0
    for a, b in _segments(points):
        total += ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5
    return round(total, 4)


def _path_normal(points: list[list[float]]) -> list[float]:
    if len(points) < 2:
        return [0.0, 0.0]
    dx = float(points[-1][0]) - float(points[0][0])
    dy = float(points[-1][1]) - float(points[0][1])
    length = (dx * dx + dy * dy) ** 0.5
    if length < 0.0001:
        return [0.0, 0.0]
    return [-dy / length, dx / length]


def _point_distance(a: list[float], b: list[float]) -> float:
    return ((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5


def _point_segment_distance(point: list[float], a: list[float], b: list[float]) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom < 0.0000001:
        return _point_distance(point, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    proj = [ax + t * dx, ay + t * dy]
    return _point_distance(point, proj)


def _max_distance_to_polyline(points: list[list[float]], reference: list[list[float]]) -> float:
    segments = _segments(reference)
    if not segments:
        return 0.0
    distances = []
    for point in points:
        distances.append(min(_point_segment_distance(point, a, b) for a, b in segments))
    return round(max(distances, default=0.0), 4)


def _lane_offset(lane_index: int, lane_count: int, tunnel_percent: float) -> float:
    if lane_count <= 1:
        return 0.0
    centered = float(lane_index) - (float(lane_count) - 1.0) / 2.0
    step = min(0.006, max(0.002, tunnel_percent / max(lane_count, 1)))
    return max(-tunnel_percent * 0.82, min(tunnel_percent * 0.82, centered * step))


def _shift_point(point: list[float], normal: list[float], offset: float) -> list[float]:
    return [
        round(min(1.0, max(0.0, float(point[0]) + normal[0] * offset)), 4),
        round(min(1.0, max(0.0, float(point[1]) + normal[1] * offset)), 4),
    ]


def _aesthetic_tunnel_path(
    points: list[list[float]],
    lane_index: int,
    lane_count: int,
    role: str,
    tunnel_percent: float = DEFAULT_REFERENCE_TUNNEL_PERCENT,
) -> tuple[list[list[float]], dict[str, Any]]:
    original = _clean_path(points)
    if len(original) < 2:
        return original, {
            "routing_algorithm": AESTHETIC_ROUTER_VERSION,
            "route_generation_status": "aesthetic_no_path",
            "reference_tunnel_percent": tunnel_percent,
            "reference_path_delta_max": 0.0,
            "reference_tunnel_preserved": True,
        }
    normal = _path_normal(original)
    offset = _lane_offset(lane_index, lane_count, tunnel_percent)
    adjusted = [list(point) for point in original]
    if abs(offset) > 0.0001 and len(original) == 2:
        mid = [
            (float(original[0][0]) + float(original[1][0])) / 2.0,
            (float(original[0][1]) + float(original[1][1])) / 2.0,
        ]
        adjusted = [original[0], _shift_point(mid, normal, offset), original[-1]]
    elif role in {"main_flow", "module_flow"}:
        # Single-lane arrows keep the exact reference geometry; the PPT renderer
        # still softens them through curve connectors, halo, and line caps.
        adjusted = original
    adjusted = _clean_path(adjusted)
    delta = _max_distance_to_polyline(adjusted, original)
    return adjusted, {
        "routing_algorithm": AESTHETIC_ROUTER_VERSION,
        "route_generation_status": "aesthetic_tunnel_adjusted" if adjusted != original else "aesthetic_style_only",
        "reference_tunnel_percent": tunnel_percent,
        "reference_path_delta_max": delta,
        "reference_tunnel_preserved": delta <= tunnel_percent,
        "lane_offset_percent": round(offset, 4),
    }


def _endpoint_lane_path(points: list[list[float]], lane_index: int, lane_count: int, role: str) -> list[list[float]]:
    """Separate fan-in/fan-out paths immediately after/before shared endpoints."""
    path = _clean_path(points)
    if len(path) < 3 or lane_count <= 1 or role not in {"branch", "convergence"}:
        return path
    centered = float(lane_index) - (float(lane_count) - 1.0) / 2.0
    offset = max(-0.012, min(0.012, centered * 0.004))
    adjusted = [list(point) for point in path]
    if role == "branch":
        adjusted[1][1] = max(0.01, min(0.99, float(adjusted[1][1]) + offset))
    else:
        adjusted[-2][1] = max(0.01, min(0.99, float(adjusted[-2][1]) + offset))
    return _clean_path(adjusted)


def _orientation(a: list[float], b: list[float], c: list[float]) -> float:
    return (float(b[1]) - float(a[1])) * (float(c[0]) - float(b[0])) - (float(b[0]) - float(a[0])) * (float(c[1]) - float(b[1]))


def _share_endpoint(a1: list[float], a2: list[float], b1: list[float], b2: list[float]) -> bool:
    pairs = ((a1, b1), (a1, b2), (a2, b1), (a2, b2))
    return any(abs(p[0] - q[0]) < 0.002 and abs(p[1] - q[1]) < 0.002 for p, q in pairs)


def _segments_cross(a1: list[float], a2: list[float], b1: list[float], b2: list[float]) -> bool:
    if _share_endpoint(a1, a2, b1, b2):
        return False
    # Fast reject by segment bounding boxes.
    if max(min(a1[0], a2[0]), min(b1[0], b2[0])) > min(max(a1[0], a2[0]), max(b1[0], b2[0])):
        return False
    if max(min(a1[1], a2[1]), min(b1[1], b2[1])) > min(max(a1[1], a2[1]), max(b1[1], b2[1])):
        return False
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)
    return o1 * o2 < 0 and o3 * o4 < 0


def _point_inside_bbox(point: list[float], bbox: dict[str, Any], pad: float = 0.0) -> bool:
    x = float(bbox["x"]) - pad
    y = float(bbox["y"]) - pad
    w = float(bbox["w"]) + pad * 2
    h = float(bbox["h"]) + pad * 2
    return x <= float(point[0]) <= x + w and y <= float(point[1]) <= y + h


def _segment_bbox_overlap(a: list[float], b: list[float], bbox: dict[str, Any], pad: float = 0.004) -> bool:
    if _point_inside_bbox(a, bbox, pad=pad) or _point_inside_bbox(b, bbox, pad=pad):
        return True
    x0 = float(bbox["x"]) - pad
    y0 = float(bbox["y"]) - pad
    x1 = float(bbox["x"]) + float(bbox["w"]) + pad
    y1 = float(bbox["y"]) + float(bbox["h"]) + pad
    edges = [
        ([x0, y0], [x1, y0]),
        ([x1, y0], [x1, y1]),
        ([x1, y1], [x0, y1]),
        ([x0, y1], [x0, y0]),
    ]
    return any(_segments_cross(a, b, e0, e1) for e0, e1 in edges)


def _candidate_lanes(source: dict, target: dict, obstacles: dict[str, dict], axis: str, limit: int = 18) -> list[float]:
    sbox = source["bbox_percent"]
    tbox = target["bbox_percent"]
    if axis == "x":
        base = [
            float(sbox["x"]),
            float(sbox["x"]) + float(sbox["w"]),
            float(sbox["x"]) + float(sbox["w"]) / 2,
            float(tbox["x"]),
            float(tbox["x"]) + float(tbox["w"]),
            float(tbox["x"]) + float(tbox["w"]) / 2,
        ]
        for obj in obstacles.values():
            bbox = obj.get("bbox_percent") if isinstance(obj.get("bbox_percent"), dict) else None
            if not bbox:
                continue
            base.extend([float(bbox["x"]) - 0.014, float(bbox["x"]) + float(bbox["w"]) + 0.014])
    else:
        base = [
            float(sbox["y"]),
            float(sbox["y"]) + float(sbox["h"]),
            float(sbox["y"]) + float(sbox["h"]) / 2,
            float(tbox["y"]),
            float(tbox["y"]) + float(tbox["h"]),
            float(tbox["y"]) + float(tbox["h"]) / 2,
        ]
        for obj in obstacles.values():
            bbox = obj.get("bbox_percent") if isinstance(obj.get("bbox_percent"), dict) else None
            if not bbox:
                continue
            base.extend([float(bbox["y"]) - 0.014, float(bbox["y"]) + float(bbox["h"]) + 0.014])
    midpoint = (sum(base[:6]) / 6) if len(base) >= 6 else 0.5
    lanes = sorted({round(min(0.985, max(0.015, value)), 4) for value in base}, key=lambda value: abs(value - midpoint))
    return sorted(lanes[:limit])


def _crossing_count(points: list[list[float]], existing_segments: list[tuple[list[float], list[float]]]) -> int:
    count = 0
    for a, b in _segments(points):
        for c, d in existing_segments:
            if _segments_cross(a, b, c, d):
                count += 1
    return count


def _obstacle_overlaps(points: list[list[float]], obstacles: dict[str, dict]) -> list[str]:
    overlaps = []
    for obj_id, obj in obstacles.items():
        bbox = obj.get("bbox_percent") if isinstance(obj.get("bbox_percent"), dict) else None
        if not bbox:
            continue
        if any(_segment_bbox_overlap(a, b, bbox) for a, b in _segments(points)):
            overlaps.append(obj_id)
    return overlaps


def _score_candidate(points: list[list[float]], obstacles: dict[str, dict], existing_segments: list[tuple[list[float], list[float]]]) -> tuple[float, dict[str, Any]]:
    overlaps = _obstacle_overlaps(points, obstacles)
    crossings = _crossing_count(points, existing_segments)
    bends = max(0, len(points) - 2)
    length = _path_length(points)
    near_edge_penalty = sum(1 for x, y in points if x < 0.012 or x > 0.988 or y < 0.012 or y > 0.988)
    score = len(overlaps) * 1000 + crossings * 80 + bends * 8 + length * 16 + near_edge_penalty * 10
    return score, {
        "candidate_score": round(score, 4),
        "obstacle_overlap_count": len(overlaps),
        "obstacle_overlap_ids": overlaps[:12],
        "crossing_count": crossings,
        "bend_count": bends,
        "path_length": length,
    }


def _best_fallback_path(
    source: dict | None,
    target: dict | None,
    obstacles: dict[str, dict],
    existing_segments: list[tuple[list[float], list[float]]],
) -> tuple[list[list[float]], dict[str, Any]]:
    if not source or not target:
        return [], {"routing_algorithm": ROUTER_VERSION, "route_generation_status": "missing_source_or_target"}
    direct = _edge_path(source, target)
    if len(direct) < 2:
        return [], {"routing_algorithm": ROUTER_VERSION, "route_generation_status": "missing_anchor_points"}

    sx, sy = direct[0]
    tx, ty = direct[-1]
    x_lanes = _candidate_lanes(source, target, obstacles, "x")
    y_lanes = _candidate_lanes(source, target, obstacles, "y")
    candidates = [_clean_path(direct)]
    candidates.extend(_clean_path([direct[0], [x, sy], [x, ty], direct[-1]]) for x in x_lanes)
    candidates.extend(_clean_path([direct[0], [sx, y], [tx, y], direct[-1]]) for y in y_lanes)
    # Two-lane orthogonal paths can move around large central obstacles without
    # changing the source-target semantics from the reference-derived program.
    for x in x_lanes[:12]:
        for y in y_lanes[:12]:
            candidates.append(_clean_path([direct[0], [x, sy], [x, y], [tx, y], direct[-1]]))
            candidates.append(_clean_path([direct[0], [sx, y], [x, y], [x, ty], direct[-1]]))

    best_path: list[list[float]] = direct
    best_meta: dict[str, Any] = {}
    best_score = float("inf")
    seen: set[tuple[tuple[float, float], ...]] = set()
    for candidate in candidates:
        if len(candidate) < 2:
            continue
        key = tuple((point[0], point[1]) for point in candidate)
        if key in seen:
            continue
        seen.add(key)
        score, meta = _score_candidate(candidate, obstacles, existing_segments)
        if score < best_score:
            best_score = score
            best_path = candidate
            best_meta = meta
    best_meta.update({
        "routing_algorithm": ROUTER_VERSION,
        "route_generation_status": "fallback_route_selected",
        "candidate_count": len(seen),
    })
    return best_path, best_meta


def _normalize_router_constraints(constraints: Any) -> dict[str, Any]:
    """Keep only the rule keys a router can honour, with the types it can read."""

    source = constraints if isinstance(constraints, dict) else {}
    rules: dict[str, Any] = {}
    for key in ("shared_vertical_channel", "forbid_crossing", "direct_when_close", "stay_inside_parent"):
        if isinstance(source.get(key), bool):
            rules[key] = source[key]
    gap = source.get("close_gap")
    if isinstance(gap, (int, float)) and not isinstance(gap, bool) and float(gap) > 0:
        rules["close_gap"] = round(float(gap), 4)
    for key in ("min_bend", "max_segments"):
        value = source.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) > 0:
            rules[key] = int(value) if key == "max_segments" else round(float(value), 4)
    return rules


def _vertical_run_indexes(path: list[list[float]]) -> list[int]:
    """Indexes whose segment runs vertically, i.e. holds one x for two points."""

    return [index for index in range(len(path) - 1) if abs(float(path[index][0]) - float(path[index + 1][0])) < 0.001]


def _shorten_path(path: list[list[float]], cap: int) -> list[list[float]]:
    """Drop the cheapest interior vertex until the path has at most ``cap`` segments."""

    while len(path) > max(2, cap + 1):
        best_index, best_cost = None, None
        for index in range(1, len(path) - 1):
            cost = _point_distance(path[index - 1], path[index + 1]) - _point_distance(path[index - 1], path[index]) - _point_distance(path[index], path[index + 1])
            if best_cost is None or cost < best_cost:
                best_index, best_cost = index, cost
        if best_index is None:
            break
        path = path[:best_index] + path[best_index + 1:]
    return path


def _collapse_short_bends(path: list[list[float]], floor: float) -> list[list[float]]:
    """Drop a corner whose two sides are both shorter than the floor the rule sets."""

    changed = True
    while changed and len(path) > 2:
        changed = False
        for index in range(1, len(path) - 1):
            if _point_distance(path[index - 1], path[index]) < floor and _point_distance(path[index], path[index + 1]) < floor:
                path = path[:index] + path[index + 1:]
                changed = True
                break
    return path


def _object_box(obj: dict | None) -> dict[str, float]:
    """The box of one object in canvas percent, or nothing when it carries none."""

    source = obj if isinstance(obj, dict) else {}
    for key in ("bbox_percent", "bbox"):
        box = source.get(key)
        if isinstance(box, dict) and all(name in box for name in ("x", "y", "w", "h")):
            return {name: float(box[name]) for name in ("x", "y", "w", "h")}
    return {}


def _box_gap(one: dict[str, float], other: dict[str, float]) -> float | None:
    """The empty space between two boxes, or nothing when one of them has no box."""

    if not one or not other:
        return None
    dx = max(float(one["x"]) - (float(other["x"]) + float(other["w"])), float(other["x"]) - (float(one["x"]) + float(one["w"])), 0.0)
    dy = max(float(one["y"]) - (float(other["y"]) + float(other["h"])), float(other["y"]) - (float(one["y"]) + float(one["h"])), 0.0)
    return round((dx * dx + dy * dy) ** 0.5, 4)


def _smallest_holder(objects: dict[str, dict], first: list[float], second: list[float], excluding: set[str]) -> dict[str, float]:
    """The smallest object holding both ends of a connector -- the panel that connector belongs to.

    A person says "this connector never leaves its panel". The arrow does not name its panel, so
    the panel is found by holding the arrow's two ends: the smallest box that contains both of them
    and is neither end itself.
    """

    best: dict[str, float] = {}
    best_area = 0.0
    for obj_id, obj in (objects or {}).items():
        if str(obj_id) in excluding:
            continue
        box = _object_box(obj)
        if not box or not (_point_inside_bbox(first, box) and _point_inside_bbox(second, box)):
            continue
        area = float(box["w"]) * float(box["h"])
        if not best or area < best_area:
            best, best_area = box, area
    return best


def _draw_as_one_connector(arrow: dict, path: list[list[float]], status: str, rule: str, applied: list[dict[str, Any]], **extra: Any) -> bool:
    """Draw one arrow as a single connector between the two points it already starts and ends at."""

    if len(path) < 3:
        return False
    first, last = path[0], path[-1]
    arrow["path_percent"] = [[round(float(first[0]), 4), round(float(first[1]), 4)], [round(float(last[0]), 4), round(float(last[1]), 4)]]
    arrow["route_generation_status"] = status
    applied.append({"arrow_id": str(arrow.get("id") or ""), "rule": rule, **extra})
    return True


def _enforce_router_constraints(
    arrows: list[dict],
    objects: dict[str, dict],
    rules: dict[str, Any],
    containers: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Draw the committed connectors by the rule instead of re-picking this figure's edges.

    Every rule is a deterministic transform on the routed paths, so it holds for whatever figure
    is being drawn. ``forbid_crossing`` re-routes a path that crosses an earlier one through the
    same obstacle-aware search the fallback already uses -- which is scored on crossings -- and
    keeps the alternative only where it really crosses fewer. ``max_segments`` and ``min_bend``
    simplify a path to the shape the rule asks for, and ``shared_vertical_channel`` snaps the
    vertical runs of connectors leaving one node onto one line. Nothing here names an edge: the
    rule travelled, not the figure that taught it.

    The last two rules are the ones a person states when they look at a figure: two boxes that sit
    close together are joined by one direct connector (``direct_when_close``), and a connector
    between two boxes of one panel does not leave that panel (``stay_inside_parent``). A person
    asked for both and neither could be written down, so the router went on drawing the detour that
    had just been objected to. They run last, so a person's sentence settles the drawing after the
    patches have said their part.
    """

    applied: list[dict[str, Any]] = []
    unmet: list[dict[str, Any]] = []
    committed: list[tuple[list[float], list[float]]] = []
    if rules.get("forbid_crossing"):
        for arrow in arrows:
            path = _clean_path(arrow.get("path_percent") or [])
            if len(path) < 2:
                continue
            arrow_id = str(arrow.get("id") or "")
            crosses = any(_segments_cross(a1, a2, b1, b2) for a1, a2 in _segments(path) for b1, b2 in committed)
            if crosses:
                source = str(arrow.get("source_id") or "")
                target = str(arrow.get("target_id") or "")
                obstacles = {obj_id: obj for obj_id, obj in objects.items() if obj_id not in {source, target}}
                candidate = _clean_path(_best_fallback_path(objects.get(source), objects.get(target), obstacles, list(committed))[0])
                still_crosses = any(_segments_cross(a1, a2, b1, b2) for a1, a2 in _segments(candidate) for b1, b2 in committed)
                if len(candidate) >= 2 and not still_crosses:
                    arrow["path_percent"] = [[round(float(point[0]), 4), round(float(point[1]), 4)] for point in candidate]
                    arrow["route_generation_status"] = "router_constraint_rerouted"
                    path = candidate
                    applied.append({"arrow_id": arrow_id, "rule": "forbid_crossing"})
                else:
                    unmet.append({"arrow_id": arrow_id, "rule": "forbid_crossing"})
            committed.extend(_segments(path))
    if rules.get("max_segments"):
        cap = int(rules["max_segments"])
        for arrow in arrows:
            path = _clean_path(arrow.get("path_percent") or [])
            if len(path) - 1 <= cap:
                continue
            shortened = _shorten_path(path, cap)
            arrow["path_percent"] = [[round(float(point[0]), 4), round(float(point[1]), 4)] for point in shortened]
            entry = {"arrow_id": str(arrow.get("id") or ""), "rule": "max_segments", "segments": len(shortened) - 1}
            (unmet if len(shortened) - 1 > cap else applied).append(entry)
    if rules.get("min_bend"):
        floor = float(rules["min_bend"])
        for arrow in arrows:
            path = _clean_path(arrow.get("path_percent") or [])
            collapsed = _collapse_short_bends(path, floor)
            if collapsed != path:
                arrow["path_percent"] = [[round(float(point[0]), 4), round(float(point[1]), 4)] for point in collapsed]
                applied.append({"arrow_id": str(arrow.get("id") or ""), "rule": "min_bend"})
    if rules.get("shared_vertical_channel"):
        groups: dict[str, list[dict]] = {}
        for arrow in arrows:
            source = str(arrow.get("source_id") or "")
            if source:
                groups.setdefault(source, []).append(arrow)
        for source, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            paths = {str(item.get("id") or ""): _clean_path(item.get("path_percent") or []) for item in members}
            lanes = [
                path[index][0]
                for path in paths.values()
                for index in _vertical_run_indexes(path)
                if index >= 1 and index + 1 <= len(path) - 2
            ]
            if not lanes:
                continue
            lanes.sort()
            channel = lanes[len(lanes) // 2]
            for arrow in members:
                arrow_id = str(arrow.get("id") or "")
                path = paths[arrow_id]
                moved = False
                for index in _vertical_run_indexes(path):
                    if index < 1 or index + 1 > len(path) - 2 or abs(float(path[index][0]) - channel) <= 0.001:
                        continue
                    path[index] = [round(channel, 4), path[index][1]]
                    path[index + 1] = [round(channel, 4), path[index + 1][1]]
                    moved = True
                if moved:
                    arrow["path_percent"] = [[round(float(point[0]), 4), round(float(point[1]), 4)] for point in path]
                    applied.append({"arrow_id": arrow_id, "rule": "shared_vertical_channel", "source": source})
    if rules.get("direct_when_close"):
        gap = float(rules.get("close_gap") or DIRECT_WHEN_CLOSE_GAP)
        for arrow in arrows:
            path = _clean_path(arrow.get("path_percent") or [])
            between = _box_gap(
                _object_box(objects.get(str(arrow.get("source_id") or ""))),
                _object_box(objects.get(str(arrow.get("target_id") or ""))),
            )
            if between is None or between > gap:
                continue
            _draw_as_one_connector(arrow, path, "person_rule_direct", "direct_when_close", applied, gap=between)
    if rules.get("stay_inside_parent"):
        # Panels are routing containers rather than obstacles, so they are not in ``objects``: a
        # connector's panel is looked for among the boxes the caller names as containers, and among
        # the visible objects as a fallback.
        boxes = {**(containers or {}), **objects}
        for arrow in arrows:
            path = _clean_path(arrow.get("path_percent") or [])
            if len(path) < 2:
                continue
            holder = _smallest_holder(
                boxes,
                path[0],
                path[-1],
                {str(arrow.get("source_id") or ""), str(arrow.get("target_id") or "")},
            )
            if not holder or all(_point_inside_bbox(point, holder, pad=PARENT_BOX_PAD) for point in path):
                continue
            _draw_as_one_connector(arrow, path, "person_rule_direct", "stay_inside_parent", applied)
    return {"router_constraint_version": ROUTER_CONSTRAINT_VERSION, "router_constraints": dict(rules), "applied": applied, "unmet": unmet}


def _feedback_outer_path(source: dict | None, target: dict | None, obstacles: dict[str, dict]) -> tuple[list[list[float]], dict[str, Any]]:
    """Route feedback relations around the outer edge of the visible graph."""
    direct = _edge_path(source, target)
    if len(direct) < 2:
        return [], {"routing_algorithm": ROUTER_VERSION, "route_generation_status": "missing_source_or_target"}
    occupied = [obj["bbox_percent"] for obj in obstacles.values() if isinstance(obj.get("bbox_percent"), dict)]
    top = max(0.015, min((float(box["y"]) for box in occupied), default=0.08) - 0.025)
    bottom = min(0.985, max((float(box["y"]) + float(box["h"]) for box in occupied), default=0.92) + 0.025)
    lane = top if top >= 0.015 and (direct[0][1] + direct[-1][1]) / 2 >= 0.5 else bottom
    points = _clean_path([direct[0], [direct[0][0], lane], [direct[-1][0], lane], direct[-1]])
    score, meta = _score_candidate(points, obstacles, [])
    meta.update({"routing_algorithm": "outer-feedback-loop-v1", "route_generation_status": "feedback_outer_loop_selected", "outer_lane_percent": round(lane, 4), "candidate_score": score})
    return points, meta


def _infer_role(arrow: dict, out_counts: dict[str, int], in_counts: dict[str, int], panel_ids: set[str]) -> str:
    kind = str(arrow.get("control_kind") or arrow.get("type") or "").lower()
    source = str(arrow.get("source_id") or arrow.get("source") or "")
    target = str(arrow.get("target_id") or arrow.get("target") or "")
    if "loop" in kind or "dashed" in kind:
        return "feedback_loop"
    # Branch/convergence semantics should win over panel membership. Some RFS
    # programs use panel-like IDs for central modules, and treating those first
    # collapses every connector into the same main-flow style.
    if out_counts.get(source, 0) > 1:
        return "branch"
    if in_counts.get(target, 0) > 1:
        return "convergence"
    if source in panel_ids or target in panel_ids:
        return "main_flow"
    return "module_flow"


def _style_for_role(role: str, arrow: dict, mode: str = "reference") -> dict[str, Any]:
    kind = str(arrow.get("control_kind") or arrow.get("type") or "").lower()
    if str(mode).lower() == "aesthetic":
        if role == "main_flow":
            return {"route_style": "soft_curve", "stroke_width_pt": 2.35, "arrowhead_size": "med", "line_cap": "round", "halo_width_pt": 4.4, "halo_color": "#FFFFFF"}
        if role == "branch":
            return {"route_style": "metro_bundle", "stroke_width_pt": 1.45, "arrowhead_size": "sm", "line_cap": "round", "halo_width_pt": 3.2, "halo_color": "#FFFFFF"}
        if role == "convergence":
            return {"route_style": "metro_bundle", "stroke_width_pt": 1.45, "arrowhead_size": "sm", "line_cap": "round", "halo_width_pt": 3.2, "halo_color": "#FFFFFF"}
        if role == "feedback_loop":
            return {"route_style": "dashed_loop", "stroke_width_pt": 1.75, "arrowhead_size": "sm", "line_cap": "round", "line_pattern": "dash", "halo_width_pt": 3.0, "halo_color": "#FFFFFF"}
        if "elbow" in kind:
            return {"route_style": "rounded_elbow", "stroke_width_pt": 1.6, "arrowhead_size": "sm", "line_cap": "round", "halo_width_pt": 3.0, "halo_color": "#FFFFFF"}
        return {"route_style": "soft_curve", "stroke_width_pt": 1.55, "arrowhead_size": "sm", "line_cap": "round", "halo_width_pt": 3.0, "halo_color": "#FFFFFF"}
    if role == "main_flow":
        return {"route_style": "soft_straight", "stroke_width_pt": 2.2, "arrowhead_size": "med", "line_cap": "round"}
    if role == "branch":
        return {"route_style": "bundled_elbow", "stroke_width_pt": 1.55, "arrowhead_size": "sm", "line_cap": "round"}
    if role == "convergence":
        return {"route_style": "bundled_elbow", "stroke_width_pt": 1.55, "arrowhead_size": "sm", "line_cap": "round"}
    if role == "feedback_loop":
        return {"route_style": "dashed_spline_like", "stroke_width_pt": 1.8, "arrowhead_size": "sm", "line_cap": "round", "line_pattern": "dash"}
    if "elbow" in kind:
        return {"route_style": "rounded_elbow", "stroke_width_pt": 1.65, "arrowhead_size": "sm", "line_cap": "round"}
    return {"route_style": "soft_straight", "stroke_width_pt": 1.45, "arrowhead_size": "sm", "line_cap": "round"}


def _length_aware_style(style: dict[str, Any], points: list[list[float]], role: str) -> dict[str, Any]:
    """Apply continuous, bounded visual compensation to connector weight.

    Semantic role supplies the base weight.  The path length then provides a
    smooth correction rather than a few hard length buckets, so two nearby
    connectors do not abruptly jump between widths.
    """
    if len(points) < 2:
        return style
    length = _path_length(points)
    width = float(style.get("stroke_width_pt") or 1.5)
    # Invert a softened length measure: short gaps receive more optical
    # support, but the correction decays continuously as the connector grows.
    support = 0.18 / (length + 0.045)
    correction = max(0.0, min(2.7, 0.62 * support - 0.22))
    if role == "main_flow":
        correction *= 0.72
    elif role == "feedback_loop":
        correction *= 0.55
    width += correction
    style["stroke_width_pt"] = round(max(1.45, min(5.0, width)), 2)
    # Arrowhead sizing belongs to apply_arrowhead_limits below.  Growing the
    # head with the extra weight of a short link is what turned 1%-wide gaps
    # into triangles with no shaft.
    return style


@dataclass(frozen=True)
class ArrowheadLimitPolicy:
    """Caps that keep an arrowhead inside the link it terminates.

    Reference framework figures draw a short link as a short stroke plus a
    compact head: the head takes roughly 40% of the link, the shaft stays
    visible, and the stroke never becomes a bar.  Short links keep the extra
    optical weight the router gives them, and the head grows with it -- up to
    a little over half the link -- so a thick short link still reads as an
    arrow instead of a bar with a nub.  Past that the head would eat the
    shaft, and a link with no shaft left is not an arrow any more.
    """

    max_stroke_share_of_link: float = 0.30
    max_head_share_of_link: float = 0.60
    max_head_share_of_canvas_width: float = 0.013
    head_length_per_stroke: float = 3.2
    head_width_per_stroke: float = 4.0
    head_width_share_of_length: float = 1.45
    max_head_width_share_of_canvas_width: float = 0.013
    min_head_length_px: float = 4.0
    min_stroke_width_pt: float = 1.45


def preview_canvas_px(program: dict[str, Any] | None = None) -> tuple[float, float]:
    """Preview pixel size of the canvas the arrow is judged on."""
    canvas = (program or {}).get("canvas") if isinstance(program, dict) else None
    canvas = canvas if isinstance(canvas, dict) else {}
    return float(canvas.get("width_px") or 1600), float(canvas.get("height_px") or 900)


def stroke_pt_to_preview_px(stroke_pt: float, canvas_height_px: float) -> float:
    """The same point-to-pixel convention the preview renderer uses."""
    return max(0.0, float(stroke_pt)) * float(canvas_height_px) / PREVIEW_POINTS_PER_CANVAS_HEIGHT


def terminal_segment_px(points: Any, width_px: float, height_px: float) -> float:
    """Length, in preview pixels, of the segment the arrowhead is drawn on."""
    if not isinstance(points, list) or len(points) < 2:
        return 0.0
    try:
        (x1, y1), (x2, y2) = points[-2], points[-1]
        return (
            ((float(x2) - float(x1)) * width_px) ** 2
            + ((float(y2) - float(y1)) * height_px) ** 2
        ) ** 0.5
    except (TypeError, ValueError, IndexError):
        return 0.0


def arrowhead_limits(
    link_px: float,
    stroke_pt: float,
    canvas_width_px: float,
    canvas_height_px: float,
    policy: ArrowheadLimitPolicy | None = None,
) -> dict[str, Any]:
    """Final stroke and head geometry for one arrow's terminal segment."""
    p = policy or ArrowheadLimitPolicy()
    pt_per_px = canvas_height_px / PREVIEW_POINTS_PER_CANVAS_HEIGHT
    stroke_px = stroke_pt_to_preview_px(stroke_pt, canvas_height_px)
    capped_stroke_px = stroke_px
    if link_px > 0:
        capped_stroke_px = min(stroke_px, p.max_stroke_share_of_link * link_px)
    capped_stroke_px = max(capped_stroke_px, p.min_stroke_width_pt * pt_per_px)
    head_cap_px = p.head_length_per_stroke * capped_stroke_px
    if link_px > 0:
        head_cap_px = min(
            head_cap_px,
            p.max_head_share_of_link * link_px,
            p.max_head_share_of_canvas_width * canvas_width_px,
        )
        head_cap_px = max(head_cap_px, min(p.min_head_length_px, p.max_head_share_of_link * link_px))
    head_px = head_cap_px
    head_width_px = min(
        p.head_width_per_stroke * capped_stroke_px,
        head_px * p.head_width_share_of_length,
        p.max_head_width_share_of_canvas_width * canvas_width_px,
    )
    ratio = head_px / capped_stroke_px if capped_stroke_px > 0 else 0.0
    size = "sm"
    for candidate in ("sm", "med", "lg"):
        if abs(POWERPOINT_ARROWHEAD_FACTORS[candidate] - ratio) < abs(POWERPOINT_ARROWHEAD_FACTORS[size] - ratio):
            size = candidate
    scaling_stroke_px = max(capped_stroke_px, POWERPOINT_MIN_HEAD_SCALING_STROKE_PT * pt_per_px)
    return {
        "link_px": round(link_px, 2),
        "stroke_width_pt": round(capped_stroke_px / pt_per_px, 2),
        "stroke_width_px": round(capped_stroke_px, 2),
        "stroke_width_pt_before": round(float(stroke_pt), 2),
        "arrowhead_length_px": round(head_px, 2),
        "arrowhead_width_px": round(head_width_px, 2),
        "arrowhead_length_percent": round(head_px / canvas_width_px, 5),
        "arrowhead_width_percent": round(head_width_px / canvas_width_px, 5),
        "arrowhead_size": size,
        "powerpoint_head_length_px": round(POWERPOINT_ARROWHEAD_FACTORS[size] * scaling_stroke_px, 2),
        "stroke_capped": capped_stroke_px < stroke_px - 1e-9,
        "head_capped": head_px < p.head_length_per_stroke * capped_stroke_px - 1e-9,
    }


def apply_arrowhead_limits(
    arrows: list[dict[str, Any]],
    canvas: dict[str, Any] | None = None,
    policy: ArrowheadLimitPolicy | None = None,
) -> dict[str, Any]:
    """Cap every arrowhead so it can never swallow the link it terminates.

    Mutates the arrow records in place and returns the report.  Block arrows
    carry no triangular head and are left untouched.
    """
    width_px, height_px = preview_canvas_px({"canvas": canvas})
    rows: list[dict[str, Any]] = []
    capped_ids: list[str] = []
    for arrow in arrows:
        if not isinstance(arrow, dict):
            continue
        if str(arrow.get("render_style") or "").lower() == "filled_block_arrow":
            continue
        link_px = terminal_segment_px(arrow.get("path_percent"), width_px, height_px)
        if link_px <= 0:
            continue
        limits = arrowhead_limits(link_px, float(arrow.get("stroke_width_pt") or 1.5), width_px, height_px, policy)
        arrow["stroke_width_pt"] = limits["stroke_width_pt"]
        arrow["arrowhead_size"] = limits["arrowhead_size"]
        arrow["arrowhead_length_percent"] = limits["arrowhead_length_percent"]
        arrow["arrowhead_width_percent"] = limits["arrowhead_width_percent"]
        arrow["arrowhead_limits"] = {
            key: limits[key]
            for key in (
                "link_px",
                "stroke_width_px",
                "stroke_width_pt_before",
                "arrowhead_length_px",
                "arrowhead_width_px",
                "powerpoint_head_length_px",
                "stroke_capped",
                "head_capped",
            )
        }
        row = {"id": arrow.get("id"), **limits}
        rows.append(row)
        if limits["stroke_capped"] or limits["head_capped"]:
            capped_ids.append(str(arrow.get("id")))
    return {
        "summary": "Arrowhead proportion limits applied to editable PPT connectors.",
        "policy": {
            "max_stroke_share_of_link": (policy or ArrowheadLimitPolicy()).max_stroke_share_of_link,
            "max_head_share_of_link": (policy or ArrowheadLimitPolicy()).max_head_share_of_link,
            "max_head_share_of_canvas_width": (policy or ArrowheadLimitPolicy()).max_head_share_of_canvas_width,
            "head_length_per_stroke": (policy or ArrowheadLimitPolicy()).head_length_per_stroke,
            "max_head_width_share_of_canvas_width": (policy or ArrowheadLimitPolicy()).max_head_width_share_of_canvas_width,
            "min_head_length_px": (policy or ArrowheadLimitPolicy()).min_head_length_px,
            "min_stroke_width_pt": (policy or ArrowheadLimitPolicy()).min_stroke_width_pt,
            "powerpoint_arrowhead_factors": dict(POWERPOINT_ARROWHEAD_FACTORS),
        },
        "canvas_px": [width_px, height_px],
        "arrow_count": len(rows),
        "capped_count": len(capped_ids),
        "capped_arrow_ids": capped_ids,
        "arrows": rows,
    }


def limit_arrowhead_proportions(
    program: dict[str, Any],
    policy: ArrowheadLimitPolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Program-level wrapper used by the composition stage and offline re-runs."""
    arrows = [item for item in program.get("arrows", []) or [] if isinstance(item, dict)]
    report = apply_arrowhead_limits(arrows, program.get("canvas"), policy)
    return program, report


def _reference_locked(arrow: dict) -> bool:
    path = arrow.get("path_percent") if isinstance(arrow.get("path_percent"), list) else []
    if len(path) < 2:
        return False
    source = str(arrow.get("binding_source") or arrow.get("detected_by") or "").lower()
    if any(term in source for term in ("reference", "opencv", "vlm", "explicit", "candidate", "layout")):
        return True
    # In reference-primary mode, existing normalized paths are treated as a
    # reference contract unless explicitly marked as fallback.
    return str(arrow.get("route_policy", "")).lower() != "fallback_reroute_allowed"


def _route_metrics(arrows: list[dict], objects: dict[str, dict]) -> tuple[dict[str, dict], int]:
    metrics: dict[str, dict] = {}
    all_segments: list[tuple[str, list[float], list[float]]] = []
    for arrow in arrows:
        points = arrow.get("path_percent") if isinstance(arrow.get("path_percent"), list) else []
        sid = str(arrow.get("source_id") or arrow.get("source") or "")
        tid = str(arrow.get("target_id") or arrow.get("target") or "")
        obstacles = []
        for obj_id, obj in objects.items():
            if obj_id in {sid, tid} or not isinstance(obj.get("bbox_percent"), dict):
                continue
            if any(_segment_bbox_overlap(a, b, obj["bbox_percent"]) for a, b in _segments(points)):
                obstacles.append(obj_id)
        metrics[str(arrow.get("id"))] = {
            "path_length": _path_length(points),
            "bend_count": max(0, len(points) - 2),
            "obstacle_overlap_count": len(obstacles),
            "obstacle_overlap_ids": obstacles[:12],
            "point_count": len(points),
        }
        for a, b in _segments(points):
            all_segments.append((str(arrow.get("id")), a, b))

    crossing_count = 0
    crossings_by_arrow = {str(arrow.get("id")): 0 for arrow in arrows}
    for idx, (aid, a1, a2) in enumerate(all_segments):
        for bid, b1, b2 in all_segments[idx + 1:]:
            if aid == bid:
                continue
            if _segments_cross(a1, a2, b1, b2):
                crossing_count += 1
                crossings_by_arrow[aid] = crossings_by_arrow.get(aid, 0) + 1
                crossings_by_arrow[bid] = crossings_by_arrow.get(bid, 0) + 1
    for aid, count in crossings_by_arrow.items():
        if aid in metrics:
            metrics[aid]["crossing_count"] = count
    return metrics, crossing_count


def count_connector_crossings(program: dict[str, Any] | None) -> int:
    """Count the pairs of connector segments that cross, from the drawn paths alone.

    The routing search already scores candidates on this number; the stage trace needs the same
    number for the figure that was actually drawn, so it is exposed rather than recomputed by a
    second, drifting implementation.
    """

    source = program if isinstance(program, dict) else {}
    segments: list[tuple[str, list[float], list[float]]] = []
    for arrow in source.get("arrows") or []:
        if not isinstance(arrow, dict):
            continue
        points = [point for point in (arrow.get("path_percent") or []) if isinstance(point, (list, tuple)) and len(point) >= 2]
        segments.extend((str(arrow.get("id")), a, b) for a, b in _segments([list(point) for point in points]))
    count = 0
    for index, (first_id, a1, a2) in enumerate(segments):
        for second_id, b1, b2 in segments[index + 1 :]:
            if first_id == second_id:
                continue
            if _segments_cross(a1, a2, b1, b2):
                count += 1
    return count


def style_and_route_arrows(
    program: dict,
    out_dir: str | Path,
    mode: str = "reference",
    repair_routing: bool = False,
    constraints: dict[str, Any] | None = None,
) -> dict:
    """Add reference-preserving arrow aesthetics and QA artifacts.

    The reference image remains the hard constraint. Existing reference-derived
    routes are not freely rerouted; this stage mostly adds style, bundling
    metadata, and diagnostics. Only missing/fallback paths are synthesized.

    ``constraints`` carries mechanism-level drawing rules a Memory row learned --
    forbidding crossings, capping segment counts, keeping corners long, sharing a
    vertical channel between the connectors leaving one node. They are rules about
    how a connector is drawn, so the same row applies to a figure it never saw.
    """
    out = Path(out_dir)
    if str(mode).lower() == "off":
        write_json(out / "arrow_style_profile.json", {"summary": "Arrow styling skipped.", "mode": "off"})
        write_json(out / "selected_arrow_routes.json", {"summary": "Arrow routing skipped.", "mode": "off", "routes": []})
        write_json(out / "arrow_quality_report.json", {"summary": "Arrow quality skipped.", "mode": "off", "status": "skipped"})
        return program

    panels = program.get("panels", []) if isinstance(program.get("panels"), list) else []
    slots = program.get("slots", []) if isinstance(program.get("slots"), list) else []
    program_style = program.get("style") if isinstance(program.get("style"), dict) else {}
    color_tokens = program_style.get("color_tokens", []) if isinstance(program_style.get("color_tokens"), list) else []
    default_style_token_id = next(
        (
            str(item.get("token_id") or "")
            for item in color_tokens
            if isinstance(item, dict) and any(term in str(item.get("usage") or "").casefold() for term in ("arrow", "connector", "line"))
        ),
        "",
    )
    if not default_style_token_id:
        default_style_token_id = next((str(item.get("token_id") or "") for item in color_tokens if isinstance(item, dict) and item.get("token_id")), "")
    # Panel bodies are routing containers, not solid obstacles.  Panel titles
    # are represented by editable text objects below and are avoided.
    visible_objects = slots + [item for key in ("semantic_nodes", "labels", "cards") for item in program.get(key, []) or []]
    visible_objects += [item for item in (program.get("text_program", {}) or {}).get("items", []) if isinstance(item, dict) and item.get("visible", True)]
    objects = {}
    for item in visible_objects:
        if not isinstance(item, dict) or not item.get("id") or not isinstance(item.get("bbox_percent"), dict):
            continue
        box = item["bbox_percent"]
        if float(box.get("x", 1)) <= 0.002 and float(box.get("y", 1)) <= 0.002 and float(box.get("w", 0)) >= 0.98 and float(box.get("h", 0)) >= 0.98:
            continue
        objects[str(item["id"])] = item
    slot_objects = {str(item.get("id")): item for item in slots if isinstance(item, dict) and item.get("id")}
    panel_ids = {str(panel.get("id")) for panel in panels if isinstance(panel, dict)}
    arrows = [dict(item) for item in program.get("arrows", []) if isinstance(item, dict)]
    out_counts: dict[str, int] = {}
    in_counts: dict[str, int] = {}
    for arrow in arrows:
        source = str(arrow.get("source_id") or arrow.get("source") or "")
        target = str(arrow.get("target_id") or arrow.get("target") or "")
        out_counts[source] = out_counts.get(source, 0) + 1
        in_counts[target] = in_counts.get(target, 0) + 1

    reference_segments: list[tuple[list[float], list[float]]] = []
    for original in arrows:
        original_points = original.get("path_percent") if isinstance(original.get("path_percent"), list) else []
        if _reference_locked(original) and len(original_points) >= 2:
            reference_segments.extend(_segments(_clean_path(original_points)))

    grouped: dict[str, list[dict]] = {}
    enriched = []
    committed_fallback_segments: list[tuple[list[float], list[float]]] = []
    for index, arrow in enumerate(arrows, start=1):
        source = str(arrow.get("source_id") or arrow.get("source") or "")
        target = str(arrow.get("target_id") or arrow.get("target") or "")
        arrow["source_id"] = source
        arrow["target_id"] = target
        arrow["source"] = source
        arrow["target"] = target
        if not str(arrow.get("style_token_id") or "").strip():
            arrow["style_token_id"] = default_style_token_id
        points = arrow.get("path_percent") if isinstance(arrow.get("path_percent"), list) else []
        role = _infer_role(arrow, out_counts, in_counts, panel_ids)
        locked = _reference_locked(arrow)
        explicit_patch = str(arrow.get("binding_source") or "").lower() == "critic_explicit_patch"
        route_meta: dict[str, Any] = {"routing_algorithm": "preserve_reference_path", "route_generation_status": "reference_locked"}
        fallback_allowed = str(arrow.get("route_policy", "")).lower() == "fallback_reroute_allowed"
        obstructed = repair_routing and bool(arrow.get("auto_reroute_allowed")) and len(points) >= 2 and any(
            _segment_bbox_overlap(a, b, obj.get("bbox_percent"))
            for a, b in _segments(_clean_path(points))
            for obj_id, obj in objects.items()
            if obj_id not in {source, target} and isinstance(obj.get("bbox_percent"), dict)
        )
        if not explicit_patch and role == "feedback_loop" and (len(points) < 2 or fallback_allowed or obstructed) and mode in {"reference", "aesthetic"}:
            obstacles = {obj_id: obj for obj_id, obj in objects.items() if obj_id not in {source, target}}
            points, route_meta = _feedback_outer_path(objects.get(source), objects.get(target), obstacles)
            locked = False
        elif not explicit_patch and (len(points) < 2 or (fallback_allowed and not locked) or obstructed) and mode in {"reference", "aesthetic"}:
            obstacles = {obj_id: obj for obj_id, obj in objects.items() if obj_id not in {source, target}}
            points, route_meta = _best_fallback_path(
                objects.get(source),
                objects.get(target),
                obstacles,
                reference_segments + committed_fallback_segments,
            )
            locked = False
            if repair_routing and obstructed:
                route_meta["route_generation_status"] = "repair_route_selected"
        else:
            points = _clean_path(points)
        arrow["path_percent"] = [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in points if isinstance(p, list) and len(p) >= 2]
        style = _length_aware_style(_style_for_role(role, arrow, mode=mode), arrow["path_percent"], role)
        # This function is the sole final owner of visual arrow fields.  The
        # preceding geometry resolver may supply paths and labels, but it must
        # not freeze route style, weight, caps or arrowhead size.
        recompute_generated_style = True
        arrow.pop("_arrow_style_pending", None)
        for key, value in style.items():
            if recompute_generated_style or key not in arrow or not str(arrow.get(key, "")).strip():
                arrow[key] = value
        if recompute_generated_style or not str(arrow.get("semantic_role", "")).strip():
            arrow["semantic_role"] = role
        arrow.setdefault("aesthetic_policy", "reference_first_soft_editable_ppt_connector")
        arrow.setdefault("reference_locked", locked)
        arrow.setdefault("reference_path_preserved", locked)
        arrow.setdefault("route_policy", "preserve_reference_path" if locked else "synthesize_missing_path_only")
        arrow.setdefault("routing_algorithm", route_meta.get("routing_algorithm", ROUTER_VERSION))
        arrow.setdefault("route_generation_status", route_meta.get("route_generation_status", "fallback_route_selected" if not locked else "reference_locked"))
        if "candidate_count" in route_meta:
            arrow.setdefault("candidate_count", route_meta["candidate_count"])
        if "candidate_score" in route_meta:
            arrow.setdefault("candidate_score", route_meta["candidate_score"])
        arrow.setdefault("corner_radius_percent", 0.018 if arrow.get("route_style") in {"rounded_elbow", "bundled_elbow"} else 0.0)
        if role == "branch":
            bundle_id = f"from_{source}"
        elif role == "convergence":
            bundle_id = f"to_{target}"
        elif role == "feedback_loop":
            bundle_id = f"loop_{source}_{target}"
        else:
            bundle_id = f"flow_{index:02d}"
        arrow.setdefault("bundle_id", bundle_id)
        grouped.setdefault(bundle_id, []).append(arrow)
        enriched.append(arrow)
        if not locked and len(arrow["path_percent"]) >= 2:
            committed_fallback_segments.extend(_segments(arrow["path_percent"]))

    for bundle_id, items in grouped.items():
        items.sort(key=lambda item: (item.get("target_id", ""), item.get("id", "")))
        for lane_index, arrow in enumerate(items):
            arrow["lane_index"] = lane_index
            arrow["lane_count"] = len(items)
            arrow["path_percent"] = _endpoint_lane_path(arrow.get("path_percent") or [], lane_index, len(items), str(arrow.get("semantic_role") or ""))
            arrow["endpoint_lane_offset_percent"] = round(float(arrow.get("lane_index") or 0) - (float(len(items) - 1) / 2.0), 4) if len(items) > 1 else 0.0

    # Mechanism-level routing rules travel on the Memory row that learned them. Applying them
    # here, after the paths are committed, is what makes the row about the drawing rather than
    # about this figure's edges: whatever was just routed is redrawn by the rule.
    router_rules = _normalize_router_constraints(constraints)
    router_constraint_report: dict[str, Any] = {}
    if router_rules:
        router_constraint_report = _enforce_router_constraints(
            enriched,
            objects,
            router_rules,
            containers={str(panel.get("id")): panel for panel in panels if isinstance(panel, dict) and panel.get("id")},
        )
        for arrow in enriched:
            arrow["router_constraints"] = dict(router_rules)
            arrow["router_constraint_version"] = ROUTER_CONSTRAINT_VERSION

    if str(mode).lower() == "aesthetic":
        tunnel_percent = DEFAULT_REFERENCE_TUNNEL_PERCENT
        for arrow in enriched:
            original_path = _clean_path(arrow.get("path_percent") if isinstance(arrow.get("path_percent"), list) else [])
            arrow["reference_original_path_percent"] = original_path
            if arrow.get("reference_locked") and len(original_path) >= 2:
                offset_allowed = bool(arrow.get("aesthetic_offset_allowed")) or str(arrow.get("route_policy", "")).lower() == "aesthetic_tunnel_allowed"
                if offset_allowed:
                    adjusted_path, aesthetic_meta = _aesthetic_tunnel_path(
                        original_path,
                        int(arrow.get("lane_index") or 0),
                        int(arrow.get("lane_count") or 1),
                        str(arrow.get("semantic_role") or ""),
                        tunnel_percent=tunnel_percent,
                    )
                    arrow["path_percent"] = adjusted_path
                    arrow["routing_algorithm"] = aesthetic_meta["routing_algorithm"]
                    arrow["route_generation_status"] = aesthetic_meta["route_generation_status"]
                    arrow["reference_tunnel_percent"] = aesthetic_meta["reference_tunnel_percent"]
                    arrow["reference_path_delta_max"] = aesthetic_meta["reference_path_delta_max"]
                    arrow["reference_tunnel_preserved"] = aesthetic_meta["reference_tunnel_preserved"]
                    arrow["lane_offset_percent"] = aesthetic_meta["lane_offset_percent"]
                    arrow["reference_path_preserved"] = bool(aesthetic_meta["reference_tunnel_preserved"])
                else:
                    arrow["routing_algorithm"] = AESTHETIC_ROUTER_VERSION
                    arrow["route_generation_status"] = "aesthetic_style_only"
                    arrow["reference_tunnel_percent"] = tunnel_percent
                    arrow["reference_path_delta_max"] = 0.0
                    arrow["reference_tunnel_preserved"] = True
                    arrow["lane_offset_percent"] = 0.0
                    arrow["reference_path_preserved"] = True

    arrowhead_report = apply_arrowhead_limits(enriched, program.get("canvas"))
    metrics, total_crossings = _route_metrics(enriched, slot_objects)
    routes = []
    for arrow in enriched:
        aid = str(arrow.get("id"))
        m = metrics.get(aid, {})
        clutter_score = (
            int(m.get("crossing_count", 0)) * 12
            + int(m.get("bend_count", 0)) * 3
            + int(m.get("obstacle_overlap_count", 0)) * 18
        )
        arrow["aesthetic_score"] = max(0, 100 - clutter_score)
        routes.append({
            "id": aid,
            "source_id": arrow.get("source_id"),
            "target_id": arrow.get("target_id"),
            "semantic_role": arrow.get("semantic_role"),
            "route_style": arrow.get("route_style"),
            "bundle_id": arrow.get("bundle_id"),
            "lane_index": arrow.get("lane_index"),
            "lane_count": arrow.get("lane_count"),
            "reference_locked": arrow.get("reference_locked"),
            "reference_path_preserved": arrow.get("reference_path_preserved"),
            "path_percent": arrow.get("path_percent"),
            "style_token_id": arrow.get("style_token_id"),
            "stroke_width_pt": arrow.get("stroke_width_pt"),
            "arrowhead_size": arrow.get("arrowhead_size"),
            "arrowhead_length_percent": arrow.get("arrowhead_length_percent"),
            "arrowhead_width_percent": arrow.get("arrowhead_width_percent"),
            "line_cap": arrow.get("line_cap"),
            "line_pattern": arrow.get("line_pattern", "solid"),
            "routing_algorithm": arrow.get("routing_algorithm"),
            "route_generation_status": arrow.get("route_generation_status"),
            "reference_original_path_percent": arrow.get("reference_original_path_percent"),
            "reference_tunnel_percent": arrow.get("reference_tunnel_percent"),
            "reference_path_delta_max": arrow.get("reference_path_delta_max"),
            "reference_tunnel_preserved": arrow.get("reference_tunnel_preserved"),
            "lane_offset_percent": arrow.get("lane_offset_percent"),
            "halo_width_pt": arrow.get("halo_width_pt"),
            "halo_color": arrow.get("halo_color"),
            "candidate_count": arrow.get("candidate_count"),
            "candidate_score": arrow.get("candidate_score"),
            "metrics": m,
            "aesthetic_score": arrow["aesthetic_score"],
        })

    arrow_style_profile = {
        "summary": "Reference-first arrow styling profile for editable PPT connectors.",
        "mode": mode,
        "reference_priority": "reference_image_hard_constraint",
        "routing_principle": "preserve reference-derived source-target logic and path geometry; only synthesize missing fallback paths",
        "routing_algorithm": ROUTER_VERSION,
        "aesthetic_routing_algorithm": AESTHETIC_ROUTER_VERSION,
        "router_constraint_version": ROUTER_CONSTRAINT_VERSION if router_rules else "",
        "router_constraints": dict(router_rules),
        "router_constraint_applied": router_constraint_report.get("applied", []),
        "router_constraint_unmet": router_constraint_report.get("unmet", []),
        "reference_tunnel_percent": DEFAULT_REFERENCE_TUNNEL_PERCENT,
        "fallback_routing_policy": "only arrows with missing paths or route_policy=fallback_reroute_allowed may use obstacle-aware orthogonal routing",
        "aesthetic_mode_policy": "mode=aesthetic may adjust reference-locked paths only inside the reference tunnel and must keep reference_tunnel_preserved=true",
        "style_rules": {
            "main_flow": {"route_style": "soft_straight", "stroke_width_pt": 2.2, "arrowhead_size": "med"},
            "branch": {"route_style": "bundled_elbow", "stroke_width_pt": 1.55, "arrowhead_size": "sm"},
            "convergence": {"route_style": "bundled_elbow", "stroke_width_pt": 1.55, "arrowhead_size": "sm"},
            "feedback_loop": {"route_style": "dashed_spline_like", "stroke_width_pt": 1.8, "arrowhead_size": "sm"},
            "module_flow": {"route_style": "soft_straight", "stroke_width_pt": 1.45, "arrowhead_size": "sm"},
        },
        "aesthetic_style_rules": {
            "main_flow": {"route_style": "soft_curve", "stroke_width_pt": 2.35, "arrowhead_size": "med", "halo_width_pt": 4.4},
            "branch": {"route_style": "metro_bundle", "stroke_width_pt": 1.45, "arrowhead_size": "sm", "halo_width_pt": 3.2},
            "convergence": {"route_style": "metro_bundle", "stroke_width_pt": 1.45, "arrowhead_size": "sm", "halo_width_pt": 3.2},
            "feedback_loop": {"route_style": "dashed_loop", "stroke_width_pt": 1.75, "arrowhead_size": "sm", "halo_width_pt": 3.0},
            "module_flow": {"route_style": "soft_curve", "stroke_width_pt": 1.55, "arrowhead_size": "sm", "halo_width_pt": 3.0},
        },
        "ppt_editability": "all arrows render as PPT connector shapes, not raster assets",
        "rounded_line_policy": "use round line caps and editable connector segments; reference-locked paths are not geometrically rewritten",
    }
    selected_routes = {
        "summary": "Selected reference-preserving arrow routes and style assignments.",
        "mode": mode,
        "route_count": len(routes),
        "routes": routes,
    }
    tunnel_violations = [route["id"] for route in routes if route.get("reference_tunnel_preserved") is False]
    route_obstacle_overlaps = sum(int(route["metrics"].get("obstacle_overlap_count", 0)) for route in routes)
    quality_report = {
        "summary": "Arrow routing and styling quality report.",
        "mode": mode,
        "status": "pass" if all(route["aesthetic_score"] >= 35 for route in routes) and not tunnel_violations else "needs_review",
        "arrow_count": len(routes),
        # The canonical crossing: two connector segments crossing each other. The other smell a
        # reader sees is a connector running through a box it does not join, and it is named
        # ``route_obstacle_overlap_count`` because the geometry report already has a
        # ``box_pass_count`` that means a line crossing a card or its text -- one name for two
        # readings is how the two came to be read as the same number. ``box_pass_count`` is still
        # written beside it for readers of the archived reports, and goes away next version. All
        # three are counted from the drawn paths, in ``_route_metrics`` above.
        "crossing_count": total_crossings,
        "route_obstacle_overlap_count": route_obstacle_overlaps,
        "box_pass_count": route_obstacle_overlaps,
        "fallback_route_count": sum(1 for route in routes if route.get("route_generation_status") == "fallback_route_selected"),
        "aesthetic_tunnel_adjusted_count": sum(1 for route in routes if route.get("route_generation_status") == "aesthetic_tunnel_adjusted"),
        "aesthetic_style_only_count": sum(1 for route in routes if route.get("route_generation_status") == "aesthetic_style_only"),
        "reference_tunnel_violations": tunnel_violations,
        "average_aesthetic_score": round(sum(float(route["aesthetic_score"]) for route in routes) / max(len(routes), 1), 2),
        "reference_path_overrides": [route["id"] for route in routes if route.get("reference_locked") and not route.get("reference_path_preserved")],
        "router_constraint_version": ROUTER_CONSTRAINT_VERSION if router_rules else "",
        "router_constraints": dict(router_rules),
        "router_constraint_applied": router_constraint_report.get("applied", []),
        "router_constraint_unmet": router_constraint_report.get("unmet", []),
        "routes": routes,
    }
    program["arrows"] = enriched
    program["control_shapes"] = [dict(item) for item in program["arrows"]]
    program.setdefault("style", {})["arrow_style_profile_path"] = "arrow_style_profile.json"
    write_json(out / "arrow_style_profile.json", arrow_style_profile)
    write_json(out / "selected_arrow_routes.json", selected_routes)
    write_json(out / "arrow_quality_report.json", quality_report)
    write_json(out / "arrowhead_limits_report.json", arrowhead_report)
    write_json(out / "figure_program.json", program)
    return program
