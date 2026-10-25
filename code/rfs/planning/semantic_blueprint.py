from __future__ import annotations

from collections import defaultdict, deque
from math import hypot
from typing import Any


_SUPPORTED_TOPOLOGIES = {"linear", "branch", "multimodal", "feedback", "dense_multiframe"}
_FEEDBACK_TYPES = {"feedback_loop", "iteration", "iterate", "return_flow"}


def _label(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("label") or item.get("title") or "").strip()


def _normalized(value: Any) -> str:
    return "".join(char for char in str(value or "").casefold() if char.isalnum())


def _visible_entities(specification: dict[str, Any]) -> list[dict[str, Any]]:
    required = {
        _normalized(value)
        for value in specification.get("required_labels", [])
        if str(value).strip()
    }
    entities: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    for field in ("inputs", "modules", "outputs", "innovations"):
        values = specification.get(field, [])
        if not isinstance(values, list):
            continue
        for index, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            label = _label(item)
            entity_id = str(item.get("id") or f"{field}_{index + 1}").strip()
            normalized_label = _normalized(label)
            if not label or not entity_id or entity_id in seen_ids or normalized_label in seen_labels:
                continue
            if required and normalized_label not in required:
                continue
            seen_ids.add(entity_id)
            seen_labels.add(normalized_label)
            entities.append({
                "id": entity_id,
                "label": label,
                "field": field,
                "role": str(item.get("role") or "").strip(),
                "evidence_ids": [str(value) for value in item.get("evidence_ids", []) if value],
                "order": len(entities),
            })
    return entities


def _dense_overview_panels(specification: dict[str, Any]) -> list[dict[str, Any]]:
    values = specification.get("overview_panels", [])
    if not isinstance(values, list):
        return []
    panels: list[dict[str, Any]] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("title") or "").strip()
        panel_id = str(item.get("id") or f"panel_{index + 1}").strip()
        if not label or not panel_id:
            continue
        panels.append({
            "id": panel_id,
            "panel_label": str(item.get("panel_label") or chr(ord("a") + index)).strip(),
            "label": label,
            "description": str(item.get("description") or "").strip(),
            "evidence_ids": [str(value) for value in item.get("evidence_ids", []) if value],
            "entity_ids": [str(value) for value in item.get("entity_ids", []) if value],
            "order": index,
        })
    return panels


def _layout_dense_panels(panels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    count = len(panels)
    positioned: list[dict[str, Any]] = []
    if count == 3:
        boxes = [
            {"x": 0.04, "y": 0.05, "w": 0.92, "h": 0.42},
            {"x": 0.04, "y": 0.53, "w": 0.44, "h": 0.42},
            {"x": 0.52, "y": 0.53, "w": 0.44, "h": 0.42},
        ]
    else:
        columns = 2 if count <= 4 else 3
        rows = (count + columns - 1) // columns
        gap_x, gap_y = 0.04, 0.06
        width = (0.92 - gap_x * (columns - 1)) / columns
        height = (0.90 - gap_y * (rows - 1)) / rows
        boxes = [
            {
                "x": round(0.04 + (index % columns) * (width + gap_x), 6),
                "y": round(0.05 + (index // columns) * (height + gap_y), 6),
                "w": round(width, 6),
                "h": round(height, 6),
            }
            for index in range(count)
        ]
    for panel, box in zip(panels, boxes):
        positioned.append({**panel, "bbox_percent": box})
    return positioned


def _layout_nodes_in_dense_panels(
    nodes: list[dict[str, Any]],
    relations: list[dict[str, str]],
    panels: list[dict[str, Any]],
    global_ranks: dict[str, int],
) -> list[dict[str, Any]]:
    if not panels:
        return _layout_nodes(nodes, _rank_nodes(nodes, relations))
    panel_by_entity = {
        entity_id: panel["id"]
        for panel in panels
        for entity_id in panel.get("entity_ids", [])
    }
    panel_by_evidence = {
        evidence_id: panel["id"]
        for panel in panels
        for evidence_id in panel.get("evidence_ids", [])
    }
    default_panel = panels[0]["id"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        panel_id = panel_by_entity.get(node["id"])
        if not panel_id:
            panel_id = next((panel_by_evidence[value] for value in node.get("evidence_ids", []) if value in panel_by_evidence), None)
        grouped[panel_id or default_panel].append(node)
    by_panel = {panel["id"]: panel for panel in panels}
    positioned: list[dict[str, Any]] = []
    for panel_id, panel_nodes in grouped.items():
        panel = by_panel.get(panel_id) or panels[0]
        node_ids = {item["id"] for item in panel_nodes}
        local_relations = [
            item for item in relations
            if item["source"] in node_ids and item["target"] in node_ids
        ]
        local_layout = _layout_nodes(panel_nodes, _rank_nodes(panel_nodes, local_relations))
        box = panel["bbox_percent"]
        inner_x = box["x"] + box["w"] * 0.035
        inner_y = box["y"] + box["h"] * 0.24
        inner_w = box["w"] * 0.93
        inner_h = box["h"] * 0.70
        for item in local_layout:
            local_box = item["bbox_percent"]
            positioned.append({
                **item,
                "panel_id": panel["id"],
                "global_rank": int(global_ranks.get(item["id"], item.get("rank", 0))),
                "bbox_percent": {
                    "x": round(inner_x + local_box["x"] * inner_w, 6),
                    "y": round(inner_y + local_box["y"] * inner_h, 6),
                    "w": round(local_box["w"] * inner_w, 6),
                    "h": round(local_box["h"] * inner_h, 6),
                },
            })
    return positioned


def _visible_relations(specification: dict[str, Any], node_ids: set[str]) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    values = specification.get("relations", [])
    if not isinstance(values, list):
        return relations
    type_priority = {
        "feedback_loop": 0,
        "branch": 1,
        "conditioning": 2,
        "feature_flow": 3,
        "data_flow": 4,
    }
    candidates: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for item in values:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        if not source or not target or source == target or source not in node_ids or target not in node_ids:
            continue
        candidates[(source, target)].append({
            "source": source,
            "target": target,
            "type": str(item.get("type") or "data_flow").strip(),
            "label": str(item.get("label") or "").strip(),
        })
    for pair, items in candidates.items():
        if pair in seen:
            continue
        selected = sorted(items, key=lambda item: type_priority.get(item["type"], 9))[0]
        labels = [item["label"] for item in items if item["label"]]
        if labels:
            selected["label"] = labels[0]
        relations.append(selected)
        seen.add(pair)
    return relations


def _rank_nodes(nodes: list[dict[str, Any]], relations: list[dict[str, str]]) -> dict[str, int]:
    node_ids = [item["id"] for item in nodes]
    incoming: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    outgoing: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    forward_relations = [
        item for item in relations
        if item["type"] not in _FEEDBACK_TYPES
    ]
    for item in forward_relations:
        incoming[item["target"]].add(item["source"])
        outgoing[item["source"]].add(item["target"])

    indegree = {node_id: len(incoming[node_id]) for node_id in node_ids}
    order_lookup = {item["id"]: item["order"] for item in nodes}
    queue = deque(sorted((node_id for node_id in node_ids if indegree[node_id] == 0), key=order_lookup.get))
    ranks = {node_id: 0 for node_id in node_ids}
    visited: set[str] = set()
    while queue:
        source = queue.popleft()
        visited.add(source)
        for target in sorted(outgoing[source], key=order_lookup.get):
            ranks[target] = max(ranks[target], ranks[source] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    if len(visited) != len(node_ids):
        for _ in range(len(node_ids)):
            changed = False
            for item in forward_relations:
                candidate = min(len(node_ids) - 1, ranks[item["source"]] + 1)
                if candidate > ranks[item["target"]]:
                    ranks[item["target"]] = candidate
                    changed = True
            if not changed:
                break

    input_ids = {item["id"] for item in nodes if item["field"] == "inputs"}
    for node_id in input_ids:
        if not incoming[node_id]:
            ranks[node_id] = 0
    relations_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for relation in forward_relations:
        relations_by_source[relation["source"]].append(relation)
    for node in nodes:
        node_id = node["id"]
        outgoing_relations = relations_by_source.get(node_id, [])
        if incoming[node_id] or not outgoing_relations:
            continue
        if all(str(item.get("type") or "").casefold() == "conditioning" for item in outgoing_relations):
            target_rank = min(ranks[item["target"]] for item in outgoing_relations)
            ranks[node_id] = max(0, target_rank - 1)
    return ranks


def _layout_nodes(nodes: list[dict[str, Any]], ranks: dict[str, int]) -> list[dict[str, Any]]:
    layers: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in nodes:
        layers[ranks[item["id"]]].append(item)
    compact_ranks = {rank: index for index, rank in enumerate(sorted(layers))}
    layer_count = max(1, len(compact_ranks))
    node_width = min(0.16, max(0.115, 0.72 / layer_count))
    x_start, x_end = 0.07, 0.93
    positioned: list[dict[str, Any]] = []
    for original_rank in sorted(layers):
        layer_index = compact_ranks[original_rank]
        items = sorted(layers[original_rank], key=lambda item: (item["field"] == "innovations", item["field"] == "outputs", item["order"]))
        count = len(items)
        center_x = (x_start + x_end) / 2 if layer_count == 1 else x_start + layer_index * (x_end - x_start) / (layer_count - 1)
        node_height = min(0.16, max(0.075, 0.66 / max(count, 1)))
        if count == 1:
            centers_y = [0.50]
        else:
            y_start, y_end = 0.15, 0.85
            centers_y = [y_start + index * (y_end - y_start) / (count - 1) for index in range(count)]
        for item, center_y in zip(items, centers_y):
            x = max(0.015, min(0.985 - node_width, center_x - node_width / 2))
            positioned.append({
                **item,
                "rank": layer_index,
                "bbox_percent": {
                    "x": round(x, 6),
                    "y": round(max(0.06, center_y - node_height / 2), 6),
                    "w": round(node_width, 6),
                    "h": round(node_height, 6),
                },
            })
    return positioned


def _connector_path(
    source: dict[str, Any],
    target: dict[str, Any],
    relation_type: str,
    source_port_fraction: float = 0.5,
    target_port_fraction: float = 0.5,
) -> tuple[list[list[float]], str]:
    source_box = source["bbox_percent"]
    target_box = target["bbox_percent"]
    source_center = (source_box["x"] + source_box["w"] / 2, source_box["y"] + source_box["h"] / 2)
    target_center = (target_box["x"] + target_box["w"] / 2, target_box["y"] + target_box["h"] / 2)
    source_rank = int(source.get("global_rank", source.get("rank", 0)))
    target_rank = int(target.get("global_rank", target.get("rank", 0)))
    is_return = relation_type in _FEEDBACK_TYPES or target_rank <= source_rank
    if is_return:
        route_y = 0.95 if source_center[1] >= 0.42 or target_center[1] >= 0.42 else 0.05
        source_anchor_y = source_box["y"] + source_box["h"] if route_y > 0.5 else source_box["y"]
        target_anchor_y = target_box["y"] + target_box["h"] if route_y > 0.5 else target_box["y"]
        return ([
            [round(source_center[0], 6), round(source_anchor_y, 6)],
            [round(source_center[0], 6), route_y],
            [round(target_center[0], 6), route_y],
            [round(target_center[0], 6), round(target_anchor_y, 6)],
        ], "outer_feedback")
    if source_rank == target_rank:
        if source_center[1] <= target_center[1]:
            start = [source_center[0], source_box["y"] + source_box["h"]]
            end = [target_center[0], target_box["y"]]
        else:
            start = [source_center[0], source_box["y"]]
            end = [target_center[0], target_box["y"] + target_box["h"]]
        return ([[round(value, 6) for value in start], [round(value, 6) for value in end]], "vertical")
    if target_center[0] >= source_center[0]:
        start = [source_box["x"] + source_box["w"], source_box["y"] + source_port_fraction * source_box["h"]]
        end = [target_box["x"], target_box["y"] + target_port_fraction * target_box["h"]]
    else:
        start = [source_box["x"], source_box["y"] + source_port_fraction * source_box["h"]]
        end = [target_box["x"] + target_box["w"], target_box["y"] + target_port_fraction * target_box["h"]]
    if target_rank - source_rank > 1:
        lane_y = max(0.035, min(0.965, min(source_box["y"], target_box["y"]) - 0.045))
        if lane_y < 0.055:
            lane_y = min(0.965, max(source_box["y"] + source_box["h"], target_box["y"] + target_box["h"]) + 0.045)
        return ([
            [round(value, 6) for value in start],
            [round(start[0] + 0.025, 6), round(start[1], 6)],
            [round(start[0] + 0.025, 6), round(lane_y, 6)],
            [round(end[0] - 0.025, 6), round(lane_y, 6)],
            [round(end[0] - 0.025, 6), round(end[1], 6)],
            [round(value, 6) for value in end],
        ], "bypass_orthogonal")
    if abs(start[1] - end[1]) < 0.025:
        return ([[round(value, 6) for value in start], [round(value, 6) for value in end]], "straight")
    lane_fraction = 0.20 + 0.60 * (0.65 * target_port_fraction + 0.35 * source_port_fraction)
    mid_x = start[0] + (end[0] - start[0]) * lane_fraction
    return ([
        [round(value, 6) for value in start],
        [round(mid_x, 6), round(start[1], 6)],
        [round(mid_x, 6), round(end[1], 6)],
        [round(value, 6) for value in end],
    ], "orthogonal")


def _segments(path: list[list[float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    return [
        ((float(start[0]), float(start[1])), (float(end[0]), float(end[1])))
        for start, end in zip(path[:-1], path[1:])
        if start != end
    ]


def _segment_length(segment: tuple[tuple[float, float], tuple[float, float]]) -> float:
    return hypot(segment[1][0] - segment[0][0], segment[1][1] - segment[0][1])


def _path_length(path: list[list[float]]) -> float:
    return sum(_segment_length(segment) for segment in _segments(path))


def _point_on_segment(
    point: tuple[float, float],
    segment: tuple[tuple[float, float], tuple[float, float]],
    *,
    tolerance: float = 1e-7,
) -> bool:
    (x, y), ((x1, y1), (x2, y2)) = point, segment
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > tolerance:
        return False
    return min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance


def _segment_crosses(
    first: tuple[tuple[float, float], tuple[float, float]],
    second: tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    (a1, a2), (b1, b2) = first, second
    first_vertical = abs(a1[0] - a2[0]) < 1e-7
    second_vertical = abs(b1[0] - b2[0]) < 1e-7
    if first_vertical == second_vertical:
        return False
    vertical, horizontal = (first, second) if first_vertical else (second, first)
    intersection = (vertical[0][0], horizontal[0][1])
    if not _point_on_segment(intersection, vertical) or not _point_on_segment(intersection, horizontal):
        return False
    endpoints = {vertical[0], vertical[1], horizontal[0], horizontal[1]}
    return intersection not in endpoints


def _shared_segment_overlap(
    first: tuple[tuple[float, float], tuple[float, float]],
    second: tuple[tuple[float, float], tuple[float, float]],
    *,
    lane_tolerance: float = 0.006,
) -> float:
    first_vertical = abs(first[0][0] - first[1][0]) < 1e-7
    second_vertical = abs(second[0][0] - second[1][0]) < 1e-7
    if first_vertical != second_vertical:
        return 0.0
    if first_vertical:
        if abs(first[0][0] - second[0][0]) > lane_tolerance:
            return 0.0
        first_range = sorted((first[0][1], first[1][1]))
        second_range = sorted((second[0][1], second[1][1]))
    else:
        if abs(first[0][1] - second[0][1]) > lane_tolerance:
            return 0.0
        first_range = sorted((first[0][0], first[1][0]))
        second_range = sorted((second[0][0], second[1][0]))
    return max(0.0, min(first_range[1], second_range[1]) - max(first_range[0], second_range[0]))


def _segment_intersects_box(
    segment: tuple[tuple[float, float], tuple[float, float]],
    box: dict[str, Any],
    *,
    padding: float = 0.006,
) -> bool:
    left = float(box["x"]) - padding
    right = float(box["x"]) + float(box["w"]) + padding
    top = float(box["y"]) - padding
    bottom = float(box["y"]) + float(box["h"]) + padding
    (x1, y1), (x2, y2) = segment
    if abs(x1 - x2) < 1e-7:
        return left < x1 < right and max(min(y1, y2), top) < min(max(y1, y2), bottom)
    if abs(y1 - y2) < 1e-7:
        return top < y1 < bottom and max(min(x1, x2), left) < min(max(x1, x2), right)
    return False


def _compact_path(path: list[list[float]]) -> list[list[float]]:
    compact: list[list[float]] = []
    for point in path:
        normalized = [round(float(point[0]), 6), round(float(point[1]), 6)]
        if compact and normalized == compact[-1]:
            continue
        compact.append(normalized)
        while len(compact) >= 3:
            a, b, c = compact[-3:]
            if (abs(a[0] - b[0]) < 1e-7 and abs(b[0] - c[0]) < 1e-7) or (abs(a[1] - b[1]) < 1e-7 and abs(b[1] - c[1]) < 1e-7):
                compact.pop(-2)
            else:
                break
    return compact


def _node_center_y(node: dict[str, Any]) -> float:
    box = node["bbox_percent"]
    return float(box["y"]) + float(box["h"]) / 2


def _relation_ports(
    relations: list[dict[str, str]],
    by_id: dict[str, dict[str, Any]],
) -> dict[int, tuple[float, float]]:
    incoming: dict[str, list[dict[str, str]]] = defaultdict(list)
    outgoing: dict[str, list[dict[str, str]]] = defaultdict(list)
    for relation in relations:
        incoming[relation["target"]].append(relation)
        outgoing[relation["source"]].append(relation)
    ports: dict[int, tuple[float, float]] = {}
    for relation in relations:
        target_items = sorted(
            incoming[relation["target"]],
            key=lambda item: (_node_center_y(by_id[item["source"]]), relations.index(item)),
        )
        source_items = sorted(
            outgoing[relation["source"]],
            key=lambda item: (_node_center_y(by_id[item["target"]]), relations.index(item)),
        )
        ports[id(relation)] = (
            (source_items.index(relation) + 1) / (len(source_items) + 1),
            (target_items.index(relation) + 1) / (len(target_items) + 1),
        )
    return ports


def _forward_endpoints(
    source: dict[str, Any],
    target: dict[str, Any],
    source_fraction: float,
    target_fraction: float,
) -> tuple[list[float], list[float]]:
    source_box = source["bbox_percent"]
    target_box = target["bbox_percent"]
    source_center_x = source_box["x"] + source_box["w"] / 2
    target_center_x = target_box["x"] + target_box["w"] / 2
    if target_center_x >= source_center_x:
        return (
            [source_box["x"] + source_box["w"], source_box["y"] + source_fraction * source_box["h"]],
            [target_box["x"], target_box["y"] + target_fraction * target_box["h"]],
        )
    return (
        [source_box["x"], source_box["y"] + source_fraction * source_box["h"]],
        [target_box["x"] + target_box["w"], target_box["y"] + target_fraction * target_box["h"]],
    )


def _route_candidates(
    source: dict[str, Any],
    target: dict[str, Any],
    relation_type: str,
    source_fraction: float,
    target_fraction: float,
) -> list[dict[str, Any]]:
    base_path, base_style = _connector_path(
        source,
        target,
        relation_type,
        source_port_fraction=source_fraction,
        target_port_fraction=target_fraction,
    )
    candidates = [{"path": base_path, "route_style": base_style, "lane_assignment": "local_default"}]
    source_rank = int(source.get("global_rank", source.get("rank", 0)))
    target_rank = int(target.get("global_rank", target.get("rank", 0)))
    is_return = relation_type in _FEEDBACK_TYPES or target_rank <= source_rank
    if is_return:
        source_box, target_box = source["bbox_percent"], target["bbox_percent"]
        source_center_x = source_box["x"] + source_box["w"] / 2
        target_center_x = target_box["x"] + target_box["w"] / 2
        for lane_name, lane_y in (("outer_top", 0.035), ("outer_bottom", 0.965)):
            source_anchor_y = source_box["y"] if lane_y < 0.5 else source_box["y"] + source_box["h"]
            target_anchor_y = target_box["y"] if lane_y < 0.5 else target_box["y"] + target_box["h"]
            candidates.append({
                "path": [[source_center_x, source_anchor_y], [source_center_x, lane_y], [target_center_x, lane_y], [target_center_x, target_anchor_y]],
                "route_style": "outer_feedback",
                "lane_assignment": lane_name,
            })
        return candidates
    if source_rank == target_rank:
        return candidates
    start, end = _forward_endpoints(source, target, source_fraction, target_fraction)
    direction = 1.0 if end[0] >= start[0] else -1.0
    if target_rank - source_rank > 1:
        source_box = source["bbox_percent"]
        target_box = target["bbox_percent"]
        source_center_x = source_box["x"] + source_box["w"] / 2
        target_center_x = target_box["x"] + target_box["w"] / 2
        for lane_index, lane_y in enumerate((0.035, 0.055, 0.945, 0.965), start=1):
            source_x = start[0] + direction * (0.020 + 0.008 * (lane_index % 2))
            target_x = end[0] - direction * (0.020 + 0.008 * ((lane_index + 1) % 2))
            candidates.append({
                "path": [start, [source_x, start[1]], [source_x, lane_y], [target_x, lane_y], [target_x, end[1]], end],
                "route_style": "bypass_orthogonal",
                "lane_assignment": ("outer_top" if lane_y < 0.5 else "outer_bottom") + f"_{lane_index}",
            })
            source_edge_y = source_box["y"] if lane_y < 0.5 else source_box["y"] + source_box["h"]
            target_edge_y = target_box["y"] if lane_y < 0.5 else target_box["y"] + target_box["h"]
            candidates.append({
                "path": [[source_center_x, source_edge_y], [source_center_x, lane_y], [target_center_x, lane_y], [target_center_x, target_edge_y]],
                "route_style": "bypass_orthogonal",
                "lane_assignment": ("outer_top_edge" if lane_y < 0.5 else "outer_bottom_edge") + f"_{lane_index}",
            })
        return candidates
    if abs(start[1] - end[1]) >= 0.025:
        for lane_index, fraction in enumerate((0.24, 0.38, 0.50, 0.62, 0.76), start=1):
            lane_x = start[0] + (end[0] - start[0]) * fraction
            candidates.append({
                "path": [start, [lane_x, start[1]], [lane_x, end[1]], end],
                "route_style": "orthogonal",
                "lane_assignment": f"inter_rank_{lane_index}",
            })
    return candidates


def _candidate_metrics(
    candidate_path: list[list[float]],
    relation: dict[str, str],
    nodes: list[dict[str, Any]],
    accepted_paths: list[list[list[float]]],
) -> dict[str, float | int]:
    candidate_segments = _segments(candidate_path)
    obstacle_intersections = 0
    for node in nodes:
        if node["id"] in {relation["source"], relation["target"]}:
            continue
        if any(_segment_intersects_box(segment, node["bbox_percent"]) for segment in candidate_segments):
            obstacle_intersections += 1
    crossings = 0
    shared_overlap = 0.0
    for accepted_path in accepted_paths:
        accepted_segments = _segments(accepted_path)
        crossings += sum(1 for first in candidate_segments for second in accepted_segments if _segment_crosses(first, second))
        shared_overlap += sum(_shared_segment_overlap(first, second) for first in candidate_segments for second in accepted_segments)
    return {
        "node_obstacle_intersections": obstacle_intersections,
        "crossing_count": crossings,
        "shared_segment_overlap": round(shared_overlap, 6),
        "bend_count": max(0, len(candidate_path) - 2),
        "path_length": round(_path_length(candidate_path), 6),
    }


def _route_connectors(
    positioned_nodes: list[dict[str, Any]],
    relations: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {item["id"]: item for item in positioned_nodes}
    ports = _relation_ports(relations, by_id)
    indexed_relations = list(enumerate(relations))

    def priority(item: tuple[int, dict[str, str]]) -> tuple[int, int, int]:
        index, relation = item
        source = by_id[relation["source"]]
        target = by_id[relation["target"]]
        source_rank = int(source.get("global_rank", source.get("rank", 0)))
        target_rank = int(target.get("global_rank", target.get("rank", 0)))
        is_return = relation["type"] in _FEEDBACK_TYPES or target_rank <= source_rank
        return (2 if is_return else 1 if target_rank - source_rank > 1 else 0, abs(target_rank - source_rank), index)

    accepted_paths: list[list[list[float]]] = []
    routed: dict[int, dict[str, Any]] = {}
    for original_index, relation in sorted(indexed_relations, key=priority):
        source_fraction, target_fraction = ports[id(relation)]
        candidates = _route_candidates(
            by_id[relation["source"]],
            by_id[relation["target"]],
            relation["type"],
            source_fraction,
            target_fraction,
        )
        scored = []
        for candidate_index, candidate in enumerate(candidates):
            path = _compact_path(candidate["path"])
            metrics = _candidate_metrics(path, relation, positioned_nodes, accepted_paths)
            score = (
                int(metrics["node_obstacle_intersections"]) * 10000
                + float(metrics["shared_segment_overlap"]) * 4000
                + int(metrics["crossing_count"]) * 250
                + int(metrics["bend_count"]) * 0.7
                + float(metrics["path_length"])
                + candidate_index * 1e-5
            )
            scored.append((score, path, candidate, metrics))
        _, path, selected, metrics = min(scored, key=lambda item: item[0])
        accepted_paths.append(path)
        routed[original_index] = {
            "id": f"semantic_connector_{original_index + 1:02d}",
            **relation,
            "path_percent": path,
            "route_style": selected["route_style"],
            "lane_assignment": selected["lane_assignment"],
            "routing_metrics": metrics,
        }

    connectors = [routed[index] for index in range(len(relations))]
    crossing_count = 0
    shared_overlap = 0.0
    for first_index, first in enumerate(connectors):
        for second in connectors[first_index + 1:]:
            first_segments = _segments(first["path_percent"])
            second_segments = _segments(second["path_percent"])
            crossing_count += sum(1 for a in first_segments for b in second_segments if _segment_crosses(a, b))
            shared_overlap += sum(_shared_segment_overlap(a, b) for a in first_segments for b in second_segments)
    obstacle_intersections = sum(int(item["routing_metrics"]["node_obstacle_intersections"]) for item in connectors)
    quality = {
        "routing_algorithm": "global_candidate_lane_router_v1",
        "crossing_count": crossing_count,
        "shared_segment_overlap": round(shared_overlap, 6),
        "node_obstacle_intersections": obstacle_intersections,
        "lane_assignments": {item["id"]: item["lane_assignment"] for item in connectors},
        "passed": obstacle_intersections == 0 and crossing_count == 0 and shared_overlap < 0.01,
    }
    return connectors, quality


def _compile_panel_graph_blueprint(
    specification: dict[str, Any],
    *,
    max_nodes: int,
) -> dict[str, Any] | None:
    raw_panels = specification.get("panel_graphs", [])
    if not isinstance(raw_panels, list) or len(raw_panels) < 2:
        return None
    panels: list[dict[str, Any]] = []
    panel_nodes: dict[str, list[dict[str, Any]]] = {}
    panel_relations: dict[str, list[dict[str, str]]] = {}
    panel_groups: dict[str, list[dict[str, Any]]] = {}
    total_nodes = 0
    total_relations = 0
    for panel_index, raw_panel in enumerate(raw_panels):
        if not isinstance(raw_panel, dict):
            continue
        panel_id = str(raw_panel.get("id") or f"panel_{panel_index + 1}").strip()
        label = str(raw_panel.get("label") or raw_panel.get("description") or "").strip()
        if not panel_id or not label:
            continue
        nodes: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for node_index, raw_node in enumerate(raw_panel.get("nodes", []) if isinstance(raw_panel.get("nodes"), list) else []):
            if not isinstance(raw_node, dict):
                continue
            instance_id = str(raw_node.get("instance_id") or raw_node.get("id") or "").strip()
            node_label = _label(raw_node)
            if not instance_id or not node_label or instance_id in seen_ids:
                continue
            role = str(raw_node.get("role") or "module").strip()
            role_key = role.casefold()
            if role_key == "group":
                groups.append({
                    "id": instance_id,
                    "label": node_label,
                    "role": role,
                    "evidence_ids": [str(value) for value in raw_node.get("evidence_ids", []) if value],
                })
                seen_ids.add(instance_id)
                continue
            field = "inputs" if "input" in role_key or "dataset" in role_key else "outputs" if any(term in role_key for term in ("output", "prediction", "objective", "loss")) else "modules"
            nodes.append({
                "id": instance_id,
                "semantic_id": str(raw_node.get("semantic_id") or "").strip() or None,
                "label": node_label,
                "field": field,
                "role": role,
                "evidence_ids": [str(value) for value in raw_node.get("evidence_ids", []) if value],
                "order": node_index,
                "panel_id": panel_id,
            })
            seen_ids.add(instance_id)
        operational_ids = {item["id"] for item in nodes}
        relations: list[dict[str, str]] = []
        for raw_relation in raw_panel.get("relations", []) if isinstance(raw_panel.get("relations"), list) else []:
            if not isinstance(raw_relation, dict):
                continue
            source = str(raw_relation.get("source") or "").strip()
            target = str(raw_relation.get("target") or "").strip()
            if not source or not target or source == target or source not in operational_ids or target not in operational_ids:
                continue
            relations.append({
                "source": source,
                "target": target,
                "type": str(raw_relation.get("type") or "data_flow").strip(),
                "label": str(raw_relation.get("label") or "").strip(),
            })
        panels.append({
            "id": panel_id,
            "panel_label": str(raw_panel.get("panel_label") or chr(ord("a") + panel_index)).strip(),
            "label": label,
            "description": str(raw_panel.get("description") or label).strip(),
            "evidence_ids": [str(value) for value in raw_panel.get("evidence_ids", []) if value],
            "entity_ids": [item["id"] for item in nodes],
            "groups": groups,
            "order": panel_index,
        })
        panel_nodes[panel_id] = nodes
        panel_relations[panel_id] = relations
        panel_groups[panel_id] = groups
        total_nodes += len(nodes)
        total_relations += len(relations)
    if len(panels) < 2 or total_nodes < 2 or total_relations < 1:
        return None
    panel_node_limit = max(24, int(max_nodes))
    if total_nodes > panel_node_limit:
        return {
            "summary": "Panel-local semantic blueprint skipped because the instance graph is too dense for one raster guide.",
            "applied": False,
            "topology": "dense_multiframe",
            "nodes": [],
            "connectors": [],
            "reason": "too_many_nodes",
            "node_count": total_nodes,
            "source": "panel_graphs",
        }

    positioned_panels = _layout_dense_panels(panels)
    by_panel = {item["id"]: item for item in positioned_panels}
    positioned_nodes: list[dict[str, Any]] = []
    connectors: list[dict[str, Any]] = []
    isolated_node_ids: list[str] = []
    connector_index = 1
    panel_route_quality: list[dict[str, Any]] = []
    for panel in positioned_panels:
        panel_id = panel["id"]
        nodes = panel_nodes.get(panel_id, [])
        relations = panel_relations.get(panel_id, [])
        ranks = _rank_nodes(nodes, relations)
        local_layout = _layout_nodes(nodes, ranks)
        box = by_panel[panel_id]["bbox_percent"]
        inner_x = box["x"] + box["w"] * 0.035
        inner_y = box["y"] + box["h"] * 0.22
        inner_w = box["w"] * 0.93
        inner_h = box["h"] * 0.72
        local_positioned: list[dict[str, Any]] = []
        for item in local_layout:
            local_box = item["bbox_percent"]
            local_positioned.append({
                **item,
                "panel_id": panel_id,
                "global_rank": int(ranks.get(item["id"], item.get("rank", 0))),
                "bbox_percent": {
                    "x": round(inner_x + local_box["x"] * inner_w, 6),
                    "y": round(inner_y + local_box["y"] * inner_h, 6),
                    "w": round(local_box["w"] * inner_w, 6),
                    "h": round(local_box["h"] * inner_h, 6),
                },
            })
        positioned_nodes.extend(local_positioned)
        connected_ids = {value for relation in relations for value in (relation["source"], relation["target"])}
        isolated_node_ids.extend(item["id"] for item in nodes if item["id"] not in connected_ids)
        local_connectors, local_quality = _route_connectors(local_positioned, relations)
        panel_route_quality.append({"panel_id": panel_id, **local_quality})
        for connector in local_connectors:
            connectors.append({
                **connector,
                "id": f"semantic_connector_{connector_index:02d}",
                "panel_id": panel_id,
            })
            connector_index += 1
    route_quality = {
        "routing_algorithm": "panel_local_global_candidate_lane_router_v1",
        "crossing_count": sum(int(item["crossing_count"]) for item in panel_route_quality),
        "shared_segment_overlap": round(sum(float(item["shared_segment_overlap"]) for item in panel_route_quality), 6),
        "node_obstacle_intersections": sum(int(item["node_obstacle_intersections"]) for item in panel_route_quality),
        "panel_reports": panel_route_quality,
        "passed": all(bool(item["passed"]) for item in panel_route_quality),
    }
    return {
        "summary": "Panel-local paper semantic graphs compiled into normalized node and connector geometry.",
        "applied": True,
        "topology": "dense_multiframe",
        "node_count": len(positioned_nodes),
        "connector_count": len(connectors),
        "isolated_node_ids": isolated_node_ids,
        "panels": positioned_panels,
        "nodes": positioned_nodes,
        "connectors": connectors,
        "route_quality": route_quality,
        "source": "panel_graphs",
    }


def compile_semantic_blueprint(
    specification: dict[str, Any] | None,
    *,
    max_nodes: int = 16,
) -> dict[str, Any]:
    specification = specification if isinstance(specification, dict) else {}
    topology = str(specification.get("topology") or "unknown")
    if topology == "dense_multiframe":
        panel_graph_blueprint = _compile_panel_graph_blueprint(specification, max_nodes=max_nodes)
        if panel_graph_blueprint is not None:
            return panel_graph_blueprint
    nodes = _visible_entities(specification)
    if topology not in _SUPPORTED_TOPOLOGIES:
        return {"summary": "Generic semantic blueprint not applicable to this topology.", "applied": False, "topology": topology, "nodes": [], "connectors": [], "reason": "unsupported_topology"}
    if len(nodes) < 2:
        return {"summary": "Generic semantic blueprint requires at least two visible entities.", "applied": False, "topology": topology, "nodes": [], "connectors": [], "reason": "too_few_nodes"}
    if len(nodes) > max(2, int(max_nodes)):
        return {"summary": "Generic semantic blueprint skipped because the visible contract is too dense for one raster guide.", "applied": False, "topology": topology, "nodes": [], "connectors": [], "reason": "too_many_nodes", "node_count": len(nodes)}

    node_ids = {item["id"] for item in nodes}
    relations = _visible_relations(specification, node_ids)
    if not relations:
        return {"summary": "Generic semantic blueprint requires at least one declared relation.", "applied": False, "topology": topology, "nodes": [], "connectors": [], "reason": "no_relations"}

    connected_ids = {value for item in relations for value in (item["source"], item["target"])}
    isolated_node_ids = [item["id"] for item in nodes if item["id"] not in connected_ids]
    connected_nodes = [item for item in nodes if item["id"] in connected_ids]
    if len(connected_nodes) >= 2:
        nodes = connected_nodes

    global_ranks = _rank_nodes(nodes, relations)
    dense_panels = _layout_dense_panels(_dense_overview_panels(specification)) if topology == "dense_multiframe" else []
    if len(dense_panels) >= 2:
        positioned = _layout_nodes_in_dense_panels(nodes, relations, dense_panels, global_ranks)
    else:
        positioned = _layout_nodes(nodes, global_ranks)
    connectors, route_quality = _route_connectors(positioned, relations)
    return {
        "summary": "Paper-grounded semantic graph compiled into normalized node and connector geometry.",
        "applied": True,
        "topology": topology,
        "node_count": len(positioned),
        "connector_count": len(connectors),
        "isolated_node_ids": isolated_node_ids,
        "panels": dense_panels,
        "nodes": positioned,
        "connectors": connectors,
        "route_quality": route_quality,
        "source": "figure_specification",
    }
