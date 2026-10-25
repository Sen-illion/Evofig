"""Deterministic defect patterns: the shape of a problem, read off a figure's structure.

A condition written as ``node_count_bucket=high`` cannot tell two figures apart that both have
many nodes -- one of which has a four-way fan-out crossing two panels and the other of which is
a tidy pipeline. So a Memory row could not recognise "a different paper with the same illness".
These patterns are that illness named: pure structure, computed the same way every time, and
carried into the condition so a row can be retrieved for a figure it has never seen.

Nothing here is planned by a model. Every token is a count or a comparison over the panels,
nodes and connectors the pipeline already wrote down.
"""

from __future__ import annotations

from typing import Any, Iterable


#: The tokens this module may emit, so callers and contracts can name them without guessing.
DEFECT_PATTERNS = (
    "fanout_1_to_n",
    "cross_panel",
    "edge_crossings",
    "dashed_cycle",
    "long_edge",
    "port_collision",
)

#: A fan-out is one-to-many once one node feeds this many others.
FANOUT_MIN = 4

#: A connector is long once it spans this much of the canvas diagonal.
LONG_EDGE_MIN = 0.5

#: Two connectors that leave the same node towards the same panel share a port.
PORT_COLLISION_MIN = 2


def _items(source: dict[str, Any], key: str, *fallbacks: str) -> list[dict[str, Any]]:
    for name in (key, *fallbacks):
        value = source.get(name)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _box(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    box = item.get("bbox_percent") if isinstance(item.get("bbox_percent"), dict) else item.get("bbox") if isinstance(item.get("bbox"), dict) else None
    if not isinstance(box, dict):
        return None
    values = []
    for axis in ("x", "y", "w", "h"):
        value = box.get(axis)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        values.append(float(value))
    return values[0], values[1], values[2], values[3]


def _centre(item: dict[str, Any]) -> tuple[float, float] | None:
    box = _box(item)
    if box is None:
        return None
    return box[0] + box[2] / 2.0, box[1] + box[3] / 2.0


def _inside(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    return box[0] <= point[0] <= box[0] + box[2] and box[1] <= point[1] <= box[1] + box[3]


def _endpoints(figure: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str]:
    """Read the figure as (nodes, edges, source key, target key), whatever it is called."""

    nodes = _items(figure, "nodes", "modules")
    if "edges" in figure and isinstance(figure.get("edges"), list):
        return nodes, _items(figure, "edges"), "source", "target"
    return nodes, _items(figure, "connectors"), "source", "target"


def defect_patterns(figure: dict[str, Any] | None) -> list[str]:
    """Name the structural defects a figure has, sorted, using only what was written down.

    An empty or unreadable figure reports no defects rather than a guess: a condition that
    fires on nothing would exclude every candidate, which is the opposite of what a pattern
    is for.
    """

    source = figure if isinstance(figure, dict) else {}
    panels = _items(source, "panels", "groups")
    nodes, edges, source_key, target_key = _endpoints(source)
    if not nodes and not edges:
        return []
    node_by_id = {str(node.get("id") or ""): node for node in nodes if str(node.get("id") or "")}
    panel_by_id = {str(panel.get("id") or ""): panel for panel in panels if str(panel.get("id") or "")}
    owner = {node_id: str(node.get("panel_id") or "") for node_id, node in node_by_id.items()}
    outgoing: dict[str, list[str]] = {}
    for edge in edges:
        outgoing.setdefault(str(edge.get(source_key) or ""), []).append(str(edge.get(target_key) or ""))
    found: set[str] = set()

    if any(len(targets) >= FANOUT_MIN for targets in outgoing.values()):
        found.add("fanout_1_to_n")

    if any(owner.get(str(edge.get(source_key) or ""), "") != owner.get(str(edge.get(target_key) or ""), "") for edge in edges):
        found.add("cross_panel")

    crossings = 0
    for edge in edges:
        start, end = _centre(node_by_id.get(str(edge.get(source_key) or ""), {})), _centre(node_by_id.get(str(edge.get(target_key) or ""), {}))
        if start is None or end is None:
            continue
        midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        owner_ids = {owner.get(str(edge.get(source_key) or ""), ""), owner.get(str(edge.get(target_key) or ""), "")}
        for panel_id, panel in panel_by_id.items():
            box = _box(panel)
            if panel_id not in owner_ids and box is not None and _inside(midpoint, box):
                crossings += 1
                break
    if crossings >= 2:
        found.add("edge_crossings")

    if _has_cycle(node_by_id, edges, source_key, target_key):
        found.add("dashed_cycle")

    for edge in edges:
        start, end = _centre(node_by_id.get(str(edge.get(source_key) or ""), {})), _centre(node_by_id.get(str(edge.get(target_key) or ""), {}))
        if start is None or end is None:
            continue
        if ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5 >= LONG_EDGE_MIN:
            found.add("long_edge")
            break

    shared: dict[tuple[str, str], int] = {}
    for edge in edges:
        key = (str(edge.get(source_key) or ""), owner.get(str(edge.get(target_key) or ""), ""))
        shared[key] = shared.get(key, 0) + 1
    if any(count >= PORT_COLLISION_MIN for (source_id, panel_id), count in shared.items() if source_id and panel_id):
        found.add("port_collision")

    return sorted(found)


def _has_cycle(nodes: dict[str, dict[str, Any]], edges: list[dict[str, Any]], source_key: str, target_key: str) -> bool:
    """Whether the connectors form a directed cycle, which is what a dashed feedback loop is."""

    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(str(edge.get(source_key) or ""), []).append(str(edge.get(target_key) or ""))
    visiting: set[str] = set()
    done: set[str] = set()

    def walk(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in done or node_id not in adjacency:
            return False
        visiting.add(node_id)
        if any(walk(target) for target in adjacency[node_id]):
            return True
        visiting.discard(node_id)
        done.add(node_id)
        return False

    return any(walk(node_id) for node_id in list(nodes))


def missing_patterns(condition: dict[str, Any] | None, present: Iterable[str]) -> list[str]:
    """The patterns a condition asks for that the figure does not have."""

    asked = [str(item) for item in ((condition or {}).get("defect_patterns") or []) if str(item)]
    have = {str(item) for item in present}
    return [pattern for pattern in asked if pattern not in have]


__all__ = ["DEFECT_PATTERNS", "FANOUT_MIN", "LONG_EDGE_MIN", "PORT_COLLISION_MIN", "defect_patterns", "missing_patterns"]
