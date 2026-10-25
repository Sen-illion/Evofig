"""Deterministic decomposability measurement for one compiled figure candidate.

The global structure plan declares that a figure must stay decomposable into slot
boundaries and cross-slot interfaces, but nothing measured that declaration: a round
that broke the panel/slot discipline only showed up later as a weak Judge score with
no attributable object. This module reads the compiled figure program, the visual
graph and the editable graph and reports seven measurements:

* ``unbound_scientific_nodes`` - scientific nodes no editable object is bound to;
* ``slot_boundary_overflows`` - slots drawn outside the object that owns them;
* ``orphan_nodes`` - semantic nodes without a declared panel;
* ``cross_panel_connectors`` - connectors whose endpoints sit in different panels;
* ``min_panel_occupancy`` - the emptiest panel, as payload area over panel area;
* ``empty_region_ratio`` - canvas area that no panel covers;
* ``max_node_slot_fanout`` - the busiest node, counting its slots and its text items.

The three hard measurements name a structural break that a local repair cannot fix;
the soft ones name crowding that a local repair may. Nothing here calls a provider:
the report is a pure function of its inputs, so the same candidate always produces
the same numbers and a run can be replayed and compared round by round.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..contracts import (
    DECOMPOSABILITY_CONTRACT,
    DECOMPOSABILITY_MAX_CROSS_PANEL_CONNECTORS,
    DECOMPOSABILITY_MAX_NODE_SLOT_FANOUT,
    DECOMPOSABILITY_MIN_PANEL_AREA,
    DECOMPOSABILITY_MIN_PANEL_OCCUPANCY,
    DECOMPOSABILITY_VERSION,
)


HARD_METRICS = ("unbound_scientific_nodes", "slot_boundary_overflows", "orphan_nodes")
SOFT_METRICS = ("cross_panel_connectors", "min_panel_occupancy")


def _items(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _area(box: Any) -> float:
    if not isinstance(box, dict):
        return 0.0
    try:
        return max(0.0, float(box.get("w") or 0.0)) * max(0.0, float(box.get("h") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _contains(outer: dict[str, Any], inner: dict[str, Any], tolerance: float = 0.005) -> bool:
    """Return whether ``inner`` sits inside ``outer`` within a small tolerance."""

    try:
        ox, oy = float(outer.get("x") or 0.0), float(outer.get("y") or 0.0)
        ow, oh = float(outer.get("w") or 0.0), float(outer.get("h") or 0.0)
        ix, iy = float(inner.get("x") or 0.0), float(inner.get("y") or 0.0)
        iw, ih = float(inner.get("w") or 0.0), float(inner.get("h") or 0.0)
    except (TypeError, ValueError):
        return True
    return (
        ix >= ox - tolerance
        and iy >= oy - tolerance
        and ix + iw <= ox + ow + tolerance
        and iy + ih <= oy + oh + tolerance
    )


def _endpoint(value: dict[str, Any], *keys: str) -> str:
    for key in keys:
        text = str(value.get(key) or "").strip()
        if text:
            return text
    return ""


def _soft_signature(violations: Iterable[dict[str, Any]]) -> frozenset[str]:
    return frozenset(str(item.get("metric") or "") for item in violations if item.get("kind") == "soft")


def _soft_streak(current: frozenset[str], history: Iterable[dict[str, Any]] | None) -> int:
    """Count this report plus the trailing prior reports carrying the same soft break."""

    if not current:
        return 0
    streak = 1
    previous = _items(history) if isinstance(history, list) else []
    for report in reversed(previous):
        violations = _items(report.get("violations"))
        if _soft_signature(violations) != current:
            break
        streak += 1
    return streak


def _union_box(boxes: list[dict[str, Any]]) -> dict[str, float]:
    """Return the bounding box of ``boxes`` in percent coordinates."""

    lefts, tops, rights, bottoms = [], [], [], []
    for box in boxes:
        try:
            x, y = float(box.get("x") or 0.0), float(box.get("y") or 0.0)
            w, h = float(box.get("w") or 0.0), float(box.get("h") or 0.0)
        except (TypeError, ValueError):
            continue
        lefts.append(x)
        tops.append(y)
        rights.append(x + w)
        bottoms.append(y + h)
    if not lefts:
        return {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
    return {
        "x": round(min(lefts), 4),
        "y": round(min(tops), 4),
        "w": round(max(rights) - min(lefts), 4),
        "h": round(max(bottoms) - min(tops), 4),
    }


def _panel_boxes(
    *,
    nodes: list[dict[str, Any]],
    declared: list[dict[str, Any]],
    node_panel: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], str]:
    """Return the panels to measure, naming where they came from.

    A heuristic run publishes no panel list at all, only the ``panel_id`` each node
    belongs to. Measuring against an empty panel list would report a clean figure for
    every candidate, so the groups the nodes already declare are measured as panels.
    """

    if declared:
        return {str(item.get("id") or ""): item for item in declared if str(item.get("id") or "")}, "declared"
    grouped: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        panel_id = node_panel.get(str(node.get("id") or ""), "")
        if panel_id:
            grouped.setdefault(panel_id, []).append(node.get("bbox_percent") or {})
    if not grouped:
        return {}, "none"
    boxes = {panel_id: {"id": panel_id, "bbox_percent": _union_box(values)} for panel_id, values in grouped.items()}
    return boxes, "synthesized"


def build_decomposability_report(
    editable_graph: dict[str, Any] | None,
    visual_graph: dict[str, Any] | None,
    composition: dict[str, Any] | None,
    *,
    panels: list[dict[str, Any]] | None = None,
    round_index: int = -1,
    history: list[dict[str, Any]] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure how cleanly one compiled candidate decomposes.

    ``panels`` overrides the program's own panel list when the composition planner
    returned no panels: the structural panels belong to the layout in that case, and
    measuring against an empty panel list would report every node as an orphan.

    ``thresholds`` lets a parameter-only prompt patch move the boundaries this module
    measures against, so "how crowded is too crowded" stays a deterministic parameter
    instead of a value baked into the code.
    """

    program = composition if isinstance(composition, dict) else {}
    limits = {
        "min_panel_occupancy": DECOMPOSABILITY_MIN_PANEL_OCCUPANCY,
        "max_cross_panel_connectors": DECOMPOSABILITY_MAX_CROSS_PANEL_CONNECTORS,
        "max_node_slot_fanout": DECOMPOSABILITY_MAX_NODE_SLOT_FANOUT,
        "min_panel_area": DECOMPOSABILITY_MIN_PANEL_AREA,
    }
    for key, value in (thresholds or {}).items():
        try:
            limits[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    nodes = _items(program.get("semantic_nodes"))
    slots = _items(program.get("slots"))
    arrows = _items(program.get("arrows"))
    text_items = _items((program.get("text_program") or {}).get("items") if isinstance(program.get("text_program"), dict) else [])
    node_panel = {str(node.get("id") or ""): str(node.get("panel_id") or "") for node in nodes}
    declared_panels = _items(panels) if isinstance(panels, list) else []
    declared_panels = declared_panels or _items(program.get("panels"))
    panel_by_id, panel_source = _panel_boxes(nodes=nodes, declared=declared_panels, node_panel=node_panel)
    violations: list[dict[str, Any]] = []

    declared_panel_ids = {str(item.get("id") or "") for item in declared_panels if str(item.get("id") or "")}
    orphan_ids = sorted(
        node_id for node_id, panel_id in node_panel.items()
        if node_id and (not panel_id or (declared_panel_ids and panel_id not in declared_panel_ids))
    )
    if orphan_ids:
        violations.append({
            "metric": "orphan_nodes",
            "value": len(orphan_ids),
            "threshold": 0,
            "kind": "hard",
            "reason": "semantic nodes are not grouped into a declared panel",
            "object_ids": orphan_ids,
            "target_node_ids": orphan_ids,
        })

    node_box = {str(node.get("id") or ""): node.get("bbox_percent") for node in nodes}
    overflow_ids: list[str] = []
    overflow_owners: list[str] = []
    for slot in slots:
        owner = str(slot.get("parent_id") or "").strip() or str(slot.get("id") or "").removeprefix("slot_")
        owner_box = node_box.get(owner)
        slot_box = slot.get("bbox_percent")
        if isinstance(owner_box, dict) and isinstance(slot_box, dict) and not _contains(owner_box, slot_box):
            overflow_ids.append(str(slot.get("id") or ""))
            if owner:
                overflow_owners.append(owner)
    if overflow_ids:
        violations.append({
            "metric": "slot_boundary_overflows",
            "value": len(overflow_ids),
            "threshold": 0,
            "kind": "hard",
            "reason": "a slot is drawn outside the object that owns it",
            "object_ids": sorted(overflow_ids),
            "target_node_ids": sorted(dict.fromkeys(overflow_owners)),
        })

    bound_scientific = {
        str(value)
        for node in _items((editable_graph or {}).get("nodes"))
        for value in (node.get("scientific_node_ids") or [])
        if str(value).strip()
    }
    declared_scientific = {
        str(value)
        for node in _items((visual_graph or {}).get("nodes"))
        for value in (node.get("scientific_node_ids") or [])
        if str(value).strip()
    }
    unbound_ids = sorted(declared_scientific - bound_scientific)
    if unbound_ids:
        violations.append({
            "metric": "unbound_scientific_nodes",
            "value": len(unbound_ids),
            "threshold": 0,
            "kind": "hard",
            "reason": "a scientific node has no editable object bound to it",
            "object_ids": unbound_ids,
            "target_node_ids": sorted({
                str(node.get("id") or "")
                for node in _items((visual_graph or {}).get("nodes"))
                if set(str(value) for value in (node.get("scientific_node_ids") or [])) & set(unbound_ids)
            } - {""}),
        })

    cross_ids: list[str] = []
    cross_nodes: list[str] = []
    for arrow in arrows:
        source = _endpoint(arrow, "source", "source_id")
        target = _endpoint(arrow, "target", "target_id")
        source_panel, target_panel = node_panel.get(source, ""), node_panel.get(target, "")
        if source_panel and target_panel and source_panel != target_panel:
            cross_ids.append(str(arrow.get("id") or ""))
            cross_nodes += [source, target]
    cross_count = len(cross_ids)
    if cross_count > limits["max_cross_panel_connectors"]:
        violations.append({
            "metric": "cross_panel_connectors",
            "value": cross_count,
            "threshold": limits["max_cross_panel_connectors"],
            "kind": "soft",
            "reason": "too many connectors cross a panel boundary to route cleanly",
            "object_ids": sorted(item for item in cross_ids if item),
            "target_connector_ids": sorted(item for item in cross_ids if item),
            "target_node_ids": sorted({item for item in cross_nodes if item}),
        })

    panel_ratios: list[float] = []
    panel_area_total = 0.0
    for panel in panel_by_id.values():
        area = _area(panel.get("bbox_percent"))
        if area < limits["min_panel_area"]:
            continue
        panel_area_total += area
        children = [node for node in nodes if node_panel.get(str(node.get("id") or "")) == str(panel.get("id") or "")]
        payload = sum(_area(node.get("bbox_percent")) for node in children)
        panel_ratios.append(round(min(1.0, payload / area), 4))
    min_panel_occupancy = round(min(panel_ratios), 4) if panel_ratios else 1.0
    low_panels = sorted(
        str(panel.get("id") or "")
        for panel in panel_by_id.values()
        if _area(panel.get("bbox_percent")) >= limits["min_panel_area"]
        and round(
            min(1.0, sum(
                _area(node.get("bbox_percent"))
                for node in nodes
                if node_panel.get(str(node.get("id") or "")) == str(panel.get("id") or "")
            ) / _area(panel.get("bbox_percent"))),
            4,
        )
        == min_panel_occupancy
    )
    if panel_ratios and min_panel_occupancy < limits["min_panel_occupancy"]:
        violations.append({
            "metric": "min_panel_occupancy",
            "value": min_panel_occupancy,
            "threshold": limits["min_panel_occupancy"],
            "kind": "soft",
            "reason": "a panel carries much less content than its area suggests",
            "object_ids": low_panels,
            "target_node_ids": sorted(
                str(node.get("id") or "")
                for node in nodes
                if node_panel.get(str(node.get("id") or "")) in set(low_panels) and str(node.get("id") or "")
            ),
        })

    empty_region_ratio = round(max(0.0, 1.0 - min(1.0, panel_area_total)), 4)

    fanout: dict[str, int] = {}
    for slot in slots:
        owner = str(slot.get("parent_id") or "").strip() or str(slot.get("id") or "").removeprefix("slot_")
        if owner:
            fanout[owner] = fanout.get(owner, 0) + 1
    for item in text_items:
        owner = _endpoint(item, "owner_node_id", "semantic_source_id", "target_id")
        if owner:
            fanout[owner] = fanout.get(owner, 0) + 1
    max_node_slot_fanout = max(fanout.values()) if fanout else 0
    if max_node_slot_fanout > limits["max_node_slot_fanout"]:
        busiest = sorted(node_id for node_id, count in fanout.items() if count == max_node_slot_fanout)
        violations.append({
            "metric": "max_node_slot_fanout",
            "value": max_node_slot_fanout,
            "threshold": limits["max_node_slot_fanout"],
            "kind": "soft",
            "reason": "one node carries more slots and text items than a single block should",
            "object_ids": busiest,
            "target_node_ids": busiest,
        })

    hard = [item for item in violations if item["kind"] == "hard"]
    soft = [item for item in violations if item["kind"] == "soft"]
    status = "hard_violations" if hard else "soft_violations" if soft else "decomposable"
    report = {
        "contract": DECOMPOSABILITY_CONTRACT,
        "version": DECOMPOSABILITY_VERSION,
        "round": int(round_index),
        "status": status,
        "metrics": {
            "unbound_scientific_nodes": len(unbound_ids),
            "slot_boundary_overflows": len(overflow_ids),
            "orphan_nodes": len(orphan_ids),
            "cross_panel_connectors": cross_count,
            "min_panel_occupancy": min_panel_occupancy,
            "empty_region_ratio": empty_region_ratio,
            "max_node_slot_fanout": max_node_slot_fanout,
        },
        "thresholds": dict(limits),
        "violations": violations,
        "repeat": {"soft_streak": _soft_streak(_soft_signature(soft), history)},
        "provenance": {
            "editable_graph_id": str((editable_graph or {}).get("graph_id") or ""),
            "visual_graph_id": str((visual_graph or {}).get("graph_id") or ""),
            "panel_count": len(panel_by_id),
            "node_count": len(nodes),
            "slot_count": len(slots),
            "connector_count": len(arrows),
            "panel_source": panel_source,
        },
    }
    return report
