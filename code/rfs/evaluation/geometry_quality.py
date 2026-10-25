"""Geometry and occupancy checks for editable figure programs.

Two names are fixed here and used everywhere else, because "crossing" was doing two jobs.

A *crossing* is two connector segments crossing each other. That number is measured once, by
``rfs.composition.arrow_routing`` (``count_connector_crossings``), from the drawn paths alone, and
it is not computed in this module.

What this module measures is a connector segment passing through a box -- a node it does not
belong to, or a piece of text -- which is a different thing a reader sees. Those are reported as
*box passes*, under their own names, so a reader never has to guess which of the two a number is.
"""

from __future__ import annotations

from typing import Any


#: A box touching the title band by a hair is a rounding artifact; a box that
#: takes a visible bite out of it is a defect worth reporting.
GROSS_TITLE_OVERLAP_AREA = 0.002

#: How far a box may poke out of the node that owns it before the overhang is a break a
#: reader can see, measured against the box it left. Two boxes meant to share an edge differ
#: by rounding; the overhangs being counted as defects were half a percent to one percent of
#: a node, which nobody looking at the figure can tell from an exact fit. A nudge inside the
#: tolerance is recorded, reported, and does not block the figure.
BOUNDARY_TOLERANCE_RATIO = 0.01


def _area(box: dict[str, Any]) -> float:
    return max(0.0, float(box.get("w", 0))) * max(0.0, float(box.get("h", 0)))


def _intersection(a: dict[str, Any], b: dict[str, Any]) -> float:
    width = max(0.0, min(float(a["x"]) + float(a["w"]), float(b["x"]) + float(b["w"])) - max(float(a["x"]), float(b["x"])))
    height = max(0.0, min(float(a["y"]) + float(a["h"]), float(b["y"]) + float(b["h"])) - max(float(a["y"]), float(b["y"])))
    return width * height


def _contains(outer: dict[str, Any], inner: dict[str, Any]) -> bool:
    return float(inner["x"]) >= float(outer["x"]) and float(inner["y"]) >= float(outer["y"]) and float(inner["x"]) + float(inner["w"]) <= float(outer["x"]) + float(outer["w"]) + 1e-6 and float(inner["y"]) + float(inner["h"]) <= float(outer["y"]) + float(outer["h"]) + 1e-6


def _overhang_ratio(outer: dict[str, Any], inner: dict[str, Any]) -> float:
    """How far ``inner`` pokes out of ``outer``, as a share of the box it leaves."""

    width = max(float(outer.get("w", 0.0)), 1e-9)
    height = max(float(outer.get("h", 0.0)), 1e-9)
    left = (float(outer["x"]) - float(inner["x"])) / width
    top = (float(outer["y"]) - float(inner["y"])) / height
    right = (float(inner["x"]) + float(inner["w"]) - float(outer["x"]) - float(outer["w"])) / width
    bottom = (float(inner["y"]) + float(inner["h"]) - float(outer["y"]) - float(outer["h"])) / height
    return max(0.0, left, top, right, bottom)


def _outside_area(outer: dict[str, Any], inner: dict[str, Any]) -> float:
    """How much of ``inner`` sticks out of ``outer``."""
    return max(0.0, _area(inner) - _intersection(outer, inner))


def _segment_hits_box(a: list[float], b: list[float], box: dict[str, Any]) -> bool:
    min_x, max_x = float(box["x"]), float(box["x"]) + float(box["w"])
    min_y, max_y = float(box["y"]), float(box["y"]) + float(box["h"])
    for step in range(21):
        t = step / 20.0
        x = float(a[0]) + (float(b[0]) - float(a[0])) * t
        y = float(a[1]) + (float(b[1]) - float(a[1])) * t
        if min_x < x < max_x and min_y < y < max_y:
            return True
    return False


def evaluate_geometry_quality(program: dict[str, Any]) -> dict[str, Any]:
    """Report Icon/text overlap, containment, and coarse content occupancy."""

    nodes = {str(item.get("id")): item for item in program.get("semantic_nodes", []) or [] if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)}
    slots = [item for item in program.get("slots", []) or [] if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)]
    overlap_records, boundary_records, node_overlap_records, connector_overlap_records, box_pass_records, hierarchy_containment_records = [], [], [], [], [], []
    # Overhangs inside the tolerance: measured, recorded, reported, and not a break.
    boundary_nudges: list[dict[str, Any]] = []
    node_list = list(nodes.values())
    for index, left in enumerate(node_list):
        for right in node_list[index + 1:]:
            if str(left.get("panel_id")) != str(right.get("panel_id")):
                continue
            amount = _intersection(left["bbox_percent"], right["bbox_percent"])
            if amount > 1e-7:
                parent, child = (left, right) if str(right.get("parent_id") or "") == str(left.get("id") or "") else ((right, left) if str(left.get("parent_id") or "") == str(right.get("id") or "") else (None, None))
                parent_is_container = parent and (
                    str(parent.get("render_policy") or "").casefold() in {"panel_only", "ppt_native_container"}
                    or str(parent.get("object_type") or "").casefold() == "container_anchor"
                )
                if parent_is_container:
                    # A declared parent and child are never two cards colliding,
                    # however far the child pokes out of its container: the
                    # overlap is the nesting itself. Only children that sit
                    # fully inside keep a zero overhang, and a container that is
                    # smaller than its payload is reported as an overhang to fix
                    # rather than as a gross collision that voids the figure.
                    hierarchy_containment_records.append({"parent_node_id": parent.get("id"), "child_node_id": child.get("id"), "contained_area": round(amount, 6), "outside_area": round(_outside_area(parent["bbox_percent"], child["bbox_percent"]), 6)})
                    continue
                node_overlap_records.append({"node_a": left.get("id"), "node_b": right.get("id"), "overlap_area": round(amount, 6)})
    for slot in slots:
        node_id = str(slot.get("parent_id") or str(slot.get("id") or "").removeprefix("slot_"))
        node = nodes.get(node_id)
        if not node:
            boundary_records.append({"slot_id": slot.get("id"), "problem": "missing_parent_node", "severity": "blocker"})
            continue
        icon_box = slot["bbox_percent"]
        label_box = node.get("label_bbox_percent")
        if isinstance(label_box, dict):
            overlap = _intersection(icon_box, label_box)
            if overlap > 1e-7:
                overlap_records.append({"slot_id": slot.get("id"), "node_id": node_id, "overlap_area": round(overlap, 6)})
        if not _contains(node["bbox_percent"], icon_box):
            overhang = _overhang_ratio(node["bbox_percent"], icon_box)
            record = {"slot_id": slot.get("id"), "node_id": node_id, "problem": "icon_outside_node", "overhang_ratio": round(overhang, 6), "severity": "minor"}
            (boundary_nudges if overhang <= BOUNDARY_TOLERANCE_RATIO else boundary_records).append(record)
        if isinstance(label_box, dict) and not _contains(node["bbox_percent"], label_box):
            overhang = _overhang_ratio(node["bbox_percent"], label_box)
            record = {"slot_id": slot.get("id"), "node_id": node_id, "problem": "label_outside_node", "overhang_ratio": round(overhang, 6), "severity": "major"}
            (boundary_nudges if overhang <= BOUNDARY_TOLERANCE_RATIO else boundary_records).append(record)
    connector_labels = [item for item in (program.get("text_program") or {}).get("items", []) if isinstance(item, dict) and item.get("visible", True) and str(item.get("role") or item.get("text_role")) == "connector_label" and isinstance(item.get("bbox_percent"), dict)]
    for label in connector_labels:
        for node in node_list:
            amount = _intersection(label["bbox_percent"], node["bbox_percent"])
            if amount > 1e-7:
                connector_overlap_records.append({"label_id": label.get("id"), "node_id": node.get("id"), "overlap_area": round(amount, 6)})
    # A line drawn across editable text is the defect a reader notices first,
    # so it is measured separately from lines crossing node boxes. This is
    # reported, not gated: routing already avoids these boxes, and the count
    # shows whether that avoidance actually held.
    title_block = program.get("title_block") if isinstance(program.get("title_block"), dict) else {}
    # A panel or card drawn across the title is the one overlap no other gate
    # sees: the cards still sit inside their panel, and the title is not an
    # object those gates know about. Measured here so a regression shows up in
    # the reports instead of only in the picture.
    title_overlap_records = []
    if isinstance(title_block.get("bbox_percent"), dict):
        for object_type, items in (("panel", program.get("panels", []) or []), ("node", node_list)):
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("bbox_percent"), dict):
                    continue
                amount = _intersection(item["bbox_percent"], title_block["bbox_percent"])
                if amount > 1e-7:
                    title_overlap_records.append({"object_type": object_type, "object_id": str(item.get("id") or ""), "overlap_area": round(amount, 6)})
    text_obstacles = [
        item for item in (program.get("text_program") or {}).get("items", [])
        if isinstance(item, dict) and item.get("visible", True) and isinstance(item.get("bbox_percent"), dict)
        and str(item.get("role") or item.get("text_role")) in {"panel_title", "connector_label"}
    ]
    if isinstance(title_block.get("bbox_percent"), dict):
        text_obstacles.append({"id": "title_block", "role": "paper_title", "bbox_percent": title_block["bbox_percent"]})
    text_pass_records = []
    for arrow in program.get("arrows", []) or []:
        if not isinstance(arrow, dict) or not isinstance(arrow.get("path_percent"), list) or len(arrow["path_percent"]) < 2:
            continue
        points = arrow["path_percent"]
        own_text_id = str(arrow.get("id") or "")
        for item in text_obstacles:
            if str(item.get("connector_id") or item.get("target_id") or "") == own_text_id:
                continue
            if any(_segment_hits_box(a, b, item["bbox_percent"]) for a, b in zip(points[:-1], points[1:])):
                text_pass_records.append({"connector_id": arrow.get("id"), "text_id": item.get("id"), "text_role": str(item.get("role") or item.get("text_role"))})
    for arrow in program.get("arrows", []) or []:
        if not isinstance(arrow, dict) or not isinstance(arrow.get("path_percent"), list):
            continue
        source_id, target_id = str(arrow.get("source") or arrow.get("source_id") or ""), str(arrow.get("target") or arrow.get("target_id") or "")
        points = arrow["path_percent"]
        for node_id, node in nodes.items():
            if node_id in {source_id, target_id}:
                continue
            if any(_segment_hits_box(a, b, node["bbox_percent"]) for a, b in zip(points[:-1], points[1:])):
                box_pass_records.append({"connector_id": arrow.get("id"), "node_id": node_id})
    occupied = min(1.0, sum(_area(item["bbox_percent"]) for item in nodes.values()))
    direct_mode = any(str(arrow.get("routing_algorithm") or "").startswith("direct-final") for arrow in (program.get("arrows", []) or []) if isinstance(arrow, dict))
    gross_title_overlap = any(item["overlap_area"] > GROSS_TITLE_OVERLAP_AREA for item in title_overlap_records)
    # What a reader can see: a real collision, a line drawn through text or a card, a box
    # taking a visible bite out of the title, and an overhang past the tolerance. A slot whose
    # icon overhangs its node by less than the tolerance is a nudge, and a figure whose only
    # finding is a nudge reads as readable.
    visible = (
        len(overlap_records)
        + len(node_overlap_records)
        + len(connector_overlap_records)
        + (len(box_pass_records) if direct_mode else 0)
        + (1 if gross_title_overlap else 0)
        + sum(1 for item in boundary_records if str(item.get("severity") or "") in {"blocker", "major"})
    )
    status = "pass" if not visible else "blocked"
    return {"summary": "Editable figure geometry quality report.", "status": status, "icon_text_overlap_count": len(overlap_records), "node_overlap_count": len(node_overlap_records), "hierarchy_containment_count": len(hierarchy_containment_records), "connector_label_overlap_count": len(connector_overlap_records), "text_pass_count": len(text_pass_records), "box_pass_count": len(box_pass_records), "node_boundary_violation_count": len(boundary_records), "title_overlap_count": len(title_overlap_records), "canvas_occupancy": round(occupied, 4), "empty_region_ratio": round(max(0.0, 1.0 - occupied), 4), "icon_text_overlaps": overlap_records, "node_overlaps": node_overlap_records, "hierarchy_containments": hierarchy_containment_records, "connector_label_overlaps": connector_overlap_records, "text_passes": text_pass_records, "box_passes": box_pass_records, "boundary_violations": boundary_records, "boundary_nudges": boundary_nudges, "boundary_nudge_count": len(boundary_nudges), "visible_break_count": visible, "title_overlaps": title_overlap_records}


def evaluate_density_quality(program: dict[str, Any], *, minimum_occupancy: float = 0.45, maximum_empty_ratio: float = 0.55) -> dict[str, Any]:
    """Report coarse node/panel occupancy without inventing semantic objects."""

    targets = program.get("density_targets") if isinstance(program.get("density_targets"), dict) else {}
    try:
        minimum_occupancy = float(targets.get("panel_occupancy_min") or targets.get("minimum_occupancy") or minimum_occupancy)
    except (TypeError, ValueError):
        pass
    try:
        maximum_empty_ratio = float(targets.get("max_empty_region_ratio") or maximum_empty_ratio)
    except (TypeError, ValueError):
        pass
    nodes = [item for item in program.get("semantic_nodes", []) or [] if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)]
    panels = [item for item in program.get("panels", []) or [] if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)]
    node_area = sum(_area(item["bbox_percent"]) for item in nodes)
    slots = [item for item in program.get("slots", []) or [] if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict)]
    text_items = [item for item in (program.get("text_program") or {}).get("items", []) if isinstance(item, dict) and item.get("visible", True) and isinstance(item.get("bbox_percent"), dict)]
    icon_area = sum(_area(item["bbox_percent"]) for item in slots)
    text_area = sum(_area(item["bbox_percent"]) for item in text_items)
    panel_reports = []
    sparse = []
    for panel in panels:
        box = panel["bbox_percent"]
        contained = sum(_intersection(box, node["bbox_percent"]) for node in nodes)
        occupancy = contained / max(_area(box), 1e-9)
        report = {"panel_id": panel.get("id"), "occupancy": round(occupancy, 4), "node_count": sum(_intersection(box, node["bbox_percent"]) > 0 for node in nodes)}
        panel_reports.append(report)
        if occupancy < minimum_occupancy:
            sparse.append(report)
    # Use a conservative union approximation: overlapping child regions should
    # not inflate density, while editable text/icons still count toward use.
    content_area = min(1.0, max(node_area, sum(_area(item["bbox_percent"]) for item in panels) * 0.05) + min(icon_area + text_area, 0.35))
    empty_ratio = max(0.0, 1.0 - content_area)
    status = "pass" if empty_ratio <= maximum_empty_ratio and not sparse else "blocked"
    return {"summary": "Editable figure density quality report.", "status": status, "canvas_occupancy": round(content_area, 4), "node_area": round(node_area, 4), "icon_area": round(icon_area, 4), "editable_text_area": round(text_area, 4), "empty_region_ratio": round(empty_ratio, 4), "panel_count": len(panel_reports), "sparse_panel_count": len(sparse), "panels": panel_reports, "sparse_panels": sparse, "targets": {"minimum_occupancy": minimum_occupancy, "maximum_empty_ratio": maximum_empty_ratio}}
