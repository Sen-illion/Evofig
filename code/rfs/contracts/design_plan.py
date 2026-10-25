"""Contract for VLM-authored semantic figure design plans."""

from __future__ import annotations

import math
from typing import Any

#: The response a run actually built its figure from. It is the model's first answer unless a plan
#: repair replaced it, and it is the artifact a replay has to start from to reproduce that figure.
DESIGN_PLAN_SEED_FILE = "design_plan_seed.json"

#: Field names the design model uses for the same four rectangle numbers.
_BBOX_ALIASES = {
    "x": ("x", "left", "x1", "xmin", "x_min"),
    "y": ("y", "top", "y1", "ymin", "y_min"),
    "w": ("w", "width", "x2", "xmax", "x_max"),
    "h": ("h", "height", "y2", "ymax", "y_max"),
}


def _bbox_number(value: Any) -> float | None:
    """Read one rectangle number, or ``None`` when the model wrote no number.

    A box the model left half-drawn used to reach the safe-area fitter and raise
    ``KeyError: 'h'``, which graded the attempt a failure and then spent the
    paper's whole retry budget on the same crash. A field with no readable number
    is now reported as absent instead of guessed at, so the design validator can
    still call the box illegal and the repair pass has something specific to ask
    for.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _normalize_bbox(value: Any) -> Any:
    """Return a bbox whose readable numbers are numbers and whose absent ones stay absent.

    The four numbers may arrive as strings, under the model's own field names, or
    as a corner list. Whatever cannot be read is left out rather than invented.
    """
    if isinstance(value, dict):
        resolved: dict[str, Any] = {}
        for target, names in _BBOX_ALIASES.items():
            for name in names:
                if name in value:
                    resolved.setdefault(target, value[name])
                    break
        box: dict[str, float] = {}
        for key in ("x", "y"):
            number = _bbox_number(resolved.get(key))
            if number is not None:
                box[key] = number
        for key, near in (("w", "x"), ("h", "y")):
            number = _bbox_number(resolved.get(key))
            if number is None:
                continue
            if any(name in value for name in _BBOX_ALIASES[key][2:]):
                origin = box.get(near)
                if origin is None:
                    continue
                number = number - origin
                if number <= 0:
                    continue
            box[key] = number
        return box
    if isinstance(value, (list, tuple)) and len(value) == 4:
        x1, y1, x2, y2 = (_bbox_number(part) for part in value)
        if None in (x1, y1, x2, y2):
            return {}
        return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
    return value


def _normalize_objects(values: Any) -> list[dict[str, Any]]:
    return [
        {**item, "bbox_percent": _normalize_bbox(item.get("bbox_percent"))}
        for item in values if isinstance(item, dict)
    ] if isinstance(values, list) else []


def _node_center_inside(node: dict[str, Any], panel: dict[str, Any]) -> bool:
    """Whether a node's centre sits inside a panel's box, which is how the contract reads one."""

    bbox = node.get("bbox_percent")
    panel_bbox = panel.get("bbox_percent")
    if not _valid_bbox(bbox) or not _valid_bbox(panel_bbox):
        return False
    center_x = float(bbox["x"]) + float(bbox["w"]) / 2
    center_y = float(bbox["y"]) + float(bbox["h"]) / 2
    left = float(panel_bbox["x"])
    top = float(panel_bbox["y"])
    return left <= center_x <= left + float(panel_bbox["w"]) and top <= center_y <= top + float(panel_bbox["h"])


def _read_parent_references(
    nodes: list[dict[str, Any]], panels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Read each node's ``parent_id`` for the reference it actually names.

    The design model writes a panel's name into ``parent_id`` when it means "this card belongs to
    that panel", and the owning node's id when it means "this card is nested inside that card".
    Only the second is node hierarchy, and a name that resolves to nothing declared is a dangling
    pointer the contract drops rather than carries any further.

    A panel's name is believed only where the box agrees with it: the card is read as a member of
    that panel when its centre sits inside the panel, and the note is dropped as stale when it does
    not. Reading a contradicted note as membership instead would hand the card a ``panel_id`` it was
    never drawn with, which moves it into that panel's payload and changes a figure the source arm
    had already compiled the other way.

    Any of these left in place is enough to leave a finished plan permanently illegal -- the
    validator refuses a node with a ``parent_id`` as a canvas-level card -- so one half-written
    membership line marked the whole figure a failure and every replay of it a stage the contract
    had outgrown.

    A card the model left inside no panel *and* inside no declared parent is not repaired here:
    that is a real orphan, the contract still calls it one, and the designer is asked for it.
    Every repair is returned so a plan that arrived half-written is never mistaken for a clean one.
    """

    node_ids = {str(node.get("id")) for node in nodes if node.get("id")}
    panels_by_id = {str(panel.get("id")): panel for panel in panels if panel.get("id")}
    repairs: list[dict[str, Any]] = []
    for node in nodes:
        parent_id = str(node.get("parent_id") or "").strip()
        if not parent_id or parent_id in node_ids:
            continue
        node_id = str(node.get("id") or "")
        panel_id = str(node.get("panel_id") or "").strip()
        if parent_id in panels_by_id:
            if panel_id in panels_by_id:
                # The node is already a member of a declared panel, and the validator never reads
                # ``parent_id`` for one: it is the model's own note of the owning container, and the
                # layout solver reads it as that owner, so it is left exactly as written.
                continue
            if _node_center_inside(node, panels_by_id[parent_id]):
                node["panel_id"] = parent_id
                repairs.append({"node_id": node_id, "field": "panel_id", "value": parent_id, "read_as": "panel_membership", "parent_id": parent_id})
                continue
            node["parent_id"] = None
            repairs.append({"node_id": node_id, "field": "parent_id", "value": parent_id, "read_as": "dropped_dangling_parent", "panel_id": parent_id})
            continue
        node["parent_id"] = None
        repairs.append({"node_id": node_id, "field": "parent_id", "value": parent_id, "read_as": "dropped_undeclared_parent"})
    return repairs


def normalize_design_plan(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    src = raw if isinstance(raw, dict) else {}
    panels = _normalize_objects(src.get("panels", []))
    nodes = _normalize_objects(src.get("nodes", []))
    provenance = src.get("provenance") if isinstance(src.get("provenance"), dict) else {}
    parent_repairs = _read_parent_references(nodes, panels)
    if parent_repairs:
        provenance = {**provenance, "repairs": [*(provenance.get("repairs") or []), *parent_repairs]}
    return {
        "contract": "rfs.design_plan",
        "version": 1,
        "canvas": (
            src.get("canvas")
            if isinstance(src.get("canvas"), dict)
            else {"aspect_ratio": "16:9", "margin": 0.04}
        ),
        "topology": str(src.get("topology") or "unknown"),
        "panels": panels,
        "nodes": nodes,
        "connectors": [x for x in src.get("connectors", []) if isinstance(x, dict)],
        "groups": _normalize_objects(src.get("groups", [])),
        "reference_adoption": (
            src.get("reference_adoption")
            if isinstance(src.get("reference_adoption"), dict)
            else {"adopted": [], "rejected": []}
        ),
        "density_targets": (
            src.get("density_targets")
            if isinstance(src.get("density_targets"), dict)
            else {}
        ),
        "provenance": provenance,
    }


#: Method-object fields that can carry content the figure has to show. The order
#: is the contract's own: what goes in and comes out first, then what it does.
_METHOD_OBJECT_FIELDS = ("inputs", "outputs", "modules", "stages", "innovations", "optional_details")


def _coverage_key(value: Any) -> str:
    """Fold a label or an id into the form two spellings of one name share."""
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum())


def method_object_index(method: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Every id-carrying object in the method contract, id -> what the contract says of it.

    The coverage check needs the objects a figure has to show; the pass that draws the
    ones a design left out needs to know which field each came from, what it is called,
    and which evidence stands behind it. Both read this one index, so an object a figure
    is missing is named the way the contract names it rather than the way a coverage
    report happened to print it.
    """

    src = method if isinstance(method, dict) else {}
    index: dict[str, dict[str, Any]] = {}
    for field in _METHOD_OBJECT_FIELDS:
        rows = src.get(field)
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("id") or "").strip()
            if not object_id or object_id in index:
                continue
            index[object_id] = {
                "id": object_id,
                "field": field,
                "label": str(item.get("label") or item.get("name") or object_id).strip(),
                "role": str(item.get("role") or field.rstrip("s") or "module"),
                "evidence_ids": [str(value) for value in (item.get("evidence_ids") or []) if value],
                "required": bool(item.get("must_show") or item.get("should_render")),
            }
    return index


def required_method_object_ids(method: dict[str, Any] | None) -> dict[str, str]:
    """Return every method object the figure has to show, id -> label.

    The reader of the method contract marks an object ``must_show`` when the
    paper evidences it and ``should_render`` when it belongs on the canvas; an
    object it leaves out of both is a handoff the figure may omit. Only an
    object that carries an id can be checked against a design, so an id-less row
    stays out of the coverage denominator instead of counting as missing.
    """

    return {
        object_id: str(info["label"])
        for object_id, info in method_object_index(method).items()
        if info["required"]
    }
def design_plan_coverage(plan: dict[str, Any] | None, method: dict[str, Any] | None) -> dict[str, Any]:
    """Measure how much of the required method content a design plan actually draws.

    A required object counts as covered when some node names it, by its id, by
    the id the node was bound to (``semantic_source_id``) or by spelling its
    label. The number is a measurement: a design that draws 9 of 38 required
    objects is not wrong about any of the 9, it is missing 29, and only the
    caller decides what a shortfall means.
    """
    required = required_method_object_ids(method)
    normalized = normalize_design_plan(plan)
    covered_keys: set[str] = set()
    for node in normalized["nodes"]:
        for value in (node.get("semantic_source_id"), node.get("id"), node.get("label")):
            key = _coverage_key(value)
            if key:
                covered_keys.add(key)
    missing_ids: list[str] = []
    missing_labels: list[str] = []
    for object_id, label in required.items():
        if _coverage_key(object_id) in covered_keys or (label and _coverage_key(label) in covered_keys):
            continue
        missing_ids.append(object_id)
        missing_labels.append(label or object_id)
    total = len(required)
    covered = total - len(missing_ids)
    return {
        "required_count": total,
        "covered_count": covered,
        "missing_ids": missing_ids,
        "missing_labels": missing_labels,
        "coverage": (covered / total) if total else 1.0,
    }


def validate_design_plan(
    plan: dict[str, Any],
    method: dict[str, Any] | None = None,
    *,
    require_method_coverage: bool = True,
) -> dict[str, Any]:
    """Check a design plan, and against the method contract when it is given.

    Geometry and reference rules are the plan's own contract. When a method
    contract is handed in the plan is also measured against it: a design that
    leaves out an object the method marks must_show or should_render is not
    illegal geometry, it is content the figure will not carry, and the caller
    that asked for coverage hears it in ``errors`` so the repair round can add
    exactly those objects back.
    """

    p = normalize_design_plan(plan)
    errors = []
    ids = {str(n.get("id")) for n in p["nodes"] if n.get("id")}
    panel_ids = {str(panel.get("id")) for panel in p["panels"] if panel.get("id")}
    if not p["nodes"]:
        errors.append("design plan requires at least one node")
    if len(p["panels"]) < 2:
        errors.append("design plan requires at least two panels")
    for i, n in enumerate(p["nodes"]):
        if not n.get("id") or not n.get("semantic_source_id"):
            errors.append(f"node[{i}] requires id and semantic_source_id")
        if not n.get("panel_id"):
            if not _valid_root_level_node(n, p["panels"]):
                errors.append(
                    f"node[{i}] without panel_id must be a root-level node outside every panel"
                )
        elif str(n.get("panel_id")) not in panel_ids:
            errors.append(f"node[{i}] references unknown panel")
        bbox = n.get("bbox_percent")
        if not _valid_bbox(bbox):
            errors.append(f"node[{i}] requires a legal bbox_percent")
    for i, panel in enumerate(p["panels"]):
        if not panel.get("id") or not _valid_bbox(panel.get("bbox_percent")):
            errors.append(f"panel[{i}] requires id and legal bbox_percent")
    for i, e in enumerate(p["connectors"]):
        if str(e.get("source")) not in ids or str(e.get("target")) not in ids:
            errors.append(f"connector[{i}] references unknown node")
    coverage = design_plan_coverage(p, method) if method is not None else None
    if coverage and require_method_coverage and coverage["required_count"] and coverage["coverage"] < 1.0:
        errors.append(
            "design plan represents "
            + str(coverage["covered_count"])
            + " of "
            + str(coverage["required_count"])
            + " required method objects; missing: "
            + ", ".join(coverage["missing_labels"][:12])
        )
    return {
        "ok": not errors,
        "errors": errors,
        "method_coverage": coverage,
        "panel_count": len(p["panels"]),
        "node_count": len(p["nodes"]),
        "connector_count": len(p["connectors"]),
    }


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, dict) or any(key not in value for key in ("x", "y", "w", "h")):
        return False
    try:
        x, y, width, height = (float(value[key]) for key in ("x", "y", "w", "h"))
    except (TypeError, ValueError):
        return False
    return 0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0 and x + width <= 1.001 and y + height <= 1.001


def _valid_root_level_node(node: dict[str, Any], panels: list[dict[str, Any]]) -> bool:
    """Allow intentional canvas-level nodes without accepting panel orphans."""
    if str(node.get("parent_id") or "").strip():
        return False
    bbox = node.get("bbox_percent")
    if not _valid_bbox(bbox):
        return False
    center_x = float(bbox["x"]) + float(bbox["w"]) / 2
    center_y = float(bbox["y"]) + float(bbox["h"]) / 2
    for panel in panels:
        panel_bbox = panel.get("bbox_percent")
        if not _valid_bbox(panel_bbox):
            continue
        left = float(panel_bbox["x"])
        top = float(panel_bbox["y"])
        right = left + float(panel_bbox["w"])
        bottom = top + float(panel_bbox["h"])
        if left <= center_x <= right and top <= center_y <= bottom:
            return False
    return True
