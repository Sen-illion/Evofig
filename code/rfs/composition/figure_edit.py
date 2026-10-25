"""Bounded, deterministic edits over a compiled figure program.

The figure program is the compilation authority for a figure; the PPTX is only
its product. Every command here reads and writes the program and nothing else,
so a hand-made edit can never leave a deck that no longer describes its own
source, and a round that has to undo that edit can do it by moving the program
back and compiling again.

Each command is JSON in / JSON out, is deterministic, and appends one record to
``figure_edits.jsonl`` beside the program. The trail is what makes an edit
checkable after the fact: it names the command, the object it touched and the
fields that moved, in the order they were issued. Nothing here calls a model,
which is what lets the cheap repair layer of a round cost a few seconds of
Python instead of another generation.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..contracts.memory_parameters import layout_rules_from, router_rules_from
from ..utils import read_json, write_json
from .arrow_routing import style_and_route_arrows
from .direct_connectors import apply_memory_routing_constraints, resolve_direct_connectors
from .layout_repairs import apply_layout_constraints
from .pptx import compile_ppt
from .preview import render_rebuild_preview

FIGURE_EDIT_CONTRACT = "rfs.figure_edit"
FIGURE_EDIT_VERSION = 1
FIGURE_EDIT_TRAIL = "figure_edits.jsonl"

#: Every command this surface accepts. The CLI builds its subcommands from this
#: tuple and the tests hold it still, so a command cannot exist in one place
#: without existing in the other.
FIGURE_EDIT_COMMANDS = (
    "show",
    "move-node",
    "resize-node",
    "set-label",
    "reroute-arrow",
    "attach-endpoint",
    "set-waypoints",
    "set-panel-bounds",
    "apply-constraints",
    "recompile",
    "render",
)

#: The commands that write the program back. ``show``, ``recompile`` and
#: ``render`` only read it, and a read that rewrote the program would make the
#: trail a lie about what changed.
MUTATING_COMMANDS = tuple(command for command in FIGURE_EDIT_COMMANDS if command not in {"show", "recompile", "render"})

#: The program keys a command may address by object id, in lookup order.
OBJECT_COLLECTIONS = ("panels", "cards", "slots", "semantic_nodes", "labels", "arrows")

#: A box narrower or shorter than this is not a box; the compiler needs a shape
#: with area, and a zero-area shape is worse than a refused edit.
MIN_BOX_SIDE = 0.005


class FigureEditError(ValueError):
    """A command that cannot be carried out on this program, and why."""


def _unit(value: Any, *, low: float = 0.0, high: float = 1.0) -> float:
    number = float(value)
    if high < low:
        high = low
    return round(min(high, max(low, number)), 4)


def _is_box(value: Any) -> bool:
    return isinstance(value, dict) and all(isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool) for key in ("x", "y", "w", "h"))


def _walk(program: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for key in OBJECT_COLLECTIONS:
        for item in program.get(key) or []:
            if isinstance(item, dict) and item.get("id"):
                yield key[:-1] if key.endswith("s") else key, item
    for item in (program.get("text_program") or {}).get("items") or []:
        if isinstance(item, dict) and item.get("id"):
            yield "text", item


def figure_object_kinds(program: dict[str, Any]) -> dict[str, str]:
    """Every addressable object id and the kind of thing it is."""

    kinds: dict[str, str] = {}
    for kind, item in _walk(program):
        kinds.setdefault(str(item["id"]), kind)
    return kinds


def figure_objects(program: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every addressable object by id, so a command can name one and find it."""

    found: dict[str, dict[str, Any]] = {}
    for _kind, item in _walk(program):
        found.setdefault(str(item["id"]), item)
    return found


def _require_object(program: dict[str, Any], object_id: str) -> dict[str, Any]:
    if not object_id:
        raise FigureEditError("this command needs an object id")
    item = figure_objects(program).get(str(object_id))
    if item is None:
        raise FigureEditError(f"no object named {object_id!r} in this figure program")
    return item


def _require_arrow(program: dict[str, Any], arrow_id: str) -> dict[str, Any]:
    item = _require_object(program, arrow_id)
    if str(item.get("id")) not in {str(arrow.get("id")) for arrow in program.get("arrows") or [] if isinstance(arrow, dict)}:
        raise FigureEditError(f"{arrow_id!r} is not an arrow in this figure program")
    return item


def _require_box(item: dict[str, Any], object_id: str) -> dict[str, Any]:
    box = item.get("bbox_percent")
    if not _is_box(box):
        raise FigureEditError(f"{object_id!r} has no bbox_percent to move")
    return box


def _points(value: Any) -> list[list[float]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise FigureEditError(f"waypoints must be JSON, not {value!r}: {exc}") from exc
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise FigureEditError("waypoints need at least two points")
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise FigureEditError(f"a waypoint is a pair, not {point!r}")
        points.append([_unit(point[0]), _unit(point[1])])
    return points


def figure_summary(program: dict[str, Any], object_id: str | None = None) -> dict[str, Any]:
    """What this program holds, and everything about one object when asked for one."""

    kinds = figure_object_kinds(program)
    objects = figure_objects(program)
    summary: dict[str, Any] = {
        "contract": FIGURE_EDIT_CONTRACT,
        "version": FIGURE_EDIT_VERSION,
        "canvas": dict(program.get("canvas") or {}),
        "counts": {key: len(program.get(key) or []) for key in OBJECT_COLLECTIONS},
        "text_items": len((program.get("text_program") or {}).get("items") or []),
    }
    summary["counts"]["objects"] = len(objects)
    listing = []
    for name in sorted(objects):
        item = objects[name]
        row: dict[str, Any] = {"id": name, "kind": kinds.get(name, "")}
        if _is_box(item.get("bbox_percent")):
            row["bbox_percent"] = dict(item["bbox_percent"])
        text = item.get("text") if isinstance(item.get("text"), str) else item.get("label")
        if isinstance(text, str) and text:
            row["text"] = text
        if kinds.get(name) == "arrow":
            row["source"] = str(item.get("source") or item.get("source_id") or "")
            row["target"] = str(item.get("target") or item.get("target_id") or "")
            path = item.get("path_percent")
            row["point_count"] = len(path) if isinstance(path, list) else 0
        listing.append(row)
    summary["objects"] = listing
    if object_id:
        item = objects.get(str(object_id))
        if item is None:
            raise FigureEditError(f"no object named {object_id!r} in this figure program")
        summary["object"] = {"id": str(object_id), "kind": kinds.get(str(object_id), ""), "record": deepcopy(item)}
    return summary


def move_object(program: dict[str, Any], object_id: str, *, dx: float = 0.0, dy: float = 0.0, x: float | None = None, y: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Move one object's box, by an offset or to a place, inside the canvas."""

    updated = deepcopy(program)
    item = _require_object(updated, object_id)
    box = _require_box(item, object_id)
    before = {"x": float(box["x"]), "y": float(box["y"])}
    target_x = float(box["x"]) if x is None else float(x)
    target_y = float(box["y"]) if y is None else float(y)
    target_x += float(dx or 0.0)
    target_y += float(dy or 0.0)
    box["x"] = _unit(target_x, high=1.0 - float(box["w"]))
    box["y"] = _unit(target_y, high=1.0 - float(box["h"]))
    after = {"x": float(box["x"]), "y": float(box["y"])}
    return updated, {"object_id": str(object_id), "field": "bbox_percent", "before": before, "after": after}


def resize_object(program: dict[str, Any], object_id: str, *, w: float | None = None, h: float | None = None, scale: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resize one object's box, and the label box that belongs to it."""

    updated = deepcopy(program)
    item = _require_object(updated, object_id)
    box = _require_box(item, object_id)
    if scale is not None and float(scale) <= 0.0:
        raise FigureEditError("scale must be greater than zero")
    factor = 1.0 if scale is None else float(scale)
    before = {"w": float(box["w"]), "h": float(box["h"]), "x": float(box["x"]), "y": float(box["y"])}
    box["w"] = _unit((float(box["w"]) if w is None else float(w)) * factor, low=MIN_BOX_SIDE, high=1.0 - float(box["x"]))
    box["h"] = _unit((float(box["h"]) if h is None else float(h)) * factor, low=MIN_BOX_SIDE, high=1.0 - float(box["y"]))
    changed = ["bbox_percent"]
    label_box = item.get("label_bbox_percent")
    if _is_box(label_box):
        grow_x = (float(box["w"]) / before["w"]) if before["w"] else 1.0
        grow_y = (float(box["h"]) / before["h"]) if before["h"] else 1.0
        offset_x = (float(label_box["x"]) - before["x"]) * grow_x
        offset_y = (float(label_box["y"]) - before["y"]) * grow_y
        label_box["w"] = _unit(float(label_box["w"]) * grow_x, low=MIN_BOX_SIDE)
        label_box["h"] = _unit(float(label_box["h"]) * grow_y, low=MIN_BOX_SIDE)
        label_box["x"] = _unit(before["x"] + offset_x, high=1.0 - float(label_box["w"]))
        label_box["y"] = _unit(before["y"] + offset_y, high=1.0 - float(label_box["h"]))
        changed.append("label_bbox_percent")
    after = {"w": float(box["w"]), "h": float(box["h"])}
    return updated, {"object_id": str(object_id), "field": "bbox_percent", "fields": changed, "before": before, "after": after}


def set_label(program: dict[str, Any], object_id: str, text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rewrite the visible text that belongs to one object, everywhere it is kept."""

    updated = deepcopy(program)
    item = _require_object(updated, object_id)
    wanted = str(text)
    changed: list[str] = []
    touched: list[dict[str, Any]] = [item]
    if isinstance(item.get("text"), str):
        item["text"] = wanted
        changed.append(f"{object_id}.text")
    if isinstance(item.get("label"), str):
        item["label"] = wanted
        changed.append(f"{object_id}.label")
    for other in (updated.get("labels") or []):
        if isinstance(other, dict) and str(other.get("target_id") or "") == str(object_id) and isinstance(other.get("text"), str):
            other["text"] = wanted
            changed.append(f"{other.get('id')}.text")
            touched.append(other)
    for block in (updated.get("text_program") or {}).get("items") or []:
        if not isinstance(block, dict):
            continue
        owner = str(block.get("owner_node_id") or block.get("target_id") or block.get("owner_id") or "")
        if owner != str(object_id) or not isinstance(block.get("text"), str):
            continue
        block["text"] = wanted
        changed.append(f"{block.get('id')}.text")
        touched.append(block)
    if not changed:
        raise FigureEditError(f"{object_id!r} carries no editable text to rewrite")
    # A line count measured for the old string would be a measurement of
    # something no longer in the figure, so the blocks that changed lose it.
    # Every block that did not change keeps the measurement it still has.
    dropped: list[str] = []
    for block in touched:
        for key in ("measured_lines", "measured_line_count"):
            if key in block:
                block.pop(key)
                dropped.append(f"{block.get('id') or object_id}.{key}")
    return updated, {"object_id": str(object_id), "field": "text", "text": wanted, "changed": sorted(set(changed)), "dropped": sorted(set(dropped))}


def set_arrow_waypoints(program: dict[str, Any], arrow_id: str, waypoints: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pin an arrow to a stated polyline, in canvas percent."""

    updated = deepcopy(program)
    arrow = _require_arrow(updated, arrow_id)
    points = _points(waypoints)
    before = arrow.get("path_percent")
    arrow["path_percent"] = points
    arrow["route_generation_status"] = "edited_waypoints"
    return updated, {
        "arrow_id": str(arrow_id),
        "field": "path_percent",
        "before_point_count": len(before) if isinstance(before, list) else 0,
        "after_point_count": len(points),
        "waypoints": points,
    }


def attach_endpoint(program: dict[str, Any], arrow_id: str, *, source: str | None = None, target: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-bind an arrow to another source or target object."""

    updated = deepcopy(program)
    arrow = _require_arrow(updated, arrow_id)
    objects = figure_objects(updated)
    if source is None and target is None:
        raise FigureEditError("attach-endpoint needs --source, --target, or both")
    for role, value in (("source", source), ("target", target)):
        if value is not None and str(value) not in objects:
            raise FigureEditError(f"no object named {value!r} to attach as the {role}")
    before = {"source": str(arrow.get("source") or arrow.get("source_id") or ""), "target": str(arrow.get("target") or arrow.get("target_id") or "")}
    if source is not None:
        arrow["source"] = str(source)
        arrow["source_id"] = str(source)
    if target is not None:
        arrow["target"] = str(target)
        arrow["target_id"] = str(target)
    # The old polyline was drawn between the old boxes. Keeping it would draw a
    # line that no longer touches what it claims to connect, so it goes and the
    # compiler falls back to the straight span between the new endpoints until a
    # reroute states a better one.
    dropped_path = isinstance(arrow.get("path_percent"), list)
    arrow.pop("path_percent", None)
    arrow["route_generation_status"] = "endpoints_changed"
    after = {"source": str(arrow.get("source") or ""), "target": str(arrow.get("target") or "")}
    return updated, {"arrow_id": str(arrow_id), "field": "endpoints", "before": before, "after": after, "dropped_path": dropped_path}


def set_panel_bounds(program: dict[str, Any], panel_id: str, *, x: float | None = None, y: float | None = None, w: float | None = None, h: float | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Set a panel's bounds, which is the container the members sit in."""

    updated = deepcopy(program)
    item = _require_object(updated, panel_id)
    kinds = figure_object_kinds(updated)
    if kinds.get(str(panel_id)) != "panel":
        raise FigureEditError(f"{panel_id!r} is not a panel in this figure program")
    box = _require_box(item, panel_id)
    before = {key: float(box[key]) for key in ("x", "y", "w", "h")}
    # A stated edge wins: the panel is placed where it was asked to be, and the
    # sides then take whatever room is left rather than pushing it off the canvas.
    if x is not None:
        box["x"] = _unit(x, high=1.0 - MIN_BOX_SIDE)
    if y is not None:
        box["y"] = _unit(y, high=1.0 - MIN_BOX_SIDE)
    if w is not None:
        box["w"] = _unit(w, low=MIN_BOX_SIDE, high=1.0 - float(box["x"]))
    else:
        box["w"] = _unit(box["w"], low=MIN_BOX_SIDE, high=1.0 - float(box["x"]))
    if h is not None:
        box["h"] = _unit(h, low=MIN_BOX_SIDE, high=1.0 - float(box["y"]))
    else:
        box["h"] = _unit(box["h"], low=MIN_BOX_SIDE, high=1.0 - float(box["y"]))
    after = {key: float(box[key]) for key in ("x", "y", "w", "h")}
    return updated, {"object_id": str(panel_id), "field": "bbox_percent", "before": before, "after": after}


def apply_constraints(program: dict[str, Any], *, layout: dict[str, Any] | None = None, routing: dict[str, Any] | None = None, out_dir: str | Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the compile-time rules a Memory row carries, in the order a run applies them.

    These are the very transforms the pipeline's compile-time layer calls, so a
    rule proved here is proved for the run: a layout rule lands through
    ``apply_layout_constraints``, a router rule through ``style_and_route_arrows``
    carrying that rule, and a stored route selection through
    ``apply_memory_routing_constraints``. Anything a rule needs to write goes to
    ``out_dir``, which is why the router pass asks for one.
    """

    if not layout and not routing:
        raise FigureEditError("apply-constraints needs --layout, --routing, or both")
    updated = deepcopy(program)
    change: dict[str, Any] = {}
    if layout:
        rules = layout_rules_from(layout)
        if not rules:
            raise FigureEditError("--layout carried no rule this compiler can honour; expected align_grid, gutter_min, or panel_occupancy_min")
        updated, layout_report = apply_layout_constraints(updated, rules)
        change["layout"] = layout_report
    if routing:
        router_rules = router_rules_from(routing)
        selections = routing.get("selections") if isinstance(routing.get("selections"), list) else []
        if not router_rules and not selections:
            raise FigureEditError("--routing carried nothing to apply; expected router rule keys or a 'selections' list")
        if router_rules:
            if out_dir is None:
                raise FigureEditError("router rules write a route report, so this command needs an output directory")
            before = {str(item.get("id")): item.get("path_percent") for item in updated.get("arrows") or [] if isinstance(item, dict)}
            updated = style_and_route_arrows(updated, out_dir, mode="reference", repair_routing=True, constraints=router_rules)
            moved = [
                str(item.get("id"))
                for item in updated.get("arrows") or []
                if isinstance(item, dict) and item.get("path_percent") != before.get(str(item.get("id")))
            ]
            change["router_rules"] = {"rules": router_rules, "moved_arrows": moved}
        if selections:
            updated, routing_report = apply_memory_routing_constraints(updated, routing)
            change["routing"] = routing_report
    return updated, change


def reroute_arrow(program: dict[str, Any], arrow_id: str, *, route_style: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-draw the connectors with the same deterministic router the run uses."""

    updated = deepcopy(program)
    arrow = _require_arrow(updated, arrow_id)
    before = {str(item.get("id")): item.get("path_percent") for item in updated.get("arrows") or [] if isinstance(item, dict)}
    if route_style:
        arrow["route_style"] = str(route_style)
    routed, report = resolve_direct_connectors(updated)
    moved = [
        str(item.get("id"))
        for item in routed.get("arrows") or []
        if isinstance(item, dict) and item.get("path_percent") != before.get(str(item.get("id")))
    ]
    return routed, {
        "arrow_id": str(arrow_id),
        "field": "path_percent",
        "route_style": str(route_style or arrow.get("route_style") or ""),
        "moved_arrows": moved,
        "router_status": str(report.get("status") or ""),
        "router_report": {key: value for key, value in report.items() if key != "connectors"},
    }


def _trail_record(command: str, arguments: dict[str, Any], changed: Any, *, ok: bool = True, error: str = "") -> dict[str, Any]:
    return {
        "contract": FIGURE_EDIT_CONTRACT,
        "version": FIGURE_EDIT_VERSION,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "arguments": arguments,
        "changed": changed,
        "ok": bool(ok),
        "error": error or None,
    }


def append_edit_trail(program_path: str | Path, record: dict[str, Any]) -> Path:
    """Append one edit to the trail beside the program, as one JSON line."""

    path = Path(program_path).parent / FIGURE_EDIT_TRAIL
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_edit_trail(program_path: str | Path) -> list[dict[str, Any]]:
    """Read back every edit made to this program, in the order they were made."""

    path = Path(program_path).parent / FIGURE_EDIT_TRAIL
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def run_figure_edit(command: str, *, program_path: str | Path, out_dir: str | Path | None = None, **options: Any) -> dict[str, Any]:
    """Carry out one command against one program and answer in JSON.

    The program is written back only by the commands that change it, and only
    after the change succeeded, so a refused command leaves the figure exactly
    as it found it.
    """

    if command not in FIGURE_EDIT_COMMANDS:
        raise FigureEditError(f"unknown figure-edit command {command!r}; expected one of {', '.join(FIGURE_EDIT_COMMANDS)}")
    path = Path(program_path)
    if not path.is_file():
        raise FigureEditError(f"no figure program at {path}")
    program = read_json(path)
    if not isinstance(program, dict):
        raise FigureEditError(f"{path} does not hold a figure program object")
    target = Path(out_dir) if out_dir else path.parent
    target.mkdir(parents=True, exist_ok=True)
    trail = path.parent / FIGURE_EDIT_TRAIL

    if command == "show":
        summary = figure_summary(program, options.get("object_id"))
        shown = write_json(target / "figure_show.json", summary)
        return {"ok": True, "summary": f"Read {summary['counts'].get('objects', 0)} objects from the figure program.", "command": command, "program": str(path), "out_dir": str(target), "figure_show": str(shown), "figure": summary, "trail": str(trail)}

    if command == "recompile":
        pptx = compile_ppt(program, target)
        report = {"contract": FIGURE_EDIT_CONTRACT, "version": FIGURE_EDIT_VERSION, "command": command, "program": str(path), "pptx": str(pptx), "nodes": len(program.get("semantic_nodes") or []), "arrows": len(program.get("arrows") or [])}
        written = write_json(target / "figure_compile_report.json", report)
        append_edit_trail(path, _trail_record(command, {"out_dir": str(target)}, {"pptx": str(pptx)}))
        return {"ok": True, "summary": f"Compiled the figure program to PowerPoint without a model call.", "command": command, "program": str(path), "out_dir": str(target), "pptx": str(pptx), "compile_report": str(written), "trail": str(trail)}

    if command == "render":
        preview_path = target / "figure_preview.png"
        preview = render_rebuild_preview(program, target, preview_path)
        report = {"contract": FIGURE_EDIT_CONTRACT, "version": FIGURE_EDIT_VERSION, "command": command, "program": str(path), "preview": str(preview)}
        written = write_json(target / "figure_render_report.json", report)
        append_edit_trail(path, _trail_record(command, {"out_dir": str(target)}, {"preview": str(preview)}))
        return {"ok": True, "summary": "Rendered the figure program without a model call.", "command": command, "program": str(path), "out_dir": str(target), "preview": str(preview), "render_report": str(written), "trail": str(trail)}

    object_id = str(options.get("object_id") or "")
    if command == "move-node":
        updated, change = move_object(program, object_id, dx=float(options.get("dx") or 0.0), dy=float(options.get("dy") or 0.0), x=options.get("x"), y=options.get("y"))
    elif command == "resize-node":
        updated, change = resize_object(program, object_id, w=options.get("w"), h=options.get("h"), scale=options.get("scale"))
    elif command == "set-label":
        updated, change = set_label(program, object_id, str(options.get("text") or ""))
    elif command == "set-waypoints":
        updated, change = set_arrow_waypoints(program, str(options.get("arrow_id") or object_id), options.get("waypoints"))
    elif command == "attach-endpoint":
        updated, change = attach_endpoint(program, str(options.get("arrow_id") or object_id), source=options.get("source"), target=options.get("target"))
    elif command == "reroute-arrow":
        updated, change = reroute_arrow(program, str(options.get("arrow_id") or object_id), route_style=options.get("route_style"))
    elif command == "set-panel-bounds":
        updated, change = set_panel_bounds(program, object_id, x=options.get("x"), y=options.get("y"), w=options.get("w"), h=options.get("h"))
    else:
        updated, change = apply_constraints(program, layout=options.get("layout"), routing=options.get("routing"), out_dir=target)

    write_json(path, updated)
    append_edit_trail(path, _trail_record(command, {key: value for key, value in options.items() if value is not None}, change))
    return {
        "ok": True,
        "summary": f"{command} applied to {object_id or options.get('arrow_id') or 'the figure program'}; the program is the only thing that changed.",
        "command": command,
        "program": str(path),
        "out_dir": str(target),
        "change": change,
        "edits_recorded": len(read_edit_trail(path)),
        "trail": str(trail),
    }


__all__ = [
    "FIGURE_EDIT_COMMANDS",
    "FIGURE_EDIT_CONTRACT",
    "FIGURE_EDIT_TRAIL",
    "FIGURE_EDIT_VERSION",
    "MUTATING_COMMANDS",
    "OBJECT_COLLECTIONS",
    "FigureEditError",
    "append_edit_trail",
    "apply_constraints",
    "attach_endpoint",
    "figure_object_kinds",
    "figure_objects",
    "figure_summary",
    "move_object",
    "read_edit_trail",
    "reroute_arrow",
    "resize_object",
    "run_figure_edit",
    "set_arrow_waypoints",
    "set_label",
    "set_panel_bounds",
]
