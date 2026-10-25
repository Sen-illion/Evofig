"""Registration helper for the bounded figure-program edit surface.

The command group exists so a repair can be issued, in JSON, against the figure
program -- the thing the deck is compiled from -- instead of against the deck.
Registering the vocabulary here keeps ``rfs/cli.py`` a dispatcher: it builds the
parser, hands the parsed namespace to :func:`figure_edit_invocation`, and calls
the composition layer with what comes back.
"""

from __future__ import annotations

import json


def register_figure_edit(subparsers) -> None:
    parser = subparsers.add_parser(
        "figure-edit",
        help="Edit a compiled figure program by command, then recompile or render it. The PPTX is never edited directly.",
        description=(
            "Every command reads and writes figure_program.json and appends one line to "
            "figure_edits.jsonl beside it. The deck is only ever produced by compiling the "
            "program again, so a deck can never drift away from the program it came from."
        ),
    )
    parser.add_argument("--program", required=True, help="Path to figure_program.json.")
    parser.add_argument("--out", help="Directory for outputs and the edit trail. Defaults to the program's own directory.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    actions = parser.add_subparsers(dest="figure_action", required=True)

    show = actions.add_parser("show", help="List what the program holds, or everything about one object.")
    show.add_argument("--object", dest="object_id", help="Object id to describe in full.")

    move = actions.add_parser("move-node", help="Move one object's box, by an offset or to a place, in canvas percent.")
    move.add_argument("--object", dest="object_id", required=True)
    move.add_argument("--dx", type=float, default=0.0, help="Horizontal offset, in canvas percent.")
    move.add_argument("--dy", type=float, default=0.0, help="Vertical offset, in canvas percent.")
    move.add_argument("--x", type=float, help="Absolute left edge, in canvas percent.")
    move.add_argument("--y", type=float, help="Absolute top edge, in canvas percent.")

    resize = actions.add_parser("resize-node", help="Resize one object's box; its own label box follows.")
    resize.add_argument("--object", dest="object_id", required=True)
    resize.add_argument("--w", type=float, help="Absolute width, in canvas percent.")
    resize.add_argument("--h", type=float, help="Absolute height, in canvas percent.")
    resize.add_argument("--scale", type=float, help="Multiplier applied to the current size.")

    label = actions.add_parser("set-label", help="Rewrite the visible text that belongs to one object.")
    label.add_argument("--object", dest="object_id", required=True)
    label.add_argument("--text", required=True)

    reroute = actions.add_parser("reroute-arrow", help="Redraw the connectors with the same deterministic router a run uses.")
    reroute.add_argument("--arrow", dest="arrow_id", required=True, help="Arrow the command is aimed at.")
    reroute.add_argument("--route-style", help="Optional route style to force on that arrow before routing.")

    attach = actions.add_parser("attach-endpoint", help="Re-bind an arrow to another source or target object.")
    attach.add_argument("--arrow", dest="arrow_id", required=True)
    attach.add_argument("--source", help="Object id the arrow should leave.")
    attach.add_argument("--target", help="Object id the arrow should reach.")

    waypoints = actions.add_parser("set-waypoints", help="Pin an arrow to a stated polyline, in canvas percent.")
    waypoints.add_argument("--arrow", dest="arrow_id", required=True)
    waypoints.add_argument("--points", required=True, help='JSON list of pairs, e.g. "[[0.1,0.2],[0.3,0.2]]".')

    panel = actions.add_parser("set-panel-bounds", help="Set a panel's bounds, the container its members sit in.")
    panel.add_argument("--panel", dest="object_id", required=True)
    panel.add_argument("--x", type=float)
    panel.add_argument("--y", type=float)
    panel.add_argument("--w", type=float)
    panel.add_argument("--h", type=float)

    constraints = actions.add_parser("apply-constraints", help="Apply compile-time layout and connector rules to the program.")
    constraints.add_argument("--layout", help='JSON object of layout rules, e.g. {"align_grid": 0.02, "gutter_min": 0.01}.')
    constraints.add_argument(
        "--routing",
        help='JSON object of router rules ("shared_vertical_channel", "forbid_crossing", "min_bend", "max_segments") or a stored "selections" list.',
    )

    actions.add_parser("recompile", help="Compile the program to editable_composition.pptx.")
    actions.add_parser("render", help="Render the program to a preview PNG.")


def _json_object(value: str | None, *, name: str) -> dict | None:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--{name} must be a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"--{name} must be a JSON object, not {type(parsed).__name__}")
    return parsed


def figure_edit_invocation(args) -> tuple[str, dict]:
    """Turn one parsed ``figure-edit`` namespace into one call to the composition layer."""

    action = str(args.figure_action)
    if action == "show":
        return action, {"object_id": args.object_id}
    if action == "move-node":
        return action, {"object_id": args.object_id, "dx": args.dx, "dy": args.dy, "x": args.x, "y": args.y}
    if action == "resize-node":
        return action, {"object_id": args.object_id, "w": args.w, "h": args.h, "scale": args.scale}
    if action == "set-label":
        return action, {"object_id": args.object_id, "text": args.text}
    if action == "reroute-arrow":
        return action, {"arrow_id": args.arrow_id, "route_style": args.route_style}
    if action == "attach-endpoint":
        return action, {"arrow_id": args.arrow_id, "source": args.source, "target": args.target}
    if action == "set-waypoints":
        return action, {"arrow_id": args.arrow_id, "waypoints": args.points}
    if action == "set-panel-bounds":
        return action, {"object_id": args.object_id, "x": args.x, "y": args.y, "w": args.w, "h": args.h}
    if action == "apply-constraints":
        return action, {"layout": _json_object(args.layout, name="layout"), "routing": _json_object(args.routing, name="routing")}
    return action, {}
