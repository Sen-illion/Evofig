"""Execution semantics for the bounded repair actions.

The advisory Judge names one of :data:`REPAIR_ACTIONS`; the executor in
``rfs.evolution.direct.controller`` is what actually moves boxes. Escalations
(``replan_structure``, ``resegment``, ``reanalyze_semantics``, ``replan_theme``)
are executed by ``upstream_retry``, not by the bbox pass. The two mechanism-level
rules (``router_constraint``, ``layout_constraint``) are in the catalog so the Judge
can name them, and their row says what they cannot do: they move no box and change
nothing here, because the rule they carry is applied by the compile that draws a
later figure.

Every row answers three questions, because a name on its own is not a menu. ``does``
is what the executor performs, and ``cannot`` is the boundary that keeps a defect the
action can never reach from being assigned to it. ``when`` says what a fault *looks
like* when this is the action to reach for -- without it the seat knows the action
exists but not when to use it, which is how a menu of twelve repairs still comes back
answered with the same one every round. :data:`CATALOG_LAYERS` groups the rows the way
faults are answered: what this round moves itself, the rules the compile applies to
figures drawn under it, then the requests that go upstream.

``heuristic_default`` is the unsigned budget the heuristic gate writes onto a
prescription it issues. The executor does not invent those numbers: a VLM answer
that omitted them is incomplete. A Memory row that carries exactly those numbers
and names no objects is indistinguishable from an arm that was given no Memory;
a row that also names the catalog targets is a different move. The prompt does not
print these numbers -- a seat that reads them back is naming the shipped default
policy, which is the arm experience has to differ from, not follow.
"""

from __future__ import annotations

from typing import Any

from .evolution import REPAIR_ACTIONS


ACTION_BUDGET_KEYS = ("factor", "dx", "dy")

#: Unsigned magnitudes the heuristic gate writes onto a geometric prescription.
#: Slot-scoped nudges scale more than a global pass.
GLOBAL_NUDGE_DEFAULT = {"factor": 1.02, "dx": 0.008, "dy": 0.0}
SLOT_NUDGE_DEFAULT = {"factor": 1.08, "dx": 0.0, "dy": 0.0}

_BUDGET_RANGES = {
    "factor": (1.0, 1.2),
    "dx": (0.0, 0.05),
    "dy": (0.0, 0.05),
}

_NO_BUDGET: tuple[str, ...] = ()
_GEOMETRY_BUDGET = ACTION_BUDGET_KEYS

_GLOBAL_NUDGE_CANNOT = (
    "does not change topology, panel count, reading order, labels, arrows, or icons; "
    "does not re-run structure planning"
)
_SLOT_NUDGE_CANNOT = (
    "does not change topology, labels, arrows, or icons; only resizes and shifts the named boxes"
)


def _entry(
    *,
    does: str,
    when: str,
    cannot: str,
    target_keys: tuple[str, ...],
    budget_keys: tuple[str, ...],
    heuristic_default: dict[str, float] | None = None,
    targets_required: bool = False,
) -> dict[str, Any]:
    return {
        "does": does,
        "when": when,
        "cannot": cannot,
        "target_keys": target_keys,
        "targets_required": targets_required,
        "budget_keys": budget_keys,
        "budget_ranges": {key: _BUDGET_RANGES[key] for key in budget_keys},
        "heuristic_default": dict(heuristic_default or {}),
    }


ACTION_SEMANTICS: dict[str, dict[str, Any]] = {
    "reanalyze_semantics": _entry(
        does="records a request to re-check the paper contract; the extraction itself is upstream_retry, not this pass",
        when="the paper contract the figure was built from looks wrong -- a module, a relation or a label is not what the paper says",
        cannot="does not move boxes, rewrite labels, or change the compiled figure on its own",
        target_keys=("semantic_node_ids",),
        budget_keys=_NO_BUDGET,
    ),
    "replan_structure": _entry(
        does="records a request to re-run global structure planning; the re-plan itself is upstream_retry, not this pass",
        when="the arrangement itself is wrong -- the right modules in the wrong order, grouping or flow, which moving boxes cannot fix",
        cannot="does not move boxes, rewrite labels, or change the compiled figure on its own; does not replan unless upstream_retry.applied",
        target_keys=(),
        budget_keys=_NO_BUDGET,
    ),
    "replan_theme": _entry(
        does="records a request to re-run theme planning; the palette rewrite is the workflow's, not this pass",
        when="the colour system is the defect -- unreadable contrast, too many hues, or colour fighting the panels",
        cannot="does not recolour panels or text here; does not move boxes",
        target_keys=(),
        budget_keys=_NO_BUDGET,
    ),
    "resegment": _entry(
        does="records a request to re-cut panels; the panel re-plan itself is upstream_retry, not this pass",
        when="the panels are cut wrongly -- a module belongs to no panel, or two groups share one panel",
        cannot="does not move boxes or re-cut panels in this pass; does not resegment unless upstream_retry.applied",
        target_keys=(),
        budget_keys=_NO_BUDGET,
    ),
    "relayout_global": _entry(
        does="nudges every node box: scale by factor about its centre, then stagger left/right by unsigned dx on even/odd index",
        when="the figure as a whole is cramped or off balance and no single box is the culprit",
        cannot=_GLOBAL_NUDGE_CANNOT,
        target_keys=(),
        budget_keys=_GEOMETRY_BUDGET,
        heuristic_default=GLOBAL_NUDGE_DEFAULT,
    ),
    "relayout_slots": _entry(
        does="nudges the named node/slot boxes: scale by factor about the centre, optional dx/dy shift",
        when="a few named boxes are the defect -- overlapping, crowding their neighbours, or sitting away from their group",
        cannot=_SLOT_NUDGE_CANNOT,
        target_keys=("semantic_node_ids", "object_ids", "slot_ids"),
        budget_keys=_GEOMETRY_BUDGET,
        heuristic_default=SLOT_NUDGE_DEFAULT,
        targets_required=True,
    ),
    "resize_nodes": _entry(
        does="nudges the named node/slot boxes with the same scale-and-shift pass as relayout_slots",
        when="the named boxes are the wrong size for what they carry -- a label barely fits, or one box swallows its panel",
        cannot=_SLOT_NUDGE_CANNOT,
        target_keys=("semantic_node_ids", "object_ids", "slot_ids"),
        budget_keys=_GEOMETRY_BUDGET,
        heuristic_default=SLOT_NUDGE_DEFAULT,
        targets_required=True,
    ),
    "reroute": _entry(
        does="clears path_percent on the named connectors so the deterministic router re-derives them",
        when="a connector crosses something it should not, or its path is otherwise wrong",
        cannot="does not invent new edges, change endpoints, or move node boxes",
        target_keys=("edge_ids", "object_ids", "semantic_node_ids"),
        budget_keys=_NO_BUDGET,
        targets_required=True,
    ),
    "repair_interfaces": _entry(
        does="clears connector paths like reroute and also nudges the named node boxes with the slot scale-and-shift pass",
        when="a connector and the boxes at its two ends are both part of the defect",
        cannot="does not add ports or change which modules connect; box nudge does not replan structure",
        target_keys=("edge_ids", "object_ids", "semantic_node_ids", "slot_ids"),
        budget_keys=_GEOMETRY_BUDGET,
        heuristic_default=SLOT_NUDGE_DEFAULT,
    ),
    "regen_assets": _entry(
        does="appends a repair sentence to the named icon prompts so Image2 regenerates those slots only",
        when="an icon is itself the defect -- the wrong picture for its slot, not a picture in the wrong place",
        cannot="does not edit labels, arrows, or layout boxes; does not generate a whole-figure image",
        target_keys=("slot_ids", "semantic_node_ids", "object_ids"),
        budget_keys=_NO_BUDGET,
        targets_required=True,
    ),
    "router_constraint": _entry(
        does=(
            "states a routing rule the compile applies while it draws a figure, written under "
            "constraints: shared_vertical_channel (boolean), forbid_crossing (boolean), "
            "min_bend (smallest number of segments), max_segments (largest number of segments)"
        ),
        when="the routing defect is one this figure will not be the last to have, so the fix is a rule about how such connectors are drawn rather than a nudge to this one",
        cannot=(
            "does not move any box, change the figure's content, or touch this round's geometry; "
            "it only takes effect when a figure is compiled, and a rule with none of those "
            "parameters is not a rule"
        ),
        target_keys=(),
        budget_keys=_NO_BUDGET,
    ),
    "layout_constraint": _entry(
        does=(
            "states a layout rule the compile applies while it draws a figure, written under "
            "params: align_grid (grid step as a fraction of the canvas), gutter_min (smallest gap "
            "between neighbouring boxes), panel_occupancy_min (share of its panel a group fills)"
        ),
        when="the layout defect is one this figure will not be the last to have, so the fix is a rule about how such panels are laid out rather than a nudge to this one",
        cannot=(
            "does not move any box, change the figure's content, or touch this round's geometry; "
            "it only takes effect when a figure is compiled, and a rule with none of those "
            "parameters is not a rule"
        ),
        target_keys=(),
        budget_keys=_NO_BUDGET,
    ),
}


def action_semantics(action_type: str) -> dict[str, Any]:
    """Return the catalog row for one bounded action, or empty if the name is unknown."""

    return dict(ACTION_SEMANTICS.get(str(action_type or "").strip()) or {})


def heuristic_default(action_type: str) -> dict[str, float]:
    """Return the unsigned budget the heuristic gate writes for this action."""

    return dict(action_semantics(action_type).get("heuristic_default") or {})


def budget_keys_for(action_type: str) -> tuple[str, ...]:
    """Return the magnitude keys this action's executor reads."""

    keys = action_semantics(action_type).get("budget_keys") or ()
    return tuple(str(key) for key in keys)


def attach_heuristic_budget(repair: dict[str, Any] | None) -> dict[str, Any]:
    """Copy the shipped first-retry magnitudes onto a prescription that named none.

    Callers that may do this are themselves deterministic: the heuristic gate, and the
    decomposability policy that already measured which objects to move. A VLM prescription
    that omitted the numbers stays incomplete; the executor does not guess them.
    """

    record = dict(repair or {})
    name = str(record.get("action") or record.get("type") or "").strip()
    for key, value in heuristic_default(name).items():
        if key not in record:
            record[key] = value
    return record


def collected_target_ids(action_type: str, *sources: dict[str, Any]) -> list[str]:
    """Collect object ids an action names, across the typed target fields the catalog lists."""

    keys = tuple(action_semantics(action_type).get("target_keys") or ())
    found: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            for item in source.get(key) or []:
                text = str(item).strip()
                if text and text not in found:
                    found.append(text)
    return found


def action_missing_required_targets(action_type: str, *sources: dict[str, Any]) -> bool:
    """Whether this action's catalog says it needs object ids and none were supplied."""

    if not action_semantics(action_type).get("targets_required"):
        return False
    return not collected_target_ids(action_type, *sources)


def action_needs_budget(action_type: str) -> bool:
    """Whether a prescription of this action is incomplete without factor/dx/dy."""

    return bool(budget_keys_for(action_type))


#: The order the menu is read in, grouped the way faults are answered. Each layer is
#: ``(title, what a fault in this layer looks like and who answers it, action names)``.
#: Rendering walks this rather than sorting the names, so the seat reads the menu in the
#: order it should be considering the layers, and a row can only fall out of the prompt by
#: falling out of the catalog.
CATALOG_LAYERS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "object-level actions",
        "the defect is in this figure's own boxes, connectors or icons, and deterministic code "
        "moves them in this round; the move is what the next round is measured on",
        (
            "relayout_global",
            "relayout_slots",
            "resize_nodes",
            "reroute",
            "repair_interfaces",
            "regen_assets",
        ),
    ),
    (
        "compile-level rules",
        "the defect is a habit this figure will not be the last to have, so the fix is written as "
        "a rule the compile applies to every figure drawn under it; nothing moves here, so this "
        "round is not re-measured on it",
        ("router_constraint", "layout_constraint"),
    ),
    (
        "stage-level requests",
        "no box in this figure can be moved to fix it, because the step that produced the figure "
        "has to run again; the request is recorded here, and upstream_retry decides whether that "
        "step re-runs",
        ("reanalyze_semantics", "replan_structure", "replan_theme", "resegment"),
    ),
)


def render_action_catalog_for_prompt() -> str:
    """Plain-language catalog for the advisory Judge prompt.

    Generated from :data:`ACTION_SEMANTICS` so the served text cannot drift from the
    executor, and grouped by :data:`CATALOG_LAYERS` so the order is the order faults are
    answered in. Acceptance prompts must not include this block.
    """

    lines = [
        "Each requested action is executed by deterministic code, not by rewriting PowerPoint. "
        "Name only an action below, with the object ids it needs and, when the action lists "
        "budget keys, the unsigned factor/dx/dy it should apply: a budget you leave out is the "
        "executor's own default, and a budget you give is your claim about this figure. A "
        "mechanism-level rule carries its own parameters instead -- constraints for "
        "router_constraint, params for layout_constraint -- and a rule with none of them is not "
        "a rule. An action whose row defers its work to upstream_retry only records the request. "
        "An empty requested_actions list means the loop must stop this round rather than invent "
        "a repair. Do not name relayout_global unless a global box nudge is what you want."
    ]
    for title, blurb, names in CATALOG_LAYERS:
        lines.append(f"{title}: {blurb}.")
        for name in names:
            row = ACTION_SEMANTICS[name]
            lines.append(f"{name}: does {row['does']}; when {row['when']}; cannot {row['cannot']}.")
    return "\n".join(lines)


def prescription_matches_heuristic_default(action: dict[str, Any] | None) -> bool:
    """Whether this action would replay the shipped first-retry nudge.

    A Memory row that matches the catalog budget and names no targets is
    indistinguishable from the OFF arm's default policy. Naming the objects the
    catalog says this action can touch is a different prescription: ON injects
    which boxes move, not another copy of the unnamed first retry.
    """

    source = action if isinstance(action, dict) else {}
    name = str(source.get("type") or source.get("action") or "").strip()
    default = heuristic_default(name)
    if not default:
        return False
    for key, expected in default.items():
        try:
            actual = float(source.get(key))
        except (TypeError, ValueError):
            return False
        if abs(actual - float(expected)) > 1e-9:
            return False
    if collected_target_ids(name, source):
        return False
    return True


def prescription_cannot_contrast(action: dict[str, Any] | None) -> str:
    """Why injecting this prescription cannot differ from the OFF arm, or empty."""

    if prescription_matches_heuristic_default(action):
        return "the experience replays the shipped default policy"
    source = action if isinstance(action, dict) else {}
    name = str(source.get("type") or source.get("action") or "").strip()
    if not action_needs_budget(name):
        return ""
    for key in budget_keys_for(name):
        try:
            float(source.get(key))
        except (TypeError, ValueError):
            return "the experience is missing the budget another paper needs to replay it"
    return ""


def validate_action_catalog() -> list[str]:
    """Return catalog holes: missing actions, extra names, or incomplete rows."""

    errors: list[str] = []
    if set(ACTION_SEMANTICS) != set(REPAIR_ACTIONS):
        missing = sorted(REPAIR_ACTIONS - set(ACTION_SEMANTICS))
        extra = sorted(set(ACTION_SEMANTICS) - REPAIR_ACTIONS)
        if missing:
            errors.append("catalog missing actions: " + ", ".join(missing))
        if extra:
            errors.append("catalog has unknown actions: " + ", ".join(extra))
    for name, row in ACTION_SEMANTICS.items():
        for field in ("does", "when", "cannot"):
            if not str(row.get(field) or "").strip():
                errors.append(f"{name} is missing {field}")
    layered = [name for _, _, names in CATALOG_LAYERS for name in names]
    if sorted(layered) != sorted(REPAIR_ACTIONS):
        errors.append(
            "catalog layers do not cover the actions exactly once: "
            f"layered={sorted(layered)} actions={sorted(REPAIR_ACTIONS)}"
        )
        for key in row.get("budget_keys") or ():
            if key not in ACTION_BUDGET_KEYS:
                errors.append(f"{name} lists unsupported budget key {key}")
        default = row.get("heuristic_default") or {}
        if set(default) - set(row.get("budget_keys") or ()):
            errors.append(f"{name} heuristic_default has keys the executor does not read")
    return errors


__all__ = [
    "ACTION_BUDGET_KEYS",
    "ACTION_SEMANTICS",
    "CATALOG_LAYERS",
    "GLOBAL_NUDGE_DEFAULT",
    "SLOT_NUDGE_DEFAULT",
    "action_missing_required_targets",
    "action_needs_budget",
    "action_semantics",
    "attach_heuristic_budget",
    "budget_keys_for",
    "collected_target_ids",
    "heuristic_default",
    "prescription_cannot_contrast",
    "prescription_matches_heuristic_default",
    "render_action_catalog_for_prompt",
    "validate_action_catalog",
]
