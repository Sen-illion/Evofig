"""Replayable Memory parameters the compilers already read as JSON.

A promoted experience is not a prompt. Theme palettes, density bands, connector
route decisions, layout rules and Icon prompt constraints have to sit on the
action as the same keys the planner and the PPT compiler consume, so an ON arm
can apply them without asking a model to invent a second copy. Paper-specific panel ids are
dropped from a theme; a missing palette, a missing budget, or an Icon row that
was not produced under Image2 is not injectable.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .color_theme import normalize_color_theme
from .repair_actions import ACTION_BUDGET_KEYS

MEMORY_PARAMETERS_CONTRACT = "rfs.memory_parameters"
MEMORY_PARAMETERS_VERSION = 1

#: Nested JSON the injector copies onto the next paper's action.
NESTED_PARAMETER_KEYS = (
    "color_theme",
    "density_targets",
    "routing_constraints",
    "layout_constraints",
    "icon_prompt_constraint",
)

#: Measured numbers that travel with a row in addition to the geometric budget.
NUMERIC_PARAMETER_KEYS = ACTION_BUDGET_KEYS + (
    "min_panel_occupancy",
    "max_cross_panel_connectors",
    "max_node_slot_fanout",
    "min_panel_area",
    "minimum_font_size",
    "node_spacing",
)

INJECTABLE_PARAMETERS = NUMERIC_PARAMETER_KEYS

IMAGE2_ICON_MODE = "image2"

_THEME_KEEP = (
    "theme_name",
    "theme_intent",
    "canvas_background",
    "text",
    "connectors",
    "roles",
    "role_mapping",
    "accents",
    "usage_policy",
    "palette_strategy",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def replayable_theme(theme: Any) -> dict[str, Any]:
    """Return the portable palette a later paper can compile, or empty if there is none.

    Role colours, text, connectors and the canvas travel. Per-panel fills do not: those
    ids belong to the figure that learned the row.
    """

    source = theme if isinstance(theme, dict) else {}
    if isinstance(source.get("color_theme"), dict) and not source.get("roles") and not source.get("text"):
        source = source["color_theme"]
    normalized = normalize_color_theme(source)
    text = normalized.get("text") if isinstance(normalized.get("text"), dict) else {}
    connectors = normalized.get("connectors") if isinstance(normalized.get("connectors"), dict) else {}
    roles = normalized.get("roles") if isinstance(normalized.get("roles"), dict) else {}
    if not roles and not (text.get("primary") and connectors.get("primary")):
        return {}
    replayed = {key: deepcopy(normalized[key]) for key in _THEME_KEEP if key in normalized}
    replayed["contract"] = "rfs.color_theme"
    replayed["version"] = 2
    return replayed


def density_targets_from(source: Any) -> dict[str, Any]:
    """Collect occupancy, spacing and type size a density advisory already knows how to read."""

    raw = source if isinstance(source, dict) else {}
    nested = raw.get("density_targets") if isinstance(raw.get("density_targets"), dict) else raw
    targets: dict[str, Any] = {}
    for key in (
        "panel_occupancy_min",
        "canvas_occupancy_min",
        "max_empty_region_ratio",
        "minimum_occupancy",
        "minimum_font_size",
        "minimum_font_size_pt",
        "node_spacing",
        "icon_area_ratio_min",
        "icon_area_ratio_max",
    ):
        value = _number(nested.get(key))
        if value is not None:
            targets[key] = value
    band = nested.get("occupancy_band") or nested.get("density_band")
    if isinstance(band, (list, tuple)) and len(band) >= 2:
        low, high = _number(band[0]), _number(band[1])
        if low is not None and high is not None:
            targets["occupancy_band"] = [low, high]
    return targets


def routing_constraints_from(source: Any) -> dict[str, Any]:
    """Keep the route decision a later compile can replay without asking the reviewer again."""

    raw = source if isinstance(source, dict) else {}
    nested = raw.get("routing_constraints") if isinstance(raw.get("routing_constraints"), dict) else raw
    selections = nested.get("selections") if isinstance(nested.get("selections"), list) else []
    paths: list[dict[str, Any]] = []
    for item in selections:
        if not isinstance(item, dict):
            continue
        record = {
            key: item.get(key)
            for key in ("arrow_id", "connector_id", "candidate_id", "relation_type", "source", "target")
            if item.get(key) not in (None, "")
        }
        path = item.get("path") or item.get("path_percent")
        if isinstance(path, list) and path:
            record["path_percent"] = deepcopy(path)
        if record:
            paths.append(record)
    for item in nested.get("paths") or []:
        if isinstance(item, dict) and isinstance(item.get("path_percent"), list):
            paths.append(deepcopy(item))
    policy = nested.get("policy") if isinstance(nested.get("policy"), dict) else {}
    router = router_rules_from(nested)
    if not paths and not policy and not router:
        return {}
    constraints: dict[str, Any] = {}
    if paths:
        constraints["selections"] = paths
    if policy:
        constraints["policy"] = deepcopy(policy)
    if router:
        constraints["router"] = router
    return constraints


#: The routing rules that stay true on the next figure, because they name a way to draw rather
#: than the edges of the figure that taught them. Everything else in a route replay names a
#: connector id, which no other paper has.
ROUTER_RULE_KEYS = (
    "shared_vertical_channel",
    "forbid_crossing",
    "min_bend",
    "max_segments",
    # The two a person states after looking at a figure. They travel the same way the drawn rules
    # do, so a person's sentence can name the geometry it wants rather than only describe it.
    "direct_when_close",
    "stay_inside_parent",
)

#: The layout rules of the same kind: a grid to sit on, a smallest gap between neighbours, and
#: the share of its panel a group of boxes has to fill.
LAYOUT_RULE_KEYS = ("panel_occupancy_min", "gutter_min", "align_grid")


def _rule_number(value: Any) -> float | None:
    """A positive measured number, or None. A boolean is not a number here."""

    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4) if number > 0 else None


def router_rules_from(source: Any) -> dict[str, Any]:
    """Read the mechanism-level routing rules an action or a Memory block states.

    A ``router_constraint`` row carries them under ``constraints``; the compiler reads them
    out of the ``routing_constraints`` block it already reads id-bound replays from, under
    ``router``. Both spellings land here, so a rule has one definition rather than two.
    """

    raw = source if isinstance(source, dict) else {}
    for candidate in (raw.get("router"), raw.get("constraints"), raw.get("params"), raw):
        if not isinstance(candidate, dict):
            continue
        rules: dict[str, Any] = {}
        for key in ("shared_vertical_channel", "forbid_crossing", "direct_when_close", "stay_inside_parent"):
            if isinstance(candidate.get(key), bool):
                rules[key] = candidate[key]
        gap = _rule_number(candidate.get("close_gap"))
        if gap is not None:
            rules["close_gap"] = gap
        for key in ("min_bend", "max_segments"):
            number = _rule_number(candidate.get(key))
            if number is not None:
                rules[key] = int(number) if key == "max_segments" else number
        if rules:
            return rules
    return {}


def layout_rules_from(source: Any) -> dict[str, Any]:
    """Read the mechanism-level layout rules an action or a Memory block states."""

    raw = source if isinstance(source, dict) else {}
    for candidate in (raw.get("params"), raw.get("layout_constraints"), raw):
        if not isinstance(candidate, dict):
            continue
        rules = {key: _rule_number(candidate.get(key)) for key in LAYOUT_RULE_KEYS}
        found = {key: value for key, value in rules.items() if value is not None}
        if found:
            return found
    return {}


def layout_constraints_from(source: Any) -> dict[str, Any]:
    """Keep the layout rule a later compile draws by, with no ids from the paper that learned it."""

    rules = layout_rules_from(source)
    return {"params": rules} if rules else {}


def icon_prompt_constraint_from(source: Any) -> dict[str, Any]:
    """Keep the slot-level Image2 wording a regen_assets row actually applied."""

    raw = source if isinstance(source, dict) else {}
    nested = raw.get("icon_prompt_constraint") if isinstance(raw.get("icon_prompt_constraint"), dict) else {}
    slot_ids = [str(item) for item in (nested.get("slot_ids") or raw.get("slot_ids") or []) if str(item).strip()]
    constraint = str(nested.get("constraint") or nested.get("prompt") or raw.get("problem") or "").strip()
    if not slot_ids and not constraint:
        return {}
    record: dict[str, Any] = {}
    if slot_ids:
        record["slot_ids"] = slot_ids
    if constraint:
        record["constraint"] = constraint
    style = str(nested.get("style") or "").strip()
    if style:
        record["style"] = style
    return record


def icon_is_transferable(action: dict[str, Any] | None, *, icon_mode: str = "") -> bool:
    """Whether a regen_assets row was produced under a real Image2 call."""

    source = action if isinstance(action, dict) else {}
    if source.get("transferable") is False:
        return False
    mode = str(icon_mode or source.get("icon_mode") or "").strip().casefold()
    if mode and mode != IMAGE2_ICON_MODE:
        return False
    if str(source.get("type") or "") == "regen_assets" and mode == IMAGE2_ICON_MODE:
        return True
    if str(source.get("type") or "") == "regen_assets" and not mode:
        return source.get("transferable") is True
    return True


def copy_injectable_parameters(source: dict[str, Any] | None, dest: dict[str, Any]) -> dict[str, Any]:
    """Copy numeric budgets and nested compiler JSON from one action onto another."""

    raw = source if isinstance(source, dict) else {}
    for key in NUMERIC_PARAMETER_KEYS:
        value = _number(raw.get(key))
        if value is not None:
            dest[key] = value
    theme = replayable_theme(raw.get("color_theme"))
    if theme:
        dest["color_theme"] = theme
    density = density_targets_from(raw)
    if density:
        dest["density_targets"] = density
    routing = routing_constraints_from(raw)
    if routing:
        dest["routing_constraints"] = routing
    layout = layout_constraints_from(raw)
    if layout:
        dest["layout_constraints"] = layout
    icon = icon_prompt_constraint_from(raw)
    if icon:
        dest["icon_prompt_constraint"] = icon
    if raw.get("transferable") is False:
        dest["transferable"] = False
    elif raw.get("transferable") is True:
        dest["transferable"] = True
    mode = str(raw.get("icon_mode") or "").strip()
    if mode:
        dest["icon_mode"] = mode
    return dest


def stamp_executable_parameters(
    action: dict[str, Any] | None,
    *,
    bounded_change: dict[str, Any] | None = None,
    color_theme: dict[str, Any] | None = None,
    layout: dict[str, Any] | None = None,
    icon_mode: str = "",
) -> dict[str, Any]:
    """Write compiler-readable JSON onto the action a later paper will inject."""

    recorded = dict(action or {})
    action_type = str(recorded.get("type") or "")
    change = bounded_change if isinstance(bounded_change, dict) else {}
    if action_type == "replan_theme":
        theme = replayable_theme(recorded.get("color_theme") or color_theme)
        if theme:
            recorded["color_theme"] = theme
    if action_type in {"resize_nodes", "relayout_slots", "relayout_global"}:
        targets = density_targets_from(recorded) or density_targets_from(layout) or density_targets_from(change)
        if targets:
            recorded["density_targets"] = targets
    if action_type in {"reroute", "repair_interfaces"}:
        constraints = routing_constraints_from(recorded) or routing_constraints_from(change)
        if constraints:
            recorded["routing_constraints"] = constraints
    if action_type == "regen_assets":
        if icon_mode:
            recorded["icon_mode"] = str(icon_mode).strip()
        recorded["transferable"] = icon_is_transferable(recorded, icon_mode=icon_mode)
        constraint = icon_prompt_constraint_from(recorded)
        if constraint:
            recorded["icon_prompt_constraint"] = constraint
    if action_type == "router_constraint":
        rules = router_rules_from(recorded) or router_rules_from(change)
        if rules:
            recorded["routing_constraints"] = {"router": rules}
    if action_type == "layout_constraint":
        rules = layout_rules_from(recorded) or layout_rules_from(change)
        if rules:
            recorded["layout_constraints"] = {"params": rules}
    return recorded


def injection_blocked_reason(action: dict[str, Any] | None) -> str:
    """Why this action cannot fork an ON arm from the shipped compiler defaults."""

    source = action if isinstance(action, dict) else {}
    action_type = str(source.get("type") or "")
    if action_type == "replan_theme" and not replayable_theme(source.get("color_theme")):
        return "the experience is missing the palette another paper needs to replay it"
    if action_type in {"resize_nodes", "relayout_slots"} and not density_targets_from(source) and _number(source.get("factor")) is None:
        return "the experience is missing the density band another paper needs to replay it"
    if action_type == "regen_assets" and not icon_is_transferable(source, icon_mode=str(source.get("icon_mode") or "")):
        return "the icon repair was not produced under Image2 and cannot migrate"
    if action_type == "router_constraint" and not router_rules_from(source):
        return "the experience names no routing rule another paper could be drawn by"
    if action_type == "layout_constraint" and not layout_rules_from(source):
        return "the experience names no layout rule another paper could be drawn by"
    return ""


__all__ = [
    "IMAGE2_ICON_MODE",
    "INJECTABLE_PARAMETERS",
    "LAYOUT_RULE_KEYS",
    "MEMORY_PARAMETERS_CONTRACT",
    "MEMORY_PARAMETERS_VERSION",
    "NESTED_PARAMETER_KEYS",
    "NUMERIC_PARAMETER_KEYS",
    "ROUTER_RULE_KEYS",
    "copy_injectable_parameters",
    "density_targets_from",
    "icon_is_transferable",
    "icon_prompt_constraint_from",
    "injection_blocked_reason",
    "layout_constraints_from",
    "layout_rules_from",
    "replayable_theme",
    "router_rules_from",
    "routing_constraints_from",
    "stamp_executable_parameters",
]
