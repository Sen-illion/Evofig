"""Deterministic execution of bounded direct-draft layout repair intents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _shift(box: dict[str, Any], dx: float, dy: float) -> None:
    box["x"] = round(float(box["x"]) + dx, 4)
    box["y"] = round(float(box["y"]) + dy, 4)


def _clamp_to_panel(box: dict[str, Any], panel_box: dict[str, Any]) -> None:
    box["w"] = round(min(float(box["w"]), float(panel_box["w"]) - 0.024), 4)
    box["h"] = round(min(float(box["h"]), float(panel_box["h"]) - 0.024), 4)
    box["x"] = round(max(float(panel_box["x"]) + 0.012, min(float(box["x"]), float(panel_box["x"]) + float(panel_box["w"]) - float(box["w"]) - 0.012)), 4)
    box["y"] = round(max(float(panel_box["y"]) + 0.012, min(float(box["y"]), float(panel_box["y"]) + float(panel_box["h"]) - float(box["h"]) - 0.012)), 4)


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return not (float(a["x"]) + float(a["w"]) <= float(b["x"]) or float(b["x"]) + float(b["w"]) <= float(a["x"]) or float(a["y"]) + float(a["h"]) <= float(b["y"]) or float(b["y"]) + float(b["h"]) <= float(a["y"]))


#: The mechanism-level layout rules a compile can honour: a grid to sit on, the smallest gap
#: between neighbours in one panel, and the share of its panel a group of boxes has to fill.
LAYOUT_CONSTRAINT_KEYS = ("panel_occupancy_min", "gutter_min", "align_grid")

#: Growth is bounded: a rule that asks for more occupancy must not blow a box up past this.
MAX_OCCUPANCY_SCALE = 1.6
#: The margin boxes keep from their panel edge, matching the other layout passes.
PANEL_MARGIN = 0.012
#: Separation sweeps a grown panel gets before its growth is called too crowded to hold.
SEPARATION_ROUNDS = 6
#: Bisection steps for the largest growth a panel can still hold apart.
GROWTH_BACKOFF_STEPS = 8

#: The layout rule the hand-written standard states, as the parameters the pass below reads.
#: The curated standard already told the planners to keep ``panel_occupancy_min`` at or above
#: 0.45, but it travelled as prose only: a run whose experience library named no layout
#: constraint applied none at all, so every panel kept whatever sparse geometry the model drew.
#: The 2026-10-10 probe measured two panels at 0.185 and 0.061 occupied against that 0.45. The
#: occupancy number is the one the standard states; the gutter is this module's own margin, so
#: the growth is separated by the same distance a box keeps from its panel edge -- without a
#: gutter in the same rule set, growing the boxes toward each other is all the pass would do.
CURATED_LAYOUT_RULES: dict[str, float] = {"panel_occupancy_min": 0.45, "gutter_min": PANEL_MARGIN}


def _valid_box(box: Any) -> bool:
    if not isinstance(box, dict):
        return False
    for key in ("x", "y", "w", "h"):
        value = box.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
    return float(box["w"]) > 0 and float(box["h"]) > 0


def _normalize_layout_constraints(params: Any) -> dict[str, float]:
    """Keep only the rule keys a layout pass can honour, with the types it can read."""

    source = params if isinstance(params, dict) else {}
    nested = source.get("params") if isinstance(source.get("params"), dict) else source
    rules: dict[str, float] = {}
    for key in LAYOUT_CONSTRAINT_KEYS:
        value = nested.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and float(value) > 0:
            rules[key] = round(float(value), 4)
    return rules


def _panels_of(program: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("id")): item["bbox_percent"]
        for item in program.get("panels", []) or []
        if isinstance(item, dict) and _valid_box(item.get("bbox_percent"))
    }


def _grouped_members(program: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group the boxes a layout rule may move by the panel that owns them."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for key in ("semantic_nodes", "slots", "cards"):
        for item in program.get(key, []) or []:
            if not isinstance(item, dict) or not _valid_box(item.get("bbox_percent")):
                continue
            owner = str(item.get("panel_id") or item.get("owner_id") or item.get("parent_id") or "")
            groups.setdefault(owner, []).append(item)
    return sorted(groups.items())


def _clamp_inside(box: dict[str, Any], panel: dict[str, Any] | None) -> None:
    """Keep a box inside its panel, or inside the canvas when no panel owns it."""

    bounds = panel if isinstance(panel, dict) else {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
    margin = PANEL_MARGIN if panel else 0.0
    box["w"] = round(min(float(box["w"]), max(margin, float(bounds["w"]) - margin * 2)), 4)
    box["h"] = round(min(float(box["h"]), max(margin, float(bounds["h"]) - margin * 2)), 4)
    box["x"] = round(min(float(bounds["x"]) + float(bounds["w"]) - margin - float(box["w"]), max(float(bounds["x"]) + margin, float(box["x"]))), 4)
    box["y"] = round(min(float(bounds["y"]) + float(bounds["h"]) - margin - float(box["h"]), max(float(bounds["y"]) + margin, float(box["y"]))), 4)


def _occupied_share(members: list[dict[str, Any]], panel: dict[str, Any]) -> float:
    """How much of its panel the members cover, counting only the part inside it."""

    panel_area = float(panel["w"]) * float(panel["h"])
    if panel_area <= 0:
        return 0.0
    covered = 0.0
    for item in members:
        box = item["bbox_percent"]
        left = max(float(box["x"]), float(panel["x"]))
        top = max(float(box["y"]), float(panel["y"]))
        right = min(float(box["x"]) + float(box["w"]), float(panel["x"]) + float(panel["w"]))
        bottom = min(float(box["y"]) + float(box["h"]), float(panel["y"]) + float(panel["h"]))
        covered += max(0.0, right - left) * max(0.0, bottom - top)
    return round(covered / panel_area, 4)


def _axis_overlap(a: dict[str, Any], b: dict[str, Any], axis: str) -> float:
    size = "w" if axis == "x" else "h"
    return max(0.0, min(float(a[axis]) + float(a[size]), float(b[axis]) + float(b[size])) - max(float(a[axis]), float(b[axis])))


def _separate_axis(a: dict[str, Any], b: dict[str, Any], axis: str, gutter: float) -> float:
    """The signed distance ``b`` has to travel along one axis to sit a gutter away from ``a``."""

    size = "w" if axis == "x" else "h"
    a_start, a_end = float(a[axis]), float(a[axis]) + float(a[size])
    b_start, b_end = float(b[axis]), float(b[axis]) + float(b[size])
    if b_start >= a_start:
        gap = b_start - a_end
        return 0.0 if gap >= gutter else round(gutter - gap, 4)
    gap = a_start - b_end
    return 0.0 if gap >= gutter else round(gap - gutter, 4)


def _separate(a: dict[str, Any], b: dict[str, Any], gutter: float, panel: dict[str, Any] | None) -> int:
    """Push ``b`` off ``a`` along the axis that needs the least travel. 1 when it moved."""

    x_overlap = _axis_overlap(a, b, "x")
    y_overlap = _axis_overlap(a, b, "y")
    if x_overlap > 0 and y_overlap > 0:
        axis = "y" if y_overlap <= x_overlap else "x"
    elif x_overlap > 0:
        axis = "y"
    elif y_overlap > 0:
        axis = "x"
    else:
        # Apart on both axes. A diagonal near-miss is not a gutter, so it is left alone.
        return 0
    delta = _separate_axis(a, b, axis, gutter)
    if not delta:
        return 0
    b[axis] = round(float(b[axis]) + delta, 4)
    _clamp_inside(b, panel)
    return 1


def _violations(members: list[dict[str, Any]], gutter: float) -> int:
    """Pairs that still overlap or sit closer than the gutter after the pass."""

    count = 0
    for index, left in enumerate(members):
        for right in members[index + 1:]:
            a, b = left["bbox_percent"], right["bbox_percent"]
            x_overlap = _axis_overlap(a, b, "x")
            y_overlap = _axis_overlap(a, b, "y")
            if x_overlap > 0 and y_overlap > 0:
                count += 1
                continue
            gap_x = max(float(a["x"]) - (float(b["x"]) + float(b["w"])), float(b["x"]) - (float(a["x"]) + float(a["w"])))
            gap_y = max(float(a["y"]) - (float(b["y"]) + float(b["h"])), float(b["y"]) - (float(a["y"]) + float(a["h"])))
            if x_overlap > 0 and 0 <= gap_y < gutter:
                count += 1
            elif y_overlap > 0 and 0 <= gap_x < gutter:
                count += 1
    return count


def _restore_boxes(members: list[dict[str, Any]], drawn: list[dict[str, Any]]) -> None:
    """Put a panel's members back on the geometry they had before a growth trial."""

    for item, box in zip(members, drawn):
        item["bbox_percent"] = dict(box)


def _place_group(members: list[dict[str, Any]], panel: dict[str, Any], scale: float, gutter: float) -> int:
    """Draw a panel's members at ``scale`` of their drawn size, then settle them apart.

    Growing each member about its own centre is what makes a sparse panel fuller, and it is also
    what puts two neighbours on top of one another, so every trial growth is followed by the same
    separation the ``gutter_min`` rule runs. The members are left at that geometry -- the caller
    restores its own snapshot to try another factor -- and what comes back is the number of pairs
    that are still overlapping or closer than ``gutter`` when the separation stops.
    """

    for item in members:
        box = item["bbox_percent"]
        centre_x = float(box["x"]) + float(box["w"]) / 2
        centre_y = float(box["y"]) + float(box["h"]) / 2
        box["w"] = round(float(box["w"]) * scale, 4)
        box["h"] = round(float(box["h"]) * scale, 4)
        box["x"] = round(centre_x - float(box["w"]) / 2, 4)
        box["y"] = round(centre_y - float(box["h"]) / 2, 4)
        _clamp_inside(box, panel)
    for _ in range(SEPARATION_ROUNDS):
        moved = 0
        for index, left in enumerate(members):
            for right in members[index + 1:]:
                moved += _separate(left["bbox_percent"], right["bbox_percent"], gutter, panel)
        if not moved:
            break
    return _violations(members, gutter)


def apply_layout_constraints(program: dict[str, Any], params: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Draw this figure's boxes by a standing rule instead of re-picking the figure.

    ``align_grid`` snaps every box onto one grid, ``gutter_min`` keeps neighbours inside a panel
    at least that far apart, and ``panel_occupancy_min`` grows a panel's members until they fill
    the stated share of it. Each is a deterministic transform on whatever geometry it is handed,
    so a rule learned on one paper holds for the next one -- which is the whole reason to learn
    the rule instead of the boxes it was measured on.
    """

    rules = _normalize_layout_constraints(params)
    updated = deepcopy(program)
    report: dict[str, Any] = {
        "summary": "Mechanism-level layout rules applied to this figure.",
        "status": "no_change",
        "rules": dict(rules),
        "applied": [],
        "unmet": [],
    }
    if not rules:
        return updated, report
    panels = _panels_of(updated)
    groups = _grouped_members(updated)
    applied: list[dict[str, Any]] = []
    unmet: list[dict[str, Any]] = []
    if "panel_occupancy_min" in rules:
        target = float(rules["panel_occupancy_min"])
        # The growth is separated by the rule's own gutter when it names one, and by this module's
        # panel margin otherwise: the distance a box already keeps from the panel edge, so a grown
        # group stands as far apart as it stands from the frame around it.
        gutter = float(rules.get("gutter_min") or PANEL_MARGIN)
        for owner, members in groups:
            panel = panels.get(owner)
            if not panel or not members:
                continue
            before = _occupied_share(members, panel)
            if before >= target:
                continue
            drawn = [dict(item["bbox_percent"]) for item in members]
            wanted = min(MAX_OCCUPANCY_SCALE, (target / before) ** 0.5) if before > 0 else MAX_OCCUPANCY_SCALE
            scale, backed_off = wanted, False
            crowded = _place_group(members, panel, wanted, gutter)
            if crowded:
                # Growing every member about its own centre lands neighbours on top of each other,
                # and a panel can be too full to pull them apart again. Take the largest factor the
                # panel still holds apart, and 1.0 -- the figure exactly as it was drawn -- when even
                # that is crowded, so the pass never hands back a figure harder to read than the one
                # it was given. What the pass could not do is recorded either way.
                low, high, scale, backed_off = 1.0, wanted, 1.0, True
                for _ in range(GROWTH_BACKOFF_STEPS):
                    middle = round((low + high) / 2, 4)
                    _restore_boxes(members, drawn)
                    if _place_group(members, panel, middle, gutter):
                        high = middle
                    else:
                        low = scale = middle
            _restore_boxes(members, drawn)
            crowded = _place_group(members, panel, scale, gutter)
            after = _occupied_share(members, panel)
            record = {"rule": "panel_occupancy_min", "panel_id": owner, "target": target, "before": before, "after": after, "scale": round(scale, 4), "capped": bool(wanted >= MAX_OCCUPANCY_SCALE), "backed_off_for_overlap": backed_off, "crowded_pairs": crowded}
            applied.append(record)
            if after < target:
                unmet.append(record)
    if "align_grid" in rules:
        grid = float(rules["align_grid"])
        snapped = 0
        for owner, members in groups:
            for item in members:
                box = item["bbox_percent"]
                box["x"] = round(round(float(box["x"]) / grid) * grid, 4)
                box["y"] = round(round(float(box["y"]) / grid) * grid, 4)
                box["w"] = max(grid, round(round(float(box["w"]) / grid) * grid, 4))
                box["h"] = max(grid, round(round(float(box["h"]) / grid) * grid, 4))
                _clamp_inside(box, panels.get(owner))
                snapped += 1
        applied.append({"rule": "align_grid", "grid": grid, "box_count": snapped})
    if "gutter_min" in rules:
        gutter = float(rules["gutter_min"])
        for owner, members in groups:
            panel = panels.get(owner)
            separations = 0
            for _ in range(2):
                for index, left in enumerate(members):
                    for right in members[index + 1:]:
                        separations += _separate(left["bbox_percent"], right["bbox_percent"], gutter, panel)
            record = {"rule": "gutter_min", "panel_id": owner, "gutter": gutter, "separations": separations}
            applied.append(record)
            if _violations(members, gutter):
                unmet.append(record)
    report["status"] = "applied" if applied else "no_change"
    report["applied"] = applied
    report["unmet"] = unmet
    return updated, report


def apply_direct_layout_repairs(program: dict[str, Any], review: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply only semantic-preserving moves inside existing panel bounds."""
    updated = deepcopy(program)
    panels = {str(x.get("id")): x for x in updated.get("panels", []) if isinstance(x, dict) and isinstance(x.get("bbox_percent"), dict)}
    nodes = {str(x.get("id")): x for x in updated.get("semantic_nodes", []) if isinstance(x, dict) and isinstance(x.get("bbox_percent"), dict)}
    slots = {str(x.get("id")): x for x in updated.get("slots", []) if isinstance(x, dict) and isinstance(x.get("bbox_percent"), dict)}
    arrows = {str(x.get("id")): x for x in updated.get("arrows", []) if isinstance(x, dict)}
    labels = {str(x.get("id")): x for x in (updated.get("text_program") or {}).get("items", []) if isinstance(x, dict) and str(x.get("role") or x.get("text_role")) == "connector_label"}
    records, skipped = [], []
    for intent in review.get("repair_intents", []):
        action = str(intent.get("action") or "")
        panel = panels.get(str(intent.get("panel_id") or ""))
        members = [nodes[x] for x in intent.get("object_ids", []) if x in nodes]
        if not members and panel:
            members = [x for x in nodes.values() if str(x.get("panel_id")) == str(panel.get("id"))]
        if action == "rebalance_panel" and panel and members:
            pb = panel["bbox_percent"]
            min_x = min(float(x["bbox_percent"]["x"]) for x in members)
            max_x = max(float(x["bbox_percent"]["x"]) + float(x["bbox_percent"]["w"]) for x in members)
            desired_center = float(pb["x"]) + float(pb["w"]) / 2
            dx = desired_center - (min_x + max_x) / 2
            lower = float(pb["x"]) + 0.012 - min_x
            upper = float(pb["x"]) + float(pb["w"]) - 0.012 - max_x
            dx = max(lower, min(upper, dx))
            before = {str(x.get("id")): dict(x["bbox_percent"]) for x in members}
            for node in members:
                _shift(node["bbox_percent"], dx, 0.0)
                _clamp_to_panel(node["bbox_percent"], pb)
            records.append({"action": action, "panel_id": panel.get("id"), "object_ids": list(before), "delta": {"x": round(dx, 4), "y": 0.0}, "before": before})
        elif action == "translate_group" and panel and members:
            amount = float(intent.get("amount") or 0.0)
            dx = amount if str(intent.get("axis")) != "vertical" else 0.0
            dy = amount if str(intent.get("axis")) == "vertical" else 0.0
            original = [dict(node["bbox_percent"]) for node in members]
            # Clamp the group as a whole so relative spacing is preserved.
            min_x = min(float(box["x"]) for box in original)
            max_x = max(float(box["x"]) + float(box["w"]) for box in original)
            min_y = min(float(box["y"]) for box in original)
            max_y = max(float(box["y"]) + float(box["h"]) for box in original)
            pb = panel["bbox_percent"]
            if dx:
                dx = max(float(pb["x"]) + 0.012 - min_x, min(float(pb["x"]) + float(pb["w"]) - 0.012 - max_x, dx))
            if dy:
                dy = max(float(pb["y"]) + 0.012 - min_y, min(float(pb["y"]) + float(pb["h"]) - 0.012 - max_y, dy))
            for node in members:
                _shift(node["bbox_percent"], dx, dy)
            records.append({"action": action, "panel_id": panel.get("id"), "object_ids": [x.get("id") for x in members], "delta": {"x": dx, "y": dy}})
        elif action in {"distribute_horizontal", "distribute_vertical"} and panel and len(members) >= 2:
            axis = "x" if action.endswith("horizontal") else "y"
            size_key = "w" if axis == "x" else "h"
            members.sort(key=lambda x: float(x["bbox_percent"][axis]))
            pb = panel["bbox_percent"]
            start = float(pb[axis]) + 0.02
            available = float(pb[size_key]) - 0.04
            occupied = sum(float(x["bbox_percent"][size_key]) for x in members)
            gap = (available - occupied) / max(1, len(members) - 1)
            if gap < 0.008:
                skipped.append({"action": action, "reason": "insufficient_panel_space"})
                continue
            cursor = start
            for node in members:
                node["bbox_percent"][axis] = round(cursor, 4)
                cursor += float(node["bbox_percent"][size_key]) + gap
                _clamp_to_panel(node["bbox_percent"], pb)
            records.append({"action": action, "panel_id": panel.get("id"), "object_ids": [x.get("id") for x in members], "gap": round(gap, 4)})
        elif action == "align_objects" and panel and len(members) >= 2:
            axis = "y" if str(intent.get("axis")) != "vertical" else "x"
            target = sum(float(x["bbox_percent"][axis]) for x in members) / len(members)
            for node in members:
                node["bbox_percent"][axis] = round(target, 4)
                _clamp_to_panel(node["bbox_percent"], panel["bbox_percent"])
            records.append({"action": action, "panel_id": panel.get("id"), "object_ids": [x.get("id") for x in members], "axis": axis})
        elif action == "resize_node" and panel and len(members) == 1 and isinstance(intent.get("target_bbox_percent"), dict):
            before = dict(members[0]["bbox_percent"])
            requested = intent["target_bbox_percent"]
            members[0]["bbox_percent"].update({key: round(float(requested.get(key, before[key])), 4) for key in ("x", "y", "w", "h")})
            _clamp_to_panel(members[0]["bbox_percent"], panel["bbox_percent"])
            records.append({"action": action, "panel_id": panel.get("id"), "object_ids": [members[0].get("id")], "before": before, "after": dict(members[0]["bbox_percent"])})
        elif action in {"reserve_connector_lane", "move_connector_label"}:
            targets = []
            for object_id in intent.get("object_ids", []):
                target = arrows.get(str(object_id)) if action == "reserve_connector_lane" else labels.get(str(object_id))
                if isinstance(target, dict):
                    targets.append(target)
            amount = float(intent.get("amount") or 0.02)
            for target in targets:
                if isinstance(target.get("bbox_percent"), dict):
                    axis = "y" if str(intent.get("axis")) != "horizontal" else "x"
                    target["bbox_percent"][axis] = round(float(target["bbox_percent"][axis]) + amount, 4)
                elif action == "reserve_connector_lane":
                    # A lane request invalidates the old route. The canonical
                    # connector router will choose a bounded candidate against
                    # final node geometry; never translate an obsolete path.
                    target["path_percent"] = []
                    target["route_lane_preference"] = {"axis": str(intent.get("axis") or "vertical"), "amount": round(amount, 4)}
            records.append({"action": action, "object_ids": list(intent.get("object_ids", [])), "amount": amount})
        else:
            skipped.append({"action": action, "reason": "unsupported_or_missing_targets"})
    # Keep slots and labels derived from final node geometry; callers rerun the
    # node-content and connector layout passes after this executor.
    changed_node_ids = sorted({str(value) for item in records if item.get("action") in {"rebalance_panel", "translate_group", "distribute_horizontal", "distribute_vertical", "align_objects", "resize_node"} for value in item.get("object_ids", [])})
    requested_connector_ids = sorted({str(value) for item in records if item.get("action") == "reserve_connector_lane" for value in item.get("object_ids", [])})
    return updated, {"summary": "Bounded direct layout repair execution.", "status": "applied" if records else "no_change", "applied_count": len(records), "skipped_count": len(skipped), "changed_node_ids": changed_node_ids, "requested_connector_ids": requested_connector_ids, "applied": records, "skipped": skipped}


__all__ = ["LAYOUT_CONSTRAINT_KEYS", "apply_direct_layout_repairs", "apply_layout_constraints"]
