"""Stable contract for visual review of direct editable PPT drafts."""

from __future__ import annotations

from typing import Any


ALLOWED_ISSUES = {
    "horizontal_imbalance",
    "vertical_imbalance",
    "crowding",
    "excess_whitespace",
    "text_too_small",
    "text_hierarchy",
    "connector_label_mismatch",
    "connector_congestion",
    "icon_text_ratio",
    "color_harmony",
}
ALLOWED_ACTIONS = {
    "rebalance_panel",
    "translate_group",
    "distribute_horizontal",
    "distribute_vertical",
    "align_objects",
    "resize_node",
    "reserve_connector_lane",
    "move_connector_label",
    "replan_structure",
    "replan_colors",
}
ALLOWED_STATUSES = {"pass", "needs_repair", "needs_replan", "blocked", "failed", "skipped"}
ALLOWED_SEVERITIES = {"minor", "major", "blocker"}


def _bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        x, y, w, h = (float(value.get(key)) for key in ("x", "y", "w", "h"))
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    x, y = max(0.0, min(0.99, x)), max(0.0, min(0.99, y))
    w, h = min(w, 1.0 - x), min(h, 1.0 - y)
    return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}


def _amount(value: Any) -> float | None:
    try:
        return max(-0.15, min(0.15, float(value)))
    except (TypeError, ValueError):
        return None


def normalize_visual_review(value: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize untrusted VLM output without inventing repair instructions."""
    source = value if isinstance(value, dict) else {}
    feedback = []
    panel_ids_hint = {str(x.get("panel_id") or x.get("id") or "") for x in source.get("panels", []) if isinstance(x, dict)}
    for item in source.get("feedback", []) if isinstance(source.get("feedback"), list) else []:
        if not isinstance(item, dict):
            continue
        issue = str(item.get("issue_type") or "").strip()
        if issue not in ALLOWED_ISSUES:
            continue
        object_ids = [str(x) for x in item.get("object_ids", []) if str(x).strip()]
        panel_id = str(item.get("panel_id") or "")
        if not panel_id:
            panel_matches = [x for x in object_ids if x in panel_ids_hint]
            if panel_matches:
                panel_id = panel_matches[0]
                object_ids = [x for x in object_ids if x != panel_id]
        feedback.append({
            "issue_type": issue,
            "severity": str(item.get("severity") or "minor") if str(item.get("severity") or "minor") in ALLOWED_SEVERITIES else "minor",
            "panel_id": panel_id,
            "object_ids": object_ids,
            "evidence": str(item.get("evidence") or "").strip(),
        })
    intents = []
    for item in source.get("repair_intents", []) if isinstance(source.get("repair_intents"), list) else []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip()
        if action not in ALLOWED_ACTIONS:
            continue
        object_ids = [str(x) for x in item.get("object_ids", []) if str(x).strip()]
        panel_id = str(item.get("panel_id") or "")
        if not panel_id:
            panel_matches = [x for x in object_ids if x in panel_ids_hint]
            if panel_matches:
                panel_id = panel_matches[0]
                object_ids = [x for x in object_ids if x != panel_id]
        intents.append({
            "action": action,
            "panel_id": panel_id,
            "object_ids": object_ids,
            "axis": str(item.get("axis") or ""),
            "amount": _amount(item.get("amount")) if action in {"translate_group", "reserve_connector_lane", "move_connector_label"} else None,
            "target_bbox_percent": _bbox(item.get("target_bbox_percent")),
            "reason": str(item.get("reason") or "").strip(),
        })
    scores = source.get("scores") if isinstance(source.get("scores"), dict) else {}
    return {
        "contract": "rfs.direct_visual_review",
        "version": 1,
        "status": str(source.get("status")) if str(source.get("status")) in ALLOWED_STATUSES else ("needs_repair" if feedback else "pass"),
        "scores": {key: max(0.0, min(1.0, float(scores.get(key)))) if isinstance(scores.get(key), (int, float)) else None for key in ("balance", "readability", "density", "flow_clarity")},
        "feedback": feedback,
        "repair_intents": intents,
        "requires_replan": bool(source.get("requires_replan")) or any(x["action"] == "replan_structure" for x in intents),
        "provider_error": str(source.get("provider_error") or "") or None,
    }


def validate_visual_review(review: dict[str, Any], *, valid_object_ids: set[str], valid_panel_ids: set[str], object_kinds: dict[str, str] | None = None) -> dict[str, Any]:
    """Validate references and executable intent fields."""
    errors = []
    global_panel_ids = {"root", "global", "canvas"}
    for index, item in enumerate(review.get("feedback", [])):
        if item.get("panel_id") and item["panel_id"] not in valid_panel_ids and item["panel_id"] not in global_panel_ids:
            errors.append(f"feedback[{index}] references unknown panel_id")
        unknown = sorted(set(item.get("object_ids", [])) - valid_object_ids - valid_panel_ids)
        if unknown:
            errors.append(f"feedback[{index}] references unknown object_ids: {', '.join(unknown)}")
    for index, item in enumerate(review.get("repair_intents", [])):
        if item.get("panel_id") and item["panel_id"] not in valid_panel_ids and item["panel_id"] not in global_panel_ids:
            errors.append(f"repair_intents[{index}] references unknown panel_id")
        unknown = sorted(set(item.get("object_ids", [])) - valid_object_ids - valid_panel_ids)
        if unknown:
            errors.append(f"repair_intents[{index}] references unknown object_ids: {', '.join(unknown)}")
        action = str(item.get("action") or "")
        kinds = {str((object_kinds or {}).get(value) or "") for value in item.get("object_ids", [])}
        if action in {"rebalance_panel", "translate_group", "distribute_horizontal", "distribute_vertical", "align_objects", "resize_node"} and kinds - {"node"}:
            errors.append(f"repair_intents[{index}] requires node object_ids")
        if action in {"reserve_connector_lane", "move_connector_label"} and kinds - {"arrow", "connector_label"}:
            errors.append(f"repair_intents[{index}] requires arrow or connector-label object_ids")
        # A visual reviewer may identify a resize need without choosing exact
        # geometry.  The compact feedback planner owns that decision; defer
        # geometry rather than blocking the whole review contract.
    return {"ok": not errors, "errors": errors, "feedback_count": len(review.get("feedback", [])), "repair_intent_count": len(review.get("repair_intents", []))}


__all__ = ["ALLOWED_ACTIONS", "normalize_visual_review", "validate_visual_review"]
