"""Compact VLM planning for bounded draft repairs after visual review."""

from __future__ import annotations

import json
from typing import Any

from ..providers import call_vlm_json, resolve_vlm_model


_ACTIONS = {"rebalance_panel", "translate_group", "distribute_horizontal", "distribute_vertical", "align_objects", "reserve_connector_lane", "move_connector_label"}


def _compact_context(program: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    affected = {str(value) for item in (review.get("feedback") or []) + (review.get("repair_intents") or []) if isinstance(item, dict) for value in item.get("object_ids", [])}
    panel_ids = {str(item.get("panel_id")) for item in (review.get("feedback") or []) + (review.get("repair_intents") or []) if isinstance(item, dict) and item.get("panel_id")}
    nodes = [item for item in program.get("semantic_nodes", []) if isinstance(item, dict) and (str(item.get("id")) in affected or str(item.get("panel_id")) in panel_ids)]
    node_ids = {str(item.get("id")) for item in nodes}
    arrows = [item for item in program.get("arrows", []) if isinstance(item, dict) and (str(item.get("id")) in affected or str(item.get("source")) in node_ids or str(item.get("target")) in node_ids)]
    text = [item for item in (program.get("text_program") or {}).get("items", []) if isinstance(item, dict) and (str(item.get("owner_node_id")) in node_ids or str(item.get("target_id")) in affected)]
    panels = [item for item in program.get("panels", []) if isinstance(item, dict) and (str(item.get("id")) in panel_ids or any(str(node.get("panel_id")) == str(item.get("id")) for node in nodes))]
    return {"panels": [{k: x.get(k) for k in ("id", "bbox_percent")} for x in panels], "nodes": [{k: x.get(k) for k in ("id", "panel_id", "bbox_percent", "internal_layout")} for x in nodes], "connectors": [{k: x.get(k) for k in ("id", "source", "target", "path_percent")} for x in arrows], "text": [{k: x.get(k) for k in ("id", "owner_node_id", "target_id", "bbox_percent", "font_size_pt")} for x in text], "feedback": review.get("feedback", [])}


def plan_feedback_repairs_vlm(program: dict[str, Any], review: dict[str, Any], *, model: str | None = None, timeout: int = 120, retries: int = 1, call_metadata: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return local/panel/global-candidate intent without rebuilding composition.

    ``call_metadata`` is an optional caller-owned dict the provider fills with the call's own
    timing, so a round can show what this seat cost instead of only that it ran.
    """
    context = _compact_context(program, review)
    prompt = "You repair an existing editable scientific PPT draft; do not redesign it. Return JSON only: {repair_scope:'local|panel|global_candidate',repair_intents:[{action:'rebalance_panel|translate_group|distribute_horizontal|distribute_vertical|align_objects|reserve_connector_lane|move_connector_label',panel_id:'',object_ids:[],axis:'horizontal|vertical',amount:0.0,reason:''}],protected_object_ids:[],reason:''}. Preserve all IDs, scientific relations, unaffected geometry, text and colors. Prefer local, then panel. Use global_candidate only when panel division itself prevents readable flow. CONTEXT=" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    raw = None
    try:
        raw = call_vlm_json(prompt, [], model=resolve_vlm_model("RFS_FEEDBACK_REPAIR_MODEL", "RFS_DIRECT_VISUAL_REVIEW_MODEL", explicit_model=model), timeout=timeout, retries=retries, call_metadata=call_metadata)
        scope = str(raw.get("repair_scope") or "local")
        if scope not in {"local", "panel", "global_candidate"}:
            scope = "local"
        intents = []
        for item in raw.get("repair_intents", []) if isinstance(raw.get("repair_intents"), list) else []:
            if not isinstance(item, dict) or str(item.get("action")) not in _ACTIONS:
                continue
            intents.append({"action": str(item.get("action")), "panel_id": str(item.get("panel_id") or ""), "object_ids": [str(x) for x in item.get("object_ids", [])], "axis": str(item.get("axis") or ""), "amount": max(-0.12, min(0.12, float(item.get("amount") or 0.0))), "reason": str(item.get("reason") or "")})
        result = {"contract": "rfs.feedback_repair_plan", "repair_scope": scope, "repair_intents": intents, "protected_object_ids": [str(x) for x in raw.get("protected_object_ids", [])] if isinstance(raw.get("protected_object_ids"), list) else [], "reason": str(raw.get("reason") or "")}
        report = {"ok": bool(intents), "mode": "vlm", "fallback_used": False, "prompt_chars": len(prompt), "estimated_tokens": (len(prompt) + 3) // 4, "affected_object_count": len(context["nodes"]) + len(context["connectors"]) + len(context["text"]), "raw_response": raw}
        if call_metadata is not None:
            report["provider"] = dict(call_metadata)
        return result, report
    except Exception as exc:
        report = {"ok": False, "mode": "fallback", "fallback_used": True, "fallback_reason": str(exc), "prompt_chars": len(prompt), "estimated_tokens": (len(prompt) + 3) // 4, "affected_object_count": len(context["nodes"]) + len(context["connectors"]) + len(context["text"]), "raw_response": raw}
        if call_metadata is not None:
            report["provider"] = dict(call_metadata)
        return {"contract": "rfs.feedback_repair_plan", "repair_scope": "local", "repair_intents": [], "protected_object_ids": [], "reason": "repair planning unavailable"}, report


__all__ = ["plan_feedback_repairs_vlm"]
