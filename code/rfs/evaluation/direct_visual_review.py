"""Visual review for compiled drafts in the direct paper-to-PPTX workflow."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from ..contracts.visual_review import normalize_visual_review, validate_visual_review
from ..providers import call_vlm_json, resolve_vlm_model


def _compact_program(program: dict[str, Any]) -> dict[str, Any]:
    return {
        "canvas": program.get("canvas"),
        "panels": [{"id": x.get("id"), "title": x.get("title"), "bbox_percent": x.get("bbox_percent")} for x in program.get("panels", []) if isinstance(x, dict)],
        "nodes": [{"id": x.get("id"), "panel_id": x.get("panel_id"), "label": x.get("label"), "bbox_percent": x.get("bbox_percent"), "font_size_pt": x.get("font_size_pt"), "internal_layout": x.get("internal_layout")} for x in program.get("semantic_nodes", []) if isinstance(x, dict)],
        "connectors": [{"id": x.get("id"), "source": x.get("source"), "target": x.get("target"), "label": x.get("label"), "path_percent": x.get("path_percent")} for x in program.get("arrows", []) if isinstance(x, dict)],
        "text": [{"id": x.get("id"), "role": x.get("role") or x.get("text_role"), "text": x.get("text"), "bbox_percent": x.get("bbox_percent"), "font_size_pt": x.get("font_size_pt"), "visible": x.get("visible", True)} for x in (program.get("text_program") or {}).get("items", []) if isinstance(x, dict)],
    }


def _heuristic_review(program: dict[str, Any]) -> dict[str, Any]:
    feedback, intents = [], []
    nodes = [x for x in program.get("semantic_nodes", []) if isinstance(x, dict) and isinstance(x.get("bbox_percent"), dict)]
    for panel in program.get("panels", []):
        if not isinstance(panel, dict) or not isinstance(panel.get("bbox_percent"), dict):
            continue
        members = [x for x in nodes if str(x.get("panel_id")) == str(panel.get("id"))]
        if not members:
            continue
        pb = panel["bbox_percent"]
        left = min(float(x["bbox_percent"]["x"]) for x in members) - float(pb["x"])
        right = float(pb["x"]) + float(pb["w"]) - max(float(x["bbox_percent"]["x"]) + float(x["bbox_percent"]["w"]) for x in members)
        top = min(float(x["bbox_percent"]["y"]) for x in members) - float(pb["y"])
        bottom = float(pb["y"]) + float(pb["h"]) - max(float(x["bbox_percent"]["y"]) + float(x["bbox_percent"]["h"]) for x in members)
        usable = max(float(pb["w"]), 1e-6)
        if abs(left - right) / usable >= 0.14:
            feedback.append({"issue_type": "horizontal_imbalance", "severity": "major", "panel_id": panel.get("id"), "object_ids": [x.get("id") for x in members], "evidence": f"left_gap={left:.3f}, right_gap={right:.3f}"})
            intents.append({"action": "rebalance_panel", "panel_id": panel.get("id"), "object_ids": [x.get("id") for x in members], "axis": "horizontal", "reason": "balance panel internal margins"})
        if abs(top - bottom) / max(float(pb["h"]), 1e-6) >= 0.16:
            feedback.append({"issue_type": "vertical_imbalance", "severity": "major", "panel_id": panel.get("id"), "object_ids": [x.get("id") for x in members], "evidence": f"top_gap={top:.3f}, bottom_gap={bottom:.3f}"})
            intents.append({"action": "translate_group", "panel_id": panel.get("id"), "object_ids": [x.get("id") for x in members], "axis": "vertical", "amount": round((bottom - top) / 2, 4), "reason": "balance panel vertical margins"})
        occupied = sum(float(x["bbox_percent"]["w"]) * float(x["bbox_percent"]["h"]) for x in members) / max(float(pb["w"]) * float(pb["h"]), 1e-9)
        if occupied < 0.18:
            feedback.append({"issue_type": "excess_whitespace", "severity": "major", "panel_id": panel.get("id"), "object_ids": [x.get("id") for x in members], "evidence": f"node_occupancy={occupied:.3f}"})
            intents.append({"action": "replan_structure", "panel_id": panel.get("id"), "object_ids": [x.get("id") for x in members], "reason": "panel content is too sparse for bounded translation"})
    return {"status": "needs_repair" if feedback else "pass", "scores": {}, "feedback": feedback, "repair_intents": intents, "requires_replan": False}


def review_direct_draft(program: dict[str, Any], preview_path: str | Path, *, reference_path: str | Path | None = None, mode: str = "vlm", model: str | None = None, timeout: int = 180, retries: int = 1, call_metadata: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Review a rendered draft and return a validated, object-bound contract.

    ``call_metadata`` is an optional caller-owned dict the provider fills with the call's own
    timing and token counts. A repeat review of the same picture is served from a cache file,
    so the dict is told that too: a call that cost nothing must not read as if it cost a call.
    """
    if mode == "off":
        raw = {"status": "skipped", "feedback": [], "repair_intents": []}
    elif mode == "heuristic":
        raw = _heuristic_review(program)
    elif mode == "vlm":
        prompt = """You are the visual layout reviewer for an editable scientific PowerPoint draft. When two images are supplied, IMAGE 1 is the optional excellent reference and IMAGE 2 is the current PPT draft; when one image is supplied it is the current draft. Judge the current draft, using the reference only for transferable visual experience. Check panel balance, whitespace, crowding, readable text hierarchy, icon/text proportion, connector congestion, and whether connector labels fit their actual path segments. Do not change scientific content, IDs, or source-target relations. Return JSON only: {status:'pass|needs_repair|needs_replan|blocked', scores:{balance:0..1,readability:0..1,density:0..1,flow_clarity:0..1}, feedback:[{issue_type:'horizontal_imbalance|vertical_imbalance|crowding|excess_whitespace|text_too_small|text_hierarchy|connector_label_mismatch|connector_congestion|icon_text_ratio',severity:'minor|major|blocker',panel_id:'',object_ids:[],evidence:''}], repair_intents:[{action:'rebalance_panel|translate_group|distribute_horizontal|distribute_vertical|align_objects|resize_node|reserve_connector_lane|move_connector_label|replan_structure',panel_id:'',object_ids:[],axis:'horizontal|vertical',amount:0.0,target_bbox_percent:null,reason:''}], requires_replan:false}. Prefer bounded local repairs. COMPACT_PROGRAM=""" + json.dumps(_compact_program(program), ensure_ascii=False)
        image_paths = [preview_path]
        if reference_path and Path(reference_path).exists():
            image_paths.insert(0, reference_path)
        cache_key = hashlib.sha256((prompt + "|" + "|".join(str(Path(x).resolve()) + ":" + str(Path(x).stat().st_mtime_ns) for x in image_paths)).encode("utf-8")).hexdigest()
        cache_path = Path(preview_path).parent / f"visual_review_cache_{cache_key[:16]}.json"
        if cache_path.exists():
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            if call_metadata is not None:
                call_metadata.update({"cache_hit": True, "success": True, "elapsed_seconds": 0.0, "attempts": 0, "retries_used": 0})
        else:
            try:
                raw = call_vlm_json(prompt, image_paths, model=resolve_vlm_model("RFS_DIRECT_VISUAL_REVIEW_MODEL", "RFS_CRITIC_MODEL", explicit_model=model), timeout=timeout, retries=retries, call_metadata=call_metadata)
                cache_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                raw = {"status": "failed", "feedback": [], "repair_intents": [], "provider_error": str(exc)}
    else:
        raise ValueError(f"Unsupported visual review mode: {mode}")
    raw = {**raw, "panels": [{"id": item.get("id")} for item in program.get("panels", []) if isinstance(item, dict) and item.get("id")]}
    review = normalize_visual_review(raw)
    object_kinds = {}
    for key, kind in (("semantic_nodes", "node"), ("arrows", "arrow"), ("slots", "slot")):
        object_kinds.update({str(x.get("id")): kind for x in program.get(key, []) if isinstance(x, dict) and x.get("id")})
    for item in (program.get("text_program") or {}).get("items", []):
        if isinstance(item, dict) and item.get("id"):
            object_kinds[str(item.get("id"))] = "connector_label" if str(item.get("role") or item.get("text_role")) == "connector_label" else "text"
    object_ids = set(object_kinds)
    panel_ids = {str(x.get("id")) for x in program.get("panels", []) if isinstance(x, dict) and x.get("id")}
    validation = validate_visual_review(review, valid_object_ids=object_ids, valid_panel_ids=panel_ids, object_kinds=object_kinds)
    if not validation.get("ok"):
        warnings = []
        for item in review.get("feedback", []) + review.get("repair_intents", []):
            if item.get("panel_id") and item["panel_id"] not in panel_ids and item["panel_id"] not in {"root", "global", "canvas"}:
                warnings.append(f"unknown_panel_id:{item['panel_id']}")
                item["panel_id"] = ""
        for item in review.get("repair_intents", []):
            action = str(item.get("action") or "")
            if action in {"rebalance_panel", "translate_group", "distribute_horizontal", "distribute_vertical", "align_objects", "resize_node"}:
                original = list(item.get("object_ids", []))
                item["object_ids"] = [value for value in original if object_kinds.get(value) == "node"]
                dropped = [value for value in original if value not in item["object_ids"]]
                if dropped:
                    warnings.append(f"incompatible_objects:{action}:{','.join(dropped)}")
            if action in {"reserve_connector_lane", "move_connector_label"}:
                original = list(item.get("object_ids", []))
                item["object_ids"] = [value for value in original if object_kinds.get(value) in {"arrow", "connector_label"}]
                dropped = [value for value in original if value not in item["object_ids"]]
                if dropped:
                    warnings.append(f"incompatible_objects:{action}:{','.join(dropped)}")
        review["repair_intents"] = [
            item for item in review.get("repair_intents", [])
            if item.get("object_ids") or item.get("action") in {"replan_structure", "replan_colors"}
        ]
        review["normalization_warnings"] = warnings
        validation = validate_visual_review(review, valid_object_ids=object_ids, valid_panel_ids=panel_ids, object_kinds=object_kinds)
    report = {"mode": mode, "provider_error": raw.get("provider_error"), **validation}
    if call_metadata is not None:
        report["provider"] = dict(call_metadata)
    return review, report


__all__ = ["review_direct_draft"]
