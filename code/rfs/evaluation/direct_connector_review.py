"""VLM-assisted preference selection for deterministic connector candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..providers import call_vlm_json, resolve_vlm_model


def review_connector_routes(program: dict[str, Any], preview_path: str | Path, *, reference_path: str | Path | None = None, mode: str = "vlm", model: str | None = None, timeout: int = 180, retries: int = 1, call_metadata: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select only among collision-free route candidates; never invent a path.

    ``call_metadata`` is an optional caller-owned dict the provider fills with the call's own
    timing, so a round can show what this seat cost instead of only that it ran.
    """
    if mode == "off":
        return {"status": "skipped", "selections": []}, {"ok": True, "mode": "off", "errors": []}
    arrows = []
    for arrow in program.get("arrows", []) or []:
        if not isinstance(arrow, dict):
            continue
        route = arrow.get("route_selection") if isinstance(arrow.get("route_selection"), dict) else {}
        candidates = [item for item in route.get("candidates", []) if isinstance(item, dict) and int(item.get("obstacle_hits") or 0) == 0]
        if len(candidates) > 1:
            arrows.append({"arrow_id": arrow.get("id"), "source": arrow.get("source"), "target": arrow.get("target"), "relation_type": arrow.get("relation_type"), "candidates": [{key: value for key, value in item.items() if key != "path"} for item in candidates]})
    if not arrows:
        return {"status": "pass", "selections": []}, {"ok": True, "mode": mode, "errors": [], "candidate_count": 0}
    if mode == "heuristic":
        selections = [{"arrow_id": item["arrow_id"], "candidate_id": min(item["candidates"], key=lambda x: float(x.get("score") or 0)).get("candidate_id")} for item in arrows]
        return {"status": "pass", "selections": selections}, {"ok": True, "mode": "heuristic", "errors": [], "candidate_count": len(arrows)}
    prompt = "Select the most visually coherent route for each connector, but only from supplied collision-free candidates. Preserve source, target, relation_type, label, and scientific meaning. Prefer a straight link; a short stroke that is slightly off axis is fine, while a long diagonal, a detour past an unrelated box, or a run out to the slide border is not. Then prefer few turns and a short length. Return JSON only: {selections:[{arrow_id:'',candidate_id:'',reason:''}]}. ROUTE_CANDIDATES=" + json.dumps(arrows, ensure_ascii=False, separators=(",", ":"))
    images = [str(preview_path)]
    if reference_path and Path(reference_path).exists():
        images.insert(0, str(reference_path))
    try:
        raw = call_vlm_json(prompt, images, model=resolve_vlm_model("RFS_CONNECTOR_REVIEW_MODEL", "RFS_CRITIC_MODEL", explicit_model=model), timeout=timeout, retries=retries, call_metadata=call_metadata)
    except Exception as exc:
        return {"status": "failed", "selections": [], "provider_error": str(exc)}, {"ok": False, "mode": "vlm", "errors": [str(exc)], "candidate_count": len(arrows)}
    allowed = {str(item["arrow_id"]): {str(candidate.get("candidate_id")) for candidate in item["candidates"]} for item in arrows}
    selections = []
    for item in raw.get("selections", []) if isinstance(raw, dict) and isinstance(raw.get("selections"), list) else []:
        if not isinstance(item, dict):
            continue
        arrow_id, candidate_id = str(item.get("arrow_id") or ""), str(item.get("candidate_id") or "")
        if arrow_id in allowed and candidate_id in allowed[arrow_id]:
            selections.append({"arrow_id": arrow_id, "candidate_id": candidate_id, "reason": str(item.get("reason") or "")})
    report = {"ok": True, "mode": "vlm", "errors": [], "candidate_count": len(arrows), "selection_count": len(selections)}
    if call_metadata is not None:
        report["provider"] = dict(call_metadata)
    return {"status": "pass", "selections": selections}, report


__all__ = ["review_connector_routes"]
