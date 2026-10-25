"""Contract validation for post-Icon VLM PowerPoint composition plans."""
from __future__ import annotations
from typing import Any
from .design_plan import _normalize_bbox, _valid_bbox


def normalize_ppt_composition_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    source = plan if isinstance(plan, dict) else {}
    normalized = dict(source)
    for key in ("panels", "nodes", "text_items", "icon_placements"):
        items = []
        for item in source.get(key, []):
            if not isinstance(item, dict):
                continue
            normalized_item = {**item, "bbox_percent": _normalize_bbox(item.get("bbox_percent"))}
            if key == "panels" and not str(normalized_item.get("title") or "").strip():
                normalized_item["title"] = str(
                    normalized_item.get("label")
                    or normalized_item.get("name")
                    or normalized_item.get("id")
                    or "Panel"
                ).replace("_", " ").title()
            items.append(normalized_item)
        normalized[key] = items
    return normalized

def validate_ppt_composition_plan(plan: dict[str, Any], *, expected_asset_ids: set[str] | None = None) -> dict[str, Any]:
    errors=[]; source=normalize_ppt_composition_plan(plan)
    if not source:
        errors.append("composition plan must be a non-empty object")
    placements=[x for x in source.get("icon_placements",[]) if isinstance(x,dict)]
    placed={str(x.get("asset_id")) for x in placements if x.get("asset_id")}; missing=sorted((expected_asset_ids or set())-placed)
    for i,item in enumerate(placements):
        bbox=item.get("bbox_percent")
        if not item.get("asset_id") or not _valid_bbox(bbox): errors.append(f"icon_placement[{i}] requires asset_id and legal bbox_percent")
    if missing: errors.append("missing icon placements: "+", ".join(missing))
    if expected_asset_ids and not placements: errors.append("composition requires icon placements")
    return {"ok":not errors,"errors":errors,"icon_placement_count":len(placements),"missing_asset_ids":missing}
