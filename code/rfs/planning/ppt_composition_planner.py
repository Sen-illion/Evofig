"""VLM final composition planner invoked after real Icon generation."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from PIL import Image
from ..providers import call_vlm_json, resolve_vlm_model
from ..contracts import normalize_ppt_composition_plan, validate_ppt_composition_plan
from ..contracts.vlm_context import compact_reference, context_stats, json_compact
from .prompt_assembly import render_patch_block


def _image_dimensions(path: Any) -> tuple[int, int]:
    try:
        with Image.open(str(path)) as image:
            return image.size
    except Exception:
        return (0, 0)


def plan_ppt_composition_vlm(
    design: dict[str, Any],
    text_plan: dict[str, Any],
    icon_report: dict[str, Any],
    reference: str | Path | None,
    *,
    reference_template: dict[str, Any] | None = None,
    exemplar_style_brief: dict[str, Any] | None = None,
    model: str | None = None,
    timeout: int = 180,
    retries: int = 1,
    visual_review: dict[str, Any] | None = None,
    color_theme: dict[str, Any] | None = None,
    prompt_patches: list[dict[str, Any]] | None = None,
):
    icons = [
        str(x.get("path"))
        for x in icon_report.get("assets", [])
        if x.get("path") and Path(str(x["path"])).exists()
    ]
    icon_manifest = [
        {"asset_id": str(item.get("asset_id")), "node_id": str(item.get("node_id") or ""), "path": str(item.get("path")), "width_px": _image_dimensions(item.get("path"))[0], "height_px": _image_dimensions(item.get("path"))[1]}
        for item in icon_report.get("assets", []) if item.get("asset_id") and item.get("path")
    ]
    if not icons:
        return {}, {"ok": False, "mode": "vlm", "fallback_used": True, "fallback_reason": "no generated icon files available", "icon_input_count": 0}
    prompt = (
        "Return exactly one non-empty JSON object matching {panels:[],nodes:[],text_items:[],icon_placements:[{asset_id,node_id,bbox_percent}],connectors:[],z_order:[],adjustments:[],reference_adoption:{adopted_patterns:[],rejected_patterns:[]}}. Place every actual generated Icon listed in ICON_MANIFEST using DESIGN and TEXT; copy asset_id exactly, include one icon_placement per icon, preserve semantic IDs, and provide legal bbox_percent for every placed object. COLOR_THEME is authoritative: do not invent, replace, or override colors. Use REFERENCE_TEMPLATE as a reusable structural prior for zone proportions, nested hierarchy, density, object patterns and feedback flow, but do not copy its scientific labels. Keep Icon and editable text regions disjoint, size Icons according to their actual pixel dimensions, and avoid oversized empty cards. Do not invent semantics or PPT code. ICON_MANIFEST="
        + json_compact(icon_manifest)
        + " DESIGN="
        + json_compact({"canvas": design.get("canvas"), "panels": design.get("panels"), "nodes": design.get("nodes"), "connectors": design.get("connectors")})
        + " TEXT="
        + json_compact({"items": text_plan.get("items", []), "text_policy": text_plan.get("text_policy", {})})
        + render_patch_block(prompt_patches)
        + " REFERENCE_TEMPLATE="
        + json_compact(reference_template or {})
        + " EXEMPLAR_STYLE_BRIEF="
        + json_compact(compact_reference({}, exemplar_style_brief))
        + " VISUAL_REVIEW_FEEDBACK="
        + json_compact(visual_review or {})
        + " COLOR_THEME="
        + json_compact(color_theme or {})
    )
    raw = None
    provider_metadata: dict[str, Any] = {}
    try:
        raw = call_vlm_json(
            prompt,
            [reference] if reference else [],
            model=resolve_vlm_model("RFS_PPT_PLAN_MODEL", explicit_model=model),
            timeout=timeout,
            retries=retries,
            call_metadata=provider_metadata,
        )
        plan = normalize_ppt_composition_plan(raw)
        validation = validate_ppt_composition_plan(plan, expected_asset_ids={str(x.get("asset_id")) for x in icon_report.get("assets", []) if x.get("asset_id")})
        if not validation["ok"]:
            raise ValueError("invalid PPT composition plan: " + "; ".join(validation["errors"]))
        return plan, {
            **validation,
            "mode": "vlm",
            "fallback_used": False,
            "icon_input_count": len(icons),
            "raw_response": raw,
            "context": context_stats(prompt, image_count=1 if reference else 0, field_names=["icon_manifest", "design_geometry", "text_geometry", "reference_template", "style_rules", "color_theme"]),
            "provider": provider_metadata,
        }
    except Exception as exc:
        return {}, {
            "ok": False,
            "mode": "vlm",
            "fallback_used": True,
            "fallback_reason": str(exc),
            "icon_input_count": len(icons),
            "raw_response": raw,
            "context": context_stats(prompt, image_count=1 if reference else 0, field_names=["icon_manifest", "design_geometry", "text_geometry", "reference_template", "style_rules", "color_theme"]),
            "provider": provider_metadata,
        }
