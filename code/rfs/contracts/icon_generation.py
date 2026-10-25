"""Stable contract for Image2-generated local visual assets."""

from __future__ import annotations

from typing import Any

ICON_GENERATION_CONTRACT_VERSION = 2

ICON_GRAMMARS = {"academic_flat_symbol", "bounded_mechanism_visual"}
ICON_COMPLEXITY_KINDS = {"standard_icon", "hero_visual"}


def normalize_icon_generation_plan(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    icons = source.get("icons") if isinstance(source.get("icons"), list) else []
    normalized = []
    for index, item in enumerate(icons, start=1):
        if not isinstance(item, dict):
            continue
        asset_id = str(item.get("asset_id") or f"icon_{index:02d}")
        complexity_kind = str(item.get("complexity_kind") or ("hero_visual" if str(item.get("visual_role") or "").casefold() in {"hero_visual", "large_visual", "wide_visual"} else "standard_icon"))
        grammar = str(item.get("icon_grammar") or ("bounded_mechanism_visual" if complexity_kind == "hero_visual" else "academic_flat_symbol"))
        normalized.append({**item, "asset_id": asset_id, "semantic_node_ids": [str(value) for value in item.get("semantic_node_ids", []) if str(value).strip()], "semantic_source_id": str(item.get("semantic_source_id") or ""), "prompt": str(item.get("prompt") or item.get("subject") or asset_id), "negative_prompt": str(item.get("negative_prompt") or "text, labels, arrows, formulas, watermark"), "panel_id": str(item.get("panel_id") or ""), "panel_background_color": str(item.get("panel_background_color") or ""), "background_policy": str(item.get("background_policy") or "transparent_panel_inherit"), "complexity_kind": complexity_kind, "icon_grammar": grammar, "primary_object": str(item.get("primary_object") or item.get("visual_metaphor") or item.get("subject") or asset_id), "supporting_object_limit": max(0, min(4, int(item.get("supporting_object_limit", 1 if complexity_kind == "standard_icon" else 3)))), "stroke_style": str(item.get("stroke_style") or "clean uniform medium outline"), "fill_style": str(item.get("fill_style") or "flat limited pastel fills"), "detail_policy": str(item.get("detail_policy") or "recognizable at small size; no decorative clutter")})
    return {"contract": "rfs.icon_generation_plan", "version": int(source.get("version") or ICON_GENERATION_CONTRACT_VERSION), "global_style": source.get("global_style") if isinstance(source.get("global_style"), dict) else {}, "icons": normalized, "provenance": source.get("provenance") if isinstance(source.get("provenance"), dict) else {}}


def validate_icon_generation_plan(plan: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_icon_generation_plan(plan)
    errors: list[str] = []
    ids = [str(item.get("asset_id") or "") for item in normalized["icons"]]
    if len(ids) != len(set(ids)):
        errors.append("icon asset IDs must be unique")
    for index, icon in enumerate(normalized["icons"]):
        if not icon["semantic_node_ids"]:
            errors.append(f"icon[{index}] requires semantic_node_ids")
        if not icon["prompt"].strip():
            errors.append(f"icon[{index}] requires prompt")
        if icon["background_policy"] not in {"transparent_panel_inherit", "exact_panel_color", "transparent_or_solid_removable"}:
            errors.append(f"icon[{index}] has unsupported background_policy")
        if icon["complexity_kind"] not in ICON_COMPLEXITY_KINDS:
            errors.append(f"icon[{index}] has unsupported complexity_kind")
        if icon["icon_grammar"] not in ICON_GRAMMARS:
            errors.append(f"icon[{index}] has unsupported icon_grammar")
        if not icon["primary_object"].strip():
            errors.append(f"icon[{index}] requires primary_object")
        if icon["complexity_kind"] == "standard_icon" and icon["supporting_object_limit"] > 2:
            errors.append(f"icon[{index}] standard_icon supporting_object_limit must be <= 2")
    return {"ok": not errors, "errors": errors, "icon_count": len(normalized["icons"])}
