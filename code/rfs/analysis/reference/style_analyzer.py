"""Deterministic reference style/layout profile extraction."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from PIL import Image
import json
from ...providers import call_vlm_json, resolve_vlm_model
from ...contracts import normalize_reference_layout_template, validate_reference_layout_template
from ...contracts.vlm_context import compact_method, json_compact
from .figure_source import reference_image_for_analysis


def _fallback_template(ratio: float) -> dict[str, Any]:
    return normalize_reference_layout_template({
        "canvas": {"aspect_ratio": ratio, "reading_direction": "left_to_right", "outer_frame": "rounded"},
        "zones": [
            {"id": "zone_left", "relative_width": 0.3, "role": "input_or_context", "internal_pattern": "stacked_cards"},
            {"id": "zone_center", "relative_width": 0.4, "role": "core_method", "internal_pattern": "hierarchy_or_queue"},
            {"id": "zone_right", "relative_width": 0.3, "role": "evaluation_or_output", "internal_pattern": "stacked_evaluation_cards"},
        ],
        "object_patterns": ["rounded outer frame", "nested cards", "pill badges", "highlighted selected item"],
        "relation_patterns": [
            {"source_zone": "zone_left", "target_zone": "zone_center", "type": "primary_flow"},
            {"source_zone": "zone_center", "target_zone": "zone_right", "type": "evaluation_flow"},
            {"source_zone": "zone_right", "target_zone": "zone_left", "type": "optional_feedback"},
        ],
        "density_profile": {"target": "dense", "canvas_occupancy_min": 0.65, "panel_occupancy_min": 0.45, "max_empty_region_ratio": 0.3},
        "adoption_guidance": {"adopt": ["zone proportions", "nested hierarchy", "flow patterns", "content density"], "avoid_copying": ["reference scientific labels"]},
    })


def _mapping(value: Any) -> dict[str, Any]:
    """Accept only object-shaped VLM sections; malformed values use empty maps."""
    return value if isinstance(value, dict) else {}


def _pixel_palette(image: Image.Image, count: int = 6) -> list[str]:
    """Extract non-near-white dominant colors for a truthful local fallback."""
    reduced = image.convert("RGB").resize((160, 160)).quantize(colors=max(3, count * 3))
    palette = reduced.getpalette() or []
    ranked = sorted(reduced.getcolors() or [], reverse=True)
    colors: list[str] = []
    for _, index in ranked:
        rgb = tuple(palette[index * 3:index * 3 + 3])
        if len(rgb) != 3 or min(rgb) > 242 or max(rgb) < 28:
            continue
        value = "#%02X%02X%02X" % rgb
        if value not in colors:
            colors.append(value)
        if len(colors) >= count:
            break
    return colors or ["#607D8B", "#78909C", "#90A4AE"]


def analyze_reference_image(reference: str | Path | None, *, method: dict[str, Any], mode: str = "heuristic", model: str | None = None, work_dir: str | Path | None = None) -> dict[str, Any]:
    """Profile one reference figure, rendering a PDF asset before reading it."""
    if not reference or not Path(reference).exists():
        return {"status": "not_found", "style": {}, "layout": {}, "influence": {"adopted": [], "rejected": ["reference_missing"]}}
    path = Path(reference).resolve()
    asset = reference_image_for_analysis(path, work_dir=Path(work_dir) if work_dir else Path(tempfile.gettempdir()) / "rfs_reference_cache")
    if asset["status"] != "ok":
        return {"status": "failed", "error": str(asset.get("error")), "reference_asset": asset, "style": {}, "layout": {}, "influence": {"adopted": [], "rejected": ["reference_analysis_failed"]}}
    raster = Path(str(asset["image"]))
    try:
        with Image.open(raster) as image:
            width, height = image.size
            sample = image.convert("RGB").resize((1, 1)).getpixel((0, 0))
            extracted_palette = _pixel_palette(image)
        ratio = round(width / max(height, 1), 4)
        topology = str(method.get("topology") or "unknown")
        if mode == "vlm":
            prompt = f"""Analyze this framework figure only for reusable visual structure, not scientific semantics. Return JSON only with EXACT object-shaped fields:
{{"layout":{{"aspect_ratio":1.8,"macro_pattern":"...","topology_hint":"..."}},"style":{{"background_rgb":[255,255,255],"palette":["#RRGGBB"],"line_style":"...","node_style":"...","icon_style":"...","contrast_policy":"..."}},"influence":{{"adopted":[],"rejected":[],"semantic_source":"paper_contract_only"}},"reference_layout_template":{{"canvas":{{"reading_direction":"...","outer_frame":"..."}},"zones":[{{"id":"...","relative_width":0.5,"role":"...","internal_pattern":"...","vertical_layers":1}}],"object_patterns":[],"relation_patterns":[],"density_profile":{{}},"style_tokens":{{}},"adoption_guidance":{{"adopt":[],"avoid_copying":[]}}}}}}.
layout, style, influence, and reference_layout_template MUST be JSON objects; style.palette MUST be an array of valid #RRGGBB strings. Capture reusable zone roles and strengths, but do not require every reference zone in the target design. Do not copy labels. Compact paper semantics: {json_compact(compact_method(method))}"""
            raw = call_vlm_json(prompt, [raster], model=resolve_vlm_model("RFS_REFERENCE_STYLE_MODEL", explicit_model=model), timeout=180, retries=1)
            raw = _mapping(raw)
            fallback_layout = {"width": width, "height": height, "aspect_ratio": ratio, "macro_pattern": "wide_horizontal" if ratio >= 1.35 else "balanced", "topology_hint": topology}
            fallback_style = {"background_rgb": list(sample), "palette": extracted_palette, "line_style": "clean_thin", "node_style": "rounded_card", "icon_style": "flat_scientific"}
            layout = _mapping(raw.get("layout")) or fallback_layout
            raw_style = _mapping(raw.get("style"))
            valid_palette = [str(value).upper() for value in raw_style.get("palette", []) if isinstance(value, str) and len(value) == 7 and value.startswith("#")]
            style = {**fallback_style, **raw_style, "palette": valid_palette or extracted_palette}
            influence = _mapping(raw.get("influence")) or {"adopted": [], "rejected": [], "semantic_source": "paper_contract_only"}
            template = normalize_reference_layout_template(_mapping(raw.get("reference_layout_template")) or _fallback_template(ratio))
            complete = bool(_mapping(raw.get("layout")) and raw_style and valid_palette and _mapping(raw.get("influence")) and _mapping(raw.get("reference_layout_template")))
            return {"status": "complete" if complete else "partial", "reference": str(path), "reference_asset": asset, "layout": layout, "style": style, "influence": influence, "reference_layout_template": template, "reference_layout_template_validation": validate_reference_layout_template(template), "sources": {"layout": "vlm" if _mapping(raw.get("layout")) else "deterministic", "style": "vlm" if raw_style and valid_palette else "pixel_extraction", "template": "vlm" if _mapping(raw.get("reference_layout_template")) else "deterministic"}}
        template = _fallback_template(ratio)
        return {"status": "fallback", "reference": str(path), "reference_asset": asset, "layout": {"width": width, "height": height, "aspect_ratio": ratio, "macro_pattern": "wide_horizontal" if ratio >= 1.35 else "portrait" if ratio <= 0.8 else "balanced", "topology_hint": topology}, "style": {"background_rgb": list(sample), "palette": extracted_palette, "line_style": "clean_thin", "node_style": "rounded_card", "icon_style": "flat_scientific"}, "influence": {"adopted": ["aspect_ratio", "clean_thin_lines", "rounded_card_nodes", "flat_scientific_icon_style"], "rejected": ["reference_specific_labels", "reference_specific_scientific_relations"], "semantic_source": "paper_contract_only"}, "reference_layout_template": template, "reference_layout_template_validation": validate_reference_layout_template(template), "sources": {"layout": "deterministic", "style": "pixel_extraction", "template": "deterministic"}}
    except Exception as exc:
        return {"status": "failed", "error": str(exc), "reference_asset": asset, "style": {}, "layout": {}, "influence": {"adopted": [], "rejected": ["reference_analysis_failed"]}}
