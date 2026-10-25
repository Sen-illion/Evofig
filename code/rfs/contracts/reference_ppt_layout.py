"""Stable contract for deterministic facts extracted from PPTX exemplars."""

from __future__ import annotations

import re
from typing import Any


EVIDENCE_MODES = {"editable_objects", "raster_only"}
HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bbox(value: Any) -> dict[str, float]:
    source = value if isinstance(value, dict) else {}
    return {key: _float(source.get(key)) for key in ("x", "y", "w", "h")}


def normalize_reference_ppt_layout_facts(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize factual PPT object evidence without adding visual judgments."""

    source = raw if isinstance(raw, dict) else {}
    slides = []
    for item in source.get("slides", []) if isinstance(source.get("slides"), list) else []:
        if not isinstance(item, dict):
            continue
        root_regions = []
        for region in item.get("root_regions", []) if isinstance(item.get("root_regions"), list) else []:
            if not isinstance(region, dict):
                continue
            root_regions.append({
                "shape_id": _integer(region.get("shape_id")),
                "name": str(region.get("name") or ""),
                "kind": str(region.get("kind") or "unknown"),
                "bbox_percent": _bbox(region.get("bbox_percent")),
                "text_preview": str(region.get("text_preview") or ""),
                "descendant_counts": {
                    str(key): _integer(value)
                    for key, value in (region.get("descendant_counts") or {}).items()
                } if isinstance(region.get("descendant_counts"), dict) else {},
            })
        picture_assets = []
        for asset in item.get("picture_assets", []) if isinstance(item.get("picture_assets"), list) else []:
            if not isinstance(asset, dict):
                continue
            picture_assets.append({
                "shape_id": _integer(asset.get("shape_id")),
                "name": str(asset.get("name") or ""),
                "containing_group_shape_id": _integer(asset.get("containing_group_shape_id")),
                "bbox_percent": _bbox(asset.get("bbox_percent")),
                "image_sha256": str(asset.get("image_sha256") or ""),
                "image_extension": str(asset.get("image_extension") or ""),
                "pixel_width": _integer(asset.get("pixel_width")),
                "pixel_height": _integer(asset.get("pixel_height")),
            })
        colors = []
        for color in item.get("object_color_frequency", []) if isinstance(item.get("object_color_frequency"), list) else []:
            if isinstance(color, dict):
                colors.append({
                    "hex": str(color.get("hex") or "").upper(),
                    "object_occurrences": _integer(color.get("object_occurrences")),
                })
        typography = item.get("typography") if isinstance(item.get("typography"), dict) else {}
        slides.append({
            "slide_index": _integer(item.get("slide_index")),
            "evidence_mode": str(item.get("evidence_mode") or ""),
            "trusted_for_layout_geometry": bool(item.get("trusted_for_layout_geometry")),
            "root_shape_count": _integer(item.get("root_shape_count")),
            "recursive_object_count": _integer(item.get("recursive_object_count")),
            "maximum_group_depth": _integer(item.get("maximum_group_depth")),
            "object_counts": {
                str(key): _integer(value)
                for key, value in (item.get("object_counts") or {}).items()
            } if isinstance(item.get("object_counts"), dict) else {},
            "editable_object_ratio": _float(item.get("editable_object_ratio")),
            "native_editable_object_count": _integer(item.get("native_editable_object_count")),
            "picture_object_count": _integer(item.get("picture_object_count")),
            "root_region_occupancy": _float(item.get("root_region_occupancy")),
            "root_regions": root_regions,
            "picture_assets": picture_assets,
            "typography": {
                "explicit_run_count": _integer(typography.get("explicit_run_count")),
                "minimum_pt": typography.get("minimum_pt"),
                "median_pt": typography.get("median_pt"),
                "maximum_pt": typography.get("maximum_pt"),
            },
            "object_color_frequency": colors,
            "learning_policy": str(item.get("learning_policy") or ""),
        })
    canvas = source.get("canvas") if isinstance(source.get("canvas"), dict) else {}
    source_info = source.get("source") if isinstance(source.get("source"), dict) else {}
    safety = source.get("safety") if isinstance(source.get("safety"), dict) else {}
    return {
        "contract": "rfs.reference_ppt_layout_facts",
        "version": 1,
        "summary": str(source.get("summary") or "Deterministic PPTX exemplar object facts."),
        "source": {
            "path": str(source_info.get("path") or ""),
            "sha256": str(source_info.get("sha256") or ""),
            "total_slide_count": _integer(source_info.get("total_slide_count")),
        },
        "canvas": {
            "width_inches": _float(canvas.get("width_inches")),
            "height_inches": _float(canvas.get("height_inches")),
            "aspect_ratio": _float(canvas.get("aspect_ratio")),
        },
        "selected_slide_count": len(slides),
        "slides": slides,
        "safety": {
            "scientific_semantics_learned": bool(safety.get("scientific_semantics_learned")),
            "exact_text_reusable": bool(safety.get("exact_text_reusable")),
            "raster_only_geometry_reusable": bool(safety.get("raster_only_geometry_reusable")),
        },
    }


def validate_reference_ppt_layout_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Validate source provenance, slide evidence modes, counts, and geometry."""

    value = normalize_reference_ppt_layout_facts(facts)
    errors: list[str] = []
    source = value["source"]
    if not re.fullmatch(r"[0-9a-fA-F]{64}", source["sha256"]):
        errors.append("source.sha256 must contain 64 hexadecimal characters")
    if source["total_slide_count"] < value["selected_slide_count"]:
        errors.append("source.total_slide_count cannot be smaller than selected_slide_count")
    canvas = value["canvas"]
    if canvas["width_inches"] <= 0 or canvas["height_inches"] <= 0 or canvas["aspect_ratio"] <= 0:
        errors.append("canvas dimensions and aspect_ratio must be positive")
    if not value["slides"]:
        errors.append("at least one selected slide is required")
    indices: set[int] = set()
    for slide in value["slides"]:
        index = slide["slide_index"]
        if index <= 0 or index in indices:
            errors.append(f"invalid or duplicate slide_index: {index}")
        indices.add(index)
        if slide["evidence_mode"] not in EVIDENCE_MODES:
            errors.append(f"slide {index} has unsupported evidence_mode")
        if slide["evidence_mode"] == "raster_only" and slide["trusted_for_layout_geometry"]:
            errors.append(f"slide {index} raster_only evidence cannot be trusted for layout geometry")
        if slide["root_shape_count"] != len(slide["root_regions"]):
            errors.append(f"slide {index} root_shape_count does not match root_regions")
        if slide["recursive_object_count"] < slide["root_shape_count"]:
            errors.append(f"slide {index} recursive_object_count is too small")
        if slide["native_editable_object_count"] + slide["picture_object_count"] != slide["recursive_object_count"]:
            errors.append(f"slide {index} native/picture counts do not sum to recursive_object_count")
        if slide["picture_object_count"] != len(slide["picture_assets"]):
            errors.append(f"slide {index} picture_object_count does not match picture_assets")
        if not 0.0 <= slide["editable_object_ratio"] <= 1.0:
            errors.append(f"slide {index} editable_object_ratio must be in 0..1")
        if not 0.0 <= slide["root_region_occupancy"] <= 1.0:
            errors.append(f"slide {index} root_region_occupancy must be in 0..1")
        for region in slide["root_regions"]:
            box = region["bbox_percent"]
            if box["w"] < 0 or box["h"] < 0 or box["x"] < 0 or box["y"] < 0:
                errors.append(f"slide {index} region {region['shape_id']} has negative geometry")
            if box["x"] + box["w"] > 1.001 or box["y"] + box["h"] > 1.001:
                errors.append(f"slide {index} region {region['shape_id']} exceeds the canvas")
        for asset in slide["picture_assets"]:
            if not re.fullmatch(r"[0-9a-fA-F]{64}", asset["image_sha256"]):
                errors.append(f"slide {index} picture {asset['shape_id']} has invalid image_sha256")
            if asset["pixel_width"] <= 0 or asset["pixel_height"] <= 0:
                errors.append(f"slide {index} picture {asset['shape_id']} has invalid pixel dimensions")
        for color in slide["object_color_frequency"]:
            if not HEX_RE.fullmatch(color["hex"]):
                errors.append(f"slide {index} contains invalid color {color['hex']}")
            if color["object_occurrences"] <= 0:
                errors.append(f"slide {index} color occurrences must be positive")
    safety = value["safety"]
    if safety["scientific_semantics_learned"] or safety["exact_text_reusable"] or safety["raster_only_geometry_reusable"]:
        errors.append("PPT layout facts must not authorize semantic/text/raster-geometry reuse")
    return {
        "ok": not errors,
        "errors": errors,
        "facts": value,
        "selected_slide_count": len(value["slides"]),
    }


__all__ = [
    "EVIDENCE_MODES",
    "normalize_reference_ppt_layout_facts",
    "validate_reference_ppt_layout_facts",
]
