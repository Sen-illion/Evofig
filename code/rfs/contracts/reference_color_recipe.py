"""Stable contract for reusable, evidence-backed reference color recipes."""

from __future__ import annotations

import re
from typing import Any


HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

#: Upper bound on a recipe's palette. It matches
#: ``rfs.analysis.reference.color_recipe.MAX_REPRESENTATIVES``: a figure that
#: carries more distinct surfaces than this (a dominant field, several panel
#: tints, a couple of accents) is summarised by folding the smallest bucket into
#: its nearest neighbour, and the contract has to allow what the extractor emits.
MAX_REPRESENTATIVE_COLORS = 16
ALLOWED_USES = {
    "canvas",
    "panel_fill",
    "panel_header",
    "node_fill",
    "border",
    "text",
    "primary_connector",
    "feedback_connector",
    "accent",
    "unassigned",
}


def normalize_reference_color_recipe(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize deterministic color evidence and optional VLM semantics."""
    src = raw if isinstance(raw, dict) else {}
    colors = []
    for index, item in enumerate(src.get("colors", []) if isinstance(src.get("colors"), list) else []):
        if not isinstance(item, dict):
            continue
        colors.append({
            "color_id": str(item.get("color_id") or f"c{index + 1:02d}"),
            "hex": str(item.get("hex") or "").upper(),
            "rgb": list(item.get("rgb") or [])[:3],
            "lab": list(item.get("lab") or [])[:3],
            "hsl": list(item.get("hsl") or [])[:3],
            "area_ratio": float(item.get("area_ratio") or 0.0),
            "spatial_zones": [str(value) for value in item.get("spatial_zones", []) if str(value).strip()],
            "used_for": [str(value) for value in item.get("used_for", []) if str(value).strip()],
            "semantic_roles": [str(value) for value in item.get("semantic_roles", []) if str(value).strip()],
            "confidence": float(item.get("confidence") or 0.0),
        })
    return {
        "contract": "rfs.reference_color_recipe",
        "version": 1,
        "recipe_id": str(src.get("recipe_id") or ""),
        "source_path": str(src.get("source_path") or ""),
        "source_sha256": str(src.get("source_sha256") or ""),
        "quality_label": str(src.get("quality_label") or "unknown"),
        "image": src.get("image") if isinstance(src.get("image"), dict) else {},
        "background_color_id": str(src.get("background_color_id") or ""),
        "colors": colors,
        "relationships": src.get("relationships") if isinstance(src.get("relationships"), dict) else {},
        "transfer_rules": [str(value) for value in src.get("transfer_rules", []) if str(value).strip()],
        "avoid_copying": [str(value) for value in src.get("avoid_copying", []) if str(value).strip()],
        "analysis": src.get("analysis") if isinstance(src.get("analysis"), dict) else {},
    }


def validate_reference_color_recipe(recipe: dict[str, Any], *, require_semantics: bool = False) -> dict[str, Any]:
    """Validate pixel evidence and, when requested, VLM usage assignments."""
    normalized = normalize_reference_color_recipe(recipe)
    errors: list[str] = []
    if not normalized["recipe_id"]:
        errors.append("recipe_id is required")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", normalized["source_sha256"]):
        errors.append("source_sha256 must contain 64 hexadecimal characters")
    colors = normalized["colors"]
    if not 2 <= len(colors) <= MAX_REPRESENTATIVE_COLORS:
        errors.append(f"colors must contain 2-{MAX_REPRESENTATIVE_COLORS} representative colors")
    ids = {item["color_id"] for item in colors}
    if len(ids) != len(colors):
        errors.append("color_id values must be unique")
    if normalized["background_color_id"] not in ids:
        errors.append("background_color_id must reference a representative color")
    area_sum = 0.0
    for item in colors:
        if not HEX_RE.fullmatch(item["hex"]):
            errors.append(f"{item['color_id']}.hex must be #RRGGBB")
        if len(item["rgb"]) != 3 or len(item["lab"]) != 3 or len(item["hsl"]) != 3:
            errors.append(f"{item['color_id']} must include RGB, Lab, and HSL triples")
        if not 0.0 <= item["area_ratio"] <= 1.0:
            errors.append(f"{item['color_id']}.area_ratio must be in 0..1")
        area_sum += item["area_ratio"]
        invalid_uses = set(item["used_for"]) - ALLOWED_USES
        if invalid_uses:
            errors.append(f"{item['color_id']} contains unsupported uses: {sorted(invalid_uses)}")
        if require_semantics and (not item["used_for"] or item["used_for"] == ["unassigned"]):
            errors.append(f"{item['color_id']} has no semantic usage assignment")
    if colors and not 0.98 <= area_sum <= 1.02:
        errors.append(f"area ratios must sum to approximately 1.0, got {area_sum:.4f}")
    return {"ok": not errors, "errors": errors, "recipe": normalized, "color_count": len(colors), "area_ratio_sum": round(area_sum, 4)}


__all__ = ["ALLOWED_USES", "MAX_REPRESENTATIVE_COLORS", "normalize_reference_color_recipe", "validate_reference_color_recipe"]
