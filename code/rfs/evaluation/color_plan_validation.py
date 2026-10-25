"""Pre-generation checks for VLM-authored color themes."""

from __future__ import annotations

import colorsys
from typing import Any

from ..contracts.color_theme import validate_color_theme
from .text_readability import _contrast


def _hls(value: str) -> tuple[float, float, float]:
    raw = str(value or "").lstrip("#")
    if len(raw) != 6:
        return 0.0, 0.0, 0.0
    try:
        rgb = tuple(int(raw[index:index + 2], 16) / 255.0 for index in (0, 2, 4))
    except ValueError:
        return 0.0, 0.0, 0.0
    return colorsys.rgb_to_hls(*rgb)


def validate_color_plan(
    theme: dict[str, Any],
    *,
    expected_panel_ids: set[str],
    expected_node_roles: set[str],
    expected_recipe_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Validate readability and palette behavior before Image2 is called."""
    contract = validate_color_theme(theme, expected_panel_ids, expected_node_roles, expected_recipe_ids)
    normalized = contract["theme"]
    errors = list(contract["errors"])
    warnings: list[str] = []
    structural_hues: set[int] = set()
    chromatic_surfaces = 0
    chromatic_structural_colors = 0
    minimum_contrast = 99.0
    for panel_id, panel in normalized["panels"].items():
        for text_key, surface_key, threshold in (("body_text", "fill", 4.5), ("header_text", "header", 3.0)):
            ratio = _contrast(str(panel.get(text_key) or "#000000"), str(panel.get(surface_key) or "#FFFFFF"))
            minimum_contrast = min(minimum_contrast, ratio)
            if ratio < threshold:
                errors.append(f"panel {panel_id} {text_key}/{surface_key} contrast {ratio:.2f} is below {threshold:.1f}")
        for key in ("fill", "border", "header"):
            hue, _lightness, saturation = _hls(str(panel.get(key) or ""))
            if saturation > 0.12:
                structural_hues.add(round(hue * 12) % 12)
                chromatic_structural_colors += 1
            if key == "fill" and saturation > 0.08:
                chromatic_surfaces += 1
    # No saturation ceiling is applied to a panel fill. The only saturation this module can
    # measure is HLS, which reads a near-white tint as strongly saturated (#F0FDFA measures
    # 0.77), so a ceiling on it rejected every pale surface a model wrote -- three of three
    # candidates in the 2026-10-10 probe -- and each paper fell back to the same
    # deterministic palette instead of the theme it had been given.
    max_hues = int((normalized.get("usage_policy") or {}).get("max_distinct_hues") or 4)
    if len(structural_hues) > max_hues:
        errors.append(f"structural hue count {len(structural_hues)} exceeds declared maximum {max_hues}")
    if normalized["panels"] and chromatic_structural_colors == 0:
        rationale = " ".join(str(value) for value in (normalized.get("palette_strategy") or {}).values()).casefold()
        if not any(token in rationale for token in ("neutral", "grayscale", "monochrome")):
            errors.append("all panel surfaces are neutral without an explicit neutral-palette rationale")
    distinct_groups = {
        str(normalized["role_mapping"].get(role) or role)
        for role in expected_node_roles
    }
    signatures = {
        (
            str((normalized["roles"].get(group) or {}).get("fill") or "").upper(),
            str((normalized["roles"].get(group) or {}).get("border") or "").upper(),
        )
        for group in distinct_groups
    }
    if len(distinct_groups) > 1 and len(signatures) == 1:
        warnings.append("all distinct semantic color groups use the same fill and border")
    self_check = normalized.get("self_check") or {}
    for key in ("all_panel_ids_covered", "all_node_roles_covered", "palette_is_harmonious", "contrast_requirements_met"):
        if self_check.get(key) is False:
            errors.append(f"VLM self_check reports {key}=false")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "theme": normalized,
        "metrics": {
            "minimum_contrast": None if minimum_contrast == 99.0 else round(minimum_contrast, 2),
            "structural_hue_count": len(structural_hues),
            "maximum_structural_hues": max_hues,
            "chromatic_panel_surface_count": chromatic_surfaces,
        },
    }


__all__ = ["validate_color_plan"]
