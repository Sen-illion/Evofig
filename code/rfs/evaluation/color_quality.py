"""Deterministic quality checks for role-aware scientific color themes."""

from __future__ import annotations

import colorsys
from typing import Any

from .text_readability import _contrast
from ..contracts.color_theme import resolve_connector_color


def _hls(value: str) -> tuple[float, float, float]:
    raw = str(value or "#000000").lstrip("#")
    if len(raw) != 6:
        return 0.0, 0.0, 0.0
    rgb = tuple(int(raw[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return colorsys.rgb_to_hls(*rgb)


def evaluate_color_quality(program: dict[str, Any]) -> dict[str, Any]:
    """Report contrast, saturation, role consistency, and theme propagation."""
    style = program.get("style") if isinstance(program.get("style"), dict) else {}
    theme = style.get("color_theme") if isinstance(style.get("color_theme"), dict) else {}
    issues: list[dict[str, Any]] = []
    panel_styles = style.get("panel_styles") if isinstance(style.get("panel_styles"), dict) else {}
    hues: set[int] = set()
    minimum_contrast = 99.0
    for panel_id, expected in (theme.get("panels") or {}).items():
        actual = panel_styles.get(panel_id, {})
        for theme_key, actual_key in (("fill", "fill_color"), ("border", "stroke_color"), ("header", "header_color"), ("body_text", "text_color")):
            if str(actual.get(actual_key) or "").upper() != str(expected.get(theme_key) or "").upper():
                issues.append({"severity": "major", "role": f"panel.{panel_id}.{theme_key}", "reason": "compiled color differs from authoritative theme"})
        ratio = _contrast(str(expected.get("body_text") or "#000000"), str(expected.get("fill") or "#FFFFFF"))
        minimum_contrast = min(minimum_contrast, ratio)
        if ratio < 4.5:
            issues.append({"severity": "major", "role": f"panel.{panel_id}.body_text", "reason": "insufficient text contrast", "contrast_ratio": round(ratio, 2)})
        for key in ("fill", "border", "header"):
            color = str(expected.get(key) or "")
            h, _l, saturation = _hls(color)
            if saturation > 0.18:
                hues.add(round(h * 12))
            if color.upper() in {"#000000", "#FF0000"}:
                issues.append({"severity": "major", "role": f"panel.{panel_id}.{key}", "reason": "pure black/red used for a large structural role"})
    mismatched_arrows = [
        str(item.get("id"))
        for item in program.get("arrows", [])
        if isinstance(item, dict)
        and str(item.get("stroke_color") or "").upper() != resolve_connector_color(theme, item).upper()
    ]
    if mismatched_arrows:
        issues.append({"severity": "major", "role": "connectors.primary", "reason": "connector colors conflict with theme", "object_ids": mismatched_arrows})
    max_hues = int((theme.get("usage_policy") or {}).get("max_distinct_hues") or 4)
    if len(hues) > max_hues:
        issues.append({"severity": "minor", "role": "theme", "reason": "too many distinct structural hues", "distinct_hues": len(hues), "maximum": max_hues})
    status = "blocked" if any(x["severity"] == "major" for x in issues) else "warning" if issues else "pass"
    return {"summary": "Deterministic color quality report.", "status": status, "theme_source": style.get("color_theme_source"), "issue_count": len(issues), "issues": issues, "metrics": {"minimum_text_contrast": None if minimum_contrast == 99.0 else round(minimum_contrast, 2), "distinct_structural_hues": len(hues), "maximum_distinct_hues": max_hues, "connector_mismatch_count": len(mismatched_arrows)}}
