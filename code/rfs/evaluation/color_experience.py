"""Audit propagation of retrieved color experience into the final figure program."""

from __future__ import annotations

from typing import Any


def evaluate_color_experience_usage(
    theme: dict[str, Any],
    figure_program: dict[str, Any] | None = None,
    *,
    supplied_recipe_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Report recipe decisions and exact theme propagation without rewriting output."""
    supplied = {str(value) for value in (supplied_recipe_ids or []) if str(value)}
    adoption = theme.get("experience_adoption") if isinstance(theme.get("experience_adoption"), dict) else {}
    adopted = [item for item in adoption.get("adopted", []) if isinstance(item, dict)]
    rejected = [item for item in adoption.get("rejected", []) if isinstance(item, dict)]
    mentioned = {str(item.get("recipe_id")) for item in adopted + rejected if item.get("recipe_id")}
    program_theme = (((figure_program or {}).get("style") or {}).get("color_theme") if isinstance((figure_program or {}).get("style"), dict) else None)
    panel_ids = set(theme.get("panels", {})) if isinstance(theme.get("panels"), dict) else set()
    program_panel_styles = (((figure_program or {}).get("style") or {}).get("panel_styles") if isinstance((figure_program or {}).get("style"), dict) else {}) or {}
    propagated = figure_program is None or program_theme == theme
    panel_propagated = figure_program is None or panel_ids.issubset(set(program_panel_styles))
    errors = []
    if supplied - mentioned:
        errors.append(f"recipes lack adoption decision: {sorted(supplied - mentioned)}")
    if mentioned - supplied:
        errors.append(f"unknown recipes referenced: {sorted(mentioned - supplied)}")
    if not propagated:
        errors.append("color_theme was not propagated unchanged into figure_program.style")
    if not panel_propagated:
        errors.append(f"panel styles missing in figure_program: {sorted(panel_ids - set(program_panel_styles))}")
    return {
        "summary": "Color experience usage and propagation audit.",
        "ok": not errors,
        "status": "pass" if not errors else "failed",
        "supplied_recipe_ids": sorted(supplied),
        "adopted": adopted,
        "rejected": rejected,
        "theme_role_mapping": theme.get("role_mapping") or {},
        "theme_panel_ids": sorted(panel_ids),
        "theme_propagated_to_figure_program": propagated,
        "panel_styles_propagated": panel_propagated,
        "errors": errors,
    }


__all__ = ["evaluate_color_experience_usage"]
