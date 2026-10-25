"""Apply one authoritative color theme to editable figure-program objects."""

from __future__ import annotations

from typing import Any

from ..contracts.color_theme import resolve_connector_color, resolve_role_style


def apply_color_theme_to_program(program: dict[str, Any], panel_style_plan: dict[str, Any] | None) -> dict[str, Any]:
    """Propagate panel, role, connector, and text colors without changing geometry."""
    plan = panel_style_plan if isinstance(panel_style_plan, dict) else {}
    theme = plan.get("color_theme") if isinstance(plan.get("color_theme"), dict) else {}
    panel_specs = {str(item.get("panel_id")): item for item in plan.get("panels", []) if isinstance(item, dict) and item.get("panel_id")}
    style = program.get("style") if isinstance(program.get("style"), dict) else {}
    style["panel_styles"] = {
        panel_id: {
            "fill_color": spec.get("fill_color_hex"),
            "stroke_color": spec.get("border_color_hex"),
            "header_color": spec.get("header_color_hex") or spec.get("accent_color_hex"),
            "header_text_color": spec.get("header_text_color_hex"),
            "text_color": spec.get("text_color_hex"),
        }
        for panel_id, spec in panel_specs.items()
    }
    style["visual_style_contract"] = plan
    if theme:
        style["color_theme"] = theme
        style["color_theme_source"] = str((theme.get("provenance") or {}).get("source") or "unknown")
        style["palette"] = [str(item.get("border")) for item in theme.get("panels", {}).values() if isinstance(item, dict) and item.get("border")]
        program.setdefault("canvas", {})["background"] = theme.get("canvas_background") or program.get("canvas", {}).get("background")
    program["style"] = style
    panels = {str(item.get("id")): item for item in program.get("panels", []) if isinstance(item, dict)}
    for panel_id, panel in panels.items():
        spec = panel_specs.get(panel_id)
        if spec:
            panel.update({"fill_color": spec.get("fill_color_hex"), "stroke_color": spec.get("border_color_hex"), "text_color": spec.get("text_color_hex"), "style_locked_for_icons": True})
    nodes = {str(item.get("id")): item for item in program.get("semantic_nodes", []) if isinstance(item, dict)}
    nodes.update({str(item.get("semantic_source_id")): item for item in program.get("semantic_nodes", []) if isinstance(item, dict) and item.get("semantic_source_id")})
    for node in program.get("semantic_nodes", []):
        if not isinstance(node, dict):
            continue
        panel_spec = panel_specs.get(str(node.get("panel_id"))) or {}
        role_style = resolve_role_style(theme, node.get("role"))
        node["text_color"] = role_style.get("text") or panel_spec.get("text_color_hex") or node.get("text_color") or "#173042"
        node["fill_color"] = role_style.get("fill") or panel_spec.get("fill_color_hex") or node.get("fill_color") or "#F5FAFC"
        node["stroke_color"] = role_style.get("border") or panel_spec.get("border_color_hex") or node.get("stroke_color") or "#1F6F8B"
    for arrow in program.get("arrows", []):
        if isinstance(arrow, dict) and theme:
            arrow["stroke_color"] = resolve_connector_color(theme, arrow)
    primary_text = str((theme.get("text") or {}).get("primary") or "#263747")
    for item in ((program.get("text_program") or {}).get("items") or []):
        if not isinstance(item, dict):
            continue
        role = str(item.get("text_role") or item.get("role") or "").casefold()
        target_id = str(item.get("target_id") or item.get("semantic_source_id") or "")
        if role == "panel_title" and target_id in panel_specs:
            color = panel_specs[target_id].get("header_text_color_hex")
        elif role in {"node_label", "primary_node", "secondary_node"} and target_id in nodes:
            color = nodes[target_id].get("text_color")
        elif role in {"connector_label", "relation_label", "arrow_label"}:
            color = (theme.get("text") or {}).get("secondary") or primary_text
        else:
            color = (panel_specs.get(str(item.get("panel_id") or "")) or {}).get("text_color_hex") or primary_text
        item["color_hex"] = str(color or primary_text)
        item["color_source"] = "authoritative_color_theme"
    return program


__all__ = ["apply_color_theme_to_program"]
