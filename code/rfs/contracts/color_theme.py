"""Stable contract and validation for VLM-authored figure color themes."""

from __future__ import annotations

import re
from typing import Any

HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def normalize_color_theme(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    src = raw if isinstance(raw, dict) else {}
    panels = src.get("panels", {})
    if isinstance(panels, list):
        panels = {str(item.get("panel_id")): item for item in panels if isinstance(item, dict) and item.get("panel_id")}
    return {
        "contract": "rfs.color_theme",
        "version": 2,
        "theme_name": str(src.get("theme_name") or "unnamed_theme"),
        "theme_intent": str(src.get("theme_intent") or "clean academic"),
        "palette_strategy": src.get("palette_strategy") if isinstance(src.get("palette_strategy"), dict) else {},
        "canvas_background": str(src.get("canvas_background") or "#F8FAFC"),
        "text": src.get("text") if isinstance(src.get("text"), dict) else {},
        "connectors": src.get("connectors") if isinstance(src.get("connectors"), dict) else {},
        "panels": panels,
        "role_mapping": src.get("role_mapping") if isinstance(src.get("role_mapping"), dict) else {},
        "roles": src.get("roles") if isinstance(src.get("roles"), dict) else {},
        "accents": src.get("accents") if isinstance(src.get("accents"), list) else [],
        "usage_policy": src.get("usage_policy") if isinstance(src.get("usage_policy"), dict) else {},
        "experience_adoption": src.get("experience_adoption") if isinstance(src.get("experience_adoption"), dict) else {"adopted": [], "rejected": []},
        "self_check": src.get("self_check") if isinstance(src.get("self_check"), dict) else {},
        "provenance": src.get("provenance") if isinstance(src.get("provenance"), dict) else {},
    }


def resolve_role_style(theme: dict[str, Any], actual_role: Any) -> dict[str, Any]:
    """Resolve one design role through the VLM-authored semantic mapping."""
    role = str(actual_role or "").strip()
    roles = theme.get("roles") if isinstance(theme.get("roles"), dict) else {}
    mapping = theme.get("role_mapping") if isinstance(theme.get("role_mapping"), dict) else {}
    semantic_group = str(mapping.get(role) or role).strip()
    style = roles.get(semantic_group)
    return style if isinstance(style, dict) else {}


def resolve_connector_color(theme: dict[str, Any], connector: dict[str, Any]) -> str:
    """Resolve primary, secondary, or feedback color from connector semantics."""
    connectors = theme.get("connectors") if isinstance(theme.get("connectors"), dict) else {}
    meaning = " ".join(str(connector.get(key) or "") for key in ("relation_type", "semantic_role", "route_kind", "kind")).casefold()
    if any(token in meaning for token in ("feedback", "loop", "iterate", "revision")):
        return str(connectors.get("feedback") or connectors.get("primary") or "#54758A")
    if any(token in meaning for token in ("secondary", "support", "context", "optional")):
        return str(connectors.get("secondary") or connectors.get("primary") or "#54758A")
    return str(connectors.get("primary") or "#54758A")


def validate_color_theme(
    theme: dict[str, Any],
    expected_panel_ids: set[str] | None = None,
    expected_node_roles: set[str] | None = None,
    expected_recipe_ids: set[str] | None = None,
) -> dict[str, Any]:
    normalized = normalize_color_theme(theme)
    errors: list[str] = []
    colors: list[tuple[str, Any]] = [("canvas_background", normalized["canvas_background"])]
    for key in ("primary", "secondary", "muted", "inverse"):
        if not normalized["text"].get(key):
            errors.append(f"text missing {key}")
    for key in ("primary", "secondary", "feedback"):
        if not normalized["connectors"].get(key):
            errors.append(f"connectors missing {key}")
    for role, semantic_group in normalized["role_mapping"].items():
        if not isinstance(role, str) or not role.strip():
            errors.append("role_mapping contains an empty role")
        if not isinstance(semantic_group, str) or not semantic_group.strip():
            errors.append(f"role_mapping {role} has an empty semantic group")
        elif not isinstance(normalized["roles"].get(semantic_group), dict):
            errors.append(f"role_mapping {role} targets missing role style: {semantic_group}")
    if expected_node_roles:
        for role in sorted(expected_node_roles):
            if not resolve_role_style(normalized, role):
                errors.append(f"missing node role theme: {role}")
    for group in ("text", "connectors", "roles"):
        for key, value in normalized[group].items():
            if isinstance(value, dict):
                colors.extend((f"{group}.{key}.{k}", v) for k, v in value.items() if isinstance(v, str) and k != "allowed_usage")
            elif isinstance(value, str):
                colors.append((f"{group}.{key}", value))
    for panel_id, panel in normalized["panels"].items():
        if not isinstance(panel, dict):
            errors.append(f"panel {panel_id} must be an object")
            continue
        for key in ("fill", "border", "header", "header_text", "body_text"):
            value = panel.get(key)
            if not value:
                errors.append(f"panel {panel_id} missing {key}")
            else:
                colors.append((f"panels.{panel_id}.{key}", value))
        if str(panel.get("fill", "")).upper() in {"#000000", "#FF0000"} or str(panel.get("border", "")).upper() in {"#000000", "#FF0000"}:
            errors.append(f"panel {panel_id} uses prohibited pure black/red surface or border")
    if expected_panel_ids:
        errors.extend(f"missing panel theme: {item}" for item in sorted(expected_panel_ids - set(normalized["panels"])))
    if expected_recipe_ids:
        adoption = normalized["experience_adoption"]
        adopted = adoption.get("adopted") if isinstance(adoption.get("adopted"), list) else []
        rejected = adoption.get("rejected") if isinstance(adoption.get("rejected"), list) else []
        mentioned: set[str] = set()
        for status, items in (("adopted", adopted), ("rejected", rejected)):
            for item in items:
                if not isinstance(item, dict):
                    errors.append(f"experience_adoption.{status} entries must be objects")
                    continue
                recipe_id = str(item.get("recipe_id") or "")
                if recipe_id not in expected_recipe_ids:
                    errors.append(f"experience_adoption references unknown recipe: {recipe_id or '<empty>'}")
                else:
                    mentioned.add(recipe_id)
                if status == "adopted" and (not item.get("principle") or not item.get("applied_to")):
                    errors.append(f"adopted recipe {recipe_id} must state principle and applied_to")
                if status == "rejected" and not item.get("reason"):
                    errors.append(f"rejected recipe {recipe_id} must state a reason")
        errors.extend(f"experience recipe not adopted or rejected: {item}" for item in sorted(expected_recipe_ids - mentioned))
    for name, value in colors:
        if not isinstance(value, str) or not HEX_RE.fullmatch(value):
            errors.append(f"{name} must be #RRGGBB")
    return {
        "ok": not errors,
        "errors": errors,
        "panel_count": len(normalized["panels"]),
        "node_role_count": len(expected_node_roles or set()),
        "theme": normalized,
    }
