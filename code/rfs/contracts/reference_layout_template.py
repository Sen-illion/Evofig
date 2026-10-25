"""Executable, semantics-free layout template extracted from a reference figure."""

from __future__ import annotations

from typing import Any


def normalize_reference_layout_template(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    return {
        "contract": "rfs.reference_layout_template",
        "version": 1,
        "canvas": source.get("canvas") if isinstance(source.get("canvas"), dict) else {},
        "zones": [item for item in source.get("zones", []) if isinstance(item, dict)],
        "object_patterns": [str(item) for item in source.get("object_patterns", []) if str(item).strip()],
        "relation_patterns": [item for item in source.get("relation_patterns", []) if isinstance(item, dict)],
        "density_profile": source.get("density_profile") if isinstance(source.get("density_profile"), dict) else {},
        "style_tokens": source.get("style_tokens") if isinstance(source.get("style_tokens"), dict) else {},
        "adoption_guidance": source.get("adoption_guidance") if isinstance(source.get("adoption_guidance"), dict) else {"adopt": [], "avoid_copying": []},
    }


def validate_reference_layout_template(template: dict[str, Any]) -> dict[str, Any]:
    value = normalize_reference_layout_template(template)
    errors: list[str] = []
    zone_ids = {str(item.get("id")) for item in value["zones"] if item.get("id")}
    if len(value["zones"]) < 2:
        errors.append("reference template requires at least two zones")
    for index, zone in enumerate(value["zones"]):
        if not zone.get("id") or not zone.get("role") or not zone.get("internal_pattern"):
            errors.append(f"zone[{index}] requires id/role/internal_pattern")
        try:
            width = float(zone.get("relative_width"))
            if width <= 0 or width > 1:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"zone[{index}] requires legal relative_width")
    for index, relation in enumerate(value["relation_patterns"]):
        if str(relation.get("source_zone")) not in zone_ids or str(relation.get("target_zone")) not in zone_ids:
            errors.append(f"relation_pattern[{index}] references unknown zone")
    return {"ok": not errors, "errors": errors, "zone_count": len(value["zones"]), "relation_pattern_count": len(value["relation_patterns"])}
