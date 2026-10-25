"""Measure whether a composition uses reusable structure from its reference template."""

from __future__ import annotations

from typing import Any


def evaluate_reference_adoption(template: dict[str, Any], composition: dict[str, Any]) -> dict[str, Any]:
    """Compare zone and relation-pattern coverage without comparing scientific labels."""

    zones = [item for item in template.get("zones", []) if isinstance(item, dict)]
    relations = [item for item in template.get("relation_patterns", []) if isinstance(item, dict)]
    panels = [item for item in composition.get("panels", []) if isinstance(item, dict)]
    connectors = [item for item in composition.get("connectors", []) if isinstance(item, dict)]
    explicit = composition.get("reference_adoption") if isinstance(composition.get("reference_adoption"), dict) else {}
    adopted = [str(item) for item in explicit.get("adopted_patterns", []) if str(item).strip()]
    zone_coverage = min(1.0, len(panels) / max(len(zones), 1)) if zones else 1.0
    relation_coverage = min(1.0, len(connectors) / max(len(relations), 1)) if relations else 1.0
    explicit_coverage = min(1.0, len(adopted) / max(len(zones) + len(relations), 1)) if zones or relations else 1.0
    score = 0.45 * zone_coverage + 0.35 * relation_coverage + 0.20 * explicit_coverage
    status = "pass" if score >= 0.55 else "blocked"
    return {"summary": "Reference structural adoption report.", "status": status, "score": round(score, 4), "zone_coverage": round(zone_coverage, 4), "relation_coverage": round(relation_coverage, 4), "explicit_adoption_coverage": round(explicit_coverage, 4), "template_zone_count": len(zones), "composition_panel_count": len(panels), "template_relation_count": len(relations), "composition_connector_count": len(connectors), "adopted_patterns": adopted}
