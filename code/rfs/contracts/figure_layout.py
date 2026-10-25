"""Stable contract for direct paper-to-PPTX figure layout plans."""

from __future__ import annotations

import math
from typing import Any

FIGURE_LAYOUT_CONTRACT_VERSION = 1

#: Canvas margin used when the plan does not carry a readable one.
DEFAULT_CANVAS_MARGIN = 0.04


def _number(value: Any, default: float | None) -> float | None:
    """Read one numeric field without trusting the shape the model returned.

    The design model is asked for a scalar margin and still answers with a
    named-edge object such as ``{"top": 0.04, "bottom": 0.04}`` on some papers.
    Losing a whole paper to that shape is not worth its cost, so the widest
    readable edge is used and the default stands in when the object carries no
    number at all.
    """
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else default
    if isinstance(value, dict):
        edges = [_number(item, None) for item in value.values()]
        numbers = [edge for edge in edges if edge is not None]
        return max(numbers) if numbers else default
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def normalize_figure_layout_plan(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one layout plan with every field the rest of the line relies on typed."""
    source = raw if isinstance(raw, dict) else {}
    def items(key: str) -> list[dict[str, Any]]:
        value = source.get(key)
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
    canvas = source.get("canvas") if isinstance(source.get("canvas"), dict) else {}
    margin = _number(canvas.get("margin"), DEFAULT_CANVAS_MARGIN)
    version = _number(source.get("version"), float(FIGURE_LAYOUT_CONTRACT_VERSION))
    return {
        "contract": "rfs.figure_layout_plan",
        "version": int(version or FIGURE_LAYOUT_CONTRACT_VERSION),
        "canvas": {"aspect_ratio": str(canvas.get("aspect_ratio") or "16:9"), "margin": float(margin or DEFAULT_CANVAS_MARGIN)},
        "topology": str(source.get("topology") or "left_to_right_pipeline"),
        "reading_order": [str(item) for item in source.get("reading_order", []) if str(item).strip()] if isinstance(source.get("reading_order"), list) else [],
        "panels": items("panels"),
        "nodes": items("nodes"),
        "connectors": items("connectors"),
        "style_tokens": source.get("style_tokens") if isinstance(source.get("style_tokens"), dict) else {},
        "provenance": source.get("provenance") if isinstance(source.get("provenance"), dict) else {},
    }


def validate_figure_layout_plan(plan: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_figure_layout_plan(plan)
    errors: list[str] = []
    node_ids = [str(item.get("id") or "") for item in normalized["nodes"]]
    if len(node_ids) != len(set(node_ids)):
        errors.append("layout node IDs must be unique")
    known = {item for item in node_ids if item}
    for index, connector in enumerate(normalized["connectors"]):
        source = str(connector.get("source") or "").strip()
        target = str(connector.get("target") or "").strip()
        if not source or not target or source not in known or target not in known:
            errors.append(f"connector[{index}] must reference declared nodes")
    for index, node in enumerate(normalized["nodes"]):
        bbox = node.get("bbox_percent")
        if not isinstance(bbox, dict) or any(key not in bbox for key in ("x", "y", "w", "h")):
            errors.append(f"node[{index}] requires bbox_percent x/y/w/h")
    return {"ok": not errors, "errors": errors, "node_count": len(known), "connector_count": len(normalized["connectors"])}