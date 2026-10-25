"""Bounded layout variants that preserve one authoritative scientific graph."""

from __future__ import annotations

import copy
from typing import Any

from .density_advisory import build_density_advisory


def _scaled(box: dict[str, Any], scale: float) -> dict[str, float]:
    x, y, w, h = (float(box.get(key, 0.0)) for key in ("x", "y", "w", "h"))
    nw, nh = min(0.96, w * scale), min(0.92, h * scale)
    cx, cy = x + w / 2, y + h / 2
    nx, ny = max(0.02, min(0.98 - nw, cx - nw / 2)), max(0.04, min(0.96 - nh, cy - nh / 2))
    return {"x": round(nx, 4), "y": round(ny, 4), "w": round(nw, 4), "h": round(nh, 4)}


def build_layout_variants(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Create three geometry-only variants without changing nodes or relations."""
    variants = []
    for variant_id in ("balanced", "compact", "hierarchy_emphasis"):
        candidate = copy.deepcopy(plan)
        for node in candidate.get("nodes", []):
            box = node.get("bbox_percent") if isinstance(node.get("bbox_percent"), dict) else None
            if not box:
                continue
            scale = 1.0
            if variant_id == "compact":
                scale = 1.10
            elif variant_id == "hierarchy_emphasis":
                scale = 1.14 if node.get("group_id") in {"aggregation_group", "output_group"} else 0.94
            node["bbox_percent"] = _scaled(box, scale)
        candidate["layout_policy"] = {**candidate.get("layout_policy", {}), "variant": variant_id}
        advisory = build_density_advisory(candidate)
        variants.append({"variant_id": variant_id, "plan": candidate, "density_advisory": advisory})
    return variants
