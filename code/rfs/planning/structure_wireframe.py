"""Render a cheap structure-only preview for density review."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def render_structure_wireframe(plan: dict[str, Any], out: str | Path, *, width: int = 1200, height: int = 675) -> str:
    """Render panels, nodes, groups and connectors without icons or final styling."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def box(value: dict[str, Any]) -> tuple[int, int, int, int]:
        return (int(float(value.get("x", 0)) * width), int(float(value.get("y", 0)) * height), int((float(value.get("x", 0)) + float(value.get("w", 0))) * width), int((float(value.get("y", 0)) + float(value.get("h", 0))) * height))

    panels = [p for p in plan.get("panels", []) if isinstance(p, dict)]
    nodes = {str(n.get("id")): n for n in plan.get("nodes", []) if isinstance(n, dict)}
    for panel in panels:
        b = box(panel.get("bbox_percent") or {})
        draw.rounded_rectangle(b, radius=10, outline="#64748B", width=3, fill="#F8FAFC")
        draw.text((b[0] + 8, b[1] + 8), str(panel.get("title") or panel.get("id") or "panel"), fill="#1E293B")
    for group in plan.get("groups", []) or []:
        if isinstance(group, dict) and isinstance(group.get("bbox_percent"), dict):
            draw.rounded_rectangle(box(group["bbox_percent"]), radius=8, outline="#94A3B8", width=2)
    for edge in plan.get("connectors", []) or []:
        if not isinstance(edge, dict):
            continue
        source, target = nodes.get(str(edge.get("source"))), nodes.get(str(edge.get("target")))
        if not source or not target:
            continue
        a, b = box(source.get("bbox_percent") or {}), box(target.get("bbox_percent") or {})
        start, end = ((a[2], (a[1] + a[3]) // 2), (b[0], (b[1] + b[3]) // 2))
        draw.line([start, end], fill="#475569", width=3)
    for node in nodes.values():
        b = box(node.get("bbox_percent") or {})
        draw.rounded_rectangle(b, radius=8, outline="#2563EB", width=3, fill="#DBEAFE")
        draw.text((b[0] + 6, b[1] + 6), str(node.get("label") or node.get("id") or "node"), fill="#0F172A")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)

