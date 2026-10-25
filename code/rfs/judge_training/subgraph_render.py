"""Deterministic PNG rendering of one node centered subgraph.

A RAG subgraph pair is judged by a model that reads figures, and it is reviewed by a
human who looks at pictures, so a subgraph has to reach both as an image and not only as
a description. The renderer here is deliberately the cheapest thing that can do that: a
layered node-and-arrow diagram drawn with Pillow, no layout model, no provider call and
no icon assets. The same subgraph always produces the same bytes, so a persisted pair
dataset stays readable after the index is rebuilt.

The picture exists to be *read as structure*: which node is the centre, how many hops
away the neighbours sit and which way every edge points. Styling is therefore flat and
label first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from ..analysis.graph.subgraphs import subgraph_edges


RENDER_VERSION = "subgraph_render_v1"
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
MARGIN = 10
NODE_HEIGHT = 26
NODE_MIN_WIDTH = 54
NODE_MAX_WIDTH = 108
LABEL_CHARS = 14
BACKGROUND = (255, 255, 255)
PANEL_OUTLINE = (203, 213, 225)
NODE_FILL = (219, 234, 254)
NODE_OUTLINE = (37, 99, 235)
CENTER_FILL = (254, 240, 138)
CENTER_OUTLINE = (180, 83, 9)
EDGE_COLOR = (71, 85, 105)
EDGE_LABEL_COLOR = (100, 116, 139)


def _font() -> Any:
    return ImageFont.load_default()


def _truncate(text: str, limit: int = LABEL_CHARS) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: max(1, limit - 1)] + "\u2026"


def _label_of(node: dict[str, Any] | None, fallback: str = "") -> str:
    item = node or {}
    for key in ("label", "name", "title", "id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return str(fallback or "").strip()


def _depths(node_ids: list[str], edges: list[dict[str, Any]], center: str) -> dict[str, int]:
    """Return each node's hop distance from the centre, ignoring edge direction."""

    neighbours: dict[str, set[str]] = {node_id: set() for node_id in node_ids}
    for edge in edges:
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        if source in neighbours and target in neighbours and source != target:
            neighbours[source].add(target)
            neighbours[target].add(source)
    start = center if center in neighbours else (node_ids[0] if node_ids else "")
    depth = {start: 0}
    frontier = [start]
    while frontier:
        following: list[str] = []
        for node_id in frontier:
            for other in sorted(neighbours.get(node_id, ())):
                if other not in depth:
                    depth[other] = depth[node_id] + 1
                    following.append(other)
        frontier = following
    # A node the centre cannot reach keeps its own column instead of overlapping column 0.
    for node_id in node_ids:
        depth.setdefault(node_id, 0)
    return depth


def _arrow(draw: Any, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill=EDGE_COLOR, width=2)
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    tip = (end[0] - ux * 3, end[1] - uy * 3)
    left = (tip[0] - ux * 7 - uy * 4, tip[1] - uy * 7 + ux * 4)
    right = (tip[0] - ux * 7 + uy * 4, tip[1] - uy * 7 - ux * 4)
    draw.polygon([tip, left, right], fill=EDGE_COLOR)


def render_subgraph(subgraph: dict[str, Any], graph: dict[str, Any] | None, out: str | Path, *, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> str:
    """Draw one subgraph to a PNG and return the path it was written to."""

    properties = subgraph.get("properties") if isinstance(subgraph.get("properties"), dict) else {}
    center = str(properties.get("center_node_id") or "")
    node_ids = sorted(str(item) for item in (subgraph.get("node_ids") or []))
    edges = subgraph_edges(subgraph, graph)
    nodes = {str(item.get("id") or ""): item for item in ((graph or {}).get("nodes") or []) if isinstance(item, dict)}
    depth = _depths(node_ids, edges, center)
    columns: dict[int, list[str]] = {}
    for node_id in node_ids:
        columns.setdefault(depth.get(node_id, 0), []).append(node_id)
    layers = sorted(columns)

    image = Image.new("RGB", (int(width), int(height)), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = _font()
    boxes: dict[str, tuple[int, int, int, int]] = {}
    span = max(1, int(width) - 2 * MARGIN)
    column_width = span / float(max(1, len(layers)))
    box_width = int(max(NODE_MIN_WIDTH, min(NODE_MAX_WIDTH, column_width - 10)))
    for index, layer in enumerate(layers):
        members = sorted(columns[layer])
        left = int(MARGIN + index * column_width + max(0, (column_width - box_width) / 2))
        usable = max(1, int(height) - 2 * MARGIN)
        step = usable / float(len(members))
        for position, node_id in enumerate(members):
            top = int(MARGIN + position * step + max(0, (step - NODE_HEIGHT) / 2))
            boxes[node_id] = (left, top, left + box_width, top + NODE_HEIGHT)
    for edge in edges:
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        if source not in boxes or target not in boxes:
            continue
        a, b = boxes[source], boxes[target]
        start = (a[2] if a[0] <= b[0] else a[0], (a[1] + a[3]) // 2)
        end = (b[0] if a[0] <= b[0] else b[2], (b[1] + b[3]) // 2)
        _arrow(draw, start, end)
    draw.rectangle((0, 0, int(width) - 1, int(height) - 1), outline=PANEL_OUTLINE)
    for node_id in node_ids:
        box = boxes.get(node_id)
        if box is None:
            continue
        is_center = node_id == center
        draw.rounded_rectangle(box, radius=5, fill=CENTER_FILL if is_center else NODE_FILL, outline=CENTER_OUTLINE if is_center else NODE_OUTLINE, width=2)
        draw.text((box[0] + 4, box[1] + 7), _truncate(_label_of(nodes.get(node_id), node_id)), fill=(15, 23, 42), font=font)
    draw.text((MARGIN, 2), _truncate(str(subgraph.get("motif") or "unknown"), 24), fill=EDGE_LABEL_COLOR, font=font)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return str(path)