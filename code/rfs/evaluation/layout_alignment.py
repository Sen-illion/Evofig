"""Reference-free alignment and spacing for one delivered framework figure.

EditableFigureBench L3 already counts overlaps, crossings, and type size from the
figure program. This module answers a different question from the delivered
``editable_composition.pptx`` alone: do the boxes share edges, are the gaps
between neighbors even, and does anything leave the canvas? It does not compare
the deck with ``reference.png``. Connectors are not boxes, so they stay out of
the alignment denominator. Grouped children are placed by adding each ancestor's
offset; that is the slide position python-pptx exposes one group level at a time.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Iterator

from pptx import Presentation

from .native_editability import CONNECTOR_PRESETS, _local_name, _preset_geometry

#: Two edges count as colinear when they differ by less than this share of the axis.
EDGE_ALIGN_RATIO = 0.005

#: A box closer to the canvas than this share of the shorter side is tight, not overflowing.
TIGHT_MARGIN_RATIO = 0.02


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _is_connector(shape: Any) -> bool:
    local = _local_name(shape._element)
    if local == "cxnSp":
        return True
    return _preset_geometry(shape) in CONNECTOR_PRESETS


def _boxes(shapes: Any, origin_x: int = 0, origin_y: int = 0) -> Iterator[tuple[int, int, int, int]]:
    """Yield slide-absolute ``(left, top, right, bottom)`` for every non-connector box."""
    for shape in shapes:
        local = _local_name(shape._element)
        left = origin_x + int(shape.left or 0)
        top = origin_y + int(shape.top or 0)
        if local == "grpSp":
            yield from _boxes(shape.shapes, left, top)
            continue
        if _is_connector(shape):
            continue
        width = int(shape.width or 0)
        height = int(shape.height or 0)
        if width <= 0 or height <= 0:
            continue
        yield (left, top, left + width, top + height)


def _aligned_edge_ratio(boxes: list[tuple[int, int, int, int]], tolerance_x: float, tolerance_y: float) -> float | None:
    """Share of edges that share a line with an edge of a different box."""
    if len(boxes) < 2:
        return None
    vertical: list[tuple[int, int]] = []
    horizontal: list[tuple[int, int]] = []
    for index, (left, top, right, bottom) in enumerate(boxes):
        vertical.append((index, left))
        vertical.append((index, right))
        horizontal.append((index, top))
        horizontal.append((index, bottom))

    def aligned(edges: list[tuple[int, int]], tolerance: float) -> int:
        count = 0
        for owner, coord in edges:
            if any(other != owner and abs(other_coord - coord) <= tolerance for other, other_coord in edges):
                count += 1
        return count

    total = len(vertical) + len(horizontal)
    if total == 0:
        return None
    return (aligned(vertical, tolerance_x) + aligned(horizontal, tolerance_y)) / total


def _vertical_overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
    return min(left[3], right[3]) - max(left[1], right[1])


def _gap_cv(boxes: list[tuple[int, int, int, int]], tolerance_x: float) -> float | None:
    """Coefficient of variation of gaps to the nearest box on the right.

    A neighbor must overlap this box vertically and sit at or to the right of
    its right edge. Fewer than three such gaps cannot support a variation.
    """
    gaps: list[float] = []
    for index, box in enumerate(boxes):
        nearest: float | None = None
        for other_index, other in enumerate(boxes):
            if other_index == index or _vertical_overlap(box, other) <= 0:
                continue
            gap = other[0] - box[2]
            if gap < -tolerance_x:
                continue
            gap = max(0.0, float(gap))
            if nearest is None or gap < nearest:
                nearest = gap
        if nearest is not None:
            gaps.append(nearest)
    if len(gaps) < 3:
        return None
    mean = statistics.fmean(gaps)
    if mean <= 0:
        return 0.0
    return statistics.pstdev(gaps) / mean


def _empty(reason: str, *, ok: bool) -> dict[str, Any]:
    return {
        "summary": "Reference-free alignment and spacing of one delivered deck.",
        "ok": ok,
        "reason": reason,
        "box_count": 0,
        "aligned_edge_ratio": None,
        "gap_cv": None,
        "overflow_count": None,
        "tight_margin_count": None,
    }


def measure_layout_alignment(pptx_path: str | Path) -> dict[str, Any]:
    """Measure colinearity, gap uniformity, overflow, and tight margins.

    A missing or unreadable deck returns ``ok: False`` and null counts. It does
    not report zeros, which would look like a clean figure.
    """
    path = Path(pptx_path)
    if not path.is_file():
        return _empty("missing editable_composition.pptx", ok=False)
    try:
        deck = Presentation(str(path))
    except Exception as exc:
        return _empty(f"{type(exc).__name__}: {exc}", ok=False)
    if not deck.slides:
        return _empty("deck has no slides", ok=False)
    slide = deck.slides[0]
    slide_width = int(deck.slide_width or 0)
    slide_height = int(deck.slide_height or 0)
    if slide_width <= 0 or slide_height <= 0:
        return _empty("slide canvas has no size", ok=False)
    boxes = list(_boxes(slide.shapes))
    tolerance_x = EDGE_ALIGN_RATIO * slide_width
    tolerance_y = EDGE_ALIGN_RATIO * slide_height
    tight_limit = TIGHT_MARGIN_RATIO * min(slide_width, slide_height)
    overflow = 0
    tight = 0
    for left, top, right, bottom in boxes:
        outside = left < -tolerance_x or top < -tolerance_y or right > slide_width + tolerance_x or bottom > slide_height + tolerance_y
        if outside:
            overflow += 1
            continue
        margin = min(left, top, slide_width - right, slide_height - bottom)
        if margin < tight_limit:
            tight += 1
    return {
        "summary": "Reference-free alignment and spacing of one delivered deck.",
        "ok": True,
        "reason": "",
        "box_count": len(boxes),
        "aligned_edge_ratio": _round(_aligned_edge_ratio(boxes, tolerance_x, tolerance_y)),
        "gap_cv": _round(_gap_cv(boxes, tolerance_x)),
        "overflow_count": overflow,
        "tight_margin_count": tight,
    }
