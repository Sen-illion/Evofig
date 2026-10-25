"""Rendered-ink density measurement for editable figures.

Box-area occupancy counts how much of a panel a rectangle covers, so a figure
can score well while its cards hold almost no visible content (the round_07 and
round_09 failure mode). This module measures the delivered preview instead: how
much of the canvas carries ink at all, and how much of the body is an empty
grid cell. Thresholds are calibrated on the reference deck's first page.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


GRID_COLUMNS = 32
GRID_ROWS = 18
TITLE_BAND_RATIO = 0.12
BACKGROUND_TOLERANCE = 18
EMPTY_CELL_INK_RATIO = 0.005
MIN_INK_COVERAGE = 0.18
MAX_BODY_EMPTY_CELL_RATIO = 0.42


def _background_color(image: Image.Image) -> tuple[int, int, int]:
    """Most common colour along the canvas border, i.e. the page background."""
    width, height = image.size
    pixels = image.load()
    samples: Counter[tuple[int, int, int]] = Counter()
    for x in range(width):
        for y in (0, height - 1):
            samples[pixels[x, y][:3]] += 1
    for y in range(height):
        for x in (0, width - 1):
            samples[pixels[x, y][:3]] += 1
    return samples.most_common(1)[0][0]


def measure_rendered_density(
    preview_path: str | Path,
    *,
    columns: int = GRID_COLUMNS,
    rows: int = GRID_ROWS,
) -> dict[str, Any]:
    """Measure visible ink coverage and empty grid cells on a rendered preview."""
    path = Path(preview_path)
    if not path.exists():
        return {
            "summary": "Rendered ink density of the delivered preview.",
            "status": "not_measured",
            "reason": "preview_missing",
            "preview": str(path),
        }
    with Image.open(path) as raw:
        image = raw.convert("RGB")
    width, height = image.size
    background = _background_color(image)
    pixels = image.load()
    ink = [[False] * width for _ in range(height)]
    ink_total = 0
    for y in range(height):
        row = ink[y]
        for x in range(width):
            r, g, b = pixels[x, y]
            if abs(r - background[0]) + abs(g - background[1]) + abs(b - background[2]) > BACKGROUND_TOLERANCE:
                row[x] = True
                ink_total += 1
    body_top = int(height * TITLE_BAND_RATIO)
    cell_w = max(1.0, width / columns)
    cell_h = max(1.0, height / rows)
    empty_cells = 0
    body_cells = 0
    body_empty = 0
    for row_index in range(rows):
        y0 = int(row_index * cell_h)
        y1 = max(y0 + 1, int((row_index + 1) * cell_h))
        for column_index in range(columns):
            x0 = int(column_index * cell_w)
            x1 = max(x0 + 1, int((column_index + 1) * cell_w))
            area = max(1, (y1 - y0) * (x1 - x0))
            covered = sum(ink[y][x0:x1].count(True) for y in range(y0, min(y1, height)))
            is_empty = covered / area < EMPTY_CELL_INK_RATIO
            empty_cells += int(is_empty)
            if y0 >= body_top:
                body_cells += 1
                body_empty += int(is_empty)
    ink_coverage = round(ink_total / max(1, width * height), 4)
    empty_cell_ratio = round(empty_cells / max(1, columns * rows), 4)
    body_empty_cell_ratio = round(body_empty / max(1, body_cells), 4)
    blocking = []
    if ink_coverage < MIN_INK_COVERAGE:
        blocking.append(f"ink_coverage_below_minimum:{MIN_INK_COVERAGE}")
    if body_empty_cell_ratio > MAX_BODY_EMPTY_CELL_RATIO:
        blocking.append(f"body_empty_cells_above_maximum:{MAX_BODY_EMPTY_CELL_RATIO}")
    return {
        "summary": "Rendered ink density of the delivered preview.",
        "status": "pass" if not blocking else "warning",
        "policy": "box-area occupancy is reported next to the ink actually visible in the delivered preview",
        "preview": str(path),
        "canvas_px": [width, height],
        "grid": [columns, rows],
        "background_rgb": list(background),
        "ink_coverage": ink_coverage,
        "empty_cell_ratio": empty_cell_ratio,
        "body_empty_cell_ratio": body_empty_cell_ratio,
        "minimum_ink_coverage": MIN_INK_COVERAGE,
        "maximum_body_empty_cell_ratio": MAX_BODY_EMPTY_CELL_RATIO,
        "issues": blocking,
    }


__all__ = ["measure_rendered_density"]
