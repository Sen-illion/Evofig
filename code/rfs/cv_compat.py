from __future__ import annotations

from typing import Any


def hough_line_coordinates(line: Any) -> tuple[int, int, int, int] | None:
    """Normalize OpenCV HoughLinesP rows across array-shape variants."""
    try:
        values = line.reshape(-1).tolist() if hasattr(line, "reshape") else line
        while isinstance(values, (list, tuple)) and len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = values[0]
        if not isinstance(values, (list, tuple)) or len(values) < 4:
            return None
        return tuple(int(value) for value in values[:4])
    except (TypeError, ValueError):
        return None
