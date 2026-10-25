"""Normalize generated Icon backgrounds for panel inheritance."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image


def _rgb(value: str) -> tuple[int, int, int]:
    text = str(value or "#FFFFFF").lstrip("#")
    return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))


def normalize_icon_background(path: str | Path, target_hex: str, *, policy: str = "transparent_panel_inherit") -> dict[str, Any]:
    """Remove connected corner background or replace it with the exact panel color."""

    source = Path(path)
    with Image.open(source).convert("RGBA") as image:
        pixels = image.load()
        width, height = image.size
        corners = [pixels[0, 0], pixels[width - 1, 0], pixels[0, height - 1], pixels[width - 1, height - 1]]
        target = _rgb(target_hex)
        changed = 0
        queue = deque([(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)])
        seen: set[tuple[int, int]] = set()
        while queue:
            x, y = queue.popleft()
            if (x, y) in seen or not (0 <= x < width and 0 <= y < height):
                continue
            seen.add((x, y))
            pixel = pixels[x, y]
            if pixel[3] == 0:
                continue
            if max(abs(pixel[channel] - corners[0][channel]) for channel in range(3)) > 24:
                continue
            pixels[x, y] = (target[0], target[1], target[2], 0) if policy == "transparent_panel_inherit" else (target[0], target[1], target[2], 255)
            changed += 1
            queue.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
        image.save(source)
        alpha_pixels = sum(1 for pixel in image.getdata() if pixel[3] == 0)
    return {"path": str(source), "target_background_hex": target_hex, "policy": policy, "corner_colors": [list(pixel[:3]) for pixel in corners], "changed_pixel_count": changed, "transparent_pixel_ratio": round(alpha_pixels / max(width * height, 1), 4), "status": "pass"}
