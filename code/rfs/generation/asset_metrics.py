"""Measure how much of a generated icon raster its visible glyph occupies."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

ALPHA_THRESHOLD = 24
COLOR_THRESHOLD = 24
_ALPHA_LUT = bytes(255 if value >= ALPHA_THRESHOLD else 0 for value in range(256))
_COLOR_LUT = bytes(255 if value >= COLOR_THRESHOLD else 0 for value in range(256))
CACHE: dict[str, dict[str, Any]] = {}


def _fallback(path: str, source: str) -> dict[str, Any]:
    return {"path": path, "width_ratio": 1.0, "height_ratio": 1.0, "ink_ratio": 1.0, "source": source}


def _bbox_ratios(bbox: tuple[int, int, int, int] | None, size: tuple[int, int]) -> tuple[float, float]:
    width, height = size
    if not bbox or width <= 0 or height <= 0:
        return 1.0, 1.0
    return (bbox[2] - bbox[0]) / width, (bbox[3] - bbox[1]) / height


def measure_ink_bbox(path: str | Path) -> dict[str, Any]:
    """Return the visible glyph bbox of one icon raster as frame ratios.

    Generated icons are drawn with contain semantics, so any padding baked into
    the raster shrinks the glyph that actually reaches the slide. The ratios let
    the layout compensate the icon slot instead of silently shrinking the mark.
    """

    target = Path(str(path))
    key = str(target)
    if key in CACHE:
        return CACHE[key]
    if not target.exists():
        CACHE[key] = _fallback(key, "missing_file")
        return CACHE[key]
    try:
        from PIL import Image, ImageChops

        with Image.open(target) as image:
            rgba = image.convert("RGBA")
            size = rgba.size
            alpha = rgba.getchannel("A")
            alpha_mask = alpha.point(_ALPHA_LUT)
            bbox = alpha_mask.getbbox()
            histogram = alpha.histogram()
            opaque = sum(histogram[128:]) / max(1, size[0] * size[1])
            if opaque >= 0.98 and bbox == (0, 0, size[0], size[1]):
                rgb = rgba.convert("RGB")
                background = Image.new("RGB", size, rgb.getpixel((0, 0)))
                difference = ImageChops.difference(rgb, background).convert("L").point(_COLOR_LUT)
                bbox = difference.getbbox()
                source = "corner_color_bbox"
            else:
                source = "alpha_bbox"
    except Exception as error:  # pragma: no cover - defensive, reported explicitly
        CACHE[key] = _fallback(key, f"unreadable_{type(error).__name__}")
        return CACHE[key]
    width_ratio, height_ratio = _bbox_ratios(bbox, size)
    CACHE[key] = {
        "path": key,
        "size": [size[0], size[1]],
        "bbox": list(bbox) if bbox else None,
        "width_ratio": round(width_ratio, 4),
        "height_ratio": round(height_ratio, 4),
        "ink_ratio": round(opaque, 4),
        "source": source,
    }
    return CACHE[key]


def measure_program_assets(assets: Any) -> dict[str, dict[str, Any]]:
    """Measure every program asset once, keyed by asset id."""

    measured: dict[str, dict[str, Any]] = {}
    for asset in assets or []:
        if not isinstance(asset, dict):
            continue
        asset_id = str(asset.get("id") or "")
        path = str(asset.get("path") or "")
        if not asset_id or not path:
            continue
        measured[asset_id] = measure_ink_bbox(path)
    return measured


GLYPH_TRIM_MARGIN = 0.03


def trim_asset_to_glyph(path: str | Path, *, margin_ratio: float = GLYPH_TRIM_MARGIN) -> dict[str, Any]:
    """Write a glyph-tight copy of one icon asset next to the original.

    Generated rasters carry a transparent frame around the mark. Inserting that
    frame with contain semantics shrinks the visible glyph, so the layout sees
    padding where the drawing is not. The trimmed copy lets a slot map onto the
    glyph itself; the untouched selection stays on disk for audit.
    """

    target = Path(str(path))
    measured = measure_ink_bbox(target)
    bbox = measured.get("bbox")
    size = measured.get("size")
    record: dict[str, Any] = {"path": str(target), "trimmed_path": None, "status": "unchanged", "bbox": bbox, "size": size}
    if not isinstance(size, list) or len(size) != 2 or not all(size):
        record["status"] = "unmeasurable"
        return record
    if not bbox:
        # A measurable raster with no distinguishable mark is already tight.
        record["status"] = "already_tight"
        return record
    try:
        from PIL import Image

        with Image.open(target) as image:
            rgba = image.convert("RGBA")
            width, height = rgba.size
            left, top, right, bottom = (int(value) for value in bbox)
            pad = max(2, int(round(max(right - left, bottom - top) * margin_ratio)))
            crop = (max(0, left - pad), max(0, top - pad), min(width, right + pad), min(height, bottom + pad))
            if crop == (0, 0, width, height):
                record["status"] = "already_tight"
                return record
            trimmed = rgba.crop(crop)
            out_path = target.with_name("glyph.png")
            trimmed.save(out_path)
    except Exception as error:  # pragma: no cover - defensive, reported explicitly
        record["status"] = f"unreadable_{type(error).__name__}"
        return record
    record.update({"trimmed_path": str(out_path), "status": "trimmed", "crop": list(crop), "trimmed_size": list(trimmed.size)})
    return record


def trim_program_icon_assets(program: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Point every program asset at a glyph-tight copy of its mark."""

    updated = deepcopy(program)
    assets = [item for item in updated.get("assets", []) or [] if isinstance(item, dict)]
    records = []
    for asset in assets:
        path = str(asset.get("path") or "")
        if not path:
            continue
        record = trim_asset_to_glyph(path)
        if record["status"] == "trimmed":
            asset["path"] = record["trimmed_path"]
            asset["glyph_trim"] = {"crop": record["crop"], "source_path": record["path"], "trimmed_size": record["trimmed_size"]}
        records.append({"asset_id": asset.get("id"), **record})
    updated["assets"] = assets
    report = {
        "summary": "Glyph-tight icon assets delivered to the figure program.",
        "status": "pass",
        "asset_count": len(assets),
        "trimmed_count": sum(1 for item in records if item["status"] == "trimmed"),
        "already_tight_count": sum(1 for item in records if item["status"] == "already_tight"),
        "unmeasurable_count": sum(1 for item in records if item["status"] not in {"trimmed", "already_tight"}),
        "margin_ratio": GLYPH_TRIM_MARGIN,
        "assets": records,
    }
    return updated, report


__all__ = ["measure_ink_bbox", "measure_program_assets", "trim_asset_to_glyph", "trim_program_icon_assets"]
