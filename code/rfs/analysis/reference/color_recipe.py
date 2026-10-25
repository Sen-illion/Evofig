"""Extract deterministic color evidence and audit cards from reference figures."""

from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ...contracts.reference_color_recipe import normalize_reference_color_recipe, validate_reference_color_recipe


#: Median-cut buckets taken before near-duplicates are merged. The palette is
#: read as design evidence, so the extraction has to look at enough buckets to
#: notice a small saturated accent sitting beside a large near-white field.
QUANTIZE_COLORS = 32

#: Lab distance under which two buckets become one colour.
MERGE_DELTA_E = 5.0

#: A bucket below this share of the figure survives only when it is saturated
#: enough to be an accent rather than a quantisation artefact.
MINIMUM_AREA_RATIO = 0.003
ACCENT_SATURATION = 0.35

#: Palette cap, mirroring the bound the reference colour recipe contract
#: enforces. A busier figure is summarised rather than truncated: the smallest
#: buckets fold into their nearest neighbour instead of disappearing.
MAX_REPRESENTATIVES = 16

#: Identity of the numbers above. It travels in every recipe and keys the reuse
#: cache, so a stored library measured by an older extractor is re-derived
#: instead of being handed back as if this code had produced it.
EXTRACTOR_ID = "deterministic.v2"


def _lab(rgb: tuple[int, int, int]) -> list[float]:
    pixel = np.uint8([[list(rgb)]])
    value = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0, 0]
    return [round(float(value[0]) * 100.0 / 255.0, 3), round(float(value[1]) - 128.0, 3), round(float(value[2]) - 128.0, 3)]


def _hsl(rgb: tuple[int, int, int]) -> list[float]:
    h, lightness, saturation = colorsys.rgb_to_hls(*(value / 255.0 for value in rgb))
    return [round(h * 360.0, 3), round(saturation, 4), round(lightness, 4)]


def _distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    a, b = _lab(left), _lab(right)
    return float(sum((a[index] - b[index]) ** 2 for index in range(3)) ** 0.5)


def _quantized_colors(image: Image.Image, count: int) -> list[tuple[int, tuple[int, int, int]]]:
    rgb = image.convert("RGB")
    quantized = rgb.quantize(colors=count, method=Image.Quantize.MEDIANCUT).convert("RGB")
    return sorted(quantized.getcolors(rgb.width * rgb.height) or [], reverse=True)


def _merge_nearby(values: list[tuple[int, tuple[int, int, int]]], threshold: float = 8.0) -> list[tuple[int, tuple[int, int, int]]]:
    merged: list[tuple[int, tuple[int, int, int]]] = []
    for count, rgb in values:
        match = next((index for index, (_weight, current) in enumerate(merged) if _distance(rgb, current) < threshold), None)
        if match is None:
            merged.append((count, rgb))
            continue
        old_count, old_rgb = merged[match]
        total = old_count + count
        combined = tuple(round((old_rgb[channel] * old_count + rgb[channel] * count) / total) for channel in range(3))
        merged[match] = (total, combined)
    return sorted(merged, reverse=True)


def _representatives(image: Image.Image, *, count: int = MAX_REPRESENTATIVES) -> list[tuple[int, tuple[int, int, int]]]:
    values = _merge_nearby(_quantized_colors(image, QUANTIZE_COLORS), MERGE_DELTA_E)
    total = image.width * image.height
    kept = []
    dropped = []
    for weight, rgb in values:
        saturation = _hsl(rgb)[1]
        (kept if weight / max(total, 1) >= MINIMUM_AREA_RATIO or saturation >= ACCENT_SATURATION else dropped).append((weight, rgb))
    if len(kept) < 2:
        kept = values[:2]
        dropped = values[2:]
    for weight, rgb in dropped:
        nearest = min(range(len(kept)), key=lambda index: _distance(rgb, kept[index][1]))
        old_weight, old_rgb = kept[nearest]
        combined_weight = old_weight + weight
        combined_rgb = tuple(round((old_rgb[channel] * old_weight + rgb[channel] * weight) / combined_weight) for channel in range(3))
        kept[nearest] = (combined_weight, combined_rgb)
    return sorted(kept[:count], reverse=True)


def _background_index(image: Image.Image, representatives: list[tuple[int, tuple[int, int, int]]]) -> int:
    rgb = np.asarray(image.convert("RGB"))
    band_y = max(1, round(rgb.shape[0] * 0.05))
    band_x = max(1, round(rgb.shape[1] * 0.05))
    border = np.concatenate((rgb[:band_y].reshape(-1, 3), rgb[-band_y:].reshape(-1, 3), rgb[:, :band_x].reshape(-1, 3), rgb[:, -band_x:].reshape(-1, 3)))
    mean = tuple(int(value) for value in np.median(border, axis=0))
    return min(range(len(representatives)), key=lambda index: _distance(mean, representatives[index][1]))


def _spatial_zones(image: Image.Image, rgb: tuple[int, int, int]) -> list[str]:
    data = np.asarray(image.convert("RGB"), dtype=np.int32)
    distance = np.sqrt(np.sum((data - np.asarray(rgb, dtype=np.int32)) ** 2, axis=2))
    result = []
    rows = ("top", "middle", "bottom")
    columns = ("left", "center", "right")
    for row in range(3):
        for column in range(3):
            crop = distance[row * data.shape[0] // 3:(row + 1) * data.shape[0] // 3, column * data.shape[1] // 3:(column + 1) * data.shape[1] // 3]
            if crop.size and float(np.mean(crop < 28.0)) >= 0.08:
                result.append(f"{rows[row]}_{columns[column]}")
    return result


def render_palette_card(recipe: dict[str, Any], path: str | Path) -> Path:
    """Render a compact human-auditable swatch card."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    colors = recipe.get("colors", [])
    card = Image.new("RGB", (900, 90 + 86 * len(colors)), "white")
    draw = ImageDraw.Draw(card)
    draw.text((24, 20), f"{recipe.get('recipe_id')}  |  {recipe.get('source_sha256', '')[:12]}", fill="#172033")
    for index, item in enumerate(colors):
        y = 68 + index * 86
        draw.rounded_rectangle((24, y, 174, y + 58), radius=8, fill=item["hex"], outline="#64748B", width=2)
        details = f"{item['color_id']}  {item['hex']}  {item['area_ratio']:.1%}  {', '.join(item['spatial_zones'][:4])}"
        usage = ", ".join(item.get("used_for") or ["unassigned"])
        roles = ", ".join((item.get("semantic_roles") or [])[:4])
        draw.text((194, y + 9), details, fill="#172033")
        draw.text((194, y + 33), f"uses: {usage}  |  roles: {roles or '-'}", fill="#3E5066")
    card.save(target)
    return target


def analyze_reference_color_recipe(image_path: str | Path, recipe_id: str, *, quality_label: str = "positive", palette_card_path: str | Path | None = None) -> dict[str, Any]:
    """Build deterministic palette evidence for one reference image."""
    source = Path(image_path).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with Image.open(source) as opened:
        original_size = opened.size
        image = opened.convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        rgb = background.convert("RGB")
        scale = min(1.0, 384.0 / max(rgb.size))
        analysis_image = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))), Image.Resampling.LANCZOS)
    representatives = _representatives(analysis_image)
    total = sum(weight for weight, _rgb in representatives)
    colors = []
    for index, (weight, value) in enumerate(representatives):
        colors.append({
            "color_id": f"c{index + 1:02d}",
            "hex": f"#{value[0]:02X}{value[1]:02X}{value[2]:02X}",
            "rgb": list(value),
            "lab": _lab(value),
            "hsl": _hsl(value),
            "area_ratio": round(weight / max(total, 1), 6),
            "spatial_zones": _spatial_zones(analysis_image, value),
            "used_for": ["unassigned"],
            "semantic_roles": [],
            "confidence": 1.0,
        })
    if colors:
        colors[-1]["area_ratio"] = round(colors[-1]["area_ratio"] + (1.0 - sum(item["area_ratio"] for item in colors)), 6)
    background_index = _background_index(analysis_image, representatives)
    colors[background_index]["used_for"] = ["canvas"]
    recipe = normalize_reference_color_recipe({
        "recipe_id": recipe_id,
        "source_path": str(source),
        "source_sha256": digest,
        "quality_label": quality_label,
        "image": {"width": original_size[0], "height": original_size[1], "analysis_width": analysis_image.width, "analysis_height": analysis_image.height},
        "background_color_id": colors[background_index]["color_id"],
        "colors": colors,
        "analysis": {
            "mode": "deterministic",
            "extractor": EXTRACTOR_ID,
            "quantize_colors": QUANTIZE_COLORS,
            "merge_delta_e": MERGE_DELTA_E,
            "minimum_area_ratio": MINIMUM_AREA_RATIO,
            "accent_saturation": ACCENT_SATURATION,
            "maximum_colors": MAX_REPRESENTATIVES,
            "palette_card": str(Path(palette_card_path).resolve()) if palette_card_path else None,
        },
    })
    validation = validate_reference_color_recipe(recipe)
    if not validation["ok"]:
        raise ValueError("invalid deterministic color recipe: " + "; ".join(validation["errors"]))
    if palette_card_path:
        render_palette_card(recipe, palette_card_path)
    return recipe


__all__ = ["EXTRACTOR_ID", "analyze_reference_color_recipe", "render_palette_card"]
