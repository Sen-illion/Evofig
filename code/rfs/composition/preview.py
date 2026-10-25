from __future__ import annotations

from pathlib import Path
import math

from PIL import Image, ImageColor, ImageDraw, ImageFont
from ..contracts.figure_text import fit_panel_title_size as _fit_panel_title_size
from .fonts import body_font_family, heading_font_family, load_font as _load_font
from ..contracts.text_ownership import node_text_owned_by_program, visible_text_items


def _hex_color(value: object, fallback: str = "#000000") -> str:
    if not isinstance(value, str):
        return fallback
    text = value.strip()
    if not text:
        return fallback
    if not text.startswith("#"):
        text = "#" + text
    try:
        ImageColor.getrgb(text)
        return text
    except ValueError:
        return fallback


def _bbox_px(bbox: dict, width: int, height: int) -> tuple[int, int, int, int]:
    x = int(round(float(bbox.get("x", 0)) * width))
    y = int(round(float(bbox.get("y", 0)) * height))
    w = int(round(float(bbox.get("w", 0)) * width))
    h = int(round(float(bbox.get("h", 0)) * height))
    return x, y, max(1, w), max(1, h)


# Roles the compiler paints with the deck's heading font; every other role
# uses the body font the program declares.
HEADING_ROLES = frozenset({"title", "subtitle", "panel_title", "panel_header"})


def _font(size_px: int, bold: bool = False, family: str | None = None) -> ImageFont.ImageFont:
    return _load_font(max(4, int(size_px)), family, bold=bold)


def _families(program: dict) -> tuple[str, str]:
    """Return (heading family, body family) exactly as the compiler paints them."""

    style = program.get("style") if isinstance(program.get("style"), dict) else {}
    return heading_font_family(style), body_font_family(style)


def _family_for_item(item: dict, families: tuple[str, str]) -> str:
    role = str(item.get("role") or item.get("text_role") or "").casefold()
    return families[0] if role in HEADING_ROLES else families[1]


def _is_panel_title_item(item: dict) -> bool:
    return str(item.get("role") or item.get("text_role")) == "panel_title"


def _is_ocr_bound(item: dict) -> bool:
    return "ocr" in str(item.get("reference_binding") or item.get("fit_strategy") or "").lower()


def _panel_title_text_item(program: dict, panel_id: object) -> dict | None:
    """Return the text program item that owns a panel heading, if any."""

    for item in visible_text_items(program):
        if not _is_panel_title_item(item):
            continue
        if str(item.get("target_id") or item.get("panel_id")) == str(panel_id):
            return item
    return None


def _draw_panel_header_title(
    draw: ImageDraw.ImageDraw,
    program: dict,
    panel: dict,
    local_style: dict,
    band_box: dict,
    canvas: dict,
    canvas_size: tuple[int, int],
    *,
    fill_color: str,
    header_color: str,
) -> None:
    """Paint a panel heading inside the header band the compiler actually draws.

    The PowerPoint compiler renders a non-OCR panel title as the text of the
    header band.  The preview used to paint the text program's own box instead,
    and that box is pinned just below the band, so every white heading landed
    on the pale panel fill and read as washed out - in the preview the user
    judges, and in every visual review made from one.
    """

    title = str(panel.get("title") or "").strip()
    item = _panel_title_text_item(program, panel.get("id"))
    if not title or (item is not None and _is_ocr_bound(item)):
        return
    canvas_w, canvas_h = canvas_size
    canvas_width_in = float(canvas.get("width_in") or 13.333)
    canvas_height_in = float(canvas.get("height_in") or 7.5)
    band_width_in = float(band_box["w"]) * canvas_width_in
    band_height_in = float(band_box["h"]) * canvas_height_in
    requested = float((item or {}).get("font_size_pt") or (9.0 if band_width_in < 2.0 else 10.0))
    color = str(
        (item or {}).get("color_hex")
        or local_style.get("header_text_color")
        or ("#FFFFFF" if header_color != fill_color else local_style.get("text_color") or "#173042")
    )
    _draw_text(
        draw,
        {
            "id": f"{panel.get('id')}_header_title",
            "text": title,
            "bbox_percent": band_box,
            "font_size_pt": _fit_panel_title_size(title, band_width_in, requested, band_height_in),
            "font_size_role": "panel_title",
            "bold": True,
            "color_hex": color,
            "center_text_in_textbox": True,
            "max_lines": int((item or {}).get("max_lines") or 2),
            "canvas_height_in": canvas_height_in,
        },
        canvas_w,
        canvas_h,
        heading_font_family((program.get("style") if isinstance(program.get("style"), dict) else {})),
    )


def _draw_text(draw: ImageDraw.ImageDraw, item: dict, canvas_w: int, canvas_h: int, family: str | None = None) -> None:
    if not item.get("visible", True):
        return
    if str(item.get("layer_ownership") or "editable_text_layer") != "editable_text_layer":
        return
    bbox = item.get("bbox_percent")
    if not isinstance(bbox, dict):
        return
    x, y, w, h = _bbox_px(bbox, canvas_w, canvas_h)
    text = str(item.get("text") or "")
    if not text:
        return
    font_size_pt = float(item.get("font_size_pt") or 8)
    canvas_height_in = float(item.get("canvas_height_in") or 7.5)
    size_px = max(4, int(round(font_size_pt * canvas_h / (canvas_height_in * 72.0))))
    font = _font(size_px, bold=bool(item.get("bold")), family=family)
    color = _hex_color(item.get("color_hex"), "#263747")
    align = "center" if bool(item.get("center_text_in_textbox", True)) else str(item.get("align") or "center").lower()
    # Mirror PowerPoint's wrapping behavior for compact panel headings.
    words = list(text) if " " not in text.strip() else text.split()
    sep = "" if " " not in text.strip() else " "
    lines, current = [], ""
    for word in words:
        candidate = f"{current}{sep if current else ''}{word}"
        if current and draw.textbbox((0, 0), candidate, font=font)[2] > w:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    max_lines = max(1, int(item.get("max_lines") or 2))
    lines = lines[:max_lines] or [""]
    line_gap = max(1, int(round(size_px * 0.16)))
    line_heights = [draw.textbbox((0, 0), line or " ", font=font)[3] for line in lines]
    total_h = sum(line_heights) + line_gap * (len(lines) - 1)
    ty = y + (h - total_h) / 2
    for line, line_h in zip(lines, line_heights):
        text_w = max(1, draw.textbbox((0, 0), line or " ", font=font)[2])
        if align == "left":
            tx = x
        elif align == "right":
            tx = x + w - text_w
        else:
            tx = x + (w - text_w) / 2
        draw.text((tx, ty), line, fill=color, font=font)
        ty += line_h + line_gap


def _arrowhead_polygon(
    arrow: dict,
    points: list[tuple[int, int]],
    canvas_w: int,
    canvas_h: int,
) -> list[tuple[float, float]]:
    """Triangle drawn on the terminal segment.

    The composition stage stores the head length and width it derived from the
    link a connector actually spans, so the preview and the exported PPTX show
    the same arrow instead of two independent guesses.
    """
    (x1, y1), (x2, y2) = points[-2], points[-1]
    dx, dy = x2 - x1, y2 - y1
    length = max(math.hypot(dx, dy), 1.0)
    ux, uy = dx / length, dy / length
    stored_length = arrow.get("arrowhead_length_percent")
    stored_width = arrow.get("arrowhead_width_percent")
    if isinstance(stored_length, (int, float)) and float(stored_length) > 0:
        size = max(1.0, float(stored_length) * canvas_w)
    else:
        line_width = max(1, int(round(float(arrow.get("stroke_width_pt") or 1.5) * canvas_h / 277.0)))
        size = float(max(5, int(round(line_width * 3.2))))
    if isinstance(stored_width, (int, float)) and float(stored_width) > 0:
        half = max(0.5, float(stored_width) * canvas_w / 2.0)
    else:
        half = size * 0.55
    px, py = -uy, ux
    return [
        (x2, y2),
        (x2 - ux * size + px * half, y2 - uy * size + py * half),
        (x2 - ux * size - px * half, y2 - uy * size - py * half),
    ]


def _draw_arrow(draw: ImageDraw.ImageDraw, arrow: dict, canvas_w: int, canvas_h: int) -> None:
    path = arrow.get("path_percent")
    if not isinstance(path, list) or len(path) < 2:
        return
    points: list[tuple[int, int]] = []
    for point in path:
        if isinstance(point, list) and len(point) >= 2:
            points.append((int(round(float(point[0]) * canvas_w)), int(round(float(point[1]) * canvas_h))))
    if len(points) < 2:
        return
    color = _hex_color(arrow.get("stroke_color") or arrow.get("outline_color"), "#3F5063")
    line_width = max(1, int(round(float(arrow.get("stroke_width_pt") or 1.5) * canvas_h / 277.0)))
    render_style = str(arrow.get("render_style") or "").lower()
    if render_style == "filled_block_arrow":
        fill = _hex_color(arrow.get("fill_color"), "#AFC6DE")
        start, end = points[0], points[-1]
        thickness = max(8, int(round(float(arrow.get("block_arrow_thickness_percent") or 0.052) * canvas_h)))
        if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
            y = int(round((start[1] + end[1]) / 2))
            left, right = sorted([start[0], end[0]])
            draw.rounded_rectangle((left, y - thickness // 2, right, y + thickness // 2), radius=thickness // 3, fill=fill, outline=color, width=line_width)
        else:
            x = int(round((start[0] + end[0]) / 2))
            top, bottom = sorted([start[1], end[1]])
            draw.rounded_rectangle((x - thickness // 2, top, x + thickness // 2, bottom), radius=thickness // 3, fill=fill, outline=color, width=line_width)
        return
    draw.line(points, fill=color, width=line_width, joint="curve")
    # Match the editable PPTX compiler: one triangular arrowhead at the
    # terminal segment, so previews communicate direction rather than a bare
    # line. Feedback paths keep the same endpoint semantics.
    draw.polygon(_arrowhead_polygon(arrow, points, canvas_w, canvas_h), fill=color)


def _draw_card(draw: ImageDraw.ImageDraw, card: dict, canvas_w: int, canvas_h: int) -> None:
    bbox = card.get("bbox_percent")
    if not isinstance(bbox, dict):
        return
    x, y, w, h = _bbox_px(bbox, canvas_w, canvas_h)
    stroke = _hex_color(card.get("stroke_color"), "#59AFCB")
    fill = None if float(card.get("fill_transparency", 1.0)) >= 1.0 else _hex_color(card.get("fill_color"), "#FFFFFF")
    line_width = max(1, int(round(float(card.get("stroke_width_pt") or 1.5) * canvas_h / 360.0)))
    radius = max(0, int(round(float(card.get("corner_radius") or 0.08) * min(w, h))))
    rect = (x, y, x + w, y + h)
    shape_kind = str(card.get("shape_kind") or "rounded_rect").lower()
    if shape_kind == "rect":
        draw.rectangle(rect, fill=fill, outline=stroke, width=line_width)
    elif shape_kind in {"oval", "circle", "ellipse"}:
        draw.ellipse(rect, fill=fill, outline=stroke, width=line_width)
    else:
        draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=stroke, width=line_width)


def _draw_title_block(draw: ImageDraw.ImageDraw, program: dict, canvas_w: int, canvas_h: int) -> None:
    """Mirror the editable PPT title block in the review preview."""
    title_block = program.get("title_block")
    if not isinstance(title_block, dict) or not isinstance(title_block.get("bbox_percent"), dict):
        return
    bbox = title_block["bbox_percent"]
    title = str(title_block.get("title") or "").strip()
    subtitle = str(title_block.get("subtitle") or "").strip()
    if not title and not subtitle:
        return
    canvas = program.get("canvas") if isinstance(program.get("canvas"), dict) else {}
    height_in = float(canvas.get("height_in") or 7.5)
    block_height_in = float(bbox.get("h") or 0) * height_in
    # Same arithmetic as the composer: title_h = min(h * 0.58, 0.42in), the
    # subtitle sits at title_h * 0.94 and takes the rest up to 0.26in.
    title_height_in = min(block_height_in * 0.58, 0.42)
    subtitle_height = max(0.18, min(block_height_in - title_height_in, 0.26)) / height_in
    title_height = title_height_in / height_in
    heading_family = _families(program)[0]
    if title:
        _draw_text(
            draw,
            {
                "text": title,
                "bbox_percent": {"x": bbox["x"], "y": bbox["y"], "w": bbox["w"], "h": title_height},
                "font_size_pt": title_block.get("title_font_size", 24),
                "bold": True,
                "color_hex": title_block.get("title_color", "#16384A"),
            },
            canvas_w,
            canvas_h,
            heading_family,
        )
    if subtitle:
        _draw_text(
            draw,
            {
                "text": subtitle,
                "bbox_percent": {"x": bbox["x"], "y": float(bbox["y"]) + title_height_in * 0.94 / height_in, "w": bbox["w"], "h": subtitle_height},
                "font_size_pt": title_block.get("subtitle_font_size", 11),
                "bold": False,
                "color_hex": title_block.get("subtitle_color", "#333333"),
            },
            canvas_w,
            canvas_h,
            heading_family,
        )


def _draw_semantic_node(draw: ImageDraw.ImageDraw, node: dict, canvas_w: int, canvas_h: int, program: dict) -> None:
    if str(node.get("render_policy") or "").casefold() in {"panel_only", "hidden"}:
        return
    bbox = node.get("bbox_percent")
    if not isinstance(bbox, dict):
        return
    x, y, w, h = _bbox_px(bbox, canvas_w, canvas_h)
    stroke = _hex_color(node.get("stroke_color"), "#1F6F8B")
    fill = None if float(node.get("fill_transparency", 0.0)) >= 1.0 else _hex_color(node.get("fill_color"), "#F5FAFC")
    line_width = max(1, int(round(float(node.get("stroke_width_pt") or 1.6) * canvas_h / 360.0)))
    draw.rounded_rectangle((x, y, x + w, y + h), radius=max(2, min(12, h // 10)), fill=fill, outline=stroke, width=line_width)
    if not node_text_owned_by_program(program, node):
        _draw_text(draw, {"bbox_percent": node.get("label_bbox_percent") or bbox, "text": node.get("label") or node.get("id"), "font_size_pt": node.get("font_size_pt") or 12, "bold": node.get("bold", True), "color_hex": node.get("text_color") or stroke}, canvas_w, canvas_h)


def _fallback_arrow_path(arrow: dict, objects: dict[str, dict]) -> list[list[float]]:
    source = objects.get(str(arrow.get("source") or arrow.get("source_id")))
    target = objects.get(str(arrow.get("target") or arrow.get("target_id")))
    if not source or not target:
        return []
    sb, tb = source.get("bbox_percent"), target.get("bbox_percent")
    if not isinstance(sb, dict) or not isinstance(tb, dict):
        return []
    return [[float(sb["x"] + sb["w"] / 2), float(sb["y"] + sb["h"] / 2)], [float(tb["x"] + tb["w"] / 2), float(tb["y"] + tb["h"] / 2)]]


def render_rebuild_preview(program: dict, out_dir: str | Path, preview_path: str | Path | None = None) -> Path:
    out = Path(out_dir)
    canvas = program.get("canvas", {}) if isinstance(program.get("canvas"), dict) else {}
    width = int(canvas.get("width_px") or 1600)
    height = int(canvas.get("height_px") or 900)
    background = _hex_color(canvas.get("background"), "#FFFFFF")
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)

    _draw_title_block(draw, program, width, height)

    style = program.get("style", {}) if isinstance(program.get("style"), dict) else {}
    panel_styles = style.get("panel_styles", {}) if isinstance(style.get("panel_styles"), dict) else {}
    palette = style.get("palette") or ["#F7F8F7", "#D2DCDE", "#A09596"]

    for idx, panel in enumerate(program.get("panels", [])):
        bbox = panel.get("bbox_percent")
        if not isinstance(bbox, dict):
            continue
        x, y, w, h = _bbox_px(bbox, width, height)
        local_style = panel_styles.get(panel.get("id"), {}) if isinstance(panel_styles.get(panel.get("id")), dict) else {}
        fill = _hex_color(local_style.get("fill_color"), palette[idx % len(palette)])
        stroke = _hex_color(local_style.get("stroke_color"), "#FFFFFF")
        draw.rounded_rectangle((x, y, x + w, y + h), radius=max(2, min(12, h // 18)), fill=fill, outline=stroke, width=2)
        header = _hex_color(local_style.get("header_color"), stroke)
        canvas_height_in = float(canvas.get("height_in") or 7.5)
        header_h = max(1, min(int(round(0.34 / canvas_height_in * height)), int(round(h * 0.17))))
        draw.rounded_rectangle((x, y, x + w, y + header_h), radius=max(2, min(12, h // 18)), fill=header, outline=header, width=1)
        _draw_panel_header_title(
            draw,
            program,
            panel,
            local_style,
            {"x": x / width, "y": y / height, "w": w / width, "h": header_h / height},
            canvas,
            (width, height),
            fill_color=fill,
            header_color=header,
        )

    for card in sorted(program.get("cards", []), key=lambda item: int(item.get("z_index") or 12)):
        if isinstance(card, dict):
            _draw_card(draw, card, width, height)

    objects = {str(item.get("id")): item for key in ("semantic_nodes", "slots", "cards", "panels") for item in program.get(key, []) if isinstance(item, dict) and item.get("id")}
    for node in sorted(program.get("semantic_nodes", []), key=lambda item: int(item.get("z_index") or 30)):
        if isinstance(node, dict):
            _draw_semantic_node(draw, node, width, height, program)

    for arrow in program.get("arrows", []):
        if isinstance(arrow, dict):
            item = dict(arrow)
            if len(item.get("path_percent") or []) < 2:
                item["path_percent"] = _fallback_arrow_path(item, objects)
            _draw_arrow(draw, item, width, height)

    asset_by_id = {str(asset.get("id")): asset for asset in program.get("assets", []) if isinstance(asset, dict)}
    for slot in sorted(program.get("slots", []), key=lambda item: int(item.get("z_index") or 20)):
        bbox = slot.get("bbox_percent")
        if not isinstance(bbox, dict):
            continue
        x, y, w, h = _bbox_px(bbox, width, height)
        asset = asset_by_id.get(str(slot.get("asset_id") or slot.get("id")))
        fallback_asset_path = out / "assets" / f"{slot.get('asset_id')}.png"
        configured_path = Path(str(asset.get("path"))) if asset and asset.get("path") else None
        asset_path = configured_path if configured_path and configured_path.exists() else fallback_asset_path
        if asset_path.exists():
            with Image.open(asset_path) as asset_img:
                asset_img = asset_img.convert("RGBA")
                asset_img.thumbnail((w, h), Image.Resampling.LANCZOS)
                left = x + (w - asset_img.width) // 2
                top = y + (h - asset_img.height) // 2
                image.paste(asset_img, (left, top), asset_img)

    text_program = program.get("text_program") if isinstance(program.get("text_program"), dict) else {}
    text_items = text_program.get("items", []) if isinstance(text_program, dict) else []
    visible_items = [item for item in text_items if isinstance(item, dict) and item.get("visible", True)]
    families = _families(program)
    for item in sorted(visible_items, key=lambda item: (float(item.get("z_index") or 80), str(item.get("id") or ""))):
        if _is_panel_title_item(item) and not _is_ocr_bound(item):
            # A non-OCR panel heading is painted by its panel's header band
            # above.  Drawing the standalone box as well puts the same words on
            # the pale panel fill below the band.
            continue
        _draw_text(draw, item, width, height, _family_for_item(item, families))

    target = Path(preview_path) if preview_path else out / "rebuild_preview.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target
