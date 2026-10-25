"""Deterministic global text measurement and fitting for editable figures."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from PIL import ImageFont

from .fonts import heading_font_family, load_font


ROLE_SIZE_RANGES = {
    "panel_title": (14.0, 20.0),
    "primary_node": (12.0, 16.0),
    "node_label": (10.0, 16.0),
    "secondary_node": (10.0, 13.0),
    "connector_label": (8.0, 10.0),
}

#: The paper title is drawn inside its own block, not among the node labels, so
#: its size is resolved here instead of by the owner-lane packing below.
TITLE_FONT_CEILING_PT = 20.0
TITLE_MIN_FONT_SIZE_PT = 9.0
TITLE_MAX_LINES = 2


def _role_range(item: dict[str, Any], default_min: float, default_max: float) -> tuple[float, float]:
    role = str(item.get("font_size_role") or item.get("text_role") or item.get("role") or "")
    if role == "node_label":
        role = "primary_node" if int(item.get("priority") or 2) <= 1 else "secondary_node"
    return ROLE_SIZE_RANGES.get(role, (default_min, default_max))


def _font(size: int, family: str | None = None, bold: bool = False) -> ImageFont.ImageFont:
    return load_font(max(1, size), family, bold=bold)


def _measure(text: str, size: float, family: str | None = None) -> tuple[float, float]:
    # PIL measures pixels while the contracts specify typographic points.
    # Use the conventional 96-DPI conversion instead of treating 1 pt as 1 px.
    font = _font(max(1, round(size * 96.0 / 72.0)), family)
    left, top, right, bottom = font.getbbox(text or " ")
    return max(1.0, float(right - left)), max(1.0, float(bottom - top))


def _wrap(text: str, max_width_px: float, size: float, family: str | None) -> list[str]:
    words = list(text) if " " not in text.strip() else text.split()
    separator = "" if " " not in text.strip() else " "
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current}{separator if current else ''}{word}"
        if current and _measure(candidate, size, family)[0] > max_width_px:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _drawn_title_height_px(title_block: dict[str, Any], height_in: float) -> float:
    """Height the PPT compiler gives the title inside its block, in pixels."""
    block_height_in = float(title_block["bbox_percent"].get("h") or 0.0) * height_in
    return max(0.0, min(block_height_in * 0.58, 0.42)) * 96.0


def fit_title_block(program: dict[str, Any], *, width_in: float, height_in: float) -> dict[str, Any] | None:
    """Shrink the paper title until its ink fits the box the compiler draws.

    The compiler gives the title the upper 58% of the title block, capped at
    0.42 in, while the block itself is specified in canvas shares rather than in
    text. A long paper title therefore wrapped to a second line whose ink landed
    below the block - and, on a full canvas, on top of the first panel. Nothing
    measured that overlap, because the title is not part of the text program.
    Measuring it here keeps one authority for text size and reports the result
    in the global text report instead of leaving it visible only in the picture.
    """

    block = program.get("title_block")
    if not isinstance(block, dict) or not isinstance(block.get("bbox_percent"), dict):
        return None
    title = str(block.get("title") or "").strip()
    if not title:
        return None
    box = block["bbox_percent"]
    style = program.get("style") if isinstance(program.get("style"), dict) else {}
    # The composer paints the paper title with the heading font, so the size
    # that fits has to be measured with that font, not with the body family.
    family = heading_font_family(style)
    available_width_px = max(8.0, float(box.get("w") or 0.0) * width_in * 96.0 * 0.96)
    available_height_px = _drawn_title_height_px(block, height_in)
    requested = float(block.get("title_font_size") or TITLE_FONT_CEILING_PT)
    selected, lines = TITLE_MIN_FONT_SIZE_PT, [title]
    for candidate in range(round(TITLE_FONT_CEILING_PT * 2), round(TITLE_MIN_FONT_SIZE_PT * 2) - 1, -1):
        size = candidate / 2
        wrapped = _wrap(title, available_width_px, size, family)
        if len(wrapped) <= TITLE_MAX_LINES and len(wrapped) * _measure("Ag", size, family)[1] * 1.22 <= available_height_px:
            selected, lines = size, wrapped
            break
    block["title_font_size"] = round(selected, 2)
    block["measured_lines"] = lines
    block["measured_line_count"] = len(lines)
    return {
        "text_id": "title_block",
        "font_size_role": "paper_title",
        "requested_font_size_pt": requested,
        "selected_font_size_pt": round(selected, 2),
        "minimum_readable_size_pt": TITLE_MIN_FONT_SIZE_PT,
        "line_count": len(lines),
        "overflow": len(lines) * _measure("Ag", selected, family)[1] * 1.22 > available_height_px,
        "vertical_wrap_risk": False,
        "bbox_expansion": None,
    }


def fit_text_program_globally(program: dict[str, Any], *, min_font_size_pt: float = 6.0, max_font_size_pt: float = 18.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit all editable text boxes against one shared font measurement model."""

    updated = deepcopy(program)
    canvas = updated.get("canvas") if isinstance(updated.get("canvas"), dict) else {}
    width_in = float(canvas.get("width_in") or 13.333)
    height_in = float(canvas.get("height_in") or 7.5)
    text_program = updated.get("text_program") if isinstance(updated.get("text_program"), dict) else {"items": []}
    owners = {
        str(item.get("id") or ""): item["bbox_percent"]
        for key in ("cards", "panels", "slots", "semantic_nodes")
        for item in updated.get(key, []) or []
        if isinstance(item, dict) and isinstance(item.get("bbox_percent"), dict) and item.get("id")
    }
    reports: list[dict[str, Any]] = []
    for item in text_program.get("items", []) if isinstance(text_program.get("items"), list) else []:
        if not isinstance(item, dict) or not item.get("visible", True) or not isinstance(item.get("bbox_percent"), dict):
            continue
        box = item["bbox_percent"]
        available_width_px = max(8.0, float(box.get("w", 0)) * width_in * 96 * 0.88)
        available_height_px = max(8.0, float(box.get("h", 0)) * height_in * 96 * 0.82)
        role_min, role_max = _role_range(item, min_font_size_pt, max_font_size_pt)
        explicit_min = item.get("min_readable_size_pt")
        absolute_min = max(8.0, float(explicit_min)) if explicit_min is not None else role_min
        requested_default = (role_min + role_max) / 2
        requested = max(absolute_min, min(role_max, float(item.get("font_size_pt") or requested_default)))
        family = str(item.get("font_family_guess") or "") or None
        text = str(item.get("text") or "")
        original_box = dict(box)
        expansion = None
        compact_text = " ".join(text.split())
        estimated_chars_per_line = max(1.0, available_width_px / max(_measure("M", requested, family)[0], 1.0))
        if len(compact_text) >= 8 and estimated_chars_per_line < 4:
            target_width = min(0.24, max(float(box.get("w", 0)), min(0.18, len(compact_text) * 0.006)))
            owner_box = owners.get(str(item.get("target_id") or item.get("owner_id") or ""))
            if owner_box:
                target_width = min(target_width, max(0.02, float(owner_box["w"]) * 0.92))
            center_x = float(box.get("x", 0)) + float(box.get("w", 0)) / 2
            new_x = max(0.0, min(1.0 - target_width, center_x - target_width / 2))
            if owner_box:
                new_x = min(max(new_x, float(owner_box["x"]) + 0.004), float(owner_box["x"]) + float(owner_box["w"]) - target_width - 0.004)
            box.update({"x": round(new_x, 4), "w": round(target_width, 4)})
            available_width_px = max(8.0, target_width * width_in * 96 * 0.88)
            expansion = {"from": original_box, "to": dict(box), "reason": "prevent_character_per_line_wrapping"}
        # A label whose longest word is wider than its box is not fitted by wrapping:
        # _wrap falls back to splitting the word, which is how a delivered deck showed
        # "Unstruc / tured" and "Questio / n". When the word does fit the box at the
        # readable floor, the size search keeps shrinking until it does; a word that
        # cannot fit even there is reported instead, because the box is what has to
        # change and a smaller size would only make it unreadable.
        words = compact_text.split() or [compact_text]
        longest_word = max(words, key=lambda word: _measure(word, absolute_min, family)[0])
        word_fits_at_floor = _measure(longest_word, absolute_min, family)[0] <= available_width_px
        selected = absolute_min
        selected_lines = [text]
        fitted = False
        for candidate_size in range(round(requested * 2), round(absolute_min * 2) - 1, -1):
            size = candidate_size / 2
            if word_fits_at_floor and _measure(longest_word, size, family)[0] > available_width_px:
                continue
            lines = _wrap(text, available_width_px, size, family)
            line_height = _measure("Ag", size, family)[1] * 1.22
            if len(lines) <= int(item.get("max_lines") or 2) and len(lines) * line_height <= available_height_px:
                selected = size
                selected_lines = lines
                fitted = True
                break
        item["font_size_pt"] = round(selected, 2)
        item["measured_lines"] = selected_lines
        item["measured_line_count"] = len(selected_lines)
        item["fit_strategy"] = "global_real_font_measurement"
        overflow = len(selected_lines) * _measure("Ag", selected, family)[1] * 1.22 > available_height_px
        vertical_wrap_risk = len(compact_text) >= 8 and max((len(line) for line in selected_lines), default=0) < 4
        # Nothing fitted at the floor means the compiler will draw the label with
        # PowerPoint's own wrap, which breaks the word; that is the same defect and
        # is reported as one.
        word_break_risk = (not fitted) or (longest_word not in "\n".join(selected_lines))
        reports.append({"text_id": item.get("id"), "font_size_role": item.get("font_size_role") or item.get("text_role") or item.get("role"), "requested_font_size_pt": requested, "selected_font_size_pt": selected, "minimum_readable_size_pt": absolute_min, "line_count": len(selected_lines), "overflow": overflow, "vertical_wrap_risk": vertical_wrap_risk, "longest_word": longest_word, "word_break_risk": word_break_risk, "bbox_expansion": expansion})
    # Pack text belonging to the same card/panel into non-overlapping lanes.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in text_program.get("items", []) if isinstance(text_program.get("items"), list) else []:
        owner = str(item.get("target_id") or item.get("owner_id") or "")
        role = str(item.get("text_role") or item.get("role") or "").casefold()
        # Node labels already have an authoritative bbox derived jointly with
        # the icon slot by arrange_node_contents(). Generic owner packing must
        # not overwrite that disjoint internal layout.
        if role == "node_label":
            continue
        if owner in owners and isinstance(item.get("bbox_percent"), dict):
            grouped.setdefault(owner, []).append(item)
    for owner_id, members in grouped.items():
        owner_box = owners[owner_id]
        members.sort(key=lambda value: (float(value["bbox_percent"].get("y", 0)), str(value.get("id") or "")))
        cursor = float(owner_box["y"]) + 0.035
        bottom = float(owner_box["y"]) + float(owner_box["h"]) - 0.012
        for item in members:
            box = item["bbox_percent"]
            height = min(float(box.get("h", 0.03)), max(0.016, bottom - cursor))
            if height <= 0.012:
                item["visible"] = False
                continue
            box["x"] = round(float(owner_box["x"]) + 0.012, 4)
            box["w"] = round(max(0.02, float(owner_box["w"]) - 0.024), 4)
            box["y"] = round(cursor, 4)
            box["h"] = round(height, 4)
            cursor += height + 0.008
    updated["text_program"] = text_program
    title_report = fit_title_block(updated, width_in=width_in, height_in=height_in)
    if title_report:
        reports.insert(0, title_report)
    blocked = any(item["overflow"] or item["vertical_wrap_risk"] for item in reports)
    # A word wider than its box is reported and counted, but it is not treated as an
    # overflow: the size that would break it is already the readable floor, so the
    # missing fix is a wider box and that belongs to the layout stage, not here.
    word_break_count = sum(bool(item.get("word_break_risk")) for item in reports)
    return updated, {"summary": "Global real-font text measurement report.", "status": "blocked" if blocked else "pass", "text_count": len(reports), "overflow_count": sum(bool(item["overflow"]) for item in reports), "vertical_wrap_risk_count": sum(bool(item["vertical_wrap_risk"]) for item in reports), "word_break_count": word_break_count, "word_break_text_ids": [item.get("text_id") for item in reports if item.get("word_break_risk")], "expanded_bbox_count": sum(bool(item["bbox_expansion"]) for item in reports), "items": reports}
