"""Deterministic readability checks for editable framework text."""

from __future__ import annotations

from typing import Any

from ..contracts.text_ownership import node_text_owned_by_program


def _rgb(value: str) -> tuple[int, int, int]:
    text = str(value or "#000000").strip().lstrip("#")
    if len(text) != 6:
        text = "000000"
    return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))


def _luminance(value: str) -> float:
    converted = []
    for channel in _rgb(value):
        raw = channel / 255
        converted.append(raw / 12.92 if raw <= 0.03928 else ((raw + 0.055) / 1.055) ** 2.4)
    return 0.2126 * converted[0] + 0.7152 * converted[1] + 0.0722 * converted[2]


def _contrast(a: str, b: str) -> float:
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def evaluate_text_readability(program: dict[str, Any]) -> dict[str, Any]:
    """Report contrast and minimum-font violations without mutating contracts."""

    style = program.get("style") if isinstance(program.get("style"), dict) else {}
    panel_styles = style.get("panel_styles") if isinstance(style.get("panel_styles"), dict) else {}
    panel_by_id = {str(item.get("id")): item for item in program.get("panels", []) if isinstance(item, dict)}
    failures, items = [], []
    text_items = list((program.get("text_program") or {}).get("items", []))
    for node in program.get("semantic_nodes", []):
        if isinstance(node, dict) and not node_text_owned_by_program(program, node):
            text_items.append({"id": f"node_{node.get('id')}", "text": node.get("label"), "font_size_pt": node.get("font_size_pt"), "color_hex": node.get("text_color"), "panel_id": node.get("panel_id"), "font_size_role": node.get("font_size_role") or "node_label"})
    for item in text_items:
        if not isinstance(item, dict) or not item.get("visible", True):
            continue
        panel_id = str(item.get("panel_id") or "")
        panel_style = panel_styles.get(panel_id, {}) if isinstance(panel_styles.get(panel_id), dict) else {}
        panel = panel_by_id.get(panel_id, {})
        role = str(item.get("font_size_role") or item.get("text_role") or item.get("role") or "")
        # A panel title sits on the header band, not on the panel body. Reading
        # the body fill as its background scores white-on-band text as all but
        # invisible, which is how a correct figure got sent back for a redo.
        on_header_band = role == "panel_title"
        background = str(
            (panel_style.get("header_color") if on_header_band else None)
            or panel_style.get("fill_color")
            or panel.get("fill_color")
            or "#FFFFFF"
        )
        foreground = str(
            item.get("color_hex")
            or item.get("resolved_color_hex")
            or (panel_style.get("header_text_color") if on_header_band else None)
            or panel_style.get("text_color")
            or "#263747"
        )
        size = float(item.get("font_size_pt") or 0)
        minimum = max(8.0, float(item.get("min_readable_size_pt") or 8.0))
        ratio = _contrast(foreground, background)
        record = {"text_id": item.get("id"), "font_size_role": item.get("font_size_role") or item.get("text_role") or item.get("role"), "font_size_pt": size, "minimum_font_size_pt": minimum, "foreground": foreground, "background": background, "contrast_ratio": round(ratio, 2)}
        if size < minimum:
            failures.append({**record, "reason": "font_below_readable_minimum"})
        if ratio < (3.0 if size >= 14 else 4.5):
            failures.append({**record, "reason": "insufficient_text_contrast"})
        items.append(record)
    blocked = any(item["reason"] in {"font_below_readable_minimum", "insufficient_text_contrast"} for item in failures)
    return {"summary": "Editable text readability quality report.", "status": "blocked" if blocked else "pass", "text_count": len(items), "failure_count": len(failures), "minimum_font_size_pt": min((item["font_size_pt"] for item in items), default=None), "failures": failures, "items": items}
