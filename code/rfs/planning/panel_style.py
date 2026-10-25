"""Deterministic panel color assignment shared by Icon and PPT planning."""

from __future__ import annotations

from typing import Any, Iterable
import colorsys


DEFAULT_PALETTE = ["#EAF1F8", "#E8F3F0", "#F4ECF8", "#F7F1E5"]
DEFAULT_BORDERS = ["#2A63A1", "#1F756F", "#7251A1", "#B06B2C"]
DEFAULT_TEXT = "#173042"

#: Text is read at the same minimum the readability check enforces, so a band
#: that cannot carry a readable title is deepened here instead of being found
#: after the figure is compiled and sent back for a redo.
MIN_TEXT_CONTRAST = 4.5
_LIGHTNESS_STEP = 0.05
_LIGHTNESS_STEPS = 20


def _hex(value: Any, fallback: str) -> str:
    """Accept a colour only when every character after the hash really is hex."""
    text = str(value or "").strip()
    body = text[1:] if text.startswith("#") else ""
    return text if len(body) == 6 and all(char in "0123456789abcdefABCDEF" for char in body) else fallback


def _rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _surface(value: str, amount: float = 0.82) -> str:
    r, g, b = (channel / 255 for channel in _rgb(value))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Preserve hue while making a color suitable for a large panel surface.
    l = max(l, amount)
    s = min(s, 0.35)
    rr, gg, bb = colorsys.hls_to_rgb(h, l, s)
    return "#%02X%02X%02X" % (round(rr * 255), round(gg * 255), round(bb * 255))


def _luminance(value: str) -> float:
    channels = []
    for channel in _rgb(value):
        c = channel / 255
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _shift_lightness(value: str, delta: float) -> str:
    """Move a surface's lightness while keeping its hue and saturation."""
    r, g, b = (channel / 255 for channel in _rgb(value))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    rr, gg, bb = colorsys.hls_to_rgb(h, min(max(l + delta, 0.0), 1.0), s)
    return "#%02X%02X%02X" % (round(rr * 255), round(gg * 255), round(bb * 255))


def _readable_on(surface: str, preferences: Iterable[Any], minimum: float = MIN_TEXT_CONTRAST) -> tuple[str, str]:
    """Return the surface and a text colour that really reads on it.

    The first preference that clears the minimum wins, so a theme keeps its own
    text colour whenever that colour is legible. Only when none of them clears
    it is the surface itself walked towards black or white: the colour keeps its
    hue, and a mid-tone band stops carrying small white text at 3.9:1.
    """
    choices = [text for text in (_hex(item, "") for item in preferences) if text]
    if not choices:
        choices = ["#FFFFFF", DEFAULT_TEXT]
    for text in choices:
        if _contrast(text, surface) >= minimum:
            return surface, text
    for step in range(1, _LIGHTNESS_STEPS + 1):
        for delta in (-_LIGHTNESS_STEP * step, _LIGHTNESS_STEP * step):
            moved = _shift_lightness(surface, delta)
            for text in choices:
                if _contrast(text, moved) >= minimum:
                    return moved, text
    return surface, choices[0]


def build_panel_style_plan(panels: list[dict[str, Any]], reference_profile: dict[str, Any] | None = None, exemplar_brief: dict[str, Any] | None = None, color_theme: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve VLM/reference colors into role-aware, readable panel styles."""

    if isinstance(color_theme, dict) and isinstance(color_theme.get("panels"), dict):
        theme = {**color_theme, "panels": dict(color_theme["panels"])}
        result = []
        for index, panel in enumerate(panels):
            panel_id = str(panel.get("id") or "")
            spec = theme["panels"].get(panel_id)
            if not isinstance(spec, dict):
                break
            fill = _hex(spec.get("fill"), DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)])
            border = _hex(spec.get("border"), DEFAULT_BORDERS[index % len(DEFAULT_BORDERS)])
            header, header_text = _readable_on(_hex(spec.get("header"), border), (spec.get("header_text"), "#FFFFFF", DEFAULT_TEXT))
            fill, body_text = _readable_on(fill, (spec.get("body_text"), DEFAULT_TEXT, "#FFFFFF"))
            # The plan is echoed into the program and the colour check compares
            # that echo with the theme, so a resolved colour is written to both.
            theme["panels"][panel_id] = {**spec, "fill": fill, "border": border, "header": header, "header_text": header_text, "body_text": body_text}
            result.append({"panel_id": panel_id, "fill_color_hex": fill, "border_color_hex": border, "accent_color_hex": header, "header_color_hex": header, "header_text_color_hex": header_text, "text_color_hex": body_text, "body_text_color_hex": body_text, "icon_background_policy": "transparent_panel_inherit", "color_rules": [], "contrast_ratio": round(_contrast(body_text, fill), 2), "header_contrast_ratio": round(_contrast(header_text, header), 2)})
        if len(result) == len(panels):
            return {"contract": "rfs.panel_style_plan", "version": 2, "panels": result, "palette_source": str((theme.get("provenance") or {}).get("source") or "vlm"), "color_theme": theme, "exemplar_color_rules": [], "palette_decision_trace": [{"panel_id": item["panel_id"], "source_color": item["fill_color_hex"], "fill_color": item["fill_color_hex"], "border_color": item["border_color_hex"], "text_color": item["text_color_hex"], "reason": "color_theme_planner"} for item in result], "contrast_policy": {"min_text_contrast": MIN_TEXT_CONTRAST, "min_large_text_contrast": 3.0}}
    style = (reference_profile or {}).get("style") or {}
    palette = [str(item) for item in style.get("palette", []) if isinstance(item, str) and item.startswith("#")] or DEFAULT_PALETTE
    rules = (exemplar_brief or {}).get("color_rules") or (reference_profile or {}).get("exemplar_color_rules") or []
    result, trace = [], []
    for index, panel in enumerate(panels):
        raw = _hex(palette[index % len(palette)], DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)])
        fill = _surface(raw) if _luminance(raw) < 0.72 else raw
        border = raw if _contrast(raw, fill) >= 1.35 else _hex(DEFAULT_BORDERS[index % len(DEFAULT_BORDERS)], DEFAULT_BORDERS[index % len(DEFAULT_BORDERS)])
        band, band_text = _readable_on(border, ("#FFFFFF", DEFAULT_TEXT))
        fill, text = _readable_on(fill, (DEFAULT_TEXT, "#FFFFFF"))
        panel_id = str(panel.get("id") or f"panel_{index + 1:02d}")
        result.append({"panel_id": panel_id, "fill_color_hex": fill, "border_color_hex": border, "accent_color_hex": band, "header_color_hex": band, "header_text_color_hex": band_text, "text_color_hex": text, "body_text_color_hex": text, "icon_background_policy": "transparent_panel_inherit", "color_rules": rules[:8], "contrast_ratio": round(_contrast(text, fill), 2), "header_contrast_ratio": round(_contrast(band_text, band), 2)})
        trace.append({"panel_id": panel_id, "source_color": raw, "fill_color": fill, "border_color": border, "header_color": band, "header_text_color": band_text, "text_color": text, "reason": "surface_for_large_area" if fill != raw else "reference_color_usable_as_surface"})
    source_name = str((reference_profile or {}).get("sources", {}).get("style") or "fallback")
    return {"contract": "rfs.panel_style_plan", "version": 2, "panels": result, "palette_source": source_name, "exemplar_color_rules": rules[:8], "palette_decision_trace": trace, "contrast_policy": {"min_text_contrast": MIN_TEXT_CONTRAST, "min_large_text_contrast": 3.0}}
