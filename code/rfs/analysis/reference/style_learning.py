from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ...utils import ensure_dir, write_json
from ...providers import call_vlm_json, vlm_credentials_available
from .library_learning import DEFAULT_REFERENCE_ANALYSIS_MODEL, SUPPORTED_IMAGE_SUFFIXES


USER_STYLE_FIELDS = [
    "color_rules",
    "line_rules",
    "shape_rules",
    "shadow_depth_rules",
    "typography_style_rules",
    "texture_rendering_rules",
    "icon_object_style_rules",
    "panel_surface_rules",
    "overall_mood",
    "avoid_style_rules",
    "anti_copy_rules",
]


def _dominant_colors(path: Path, count: int = 6) -> list[str]:
    with Image.open(path) as image:
        thumb = image.convert("RGB").resize((96, 96)).quantize(colors=count).convert("RGB")
        colors = sorted(thumb.getcolors(96 * 96) or [], reverse=True)
    return [f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}" for _amount, rgb in colors[:count]]


def empty_user_style_brief() -> dict[str, Any]:
    return {
        "color_rules": [],
        "line_rules": [],
        "shape_rules": [],
        "shadow_depth_rules": [],
        "typography_style_rules": [],
        "texture_rendering_rules": [],
        "icon_object_style_rules": [],
        "panel_surface_rules": [],
        "overall_mood": "unknown",
        "avoid_style_rules": [],
        "anti_copy_rules": [],
    }


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_user_style_brief(raw: dict[str, Any] | None) -> dict[str, Any]:
    data = raw if isinstance(raw, dict) else {}
    result = empty_user_style_brief()
    for key in USER_STYLE_FIELDS:
        if key == "overall_mood":
            result[key] = str(data.get(key) or result[key])
        else:
            result[key] = _list(data.get(key))
    return result


def _heuristic_brief(path: Path) -> dict[str, Any]:
    brief = empty_user_style_brief()
    colors = _dominant_colors(path)
    brief["color_rules"] = [f"use dominant style colors {', '.join(colors[:4])}"]
    brief["line_rules"] = ["unknown line style; keep academic thin strokes"]
    brief["shape_rules"] = ["unknown shape language; keep clean diagram primitives"]
    brief["shadow_depth_rules"] = ["unknown depth; keep shadows restrained"]
    brief["typography_style_rules"] = ["short readable academic labels"]
    brief["texture_rendering_rules"] = ["clean raster rendering without noisy textures"]
    brief["icon_object_style_rules"] = ["consistent scientific icon/object rendering"]
    brief["panel_surface_rules"] = ["clean panel surfaces with clear contrast"]
    brief["overall_mood"] = "user style inferred heuristically from image colors"
    brief["avoid_style_rules"] = ["do not copy user image content, labels, captions, logos, formulas, or paper-specific objects"]
    return brief


def _style_prompt(record: dict[str, Any]) -> str:
    return f"""
# Summary

Analyze this image only as a user visual style preference for future scientific figure generation. Return JSON only.

Extract reusable style, not content. The current project will analyze the paper, choose layout/structure, select project exemplars, and validate scientific correctness separately.

Style reference metadata:
{json.dumps(record, ensure_ascii=False, indent=2)}

Return exactly these style-only fields:
{{
  "color_rules": [],
  "line_rules": [],
  "shape_rules": [],
  "shadow_depth_rules": [],
  "typography_style_rules": [],
  "texture_rendering_rules": [],
  "icon_object_style_rules": [],
  "panel_surface_rules": [],
  "overall_mood": "...",
  "avoid_style_rules": [],
  "anti_copy_rules": []
}}

Rules:
- Do not extract or reuse paper modules, labels, formulas, captions, logos, dataset names, method names, or unique scientific objects.
- Do not infer layout topology, section count, reading order, or scientific relations from this image.
- Fill anti_copy_rules with visible terms, logos, captions, formulas, method names, dataset names, and unique objects that must not be copied.
- Use short reusable visual-style phrases.
""".strip()


def learn_reference_style(
    style_reference: str | Path,
    out: str | Path,
    mode: str = "vlm",
    model: str | None = None,
    vlm_adapter: Callable[[str, list[str | Path], str | None, int | None, int | None], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode not in {"heuristic", "vlm"}:
        raise ValueError("User style learning mode must be heuristic or vlm")
    source = Path(style_reference).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Style reference does not exist: {source}")
    if source.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported style reference image suffix: {source.suffix}")
    if mode == "vlm" and not vlm_adapter and not vlm_credentials_available():
        raise RuntimeError("VLM user style learning requires API_BASE and API_KEY/GEMINI_API_KEY environment variables")

    output = ensure_dir(out).resolve()
    resolved_model = model or DEFAULT_REFERENCE_ANALYSIS_MODEL
    with Image.open(source) as image:
        width, height = image.size
    record = {
        "summary": "User style reference analysis record.",
        "source_path": str(source),
        "analysis_mode": "heuristic",
        "image": {
            "width": width,
            "height": height,
            "aspect_ratio": round(width / max(height, 1), 6),
            "dominant_colors": _dominant_colors(source),
        },
    }
    brief = _heuristic_brief(source)
    failures: list[dict[str, str]] = []
    prompt = _style_prompt(record)
    if mode == "vlm":
        try:
            raw = vlm_adapter(prompt, [source], resolved_model, None, None) if vlm_adapter else call_vlm_json(prompt, [source], model=resolved_model)
            brief = normalize_user_style_brief(raw)
            record["analysis_mode"] = "vlm"
        except Exception as exc:
            record["analysis_mode"] = "heuristic_after_vlm_failure"
            failures.append({"path": str(source), "error": str(exc)})

    write_json(output / "user_style_record.json", record)
    write_json(output / "user_style_brief.json", brief)
    report = {
        "summary": "User style preference learning completed.",
        "ok": not failures,
        "style_reference": str(source),
        "out_dir": str(output),
        "analysis_mode": mode,
        "model": resolved_model if mode == "vlm" else None,
        "api_base_present": bool(os.getenv("API_BASE", "").strip()) if mode == "vlm" else None,
        "api_key_present": bool((os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()) if mode == "vlm" else None,
        "record_path": str(output / "user_style_record.json"),
        "brief_path": str(output / "user_style_brief.json"),
        "report_path": str(output / "user_style_report.json"),
        "failure_count": len(failures),
        "failures": failures,
    }
    write_json(output / "user_style_report.json", report)
    return report
