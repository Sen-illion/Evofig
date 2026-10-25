from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from ...utils import ensure_dir, write_json
from ...providers import call_vlm_json, resolve_vlm_retries, resolve_vlm_timeout, vlm_credentials_available
from ...contracts.reference_color_recipe import ALLOWED_USES, validate_reference_color_recipe
from .color_recipe import EXTRACTOR_ID, analyze_reference_color_recipe, render_palette_card


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
DEFAULT_REFERENCE_ANALYSIS_MODEL = "gpt-5.6-sol"


def _dominant_colors(path: Path, count: int = 6) -> list[str]:
    with Image.open(path) as image:
        thumb = image.convert("RGB").resize((96, 96)).quantize(colors=count).convert("RGB")
        colors = sorted(thumb.getcolors(96 * 96) or [], reverse=True)
    return [f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}" for _amount, rgb in colors[:count]]


def _classify_by_ratio(ratio: float) -> str:
    if ratio <= 0:
        return "unknown"
    if ratio >= 3.5:
        return "linear"
    if ratio >= 2.55:
        return "tripanel"
    if ratio >= 2.05:
        return "dense-multimodal"
    return "arbor"


def _empty_design_contract() -> dict[str, Any]:
    return {
        "layout": {
            "macro_pattern": "unknown",
            "regions": [],
            "reading_order": [],
        },
        "structure": {
            "section_count": None,
            "section_roles": [],
            "grouping_style": "unknown",
        },
        "density": {
            "level": "unknown",
            "module_count_estimate": None,
            "text_density": "unknown",
            "whitespace_policy": "unknown",
        },
        "hierarchy": {
            "primary_focus": "unknown",
            "secondary_focus": [],
            "emphasis_methods": [],
        },
        "color": {
            "palette": [],
            "background": "unknown",
            "contrast_policy": "unknown",
        },
        "color_recipe": {},
        "text": {
            "label_style": "unknown",
            "title_style": "unknown",
            "forbidden": [],
        },
        "connectors": {
            "grammar": [],
            "arrow_density": "unknown",
        },
        "safety": {
            "forbidden_copy_terms": [],
            "forbidden_copy_objects": [],
        },
        "why_good": [],
        "warnings": [],
    }


def _reference_analysis_prompt(record: dict[str, Any]) -> str:
    quality_guidance = (
        "Explain why this figure is a strong human-quality scientific framework figure."
        if record["quality_label"] == "positive"
        else "Identify reusable anti-patterns and design problems to avoid; why_good may be empty."
    )
    return f"""
# Summary

Analyze this scientific framework figure as reusable visual-design knowledge for future direct paper-to-PPTX generation. Return JSON only.

Do not copy or recommend paper-specific scientific content. Extract reusable design rules only.

Reference metadata:
{json.dumps({key: record.get(key) for key in ["id", "quality_label", "archetype", "image"]}, ensure_ascii=False, indent=2)}

Quality task:
{quality_guidance}

Return this JSON shape:
{{
  "layout": {{"macro_pattern": "...", "regions": [], "reading_order": []}},
  "structure": {{"section_count": 0, "section_roles": [], "grouping_style": "..."}},
  "density": {{"level": "low|medium|high|very_high|unknown", "module_count_estimate": 0, "text_density": "low|medium|high|unknown", "whitespace_policy": "..."}},
  "hierarchy": {{"primary_focus": "...", "secondary_focus": [], "emphasis_methods": []}},
  "color": {{"palette": [], "background": "...", "contrast_policy": "..."}},
  "color_recipe_semantics": {{
    "assignments": [{{"color_id": "c01", "used_for": ["panel_fill"], "semantic_roles": ["primary_processing"], "confidence": 0.0}}],
    "relationships": {{"harmony": "...", "dominant_supporting_accent_ratio": [], "saturation_hierarchy": "...", "pairings": []}},
    "transfer_rules": [],
    "avoid_copying": []
  }},
  "text": {{"label_style": "...", "title_style": "...", "forbidden": []}},
  "connectors": {{"grammar": [], "arrow_density": "low|medium|high|unknown"}},
  "safety": {{"forbidden_copy_terms": ["visible paper-specific labels, method names, dataset names, captions, or branded text"], "forbidden_copy_objects": ["paper-specific logos or unique objects"]}},
  "why_good": []
}}

Rules:
- Use short reusable phrases, not long prose.
- Preserve the schema keys exactly.
- Never recommend copying visible reference labels, equations, names, logos, or unique scientific objects.
- COLOR_RECIPE below is deterministic pixel evidence. Assign uses only to its existing color_id values; never invent, replace, or recolor them.
- Assign every significant color to one or more of: canvas, panel_fill, panel_header, node_fill, border, text, primary_connector, feedback_connector, accent.
- For positive examples, fill why_good with concrete design reasons.
- For negative examples, focus on what future generations should avoid.

COLOR_RECIPE:
{json.dumps(record.get("color_recipe") or {}, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _record_for_image(path: Path, record_id: str, quality_label: str, mode: str, palette_card_path: Path | None = None) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    ratio = width / max(height, 1)
    contract = _empty_design_contract()
    recipe = analyze_reference_color_recipe(path, f"{record_id}.color.v1", quality_label=quality_label, palette_card_path=palette_card_path)
    contract["color"]["palette"] = [item["hex"] for item in recipe["colors"]]
    contract["color_recipe"] = recipe
    return {
        "id": record_id,
        "quality_label": quality_label,
        "source_path": str(path.resolve()),
        "analysis_mode": mode,
        "image": {
            "width": width,
            "height": height,
            "aspect_ratio": round(ratio, 6),
            "dominant_colors": contract["color"]["palette"],
        },
        "archetype": _classify_by_ratio(ratio),
        **contract,
    }


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _merge_vlm_analysis(base: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    layout = _dict(raw.get("layout"))
    result["layout"] = {
        "macro_pattern": str(layout.get("macro_pattern") or result["layout"]["macro_pattern"]),
        "regions": _list(layout.get("regions")),
        "reading_order": _list(layout.get("reading_order")),
    }
    structure = _dict(raw.get("structure"))
    result["structure"] = {
        "section_count": structure.get("section_count"),
        "section_roles": _list(structure.get("section_roles")),
        "grouping_style": str(structure.get("grouping_style") or result["structure"]["grouping_style"]),
    }
    density = _dict(raw.get("density"))
    result["density"] = {
        "level": str(density.get("level") or result["density"]["level"]),
        "module_count_estimate": density.get("module_count_estimate"),
        "text_density": str(density.get("text_density") or result["density"]["text_density"]),
        "whitespace_policy": str(density.get("whitespace_policy") or result["density"]["whitespace_policy"]),
    }
    hierarchy = _dict(raw.get("hierarchy"))
    result["hierarchy"] = {
        "primary_focus": str(hierarchy.get("primary_focus") or result["hierarchy"]["primary_focus"]),
        "secondary_focus": _list(hierarchy.get("secondary_focus")),
        "emphasis_methods": _list(hierarchy.get("emphasis_methods")),
    }
    color = _dict(raw.get("color"))
    result["color"] = {
        "palette": _list(color.get("palette")) or result["color"]["palette"],
        "background": str(color.get("background") or result["color"]["background"]),
        "contrast_policy": str(color.get("contrast_policy") or result["color"]["contrast_policy"]),
    }
    semantic_recipe = _dict(raw.get("color_recipe_semantics"))
    assignments = {str(item.get("color_id")): item for item in _list(semantic_recipe.get("assignments")) if isinstance(item, dict)}
    known_ids = {str(item.get("color_id")) for item in result["color_recipe"].get("colors", [])}
    invented = sorted(set(assignments) - known_ids)
    if invented:
        raise ValueError(f"VLM color semantics invented color ids: {invented}")
    for item in result["color_recipe"].get("colors", []):
        assignment = assignments.get(str(item.get("color_id")))
        if assignment:
            uses = [str(value) for value in _list(assignment.get("used_for")) if str(value) in ALLOWED_USES and str(value) != "unassigned"]
            item["used_for"] = uses or item.get("used_for") or ["unassigned"]
            item["semantic_roles"] = [str(value) for value in _list(assignment.get("semantic_roles")) if str(value).strip()]
            item["confidence"] = max(0.0, min(1.0, float(assignment.get("confidence") or 0.0)))
    result["color_recipe"]["relationships"] = _dict(semantic_recipe.get("relationships"))
    result["color_recipe"]["transfer_rules"] = [str(value) for value in _list(semantic_recipe.get("transfer_rules")) if str(value).strip()]
    result["color_recipe"]["avoid_copying"] = [str(value) for value in _list(semantic_recipe.get("avoid_copying")) if str(value).strip()]
    result["color_recipe"].setdefault("analysis", {})["semantic_mode"] = "vlm"
    recipe_validation = validate_reference_color_recipe(result["color_recipe"], require_semantics=True)
    if not recipe_validation["ok"]:
        raise ValueError("invalid VLM color semantics: " + "; ".join(recipe_validation["errors"]))
    text = _dict(raw.get("text"))
    result["text"] = {
        "label_style": str(text.get("label_style") or result["text"]["label_style"]),
        "title_style": str(text.get("title_style") or result["text"]["title_style"]),
        "forbidden": _list(text.get("forbidden")),
    }
    connectors = _dict(raw.get("connectors"))
    result["connectors"] = {
        "grammar": _list(connectors.get("grammar")),
        "arrow_density": str(connectors.get("arrow_density") or result["connectors"]["arrow_density"]),
    }
    safety = _dict(raw.get("safety"))
    result["safety"] = {
        "forbidden_copy_terms": _list(safety.get("forbidden_copy_terms")),
        "forbidden_copy_objects": _list(safety.get("forbidden_copy_objects")),
    }
    result["why_good"] = _list(raw.get("why_good"))
    result["analysis_mode"] = "vlm"
    return result


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )


def _existing_by_sha(path: Path) -> dict[str, dict[str, Any]]:
    """Index reusable records by source digest.

    Reuse is keyed on the image bytes *and* on the extractor that measured them:
    the deterministic numbers behind the palette changed after this library was
    first built, and a record carrying an older ``analysis.extractor`` would
    otherwise be handed back as if today's code had produced it.
    """
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            recipe = record.get("color_recipe") if isinstance(record, dict) else None
            if str((recipe or {}).get("analysis", {}).get("extractor") or "") != EXTRACTOR_ID:
                continue
            validation = validate_reference_color_recipe(recipe or {}, require_semantics=True)
            if validation["ok"]:
                result[str(recipe.get("source_sha256"))] = record
        except Exception:
            continue
    return result


def _iter_reference_files(root: Path, folder_name: str) -> list[Path]:
    folder = root / folder_name
    if not folder.exists():
        return []
    return sorted((item for item in folder.rglob("*") if item.is_file()), key=lambda item: str(item).casefold())


def analyze_reference_figures(
    references: str | Path = "reference-figures",
    out: str | Path = "output/reference_learning_stage1",
    mode: str = "heuristic",
    model: str | None = None,
    vlm_timeout: int | None = None,
    vlm_retries: int | None = None,
    vlm_adapter: Callable[[str, list[str | Path], str | None, int | None, int | None], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode not in {"heuristic", "vlm"}:
        raise ValueError("Reference figure analysis mode must be heuristic or vlm")
    vlm_timeout = resolve_vlm_timeout(vlm_timeout)
    vlm_retries = resolve_vlm_retries(vlm_retries)
    if mode == "vlm" and not vlm_adapter and not vlm_credentials_available():
        raise RuntimeError("VLM reference analysis requires API_BASE and API_KEY/GEMINI_API_KEY environment variables")

    reference_root = Path(references).resolve()
    output_root = ensure_dir(out).resolve()
    existing_records = _existing_by_sha(output_root / "reference_library.jsonl") if mode == "vlm" else {}
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    skipped: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    scanned_file_count = 0
    reused_count = 0
    resolved_model = model or DEFAULT_REFERENCE_ANALYSIS_MODEL

    groups = [
        ("excellent", "positive", "excellent"),
        ("negative", "negative", "negative"),
    ]
    for folder_name, quality_label, id_prefix in groups:
        counter = 0
        for path in _iter_reference_files(reference_root, folder_name):
            if path.name == ".gitkeep":
                continue
            scanned_file_count += 1
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_IMAGE_SUFFIXES:
                message = f"skipped non-image file: {path}"
                warnings.append(message)
                skipped.append({"path": str(path.resolve()), "reason": "unsupported_suffix"})
                continue
            record_id = f"{id_prefix}_{counter + 1:04d}"
            try:
                palette_card_path = output_root / "palette_cards" / f"{record_id}.png"
                record = _record_for_image(path, record_id, quality_label, "heuristic", palette_card_path)
                if mode == "vlm":
                    digest = str((record.get("color_recipe") or {}).get("source_sha256") or "")
                    cached = existing_records.get(digest)
                    if cached:
                        cached = json.loads(json.dumps(cached))
                        cached.update({"id": record_id, "source_path": str(path.resolve()), "quality_label": quality_label, "image": record["image"], "archetype": record["archetype"]})
                        cached["color_recipe"]["recipe_id"] = f"{record_id}.color.v1"
                        cached["color_recipe"]["source_path"] = str(path.resolve())
                        cached["color_recipe"].setdefault("analysis", {})["palette_card"] = str(palette_card_path.resolve())
                        render_palette_card(cached["color_recipe"], palette_card_path)
                        records.append(cached)
                        reused_count += 1
                        counter += 1
                        continue
                    prompt = _reference_analysis_prompt(record)
                    try:
                        if vlm_adapter:
                            raw = vlm_adapter(prompt, [path], resolved_model, vlm_timeout, vlm_retries)
                        else:
                            raw = call_vlm_json(prompt, [path], model=resolved_model, timeout=vlm_timeout, retries=vlm_retries)
                        record = _merge_vlm_analysis(record, raw)
                        render_palette_card(record["color_recipe"], palette_card_path)
                    except Exception as exc:
                        error = str(exc)
                        record["analysis_mode"] = "heuristic_after_vlm_failure"
                        record["warnings"].append(f"VLM analysis failed: {error}")
                        record["color_recipe"].setdefault("analysis", {})["semantic_mode"] = "failed"
                        failures.append({"path": str(path.resolve()), "record_id": record_id, "error": error})
                records.append(record)
                counter += 1
            except Exception as exc:
                message = f"skipped unreadable image: {path}: {exc}"
                warnings.append(message)
                skipped.append({"path": str(path.resolve()), "reason": "unreadable_image", "error": str(exc)})

    records_path = output_root / "reference_analysis_records.jsonl"
    library_path = output_root / "reference_library.jsonl"
    _write_jsonl(records_path, records)
    _write_jsonl(library_path, records)

    report = {
        "summary": "Reference figure analysis completed.",
        "ok": not failures,
        "references": str(reference_root),
        "out_dir": str(output_root),
        "analysis_mode": mode,
        "model": resolved_model if mode == "vlm" else None,
        "api_base_present": bool(os.getenv("API_BASE", "").strip()) if mode == "vlm" else None,
        "api_key_present": bool((os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()) if mode == "vlm" else None,
        "input_count": len(records),
        "scanned_file_count": scanned_file_count,
        "skipped_count": len(skipped),
        "failure_count": len(failures),
        "reused_count": reused_count,
        "records_path": str(records_path),
        "library_path": str(library_path),
        "report_path": str(output_root / "reference_analysis_report.json"),
        "supported_image_suffixes": sorted(SUPPORTED_IMAGE_SUFFIXES),
        "quality_counts": {
            "positive": sum(1 for record in records if record["quality_label"] == "positive"),
            "negative": sum(1 for record in records if record["quality_label"] == "negative"),
        },
        "skipped": skipped,
        "failures": failures,
        "warnings": warnings,
    }
    write_json(output_root / "reference_analysis_report.json", report)
    return report
