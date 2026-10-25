"""OCR backends and isolated image OCR execution for paper analysis."""

from __future__ import annotations

import math
import os
import json
import subprocess
import sys
import tempfile
import threading
import shutil
from pathlib import Path
from typing import Any, Callable

from PIL import Image


OCR_LOW_CONFIDENCE = 0.65
_EASYOCR_READERS: dict[tuple[str, ...], Any] = {}
_RAPIDOCR_LOCAL = threading.local()


def run_ocr_isolated(image_path: str | Path, engine: str, lang: str) -> list[dict[str, Any]]:
    """Run a local OCR backend outside the product process on Windows.

    OpenCV/ONNXRuntime and EasyOCR/PyTorch can ship different Intel OpenMP
    runtimes.  Loading them into the same Windows process may abort with OMP
    Error #15 before Python can raise an exception.  A short-lived worker keeps
    that native state out of the workflow and turns crashes into normal errors.
    """
    requested = str(engine or "rapidocr").lower()
    backends = {
        "rapidocr": run_rapidocr,
        "easyocr": run_easyocr,
        "paddle": run_paddle_ocr,
    }
    backend = backends.get(requested)
    if backend is None:
        raise ValueError(f"Unsupported OCR engine: {requested}")
    # Preserve dependency injection and compatibility-layer patching in tests
    # and downstream integrations; only real native backends need isolation.
    backend_is_overridden = getattr(backend, "__module__", __name__) != __name__
    if os.name != "nt" or backend_is_overridden:
        return backend(image_path, lang)
    with tempfile.TemporaryDirectory(prefix="rfs_ocr_") as temp_dir:
        output_path = Path(temp_dir) / "result.json"
        # Windows native OCR/OpenCV stacks can mangle non-ASCII paths when the
        # image path crosses a subprocess boundary. Keep the source artifact
        # untouched, but give every worker a binary-identical ASCII-path copy.
        source_path = Path(image_path).resolve()
        ascii_image_path = Path(temp_dir) / f"input{source_path.suffix.lower() or '.png'}"
        shutil.copyfile(source_path, ascii_image_path)
        worker_env = os.environ.copy()
        worker_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        worker_env.setdefault("OMP_NUM_THREADS", "1")
        worker_env.setdefault("MKL_NUM_THREADS", "1")
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "rfs.analysis.paper.ocr_subprocess",
                "--image",
                str(ascii_image_path),
                "--engine",
                requested,
                "--lang",
                str(lang),
                "--out",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            env=worker_env,
            timeout=max(30, int(os.getenv("RFS_OCR_IMAGE_TIMEOUT") or 180)),
            check=False,
        )
        payload = None
        if output_path.exists():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception:
                payload = None
        if completed.returncode != 0 or not output_path.exists():
            detail = (
                str(payload.get("error"))
                if isinstance(payload, dict) and payload.get("error")
                else completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
            )
            raise RuntimeError(f"{requested} OCR worker failed: {detail}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"{requested} OCR worker returned invalid JSON")
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return list(payload.get("records") or [])


def _positive_ocr_setting(explicit: int | None, env_name: str, default: int) -> int:
    try:
        return max(1, int(explicit if explicit is not None else os.getenv(env_name) or default))
    except (TypeError, ValueError):
        return max(1, int(default))


def _rapidocr_detector_limit(explicit: int | None = None) -> int:
    """Return a bounded detector resize limit suitable for RapidOCR."""
    value = _positive_ocr_setting(explicit, "RFS_RAPIDOCR_DET_LIMIT", 512)
    return max(256, min(2048, value))


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp_bbox(bbox: dict[str, float]) -> dict[str, float]:
    x = max(0.0, min(0.995, float(bbox["x"])))
    y = max(0.0, min(0.995, float(bbox["y"])))
    w = max(0.001, min(float(bbox["w"]), 1.0 - x))
    h = max(0.001, min(float(bbox["h"]), 1.0 - y))
    return {"x": _round4(x), "y": _round4(y), "w": _round4(w), "h": _round4(h)}


def _bbox_center(bbox: dict[str, float]) -> dict[str, float]:
    return {"x": _round4(float(bbox["x"]) + float(bbox["w"]) / 2), "y": _round4(float(bbox["y"]) + float(bbox["h"]) / 2)}


def _luminance(rgb: tuple[int, int, int]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _contrast(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return abs(_luminance(a) - _luminance(b))


def _sample_text_color(image: Image.Image, bbox: dict[str, float], fallback: str = "#263747") -> tuple[str, str]:
    try:
        rgb_image = image.convert("RGB")
        iw, ih = rgb_image.size
        pad_x = max(1, int(float(bbox["w"]) * iw * 0.04))
        pad_y = max(1, int(float(bbox["h"]) * ih * 0.10))
        x0 = max(0, min(iw - 1, int(float(bbox["x"]) * iw) - pad_x))
        y0 = max(0, min(ih - 1, int(float(bbox["y"]) * ih) - pad_y))
        x1 = max(x0 + 1, min(iw, int((float(bbox["x"]) + float(bbox["w"])) * iw) + pad_x))
        y1 = max(y0 + 1, min(ih, int((float(bbox["y"]) + float(bbox["h"])) * ih) + pad_y))
        crop = rgb_image.crop((x0, y0, x1, y1))
        pixels = list(crop.get_flattened_data() if hasattr(crop, "get_flattened_data") else crop.getdata())
    except Exception:
        return fallback, "sample_failed"
    if not pixels:
        return fallback, "empty_crop"
    pixels = sorted(pixels, key=_luminance)
    dark = pixels[: max(1, len(pixels) // 5)]
    light = pixels[-max(1, len(pixels) // 5):]
    dark_avg = tuple(int(sum(px[i] for px in dark) / len(dark)) for i in range(3))
    light_avg = tuple(int(sum(px[i] for px in light) / len(light)) for i in range(3))
    if _contrast(dark_avg, light_avg) < 32:
        return fallback, "low_contrast_fallback"
    text_rgb = dark_avg if _luminance(dark_avg) < _luminance(light_avg) else light_avg
    return _hex(text_rgb), "sampled_foreground"


def _bbox_overlap(a: dict[str, float], b: dict[str, float]) -> float:
    ax0, ay0 = float(a["x"]), float(a["y"])
    ax1, ay1 = ax0 + float(a["w"]), ay0 + float(a["h"])
    bx0, by0 = float(b["x"]), float(b["y"])
    bx1, by1 = bx0 + float(b["w"]), by0 + float(b["h"])
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    area = ix * iy
    return area / max(float(a["w"]) * float(a["h"]), 0.000001)


def _nearest_target(bbox: dict[str, float], program: dict) -> tuple[str, str]:
    candidates: list[tuple[float, int, str, str]] = []
    for collection, priority in (("cards", 3), ("slots", 2), ("panels", 1)):
        for obj in program.get(collection, []):
            if not isinstance(obj, dict) or not isinstance(obj.get("bbox_percent"), dict):
                continue
            overlap = _bbox_overlap(bbox, obj["bbox_percent"])
            if overlap <= 0:
                continue
            obox = obj["bbox_percent"]
            relative_y = (float(bbox["y"]) - float(obox["y"])) / max(float(obox["h"]), 0.001)
            if collection == "cards":
                role = "card_title" if relative_y <= 0.28 else "card_body"
            elif collection == "slots":
                role = "slot_caption"
            else:
                role = "panel_title" if relative_y <= 0.12 else "panel_body"
            area = max(float(obox["w"]) * float(obox["h"]), 0.000001)
            specificity = 1.0 / area
            candidates.append((overlap + min(0.25, specificity * 0.0005), priority, role, str(obj.get("id") or "")))
    if candidates:
        _overlap, _priority, role, target_id = max(candidates, key=lambda item: (item[0], item[1]))
        return role, target_id
    return "free_text", "canvas"


def _font_family_guess(text: str) -> str:
    if any(ord(char) > 127 for char in text):
        return "Microsoft YaHei"
    return "Comic Sans MS"


def _normalize_quad(quad: Any) -> list[list[float]] | None:
    if hasattr(quad, "tolist"):
        quad = quad.tolist()
    if not isinstance(quad, (list, tuple)) or len(quad) < 4:
        return None
    points: list[list[float]] = []
    for point in quad[:4]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        points.append([float(point[0]), float(point[1])])
    return points


def _flatten_paddle_result(result: Any) -> list[Any]:
    if not isinstance(result, list):
        return []
    if result and all(isinstance(item, list) and len(item) >= 2 and _normalize_quad(item[0]) for item in result):
        return result
    flattened: list[Any] = []
    for page in result:
        if isinstance(page, list):
            flattened.extend(_flatten_paddle_result(page))
    return flattened


def _mapping_value(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _parse_paddle_mapping(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    payload = mapping.get("res") if isinstance(mapping.get("res"), dict) else mapping
    texts = _mapping_value(payload, ("rec_texts", "texts", "text"))
    quads = _mapping_value(payload, ("rec_polys", "dt_polys", "det_polys", "polys", "boxes"))
    scores = _mapping_value(payload, ("rec_scores", "scores", "confidences"))
    if hasattr(texts, "tolist"):
        texts = texts.tolist()
    if hasattr(quads, "tolist"):
        quads = quads.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not isinstance(quads, list):
        return []
    records = []
    score_values = scores if isinstance(scores, list) else []
    for index, text in enumerate(texts):
        if index >= len(quads):
            break
        quad = _normalize_quad(quads[index])
        raw_text = str(text or "").strip()
        if not quad or not raw_text:
            continue
        confidence = 0.0
        if index < len(score_values):
            try:
                confidence = float(score_values[index])
            except Exception:
                confidence = 0.0
        records.append({"text": raw_text, "confidence": confidence, "quad": quad})
    return records


def _parse_ocr_result(result: Any) -> list[dict[str, Any]]:
    records = []
    if isinstance(result, dict):
        return _parse_paddle_mapping(result)
    if isinstance(result, list):
        for page in result:
            value = page
            if hasattr(value, "json"):
                value = value.json() if callable(value.json) else value.json
            if isinstance(value, dict):
                records.extend(_parse_paddle_mapping(value))
        if records:
            return records
    for item in _flatten_paddle_result(result):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        quad = _normalize_quad(item[0])
        payload = item[1]
        if not quad:
            continue
        text = ""
        confidence = 0.0
        if isinstance(payload, (list, tuple)) and payload:
            text = str(payload[0] or "").strip()
            if len(payload) > 1:
                try:
                    confidence = float(payload[1])
                except Exception:
                    confidence = 0.0
        else:
            text = str(payload or "").strip()
        if text:
            records.append({"text": text, "confidence": confidence, "quad": quad})
    return records


def run_paddle_ocr(image_path: str | Path, lang: str) -> list[dict[str, Any]]:
    from paddleocr import PaddleOCR  # type: ignore

    lang_map = {"en_ch": "ch", "ch": "ch", "en": "en"}
    mapped_lang = lang_map.get(str(lang), "ch")
    init_options = [
        {"use_angle_cls": True, "lang": mapped_lang, "show_log": False},
        {"use_angle_cls": True, "lang": mapped_lang},
        {"use_textline_orientation": True, "lang": mapped_lang},
        {"lang": mapped_lang},
    ]
    last_error: Exception | None = None
    for options in init_options:
        try:
            ocr = PaddleOCR(**options)
            break
        except Exception as exc:
            last_error = exc
    else:
        assert last_error is not None
        raise last_error
    try:
        result = ocr.ocr(str(image_path), cls=True)
    except Exception:
        result = ocr.ocr(str(image_path))
    return _parse_ocr_result(result)


def run_easyocr(image_path: str | Path, lang: str) -> list[dict[str, Any]]:
    import easyocr  # type: ignore

    lang_map = {"en_ch": ["ch_sim", "en"], "ch": ["ch_sim"], "en": ["en"]}
    languages = tuple(lang_map.get(str(lang), ["ch_sim", "en"]))
    reader = _EASYOCR_READERS.get(languages)
    if reader is None:
        allow_download = str(os.getenv("RFS_OCR_ALLOW_DOWNLOAD") or "").strip().casefold() in {"1", "true", "yes", "on"}
        try:
            reader = easyocr.Reader(list(languages), gpu=False, download_enabled=allow_download, verbose=False)
        except Exception as exc:
            hint = " Set RFS_OCR_ALLOW_DOWNLOAD=1 for an explicit one-time model download." if not allow_download else ""
            raise RuntimeError(f"EasyOCR model is not ready for languages {list(languages)}.{hint} Original error: {exc}") from exc
        _EASYOCR_READERS[languages] = reader
    records = []
    for quad, text, confidence in reader.readtext(str(image_path)):
        points = _normalize_quad(quad)
        if points and str(text).strip():
            records.append({"text": str(text).strip(), "confidence": float(confidence), "quad": points})
    return records


def run_rapidocr_detailed(
    image_path: str | Path,
    lang: str,
    *,
    threads: int | None = None,
    batch_size: int | None = None,
    detector_limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del lang
    thread_count = _positive_ocr_setting(threads, "RFS_RAPIDOCR_THREADS", 1)
    recognition_batch = _positive_ocr_setting(batch_size, "RFS_RAPIDOCR_BATCH", 6)
    detector_side_limit = _rapidocr_detector_limit(detector_limit)
    engines = getattr(_RAPIDOCR_LOCAL, "engines", None)
    if engines is None:
        engines = {}
        _RAPIDOCR_LOCAL.engines = engines
    engine_key = (thread_count, recognition_batch, detector_side_limit)
    if engine_key not in engines:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
        options = [
            {"intra_op_num_threads": thread_count, "inter_op_num_threads": 1, "det_limit_side_len": detector_side_limit, "rec_batch_num": recognition_batch},
            {"intra_op_num_threads": thread_count, "det_limit_side_len": detector_side_limit},
            {"det_limit_side_len": detector_side_limit},
            {},
        ]
        last_error: Exception | None = None
        for option in options:
            try:
                engines[engine_key] = RapidOCR(**option)
                break
            except Exception as exc:
                last_error = exc
        else:
            assert last_error is not None
            raise last_error
    result, timings = engines[engine_key](str(image_path))
    records = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        quad = _normalize_quad(item[0])
        text = str(item[1] or "").strip()
        if quad and text:
            records.append({"text": text, "confidence": float(item[2] or 0.0), "quad": quad})
    timing_values = [float(value or 0.0) for value in (timings or [])]
    while len(timing_values) < 3:
        timing_values.append(0.0)
    diagnostics = {
        "detector_limit": detector_side_limit,
        "thread_count": thread_count,
        "recognition_batch": recognition_batch,
        "detection_seconds": round(timing_values[0], 4),
        "classification_seconds": round(timing_values[1], 4),
        "recognition_seconds": round(timing_values[2], 4),
        "inference_seconds": round(sum(timing_values[:3]), 4),
        "region_count": len(records),
    }
    return records, diagnostics


def run_rapidocr(image_path: str | Path, lang: str, *, threads: int | None = None, batch_size: int | None = None, detector_limit: int | None = None) -> list[dict[str, Any]]:
    records, _diagnostics = run_rapidocr_detailed(
        image_path,
        lang,
        threads=threads,
        batch_size=batch_size,
        detector_limit=detector_limit,
    )
    return records


def _record_to_region(record: dict[str, Any], image: Image.Image, program: dict, index: int, canvas_height_in: float, engine: str) -> dict:
    width, height = image.size
    xs = [point[0] for point in record["quad"]]
    ys = [point[1] for point in record["quad"]]
    bbox = _clamp_bbox({
        "x": min(xs) / max(width, 1),
        "y": min(ys) / max(height, 1),
        "w": (max(xs) - min(xs)) / max(width, 1),
        "h": (max(ys) - min(ys)) / max(height, 1),
    })
    text = str(record.get("text") or "").strip()
    confidence = float(record.get("confidence") or 0.0)
    color_hex, color_status = _sample_text_color(image, bbox)
    role, target_id = _nearest_target(bbox, program)
    font_size = max(1.0, float(bbox["h"]) * canvas_height_in * 72 * 0.72)
    return {
        "id": f"ref_text_ocr_{index:03d}",
        "text": text,
        "raw_text": text,
        "role": role,
        "target_id": target_id,
        "bbox_percent": bbox,
        "line_bbox_percent": bbox,
        "word_bbox_percent": [bbox],
        "center_percent": _bbox_center(bbox),
        "width_percent": _round4(bbox["w"]),
        "height_percent": _round4(bbox["h"]),
        "estimated_font_ratio": _round4(float(bbox["h"]) * 0.72),
        "font_size_pt": round(font_size, 2),
        "font_family_guess": _font_family_guess(text),
        "font_weight_guess": "bold" if role == "panel_title" and float(bbox["h"]) > 0.025 else "regular",
        "color_hex": color_hex,
        "color_sampling_status": color_status,
        "confidence": round(confidence, 4),
        "ocr_engine": engine,
        "source": "reference_ocr_text_region",
        "editable_in": "pptx",
    }


def extract_reference_text(
    reference_path: str | Path,
    program: dict,
    *,
    mode: str = "ocr",
    engine: str = "easyocr",
    lang: str = "en_ch",
    ocr_adapter: Callable[[str | Path, str], list[dict[str, Any]]] | None = None,
) -> tuple[list[dict], dict]:
    requested_mode = str(mode or "ocr").lower()
    requested_engine = str(engine or "easyocr").lower()
    report = {
        "summary": "OCR text extraction quality report.",
        "mode": requested_mode,
        "ocr_engine": requested_engine,
        "ocr_engine_effective": requested_engine,
        "ocr_lang": lang,
        "status": "not_run",
        "text_region_count": 0,
        "low_confidence_count": 0,
        "warnings": [],
        "fallback_reason": None,
    }
    if requested_mode != "ocr" or requested_engine == "off":
        report.update({"status": "fallback", "fallback_reason": "ocr_disabled"})
        return [], report
    effective_engine = requested_engine
    try:
        if ocr_adapter:
            raw_records = ocr_adapter(reference_path, lang)
        elif requested_engine == "rapidocr":
            raw_records = run_ocr_isolated(reference_path, "rapidocr", lang)
        elif requested_engine == "easyocr":
            raw_records = run_ocr_isolated(reference_path, "easyocr", lang)
        else:
            try:
                raw_records = run_ocr_isolated(reference_path, "paddle", lang)
            except Exception as paddle_exc:
                try:
                    raw_records = run_ocr_isolated(reference_path, "easyocr", lang)
                    effective_engine = "easyocr"
                    report["ocr_engine_effective"] = effective_engine
                    report["warnings"].append(f"paddle_unavailable_used_easyocr:{paddle_exc}")
                    report["fallback_reason"] = f"paddle_unavailable_used_easyocr:{paddle_exc}"
                except Exception as easy_exc:
                    raise RuntimeError(f"paddle_failed:{paddle_exc}; easyocr_failed:{easy_exc}") from easy_exc
    except Exception as exc:
        report.update({"status": "fallback", "fallback_reason": f"ocr_unavailable:{exc}"})
        report["warnings"].append(str(exc))
        return [], report
    try:
        with Image.open(reference_path) as image:
            rgb_image = image.convert("RGB")
            canvas = program.get("canvas", {}) if isinstance(program.get("canvas"), dict) else {}
            canvas_height_in = float(canvas.get("height_in") or 7.5)
            regions = [_record_to_region(record, rgb_image, program, index, canvas_height_in, effective_engine) for index, record in enumerate(raw_records, start=1)]
    except Exception as exc:
        report.update({"status": "fallback", "fallback_reason": f"ocr_postprocess_failed:{exc}"})
        report["warnings"].append(str(exc))
        return [], report
    regions = [region for region in regions if str(region.get("text", "")).strip()]
    low_conf = [region for region in regions if float(region.get("confidence") or 0.0) < OCR_LOW_CONFIDENCE]
    overlap_warnings = []
    for region in regions:
        if region.get("role") == "slot_caption":
            overlap_warnings.append(region["id"])
    if not regions:
        report.update({"status": "fallback", "fallback_reason": "ocr_returned_no_text", "text_region_count": 0})
        return [], report
    report.update({
        "status": "pass",
        "ocr_engine_effective": effective_engine,
        "text_region_count": len(regions),
        "low_confidence_count": len(low_conf),
        "low_confidence_text_ids": [region["id"] for region in low_conf],
        "slot_overlap_text_ids": overlap_warnings,
        "warnings": report["warnings"] + ([f"{len(low_conf)} low-confidence OCR text region(s)"] if low_conf else []),
    })
    return regions, report
