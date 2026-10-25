"""Deterministic local corruptions for auditable Judge negative examples."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageStat


# Ordered gate classes first: module_order_error and overlap are gated defect classes,
# and missing_relation is the local arrow-class source that feeds the same gate.
LOCAL_CORRUPTION_TYPES = (
    "module_order_error",
    "overlap",
    "terminology_corruption",
    "missing_core_entity",
    "missing_core_stage",
    "missing_relation",
    "correct_but_ugly",
)
ELIGIBLE_LABELS = {"好（科研）", "好（生动）"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokens(value: str) -> set[str]:
    return {token for token in re.sub(r"[^a-z0-9]+", " ", value.casefold()).split() if len(token) >= 2}


def _bbox_from_quad(quad: list[list[float]], width: int, height: int) -> dict[str, float]:
    xs = [float(point[0]) for point in quad]
    ys = [float(point[1]) for point in quad]
    return {"x": min(xs) / width, "y": min(ys) / height, "w": (max(xs) - min(xs)) / width, "h": (max(ys) - min(ys)) / height}


def _union(boxes: list[dict[str, float]]) -> dict[str, float]:
    left = min(box["x"] for box in boxes)
    top = min(box["y"] for box in boxes)
    right = max(box["x"] + box["w"] for box in boxes)
    bottom = max(box["y"] + box["h"] for box in boxes)
    return {"x": left, "y": top, "w": right - left, "h": bottom - top}


def _center(box: dict[str, float]) -> tuple[float, float]:
    return box["x"] + box["w"] / 2, box["y"] + box["h"] / 2


def _axis_overlap(left: dict[str, float], right: dict[str, float], axis: str) -> float:
    """Return how much two boxes share along the axis they do not travel on."""

    if axis == "horizontal":
        top, bottom = max(left["y"], right["y"]), min(left["y"] + left["h"], right["y"] + right["h"])
        return max(0.0, bottom - top) / max(1e-6, min(left["h"], right["h"]))
    left_edge, right_edge = max(left["x"], right["x"]), min(left["x"] + left["w"], right["x"] + right["w"])
    return max(0.0, right_edge - left_edge) / max(1e-6, min(left["w"], right["w"]))


def _reading_order(entries: list[tuple[str, dict[str, float]]]) -> list[tuple[str, dict[str, float]]]:
    return sorted(entries, key=lambda item: (round(item[1]["y"], 2), round(item[1]["x"], 2), item[0]))


def _adjacent_pairs(ordered: list[tuple[str, dict[str, float]]], *, min_overlap: float = 0.5) -> list[tuple[str, dict[str, float], str, dict[str, float], str]]:
    """Pair neighbouring modules that sit side by side or on top of each other."""

    pairs = []
    for (first_id, first), (second_id, second) in zip(ordered, ordered[1:]):
        first_x, first_y = _center(first)
        second_x, second_y = _center(second)
        axis = "horizontal" if abs(second_x - first_x) >= abs(second_y - first_y) else "vertical"
        if _axis_overlap(first, second, axis) >= min_overlap:
            pairs.append((first_id, first, second_id, second, axis))
    return pairs


def _candidate(candidate_id: str, figure: dict[str, Any], kind: str, target_id: str, bbox: dict[str, float], confidence: float, source: str, **extra: Any) -> dict[str, Any]:
    return {"candidate_id": candidate_id, "paper_id": str(figure["paper_id"]), "figure_id": str(figure["figure_id"]), "split": str(figure["split"]), "final_label": str(figure["final_label"]), "source_image": str(Path(figure["figure_path"]).resolve()), "corruption_type": kind, "target_id": target_id, "bbox_percent": {key: round(float(value), 6) for key, value in bbox.items()}, "region_source": source, "localization_confidence": round(float(confidence), 4), "review_status": "pending", **extra}


def _figure_ocr_worker(payload: tuple[str, Callable[..., list[dict[str, Any]]]]) -> dict[str, Any]:
    """Read one figure's pixel size and OCR records inside a worker process.

    An OCR engine keeps its model in the process that loaded it, so a pool of these workers
    loads the model once each and then reuses it. The worker never raises: a failure comes back
    as the same text the serial path would have recorded, so a failed figure reads the same
    whether or not workers ran.
    """
    source_text, ocr_function = payload
    source = Path(source_text)
    try:
        with Image.open(source) as image:
            width, height = image.size
        try:
            records = ocr_function(source, "en_ch")
        except Exception:
            from rfs.analysis.paper.ocr import run_rapidocr
            records = run_rapidocr(source, "en_ch")
    except Exception as exc:  # noqa: BLE001 - the failure text is the report
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"image_size": [int(width), int(height)], "ocr": list(records)}


def _ocr_reader_can_cross_processes(ocr_function: Callable[..., Any] | None) -> bool:
    """Say whether one OCR callable may be sent to a worker process.

    Only a module-level function can be imported back by another process. A closure or a bound
    method cannot, so those readers keep running here and the candidates stay the same.
    """
    if ocr_function is None or getattr(ocr_function, "__self__", None) is not None:
        return False
    qualname = str(getattr(ocr_function, "__qualname__", ""))
    return bool(getattr(ocr_function, "__module__", None)) and "<locals>" not in qualname


def _ocr_results_in_order(
    paths: list[Path],
    ocr_function: Callable[[str | Path, str], list[dict[str, Any]]],
    workers: int,
) -> Iterator[dict[str, Any]]:
    """Yield one OCR outcome per figure in the caller's order, reading ahead in a pool.

    Reading ahead is what makes a corpus-scale run practical: one figure costs seconds of one
    core, and the loop only stops once its rarest class has filled up, which happens near the
    end of the corpus. Neither the order nor the outcomes depend on how many workers ran.
    """
    if workers <= 1 or not _ocr_reader_can_cross_processes(ocr_function):
        for path in paths:
            yield _figure_ocr_worker((str(path), ocr_function))
        return
    pending: deque[Any] = deque()
    remaining = iter(paths)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in range(max(1, workers)):
            path = next(remaining, None)
            if path is None:
                break
            pending.append(pool.submit(_figure_ocr_worker, (str(path), ocr_function)))
        while pending:
            yield pending.popleft().result()
            path = next(remaining, None)
            if path is not None:
                pending.append(pool.submit(_figure_ocr_worker, (str(path), ocr_function)))


def generate_region_candidates(
    figures_path: str | Path,
    structured_dir: str | Path,
    out_path: str | Path,
    *,
    per_type: int = 100,
    ocr_function: Callable[[str | Path, str], list[dict[str, Any]]] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Generate conservative OCR-grounded local-region candidates from good figures.

    `workers` only decides how many OCR reads run ahead in parallel; it never changes
    which candidates come out.
    """
    if ocr_function is None:
        from rfs.analysis.paper.ocr import run_easyocr
        ocr_function = run_easyocr
    figures = sorted(
        (row for row in _jsonl(Path(figures_path)) if str(row.get("final_label")) in ELIGIBLE_LABELS),
        key=lambda row: hashlib.sha256(str(row["figure_id"]).encode()).hexdigest(),
    )
    contracts = Path(structured_dir)
    rows: list[dict[str, Any]] = []
    counts = {kind: 0 for kind in LOCAL_CORRUPTION_TYPES}
    failures: list[dict[str, str]] = []
    outcomes = _ocr_results_in_order([Path(str(row["figure_path"])) for row in figures], ocr_function, workers)
    for figure, outcome in zip(figures, outcomes):
        if all(count >= per_type for count in counts.values()):
            break
        if counts["correct_but_ugly"] < per_type:
            rows.append(_candidate(f"{figure['figure_id']}__style", figure, "correct_but_ugly", "whole_figure", {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}, 1.0, "deterministic_global"))
            counts["correct_but_ugly"] += 1
        if "error" in outcome:
            failures.append({"figure_id": str(figure["figure_id"]), "error": str(outcome["error"])})
            continue
        try:
            contract = json.loads((contracts / f"{figure['paper_id']}.json").read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append({"figure_id": str(figure["figure_id"]), "error": f"{type(exc).__name__}: {exc}"})
            continue
        width, height = outcome["image_size"]
        ocr = outcome["ocr"]
        regions = [{**record, "bbox": _bbox_from_quad(record["quad"], width, height)} for record in ocr if record.get("quad") and float(record.get("confidence") or 0) >= 0.55]
        entities = contract.get("entities") if isinstance(contract.get("entities"), list) else []
        matched: dict[str, dict[str, Any]] = {}
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            entity_id = str(entity.get("entity_id") or "")
            target_tokens = _tokens(entity_id + " " + str(entity.get("name") or ""))
            ranked = []
            for region in regions:
                overlap = target_tokens & _tokens(str(region.get("text") or ""))
                score = len(overlap) / max(1, len(target_tokens))
                if overlap:
                    ranked.append((score * float(region.get("confidence") or 0), region))
            if ranked:
                score, region = max(ranked, key=lambda item: item[0])
                matched[entity_id] = {"score": score, "region": region}
        required = ((contract.get("expected_figure_content") or {}).get("required_entities") or [])
        pairs = _adjacent_pairs(_reading_order([(str(entity_id), matched[str(entity_id)]["region"]["bbox"]) for entity_id in required if str(entity_id) in matched]))
        if pairs and counts["module_order_error"] < per_type:
            first_id, first_box, second_id, second_box, axis = pairs[0]
            confidence = min(matched[first_id]["score"], matched[second_id]["score"])
            rows.append(_candidate(f"{figure['figure_id']}__order__{first_id}__{second_id}", figure, "module_order_error", f"{first_id}<->{second_id}", first_box, confidence, "ocr_adjacent_pair", secondary_bbox_percent=second_box, secondary_target_id=second_id, swap_axis=axis))
            counts["module_order_error"] += 1
        if pairs and counts["overlap"] < per_type:
            first_id, first_box, second_id, second_box, axis = pairs[-1]
            confidence = min(matched[first_id]["score"], matched[second_id]["score"])
            rows.append(_candidate(f"{figure['figure_id']}__overlap__{first_id}__{second_id}", figure, "overlap", first_id, first_box, confidence, "ocr_adjacent_pair", secondary_bbox_percent=second_box, secondary_target_id=second_id, swap_axis=axis))
            counts["overlap"] += 1
        for entity_id in required:
            match = matched.get(str(entity_id))
            if not match:
                continue
            region = match["region"]
            if counts["terminology_corruption"] < per_type:
                rows.append(_candidate(f"{figure['figure_id']}__term__{entity_id}", figure, "terminology_corruption", str(entity_id), region["bbox"], match["score"], "ocr", original_text=region.get("text"), replacement_text="Unsupported Module"))
                counts["terminology_corruption"] += 1
            if counts["missing_core_entity"] < per_type:
                rows.append(_candidate(f"{figure['figure_id']}__entity__{entity_id}", figure, "missing_core_entity", str(entity_id), region["bbox"], match["score"], "ocr"))
                counts["missing_core_entity"] += 1
        flows = list(contract.get("training_flow") or []) + list(contract.get("inference_flow") or [])
        for index, step in enumerate(flows):
            description = str((step or {}).get("step_description") or "") if isinstance(step, dict) else str(step)
            boxes = [value["region"]["bbox"] for entity_id, value in matched.items() if _tokens(entity_id) & _tokens(description)]
            union = _union(boxes) if boxes else None
            confidence = min((matched[e]["score"] for e in matched if _tokens(e) & _tokens(description)), default=0.0)
            if union and union["w"] * union["h"] <= 0.22 and confidence >= 0.4 and counts["missing_core_stage"] < per_type:
                rows.append(_candidate(f"{figure['figure_id']}__stage__{index}", figure, "missing_core_stage", f"flow_{index}", union, confidence, "ocr_union", step_description=description))
                counts["missing_core_stage"] += 1
                break
        for index, relation in enumerate(contract.get("relations") or []):
            if not isinstance(relation, dict):
                continue
            source_match = matched.get(str(relation.get("source")))
            target_match = matched.get(str(relation.get("target")))
            if source_match and target_match and counts["missing_relation"] < per_type:
                a, b = source_match["region"]["bbox"], target_match["region"]["bbox"]
                ax, ay = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
                bx, by = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
                center_x, center_y = (ax + bx) / 2, (ay + by) / 2
                bbox = {"x": max(0.0, center_x - 0.025), "y": max(0.0, center_y - 0.025), "w": min(0.05, 1.0 - max(0.0, center_x - 0.025)), "h": min(0.05, 1.0 - max(0.0, center_y - 0.025))}
                rows.append(_candidate(f"{figure['figure_id']}__relation__{index}", figure, "missing_relation", f"{relation.get('source')}->{relation.get('target')}", bbox, min(source_match["score"], target_match["score"]), "ocr_corridor", relation=relation))
                counts["missing_relation"] += 1
                break
    rows.sort(key=lambda row: row["candidate_id"])
    _write_jsonl(Path(out_path), rows)
    report = {"candidates": len(rows), "counts": counts, "eligible_figures": len(figures), "ocr_failures": failures, "status": "pending_human_region_review"}
    Path(out_path).with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _pixel_bbox(box: dict[str, float], size: tuple[int, int], padding: int = 0) -> tuple[int, int, int, int]:
    width, height = size
    left = max(0, int(box["x"] * width) - padding)
    top = max(0, int(box["y"] * height) - padding)
    right = min(width, int((box["x"] + box["w"]) * width) + padding)
    bottom = min(height, int((box["y"] + box["h"]) * height) + padding)
    return left, top, max(left + 1, right), max(top + 1, bottom)


def _background(image: Image.Image, bbox: tuple[int, int, int, int]) -> tuple[int, int, int]:
    left, top, right, bottom = bbox
    ring = image.crop((max(0, left - 5), max(0, top - 5), min(image.width, right + 5), min(image.height, bottom + 5))).convert("RGB")
    stat = ImageStat.Stat(ring)
    return tuple(int(value) for value in stat.median[:3])


def _font(size: int) -> ImageFont.ImageFont:
    for path in (Path(r"<PATH>"), Path(r"<PATH>")):
        if path.exists():
            return ImageFont.truetype(str(path), max(8, size))
    return ImageFont.load_default()


def _region_pair(review: dict[str, Any], size: tuple[int, int]) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    secondary = review.get("secondary_bbox_percent")
    if not isinstance(secondary, dict):
        raise ValueError(f"{review.get('corruption_type')} needs a secondary_bbox_percent region")
    return _pixel_bbox(review["bbox_percent"], size), _pixel_bbox(secondary, size)


def _refit(image: Image.Image, box: tuple[int, int, int, int], size: tuple[int, int]) -> Image.Image:
    crop = image.crop(box)
    return crop if crop.size == size else crop.resize(size, Image.LANCZOS)


def _swap_regions(image: Image.Image, first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> None:
    """Exchange the pixel content of two adjacent module boxes in place."""

    first_crop = _refit(image, first, (second[2] - second[0], second[3] - second[1]))
    second_crop = _refit(image, second, (first[2] - first[0], first[3] - first[1]))
    image.paste(first_crop, (second[0], second[1]))
    image.paste(second_crop, (first[0], first[1]))


def _move_onto_neighbour(image: Image.Image, moving: tuple[int, int, int, int], fixed: tuple[int, int, int, int], axis: str, *, overlap_fraction: float = 0.65) -> tuple[int, int, int, int]:
    """Shift one module onto its neighbour and clear the position it left behind."""

    width, height = moving[2] - moving[0], moving[3] - moving[1]
    moving_x, moving_y = (moving[0] + moving[2]) / 2, (moving[1] + moving[3]) / 2
    fixed_x, fixed_y = (fixed[0] + fixed[2]) / 2, (fixed[1] + fixed[3]) / 2
    shift_x = round((fixed_x - moving_x) * overlap_fraction) if axis == "horizontal" else 0
    shift_y = round((fixed_y - moving_y) * overlap_fraction) if axis == "vertical" else 0
    left = min(max(0, moving[0] + shift_x), max(0, image.width - width))
    top = min(max(0, moving[1] + shift_y), max(0, image.height - height))
    crop = image.crop(moving)
    ImageDraw.Draw(image).rectangle((moving[0], moving[1], moving[2] - 1, moving[3] - 1), fill=_background(image, moving))
    image.paste(crop, (left, top))
    return left, top, left + width, top + height


def _apply_edit(review: dict[str, Any], image: Image.Image) -> tuple[Image.Image, str, list[tuple[int, int, int, int]]]:
    """Apply one deterministic edit and report the boxes the edit may touch."""

    kind = str(review["corruption_type"])
    bbox = _pixel_bbox(review["bbox_percent"], image.size)
    if kind == "correct_but_ugly":
        return ImageEnhance.Contrast(image).enhance(0.48).filter(ImageFilter.GaussianBlur(radius=0.65)), "degrade_style", []
    if kind in {"module_order_error", "overlap"}:
        first, second = _region_pair(review, image.size)
        axis = "vertical" if str(review.get("swap_axis") or "horizontal").startswith("v") else "horizontal"
        if kind == "module_order_error":
            _swap_regions(image, first, second)
            return image, "swap_module_positions", [first, second]
        return image, "shift_module_into_neighbour", [first, _move_onto_neighbour(image, first, second, axis)]
    fill = _background(image, bbox)
    ImageDraw.Draw(image).rectangle((bbox[0], bbox[1], bbox[2] - 1, bbox[3] - 1), fill=fill)
    operation = "text_replace" if kind == "terminology_corruption" else "remove_relation" if kind == "missing_relation" else "mask_stage" if kind == "missing_core_stage" else "mask_entity"
    if kind == "terminology_corruption":
        text = str(review.get("replacement_text") or "Unsupported Module")
        height = max(8, bbox[3] - bbox[1])
        ImageDraw.Draw(image).text((bbox[0] + 2, bbox[1] + 1), text, fill=(120, 20, 20), font=_font(int(height * 0.58)))
    return image, operation, [bbox]


def _changed_pixels(original: Image.Image, image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, float]:
    """Count changed pixels inside the allowed boxes and outside them."""

    changed = ImageChops.difference(original, image).convert("L").point(lambda value: 255 if value > 8 else 0)
    total = sum(changed.histogram()[1:])
    if not boxes:
        return total, total, 0.0
    mask = Image.new("L", image.size, 0)
    painter = ImageDraw.Draw(mask)
    for box in boxes:
        painter.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), fill=255)
    target = sum(ImageChops.multiply(changed, mask).histogram()[1:])
    return total, target, max(0, total - target) / max(1, original.width * original.height)


def apply_local_corruption(review: dict[str, Any], output_path: str | Path) -> dict[str, Any]:
    """Apply one approved deterministic corruption and enforce a pixel-locality gate."""
    source = Path(review["source_image"])
    output = Path(output_path)
    with Image.open(source) as opened:
        original = opened.convert("RGB")
    image, operation, boxes = _apply_edit(review, original.copy())
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    changed_total, target_changed, outside_ratio = _changed_pixels(original, image, boxes)
    locality_passed = not boxes or outside_ratio <= 0.001
    passed = changed_total > 0 and locality_passed
    result = {**review, "operation": operation, "output_image": str(output.resolve()), "source_sha256": _sha256(source), "output_sha256": _sha256(output), "changed_pixel_ratio": round(changed_total / max(1, original.width * original.height), 6), "outside_target_changed_ratio": round(outside_ratio, 6), "automatic_gate": {"size_preserved": image.size == original.size, "target_changed": target_changed > 0, "locality_passed": locality_passed}, "status": "generated" if passed else "rejected"}
    return result


def generate_approved_corruptions(reviews_path: str | Path, out_dir: str | Path, results_path: str | Path, *, per_type: int = 100) -> dict[str, Any]:
    """Generate approved/adjusted candidates with deterministic per-type limits."""
    latest = {row["candidate_id"]: row for row in _jsonl(Path(reviews_path))}
    selected = [row for row in latest.values() if row.get("review_status") in {"approved", "adjusted"}]
    counts = {kind: 0 for kind in LOCAL_CORRUPTION_TYPES}
    results: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: item["candidate_id"]):
        kind = str(row["corruption_type"])
        if kind not in counts or counts[kind] >= per_type:
            continue
        target = Path(out_dir) / str(row["paper_id"]) / f"{row['candidate_id']}.png"
        try:
            result = apply_local_corruption(row, target)
        except Exception as exc:
            result = {**row, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        results.append(result)
        counts[kind] += 1
    _write_jsonl(Path(results_path), results)
    return {"selected": len(selected), "generated": sum(row.get("status") == "generated" for row in results), "rejected": sum(row.get("status") == "rejected" for row in results), "failed": sum(row.get("status") == "failed" for row in results), "counts": counts}
