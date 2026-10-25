"""Conservative VLM-assisted review of local-corruption regions for Kira1 audit."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from rfs.providers.client import call_vlm_json, resolve_vlm_model


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _font(size: int) -> ImageFont.ImageFont:
    path = Path(r"<PATH>")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def _overlay(source: Path, candidates: list[dict[str, Any]], output: Path) -> Path:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, round(min(width, height) / 250))
    for index, row in enumerate(candidates, start=1):
        box = row["bbox_percent"]
        coords = (round(box["x"] * width), round(box["y"] * height), round((box["x"] + box["w"]) * width), round((box["y"] + box["h"]) * height))
        color = (230, 30, 30) if row["corruption_type"] != "missing_relation" else (20, 90, 240)
        draw.rectangle(coords, outline=color, width=line_width)
        label = str(index)
        label_box = draw.textbbox((coords[0], coords[1]), label, font=_font(max(14, line_width * 5)))
        draw.rectangle(label_box, fill=(255, 255, 255))
        draw.text((coords[0], coords[1]), label, fill=color, font=_font(max(14, line_width * 5)))
        secondary = row.get("secondary_bbox_percent")
        if isinstance(secondary, dict):
            neighbour = (round(secondary["x"] * width), round(secondary["y"] * height), round((secondary["x"] + secondary["w"]) * width), round((secondary["y"] + secondary["h"]) * height))
            draw.rectangle(neighbour, outline=(240, 140, 20), width=line_width)
            draw.text((neighbour[0], neighbour[1]), f"{label}n", fill=(240, 140, 20), font=_font(max(14, line_width * 5)))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def _prompt(candidates: list[dict[str, Any]], contract: dict[str, Any]) -> str:
    items = []
    for index, row in enumerate(candidates, start=1):
        items.append({"index": index, "candidate_id": row["candidate_id"], "corruption_type": row["corruption_type"], "target_id": row["target_id"], "current_bbox_percent": row["bbox_percent"], "current_secondary_bbox_percent": row.get("secondary_bbox_percent"), "secondary_target_id": row.get("secondary_target_id"), "swap_axis": row.get("swap_axis"), "original_text": row.get("original_text"), "step_description": row.get("step_description"), "relation": row.get("relation")})
    evidence = {"entities": contract.get("entities", []), "relations": contract.get("relations", []), "training_flow": contract.get("training_flow", []), "inference_flow": contract.get("inference_flow", [])}
    return f"""You are assisting Kira1 with a conservative human-quality audit of scientific diagram regions.
The image has numbered candidate rectangles. Check visible pixels, not merely the requested target.
Paper evidence contract: {json.dumps(evidence, ensure_ascii=False)}
Candidates: {json.dumps(items, ensure_ascii=False)}

Rules:
- terminology_corruption: approve only a tight rectangle around the exact visible target text. Adjust to the exact text if needed.
- missing_core_entity: the rectangle must cover the complete visible module/card for the target, not only its label. Adjust only if module boundaries are unambiguous; otherwise reject.
- missing_core_stage: the rectangle must cover one complete, scientifically meaningful stage and no unrelated stage. Reject broad or ambiguous regions.
- missing_relation: the rectangle must tightly cover one clearly visible arrow/connector representing the specified relation, without covering endpoint modules. Reject if the relation cannot be visually identified.
- module_order_error: rectangle 1 (red, suffix n = orange partner) must cover the first module of the pair, the partner box must cover the second module, and both must be complete, mutually disjoint modules that a reader could mistake for swapped order. Reject if either module is partial or the pair is not visually sequential.
- overlap: the primary box must cover the module that is meant to be moved, the partner box the neighbour it would cover, and both must be complete modules sitting side by side or stacked. Reject if the two boxes already overlap or the pair is unrelated.
- Reject rather than guess. Visual confidence must be at least 0.85 for approval or adjustment.

Return JSON: {{"decisions":[{{"candidate_id":"...","review_status":"approved|adjusted|rejected","bbox_percent":{{"x":0,"y":0,"w":0,"h":0}},"confidence":0.0,"reason":"visible reason"}}]}}.
Coordinates are normalized to the full image. Return exactly one decision per candidate."""


def _valid_bbox(value: Any, kind: str) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        box = {key: float(value[key]) for key in ("x", "y", "w", "h")}
    except (KeyError, TypeError, ValueError):
        return None
    if box["x"] < 0 or box["y"] < 0 or box["w"] <= 0 or box["h"] <= 0 or box["x"] + box["w"] > 1 or box["y"] + box["h"] > 1:
        return None
    max_area = {"terminology_corruption": 0.06, "missing_core_entity": 0.25, "missing_core_stage": 0.30, "missing_relation": 0.06, "module_order_error": 0.25, "overlap": 0.25}.get(kind, 0.3)
    return box if box["w"] * box["h"] <= max_area else None


def review_local_regions(
    candidates_path: str | Path,
    structured_dir: str | Path,
    reviews_path: str | Path,
    overlays_dir: str | Path,
    *,
    workers: int = 3,
    batch_size: int = 5,
    model: str | None = None,
    timeout: int = 180,
    retries: int = 1,
) -> dict[str, Any]:
    """Review pending regions, auto-approve deterministic style cases, and resume safely."""
    candidates = _jsonl(Path(candidates_path))
    output = Path(reviews_path)
    existing = {row["candidate_id"]: row for row in _jsonl(output)}
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    for row in candidates:
        if row["candidate_id"] not in existing and row["corruption_type"] == "correct_but_ugly":
            existing[row["candidate_id"]] = {**row, "review_status": "approved", "reviewer": "Kira1", "review_method": "deterministic_whole_figure", "review_confidence": 1.0, "review_reason": "The corruption intentionally applies a deterministic style degradation to the whole figure.", "updated_at": now}
    _write_jsonl(output, sorted(existing.values(), key=lambda row: row["candidate_id"]))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        if row["candidate_id"] not in existing:
            grouped.setdefault(str(row["figure_id"]), []).append(row)
    batches = []
    for figure_id, rows in grouped.items():
        for start in range(0, len(rows), max(1, batch_size)):
            batches.append((figure_id, rows[start:start + max(1, batch_size)]))
    lock = threading.Lock()
    contracts = Path(structured_dir)
    overlays = Path(overlays_dir)
    resolved_model = resolve_vlm_model("RFS_CRITIC_MODEL", explicit_model=model)

    def persist(rows: list[dict[str, Any]]) -> None:
        with lock:
            for row in rows:
                existing[row["candidate_id"]] = row
            _write_jsonl(output, sorted(existing.values(), key=lambda row: row["candidate_id"]))

    def run_batch(payload: tuple[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        figure_id, rows = payload
        source = Path(rows[0]["source_image"])
        overlay = _overlay(source, rows, overlays / f"{figure_id}__{rows[0]['candidate_id'].split('__')[-2]}.png")
        contract = json.loads((contracts / f"{rows[0]['paper_id']}.json").read_text(encoding="utf-8"))
        metadata: dict[str, Any] = {}
        try:
            raw = call_vlm_json(_prompt(rows, contract), [overlay], model=resolved_model, timeout=timeout, retries=retries, call_metadata=metadata)
            by_id = {str(item.get("candidate_id")): item for item in raw.get("decisions", []) if isinstance(item, dict)}
        except Exception as exc:
            by_id = {}
            metadata["error"] = f"{type(exc).__name__}: {exc}"
        decisions = []
        for candidate in rows:
            decision = by_id.get(candidate["candidate_id"], {})
            status = str(decision.get("review_status") or "rejected")
            confidence = max(0.0, min(1.0, float(decision.get("confidence") or 0.0)))
            bbox = _valid_bbox(decision.get("bbox_percent"), str(candidate["corruption_type"]))
            if status not in {"approved", "adjusted"} or confidence < 0.85 or bbox is None:
                status = "rejected"
                bbox = candidate["bbox_percent"]
            decisions.append({**candidate, "review_status": status, "bbox_percent": bbox, "region_source": "vlm_adjusted_kira1_review" if status == "adjusted" else candidate.get("region_source"), "reviewer": "Kira1", "review_method": "vlm_assisted_visual_audit", "review_model": resolved_model, "review_confidence": confidence, "review_reason": str(decision.get("reason") or metadata.get("error") or "Rejected because the candidate could not be verified conservatively."), "provider": metadata, "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        persist(decisions)
        return decisions

    completed = []
    with ThreadPoolExecutor(max_workers=max(1, min(5, workers))) as executor:
        futures = [executor.submit(run_batch, batch) for batch in batches]
        for future in as_completed(futures):
            completed.extend(future.result())
    rows = list(existing.values())
    return {"candidates": len(candidates), "reviewed_this_run": len(completed), "approved": sum(row.get("review_status") == "approved" for row in rows), "adjusted": sum(row.get("review_status") == "adjusted" for row in rows), "rejected": sum(row.get("review_status") == "rejected" for row in rows), "pending": len(candidates) - len(rows), "reviews_path": str(output.resolve())}
#: A screened region whose OCR match is weaker than this is rejected rather than edited. A
#: middleware concept is named in a full sentence while the figure labels it in a few words,
#: so one shared content token already scores low; the floor only drops a region whose match
#: carries essentially nothing, and the score travels with the row for the later audit.
MIN_SCREEN_CONFIDENCE = 0.05


def screen_regions_geometrically(candidates_path: str | Path, reviews_path: str | Path) -> dict[str, Any]:
    """Approve or reject every pending region by geometry alone, without a provider call.

    The screen checks only that a region lies inside the image, stays inside its corruption
    class's area budget, and was located with a usable OCR match. It cannot confirm that the
    text or module the contract asked for is really the one inside the box, so every row it
    writes is stamped ``review_method = geometry_screen``; a reader can always tell it apart
    from the VLM-assisted audit. The whole-figure style degradation stays deterministic and
    is approved the same way :func:`review_local_regions` approves it.
    """
    candidates = _jsonl(Path(candidates_path))
    output = Path(reviews_path)
    existing = {row["candidate_id"]: row for row in _jsonl(output)}
    known_before = set(existing)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    for row in candidates:
        if row["candidate_id"] in existing:
            continue
        kind = str(row["corruption_type"])
        confidence = float(row.get("localization_confidence") or 0.0)
        status = "approved"
        reason = "Approved because the located region lies inside the figure and inside its class area budget."
        if kind != "correct_but_ugly":
            partner = row.get("secondary_bbox_percent")
            if _valid_bbox(row.get("bbox_percent"), kind) is None:
                status, reason = "rejected", "Rejected because the located region leaves the image or exceeds its class area budget."
            elif isinstance(partner, dict) and _valid_bbox(partner, kind) is None:
                status, reason = "rejected", "Rejected because the partner region leaves the image or exceeds its class area budget."
            elif confidence < MIN_SCREEN_CONFIDENCE:
                status, reason = "rejected", f"Rejected because the OCR match scored {confidence:.2f}, below the {MIN_SCREEN_CONFIDENCE:.2f} screening floor."
        existing[row["candidate_id"]] = {**row, "review_status": status, "reviewer": None, "review_method": "geometry_screen", "review_confidence": 1.0 if kind == "correct_but_ugly" else round(confidence, 4), "review_reason": reason, "updated_at": now}
    rows = list(existing.values())
    _write_jsonl(output, sorted(rows, key=lambda row: row["candidate_id"]))
    return {"candidates": len(candidates), "reviewed_this_run": sum(row["candidate_id"] not in known_before for row in candidates), "approved": sum(row.get("review_status") == "approved" for row in rows), "adjusted": 0, "rejected": sum(row.get("review_status") == "rejected" for row in rows), "pending": 0, "review_method": "geometry_screen", "reviews_path": str(output.resolve())}
