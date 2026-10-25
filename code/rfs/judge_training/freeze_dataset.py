"""Freeze real and deterministic-local JudgeExamples into an immutable D1 dataset."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .schemas import normalize_example, validate_example


FROZEN_DATASET_VERSION = "judge-d1-v002"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_assets(row: dict[str, Any]) -> list[tuple[str, str]]:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    source = str(provenance.get("source") or "")
    task = str(row.get("task_type") or "")
    if source == "deterministic_local_corruption":
        if task in {"pairwise_preference", "style_preference"}:
            return [(str(row.get("candidate_a") or ""), str(provenance.get("source_sha256") or "")), (str(row.get("candidate_b") or ""), str(provenance.get("output_sha256") or ""))]
        return [(str(row.get("candidate_a") or ""), str(provenance.get("output_sha256") or ""))]
    return [(str(row.get("candidate_a") or ""), str(provenance.get("figure_sha256") or ""))]


def validate_frozen_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Validate schema, assets, hashes, IDs, and paper-level split isolation."""
    root = Path(dataset_dir)
    rows = _jsonl(root / "judge_examples.jsonl")
    errors: list[dict[str, Any]] = []
    ids: set[str] = set()
    paper_splits: dict[str, set[str]] = {}
    assets: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        item_errors = validate_example(row)
        example_id = str(row.get("example_id") or "")
        if example_id in ids:
            item_errors.append("duplicate example_id")
        ids.add(example_id)
        if row.get("dataset_version") != FROZEN_DATASET_VERSION:
            item_errors.append("incorrect frozen dataset_version")
        paper_splits.setdefault(str(row.get("ground_truth_id") or ""), set()).add(str(row.get("split") or ""))
        for asset_path, expected_hash in _expected_assets(row):
            path = Path(asset_path)
            if not asset_path or not path.is_file():
                item_errors.append(f"asset does not exist: {asset_path}")
                continue
            actual_hash = assets.get(str(path)) or _sha256(path)
            assets[str(path)] = actual_hash
            if not expected_hash:
                item_errors.append(f"asset hash missing: {asset_path}")
            elif actual_hash != expected_hash:
                item_errors.append(f"asset hash mismatch: {asset_path}")
        if item_errors:
            errors.append({"line": index, "example_id": example_id, "errors": item_errors})
    leakage = {paper: sorted(splits) for paper, splits in paper_splits.items() if len(splits) > 1}
    if leakage:
        errors.append({"paper_split_leakage": leakage})
    split_files = {"train": "train.jsonl", "validation": "validation.jsonl", "test": "hidden_test.jsonl"}
    for split, filename in split_files.items():
        expected = [row["example_id"] for row in rows if row.get("split") == split]
        actual = [row.get("example_id") for row in _jsonl(root / filename)]
        if expected != actual:
            errors.append({"split_file_mismatch": split})
    return {"records": len(rows), "papers": len(paper_splits), "unique_assets": len(assets), "errors": errors, "ok": not errors}


def freeze_d1_dataset(
    real_examples_path: str | Path,
    local_examples_path: str | Path,
    out_dir: str | Path,
    *,
    region_reviews_path: str | Path | None = None,
    local_results_path: str | Path | None = None,
    audit_queue_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create the final frozen D1 files and fail if strict validation finds errors."""
    real_path = Path(real_examples_path)
    local_path = Path(local_examples_path)
    output = Path(out_dir)
    real = _jsonl(real_path)
    local = _jsonl(local_path)
    rows = []
    for raw in real + local:
        normalized = normalize_example({**raw, "dataset_version": FROZEN_DATASET_VERSION})
        rows.append(normalized)
    rows.sort(key=lambda row: row["example_id"])
    _write_jsonl(output / "judge_examples.jsonl", rows)
    _write_jsonl(output / "train.jsonl", [row for row in rows if row["split"] == "train"])
    _write_jsonl(output / "validation.jsonl", [row for row in rows if row["split"] == "validation"])
    _write_jsonl(output / "hidden_test.jsonl", [row for row in rows if row["split"] == "test"])

    asset_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        for asset_path, expected_hash in _expected_assets(row):
            if asset_path:
                asset_rows.setdefault(asset_path, {"path": asset_path, "sha256": expected_hash, "exists": Path(asset_path).is_file()})
    _write_jsonl(output / "asset_manifest.jsonl", sorted(asset_rows.values(), key=lambda item: item["path"]))

    reviews = _jsonl(Path(region_reviews_path)) if region_reviews_path else []
    results = _jsonl(Path(local_results_path)) if local_results_path else []
    audit = _jsonl(Path(audit_queue_path)) if audit_queue_path else []
    provenance_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    corruption_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for row in rows:
        source = str((row.get("provenance") or {}).get("source") or "unknown")
        task = str(row.get("task_type") or "unknown")
        corruption = str((row.get("provenance") or {}).get("corruption_type") or "none")
        label = str((row.get("labels") or {}).get("quality_label") or "unknown")
        provenance_counts[source] = provenance_counts.get(source, 0) + 1
        task_counts[task] = task_counts.get(task, 0) + 1
        corruption_counts[corruption] = corruption_counts.get(corruption, 0) + 1
        label_counts[label] = label_counts.get(label, 0) + 1
    manifest = {
        "schema_version": "judge-frozen-dataset-manifest-v1",
        "dataset_version": FROZEN_DATASET_VERSION,
        "task": "scientific_framework_judge",
        "status": "frozen",
        "frozen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "records": len(rows),
        "papers": len({row["ground_truth_id"] for row in rows}),
        "unique_assets": len(asset_rows),
        "source_records": {"real": len(real), "deterministic_local": len(local)},
        "split_counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")},
        "task_counts": dict(sorted(task_counts.items())),
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "corruption_counts": dict(sorted(corruption_counts.items())),
        "quality_label_counts": dict(sorted(label_counts.items())),
        "source_files": {"real_examples": str(real_path.resolve()), "real_examples_sha256": _sha256(real_path), "local_examples": str(local_path.resolve()), "local_examples_sha256": _sha256(local_path)},
        "region_review": {"records": len(reviews), "approved": sum(row.get("review_status") == "approved" for row in reviews), "adjusted": sum(row.get("review_status") == "adjusted" for row in reviews), "rejected": sum(row.get("review_status") == "rejected" for row in reviews), "reviewer": "Kira1"},
        "local_generation": {"records": len(results), "generated": sum(row.get("status") == "generated" for row in results), "rejected": sum(row.get("status") == "rejected" for row in results), "failed": sum(row.get("status") == "failed" for row in results)},
        "stratified_spot_check": {"queue_records": len(audit), "status": "pending_optional_second_review", "note": "The regions were fully reviewed by Kira1 and passed deterministic gates; this queue remains available for <AUTHOR>'s optional second review."},
    }
    validation = validate_frozen_dataset(output)
    manifest["validation"] = validation
    manifest["files_sha256"] = {name: _sha256(output / name) for name in ("judge_examples.jsonl", "train.jsonl", "validation.jsonl", "hidden_test.jsonl", "asset_manifest.jsonl")}
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "validation_report.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text("# Judge D1 v002\n\nFrozen Judge dataset built from reviewed real framework figures and deterministic local corruptions. Paper-level splits are preserved. Creator training is out of scope. The stratified spot-check queue remains available as an optional second review.\n", encoding="utf-8")
    if not validation["ok"]:
        raise RuntimeError(f"Frozen D1 validation failed with {len(validation['errors'])} error groups")
    return manifest
