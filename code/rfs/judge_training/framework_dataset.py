"""Compile reviewed framework figures and paper contracts into D1 JudgeExample data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .judge_schema import GROUNDING_TASK_TYPES, entity_ids, human_verdict_quality_scores
from .schemas import normalize_example, validate_example


DATASET_VERSION = "judge-dataset-v002"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ground_truth(contract: dict[str, Any], final_label: str) -> dict[str, Any]:
    expected = contract.get("expected_figure_content") if isinstance(contract.get("expected_figure_content"), dict) else {}
    return {
        "final_label": final_label,
        "method_summary": contract.get("method_summary"),
        "required_entities": expected.get("required_entities", []),
        "required_relations": expected.get("required_relations", []),
        "entities": contract.get("entities", []),
        "relations": contract.get("relations", []),
        "training_flow": contract.get("training_flow", []),
        "inference_flow": contract.get("inference_flow", []),
        "must_not_invent": expected.get("must_not_invent", []),
    }


def _base_example(figure: dict[str, Any], raw: dict[str, Any], contract: dict[str, Any], task_type: str) -> dict[str, Any]:
    aid = str(figure["arxiv_id"])
    figure_path = Path(figure["figure_path"]).resolve()
    label = str(figure["final_label"])
    source = "real_negative" if label.strip() == "坏" else "human_framework_review"
    ground_truth = _ground_truth(contract, label)
    labels: dict[str, Any] = {
        "quality_label": label,
        "quality_scores": human_verdict_quality_scores(label),
        "diagnosis": [],
        "confidence": 1.0,
    }
    if task_type in GROUNDING_TASK_TYPES:
        labels["grounding"] = {"grounded_entities": entity_ids(ground_truth.get("required_entities"))}
    return normalize_example({
        "example_id": f"{task_type}-{figure['figure_id']}",
        "dataset_version": DATASET_VERSION,
        "ground_truth_id": f"arxiv-{aid}",
        "task_type": task_type,
        "split": figure["split"],
        "paper": {"paper_id": aid, "title": raw.get("title"), "abstract": raw.get("abstract"), "evidence_pack": f"evidence/structured/{aid}.json", "evidence_spans": contract.get("evidence_spans", [])},
        "ground_truth": ground_truth,
        "candidate_a": str(figure_path),
        "labels": labels,
        "provenance": {"source": source, "source_arxiv_id": aid, "figure_id": figure["figure_id"], "figure_sha256": _sha256(figure_path), "human_audited": True, "corruption_type": None},
        "judge_metadata": {"judge_role": "bootstrap", "trained": False, "human_audited": True},
    })


def build_framework_judge_dataset(framework_root: str | Path, evidence_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Build real D1 examples while preserving frozen paper-level splits."""
    framework = Path(framework_root)
    evidence = Path(evidence_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figures = _jsonl(framework / "figures.jsonl")
    raw_root = framework / "evidence" / "raw"
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for figure in figures:
        aid = str(figure["arxiv_id"])
        raw = _json(raw_root / f"{aid}.json")
        contract = _json(evidence / "structured" / f"{aid}.json")
        for task_type in ("method_grounding", "scientific_quality", "error_audit"):
            example = _base_example(figure, raw, contract, task_type)
            item_errors = validate_example(example)
            if item_errors:
                errors.append({"example_id": example.get("example_id"), "errors": item_errors})
            else:
                records.append(example)
    records.sort(key=lambda item: item["example_id"])
    for name, rows in {"judge_examples": records, "train": [x for x in records if x["split"] == "train"], "validation": [x for x in records if x["split"] == "validation"], "hidden_test": [x for x in records if x["split"] == "test"]}.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in rows), encoding="utf-8")
    manifest = {
        "schema_version": "judge-dataset-manifest-v1",
        "dataset_version": DATASET_VERSION,
        "source_framework_manifest": str((framework / "dataset_manifest.json").resolve()),
        "records": len(records),
        "paper_count": len({item["ground_truth_id"] for item in records}),
        "figure_count": len(figures),
        "task_counts": {task: sum(x["task_type"] == task for x in records) for task in sorted({x["task_type"] for x in records})},
        "split_counts": {split: sum(x["split"] == split for x in records) for split in ("train", "validation", "test")},
        "errors": errors,
    }
    data_path = out / "judge_examples.jsonl"
    manifest["dataset_sha256"] = _sha256(data_path)
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dataset_report.json").write_text(json.dumps({**manifest, "ok": not errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def validate_framework_judge_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Validate schema, assets, evidence links, hashes, and paper split isolation."""
    root = Path(dataset_dir)
    rows = _jsonl(root / "judge_examples.jsonl")
    errors: list[dict[str, Any]] = []
    paper_splits: dict[str, set[str]] = {}
    for index, row in enumerate(rows, start=1):
        item_errors = validate_example(row)
        candidate = Path(str(row.get("candidate_a") or ""))
        if not candidate.exists():
            item_errors.append("candidate_a does not exist")
        expected_hash = str((row.get("provenance") or {}).get("figure_sha256") or "")
        if candidate.exists() and expected_hash and _sha256(candidate) != expected_hash:
            item_errors.append("candidate_a sha256 mismatch")
        paper_splits.setdefault(str(row.get("ground_truth_id")), set()).add(str(row.get("split")))
        if item_errors:
            errors.append({"line": index, "example_id": row.get("example_id"), "errors": item_errors})
    leakage = {paper: sorted(splits) for paper, splits in paper_splits.items() if len(splits) > 1}
    if leakage:
        errors.append({"split_leakage": leakage})
    return {"records": len(rows), "papers": len(paper_splits), "errors": errors, "ok": not errors}
