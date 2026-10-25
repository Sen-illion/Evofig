"""Compile independently verified corruptions into JudgeExamples and audit queues."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .framework_dataset import DATASET_VERSION, _ground_truth
from .judge_schema import GROUNDING_TASK_TYPES, PAIRWISE_TASK_TYPES, defect_label, entity_ids
from .schemas import normalize_example, validate_example


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _target_text(value: Any) -> str:
    """Render a corruption target that may be an id, a stage pair or a list."""

    if isinstance(value, dict):
        return "; ".join(f"{str(key).strip()}={str(item).strip()}" for key, item in value.items() if str(key).strip())
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip() if value is not None else ""


def _verification_evidence(verdict: dict[str, Any], kind: str, target: str) -> str:
    evidence = verdict.get("evidence") if isinstance(verdict.get("evidence"), list) else []
    text = "; ".join(str(item).strip() for item in evidence if str(item).strip())
    return text or str(verdict.get("reason") or "").strip() or f"independently verified {kind} at {target}"


def _diagnosis_labels(kind: str, task_type: str, ground_truth: dict[str, Any], *, target: str, evidence: str, figure: str | None = None) -> dict[str, Any]:
    """Build the contract-shaped label of one injected defect."""

    defect = defect_label(kind)
    diagnosis = {"type": defect["diagnosis_type"], "target": target, "severity": defect["severity"], "evidence": evidence}
    if figure:
        diagnosis["figure"] = figure
    labels: dict[str, Any] = {
        "quality_label": "坏",
        "quality_scores": dict(defect["quality_scores"]),
        "diagnosis": [diagnosis],
        "confidence": 1.0,
    }
    if task_type in PAIRWISE_TASK_TYPES:
        labels["preferred_candidate"] = "A"
    if task_type in GROUNDING_TASK_TYPES:
        labels["grounding"] = {"grounded_entities": entity_ids(ground_truth.get("required_entities"))}
    return labels


def build_corrupted_examples(
    verification_path: str | Path,
    generated_results_path: str | Path,
    structured_dir: str | Path,
    raw_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Build training records only from strict independently verified edits."""
    verified = {row["corruption_id"]: row for row in _jsonl(Path(verification_path)) if row.get("status") == "verified"}
    generated = {row["corruption_id"]: row for row in _jsonl(Path(generated_results_path))}
    contracts = Path(structured_dir)
    raw_root = Path(raw_dir)
    output = Path(out_dir)
    examples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for corruption_id, verdict in verified.items():
        source = generated.get(corruption_id)
        if not source:
            errors.append({"corruption_id": corruption_id, "error": "missing generation result"})
            continue
        paper_id = str(source["paper_id"])
        contract = _json(contracts / f"{paper_id}.json")
        raw = _json(raw_root / f"{paper_id}.json")
        kind = str(source["corruption_type"])
        ground_truth = _ground_truth(contract, "坏")
        target = _target_text(source.get("target")) or str(source["figure_id"])
        evidence = _verification_evidence(verdict.get("verification") if isinstance(verdict.get("verification"), dict) else {}, kind, target)
        common = {
            "dataset_version": DATASET_VERSION,
            "ground_truth_id": f"arxiv-{paper_id}",
            "split": source["split"],
            "paper": {"paper_id": paper_id, "title": raw.get("title"), "abstract": raw.get("abstract"), "evidence_pack": f"evidence/structured/{paper_id}.json", "evidence_spans": contract.get("evidence_spans", [])},
            "ground_truth": ground_truth,
            "provenance": {"source": "synthetic_verified_corruption", "source_arxiv_id": paper_id, "figure_id": source["figure_id"], "output_sha256": source["output_sha256"], "human_audited": False, "corruption_type": kind, "target": source.get("target"), "verification": verdict.get("verification")},
            "judge_metadata": {"judge_role": "bootstrap", "trained": False, "human_audited": False},
        }
        task_specs = [
            ("scientific_quality", source["output_image"], None),
            ("error_audit", source["output_image"], None),
            ("pairwise_preference", source["source_image"], source["output_image"]),
        ]
        if kind == "correct_but_ugly":
            task_specs.append(("style_preference", source["source_image"], source["output_image"]))
        for task, candidate_a, candidate_b in task_specs:
            labels = _diagnosis_labels(kind, task, ground_truth, target=target, evidence=evidence, figure="B" if task in PAIRWISE_TASK_TYPES else None)
            example = normalize_example({**common, "example_id": f"{task}-{corruption_id}", "task_type": task, "candidate_a": candidate_a, "candidate_b": candidate_b, "labels": labels})
            item_errors = validate_example(example)
            if item_errors:
                errors.append({"example_id": example["example_id"], "errors": item_errors})
            else:
                examples.append(example)
    examples.sort(key=lambda row: row["example_id"])
    _write_jsonl(output / "corrupted_examples.jsonl", examples)
    adversarial = output / "adversarial"
    kinds = sorted({str((row.get("provenance") or {}).get("corruption_type")) for row in examples})
    for kind in kinds:
        _write_jsonl(adversarial / f"{kind}.jsonl", [row for row in examples if (row.get("provenance") or {}).get("corruption_type") == kind])
    report = {"verified_corruptions": len(verified), "examples": len(examples), "task_counts": {task: sum(row["task_type"] == task for row in examples) for task in sorted({row["task_type"] for row in examples})}, "errors": errors, "ok": not errors}
    (output / "corrupted_examples.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_stratified_audit_queue(real_examples_path: str | Path, corrupted_examples_path: str | Path, out_path: str | Path, *, per_stratum: int = 5) -> dict[str, Any]:
    """Select deterministic split-by-label/type samples for the pre-freeze human audit."""
    real = [row for row in _jsonl(Path(real_examples_path)) if row.get("task_type") == "scientific_quality"]
    synthetic = [row for row in _jsonl(Path(corrupted_examples_path)) if row.get("task_type") == "scientific_quality"]
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in real:
        key = f"{row['split']}|real|{(row.get('labels') or {}).get('quality_label')}"
        strata.setdefault(key, []).append(row)
    for row in synthetic:
        key = f"{row['split']}|synthetic|{(row.get('provenance') or {}).get('corruption_type')}"
        strata.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    for key, rows in sorted(strata.items()):
        ranked = sorted(rows, key=lambda row: hashlib.sha256(f"{key}|{row['example_id']}".encode()).hexdigest())
        for row in ranked[:max(1, int(per_stratum))]:
            selected.append({"audit_id": f"audit-{row['example_id']}", "stratum": key, "example_id": row["example_id"], "split": row["split"], "candidate_a": row["candidate_a"], "candidate_b": row.get("candidate_b"), "expected_labels": row.get("labels"), "human_decision": None, "human_notes": None, "status": "pending"})
    _write_jsonl(Path(out_path), selected)
    return {"strata": len(strata), "audit_items": len(selected), "per_stratum": int(per_stratum)}


def build_local_corrupted_examples(
    results_path: str | Path,
    structured_dir: str | Path,
    raw_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Compile generated deterministic local corruptions into JudgeExamples."""
    results = [row for row in _jsonl(Path(results_path)) if row.get("status") == "generated" and row.get("review_status") in {"approved", "adjusted"}]
    contracts, raw_root, output = Path(structured_dir), Path(raw_dir), Path(out_dir)
    examples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for result in results:
        paper_id = str(result["paper_id"])
        contract = _json(contracts / f"{paper_id}.json")
        raw = _json(raw_root / f"{paper_id}.json")
        kind = str(result["corruption_type"])
        ground_truth = _ground_truth(contract, "坏")
        target = _target_text(result.get("target_id")) or str(result["candidate_id"])
        evidence = f"{result.get('operation') or kind} on {target} via {result.get('region_source') or 'deterministic_local_edit'}"
        common = {
            "dataset_version": DATASET_VERSION,
            "ground_truth_id": f"arxiv-{paper_id}",
            "split": result["split"],
            "paper": {"paper_id": paper_id, "title": raw.get("title"), "abstract": raw.get("abstract"), "evidence_pack": f"evidence/structured/{paper_id}.json", "evidence_spans": contract.get("evidence_spans", [])},
            "ground_truth": ground_truth,
            "preview": result["source_image"],
            "provenance": {"source": "deterministic_local_corruption", "source_arxiv_id": paper_id, "figure_id": result["figure_id"], "source_sha256": result["source_sha256"], "output_sha256": result["output_sha256"], "corruption_type": kind, "target": result.get("target_id"), "bbox_percent": result.get("bbox_percent"), "operation": result.get("operation"), "human_audited": True, "automatic_gate": result.get("automatic_gate")},
            "judge_metadata": {"judge_role": "bootstrap", "trained": False, "human_audited": True},
        }
        specs = [
            ("scientific_quality", result["output_image"], None),
            ("error_audit", result["output_image"], None),
            ("pairwise_preference", result["source_image"], result["output_image"]),
        ]
        if kind == "correct_but_ugly":
            specs.append(("style_preference", result["source_image"], result["output_image"]))
        for task, candidate_a, candidate_b in specs:
            labels = _diagnosis_labels(kind, task, ground_truth, target=target, evidence=evidence, figure="B" if task in PAIRWISE_TASK_TYPES else None)
            example = normalize_example({**common, "example_id": f"{task}-{result['candidate_id']}", "task_type": task, "candidate_a": candidate_a, "candidate_b": candidate_b, "labels": labels})
            item_errors = validate_example(example)
            if item_errors:
                errors.append({"example_id": example["example_id"], "errors": item_errors})
            else:
                examples.append(example)
    examples.sort(key=lambda row: row["example_id"])
    _write_jsonl(output / "local_corrupted_examples.jsonl", examples)
    for kind in sorted({(row.get("provenance") or {}).get("corruption_type") for row in examples}):
        _write_jsonl(output / "adversarial" / f"{kind}.jsonl", [row for row in examples if (row.get("provenance") or {}).get("corruption_type") == kind])
    report = {"generated_corruptions": len(results), "examples": len(examples), "task_counts": {task: sum(row["task_type"] == task for row in examples) for task in sorted({row["task_type"] for row in examples})}, "split_counts": {split: sum(row["split"] == split for row in examples) for split in ("train", "validation", "test")}, "errors": errors, "ok": not errors}
    (output / "local_corrupted_examples.report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
