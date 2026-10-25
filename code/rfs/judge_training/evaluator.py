"""Offline Judge evaluation, suite manifests, gates, and shadow comparison.

Every judgement field is read through :mod:`rfs.judge_training.judge_schema`, so the
metrics, the prompt contract, and the dataset validator stay in step.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rfs.utils import write_json, write_text

from .judge_schema import grounding_field, has_blocking_diagnosis, normalize_judgement, pairwise_choice, scientific_score

DEFAULT_THRESHOLDS = {
    "scientific_error_recall": 0.95, "scientific_error_miss_rate": 0.05,
    "beautiful_but_wrong_block_rate": 0.95, "required_entity_recall": 0.90,
    "invented_entity_recall": 0.95, "relation_direction_accuracy": 0.90,
    "pairwise_accuracy": 0.80, "aesthetic_preference_accuracy": 0.75,
    "repair_effectiveness": 0.70, "repeat_score_std": 0.03,
    "cross_domain_pairwise_accuracy": 0.70, "json_valid_rate": 0.99,
}

def _jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists(): raise FileNotFoundError(source)
    rows = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        try: value = json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(value, dict): value["_line"] = number; rows.append(value)
    return rows

def _prediction(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("prediction") if isinstance(row.get("prediction"), dict) else row
    return normalize_judgement(value)

def _labels(row: dict[str, Any]) -> dict[str, Any]:
    return normalize_judgement(row.get("labels") if isinstance(row.get("labels"), dict) else {})

def _list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)): return [str(x) for x in value if str(x).strip()]
    return [] if value in (None, "") else [str(value)]

def _rate(n: int, d: int) -> float | None: return round(n / d, 4) if d else None
def _mean(v: list[float]) -> float | None: return round(sum(v) / len(v), 4) if v else None

def _metric(name: str, value: float | None, threshold: float | None, *, direction: str = "ge", count: int = 0) -> dict[str, Any]:
    passed = None if value is None or threshold is None else (value >= threshold if direction == "ge" else value <= threshold)
    return {"name": name, "value": value, "threshold": threshold, "direction": direction, "sample_count": count, "passed": passed, "unavailable_reason": "no_compatible_samples" if value is None else None}

def _group_breakdown(gold: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[str]] = {}
    for key, row in gold.items():
        labels = _labels(row); value = row.get(field) or labels.get(field)
        if value: groups.setdefault(str(value), []).append(key)
    result = {}
    for name, ids in sorted(groups.items()):
        expected_errors = detected_errors = pair_total = pair_correct = 0
        for key in ids:
            labels, output = _labels(gold[key]), predictions.get(key, {})
            if has_blocking_diagnosis(labels): expected_errors += 1; detected_errors += int(has_blocking_diagnosis(output))
            if "preferred_candidate" in labels: pair_total += 1; pair_correct += int(pairwise_choice(output) == pairwise_choice(labels))
        result[name] = {"sample_count": len(ids), "scientific_error_recall": _rate(detected_errors, expected_errors), "pairwise_accuracy": _rate(pair_correct, pair_total)}
    return result

def load_suite_manifest(path: str | Path | None, dataset: str | Path, suite: str = "hidden_test") -> dict[str, Any]:
    """Load a frozen suite manifest while supporting bare legacy JSONL datasets."""
    source = Path(path) if path else None
    raw = json.loads(source.read_text(encoding="utf-8")) if source and source.exists() else {}
    if not isinstance(raw, dict): raise ValueError("suite manifest must be a JSON object")
    data = Path(dataset)
    if data.is_dir(): data = data / "judge_examples.jsonl"
    digest = hashlib.sha256(data.read_bytes()).hexdigest() if data.exists() else None
    expected = raw.get("dataset_sha256")
    thresholds = dict(DEFAULT_THRESHOLDS); thresholds.update(raw.get("thresholds") if isinstance(raw.get("thresholds"), dict) else {})
    return {"suite_id": str(raw.get("suite_id") or suite), "suite": suite, "manifest_path": str(source.resolve()) if source else None, "dataset": str(data.resolve()), "dataset_version": raw.get("dataset_version"), "dataset_sha256": digest, "expected_dataset_sha256": expected, "dataset_integrity": expected in (None, digest), "thresholds": thresholds, "example_ids": raw.get("example_ids"), "adversarial_types": raw.get("adversarial_types", [])}

def build_suite_manifest(dataset: str | Path, out: str | Path, *, suite: str = "hidden_test", dataset_version: str | None = None, example_ids: list[str] | None = None, thresholds: dict[str, float] | None = None, adversarial_types: list[str] | None = None) -> dict[str, Any]:
    """Write a content-addressed frozen suite manifest for a JSONL dataset."""
    data = Path(dataset)
    if data.is_dir(): data = data / "judge_examples.jsonl"
    payload = {"schema_version": "judge-suite-v1", "suite_id": suite, "suite": suite, "dataset": str(data.resolve()), "dataset_version": dataset_version, "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(), "example_ids": example_ids or [], "thresholds": {**DEFAULT_THRESHOLDS, **(thresholds or {})}, "adversarial_types": adversarial_types or []}
    write_json(out, payload)
    return payload

def _gold(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ids = {str(x) for x in manifest.get("example_ids") or []}; result = {}
    for row in rows:
        key = str(row.get("example_id") or "")
        if key and (not ids or key in ids): result[key] = row
    return result

def _set_recall(expected: list[list[str]], actual: list[list[str]]) -> tuple[float | None, int]:
    hits = total = 0
    for left, right in zip(expected, actual):
        a, b = set(left), set(right); total += len(a); hits += len(a & b)
    return _rate(hits, total), total

def _evaluate(gold: dict[str, dict[str, Any]], prediction_rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    predictions = {str(x.get("example_id")): _prediction(x) for x in prediction_rows if x.get("example_id")}
    scores: list[float] = []; blocker_tp = blocker_fn = blocker_fp = 0; json_valid = 0
    required_g: list[list[str]] = []; required_p: list[list[str]] = []; invented_g: list[list[str]] = []; invented_p: list[list[str]] = []
    direction_hits = direction_total = pair_ok = pair_n = aesthetic_ok = aesthetic_n = repair_ok = repair_n = 0
    beautiful_ok = beautiful_n = cross_domain_ok = cross_domain_n = terminology_hits = terminology_total = 0
    missing_rel_g: list[list[str]] = []; missing_rel_p: list[list[str]] = []
    pair_orders: dict[str, list[str]] = {}
    repeats: dict[str, list[float]] = {}
    for key, row in gold.items():
        labels, output = _labels(row), predictions.get(key, {})
        if output: json_valid += 1
        label_score, output_score = scientific_score(labels), scientific_score(output)
        if label_score is not None and output_score is not None:
            scores.append(abs(label_score - output_score))
            repeats.setdefault(str(row.get("repeat_group") or key), []).append(output_score)
        expected_block = has_blocking_diagnosis(labels); actual_block = has_blocking_diagnosis(output)
        if expected_block and actual_block: blocker_tp += 1
        elif expected_block: blocker_fn += 1
        elif actual_block: blocker_fp += 1
        ground_truth = row.get("ground_truth") if isinstance(row.get("ground_truth"), dict) else {}
        required_g.append(_list(grounding_field(labels, "grounded_entities") or ground_truth.get("required_entities"))); required_p.append(_list(grounding_field(output, "grounded_entities")))
        invented_g.append(_list(grounding_field(labels, "ungrounded_entities"))); invented_p.append(_list(grounding_field(output, "ungrounded_entities")))
        missing_rel_g.append(_list(grounding_field(labels, "missing_relations"))); missing_rel_p.append(_list(grounding_field(output, "missing_relations")))
        expected_terms = _list(grounding_field(labels, "terminology_errors")); actual_terms = set(_list(grounding_field(output, "terminology_errors")))
        terminology_total += len(expected_terms); terminology_hits += sum(term in actual_terms for term in expected_terms)
        expected_rel = grounding_field(labels, "relation_directions"); actual_rel = grounding_field(output, "relation_directions")
        for rel, direction in expected_rel.items(): direction_total += 1; direction_hits += int(actual_rel.get(rel) == direction)
        if row.get("task_type") in {"image_pairwise", "pairwise_preference"}:
            correct = int(pairwise_choice(output) == pairwise_choice(labels)); pair_n += 1; pair_ok += correct
            if row.get("domain") or labels.get("domain"): cross_domain_n += 1; cross_domain_ok += correct
            if row.get("pair_group"): pair_orders.setdefault(str(row["pair_group"]), []).append(pairwise_choice(output))
        if row.get("task_type") == "style_preference":
            aesthetic_n += 1; aesthetic_ok += int(pairwise_choice(output) == pairwise_choice(labels))
        if "feedback_effective" in labels: repair_n += 1; repair_ok += int(bool(output.get("feedback_effective")) == bool(labels.get("feedback_effective")))
        adversarial_type = str(row.get("adversarial_type") or labels.get("adversarial_type") or "")
        if adversarial_type == "beautiful_but_wrong": beautiful_n += 1; beautiful_ok += int(actual_block)
    required, required_n = _set_recall(required_g, required_p); invented, invented_n = _set_recall(invented_g, invented_p); missing_rel, missing_rel_n = _set_recall(missing_rel_g, missing_rel_p)
    stds = []
    for values in repeats.values():
        if len(values) > 1:
            mean = sum(values) / len(values); stds.append(math.sqrt(sum((x - mean) ** 2 for x in values) / len(values)))
    order_groups = [values for values in pair_orders.values() if len(values) > 1]
    order_consistent = _rate(sum(len(set(values)) == 1 for values in order_groups), len(order_groups))
    precision = _rate(blocker_tp, blocker_tp + blocker_fp)
    recall = _rate(blocker_tp, blocker_tp + blocker_fn)
    f1 = round(2 * precision * recall / (precision + recall), 4) if precision is not None and recall is not None and precision + recall else None
    metrics = {
        "scientific_mae": _metric("scientific_mae", _mean(scores), None, count=len(scores)),
        "scientific_error_recall": _metric("scientific_error_recall", _rate(blocker_tp, blocker_tp + blocker_fn), thresholds.get("scientific_error_recall"), count=blocker_tp + blocker_fn),
        "scientific_error_precision": _metric("scientific_error_precision", precision, None, count=blocker_tp + blocker_fp),
        "scientific_error_f1": _metric("scientific_error_f1", f1, None, count=blocker_tp + blocker_fn + blocker_fp),
        "scientific_error_miss_rate": _metric("scientific_error_miss_rate", _rate(blocker_fn, blocker_tp + blocker_fn), thresholds.get("scientific_error_miss_rate"), direction="le", count=blocker_tp + blocker_fn),
        "beautiful_but_wrong_block_rate": _metric("beautiful_but_wrong_block_rate", _rate(beautiful_ok, beautiful_n), thresholds.get("beautiful_but_wrong_block_rate"), count=beautiful_n),
        "required_entity_recall": _metric("required_entity_recall", required, thresholds.get("required_entity_recall"), count=required_n),
        "invented_entity_recall": _metric("invented_entity_recall", invented, thresholds.get("invented_entity_recall"), count=invented_n),
        "relation_direction_accuracy": _metric("relation_direction_accuracy", _rate(direction_hits, direction_total), thresholds.get("relation_direction_accuracy"), count=direction_total),
        "missing_relation_recall": _metric("missing_relation_recall", missing_rel, None, count=missing_rel_n),
        "terminology_error_recall": _metric("terminology_error_recall", _rate(terminology_hits, terminology_total), None, count=terminology_total),
        "pairwise_accuracy": _metric("pairwise_accuracy", _rate(pair_ok, pair_n), thresholds.get("pairwise_accuracy"), count=pair_n),
        "aesthetic_preference_accuracy": _metric("aesthetic_preference_accuracy", _rate(aesthetic_ok, aesthetic_n), thresholds.get("aesthetic_preference_accuracy"), count=aesthetic_n),
        "repair_effectiveness": _metric("repair_effectiveness", _rate(repair_ok, repair_n), thresholds.get("repair_effectiveness"), count=repair_n),
        "repeat_score_std": _metric("repeat_score_std", _mean(stds), thresholds.get("repeat_score_std"), direction="le", count=len(stds)),
        "pair_order_consistency": _metric("pair_order_consistency", order_consistent, None, count=len(order_groups)),
        "cross_domain_pairwise_accuracy": _metric("cross_domain_pairwise_accuracy", _rate(cross_domain_ok, cross_domain_n), thresholds.get("cross_domain_pairwise_accuracy"), count=cross_domain_n),
        "json_valid_rate": _metric("json_valid_rate", _rate(json_valid, len(gold)), thresholds.get("json_valid_rate"), count=len(gold)),
    }
    for item in metrics.values(): item["coverage"] = round(item["sample_count"] / len(gold), 4) if gold else 0.0
    failures = [name for name, item in metrics.items() if item["threshold"] is not None and item["passed"] is not True]
    return {"metrics": metrics, "scientific_mae": metrics["scientific_mae"]["value"], "gate_pass": not failures, "gate_failures": failures, "gold_records": len(gold), "prediction_records": len(prediction_rows), "matched_records": len(set(gold) & set(predictions)), "breakdowns": {"task_type": _group_breakdown(gold, predictions, "task_type"), "domain": _group_breakdown(gold, predictions, "domain"), "adversarial_type": _group_breakdown(gold, predictions, "adversarial_type")}}

def _write_report(result: dict[str, Any], report_out: str | Path | None) -> dict[str, Any]:
    if report_out:
        root = Path(report_out); root.mkdir(parents=True, exist_ok=True)
        lines = ["# Judge Evaluation Report", "", f"- Suite: `{result.get('suite')}`", f"- Gate pass: `{result.get('gate_pass')}`", f"- Matched records: `{result.get('matched_records')}`", "", "| Metric | Value | Threshold | Passed |", "|---|---:|---:|:---:|"]
        for name, item in result.get("metrics", {}).items(): lines.append(f"| {name} | {item.get('value')} | {item.get('threshold')} | {item.get('passed')} |")
        if result.get("gate_failures"): lines += ["", "## Gate failures", ""] + [f"- `{x}`" for x in result["gate_failures"]]
        result["report_json"] = str((root / "judge_evaluation_report.json").resolve()); result["report_markdown"] = str((root / "judge_evaluation_report.md").resolve())
        write_json(root / "judge_evaluation_report.json", result)
        write_text(root / "judge_evaluation_report.md", "\n".join(lines) + "\n")
    return result

def evaluate_predictions(dataset: str | Path, predictions: str | Path, *, checkpoint: str | Path | None = None, suite: str = "hidden_test", suite_manifest: str | Path | None = None, report_out: str | Path | None = None, strict_gates: bool = False) -> dict[str, Any]:
    manifest = load_suite_manifest(suite_manifest, dataset, suite); gold = _gold(_jsonl(manifest["dataset"]), manifest)
    result = {"summary": "Offline Judge prediction evaluation.", "ok": bool(manifest["dataset_integrity"]), "checkpoint": str(checkpoint) if checkpoint else None, "suite": suite, "suite_manifest": manifest, **_evaluate(gold, _jsonl(predictions), manifest["thresholds"])}
    result = _write_report(result, report_out)
    if strict_gates and not result["gate_pass"]: result["ok"] = False
    return result

def compare_shadow_predictions(dataset: str | Path, trained_predictions: str | Path, bootstrap_predictions: str | Path, *, checkpoint: str | Path | None = None, suite_manifest: str | Path | None = None, report_out: str | Path | None = None, strict_gates: bool = False) -> dict[str, Any]:
    manifest = load_suite_manifest(suite_manifest, dataset, "shadow"); gold = _gold(_jsonl(manifest["dataset"]), manifest); trained_rows, bootstrap_rows = _jsonl(trained_predictions), _jsonl(bootstrap_predictions)
    trained = {str(x.get("example_id")): _prediction(x) for x in trained_rows if x.get("example_id")}; bootstrap = {str(x.get("example_id")): _prediction(x) for x in bootstrap_rows if x.get("example_id")}; common = sorted(set(gold) & set(trained) & set(bootstrap)); disagreements = []
    for key in common:
        if json.dumps(trained[key], sort_keys=True, ensure_ascii=False) != json.dumps(bootstrap[key], sort_keys=True, ensure_ascii=False): disagreements.append({"example_id": key, "task_type": gold[key].get("task_type"), "trained": trained[key], "bootstrap": bootstrap[key]})
    result = {"summary": "Shadow Judge comparison against Bootstrap predictions.", "ok": bool(manifest["dataset_integrity"]), "checkpoint": str(checkpoint) if checkpoint else None, "suite": "shadow", "suite_manifest": manifest, "common_records": len(common), "disagreement_count": len(disagreements), "disagreement_rate": _rate(len(disagreements), len(common)), "disagreements": disagreements, "trained_evaluation": _evaluate(gold, trained_rows, manifest["thresholds"]), "bootstrap_evaluation": _evaluate(gold, bootstrap_rows, manifest["thresholds"])}; result["gate_pass"] = result["trained_evaluation"]["gate_pass"]
    result = _write_report(result, report_out)
    if strict_gates and not result["gate_pass"]: result["ok"] = False
    return result
