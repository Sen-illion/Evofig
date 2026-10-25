"""Per-class gates, the fixed adversarial set and the Judge promotion decision.

The evaluator measures a Judge as a whole. Promotion needs more than that: each of the
three gating defect classes is gated on its own, a fixed adversarial checklist has to be
covered and passed, the shadow run has to hold against the bootstrap, and only then may
a checkpoint claim the online or frozen role that
:func:`rfs.judge_training.registry.validate_manifest` demands four elements for.

The chain is written down in one place - baseline, training, hidden test, shadow, online,
frozen - so one call answers "where is this checkpoint" with a full gate table, the group
breakdowns and an explicit promotion recommendation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from rfs.utils import write_json, write_text

from .evaluator import _gold, _jsonl, compare_shadow_predictions, evaluate_predictions, load_suite_manifest
from .judge_schema import GATE_DIAGNOSIS_TYPES, diagnosis_types, normalize_judgement, pairwise_choice
from .preference_pairs import known_defects
from .prompts import PROMPT_VERSION


PROMOTION_SCHEMA_VERSION = "judge-promotion-v1"
GATE_CLASSES = GATE_DIAGNOSIS_TYPES
DEFAULT_CLASS_THRESHOLD = 0.9
DEFAULT_ADVERSARIAL_THRESHOLD = 0.9
POLISHED_LAYOUT_SCORE = 0.7
ROLE_ORDER = ("experimental", "shadow", "online", "frozen")

# The fixed adversarial checklist. Each entry states what has to hold; a case that the
# dataset does not cover cannot be passed, so an uncovered checklist fails the gate
# instead of quietly shrinking to what happened to be available.
ADVERSARIAL_CASES: tuple[dict[str, Any], ...] = (
    {"case_id": "beautiful_but_order_error", "kind": "defect", "defect": "module_order_error", "polished": True, "expectation": "a polished figure whose method stages are swapped is named as module_order_error"},
    {"case_id": "beautiful_but_wrong_arrow", "kind": "defect", "defect": "wrong_arrow_target", "polished": True, "expectation": "a polished figure whose arrow targets the wrong module is named as wrong_arrow_target"},
    # ``overlap`` is one case on purpose: the diagnosis vocabulary pins the overlap defect to
    # severity ``medium``, so a severity split would demand a sample that cannot exist and the
    # checklist would fail no matter how good the Judge is. Split it again only once the
    # vocabulary carries a high-severity overlap.
    {"case_id": "overlap", "kind": "defect", "defect": "overlap", "expectation": "a figure whose modules overlap is named as overlap"},
    {"case_id": "ab_swapped", "kind": "pair_order", "expectation": "the same figure is preferred with A and B exchanged"},
    {"case_id": "truth_not_in_A", "kind": "certified_in_b", "expectation": "the certified figure is picked when it sits in position B"},
)


def _labels(row: dict[str, Any]) -> dict[str, Any]:
    return normalize_judgement(row.get("labels") if isinstance(row.get("labels"), dict) else {})


def _prediction(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    source = value.get("prediction") if isinstance(value, dict) and isinstance(value.get("prediction"), dict) else value
    return normalize_judgement(source if isinstance(source, dict) else {})


def _injected_defects(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the injected defect of one example: the provenance first, then the label."""

    defects = known_defects(row)
    if defects:
        return defects
    labels = _labels(row)
    return [{"diagnosis_type": name, "severity": next((entry.get("severity") for entry in labels.get("diagnosis") or [] if entry.get("type") == name), "medium"), "target": "", "source": "label"} for name in diagnosis_types(labels)]


def _layout_score(row: dict[str, Any]) -> float | None:
    scores = _labels(row).get("quality_scores") or {}
    value = scores.get("layout_quality")
    return float(value) if isinstance(value, (int, float)) else None


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def per_class_gates(gold: dict[str, dict[str, Any]], prediction_rows: Sequence[dict[str, Any]], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Gate each of the three defect classes on its own recall and precision."""

    settings = {name: float((thresholds or {}).get(name, DEFAULT_CLASS_THRESHOLD)) for name in GATE_CLASSES}
    predictions = {str(row.get("example_id")): _prediction(row) for row in prediction_rows if row.get("example_id")}
    classes: dict[str, dict[str, Any]] = {name: {"injected": 0, "detected": 0, "predicted": 0, "correct": 0, "missed_example_ids": [], "threshold": settings[name]} for name in GATE_CLASSES}
    for key, row in sorted(gold.items()):
        injected = {str(item.get("diagnosis_type")) for item in _injected_defects(row)} & set(GATE_CLASSES)
        named = set(diagnosis_types(predictions.get(key) or {})) & set(GATE_CLASSES)
        for name in sorted(injected):
            entry = classes[name]
            entry["injected"] += 1
            if name in named:
                entry["detected"] += 1
            else:
                entry["missed_example_ids"].append(key)
        for name in sorted(named):
            entry = classes[name]
            entry["predicted"] += 1
            entry["correct"] += 1 if name in injected else 0
    result: dict[str, Any] = {}
    for name, entry in classes.items():
        recall = _rate(entry["detected"], entry["injected"])
        entry["recall"] = recall
        entry["precision"] = _rate(entry["correct"], entry["predicted"])
        entry["passed"] = recall is not None and recall >= entry["threshold"]
        entry["unavailable_reason"] = "no_injected_sample" if not entry["injected"] else None
        result[name] = entry
    return {"classes": result, "passed": all(bool(entry["passed"]) for entry in result.values()), "coverable": any(entry["injected"] for entry in result.values())}


def adversarial_case_ids(row: dict[str, Any]) -> list[str]:
    """Return the checklist cases one example belongs to."""

    defects = _injected_defects(row)
    classes = {str(item.get("diagnosis_type")) for item in defects}
    severities = {str(item.get("severity") or "medium") for item in defects}
    polished = (_layout_score(row) or 0.0) >= POLISHED_LAYOUT_SCORE
    labels = _labels(row)
    cases: list[str] = []
    for case in ADVERSARIAL_CASES:
        if case["kind"] == "defect":
            if case["defect"] not in classes:
                continue
            if case.get("polished") and not polished:
                continue
            if case.get("severity") and severities.isdisjoint(set(case["severity"])):
                continue
            cases.append(str(case["case_id"]))
        elif case["kind"] == "certified_in_b":
            if "preferred_candidate" in labels and pairwise_choice(labels) == "B":
                cases.append(str(case["case_id"]))
    return cases


def _pair_groups(gold: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for key, row in sorted(gold.items()):
        group = str(row.get("pair_group") or "")
        if group:
            groups.setdefault(group, []).append(key)
    return {group: keys for group, keys in groups.items() if len(keys) > 1}


def adversarial_gates(gold: dict[str, dict[str, Any]], prediction_rows: Sequence[dict[str, Any]], *, threshold: float = DEFAULT_ADVERSARIAL_THRESHOLD) -> dict[str, Any]:
    """Check the fixed adversarial checklist, coverage included."""

    predictions = {str(row.get("example_id")): _prediction(row) for row in prediction_rows if row.get("example_id")}
    samples: dict[str, list[str]] = {}
    for key, row in sorted(gold.items()):
        for case_id in adversarial_case_ids(row):
            samples.setdefault(case_id, []).append(key)
    swapped_groups = _pair_groups(gold)
    samples["ab_swapped"] = [key for keys in swapped_groups.values() for key in keys]
    cases: dict[str, dict[str, Any]] = {}
    for case in ADVERSARIAL_CASES:
        case_id = str(case["case_id"])
        keys = samples.get(case_id, [])
        passed = 0
        failures: list[str] = []
        if case["kind"] == "pair_order":
            for group, group_keys in sorted(swapped_groups.items()):
                picks = {pairwise_choice(predictions.get(key) or {}) for key in group_keys}
                if picks == {"A", "B"} and "TIE" not in picks and "" not in picks:
                    passed += 1
                else:
                    failures.append(group + ": " + ",".join(sorted(pick or "none" for pick in picks)))
            total = len(swapped_groups)
        else:
            for key in keys:
                output = predictions.get(key) or {}
                if case["kind"] == "certified_in_b":
                    hit = pairwise_choice(output) == pairwise_choice(_labels(gold[key]))
                else:
                    hit = str(case["defect"]) in set(diagnosis_types(output))
                passed += 1 if hit else 0
                if not hit:
                    failures.append(key)
            total = len(keys)
        rate = _rate(passed, total)
        cases[case_id] = {
            "kind": str(case["kind"]),
            "expectation": str(case["expectation"]),
            "covered": bool(total),
            "samples": total,
            "passed": passed,
            "rate": rate,
            "threshold": threshold,
            "gate_passed": bool(total) and rate is not None and rate >= threshold,
            "failures": failures[:20],
            "unavailable_reason": None if total else "case_not_covered_by_this_dataset",
        }
    return {"cases": cases, "passed": all(entry["gate_passed"] for entry in cases.values()), "uncovered": sorted(case_id for case_id, entry in cases.items() if not entry["covered"])}


def shadow_gate(trained: dict[str, Any], bootstrap: dict[str, Any]) -> dict[str, Any]:
    """Decide shadow stability: the trained Judge must not lose to the bootstrap."""

    reasons: list[str] = []
    for metric in ("scientific_error_recall", "pairwise_accuracy"):
        trained_value = (trained.get("metrics") or {}).get(metric, {}).get("value")
        bootstrap_value = (bootstrap.get("metrics") or {}).get(metric, {}).get("value")
        if trained_value is None or bootstrap_value is None:
            continue
        if float(trained_value) < float(bootstrap_value):
            reasons.append(f"trained {metric} {trained_value} is below bootstrap {bootstrap_value}")
    if not trained.get("gate_pass"):
        reasons.append("the trained Judge does not pass the shadow suite gates")
    return {"stable": not reasons, "reasons": reasons}


def promotion_decision(*, hidden_test_passed: bool, class_gates_passed: bool, adversarial_passed: bool, shadow_stable: bool, baseline_passed: bool | None = None) -> dict[str, Any]:
    """Build the four-element promotion block the registry validates."""

    reasons: list[str] = []
    if not hidden_test_passed:
        reasons.append("the hidden test gates did not pass")
    if not class_gates_passed:
        reasons.append("at least one gating defect class is below its threshold")
    if not adversarial_passed:
        reasons.append("the fixed adversarial checklist did not pass")
    if baseline_passed is False:
        reasons.append("the baseline did not pass its own run")
    approved = not reasons
    # Shadow stability is the fourth element: the registry demands it for the frozen role
    # on top of approval, so an approved but unstable checkpoint may serve online only.
    notes = list(reasons)
    if not shadow_stable:
        notes.append("the shadow comparison against the bootstrap is not stable")
    roles = ["experimental"]
    if approved:
        roles.append("shadow")
    if approved and hidden_test_passed and adversarial_passed and class_gates_passed:
        roles.append("online")
    if "online" in roles and shadow_stable:
        roles.append("frozen")
    if approved and shadow_stable:
        advice = "promote to frozen"
    elif approved:
        advice = "promote to online and keep it in shadow until the shadow comparison is stable"
    else:
        advice = "hold and repair: " + "; ".join(reasons)
    return {
        "approved": approved,
        "hidden_test_passed": bool(hidden_test_passed),
        "adversarial_passed": bool(adversarial_passed),
        "shadow_stable": bool(shadow_stable),
        "class_gates_passed": bool(class_gates_passed),
        "roles": [role for role in ROLE_ORDER if role in roles],
        "advice": advice,
        "reasons": reasons,
        "notes": notes,
    }


def _chain(baseline: dict[str, Any] | None, training: dict[str, Any] | None, hidden: dict[str, Any], shadow: dict[str, Any] | None, promotion: dict[str, Any]) -> list[dict[str, Any]]:
    def row(step: str, status: str, detail: str) -> dict[str, Any]:
        return {"step": step, "status": status, "detail": detail}

    entries = [
        row("baseline", "not_run" if baseline is None else ("passed" if baseline.get("gate_pass") else "failed"), "bootstrap VLM baseline on the same suite" if baseline else "no baseline predictions supplied"),
        row("training", "not_run" if training is None else "reported", f"{training.get('metrics', {}).get('steps')} steps, mean margin {training.get('metrics', {}).get('mean_margin')}" if training else "no training report supplied"),
        row("hidden_test", "passed" if hidden.get("gate_pass") else "failed", "; ".join(hidden.get("gate_failures") or []) or "all gates passed"),
    ]
    if shadow is None:
        entries.append(row("shadow", "not_run", "no bootstrap predictions supplied"))
    else:
        entries.append(row("shadow", "passed" if shadow.get("stable") else "failed", "; ".join(shadow.get("reasons") or []) or "trained Judge holds against the bootstrap"))
    entries.append(row("online", "ready" if "online" in promotion["roles"] else "blocked", promotion["advice"]))
    entries.append(row("frozen", "ready" if "frozen" in promotion["roles"] else "blocked", "needs an approved, stable checkpoint" if "frozen" not in promotion["roles"] else "all four promotion elements hold"))
    return entries


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Judge promotion report",
        "",
        f"- Suite: `{report['suite']['suite_id']}`",
        f"- Gate pass: `{report['hidden_test']['gate_pass']}`",
        f"- Promotion: `{report['promotion']['advice']}`",
        f"- Prompt version: `{PROMPT_VERSION}`",
        "",
        "## Chain",
        "",
        "| Step | Status | Detail |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {row['step']} | {row['status']} | {row['detail']} |" for row in report["chain"]]
    lines += ["", "## Gates", "", "| Metric | Value | Threshold | Passed |", "| --- | ---: | ---: | :---: |"]
    lines += [f"| {name} | {entry.get('value')} | {entry.get('threshold')} | {entry.get('passed')} |" for name, entry in sorted(report["hidden_test"]["metrics"].items())]
    lines += ["", "## Gating defect classes", "", "| Class | Injected | Detected | Recall | Precision | Passed |", "| --- | ---: | ---: | ---: | ---: | :---: |"]
    lines += [f"| {name} | {entry['injected']} | {entry['detected']} | {entry['recall']} | {entry['precision']} | {entry['passed']} |" for name, entry in sorted(report["class_gates"]["classes"].items())]
    lines += ["", "## Adversarial checklist", "", "| Case | Samples | Passed | Rate | Passed gate |", "| --- | ---: | ---: | ---: | :---: |"]
    lines += [f"| {name} | {entry['samples']} | {entry['passed']} | {entry['rate']} | {entry['gate_passed']} |" for name, entry in sorted(report["adversarial"]["cases"].items())]
    lines += ["", "## Groups", "", "| Group | Samples | Error recall | Pairwise accuracy |", "| --- | ---: | ---: | ---: |"]
    for field in sorted(report["hidden_test"]["breakdowns"]):
        for name, entry in sorted(report["hidden_test"]["breakdowns"][field].items()):
            lines.append(f"| {field}:{name} | {entry['sample_count']} | {entry['scientific_error_recall']} | {entry['pairwise_accuracy']} |")
    if report["promotion"]["reasons"]:
        lines += ["", "## Blockers", ""] + [f"- {reason}" for reason in report["promotion"]["reasons"]]
    return chr(10).join(lines) + chr(10)


def build_promotion_report(*, dataset: str | Path, predictions: str | Path, suite_manifest: str | Path | None = None, suite: str = "hidden_test", checkpoint: str | Path | None = None, baseline_predictions: str | Path | None = None, bootstrap_predictions: str | Path | None = None, training_report: str | Path | None = None, class_thresholds: dict[str, float] | None = None, adversarial_threshold: float = DEFAULT_ADVERSARIAL_THRESHOLD, report_out: str | Path | None = None, strict_gates: bool = False) -> dict[str, Any]:
    """Turn one checkpoint's predictions into the full gate table and a promotion call.

    ``gate_pass`` carries the verdict and ``promotion.approved`` the recommendation;
    ``ok`` only reports whether the call itself succeeded on a trusted dataset, so a
    blocked checkpoint stays exit code zero until ``strict_gates`` is asked for.
    """

    manifest = load_suite_manifest(suite_manifest, dataset, suite)
    rows = _jsonl(dataset)
    gold = _gold(rows, manifest)
    prediction_rows = _jsonl(predictions)
    hidden = evaluate_predictions(dataset, predictions, checkpoint=checkpoint, suite=suite, suite_manifest=suite_manifest)
    classes = per_class_gates(gold, prediction_rows, class_thresholds)
    adversarial = adversarial_gates(gold, prediction_rows, threshold=adversarial_threshold)
    baseline = evaluate_predictions(dataset, baseline_predictions, suite=f"{suite}_baseline", suite_manifest=suite_manifest) if baseline_predictions else None
    shadow = None
    if bootstrap_predictions:
        comparison = compare_shadow_predictions(dataset, predictions, bootstrap_predictions, checkpoint=checkpoint, suite_manifest=suite_manifest)
        shadow = {**shadow_gate(comparison["trained_evaluation"], comparison["bootstrap_evaluation"]), "disagreement_rate": comparison.get("disagreement_rate"), "common_records": comparison.get("common_records")}
    training = json.loads(Path(training_report).read_text(encoding="utf-8")) if training_report and Path(training_report).is_file() else None
    promotion = promotion_decision(
        hidden_test_passed=bool(hidden.get("gate_pass")) and bool(manifest["dataset_integrity"]),
        class_gates_passed=bool(classes["passed"]),
        adversarial_passed=bool(adversarial["passed"]),
        shadow_stable=bool(shadow["stable"]) if shadow is not None else False,
        baseline_passed=bool(baseline.get("gate_pass")) if baseline is not None else None,
    )
    report: dict[str, Any] = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "checkpoint": str(checkpoint) if checkpoint else "",
        "dataset": str(dataset),
        "predictions": str(predictions),
        "suite": {"suite_id": manifest["suite_id"], "dataset_integrity": bool(manifest["dataset_integrity"]), "gold_records": len(gold), "prediction_records": len(prediction_rows)},
        "hidden_test": hidden,
        "class_gates": classes,
        "adversarial": adversarial,
        "baseline": baseline,
        "shadow": shadow,
        "training": {"steps": (training or {}).get("metrics", {}).get("steps"), "mean_margin": (training or {}).get("metrics", {}).get("mean_margin"), "pairwise_accuracy": (training or {}).get("metrics", {}).get("pairwise_accuracy"), "gated_recall": (training or {}).get("metrics", {}).get("gated_recall")} if training else None,
        "promotion": promotion,
        "summary": f"Judge promotion gate table for {manifest['suite_id']}.",
        "ok": bool(manifest["dataset_integrity"]),
    }
    gates = (("hidden_test", hidden.get("gate_pass")), ("class_gates", classes["passed"]), ("adversarial", adversarial["passed"]), ("shadow", None if shadow is None else shadow["stable"]))
    report["gate_pass"] = all(passed is not False for _, passed in gates)
    report["chain"] = _chain(baseline, training, hidden, shadow, promotion)
    if strict_gates and not report["gate_pass"]:
        report["ok"] = False
        report["strict_gates_failed"] = [name for name, passed in gates if passed is False]
    if report_out:
        target = Path(report_out)
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "promotion_report.json", report)
        write_text(target / "promotion_report.md", _markdown(report))
    return report
