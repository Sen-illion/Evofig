"""Choice-exam gate table: the promotion decision read from two-field answers.

The judgement-format gate table in :mod:`rfs.judge_training.promotion` reads
``labels.diagnosis`` and ``preferred_candidate``. The choice Judge answers
``{"wrong_figure": "A or B", "defect": "..."}`` instead, so it cannot be scored by that
table: ``judge promote`` would find no diagnosis in a choice prediction and report every
gate as failed, which reads as a bad Judge rather than as a format mismatch.

This module computes the same four gate groups from the predictions the choice scorer
already produces, and hands them to the shared :func:`promotion_decision`, so both answer
formats land on one promotion contract and the registry reads one shape.

Gate groups, and what each one refuses:

- ``hidden``: the headline arithmetic on the held-out exam - figure accuracy, defect
  accuracy and answer legality.
- ``classes``: recall of each gating defect (order, arrow, overlap) computed on its own
  samples, so a high average cannot hide a class the Judge never names.
- ``adversarial``: the ways a two-field answer can be right by accident - naming the same
  slot every time, and passing one orientation of a pair while failing its mirror.
- ``shadow``: not computable here. The shadow comparison compares two runs on a shared
  judgement suite; choice scoring has no bootstrap counterpart, so this gate reports
  ``None`` with a reason instead of a pass, and the promotion decision keeps the frozen
  role blocked until a shadow comparison exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rfs.utils import write_json, write_text

from .judge_schema import GATE_DIAGNOSIS_TYPES
from .choice_items import _side_orientation
from .promotion import PROMOTION_SCHEMA_VERSION, promotion_decision

CHOICE_GATE_SCHEMA_VERSION = "judge-choice-gate-v1"

#: The published thresholds this project gates its Judge on. ``pairwise`` is the figure
#: pick, ``classes`` are the three gating defects, and the adversarial pair mirrors the
#: per-position checks. Override any of them through the ``thresholds`` argument.
CHOICE_GATE_THRESHOLDS: dict[str, float] = {
    "figure_accuracy": 0.80,
    "defect_accuracy": 0.80,
    "legal_rate": 0.99,
    "module_order_error": 0.90,
    "wrong_arrow_target": 0.95,
    "overlap": 0.95,
    "ab_swapped": 0.90,
    "certified_in_a": 0.90,
    "certified_in_b": 0.90,
    "pairwise_both_sides": 0.80,
}

CHOICE_ADVERSARIAL_CASES: tuple[dict[str, str], ...] = (
    {"case_id": "ab_swapped", "expectation": "the same pair is named correctly with the certified figure in A and in B"},
    {"case_id": "certified_in_a", "expectation": "the wrong figure is named when the certified figure sits in slot A"},
    {"case_id": "certified_in_b", "expectation": "the wrong figure is named when the certified figure sits in slot B"},
    {"case_id": "pairwise_both_sides", "expectation": "figure and defect are both right on both orientations of a pair"},
)


def _predictions(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"no choice predictions at {source}")
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 4) if total else None


def _orientation(row: Mapping[str, Any]) -> str:
    """One vocabulary for the slot, whether predicted now or read from an older run."""

    return _side_orientation(row)


def _gate(metric: str, value: float | None, threshold: float, *, samples: int) -> dict[str, Any]:
    return {
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "samples": samples,
        "passed": bool(value is not None and value >= threshold),
        "unavailable_reason": "no_samples" if not samples else None,
    }


def pair_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group predicted sides into pairs and decide each pair's figure and defect verdicts."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = str(row.get("pair_id") or row.get("example_id") or "")
        if key:
            grouped.setdefault(key, []).append(row)
    table: dict[str, dict[str, Any]] = {}
    for key, sides in grouped.items():
        # A pair counts as right only when every orientation is legal and correct, which is
        # the same rule the scorer reports, so the gate table cannot disagree with the score.
        figure_ok = all(bool(side.get("legal")) and str(side.get("pred_figure")) == str(side.get("gold_figure")) for side in sides)
        defect_ok = all(bool(side.get("legal")) and str(side.get("pred_defect")) == str(side.get("gold_defect")) for side in sides)
        table[key] = {"gold_defect": str(sides[0].get("gold_defect") or ""), "figure_ok": figure_ok, "defect_ok": defect_ok, "sides": sides}
    return table


def _class_gates(table: Mapping[str, Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    classes: dict[str, Any] = {}
    for name in GATE_DIAGNOSIS_TYPES:
        keys = [key for key, entry in table.items() if entry["gold_defect"] == name]
        detected = sum(1 for key in keys if table[key]["defect_ok"])
        classes[name] = {**_gate(name, _rate(detected, len(keys)), float(thresholds[name]), samples=len(keys)), "detected": detected}
    return {"classes": classes, "passed": all(entry["passed"] for entry in classes.values())}


def _named_correctly(side: Mapping[str, Any]) -> bool:
    """Whether this side named the wrong figure, which is the only accepted answer."""

    return bool(side.get("legal")) and str(side.get("pred_figure")) == str(side.get("gold_figure"))


def _adversarial_gates(rows: Sequence[Mapping[str, Any]], table: Mapping[str, Mapping[str, Any]], thresholds: Mapping[str, float]) -> dict[str, Any]:
    mirrored_pairs = [key for key, entry in table.items() if len(entry["sides"]) > 1]
    swapped_hits = sum(1 for key in mirrored_pairs if all(_named_correctly(side) for side in table[key]["sides"]))
    both_hits = sum(1 for key in mirrored_pairs if table[key]["figure_ok"] and table[key]["defect_ok"])
    in_a = [side for side in rows if _orientation(side) == "gold_in_A"]
    in_b = [side for side in rows if _orientation(side) == "gold_in_B"]
    values = {
        "ab_swapped": (swapped_hits, len(mirrored_pairs)),
        "certified_in_a": (sum(1 for side in in_a if _named_correctly(side)), len(in_a)),
        "certified_in_b": (sum(1 for side in in_b if _named_correctly(side)), len(in_b)),
        "pairwise_both_sides": (both_hits, len(mirrored_pairs)),
    }
    cases: dict[str, Any] = {}
    for case in CHOICE_ADVERSARIAL_CASES:
        case_id = case["case_id"]
        hits, total = values[case_id]
        cases[case_id] = {**_gate(case_id, _rate(hits, total), float(thresholds[case_id]), samples=total), "expectation": case["expectation"], "hits": hits}
    uncovered = [case_id for case_id, entry in cases.items() if not entry["samples"]]
    return {"cases": cases, "passed": all(entry["passed"] for entry in cases.values()), "uncovered": uncovered}


def _chain(report: Mapping[str, Any]) -> list[dict[str, str]]:
    def row(step: str, status: str, detail: str) -> dict[str, str]:
        return {"step": step, "status": status, "detail": detail}

    hidden, classes, adversarial, shadow = report["hidden"], report["class_gates"], report["adversarial"], report["shadow"]
    return [
        row("predict", "passed" if report["predictions"] else "not_run", f"{report['samples']['sides']} sides over {report['samples']['pairs']} pairs"),
        row("hidden", "passed" if hidden["passed"] else "failed", "; ".join(entry["metric"] for entry in hidden["gates"] if not entry["passed"]) or "figure, defect and legality all clear"),
        row("classes", "passed" if classes["passed"] else "failed", "; ".join(name for name, entry in classes["classes"].items() if not entry["passed"]) or "every gating defect is named"),
        row("adversarial", "passed" if adversarial["passed"] else "failed", "; ".join(adversarial["uncovered"]) or "; ".join(name for name, entry in adversarial["cases"].items() if not entry["passed"]) or "no case is passed by accident"),
        row("shadow", "not_run" if not (shadow or {}).get("measured") else ("passed" if shadow["stable"] else "failed"), (shadow or {}).get("reason", "no shadow comparison was run")),
        row("promotion", "ready" if report["promotion"]["approved"] else "blocked", report["promotion"]["advice"]),
    ]


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Choice Judge gate table",
        "",
        f"- Predictions: `{report['predictions']}`",
        f"- Dataset: `{report['dataset'] or 'unnamed'}`",
        f"- Checkpoint: `{report['checkpoint'] or 'base model'}`",
        f"- Promotion: `{report['promotion']['advice']}`",
        "",
        "## Metrics",
        "",
        "| Group | Metric | Value | Threshold | Samples | Passed |",
        "| --- | --- | ---: | ---: | ---: | :---: |",
    ]
    for group, entries in (("hidden", report["hidden"]["gates"]), ("classes", list(report["class_gates"]["classes"].values())), ("adversarial", list(report["adversarial"]["cases"].values()))):
        for entry in entries:
            lines.append(f"| {group} | {entry['metric']} | {entry['value']} | {entry['threshold']} | {entry['samples']} | {entry['passed']} |")
    lines += ["", "## Chain", "", "| Step | Status | Detail |", "| --- | --- | --- |"]
    lines += [f"| {row['step']} | {row['status']} | {row['detail']} |" for row in report["chain"]]
    if report["promotion"]["reasons"]:
        lines += ["", "## Blockers", ""] + [f"- {reason}" for reason in report["promotion"]["reasons"]]
    if report["promotion"]["notes"]:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in report["promotion"]["notes"]]
    if report.get("next_blockers"):
        lines += ["", "## What is furthest from the bar", ""] + [f"- `{entry['metric']}`: {entry['value']} against {entry['threshold']} ({entry['gap']:+.3f}, {entry['samples']} samples)" for entry in report["next_blockers"]]
    return "\n".join(lines) + "\n"


def _shortfalls(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rank the gates that failed by how far short they are, so the next run has a target."""

    entries = [*report["hidden"]["gates"], *report["class_gates"]["classes"].values(), *report["adversarial"]["cases"].values()]
    gaps = [
        {"metric": entry["metric"], "value": entry["value"], "threshold": entry["threshold"], "samples": entry["samples"], "gap": round(float(entry["value"] or 0.0) - float(entry["threshold"]), 4)}
        for entry in entries
        if not entry["passed"] and entry["value"] is not None
    ]
    return sorted(gaps, key=lambda item: item["gap"])


def build_choice_gate_report(
    *,
    predictions: str | Path,
    dataset: str | Path = "",
    checkpoint: str | Path = "",
    model_id: str = "",
    dataset_version: str = "",
    git_commit: str = "",
    thresholds: Mapping[str, float] | None = None,
    report_out: str | Path | None = None,
    strict_gates: bool = False,
) -> dict[str, Any]:
    """Gate a choice checkpoint on one exam's predictions and write the promotion table.

    ``ok`` means the table was computed. ``strict_gates`` additionally makes it follow the
    promotion decision, so a caller can fail a run whose gates did not pass.
    """

    settings = {**CHOICE_GATE_THRESHOLDS, **{str(key): float(value) for key, value in (thresholds or {}).items()}}
    rows = _predictions(predictions)
    table = pair_table(rows)
    legal = sum(1 for row in rows if row.get("legal"))
    figure_hits = sum(1 for entry in table.values() if entry["figure_ok"])
    defect_hits = sum(1 for entry in table.values() if entry["defect_ok"])

    hidden_gates = [
        _gate("figure_accuracy", _rate(figure_hits, len(table)), settings["figure_accuracy"], samples=len(table)),
        _gate("defect_accuracy", _rate(defect_hits, len(table)), settings["defect_accuracy"], samples=len(table)),
        _gate("legal_rate", _rate(legal, len(rows)), settings["legal_rate"], samples=len(rows)),
    ]
    hidden = {"gates": hidden_gates, "passed": all(entry["passed"] for entry in hidden_gates)}
    classes = _class_gates(table, settings)
    adversarial = _adversarial_gates(rows, table, settings)

    # The shadow comparison compares two runs on a shared judgement suite; choice scoring has
    # no bootstrap counterpart, so this gate is reported as unmeasured rather than failed or
    # silently dropped. The promotion decision leaves frozen blocked while it is unmeasured.
    shadow = None if not rows else {"reason": "choice scoring has no bootstrap counterpart; run the shadow gate on a judgement suite to measure it", "measured": False, "stable": False}

    promotion = promotion_decision(
        hidden_test_passed=bool(hidden["passed"]),
        class_gates_passed=bool(classes["passed"]),
        adversarial_passed=bool(adversarial["passed"]),
        shadow_stable=False,
    )

    report: dict[str, Any] = {
        "schema_version": CHOICE_GATE_SCHEMA_VERSION,
        "predictions": str(predictions),
        "dataset": str(dataset or ""),
        "checkpoint": str(checkpoint or ""),
        "thresholds": settings,
        "samples": {"sides": len(rows), "pairs": len(table)},
        "hidden": hidden,
        "class_gates": classes,
        "adversarial": adversarial,
        "shadow": shadow,
        "promotion": promotion,
        "summary": f"Choice Judge gate table for {Path(str(predictions)).name}.",
    }
    report["chain"] = _chain(report)
    report["next_blockers"] = _shortfalls(report)
    measured = bool(rows)
    report["ok"] = (measured and bool(promotion["approved"])) if strict_gates else measured
    manifest = registry_manifest(report, model_id=model_id, dataset_version=dataset_version, checkpoint=checkpoint, git_commit=git_commit)
    report["registry_manifest"] = manifest
    if report_out:
        target = Path(report_out)
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "choice_gate.json", report)
        write_text(target / "choice_gate.md", _markdown(report))
        if manifest.get("model_id"):
            write_json(target / "registry_manifest.json", manifest)
    return report


def registry_manifest(report: Mapping[str, Any], *, model_id: str, dataset_version: str, checkpoint: str | Path, git_commit: str) -> dict[str, Any]:
    """Shape the report into the manifest ``rfs judge register`` validates.

    The roles come from the promotion decision, so a checkpoint that did not clear the
    gates is offered as ``experimental`` and the registry refuses the serving roles for it
    rather than leaving the caller to remember which gates were run.
    """

    promotion = report.get("promotion") or {}
    roles = [str(role) for role in promotion.get("roles") or ["experimental"]]
    return {
        "model_id": str(model_id or ""),
        "dataset_version": str(dataset_version or ""),
        "roles": roles,
        "git_commit": str(git_commit or ""),
        "checkpoint": str(checkpoint or ""),
        "promotion": {
            "approved": bool(promotion.get("approved")),
            "hidden_test_passed": bool(promotion.get("hidden_test_passed")),
            "adversarial_passed": bool(promotion.get("adversarial_passed")),
            "shadow_stable": bool(promotion.get("shadow_stable")),
            "class_gates_passed": bool(promotion.get("class_gates_passed")),
        },
        "choice_gate_schema": CHOICE_GATE_SCHEMA_VERSION,
        "promotion_schema": PROMOTION_SCHEMA_VERSION,
    }


__all__ = [
    "CHOICE_ADVERSARIAL_CASES",
    "CHOICE_GATE_SCHEMA_VERSION",
    "CHOICE_GATE_THRESHOLDS",
    "build_choice_gate_report",
    "pair_table",
    "registry_manifest",
]
