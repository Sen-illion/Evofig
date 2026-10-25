"""Score records for SciFlow-Bench and FigureBench.

SciFlow's official evaluator is not vendored. Its columns stay null until
that evaluator is connected, and the reason says so. FigureBench stores the
official test and the development set separately. They are not averaged.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

SCIFLOW_FIELDS = ("node_f1", "edge_f1", "text", "clip", "lpips", "flow")
SCIFLOW_NOT_CONNECTED = "official_evaluator_not_connected"

FIGUREBENCH_DESIGN = ("aesthetic_quality", "expressiveness", "polish")
FIGUREBENCH_COMMUNICATION = ("clarity", "logical_flow")
FIGUREBENCH_FIDELITY = ("accuracy", "completeness", "appropriateness")
FIGUREBENCH_SPLITS = ("official_test", "development")


def sciflow_record(*, reason: str = SCIFLOW_NOT_CONNECTED) -> dict[str, Any]:
    """Empty SciFlow columns. Connecting the official evaluator is a later step."""
    report: dict[str, Any] = {
        "summary": "SciFlow-Bench columns. The official evaluator is not connected.",
        "available": False,
        "reason": reason,
    }
    for name in SCIFLOW_FIELDS:
        report[name] = None
    return report


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _group(payload: Mapping[str, Any] | None, names: Sequence[str]) -> float | None:
    if payload is None:
        return None
    values: list[float] = []
    for name in names:
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        values.append(float(value))
    return _mean(values)


def figurebench_split_score(submetrics: Mapping[str, Any] | None) -> dict[str, Any]:
    """Collapse one split's sub-metrics into overall, design, communication, fidelity."""
    design = _group(submetrics, FIGUREBENCH_DESIGN)
    communication = _group(submetrics, FIGUREBENCH_COMMUNICATION)
    fidelity = _group(submetrics, FIGUREBENCH_FIDELITY)
    parts = [value for value in (design, communication, fidelity) if value is not None]
    overall = _mean(parts) if len(parts) == 3 else None
    return {
        "available": overall is not None,
        "overall": overall,
        "design": design,
        "communication": communication,
        "fidelity": fidelity,
    }


def figurebench_record(
    *,
    official_test: Mapping[str, Any] | None = None,
    development: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep the 300-row test and the 3,000-row development set in separate cells."""
    return {
        "summary": "FigureBench scores. The official test and the development set are not averaged.",
        "official_test": figurebench_split_score(official_test),
        "development": figurebench_split_score(development),
        "combined": None,
        "note": "official_test is the 300-row test; development is the 3,000-row set; combined stays null.",
    }
