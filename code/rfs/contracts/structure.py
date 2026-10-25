"""Stable contracts for paper-grounded global figure structure planning."""

from __future__ import annotations

from typing import Any


STRUCTURE_CONTRACT_VERSION = 1


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalize_global_structure_plan(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a stable, JSON-safe global structure plan envelope."""

    source = raw if isinstance(raw, dict) else {}
    panels = _as_list(source.get("panels"))
    groups = _as_list(source.get("groups"))
    nodes = _as_list(source.get("nodes"))
    relations = _as_list(source.get("relations"))
    return {
        "contract": "rfs.global_structure_plan",
        "version": int(source.get("version") or STRUCTURE_CONTRACT_VERSION),
        "summary": str(source.get("summary") or "Paper-grounded global figure structure plan."),
        "topology": str(source.get("topology") or "unknown"),
        "reading_order": [str(item) for item in _as_list(source.get("reading_order")) if str(item).strip()],
        "panels": [item for item in panels if isinstance(item, dict)],
        "groups": [item for item in groups if isinstance(item, dict)],
        "nodes": [item for item in nodes if isinstance(item, dict)],
        "relations": [item for item in relations if isinstance(item, dict)],
        "layout_policy": source.get("layout_policy") if isinstance(source.get("layout_policy"), dict) else {},
        "decomposability": source.get("decomposability") if isinstance(source.get("decomposability"), dict) else {},
        "uncertainties": [str(item) for item in _as_list(source.get("uncertainties")) if str(item).strip()],
        "provenance": source.get("provenance") if isinstance(source.get("provenance"), dict) else {},
    }


def validate_global_structure_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate IDs and relation endpoints without mutating the plan."""

    normalized = normalize_global_structure_plan(plan)
    errors: list[str] = []
    node_ids = [str(item.get("id") or "").strip() for item in normalized["nodes"]]
    panel_ids = [str(item.get("id") or "").strip() for item in normalized["panels"]]
    group_ids = [str(item.get("id") or "").strip() for item in normalized["groups"]]
    if len(node_ids) != len(set(node_ids)):
        errors.append("global structure node IDs must be unique")
    if len(panel_ids) != len(set(panel_ids)):
        errors.append("global structure panel IDs must be unique")
    if len(group_ids) != len(set(group_ids)):
        errors.append("global structure group IDs must be unique")
    known_nodes = {item for item in node_ids if item}
    for index, relation in enumerate(normalized["relations"]):
        source = str(relation.get("source") or relation.get("source_id") or "").strip()
        target = str(relation.get("target") or relation.get("target_id") or "").strip()
        if not source or not target:
            errors.append(f"relation[{index}] requires source and target")
        elif known_nodes and (source not in known_nodes or target not in known_nodes):
            errors.append(f"relation[{index}] endpoint is not a declared node: {source}->{target}")
    return {"ok": not errors, "errors": errors, "node_count": len(known_nodes), "relation_count": len(normalized["relations"])}


DECOMPOSABILITY_CONTRACT = "rfs.decomposability_report"
DECOMPOSABILITY_VERSION = 1
DECOMPOSABILITY_METRICS = (
    "unbound_scientific_nodes",
    "slot_boundary_overflows",
    "orphan_nodes",
    "cross_panel_connectors",
    "min_panel_occupancy",
    "empty_region_ratio",
    "max_node_slot_fanout",
)
DECOMPOSABILITY_STATUSES = {"decomposable", "soft_violations", "hard_violations"}
DECOMPOSABILITY_VIOLATION_KINDS = {"hard", "soft"}
DECOMPOSABILITY_MIN_PANEL_OCCUPANCY = 0.45
DECOMPOSABILITY_MAX_CROSS_PANEL_CONNECTORS = 2
DECOMPOSABILITY_MAX_NODE_SLOT_FANOUT = 3
DECOMPOSABILITY_MIN_PANEL_AREA = 0.02


def normalize_decomposability_report(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a stable envelope for one measured decomposability report.

    The global structure plan only *declares* that a figure stays decomposable into
    slot boundaries and cross-slot interfaces. This envelope carries the measurement
    of that declaration against a compiled candidate, so two rounds of one paper can
    be compared and a repair can be rolled back on the same numbers.
    """

    source = raw if isinstance(raw, dict) else {}
    metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
    thresholds = source.get("thresholds") if isinstance(source.get("thresholds"), dict) else {}
    violations: list[dict[str, Any]] = []
    for item in _as_list(source.get("violations")):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip()
        violations.append({
            "metric": str(item.get("metric") or ""),
            "value": item.get("value"),
            "threshold": item.get("threshold"),
            "kind": kind if kind in DECOMPOSABILITY_VIOLATION_KINDS else "soft",
            "reason": str(item.get("reason") or ""),
            "object_ids": [str(value) for value in _as_list(item.get("object_ids")) if str(value).strip()],
        })
    status = str(source.get("status") or "decomposable").strip()
    round_value = str(source.get("round") or "")
    return {
        "contract": DECOMPOSABILITY_CONTRACT,
        "version": int(source.get("version") or DECOMPOSABILITY_VERSION),
        "round": int(round_value) if round_value.lstrip("-").isdigit() else None,
        "status": status if status in DECOMPOSABILITY_STATUSES else "decomposable",
        "metrics": {name: metrics.get(name) for name in DECOMPOSABILITY_METRICS},
        "thresholds": dict(thresholds),
        "violations": violations,
        "repeat": source.get("repeat") if isinstance(source.get("repeat"), dict) else {},
        "provenance": source.get("provenance") if isinstance(source.get("provenance"), dict) else {},
    }


def validate_decomposability_report(report: dict[str, Any]) -> dict[str, Any]:
    """Check that a report carries every measurement the decomposition policy reads."""

    normalized = normalize_decomposability_report(report)
    errors: list[str] = []
    for name in DECOMPOSABILITY_METRICS:
        value = normalized["metrics"].get(name)
        if value is None:
            errors.append(f"decomposability metric {name} is missing")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"decomposability metric {name} must be a number")
    for index, violation in enumerate(normalized["violations"]):
        if not violation["metric"]:
            errors.append(f"violation[{index}] requires a metric")
        elif violation["metric"] not in DECOMPOSABILITY_METRICS:
            errors.append(f"violation[{index}] names an unknown metric: {violation['metric']}")
    return {
        "ok": not errors,
        "errors": errors,
        "metric_count": len(DECOMPOSABILITY_METRICS),
        "violation_count": len(normalized["violations"]),
    }
