"""Grounding Gate: bind scientific objects to editable output before treating a figure as grounded.

The gate is an instrument, not a substitute Judge. VLM advisory and acceptance roles still
speak after it runs; its findings are merged into their answers. Heuristic judges have no
model to ask, so they still return the gate as the whole reply.
"""

from __future__ import annotations

from typing import Any


def evaluate_grounding_gate(alignment_report: dict[str, Any]) -> dict[str, Any]:
    """Convert graph alignment evidence into named findings the Judge still answers."""

    failures = list(alignment_report.get("errors") or [])
    checks = {
        "graph_contracts_valid": all((alignment_report.get("layer_validation") or {}).values()),
        "required_nodes_grounded": not alignment_report.get("missing_required_node_ids"),
        "required_edges_grounded": not alignment_report.get("missing_required_edge_ids"),
        "forbidden_labels_absent": not alignment_report.get("forbidden_label_hits"),
    }
    for name, passed in checks.items():
        if not passed and not any(name in item for item in failures):
            failures.append(f"grounding check failed: {name}")
    passed = bool(alignment_report.get("ok")) and all(checks.values())
    return {
        "contract": "rfs.grounding_gate",
        "version": 1,
        "status": "pass" if passed else "blocked",
        "ok": passed,
        "checks": checks,
        "blocking_issues": failures,
        "node_coverage": float(alignment_report.get("node_coverage") or 0.0),
        "edge_coverage": float(alignment_report.get("edge_coverage") or 0.0),
        "alignment_graph_ids": alignment_report.get("graph_ids") or {},
        "policy": "findings_merged_into_judge",
    }
