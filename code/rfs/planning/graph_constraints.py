"""Deterministic scientific-graph context for structure planning.

The structure planner has always seen the method contract alone: the scientific
graph and its subgraphs are built next to the plan but were never handed to it,
so a layout could ignore the paper's own topology. This module turns those
graphs into a compact summary the planner can read, and checks a produced design
plan back against that summary. Everything here is deterministic and calls no
provider.

The summary is stated in *contract id* space -- the ``source_contract_id`` a
graph node carries and the ``source_relation_id`` an edge carries -- because the
design plan's ``semantic_source_id`` and connector endpoints are the only
identifiers it is allowed to speak. The graph's own hashed node and edge ids
never reach a prompt, so a change to the id scheme cannot invalidate a plan.
"""

from __future__ import annotations

import json
from typing import Any

from ..analysis.graph import build_scientific_graph, extract_subgraphs


def _contract_id(node: dict[str, Any]) -> str:
    """The paper contract's own id for one graph node, falling back to the node id."""

    properties = node.get("properties") if isinstance(node.get("properties"), dict) else {}
    value = str(properties.get("source_contract_id") or "").strip()
    return value or str(node.get("id") or "").strip()


def _short(value: Any, limit: int = 60) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _degree(edges: list[list[Any]]) -> dict[str, int]:
    degree: dict[str, int] = {}
    for edge in edges:
        for endpoint in (str(edge[0]), str(edge[1])):
            degree[endpoint] = degree.get(endpoint, 0) + 1
    return degree


def graph_context(
    method: dict[str, Any],
    *,
    paper_id: str = "",
    evidence_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarise the scientific graph and its subgraphs in contract-id space.

    ``paper_id`` only reaches the stable-hash node ids, which this summary never
    exposes, so the returned contract ids depend on the method alone. When the
    caller names no paper the graph is built under a fixed placeholder, because a
    subgraph is only valid under a named paper and the planner does not always
    hold one.
    """

    resolved_paper_id = str(paper_id or "").strip() or "planning-graph-context"
    graph, _ground_truth, report = build_scientific_graph(
        method if isinstance(method, dict) else {},
        paper_id=resolved_paper_id,
        evidence_map=evidence_map,
    )
    subgraphs, subgraph_report = extract_subgraphs(graph)
    motif_by_node = {
        str((item.get("properties") or {}).get("center_node_id") or ""): str(item.get("motif") or "unknown")
        for item in subgraphs
        if isinstance(item, dict)
    }
    graph_nodes = [item for item in graph.get("nodes", []) if isinstance(item, dict)]
    contract_by_node = {str(item.get("id")): _contract_id(item) for item in graph_nodes}
    nodes = [
        {
            "id": _contract_id(item),
            "label": str(item.get("label") or ""),
            "role": str(item.get("role") or ""),
            "phase": str((item.get("properties") or {}).get("phase") or ""),
            "required": bool(item.get("required")),
            "motif": motif_by_node.get(str(item.get("id") or ""), "unknown"),
        }
        for item in graph_nodes
    ]
    edges = [
        [
            contract_by_node.get(str(item.get("source")), str(item.get("source"))),
            contract_by_node.get(str(item.get("target")), str(item.get("target"))),
            str(item.get("relation_type") or ""),
            bool(item.get("required")),
        ]
        for item in graph.get("edges", [])
        if isinstance(item, dict)
    ]
    required_node_ids = [item["id"] for item in nodes if item["required"] and item["id"]]
    required_edge_ids = [[str(edge[0]), str(edge[1]), str(edge[2])] for edge in edges if edge[3]]
    motif_histogram = dict(subgraph_report.get("motif_counts") or {})
    errors = [*report.get("errors", []), *subgraph_report.get("errors", [])]
    return {
        "contract": "rfs.planning_graph_context",
        "version": 1,
        "paper_id": str(graph.get("paper_id") or ""),
        "nodes": nodes,
        "edges": edges,
        "motif_histogram": motif_histogram,
        "required": {"node_ids": required_node_ids, "edge_ids": required_edge_ids},
        "counts": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "required_node_count": len(required_node_ids),
            "required_edge_count": len(required_edge_ids),
            "subgraph_count": int(subgraph_report.get("subgraph_count") or 0),
        },
        "errors": errors,
    }


def _node_line(node: dict[str, Any]) -> str:
    return "%s | %s | %s | phase=%s | %s | %s" % (
        node.get("id"),
        _short(node.get("label")),
        node.get("role") or "-",
        node.get("phase") or "-",
        node.get("motif") or "unknown",
        "required" if node.get("required") else "optional",
    )


def _edge_line(edge: list[Any]) -> str:
    return "%s -> %s | %s | %s" % (
        edge[0],
        edge[1],
        edge[2] or "-",
        "required" if len(edge) > 3 and edge[3] else "optional",
    )


def compact_graph_context(context: dict[str, Any], *, max_chars: int = 2400) -> str:
    """Render the summary as a compact, budgeted prompt block.

    Required items are kept first; whatever budget is left goes to the rest in
    descending endpoint degree, so a drop always removes the least connected
    item first. What survives a tight budget is the paper's must-show topology.
    """

    if not isinstance(context, dict) or not context.get("nodes"):
        return ""
    nodes = [item for item in context.get("nodes") or [] if isinstance(item, dict)]
    edges = [item for item in context.get("edges") or [] if isinstance(item, list) and len(item) >= 3]
    degree = _degree(edges)
    required = [item for item in nodes if item.get("required")]
    optional = [item for item in nodes if not item.get("required")]
    optional.sort(key=lambda item: (-degree.get(str(item.get("id")), 0), str(item.get("id"))))
    required_edges = [item for item in edges if len(item) > 3 and item[3]]
    optional_edges = [item for item in edges if not (len(item) > 3 and item[3])]
    optional_edges.sort(
        key=lambda item: (
            -(degree.get(str(item[0]), 0) + degree.get(str(item[1]), 0)),
            str(item[0]),
            str(item[1]),
        )
    )
    counts = context.get("counts") or {}
    header = "nodes=%s edges=%s required_nodes=%s required_edges=%s motifs=%s" % (
        counts.get("node_count"),
        counts.get("edge_count"),
        counts.get("required_node_count"),
        counts.get("required_edge_count"),
        json.dumps(context.get("motif_histogram") or {}, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    )
    ordered = (
        [_node_line(item) for item in required]
        + [_edge_line(item) for item in required_edges]
        + [_node_line(item) for item in optional]
        + [_edge_line(item) for item in optional_edges]
    )
    lines: list[str] = []
    used = len(header)
    for line in ordered:
        if used + len(line) + 1 > max_chars and lines:
            continue
        lines.append(line)
        used += len(line) + 1
    if not lines:
        return ""
    return "\n".join([header, *lines])


def check_design_against_graph(design: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Measure how much of the graph's required topology a design plan carries.

    Coverage is computed in contract-id space: a design node matches through its
    ``semantic_source_id`` and a design connector matches through the contract ids
    of its two endpoints. Matching is by endpoint pair, not relation type, so a
    grounded connector and an ungrounded one count the same.
    """

    plan = design if isinstance(design, dict) else {}
    context = context if isinstance(context, dict) else {}
    design_ids = {
        str(item.get("id")): str(item.get("semantic_source_id") or item.get("id"))
        for item in plan.get("nodes", [])
        if isinstance(item, dict)
    }
    present_nodes = {value for value in design_ids.values() if value}
    present_edges: set[tuple[str, str]] = set()
    for connector in plan.get("connectors", []) or []:
        if not isinstance(connector, dict):
            continue
        source = design_ids.get(str(connector.get("source")), str(connector.get("source")))
        target = design_ids.get(str(connector.get("target")), str(connector.get("target")))
        if source and target:
            present_edges.add((source, target))
    required = context.get("required") if isinstance(context.get("required"), dict) else {}
    required_nodes = [str(item) for item in (required.get("node_ids") or []) if str(item)]
    required_edges = [
        [str(item[0]), str(item[1]), str(item[2]) if len(item) > 2 else ""]
        for item in (required.get("edge_ids") or [])
        if isinstance(item, list) and len(item) >= 2
    ]
    missing_nodes = sorted(item for item in required_nodes if item not in present_nodes)
    missing_edges = sorted(
        [item for item in required_edges if (item[0], item[1]) not in present_edges]
    )
    node_coverage = 1.0 if not required_nodes else round((len(required_nodes) - len(missing_nodes)) / len(required_nodes), 4)
    edge_coverage = 1.0 if not required_edges else round((len(required_edges) - len(missing_edges)) / len(required_edges), 4)
    errors = [f"missing required node: {item}" for item in missing_nodes]
    errors.extend(f"missing required relation: {item[0]}->{item[1]} ({item[2] or '-'})" for item in missing_edges)
    return {
        "ok": not missing_nodes and not missing_edges,
        "missing_required_node_ids": missing_nodes,
        "missing_required_edge_ids": missing_edges,
        "node_coverage": node_coverage,
        "edge_coverage": edge_coverage,
        "required_node_count": len(required_nodes),
        "required_edge_count": len(required_edges),
        "errors": errors,
    }


__all__ = ["check_design_against_graph", "compact_graph_context", "graph_context"]
