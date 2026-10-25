"""Extract reusable one-hop structural subgraphs from Graph Middleware layers."""

from __future__ import annotations

from typing import Any

from ...contracts import normalize_graph, normalize_subgraph, stable_graph_id, validate_subgraph
from .motifs import classify_node_motif


def extract_subgraphs(graph: dict[str, Any], *, include_unknown: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return deterministic node-centered subgraphs with motif metadata."""

    source = normalize_graph(graph)
    nodes_by_id = {str(item.get("id")): item for item in source["nodes"]}
    subgraphs: list[dict[str, Any]] = []
    errors: list[str] = []
    for node_id in sorted(nodes_by_id):
        incident = [
            item for item in source["edges"]
            if str(item.get("source")) == node_id or str(item.get("target")) == node_id
        ]
        neighbor_ids = sorted({
            endpoint
            for item in incident
            for endpoint in (str(item.get("source") or ""), str(item.get("target") or ""))
            if endpoint
        })
        motif = classify_node_motif(node_id, source["edges"], node=nodes_by_id[node_id], nodes_by_id=nodes_by_id)
        if motif == "unknown" and not include_unknown:
            continue
        edge_ids = sorted(str(item.get("id")) for item in incident)
        boundary = sorted(value for value in neighbor_ids if value != node_id)
        subgraph = normalize_subgraph({
            "subgraph_id": stable_graph_id("subgraph", source["graph_id"], node_id, motif, edge_ids),
            "graph_id": source["graph_id"],
            "paper_id": source["paper_id"],
            "graph_kind": source["kind"],
            "motif": motif,
            "node_ids": neighbor_ids or [node_id],
            "edge_ids": edge_ids,
            "boundary_node_ids": boundary,
            "properties": {
                "center_node_id": node_id,
                "node_count": len(neighbor_ids or [node_id]),
                "edge_count": len(edge_ids),
                "center_role": nodes_by_id[node_id].get("role"),
                "center_kind": nodes_by_id[node_id].get("kind"),
                "required": bool(nodes_by_id[node_id].get("required")),
            },
            "provenance": {"extractor": "node_centered_one_hop_v1"},
        })
        validation = validate_subgraph(subgraph, graph=source)
        if validation["ok"]:
            subgraphs.append(subgraph)
        else:
            errors.extend(validation["errors"])
    return subgraphs, {
        "ok": not errors,
        "errors": errors,
        "graph_id": source["graph_id"],
        "subgraph_count": len(subgraphs),
        "motif_counts": motif_histogram(subgraphs),
    }


def motif_histogram(subgraphs: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in subgraphs:
        motif = str(item.get("motif") or "unknown")
        result[motif] = result.get(motif, 0) + 1
    return dict(sorted(result.items()))


def subgraph_edges(subgraph: dict[str, Any], graph: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Return the edges of one subgraph, read from its own body or from its graph.

    A locator row is not materialized, so most callers only have the edge ids and the
    graph the subgraph can be re-extracted from. Both shapes are accepted here so the
    signature below cannot depend on which one the caller happens to hold.
    """

    body = subgraph.get("edges") if isinstance(subgraph.get("edges"), list) else None
    if body:
        return [dict(item) for item in body if isinstance(item, dict)]
    if not isinstance(graph, dict):
        return []
    wanted = {str(item) for item in (subgraph.get("edge_ids") or [])}
    found = [item for item in (graph.get("edges") or []) if isinstance(item, dict) and str(item.get("id") or "") in wanted]
    return sorted(found, key=lambda item: str(item.get("id") or ""))


def _has_cycle(node_ids: list[str], edges: list[dict[str, Any]]) -> bool:
    """Return whether the subgraph's own edges close a loop over its own nodes."""

    parent = {node_id: node_id for node_id in node_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for edge in edges:
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        if source not in parent or target not in parent:
            continue
        left, right = find(source), find(target)
        if left == right:
            return True
        parent[left] = right
    return False


def structure_signature(subgraph: dict[str, Any], graph: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the structural identity of one node centered subgraph.

    Two subgraphs are the same structure when their motif and their node and edge
    counts agree. The degree histogram, the cycle flag and the center's own degrees are
    reported next to that verdict, because the pair builder has to name *how* two
    subgraphs differ and not only that they do.
    """

    properties = subgraph.get("properties") if isinstance(subgraph.get("properties"), dict) else {}
    node_ids = [str(item) for item in (subgraph.get("node_ids") or [])]
    center = str(properties.get("center_node_id") or "")
    edges = subgraph_edges(subgraph, graph)
    in_degree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    out_degree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source, target = str(edge.get("source") or ""), str(edge.get("target") or "")
        if source in out_degree:
            out_degree[source] += 1
        if target in in_degree:
            in_degree[target] += 1
    histogram: dict[str, int] = {}
    for node_id in node_ids:
        key = f"{in_degree[node_id]}:{out_degree[node_id]}"
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "motif": str(subgraph.get("motif") or "unknown"),
        "node_count": int(properties.get("node_count") or len(node_ids)),
        "edge_count": int(properties.get("edge_count") or len(edges)),
        "center_node_id": center,
        "center_in_degree": in_degree.get(center, 0),
        "center_out_degree": out_degree.get(center, 0),
        "degree_histogram": dict(sorted(histogram.items())),
        "has_cycle": _has_cycle(node_ids, edges),
        "required": bool(properties.get("required")),
        "required_edges": sum(1 for edge in edges if edge.get("required")),
    }


def same_structure(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether two :func:`structure_signature` values are the same structure.

    The rule is deliberately the strict one: identical motif plus identical node and
    edge counts. A looser rule would let a four node chain stand in for a three node
    fan, which is exactly the confusion the Judge is trained against.
    """

    return (
        str(left.get("motif") or "") == str(right.get("motif") or "")
        and int(left.get("node_count") or 0) == int(right.get("node_count") or 0)
        and int(left.get("edge_count") or 0) == int(right.get("edge_count") or 0)
    )