"""Deterministic alignment checks across scientific, visual, and editable graphs."""

from __future__ import annotations

from typing import Any

from ..contracts import validate_graph, validate_graph_bindings


def _bound_ids(bindings: dict[str, Any], collection: str, key: str) -> set[str]:
    return {
        str(item.get(key) or "")
        for item in bindings.get(collection, [])
        if isinstance(item, dict) and item.get("status") == "bound"
    }


def align_graph_layers(
    scientific_graph: dict[str, Any],
    visual_graph: dict[str, Any],
    editable_graph: dict[str, Any],
    bindings: dict[str, Any],
) -> dict[str, Any]:
    """Check required semantic coverage and cross-layer object references."""

    scientific_validation = validate_graph(scientific_graph, expected_kind="scientific")
    visual_validation = validate_graph(visual_graph, expected_kind="visual")
    editable_validation = validate_graph(editable_graph, expected_kind="editable")
    binding_validation = validate_graph_bindings(
        bindings,
        scientific_graph=scientific_graph,
        visual_graph=visual_graph,
        editable_graph=editable_graph,
    )
    required_nodes = set(scientific_graph.get("required_node_ids") or [])
    required_edges = set(scientific_graph.get("required_edge_ids") or [])
    bound_nodes = _bound_ids(bindings, "node_bindings", "scientific_node_id")
    bound_edges = _bound_ids(bindings, "edge_bindings", "scientific_edge_id")
    missing_nodes = sorted(required_nodes - bound_nodes)
    missing_edges = sorted(required_edges - bound_edges)

    visible_labels = [
        str(item.get("label") or "").strip().casefold()
        for item in editable_graph.get("nodes", [])
        if isinstance(item, dict) and str(item.get("label") or "").strip()
    ]
    forbidden_hits = []
    for forbidden in scientific_graph.get("forbidden_labels", []):
        needle = str(forbidden or "").strip().casefold()
        if needle and any(needle == label or needle in label for label in visible_labels):
            forbidden_hits.append(str(forbidden))

    errors = [
        *scientific_validation["errors"],
        *visual_validation["errors"],
        *editable_validation["errors"],
        *binding_validation["errors"],
    ]
    errors.extend(f"required scientific node is not bound through editable output: {item}" for item in missing_nodes)
    errors.extend(f"required scientific edge is not bound through editable output: {item}" for item in missing_edges)
    errors.extend(f"forbidden scientific label appears in editable output: {item}" for item in forbidden_hits)
    node_denominator = max(1, len(required_nodes))
    edge_denominator = max(1, len(required_edges))
    return {
        "contract": "rfs.graph_alignment_report",
        "version": 1,
        "ok": not errors,
        "errors": errors,
        "graph_ids": {
            "scientific": scientific_graph.get("graph_id"),
            "visual": visual_graph.get("graph_id"),
            "editable": editable_graph.get("graph_id"),
        },
        "required_node_count": len(required_nodes),
        "required_edge_count": len(required_edges),
        "bound_required_node_count": len(required_nodes & bound_nodes),
        "bound_required_edge_count": len(required_edges & bound_edges),
        "node_coverage": 1.0 if not required_nodes else round(len(required_nodes & bound_nodes) / node_denominator, 4),
        "edge_coverage": 1.0 if not required_edges else round(len(required_edges & bound_edges) / edge_denominator, 4),
        "missing_required_node_ids": missing_nodes,
        "missing_required_edge_ids": missing_edges,
        "forbidden_label_hits": forbidden_hits,
        "layer_validation": {
            "scientific": scientific_validation["ok"],
            "visual": visual_validation["ok"],
            "editable": editable_validation["ok"],
            "bindings": binding_validation["ok"],
        },
    }
