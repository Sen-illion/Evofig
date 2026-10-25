"""Build editable-object graphs and cross-layer bindings from figure programs."""

from __future__ import annotations

from typing import Any

from ..contracts import (
    normalize_graph,
    normalize_graph_bindings,
    stable_graph_id,
    validate_graph,
    validate_graph_bindings,
)


_OBJECT_COLLECTIONS = (
    ("panels", "panel"),
    ("cards", "card"),
    ("semantic_nodes", "shape"),
    ("slots", "slot"),
    ("labels", "label"),
    ("groups", "group"),
)


def _visual_maps(visual_graph: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    nodes = {
        str(item.get("properties", {}).get("source_object_id") or ""): item
        for item in visual_graph.get("nodes", []) if isinstance(item, dict)
    }
    edges = {
        str(item.get("properties", {}).get("source_object_id") or ""): item
        for item in visual_graph.get("edges", []) if isinstance(item, dict)
    }
    return nodes, edges


def build_editable_graph(
    figure_program: dict[str, Any],
    *,
    visual_graph: dict[str, Any],
    scientific_graph: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Represent native PPT objects and produce scientific-to-editable bindings."""

    paper_id = str(scientific_graph.get("paper_id") or visual_graph.get("paper_id") or "").strip()
    graph_id = stable_graph_id("editable_graph", paper_id)
    visual_nodes, visual_edges = _visual_maps(visual_graph)
    editable_by_object_id: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []

    collections = list(_OBJECT_COLLECTIONS)
    text_program = figure_program.get("text_program") if isinstance(figure_program.get("text_program"), dict) else {}
    if isinstance(text_program.get("items"), list):
        collections.append(("text_program.items", "text"))

    for collection, kind in collections:
        values = text_program.get("items", []) if collection == "text_program.items" else figure_program.get(collection, [])
        if not isinstance(values, list):
            continue
        for occurrence, item in enumerate(values):
            if not isinstance(item, dict):
                continue
            object_id = str(item.get("id") or item.get("slot_id") or "").strip()
            if not object_id:
                object_id = f"{kind}_{occurrence + 1}"
            editable_id = stable_graph_id("pobject", paper_id, collection, object_id)
            editable_by_object_id[object_id] = editable_id
            target_id = str(item.get("target_id") or item.get("node_id") or "").strip()
            visual = visual_nodes.get(object_id) or visual_nodes.get(target_id) or {}
            visual_id = str(visual.get("id") or "")
            scientific_ids = [str(value) for value in visual.get("scientific_node_ids", []) if str(value).strip()]
            nodes.append({
                "id": editable_id,
                "kind": kind,
                "label": str(item.get("label") or item.get("title") or item.get("text") or object_id),
                "role": str(item.get("role") or item.get("semantic_role") or kind),
                "required": bool(visual.get("required", False)),
                "evidence_ids": item.get("evidence_ids") or [],
                "scientific_node_ids": scientific_ids,
                "visual_node_ids": [visual_id] if visual_id else [],
                "panel_ids": [str(item.get("panel_id"))] if item.get("panel_id") else [],
                "slot_ids": [object_id] if kind == "slot" else [],
                "ppt_object_ids": [editable_id],
                "properties": {
                    "source_object_id": object_id,
                    "source_collection": collection,
                    "target_object_id": target_id,
                    "bbox_percent": item.get("bbox_percent"),
                    "editable_in": item.get("editable_in") or "pptx",
                },
                "provenance": {"source_contract": f"figure_program.{collection}", "source_object_id": object_id},
            })

    edges: list[dict[str, Any]] = []
    for occurrence, item in enumerate(figure_program.get("arrows", [])):
        if not isinstance(item, dict):
            continue
        object_id = str(item.get("id") or f"arrow_{occurrence + 1}")
        visual = visual_edges.get(object_id) or {}
        visual_id = str(visual.get("id") or "")
        scientific_ids = [str(value) for value in visual.get("scientific_edge_ids", []) if str(value).strip()]
        source_object = str(item.get("source") or item.get("source_id") or "")
        target_object = str(item.get("target") or item.get("target_id") or "")
        edge_id = stable_graph_id("pconnector", paper_id, object_id)
        edges.append({
            "id": edge_id,
            "source": editable_by_object_id.get(source_object, source_object),
            "target": editable_by_object_id.get(target_object, target_object),
            "relation_type": str(item.get("type") or item.get("semantic_role") or "related_to"),
            "required": bool(visual.get("required", False)),
            "evidence_ids": item.get("evidence_ids") or [],
            "scientific_edge_ids": scientific_ids,
            "visual_edge_ids": [visual_id] if visual_id else [],
            "connector_ids": [edge_id],
            "properties": {
                "source_object_id": object_id,
                "path_percent": item.get("path_percent") or [],
                "editable_in": item.get("editable_in") or "pptx",
            },
            "provenance": {"source_contract": "figure_program.arrows", "source_object_id": object_id},
        })

    graph = normalize_graph({
        "graph_id": graph_id,
        "paper_id": paper_id,
        "kind": "editable",
        "summary": "Native editable PowerPoint object graph.",
        "nodes": nodes,
        "edges": edges,
        "required_node_ids": [item["id"] for item in nodes if item.get("required")],
        "required_edge_ids": [item["id"] for item in edges if item.get("required")],
        "provenance": {"source_contract": "figure_program", "visual_graph_id": visual_graph.get("graph_id")},
    }, kind="editable")
    graph_validation = validate_graph(graph, expected_kind="editable")
    bindings = _build_bindings(scientific_graph, visual_graph, graph)
    binding_validation = validate_graph_bindings(
        bindings,
        scientific_graph=scientific_graph,
        visual_graph=visual_graph,
        editable_graph=graph,
    )
    errors = [*graph_validation["errors"], *binding_validation["errors"]]
    return graph, bindings, {
        "ok": not errors,
        "errors": errors,
        "object_count": len(nodes),
        "connector_count": len(edges),
        "source_to_editable_ids": editable_by_object_id,
    }


def _build_bindings(
    scientific_graph: dict[str, Any], visual_graph: dict[str, Any], editable_graph: dict[str, Any]
) -> dict[str, Any]:
    node_bindings = []
    for scientific in scientific_graph.get("nodes", []):
        scientific_id = str(scientific.get("id") or "")
        visual_ids = [
            str(item.get("id")) for item in visual_graph.get("nodes", [])
            if scientific_id in item.get("scientific_node_ids", [])
        ]
        editable_ids = [
            str(item.get("id")) for item in editable_graph.get("nodes", [])
            if scientific_id in item.get("scientific_node_ids", [])
        ]
        node_bindings.append({
            "binding_id": stable_graph_id("node_binding", scientific_graph.get("paper_id"), scientific_id),
            "scientific_node_id": scientific_id,
            "visual_node_ids": visual_ids,
            "panel_ids": sorted({value for item in editable_graph.get("nodes", []) if scientific_id in item.get("scientific_node_ids", []) for value in item.get("panel_ids", [])}),
            "slot_ids": sorted({value for item in editable_graph.get("nodes", []) if scientific_id in item.get("scientific_node_ids", []) for value in item.get("slot_ids", [])}),
            "ppt_object_ids": editable_ids,
            "status": "bound" if visual_ids and editable_ids else "unbound",
        })
    edge_bindings = []
    for scientific in scientific_graph.get("edges", []):
        scientific_id = str(scientific.get("id") or "")
        visual_ids = [
            str(item.get("id")) for item in visual_graph.get("edges", [])
            if scientific_id in item.get("scientific_edge_ids", [])
        ]
        connector_ids = [
            str(item.get("id")) for item in editable_graph.get("edges", [])
            if scientific_id in item.get("scientific_edge_ids", [])
        ]
        edge_bindings.append({
            "binding_id": stable_graph_id("edge_binding", scientific_graph.get("paper_id"), scientific_id),
            "scientific_edge_id": scientific_id,
            "visual_edge_ids": visual_ids,
            "connector_ids": connector_ids,
            "status": "bound" if visual_ids and connector_ids else "unbound",
        })
    return normalize_graph_bindings({
        "binding_set_id": stable_graph_id("graph_bindings", scientific_graph.get("paper_id")),
        "paper_id": scientific_graph.get("paper_id"),
        "node_bindings": node_bindings,
        "edge_bindings": edge_bindings,
        "provenance": {
            "scientific_graph_id": scientific_graph.get("graph_id"),
            "visual_graph_id": visual_graph.get("graph_id"),
            "editable_graph_id": editable_graph.get("graph_id"),
        },
    })
