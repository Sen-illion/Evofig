"""Stable Graph Middleware contracts and deterministic identifiers."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


GRAPH_CONTRACT_VERSION = 1
GRAPH_KINDS = {"scientific", "visual", "editable"}
MOTIF_KINDS = {
    "linear",
    "branch",
    "merge",
    "feedback",
    "repeated",
    "multi_input",
    "multi_output",
    "cross_panel",
    "training_inference",
    "unknown",
}


def _records(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _identifier(value: Any) -> str:
    return str(value or "").strip()


def stable_graph_id(namespace: str, *parts: Any) -> str:
    """Return a readable, deterministic ID for one graph-layer object."""

    prefix = re.sub(r"[^a-z0-9_.-]+", "_", str(namespace).strip().casefold()).strip("_.-")
    prefix = prefix or "graph"
    canonical = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}.{digest}"


def normalize_graph_node(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize one node shared by scientific, visual, and editable graphs."""

    source = raw if isinstance(raw, dict) else {}
    return {
        **source,
        "id": _identifier(source.get("id")),
        "kind": _identifier(source.get("kind") or source.get("node_type") or "unknown"),
        "label": _identifier(source.get("label") or source.get("name") or source.get("id")),
        "role": _identifier(source.get("role") or "unknown"),
        "required": bool(source.get("required", source.get("must_show", False))),
        "evidence_ids": _strings(source.get("evidence_ids")),
        "scientific_node_ids": _strings(source.get("scientific_node_ids")),
        "visual_node_ids": _strings(source.get("visual_node_ids")),
        "panel_ids": _strings(source.get("panel_ids")),
        "slot_ids": _strings(source.get("slot_ids")),
        "port_ids": _strings(source.get("port_ids")),
        "ppt_object_ids": _strings(source.get("ppt_object_ids")),
        "properties": source.get("properties") if isinstance(source.get("properties"), dict) else {},
        "provenance": source.get("provenance") if isinstance(source.get("provenance"), dict) else {},
    }


def normalize_graph_edge(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize one directed relation shared by all graph layers."""

    source = raw if isinstance(raw, dict) else {}
    return {
        **source,
        "id": _identifier(source.get("id")),
        "source": _identifier(source.get("source") or source.get("source_id")),
        "target": _identifier(source.get("target") or source.get("target_id")),
        "relation_type": _identifier(source.get("relation_type") or source.get("type") or "related_to"),
        "required": bool(source.get("required", source.get("must_show", False))),
        "evidence_ids": _strings(source.get("evidence_ids")),
        "scientific_edge_ids": _strings(source.get("scientific_edge_ids")),
        "visual_edge_ids": _strings(source.get("visual_edge_ids")),
        "connector_ids": _strings(source.get("connector_ids")),
        "properties": source.get("properties") if isinstance(source.get("properties"), dict) else {},
        "provenance": source.get("provenance") if isinstance(source.get("provenance"), dict) else {},
    }


def normalize_graph(raw: dict[str, Any] | None = None, *, kind: str | None = None) -> dict[str, Any]:
    """Return a stable graph envelope without inventing semantic content."""

    source = raw if isinstance(raw, dict) else {}
    resolved_kind = _identifier(kind or source.get("kind") or "scientific").casefold()
    if resolved_kind not in GRAPH_KINDS:
        resolved_kind = "scientific"
    return {
        "contract": f"rfs.{resolved_kind}_graph",
        "version": int(source.get("version") or GRAPH_CONTRACT_VERSION),
        "graph_id": _identifier(source.get("graph_id")),
        "paper_id": _identifier(source.get("paper_id")),
        "kind": resolved_kind,
        "summary": _identifier(source.get("summary") or f"EvoFig {resolved_kind} graph."),
        "nodes": [normalize_graph_node(item) for item in _records(source.get("nodes"))],
        "edges": [normalize_graph_edge(item) for item in _records(source.get("edges"))],
        "required_node_ids": _strings(source.get("required_node_ids")),
        "required_edge_ids": _strings(source.get("required_edge_ids")),
        "forbidden_labels": _strings(source.get("forbidden_labels")),
        "uncertainties": _strings(source.get("uncertainties")),
        "provenance": source.get("provenance") if isinstance(source.get("provenance"), dict) else {},
    }


def validate_graph(graph: dict[str, Any], *, expected_kind: str | None = None) -> dict[str, Any]:
    """Validate graph identity, endpoints, required objects, and evidence."""

    normalized = normalize_graph(graph)
    errors: list[str] = []
    if expected_kind and normalized["kind"] != expected_kind:
        errors.append(f"graph kind must be {expected_kind}")
    if not normalized["graph_id"]:
        errors.append("graph_id is required")
    if not normalized["paper_id"]:
        errors.append("paper_id is required")
    node_ids = [item["id"] for item in normalized["nodes"]]
    edge_ids = [item["id"] for item in normalized["edges"]]
    if any(not item for item in node_ids):
        errors.append("every graph node requires id")
    if any(not item for item in edge_ids):
        errors.append("every graph edge requires id")
    if len(node_ids) != len(set(node_ids)):
        errors.append("graph node IDs must be unique")
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("graph edge IDs must be unique")
    known_nodes = {item for item in node_ids if item}
    known_edges = {item for item in edge_ids if item}
    for index, edge in enumerate(normalized["edges"]):
        if not edge["source"] or not edge["target"]:
            errors.append(f"edge[{index}] requires source and target")
        elif edge["source"] not in known_nodes or edge["target"] not in known_nodes:
            errors.append(f"edge[{index}] endpoint is not declared: {edge['source']}->{edge['target']}")
    missing_nodes = sorted(set(normalized["required_node_ids"]) - known_nodes)
    missing_edges = sorted(set(normalized["required_edge_ids"]) - known_edges)
    if missing_nodes:
        errors.append(f"required graph node is missing: {missing_nodes[0]}")
    if missing_edges:
        errors.append(f"required graph edge is missing: {missing_edges[0]}")
    if normalized["kind"] == "scientific":
        for item in normalized["nodes"]:
            if (item["required"] or item["id"] in normalized["required_node_ids"]) and not item["evidence_ids"]:
                errors.append(f"required scientific node has no evidence: {item['id']}")
        for item in normalized["edges"]:
            if (item["required"] or item["id"] in normalized["required_edge_ids"]) and not item["evidence_ids"]:
                errors.append(f"required scientific edge has no evidence: {item['id']}")
    return {
        "ok": not errors,
        "errors": errors,
        "kind": normalized["kind"],
        "node_count": len(known_nodes),
        "edge_count": len(known_edges),
        "graph": normalized,
    }


def normalize_graph_bindings(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize cross-layer node and edge bindings."""

    source = raw if isinstance(raw, dict) else {}
    node_bindings = []
    for item in _records(source.get("node_bindings")):
        node_bindings.append({
            **item,
            "binding_id": _identifier(item.get("binding_id") or item.get("id")),
            "scientific_node_id": _identifier(item.get("scientific_node_id")),
            "visual_node_ids": _strings(item.get("visual_node_ids")),
            "panel_ids": _strings(item.get("panel_ids")),
            "slot_ids": _strings(item.get("slot_ids")),
            "port_ids": _strings(item.get("port_ids")),
            "ppt_object_ids": _strings(item.get("ppt_object_ids")),
            "status": _identifier(item.get("status") or "bound"),
            "provenance": item.get("provenance") if isinstance(item.get("provenance"), dict) else {},
        })
    edge_bindings = []
    for item in _records(source.get("edge_bindings")):
        edge_bindings.append({
            **item,
            "binding_id": _identifier(item.get("binding_id") or item.get("id")),
            "scientific_edge_id": _identifier(item.get("scientific_edge_id")),
            "visual_edge_ids": _strings(item.get("visual_edge_ids")),
            "connector_ids": _strings(item.get("connector_ids")),
            "status": _identifier(item.get("status") or "bound"),
            "provenance": item.get("provenance") if isinstance(item.get("provenance"), dict) else {},
        })
    return {
        "contract": "rfs.graph_bindings",
        "version": int(source.get("version") or GRAPH_CONTRACT_VERSION),
        "binding_set_id": _identifier(source.get("binding_set_id")),
        "paper_id": _identifier(source.get("paper_id")),
        "node_bindings": node_bindings,
        "edge_bindings": edge_bindings,
        "provenance": source.get("provenance") if isinstance(source.get("provenance"), dict) else {},
    }


def validate_graph_bindings(
    bindings: dict[str, Any],
    *,
    scientific_graph: dict[str, Any] | None = None,
    visual_graph: dict[str, Any] | None = None,
    editable_graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate binding IDs and optional references into the three layers."""

    normalized = normalize_graph_bindings(bindings)
    errors: list[str] = []
    if not normalized["binding_set_id"]:
        errors.append("binding_set_id is required")
    if not normalized["paper_id"]:
        errors.append("paper_id is required")
    all_bindings = normalized["node_bindings"] + normalized["edge_bindings"]
    binding_ids = [item["binding_id"] for item in all_bindings]
    if any(not item for item in binding_ids):
        errors.append("every graph binding requires binding_id")
    if len(binding_ids) != len(set(binding_ids)):
        errors.append("graph binding IDs must be unique")

    scientific = normalize_graph(scientific_graph, kind="scientific") if scientific_graph is not None else None
    visual = normalize_graph(visual_graph, kind="visual") if visual_graph is not None else None
    editable = normalize_graph(editable_graph, kind="editable") if editable_graph is not None else None
    scientific_nodes = {item["id"] for item in scientific["nodes"]} if scientific else set()
    scientific_edges = {item["id"] for item in scientific["edges"]} if scientific else set()
    visual_nodes = {item["id"] for item in visual["nodes"]} if visual else set()
    visual_edges = {item["id"] for item in visual["edges"]} if visual else set()
    editable_nodes = {item["id"] for item in editable["nodes"]} if editable else set()
    editable_edges = {item["id"] for item in editable["edges"]} if editable else set()

    for index, item in enumerate(normalized["node_bindings"]):
        if not item["scientific_node_id"]:
            errors.append(f"node_binding[{index}] requires scientific_node_id")
        elif scientific is not None and item["scientific_node_id"] not in scientific_nodes:
            errors.append(f"node_binding[{index}] references unknown scientific node: {item['scientific_node_id']}")
        for value in item["visual_node_ids"]:
            if visual is not None and value not in visual_nodes:
                errors.append(f"node_binding[{index}] references unknown visual node: {value}")
        for value in item["ppt_object_ids"]:
            if editable is not None and value not in editable_nodes:
                errors.append(f"node_binding[{index}] references unknown PPT object: {value}")
    for index, item in enumerate(normalized["edge_bindings"]):
        if not item["scientific_edge_id"]:
            errors.append(f"edge_binding[{index}] requires scientific_edge_id")
        elif scientific is not None and item["scientific_edge_id"] not in scientific_edges:
            errors.append(f"edge_binding[{index}] references unknown scientific edge: {item['scientific_edge_id']}")
        for value in item["visual_edge_ids"]:
            if visual is not None and value not in visual_edges:
                errors.append(f"edge_binding[{index}] references unknown visual edge: {value}")
        for value in item["connector_ids"]:
            if editable is not None and value not in editable_edges:
                errors.append(f"edge_binding[{index}] references unknown connector: {value}")
    return {"ok": not errors, "errors": errors, "bindings": normalized}


def normalize_subgraph(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize a dynamically extracted graph motif reference."""

    source = raw if isinstance(raw, dict) else {}
    motif = _identifier(source.get("motif") or "unknown").casefold()
    if motif not in MOTIF_KINDS:
        motif = "unknown"
    return {
        "contract": "rfs.graph_subgraph",
        "version": int(source.get("version") or GRAPH_CONTRACT_VERSION),
        "subgraph_id": _identifier(source.get("subgraph_id")),
        "graph_id": _identifier(source.get("graph_id")),
        "paper_id": _identifier(source.get("paper_id")),
        "graph_kind": _identifier(source.get("graph_kind") or "scientific"),
        "motif": motif,
        "node_ids": _strings(source.get("node_ids")),
        "edge_ids": _strings(source.get("edge_ids")),
        "boundary_node_ids": _strings(source.get("boundary_node_ids")),
        "properties": source.get("properties") if isinstance(source.get("properties"), dict) else {},
        "provenance": source.get("provenance") if isinstance(source.get("provenance"), dict) else {},
    }


def validate_subgraph(subgraph: dict[str, Any], *, graph: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate one subgraph and, when supplied, its source graph references."""

    normalized = normalize_subgraph(subgraph)
    errors: list[str] = []
    for field in ("subgraph_id", "graph_id", "paper_id"):
        if not normalized[field]:
            errors.append(f"{field} is required")
    if not normalized["node_ids"]:
        errors.append("subgraph requires at least one node")
    if len(normalized["node_ids"]) != len(set(normalized["node_ids"])):
        errors.append("subgraph node IDs must be unique")
    if len(normalized["edge_ids"]) != len(set(normalized["edge_ids"])):
        errors.append("subgraph edge IDs must be unique")
    if graph is not None:
        source = normalize_graph(graph)
        known_nodes = {item["id"] for item in source["nodes"]}
        known_edges = {item["id"] for item in source["edges"]}
        missing_nodes = sorted(set(normalized["node_ids"]) - known_nodes)
        missing_edges = sorted(set(normalized["edge_ids"]) - known_edges)
        if normalized["graph_id"] != source["graph_id"]:
            errors.append("subgraph graph_id does not match source graph")
        if missing_nodes:
            errors.append(f"subgraph references unknown node: {missing_nodes[0]}")
        if missing_edges:
            errors.append(f"subgraph references unknown edge: {missing_edges[0]}")
    return {"ok": not errors, "errors": errors, "subgraph": normalized}
