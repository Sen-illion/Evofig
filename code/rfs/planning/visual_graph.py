"""Build the visual topology graph from the global design plan."""

from __future__ import annotations

from typing import Any
import re

from ..contracts import normalize_design_plan, normalize_graph, stable_graph_id, validate_graph


def _scientific_maps(scientific_graph: dict[str, Any]) -> tuple[dict[str, str], dict[tuple[str, str, str], str]]:
    source_nodes = {
        str(item.get("properties", {}).get("source_contract_id") or ""): str(item.get("id") or "")
        for item in scientific_graph.get("nodes", [])
        if isinstance(item, dict) and item.get("id")
    }
    source_nodes = {key: value for key, value in source_nodes.items() if key}
    source_edges: dict[tuple[str, str, str], str] = {}
    for item in scientific_graph.get("edges", []):
        if not isinstance(item, dict):
            continue
        properties = item.get("properties") if isinstance(item.get("properties"), dict) else {}
        key = (
            str(properties.get("source_node_contract_id") or ""),
            str(properties.get("target_node_contract_id") or ""),
            str(item.get("relation_type") or "related_to"),
        )
        source_edges[key] = str(item.get("id") or "")
    return source_nodes, source_edges


def _normalized_label(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _panel_contract_aliases(panel_id: str) -> list[str]:
    """Names a panel id may share with a scientific ``source_contract_id``.

    Planners often prefix the contract id (``panel_stage_tasks`` for ``stage_tasks``).
    The prefix is structural, not a paper-specific spelling.
    """

    aliases = [panel_id]
    if panel_id.startswith("panel_") and panel_id[6:]:
        aliases.append(panel_id[6:])
    return aliases


def _panel_scientific_bindings(panel: dict[str, Any], scientific_graph: dict[str, Any], source_nodes: dict[str, str]) -> list[str]:
    """Bind explicit panel IDs plus distinctive required stage/innovation titles."""
    panel_id = str(panel.get("id") or "").strip()
    bindings = [source_nodes[alias] for alias in _panel_contract_aliases(panel_id) if alias in source_nodes]
    panel_label = f" {_normalized_label(panel.get('title') or panel.get('label'))} "
    for item in scientific_graph.get("nodes", []):
        if not isinstance(item, dict):
            continue
        if not item.get("required") or str(item.get("kind") or "") not in {"stage", "innovation"}:
            continue
        label = _normalized_label(item.get("label"))
        if not label:
            continue
        compact = label.replace(" ", "")
        exact = panel_label.strip() == label
        # Short labels may still bind when they are the whole panel title. Substring
        # matching stays gated so a four-letter token cannot hitch onto a longer title.
        contained = len(compact) >= 6 and f" {label} " in panel_label
        if exact or contained:
            bindings.append(str(item.get("id") or ""))
    return list(dict.fromkeys(value for value in bindings if value))


def build_visual_graph(
    design_plan: dict[str, Any],
    *,
    scientific_graph: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind planned panels, nodes, and connectors to scientific objects."""

    plan = normalize_design_plan(design_plan)
    paper_id = str(scientific_graph.get("paper_id") or "").strip()
    graph_id = stable_graph_id("visual_graph", paper_id)
    scientific_nodes, scientific_edges = _scientific_maps(scientific_graph)
    visual_by_plan_id: dict[str, str] = {}
    semantic_by_plan_id: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []

    for panel in plan["panels"]:
        panel_id = str(panel.get("id") or "").strip()
        scientific_ids = _panel_scientific_bindings(panel, scientific_graph, scientific_nodes)
        visual_id = stable_graph_id("vpanel", paper_id, panel_id)
        visual_by_plan_id[panel_id] = visual_id
        nodes.append({
            "id": visual_id,
            "kind": "panel",
            "label": str(panel.get("title") or panel_id),
            "role": str(panel.get("role") or "panel"),
            "required": bool(set(scientific_ids).intersection(scientific_graph.get("required_node_ids", []))),
            "scientific_node_ids": scientific_ids,
            "panel_ids": [panel_id] if panel_id else [],
            "properties": {"source_object_id": panel_id, "bbox_percent": panel.get("bbox_percent")},
            "provenance": {"source_contract": "rfs.design_plan.panels", "source_object_id": panel_id},
        })

    for occurrence, item in enumerate(plan["nodes"]):
        plan_id = str(item.get("id") or "").strip()
        semantic_id = str(item.get("semantic_source_id") or plan_id).strip()
        scientific_id = scientific_nodes.get(semantic_id, "")
        visual_id = stable_graph_id("vnode", paper_id, plan_id or occurrence)
        visual_by_plan_id[plan_id] = visual_id
        semantic_by_plan_id[plan_id] = semantic_id
        nodes.append({
            "id": visual_id,
            "kind": "visual_node",
            "label": str(item.get("label") or plan_id),
            "role": str(item.get("role") or "module"),
            "required": bool(scientific_id and scientific_id in scientific_graph.get("required_node_ids", [])),
            "evidence_ids": item.get("evidence_ids") or [],
            "scientific_node_ids": [scientific_id] if scientific_id else [],
            "panel_ids": [str(item.get("panel_id"))] if item.get("panel_id") else [],
            "port_ids": [str(value) for value in item.get("port_ids", []) if str(value).strip()] if isinstance(item.get("port_ids"), list) else [],
            "properties": {
                "source_object_id": plan_id,
                "semantic_source_id": semantic_id,
                "bbox_percent": item.get("bbox_percent"),
                "display_policy": item.get("display_policy"),
            },
            "provenance": {"source_contract": "rfs.design_plan.nodes", "source_object_id": plan_id},
        })

    edges: list[dict[str, Any]] = []
    unknown_semantic_node_ids = sorted({
        value for value in semantic_by_plan_id.values() if value and value not in scientific_nodes
    })
    unbound_edge_ids: list[str] = []
    for occurrence, item in enumerate(plan["connectors"]):
        plan_id = str(item.get("id") or "").strip()
        source_plan_id = str(item.get("source") or item.get("source_id") or "").strip()
        target_plan_id = str(item.get("target") or item.get("target_id") or "").strip()
        relation_type = str(item.get("type") or item.get("relation_type") or "related_to")
        source_semantic = semantic_by_plan_id.get(source_plan_id, source_plan_id)
        target_semantic = semantic_by_plan_id.get(target_plan_id, target_plan_id)
        scientific_edge_id = scientific_edges.get((source_semantic, target_semantic, relation_type), "")
        edge_id = stable_graph_id("vedge", paper_id, plan_id or occurrence, source_plan_id, target_plan_id)
        if not scientific_edge_id:
            unbound_edge_ids.append(edge_id)
        edges.append({
            "id": edge_id,
            "source": visual_by_plan_id.get(source_plan_id, source_plan_id),
            "target": visual_by_plan_id.get(target_plan_id, target_plan_id),
            "relation_type": relation_type,
            "required": bool(scientific_edge_id and scientific_edge_id in scientific_graph.get("required_edge_ids", [])),
            "evidence_ids": item.get("evidence_ids") or [],
            "scientific_edge_ids": [scientific_edge_id] if scientific_edge_id else [],
            "properties": {
                "source_object_id": plan_id,
                "source_port_id": item.get("source_port_id"),
                "target_port_id": item.get("target_port_id"),
                "path_percent": item.get("path_percent") or [],
            },
            "provenance": {"source_contract": "rfs.design_plan.connectors", "source_object_id": plan_id},
        })

    graph = normalize_graph({
        "graph_id": graph_id,
        "paper_id": paper_id,
        "kind": "visual",
        "summary": "Visual topology planned for the editable scientific figure.",
        "nodes": nodes,
        "edges": edges,
        "required_node_ids": [item["id"] for item in nodes if item.get("required")],
        "required_edge_ids": [item["id"] for item in edges if item.get("required")],
        "forbidden_labels": scientific_graph.get("forbidden_labels") or [],
        "uncertainties": scientific_graph.get("uncertainties") or [],
        "provenance": {"source_contract": "rfs.design_plan", "scientific_graph_id": scientific_graph.get("graph_id")},
    }, kind="visual")
    validation = validate_graph(graph, expected_kind="visual")
    errors = list(validation["errors"])
    errors.extend(f"visual node references unknown scientific source: {item}" for item in unknown_semantic_node_ids)
    return graph, {
        "ok": not errors,
        "errors": errors,
        "unknown_semantic_node_ids": unknown_semantic_node_ids,
        "unbound_visual_edge_ids": unbound_edge_ids,
        "plan_to_visual_ids": visual_by_plan_id,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def resolve_design_required_edge_ids(
    design_plan: dict[str, Any],
    scientific_graph: dict[str, Any],
) -> dict[str, Any]:
    """Map the relations the design stage decided to draw onto scientific edges."""

    plan = normalize_design_plan(design_plan)
    semantic_by_plan_id = {
        str(item.get("id")): str(item.get("semantic_source_id") or item.get("id") or "")
        for item in plan["nodes"]
        if isinstance(item, dict) and item.get("id")
    }
    by_relation_id: dict[str, str] = {}
    by_pair: dict[tuple[str, str, str], str] = {}
    by_pair_any: dict[tuple[str, str], list[str]] = {}
    for item in scientific_graph.get("edges", []):
        if not isinstance(item, dict) or not item.get("id"):
            continue
        properties = item.get("properties") if isinstance(item.get("properties"), dict) else {}
        edge_id = str(item.get("id"))
        relation_id = str(properties.get("source_relation_id") or "").strip()
        if relation_id:
            by_relation_id[relation_id] = edge_id
        source_id = str(properties.get("source_node_contract_id") or "").strip()
        target_id = str(properties.get("target_node_contract_id") or "").strip()
        relation_type = str(item.get("relation_type") or "related_to")
        by_pair.setdefault((source_id, target_id, relation_type), edge_id)
        by_pair_any.setdefault((source_id, target_id), []).append(edge_id)

    required: list[str] = []
    matched: list[str] = []
    unmatched: list[str] = []
    match_sources: dict[str, str] = {}
    for connector in plan["connectors"]:
        connector_id = str(connector.get("id") or "")
        relation_id = str(connector.get("paper_relation_id") or "").strip()
        edge_id = by_relation_id.get(relation_id, "") if relation_id else ""
        match_source = "paper_relation_id"
        if not edge_id:
            source_semantic = semantic_by_plan_id.get(str(connector.get("source") or ""), str(connector.get("source") or ""))
            target_semantic = semantic_by_plan_id.get(str(connector.get("target") or ""), str(connector.get("target") or ""))
            relation_type = str(connector.get("type") or connector.get("relation_type") or "related_to")
            edge_id = by_pair.get((source_semantic, target_semantic, relation_type), "")
            match_source = "source_node_contract_pair"
            if not edge_id:
                candidates = by_pair_any.get((source_semantic, target_semantic)) or []
                edge_id = candidates[0] if len(candidates) == 1 else ""
        if edge_id:
            key = connector_id or edge_id
            matched.append(key)
            match_sources[key] = match_source
            if edge_id not in required:
                required.append(edge_id)
        else:
            unmatched.append(connector_id)
    return {
        "contract": "rfs.design_plan_required_edges",
        "version": 1,
        "required_edge_ids": required,
        "matched_connector_ids": matched,
        "unmatched_connector_ids": unmatched,
        "match_sources": match_sources,
    }
