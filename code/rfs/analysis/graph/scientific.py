"""Build evidence-grounded scientific graphs from stored method material.

The content may be the generation-facing method plan or the paper method
inventory the middleware keeps beside the review: both share one shape, and the
``source`` argument only decides which document the graph's provenance names.
"""

from __future__ import annotations

from typing import Any

from ...contracts import (
    normalize_graph,
    normalize_method_figure_content,
    stable_graph_id,
    validate_graph,
    validate_method_figure_content,
)


_NODE_SECTIONS = (
    ("inputs", "input"),
    ("outputs", "output"),
    ("modules", "module"),
    ("stages", "stage"),
    ("innovations", "innovation"),
    ("optional_details", "optional_detail"),
)


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _known_evidence_ids(evidence_map: dict[str, Any] | None) -> set[str] | None:
    if evidence_map is None:
        return None
    known: set[str] = set()
    records = evidence_map.get("evidence") if isinstance(evidence_map, dict) else None
    if isinstance(records, list):
        for item in records:
            if not isinstance(item, dict):
                continue
            for field in ("id", "legacy_id"):
                value = str(item.get(field) or "").strip()
                if value:
                    known.add(value)
    return known


def _phase(item: dict[str, Any]) -> str:
    for field in ("phase", "execution_phase", "lifecycle_phase"):
        value = str(item.get(field) or "").strip().casefold()
        if value:
            return value
    text = " ".join(str(item.get(field) or "") for field in ("label", "role", "name")).casefold()
    if "train" in text or "训练" in text:
        return "training"
    if "infer" in text or "inference" in text or "推理" in text:
        return "inference"
    return ""


def _required(item: dict[str, Any]) -> bool:
    return bool(item.get("must_show", False) or item.get("should_render", False))


def build_scientific_graph(
    method_content: dict[str, Any],
    *,
    paper_id: str,
    evidence_map: dict[str, Any] | None = None,
    source: str = "method_figure_content",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Translate one normalized method document without adding scientific claims.

    Only the facts the document states are copied: an id, a label, a role, its
    evidence and its relations. Anything a generation-side plan carries beyond
    that — a parent pointer, input/output shortcuts — is dropped on purpose:
    nothing reads those fields, and keeping them would smuggle one model's
    judgement into the scientific layer.
    """

    normalized_method = normalize_method_figure_content(method_content)
    method_validation = validate_method_figure_content(normalized_method)
    resolved_paper_id = str(paper_id or "").strip()
    graph_id = stable_graph_id("scientific_graph", resolved_paper_id)
    source_to_scientific: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []

    for section, kind in _NODE_SECTIONS:
        for occurrence, item in enumerate(normalized_method[section]):
            source_id = str(item.get("id") or "").strip()
            node_id = stable_graph_id("snode", resolved_paper_id, section, source_id or occurrence)
            if source_id:
                source_to_scientific[source_id] = node_id
            phase = _phase(item)
            properties = {
                "source_contract_id": source_id,
                "source_section": section,
                "detail_level": str(item.get("detail_level") or ""),
            }
            if phase:
                properties["phase"] = phase
            nodes.append({
                "id": node_id,
                "kind": kind,
                "label": str(item.get("label") or source_id),
                "role": str(item.get("role") or kind),
                "required": _required(item),
                "evidence_ids": _strings(item.get("evidence_ids")),
                "properties": properties,
                "provenance": {
                    "source_contract": source,
                    "source_contract_id": source_id,
                    "source_section": section,
                },
            })

    edges: list[dict[str, Any]] = []
    relation_mapping: dict[str, str] = {}
    for occurrence, relation in enumerate(normalized_method["relations"]):
        source_id = str(relation.get("source") or relation.get("source_id") or "").strip()
        target_id = str(relation.get("target") or relation.get("target_id") or "").strip()
        relation_id = str(relation.get("id") or "").strip()
        relation_type = str(relation.get("type") or relation.get("relation_type") or "related_to").strip()
        edge_id = stable_graph_id(
            "sedge", resolved_paper_id, source_id, target_id, relation_type, relation_id or occurrence
        )
        if relation_id:
            relation_mapping[relation_id] = edge_id
        edges.append({
            "id": edge_id,
            "source": source_to_scientific.get(source_id, source_id),
            "target": source_to_scientific.get(target_id, target_id),
            "relation_type": relation_type,
            "required": _required(relation),
            "evidence_ids": _strings(relation.get("evidence_ids")),
            "properties": {
                "source_relation_id": relation_id,
                "source_node_contract_id": source_id,
                "target_node_contract_id": target_id,
                "label": str(relation.get("label") or ""),
            },
            "provenance": {
                "source_contract": source + ".relations",
                "source_contract_id": relation_id,
                "source_occurrence": occurrence,
            },
        })

    graph = normalize_graph({
        "graph_id": graph_id,
        "paper_id": resolved_paper_id,
        "kind": "scientific",
        "summary": normalized_method["figure_goal"],
        "nodes": nodes,
        "edges": edges,
        "required_node_ids": [item["id"] for item in nodes if item["required"]],
        "required_edge_ids": [item["id"] for item in edges if item["required"]],
        "forbidden_labels": normalized_method["forbidden_inventions"],
        "uncertainties": normalized_method["uncertainties"],
        "provenance": {
            **normalized_method["provenance"],
            "source_contract": "rfs." + source,
        },
    }, kind="scientific")
    graph_validation = validate_graph(graph, expected_kind="scientific")

    known_evidence = _known_evidence_ids(evidence_map)
    evidence_references = {
        item["id"]: list(item["evidence_ids"])
        for item in [*graph["nodes"], *graph["edges"]]
        if item["evidence_ids"]
    }
    referenced_evidence = {value for values in evidence_references.values() for value in values}
    unknown_evidence = sorted(referenced_evidence - known_evidence) if known_evidence is not None else []
    errors = [*method_validation["errors"], *graph_validation["errors"]]
    errors.extend(f"unknown evidence ID: {item}" for item in unknown_evidence)

    ground_truth = {
        "contract": "rfs.scientific_ground_truth",
        "version": 1,
        "paper_id": resolved_paper_id,
        "graph_id": graph_id,
        "required_node_ids": list(graph["required_node_ids"]),
        "required_edge_ids": list(graph["required_edge_ids"]),
        "forbidden_labels": list(graph["forbidden_labels"]),
        "node_evidence": {item["id"]: list(item["evidence_ids"]) for item in graph["nodes"]},
        "edge_evidence": {item["id"]: list(item["evidence_ids"]) for item in graph["edges"]},
    }
    report = {
        "ok": not errors,
        "errors": errors,
        "method_contract_ok": method_validation["ok"],
        "graph_contract_ok": graph_validation["ok"],
        "unknown_evidence_ids": unknown_evidence,
        "source_to_scientific_node_ids": source_to_scientific,
        "source_to_scientific_edge_ids": relation_mapping,
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
    }
    return graph, ground_truth, report
