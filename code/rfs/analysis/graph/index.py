"""Portable JSONL graph index used for retrieval and validation-case selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ...contracts import normalize_graph, stable_graph_id
from .subgraphs import extract_subgraphs, motif_histogram


def build_graph_index_record(
    graph: dict[str, Any],
    *,
    domain: str = "unknown",
    content_tags: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
    embedding: list[float] | None = None,
    defect_patterns: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build one deterministic, API-independent retrieval record.

    ``defect_patterns`` carries the structural shapes a caller already read off the figure, so
    a Memory condition can be matched on the illness rather than on a node count.
    """

    source = normalize_graph(graph)
    subgraphs, report = extract_subgraphs(source, include_unknown=True)
    tags = sorted({str(item).strip().casefold() for item in content_tags if str(item).strip()})
    shapes = sorted({str(item) for item in (defect_patterns or []) if str(item)})
    return {
        "contract": "rfs.graph_index_record",
        "version": 1,
        "record_id": stable_graph_id("graph_index", source["paper_id"], source["kind"], source["graph_id"]),
        "paper_id": source["paper_id"],
        "graph_id": source["graph_id"],
        "graph_kind": source["kind"],
        "domain": str(domain or "unknown").strip().casefold(),
        "content_tags": tags,
        "features": {
            "node_count": len(source["nodes"]),
            "edge_count": len(source["edges"]),
            "required_node_count": len(source["required_node_ids"]),
            "required_edge_count": len(source["required_edge_ids"]),
            "motif_histogram": motif_histogram(subgraphs),
            "defect_patterns": shapes,
        },
        "subgraphs": subgraphs,
        "embedding": [float(item) for item in embedding] if isinstance(embedding, list) else [],
        "metadata": metadata if isinstance(metadata, dict) else {},
        "index_report": report,
    }


class GraphIndex:
    """Small append-safe JSONL index; no external vector database required."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict) and item.get("record_id"):
                records.append(item)
        return records

    def upsert(self, record: dict[str, Any]) -> None:
        if not record.get("record_id"):
            raise ValueError("graph index record requires record_id")
        by_id = {str(item["record_id"]): item for item in self.records()}
        by_id[str(record["record_id"])] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(json.dumps(by_id[key], ensure_ascii=False, sort_keys=True) + "\n" for key in sorted(by_id))
        self.path.write_text(payload, encoding="utf-8")
