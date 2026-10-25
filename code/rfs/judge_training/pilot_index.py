"""Deterministic retrieval index, motif index, subgraph locator and probe.

The index stage closes Phase 3. It never calls a provider: it reads the stored
scientific and visual layers of every Pilot paper and writes four version level
artifacts.

graph_index.jsonl
    One rfs.graph_index_record per paper layer, reused from rfs.analysis.graph.
    Subgraph bodies are deliberately not stored: the workflow spec keeps one
    full graph plus its main motifs per paper and extracts subgraphs on demand,
    so a record points at subgraph_index.jsonl instead of copying subgraph
    payloads into the retrieval index.
motif_index.jsonl
    One row per motif with the papers, domains and layers that carry it, which
    is what "search by structure and domain" is answered from.
subgraph_index.jsonl
    One locator row per node centred subgraph: motif, centre, size and the
    stored graph it can be re-extracted from. Still no subgraph body.
retrieval_probe.json
    Probe of the index: for a deterministic sample of papers, the structurally
    closest cases from other papers, with cross domain statistics and guards
    (no self hit, no hidden test record, no provider call).

Every artifact is a pure function of the stored layers plus this module, so two
runs over the same version produce byte identical files.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable

from rfs.analysis.graph import build_graph_index_record, extract_subgraphs, structure_signature, subgraph_edges
from rfs.analysis.graph.retrieval import RECALL_LIMIT, rank_validation_cases_with_recall
from rfs.judge_training.source_snapshot_io import strict_jsonl
from rfs.providers.embeddings import TextEmbedder, resolve_embedder

LEDGER_SCHEMA_VERSION = "pilot-graph-v1"
INDEX_STEP = "index"
INDEX_ARTIFACTS = ("graph_index.jsonl", "motif_index.jsonl", "subgraph_index.jsonl", "retrieval_probe.json")
INDEX_CONTRACT_VERSION = 1
MOTIF_INDEX_CONTRACT = "rfs.graph_motif_index"
SUBGRAPH_INDEX_CONTRACT = "rfs.graph_subgraph_locator"
PROBE_CONTRACT = "rfs.graph_retrieval_probe"
SUBGRAPH_EXTRACTOR = "node_centered_one_hop_v1"
#: Tag prefixes that describe one paper as a whole rather than one subgraph.
GRAPH_SCOPED_TAG_PREFIXES = ("role:", "relation:")
DEFAULT_PROBE_TOP_K = 5
DEFAULT_PROBE_LIMIT = 64
LAYERS = ("scientific", "visual")
LAYER_FILES = {"scientific": "scientific_graph.json", "visual": "visual_graph.json"}
LAYER_KEYS = {"scientific": "scientific_graph", "visual": "visual_graph"}
TAG_ROW_FIELDS = ("figure_type", "structure", "complexity", "panel_count", "has_arrows")
PROBE_SPLITS = ("train", "validation")
HIDDEN_TEST_SPLIT = "hidden_test"


def layer_file(layer: str, directory: str) -> str:
    """Return the version relative path of one stored graph layer."""
    name = LAYER_FILES.get(str(layer), f"{layer}_graph.json")
    folder = str(directory or "").strip() or "unknown-paper"
    return f"graphs/{folder}/{name}"


def graph_for_layer(entry: dict[str, Any] | None, layer: str) -> dict[str, Any] | None:
    """Return one stored layer of a paper, or None when it is absent."""
    value = (entry or {}).get(LAYER_KEYS.get(str(layer), ""))
    return value if isinstance(value, dict) else None


def _split_of(row: dict[str, Any]) -> str:
    return str(row.get("split") or "").strip().casefold()


def _domain_of(row: dict[str, Any]) -> str:
    return str(row.get("domain") or "unknown").strip().casefold() or "unknown"


def paper_tags(row: dict[str, Any]) -> list[str]:
    """Return the tags one Pilot ledger row contributes on its own.

    Only the row's own fields are read, so the same list describes a paper's whole
    graph and any single subgraph of it.
    """

    tags: list[str] = [f"domain:{_domain_of(row)}"]
    for field in TAG_ROW_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, bool):
            tags.append(f"{field}:{'yes' if value else 'no'}")
            continue
        label = str(value).strip().casefold()
        if label:
            tags.append(f"{field}:{label}")
    return tags


def content_tags(row: dict[str, Any], graph: dict[str, Any]) -> list[str]:
    """Return the deterministic content tags used for diversity in retrieval.

    Tags come from the Pilot ledger row plus the stored graph itself, so the
    same rule still works at full D0 scale where no Pilot label exists.
    """

    tags: list[str] = paper_tags(row)
    for node in graph.get("nodes") or []:
        role = str((node or {}).get("role") or "").strip().casefold()
        if role:
            tags.append(f"role:{role}")
    for edge in graph.get("edges") or []:
        relation = str((edge or {}).get("relation_type") or "").strip().casefold()
        if relation:
            tags.append(f"relation:{relation}")
    return tags


def record_text(row: dict[str, Any], graph: dict[str, Any], tags: Iterable[str]) -> str:
    """Return the deterministic text one retrieval record is embedded from.

    The text is the record's own content: its domain and figure tags, the labels of
    its nodes, its relation types and its motifs. Nothing is read from a provider, so
    the same stored graph always produces the same embedding.
    """

    parts: list[str] = [_domain_of(row)]
    for field in TAG_ROW_FIELDS:
        value = row.get(field)
        if value is not None:
            parts.append(str(value))
    parts.extend(tags)
    for node in graph.get("nodes") or []:
        label = (node or {}).get("label") or (node or {}).get("name") or (node or {}).get("id")
        if label:
            parts.append(str(label))
    for edge in graph.get("edges") or []:
        relation = str((edge or {}).get("relation_type") or "").strip()
        if relation:
            parts.append(relation)
    return " | ".join(part for part in parts if str(part).strip())


def _node_label(node: dict[str, Any] | None, fallback: str = "") -> str:
    """Return the human readable label of one node, falling back to its id."""

    item = node or {}
    for key in ("label", "name", "title", "id"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return str(fallback or "").strip()


def _mapping_text(values: dict[str, Any]) -> str:
    """Render a small count map as one deterministic token."""

    return ",".join(f"{key}={values[key]}" for key in sorted(values))


def subgraph_text(locator: dict[str, Any], subgraph: dict[str, Any], graph: dict[str, Any] | None = None, tags: Iterable[str] = ()) -> str:
    """Return the deterministic text one subgraph locator is embedded from.

    The text is the subgraph's own structure - motif, size, degree shape, whether it
    closes a loop - plus the labels of its nodes and the types of its edges, so two
    subgraphs with the same motif and size still separate when their content differs.
    Nothing is read from a provider: the same stored graph always produces the same
    text and therefore the same vector.
    """

    signature = structure_signature(subgraph, graph)
    nodes = {str(item.get("id") or ""): item for item in ((graph or {}).get("nodes") or []) if isinstance(item, dict)}
    center = str(locator.get("center_node_id") or signature.get("center_node_id") or "")
    body = subgraph.get("properties") if isinstance(subgraph.get("properties"), dict) else {}
    parts: list[str] = [
        f"layer:{locator.get('graph_kind') or body.get('layer') or 'unknown'}",
        f"motif:{locator.get('motif') or signature['motif']}",
        f"nodes:{signature['node_count']}",
        f"edges:{signature['edge_count']}",
        f"degrees:{_mapping_text(signature['degree_histogram'])}",
        f"cycle:{'yes' if signature['has_cycle'] else 'no'}",
        f"center_degrees:{signature['center_in_degree']}:{signature['center_out_degree']}",
        f"center_role:{str(body.get('center_role') or '').strip().casefold()}",
        f"center_kind:{str(body.get('center_kind') or '').strip().casefold()}",
        f"center_label:{_node_label(nodes.get(center), center)}",
    ]
    neighbors = sorted(
        _node_label(nodes.get(str(node_id)), str(node_id))
        for node_id in (subgraph.get("node_ids") or [])
        if str(node_id) != center
    )
    parts.extend(f"node:{label}" for label in neighbors if label)
    relations = sorted({str(edge.get("relation_type") or "").strip().casefold() for edge in subgraph_edges(subgraph, graph)} - {""})
    parts.extend(f"relation:{relation}" for relation in relations)
    parts.extend(str(tag) for tag in tags)
    return " | ".join(part for part in parts if str(part).strip())


def index_record_for(graph: dict[str, Any], *, row: dict[str, Any], layer: str, directory: str, embedder: TextEmbedder | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one retrieval record, its embedding and the subgraph locator rows."""
    paper_id = str(row.get("paper_id") or graph.get("paper_id") or "")
    source_graph = layer_file(layer, directory)
    encoder = embedder or resolve_embedder()
    tags = content_tags(row, graph)
    record = build_graph_index_record(
        graph,
        domain=_domain_of(row),
        content_tags=tags,
        embedding=encoder.encode([record_text(row, graph, tags)])[0],
        metadata={
            "embedding": encoder.fingerprint(),
            "split": _split_of(row),
            "domain": _domain_of(row),
            "layer": str(layer),
            "figure_type": row.get("figure_type"),
            "structure": row.get("structure"),
            "complexity": row.get("complexity"),
            "panel_count": row.get("panel_count"),
            "source_graph": source_graph,
            "subgraph_payloads_materialized": False,
        },
    )
    subgraphs, subgraph_report = extract_subgraphs(graph, include_unknown=True)
    fingerprint = encoder.fingerprint()
    row_tags = paper_tags(row)
    locators: list[dict[str, Any]] = []
    for item in subgraphs:
        locator = {
            "contract": SUBGRAPH_INDEX_CONTRACT,
            "version": INDEX_CONTRACT_VERSION,
            "subgraph_id": str(item.get("subgraph_id") or ""),
            "graph_id": str(item.get("graph_id") or ""),
            "paper_id": paper_id,
            "split": _split_of(row),
            "domain": _domain_of(row),
            "graph_kind": str(item.get("graph_kind") or layer),
            "motif": str(item.get("motif") or "unknown"),
            "center_node_id": str((item.get("properties") or {}).get("center_node_id") or ""),
            "node_count": int((item.get("properties") or {}).get("node_count") or 0),
            "edge_count": int((item.get("properties") or {}).get("edge_count") or 0),
            "required": bool((item.get("properties") or {}).get("required")),
            "extractor": str((item.get("provenance") or {}).get("extractor") or SUBGRAPH_EXTRACTOR),
            "source_graph": source_graph,
        }
        # The locator row is the only place a subgraph vector is stored, so it records
        # the encoder that produced it; an unembedded locator is an error, not a
        # silent fall back to structure only.
        locator["embedding"] = encoder.encode([subgraph_text(locator, item, graph, row_tags)])[0]
        locator["embedding_fingerprint"] = dict(fingerprint)
        locators.append(locator)
    stored_report = record.get("index_report") if isinstance(record.get("index_report"), dict) else {}
    record["subgraphs"] = []
    record["subgraph_locator"] = {
        "index": "subgraph_index.jsonl",
        "materialized": False,
        "count": len(locators),
        "motif_counts": dict(subgraph_report.get("motif_counts") or {}),
        "source_graph": source_graph,
        "extractor": SUBGRAPH_EXTRACTOR,
        "note": "subgraphs are re-extracted from the stored full graph on demand",
    }
    record["index_report"] = {
        **stored_report,
        "ok": bool(stored_report.get("ok", True)) and bool(subgraph_report.get("ok", True)),
        "errors": list(stored_report.get("errors") or []) + list(subgraph_report.get("errors") or []),
        "subgraph_count": len(locators),
        "motif_counts": dict(subgraph_report.get("motif_counts") or {}),
        "subgraph_payloads_materialized": False,
    }
    return record, locators


def build_motif_rows(records: Iterable[dict[str, Any]], locators: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate every motif into one searchable row."""
    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        paper_id = str(record.get("paper_id") or "")
        domain = str(record.get("domain") or "unknown")
        kind = str(record.get("graph_kind") or "unknown")
        histogram = (record.get("features") or {}).get("motif_histogram") or {}
        for motif, count in histogram.items():
            bucket = buckets.setdefault(str(motif), {"subgraph_count": 0, "papers": set(), "domains": {}, "graph_kinds": {}, "required_subgraphs": 0})
            bucket["subgraph_count"] += int(count or 0)
            bucket["papers"].add(paper_id)
            bucket["domains"][domain] = bucket["domains"].get(domain, 0) + int(count or 0)
            bucket["graph_kinds"][kind] = bucket["graph_kinds"].get(kind, 0) + int(count or 0)
    for item in locators:
        bucket = buckets.get(str(item.get("motif") or "unknown"))
        if bucket is not None and item.get("required"):
            bucket["required_subgraphs"] += 1
    rows = []
    for motif in sorted(buckets):
        bucket = buckets[motif]
        rows.append({
            "contract": MOTIF_INDEX_CONTRACT,
            "version": INDEX_CONTRACT_VERSION,
            "motif": motif,
            "subgraph_count": bucket["subgraph_count"],
            "required_subgraphs": bucket["required_subgraphs"],
            "paper_count": len(bucket["papers"]),
            "papers": sorted(bucket["papers"]),
            "domains": dict(sorted(bucket["domains"].items())),
            "graph_kinds": dict(sorted(bucket["graph_kinds"].items())),
        })
    return rows


def sampled_queries(queries: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Pick a deterministic, evenly spread sample of the probe queries."""
    if not limit or limit <= 0 or len(queries) <= limit:
        return list(queries)
    step = len(queries) / float(limit)
    return [queries[int(index * step)] for index in range(int(limit))]


def build_retrieval_probe(records: list[dict[str, Any]], rows: list[dict[str, Any]], *, top_k: int = DEFAULT_PROBE_TOP_K, limit: int = DEFAULT_PROBE_LIMIT, graph_id: str = "", embedder: TextEmbedder | None = None, recall_limit: int = RECALL_LIMIT) -> dict[str, Any]:
    """Recall by embedding, then rank structurally for a deterministic sample of papers."""
    by_paper = {str(row.get("paper_id") or ""): row for row in rows}
    encoder = embedder or resolve_embedder()
    queries = [record for record in records if str(record.get("graph_kind") or "") == "scientific"]
    picked = sampled_queries(queries, int(limit or 0))
    query_rows: list[dict[str, Any]] = []
    for record in picked:
        paper_id = str(record.get("paper_id") or "")
        hits = rank_validation_cases_with_recall(record, records, top_k=int(top_k), recall_limit=int(recall_limit), exclude_paper_ids=(paper_id,))
        hit_rows = [
            {
                "paper_id": str((item.get("record") or {}).get("paper_id") or ""),
                "domain": str((item.get("record") or {}).get("domain") or "unknown"),
                "graph_kind": str((item.get("record") or {}).get("graph_kind") or "unknown"),
                "score": item.get("score"),
                "structure_similarity": item.get("structure_similarity"),
                "motif_similarity": item.get("motif_similarity"),
                "content_diversity": item.get("content_diversity"),
                "cross_domain": bool(item.get("cross_domain")),
                "selection_reason": item.get("selection_reason"),
            }
            for item in hits
        ]
        query_rows.append({
            "paper_id": paper_id,
            "split": str((by_paper.get(paper_id) or {}).get("split") or ""),
            "domain": str(record.get("domain") or "unknown"),
            "node_count": int((record.get("features") or {}).get("node_count") or 0),
            "edge_count": int((record.get("features") or {}).get("edge_count") or 0),
            "motif_counts": dict((record.get("features") or {}).get("motif_histogram") or {}),
            "candidate_records": len(records),
            "recall_limit": max(int(recall_limit), int(top_k)),
            "hit_count": len(hit_rows),
            "cross_domain_hits": sum(1 for item in hit_rows if item["cross_domain"]),
            "top_score": hit_rows[0]["score"] if hit_rows else 0.0,
            "top_paper_id": hit_rows[0]["paper_id"] if hit_rows else "",
            "hits": hit_rows,
        })
    scores = [float(row["top_score"]) for row in query_rows if row["hit_count"]]
    hit_total = sum(int(row["hit_count"]) for row in query_rows)
    cross_total = sum(int(row["cross_domain_hits"]) for row in query_rows)
    domains: dict[str, int] = {}
    layers: dict[str, int] = {}
    hit_domains: dict[str, int] = {}
    for row in query_rows:
        domains[row["domain"]] = domains.get(row["domain"], 0) + 1
        for hit in row["hits"]:
            layers[hit["graph_kind"]] = layers.get(hit["graph_kind"], 0) + 1
            hit_domains[hit["domain"]] = hit_domains.get(hit["domain"], 0) + 1
    split_counts: dict[str, int] = {}
    for record in records:
        split = str((record.get("metadata") or {}).get("split") or "")
        split_counts[split] = split_counts.get(split, 0) + 1
    unexpected_splits = sorted(split for split in split_counts if split not in PROBE_SPLITS + (HIDDEN_TEST_SPLIT,))
    without_motifs = sorted(
        str(record.get("paper_id") or "")
        for record in queries
        if not ((record.get("features") or {}).get("motif_histogram") or {})
    )
    return {
        "contract": PROBE_CONTRACT,
        "version": INDEX_CONTRACT_VERSION,
        "graph_id": str(graph_id or ""),
        "top_k": int(top_k),
        "probe_limit": int(limit or 0),
        "queries": query_rows,
        "embedding": encoder.fingerprint(),
        "recall": {"limit": max(int(recall_limit), int(top_k)), "strategy": "vector_recall_then_structure_rerank"},
        "summary": {
            "probe_papers": len(query_rows),
            "available_queries": len(queries),
            "candidate_records": len(records),
            "queries_with_hits": len(scores),
            "queries_without_hits": [row["paper_id"] for row in query_rows if not row["hit_count"]],
            "hits": hit_total,
            "cross_domain_hits": cross_total,
            "cross_domain_hit_rate": round(cross_total / hit_total, 4) if hit_total else 0.0,
            "mean_top_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
            "median_top_score": round(statistics.median(scores), 6) if scores else 0.0,
            "distinct_papers_retrieved": len({hit["paper_id"] for row in query_rows for hit in row["hits"]}),
            "retrieved_layer_counts": dict(sorted(layers.items())),
            "retrieved_domain_counts": dict(sorted(hit_domains.items())),
            "queries_by_domain": dict(sorted(domains.items())),
            "motifs_covered": sorted({motif for row in query_rows for motif in row["motif_counts"]}),
            "queries_with_motifs": len(query_rows) - len(without_motifs),
        },
        "guards": {
            "provider_calls": 0,
            "self_hits": sum(1 for row in query_rows for hit in row["hits"] if hit["paper_id"] == row["paper_id"]),
            "hidden_test_records": int(split_counts.get(HIDDEN_TEST_SPLIT, 0)),
            "hidden_test_hits": sum(
                1 for row in query_rows for hit in row["hits"]
                if str((by_paper.get(hit["paper_id"]) or {}).get("split") or "").casefold() == HIDDEN_TEST_SPLIT
            ),
            "splits": dict(sorted(split_counts.items())),
            "unexpected_splits": unexpected_splits,
            "records_without_motifs": without_motifs,
        },
    }


def index_report(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Small audit block reused by coverage, validation and the manifest."""
    records = list((payload or {}).get("records") or [])
    locators = list((payload or {}).get("subgraphs") or [])
    probe = (payload or {}).get("probe") or {}
    layers: dict[str, int] = {}
    for record in records:
        layer = str(record.get("graph_kind") or "unknown")
        layers[layer] = layers.get(layer, 0) + 1
    without_embedding = sum(1 for record in records if not (record.get("embedding") or []))
    subgraphs_without_embedding = sum(1 for item in locators if not (item.get("embedding") or []))
    errors = []
    if without_embedding:
        errors.append(f"{without_embedding} retrieval record(s) carry no stored embedding; rebuild the index instead of ranking them structurally")
    if subgraphs_without_embedding:
        errors.append(f"{subgraphs_without_embedding} subgraph locator(s) carry no stored embedding; rebuild the index instead of pairing them structurally")
    return {
        "papers": len({str(record.get("paper_id") or "") for record in records}),
        "records": len(records),
        "layers": dict(sorted(layers.items())),
        "motifs": len(list((payload or {}).get("motifs") or [])),
        "subgraph_locations": len(locators),
        "probe_queries": len(list((probe or {}).get("queries") or [])),
        "probe_guards": dict((probe or {}).get("guards") or {}),
        "embedding": dict((probe or {}).get("embedding") or {}),
        "recall": dict((probe or {}).get("recall") or {}),
        "records_without_embedding": without_embedding,
        "subgraphs_without_embedding": subgraphs_without_embedding,
        "provider_calls": 0,
        "errors": errors,
    }


def paper_index_report(ledger_row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one paper's index audit from its ledger row.

    The audit is derived instead of stored a second time, so a resumed run stays
    byte identical to a fresh one.
    """
    error = ledger_row.get("error")
    return {
        "paper_id": str(ledger_row.get("paper_id") or ""),
        "ok": str(ledger_row.get("status") or "") == "ok",
        "record_count": int(ledger_row.get("record_count") or 0),
        "record_ids": list(ledger_row.get("record_ids") or []),
        "node_count": int(ledger_row.get("node_count") or 0),
        "edge_count": int(ledger_row.get("edge_count") or 0),
        "motif_count": int(ledger_row.get("motif_count") or 0),
        "subgraph_count": int(ledger_row.get("subgraph_count") or 0),
        "provider_calls": 0,
        "errors": [str(error)] if error else [],
        "warnings": list(ledger_row.get("warnings") or []),
    }


def index_all(rows: list[dict[str, Any]], *, graphs: dict[str, dict[str, Any]], visuals: dict[str, dict[str, Any]], graph_id: str = "", directories: dict[str, str] | None = None, probe_top_k: int = DEFAULT_PROBE_TOP_K, probe_limit: int = DEFAULT_PROBE_LIMIT, embedder: TextEmbedder | None = None, recall_limit: int = RECALL_LIMIT) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Index every paper that has both stored layers. No provider call."""
    encoder = embedder or resolve_embedder()
    folder_by_paper = {str(key): str(value) for key, value in (directories or {}).items()}
    records: list[dict[str, Any]] = []
    locators: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for row in rows:
        started = time.monotonic()
        paper_id = str(row.get("paper_id") or "")
        directory = folder_by_paper.get(paper_id) or paper_id
        stores = {"scientific": graphs or {}, "visual": visuals or {}}
        layer_graphs = {layer: graph_for_layer(stores[layer].get(paper_id), layer) for layer in LAYERS}
        missing = [layer for layer in LAYERS if layer_graphs[layer] is None]
        paper_records: list[dict[str, Any]] = []
        paper_locators: list[dict[str, Any]] = []
        if not missing:
            for layer in LAYERS:
                record, layer_locators = index_record_for(layer_graphs[layer], row=row, layer=layer, directory=directory, embedder=encoder)
                paper_records.append(record)
                paper_locators.extend(layer_locators)
        records.extend(paper_records)
        locators.extend(paper_locators)
        report = {
            "paper_id": paper_id,
            "ok": not missing,
            "errors": [f"no stored {layer} layer for this paper" for layer in missing],
            "warnings": [],
            "record_ids": [str(item.get("record_id") or "") for item in paper_records],
            "record_count": len(paper_records),
            "node_count": sum(int((item.get("features") or {}).get("node_count") or 0) for item in paper_records),
            "edge_count": sum(int((item.get("features") or {}).get("edge_count") or 0) for item in paper_records),
            "motif_count": len({str(item.get("motif") or "unknown") for item in paper_locators}),
            "subgraph_count": len(paper_locators),
            "provider_calls": 0,
        }
        ledger.append({
            "schema_version": LEDGER_SCHEMA_VERSION,
            "paper_id": paper_id,
            "split": str(row.get("split") or ""),
            "domain": str(row.get("domain") or ""),
            "step": INDEX_STEP,
            "status": "ok" if not missing else "failed",
            "attempts": 0,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "image_count": 0,
            "image_bytes": 0,
            "prompt_chars": None,
            "estimated_tokens": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "token_source": "not_called",
            "failure_categories": [],
            "error": None if not missing else report["errors"][0],
            "record_ids": list(report["record_ids"]),
            "record_count": report["record_count"],
            "node_count": report["node_count"],
            "edge_count": report["edge_count"],
            "motif_count": report["motif_count"],
            "subgraph_count": report["subgraph_count"],
            "artifact_sha256": {},
            "warnings": [] if not missing else list(report["errors"]),
        })
    probe = build_retrieval_probe(records, rows, top_k=probe_top_k, limit=probe_limit, graph_id=graph_id, embedder=encoder, recall_limit=recall_limit)
    payload: dict[str, Any] = {
        "graph_id": str(graph_id or ""),
        "records": sorted(records, key=lambda item: str(item.get("record_id") or "")),
        "motifs": build_motif_rows(records, locators),
        "subgraphs": sorted(locators, key=lambda item: str(item.get("subgraph_id") or "")),
        "probe": probe,
        "report": {},
    }
    payload["report"] = index_report(payload)
    return payload, ledger


def index_coverage(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Summarize how much of the version the index covers."""
    records = list((payload or {}).get("records") or [])
    locators = list((payload or {}).get("subgraphs") or [])
    motifs = list((payload or {}).get("motifs") or [])
    probe = (payload or {}).get("probe") or {}
    summary = (probe or {}).get("summary") or {}
    kinds: dict[str, set[str]] = {}
    layers: dict[str, int] = {}
    domains: dict[str, int] = {}
    splits: dict[str, int] = {}
    for record in records:
        paper_id = str(record.get("paper_id") or "")
        layer = str(record.get("graph_kind") or "unknown")
        kinds.setdefault(paper_id, set()).add(layer)
        layers[layer] = layers.get(layer, 0) + 1
        domain = str(record.get("domain") or "unknown")
        domains[domain] = domains.get(domain, 0) + 1
        split = str((record.get("metadata") or {}).get("split") or "")
        splits[split] = splits.get(split, 0) + 1
    return {
        "papers": len(kinds),
        "records": len(records),
        "layers": dict(sorted(layers.items())),
        "domains": dict(sorted(domains.items())),
        "splits": dict(sorted(splits.items())),
        "papers_missing_a_layer": sorted(paper_id for paper_id, seen in kinds.items() if set(LAYERS) - seen),
        "motifs": len(motifs),
        "subgraph_locations": len(locators),
        "subgraph_embeddings": sum(1 for item in locators if item.get("embedding")),
        "hidden_test_records": int(((probe or {}).get("guards") or {}).get("hidden_test_records") or 0),
        "probe": {
            "queries": int(summary.get("probe_papers") or 0),
            "queries_with_hits": int(summary.get("queries_with_hits") or 0),
            "cross_domain_hit_rate": summary.get("cross_domain_hit_rate"),
            "mean_top_score": summary.get("mean_top_score"),
            "distinct_papers_retrieved": summary.get("distinct_papers_retrieved"),
        },
    }


def load_index_state(directory: str | Path) -> dict[str, Any] | None:
    """Read a stored index payload, or None when the version has none yet."""
    target = Path(directory)
    index_path = target / "graph_index.jsonl"
    probe_path = target / "retrieval_probe.json"
    if not index_path.is_file() or not probe_path.is_file():
        return None
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    motif_path = target / "motif_index.jsonl"
    subgraph_path = target / "subgraph_index.jsonl"
    payload: dict[str, Any] = {
        "graph_id": str((probe or {}).get("graph_id") or ""),
        "records": strict_jsonl(index_path),
        "motifs": strict_jsonl(motif_path) if motif_path.is_file() else [],
        "subgraphs": strict_jsonl(subgraph_path) if subgraph_path.is_file() else [],
        "probe": probe if isinstance(probe, dict) else {},
        "report": {},
    }
    payload["report"] = index_report(payload)
    return payload


__all__ = [
    "DEFAULT_PROBE_LIMIT",
    "DEFAULT_PROBE_TOP_K",
    "INDEX_ARTIFACTS",
    "INDEX_STEP",
    "LAYERS",
    "LEDGER_SCHEMA_VERSION",
    "MOTIF_INDEX_CONTRACT",
    "PROBE_CONTRACT",
    "SUBGRAPH_INDEX_CONTRACT",
    "build_motif_rows",
    "build_retrieval_probe",
    "content_tags",
    "graph_for_layer",
    "index_all",
    "index_coverage",
    "index_record_for",
    "index_report",
    "record_text",
    "layer_file",
    "load_index_state",
    "paper_index_report",
    "paper_tags",
    "sampled_queries",
    "subgraph_text",
]
