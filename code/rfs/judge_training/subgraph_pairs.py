"""Build RAG subgraph preference examples: which candidate is the same structure.

This is the third deterministic preference source. The middleware publishes one locator
row per node centred subgraph next to the graph it can be re-extracted from; the builder
takes one subgraph as the structural query, then chooses two neighbours of it out of the
other papers of the same split - the one the strict structure rule certifies and the
closest one it refuses. The pair to judge is therefore "which of these two subgraphs is
the same structure as the query", and the answer is decided by the graph, never by eye.

Nothing here reads a provider. The strict rule is motif plus node and edge count, so a
chosen subgraph really is the same structure and not merely a plausible one. The refused
candidate is the nearest neighbour that breaks the rule, which is where the signal is:
it is what a Judge that matched on wording instead of on structure would pick. The
difference is named in the vocabulary the Judge already speaks, through
:func:`rfs.judge_training.judge_schema.defect_label`, so a structure mismatch is labelled
exactly like the controlled defect of the same kind and no diagnosis type is invented.

Splits stay apart by construction: candidates come from the query's own split, hidden
test never appears, and no subgraph is ever paired with its own paper. The report
re-derives those three properties from the written rows instead of asserting them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..analysis.graph.retrieval import vector_recall_subgraphs
from ..analysis.graph.subgraphs import extract_subgraphs, same_structure, structure_signature
from .judge_schema import defect_label
from .pilot_index import subgraph_text
from .preference_pairs import _write_jsonl, mirror_judgement
from .preference_sources import MiddlewareSource, SourceUnavailable
from .subgraph_render import render_subgraph


SUBGRAPH_PAIR_CONTRACT = "rfs.judge_rag_subgraph_pair"
SUBGRAPH_PAIR_VERSION = 1
DATASET_VERSION = "judge-dataset-v003"
TASK_TYPE = "pairwise_preference"
PROVENANCE_SOURCE = "rag_subgraph"
DEFECT_SOURCE = "structure_rule"
CERTIFIED_CONFIDENCE = 1.0
STRICT_STRUCTURE_RULE = "same motif with identical node and edge counts"
MIN_NODE_COUNT = 2
DEFAULT_RECALL_LIMIT = 64
HIDDEN_TEST_SPLIT = "hidden_test"
TRAINABLE_SPLITS = ("train", "validation")
IMAGE_DIR = "subgraphs"
EXAMPLES_NAME = "rag_subgraph_examples.jsonl"
REPORT_NAME = "rag_subgraph_pairs.report.json"
SKIPPED_SAMPLE = 50
#: A query is refused for exactly one of these reasons, and every refusal is recorded.
EXCLUDED_REASONS = (
    "query_has_no_structure",
    "query_split_has_no_candidate",
    "query_has_no_strict_match",
    "query_has_no_mismatch_candidate",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _sha256(path: str | Path) -> str:
    target = Path(path)
    if not target.is_file():
        return ""
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = _text(value) or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _short_id(subgraph_id: str, *, length: int = 0) -> str:
    """Return the stable hash part of a subgraph id, which is what an example id needs."""

    text = _text(subgraph_id)
    value = text.rsplit(".", 1)[-1] or text
    return value[: int(length)] if length else value


def mismatch_between(query: dict[str, Any], other: dict[str, Any]) -> tuple[str, str]:
    """Name how one candidate differs from the query, in the Judge's own vocabulary.

    The verdict is structural, so it is a property of the two graphs and not of anyone's
    reading of them. Only defect keys that already exist are returned: the three gating
    classes and the auxiliary ones that describe a stage the query does not have.
    """

    query_nodes, other_nodes = int(query.get("node_count") or 0), int(other.get("node_count") or 0)
    query_edges, other_edges = int(query.get("edge_count") or 0), int(other.get("edge_count") or 0)
    if other_nodes < query_nodes:
        return "missing_core_stage", "fewer_stages_than_the_query"
    if other_nodes > query_nodes:
        return "invented_module", "more_stages_than_the_query"
    if other_edges != query_edges:
        return "wrong_arrow_target", "different_number_of_relations"
    if dict(query.get("degree_histogram") or {}) != dict(other.get("degree_histogram") or {}):
        return "relation_reversal", "same_size_but_differently_wired"
    return "module_order_error", "same_size_and_wiring_in_another_order"


def structure_phrase(signature: dict[str, Any], *, prefix: str = "") -> str:
    """Render one signature as the short structural phrase a diagnosis points at."""

    phrase = f"{_text(signature.get('motif')) or 'unknown'} over {int(signature.get('node_count') or 0)} nodes and {int(signature.get('edge_count') or 0)} relations"
    return f"{prefix}{phrase}" if prefix else phrase


def certified_judgement(query: dict[str, Any], other: dict[str, Any], *, defect_key: str, kind: str) -> dict[str, Any]:
    """Return the judgement the deterministic structure rule certifies for one pair."""

    label = defect_label(defect_key)
    return {
        "preferred_candidate": "A",
        "quality_scores": dict(label["quality_scores"]),
        "diagnosis": [{
            "type": str(label["diagnosis_type"]),
            "target": structure_phrase(other, prefix="candidate B: "),
            "severity": str(label["severity"]),
            "evidence": f"the query is {structure_phrase(query)}; candidate B is {structure_phrase(other)} ({kind})",
            "figure": "B",
        }],
        "confidence": CERTIFIED_CONFIDENCE,
    }


def _subgraph_block(entry: dict[str, Any], image: str) -> dict[str, Any]:
    """Describe one side of a pair for the report and for the human reviewer."""

    locator = entry["locator"]
    return {
        "subgraph_id": _text(locator.get("subgraph_id")),
        "graph_id": _text(locator.get("graph_id")),
        "paper_id": _text(locator.get("paper_id")),
        "domain": _text(locator.get("domain")),
        "split": _text(locator.get("split")),
        "graph_kind": _text(locator.get("graph_kind")),
        "motif": _text(locator.get("motif")),
        "node_count": int(locator.get("node_count") or 0),
        "edge_count": int(locator.get("edge_count") or 0),
        "required": bool(locator.get("required")),
        "source_graph": _text(locator.get("source_graph")),
        "image": image,
    }


@dataclass(frozen=True)
class SubgraphCatalogue:
    """Every usable subgraph of one middleware version, with its vector and its graph."""

    version: MiddlewareSource
    entries: dict[str, dict[str, Any]]
    missing_graphs: tuple[str, ...] = ()
    unmapped_locators: tuple[str, ...] = ()

    @classmethod
    def from_version(cls, version: MiddlewareSource) -> "SubgraphCatalogue":
        """Join the locator index with the graph bodies it points at.

        A locator whose graph file is absent, or whose subgraph the extractor no longer
        reproduces, is reported instead of being dropped quietly: an index that no longer
        matches its graphs must not look like a shorter index.
        """

        graphs: dict[str, dict[str, Any]] = {}
        bodies: dict[str, dict[str, dict[str, Any]]] = {}
        entries: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        unmapped: list[str] = []
        for locator in version.subgraph_locators():
            subgraph_id = _text(locator.get("subgraph_id"))
            source_graph = _text(locator.get("source_graph"))
            if not subgraph_id or not source_graph:
                continue
            if source_graph not in bodies:
                path = version.root / source_graph
                if not path.is_file():
                    missing.append(source_graph)
                    graphs[source_graph] = {}
                    bodies[source_graph] = {}
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                graph = payload if isinstance(payload, dict) else {}
                graphs[source_graph] = graph
                bodies[source_graph] = {_text(item.get("subgraph_id")): item for item in extract_subgraphs(graph)[0]}
            body = bodies[source_graph].get(subgraph_id)
            if body is None:
                unmapped.append(subgraph_id)
                continue
            entries[subgraph_id] = {
                "locator": locator,
                "subgraph": body,
                "graph": graphs[source_graph],
                "source_graph": source_graph,
                "signature": structure_signature(body, graphs[source_graph]),
                "version_root": str(version.root),
            }
        return cls(version=version, entries=entries, missing_graphs=tuple(sorted(set(missing))), unmapped_locators=tuple(sorted(unmapped)))

    def ordered_queries(self, split: str = "") -> list[dict[str, Any]]:
        """Return the subgraphs a pair may be built around, required stages first.

        A subgraph with no motif or with a single node carries no structure to match, so
        it can never be a query; that filtering is why a query without a match is
        reported rather than silently retried on a weaker subgraph.
        """

        selected: list[dict[str, Any]] = []
        for subgraph_id in sorted(self.entries):
            entry = self.entries[subgraph_id]
            signature = entry["signature"]
            if int(signature.get("node_count") or 0) < MIN_NODE_COUNT or _text(signature.get("motif")) in {"", "unknown"}:
                continue
            locator_split = _text(entry["locator"].get("split"))
            if locator_split == HIDDEN_TEST_SPLIT or locator_split not in TRAINABLE_SPLITS:
                continue
            if split and locator_split != split:
                continue
            if not entry["locator"].get("embedding"):
                continue
            selected.append(entry)
        selected.sort(key=lambda entry: (not bool(entry["signature"].get("required")), _text(entry["locator"].get("subgraph_id"))))
        return selected

    def candidates_for(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the subgraphs one query may be paired with."""

        paper_id = _text(query["locator"].get("paper_id"))
        split = _text(query["locator"].get("split"))
        pool = []
        for subgraph_id in sorted(self.entries):
            entry = self.entries[subgraph_id]
            locator = entry["locator"]
            if _text(locator.get("paper_id")) == paper_id:
                continue
            if _text(locator.get("split")) != split or split == HIDDEN_TEST_SPLIT:
                continue
            if not locator.get("embedding"):
                continue
            pool.append(entry)
        return pool


def _image_for(entry: dict[str, Any], directory: Path) -> str:
    """Render one subgraph once and return its stored image path."""

    path = directory / IMAGE_DIR / f"{_short_id(_text(entry['locator'].get('subgraph_id')))}.png"
    if not path.is_file():
        render_subgraph(entry["subgraph"], entry["graph"], path)
    return str(path)


def _graph_labels(entry: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return the query's own node labels and relation sentences."""

    subgraph, graph = entry["subgraph"], entry["graph"]
    body = subgraph.get("properties") if isinstance(subgraph.get("properties"), dict) else {}
    center = _text(body.get("center_node_id"))
    nodes = {_text(item.get("id")): item for item in (graph.get("nodes") or []) if isinstance(item, dict)}
    ordered = [str(item) for item in (subgraph.get("node_ids") or [])]
    labels = []
    for node_id in [center, *[item for item in ordered if item != center]]:
        node = nodes.get(node_id) or {}
        label = _text(node.get("label")) or _text(node.get("name")) or node_id
        if label and label not in labels:
            labels.append(label)
    wanted = {str(item) for item in (subgraph.get("edge_ids") or [])}
    relations = []
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict) or _text(edge.get("id")) not in wanted:
            continue
        source = nodes.get(_text(edge.get("source"))) or {}
        target = nodes.get(_text(edge.get("target"))) or {}
        source_label = _text(source.get("label")) or _text(edge.get("source"))
        target_label = _text(target.get("label")) or _text(edge.get("target"))
        relation = _text(edge.get("relation_type"))
        relations.append(f"{source_label} -> {target_label}" + (f" ({relation})" if relation else ""))
    return labels, sorted(relations)


def _example_for(query: dict[str, Any], chosen: dict[str, Any], rejected: dict[str, Any], *, directory: Path, include_swapped: bool, recall_limit: int, same_layer_only: bool) -> list[dict[str, Any]]:
    """Write one query's structural pair, in both orientations when asked."""

    locator = query["locator"]
    defect_key, kind = mismatch_between(query["signature"], rejected["signature"])
    query_image = _image_for(query, directory)
    chosen_image = _image_for(chosen, directory)
    rejected_image = _image_for(rejected, directory)
    labels, relations = _graph_labels(query)
    query_text = subgraph_text(locator, query["subgraph"], query["graph"], ())
    # The example id stays short enough for the review sheet to show the pair's own
    # verdict next to it; the two full subgraph ids live in the provenance block.
    example_id = f"rag-subgraph-{_short_id(_text(locator.get('subgraph_id')))}-{_short_id(_text(rejected['locator'].get('subgraph_id')), length=8)}"
    row: dict[str, Any] = {
        "example_id": example_id,
        "dataset_version": DATASET_VERSION,
        "ground_truth_id": _text(locator.get("paper_id")),
        "paper_id": _text(locator.get("paper_id")),
        "domain": _text(locator.get("domain")),
        "split": _text(locator.get("split")),
        "task_type": TASK_TYPE,
        "candidate_a": chosen_image,
        "candidate_b": rejected_image,
        "preview": query_image,
        "labels": certified_judgement(query["signature"], rejected["signature"], defect_key=defect_key, kind=kind),
        "pair_group": f"{_text(locator.get('paper_id'))}|rag_subgraph|{_text(locator.get('subgraph_id'))}",
        "paper": {
            "title": f"{_text(locator.get('paper_id'))} structural query: {_text(locator.get('motif'))}",
            "abstract": query_text,
            "evidence_spans": [{"evidence_id": _text(locator.get("subgraph_id")), "quote": query_text, "kind": "subgraph"}],
            "source_path": str(directory),
        },
        "ground_truth": {
            "method_summary": f"which of the two supplied subgraphs is the same structure as the {_text(locator.get('motif'))} query",
            "required_entities": labels,
            "required_relations": relations,
            "graph_id": _text(locator.get("graph_id")),
            "contract_path": _text(locator.get("source_graph")),
            "contract_sha256": "",
        },
        "provenance": {
            "source": PROVENANCE_SOURCE,
            "defect_source": DEFECT_SOURCE,
            "corruption_type": defect_key,
            "target": structure_phrase(rejected["signature"], prefix="candidate B: "),
            "mismatch": {
                "kind": kind,
                "query": structure_phrase(query["signature"]),
                "candidate": structure_phrase(rejected["signature"]),
                "node_delta": int(rejected["signature"].get("node_count") or 0) - int(query["signature"].get("node_count") or 0),
                "edge_delta": int(rejected["signature"].get("edge_count") or 0) - int(query["signature"].get("edge_count") or 0),
            },
            "strict_structure_rule": STRICT_STRUCTURE_RULE,
            "recall_limit": int(recall_limit),
            "same_layer_only": bool(same_layer_only),
            "middleware_version": str(query["version_root"]),
            "query": _subgraph_block(query, query_image),
            "chosen": _subgraph_block(chosen, chosen_image),
            "rejected": _subgraph_block(rejected, rejected_image),
        },
    }
    rows = [row]
    if include_swapped:
        mirrored = dict(row)
        mirrored["example_id"] = f"{example_id}__swap"
        mirrored["candidate_a"] = row["candidate_b"]
        mirrored["candidate_b"] = row["candidate_a"]
        mirrored["labels"] = mirror_judgement(row["labels"])
        rows.append(mirrored)
    return rows


def _pair_for(query: dict[str, Any], catalogue: SubgraphCatalogue, *, directory: Path, include_swapped: bool, recall_limit: int, same_layer_only: bool) -> tuple[list[dict[str, Any]], str]:
    """Return the examples of one query, or the single reason it was refused."""

    pool = catalogue.candidates_for(query)
    if not pool:
        return [], "query_split_has_no_candidate"
    recalled = vector_recall_subgraphs(
        query["locator"],
        [entry["locator"] for entry in pool],
        limit=max(int(recall_limit), 2),
        exclude_paper_ids=(_text(query["locator"].get("paper_id")),),
        same_layer_only=bool(same_layer_only),
    )
    by_id = {_text(entry["locator"].get("subgraph_id")): entry for entry in pool}
    ordered = [by_id[_text(item.get("subgraph_id"))] for item in recalled if _text(item.get("subgraph_id")) in by_id]
    if not ordered:
        return [], "query_split_has_no_candidate"
    chosen = next((entry for entry in ordered if same_structure(query["signature"], entry["signature"])), None)
    if chosen is None:
        return [], "query_has_no_strict_match"
    rejected = next((entry for entry in ordered if not same_structure(query["signature"], entry["signature"])), None)
    if rejected is None:
        return [], "query_has_no_mismatch_candidate"
    return _example_for(query, chosen, rejected, directory=directory, include_swapped=include_swapped, recall_limit=recall_limit, same_layer_only=same_layer_only), ""


def _discipline(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Re-derive the pairing discipline from the written rows.

    Nothing here trusts the builder: the three properties that keep a pair honest are
    recomputed from the rows that were actually written, so a report can never claim a
    discipline the data does not have.
    """

    leaks: list[str] = []
    hidden: list[str] = []
    same_paper: list[str] = []
    cross_paper = cross_domain = same_domain = cross_layer = same_layer = 0
    for row in examples:
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        query = provenance.get("query") if isinstance(provenance.get("query"), dict) else {}
        for side in ("chosen", "rejected"):
            block = provenance.get(side) if isinstance(provenance.get(side), dict) else {}
            if _text(block.get("split")) != _text(query.get("split")):
                leaks.append(f"{row.get('example_id')}:{side}")
            if _text(block.get("split")) == HIDDEN_TEST_SPLIT:
                hidden.append(f"{row.get('example_id')}:{side}")
            if _text(block.get("paper_id")) == _text(query.get("paper_id")):
                same_paper.append(f"{row.get('example_id')}:{side}")
                continue
            cross_paper += 1
            if _text(block.get("domain")) != _text(query.get("domain")):
                cross_domain += 1
            else:
                same_domain += 1
            if _text(block.get("graph_kind")) != _text(query.get("graph_kind")):
                cross_layer += 1
            else:
                same_layer += 1
    return {
        "candidate_split_leaks": leaks,
        "hidden_test_candidates": hidden,
        "same_paper_candidates": same_paper,
        "cross_paper_sides": cross_paper,
        "cross_domain_sides": cross_domain,
        "same_domain_sides": same_domain,
        "cross_layer_sides": cross_layer,
        "same_layer_sides": same_layer,
        "ok": not leaks and not hidden and not same_paper,
    }


def _interleave_by_domain(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin queries so a domain cap is reached together, not by id order."""

    buckets: dict[str, list[dict[str, Any]]] = {}
    for entry in queries:
        domain = _text(entry["locator"].get("domain")) or "unknown"
        buckets.setdefault(domain, []).append(entry)
    mixed: list[dict[str, Any]] = []
    cursors = {name: 0 for name in sorted(buckets)}
    while True:
        progressed = False
        for name in cursors:
            index = cursors[name]
            if index >= len(buckets[name]):
                continue
            mixed.append(buckets[name][index])
            cursors[name] = index + 1
            progressed = True
        if not progressed:
            return mixed


def build_subgraph_pairs(version: MiddlewareSource | str | Path, out_dir: str | Path, *, split: str = "", limit: int = 0, per_domain: int = 0, include_swapped: bool = True, recall_limit: int = DEFAULT_RECALL_LIMIT, same_layer_only: bool = False) -> dict[str, Any]:
    """Write the RAG subgraph examples of one middleware version.

    Every query is either paired or refused with one recorded reason, so a run can never
    look complete while a whole class of queries was silently dropped. An empty result is
    a failure: an empty dataset must not pass for a finished build.
    """

    source = version if isinstance(version, MiddlewareSource) else MiddlewareSource(Path(version))
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    catalogue = SubgraphCatalogue.from_version(source)
    queries = catalogue.ordered_queries(split)
    domain_cap = max(int(per_domain), 0)
    if domain_cap:
        queries = _interleave_by_domain(queries)
    examples: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    paired = 0
    paired_by_domain: dict[str, int] = {}
    domains_present = {_text(entry["locator"].get("domain")) or "unknown" for entry in queries}
    capped_skipped = 0
    for entry in queries:
        if limit and paired >= int(limit):
            break
        domain = _text(entry["locator"].get("domain")) or "unknown"
        if domain_cap and paired_by_domain.get(domain, 0) >= domain_cap:
            capped_skipped += 1
            if all(paired_by_domain.get(name, 0) >= domain_cap for name in domains_present):
                break
            continue
        rows, reason = _pair_for(entry, catalogue, directory=dest, include_swapped=include_swapped, recall_limit=recall_limit, same_layer_only=same_layer_only)
        if reason:
            exclusions.append({
                "subgraph_id": _text(entry["locator"].get("subgraph_id")),
                "paper_id": _text(entry["locator"].get("paper_id")),
                "split": _text(entry["locator"].get("split")),
                "motif": _text(entry["signature"].get("motif")),
                "reason": reason,
            })
            continue
        paired += 1
        paired_by_domain[domain] = paired_by_domain.get(domain, 0) + 1
        examples.extend(rows)
        if domain_cap and paired % 50 == 0:
            print(f"paired={paired} by_domain={paired_by_domain}", flush=True)
    if not examples:
        raise SourceUnavailable("no subgraph of this middleware version has both a strict structural match and a mismatching neighbour; nothing was written")
    target = dest / EXAMPLES_NAME
    _write_jsonl(target, examples)
    discipline = _discipline(examples)
    report = {
        "contract": SUBGRAPH_PAIR_CONTRACT,
        "version": SUBGRAPH_PAIR_VERSION,
        "summary": "Structural query pairs of one middleware version: the same structure against its nearest mismatch.",
        "ok": bool(discipline["ok"]) and bool(examples),
        "middleware_version": str(source.root),
        "strict_structure_rule": STRICT_STRUCTURE_RULE,
        "recall": {"strategy": "vector_recall_then_strict_structure_match", "limit": int(recall_limit), "same_layer_only": bool(same_layer_only)},
        "catalogue": {
            "locators": len(source.subgraph_locators()),
            "usable_subgraphs": len(catalogue.entries),
            "papers": len({_text(entry["locator"].get("paper_id")) for entry in catalogue.entries.values()}),
            "domains": _distribution(entry["locator"].get("domain") for entry in catalogue.entries.values()),
            "layers": _distribution(entry["locator"].get("graph_kind") for entry in catalogue.entries.values()),
            "splits": _distribution(entry["locator"].get("split") for entry in catalogue.entries.values()),
            "motifs": _distribution(entry["signature"].get("motif") for entry in catalogue.entries.values()),
            "missing_graphs": list(catalogue.missing_graphs),
            "unmapped_locators": list(catalogue.unmapped_locators[:SKIPPED_SAMPLE]),
            "unmapped_locator_count": len(catalogue.unmapped_locators),
        },
        "counts": {
            "queries_considered": len(queries),
            "queries_paired": paired,
            "queries_paired_by_domain": dict(paired_by_domain),
            "per_domain": domain_cap,
            "domain_cap_skipped": capped_skipped,
            "examples": len(examples),
            "swapped": sum(1 for row in examples if str(row["example_id"]).endswith("__swap")),
            "subgraph_images": len(list((dest / IMAGE_DIR).glob("*.png"))) if (dest / IMAGE_DIR).is_dir() else 0,
        },
        "mismatch_distribution": _distribution(row["provenance"]["corruption_type"] for row in examples if not str(row["example_id"]).endswith("__swap")),
        "diagnosis_distribution": _distribution(entry["type"] for row in examples if not str(row["example_id"]).endswith("__swap") for entry in row["labels"]["diagnosis"]),
        "query_split_distribution": _distribution(row.get("split") for row in examples if not str(row["example_id"]).endswith("__swap")),
        "exclusions": {"by_reason": _distribution(item["reason"] for item in exclusions), "count": len(exclusions), "sample": exclusions[:SKIPPED_SAMPLE]},
        "discipline": discipline,
        "sources": {
            "subgraph_index_sha256": _sha256(source.root / "subgraph_index.jsonl"),
            "graph_index_sha256": _sha256(source.root / "graph_index.jsonl"),
            "examples_sha256": _sha256(target),
        },
        "out": str(target),
        "split": split,
    }
    (dest / REPORT_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report