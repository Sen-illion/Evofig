"""Unit tests for the deterministic Pilot retrieval index and probe."""

import json
from pathlib import Path
import tempfile
import unittest

from rfs.judge_training import pilot_graphs
from rfs.judge_training.pilot_index import (
    DEFAULT_PROBE_LIMIT,
    INDEX_ARTIFACTS,
    LEDGER_SCHEMA_VERSION,
    build_motif_rows,
    build_retrieval_probe,
    content_tags,
    graph_for_layer,
    index_all,
    index_coverage,
    index_record_for,
    index_report,
    layer_file,
    load_index_state,
    paper_index_report,
    paper_tags,
    record_text,
    sampled_queries,
    subgraph_text,
)
from rfs.judge_training.source_snapshot_io import write_jsonl
from rfs.providers.embeddings import DEFAULT_DIMENSION
from rfs.utils import write_json
from rfs.analysis.graph import extract_subgraphs


def node(node_id, label, *, kind="module", role="module", required=True):
    return {"id": node_id, "kind": kind, "label": label, "role": role, "required": required}


def edge(edge_id, source, target, *, relation="data_flow", required=True):
    return {"id": edge_id, "source": source, "target": target, "relation_type": relation, "required": required}


def scientific_graph(nodes, edges):
    return {
        "graph_id": "scientific-p0001",
        "paper_id": "p0001",
        "kind": "scientific",
        "nodes": nodes,
        "edges": edges,
        "required_node_ids": [item["id"] for item in nodes if item.get("required")],
        "required_edge_ids": [item["id"] for item in edges if item.get("required")],
    }


def visual_graph(nodes, edges):
    return {
        "graph_id": "visual-p0001",
        "paper_id": "p0001",
        "kind": "visual",
        "nodes": nodes,
        "edges": edges,
    }


def chain_graph(*, graph_id, paper_id, kind):
    nodes = [node("m1", "Encoder"), node("m2", "Decoder"), node("out1", "Output", kind="artifact")]
    edges = [edge("e1", "m1", "m2"), edge("e2", "m2", "out1")]
    return {
        "graph_id": graph_id,
        "paper_id": paper_id,
        "kind": kind,
        "nodes": nodes,
        "edges": edges,
        "required_node_ids": [item["id"] for item in nodes],
        "required_edge_ids": [item["id"] for item in edges],
    }


def fan_in_graph(*, graph_id, paper_id, kind):
    nodes = [node("a", "Stream A"), node("b", "Stream B"), node("c", "Fusion"), node("d", "Report", kind="artifact")]
    edges = [edge("e1", "a", "c"), edge("e2", "b", "c"), edge("e3", "c", "d")]
    return {
        "graph_id": graph_id,
        "paper_id": paper_id,
        "kind": kind,
        "nodes": nodes,
        "edges": edges,
        "required_node_ids": [item["id"] for item in nodes],
        "required_edge_ids": [item["id"] for item in edges],
    }


def row(paper_id, *, split="train", domain="cv", **extra):
    payload = {"paper_id": paper_id, "split": split, "domain": domain}
    payload.update(extra)
    return payload


def entry(paper_id, *, graph, layer="scientific"):
    return {"paper_id": paper_id, ("scientific_graph" if layer == "scientific" else "visual_graph"): graph}


class PilotIndexRecordTests(unittest.TestCase):
    def test_content_tags_are_deterministic_and_casefolded(self):
        graph = chain_graph(graph_id="g", paper_id="p0001", kind="scientific")
        tags = content_tags(row("p0001", domain="CV", figure_type="Framework", panel_count=4, has_arrows=True), graph)
        self.assertEqual(tags[0], "domain:cv")
        self.assertIn("figure_type:framework", tags)
        self.assertIn("panel_count:4", tags)
        self.assertIn("has_arrows:yes", tags)
        self.assertIn("role:module", tags)
        self.assertIn("relation:data_flow", tags)
        self.assertEqual(tags, content_tags(row("p0001", domain="CV", figure_type="Framework", panel_count=4, has_arrows=True), graph))

    def test_layer_file_points_at_the_stored_graph(self):
        self.assertEqual(layer_file("scientific", "p0001"), "graphs/p0001/scientific_graph.json")
        self.assertEqual(layer_file("visual", "iclr_2026_1"), "graphs/iclr_2026_1/visual_graph.json")
        self.assertEqual(layer_file("scientific", ""), "graphs/unknown-paper/scientific_graph.json")

    def test_graph_for_layer_ignores_a_missing_or_broken_layer(self):
        self.assertIsNone(graph_for_layer(None, "scientific"))
        self.assertIsNone(graph_for_layer({}, "visual"))
        self.assertIsNone(graph_for_layer({"visual_graph": []}, "visual"))
        graph = chain_graph(graph_id="g", paper_id="p0001", kind="visual")
        self.assertIs(graph_for_layer({"visual_graph": graph}, "visual"), graph)

    def test_record_keeps_a_locator_and_drops_subgraph_bodies(self):
        graph = chain_graph(graph_id="g", paper_id="p0001", kind="scientific")
        record, locators = index_record_for(graph, row=row("p0001"), layer="scientific", directory="p0001")

        self.assertEqual(record["contract"], "rfs.graph_index_record")
        self.assertEqual(record["graph_kind"], "scientific")
        self.assertEqual(record["domain"], "cv")
        self.assertEqual(record["subgraphs"], [])
        self.assertFalse(record["subgraph_locator"]["materialized"])
        self.assertEqual(record["subgraph_locator"]["source_graph"], "graphs/p0001/scientific_graph.json")
        self.assertEqual(record["subgraph_locator"]["count"], len(locators))
        self.assertEqual(record["subgraph_locator"]["count"], sum(record["index_report"]["motif_counts"].values()))
        self.assertTrue(record["index_report"]["ok"])
        self.assertFalse(record["index_report"]["subgraph_payloads_materialized"])
        self.assertEqual(record["metadata"]["layer"], "scientific")
        self.assertEqual(record["metadata"]["source_graph"], "graphs/p0001/scientific_graph.json")
        self.assertEqual(record["features"]["node_count"], 3)
        self.assertTrue(record["record_id"])
        for item in locators:
            self.assertEqual(item["contract"], "rfs.graph_subgraph_locator")
            self.assertEqual(item["paper_id"], "p0001")
            self.assertEqual(item["split"], "train")
            self.assertEqual(item["source_graph"], "graphs/p0001/scientific_graph.json")
            self.assertEqual(item["extractor"], "node_centered_one_hop_v1")
            self.assertTrue(item["center_node_id"])
            self.assertGreaterEqual(item["node_count"], 1)
            # A locator row is the only place a subgraph vector is stored, so it has to
            # carry the vector and the identity of the encoder that produced it.
            self.assertEqual(len(item["embedding"]), DEFAULT_DIMENSION)
            self.assertEqual(item["embedding_fingerprint"]["kind"], "hashing_ngram")
            self.assertEqual(item["embedding_fingerprint"]["model"], "rfs-hashing-ngram-v1")
            self.assertIn("dimension", item["embedding_fingerprint"])


def coverage_subgraph_embeddings(payload):
    return index_coverage(payload)["subgraph_embeddings"]


class PilotIndexAggregationTests(unittest.TestCase):
    def build(self, rows, graphs, visuals, **kwargs):
        return index_all(rows, graphs=graphs, visuals=visuals, graph_id="paper-d0-pilot-graph-v001", **kwargs)

    def test_index_all_indexes_both_layers_of_every_paper(self):
        rows = [row("p0001"), row("p0002", split="validation", domain="biology")]
        graphs = {
            "p0001": entry("p0001", graph=chain_graph(graph_id="s1", paper_id="p0001", kind="scientific")),
            "p0002": entry("p0002", graph=fan_in_graph(graph_id="s2", paper_id="p0002", kind="scientific")),
        }
        visuals = {
            "p0001": entry("p0001", graph=chain_graph(graph_id="v1", paper_id="p0001", kind="visual"), layer="visual"),
            "p0002": entry("p0002", graph=fan_in_graph(graph_id="v2", paper_id="p0002", kind="visual"), layer="visual"),
        }
        payload, ledger = self.build(rows, graphs, visuals)

        self.assertEqual(len(payload["records"]), 4)
        self.assertEqual([item["record_id"] for item in payload["records"]], sorted(item["record_id"] for item in payload["records"]))
        self.assertEqual([item["subgraph_id"] for item in payload["subgraphs"]], sorted(item["subgraph_id"] for item in payload["subgraphs"]))
        self.assertEqual(payload["report"]["papers"], 2)
        self.assertEqual(payload["report"]["layers"], {"scientific": 2, "visual": 2})
        self.assertEqual(payload["report"]["provider_calls"], 0)
        self.assertEqual(payload["report"]["probe_queries"], 2)
        self.assertEqual([item["step"] for item in ledger], ["index", "index"])
        self.assertEqual([item["status"] for item in ledger], ["ok", "ok"])
        self.assertEqual([item["record_count"] for item in ledger], [2, 2])
        self.assertEqual(payload["graph_id"], "paper-d0-pilot-graph-v001")
        coverage = index_coverage(payload)
        self.assertEqual(coverage["papers"], 2)
        self.assertEqual(coverage["records"], 4)
        self.assertEqual(coverage["splits"], {"train": 2, "validation": 2})
        self.assertEqual(coverage["domains"], {"biology": 2, "cv": 2})
        self.assertEqual(coverage["papers_missing_a_layer"], [])
        self.assertEqual(coverage["hidden_test_records"], 0)
        self.assertGreater(coverage["subgraph_locations"], 0)
        self.assertEqual(coverage["probe"]["queries"], 2)

    def test_records_carry_a_traceable_embedding_and_the_probe_recalls_by_it(self):
        rows = [row("p0001"), row("p0002", domain="biology"), row("p0003", domain="materials")]
        graphs = {item["paper_id"]: entry(item["paper_id"], graph=fan_in_graph(graph_id="s-" + item["paper_id"], paper_id=item["paper_id"], kind="scientific")) for item in rows}
        visuals = {item["paper_id"]: entry(item["paper_id"], graph=fan_in_graph(graph_id="v-" + item["paper_id"], paper_id=item["paper_id"], kind="visual"), layer="visual") for item in rows}
        payload, _ = self.build(rows, graphs, visuals)

        self.assertTrue(payload["subgraphs"])
        self.assertTrue(all(len(item["embedding"]) == DEFAULT_DIMENSION for item in payload["subgraphs"]))
        self.assertEqual(coverage_subgraph_embeddings(payload), len(payload["subgraphs"]))
        for record in payload["records"]:
            self.assertEqual(len(record["embedding"]), 256)
            self.assertEqual(record["metadata"]["embedding"]["model"], "rfs-hashing-ngram-v1")
        self.assertEqual(index_report(payload)["records_without_embedding"], 0)
        self.assertEqual(payload["probe"]["embedding"]["kind"], "hashing_ngram")
        self.assertEqual(payload["probe"]["recall"]["strategy"], "vector_recall_then_structure_rerank")
        self.assertGreater(payload["probe"]["queries"][0]["recall_limit"], 0)

    def test_record_text_is_deterministic_and_provider_free(self):
        graph = fan_in_graph(graph_id="g", paper_id="p0001", kind="scientific")
        tags = content_tags(row("p0001", figure_type="Framework"), graph)
        text = record_text(row("p0001", figure_type="Framework"), graph, tags)
        self.assertEqual(text, record_text(row("p0001", figure_type="Framework"), graph, tags))
        self.assertIn("Stream A", text)
        self.assertIn("data_flow", text)

    def test_index_all_fails_a_paper_that_misses_a_layer(self):
        rows = [row("p0001"), row("p0002")]
        graphs = {"p0001": entry("p0001", graph=chain_graph(graph_id="s1", paper_id="p0001", kind="scientific"))}
        payload, ledger = self.build(rows, graphs, {})

        self.assertEqual(payload["records"], [])
        self.assertEqual([item["status"] for item in ledger], ["failed", "failed"])
        self.assertIn("no stored visual layer", ledger[0]["error"])
        self.assertEqual(ledger[0]["token_source"], "not_called")
        self.assertIsNone(ledger[0]["total_tokens"])
        report = paper_index_report(ledger[0])
        self.assertFalse(report["ok"])
        self.assertEqual(report["errors"], ["no stored visual layer for this paper"])

    def test_index_all_is_deterministic(self):
        rows = [row("p0001"), row("p0002", domain="materials")]
        graphs = {item["paper_id"]: entry(item["paper_id"], graph=chain_graph(graph_id="s-" + item["paper_id"], paper_id=item["paper_id"], kind="scientific")) for item in rows}
        visuals = {item["paper_id"]: entry(item["paper_id"], graph=chain_graph(graph_id="v-" + item["paper_id"], paper_id=item["paper_id"], kind="visual"), layer="visual") for item in rows}
        first, _ = self.build(rows, graphs, visuals)
        second, _ = self.build(rows, graphs, visuals)
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )

    def test_paper_index_report_rebuilds_the_audit_from_the_ledger_row(self):
        report = paper_index_report({
            "paper_id": "p0001",
            "status": "ok",
            "record_count": 2,
            "record_ids": ["b", "a"],
            "node_count": 7,
            "edge_count": 6,
            "motif_count": 3,
            "subgraph_count": 11,
        })
        self.assertTrue(report["ok"])
        self.assertEqual(report["record_ids"], ["b", "a"])
        self.assertEqual(report["subgraph_count"], 11)
        self.assertEqual(report["provider_calls"], 0)
        self.assertEqual(report["errors"], [])
        failed = paper_index_report({"paper_id": "p0002", "status": "failed", "error": "boom", "warnings": ["boom"]})
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["errors"], ["boom"])

    def test_motif_rows_aggregate_papers_domains_and_layers(self):
        cases = (
            (chain_graph(graph_id="s1", paper_id="p0001", kind="scientific"), row("p0001"), "scientific", "p0001"),
            (chain_graph(graph_id="v1", paper_id="p0001", kind="visual"), row("p0001"), "visual", "p0001"),
            (fan_in_graph(graph_id="s2", paper_id="p0002", kind="scientific"), row("p0002", domain="biology"), "scientific", "p0002"),
        )
        records = []
        locators = []
        for graph, paper_row, layer, directory in cases:
            record, layer_locators = index_record_for(graph, row=paper_row, layer=layer, directory=directory)
            records.append(record)
            locators.extend(layer_locators)
        rows = build_motif_rows(records, locators)
        self.assertTrue(rows)
        self.assertEqual(sum(item["subgraph_count"] for item in rows), len(locators))
        for item in rows:
            self.assertEqual(item["contract"], "rfs.graph_motif_index")
            self.assertEqual(item["papers"], sorted(item["papers"]))
            self.assertEqual(sum(item["domains"].values()), item["subgraph_count"])
            self.assertEqual(sum(item["graph_kinds"].values()), item["subgraph_count"])
            self.assertEqual(item["paper_count"], len(item["papers"]))
            self.assertLessEqual(item["required_subgraphs"], item["subgraph_count"])


class PilotIndexProbeTests(unittest.TestCase):
    def records(self):
        return [
            index_record_for(chain_graph(graph_id="s1", paper_id="p0001", kind="scientific"), row=row("p0001"), layer="scientific", directory="p0001")[0],
            index_record_for(chain_graph(graph_id="s2", paper_id="p0002", kind="scientific"), row=row("p0002", domain="biology"), layer="scientific", directory="p0002")[0],
            index_record_for(fan_in_graph(graph_id="s3", paper_id="p0003", kind="scientific"), row=row("p0003", domain="materials"), layer="scientific", directory="p0003")[0],
        ]

    def test_sampled_queries_keeps_small_lists_and_spreads_large_ones(self):
        items = [{"paper_id": str(index)} for index in range(10)]
        self.assertEqual(sampled_queries(items, 0), items)
        self.assertEqual(sampled_queries(items, 20), items)
        picked = sampled_queries(items, 3)
        self.assertEqual([item["paper_id"] for item in picked], ["0", "3", "6"])
        self.assertEqual(picked, sampled_queries(items, 3))

    def test_probe_ranks_a_structurally_similar_paper_of_another_domain(self):
        probe = build_retrieval_probe(self.records(), [row("p0001"), row("p0002", domain="biology"), row("p0003", domain="materials")], top_k=2, graph_id="g1")

        self.assertEqual(probe["graph_id"], "g1")
        self.assertEqual(probe["summary"]["probe_papers"], 3)
        self.assertEqual(probe["summary"]["queries_with_hits"], 3)
        self.assertEqual(probe["summary"]["queries_without_hits"], [])
        self.assertEqual(probe["guards"]["self_hits"], 0)
        self.assertEqual(probe["guards"]["provider_calls"], 0)
        self.assertEqual(probe["guards"]["hidden_test_records"], 0)
        self.assertEqual(probe["guards"]["unexpected_splits"], [])
        first = next(item for item in probe["queries"] if item["paper_id"] == "p0001")
        self.assertEqual(first["hits"][0]["paper_id"], "p0002")
        self.assertTrue(first["hits"][0]["cross_domain"])
        self.assertEqual(first["hits"][0]["selection_reason"], "similar_subgraph_logic_with_domain_or_content_variation")
        self.assertGreaterEqual(first["hits"][0]["score"], first["hits"][-1]["score"])
        self.assertTrue(first["motif_counts"])

    def test_probe_guards_flag_hidden_test_and_unknown_splits(self):
        records = self.records()
        records[1]["metadata"]["split"] = "hidden_test"
        records[2]["metadata"]["split"] = "test"
        probe = build_retrieval_probe(records, [row("p0001")], top_k=2)

        self.assertEqual(probe["guards"]["hidden_test_records"], 1)
        self.assertEqual(probe["guards"]["unexpected_splits"], ["test"])
        self.assertEqual(probe["guards"]["splits"], {"hidden_test": 1, "test": 1, "train": 1})
        self.assertEqual(probe["queries"][0]["split"], "train")

    def test_probe_without_another_paper_reports_no_hit(self):
        records = self.records()[:1]
        probe = build_retrieval_probe(records, [row("p0001")], top_k=5)

        self.assertEqual(probe["queries"][0]["hit_count"], 0)
        self.assertEqual(probe["queries"][0]["top_paper_id"], "")
        self.assertEqual(probe["summary"]["queries_with_hits"], 0)
        self.assertEqual(probe["summary"]["queries_without_hits"], ["p0001"])
        self.assertEqual(probe["summary"]["mean_top_score"], 0.0)

    def test_paper_tags_describe_the_paper_and_not_one_subgraph(self):
        graph = chain_graph(graph_id="g", paper_id="p0001", kind="scientific")
        tags = paper_tags(row("p0001", domain="CV", figure_type="Framework"))

        self.assertEqual(tags, ["domain:cv", "figure_type:framework"])
        self.assertTrue(all(not tag.startswith("role:") for tag in tags))
        self.assertTrue(all(not tag.startswith("relation:") for tag in tags))
        self.assertIn("role:module", content_tags(row("p0001"), graph))

    def test_subgraph_text_is_deterministic_and_reads_like_a_structure(self):
        graph = fan_in_graph(graph_id="g", paper_id="p0001", kind="scientific")
        record, locators = index_record_for(graph, row=row("p0001", figure_type="Framework"), layer="scientific", directory="p0001")
        locator = next(item for item in locators if item["center_node_id"] == "c")
        body = extract_subgraphs(graph)[0]
        chosen = next(item for item in body if item["properties"]["center_node_id"] == "c")

        first = subgraph_text(locator, chosen, graph, paper_tags(row("p0001")))
        second = subgraph_text(locator, chosen, graph, paper_tags(row("p0001")))

        self.assertEqual(first, second)
        self.assertIn("layer:scientific", first)
        self.assertIn("motif:merge", first)
        self.assertIn("nodes:4", first)
        self.assertIn("edges:3", first)
        self.assertIn("degrees:0:1=2,1:0=1,2:1=1", first)
        self.assertIn("center_degrees:2:1", first)
        self.assertIn("center_label:Fusion", first)
        self.assertIn("node:Stream A", first)
        self.assertIn("relation:data_flow", first)
        self.assertIn("domain:cv", first)
        self.assertEqual(first.count("center_label:"), 1)

    def test_subgraph_text_separates_two_shapes_of_the_same_size(self):
        merge_graph = fan_in_graph(graph_id="m", paper_id="p0001", kind="scientific")
        chain = chain_graph(graph_id="c", paper_id="p0002", kind="scientific")
        merge_body = next(item for item in extract_subgraphs(merge_graph)[0] if item["properties"]["center_node_id"] == "c")
        chain_body = next(item for item in extract_subgraphs(chain)[0] if item["properties"]["center_node_id"] == "m2")
        merge_locator = {"graph_kind": "scientific", "motif": "merge", "center_node_id": "c"}
        chain_locator = {"graph_kind": "scientific", "motif": "linear", "center_node_id": "m2"}

        self.assertNotEqual(
            subgraph_text(merge_locator, merge_body, merge_graph),
            subgraph_text(chain_locator, chain_body, chain),
        )

    def test_index_report_counts_no_provider_calls(self):
        report = index_report({"records": [{"paper_id": "p0001", "graph_kind": "scientific"}], "motifs": [{}], "subgraphs": [{}, {}], "probe": {"queries": [{}], "guards": {"self_hits": 0}}})
        self.assertEqual(report["provider_calls"], 0)
        self.assertEqual(report["papers"], 1)
        self.assertEqual(report["records"], 1)
        self.assertEqual(report["motifs"], 1)
        self.assertEqual(report["subgraph_locations"], 2)
        self.assertEqual(report["probe_queries"], 1)
        self.assertEqual(report["probe_guards"], {"self_hits": 0})
        self.assertEqual(index_report(None)["records"], 0)

    def test_an_index_without_embeddings_is_reported_as_an_error(self):
        legacy = index_report({"records": [{"record_id": "r1", "paper_id": "p0001", "graph_kind": "scientific", "embedding": []}], "probe": {"queries": []}})
        self.assertEqual(1, legacy["records_without_embedding"])
        self.assertEqual(1, len(legacy["errors"]))
        self.assertIn("no stored embedding", legacy["errors"][0])
        rebuilt = index_report({"records": [{"record_id": "r1", "paper_id": "p0001", "graph_kind": "scientific", "embedding": [0.5, 0.5]}], "probe": {"queries": []}})
        self.assertEqual(0, rebuilt["records_without_embedding"])
        self.assertEqual([], rebuilt["errors"])

    def test_a_locator_without_an_embedding_is_reported_as_an_error(self):
        payload = index_report({"records": [], "subgraphs": [{"subgraph_id": "s1"}, {"subgraph_id": "s2", "embedding": [0.5]}], "probe": {"queries": []}})

        self.assertEqual(1, payload["subgraphs_without_embedding"])
        self.assertEqual(1, len(payload["errors"]))
        self.assertIn("subgraph locator", payload["errors"][0])
        self.assertEqual([], index_report({"subgraphs": [{"subgraph_id": "s1", "embedding": [0.5]}]})["errors"])


class PilotIndexStorageTests(unittest.TestCase):
    def test_load_index_state_reads_the_written_artifacts(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_jsonl(root / "graph_index.jsonl", [{"record_id": "a", "paper_id": "p0001", "graph_kind": "scientific"}])
            write_jsonl(root / "motif_index.jsonl", [{"motif": "chain"}])
            write_jsonl(root / "subgraph_index.jsonl", [{"subgraph_id": "s1"}])
            write_json(root / "retrieval_probe.json", {"graph_id": "g1", "queries": [{"paper_id": "p0001"}], "guards": {"self_hits": 0}})
            state = load_index_state(root)

        self.assertEqual(state["graph_id"], "g1")
        self.assertEqual(len(state["records"]), 1)
        self.assertEqual(len(state["motifs"]), 1)
        self.assertEqual(len(state["subgraphs"]), 1)
        self.assertEqual(state["report"]["probe_queries"], 1)

    def test_load_index_state_without_artifacts_returns_none(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.assertIsNone(load_index_state(root))
            write_jsonl(root / "graph_index.jsonl", [{"record_id": "a"}])
            self.assertIsNone(load_index_state(root))

    def test_artifact_names_and_schema_match_the_orchestration(self):
        self.assertEqual(LEDGER_SCHEMA_VERSION, pilot_graphs.SCHEMA_VERSION)
        self.assertEqual(INDEX_ARTIFACTS, ("graph_index.jsonl", "motif_index.jsonl", "subgraph_index.jsonl", "retrieval_probe.json"))
        self.assertEqual(DEFAULT_PROBE_LIMIT, 64)
        self.assertEqual(pilot_graphs.INDEX_STEP, "index")
        self.assertIn(pilot_graphs.INDEX_STEP, pilot_graphs.IMPLEMENTED_STAGES)
        self.assertEqual(pilot_graphs.STAGE_ORDER[-1], pilot_graphs.INDEX_STEP)
        self.assertLess(
            pilot_graphs.STAGE_ORDER.index(pilot_graphs.BIND_STEP),
            pilot_graphs.STAGE_ORDER.index(pilot_graphs.INDEX_STEP),
        )


if __name__ == "__main__":
    unittest.main()
