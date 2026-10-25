"""Tests for the RAG subgraph preference example builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from rfs.judge_training.judge_schema import DIAGNOSIS_TYPES, DEFECT_LABELS, validate_judgement
from rfs.judge_training.pilot_index import index_all
from rfs.judge_training.preference_pairs import chosen_source, known_defects
from rfs.judge_training.preference_sources import MiddlewareSource, SourceUnavailable
from rfs.judge_training.schemas import validate_example
from rfs.judge_training.source_snapshot_io import write_jsonl
from rfs.judge_training.subgraph_pairs import (
    EXCLUDED_REASONS,
    SubgraphCatalogue,
    build_subgraph_pairs,
    mismatch_between,
    structure_phrase,
)
from rfs.judge_training.subgraph_render import render_subgraph
from rfs.utils import write_json


TASK = "pairwise_preference"


def node(node_id, label):
    return {"id": node_id, "kind": "module", "label": label, "role": "module", "required": True}


def edge(edge_id, source, target, relation="data_flow"):
    return {"id": edge_id, "source": source, "target": target, "relation_type": relation, "required": True}


def layer_graph(paper_id, kind, nodes, edges):
    return {
        "graph_id": f"{kind}-{paper_id}",
        "paper_id": paper_id,
        "kind": kind,
        "nodes": nodes,
        "edges": edges,
        "required_node_ids": [item["id"] for item in nodes],
        "required_edge_ids": [item["id"] for item in edges],
    }


def chain(paper_id, kind):
    return layer_graph(paper_id, kind, [node("m1", "Encoder"), node("m2", "Decoder"), node("m3", "Output")], [edge("e1", "m1", "m2"), edge("e2", "m2", "m3")])


def fork(paper_id, kind):
    return layer_graph(paper_id, kind, [node("f1", "Input"), node("f2", "Branch A"), node("f3", "Branch B")], [edge("e1", "f1", "f2"), edge("e2", "f1", "f3")])


def signature(motif, nodes, edges, *, histogram=None, center="c"):
    return {"motif": motif, "node_count": nodes, "edge_count": edges, "degree_histogram": histogram or {}, "center_node_id": center}


def build_version(root: Path, papers: dict[str, dict], *, embedder=None) -> Path:
    """Write a minimal middleware version the builder can be pointed at."""

    rows = []
    graphs: dict[str, dict] = {}
    visuals: dict[str, dict] = {}
    for paper_id, spec in papers.items():
        rows.append({"paper_id": paper_id, "split": spec["split"], "domain": spec.get("domain", "cv")})
        layout = fork if spec.get("shape") == "fork" else chain
        graphs[paper_id] = {"paper_id": paper_id, "scientific_graph": layout(paper_id, "scientific")}
        visuals[paper_id] = {"paper_id": paper_id, "visual_graph": layout(paper_id, "visual")}
        folder = root / "graphs" / paper_id
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / "scientific_graph.json", graphs[paper_id]["scientific_graph"])
        write_json(folder / "visual_graph.json", visuals[paper_id]["visual_graph"])
    payload, _ = index_all(
        rows,
        graphs=graphs,
        visuals=visuals,
        graph_id="test-version",
        directories={paper_id: paper_id for paper_id in papers},
        embedder=embedder,
    )
    write_jsonl(root / "graph_index.jsonl", payload["records"])
    write_jsonl(root / "motif_index.jsonl", payload["motifs"])
    write_jsonl(root / "subgraph_index.jsonl", payload["subgraphs"])
    write_json(root / "retrieval_probe.json", payload["probe"])
    return root


def load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class MismatchTests(unittest.TestCase):
    def test_the_five_structural_differences_map_onto_existing_defect_keys(self):
        query = signature("linear", 3, 2, histogram={"0:1": 1, "1:0": 1, "1:1": 1})
        cases = [
            (signature("linear", 2, 1, histogram={"0:1": 1, "1:0": 1}), "missing_core_stage"),
            (signature("linear", 4, 3, histogram={}), "invented_module"),
            (signature("linear", 3, 3, histogram={}), "wrong_arrow_target"),
            (signature("linear", 3, 2, histogram={"0:0": 1, "2:0": 1, "0:2": 1}), "relation_reversal"),
            (signature("merge", 3, 2, histogram={"0:1": 1, "1:0": 1, "1:1": 1}), "module_order_error"),
        ]
        for other, expected in cases:
            with self.subTest(expected=expected):
                key, kind = mismatch_between(query, other)
                self.assertEqual(key, expected)
                self.assertTrue(kind)
                self.assertIn(key, DEFECT_LABELS)

    def test_the_reported_diagnosis_stays_inside_the_frozen_ten(self):
        query = signature("linear", 3, 2)
        for motif, nodes, edges in (("linear", 2, 1), ("linear", 4, 3), ("linear", 3, 3), ("merge", 3, 2)):
            key, _ = mismatch_between(query, signature(motif, nodes, edges))
            self.assertIn(DEFECT_LABELS[key]["diagnosis_type"], DIAGNOSIS_TYPES)

    def test_a_signature_reads_as_one_structural_phrase(self):
        self.assertEqual(structure_phrase(signature("linear", 3, 2)), "linear over 3 nodes and 2 relations")
        self.assertEqual(structure_phrase(signature("linear", 3, 2), prefix="candidate B: "), "candidate B: linear over 3 nodes and 2 relations")


class CatalogueTests(unittest.TestCase):
    def test_the_catalogue_joins_locators_with_their_graphs(self):
        with tempfile.TemporaryDirectory() as name:
            root = build_version(Path(name), {"p0001": {"split": "train"}, "p0002": {"split": "train"}})
            catalogue = SubgraphCatalogue.from_version(MiddlewareSource(root))

        self.assertTrue(catalogue.entries)
        self.assertEqual(catalogue.missing_graphs, ())
        self.assertEqual(catalogue.unmapped_locators, ())
        for entry in catalogue.entries.values():
            self.assertTrue(entry["subgraph"]["node_ids"])
            self.assertTrue(entry["signature"]["motif"])
            self.assertTrue(entry["locator"]["embedding"])

    def test_a_locator_without_a_graph_is_reported_instead_of_dropped_quietly(self):
        with tempfile.TemporaryDirectory() as name:
            root = build_version(Path(name), {"p0001": {"split": "train"}, "p0002": {"split": "train"}})
            (root / "graphs" / "p0002" / "scientific_graph.json").unlink()
            catalogue = SubgraphCatalogue.from_version(MiddlewareSource(root))

        self.assertEqual(list(catalogue.missing_graphs), ["graphs/p0002/scientific_graph.json"])
        self.assertTrue(catalogue.entries)
        self.assertTrue(all(entry["source_graph"] != "graphs/p0002/scientific_graph.json" for entry in catalogue.entries.values()))

    def test_queries_skip_single_node_subgraphs_and_the_hidden_test_split(self):
        with tempfile.TemporaryDirectory() as name:
            root = build_version(Path(name), {"p0001": {"split": "train"}, "p0002": {"split": "hidden_test"}})
            catalogue = SubgraphCatalogue.from_version(MiddlewareSource(root))
            queries = catalogue.ordered_queries()

        self.assertTrue(queries)
        self.assertTrue(all(int(entry["signature"]["node_count"]) >= 2 for entry in queries))
        self.assertTrue(all(entry["signature"]["motif"] != "unknown" for entry in queries))
        self.assertEqual({entry["locator"]["split"] for entry in queries}, {"train"})

    def test_a_query_may_not_be_paired_with_its_own_paper_or_another_split(self):
        with tempfile.TemporaryDirectory() as name:
            root = build_version(Path(name), {"p0001": {"split": "train"}, "p0002": {"split": "validation"}})
            catalogue = SubgraphCatalogue.from_version(MiddlewareSource(root))
            query = catalogue.ordered_queries()[0]

        self.assertEqual(catalogue.candidates_for(query), [])


class BuilderTests(unittest.TestCase):
    def version(self, root: Path, *, extra: bool = True, embedder=None) -> Path:
        papers = {
            "p0001": {"split": "train", "domain": "cv"},
            "p0002": {"split": "train", "domain": "biology"},
        }
        if extra:
            papers["p0003"] = {"split": "train", "domain": "materials", "shape": "fork"}
        return build_version(root, papers, embedder=embedder)

    def test_a_pair_puts_a_strict_match_against_its_nearest_mismatch(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = self.version(root / "version")
            report = build_subgraph_pairs(version, root / "out", split="train", include_swapped=False)
            rows = load_examples(Path(report["out"]))

        self.assertTrue(report["ok"])
        self.assertTrue(rows)
        for row in rows:
            provenance = row["provenance"]
            query = provenance["query"]
            chosen = provenance["chosen"]
            rejected = provenance["rejected"]
            self.assertEqual((chosen["motif"], chosen["node_count"], chosen["edge_count"]), (query["motif"], query["node_count"], query["edge_count"]))
            self.assertNotEqual((rejected["motif"], rejected["node_count"], rejected["edge_count"]), (query["motif"], query["node_count"], query["edge_count"]))
            self.assertEqual(chosen["paper_id"], rejected["paper_id"])
            self.assertNotEqual(query["paper_id"], chosen["paper_id"])
            self.assertEqual(row["pair_group"], f"{query['paper_id']}|rag_subgraph|{query['subgraph_id']}")
            self.assertEqual(provenance["source"], "rag_subgraph")
            self.assertEqual(row["task_type"], TASK)

    def test_every_example_satisfies_the_frozen_envelope_and_judgement_contract(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = self.version(root / "version")
            report = build_subgraph_pairs(version, root / "out", split="train", include_swapped=True)
            rows = load_examples(Path(report["out"]))

        self.assertTrue(rows)
        self.assertEqual(sum(1 for row in rows if str(row["example_id"]).endswith("__swap")), len(rows) // 2)
        for row in rows:
            self.assertEqual([], validate_example(row))
            self.assertEqual([], validate_judgement(row["labels"], task_type=TASK))
            self.assertIn(row["labels"]["preferred_candidate"], {"A", "B"})
            self.assertIn(row["labels"]["diagnosis"][0]["type"], DIAGNOSIS_TYPES)

    def test_the_pair_source_carries_a_traceable_defect_for_the_pair_builder(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = self.version(root / "version")
            report = build_subgraph_pairs(version, root / "out", split="train", include_swapped=False)
            rows = load_examples(Path(report["out"]))

        for row in rows:
            self.assertEqual(chosen_source(row), "subgraph_structure_rule")
            defects = known_defects(row)
            self.assertEqual(1, len(defects))
            self.assertTrue(defects[0]["diagnosis_type"] in DIAGNOSIS_TYPES)
            self.assertEqual(defects[0]["diagnosis_type"], row["labels"]["diagnosis"][0]["type"])

    def test_both_candidates_and_the_query_are_rendered_and_distinct(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = self.version(root / "version")
            report = build_subgraph_pairs(version, root / "out", split="train", include_swapped=False)
            rows = load_examples(Path(report["out"]))

            for row in rows:
                paths = [Path(row["candidate_a"]), Path(row["candidate_b"]), Path(row["preview"])]
                for path in paths:
                    self.assertTrue(path.is_file(), path)
                    self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(len({str(path) for path in paths}), 3)
            referenced = {Path(row[key]).name for row in rows for key in ("candidate_a", "candidate_b", "preview")}
        self.assertEqual(report["counts"]["subgraph_images"], len(referenced))

    def test_the_discipline_is_re_derived_from_the_written_rows(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = build_version(root / "version", {
                "p0001": {"split": "train", "domain": "cv"},
                "p0002": {"split": "train", "domain": "biology"},
                "p0003": {"split": "validation", "domain": "materials", "shape": "fork"},
                "p0004": {"split": "validation", "domain": "rag"},
                "p0005": {"split": "hidden_test", "domain": "agent"},
            })
            report = build_subgraph_pairs(version, root / "out", include_swapped=False)
            rows = load_examples(Path(report["out"]))

        discipline = report["discipline"]
        self.assertTrue(discipline["ok"])
        self.assertEqual(discipline["candidate_split_leaks"], [])
        self.assertEqual(discipline["hidden_test_candidates"], [])
        self.assertEqual(discipline["same_paper_candidates"], [])
        self.assertGreater(discipline["cross_paper_sides"], 0)
        self.assertEqual(set(row["split"] for row in rows), {"train", "validation"})
        papers = {row["provenance"]["query"]["paper_id"] for row in rows}
        self.assertNotIn("p0005", papers)
        for row in rows:
            for side in ("chosen", "rejected"):
                self.assertNotEqual(row["provenance"][side]["paper_id"], "p0005")

    def test_a_same_domain_candidate_is_allowed(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = build_version(root / "version", {
                "p0001": {"split": "train", "domain": "cv"},
                "p0002": {"split": "train", "domain": "cv"},
                "p0003": {"split": "train", "domain": "cv", "shape": "fork"},
            })
            report = build_subgraph_pairs(version, root / "out", split="train", include_swapped=False)
            rows = load_examples(Path(report["out"]))

        self.assertTrue(rows)
        self.assertEqual(report["discipline"]["same_domain_sides"], 2 * len(rows))
        self.assertEqual(report["discipline"]["cross_domain_sides"], 0)

    def test_a_query_without_a_match_is_refused_with_one_recorded_reason(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = build_version(root / "version", {
                "p0001": {"split": "train", "domain": "cv"},
                "p0002": {"split": "train", "domain": "cv"},
                "p0003": {"split": "validation", "domain": "rag"},
            })
            report = build_subgraph_pairs(version, root / "out")
            rows = load_examples(Path(report["out"]))

        reasons = report["exclusions"]["by_reason"]
        self.assertEqual(list(reasons), ["query_split_has_no_candidate"])
        self.assertTrue(all(reason in EXCLUDED_REASONS for reason in reasons))
        refused = {item["paper_id"] for item in report["exclusions"]["sample"]}
        self.assertEqual(refused, {"p0003"})
        self.assertTrue(all(row["split"] == "train" for row in rows))
        self.assertEqual(report["counts"]["queries_paired"], report["counts"]["queries_considered"] - report["exclusions"]["count"])

    def test_an_empty_result_is_a_failure_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = build_version(root / "version", {"p0001": {"split": "train", "domain": "cv"}})
            with self.assertRaises(SourceUnavailable):
                build_subgraph_pairs(version, root / "out")

        self.assertFalse((root / "out" / "rag_subgraph_examples.jsonl").exists())

    def test_the_limit_counts_queries_and_the_swap_is_optional(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = self.version(root / "version")
            report = build_subgraph_pairs(version, root / "out", split="train", limit=2, include_swapped=False)
            rows = load_examples(Path(report["out"]))

        self.assertEqual(report["counts"]["queries_paired"], 2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(report["counts"]["swapped"], 0)

    def test_a_domain_cap_stops_one_domain_without_dropping_the_others(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = self.version(root / "version")
            report = build_subgraph_pairs(version, root / "out", split="train", per_domain=1, include_swapped=False)
            rows = load_examples(Path(report["out"]))

        by_domain: dict[str, int] = {}
        for row in rows:
            domain = str(row.get("domain") or "")
            by_domain[domain] = by_domain.get(domain, 0) + 1
        self.assertTrue(by_domain)
        self.assertTrue(all(count <= 1 for count in by_domain.values()))
        self.assertLessEqual(report["counts"]["queries_paired"], len(report["catalogue"]["domains"]))
        self.assertEqual(report["counts"]["per_domain"], 1)

    def test_a_rebuild_writes_byte_identical_examples_and_images(self):
        """An example names the directory it was built into, so the rebuild is repeated there."""

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = self.version(root / "version")
            target = root / "out"

            def digest(directory: Path) -> dict[str, str]:
                return {
                    str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(directory.rglob("*"))
                    if path.is_file() and path.suffix in {".jsonl", ".png"}
                }

            first = build_subgraph_pairs(version, target, split="train")
            before = digest(target)
            shutil.rmtree(target)
            second = build_subgraph_pairs(version, target, split="train")
            after = digest(target)

        self.assertTrue(before)
        self.assertEqual(before, after)
        self.assertEqual(first["sources"]["examples_sha256"], second["sources"]["examples_sha256"])

    def test_the_report_names_the_encoder_that_embedded_the_index(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            version = self.version(root / "version")
            report = build_subgraph_pairs(version, root / "out", split="train")

        self.assertEqual(report["strict_structure_rule"], "same motif with identical node and edge counts")
        self.assertEqual(report["recall"]["strategy"], "vector_recall_then_strict_structure_match")
        self.assertTrue(report["catalogue"]["locators"] >= report["catalogue"]["usable_subgraphs"])
        self.assertEqual(report["catalogue"]["missing_graphs"], [])
        self.assertEqual(report["discipline"]["cross_paper_sides"] > 0, True)


class SubgraphRenderTests(unittest.TestCase):
    def one_subgraph(self):
        graph = chain("p0001", "scientific")
        graph["nodes"].append(node("m4", "Loss"))
        subgraph = {
            "subgraph_id": "subgraph.p0001.scientific.m1.aaaa",
            "node_ids": ["m1", "m2", "m3"],
            "edge_ids": ["e1", "e2"],
            "motif": "linear",
            "properties": {"center_node_id": "m2"},
        }
        return subgraph, graph

    def test_the_same_subgraph_always_renders_the_same_bytes(self):
        subgraph, graph = self.one_subgraph()
        with tempfile.TemporaryDirectory() as name:
            first = Path(render_subgraph(subgraph, graph, Path(name) / "a.png"))
            second = Path(render_subgraph(subgraph, graph, Path(name) / "b.png"))

            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_an_unreachable_node_gets_its_own_column_instead_of_overlapping(self):
        subgraph, graph = self.one_subgraph()
        subgraph["node_ids"] = ["m1", "m2", "m3", "m4"]
        subgraph["edge_ids"] = ["e1", "e2"]
        with tempfile.TemporaryDirectory() as name:
            out = Path(render_subgraph(subgraph, graph, Path(name) / "spread.png", width=200, height=200))

            self.assertEqual(out.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(out.stat().st_size, 0)

    def test_a_subgraph_without_a_stored_graph_still_renders_its_nodes(self):
        subgraph, _ = self.one_subgraph()
        with tempfile.TemporaryDirectory() as name:
            out = Path(render_subgraph(subgraph, None, Path(name) / "no-graph.png"))

            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
