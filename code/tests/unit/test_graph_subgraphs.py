"""Tests for deterministic graph motif and subgraph extraction."""

from __future__ import annotations

import unittest

from rfs.analysis.graph import classify_node_motif, extract_subgraphs, same_structure, structure_signature, subgraph_edges
from rfs.contracts import normalize_graph


def _graph(edges: list[tuple[str, str, str]], *, panels: dict[str, list[str]] | None = None, phases: dict[str, str] | None = None) -> dict:
    node_ids = sorted({value for edge in edges for value in edge[:2]})
    return normalize_graph({
        "graph_id": "g",
        "paper_id": "p",
        "kind": "scientific",
        "nodes": [{"id": item, "panel_ids": (panels or {}).get(item, []), "properties": {"phase": (phases or {}).get(item, "")}} for item in node_ids],
        "edges": [{"id": f"e{index}", "source": source, "target": target, "relation_type": kind} for index, (source, target, kind) in enumerate(edges)],
    })


class GraphSubgraphTests(unittest.TestCase):
    def test_extracts_linear_branch_merge_and_feedback(self) -> None:
        cases = [
            (_graph([("a", "b", "data"), ("b", "c", "data")]), "linear"),
            (_graph([("a", "b", "data"), ("a", "c", "data")]), "branch"),
            (_graph([("a", "c", "data"), ("b", "c", "data")]), "merge"),
            (_graph([("a", "b", "data"), ("b", "a", "feedback")]), "feedback"),
        ]
        for graph, expected in cases:
            with self.subTest(expected=expected):
                subgraphs, report = extract_subgraphs(graph)
                self.assertTrue(report["ok"], report["errors"])
                self.assertIn(expected, {item["motif"] for item in subgraphs})

    def test_extracts_training_inference_and_cross_panel(self) -> None:
        phased = _graph(
            [("train", "bridge", "data"), ("bridge", "infer", "data")],
            phases={"train": "training", "infer": "inference"},
        )
        subgraphs, _ = extract_subgraphs(phased)
        self.assertIn("training_inference", {item["motif"] for item in subgraphs})
        paneled = _graph([("a", "b", "data")], panels={"a": ["p1"], "b": ["p2"]})
        subgraphs, _ = extract_subgraphs(paneled)
        self.assertIn("cross_panel", {item["motif"] for item in subgraphs})


class StructureSignatureTests(unittest.TestCase):
    def _signature(self, edges, *, center="b"):
        graph = _graph(edges)
        subgraphs, _ = extract_subgraphs(graph)
        body = next(item for item in subgraphs if item["properties"]["center_node_id"] == center)
        return structure_signature(body, graph), body, graph

    def test_the_signature_reports_size_degrees_and_whether_a_loop_closes(self):
        signature, _, _ = self._signature([("a", "b", "data"), ("b", "c", "data")], center="b")

        self.assertEqual(signature["motif"], "linear")
        self.assertEqual((signature["node_count"], signature["edge_count"]), (3, 2))
        self.assertEqual(signature["center_in_degree"], 1)
        self.assertEqual(signature["center_out_degree"], 1)
        self.assertEqual(signature["degree_histogram"], {"0:1": 1, "1:0": 1, "1:1": 1})
        self.assertFalse(signature["has_cycle"])

        looped, _, _ = self._signature([("a", "b", "data"), ("b", "a", "feedback")], center="b")
        self.assertTrue(looped["has_cycle"])

    def test_the_signature_does_not_depend_on_a_materialized_body(self):
        graph = _graph([("a", "b", "data"), ("b", "c", "data")])
        subgraphs, _ = extract_subgraphs(graph)
        body = next(item for item in subgraphs if item["properties"]["center_node_id"] == "b")
        locator_like = {key: value for key, value in body.items() if key != "edges"}

        self.assertEqual(structure_signature(body, graph), structure_signature(locator_like, graph))
        self.assertEqual(len(subgraph_edges(locator_like, graph)), 2)

    def test_the_same_structure_rule_is_strict(self):
        chain, _, _ = self._signature([("a", "b", "data"), ("b", "c", "data")], center="b")
        same, _, _ = self._signature([("a", "b", "data"), ("b", "c", "data")], center="b")
        merge, _, _ = self._signature([("a", "c", "data"), ("b", "c", "data")], center="c")
        smaller, _, _ = self._signature([("a", "b", "data"), ("a", "c", "data")], center="a")

        self.assertTrue(same_structure(chain, same))
        self.assertFalse(same_structure(chain, merge))
        self.assertFalse(same_structure(chain, smaller))
        self.assertFalse(same_structure(chain, {"motif": "linear", "node_count": 3, "edge_count": 3}))

    def test_an_isolated_node_is_carried_as_unknown_rather_than_guessed(self):
        graph = normalize_graph({"graph_id": "g", "paper_id": "p", "kind": "scientific", "nodes": [{"id": "a"}], "edges": []})

        self.assertEqual([], extract_subgraphs(graph)[0])
        subgraphs, report = extract_subgraphs(graph, include_unknown=True)
        signature = structure_signature(subgraphs[0], graph)

        self.assertTrue(report["ok"])
        self.assertEqual(signature["motif"], "unknown")
        self.assertEqual((signature["node_count"], signature["edge_count"]), (1, 0))

    def test_a_drawing_the_reading_called_a_repetition_is_its_own_motif(self):
        edges = [
            {"id": "e1", "source": "a", "target": "b", "relation_type": "forward"},
            {"id": "e2", "source": "b", "target": "a", "relation_type": "forward"},
        ]
        repeated = {"id": "a", "properties": {"repeat_group": "pvcnn diffusion"}}

        self.assertEqual("repeated", classify_node_motif("a", edges, node=repeated))
        self.assertEqual("linear", classify_node_motif("b", edges))


if __name__ == "__main__":
    unittest.main()
