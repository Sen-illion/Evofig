"""Tests for the deterministic graph context the structure planner reads."""

from __future__ import annotations

import unittest

from rfs.planning.graph_constraints import (
    check_design_against_graph,
    compact_graph_context,
    graph_context,
)


METHOD = {
    "figure_goal": "Show the pipeline",
    "inputs": [{"id": "input", "label": "Paper PDF", "must_show": True, "evidence_ids": ["E1"]}],
    "modules": [
        {"id": "encode", "label": "Encoder", "must_show": True, "evidence_ids": ["E2"]},
        {"id": "decode", "label": "Decoder", "should_render": True, "evidence_ids": ["E3"]},
    ],
    "outputs": [{"id": "result", "label": "Figure", "evidence_ids": ["E4"]}],
    "relations": [
        {"id": "r1", "source": "input", "target": "encode", "type": "data_flow", "must_show": True, "evidence_ids": ["E5"]},
        {"id": "r2", "source": "encode", "target": "decode", "type": "data_flow", "must_show": True, "evidence_ids": ["E6"]},
        {"id": "r3", "source": "decode", "target": "result", "type": "data_flow", "evidence_ids": ["E7"]},
    ],
}


def _design(*, nodes: list[str], edges: list[tuple[str, str]]) -> dict:
    """A design plan in contract-id space, with node ids equal to contract ids."""
    bbox = {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.15}
    return {
        "panels": [
            {"id": "p1", "bbox_percent": {"x": 0.05, "y": 0.15, "w": 0.4, "h": 0.4}},
            {"id": "p2", "bbox_percent": {"x": 0.55, "y": 0.15, "w": 0.4, "h": 0.4}},
        ],
        "nodes": [
            {"id": item, "semantic_source_id": item, "panel_id": "p1", "bbox_percent": dict(bbox)}
            for item in nodes
        ],
        "connectors": [
            {"id": "c%d" % index, "source": source, "target": target}
            for index, (source, target) in enumerate(edges)
        ],
    }


class GraphContextTests(unittest.TestCase):
    def test_the_summary_is_deterministic_and_stated_in_contract_ids(self):
        first = graph_context(METHOD, paper_id="paper-1")
        second = graph_context(METHOD, paper_id="paper-1")
        self.assertEqual(first, second)
        self.assertEqual(
            {item["id"] for item in first["nodes"]},
            {"input", "encode", "decode", "result"},
        )
        self.assertFalse([item for item in first["nodes"] if str(item["id"]).startswith("snode")])

    def test_required_nodes_and_relations_are_listed(self):
        context = graph_context(METHOD, paper_id="paper-1")
        self.assertEqual(sorted(context["required"]["node_ids"]), ["decode", "encode", "input"])
        self.assertEqual(
            context["required"]["edge_ids"],
            [["input", "encode", "data_flow"], ["encode", "decode", "data_flow"]],
        )
        self.assertEqual(context["counts"]["required_node_count"], 3)
        self.assertEqual(context["counts"]["required_edge_count"], 2)
        self.assertEqual(context["errors"], [])

    def test_motifs_are_reported_for_the_papers_chain(self):
        context = graph_context(METHOD, paper_id="paper-1")
        self.assertIn("linear", context["motif_histogram"])
        by_id = {item["id"]: item for item in context["nodes"]}
        self.assertEqual(by_id["encode"]["motif"], "linear")
        self.assertEqual(by_id["input"]["motif"], "multi_output")


class CompactContextTests(unittest.TestCase):
    CONTEXT = {
        "nodes": [
            {"id": "req1", "label": "Required", "role": "module", "phase": "", "required": True, "motif": "linear"},
            {"id": "hub", "label": "Hub", "role": "module", "phase": "", "required": False, "motif": "branch"},
            {"id": "leafA", "label": "Leaf A", "role": "module", "phase": "", "required": False, "motif": "linear"},
            {"id": "leafZ", "label": "Leaf Z", "role": "module", "phase": "", "required": False, "motif": "linear"},
        ],
        "edges": [
            ["req1", "hub", "data_flow", True],
            ["hub", "leafA", "data_flow", True],
            ["hub", "leafZ", "data_flow", False],
        ],
        "motif_histogram": {"branch": 1, "linear": 3},
        "counts": {"node_count": 4, "edge_count": 3, "required_node_count": 1, "required_edge_count": 2},
    }

    def test_a_wide_budget_carries_every_item(self):
        text = compact_graph_context(self.CONTEXT, max_chars=100000)
        for token in ("req1", "hub", "leafA", "leafZ", "motifs="):
            self.assertIn(token, text)

    def test_a_tight_budget_keeps_the_required_item_first(self):
        full = compact_graph_context(self.CONTEXT, max_chars=100000)
        header, *lines = full.splitlines()
        required_line = next(line for line in lines if "req1" in line)
        tight = compact_graph_context(self.CONTEXT, max_chars=len(header) + len(required_line) + 2)
        self.assertIn("req1", tight)
        self.assertNotIn("hub", tight)
        self.assertNotIn("leafA", tight)
        self.assertNotIn("leafZ", tight)

    def test_degree_decides_which_optional_item_is_dropped_next(self):
        full = compact_graph_context(self.CONTEXT, max_chars=100000)
        header, *lines = full.splitlines()
        required_line = next(line for line in lines if "req1" in line)
        hub_line = next(line for line in lines if "hub" in line and "leaf" not in line)
        budget = len(header) + len(required_line) + len(hub_line) + 4
        tighter = compact_graph_context(self.CONTEXT, max_chars=budget)
        self.assertIn("hub", tighter)
        self.assertNotIn("leafA", tighter)
        self.assertNotIn("leafZ", tighter)

    def test_a_context_with_no_nodes_renders_nothing(self):
        self.assertEqual(compact_graph_context({}, max_chars=1000), "")
        self.assertEqual(compact_graph_context({"nodes": []}, max_chars=1000), "")


class DesignCoverageTests(unittest.TestCase):
    def setUp(self):
        self.context = graph_context(METHOD, paper_id="paper-1")

    def test_a_design_that_carries_every_required_item_passes(self):
        design = _design(nodes=["input", "encode", "decode", "result"], edges=[("input", "encode"), ("encode", "decode"), ("decode", "result")])
        check = check_design_against_graph(design, self.context)
        self.assertTrue(check["ok"], check["errors"])
        self.assertEqual(check["missing_required_node_ids"], [])
        self.assertEqual(check["missing_required_edge_ids"], [])
        self.assertEqual(check["node_coverage"], 1.0)
        self.assertEqual(check["edge_coverage"], 1.0)

    def test_a_design_missing_a_required_module_is_reported(self):
        design = _design(nodes=["input", "encode", "result"], edges=[("input", "encode"), ("encode", "result")])
        check = check_design_against_graph(design, self.context)
        self.assertFalse(check["ok"])
        self.assertEqual(check["missing_required_node_ids"], ["decode"])
        self.assertIn("decode", check["errors"][0])

    def test_a_design_missing_a_required_relation_is_reported(self):
        design = _design(nodes=["input", "encode", "decode", "result"], edges=[("input", "encode"), ("decode", "result")])
        check = check_design_against_graph(design, self.context)
        self.assertFalse(check["ok"])
        self.assertEqual(check["missing_required_edge_ids"], [["encode", "decode", "data_flow"]])
        self.assertIn("encode->decode", check["errors"][0])
        self.assertEqual(check["node_coverage"], 1.0)

    def test_nothing_required_means_full_coverage(self):
        empty = graph_context({}, paper_id="paper-1")
        check = check_design_against_graph(_design(nodes=["a", "b"], edges=[("a", "b")]), empty)
        self.assertTrue(check["ok"])
        self.assertEqual(check["node_coverage"], 1.0)
        self.assertEqual(check["edge_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
