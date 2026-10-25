"""Tests for scientific-to-visual graph construction."""

from __future__ import annotations

import unittest

from rfs.analysis.graph import build_scientific_graph
from rfs.planning import build_visual_graph, resolve_design_required_edge_ids


def _scientific() -> dict:
    method = {
        "modules": [
            {"id": "a", "label": "A", "must_show": True, "evidence_ids": ["E1"]},
            {"id": "b", "label": "B", "must_show": True, "evidence_ids": ["E2"]},
        ],
        "relations": [{"id": "r", "source": "a", "target": "b", "type": "data_flow", "must_show": True, "evidence_ids": ["E3"]}],
    }
    return build_scientific_graph(method, paper_id="P", evidence_map={"evidence": [{"id": f"E{i}"} for i in range(1, 4)]})[0]


def _design() -> dict:
    return {
        "panels": [
            {"id": "p1", "title": "First", "bbox_percent": {"x": .02, "y": .05, "w": .45, "h": .9}},
            {"id": "p2", "title": "Second", "bbox_percent": {"x": .53, "y": .05, "w": .45, "h": .9}},
        ],
        "nodes": [
            {"id": "va", "semantic_source_id": "a", "label": "A", "panel_id": "p1", "bbox_percent": {"x": .1, "y": .2, "w": .2, "h": .2}, "evidence_ids": ["E1"]},
            {"id": "vb", "semantic_source_id": "b", "label": "B", "panel_id": "p2", "bbox_percent": {"x": .7, "y": .2, "w": .2, "h": .2}, "evidence_ids": ["E2"]},
        ],
        "connectors": [{"id": "vr", "source": "va", "target": "vb", "type": "data_flow", "evidence_ids": ["E3"]}],
    }


class VisualGraphTests(unittest.TestCase):
    def test_stage_contract_object_binds_to_same_id_panel(self) -> None:
        method = {"stages": [{"id": "stage_one", "label": "Stage One", "must_show": True, "evidence_ids": ["E1"]}], "modules": [{"id": "a", "label": "A", "parent_id": "stage_one", "evidence_ids": ["E1"]}]}
        scientific = build_scientific_graph(method, paper_id="P", evidence_map={"evidence": [{"id": "E1"}]})[0]
        design = {"panels": [{"id": "stage_one", "title": "Stage One", "bbox_percent": {"x": .05, "y": .05, "w": .9, "h": .9}}], "nodes": [{"id": "a", "semantic_source_id": "a", "panel_id": "stage_one", "bbox_percent": {"x": .2, "y": .2, "w": .2, "h": .2}}], "connectors": []}
        graph, report = build_visual_graph(design, scientific_graph=scientific)
        panel = next(item for item in graph["nodes"] if item["kind"] == "panel")
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(len(panel["scientific_node_ids"]), 1)
        self.assertTrue(panel["required"])

    def test_distinctive_required_label_can_bind_to_panel_title(self) -> None:
        method = {"modules": [{"id": "core", "label": "Core", "evidence_ids": ["E1"]}], "innovations": [{"id": "innovation_puriflow", "label": "PuriFlow", "must_show": True, "evidence_ids": ["E2"]}]}
        scientific = build_scientific_graph(method, paper_id="P", evidence_map={"evidence": [{"id": "E1"}, {"id": "E2"}]})[0]
        design = {"panels": [{"id": "pipeline", "title": "PuriFlow Purification Pipeline", "bbox_percent": {"x": .05, "y": .05, "w": .9, "h": .9}}], "nodes": [{"id": "core", "semantic_source_id": "core", "panel_id": "pipeline", "bbox_percent": {"x": .2, "y": .2, "w": .2, "h": .2}}], "connectors": []}
        graph, report = build_visual_graph(design, scientific_graph=scientific)
        panel = next(item for item in graph["nodes"] if item["kind"] == "panel")
        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(panel["required"])
        self.assertEqual(1, len(panel["scientific_node_ids"]))

    def test_panel_can_bind_enclosing_stage_and_named_innovation(self) -> None:
        method = {"stages": [{"id": "pipeline", "label": "PuriFlow Purification Pipeline", "must_show": True, "evidence_ids": ["E1"]}], "modules": [{"id": "core", "label": "Core", "parent_id": "pipeline", "evidence_ids": ["E1"]}], "innovations": [{"id": "innovation_puriflow", "label": "PuriFlow", "must_show": True, "evidence_ids": ["E2"]}]}
        scientific = build_scientific_graph(method, paper_id="P", evidence_map={"evidence": [{"id": "E1"}, {"id": "E2"}]})[0]
        design = {"panels": [{"id": "pipeline", "title": "PuriFlow Purification Pipeline", "bbox_percent": {"x": .05, "y": .05, "w": .9, "h": .9}}], "nodes": [{"id": "core", "semantic_source_id": "core", "panel_id": "pipeline", "bbox_percent": {"x": .2, "y": .2, "w": .2, "h": .2}}], "connectors": []}
        graph = build_visual_graph(design, scientific_graph=scientific)[0]
        panel = next(item for item in graph["nodes"] if item["kind"] == "panel")
        self.assertEqual(2, len(panel["scientific_node_ids"]))

    def test_visual_nodes_and_edges_bind_to_science(self) -> None:
        graph, report = build_visual_graph(_design(), scientific_graph=_scientific())
        self.assertTrue(report["ok"], report["errors"])
        semantic_nodes = [item for item in graph["nodes"] if item["kind"] == "visual_node"]
        self.assertEqual(2, len(semantic_nodes))
        self.assertTrue(all(len(item["scientific_node_ids"]) == 1 for item in semantic_nodes))
        self.assertEqual(1, len(graph["edges"][0]["scientific_edge_ids"]))
        self.assertTrue(all(item["required"] for item in semantic_nodes))

    def test_unknown_semantic_source_fails(self) -> None:
        design = _design()
        design["nodes"][0]["semantic_source_id"] = "invented"
        _, report = build_visual_graph(design, scientific_graph=_scientific())
        self.assertFalse(report["ok"])
        self.assertEqual(["invented"], report["unknown_semantic_node_ids"])


    def test_design_selected_relation_becomes_a_required_scientific_edge(self) -> None:
        scientific = _scientific()
        design = _design()
        design["connectors"][0]["paper_relation_id"] = "r"
        resolved = resolve_design_required_edge_ids(design, scientific)
        self.assertEqual([scientific["edges"][0]["id"]], resolved["required_edge_ids"])
        self.assertEqual(["vr"], resolved["matched_connector_ids"])
        self.assertEqual([], resolved["unmatched_connector_ids"])
        self.assertEqual("paper_relation_id", resolved["match_sources"]["vr"])

    def test_connector_without_a_paper_relation_matches_by_endpoint_pair(self) -> None:
        scientific = _scientific()
        resolved = resolve_design_required_edge_ids(_design(), scientific)
        self.assertEqual([scientific["edges"][0]["id"]], resolved["required_edge_ids"])
        self.assertEqual("source_node_contract_pair", resolved["match_sources"]["vr"])

    def test_a_connector_the_paper_never_claimed_stays_unmatched(self) -> None:
        scientific = _scientific()
        design = _design()
        design["connectors"].append({"id": "ghost", "source": "vb", "target": "va", "type": "related_to"})
        resolved = resolve_design_required_edge_ids(design, scientific)
        self.assertEqual(["ghost"], resolved["unmatched_connector_ids"])
        self.assertEqual([scientific["edges"][0]["id"]], resolved["required_edge_ids"])

    def test_a_short_stage_title_binds_when_it_equals_the_panel_title(self) -> None:
        method = {"stages": [{"id": "stage_tasks", "label": "Tasks", "must_show": True, "evidence_ids": ["E1"]}], "modules": [{"id": "a", "label": "Align", "parent_id": "stage_tasks", "evidence_ids": ["E1"]}]}
        scientific = build_scientific_graph(method, paper_id="P", evidence_map={"evidence": [{"id": "E1"}]})[0]
        design = {
            "panels": [{"id": "panel_stage_tasks", "title": "Tasks", "bbox_percent": {"x": .05, "y": .05, "w": .9, "h": .9}}],
            "nodes": [{"id": "a", "semantic_source_id": "a", "panel_id": "panel_stage_tasks", "bbox_percent": {"x": .2, "y": .2, "w": .2, "h": .2}}],
            "connectors": [],
        }
        graph, report = build_visual_graph(design, scientific_graph=scientific)
        panel = next(item for item in graph["nodes"] if item["kind"] == "panel")
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(1, len(panel["scientific_node_ids"]))
        self.assertTrue(panel["required"])

    def test_a_short_stage_token_does_not_bind_as_a_substring(self) -> None:
        method = {"stages": [{"id": "stage_data", "label": "Data", "must_show": True, "evidence_ids": ["E1"]}], "modules": [{"id": "a", "label": "A", "parent_id": "stage_data", "evidence_ids": ["E1"]}]}
        scientific = build_scientific_graph(method, paper_id="P", evidence_map={"evidence": [{"id": "E1"}]})[0]
        design = {
            "panels": [{"id": "pipeline", "title": "Database Pipeline", "bbox_percent": {"x": .05, "y": .05, "w": .9, "h": .9}}],
            "nodes": [{"id": "a", "semantic_source_id": "a", "panel_id": "pipeline", "bbox_percent": {"x": .2, "y": .2, "w": .2, "h": .2}}],
            "connectors": [],
        }
        graph, report = build_visual_graph(design, scientific_graph=scientific)
        panel = next(item for item in graph["nodes"] if item["kind"] == "panel")
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual([], panel["scientific_node_ids"])

if __name__ == "__main__":
    unittest.main()
