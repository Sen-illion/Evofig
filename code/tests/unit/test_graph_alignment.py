"""Tests for cross-layer graph alignment and the Grounding Gate."""

from __future__ import annotations

import unittest

from rfs.composition import build_editable_graph
from rfs.evaluation import align_graph_layers, evaluate_grounding_gate
from rfs.analysis.graph import build_scientific_graph
from rfs.planning import build_visual_graph, resolve_design_required_edge_ids
from tests.unit.test_editable_graph import _layers


class GraphAlignmentTests(unittest.TestCase):
    def test_required_stage_is_grounded_by_editable_panel(self) -> None:
        method = {"stages": [{"id": "stage_one", "label": "Stage One", "must_show": True, "evidence_ids": ["E1"]}], "modules": [{"id": "m", "label": "Module", "parent_id": "stage_one", "must_show": True, "evidence_ids": ["E2"]}]}
        scientific = build_scientific_graph(method, paper_id="P", evidence_map={"evidence": [{"id": "E1"}, {"id": "E2"}]})[0]
        design = {"panels": [{"id": "stage_one", "title": "Stage One", "bbox_percent": {"x": .05, "y": .05, "w": .9, "h": .9}}], "nodes": [{"id": "m", "semantic_source_id": "m", "panel_id": "stage_one", "bbox_percent": {"x": .2, "y": .2, "w": .2, "h": .2}}], "connectors": []}
        visual = build_visual_graph(design, scientific_graph=scientific)[0]
        program = {"panels": design["panels"], "semantic_nodes": design["nodes"], "cards": [], "slots": [], "labels": [], "groups": [], "arrows": [], "text_program": {"items": []}}
        editable, bindings, report = build_editable_graph(program, visual_graph=visual, scientific_graph=scientific)
        alignment = align_graph_layers(scientific, visual, editable, bindings)
        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(alignment["ok"], alignment["errors"])
        self.assertEqual(1.0, alignment["node_coverage"])

    def test_complete_required_chain_passes_grounding(self) -> None:
        scientific, visual, program = _layers()
        scientific["required_node_ids"] = [item["id"] for item in scientific["nodes"]]
        scientific["required_edge_ids"] = [item["id"] for item in scientific["edges"]]
        editable, bindings, report = build_editable_graph(program, visual_graph=visual, scientific_graph=scientific)
        self.assertTrue(report["ok"], report["errors"])
        alignment = align_graph_layers(scientific, visual, editable, bindings)
        gate = evaluate_grounding_gate(alignment)
        self.assertTrue(alignment["ok"], alignment["errors"])
        self.assertEqual("pass", gate["status"])
        self.assertEqual(1.0, alignment["node_coverage"])
        self.assertEqual(1.0, alignment["edge_coverage"])

    def test_missing_required_binding_blocks(self) -> None:
        scientific, visual, program = _layers()
        required = scientific["nodes"][0]["id"]
        scientific["required_node_ids"] = [required]
        editable, bindings, _ = build_editable_graph(program, visual_graph=visual, scientific_graph=scientific)
        binding = next(item for item in bindings["node_bindings"] if item["scientific_node_id"] == required)
        binding["status"] = "unbound"
        alignment = align_graph_layers(scientific, visual, editable, bindings)
        gate = evaluate_grounding_gate(alignment)
        self.assertFalse(alignment["ok"])
        self.assertEqual("blocked", gate["status"])
        self.assertEqual([required], alignment["missing_required_node_ids"])


    def test_the_relations_the_design_chose_are_the_ones_the_gate_audits(self) -> None:
        scientific, visual, program = _layers()
        design = {
            "nodes": program["semantic_nodes"],
            "connectors": [{"id": "vr", "source": "va", "target": "vb", "type": "data_flow"}],
        }
        resolved = resolve_design_required_edge_ids(design, scientific)
        self.assertEqual([item["id"] for item in scientific["edges"]], resolved["required_edge_ids"])
        scientific["required_edge_ids"] = list(resolved["required_edge_ids"])
        editable, bindings, _ = build_editable_graph(program, visual_graph=visual, scientific_graph=scientific)
        gate = evaluate_grounding_gate(align_graph_layers(scientific, visual, editable, bindings))
        self.assertEqual("pass", gate["status"])
        self.assertEqual(1.0, gate["edge_coverage"])
        # Dropping the very arrow the design selected leaves the relation unmet.
        dropped = {**program, "arrows": []}
        editable, bindings, _ = build_editable_graph(dropped, visual_graph=visual, scientific_graph=scientific)
        alignment = align_graph_layers(scientific, visual, editable, bindings)
        gate = evaluate_grounding_gate(alignment)
        self.assertEqual(resolved["required_edge_ids"], alignment["missing_required_edge_ids"])
        self.assertFalse(gate["checks"]["required_edges_grounded"])
        self.assertEqual("blocked", gate["status"])

if __name__ == "__main__":
    unittest.main()
