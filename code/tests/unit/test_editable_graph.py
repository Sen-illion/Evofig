"""Tests for editable PowerPoint object graph construction."""

from __future__ import annotations

import unittest

from rfs.analysis.graph import build_scientific_graph
from rfs.composition import build_editable_graph
from rfs.planning import build_visual_graph


def _layers() -> tuple[dict, dict, dict]:
    method = {
        "modules": [
            {"id": "a", "label": "A", "evidence_ids": ["E1"]},
            {"id": "b", "label": "B", "evidence_ids": ["E2"]},
        ],
        "relations": [{"id": "r", "source": "a", "target": "b", "type": "data_flow", "evidence_ids": ["E3"]}],
    }
    scientific = build_scientific_graph(method, paper_id="P", evidence_map={"evidence": [{"id": f"E{i}"} for i in range(1, 4)]})[0]
    design = {
        "panels": [
            {"id": "p1", "bbox_percent": {"x": 0, "y": 0, "w": .5, "h": 1}},
            {"id": "p2", "bbox_percent": {"x": .5, "y": 0, "w": .5, "h": 1}},
        ],
        "nodes": [
            {"id": "va", "semantic_source_id": "a", "panel_id": "p1", "bbox_percent": {"x": .1, "y": .2, "w": .2, "h": .2}},
            {"id": "vb", "semantic_source_id": "b", "panel_id": "p2", "bbox_percent": {"x": .7, "y": .2, "w": .2, "h": .2}},
        ],
        "connectors": [{"id": "vr", "source": "va", "target": "vb", "type": "data_flow"}],
    }
    visual = build_visual_graph(design, scientific_graph=scientific)[0]
    program = {
        "panels": design["panels"],
        "cards": [],
        "semantic_nodes": design["nodes"],
        "slots": [{"id": "slot_a", "node_id": "va", "panel_id": "p1", "bbox_percent": {"x": .12, "y": .3, "w": .1, "h": .1}}],
        "labels": [{"id": "label_a", "target_id": "va", "text": "A", "bbox_percent": {"x": .1, "y": .2, "w": .2, "h": .05}}],
        "groups": [],
        "arrows": [{"id": "vr", "source": "va", "target": "vb", "type": "data_flow", "editable_in": "pptx"}],
    }
    return scientific, visual, program


class EditableGraphTests(unittest.TestCase):
    def test_native_objects_and_connectors_are_graph_objects(self) -> None:
        scientific, visual, program = _layers()
        graph, bindings, report = build_editable_graph(program, visual_graph=visual, scientific_graph=scientific)
        self.assertTrue(report["ok"], report["errors"])
        self.assertGreaterEqual(len(graph["nodes"]), 6)
        self.assertEqual(1, len(graph["edges"]))
        self.assertEqual("pptx", graph["edges"][0]["properties"]["editable_in"])
        self.assertTrue(bindings["node_bindings"])
        self.assertTrue(bindings["edge_bindings"])

    def test_dangling_program_connector_fails_graph(self) -> None:
        scientific, visual, program = _layers()
        program["arrows"][0]["target"] = "missing"
        _, _, report = build_editable_graph(program, visual_graph=visual, scientific_graph=scientific)
        self.assertFalse(report["ok"])
        self.assertTrue(any("endpoint is not declared" in item for item in report["errors"]))


if __name__ == "__main__":
    unittest.main()
