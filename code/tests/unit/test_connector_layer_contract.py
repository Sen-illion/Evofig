"""Regression tests for panel-title ownership and connector layer policy."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from rfs.composition.direct_connectors import resolve_direct_connectors
from rfs.composition.pptx import compile_ppt
from rfs.contracts.text_ownership import canonicalize_node_text_program


class ConnectorLayerContractTests(unittest.TestCase):
    def test_missing_panel_title_role_is_normalized_and_rendered_once(self):
        program, _ = canonicalize_node_text_program({
            "panels": [{"id": "panel_moe", "title": "Dual Space MoE", "bbox_percent": {"x": .05, "y": .05, "w": .9, "h": .8}}],
            "semantic_nodes": [],
            "text_program": {"items": [{"id": "panel_moe_title", "text": "Dual Space MoE", "target_id": "panel_moe", "bbox_percent": {"x": .05, "y": .05, "w": .9, "h": .05}}]},
        })
        item = program["text_program"]["items"][0]
        self.assertEqual(item["text_role"], "panel_title")
        with tempfile.TemporaryDirectory() as tmp:
            program.update({"canvas": {"width_in": 10.0, "height_in": 5.0}, "style": {"palette": ["#FFFFFF", "#4A90C2"]}, "cards": [], "slots": [], "arrows": [], "labels": [], "export_targets": [{"type": "pptx", "path": "editable_composition.pptx"}]})
            pptx = compile_ppt(program, Path(tmp))
            with zipfile.ZipFile(pptx) as archive:
                xml = archive.read("ppt/slides/slide1.xml").decode("utf-8")
            self.assertEqual(xml.count("Dual Space MoE"), 1)

    def test_data_flow_prefers_axis_aligned_route_over_clear_diagonal(self):
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .10, "w": .10, "h": .10}},
                {"id": "b", "bbox_percent": {"x": .70, "y": .60, "w": .10, "h": .10}},
            ],
            "arrows": [{"id": "edge", "source": "a", "target": "b", "relation_type": "data_flow"}],
            "text_program": {"items": []},
        }
        updated, _ = resolve_direct_connectors(program)
        selection = updated["arrows"][0]["route_selection"]
        self.assertNotEqual(selection["selected_candidate"], "direct")
        self.assertTrue(next(item for item in selection["candidates"] if item["candidate_id"] == selection["selected_candidate"])["axis_aligned"])


if __name__ == "__main__":
    unittest.main()
