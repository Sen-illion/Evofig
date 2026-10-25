"""Tests for single-owner editable node text."""

import unittest

from rfs.contracts.text_ownership import canonicalize_node_text_program, node_text_owned_by_program


class CanonicalTextOwnershipTests(unittest.TestCase):
    def test_panel_heading_returned_as_the_whole_panel_moves_into_the_header_lane(self):
        program, _ = canonicalize_node_text_program({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "panels": [{"id": "panel_intrinsic", "title": "Intrinsic Bias Measurement", "bbox_percent": {"x": .2, "y": .12, "w": .3, "h": .3}}],
            "semantic_nodes": [],
            "text_program": {"items": [{"id": "panel_intrinsic", "text": "Intrinsic Bias Measurement", "role": "panel_title", "bbox_percent": {"x": .2, "y": .12, "w": .3, "h": .3}, "font_size_pt": 16}]},
        })
        title = program["text_program"]["items"][0]
        self.assertEqual(title["target_id"], "panel_intrinsic")
        self.assertEqual(title["bbox_percent"], {"x": .2, "y": .12, "w": .3, "h": round(0.34 / 7.5, 4)})
        self.assertEqual(title["max_lines"], 1)
        self.assertEqual(title["font_size_pt"], 16.0)

    def test_panel_heading_is_shrunk_to_fit_a_narrow_lane_and_may_wrap_once(self):
        program, _ = canonicalize_node_text_program({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "panels": [{"id": "p", "title": "Fine-grained Attribution and Question Answering Generation", "bbox_percent": {"x": .05, "y": .12, "w": .2, "h": .4}}],
            "semantic_nodes": [],
            "text_program": {"items": [{"id": "p", "text": "Fine-grained Attribution and Question Answering Generation", "role": "panel_title", "bbox_percent": {"x": .05, "y": .12, "w": .2, "h": .4}, "font_size_pt": 16}]},
        })
        title = program["text_program"]["items"][0]
        self.assertLess(float(title["font_size_pt"]), 16.0)
        self.assertEqual(title["max_lines"], 2)

    def test_panel_heading_is_still_bound_when_the_planner_drops_the_target_id(self):
        program, _ = canonicalize_node_text_program({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "panels": [{"id": "panel_inputs", "title": "Input and Grounding", "bbox_percent": {"x": .01, "y": .12, "w": .16, "h": .82}}],
            "semantic_nodes": [],
            "text_program": {"items": [{"id": "heading", "text": "Input and Grounding", "text_role": "panel_title", "target_id": "", "bbox_percent": {"x": .01, "y": .12, "w": .16, "h": .82}, "font_size_pt": 16}]},
        })
        title = program["text_program"]["items"][0]
        self.assertEqual(title["panel_id"], "panel_inputs")
        self.assertLess(float(title["bbox_percent"]["h"]), 0.1)

    def test_duplicate_and_nonstandard_node_roles_collapse_to_one_owner(self):
        program, report = canonicalize_node_text_program({
            "semantic_nodes": [{"id": "agent", "semantic_source_id": "agent", "label": "Agent", "bbox_percent": {"x": .1, "y": .1, "w": .2, "h": .1}, "label_bbox_percent": {"x": .16, "y": .12, "w": .12, "h": .06}}],
            "text_program": {"items": [
                {"id": "vlm_agent", "text": "Agent", "role": "node", "semantic_source_id": "agent", "bbox_percent": {"x": .1, "y": .1, "w": .2, "h": .1}},
                {"id": "duplicate", "text": "Agent", "text_role": "node_label", "target_id": "agent", "bbox_percent": {"x": .1, "y": .1, "w": .2, "h": .1}},
            ]},
        })
        labels = [item for item in program["text_program"]["items"] if item.get("text_role") == "node_label"]
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["owner_node_id"], "agent")
        self.assertEqual(labels[0]["bbox_percent"], program["semantic_nodes"][0]["label_bbox_percent"])
        self.assertTrue(node_text_owned_by_program(program, program["semantic_nodes"][0]))
        self.assertEqual(report["owned_node_count"], 1)


if __name__ == "__main__":
    unittest.main()
