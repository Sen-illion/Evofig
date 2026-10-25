"""Tests for geometry-aware node and connector typography."""

import unittest

from rfs.composition import optimize_direct_typography


class DirectTypographyTests(unittest.TestCase):
    def test_node_font_is_selected_from_role_range_instead_of_forced_to_ten(self):
        program, report = optimize_direct_typography({
            "canvas": {"width_in": 13.5, "height_in": 7.5},
            "semantic_nodes": [{
                "id": "core", "label": "Core Agent", "font_size_role": "primary_node", "bold": True,
                "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.1},
                "label_bbox_percent": {"x": 0.16, "y": 0.11, "w": 0.13, "h": 0.08},
            }],
            "arrows": [], "text_program": {"items": []},
        })
        node = program["semantic_nodes"][0]
        self.assertGreaterEqual(node["font_size_pt"], 12)
        self.assertEqual(report["status"], "pass")

    def test_connector_label_uses_actual_longest_segment(self):
        program, report = optimize_direct_typography({
            "canvas": {"width_in": 13.5, "height_in": 7.5}, "semantic_nodes": [],
            "arrows": [{"id": "edge", "path_percent": [[0.1, 0.2], [0.5, 0.2]]}],
            "text_program": {"items": [{"id": "label", "role": "connector_label", "connector_id": "edge", "text": "transfers evidence"}]},
        })
        item = program["text_program"]["items"][0]
        self.assertEqual(item["font_size_pt"], 10)
        self.assertGreater(report["connector_labels"][0]["segment_length_px"], 400)


    def test_a_lane_narrower_than_one_word_is_not_a_fit(self):
        """Counting lines alone called an unbreakable word a fit; the deck then split it."""

        program, report = optimize_direct_typography({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "semantic_nodes": [{
                "id": "input_docs", "label": "Unstructured", "font_size_role": "primary_node", "bold": True,
                "bbox_percent": {"x": 0.078, "y": 0.25, "w": 0.1011, "h": 0.1033},
                "label_bbox_percent": {"x": 0.1244, "y": 0.2583, "w": 0.0467, "h": 0.0868},
            }],
            "arrows": [], "text_program": {"items": []},
        })
        node = program["semantic_nodes"][0]
        self.assertEqual(node["typography_status"], "requires_relayout")
        self.assertEqual(report["unresolved_node_count"], 1)
        self.assertEqual(report["status"], "needs_relayout")


if __name__ == "__main__":
    unittest.main()
