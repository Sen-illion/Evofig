"""Tests for global real-font text fitting."""

import unittest

from rfs.composition import fit_text_program_globally


class GlobalTextLayoutTests(unittest.TestCase):
    def test_long_text_is_measured_and_reduced_to_fit(self):
        program, report = fit_text_program_globally({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "text_program": {"items": [{"id": "long", "text": "A very long scientific framework label that requires global measurement", "font_size_pt": 18, "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.12, "h": 0.08}}]},
        })
        item = program["text_program"]["items"][0]
        self.assertLess(item["font_size_pt"], 18)
        self.assertEqual(item["fit_strategy"], "global_real_font_measurement")
        self.assertEqual(report["text_count"], 1)

    def test_narrow_long_text_bbox_is_expanded_before_wrapping(self):
        program, report = fit_text_program_globally({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "text_program": {"items": [{"id": "narrow", "text": "Retrieve relevant disease neighbors", "font_size_pt": 10, "bbox_percent": {"x": 0.9, "y": 0.2, "w": 0.015, "h": 0.16}}]},
        })
        item = program["text_program"]["items"][0]
        self.assertGreater(item["bbox_percent"]["w"], 0.015)
        self.assertEqual(report["expanded_bbox_count"], 1)
        self.assertEqual(report["vertical_wrap_risk_count"], 0)

    def test_a_label_is_shrunk_rather_than_broken_mid_word(self):
        """A word that fits the box at the readable floor must not be split in two."""

        program, report = fit_text_program_globally({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "text_program": {"items": [{
                "id": "node_label",
                "text": "Unstructured",
                "text_role": "node_label",
                "font_size_pt": 14,
                "max_lines": 2,
                "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.075, "h": 0.06},
            }]},
        })
        item = program["text_program"]["items"][0]
        self.assertEqual(item["measured_lines"], ["Unstructured"])
        self.assertLess(item["font_size_pt"], 14)
        self.assertEqual(report["word_break_count"], 0)

    def test_a_word_wider_than_its_box_is_reported_instead_of_hidden(self):
        """The box is what has to change, so the report names the word it cannot fit."""

        _program, report = fit_text_program_globally({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "semantic_nodes": [{"id": "n1", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.05, "h": 0.06}}],
            "text_program": {"items": [{
                "id": "node_label",
                "target_id": "n1",
                "text": "Unstructuredness",
                "text_role": "node_label",
                "font_size_pt": 14,
                "max_lines": 2,
                "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.03, "h": 0.06},
            }]},
        })
        self.assertEqual(report["word_break_count"], 1)
        self.assertEqual(report["word_break_text_ids"], ["node_label"])
        self.assertEqual(report["items"][0]["longest_word"], "Unstructuredness")

    def test_point_sizes_are_converted_to_pixels_during_measurement(self):
        program, _report = fit_text_program_globally({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "text_program": {"items": [{
                "id": "panel_title",
                "text": "Regulatory Outcomes & Summaries",
                "text_role": "panel_title",
                "font_size_pt": 16,
                "min_readable_size_pt": 14,
                "max_lines": 2,
                "bbox_percent": {"x": 0.78, "y": 0.05, "w": 0.186, "h": 0.08},
            }]},
        })
        item = program["text_program"]["items"][0]
        self.assertLessEqual(item["measured_line_count"], 2)
        self.assertLessEqual(item["font_size_pt"], 16)


if __name__ == "__main__":
    unittest.main()
