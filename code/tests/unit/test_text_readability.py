"""Tests for editable figure text readability evaluation."""

from __future__ import annotations

import unittest

from rfs.evaluation.text_readability import evaluate_text_readability


class TextReadabilityTests(unittest.TestCase):
    def test_reports_low_contrast_and_tiny_text(self):
        program = {
            "panels": [{"id": "panel"}],
            "style": {"panel_styles": {"panel": {"fill_color": "#1F756F", "text_color": "#173042"}}},
            "text_program": {"items": [{"id": "label", "text": "Label", "panel_id": "panel", "font_size_pt": 6, "min_readable_size_pt": 8}]},
        }
        report = evaluate_text_readability(program)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual({item["reason"] for item in report["failures"]}, {"font_below_readable_minimum", "insufficient_text_contrast"})

    def test_accepts_readable_high_contrast_text(self):
        program = {
            "panels": [{"id": "panel"}],
            "style": {"panel_styles": {"panel": {"fill_color": "#E8F3F0", "text_color": "#173042"}}},
            "text_program": {"items": [{"id": "label", "text": "Label", "panel_id": "panel", "font_size_pt": 10, "min_readable_size_pt": 8}]},
        }
        self.assertEqual(evaluate_text_readability(program)["status"], "pass")

    def test_a_panel_title_is_measured_against_its_band(self):
        # The title is white on the coloured header band; the panel body is a
        # pale tint. Measuring the title against the body scored a correct
        # figure as unreadable and bought it a redo.
        styled = {"fill_color": "#EDF3F7", "header_color": "#477FA8", "header_text_color": "#FFFFFF", "text_color": "#193247"}
        program = {
            "panels": [{"id": "panel_attack_generation", "fill_color": "#EDF3F7"}],
            "style": {"panel_styles": {"panel_attack_generation": styled}},
            "text_program": {
                "items": [
                    {"id": "text_panel_attack_generation", "text": "Attack Generation", "panel_id": "panel_attack_generation", "font_size_pt": 16.0, "color_hex": "#FFFFFF", "font_size_role": "panel_title"},
                ]
            },
        }
        report = evaluate_text_readability(program)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["items"][0]["background"], "#477FA8")

    def test_body_text_is_still_measured_against_the_panel_fill(self):
        styled = {"fill_color": "#EDF3F7", "header_color": "#477FA8", "header_text_color": "#FFFFFF", "text_color": "#193247"}
        program = {
            "panels": [{"id": "panel_attack_generation", "fill_color": "#EDF3F7"}],
            "style": {"panel_styles": {"panel_attack_generation": styled}},
            "text_program": {
                "items": [
                    {"id": "text_body", "text": "Body", "panel_id": "panel_attack_generation", "font_size_pt": 12.0, "color_hex": "#FFFFFF", "font_size_role": "panel_body"},
                ]
            },
        }
        report = evaluate_text_readability(program)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["items"][0]["background"], "#EDF3F7")


if __name__ == "__main__":
    unittest.main()
