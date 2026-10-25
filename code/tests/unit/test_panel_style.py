"""Tests for role-aware paper figure color planning."""

from __future__ import annotations

import unittest

from rfs.composition.color_application import apply_color_theme_to_program
from rfs.evaluation.color_quality import evaluate_color_quality
from rfs.evaluation.text_readability import evaluate_text_readability
from rfs.planning.panel_style import build_panel_style_plan


class PanelStyleTests(unittest.TestCase):
    def test_dark_reference_color_is_not_blindly_used_as_large_surface(self):
        plan = build_panel_style_plan(
            [{"id": "main"}],
            {"style": {"palette": ["#1F756F"]}},
            {"color_rules": ["reserve saturated colors for emphasis"]},
        )
        panel = plan["panels"][0]
        self.assertNotEqual(panel["fill_color_hex"], "#1F756F")
        self.assertEqual(panel["border_color_hex"], "#1F756F")
        self.assertGreaterEqual(panel["contrast_ratio"], 4.5)
        self.assertEqual(plan["exemplar_color_rules"], ["reserve saturated colors for emphasis"])

    def test_a_word_that_merely_looks_like_a_colour_is_replaced(self):
        # A model once answered a colour question with "No". The old check only
        # counted characters, so "#Nodata" reached the hex parser and took the
        # whole paper down with it.
        plan = build_panel_style_plan([{"id": "main"}], {"style": {"palette": ["#Nodata"]}})
        panel = plan["panels"][0]
        self.assertTrue(panel["fill_color_hex"].startswith("#"))
        self.assertEqual(len(panel["fill_color_hex"]), 7)
        int(panel["fill_color_hex"][1:], 16)

    def test_light_reference_color_remains_a_valid_surface(self):
        plan = build_panel_style_plan([{"id": "main"}], {"style": {"palette": ["#EEF4FA"]}})
        self.assertEqual(plan["panels"][0]["fill_color_hex"], "#EEF4FA")

    #: The theme iclr:2026:10006687 was compiled with: the header bands are
    #: mid-tone, and small white titles on them scored 3.97:1 and 4.43:1.
    MID_TONE_THEME = {
        "provenance": {"source": "safe_fallback"},
        "text": {"primary": "#193247", "secondary": "#526676", "muted": "#71808B", "inverse": "#FFFFFF"},
        "panels": {
            "panel_inputs": {"fill": "#EDF3F7", "border": "#477FA8", "header": "#477FA8", "header_text": "#FFFFFF", "body_text": "#193247"},
            "panel_feature_extraction": {"fill": "#F2F6F2", "border": "#4D8B7B", "header": "#4D8B7B", "header_text": "#FFFFFF", "body_text": "#193247"},
            "panel_stage_2": {"fill": "#F0EEF7", "border": "#6C72B8", "header": "#6C72B8", "header_text": "#FFFFFF", "body_text": "#193247"},
        },
    }

    def test_a_mid_tone_band_is_deepened_until_its_title_reads(self):
        panels = [{"id": panel_id} for panel_id in self.MID_TONE_THEME["panels"]]
        plan = build_panel_style_plan(panels, color_theme=self.MID_TONE_THEME)
        for panel in plan["panels"]:
            self.assertGreaterEqual(panel["header_contrast_ratio"], 4.5, panel["panel_id"])
            self.assertGreaterEqual(panel["contrast_ratio"], 4.5, panel["panel_id"])
        self.assertNotEqual(plan["panels"][1]["header_color_hex"], "#4D8B7B")

    def test_the_resolved_band_survives_the_compiled_figure_checks(self):
        panel_id = "panel_feature_extraction"
        theme = {**self.MID_TONE_THEME, "panels": {panel_id: dict(self.MID_TONE_THEME["panels"][panel_id])}}
        plan = build_panel_style_plan([{"id": panel_id}], color_theme=theme)
        program = {
            "canvas": {},
            "panels": [{"id": panel_id}],
            "semantic_nodes": [],
            "arrows": [],
            "text_program": {"items": [{"id": f"text_{panel_id}", "text": "Feature Extraction", "text_role": "panel_title", "target_id": panel_id, "panel_id": panel_id, "font_size_pt": 10.5}]},
        }
        compiled = apply_color_theme_to_program(program, plan)
        readability = evaluate_text_readability(compiled)
        self.assertEqual(readability["status"], "pass", readability["failures"])
        quality = evaluate_color_quality(compiled)
        self.assertEqual([issue for issue in quality["issues"] if "authoritative theme" in issue["reason"]], [])

    def test_a_fallback_band_gets_a_title_colour_that_reads(self):
        plan = build_panel_style_plan([{"id": "main"}], {"style": {"palette": ["#4D8B7B"]}})
        panel = plan["panels"][0]
        self.assertGreaterEqual(panel["header_contrast_ratio"], 4.5)
        self.assertEqual(panel["header_text_color_hex"], "#FFFFFF")


if __name__ == "__main__":
    unittest.main()
