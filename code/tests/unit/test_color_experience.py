"""Tests for color recipe compression, adoption, and propagation."""

from __future__ import annotations

import unittest

from rfs.contracts.color_theme import normalize_color_theme, validate_color_theme
from rfs.contracts.vlm_context import compact_color_experience
from rfs.evaluation.color_experience import evaluate_color_experience_usage


class ColorExperienceTests(unittest.TestCase):
    def test_compaction_is_bounded(self):
        brief = {"color_experience": {"selected_recipes": [{"recipe_id": f"r{i}"} for i in range(5)]}}
        self.assertEqual([item["recipe_id"] for item in compact_color_experience(brief)["selected_recipes"]], ["r0", "r1", "r2"])

    def test_adoption_must_account_for_every_recipe(self):
        theme = normalize_color_theme({
            "canvas_background": "#FFFFFF",
            "text": {"primary": "#111111", "secondary": "#222222", "muted": "#555555", "inverse": "#FFFFFF"},
            "connectors": {"primary": "#111111", "secondary": "#333333", "feedback": "#555555"},
            "panels": {"p1": {"fill": "#EEEEEE", "border": "#333333", "header": "#333333", "header_text": "#FFFFFF", "body_text": "#111111"}},
            "role_mapping": {"module": "process"},
            "roles": {"process": {"fill": "#EEEEEE", "border": "#333333", "text": "#111111"}},
            "experience_adoption": {"adopted": [{"recipe_id": "r1", "principle": "pale surfaces", "applied_to": ["panel:p1"]}], "rejected": []},
        })
        result = validate_color_theme(theme, {"p1"}, {"module"}, {"r1", "r2"})
        self.assertFalse(result["ok"])
        self.assertTrue(any("r2" in value for value in result["errors"]))

    def test_usage_report_detects_exact_program_propagation(self):
        theme = normalize_color_theme({"experience_adoption": {"adopted": [], "rejected": [{"recipe_id": "r1", "reason": "incompatible background"}]}})
        program = {"style": {"color_theme": theme, "panel_styles": {}}}
        report = evaluate_color_experience_usage(theme, program, supplied_recipe_ids=["r1"])
        self.assertTrue(report["ok"])
        self.assertTrue(report["theme_propagated_to_figure_program"])


if __name__ == "__main__":
    unittest.main()
