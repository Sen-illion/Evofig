"""Tests for the deterministic reference-measured fallback color palette."""

from __future__ import annotations

import unittest

from rfs.evaluation.color_plan_validation import validate_color_plan
from rfs.evaluation.text_readability import _contrast
from rfs.planning.color_theme_planner import (
    REFERENCE_FALLBACK_FAMILIES,
    REFERENCE_FALLBACK_TEXT,
    plan_color_theme_vlm,
)

#: Fills the fallback is allowed to use, each one measured from the pixels of
#: the curated excellent-figure library (``reference-figures/excellent``).
MEASURED_REFERENCE_FILLS = {"#DCEAF7", "#E3F0D8", "#FCEFD2", "#F6DCE3"}


class FallbackColorThemeTests(unittest.TestCase):
    def _plan(self, panel_ids, node_roles):
        design = {"panels": [{"id": pid} for pid in panel_ids], "nodes": [{"role": role} for role in node_roles]}
        return plan_color_theme_vlm({}, design, {"items": []}, mode="fallback")

    def test_fallback_is_the_reference_measured_palette(self):
        theme, report = self._plan(["p1", "p2", "p3", "p4"], ["input", "module", "output"])
        self.assertEqual("reference_measured_fallback", theme["theme_name"])
        self.assertTrue(report["fallback_used"])
        self.assertEqual("#F6F7F8", theme["canvas_background"])
        # Families cycle in the documented order so adjacent panels differ.
        for index, panel_id in enumerate(["p1", "p2", "p3", "p4"]):
            family = REFERENCE_FALLBACK_FAMILIES[index % len(REFERENCE_FALLBACK_FAMILIES)]
            self.assertEqual(family["fill"], theme["panels"][panel_id]["fill"])
            self.assertEqual(family["border"], theme["panels"][panel_id]["border"])

    def test_every_fill_traces_back_to_a_measured_reference_color(self):
        self.assertEqual(
            MEASURED_REFERENCE_FILLS,
            {family["fill"] for family in REFERENCE_FALLBACK_FAMILIES},
        )

    def test_fallback_passes_the_color_plan_gate(self):
        panel_ids = ["p1", "p2", "p3", "p4"]
        node_roles = {"input", "module", "output"}
        theme, _report = self._plan(panel_ids, sorted(node_roles))
        result = validate_color_plan(theme, expected_panel_ids=set(panel_ids), expected_node_roles=node_roles)
        self.assertTrue(result["ok"], result["errors"])
        self.assertLessEqual(
            result["metrics"]["structural_hue_count"],
            theme["usage_policy"]["max_distinct_hues"],
        )

    def test_fallback_text_contrast_holds_on_every_surface(self):
        panel_ids = ["p1", "p2", "p3", "p4"]
        theme, _report = self._plan(panel_ids, ["input", "module", "output"])
        for panel_id in panel_ids:
            panel = theme["panels"][panel_id]
            self.assertGreaterEqual(_contrast(panel["body_text"], panel["fill"]), 4.5)
            self.assertGreaterEqual(_contrast(panel["header_text"], panel["header"]), 3.0)

    def test_unknown_role_falls_back_to_module(self):
        theme, _report = self._plan(["p1"], ["evaluation"])
        self.assertEqual("module", theme["role_mapping"]["evaluation"])
        self.assertEqual(REFERENCE_FALLBACK_TEXT, theme["roles"]["module"]["text"])


if __name__ == "__main__":
    unittest.main()
