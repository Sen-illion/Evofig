"""Tests for pre-generation color plan validation."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from rfs.evaluation.color_plan_validation import validate_color_plan
from rfs.planning.color_theme_planner import plan_color_theme_vlm


def _theme(fill: str = "#E8F3F0", body: str = "#193247") -> dict:
    return {
        "theme_name": "test",
        "palette_strategy": {"dominant_family": "teal", "harmony": "analogous", "rationale": "pale teal surfaces"},
        "canvas_background": "#FFFFFF",
        "text": {"primary": "#193247", "secondary": "#526676", "muted": "#71808B", "inverse": "#FFFFFF"},
        "connectors": {"primary": "#54758A", "secondary": "#7B929F", "feedback": "#A17B55"},
        "panels": {"p1": {"fill": fill, "border": "#4D8B7B", "header": "#315C78", "header_text": "#FFFFFF", "body_text": body}},
        "role_mapping": {"module": "process"},
        "roles": {"process": {"fill": "#E8F3F0", "border": "#4D8B7B", "text": "#193247"}},
        "usage_policy": {"max_distinct_hues": 4},
        "self_check": {"all_panel_ids_covered": True, "all_node_roles_covered": True, "palette_is_harmonious": True, "contrast_requirements_met": True},
    }


class ColorPlanValidationTests(unittest.TestCase):
    def test_rejects_unreadable_panel_text(self):
        result = validate_color_plan(_theme(fill="#FFFFFF", body="#FFFFFF"), expected_panel_ids={"p1"}, expected_node_roles={"module"})
        self.assertFalse(result["ok"])
        self.assertTrue(any("contrast" in item for item in result["errors"]))

    def test_a_pale_panel_fill_is_not_rejected_for_saturation(self):
        # HLS reads a near-white tint as strongly saturated (#F0FDFA measures 0.77), so the
        # module must not put a saturation ceiling on a panel fill.
        result = validate_color_plan(_theme(fill="#F0FDFA"), expected_panel_ids={"p1"}, expected_node_roles={"module"})
        self.assertTrue(result["ok"], result["errors"])
        self.assertFalse(any("saturated" in item for item in result["errors"]))

    @patch("rfs.planning.color_theme_planner.call_vlm_json")
    def test_invalid_theme_gets_targeted_retry(self, mocked):
        mocked.side_effect = [_theme(fill="#FFFFFF", body="#FFFFFF"), _theme()]
        theme, report = plan_color_theme_vlm({}, {"panels": [{"id": "p1"}], "nodes": [{"role": "module"}]}, {"items": []})
        self.assertTrue(report["ok"])
        self.assertEqual(len(report["attempts"]), 2)
        self.assertIn("VALIDATION_FEEDBACK", mocked.call_args_list[1].args[0])
        self.assertEqual(theme["panels"]["p1"]["fill"], "#E8F3F0")

    @patch("rfs.planning.color_theme_planner.call_vlm_json", side_effect=TimeoutError("timeout"))
    def test_vlm_failure_does_not_silently_fallback(self, mocked):
        theme, report = plan_color_theme_vlm({}, {"panels": [{"id": "p1"}]}, {"items": []})
        self.assertFalse(report["ok"])
        self.assertFalse(report["fallback_used"])
        self.assertEqual(mocked.call_count, 3)
        self.assertNotEqual(theme.get("theme_name"), "safe_academic_fallback")


if __name__ == "__main__":
    unittest.main()
