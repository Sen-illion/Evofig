"""Tests for the figure layout contract's tolerance of model-shaped fields."""

import unittest

from rfs.contracts.figure_layout import (
    DEFAULT_CANVAS_MARGIN,
    FIGURE_LAYOUT_CONTRACT_VERSION,
    normalize_figure_layout_plan,
    validate_figure_layout_plan,
)


class LayoutNumberTests(unittest.TestCase):
    def test_a_named_edge_margin_object_yields_its_widest_edge(self):
        # The design model answers with this shape on the papers that used to crash
        # the whole run inside normalize_figure_layout_plan.
        plan = normalize_figure_layout_plan(
            {"canvas": {"margin": {"top": 0.04, "bottom": 0.06, "left": 0.02, "right": 0.03}}}
        )

        self.assertEqual(plan["canvas"]["margin"], 0.06)

    def test_a_margin_object_without_numbers_falls_back_to_the_default(self):
        plan = normalize_figure_layout_plan({"canvas": {"margin": {"top": "wide", "left": None}}})

        self.assertEqual(plan["canvas"]["margin"], DEFAULT_CANVAS_MARGIN)

    def test_a_numeric_string_margin_is_read_and_junk_is_ignored(self):
        self.assertEqual(normalize_figure_layout_plan({"canvas": {"margin": "0.07"}})["canvas"]["margin"], 0.07)
        self.assertEqual(normalize_figure_layout_plan({"canvas": {"margin": "auto"}})["canvas"]["margin"], DEFAULT_CANVAS_MARGIN)

    def test_a_version_that_is_not_a_number_falls_back_to_the_contract_version(self):
        self.assertEqual(
            normalize_figure_layout_plan({"version": "v2"})["version"],
            FIGURE_LAYOUT_CONTRACT_VERSION,
        )
        self.assertEqual(normalize_figure_layout_plan({"version": "3"})["version"], 3)


class LayoutPlanShapeTests(unittest.TestCase):
    def test_a_plan_with_a_bad_canvas_margin_keeps_the_rest_of_its_content(self):
        plan = normalize_figure_layout_plan({
            "canvas": {"margin": {"top": 0.05}},
            "topology": "left_to_right_pipeline",
            "reading_order": ["a", "b"],
            "panels": [{"id": "p1"}],
            "nodes": [{"id": "n1", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.2}}],
            "connectors": [{"source": "n1", "target": "n1"}],
            "style_tokens": {"ribbon": 0.5},
        })

        self.assertEqual(plan["canvas"]["margin"], 0.05)
        self.assertEqual(plan["reading_order"], ["a", "b"])
        self.assertEqual([item["id"] for item in plan["panels"]], ["p1"])
        self.assertEqual([item["id"] for item in plan["nodes"]], ["n1"])
        self.assertEqual(plan["style_tokens"], {"ribbon": 0.5})
        self.assertTrue(validate_figure_layout_plan(plan)["ok"])

    def test_a_node_without_a_full_bbox_is_still_rejected(self):
        report = validate_figure_layout_plan({
            "canvas": {"margin": {"top": 0.05}},
            "nodes": [
                {"id": "n1", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.2}},
                {"id": "n2", "bbox_percent": {"x": 0.1, "y": 0.2}},
            ],
        })

        self.assertFalse(report["ok"])
        self.assertEqual(len(report["errors"]), 1)


if __name__ == "__main__":
    unittest.main()