"""Tests for global title and page safe-area constraints."""

import unittest

from rfs.planning.canvas_constraints import fit_design_to_vertical_safe_area


class CanvasConstraintTests(unittest.TestCase):
    def test_intruding_layout_is_fit_without_changing_horizontal_geometry(self):
        plan = {
            "panels": [{"id": "p", "bbox_percent": {"x": .1, "y": .05, "w": .8, "h": .9}}],
            "nodes": [{"id": "n", "bbox_percent": {"x": .2, "y": .1, "w": .3, "h": .2}}],
            "connectors": [{"id": "e", "path_percent": [[.2, .1], [.5, .9]]}],
        }
        result, report = fit_design_to_vertical_safe_area(plan, top=.11, bottom=.96)
        self.assertTrue(report["transformed"])
        self.assertEqual(result["panels"][0]["bbox_percent"]["x"], .1)
        self.assertEqual(result["panels"][0]["bbox_percent"]["w"], .8)
        self.assertAlmostEqual(result["panels"][0]["bbox_percent"]["y"], .11, places=4)
        self.assertAlmostEqual(
            result["panels"][0]["bbox_percent"]["y"] + result["panels"][0]["bbox_percent"]["h"],
            .96,
            places=4,
        )

    def test_compliant_layout_is_not_modified(self):
        plan = {"panels": [{"id": "p", "bbox_percent": {"x": .1, "y": .15, "w": .8, "h": .7}}]}
        result, report = fit_design_to_vertical_safe_area(plan, top=.11, bottom=.96)
        self.assertFalse(report["transformed"])
        self.assertEqual(result, plan)


if __name__ == "__main__":
    unittest.main()
