"""Tests for semantic adaptation of reference-inspired layout zones."""

import unittest

from rfs.planning.panel_adaptation import adapt_design_panels
from rfs.planning.paper_figure_structure import structure_from_design_plan


class PanelAdaptationTests(unittest.TestCase):
    def test_empty_reference_sidebar_is_removed_and_container_marked(self):
        plan = {"panels": [
            {"id": "main", "title": "Agent Layer", "bbox_percent": {"x": 0, "y": 0, "w": .7, "h": 1}},
            {"id": "sidebar", "title": "Context", "bbox_percent": {"x": .72, "y": 0, "w": .28, "h": 1}},
        ], "nodes": [
            {"id": "root", "label": "Agent Layer", "role": "module", "panel_id": "main", "bbox_percent": {"x": .02, "y": .02, "w": .66, "h": .94}},
            {"id": "child", "label": "Capabilities", "role": "module", "panel_id": "main", "bbox_percent": {"x": .1, "y": .6, "w": .2, "h": .1}},
        ], "connectors": []}
        adapted, report = adapt_design_panels(plan)
        self.assertEqual(report["removed"][0]["panel_id"], "sidebar")
        self.assertEqual(report["container_node_ids"], ["root"])
        self.assertEqual([p["id"] for p in adapted["panels"]], ["main"])

    def test_single_low_information_panel_is_compacted(self):
        plan = {"panels": [
            {"id": "input", "title": "Input", "role": "input", "bbox_percent": {"x": .02, "y": .2, "w": .2, "h": .6}},
        ], "nodes": [
            {"id": "paper", "label": "Paper", "role": "input", "panel_id": "input", "bbox_percent": {"x": .07, "y": .4, "w": .08, "h": .18}},
        ], "connectors": []}
        adapted, report = adapt_design_panels(plan)
        self.assertEqual(len(report["compacted"]), 1)
        self.assertLess(adapted["panels"][0]["bbox_percent"]["h"], .6)
        self.assertGreater(adapted["panels"][0]["bbox_percent"]["h"], .12)

    def test_visible_nested_child_turns_parent_into_native_container(self):
        plan = {
            "panels": [
                {"id": "method", "title": "Method", "bbox_percent": {"x": .1, "y": .1, "w": .8, "h": .8}},
            ],
            "nodes": [
                {"id": "search", "label": "Search", "panel_id": "method", "bbox_percent": {"x": .2, "y": .2, "w": .4, "h": .5}},
                {"id": "tree", "label": "Tree Search", "panel_id": "method", "parent_id": "search", "bbox_percent": {"x": .25, "y": .35, "w": .3, "h": .25}},
            ],
            "connectors": [],
        }
        adapted, report = adapt_design_panels(plan)
        parent = next(item for item in adapted["nodes"] if item["id"] == "search")
        self.assertEqual(parent["render_policy"], "panel_only")
        self.assertEqual(parent["visual_role"], "semantic_container")
        self.assertEqual(parent["container_child_ids"], ["tree"])
        self.assertEqual(report["hierarchy_containers"][0]["node_id"], "search")
        structure = structure_from_design_plan(adapted)
        icon_ids = {item["asset_id"] for item in structure["icon_generation_plan"]["icons"]}
        self.assertNotIn("icon_search", icon_ids)
        self.assertIn("icon_tree", icon_ids)


if __name__ == "__main__":
    unittest.main()
