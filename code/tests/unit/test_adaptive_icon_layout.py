"""Tests for visual-role-aware icon slots."""

import unittest

from rfs.composition.node_layout import arrange_node_contents


class AdaptiveIconLayoutTests(unittest.TestCase):
    def test_large_visual_uses_prominent_non_shallow_layout(self):
        program, report = arrange_node_contents({
            "canvas": {"width_in": 13.5, "height_in": 7.5},
            "semantic_nodes": [{"id": "hub", "label": "Central Hub", "visual_role": "large_visual", "visual_priority": "high", "bbox_percent": {"x": .2, "y": .2, "w": .3, "h": .35}}],
            "slots": [{"id": "slot_hub", "asset_id": "icon_hub", "bbox_percent": {"x": .2, "y": .2, "w": .3, "h": .35}}],
        })
        self.assertEqual(program["semantic_nodes"][0]["internal_layout"], "large_visual_label_bottom")
        self.assertGreater(program["slots"][0]["bbox_percent"]["w"], .20)
        self.assertEqual(report["nodes"][0]["visual_role"], "large_visual")


if __name__ == "__main__":
    unittest.main()
