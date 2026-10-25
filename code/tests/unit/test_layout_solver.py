"""Tests for the deterministic final layout pass."""

import tempfile
import unittest
from pathlib import Path

from rfs.composition import solve_layout_constraints


class LayoutSolverTests(unittest.TestCase):
    def test_relocates_text_and_repairs_blocked_connector(self):
        program = {"panels": [], "slots": [{"id": "source", "bbox_percent": {"x": .05, "y": .4, "w": .1, "h": .1}}, {"id": "target", "bbox_percent": {"x": .8, "y": .4, "w": .1, "h": .1}}, {"id": "obstacle", "bbox_percent": {"x": .42, "y": .35, "w": .12, "h": .2}}], "text_program": {"items": [{"id": "label", "text": "A", "bbox_percent": {"x": .42, "y": .39, "w": .12, "h": .05}}, {"id": "label_2", "text": "B", "bbox_percent": {"x": .42, "y": .39, "w": .12, "h": .05}}]}, "arrows": [{"id": "edge", "source_id": "source", "target_id": "target", "path_percent": [[.15, .45], [.8, .45]]}], "style": {"color_tokens": []}}
        with tempfile.TemporaryDirectory() as temp:
            solved, report = solve_layout_constraints(program, Path(temp))
        self.assertGreaterEqual(report["text_relocation_count"], 1)
        self.assertIn("edge", report["rerouted_edge_ids"])
        self.assertEqual(solved["arrows"][0]["route_generation_status"], "repair_route_selected")

    def test_relation_label_collision_is_relocated_when_space_exists(self):
        program = {
            "slots": [{"id": "node", "bbox_percent": {"x": .4, "y": .4, "w": .2, "h": .2}}],
            "relation_labels": [{"id": "relation", "bbox_percent": {"x": .45, "y": .45, "w": .1, "h": .04}}],
            "text_program": {"items": []}, "arrows": [], "style": {"color_tokens": []},
        }
        with tempfile.TemporaryDirectory() as temp:
            _, report = solve_layout_constraints(program, Path(temp), reroute=False)
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["label_collisions"], [])
        self.assertGreaterEqual(report["object_relocation_count"], 1)


if __name__ == "__main__":
    unittest.main()
