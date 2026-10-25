"""Tests for the pre-routing joint node/asset geometry pass."""

import unittest

from rfs.composition.joint_geometry import resolve_joint_geometry


class JointGeometryTests(unittest.TestCase):
    def test_container_frames_do_not_displace_their_own_descendants(self):
        # The planner draws a stage container around a module group, and the
        # group container around its cards. Those boxes overlap by design; the
        # old pass read the overlap as a collision, pushed every card out to
        # the panel edge and piled them on the same coordinate.
        program = {
            "panels": [{"id": "p", "bbox_percent": {"x": .52, "y": .15, "w": .46, "h": .78}}],
            "semantic_nodes": [
                {"id": "qa_gen", "panel_id": "p", "render_policy": "panel_only", "object_type": "container_anchor", "visual_role": "semantic_container", "bbox_percent": {"x": .54, "y": .25, "w": .26, "h": .65}},
                {"id": "attribution", "panel_id": "p", "parent_id": "qa_gen", "render_policy": "panel_only", "object_type": "container_anchor", "visual_role": "semantic_container", "bbox_percent": {"x": .56, "y": .33, "w": .22, "h": .52}},
                {"id": "text_tamp", "panel_id": "p", "parent_id": "attribution", "bbox_percent": {"x": .57, "y": .4, "w": .09, "h": .125}},
                {"id": "video_tamp", "panel_id": "p", "parent_id": "attribution", "bbox_percent": {"x": .68, "y": .4, "w": .09, "h": .125}},
            ],
            "arrows": [],
        }
        updated, report = resolve_joint_geometry(program)
        self.assertEqual(report["change_count"], 0)
        self.assertEqual([node["bbox_percent"] for node in updated["semantic_nodes"]], [node["bbox_percent"] for node in program["semantic_nodes"]])

    def test_separation_never_stacks_two_cards_on_the_same_box(self):
        program = {
            "panels": [{"id": "p", "bbox_percent": {"x": .5, "y": .1, "w": .12, "h": .8}}],
            "semantic_nodes": [
                {"id": "a", "panel_id": "p", "bbox_percent": {"x": .52, "y": .2, "w": .09, "h": .12}},
                {"id": "b", "panel_id": "p", "bbox_percent": {"x": .53, "y": .25, "w": .09, "h": .12}},
            ],
            "arrows": [],
        }
        updated, report = resolve_joint_geometry(program)
        first, second = (node["bbox_percent"] for node in updated["semantic_nodes"])
        self.assertNotEqual(first, second)
        overlap_x = min(float(first["x"]) + float(first["w"]), float(second["x"]) + float(second["w"])) - max(float(first["x"]), float(second["x"]))
        overlap_y = min(float(first["y"]) + float(first["h"]), float(second["y"]) + float(second["h"])) - max(float(first["y"]), float(second["y"]))
        self.assertLessEqual(min(overlap_x, overlap_y), 0.0)
        self.assertIn("unresolved_count", report)

    def test_overlapping_peer_nodes_are_separated_and_routes_invalidated(self):
        program = {
            "panels": [{"id": "p", "bbox_percent": {"x": .05, "y": .05, "w": .9, "h": .9}}],
            "semantic_nodes": [
                {"id": "a", "panel_id": "p", "bbox_percent": {"x": .2, "y": .2, "w": .3, "h": .3}},
                {"id": "b", "panel_id": "p", "bbox_percent": {"x": .3, "y": .3, "w": .3, "h": .3}},
            ],
            "slots": [{"id": "slot_b", "parent_id": "b", "bbox_percent": {"x": .38, "y": .35, "w": .1, "h": .1}}],
            "arrows": [{"id": "edge", "source": "a", "target": "b", "path_percent": [[.5, .5], [.6, .6]]}],
        }
        updated, report = resolve_joint_geometry(program)
        a, b = updated["semantic_nodes"]
        self.assertEqual(report["status"], "applied")
        self.assertEqual(updated["arrows"][0]["path_percent"], [])
        self.assertLessEqual(float(a["bbox_percent"]["x"]) + float(a["bbox_percent"]["w"]), float(b["bbox_percent"]["x"]) + 1e-6)
        self.assertEqual(updated["slots"][0]["bbox_percent"]["x"], round(float(program["slots"][0]["bbox_percent"]["x"]) + float(b["bbox_percent"]["x"]) - float(program["semantic_nodes"][1]["bbox_percent"]["x"]), 4))


if __name__ == "__main__":
    unittest.main()
