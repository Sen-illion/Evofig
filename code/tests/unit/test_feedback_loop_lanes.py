"""Feedback arrows must hug their endpoints instead of the canvas border."""

from __future__ import annotations

import unittest

from rfs.composition.direct_connectors import resolve_direct_connectors


def _length(path: list[list[float]]) -> float:
    return sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(path[:-1], path[1:]))


class FeedbackLoopLaneTests(unittest.TestCase):
    def test_adjacent_boxes_keep_the_feedback_arrow_local(self) -> None:
        program = {
            "semantic_nodes": [
                {"id": "tools", "bbox_percent": {"x": .5, "y": .233, "w": .34, "h": .117}},
                {"id": "search", "bbox_percent": {"x": .5, "y": .4, "w": .15, "h": .15}},
                {"id": "reviser", "bbox_percent": {"x": .69, "y": .4, "w": .15, "h": .3}},
                {"id": "revision", "bbox_percent": {"x": .88, "y": .4, "w": .1, "h": .2}},
            ],
            "arrows": [{"id": "rel_tools_to_reviser", "source": "tools", "target": "reviser", "relation_type": "feedback", "label": "feedback"}],
            "text_program": {"items": []},
        }
        updated, _ = resolve_direct_connectors(program)
        arrow = updated["arrows"][0]
        path = arrow["path_percent"]
        self.assertEqual(arrow["route_selection"]["obstacle_hits"], 0)
        self.assertLess(_length(path), 0.25, path)
        self.assertLess(max(point[1] for point in path), 0.5, path)
        self.assertLess(max(point[0] for point in path), 0.87, path)

    def test_a_blocked_direct_route_still_loops_close_to_the_boxes(self) -> None:
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .1, "y": .2, "w": .1, "h": .1}},
                {"id": "wall", "bbox_percent": {"x": .3, "y": .15, "w": .3, "h": .35}},
                {"id": "b", "bbox_percent": {"x": .7, "y": .2, "w": .1, "h": .1}},
            ],
            "arrows": [{"id": "feedback", "source": "b", "target": "a", "relation_type": "feedback"}],
            "text_program": {"items": []},
        }
        updated, _ = resolve_direct_connectors(program)
        arrow = updated["arrows"][0]
        path = arrow["path_percent"]
        self.assertEqual(arrow["route_selection"]["obstacle_hits"], 0)
        self.assertTrue(arrow["route_selection"]["selected_candidate"].startswith("local_loop"))
        self.assertLess(max(point[1] for point in path), 0.6, path)


if __name__ == "__main__":
    unittest.main()
