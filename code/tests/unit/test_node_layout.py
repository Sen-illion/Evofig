"""Tests for deterministic icon and label layout inside semantic nodes."""

import unittest

from rfs.composition import arrange_node_contents


class NodeLayoutTests(unittest.TestCase):
    def test_shallow_node_uses_horizontal_layout_with_readable_label_lane(self):
        program, report = arrange_node_contents({
            "canvas": {"width_in": 13.5, "height_in": 7.5},
            "semantic_nodes": [{
                "id": "agent",
                "label": "Agent Layer Processing",
                "bbox_percent": {"x": 0.3, "y": 0.5, "w": 0.16, "h": 0.06},
            }],
            "slots": [{
                "id": "slot_agent",
                "asset_id": "icon_agent",
                "bbox_percent": {"x": 0.3, "y": 0.5, "w": 0.16, "h": 0.06},
            }],
        })
        node = program["semantic_nodes"][0]
        slot = program["slots"][0]
        self.assertEqual(node["internal_layout"], "icon_left_editable_label_right")
        self.assertGreater(node["label_bbox_percent"]["h"], 0.04)
        self.assertLess(slot["bbox_percent"]["x"], node["label_bbox_percent"]["x"])
        self.assertEqual(report["node_count"], 1)


    def test_the_lane_holds_the_labels_longest_word(self):
        """A lane narrower than one word is what drew "Unstruc/tured".

        There is no space to break on inside a word, so the lane has to be cut wide
        enough for the longest one instead of taking a fixed share of the card.
        """

        box = {"x": 0.078, "y": 0.25, "w": 0.1011, "h": 0.1033}
        program, report = arrange_node_contents({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "style": {"font_family": "Arial"},
            "semantic_nodes": [{"id": "input_docs", "label": "Unstructured", "bold": True, "bbox_percent": dict(box)}],
            "slots": [{"id": "slot_input_docs", "asset_id": "icon_docs", "bbox_percent": dict(box)}],
        })
        node = program["semantic_nodes"][0]
        slot = program["slots"][0]
        self.assertEqual(node["internal_layout"], "icon_left_editable_label_right")
        self.assertGreaterEqual(node["label_bbox_percent"]["w"] * 13.333, 0.785)
        self.assertLess(slot["bbox_percent"]["w"], 0.1011 * 0.34)

    def test_a_priority_written_as_a_word_lays_the_card_out_as_it_did(self):
        """The layout reads its priority through the same table the other two callers use.

        The word table moved to one shared module; this pins the reading the layout had
        before it moved, so the move cannot quietly change how a card is laid out.
        """

        box = {"x": 0.3, "y": 0.5, "w": 0.16, "h": 0.06}
        program, report = arrange_node_contents({
            "canvas": {"width_in": 13.5, "height_in": 7.5},
            "semantic_nodes": [{"id": "agent", "label": "Agent Layer", "visual_priority": "critical", "bbox_percent": dict(box)}],
            "slots": [{"id": "slot_agent", "asset_id": "icon_agent", "bbox_percent": dict(box)}],
        })
        self.assertEqual(program["semantic_nodes"][0]["id"], "agent")
        self.assertEqual(report["node_count"], 1)

    def test_a_card_that_cannot_afford_both_is_stacked_instead(self):
        box = {"x": 0.1, "y": 0.3, "w": 0.08, "h": 0.09}
        program, report = arrange_node_contents({
            "canvas": {"width_in": 13.333, "height_in": 7.5},
            "style": {"font_family": "Arial"},
            "semantic_nodes": [{"id": "input_docs", "label": "Unstructured", "bold": True, "bbox_percent": dict(box)}],
            "slots": [{"id": "slot_input_docs", "asset_id": "icon_docs", "bbox_percent": dict(box)}],
        })
        node = program["semantic_nodes"][0]
        self.assertEqual(node["internal_layout"], "icon_above_editable_label")
        self.assertGreater(node["label_bbox_percent"]["w"], 0.08 * 0.85)


if __name__ == "__main__":
    unittest.main()
