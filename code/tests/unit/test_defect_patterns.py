"""P7: a Memory condition names the shape of the problem, not a node count.

``node_count_bucket=high`` cannot tell a four-way fan-out crossing two panels from a tidy
pipeline that happens to have as many boxes, so a row could not recognise "a different paper
with the same illness". These patterns are that illness, read off the structure the pipeline
already wrote down.
"""

import unittest

from rfs.analysis.graph.defect_patterns import defect_patterns, missing_patterns
from rfs.analysis.graph.retrieval import count_bucket, matched_condition
from rfs.contracts.prompt_patch import render_condition_summary
from rfs.evolution.direct import count_bucket as written_bucket


def node(node_id, panel, x, y, w=0.1, h=0.1):
    return {"id": node_id, "panel_id": panel, "bbox_percent": {"x": x, "y": y, "w": w, "h": h}}


def box(panel_id, x, y, w, h):
    return {"id": panel_id, "bbox_percent": {"x": x, "y": y, "w": w, "h": h}}


SIDE_BY_SIDE = [box("p1", 0.02, 0.05, 0.4, 0.9), box("p2", 0.55, 0.05, 0.4, 0.9)]


class DefectPatternTests(unittest.TestCase):
    def test_a_four_way_fan_out_is_named(self):
        figure = {
            "panels": [box("p1", 0.02, 0.05, 0.5, 0.9)],
            "nodes": [node("hub", "p1", 0.05, 0.1)] + [node(f"leaf{index}", "p1", 0.3, 0.1 + index * 0.2) for index in range(4)],
            "connectors": [{"id": f"rel_{index}", "source": "hub", "target": f"leaf{index}"} for index in range(4)],
        }

        self.assertIn("fanout_1_to_n", defect_patterns(figure))

    def test_a_three_way_fan_out_is_not(self):
        figure = {
            "panels": [box("p1", 0.02, 0.05, 0.5, 0.9)],
            "nodes": [node("hub", "p1", 0.05, 0.1)] + [node(f"leaf{index}", "p1", 0.3, 0.3 + index * 0.2) for index in range(3)],
            "connectors": [{"id": f"rel_{index}", "source": "hub", "target": f"leaf{index}"} for index in range(3)],
        }

        self.assertNotIn("fanout_1_to_n", defect_patterns(figure))

    def test_connectors_between_panels_are_named(self):
        figure = {
            "panels": SIDE_BY_SIDE,
            "nodes": [node("a", "p1", 0.05, 0.2), node("b", "p2", 0.6, 0.2)],
            "connectors": [{"id": "rel_1", "source": "a", "target": "b"}],
        }

        self.assertIn("cross_panel", defect_patterns(figure))

    def test_edges_that_cross_a_panel_they_do_not_belong_to_are_named(self):
        figure = {
            "panels": [box("left", 0.02, 0.05, 0.2, 0.9), box("middle", 0.3, 0.05, 0.4, 0.9), box("right", 0.78, 0.05, 0.2, 0.9)],
            "nodes": [node("a1", "left", 0.03, 0.1), node("a2", "left", 0.03, 0.5), node("b1", "right", 0.8, 0.1), node("b2", "right", 0.8, 0.5)],
            "connectors": [{"id": "rel_1", "source": "a1", "target": "b1"}, {"id": "rel_2", "source": "a2", "target": "b2"}],
        }

        self.assertIn("edge_crossings", defect_patterns(figure))

    def test_a_feedback_loop_is_named(self):
        figure = {
            "panels": [box("p1", 0.02, 0.05, 0.5, 0.9)],
            "nodes": [node("a", "p1", 0.05, 0.1), node("b", "p1", 0.25, 0.1)],
            "connectors": [{"id": "rel_1", "source": "a", "target": "b"}, {"id": "rel_2", "source": "b", "target": "a"}],
        }

        self.assertIn("dashed_cycle", defect_patterns(figure))

    def test_a_tidy_figure_has_no_defects(self):
        figure = {
            "panels": [box("p1", 0.02, 0.05, 0.5, 0.9)],
            "nodes": [node("a", "p1", 0.05, 0.1), node("b", "p1", 0.2, 0.1)],
            "connectors": [{"id": "rel_1", "source": "a", "target": "b"}],
        }

        self.assertEqual([], defect_patterns(figure))

    def test_nothing_readable_reports_no_defects(self):
        self.assertEqual([], defect_patterns(None))
        self.assertEqual([], defect_patterns({}))


class DefectConditionTests(unittest.TestCase):
    def _record(self, patterns):
        return {"paper_id": "p", "domain": "ai_method", "features": {"node_count": 9, "edge_count": 9, "defect_patterns": patterns}}

    def test_a_condition_matches_a_record_that_has_the_shape(self):
        condition = {"defect_patterns": ["fanout_1_to_n", "cross_panel"]}

        self.assertTrue(matched_condition(self._record(["fanout_1_to_n", "cross_panel", "long_edge"]), [condition]))
        self.assertFalse(matched_condition(self._record(["fanout_1_to_n"]), [condition]))

    def test_a_condition_the_record_cannot_answer_never_matches(self):
        self.assertFalse(matched_condition(self._record(["fanout_1_to_n"]), [{"defect_patterns": ["port_collision"]}]))
        self.assertEqual(["port_collision"], missing_patterns({"defect_patterns": ["port_collision"]}, ["fanout_1_to_n"]))

    def test_the_shape_is_read_off_the_figure_when_the_record_carries_one(self):
        figure = {
            "panels": SIDE_BY_SIDE,
            "nodes": [node("a", "p1", 0.05, 0.2), node("b", "p2", 0.6, 0.2)],
            "connectors": [{"id": "rel_1", "source": "a", "target": "b"}],
        }

        self.assertTrue(matched_condition({"paper_id": "p", "figure": figure}, [{"defect_patterns": ["cross_panel"]}]))

    def test_the_coarse_buckets_still_match(self):
        condition = {"domain": "ai_method", "node_count_bucket": "high"}

        self.assertTrue(matched_condition(self._record([]), [condition]))
        self.assertFalse(matched_condition({"domain": "biology", "features": {"node_count": 9}}, [condition]))

    def test_a_condition_naming_a_count_can_match_the_record_it_came_from(self):
        """The writer and the reader have to bucket a count the same way, or nothing matches."""

        for count in (1, 4, 5, 8, 9, 40):
            record = {"features": {"node_count": count, "edge_count": count}}
            condition = {"topology": "", "node_count_bucket": count_bucket(count)}

            self.assertTrue(matched_condition(record, [condition]), count)

    def test_the_writer_and_the_reader_agree_on_every_bucket(self):
        self.assertEqual([written_bucket(index) for index in range(0, 20)], [count_bucket(index) for index in range(0, 20)])


class DefectRenderingTests(unittest.TestCase):
    def test_a_defect_condition_reads_as_the_shape(self):
        summary = render_condition_summary({"defect_patterns": ["fanout_1_to_n", "cross_panel"]})

        self.assertEqual("one node feeds four or more others and the connectors cross panels", summary)

    def test_the_buckets_still_render_as_they_did(self):
        self.assertEqual("node_count_bucket=high, topology=pipeline", render_condition_summary({"topology": "pipeline", "node_count_bucket": "high"}))


if __name__ == "__main__":
    unittest.main()
