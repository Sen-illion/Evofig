"""Tests for arrowhead proportion limits on editable connectors."""

import math
import unittest

from rfs.composition.arrow_routing import (
    POWERPOINT_ARROWHEAD_FACTORS,
    POWERPOINT_MIN_HEAD_SCALING_STROKE_PT,
    ArrowheadLimitPolicy,
    apply_arrowhead_limits,
    arrowhead_limits,
    limit_arrowhead_proportions,
    stroke_pt_to_preview_px,
    terminal_segment_px,
)
from rfs.composition.preview import _arrowhead_polygon


CANVAS = {"width_px": 1600, "height_px": 900}
POLICY = ArrowheadLimitPolicy()


def _arrow(arrow_id, points, stroke_pt):
    return {
        "id": arrow_id,
        "path_percent": points,
        "stroke_width_pt": stroke_pt,
    }


class ArrowheadLimitTests(unittest.TestCase):
    def test_short_link_head_stays_inside_the_link(self):
        limits = arrowhead_limits(16.0, 3.46, 1600, 900)
        self.assertLessEqual(limits["arrowhead_length_px"], POLICY.max_head_share_of_link * 16.0 + 1e-6)
        self.assertLessEqual(limits["stroke_width_px"], POLICY.max_stroke_share_of_link * 16.0 + 0.6)
        self.assertGreaterEqual(limits["stroke_width_px"], stroke_pt_to_preview_px(POLICY.min_stroke_width_pt, 900))
        self.assertLess(limits["arrowhead_width_px"], 16.0)
        self.assertTrue(limits["stroke_capped"])
        self.assertTrue(limits["head_capped"])

    def test_short_link_keeps_a_visible_shaft(self):
        limits = arrowhead_limits(16.0, 3.46, 1600, 900)
        shaft_px = limits["link_px"] - limits["arrowhead_length_px"]
        self.assertGreaterEqual(shaft_px, 0.35 * limits["link_px"])

    def test_thick_short_link_head_grows_with_the_shaft(self):
        """A short link keeps its extra weight, so the head has to grow too."""
        limits = arrowhead_limits(16.0, 3.46, 1600, 900)
        self.assertGreaterEqual(limits["arrowhead_length_px"], 0.55 * limits["link_px"])
        self.assertGreaterEqual(limits["arrowhead_width_px"], 2.5 * limits["stroke_width_px"])

    def test_long_link_is_left_alone(self):
        limits = arrowhead_limits(400.0, 1.55, 1600, 900)
        self.assertFalse(limits["stroke_capped"])
        self.assertFalse(limits["head_capped"])
        self.assertAlmostEqual(limits["stroke_width_pt"], 1.55, places=2)
        self.assertAlmostEqual(
            limits["arrowhead_length_px"],
            POLICY.head_length_per_stroke * stroke_pt_to_preview_px(1.55, 900),
            places=1,
        )
        self.assertLessEqual(limits["arrowhead_width_px"], 1.3 * limits["arrowhead_length_px"])

    def test_head_never_exceeds_the_canvas_share(self):
        limits = arrowhead_limits(600.0, 5.0, 1600, 900)
        self.assertLessEqual(limits["arrowhead_length_px"], POLICY.max_head_share_of_canvas_width * 1600 + 1e-6)

    def test_size_band_tracks_the_target_head(self):
        for link_px, stroke_pt in ((16.0, 3.46), (60.0, 2.0), (400.0, 1.55)):
            limits = arrowhead_limits(link_px, stroke_pt, 1600, 900)
            factor = POWERPOINT_ARROWHEAD_FACTORS[limits["arrowhead_size"]]
            implied = factor * max(
                limits["stroke_width_px"],
                stroke_pt_to_preview_px(POWERPOINT_MIN_HEAD_SCALING_STROKE_PT, 900),
            )
            self.assertLess(abs(implied - limits["arrowhead_length_px"]), 0.45 * limits["arrowhead_length_px"] + 1.0)

    def test_terminal_segment_uses_the_last_leg_only(self):
        points = [[0.10, 0.10], [0.40, 0.10], [0.40, 0.13]]
        self.assertAlmostEqual(terminal_segment_px(points, 1600, 900), 0.03 * 900, places=4)
        self.assertEqual(terminal_segment_px([[0.1, 0.1]], 1600, 900), 0.0)

    def test_apply_limits_writes_geometry_and_reports_caps(self):
        short = _arrow("short", [[0.10, 0.20], [0.11, 0.20]], 3.4)
        long = _arrow("long", [[0.10, 0.50], [0.40, 0.50]], 1.55)
        block = _arrow("block", [[0.10, 0.70], [0.11, 0.70]], 3.0)
        block["render_style"] = "filled_block_arrow"
        report = apply_arrowhead_limits([short, long, block], CANVAS)
        self.assertEqual(report["arrow_count"], 2)
        self.assertEqual(report["capped_arrow_ids"], ["short"])
        self.assertIn("arrowhead_length_percent", short)
        self.assertIn("arrowhead_width_percent", short)
        self.assertNotIn("arrowhead_length_percent", block)
        self.assertAlmostEqual(short["stroke_width_pt"], round(min(3.4, 4.8 * 277 / 900), 2), places=2)

    def test_program_wrapper_limits_every_arrow(self):
        program = {
            "canvas": dict(CANVAS),
            "arrows": [
                _arrow("a", [[0.10, 0.20], [0.11, 0.20]], 3.4),
                _arrow("b", [[0.10, 0.50], [0.40, 0.50]], 1.55),
            ],
        }
        program, report = limit_arrowhead_proportions(program)
        self.assertEqual(report["arrow_count"], 2)
        for arrow in program["arrows"]:
            limits = arrow["arrowhead_limits"]
            self.assertLessEqual(
                limits["arrowhead_length_px"], POLICY.max_head_share_of_link * limits["link_px"] + 1e-6
            )
            self.assertGreater(limits["powerpoint_head_length_px"], 0)

    def test_preview_polygon_uses_the_stored_head(self):
        points = [(100, 100), (200, 100)]
        arrow = {
            "stroke_width_pt": 3.4,
            "arrowhead_length_percent": 0.01,
            "arrowhead_width_percent": 0.012,
        }
        tip, left, right = _arrowhead_polygon(arrow, points, 1600, 900)
        self.assertEqual(tip, (200, 100))
        self.assertAlmostEqual(tip[0] - left[0], 0.01 * 1600, places=3)
        self.assertAlmostEqual(abs(left[1] - right[1]), 0.012 * 1600, places=3)

    def test_preview_polygon_falls_back_to_the_line_width_rule(self):
        points = [(100, 100), (200, 100)]
        tip, left, right = _arrowhead_polygon({"stroke_width_pt": 2.0}, points, 1600, 900)
        line_width = round(2.0 * 900 / 277.0)
        self.assertAlmostEqual(math.dist(tip, left), math.dist(tip, right), places=3)
        self.assertAlmostEqual(tip[0] - left[0], line_width * 3.2, places=0)


if __name__ == "__main__":
    unittest.main()
