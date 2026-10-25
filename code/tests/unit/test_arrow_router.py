import json
import tempfile
import unittest
from pathlib import Path

from rfs.composition.arrow_routing import (
    AESTHETIC_ROUTER_VERSION,
    _enforce_router_constraints,
    _normalize_router_constraints,
    _segment_bbox_overlap,
    _segments,
    style_and_route_arrows,
)


class ArrowRouterTests(unittest.TestCase):
    def test_pending_compatibility_style_is_replaced_by_canonical_role_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = style_and_route_arrows(
                {
                    "canvas": {"width_in": 10, "height_in": 5},
                    "style": {"color_tokens": [{"token_id": "arrow_001", "hex": "#111111"}]},
                    "panels": [],
                    "slots": [
                        {"id": "a", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1}},
                        {"id": "b", "bbox_percent": {"x": 0.7, "y": 0.2, "w": 0.1, "h": 0.1}},
                    ],
                    "arrows": [{
                        "id": "edge",
                        "source_id": "a",
                        "target_id": "b",
                        "path_percent": [[0.2, 0.25], [0.7, 0.25]],
                        "stroke_width_pt": 1.8,
                        "route_style": "rounded_elbow",
                        "arrowhead_size": "med",
                        "_arrow_style_pending": True,
                    }],
                },
                Path(tmp),
                mode="reference",
            )
            arrow = result["arrows"][0]
            self.assertEqual(arrow["semantic_role"], "module_flow")
            self.assertEqual(arrow["stroke_width_pt"], 1.45)
            self.assertEqual(arrow["route_style"], "soft_straight")
            # The head band now follows the head/link ratio instead of a fixed
            # label, so the exported PPTX stub matches the previewed head.
            self.assertEqual(arrow["arrowhead_size"], "med")
            self.assertGreater(arrow["arrowhead_length_percent"], 0)
            self.assertNotIn("_arrow_style_pending", arrow)

    def test_short_connector_gets_bounded_weight_compensation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = style_and_route_arrows(
                {
                    "canvas": {"width_in": 10, "height_in": 5},
                    "style": {"color_tokens": []}, "panels": [],
                    "slots": [
                        {"id": "a", "bbox_percent": {"x": .1, "y": .2, "w": .1, "h": .1}},
                        {"id": "b", "bbox_percent": {"x": .16, "y": .2, "w": .1, "h": .1}},
                    ],
                    "arrows": [{"id": "short", "source_id": "a", "target_id": "b", "path_percent": [[.2, .25], [.21, .25]]}],
                }, Path(tmp), mode="reference",
            )
            arrow = result["arrows"][0]
            self.assertEqual(arrow["semantic_role"], "module_flow")
            # A 1%-wide link keeps extra weight only up to 30% of its own
            # length, and its head grows with that weight without eating the
            # whole link.
            self.assertEqual(arrow["stroke_width_pt"], 1.48)
            self.assertEqual(arrow["arrowhead_size"], "sm")
            limits = arrow["arrowhead_limits"]
            self.assertLessEqual(limits["stroke_width_px"], 0.30 * limits["link_px"] + 0.01)
            self.assertLessEqual(limits["arrowhead_length_px"], 0.60 * limits["link_px"] + 0.01)
            self.assertGreater(limits["link_px"] - limits["arrowhead_length_px"], 0.35 * limits["link_px"])

    def test_reference_paths_are_preserved_and_styled(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            program = {
                "summary": "Figure program.",
                "canvas": {"width_in": 10, "height_in": 5},
                "style": {"color_tokens": [{"token_id": "arrow_001", "hex": "#111111"}]},
                "panels": [],
                "slots": [
                    {"id": "a", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1}},
                    {"id": "b", "bbox_percent": {"x": 0.4, "y": 0.1, "w": 0.1, "h": 0.1}},
                    {"id": "c", "bbox_percent": {"x": 0.4, "y": 0.3, "w": 0.1, "h": 0.1}},
                ],
                "arrows": [
                    {
                        "id": "flow_a",
                        "source_id": "a",
                        "target_id": "b",
                        "path_percent": [[0.2, 0.15], [0.4, 0.15]],
                        "style_token_id": "arrow_001",
                        "editable_in": "pptx",
                        "render_policy": "ppt_shape_not_image_asset",
                        "binding_source": "reference_control_candidates",
                    },
                    {
                        "id": "flow_b",
                        "source_id": "a",
                        "target_id": "c",
                        "path_percent": [[0.2, 0.15], [0.3, 0.15], [0.3, 0.35], [0.4, 0.35]],
                        "style_token_id": "arrow_001",
                        "editable_in": "pptx",
                        "render_policy": "ppt_shape_not_image_asset",
                        "binding_source": "reference_control_candidates",
                    },
                ],
            }

            result = style_and_route_arrows(program, out, mode="reference")
            by_id = {item["id"]: item for item in result["arrows"]}

            self.assertEqual(by_id["flow_a"]["path_percent"], [[0.2, 0.15], [0.4, 0.15]])
            self.assertTrue(by_id["flow_a"]["reference_locked"])
            self.assertTrue(by_id["flow_a"]["reference_path_preserved"])
            self.assertEqual(by_id["flow_a"]["semantic_role"], "branch")
            self.assertEqual(by_id["flow_a"]["route_style"], "bundled_elbow")
            self.assertEqual(result["style"]["arrow_style_profile_path"], "arrow_style_profile.json")

            quality = json.loads((out / "arrow_quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["summary"], "Arrow routing and styling quality report.")
            self.assertEqual(quality["arrow_count"], 2)
            self.assertFalse(quality["reference_path_overrides"])
            # The obstacle total has its own name now; the archived name is written beside it so
            # both readings of the same number are the same number.
            self.assertEqual(quality["route_obstacle_overlap_count"], quality["box_pass_count"])
            self.assertIn("crossing_count", quality)
            self.assertNotIn("total_crossing_count", quality)

    def test_fallback_route_avoids_slot_obstacle_without_changing_reference_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            program = {
                "summary": "Figure program.",
                "canvas": {"width_in": 10, "height_in": 5},
                "style": {"color_tokens": [{"token_id": "arrow_001", "hex": "#111111"}]},
                "panels": [],
                "slots": [
                    {"id": "source", "bbox_percent": {"x": 0.10, "y": 0.42, "w": 0.10, "h": 0.12}},
                    {"id": "target", "bbox_percent": {"x": 0.78, "y": 0.42, "w": 0.10, "h": 0.12}},
                    {"id": "obstacle", "bbox_percent": {"x": 0.42, "y": 0.38, "w": 0.14, "h": 0.20}},
                    {"id": "locked_a", "bbox_percent": {"x": 0.10, "y": 0.12, "w": 0.10, "h": 0.10}},
                    {"id": "locked_b", "bbox_percent": {"x": 0.78, "y": 0.12, "w": 0.10, "h": 0.10}},
                ],
                "arrows": [
                    {
                        "id": "locked_flow",
                        "source_id": "locked_a",
                        "target_id": "locked_b",
                        "path_percent": [[0.20, 0.17], [0.78, 0.17]],
                        "style_token_id": "arrow_001",
                        "editable_in": "pptx",
                        "render_policy": "ppt_shape_not_image_asset",
                        "binding_source": "reference_control_candidates",
                    },
                    {
                        "id": "fallback_flow",
                        "source_id": "source",
                        "target_id": "target",
                        "path_percent": [],
                        "style_token_id": "arrow_001",
                        "editable_in": "pptx",
                        "render_policy": "ppt_shape_not_image_asset",
                        "route_policy": "fallback_reroute_allowed",
                    },
                ],
            }

            result = style_and_route_arrows(program, out, mode="reference")
            by_id = {item["id"]: item for item in result["arrows"]}

            self.assertEqual(by_id["locked_flow"]["path_percent"], [[0.2, 0.17], [0.78, 0.17]])
            self.assertTrue(by_id["locked_flow"]["reference_locked"])
            self.assertTrue(by_id["locked_flow"]["reference_path_preserved"])

            fallback = by_id["fallback_flow"]
            self.assertFalse(fallback["reference_locked"])
            self.assertEqual(fallback["route_generation_status"], "fallback_route_selected")
            self.assertEqual(fallback["routing_algorithm"], "reference-constrained-orthogonal-v1")
            self.assertGreaterEqual(len(fallback["path_percent"]), 3)
            obstacle_box = by_id.get("obstacle", {}).get("bbox_percent") or {"x": 0.42, "y": 0.38, "w": 0.14, "h": 0.20}
            self.assertFalse(any(_segment_bbox_overlap(a, b, obstacle_box) for a, b in _segments(fallback["path_percent"])))

            routes = json.loads((out / "selected_arrow_routes.json").read_text(encoding="utf-8"))["routes"]
            route_by_id = {item["id"]: item for item in routes}
            self.assertEqual(route_by_id["fallback_flow"]["route_generation_status"], "fallback_route_selected")
            self.assertGreater(route_by_id["fallback_flow"]["candidate_count"], 1)

    def test_aesthetic_mode_uses_reference_tunnel_for_bundle_offsets(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            arrows = []
            for index, y in enumerate([0.18, 0.30, 0.42], start=1):
                arrows.append({
                    "id": f"branch_{index}",
                    "source_id": "source",
                    "target_id": f"target_{index}",
                    "path_percent": [[0.20, 0.30], [0.70, y]],
                    "style_token_id": "arrow_001",
                    "editable_in": "pptx",
                    "render_policy": "ppt_shape_not_image_asset",
                    "binding_source": "reference_control_candidates",
                    "aesthetic_offset_allowed": True,
                })
            program = {
                "summary": "Figure program.",
                "canvas": {"width_in": 10, "height_in": 5},
                "style": {"color_tokens": [{"token_id": "arrow_001", "hex": "#111111"}]},
                "panels": [],
                "slots": [
                    {"id": "source", "bbox_percent": {"x": 0.10, "y": 0.25, "w": 0.10, "h": 0.10}},
                    {"id": "target_1", "bbox_percent": {"x": 0.70, "y": 0.13, "w": 0.10, "h": 0.10}},
                    {"id": "target_2", "bbox_percent": {"x": 0.70, "y": 0.25, "w": 0.10, "h": 0.10}},
                    {"id": "target_3", "bbox_percent": {"x": 0.70, "y": 0.37, "w": 0.10, "h": 0.10}},
                ],
                "arrows": arrows,
            }

            result = style_and_route_arrows(program, out, mode="aesthetic")
            by_id = {item["id"]: item for item in result["arrows"]}
            adjusted = 0
            for arrow_id, arrow in by_id.items():
                self.assertEqual(arrow["semantic_role"], "branch")
                self.assertEqual(arrow["route_style"], "metro_bundle")
                self.assertEqual(arrow["routing_algorithm"], AESTHETIC_ROUTER_VERSION)
                self.assertIn(arrow["route_generation_status"], {"aesthetic_tunnel_adjusted", "aesthetic_style_only"})
                self.assertTrue(arrow["reference_tunnel_preserved"], arrow_id)
                self.assertLessEqual(arrow["reference_path_delta_max"], arrow["reference_tunnel_percent"])
                self.assertTrue(arrow["reference_path_preserved"])
                self.assertTrue(arrow["reference_original_path_percent"])
                self.assertGreater(float(arrow["halo_width_pt"]), 0)
                if arrow["route_generation_status"] == "aesthetic_tunnel_adjusted":
                    adjusted += 1
            self.assertGreaterEqual(adjusted, 2)

            quality = json.loads((out / "arrow_quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["mode"], "aesthetic")
            self.assertEqual(quality["reference_tunnel_violations"], [])
            self.assertGreaterEqual(quality["aesthetic_tunnel_adjusted_count"], 2)

    def test_feedback_loop_is_styled_as_dashed_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = style_and_route_arrows(
                {
                    "canvas": {"width_in": 10, "height_in": 5},
                    "style": {"color_tokens": []},
                    "panels": [],
                    "slots": [
                        {"id": "a", "bbox_percent": {"x": .1, "y": .2, "w": .1, "h": .1}},
                        {"id": "b", "bbox_percent": {"x": .7, "y": .2, "w": .1, "h": .1}},
                    ],
                    "arrows": [{
                        "id": "loop",
                        "source_id": "b",
                        "target_id": "a",
                        "control_kind": "dashed_loop",
                        "path_percent": [[.7, .25], [.7, .04], [.2, .04], [.2, .25]],
                    }],
                },
                Path(tmp),
                mode="reference",
            )
            arrow = result["arrows"][0]
            self.assertEqual(arrow["semantic_role"], "feedback_loop")
            self.assertEqual(arrow["route_style"], "dashed_spline_like")
            self.assertEqual(arrow["line_pattern"], "dash")

class PersonConnectorRuleTests(unittest.TestCase):
    """The two connector rules a person states, and the router honouring them.

    A person looking at a figure says "these two boxes are close, join them with one connector" and
    "this connector does not leave its panel". Neither could be written down before, so the router
    went on drawing the detour the person had just objected to. They are opt-in: a figure drawn
    without them is byte-identical to one drawn before they existed.
    """

    def _objects(self) -> dict:
        return {
            # panel_b holds both boxes of the connector below.
            "panel_b": {"bbox_percent": {"x": 0.55, "y": 0.15, "w": 0.42, "h": 0.35}},
            "response": {"bbox_percent": {"x": 0.72, "y": 0.28, "w": 0.12, "h": 0.06}},
            "limits": {"bbox_percent": {"x": 0.62, "y": 0.40, "w": 0.26, "h": 0.07}},
        }

    def _detour_arrow(self, target: str = "limits") -> dict:
        return {
            "id": "edge",
            "source_id": "response",
            "target_id": target,
            # Out of the panel to the right, down, and back in: the shape a person objects to.
            "path_percent": [[0.78, 0.34], [0.99, 0.34], [0.99, 0.435], [0.75, 0.435]],
        }

    def test_a_detour_between_two_close_boxes_becomes_one_connector(self):
        arrows = [self._detour_arrow()]
        result = _enforce_router_constraints(arrows, self._objects(), {"direct_when_close": True})
        self.assertEqual([[0.78, 0.34], [0.75, 0.435]], arrows[0]["path_percent"])
        self.assertEqual("person_rule_direct", arrows[0]["route_generation_status"])
        self.assertEqual(["direct_when_close"], [entry["rule"] for entry in result["applied"]])
        self.assertEqual([], result["unmet"])

    def test_a_long_connector_is_left_alone(self):
        objects = self._objects()
        objects["far"] = {"bbox_percent": {"x": 0.05, "y": 0.90, "w": 0.10, "h": 0.06}}
        arrows = [self._detour_arrow("far")]
        _enforce_router_constraints(arrows, objects, {"direct_when_close": True})
        self.assertEqual(4, len(arrows[0]["path_percent"]))

    def test_a_connector_that_leaves_its_panel_comes_back_inside(self):
        arrows = [self._detour_arrow()]
        result = _enforce_router_constraints(arrows, self._objects(), {"stay_inside_parent": True})
        self.assertEqual(2, len(arrows[0]["path_percent"]))
        self.assertEqual(["stay_inside_parent"], [entry["rule"] for entry in result["applied"]])

    def test_a_connector_that_stays_in_its_panel_is_left_alone(self):
        arrows = [{"id": "edge", "source_id": "response", "target_id": "limits", "path_percent": [[0.78, 0.34], [0.78, 0.435], [0.75, 0.435]]}]
        result = _enforce_router_constraints(arrows, self._objects(), {"stay_inside_parent": True})
        self.assertEqual(3, len(arrows[0]["path_percent"]))
        self.assertEqual([], result["applied"])

    def test_the_rules_survive_normalization_and_junk_does_not(self):
        rules = _normalize_router_constraints(
            {"direct_when_close": True, "stay_inside_parent": True, "close_gap": 0.05, "invent_a_rule": True}
        )
        self.assertEqual({"direct_when_close": True, "stay_inside_parent": True, "close_gap": 0.05}, rules)

    def test_the_panel_a_connector_belongs_to_is_named_as_a_container(self):
        # Panels are routing containers, not obstacles, so they are not among the objects a
        # connector is routed around. Without being named as containers a rule could never see the
        # panel it was asked about, and the detour the person objected to survived every round.
        arrows = [self._detour_arrow()]
        objects = {key: value for key, value in self._objects().items() if key != "panel_b"}
        result = _enforce_router_constraints(
            arrows,
            objects,
            {"stay_inside_parent": True},
            containers={"panel_b": self._objects()["panel_b"]},
        )
        self.assertEqual(2, len(arrows[0]["path_percent"]))
        self.assertEqual(["stay_inside_parent"], [entry["rule"] for entry in result["applied"]])

    def test_a_box_gap_the_rule_was_given_a_number_for(self):
        # The two boxes a person called "close together" sit 0.107 apart, which is further than the
        # rule's own default, so the rule carries the number it means.
        arrows = [self._detour_arrow()]
        close = _enforce_router_constraints(arrows, self._objects(), {"direct_when_close": True, "close_gap": 0.12})
        self.assertEqual(2, len(arrows[0]["path_percent"]))
        self.assertIn("direct_when_close", [entry["rule"] for entry in close["applied"]])

    def test_the_default_reading_of_close_covers_the_boxes_a_person_named(self):
        # The first sentence a person wrote named two boxes 0.107 apart, stacked neighbours of one
        # group. The rule's default was 0.08, so the rule matched no connector on the figure at all
        # and the detour survived the round that had already been told about it.
        objects = {
            "panel_b_model": {"bbox_percent": {"x": 0.79, "y": 0.205, "w": 0.165, "h": 0.068}},
            "panel_b_limits": {"bbox_percent": {"x": 0.60, "y": 0.38, "w": 0.355, "h": 0.095}},
        }
        arrows = [
            {
                "id": "edge",
                "source_id": "panel_b_model",
                "target_id": "panel_b_limits",
                "path_percent": [[0.8725, 0.273], [0.975, 0.275], [0.975, 0.38], [0.8725, 0.38]],
            }
        ]
        result = _enforce_router_constraints(arrows, objects, {"direct_when_close": True})
        self.assertEqual([[0.8725, 0.273], [0.8725, 0.38]], arrows[0]["path_percent"])
        self.assertEqual(["direct_when_close"], [entry["rule"] for entry in result["applied"]])


if __name__ == "__main__":
    unittest.main()
