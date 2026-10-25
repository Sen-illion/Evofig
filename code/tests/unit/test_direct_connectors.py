"""Tests for authoritative direct-workflow connector geometry."""

from __future__ import annotations

import unittest

from rfs.composition.direct_connectors import _label_box, apply_connector_route_selections, resolve_direct_connectors


def _path_enters(path, box, margin: float = 0.006) -> bool:
    for start, end in zip(path[:-1], path[1:]):
        for index in range(21):
            x = start[0] + (end[0] - start[0]) * index / 20
            y = start[1] + (end[1] - start[1]) * index / 20
            if box["x"] - margin <= x <= box["x"] + box["w"] + margin and box["y"] - margin <= y <= box["y"] + box["h"] + margin:
                return True
    return False


class DirectConnectorTests(unittest.TestCase):

    def test_a_program_can_retune_the_routing_weights(self):
        # The weights are a policy, not hard-coded behaviour: a caller (or the
        # self-evolution loop) may override them per program, and the report says
        # which values produced the route.
        program = {
            "route_policy": {"slant_penalty": 0.0},
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .10, "w": .10, "h": .10}},
                {"id": "b", "bbox_percent": {"x": .70, "y": .60, "w": .10, "h": .10}},
            ],
            "arrows": [{"id": "edge", "source": "a", "target": "b", "relation_type": "data_flow"}],
            "text_program": {"items": []},
        }
        updated, report = resolve_direct_connectors(program)
        self.assertEqual("direct", updated["arrows"][0]["route_selection"]["selected_candidate"])
        self.assertEqual(0.0, report["route_policy"]["slant_penalty"])

    def test_the_slant_charge_grows_with_how_far_a_stroke_runs_off_axis(self):
        # A short off-axis stroke must not carry the same charge as a long lazy
        # diagonal, so the same pair of boxes scores differently once the boxes
        # are only slightly out of line with each other.
        from rfs.composition.direct_connectors import _slant_extent
        self.assertEqual(0.0, _slant_extent([[.1, .5], [.5, .5]]))
        self.assertEqual(0.0, _slant_extent([[.1, .5], [.1, .9]]))
        self.assertAlmostEqual(0.02, _slant_extent([[.1, .5], [.5, .52]]), places=6)
        self.assertAlmostEqual(0.40, _slant_extent([[.1, .5], [.5, .90]]), places=6)
        self.assertEqual(0.0, _slant_extent([[.1, .5], [.3, .7], [.5, .5]]))

    def test_route_is_not_drawn_across_the_paper_heading(self):
        title = {"x": .04, "y": .015, "w": .92, "h": .1}
        program = {
            "title_block": {"title": "A paper", "bbox_percent": title},
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .1, "y": .16, "w": .12, "h": .12}},
                {"id": "b", "bbox_percent": {"x": .78, "y": .16, "w": .12, "h": .12}},
            ],
            "arrows": [{"id": "edge", "source": "a", "target": "b", "relation_type": "feedback", "label": "loops back"}],
            "text_program": {"items": []},
        }
        updated, report = resolve_direct_connectors(program)
        self.assertFalse(_path_enters(updated["arrows"][0]["path_percent"], title))
        self.assertEqual(report["connectors"][0]["route_selection"]["obstacle_hits"], 0)

    def test_a_label_offset_that_sits_on_another_routed_line_is_rejected(self):
        # The first candidate lane is the one a reader would read as "on" the
        # other connector, so an already routed line must veto it.
        own_path = [[.2, .5], [.5, .5]]
        other_path = [[.25, .465], [.45, .465]]
        first_choice, _ = _label_box(own_path, "clearly", [], 9.0, None)
        resolved, payload = _label_box(own_path, "clearly", [], 9.0, [other_path])
        self.assertTrue(_path_enters(other_path, first_choice))
        self.assertFalse(_path_enters(other_path, resolved))
        self.assertNotIn("collision_unresolved", payload)

    def test_label_is_derived_from_final_path_and_old_absolute_label_removed(self):
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .2, "w": .15, "h": .15}},
                {"id": "b", "bbox_percent": {"x": .7, "y": .5, "w": .15, "h": .15}},
            ],
            "arrows": [{"id": "edge", "source": "a", "target": "b", "relation_type": "prediction", "label": "produces representation", "evidence_ids": ["E1"]}],
            "labels": [{"id": "old", "target_id": "edge", "text": "old", "bbox_percent": {"x": .4, "y": .1, "w": .1, "h": .04}}],
            "text_program": {"items": []},
        }
        updated, report = resolve_direct_connectors(program)
        arrow = updated["arrows"][0]
        label = updated["text_program"]["items"][0]
        self.assertGreaterEqual(len(arrow["path_percent"]), 2)
        self.assertEqual(arrow["arrowhead"], "triangle")
        self.assertEqual(label["target_id"], "edge")
        self.assertEqual(label["evidence_ids"], ["E1"])
        self.assertEqual(label["path_binding"]["placement"], "near_segment")
        self.assertEqual(updated["labels"], [])
        self.assertEqual(report["path_bound_label_count"], 1)

    def test_upstream_connector_style_is_preserved_and_never_synthesized(self):
        style = {"route_style": "soft_curve", "stroke_color": "#2A63A1", "stroke_width_pt": 2.2, "line_pattern": "solid"}
        nodes = [
            {"id": "a", "bbox_percent": {"x": .05, "y": .2, "w": .15, "h": .15}},
            {"id": "b", "bbox_percent": {"x": .7, "y": .5, "w": .15, "h": .15}},
        ]
        styled, _ = resolve_direct_connectors({
            "semantic_nodes": nodes,
            "arrows": [{"id": "edge", "source": "a", "target": "b", "relation_type": "prediction", **style}],
            "text_program": {"items": []},
        })
        for key, value in style.items():
            self.assertEqual(styled["arrows"][0][key], value)
        bare, _ = resolve_direct_connectors({
            "semantic_nodes": nodes,
            "arrows": [{"id": "edge", "source": "a", "target": "b", "relation_type": "prediction"}],
            "text_program": {"items": []},
        })
        self.assertFalse({"route_style", "stroke_color", "stroke_width_pt", "line_pattern"} & set(bare["arrows"][0]))

    def test_feedback_route_prefers_the_local_channel(self):
        # <AUTHOR>'s ruling: two adjacent boxes get a short local loop between
        # them; only a blocked local lane may fall back to the canvas-edge lane.
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .1, "y": .2, "w": .1, "h": .1}},
                {"id": "b", "bbox_percent": {"x": .7, "y": .2, "w": .1, "h": .1}},
            ],
            "arrows": [{"id": "feedback", "source": "b", "target": "a", "relation_type": "feedback"}],
            "text_program": {"items": []},
        }
        updated, _ = resolve_direct_connectors(program)
        arrow = updated["arrows"][0]
        route = arrow["route_selection"]
        path = arrow["path_percent"]
        self.assertTrue(route["selected_candidate"].startswith("local_loop"))
        self.assertEqual(route["obstacle_hits"], 0)
        self.assertGreaterEqual(len(path), 4)
        ys = [point[1] for point in path]
        self.assertLess(min(ys), .2)

    def test_a_link_between_offset_boxes_lands_on_the_lane_they_share(self):
        # Reference figures line boxes up so the connector is one clean stroke.
        # These two faces overlap between y .32 and .40, so the arrow leaves the
        # tall box and enters the short one on that shared lane instead of
        # adding a corner only because the two centres disagree.
        program = {
            "semantic_nodes": [
                {"id": "tall", "bbox_percent": {"x": .10, "y": .20, "w": .16, "h": .30}},
                {"id": "short", "bbox_percent": {"x": .70, "y": .32, "w": .16, "h": .08}},
            ],
            "arrows": [{"id": "edge", "source": "tall", "target": "short", "relation_type": "data_flow"}],
            "text_program": {"items": []},
        }
        updated, _ = resolve_direct_connectors(program)
        path = updated["arrows"][0]["path_percent"]
        self.assertEqual(2, len(path), path)
        self.assertEqual(round(path[0][1], 4), round(path[1][1], 4), path)
        self.assertEqual([.26, .36], path[0])
        self.assertEqual([.70, .36], path[1])

    def test_faces_that_do_not_overlap_still_get_an_elbow(self):
        # Without a shared lane the old behaviour stands: the two faces are
        # entered at their mid points and the router decides the corners.
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .10, "w": .10, "h": .10}},
                {"id": "b", "bbox_percent": {"x": .70, "y": .60, "w": .10, "h": .10}},
            ],
            "arrows": [{"id": "edge", "source": "a", "target": "b", "relation_type": "data_flow"}],
            "text_program": {"items": []},
        }
        updated, _ = resolve_direct_connectors(program)
        arrow = updated["arrows"][0]
        self.assertEqual(4, len(arrow["path_percent"]), arrow["path_percent"])
        self.assertAlmostEqual(.15, arrow["path_percent"][0][0], places=4)
        self.assertAlmostEqual(.15, arrow["path_percent"][0][1], places=4)

    def test_a_blocked_link_steps_past_the_obstacle_instead_of_the_canvas_edge(self):
        # A card parked between two near neighbours used to send the arrow to the
        # slide border and back. It now steps just past the card, and the wide
        # canvas lanes stay a last resort.
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .40, "y": .45, "w": .10, "h": .10}},
                {"id": "card", "bbox_percent": {"x": .52, "y": .42, "w": .06, "h": .16}},
                {"id": "b", "bbox_percent": {"x": .60, "y": .45, "w": .10, "h": .10}},
            ],
            "arrows": [{"id": "edge", "source": "a", "target": "b", "relation_type": "data_flow"}],
            "text_program": {"items": []},
        }
        updated, _ = resolve_direct_connectors(program)
        arrow = updated["arrows"][0]
        selection = arrow["route_selection"]
        self.assertEqual(0, selection["obstacle_hits"])
        self.assertTrue(selection["selected_candidate"].startswith("detour"), selection["selected_candidate"])
        self.assertLess(sum(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in zip(arrow["path_percent"][:-1], arrow["path_percent"][1:])), 1.0)
        self.assertGreater(min(point[1] for point in arrow["path_percent"]), .05)


    def test_a_route_that_would_cross_a_routed_line_is_penalised(self):
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .1, "y": .25, "w": .12, "h": .12}},
                {"id": "b", "bbox_percent": {"x": .7, "y": .25, "w": .12, "h": .12}},
                {"id": "c", "bbox_percent": {"x": .1, "y": .65, "w": .12, "h": .12}},
                {"id": "d", "bbox_percent": {"x": .7, "y": .65, "w": .12, "h": .12}},
            ],
            "arrows": [
                {"id": "top", "source": "a", "target": "b", "relation_type": "data_flow"},
                {"id": "bottom", "source": "c", "target": "d", "relation_type": "data_flow"},
                {"id": "cross", "source": "b", "target": "c", "relation_type": "data_flow"},
            ],
            "text_program": {"items": []},
        }
        updated, _ = resolve_direct_connectors(program)
        arrows = {str(arrow.get("id")): arrow for arrow in updated["arrows"]}
        candidates = arrows["cross"]["route_selection"]["candidates"]
        self.assertTrue(any(int(item.get("crossings") or 0) >= 0 for item in candidates))
        # The crossing count is part of the deterministic score, so a clean
        # candidate always wins over a cheaper one that slices another line.
        best = min(candidates, key=lambda item: item["score"])
        self.assertEqual(best["crossings"], min(int(item.get("crossings") or 0) for item in candidates if item["obstacle_hits"] == 0))

    def test_prediction_generates_editable_bezier_samples(self):
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .2, "w": .1, "h": .1}},
                {"id": "b", "bbox_percent": {"x": .7, "y": .6, "w": .1, "h": .1}},
            ],
            "arrows": [{"id": "curve", "source": "a", "target": "b", "relation_type": "prediction"}],
            "text_program": {"items": []},
        }
        updated, _ = resolve_direct_connectors(program)
        arrow = updated["arrows"][0]
        self.assertEqual(arrow["path_kind"], "bezier_polyline")
        self.assertGreaterEqual(arrow["curve_sample_count"], 8)

    def test_existing_planned_path_is_preserved_when_clear(self):
        path = [[.15, .25], [.4, .15], [.75, .65]]
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .2, "w": .1, "h": .1}},
                {"id": "b", "bbox_percent": {"x": .7, "y": .6, "w": .1, "h": .1}},
            ],
            "arrows": [{"id": "planned", "source": "a", "target": "b", "relation_type": "data_flow", "path_percent": path}],
            "text_program": {"items": []},
        }
        updated, _ = resolve_direct_connectors(program)
        self.assertEqual(updated["arrows"][0]["path_percent"], path)


    def test_an_author_label_on_an_arrow_reaches_the_editable_text_program(self):
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .2, "w": .15, "h": .15}},
                {"id": "b", "bbox_percent": {"x": .7, "y": .5, "w": .15, "h": .15}},
            ],
            "arrows": [
                {"id": "conditional", "source": "a", "target": "b", "relation_type": "data_flow", "label": "if R == image_to_text", "label_evidence_ids": ["E7"]},
                {"id": "plain", "source": "b", "target": "a", "relation_type": "data_flow"},
            ],
            "text_program": {"items": []},
        }
        updated, _ = resolve_direct_connectors(program)
        labels = [item for item in updated["text_program"]["items"] if item.get("role") == "connector_label"]
        self.assertEqual(1, len(labels))
        self.assertEqual("if R == image_to_text", labels[0]["text"])
        self.assertEqual("conditional", labels[0]["connector_id"])
        self.assertEqual("conditional", labels[0]["target_id"])
        self.assertEqual(["E7"], labels[0]["evidence_ids"])
        self.assertIn("path_binding", labels[0])

    def test_an_arrow_without_a_label_never_produces_label_text(self):
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .2, "w": .15, "h": .15}},
                {"id": "b", "bbox_percent": {"x": .7, "y": .5, "w": .15, "h": .15}},
            ],
            "arrows": [{"id": "plain", "source": "a", "target": "b", "relation_type": "data_flow", "label": ""}],
            "text_program": {"items": []},
        }
        updated, report = resolve_direct_connectors(program)
        self.assertEqual([], [item for item in updated["text_program"]["items"] if item.get("role") == "connector_label"])
        self.assertEqual(0, report["labeled_connector_count"])

    def test_a_label_without_a_route_is_kept_instead_of_being_dropped(self):
        # The arrow survives but its endpoint did, so no path exists and no label
        # text can be written yet. The planned relation text must not vanish.
        program = {
            "semantic_nodes": [
                {"id": "a", "bbox_percent": {"x": .05, "y": .2, "w": .15, "h": .15}},
                {"id": "b", "bbox_percent": {"x": .7, "y": .5, "w": .15, "h": .15}},
            ],
            "arrows": [{"id": "dropped", "source": "a", "target": "vanished", "relation_type": "data_flow", "label": "left over relation"}],
            "labels": [{"id": "planned", "target_id": "dropped", "text": "left over relation", "bbox_percent": {"x": .4, "y": .3, "w": .1, "h": .04}}],
            "text_program": {"items": []},
        }
        updated, report = resolve_direct_connectors(program)
        self.assertEqual(0, report["connectors"][0]["point_count"])
        self.assertEqual([], [item for item in updated["text_program"]["items"] if item.get("role") == "connector_label"])
        self.assertEqual(["planned"], [item["id"] for item in updated["labels"]])


class ThePersonRuleOutranksTheRoutePickerTests(unittest.TestCase):
    """A route the picker offers cannot undo a connector a person's sentence settled.

    The candidates the picker chooses from were scored from the plan, before the person's
    sentence was read, so every one of them is a shape the person may have just rejected.
    """

    def _program(self, status: str = "person_rule_direct") -> dict:
        return {
            "arrows": [
                {
                    "id": "edge",
                    "path_percent": [[0.8725, 0.273], [0.8725, 0.38]],
                    "route_generation_status": status,
                    "route_selection": {
                        "selected_candidate": "planned",
                        "candidates": [
                            {
                                "candidate_id": "side_lane",
                                "path": [[0.8725, 0.273], [0.975, 0.275], [0.975, 0.38], [0.8725, 0.38]],
                                "obstacle_hits": 0,
                            }
                        ],
                    },
                }
            ]
        }

    def test_the_detour_the_person_rejected_cannot_come_back_through_the_picker(self):
        updated, report = apply_connector_route_selections(self._program(), [{"arrow_id": "edge", "candidate_id": "side_lane"}])
        self.assertEqual([[0.8725, 0.273], [0.8725, 0.38]], updated["arrows"][0]["path_percent"])
        self.assertEqual("person_rule_direct", updated["arrows"][0]["route_generation_status"])
        self.assertEqual(["edge"], report["person_rule_kept"])
        self.assertEqual([], report["applied_arrow_ids"])

    def test_an_arrow_the_person_never_spoke_about_is_still_the_pickers_to_choose(self):
        updated, report = apply_connector_route_selections(self._program("deterministic_reroute"), [{"arrow_id": "edge", "candidate_id": "side_lane"}])
        self.assertEqual(4, len(updated["arrows"][0]["path_percent"]))
        self.assertEqual(["edge"], report["applied_arrow_ids"])
        self.assertEqual([], report["person_rule_kept"])


if __name__ == "__main__":
    unittest.main()
