"""Tests for the design plan contract's tolerance of model-shaped rectangles."""

import unittest

from rfs.contracts.design_plan import normalize_design_plan, validate_design_plan
from rfs.planning.canvas_constraints import fit_design_to_vertical_safe_area


def _plan(**extra):
    base = {
        "panels": [
            {"id": "panel_a", "bbox_percent": {"x": 0.05, "y": 0.2, "w": 0.4, "h": 0.5}},
            {"id": "panel_b", "bbox_percent": {"x": 0.55, "y": 0.2, "w": 0.4, "h": 0.5}},
        ],
        "nodes": [
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.1}},
        ],
    }
    base.update(extra)
    return normalize_design_plan(base)


class BBoxReadingTests(unittest.TestCase):
    def test_a_half_drawn_box_keeps_the_numbers_it_has_and_gains_none(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2}},
        ])

        self.assertEqual(plan["nodes"][0]["bbox_percent"], {"x": 0.1, "y": 0.3, "w": 0.2})

    def test_string_numbers_and_the_models_own_field_names_are_read(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"left": "0.1", "top": "0.3", "width": "0.2", "height": "0.1"}},
        ])

        self.assertEqual(plan["nodes"][0]["bbox_percent"], {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.1})

    def test_a_corner_box_is_read_through_its_own_field_names(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"x1": 0.1, "y1": 0.3, "x2": 0.3, "y2": 0.4}},
        ])

        self.assertAlmostEqual(plan["nodes"][0]["bbox_percent"]["w"], 0.2)
        self.assertAlmostEqual(plan["nodes"][0]["bbox_percent"]["h"], 0.1)

    def test_a_corner_list_still_becomes_a_box(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": [0.1, 0.3, 0.3, 0.4]},
        ])

        box = plan["nodes"][0]["bbox_percent"]

        self.assertAlmostEqual(box["x"], 0.1)
        self.assertAlmostEqual(box["w"], 0.2)

    def test_a_nonsense_field_is_reported_as_absent_not_invented(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2, "h": {"value": 0.1}}},
        ])

        self.assertNotIn("h", plan["nodes"][0]["bbox_percent"])


class ParentReferenceTests(unittest.TestCase):
    """The model writes two different things into ``parent_id`` in one response."""

    def test_a_parent_id_naming_a_panel_is_read_as_the_membership_it_states(self):
        # Half-written membership used to leave a finished plan permanently illegal: the validator
        # refuses a node without panel_id that a panel contains, so one missing field marked the
        # whole figure a failure and every replay of it a stage the contract had outgrown.
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "parent_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.1}},
        ])

        self.assertEqual(plan["nodes"][0]["panel_id"], "panel_a")
        self.assertTrue(validate_design_plan(plan)["ok"])
        self.assertEqual(
            plan["provenance"]["repairs"],
            [{"node_id": "node_a", "field": "panel_id", "value": "panel_a",
              "read_as": "panel_membership", "parent_id": "panel_a"}],
        )

    def test_a_member_that_already_declares_its_panel_keeps_its_own_owner_note(self):
        # The validator never reads parent_id once panel_id is set, and the layout solver reads it
        # as the owning container, so a node that already says which panel it belongs to is left
        # exactly as the model wrote it.
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "parent_id": "panel_a", "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.1}},
        ])

        self.assertEqual(plan["nodes"][0]["parent_id"], "panel_a")
        self.assertNotIn("repairs", plan["provenance"])
        self.assertTrue(validate_design_plan(plan)["ok"])

    def test_a_parent_id_no_declared_node_owns_is_dropped_and_recorded(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "parent_id": "ghost_module",
             "bbox_percent": {"x": 0.9, "y": 0.05, "w": 0.05, "h": 0.05}},
        ])

        self.assertIsNone(plan["nodes"][0]["parent_id"])
        self.assertTrue(validate_design_plan(plan)["ok"])
        self.assertEqual(plan["provenance"]["repairs"][0]["read_as"], "dropped_undeclared_parent")

    def test_a_panel_note_the_box_contradicts_is_dropped_rather_than_believed(self):
        # The shape this contract actually met: the model named a panel in parent_id for a card
        # whose box sits outside that panel. Believing the note would hand the card a panel_id it
        # was never drawn with, which moves it into that panel's payload and changes a figure the
        # source arm had already compiled the other way. The stale note is dropped instead, and the
        # card stays the canvas-level box the arm drew.
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "parent_id": "panel_a",
             "bbox_percent": {"x": 0.92, "y": 0.45, "w": 0.12, "h": 0.12}},
        ])

        node = plan["nodes"][0]
        self.assertIsNone(node["parent_id"])
        self.assertIsNone(node.get("panel_id"))
        self.assertEqual(plan["provenance"]["repairs"][0]["read_as"], "dropped_dangling_parent")

    def test_an_overhanging_rectangle_still_has_to_be_decided_by_the_model(self):
        # Dropping the note is not a licence to accept the box: an overhanging rectangle cannot be
        # read as anything the model said, so the designer is asked for it and the plan stays
        # illegal until it answers.
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "parent_id": "panel_a",
             "bbox_percent": {"x": 0.92, "y": 0.45, "w": 0.12, "h": 0.12}},
        ])

        validation = validate_design_plan(plan)
        self.assertFalse(validation["ok"])
        self.assertIn("node[0] requires a legal bbox_percent", validation["errors"])

    def test_a_node_outside_every_panel_and_every_parent_is_still_an_orphan(self):
        # A card the model left inside no panel and inside no declared parent is a real orphan: the
        # contract keeps calling it one instead of inventing a panel for it.
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a",
             "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.1}},
        ])

        validation = validate_design_plan(plan)
        self.assertFalse(validation["ok"])
        self.assertIn("node[0] without panel_id must be a root-level node outside every panel", validation["errors"])

    def test_a_parent_id_naming_a_declared_node_is_left_as_hierarchy(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.4, "h": 0.2}},
            {"id": "node_b", "semantic_source_id": "module_b", "panel_id": "panel_a",
             "parent_id": "node_a", "bbox_percent": {"x": 0.15, "y": 0.35, "w": 0.1, "h": 0.1}},
        ])

        self.assertEqual(plan["nodes"][1]["parent_id"], "node_a")
        self.assertNotIn("repairs", plan["provenance"])
        self.assertTrue(validate_design_plan(plan)["ok"])


class SafeAreaFitTests(unittest.TestCase):
    def test_a_half_drawn_box_no_longer_crashes_the_fitter(self):
        plan = _plan(panels=[
            {"id": "panel_a", "bbox_percent": {"x": 0.05, "y": 0.2, "w": 0.4}},
            {"id": "panel_b", "bbox_percent": {"x": 0.55, "y": 0.2, "w": 0.4, "h": 0.5}},
        ])

        fitted, report = fit_design_to_vertical_safe_area(plan)

        self.assertEqual(report["unreadable_boxes"]["count"], 1)
        self.assertEqual(report["unreadable_boxes"]["objects"][0]["id"], "panel_a")
        self.assertEqual(fitted["panels"][0]["bbox_percent"], {"x": 0.05, "y": 0.2, "w": 0.4})

    def test_the_readable_boxes_are_still_fitted_together(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.02, "w": 0.2, "h": 0.1}},
            {"id": "node_b", "semantic_source_id": "module_b", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.99, "w": 0.2, "h": 0.1}},
        ])

        fitted, report = fit_design_to_vertical_safe_area(plan)

        self.assertTrue(report["transformed"])
        self.assertGreaterEqual(fitted["nodes"][0]["bbox_percent"]["y"], 0.11)
        bottom = fitted["nodes"][1]["bbox_percent"]["y"] + fitted["nodes"][1]["bbox_percent"]["h"]
        self.assertLessEqual(round(bottom, 4), 0.96)

    def test_a_half_drawn_box_is_still_illegal_so_the_designer_is_asked_for_it(self):
        plan = _plan(panels=[
            {"id": "panel_a", "bbox_percent": {"x": 0.05, "y": 0.2, "w": 0.4}},
            {"id": "panel_b", "bbox_percent": {"x": 0.55, "y": 0.2, "w": 0.4, "h": 0.5}},
        ])

        fitted, _report = fit_design_to_vertical_safe_area(plan)
        validation = validate_design_plan(fitted)

        self.assertFalse(validation["ok"])
        self.assertIn("panel[0] requires id and legal bbox_percent", validation["errors"])


def _method(**extra):
    """A method contract whose objects are the ones a figure has to carry."""
    base = {
        "modules": [
            {"id": "module_a", "label": "Encoder", "must_show": True, "should_render": True},
            {"id": "module_b", "label": "Retriever", "must_show": True, "should_render": True},
            {"id": "module_c", "label": "Reranker", "must_show": True, "should_render": True},
        ],
        "optional_details": [
            {"id": "detail_a", "label": "temperature", "must_show": False, "should_render": False},
        ],
    }
    base.update(extra)
    return base


class MethodCoverageTests(unittest.TestCase):
    def test_a_design_that_draws_one_of_three_required_objects_is_named_incomplete(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.1}},
        ])

        validation = validate_design_plan(plan, _method())

        self.assertFalse(validation["ok"])
        self.assertIn(
            "design plan represents 1 of 3 required method objects; missing: Retriever, Reranker",
            validation["errors"],
        )

    def test_a_design_that_draws_every_required_object_passes(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.1}},
            {"id": "node_b", "semantic_source_id": "module_b", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.45, "w": 0.2, "h": 0.1}},
            {"id": "node_c", "label": "Reranker", "semantic_source_id": "module_c", "panel_id": "panel_b",
             "bbox_percent": {"x": 0.6, "y": 0.3, "w": 0.2, "h": 0.1}},
        ])

        validation = validate_design_plan(plan, _method())

        self.assertTrue(validation["ok"])
        self.assertEqual(validation["method_coverage"]["coverage"], 1.0)

    def test_a_required_object_the_figure_only_spells_by_label_still_counts(self):
        # The model is allowed to keep its own node names: what has to match the
        # method contract is the content, not the identifier.
        plan = _plan(nodes=[
            {"id": "node_a", "label": "encoder", "semantic_source_id": "some_other_module",
             "panel_id": "panel_a", "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.1}},
        ])

        coverage = validate_design_plan(plan, _method())["method_coverage"]

        self.assertNotIn("module_a", coverage["missing_ids"])
        self.assertEqual(coverage["missing_ids"], ["module_b", "module_c"])

    def test_an_object_the_method_does_not_mark_for_the_figure_is_not_required(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.1}},
            {"id": "node_b", "semantic_source_id": "module_b", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.45, "w": 0.2, "h": 0.1}},
            {"id": "node_c", "semantic_source_id": "module_c", "panel_id": "panel_b",
             "bbox_percent": {"x": 0.6, "y": 0.3, "w": 0.2, "h": 0.1}},
        ])

        validation = validate_design_plan(plan, _method())

        self.assertTrue(validation["ok"])
        self.assertEqual(validation["method_coverage"]["required_count"], 3)

    def test_an_id_less_and_a_duplicated_row_stay_out_of_the_denominator(self):
        method = _method(
            modules=[
                {"label": "Encoder", "must_show": True, "should_render": True},
                {"id": "module_a", "label": "Encoder", "must_show": True, "should_render": True},
                {"id": "module_a", "label": "Encoder again", "must_show": True, "should_render": True},
            ],
            optional_details=[],
        )
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.1}},
        ])

        validation = validate_design_plan(plan, method)

        self.assertTrue(validation["ok"])
        self.assertEqual(validation["method_coverage"]["required_count"], 1)

    def test_a_plan_checked_without_a_method_contract_reports_no_coverage(self):
        plan = _plan()

        validation = validate_design_plan(plan)

        self.assertTrue(validation["ok"])
        self.assertIsNone(validation["method_coverage"])

    def test_a_caller_can_measure_coverage_without_gating_on_it(self):
        plan = _plan(nodes=[
            {"id": "node_a", "semantic_source_id": "module_a", "panel_id": "panel_a",
             "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.2, "h": 0.1}},
        ])

        validation = validate_design_plan(plan, _method(), require_method_coverage=False)

        self.assertTrue(validation["ok"])
        self.assertEqual(validation["method_coverage"]["missing_ids"], ["module_b", "module_c"])

if __name__ == "__main__":
    unittest.main()
