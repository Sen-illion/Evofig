"""The bounded actions have one execution-semantics catalog."""

from __future__ import annotations

import unittest

from rfs.contracts.evolution import REPAIR_ACTIONS
from rfs.contracts.repair_actions import (
    ACTION_SEMANTICS,
    CATALOG_LAYERS,
    GLOBAL_NUDGE_DEFAULT,
    SLOT_NUDGE_DEFAULT,
    action_needs_budget,
    attach_heuristic_budget,
    heuristic_default,
    prescription_cannot_contrast,
    prescription_matches_heuristic_default,
    render_action_catalog_for_prompt,
    validate_action_catalog,
)
from rfs.evolution.direct.controller import apply_bounded_action


class RepairActionSemanticsTests(unittest.TestCase):
    def test_every_contract_action_has_a_catalog_row(self):
        self.assertEqual([], validate_action_catalog())
        self.assertEqual(set(ACTION_SEMANTICS), set(REPAIR_ACTIONS))

    def test_global_nudge_actions_share_the_executor_default(self):
        self.assertEqual(heuristic_default("relayout_global"), GLOBAL_NUDGE_DEFAULT)
        self.assertTrue(action_needs_budget("relayout_global"))

    def test_replan_structure_is_an_upstream_request_not_a_nudge(self):
        for name in ("replan_structure", "resegment"):
            self.assertEqual(heuristic_default(name), {})
            self.assertFalse(action_needs_budget(name))
            self.assertIn("upstream_retry", ACTION_SEMANTICS[name]["does"])
        cannot = ACTION_SEMANTICS["replan_structure"]["cannot"]
        self.assertIn("does not move boxes", cannot)

    def test_slot_nudge_default_matches_first_retry(self):
        self.assertEqual(heuristic_default("relayout_slots"), SLOT_NUDGE_DEFAULT)
        self.assertEqual(heuristic_default("resize_nodes"), SLOT_NUDGE_DEFAULT)

    def test_non_geometric_actions_carry_no_budget(self):
        for name in ("reanalyze_semantics", "replan_theme", "reroute", "regen_assets"):
            self.assertFalse(action_needs_budget(name))
            self.assertEqual(heuristic_default(name), {})

    def test_heuristic_default_is_what_the_gate_writes_not_what_the_executor_invents(self):
        layout = {
            "nodes": [
                {"id": "a", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}},
                {"id": "b", "bbox_percent": {"x": 0.5, "y": 0.1, "w": 0.2, "h": 0.2}},
            ]
        }
        _, _, empty = apply_bounded_action(layout, {}, {"type": "relayout_global"})
        self.assertEqual(empty["applied_budget"], {"factor": 1.0, "dx": 0.0, "dy": 0.0})

        prescribed = attach_heuristic_budget({"type": "relayout_global"})
        _, _, report = apply_bounded_action(layout, {}, prescribed)
        self.assertEqual(report["applied_budget"], GLOBAL_NUDGE_DEFAULT)

        slot_layout = {"nodes": [{"id": "n", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]}
        slot = attach_heuristic_budget({"type": "relayout_slots", "semantic_node_ids": ["n"]})
        _, _, slot_report = apply_bounded_action(slot_layout, {}, slot)
        self.assertEqual(slot_report["applied_budget"], SLOT_NUDGE_DEFAULT)

    def test_a_row_with_the_shipped_numbers_is_the_default_policy(self):
        self.assertTrue(prescription_matches_heuristic_default({"type": "relayout_global", **GLOBAL_NUDGE_DEFAULT}))
        self.assertFalse(prescription_matches_heuristic_default({"type": "relayout_global", "factor": 1.02, "dx": 0.016, "dy": 0.0}))
        self.assertFalse(prescription_matches_heuristic_default({"type": "relayout_global"}))
        self.assertTrue(prescription_matches_heuristic_default({"type": "relayout_slots", **SLOT_NUDGE_DEFAULT}))
        self.assertFalse(prescription_matches_heuristic_default({"type": "relayout_slots", "semantic_node_ids": ["n"], **SLOT_NUDGE_DEFAULT}))

    def test_a_row_that_cannot_contrast_names_why(self):
        self.assertIn("default policy", prescription_cannot_contrast({"type": "relayout_global", **GLOBAL_NUDGE_DEFAULT}))
        self.assertIn("missing the budget", prescription_cannot_contrast({"type": "relayout_global"}))
        self.assertEqual("", prescription_cannot_contrast({"type": "relayout_global", "factor": 1.02, "dx": 0.016, "dy": 0.0}))
        self.assertEqual("", prescription_cannot_contrast({"type": "reroute", "edge_ids": ["e1"]}))

    def test_the_prompt_catalog_is_generated_from_the_same_rows(self):
        text = render_action_catalog_for_prompt()
        for name in REPAIR_ACTIONS:
            self.assertIn(name + ":", text)
        self.assertIn("empty requested_actions list", text)
        self.assertIn("upstream_retry", text)

    def test_every_row_says_when_it_is_the_answer(self):
        for name, row in ACTION_SEMANTICS.items():
            self.assertTrue(str(row.get("when") or "").strip(), f"{name} names no fault it answers")
        self.assertEqual([], validate_action_catalog())

    def test_the_layers_cover_the_catalog_exactly_once(self):
        layered = [name for _, _, names in CATALOG_LAYERS for name in names]
        self.assertEqual(sorted(layered), sorted(REPAIR_ACTIONS))

    def test_the_menu_is_read_in_the_order_faults_are_answered(self):
        text = render_action_catalog_for_prompt()
        object_level = [text.index(name + ":") for name in CATALOG_LAYERS[0][2]]
        self.assertEqual(object_level, sorted(object_level))
        self.assertLess(text.index("object-level actions"), text.index("compile-level rules"))
        self.assertLess(text.index("compile-level rules"), text.index("stage-level requests"))
        # The shipped default budgets belong to the executor. A seat that reads them back is
        # naming the policy experience has to differ from, so the prompt must not print them.
        self.assertNotIn("heuristic_default", text)
        self.assertNotIn("budget_keys=", text)


if __name__ == "__main__":
    unittest.main()
