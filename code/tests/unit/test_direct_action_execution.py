"""Tests for real bounded direct-evolution action execution."""

from __future__ import annotations

from copy import deepcopy
import unittest

from rfs.contracts.repair_actions import attach_heuristic_budget
from rfs.workflows.direct_repair_execution import execute_direct_repair


def _state() -> dict:
    return {
        "method": {"modules": [{"id": "n1", "label": "Science"}]},
        "layout": {
            "nodes": [
                {"id": "n1", "bbox_percent": {"x": .1, "y": .1, "w": .2, "h": .2}},
                {"id": "n2", "bbox_percent": {"x": .5, "y": .1, "w": .2, "h": .2}},
            ],
            "connectors": [{"id": "e1", "source": "n1", "target": "n2", "path_percent": [[.3, .2], [.5, .2]]}],
        },
        "icons": {"icons": [{"asset_id": "icon_n1", "semantic_node_ids": ["n1"], "prompt": "icon"}]},
    }


class DirectActionExecutionTests(unittest.TestCase):
    def test_every_supported_action_is_auditable_and_preserves_input_science(self) -> None:
        # The escalations are executed by `upstream_retry`, not here, so this pass has nothing
        # to change for them. They are listed apart rather than dropped, because the audit
        # still has to carry the request.
        executed_here = [
            attach_heuristic_budget({"type": "relayout_global"}),
            attach_heuristic_budget({"type": "relayout_slots", "semantic_node_ids": ["n1"]}),
            attach_heuristic_budget({"type": "resize_nodes", "semantic_node_ids": ["n1"]}),
            {"type": "reroute", "edge_ids": ["e1"]},
            attach_heuristic_budget({"type": "repair_interfaces", "edge_ids": ["e1"]}),
            {"type": "regen_assets", "slot_ids": ["icon_n1"]},
        ]
        recorded_only = [
            {"type": "reanalyze_semantics", "semantic_node_ids": ["n1"]},
            {"type": "replan_theme", "target_roles": ["panel"]},
            {"type": "replan_structure"},
            {"type": "resegment"},
        ]
        for action in executed_here + recorded_only:
            with self.subTest(action=action["type"]):
                original = _state()
                frozen = deepcopy(original)
                updated, report, before, after = execute_direct_repair(original, action, round_index=1)
                self.assertEqual(action in executed_here, report["applied"])
                self.assertTrue(report["preserve_checks"]["scientific_contract_unchanged"])
                self.assertEqual(frozen, original)
                self.assertEqual(before["state_hash"], report["object_diff"]["before_hash"])
                self.assertEqual(after["state_hash"], report["object_diff"]["after_hash"])
                self.assertEqual(frozen["method"], updated["method"])

    def test_an_escalation_this_pass_cannot_execute_leaves_the_state_alone(self) -> None:
        # These used to append a request list to the state. No module read it, and it made the
        # state diff report a change, so an escalation the workflow had declined still read as
        # an applied repair.
        for action in (
            {"type": "reanalyze_semantics", "semantic_node_ids": ["n1"]},
            {"type": "replan_theme"},
            {"type": "replan_structure"},
            {"type": "resegment"},
        ):
            with self.subTest(action=action["type"]):
                updated, report, _, _ = execute_direct_repair(_state(), action, round_index=1)
                self.assertEqual(_state(), updated)
                self.assertFalse(report["applied"])
                self.assertFalse(report["program_applied"])
                self.assertEqual([], report["object_diff"]["changed_fields"])
                self.assertTrue(report["program_changes"])

    def test_a_declined_escalation_is_audited_without_moving_boxes(self):
        from rfs.workflows.direct_repair_execution import record_unexecuted_repair

        original = _state()
        updated, report, _, _ = record_unexecuted_repair(original, {"type": "replan_structure"}, round_index=1, reason="planner_mode_heuristic")
        self.assertIs(updated, original)
        self.assertFalse(report["applied"])
        self.assertEqual("planner_mode_heuristic", report["declined_reason"])
        self.assertEqual(original["layout"]["nodes"], updated["layout"]["nodes"])


    def test_recompile_is_no_longer_an_action_the_controller_will_run(self) -> None:
        # Every round compiles its candidate, and the compile is deterministic in the state it
        # is handed, so a recompile over unchanged state re-emits the same figure. It is gone
        # from the vocabulary rather than kept as an action that is always refused.
        with self.assertRaises(ValueError):
            execute_direct_repair(_state(), {"type": "recompile"}, round_index=1)

    def test_reroute_invalidates_only_target_path(self) -> None:
        state = _state()
        state["layout"]["connectors"].append({"id": "e2", "source": "n2", "target": "n1", "path_percent": [[.5, .2], [.3, .2]]})
        updated, report, _, _ = execute_direct_repair(state, {"type": "reroute", "edge_ids": ["e1"]}, round_index=2)
        self.assertEqual([], updated["layout"]["connectors"][0]["path_percent"])
        self.assertNotEqual([], updated["layout"]["connectors"][1]["path_percent"])
        self.assertEqual("e1", report["program_changes"][0]["object_id"])

    def test_a_node_action_handed_no_matching_node_reports_no_change(self) -> None:
        # A connector id cannot be laid out. The report has to say nothing moved rather than
        # report success because the action was requested.
        _, report, _, _ = execute_direct_repair(
            _state(),
            {"type": "relayout_slots", "object_ids": ["e1"], "scope_type": "slot"},
            round_index=1,
        )
        self.assertFalse(report["applied"])
        self.assertFalse(report["geometric_applied"])
        self.assertFalse(report["object_diff"]["changed"])
        self.assertEqual([], report["changed_node_ids"])

    def test_a_program_action_that_matched_nothing_reports_no_change(self) -> None:
        _, report, _, _ = execute_direct_repair(_state(), {"type": "reroute", "edge_ids": ["e9"]}, round_index=1)
        self.assertFalse(report["applied"])
        self.assertFalse(report["program_applied"])
        self.assertEqual([], report["program_changes"])

    def test_the_report_separates_the_geometric_pass_from_the_program_pass(self) -> None:
        _, report, _, _ = execute_direct_repair(
            _state(),
            attach_heuristic_budget({"type": "relayout_slots", "semantic_node_ids": ["n1"]}),
            round_index=1,
        )
        self.assertTrue(report["applied"])
        self.assertTrue(report["geometric_applied"])
        self.assertFalse(report["program_applied"])

    def test_the_execution_report_stamps_the_budget_onto_the_action(self):
        _, report, _, _ = execute_direct_repair(_state(), {"type": "relayout_global", "factor": 1.02, "dx": 0.016, "dy": 0.0}, round_index=1)
        self.assertEqual({"factor": 1.02, "dx": 0.016, "dy": 0.0}, report["applied_budget"])
        self.assertEqual(0.016, report["action"]["dx"])
        self.assertEqual("relayout_global", report["action"]["type"])


if __name__ == "__main__":
    unittest.main()
