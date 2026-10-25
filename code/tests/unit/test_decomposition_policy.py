"""Tests for the deterministic decomposability -> bounded replan policy."""

import unittest

from rfs.contracts.repair_actions import SLOT_NUDGE_DEFAULT
from rfs.evolution.direct import action_is_executable
from rfs.planning.decomposition_policy import (
    DECOMPOSITION_POLICY_CONTRACT,
    SOFT_ESCALATION_STREAK,
    plan_decomposition_action,
)
from rfs.workflows.paper_evolve_editable import _decomposition_proposal


def _report(*, hard: bool = False, soft: bool = False, streak: int = 1, object_ids: list[str] | None = None, metric: str = "cross_panel_connectors") -> dict:
    violations: list[dict] = []
    if hard:
        violations.append({
            "metric": "orphan_nodes",
            "value": 1,
            "threshold": 0,
            "kind": "hard",
            "reason": "unpaired",
            "object_ids": object_ids if object_ids is not None else ["n1"],
        })
    if soft:
        if metric == "cross_panel_connectors":
            violations.append({
                "metric": "cross_panel_connectors",
                "value": 3,
                "threshold": 2,
                "kind": "soft",
                "reason": "crowded",
                "object_ids": ["e1", "e2", "e3"],
                "target_connector_ids": ["e1", "e2", "e3"],
                "target_node_ids": ["n1", "n2"],
            })
        else:
            violations.append({
                "metric": "max_node_slot_fanout",
                "value": 5,
                "threshold": 3,
                "kind": "soft",
                "reason": "one node carries too much",
                "object_ids": ["n7"],
                "target_node_ids": ["n7"],
            })
    status = "hard_violations" if hard else "soft_violations" if soft else "decomposable"
    return {
        "contract": "rfs.decomposability_report",
        "version": 1,
        "round": 0,
        "status": status,
        "violations": violations,
        "repeat": {"soft_streak": streak if soft else 0},
    }


class DecompositionPolicyTests(unittest.TestCase):
    def test_clean_report_asks_for_nothing(self):
        decision = plan_decomposition_action(_report())
        self.assertEqual(decision["contract"], DECOMPOSITION_POLICY_CONTRACT)
        self.assertFalse(decision["required"])
        self.assertEqual(decision["action"], "")
        self.assertEqual(decision["violations"], [])

    def test_hard_break_escalates_at_once(self):
        decision = plan_decomposition_action(_report(hard=True))
        self.assertTrue(decision["required"])
        self.assertEqual(decision["action"], "resegment")
        self.assertEqual(decision["scope_type"], "global")
        self.assertTrue(decision["escalation"])
        self.assertEqual(decision["object_ids"], ["n1"])

    def test_a_crossed_connector_break_is_routed_not_relaid_out(self):
        # `relayout_slots` only ever moves nodes, so handing it connector ids is an action
        # that cannot touch any of its own targets.
        decision = plan_decomposition_action(_report(soft=True, streak=1))
        self.assertEqual(decision["action"], "reroute")
        self.assertEqual(decision["scope_type"], "connector")
        self.assertFalse(decision["escalation"])
        self.assertEqual(decision["target_connector_ids"], ["e1", "e2", "e3"])
        self.assertEqual(decision["object_ids"], ["e1", "e2", "e3"])

    def test_a_node_break_is_relaid_out_locally(self):
        decision = plan_decomposition_action(_report(soft=True, streak=1, metric="max_node_slot_fanout"))
        self.assertEqual(decision["action"], "relayout_slots")
        self.assertEqual(decision["scope_type"], "slot")
        self.assertFalse(decision["escalation"])
        self.assertEqual(decision["target_node_ids"], ["n7"])

    def test_block_scope_still_routes_a_connector_break(self):
        decision = plan_decomposition_action(_report(soft=True, streak=1), evolution_scope="block")
        self.assertEqual(decision["action"], "reroute")
        self.assertFalse(decision["escalation"])

    def test_repeated_soft_break_escalates(self):
        decision = plan_decomposition_action(_report(soft=True, streak=SOFT_ESCALATION_STREAK))
        self.assertEqual(decision["action"], "replan_structure")
        self.assertTrue(decision["escalation"])

    def test_block_scope_never_escalates_a_soft_break(self):
        decision = plan_decomposition_action(_report(soft=True, streak=5), evolution_scope="block")
        self.assertEqual(decision["action"], "relayout_slots")
        self.assertFalse(decision["escalation"])
        self.assertIn("block-only", decision["reason"])

    def test_block_scope_declines_a_hard_break_it_cannot_repair(self):
        decision = plan_decomposition_action(_report(hard=True), evolution_scope="block")
        self.assertFalse(decision["required"])
        self.assertEqual(decision["action"], "")
        self.assertFalse(decision["escalation"])
        self.assertIn("block-only", decision["reason"])

    def test_decision_is_reproducible(self):
        report = _report(soft=True, streak=2)
        self.assertEqual(
            plan_decomposition_action(report, evolution_scope="joint"),
            plan_decomposition_action(report, evolution_scope="joint"),
        )

    def test_missing_report_is_inert(self):
        decision = plan_decomposition_action(None)
        self.assertFalse(decision["required"])
        self.assertEqual(decision["action"], "")

    def test_a_local_reflow_proposal_carries_the_catalog_budget(self):
        policy = plan_decomposition_action(_report(soft=True, streak=1, metric="max_node_slot_fanout"))
        proposal = _decomposition_proposal(policy)
        self.assertEqual("relayout_slots", proposal["type"])
        self.assertTrue(action_is_executable(proposal))
        for key, value in SLOT_NUDGE_DEFAULT.items():
            self.assertEqual(value, proposal[key])

    def test_a_routing_proposal_stays_executable_without_a_geometry_budget(self):
        policy = plan_decomposition_action(_report(soft=True, streak=1))
        proposal = _decomposition_proposal(policy)
        self.assertEqual("reroute", proposal["type"])
        self.assertTrue(action_is_executable(proposal))


if __name__ == "__main__":
    unittest.main()
