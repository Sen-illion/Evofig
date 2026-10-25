"""P6: a Memory row is a rule that travels, and the library keeps only the rule.

A row whose rule names ``rel_12`` cannot be retrieved for another paper, because the next
paper has no ``rel_12``; a row that restates the move the run would have made anyway is not a
finding. The gate that reads a row this way sits in the one place every write goes through.
"""

import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import ExperienceStore
from rfs.evolution.direct.controller import action_execution_fingerprint
from rfs.evolution.direct.rule_binding import (
    figure_labels,
    render_rule_statement,
    rule_binding_gate,
    split_rule_binding,
)
from rfs.evolution.direct.store import ALLOWED_APPEND_STATUSES, DEFAULT_RETRIEVAL_STATUSES


#: A figure that names its own parts, which is what the label check compares a rule against.
FIGURE = {
    "nodes": [{"id": "node_a", "label": "New Query"}, {"id": "node_b", "label": "Memory"}],
    "panels": [{"id": "panel_a", "label": "Multi-hop Personalized Reasoning Tasks"}],
}


def row(**overrides) -> dict:
    base = {
        "experience_id": "direct-exp-1",
        "domain": "direct_editable",
        "scope": "global",
        "apply_at": "ppt_compilation",
        "memory_type": "connector",
        "hypothesis": "a shared vertical channel keeps a one-to-many fan-out readable",
        "guidance": "Give the fan-out one vertical channel instead of letting each edge pick its own lane.",
        "action": {"type": "reroute", "constraints": {"shared_vertical_channel": True}},
        "condition": {"topology": "pipeline", "node_count_bucket": "high"},
        "graph_scope": {"edge_ids": ["rel_12"]},
        "accepted": True,
        "repair_gain": 0.12,
        "status": "candidate",
        "applicability": "common",
        "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.62},
    }
    base.update(overrides)
    return base


class SplitRuleBindingTests(unittest.TestCase):
    def test_the_move_travels_and_the_objects_stay_behind(self):
        rule, binding = split_rule_binding(row(action={"type": "reroute", "edge_ids": ["rel_12"], "constraints": {"shared_vertical_channel": True}}))

        self.assertNotIn("edge_ids", rule["action"])
        self.assertEqual({"shared_vertical_channel": True}, rule["action"]["constraints"])
        self.assertEqual(["rel_12"], binding["action"]["edge_ids"])
        self.assertEqual({"edge_ids": ["rel_12"]}, binding["graph_scope"])

    def test_the_coarse_statistics_stay_in_the_rule(self):
        rule, _ = split_rule_binding(row())

        self.assertEqual({"topology": "pipeline", "node_count_bucket": "high"}, rule["condition"])

    def test_the_sentence_about_this_figure_stays_behind(self):
        """The defect sentence is evidence about one drawing, so the library never sees it."""

        rule, binding = split_rule_binding(row(
            problem_type="the New Query box is too small",
            action={"type": "resize_nodes", "factor": 1.08, "problem": "the New Query box is too small", "expected_effect": "make it taller"},
        ))

        self.assertNotIn("problem_type", rule)
        self.assertEqual("the New Query box is too small", binding["problem_type"])
        self.assertNotIn("problem", rule["action"])
        self.assertEqual("make it taller", binding["action"]["expected_effect"])
        self.assertEqual(1.08, rule["action"]["factor"])

    def test_the_rendered_statement_names_the_condition_and_the_move(self):
        statement = render_rule_statement(
            condition={"defect_patterns": ["cross_panel"], "node_count_bucket": "high"},
            action={"type": "router_constraint", "constraints": {"shared_vertical_channel": True}},
            guidance="Route the fan-out down one channel.",
        )

        self.assertIn("the figure shows cross_panel", statement)
        self.assertIn("router_constraint", statement)
        self.assertTrue(statement.endswith("Route the fan-out down one channel."))

    def test_the_figures_labels_are_read_from_the_plan_it_wrote(self):
        self.assertEqual(["Memory", "Multi-hop Personalized Reasoning Tasks", "New Query"], figure_labels(FIGURE))


class RuleBindingGateTests(unittest.TestCase):
    def test_a_rule_that_names_a_case_object_is_refused(self):
        gate = rule_binding_gate(row(hypothesis="reroute rel_12 around the panel"))

        self.assertFalse(gate["ok"])
        self.assertEqual("rule_binds_to_case_objects", gate["reason"])
        self.assertEqual("rel_12", gate["evidence"][0]["value"])

    def test_a_condition_that_names_case_objects_is_refused(self):
        gate = rule_binding_gate(row(condition={"topology": "pipeline", "slot_ids": ["slot_3"]}))

        self.assertFalse(gate["ok"])
        self.assertEqual("rule_binds_to_case_objects", gate["reason"])
        self.assertEqual("slot_ids", gate["evidence"][0]["value"])

    def test_a_rule_that_names_coordinates_is_refused(self):
        gate = rule_binding_gate(row(hypothesis="the box at 0.42, 0.30 should move left"))

        self.assertFalse(gate["ok"])
        self.assertEqual("rule_binds_to_case_objects", gate["reason"])
        self.assertEqual("coordinate", gate["evidence"][0]["kind"])

    def test_the_shipped_statistic_names_are_not_case_objects(self):
        """``node_count_bucket`` is a shape a rule may talk about; ``node_alpha`` is not."""

        self.assertTrue(rule_binding_gate(row())["ok"])

    def test_a_rule_that_names_one_of_the_figures_own_labels_is_refused(self):
        """``New Query`` is a real node of the source figure, and no id pattern sees it."""

        gate = rule_binding_gate(row(guidance="The New Query box is too small; make it taller."), figure=FIGURE)

        self.assertFalse(gate["ok"])
        self.assertEqual("rule_binds_to_case_objects", gate["reason"])
        self.assertEqual("case_figure_label", gate["evidence"][0]["kind"])
        self.assertEqual("New Query", gate["evidence"][0]["value"])

    def test_a_label_the_figure_shares_with_ordinary_prose_is_not_a_binding(self):
        """``Memory`` is also a node label, but it is how a rule talks about the idea."""

        gate = rule_binding_gate(row(guidance="Keep memory beside the agent rather than above it."), figure=FIGURE)

        self.assertTrue(gate["ok"], gate)

    def test_a_rule_with_nothing_that_travels_is_not_a_finding(self):
        gate = rule_binding_gate(row(guidance="", action={"type": "reroute", "constraints": {"forbid_crossing": True}}))

        self.assertFalse(gate["ok"])
        self.assertEqual("no_transferable_rule", gate["reason"])

    def test_a_mechanism_move_travels_without_any_guidance(self):
        gate = rule_binding_gate(row(guidance="", action={"type": "router_constraint", "constraints": {"forbid_crossing": True}}))

        self.assertTrue(gate["ok"], gate)

    def test_a_rule_that_restates_the_default_move_is_common_knowledge(self):
        action = {"type": "reroute", "constraints": {"shared_vertical_channel": True}}

        gate = rule_binding_gate(row(action=action), default_action_fingerprint=action_execution_fingerprint(action))

        self.assertFalse(gate["ok"])
        self.assertEqual("common_knowledge", gate["reason"])

    def test_a_rule_that_differs_from_the_default_move_is_kept(self):
        default = action_execution_fingerprint({"type": "relayout_global", "factor": 1.02, "dx": 0.008})

        self.assertTrue(rule_binding_gate(row(), default_action_fingerprint=default)["ok"])


class StoreGateTests(unittest.TestCase):
    def test_a_bound_row_never_reaches_the_library(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(temp)

            stored = store.append_result(row(hypothesis="reroute rel_12 around the panel"))

            self.assertFalse(stored["ok"])
            self.assertEqual("rule_binds_to_case_objects", stored["reason"])
            self.assertEqual([], store.records())
            self.assertIsNone(store.lifecycle_status("direct-exp-1"))

    def test_a_travelling_row_is_still_appended(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(temp)

            stored = store.append_result(row())

            self.assertTrue(stored["ok"], stored)
            self.assertEqual(1, len(store.records()))

    def test_the_new_states_are_writable_and_never_retrieved(self):
        names = {"common_knowledge", "measurement_noise"}

        self.assertTrue(names <= ALLOWED_APPEND_STATUSES)
        self.assertEqual(set(), names & set(DEFAULT_RETRIEVAL_STATUSES))


if __name__ == "__main__":
    unittest.main()
