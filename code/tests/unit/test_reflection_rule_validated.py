"""A reflection's rule is validated where it is written, not silently dropped later."""

from __future__ import annotations

import unittest

from rfs.evolution.direct import validate_reflection_rule
from rfs.evolution.direct.controller import apply_reflection_policy

FIGURE = {
    "panels": [{"id": "panel_a", "label": "Processing"}],
    "nodes": [{"id": "node_alpha", "label": "New Query"}, {"id": "node_beta", "label": "Retriever"}],
    "connectors": [{"id": "rel_12"}],
}

GOOD_RULE = {"type": "layout_constraint", "params": {"align_grid": 0.05, "gutter_min": 0.02}}
LABELLED_RULE = {
    "type": "router_constraint",
    "constraints": {"forbid_crossing": True},
    "guidance": "the New Query box must not cross the panel border",
}
BAD_RULE = {"type": "resize_nodes", "params": {"factor": 1.1}}


class ReflectionRuleValidationTests(unittest.TestCase):
    def test_a_rule_that_travels_is_normalized_and_kept(self):
        reflection, rejection = validate_reflection_rule({"rule": dict(GOOD_RULE), "next_policy": "revise_rule"}, figure=FIGURE)
        self.assertEqual({}, rejection)
        self.assertEqual("layout_constraint", reflection["rule"]["type"])
        self.assertEqual("ppt_compilation", reflection["rule"]["apply_at"])
        self.assertIn("align_grid", reflection["rule"]["params"])

    def test_the_prose_the_model_wrote_survives_the_normalization(self):
        rule = {
            "type": "router_constraint",
            "constraints": {"forbid_crossing": True},
            "guidance": "share a vertical channel when two panels exchange more than one arrow",
        }
        reflection, rejection = validate_reflection_rule({"rule": rule}, figure=FIGURE)
        self.assertEqual({}, rejection)
        self.assertEqual("share a vertical channel when two panels exchange more than one arrow", reflection["rule"]["guidance"])
        self.assertEqual("router_constraint", reflection["rule"]["type"])

    def test_a_rule_that_names_the_figures_own_label_is_refused_and_explained(self):
        reflection, rejection = validate_reflection_rule({"rule": dict(LABELLED_RULE)}, figure=FIGURE)
        self.assertEqual({}, reflection["rule"])
        self.assertEqual("rule_binds_to_case_objects", rejection["reason"])
        self.assertTrue(rejection["evidence"])

    def test_a_rule_that_carries_an_object_id_is_refused_and_explained(self):
        rule = {"type": "router_constraint", "constraints": {"forbid_crossing": True}, "guidance": "restate rel_12 so it no longer crosses"}
        reflection, rejection = validate_reflection_rule({"rule": rule}, figure=FIGURE)
        self.assertEqual({}, reflection["rule"])
        self.assertEqual("rule_binds_to_case_objects", rejection["reason"])

    def test_a_rule_the_compiler_cannot_run_is_refused_and_explained(self):
        reflection, rejection = validate_reflection_rule({"rule": dict(BAD_RULE)}, figure=FIGURE)
        self.assertEqual({}, reflection["rule"])
        self.assertEqual("not_a_mechanism_rule", rejection["reason"])

    def test_a_reflection_with_no_rule_reports_no_refusal(self):
        reflection, rejection = validate_reflection_rule({"next_policy": "keep"}, figure=FIGURE)
        self.assertEqual({}, rejection)
        self.assertEqual({}, reflection["rule"])

    def test_the_refused_rule_never_reaches_the_policy_that_promotes_it(self):
        reflection, _ = validate_reflection_rule({"rule": dict(LABELLED_RULE), "next_policy": "revise_rule"}, figure=FIGURE)
        policy = apply_reflection_policy({"accepted": True}, {"type": "reroute"}, reflection)
        self.assertEqual({}, policy["revised_rule"])
        promoted = [event for event in policy["events"] if event.get("policy") == "revise_rule" and event.get("rule")]
        self.assertEqual([], promoted)


if __name__ == "__main__":
    unittest.main()
