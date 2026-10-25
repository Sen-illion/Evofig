"""A losing rule gets its reason checked, and a cheap attempt before it is written off.

The two readings this covers come from the same run: a rule the compiler reported as unmet, and a
rule the compiler carried out whose figure the comparison then faulted. They need different
answers, and the loop may only tell them apart from the compile's own report, so every claim here
is weighed against a record rather than trusted.
"""

from __future__ import annotations

import unittest

from rfs.contracts.evolution import FIGURE_MOVE_EVIDENCE_CONTRACT, normalize_reflection
from rfs.evolution.direct import (
    FOLLOW_UP_ATTEMPTS,
    apply_reflection_policy,
    next_follow_up_turn,
    round_proposal_order,
    verify_attribution,
)
from rfs.workflows.direct_repair_execution import figure_move_evidence

RULE = {"type": "layout_constraint", "params": {"panel_occupancy_min": 0.5}}


class VerifyAttributionTests(unittest.TestCase):
    def test_the_rule_being_wrong_needs_no_evidence(self):
        self.assertEqual(("rule_is_wrong", ""), verify_attribution({"attribution": "rule_is_wrong"}, {}))

    def test_an_unstated_attribution_returns_nothing(self):
        self.assertEqual(("", ""), verify_attribution({}, {"figure_changed": True}))

    def test_a_word_outside_the_contract_is_refused(self):
        attribution, reason = verify_attribution({"attribution": "under_delivered"}, {"figure_changed": True})
        self.assertEqual("", attribution)
        self.assertIn("is not one of", reason)

    def test_no_report_means_the_claim_cannot_be_checked(self):
        attribution, reason = verify_attribution({"attribution": "not_carried_out"}, {})
        self.assertEqual("", attribution)
        self.assertIn("no implementation report", reason)

    def test_not_carried_out_survives_an_unmet_target(self):
        report = {"compile_applied": [], "compile_unmet": [{"rule": "panel_occupancy_min"}], "figure_changed": True}
        self.assertEqual(("not_carried_out", ""), verify_attribution({"attribution": "not_carried_out"}, report))

    def test_not_carried_out_survives_a_figure_that_did_not_change(self):
        report = {"compile_applied": [{"rule": "max_segments"}], "compile_unmet": [], "figure_changed": False}
        self.assertEqual(("not_carried_out", ""), verify_attribution({"attribution": "not_carried_out"}, report))

    def test_not_carried_out_is_refused_when_the_rule_was_carried_out(self):
        report = {"compile_applied": [{"rule": "max_segments"}], "compile_unmet": [], "figure_changed": True}
        attribution, reason = verify_attribution({"attribution": "not_carried_out"}, report)
        self.assertEqual("", attribution)
        self.assertIn("was carried out", reason)

    def test_local_side_effect_needs_a_change_and_a_named_defect(self):
        report = {"figure_changed": True}
        self.assertEqual(
            ("local_side_effect", ""),
            verify_attribution({"attribution": "local_side_effect"}, report, {"diagnosis": [{"type": "overlap"}]}),
        )
        attribution, reason = verify_attribution({"attribution": "local_side_effect"}, report, {"diagnosis": []})
        self.assertEqual("", attribution)
        self.assertIn("named no defect", reason)

    def test_local_side_effect_is_refused_when_the_figure_never_moved(self):
        attribution, reason = verify_attribution(
            {"attribution": "local_side_effect"},
            {"figure_changed": False},
            {"diagnosis": [{"type": "overlap"}]},
        )
        self.assertEqual("", attribution)
        self.assertIn("did not change", reason)

    def test_unmeasurable_needs_a_comparison_that_did_not_call_it_clear(self):
        self.assertEqual(("unmeasurable", ""), verify_attribution({"attribution": "unmeasurable"}, {"figure_changed": True}, {"margin": "slight"}))
        attribution, _ = verify_attribution({"attribution": "unmeasurable"}, {"figure_changed": True}, {"margin": "clear"})
        self.assertEqual("", attribution)


class FigureMoveAttributionTests(unittest.TestCase):
    """A move on the figure answers the same two claims a compile report answers.

    It used to answer neither: the only record an attribution was read against was the compile's,
    so both claims that earn another attempt were refused for every round that ran no rule.
    """

    def _evidence(self, **figure_diff):
        return figure_move_evidence(
            round_index=3,
            action={"type": "reroute"},
            figure_diff={"changed": True, "changed_object_ids": ["edge_01"], **figure_diff},
        )

    def test_the_evidence_says_what_the_move_did_not_what_a_compile_did(self):
        evidence = self._evidence()

        self.assertEqual(FIGURE_MOVE_EVIDENCE_CONTRACT, evidence["contract"])
        self.assertTrue(evidence["figure_changed"])
        self.assertEqual(["edge_01"], evidence["changed_object_ids"])
        self.assertEqual([], evidence["compile_unmet"])
        self.assertEqual("figure_program_diff", evidence["source"])

    def test_a_move_that_never_reached_the_figure_is_accepted_as_not_carried_out(self):
        self.assertEqual(("not_carried_out", ""), verify_attribution({"attribution": "not_carried_out"}, self._evidence(changed=False)))

    def test_a_move_that_did_land_is_refused_for_saying_it_never_did(self):
        attribution, reason = verify_attribution({"attribution": "not_carried_out"}, self._evidence())

        self.assertEqual("", attribution)
        self.assertIn("carried out", reason)

    def test_a_named_defect_earns_the_second_attempt(self):
        self.assertEqual(
            ("local_side_effect", ""),
            verify_attribution({"attribution": "local_side_effect"}, self._evidence(), {"diagnosis": [{"type": "overlap"}]}),
        )

    def test_a_comparison_that_names_no_defect_still_refuses(self):
        attribution, reason = verify_attribution({"attribution": "local_side_effect"}, self._evidence(), {"diagnosis": []})

        self.assertEqual("", attribution)
        self.assertIn("named no defect", reason)

    def test_a_move_that_never_landed_cannot_have_broken_anything(self):
        attribution, reason = verify_attribution(
            {"attribution": "local_side_effect"},
            self._evidence(changed=False),
            {"diagnosis": [{"type": "overlap"}]},
        )

        self.assertEqual("", attribution)
        self.assertIn("did not change", reason)

    def test_a_round_that_recorded_nothing_is_still_refused(self):
        # The relaxation is about which record is read, never about reading none: a round with no
        # report at all still cannot have its claim checked, mechanism round or not.
        attribution, reason = verify_attribution({"attribution": "not_carried_out"}, {})

        self.assertEqual("", attribution)
        self.assertIn("no implementation report", reason)


class FollowUpTurnTests(unittest.TestCase):
    def test_a_rule_gets_three_cheap_attempts_and_then_none(self):
        tried: dict[str, int] = {}
        for _ in range(FOLLOW_UP_ATTEMPTS):
            offered, fingerprint = next_follow_up_turn({"type": "reroute", "scope_type": "global"}, rule_fingerprint="rule-1", tried=tried)
            self.assertTrue(offered)
            self.assertTrue(fingerprint)
            tried["rule-1"] = tried.get("rule-1", 0) + 1
        self.assertEqual(({}, ""), next_follow_up_turn({"type": "reroute", "scope_type": "global"}, rule_fingerprint="rule-1", tried=tried))

    def test_the_budget_belongs_to_the_rule_not_to_the_wording(self):
        tried = {"rule-1": FOLLOW_UP_ATTEMPTS}
        self.assertEqual(({}, ""), next_follow_up_turn({"type": "relayout_global"}, rule_fingerprint="rule-1", tried=tried))

    def test_a_repair_that_re_runs_a_stage_is_never_offered(self):
        self.assertEqual(({}, ""), next_follow_up_turn({"type": "replan_structure", "scope_type": "global"}, rule_fingerprint="rule-1", tried={}))

    def test_nothing_lost_means_nothing_to_repair(self):
        self.assertEqual(({}, ""), next_follow_up_turn({"type": "reroute", "scope_type": "global"}, rule_fingerprint="", tried={}))


class ReflectionFollowUpPolicyTests(unittest.TestCase):
    def test_a_supported_claim_keeps_the_rule_and_schedules_the_attempt(self):
        reflection = normalize_reflection(
            {
                "outcome": "regressed",
                "attribution": "not_carried_out",
                "next_policy": "follow_up",
                "refined_rule": {"type": "layout_constraint", "params": {"panel_occupancy_min": 0.45}},
            }
        )
        report = {"compile_applied": [], "compile_unmet": [{"rule": "panel_occupancy_min"}], "figure_changed": True}
        policy = apply_reflection_policy({"accepted": False, "reason": "pairwise_not_preferred"}, RULE, reflection, implementation=report)
        self.assertEqual("not_carried_out", policy["attribution"])
        self.assertEqual("", policy["attribution_rejection"])
        self.assertEqual("layout_constraint", policy["follow_up_action"]["type"])
        self.assertEqual({"panel_occupancy_min": 0.45}, policy["follow_up_action"]["params"])
        self.assertEqual("reflection_refined_rule", policy["follow_up_action"]["source"])
        self.assertTrue(policy["revised_rule"])

    def test_a_local_side_effect_schedules_the_named_repair(self):
        reflection = normalize_reflection(
            {
                "outcome": "regressed",
                "attribution": "local_side_effect",
                "next_policy": "follow_up",
                "follow_up": {"type": "reroute", "scope_type": "global", "object_ids": ["rel_memory_agent"]},
            }
        )
        policy = apply_reflection_policy(
            {"accepted": False},
            RULE,
            reflection,
            implementation={"figure_changed": True},
            pairwise={"diagnosis": [{"figure": "B", "type": "overlap", "evidence": "two arrows cross"}]},
        )
        self.assertEqual("reroute", policy["follow_up_action"]["type"])
        self.assertEqual("reflection_follow_up", policy["follow_up_action"]["source"])
        self.assertEqual(["rel_memory_agent"], policy["follow_up_action"]["object_ids"])

    def test_an_unsupported_claim_is_recorded_and_buys_nothing(self):
        reflection = normalize_reflection(
            {"outcome": "regressed", "attribution": "not_carried_out", "next_policy": "follow_up", "follow_up": {"type": "reroute", "scope_type": "global"}}
        )
        policy = apply_reflection_policy({"accepted": False}, RULE, reflection, implementation={"compile_unmet": [], "figure_changed": True})
        self.assertEqual({}, policy["follow_up_action"])
        events = [event["event"] for event in policy["events"]]
        self.assertIn("attribution_unsupported", events)
        self.assertIn("follow_up_rejected", events)

    def test_a_follow_up_that_re_runs_a_stage_is_refused_with_its_reason(self):
        reflection = normalize_reflection(
            {"outcome": "regressed", "attribution": "local_side_effect", "next_policy": "follow_up", "follow_up": {"type": "replan_structure", "scope_type": "global"}}
        )
        policy = apply_reflection_policy(
            {"accepted": False},
            RULE,
            reflection,
            implementation={"figure_changed": True},
            pairwise={"diagnosis": [{"type": "overlap"}]},
        )
        self.assertEqual({}, policy["follow_up_action"])
        rejected = [event for event in policy["events"] if event["event"] == "follow_up_rejected"]
        self.assertEqual(1, len(rejected))
        self.assertIn("minutes to hours", rejected[0]["reason"])

    def test_a_follow_up_rule_without_parameters_is_refused(self):
        reflection = normalize_reflection(
            {"outcome": "regressed", "attribution": "local_side_effect", "next_policy": "follow_up", "follow_up": {"type": "router_constraint"}}
        )
        policy = apply_reflection_policy(
            {"accepted": False},
            RULE,
            reflection,
            implementation={"figure_changed": True},
            pairwise={"diagnosis": [{"type": "overlap"}]},
        )
        self.assertEqual({}, policy["follow_up_action"])
        rejected = [event for event in policy["events"] if event["event"] == "follow_up_rejected"]
        self.assertIn("none of its parameters", rejected[0]["reason"])

    def test_the_repair_is_offered_ahead_of_everything_already_waiting(self):
        order = round_proposal_order(
            follow_up_action={"type": "reroute", "scope_type": "global"},
            revised_action={"type": "resize_nodes", "scope_type": "global", "factor": 1.1},
        )
        self.assertEqual("reroute", order[0]["type"])


class ReflectionFieldTests(unittest.TestCase):
    def test_a_row_written_before_the_three_fields_reads_as_an_empty_trio(self):
        reflection = normalize_reflection({"outcome": "regressed", "failure_type": "global_regression"})
        self.assertEqual("", reflection["attribution"])
        self.assertEqual({}, reflection["refined_rule"])
        self.assertEqual({}, reflection["follow_up"])


if __name__ == "__main__":
    unittest.main()
