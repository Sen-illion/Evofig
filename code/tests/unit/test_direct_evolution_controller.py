"""Tests for direct evolution state decisions and bounded repairs."""

import unittest

from rfs.evolution.direct.controller import ACTION_BUDGET_KEYS, action_execution_fingerprint, action_is_executable, apply_bounded_action, apply_reflection_policy, color_feedback_proposal, compare_candidate, default_reflection, derived_score, measured_score, select_bounded_action, select_next_round_proposal


class DirectEvolutionControllerTests(unittest.TestCase):
    def test_slot_feedback_selects_targeted_asset_regeneration(self):
        action = select_bounded_action({"stage_feedback": [{"scope_type": "slot", "slot_ids": ["icon_n"], "problem": "wrong style", "requested_actions": [{"action": "regen_assets"}]}]})
        self.assertEqual(action["type"], "regen_assets")
        self.assertEqual(action["slot_ids"], ["icon_n"])

    def test_the_advisory_applicability_travels_with_the_repair_it_proposed(self):
        action = select_bounded_action({
            "applicability": "domain",
            "stage_feedback": [{"scope_type": "slot", "slot_ids": ["icon_n"], "problem": "wrong style", "requested_actions": [{"action": "regen_assets"}]}],
        })
        self.assertEqual("domain", action["applicability"])
        unlabeled = select_bounded_action({
            "applicability": "everywhere",
            "stage_feedback": [{"scope_type": "slot", "slot_ids": ["icon_n"], "problem": "wrong style", "requested_actions": [{"action": "regen_assets"}]}],
        })
        self.assertNotIn("applicability", unlabeled)

    def test_empty_feedback_is_not_a_global_nudge(self):
        self.assertIsNone(select_bounded_action({"stage_feedback": []}))
        self.assertIsNone(select_bounded_action({}))
        self.assertFalse(action_is_executable(select_bounded_action({})))

    def test_a_named_budget_travels_with_the_selected_action(self):
        action = select_bounded_action({
            "stage_feedback": [{
                "scope_type": "global",
                "problem": "sparse",
                "requested_actions": [{"action": "relayout_global", "factor": 1.05, "dx": 0.02, "dy": 0.0}],
            }]
        })
        self.assertEqual(action["type"], "relayout_global")
        self.assertEqual(action["factor"], 1.05)
        self.assertEqual(action["dx"], 0.02)
        self.assertEqual(action["dy"], 0.0)

    def test_bounded_icon_repair_changes_only_target(self):
        icons = {"icons": [{"asset_id": "a", "semantic_node_ids": ["a"], "prompt": "A"}, {"asset_id": "b", "semantic_node_ids": ["b"], "prompt": "B"}]}
        _, updated, report = apply_bounded_action({"nodes": []}, icons, {"type": "regen_assets", "slot_ids": ["a"], "problem": "style"}, domain_profile={"icon_style": "biomedical"})
        self.assertNotEqual(updated["icons"][0]["prompt"], "A")
        self.assertEqual(updated["icons"][1]["prompt"], "B")
        self.assertEqual(report["changed_asset_ids"], ["a"])

    def test_a_global_relayout_records_the_unsigned_budget_it_actually_used(self):
        layout = {"nodes": [
            {"id": "n1", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}},
            {"id": "n2", "bbox_percent": {"x": 0.5, "y": 0.1, "w": 0.2, "h": 0.2}},
        ]}
        _, _, report = apply_bounded_action(layout, {}, {"type": "relayout_global", "factor": 1.02, "dx": 0.016, "dy": 0.0})
        self.assertEqual({"factor": 1.02, "dx": 0.016, "dy": 0.0}, report["applied_budget"])
        self.assertEqual(0.016, report["action"]["dx"])
        default, _, _ = apply_bounded_action(layout, {}, {"type": "relayout_global"})
        replayed, _, _ = apply_bounded_action(layout, {}, {"type": "relayout_global", **report["applied_budget"]})
        self.assertNotEqual(default["nodes"], replayed["nodes"])
        self.assertEqual(ACTION_BUDGET_KEYS, ("factor", "dx", "dy"))

    def test_retry_index_does_not_change_the_budget(self):
        layout = {"nodes": [{"id": "n", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]}
        first, _, first_report = apply_bounded_action(layout, {}, {"type": "relayout_global"}, retry_index=0)
        second, _, second_report = apply_bounded_action(layout, {}, {"type": "relayout_global"}, retry_index=3)
        self.assertEqual(first_report["applied_budget"], second_report["applied_budget"])
        self.assertEqual(first["nodes"], second["nodes"])

    def test_an_escalation_does_not_nudge_boxes_in_this_pass(self):
        layout = {"nodes": [{"id": "n", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}}]}
        for name in ("replan_structure", "resegment"):
            updated, _, report = apply_bounded_action(layout, {}, {"type": name, "factor": 1.08, "dx": 0.04, "dy": 0.0})
            self.assertEqual(layout["nodes"], updated["nodes"])
            self.assertEqual([], report["changed_node_ids"])


    def test_a_geometric_name_without_a_budget_is_not_selected(self):
        self.assertIsNone(select_bounded_action({
            "stage_feedback": [{"scope_type": "global", "problem": "sparse", "requested_actions": [{"action": "relayout_global"}]}],
        }))
        self.assertFalse(action_is_executable({"type": "relayout_global"}))
        self.assertTrue(action_is_executable({"type": "relayout_global", "factor": 1.02, "dx": 0.008, "dy": 0.0}))

    def test_two_relayouts_that_differ_only_in_step_size_are_different_moves(self):
        self.assertNotEqual(
            action_execution_fingerprint({"type": "relayout_global", "dx": 0.008}),
            action_execution_fingerprint({"type": "relayout_global", "dx": 0.016}),
        )
        self.assertEqual("relayout_global|dx=0.016", action_execution_fingerprint({"type": "relayout_global", "dx": 0.016}))

    def test_a_round_that_moved_nothing_is_refused_however_it_scored(self):
        # A reroute whose router re-derives the same paths, and an escalation the workflow
        # declined, both leave the figure as it was. The measurement cannot settle such a
        # round: the same figure scored twice returns two numbers, and the higher one used to
        # be banked as a repair.
        unchanged = {"available": True, "changed": False, "changed_object_ids": []}
        nothing_touched = {"applied": False, "changed_node_ids": [], "changed_asset_ids": []}
        by_score = compare_candidate(
            {"overall_score": 0.5, "blocking_issue_count": 0},
            {"overall_score": 0.9, "blocking_issue_count": 0},
            action={"type": "reroute", "edge_ids": ["e1"]},
            bounded_change=nothing_touched,
            figure_diff=unchanged,
        )
        by_pairwise = compare_candidate(
            {"overall_score": 0.5, "blocking_issue_count": 0},
            {"overall_score": 0.5, "blocking_issue_count": 0},
            action={"type": "reanalyze_semantics"},
            bounded_change=nothing_touched,
            figure_diff=unchanged,
            after_pairwise={"judgement_kind": "pairwise", "pairwise": {"choice": "B", "margin": "clear", "confidence": 1.0}, "preferred_candidate": "B"},
        )
        for decision in (by_score, by_pairwise):
            self.assertFalse(decision["accepted"])
            self.assertEqual("figure_unchanged", decision["reason"])
            self.assertEqual("figure_unchanged", decision["comparison"])
            self.assertIsNone(decision["score_delta"])

    def test_a_figure_diff_that_was_never_taken_does_not_veto_a_round(self):
        # Older runs and synthesised rounds carry no figure diff. Reading a missing diff as
        # "nothing moved" would refuse every one of them.
        decision = compare_candidate(
            {"overall_score": 0.5, "blocking_issue_count": 0},
            {"overall_score": 0.9, "blocking_issue_count": 0},
            action={"type": "relayout_global"},
            bounded_change={"applied": True, "changed_node_ids": [], "changed_asset_ids": []},
            figure_diff={"available": False, "changed": None, "changed_object_ids": []},
        )
        self.assertTrue(decision["accepted"])
        self.assertEqual("target_improved", decision["reason"])

    def test_regenerating_an_asset_survives_a_figure_diff_that_only_compares_geometry(self):
        # The figure diff compares panels, nodes and arrows, none of which carry the icon a
        # regenerated asset produced, so a real asset repair must not be read as a no-op.
        decision = compare_candidate(
            {"overall_score": 0.5, "blocking_issue_count": 0},
            {"overall_score": 0.9, "blocking_issue_count": 0},
            action={"type": "regen_assets", "slot_ids": ["slot_a"]},
            bounded_change={"applied": True, "changed_node_ids": [], "changed_asset_ids": ["slot_a"]},
            figure_diff={"available": True, "changed": False, "changed_object_ids": []},
        )
        self.assertTrue(decision["accepted"])
        self.assertEqual("target_improved", decision["reason"])

    def test_new_blockers_is_a_count_on_both_decision_branches(self):
        initial = compare_candidate(None, {"overall_score": 0.5, "blocking_issue_count": 0})
        later = compare_candidate({"overall_score": 0.5, "blocking_issue_count": 0}, {"overall_score": 0.4, "blocking_issue_count": 3})
        self.assertIsInstance(initial["new_blockers"], int)
        self.assertIsInstance(later["new_blockers"], int)
        self.assertEqual(later["new_blockers"], 3)

    def _preferred(self, before: dict, after: dict) -> dict:
        """One round a clear comparison preferred, so only a regression can refuse it."""

        return compare_candidate(
            before,
            after,
            action={"type": "regen_assets", "slot_ids": ["slot_a"]},
            bounded_change={"applied": True, "changed_node_ids": [], "changed_asset_ids": ["slot_a"]},
            figure_diff={"available": True, "changed": True, "changed_object_ids": ["slot_a"]},
            after_pairwise={"judgement_kind": "pairwise", "pairwise": {"choice": "B", "margin": "clear", "confidence": 0.9}, "preferred_candidate": "B"},
        )

    def test_one_more_sentence_from_the_model_is_not_a_regression(self):
        # The model writes its blocker list fresh on every call. Asked about round 0 it named
        # the checkered icon background; asked about round 1 it named that and the two icons
        # sharing a metaphor, which was true of both figures. Counting the sentences refused a
        # round the comparison had preferred with a clear margin.
        decision = self._preferred(
            {"overall_score": 0.7, "blocking_issue_count": 1, "deterministic_blockers": []},
            {"overall_score": 0.0, "blocking_issue_count": 2, "deterministic_blockers": []},
        )

        self.assertTrue(decision["accepted"])
        self.assertEqual("pairwise_preferred", decision["reason"])
        self.assertEqual(0, decision["new_blockers"])
        self.assertEqual([], decision["new_blocking_defects"])

    def test_a_defect_the_gate_did_not_see_before_is_recorded_not_a_refusal(self):
        # The mechanism no longer vetoes. A defect only the gate can name is written beside the
        # round as evidence -- work for the next round -- while the comparison's preference still
        # decides whether the round is kept.
        decision = self._preferred(
            {"overall_score": 0.7, "blocking_issue_count": 1, "deterministic_blockers": ["interface:relayout_slots"]},
            {"overall_score": 0.8, "blocking_issue_count": 1, "deterministic_blockers": ["interface:relayout_slots", "compile:replan_structure"]},
        )

        self.assertTrue(decision["accepted"])
        self.assertEqual("pairwise_preferred", decision["reason"])
        self.assertEqual(["compile:replan_structure"], decision["new_blocking_defects"])
        self.assertEqual(["compile:replan_structure"], decision["evidence"]["new_blocking_defects"])

    def test_the_same_gate_defect_measured_differently_is_not_a_new_one(self):
        # The gate writes its counts into the sentence, so comparing the prose would read
        # "overlap=3" and "overlap=2" as two defects and refuse a round that halved one.
        decision = self._preferred(
            {"overall_score": 0.4, "blocking_issue_count": 2, "deterministic_blockers": ["interface:relayout_slots"]},
            {"overall_score": 0.5, "blocking_issue_count": 3, "deterministic_blockers": ["interface:relayout_slots"]},
        )

        self.assertTrue(decision["accepted"])
        self.assertEqual(0, decision["new_blockers"])

    def _moved(self, before: dict, after: dict, *, choice: str, before_gate: dict, after_gate: dict) -> dict:
        return compare_candidate(
            before,
            after,
            action={"type": "relayout_slots", "semantic_node_ids": ["n1"], "factor": 1.08, "dx": 0.0, "dy": 0.0},
            bounded_change={"applied": True, "changed_node_ids": ["n1"], "changed_asset_ids": []},
            figure_diff={"available": True, "changed": True, "changed_object_ids": ["n1"]},
            after_pairwise={"judgement_kind": "pairwise", "pairwise": {"choice": choice, "margin": "clear", "confidence": 0.9}, "preferred_candidate": choice},
            before_grounding=before_gate,
            after_grounding=after_gate,
        )

    def test_a_coverage_gain_is_recorded_beside_a_round_the_comparison_refused(self):
        # The comparison is the only ruler. A reading that moved the other way is written onto the
        # round and never spends it, so a round the seat refused reads as refused and the number
        # that disagreed is still there to be read.
        decision = self._moved(
            {"overall_score": 0.4, "blocking_issue_count": 1, "deterministic_blockers": ["grounding_gate"]},
            {"overall_score": 0.3, "blocking_issue_count": 1, "deterministic_blockers": ["grounding_gate"]},
            choice="A",
            before_gate={"node_coverage": 0.92, "edge_coverage": 0.18},
            after_gate={"node_coverage": 0.92, "edge_coverage": 1.0},
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual("pairwise_not_preferred", decision["reason"])
        self.assertEqual("before", decision["pairwise_verdict"])
        self.assertTrue(decision["coverage_improved"])
        self.assertTrue(decision["coverage_disagreed"])

    def test_flat_coverage_does_not_overrule_a_comparison_that_prefers_the_old_figure(self):
        decision = self._moved(
            {"overall_score": 0.4, "blocking_issue_count": 1, "deterministic_blockers": ["grounding_gate"]},
            {"overall_score": 0.3, "blocking_issue_count": 1, "deterministic_blockers": ["grounding_gate"]},
            choice="A",
            before_gate={"node_coverage": 0.92, "edge_coverage": 0.18},
            after_gate={"node_coverage": 0.92, "edge_coverage": 0.18},
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual("pairwise_not_preferred", decision["reason"])

    def test_a_new_deterministic_blocker_is_recorded_beside_an_accepted_round(self):
        # The comparison preferred the new figure, so the round is kept; the defect the gate found
        # is kept as the name of what to work on next rather than as a reason to drop the round.
        decision = self._moved(
            {"overall_score": 0.4, "blocking_issue_count": 1, "deterministic_blockers": ["grounding_gate"]},
            {"overall_score": 0.3, "blocking_issue_count": 2, "deterministic_blockers": ["grounding_gate", "interface:relayout_slots"]},
            choice="B",
            before_gate={"node_coverage": 0.92, "edge_coverage": 0.18},
            after_gate={"node_coverage": 0.92, "edge_coverage": 1.0},
        )
        self.assertTrue(decision["accepted"])
        self.assertEqual("pairwise_preferred", decision["reason"])
        self.assertFalse(decision["coverage_disagreed"])
        self.assertEqual(["interface:relayout_slots"], decision["new_blocking_defects"])
        # A missing field must not read as "nothing got worse": a stored round, or one side of a
        # pair that predates the split, is still measured by the tally it does carry, and that
        # tally is evidence on the round rather than a refusal.
        decision = self._preferred(
            {"overall_score": 0.7, "blocking_issue_count": 0},
            {"overall_score": 0.8, "blocking_issue_count": 2, "deterministic_blockers": []},
        )

        self.assertTrue(decision["accepted"])
        self.assertEqual(2, decision["new_blockers"])
        # Neither answer separated the defects, so the tally is all this round reports and the
        # evidence keeps the named count -- which is none -- beside it rather than guessing.
        self.assertEqual(0, decision["evidence"]["new_blocking_defect_count"])
        self.assertEqual(2, decision["evidence"]["after_blockers"])

    def test_the_first_candidate_reports_no_comparison_instead_of_a_gain(self):
        # A first draft has nothing to be an improvement over. Reporting its own score as a
        # delta made every consumer of the record -- the training export included -- read a
        # starting point as "improved by 0.75".
        initial = compare_candidate(None, {"overall_score": 0.75, "blocking_issue_count": 0})
        compared = compare_candidate({"overall_score": 0.5, "blocking_issue_count": 0}, {"overall_score": 0.75, "blocking_issue_count": 0})
        self.assertTrue(initial["accepted"])
        self.assertEqual("initial_candidate", initial["reason"])
        self.assertIsNone(initial["score_delta"])
        self.assertEqual("none", initial["comparison"])
        self.assertEqual(0.25, compared["score_delta"])
        self.assertEqual("compared", compared["comparison"])

    def test_regression_rolls_back(self):
        decision = compare_candidate({"overall_score": 0.8, "blocking_issue_count": 0}, {"overall_score": 0.7, "blocking_issue_count": 0})
        self.assertFalse(decision["accepted"])
        self.assertEqual(default_reflection(decision, {"type": "relayout_global"}, retry_index=2)["next_policy"], "disable")

    def test_an_unmeasured_candidate_cannot_improve_and_an_unmeasured_baseline_cannot_be_beat(self):
        unmeasured = {"overall_score": None, "score_measured": False, "blocking_issue_count": 0}
        scored = {"overall_score": 0.8, "score_measured": True, "blocking_issue_count": 0}

        failed = compare_candidate(scored, unmeasured)
        self.assertFalse(failed["accepted"])
        self.assertEqual("score_measurement_failed", failed["reason"])
        self.assertEqual("measurement_failed", failed["comparison"])
        self.assertIsNone(failed["score_delta"])

        # A caller that knows the earlier state was never measured says so, and the round is
        # refused for the missing baseline rather than read against a placeholder zero.
        no_baseline = compare_candidate(scored, {"overall_score": 0.9, "blocking_issue_count": 0}, baseline_ready=False)
        self.assertFalse(no_baseline["accepted"])
        self.assertEqual("baseline_unavailable", no_baseline["reason"])
        self.assertEqual("baseline_unavailable", no_baseline["comparison"])
        self.assertIsNone(no_baseline["score_delta"])
        self.assertEqual(0.9, no_baseline["after_score"])

    def test_a_missing_score_is_read_as_missing_and_never_as_zero(self):
        self.assertIsNone(measured_score({"overall_score": None, "score_measured": False}))
        self.assertIsNone(measured_score({"overall_score": None}))
        self.assertIsNone(measured_score({}))
        self.assertEqual(0.0, measured_score({"overall_score": 0.0, "score_measured": True}))

    def test_a_defect_table_score_is_never_read_as_a_measurement(self):
        """The choice comparison's number is derived, so the gate's own reading is used."""

        derived = {"overall_score": 1.0, "score_source": "derived_from_defect", "score_measured": False, "absolute_score": 0.62}
        self.assertEqual(0.62, measured_score(derived))
        self.assertEqual(1.0, derived_score(derived))
        # With no absolute reading to fall back on, the state measures nothing at all rather
        # than borrowing the number that came out of the defect table.
        self.assertIsNone(measured_score({"overall_score": 1.0, "score_source": "derived_from_defect"}))
        # And a measured score is not reported as a derived one.
        self.assertIsNone(derived_score({"overall_score": 0.8, "score_measured": True}))



class ReflectionPolicyTests(unittest.TestCase):
    """The reflection is a claim: acting on it may tighten a decision and never widen one.

    It also may not take away a decision the comparison made. A doubt about the measurement is
    recorded on the round and settled later; withdrawing an accept on the model's behalf -- which
    is what the fallback reflection's noise flags did on every round that did not improve -- is
    what the mechanism no longer does.
    """

    def _accepted(self) -> dict:
        return compare_candidate({"overall_score": 0.4, "blocking_issue_count": 0}, {"overall_score": 0.9, "blocking_issue_count": 0})

    def test_a_randomness_flag_is_recorded_and_does_not_withdraw_an_accept(self):
        result = apply_reflection_policy(self._accepted(), {"type": "reroute"}, {"randomness_risk": True, "next_policy": "keep"})
        self.assertTrue(result["decision"]["accepted"])
        self.assertEqual("target_improved", result["decision"]["reason"])
        self.assertTrue(result["decision"]["noise_doubt"])
        self.assertTrue(result["policy"]["measurement_noise_doubt"])
        self.assertEqual(["measurement_noise_doubt_recorded"], [event["event"] for event in result["events"]])
        self.assertNotIn("randomness_risk", result["policy"]["applied"])

    def test_a_randomness_flag_never_turns_a_rollback_into_an_accept(self):
        rolled_back = compare_candidate({"overall_score": 0.9, "blocking_issue_count": 0}, {"overall_score": 0.5, "blocking_issue_count": 0})
        result = apply_reflection_policy(rolled_back, {"type": "reroute"}, {"randomness_risk": True, "next_policy": "keep"})
        self.assertFalse(result["decision"]["accepted"])
        self.assertEqual("no_significant_improvement", result["decision"]["reason"])
        self.assertNotIn("randomness_risk", result["policy"]["applied"])

    def _accepted_on_a_clear_comparison(self) -> dict:
        pairwise = {"judgement_kind": "pairwise", "pairwise": {"choice": "B", "margin": "clear", "confidence": 1.0}, "preferred_candidate": "B"}
        return compare_candidate(
            {"overall_score": 0.4, "blocking_issue_count": 2},
            {"overall_score": 0.9, "blocking_issue_count": 0},
            after_pairwise=pairwise,
            figure_diff={"available": True, "changed": True, "changed_object_ids": ["node_a"]},
        )

    def test_a_measurement_noise_claim_is_recorded_and_does_not_withdraw_a_clear_comparison(self):
        reflection = {"measurement_noise_risk": True, "next_policy": "keep"}
        result = apply_reflection_policy(self._accepted_on_a_clear_comparison(), {"type": "reroute"}, reflection)
        self.assertTrue(result["decision"]["accepted"])
        self.assertEqual("pairwise_preferred", result["decision"]["reason"])
        self.assertTrue(result["policy"]["measurement_noise_doubt"])
        self.assertEqual(["measurement_noise_doubt_recorded"], [event["event"] for event in result["events"]])

    def test_a_bare_randomness_flag_is_recorded_on_a_clear_comparison(self):
        # This reflection answers "was the advisory request effective", so the model used the
        # older flag for advice it distrusted, not for the verdict. On a round the acceptance
        # role decided by calling one state clearly better, that claim is recorded, not acted on.
        result = apply_reflection_policy(self._accepted_on_a_clear_comparison(), {"type": "reroute"}, {"randomness_risk": True, "next_policy": "keep"})
        self.assertTrue(result["decision"]["accepted"])
        self.assertEqual("pairwise_preferred", result["decision"]["reason"])
        self.assertTrue(result["policy"]["randomness_flagged"])
        self.assertTrue(result["policy"]["measurement_noise_doubt"])
        self.assertEqual(
            "the round was decided by a paired judgement the acceptance role called clear",
            result["policy"]["measurement_noise_note"],
        )
        self.assertEqual(["measurement_noise_doubt_recorded"], [event["event"] for event in result["events"]])
        self.assertNotIn("randomness_risk", result["policy"]["applied"])

    def test_a_bare_randomness_flag_does_not_withdraw_a_comparison_that_was_not_decisive(self):
        pairwise = {"judgement_kind": "pairwise", "pairwise": {"choice": "B", "margin": "slight", "confidence": 0.4}, "preferred_candidate": "B"}
        decision = compare_candidate(
            {"overall_score": 0.4, "blocking_issue_count": 0},
            {"overall_score": 0.5, "blocking_issue_count": 0},
            after_pairwise=pairwise,
            figure_diff={"available": True, "changed": True, "changed_object_ids": ["node_a"]},
        )
        result = apply_reflection_policy(decision, {"type": "reroute"}, {"randomness_risk": True, "next_policy": "keep"})
        self.assertTrue(result["decision"]["accepted"])
        self.assertEqual("pairwise_preferred", result["decision"]["reason"])
        self.assertEqual(
            "a doubt about the drawing is recorded; the comparison still decides the round",
            result["policy"]["measurement_noise_note"],
        )

    def test_disable_targets_the_experience_this_round_ran_under(self):
        result = apply_reflection_policy(self._accepted(), {"type": "reroute"}, {"next_policy": "disable"}, injected_experience={"experience_id": "direct-exp-1"})
        self.assertEqual([{"kind": "disable", "experience_id": "direct-exp-1", "reason": "reflection_next_policy_disable"}], result["experience_effects"])
        self.assertEqual("disable", result["policy"]["next_policy"])

    def test_narrow_scope_appends_the_condition_the_round_measured(self):
        result = apply_reflection_policy(
            self._accepted(),
            {"type": "reroute"},
            {"next_policy": "narrow_scope"},
            injected_experience={"experience_id": "direct-exp-2"},
            condition={"topology": "pipeline", "node_count_bucket": "medium"},
        )
        effect = result["experience_effects"][0]
        self.assertEqual("exclude_condition", effect["kind"])
        self.assertEqual("direct-exp-2", effect["experience_id"])
        self.assertEqual({"topology": "pipeline", "node_count_bucket": "medium", "reason": "reflection_next_policy_narrow_scope"}, effect["condition"])

    def test_a_policy_without_an_injected_experience_only_records_intent(self):
        result = apply_reflection_policy(self._accepted(), {"type": "reroute"}, {"next_policy": "disable"}, injected_experience={})
        self.assertEqual([], result["experience_effects"])
        self.assertEqual([], result["policy"]["applied"])
        self.assertEqual("disable", result["policy"]["next_policy"])

    def test_a_legal_revision_becomes_the_next_proposal_without_an_experience_id(self):
        result = apply_reflection_policy(
            self._accepted(),
            {"type": "relayout_global"},
            {"next_policy": "revise", "revised_action": {"type": "reroute", "scope_type": "connector", "edge_ids": ["e1"], "experience_id": "direct-exp-3"}},
        )
        self.assertEqual({"type": "reroute", "scope_type": "connector", "edge_ids": ["e1"]}, result["revised_action"])
        self.assertEqual("revise", result["policy"]["next_policy"])

    def test_an_unsupported_revision_is_rejected_and_recorded(self):
        result = apply_reflection_policy(self._accepted(), {"type": "relayout_global"}, {"next_policy": "revise", "revised_action": {"type": "redraw_everything"}})
        self.assertEqual({}, result["revised_action"])
        self.assertEqual(["revised_action_rejected"], [event["event"] for event in result["events"]])
        self.assertEqual("redraw_everything", result["events"][0]["requested_action"])
        self.assertEqual("the revised action is not one the loop may run", result["events"][0]["reason"])
        self.assertNotIn("revise", result["policy"]["applied"])

    def test_a_revision_that_names_nothing_is_recorded_as_such(self):
        result = apply_reflection_policy(self._accepted(), {"type": "relayout_global"}, {"next_policy": "revise", "revised_action": {}})
        self.assertEqual("the reflection named no revised action", result["events"][0]["reason"])

    def test_an_unknown_policy_reads_as_keep(self):
        result = apply_reflection_policy(self._accepted(), {"type": "reroute"}, {"next_policy": "rethink_everything"})
        self.assertEqual("keep", result["policy"]["next_policy"])
        self.assertEqual([], result["events"])

    def test_a_revision_that_names_a_rule_without_its_parameters_is_refused(self):
        # A mechanism-level rule carries parameters where a repair carries object ids, so a bare
        # name is refused for the same reason an unknown name is -- and told apart from it.
        result = apply_reflection_policy(
            self._accepted(),
            {"type": "relayout_global"},
            {"next_policy": "revise", "revised_action": {"type": "layout_constraint"}},
        )
        self.assertEqual({}, result["revised_action"])
        self.assertEqual(["revised_action_rejected"], [event["event"] for event in result["events"]])
        self.assertEqual("layout_constraint", result["events"][0]["requested_action"])
        self.assertIn("parameters", result["events"][0]["reason"])
        self.assertNotIn("revise", result["policy"]["applied"])

    def test_a_mechanism_rule_is_kept_without_touching_this_rounds_move(self):
        result = apply_reflection_policy(
            self._accepted(),
            {"type": "reroute"},
            {
                "next_policy": "revise_rule",
                "rule": {"type": "router_constraint", "constraints": {"forbid_crossing": True, "shared_vertical_channel": True}},
                "binding": {"edge_ids": ["rel_12"]},
            },
            injected_experience={"experience_id": "direct-exp-3"},
        )
        self.assertEqual(
            {"type": "router_constraint", "apply_at": "ppt_compilation", "constraints": {"forbid_crossing": True, "shared_vertical_channel": True}},
            result["revised_rule"],
        )
        self.assertEqual(result["revised_rule"], result["policy"]["revised_rule"])
        # The round's own bounded move is decided elsewhere and stays exactly what it was.
        self.assertEqual({}, result["revised_action"])
        self.assertIn("revise_rule", result["policy"]["applied"])

    def test_a_layout_rule_is_kept_the_same_way(self):
        result = apply_reflection_policy(
            self._accepted(),
            {"type": "relayout_global"},
            {"next_policy": "revise_rule", "rule": {"type": "layout_constraint", "params": {"align_grid": 0.02, "gutter_min": 0.01}}},
        )
        self.assertEqual(
            {"type": "layout_constraint", "apply_at": "ppt_compilation", "params": {"align_grid": 0.02, "gutter_min": 0.01}},
            result["revised_rule"],
        )

    def test_a_rule_that_names_no_parameter_is_refused(self):
        for rule in ({"type": "router_constraint"}, {"type": "layout_constraint", "params": {}}, {"type": "reroute", "edge_ids": ["rel_12"]}):
            result = apply_reflection_policy(self._accepted(), {"type": "reroute"}, {"next_policy": "revise_rule", "rule": rule})
            self.assertEqual({}, result["revised_rule"], rule)
            self.assertEqual(["revised_rule_rejected"], [event["event"] for event in result["events"]], rule)
            self.assertNotIn("revise_rule", result["policy"]["applied"])

    def test_a_rule_cannot_carry_this_figures_objects(self):
        # The rule channel reads only the mechanism vocabulary and its parameters, so ids a
        # reflection wrote beside them never reach the compiler that draws the next figure.
        result = apply_reflection_policy(
            self._accepted(),
            {"type": "reroute"},
            {"next_policy": "revise_rule", "rule": {"type": "router_constraint", "constraints": {"max_segments": 2}, "edge_ids": ["rel_12"]}},
        )
        self.assertEqual({"type": "router_constraint", "apply_at": "ppt_compilation", "constraints": {"max_segments": 2}}, result["revised_rule"])

    def test_a_round_with_no_rule_keeps_its_old_path(self):
        result = apply_reflection_policy(self._accepted(), {"type": "reroute"}, {"next_policy": "keep"})
        self.assertIsNone(result["policy"]["revised_rule"])
        self.assertEqual([], result["policy"]["applied"])
        self.assertEqual({}, result["revised_rule"])

    def test_a_revision_named_in_the_loops_own_spelling_is_aliased_into_the_whitelist(self):
        # ``relayout`` with a slot scope is the same request as ``relayout_slots``; the alias is
        # recorded so the round's record still shows what the model actually wrote.
        result = apply_reflection_policy(
            self._accepted(),
            {"type": "reroute"},
            {"next_policy": "revise", "revised_action": {"type": "Relayout", "scope_type": "slots"}},
        )
        self.assertEqual("relayout_slots", result["revised_action"]["type"])
        self.assertEqual("slot", result["revised_action"]["scope_type"])
        self.assertEqual("Relayout", result["events"][0]["aliased_from"])

    def test_a_revision_whose_name_matches_nothing_is_still_refused(self):
        result = apply_reflection_policy(
            self._accepted(),
            {"type": "reroute"},
            {"next_policy": "revise", "revised_action": {"type": "relayout", "scope_type": "panel"}},
        )
        self.assertEqual({}, result["revised_action"])
        self.assertEqual("revised_action_rejected", result["events"][0]["event"])


class NextRoundProposalTests(unittest.TestCase):
    def _soft_policy(self):
        return {
            "required": True,
            "action": "relayout_slots",
            "scope_type": "slot",
            "escalation": False,
            "violations": [{"kind": "soft", "metric": "min_panel_occupancy"}],
        }

    def _hard_policy(self):
        return {
            "required": True,
            "action": "resegment",
            "scope_type": "global",
            "escalation": True,
            "violations": [{"kind": "hard", "metric": "orphan_nodes"}],
        }

    def _judge(self, action="reroute", **payload):
        requested = {"action": action, **payload}
        return {
            "stage_feedback": [{
                "scope_type": "connector",
                "edge_ids": ["e1"],
                "problem": "awkward corner",
                "requested_actions": [requested],
            }]
        }

    def test_color_feedback_is_a_theme_replan_only_when_repair_was_required(self):
        self.assertEqual({}, color_feedback_proposal({}))
        self.assertEqual({}, color_feedback_proposal({"repair_required": False}))
        proposal = color_feedback_proposal({"repair_required": True, "feedback": [{"target_role": "canvas"}]})
        self.assertEqual("replan_theme", proposal["type"])
        self.assertTrue(action_is_executable(proposal))

    def test_a_soft_occupancy_break_does_not_starve_the_judge(self):
        chosen = select_next_round_proposal(
            decomposition_action=self._soft_policy(),
            decomposition_proposal={"type": "relayout_slots", "scope_type": "slot", "semantic_node_ids": ["n1"], "factor": 1.08, "dx": 0.0, "dy": 0.0},
            judgement=self._judge(),
        )
        self.assertEqual("reroute", chosen["type"])
        self.assertEqual(["e1"], chosen["edge_ids"])

    def test_measured_color_repair_beats_the_judge_and_a_soft_break(self):
        chosen = select_next_round_proposal(
            decomposition_action=self._soft_policy(),
            decomposition_proposal={"type": "relayout_slots", "scope_type": "slot", "factor": 1.08, "dx": 0.0, "dy": 0.0},
            judgement=self._judge(),
            pending_color_feedback={"repair_required": True},
        )
        self.assertEqual("replan_theme", chosen["type"])

    def test_a_hard_break_still_goes_before_color_and_the_judge(self):
        chosen = select_next_round_proposal(
            decomposition_action=self._hard_policy(),
            decomposition_proposal={"type": "resegment", "scope_type": "global"},
            judgement=self._judge(),
            pending_color_feedback={"repair_required": True},
        )
        self.assertEqual("resegment", chosen["type"])

    def test_soft_decomposition_is_the_fallback_when_the_judge_named_nothing(self):
        chosen = select_next_round_proposal(
            decomposition_action=self._soft_policy(),
            decomposition_proposal={"type": "relayout_slots", "scope_type": "slot", "semantic_node_ids": ["n1"], "factor": 1.08, "dx": 0.0, "dy": 0.0},
            judgement={"stage_feedback": []},
        )
        self.assertEqual("relayout_slots", chosen["type"])

    def test_a_legal_revision_still_wins(self):
        chosen = select_next_round_proposal(
            revised_action={"type": "regen_assets", "scope_type": "slot", "slot_ids": ["icon_n"]},
            decomposition_action=self._hard_policy(),
            decomposition_proposal={"type": "resegment", "scope_type": "global"},
            judgement=self._judge(),
            pending_color_feedback={"repair_required": True},
        )
        self.assertEqual("regen_assets", chosen["type"])

    def test_a_just_rolled_back_type_gives_the_other_axes_a_turn(self):
        chosen = select_next_round_proposal(
            revised_action={"type": "repair_interfaces", "scope_type": "interface", "factor": 0.9, "dx": 0.0, "dy": 0.0, "semantic_node_ids": ["n1"]},
            decomposition_action=self._soft_policy(),
            decomposition_proposal={"type": "relayout_slots", "scope_type": "slot", "semantic_node_ids": ["n1"], "factor": 1.08, "dx": 0.0, "dy": 0.0},
            judgement=self._judge("repair_interfaces", factor=1.08, dx=0.0, dy=0.0),
            skipped_action_types={"repair_interfaces"},
        )
        self.assertEqual("relayout_slots", chosen["type"])


if __name__ == "__main__":
    unittest.main()
