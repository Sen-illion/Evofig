"""Tests for the four-rung promotion ladder."""

import unittest

from rfs.analysis.graph import CASE_GROUPS
from rfs.contracts.evolution import normalize_pairwise_result
from rfs.evolution.direct.validation import ValidationCase, ladder_status, validate_experience_cases


def _case(split, passed, diverged=True):
    return {"split": split, "passed": passed, "action_diverged": diverged}


def _ladder(same_domain_passes, holdout_passes, *, pairs_isolated=True, all_safe=True, expected_groups=(), applicability="common"):
    same = [_case("same_domain", item) for item in same_domain_passes]
    hold = [_case("holdout", item) for item in holdout_passes]
    return ladder_status(same + hold, same, hold, pairs_isolated=pairs_isolated, all_safe=all_safe, expected_groups=expected_groups, applicability=applicability)


class LadderRungTests(unittest.TestCase):
    def test_broken_pairing_fails_before_any_rung(self):
        self.assertEqual(_ladder([True, True], [True], pairs_isolated=False)["status"], "overfit_or_failed")
        self.assertEqual(_ladder([True, True], [True], all_safe=False)["status"], "overfit_or_failed")

    def test_a_single_same_domain_case_cannot_prove_reproduction(self):
        result = _ladder([True], [True])
        self.assertEqual(result["status"], "current_case_validated")
        self.assertEqual(result["stopped_because"], "same_domain_reproduction_not_shown")

    def test_same_domain_reproduction_without_holdout_stops_at_that_rung(self):
        result = _ladder([True, True], [])
        self.assertEqual(result["status"], "same_domain_validated")
        self.assertEqual(result["stopped_because"], "cross_domain_generalization_not_shown")

    def test_failing_holdout_stops_at_same_domain(self):
        self.assertEqual(_ladder([True, True], [False])["status"], "same_domain_validated")

    def test_one_regressing_case_stops_short_of_promotion(self):
        result = _ladder([True, False], [True])
        self.assertEqual(result["status"], "cross_domain_validated")
        self.assertEqual(result["stopped_because"], "not_every_case_improved")

    def test_runs_with_no_contrast_are_reported_as_inconclusive_not_as_failure(self):
        same = [_case("same_domain", False, diverged=False), _case("same_domain", False, diverged=False)]
        hold = [_case("holdout", False, diverged=False)]
        result = ladder_status(same + hold, same, hold, pairs_isolated=True, all_safe=True)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["stopped_because"], "no_contrast_between_on_and_off")
        self.assertEqual(result["inconclusive_cases"], 3)

    def test_partial_contrast_still_reaches_the_ordinary_rungs(self):
        same = [_case("same_domain", True), _case("same_domain", True, diverged=False)]
        hold = [_case("holdout", True)]
        result = ladder_status(same + hold, same, hold, pairs_isolated=True, all_safe=True, expected_groups=())
        self.assertEqual(result["inconclusive_cases"], 1)
        self.assertEqual(result["status"], "promoted")

    def test_unreported_action_sequences_do_not_count_as_no_contrast(self):
        same = [{"split": "same_domain", "passed": True}, {"split": "same_domain", "passed": True}]
        hold = [{"split": "holdout", "passed": True}]
        result = ladder_status(same + hold, same, hold, pairs_isolated=True, all_safe=True, expected_groups=())
        self.assertEqual(result["inconclusive_cases"], 0)
        self.assertEqual(result["status"], "promoted")

    def test_full_evidence_promotes(self):
        result = _ladder([True, True], [True])
        self.assertEqual(result["status"], "promoted")
        self.assertEqual(result["stopped_because"], "")
        self.assertEqual(result["total_passed"], 3)

    def test_missing_cross_domain_isomorphism_stops_at_same_domain(self):
        same = [
            {"split": "same_domain", "group": "same_domain_same_structure", "passed": True, "action_diverged": True},
            {"split": "same_domain", "group": "same_domain_same_structure", "passed": True, "action_diverged": True},
        ]
        hold = [{"split": "holdout", "group": "same_domain_cross_structure", "passed": True, "action_diverged": True}]
        result = ladder_status(same + hold, same, hold, pairs_isolated=True, all_safe=True)
        self.assertEqual(result["status"], "same_domain_validated")
        self.assertEqual(result["stopped_because"], "cross_domain_same_structure_uncovered")

    def test_missing_boundary_cannot_promote(self):
        same = [
            {"split": "same_domain", "group": "same_domain_same_structure", "passed": True, "action_diverged": True},
            {"split": "same_domain", "group": "same_domain_same_structure", "passed": True, "action_diverged": True},
        ]
        hold = [{"split": "holdout", "group": "cross_domain_same_structure", "passed": True, "action_diverged": True}]
        result = ladder_status(same + hold, same, hold, pairs_isolated=True, all_safe=True)
        self.assertEqual(result["status"], "cross_domain_validated")
        self.assertEqual(result["stopped_because"], "boundary_uncovered")

    def test_four_groups_including_a_safe_boundary_promote(self):
        same = [
            {"split": "same_domain", "group": "same_domain_same_structure", "passed": True, "action_diverged": True},
            {"split": "same_domain", "group": "same_domain_cross_structure", "passed": True, "action_diverged": True},
        ]
        hold = [{"split": "holdout", "group": "cross_domain_same_structure", "passed": True, "action_diverged": True}]
        boundary = [{"split": "boundary", "group": "boundary", "passed": True, "action_diverged": True}]
        result = ladder_status(same + hold + boundary, same, hold, pairs_isolated=True, all_safe=True)
        self.assertEqual(result["status"], "promoted")
        self.assertEqual(result["group_coverage"]["boundary"], 1)
        self.assertEqual(CASE_GROUPS, ("same_domain_same_structure", "cross_domain_same_structure", "same_domain_cross_structure", "boundary"))

    def test_a_domain_experience_with_a_safe_boundary_is_domain_validated(self):
        same = [{**_case("same_domain", True), "measured": True}, {**_case("same_domain", True), "measured": True}]
        boundary = [{"split": "boundary", "group": "boundary", "passed": True, "action_diverged": True, "measured": True}]
        result = ladder_status(same + boundary, same, [], pairs_isolated=True, all_safe=True, applicability="domain")
        self.assertEqual(result["status"], "domain_validated")
        self.assertEqual(result["stopped_because"], "")

    def test_a_domain_experience_without_a_boundary_cannot_be_marked_usable(self):
        same = [{**_case("same_domain", True), "measured": True}, {**_case("same_domain", True), "measured": True}]
        result = ladder_status(same, same, [], pairs_isolated=True, all_safe=True, applicability="domain")
        self.assertEqual(result["status"], "current_case_validated")
        self.assertEqual(result["stopped_because"], "boundary_uncovered")

    def test_a_domain_experience_that_does_worse_in_another_domain_is_not_overfit(self):
        same = [{**_case("same_domain", True), "measured": True}, {**_case("same_domain", True), "measured": True}]
        hold = [{**_case("holdout", False), "measured": True}]
        boundary = [{"split": "boundary", "group": "boundary", "passed": True, "action_diverged": True, "measured": True}]
        result = ladder_status(same + hold + boundary, same, hold, pairs_isolated=True, all_safe=False, applicability="domain")
        self.assertEqual(result["status"], "domain_validated")

    def test_an_unlabeled_experience_does_not_climb(self):
        result = _ladder([True, True], [True], applicability="")
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["stopped_because"], "applicability_unspecified")


class LadderThroughValidationTests(unittest.TestCase):
    def _run(self, scores, *, target_metric="overall_score"):
        cases = [ValidationCase("a", "ai_method", "same_domain"), ValidationCase("b", "ai_method", "same_domain"), ValidationCase("c", "biology", "holdout")]
        audit = {"paired_config_hash": "h", "initial_state_hash": "h", "candidate_policy_hash": "h", "evaluator_hash": "h"}

        def runner(case, enabled, record):
            value = scores[case.case_id][1 if enabled else 0]
            return {**audit, "overall_score": value, "dimension_scores": {"density_fitness": value}, "blocking_issue_count": 0}

        return validate_experience_cases({"experience_id": "e1", "applicability": "common"}, cases, runner, target_metric=target_metric, expected_groups=())

    def test_uniform_gain_reaches_promoted(self):
        report = self._run({"a": (0.8, 0.9), "b": (0.8, 0.9), "c": (0.8, 0.9)})
        self.assertEqual(report["status"], "promoted")
        self.assertEqual(report["ladder"]["status"], "promoted")

    def test_holdout_without_gain_stops_at_same_domain(self):
        report = self._run({"a": (0.8, 0.9), "b": (0.8, 0.9), "c": (0.8, 0.8)})
        self.assertEqual(report["status"], "same_domain_validated")

    def test_a_continuous_target_metric_is_reported_on_the_verdict(self):
        report = self._run({"a": (0.8, 0.84), "b": (0.8, 0.84), "c": (0.8, 0.84)}, target_metric="density_fitness")
        self.assertEqual(report["target_metric"], "density_fitness")
        self.assertEqual(report["status"], "promoted")

    def test_a_compared_case_whose_arm_left_no_score_is_still_summarised(self):
        """The comparison decides the case; the summary must not demand a number from it."""

        cases = [ValidationCase("a", "ai_method", "same_domain")]
        audit = {"paired_config_hash": "h", "initial_state_hash": "h", "candidate_policy_hash": "h", "evaluator_hash": "h"}

        def runner(case, enabled, record):
            if enabled:
                return {**audit, "overall_score": 1.0, "measured": True, "dimension_scores": {}, "blocking_issue_count": 0}
            return {**audit, "overall_score": None, "measured": False, "dimension_scores": {}, "blocking_issue_count": 0}

        comparison = normalize_pairwise_result({"preferred_candidate": "B", "candidate_a": "without", "candidate_b": "with"})
        report = validate_experience_cases(
            {"experience_id": "e1", "applicability": "common"}, cases, runner, pairwise=lambda off, on: comparison
        )
        self.assertEqual(report["summary"]["measured_cases"], 1)
        self.assertIsNone(report["summary"]["mean_target_delta"])
        self.assertEqual(report["status"], "current_case_validated")



def _measured(split, passed=True, *, group="", diverged=True):
    return {"split": split, "passed": passed, "action_diverged": diverged, "measured": True, "group": group}


class CarriedNewDefectTests(unittest.TestCase):
    """A row that introduced a defect may be filed, and may not travel until the defect clears."""

    def _domain_run(self, *, carried, all_safe):
        same = [_measured("same_domain"), _measured("same_domain")]
        boundary = [_measured("boundary", group="boundary")]
        return ladder_status(
            same + boundary,
            same,
            [],
            pairs_isolated=True,
            all_safe=all_safe,
            expected_groups=(),
            applicability="domain",
            carried_new_defect=carried,
        )

    def test_a_clean_row_reaches_the_domain_rung(self):
        result = self._domain_run(carried=False, all_safe=False)
        self.assertEqual(result["status"], "domain_validated")

    def test_a_row_carrying_a_defect_is_held_at_the_rung_it_had(self):
        result = self._domain_run(carried=True, all_safe=False)
        self.assertEqual(result["status"], "current_case_validated")
        self.assertEqual(result["stopped_because"], "carried_new_defect")
        self.assertFalse(result["carried_new_defect_cleared"])

    def test_the_mark_is_spent_when_the_set_shows_the_defect_did_not_reproduce(self):
        result = self._domain_run(carried=True, all_safe=True)
        self.assertEqual(result["status"], "domain_validated")
        self.assertTrue(result["carried_new_defect_cleared"])

    def test_a_row_that_carries_nothing_is_never_reported_as_cleared(self):
        result = self._domain_run(carried=False, all_safe=True)
        self.assertNotIn("carried_new_defect_cleared", result)

if __name__ == "__main__":
    unittest.main()
