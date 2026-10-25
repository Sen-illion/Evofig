"""Tests for the ``quick`` validation profile and the rung it writes.

The quick profile is not a faster full ladder: it asks one matched paper and nothing else, and
marks the row on a rung of its own so a reader can tell it apart from a promoted one. These tests
pin both halves -- that it really asks one case, and that the rung it writes is visible only to a
caller that named the profile.
"""

import unittest

from rfs.analysis.graph import (
    DOMAIN_QUICK_CASE_QUOTA,
    FULL_PROFILE,
    QUICK_CASE_QUOTA,
    QUICK_PROFILE,
    case_set_qualification,
    quick_case_quota,
    required_case_groups,
)
from rfs.analysis.graph.retrieval import REQUIRED_CASE_GROUPS
from rfs.contracts.evolution import EXPERIENCE_STATUSES
from rfs.evolution.direct.store import (
    ALLOWED_APPEND_STATUSES,
    DEFAULT_RETRIEVAL_STATUSES,
    RELAXED_RETRIEVAL_STATUSES,
    resolve_retrieval_statuses,
)
from rfs.evolution.direct.validation import ValidationCase, ladder_status, validate_experience_cases


def _quick(results, *, pairs_isolated=True, all_safe=True):
    return ladder_status(
        results,
        [item for item in results if item.get("split") == "same_domain"],
        [item for item in results if item.get("split") == "holdout"],
        pairs_isolated=pairs_isolated,
        all_safe=all_safe,
        applicability="common",
        profile=QUICK_PROFILE,
    )


class QuickQuotaTests(unittest.TestCase):
    def test_the_quick_quota_is_one_matched_paper_and_nothing_else(self):
        self.assertEqual(quick_case_quota("common"), QUICK_CASE_QUOTA)
        self.assertEqual(sum(QUICK_CASE_QUOTA.values()), 1)
        self.assertEqual(QUICK_CASE_QUOTA["cross_domain_same_structure"], 1)

    def test_a_domain_row_is_asked_about_its_own_domain(self):
        self.assertEqual(quick_case_quota("domain"), DOMAIN_QUICK_CASE_QUOTA)
        self.assertEqual(sum(DOMAIN_QUICK_CASE_QUOTA.values()), 1)
        self.assertEqual(DOMAIN_QUICK_CASE_QUOTA["same_domain_cross_structure"], 1)
        self.assertEqual(DOMAIN_QUICK_CASE_QUOTA["cross_domain_same_structure"], 0)

    def test_the_quick_profile_requires_no_group(self):
        self.assertEqual(required_case_groups("common", profile=QUICK_PROFILE), ())
        self.assertEqual(required_case_groups("domain", profile=QUICK_PROFILE), ())

    def test_every_other_profile_keeps_the_contract(self):
        self.assertEqual(required_case_groups("common"), REQUIRED_CASE_GROUPS["common"])
        self.assertEqual(required_case_groups("domain"), REQUIRED_CASE_GROUPS["domain"])
        self.assertEqual(required_case_groups("common", profile=FULL_PROFILE), REQUIRED_CASE_GROUPS["common"])

    def test_a_set_that_could_never_fill_the_full_contract_qualifies_for_quick(self):
        quota = quick_case_quota("common")
        counts = {"cross_domain_same_structure": 1}
        self.assertTrue(case_set_qualification("common", quota=quota, pool_counts=counts, profile=QUICK_PROFILE)["ok"])
        full = case_set_qualification("common", quota=quota, pool_counts=counts)
        self.assertFalse(full["ok"])
        self.assertEqual(full["profile"], FULL_PROFILE)

    def test_an_unlabeled_row_still_does_not_qualify(self):
        result = case_set_qualification("", quota=quick_case_quota(""), pool_counts={}, profile=QUICK_PROFILE)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "applicability_unspecified")


class QuickLadderTests(unittest.TestCase):
    def test_one_matched_case_that_prefers_the_row_writes_the_quick_rung(self):
        result = _quick([{"split": "holdout", "passed": True, "action_diverged": True, "measured": True}])
        self.assertEqual(result["status"], "quick_validated")
        self.assertEqual(result["stopped_because"], "")

    def test_one_matched_case_that_refuses_it_leaves_the_row_where_it_was(self):
        result = _quick([{"split": "holdout", "passed": False, "action_diverged": True, "measured": True}])
        self.assertEqual(result["status"], "current_case_validated")
        self.assertEqual(result["stopped_because"], "matched_case_did_not_improve")

    def test_no_contrast_is_inconclusive_rather_than_a_failure(self):
        result = _quick([{"split": "holdout", "passed": False, "action_diverged": False, "measured": True}])
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["stopped_because"], "no_contrast_between_on_and_off")

    def test_a_pair_whose_arms_never_differed_is_inconclusive_not_a_failure(self):
        result = _quick([{"split": "holdout", "passed": False, "action_diverged": False, "measured": True}], all_safe=False)
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["stopped_because"], "no_contrast_between_on_and_off")

    def test_a_case_nobody_measured_is_refused_for_its_own_reason(self):
        result = _quick([{"split": "holdout", "passed": False, "action_diverged": True, "measured": False}])
        self.assertEqual(result["status"], "overfit_or_failed")
        self.assertTrue(result["stopped_because"])

    def test_a_broken_pairing_is_refused_before_the_rung(self):
        self.assertEqual(_quick([{"split": "holdout", "passed": True, "measured": True}], pairs_isolated=False)["status"], "overfit_or_failed")
        self.assertEqual(_quick([{"split": "holdout", "passed": True, "measured": True}], all_safe=False)["status"], "overfit_or_failed")

    def test_the_full_profile_is_untouched_by_the_quick_rung(self):
        result = ladder_status(
            [{"split": "holdout", "passed": True, "action_diverged": True, "measured": True}],
            [],
            [{"split": "holdout", "passed": True, "action_diverged": True, "measured": True}],
            pairs_isolated=True,
            all_safe=True,
            applicability="common",
        )
        self.assertEqual(result["status"], "current_case_validated")


class QuickThroughValidationTests(unittest.TestCase):
    def _run(self, scores, *, profile=QUICK_PROFILE):
        cases = [ValidationCase("c", "biology", "holdout")]
        audit = {"paired_config_hash": "h", "initial_state_hash": "h", "candidate_policy_hash": "h", "evaluator_hash": "h"}

        def runner(case, enabled, record):
            value = scores[case.case_id][1 if enabled else 0]
            return {**audit, "overall_score": value, "dimension_scores": {}, "blocking_issue_count": 0}

        return validate_experience_cases({"experience_id": "e1", "applicability": "common"}, cases, runner, expected_groups=(), profile=profile)

    def test_a_gain_on_the_one_case_is_reported_on_the_quick_rung(self):
        report = self._run({"c": (0.8, 0.9)})
        self.assertEqual(report["status"], "quick_validated")
        self.assertEqual(report["validation_profile"], QUICK_PROFILE)

    def test_no_gain_stops_at_the_row_it_started_from(self):
        report = self._run({"c": (0.8, 0.8)})
        self.assertEqual(report["status"], "current_case_validated")

    def test_the_profile_is_reported_even_when_none_was_named(self):
        report = self._run({"c": (0.8, 0.9)}, profile="")
        self.assertEqual(report["validation_profile"], FULL_PROFILE)


class QuickRungVisibilityTests(unittest.TestCase):
    def test_the_quick_rung_is_a_known_status(self):
        self.assertIn("quick_validated", EXPERIENCE_STATUSES)
        self.assertIn("quick_validated", ALLOWED_APPEND_STATUSES)

    def test_no_run_draws_from_the_quick_rung_by_default(self):
        self.assertNotIn("quick_validated", DEFAULT_RETRIEVAL_STATUSES)
        self.assertNotIn("quick_validated", resolve_retrieval_statuses("strict"))

    def test_a_caller_that_names_the_relaxed_bar_may_draw_from_it(self):
        self.assertIn("quick_validated", RELAXED_RETRIEVAL_STATUSES)
        self.assertIn("quick_validated", resolve_retrieval_statuses("relaxed"))


if __name__ == "__main__":
    unittest.main()
