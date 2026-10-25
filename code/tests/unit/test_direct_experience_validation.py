"""Tests for paired experience validation and promotion."""

import unittest

from rfs.evolution.direct import ValidationCase, validate_experience_cases


class DirectExperienceValidationTests(unittest.TestCase):
    def test_promotes_when_two_same_domain_and_one_holdout_improve(self):
        cases = [
            ValidationCase("a", "ai_method", "same_domain", "same_domain_same_structure"),
            ValidationCase("b", "ai_method", "same_domain", "same_domain_same_structure"),
            ValidationCase("c", "system", "holdout", "cross_domain_same_structure"),
            ValidationCase("d", "ai_method", "boundary", "boundary"),
        ]

        def runner(case, enabled, experience):
            delta = 0.0 if case.case_id == "d" else (0.03 if enabled else 0.0)
            return {"overall_score": 0.8 + delta, "blocking_issue_count": 0, "paired_config_hash": "cfg", "initial_state_hash": case.case_id, "candidate_policy_hash": "candidates", "evaluator_hash": "judge"}

        result = validate_experience_cases({"experience_id": "e1", "applicability": "common"}, cases, runner)
        self.assertEqual(result["status"], "promoted")

    def test_rejects_when_holdout_regresses(self):
        cases = [ValidationCase("a", "ai_method"), ValidationCase("b", "ai_method"), ValidationCase("c", "system", "holdout")]

        def runner(case, enabled, experience):
            delta = -0.02 if enabled and case.case_id == "c" else 0.03 if enabled else 0.0
            return {"overall_score": 0.8 + delta, "blocking_issue_count": 0, "paired_config_hash": "cfg", "initial_state_hash": case.case_id, "candidate_policy_hash": "candidates", "evaluator_hash": "judge"}

        result = validate_experience_cases({"experience_id": "e1", "applicability": "common"}, cases, runner)
        self.assertEqual(result["status"], "overfit_or_failed")

    def test_rejects_non_isolated_pair(self):
        cases = [ValidationCase("a", "ai_method"), ValidationCase("b", "ai_method"), ValidationCase("c", "system", "holdout")]

        def runner(case, enabled, experience):
            return {"overall_score": 0.8 + (0.03 if enabled else 0.0), "blocking_issue_count": 0, "paired_config_hash": "on" if enabled else "off", "initial_state_hash": case.case_id, "candidate_policy_hash": "candidates", "evaluator_hash": "judge"}

        result = validate_experience_cases({"experience_id": "e1", "applicability": "common"}, cases, runner)
        self.assertEqual("overfit_or_failed", result["status"])
        self.assertFalse(result["all_pairs_isolated"])


if __name__ == "__main__":
    unittest.main()
