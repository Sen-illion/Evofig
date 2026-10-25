"""P5: a pair the Judge decides differently each time is a noise reading, not a verdict.

One judgement of a pair is one sample of a model that rounds. Asked once, the same pair has
read ``after`` on one run and ``before`` on the next, and either reading was then counted as
evidence about the experience. The case now counts only when the readings repeat.
"""

import unittest

from rfs.evolution.direct import ValidationCase, validate_experience_cases


def case_set():
    return [
        ValidationCase("a", "ai_method", "same_domain", "same_domain_same_structure"),
        ValidationCase("b", "ai_method", "same_domain", "same_domain_same_structure"),
        ValidationCase("c", "system", "holdout", "cross_domain_same_structure"),
        ValidationCase("d", "ai_method", "boundary", "boundary"),
    ]


def runner(case, enabled, experience):
    return {
        "overall_score": 0.83 if enabled else 0.8,
        "blocking_issue_count": 0,
        "paired_config_hash": "cfg",
        "initial_state_hash": "start",
        "candidate_policy_hash": "candidates",
        "evaluator_hash": "judge",
    }


def comparison(choice: str) -> dict:
    return {"judgement_kind": "pairwise", "pairwise": {"choice": choice}}


EXPERIENCE = {"experience_id": "e1", "applicability": "common"}


class MeasurementSamplingTests(unittest.TestCase):
    def test_a_pair_the_judge_cannot_decide_the_same_way_twice_is_noise(self):
        # Three readings per case: the first case is answered B, A, B, the rest repeat B.
        votes = ["B", "A", "B", "B", "B", "B", "B", "B", "B", "B", "B", "B"]
        seen: list[int] = []

        def pairwise(off, on):
            seen.append(len(seen))
            return comparison(votes[len(seen) - 1])

        result = validate_experience_cases(EXPERIENCE, case_set(), runner, pairwise=pairwise)

        self.assertNotEqual(result["status"], "promoted")
        self.assertEqual(result["ladder"]["stopped_because"], "measurement_noise")
        self.assertEqual(result["ladder"]["noise_cases"], 1)
        self.assertEqual(result["summary"]["measured_cases"], 3)
        noisy = [item for item in result["results"] if item["noise"]]
        self.assertEqual([item["case_id"] for item in noisy], ["a"])
        self.assertFalse(noisy[0]["measured"])
        self.assertFalse(noisy[0]["passed"])

    def test_a_pair_the_judge_decides_the_same_way_every_time_is_a_verdict(self):
        def pairwise(off, on):
            return comparison("B")

        result = validate_experience_cases(EXPERIENCE, case_set(), runner, pairwise=pairwise)

        self.assertEqual(result["status"], "promoted")
        self.assertEqual(result["ladder"]["noise_cases"], 0)
        self.assertEqual(result["summary"]["measured_cases"], 4)

    def test_the_sampling_can_be_turned_down_to_one_reading_per_pair(self):
        """A single reading cannot disagree with itself, which is exactly the old behaviour."""

        seen: list[str] = []

        def pairwise(off, on):
            seen.append(off["overall_score"])
            return comparison("A")

        result = validate_experience_cases(EXPERIENCE, case_set(), runner, pairwise=pairwise, judge_samples=1)

        self.assertEqual(len(seen), 4)
        self.assertEqual(result["ladder"]["noise_cases"], 0)
        self.assertNotEqual(result["status"], "promoted")

    def test_a_judge_that_can_answer_nothing_is_unmeasured_not_noisy(self):
        def pairwise(off, on):
            return {}

        result = validate_experience_cases(EXPERIENCE, case_set(), runner, pairwise=pairwise)

        self.assertEqual(result["ladder"]["noise_cases"], 0)
        self.assertEqual(result["ladder"]["stopped_because"], "pair_not_measured")


if __name__ == "__main__":
    unittest.main()
