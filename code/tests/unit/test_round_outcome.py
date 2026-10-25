"""Tests for the outcome one bounded repair round is recorded as having achieved.

A round is decided by a comparison, not by subtracting two absolute scores: an accepted round
whose candidate the acceptance role preferred is an improvement even when the number went down,
and a rollback the comparison called worse is a rollback even when the number went up. These
tests pin that reading, and the fallback older records still need.
"""

import unittest

from rfs.judge_training.preference_pairs import (
    _is_measured_improvement,
    _is_measured_rollback,
    round_outcome,
)


def _observed(**overrides):
    base = {
        "action_is_repair": True,
        "decision": "accept",
        "repair_gain": 0.2,
        "comparison": "",
        "pairwise_verdict": "",
        "figure_changed": None,
        "judge": {"gate_only": False},
    }
    base.update(overrides)
    return base


class RoundOutcomeTests(unittest.TestCase):
    def test_the_comparison_outweighs_the_two_absolute_scores(self):
        accepted_with_lower_score = _observed(decision="accept", repair_gain=-0.35, comparison="pairwise", pairwise_verdict="after", figure_changed=True)
        rolled_back_with_higher_score = _observed(decision="rollback", repair_gain=0.2, comparison="pairwise", pairwise_verdict="before", figure_changed=True)

        self.assertEqual("improvement", round_outcome(accepted_with_lower_score))
        self.assertEqual("rollback", round_outcome(rolled_back_with_higher_score))

    def test_every_accept_rollback_and_score_combination_reads_off_the_comparison(self):
        cases = (
            # decision, score delta, pairwise verdict, expected outcome
            ("accept", 0.4, "after", "improvement"),
            ("accept", -0.35, "after", "improvement"),
            ("rollback", 0.2, "before", "rollback"),
            ("rollback", -0.5, "before", "rollback"),
            ("accept", 0.4, "before", "other"),
            ("rollback", -0.5, "after", "other"),
        )
        for decision, gain, verdict, expected in cases:
            observed = _observed(decision=decision, repair_gain=gain, comparison="pairwise", pairwise_verdict=verdict, figure_changed=True)
            self.assertEqual(expected, round_outcome(observed), (decision, gain, verdict))

    def test_a_figure_the_diff_says_did_not_move_is_no_outcome(self):
        self.assertEqual("other", round_outcome(_observed(comparison="pairwise", pairwise_verdict="after", figure_changed=False)))
        self.assertEqual("other", round_outcome(_observed(decision="rollback", comparison="pairwise", pairwise_verdict="before", figure_changed=False)))
        self.assertEqual("other", round_outcome(_observed(comparison="pairwise", pairwise_verdict="tie", figure_changed=True)))

    def test_a_diff_that_was_never_taken_does_not_veto_the_comparison(self):
        # Older runs and synthesised rounds carry no figure_object_diff; their comparison stands.
        self.assertEqual("improvement", round_outcome(_observed(comparison="pairwise", pairwise_verdict="after", figure_changed=None)))
        self.assertEqual("improvement", round_outcome(_observed(comparison="compared", repair_gain=0.2, figure_changed=None)))

    def test_a_comparison_taken_on_scores_reads_the_score_delta(self):
        self.assertEqual("improvement", round_outcome(_observed(comparison="compared", repair_gain=0.2, figure_changed=True)))
        self.assertEqual("rollback", round_outcome(_observed(decision="rollback", comparison="compared", repair_gain=-0.1, figure_changed=True)))
        self.assertEqual("other", round_outcome(_observed(decision="rollback", comparison="compared", repair_gain=0.0, figure_changed=True)))

    def test_without_a_comparison_the_recorded_score_delta_is_all_there_is(self):
        self.assertEqual("improvement", round_outcome(_observed(repair_gain=0.2)))
        self.assertEqual("rollback", round_outcome(_observed(decision="rollback", repair_gain=-0.1)))
        self.assertEqual("other", round_outcome(_observed(decision="rollback", repair_gain=0.0)))

    def test_a_gate_only_round_never_becomes_an_improvement(self):
        self.assertEqual("other", round_outcome(_observed(judge={"gate_only": True})))
        self.assertEqual("other", round_outcome(_observed(judge={"gate_only": True}, comparison="pairwise", pairwise_verdict="after", figure_changed=True)))

    def test_something_that_is_not_a_repair_is_no_outcome(self):
        self.assertEqual("other", round_outcome(_observed(action_is_repair=False)))

    def test_the_pair_builders_predicates_read_the_same_outcome(self):
        cases = (
            (_observed(comparison="pairwise", pairwise_verdict="after", figure_changed=True), True, False),
            (_observed(decision="rollback", comparison="pairwise", pairwise_verdict="before", figure_changed=True), False, True),
            (_observed(comparison="baseline_unavailable", repair_gain=0.0), False, False),
            (_observed(decision="rollback", repair_gain=-0.1), False, True),
        )
        for observed, improvement, rollback in cases:
            self.assertEqual(improvement, _is_measured_improvement(observed), observed)
            self.assertEqual(rollback, _is_measured_rollback(observed), observed)


if __name__ == "__main__":
    unittest.main()
