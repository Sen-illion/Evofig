"""The run report says whether a trained model judged the run.

``trained_judge`` is the one field a reader can check to tell a run judged by the trained
checkpoint from one judged by the heuristic gate, so it has to be derived from the answers
that were actually recorded rather than pinned to a constant.
"""

import unittest

from rfs.workflows.paper_evolve_editable import _trained_judge_flag


class TrainedJudgeFlagTests(unittest.TestCase):
    def test_the_gate_on_both_seats_reports_no_trained_judge(self):
        self.assertFalse(_trained_judge_flag({"trained": False}, {"trained": False}))

    def test_a_seated_checkpoint_on_either_seat_reports_a_trained_judge(self):
        self.assertTrue(_trained_judge_flag({"trained": True}, {"trained": False}))
        self.assertTrue(_trained_judge_flag({"trained": False}, {"trained": True}))

    def test_an_answer_without_the_flag_counts_as_untrained(self):
        self.assertFalse(_trained_judge_flag({}, {"judge_role": "frozen_heuristic"}))
        self.assertFalse(_trained_judge_flag(None, None))


if __name__ == "__main__":
    unittest.main()
