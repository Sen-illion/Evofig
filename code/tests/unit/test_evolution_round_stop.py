"""When the round loop is allowed to finish early.

The loop stops by itself when a round reaches a passing figure, which keeps a run from paying
for rounds that cannot add anything. That answer belongs to the state the round kept, so a
round that was rolled back may not end the run on the candidate it just rejected: doing so
stopped a real run on a figure the acceptance role had scored 0.0 with two blocking issues,
and it threw away the escalation the round had already queued for the next one.
"""

import unittest

from rfs.workflows.paper_evolve_editable import _evolution_should_stop


class EvolutionRoundStopTests(unittest.TestCase):
    def _stop(self, **overrides):
        arguments = {
            "evolution_enabled": True,
            "fixed_rounds": False,
            "accepted": True,
            "judgement": {"status": "pass"},
            "pending_seed_action": None,
        }
        return _evolution_should_stop(**{**arguments, **overrides})

    def test_an_accepted_round_that_passes_ends_the_loop(self):
        self.assertTrue(self._stop())

    def test_a_rolled_back_round_never_ends_the_loop_even_when_its_candidate_passed(self):
        self.assertFalse(self._stop(accepted=False))

    def test_an_accepted_round_that_did_not_pass_keeps_going(self):
        self.assertFalse(self._stop(judgement={"status": "fault"}))
        self.assertFalse(self._stop(judgement=None))

    def test_fixed_rounds_and_a_queued_seed_both_keep_the_loop_going(self):
        self.assertFalse(self._stop(fixed_rounds=True))
        self.assertFalse(self._stop(pending_seed_action={"experience_id": "direct-exp-1"}))

    def test_a_run_without_evolution_finishes_after_one_round(self):
        self.assertTrue(self._stop(evolution_enabled=False))


if __name__ == "__main__":
    unittest.main()
