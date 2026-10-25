"""Unit tests for scripts/replay_round_decisions.py.

The replay re-judges decisions a run already recorded, so these tests drive it with synthetic
records that spell out each path the rule can take: a comparison the old regression veto refused,
a comparison the model called worse, a noise flag that used to withdraw an accept, a round that
moved nothing, and a score path. No rounds are run and no historical data is needed.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_replay_module():
    spec = importlib.util.spec_from_file_location("replay_round_decisions", ROOT / "scripts" / "replay_round_decisions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


replay = load_replay_module()


def record(**overrides):
    base = {
        "accepted": False,
        "reason": "pairwise_not_preferred",
        "comparison": "pairwise",
        "pairwise_verdict": "before",
        "coverage_improved": False,
        "score_delta": None,
    }
    base.update(overrides)
    return base


class RejudgeDecisionTest(unittest.TestCase):
    def test_a_round_the_comparison_preferred_is_accepted_though_the_gate_named_a_blocker(self):
        # The deadlock in one line: the model said the new figure was better, a deterministic
        # check named one more defect, and the round was refused. The gate only reports now.
        answer = replay.rejudge_decision(
            record(reason="new_blocker", pairwise_verdict="after", new_blocking_defects=[{"defect_key": "interface:box_pass"}])
        )
        self.assertTrue(answer["accepted"])
        self.assertEqual(answer["reason"], "pairwise_preferred")
        self.assertTrue(answer["changed"])

    def test_a_round_the_comparison_called_worse_stays_refused(self):
        answer = replay.rejudge_decision(record(pairwise_verdict="before"))
        self.assertFalse(answer["accepted"])
        self.assertEqual(answer["reason"], "pairwise_not_preferred")
        self.assertFalse(answer["changed"])

    def test_a_coverage_gain_no_longer_keeps_a_round_the_comparison_did_not_prefer(self):
        # The old rule accepted this round on the coverage reading. The one ruler refuses it and
        # says the two readings disagreed, so the round the old rule kept is visible in the report.
        answer = replay.rejudge_decision(record(accepted=True, reason="coverage_improved_over_pairwise", pairwise_verdict="before", coverage_improved=True))
        self.assertFalse(answer["accepted"])
        self.assertEqual(answer["reason"], "pairwise_not_preferred")
        self.assertTrue(answer["coverage_disagreed"])
        self.assertTrue(answer["changed"])

    def test_a_noise_flag_no_longer_withdraws_a_preferred_round(self):
        answer = replay.rejudge_decision(record(reason="measurement_flagged_randomness", pairwise_verdict="after"))
        self.assertTrue(answer["accepted"])
        self.assertEqual(answer["reason"], "pairwise_preferred")
        self.assertTrue(answer["withdrawn_by_noise"])

    def test_a_withdrawn_first_candidate_goes_back_to_being_accepted(self):
        answer = replay.rejudge_decision(record(accepted=False, reason="measurement_flagged_randomness", comparison="none", pairwise_verdict=None))
        self.assertTrue(answer["accepted"])
        self.assertEqual(answer["reason"], "initial_candidate")
        self.assertTrue(answer["recovered_from_noise"])

    def test_an_ordinary_first_candidate_is_accepted_without_change(self):
        answer = replay.rejudge_decision(record(accepted=True, reason="initial_candidate", comparison="none", pairwise_verdict=None))
        self.assertTrue(answer["accepted"])
        self.assertFalse(answer["changed"])

    def test_a_round_that_moved_nothing_is_still_refused(self):
        answer = replay.rejudge_decision(record(accepted=False, reason="figure_unchanged", comparison="figure_unchanged", pairwise_verdict=None))
        self.assertFalse(answer["accepted"])
        self.assertEqual(answer["reason"], "figure_unchanged")
        self.assertFalse(answer["changed"])

    def test_the_score_path_reads_the_absolute_reading_against_the_tolerance(self):
        kept = replay.rejudge_decision(record(reason="no_significant_improvement", comparison="compared", pairwise_verdict=None, score_delta=0.4))
        self.assertTrue(kept["accepted"])
        self.assertEqual(kept["reason"], "target_improved")
        flat = replay.rejudge_decision(record(reason="no_significant_improvement", comparison="compared", pairwise_verdict=None, score_delta=0.004))
        self.assertFalse(flat["accepted"])

    def test_a_record_with_nothing_to_re_judge_is_left_unresolved(self):
        answer = replay.rejudge_decision({"accepted": False, "reason": "something_else"})
        self.assertTrue(answer["unresolved"])
        self.assertIsNone(answer["accepted"])


class ReplayDecisionsTest(unittest.TestCase):
    def test_a_run_directory_reports_what_moved_and_that_the_rule_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            better = root / "iter_05" / "round_01"
            worse = root / "iter_05" / "round_02"
            withdrawn = root / "iter_06" / "round_01"
            for path, payload in (
                (better, record(reason="new_blocker", pairwise_verdict="after")),
                (worse, record(pairwise_verdict="before")),
                (withdrawn, record(reason="measurement_flagged_randomness", pairwise_verdict="after")),
            ):
                path.mkdir(parents=True)
                (path / "repair_decision.json").write_text(json.dumps(payload), encoding="utf-8")
            report = replay.replay_decisions(sorted(root.glob("**/repair_decision.json")))
            self.assertTrue(report["ok"], report["checks"])
            self.assertEqual(report["decision_count"], 3)
            self.assertEqual(report["recorded_accepted_count"], 0)
            self.assertEqual(report["rejudged_accepted_count"], 2)
            self.assertEqual(report["flipped_to_accept_count"], 2)
            self.assertEqual(report["flipped_to_refuse_count"], 0)
            self.assertEqual(report["model_called_better_refused_count"], 2)
            self.assertEqual(report["noise_killed_after_count"], 1)
            self.assertTrue(all(check["ok"] for check in report["checks"]))

    def test_a_broken_record_is_reported_rather_than_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "iter_01" / "round_00"
            path.mkdir(parents=True)
            (path / "repair_decision.json").write_text("{ not json", encoding="utf-8")
            report = replay.replay_decisions(sorted(pathlib.Path(tmp).glob("**/repair_decision.json")))
            self.assertFalse(report["ok"])
            self.assertTrue(any(not check["ok"] for check in report["checks"]))


if __name__ == "__main__":
    unittest.main()