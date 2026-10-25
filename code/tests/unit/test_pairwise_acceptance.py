"""Tests for deciding a round and a memory pair by comparing two states.

The loop used to subtract two absolute scores the model had itself rounded to a 0.1 grid: of the
74 numbers the served model produced, every one but a single 0.65 sat on that grid. These tests
pin the replacement -- the two states are shown together, the answer is a preference, and no
number is invented anywhere along the way.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rfs.contracts.evolution import normalize_pairwise_result, pairwise_verdict, validate_pairwise_result
from rfs.evaluation.direct_judges import BootstrapVLMDirectJudge, DirectJudgeInput, DirectPairJudgeInput
from rfs.evolution.direct import ValidationCase, compare_candidate, validate_experience_cases


def _input(preview: Path, *, gate: dict | None = None) -> DirectJudgeInput:
    return DirectJudgeInput(
        method={"modules": [{"id": "n"}], "relations": []},
        layout={"panels": [], "nodes": [], "connectors": []},
        icons={"icons": []},
        program={"panels": [], "semantic_nodes": [], "arrows": []},
        preview=preview,
        grounding_gate=gate,
    )


def _comparison(choice: str, *, role: str = "frozen_bootstrap") -> dict:
    return normalize_pairwise_result(
        {"preferred_candidate": choice, "margin": "clear"},
        role=role,
        candidate_a="the state the loop had already accepted",
        candidate_b="the state this round produced",
    )


class PairwiseContractTests(unittest.TestCase):
    def test_a_comparison_answers_with_a_preference_and_invents_no_score(self) -> None:
        result = normalize_pairwise_result(
            {"preferred_candidate": "B", "margin": "clear", "confidence": 0.8, "blocking_issues": ["connector crosses a panel"]},
            role="frozen_bootstrap",
            candidate_a="the accepted state",
            candidate_b="this round",
        )
        self.assertEqual("pairwise", result["judgement_kind"])
        self.assertIsNone(result["overall_score"])
        self.assertFalse(result["score_measured"])
        self.assertEqual("after", pairwise_verdict(result))
        # The mapping travels with the answer, because "B was preferred" means nothing without it.
        self.assertEqual("the accepted state", result["pairwise"]["candidate_a"])
        self.assertEqual(1, result["blocking_issue_count"])
        self.assertTrue(validate_pairwise_result(result)["ok"])
        # A stored record re-validates after a round trip through JSON, without being handed the
        # mapping again: the names are inside the record.
        self.assertTrue(validate_pairwise_result(json.loads(json.dumps(result)))["ok"])

    def test_a_tie_is_an_answer_and_a_word_nobody_defined_is_not(self) -> None:
        tied = _comparison("tie")
        self.assertEqual("tie", pairwise_verdict(tied))
        self.assertTrue(validate_pairwise_result(tied)["ok"])
        junk = _comparison("maybe")
        self.assertEqual("", pairwise_verdict(junk))
        self.assertFalse(validate_pairwise_result(junk)["ok"])

    def test_a_comparison_must_name_the_two_states_it_saw(self) -> None:
        anonymous = normalize_pairwise_result({"preferred_candidate": "A"}, role="frozen_bootstrap")
        self.assertIn("a comparison must name both states it compared", validate_pairwise_result(anonymous)["errors"])


class PairwiseRoundDecisionTests(unittest.TestCase):
    def test_a_comparison_that_prefers_the_new_state_is_accepted(self) -> None:
        decision = compare_candidate(
            {"overall_score": 0.4, "blocking_issue_count": 0},
            {"overall_score": 0.45, "blocking_issue_count": 0},
            after_pairwise=_comparison("B"),
        )
        self.assertTrue(decision["accepted"])
        self.assertEqual("pairwise", decision["comparison"])
        self.assertEqual("pairwise_preferred", decision["reason"])
        self.assertEqual("after", decision["pairwise_verdict"])
        # Both scores stay in the record: they are the trend, no longer the decision.
        self.assertEqual(0.05, decision["score_delta"])
        self.assertEqual(0.4, decision["before_score"])

    def test_a_comparison_that_prefers_the_old_state_rolls_the_round_back(self) -> None:
        decision = compare_candidate(
            {"overall_score": 0.4, "blocking_issue_count": 0},
            {"overall_score": 0.9, "blocking_issue_count": 0},
            after_pairwise=_comparison("A"),
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual("pairwise_not_preferred", decision["reason"])
        # The score still rose, and the record says so rather than hiding it.
        self.assertEqual(0.5, decision["score_delta"])

    def test_a_tie_does_not_pass_as_an_improvement(self) -> None:
        decision = compare_candidate({"overall_score": 0.4, "blocking_issue_count": 0}, {"overall_score": 0.9, "blocking_issue_count": 0}, after_pairwise=_comparison("tie"))
        self.assertFalse(decision["accepted"])
        self.assertEqual("pairwise_not_preferred", decision["reason"])

    def test_a_new_blocker_is_evidence_beside_a_preference_not_a_veto(self) -> None:
        # The mechanism does not refuse a round the comparison preferred. The blocker is named
        # on the round -- and in its evidence -- as the work the next round inherits.
        decision = compare_candidate(
            {"overall_score": 0.4, "blocking_issue_count": 0},
            {"overall_score": 0.9, "blocking_issue_count": 1},
            after_pairwise=_comparison("B"),
        )
        self.assertTrue(decision["accepted"])
        self.assertEqual("pairwise_preferred", decision["reason"])
        self.assertEqual(1, decision["new_blockers"])
        self.assertEqual(1, decision["evidence"]["after_blockers"])

    def test_a_comparison_nobody_could_take_refuses_the_round(self) -> None:
        decision = compare_candidate(
            {"overall_score": 0.4, "blocking_issue_count": 0},
            {"overall_score": 0.9, "blocking_issue_count": 0},
            after_pairwise=normalize_pairwise_result({}, role="frozen_bootstrap", candidate_a="a", candidate_b="b"),
        )
        self.assertFalse(decision["accepted"])
        self.assertEqual("pairwise_unavailable", decision["reason"])

    def test_the_measured_score_path_still_runs_when_no_comparison_was_asked_for(self) -> None:
        decision = compare_candidate({"overall_score": 0.8}, {"overall_score": 0.82})
        self.assertTrue(decision["accepted"])
        self.assertEqual("compared", decision["comparison"])
        self.assertNotIn("pairwise_verdict", decision)


class PairwiseJudgeTests(unittest.TestCase):
    def test_both_states_go_into_one_call_and_the_answer_is_a_preference(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            before, after = tmp / "before.png", tmp / "after.png"
            before.write_bytes(b"before")
            after.write_bytes(b"after")
            judge = BootstrapVLMDirectJudge(model="ordinary-vlm", frozen=True)
            with patch("rfs.evaluation.direct_judges.call_vlm_json", return_value={"preferred_candidate": "B", "margin": "slight", "confidence": 0.6}) as model:
                result = judge.evaluate_pairwise(DirectPairJudgeInput(before=_input(before), after=_input(after)))
            self.assertEqual(1, model.call_count)
            question, images = model.call_args.args
            self.assertEqual([str(before), str(after)], images)
            self.assertIn("the first is A", question)
            self.assertIn("margin", question)
            self.assertEqual("after", pairwise_verdict(result))
            self.assertIsNone(result["overall_score"])
            self.assertFalse(result["gate_only"])

    def test_a_missing_state_is_recorded_rather_than_compared(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            before = Path(folder) / "before.png"
            before.write_bytes(b"before")
            judge = BootstrapVLMDirectJudge(model="ordinary-vlm", frozen=True)
            with patch("rfs.evaluation.direct_judges.call_vlm_json") as model:
                result = judge.evaluate_pairwise(DirectPairJudgeInput(before=_input(before), after=_input(Path(folder) / "after.png")))
            self.assertFalse(model.called)
            self.assertTrue(result["pairwise_unavailable"])
            self.assertEqual("", pairwise_verdict(result))

    def test_the_grounding_gate_does_not_refuse_a_pairwise_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            before, after = tmp / "before.png", tmp / "after.png"
            before.write_bytes(b"before")
            after.write_bytes(b"after")
            gate = {"status": "blocked", "blocking_issues": ["graph does not bind back to the paper"]}
            for frozen in (False, True):
                judge = BootstrapVLMDirectJudge(model="ordinary-vlm", frozen=frozen)
                with patch("rfs.evaluation.direct_judges.call_vlm_json", return_value={"preferred_candidate": "B"}) as model:
                    result = judge.evaluate_pairwise(DirectPairJudgeInput(before=_input(before), after=_input(after, gate=gate)))
                self.assertTrue(model.called)
                self.assertFalse(result["gate_only"])
                self.assertEqual("after", pairwise_verdict(result))
                # The gate's own finding still counts as a blocker of the state it looked at.
                self.assertIn("graph does not bind back to the paper", result["blocking_issues"])


class PairwiseMemoryTest(unittest.TestCase):
    def _runner(self, case: ValidationCase, enabled: bool, record: dict) -> dict:
        return {
            "overall_score": 0.4 if enabled else 0.9,
            "blocking_issue_count": 0,
            "paired_config_hash": "cfg",
            "initial_state_hash": case.case_id,
            "candidate_policy_hash": "candidates",
            "evaluator_hash": "judge",
            "run_dir": f"/runs/{case.case_id}/{'on' if enabled else 'off'}",
        }

    def _cases(self) -> list[ValidationCase]:
        return [ValidationCase("c1", "ai_method"), ValidationCase("c2", "ai_method"), ValidationCase("c3", "cv", "holdout")]

    def test_the_pair_is_decided_by_comparing_the_two_arms(self) -> None:
        seen: list[tuple[str, str]] = []

        def pairwise(off: dict, on: dict) -> dict:
            seen.append((off["run_dir"], on["run_dir"]))
            return _comparison("B")

        report = validate_experience_cases({"experience_id": "e1", "applicability": "common"}, self._cases(), self._runner, pairwise=pairwise, expected_groups=())
        self.assertEqual("pairwise", report["comparison"])
        # Each of the three pairs is put to the Judge three times, and the same pair each time:
        # one reading of a rounding is not a measurement of the figure (P5).
        self.assertEqual(9, len(seen))
        self.assertEqual(3, len(set(seen)))
        self.assertTrue(all(item["passed"] for item in report["results"]))
        # Every score moved the wrong way and none of them decided anything.
        self.assertTrue(all(item["target_delta"] < 0 for item in report["results"]))
        self.assertEqual("promoted", report["status"])

    def test_a_comparison_that_prefers_the_control_refuses_the_experience(self) -> None:
        report = validate_experience_cases({"experience_id": "e1", "applicability": "common"}, self._cases(), self._runner, pairwise=lambda off, on: _comparison("A"))
        self.assertEqual("overfit_or_failed", report["status"])
        self.assertFalse(report["all_cases_safe"])
        self.assertTrue(all(item["pairwise_verdict"] == "before" for item in report["results"]))

    def test_an_unanswered_comparison_leaves_the_pair_unmeasured(self) -> None:
        report = validate_experience_cases({"experience_id": "e1", "applicability": "common"}, self._cases(), self._runner, pairwise=lambda off, on: normalize_pairwise_result({}, role="frozen_bootstrap", candidate_a="off", candidate_b="on"))
        self.assertEqual("overfit_or_failed", report["status"])
        self.assertEqual("pair_not_measured", report["ladder"]["stopped_because"])
        self.assertFalse(report["all_cases_measured"])


if __name__ == "__main__":
    unittest.main()
