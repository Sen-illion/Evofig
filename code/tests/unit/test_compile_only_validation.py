"""A compile-time rule is validated by drawing one figure and compiling it twice."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import ValidationCase
from rfs.evolution.direct.validation import COMPILE_ONLY_PATH, PAIRED_RUN_PATH, validate_experience_cases


def _case(case_id: str, split: str = "same_domain", group: str = "same_domain_same_structure") -> ValidationCase:
    return ValidationCase(case_id, "ai_method", split, group)


def _arm(run_dir: str, *, program_hash: str = "shared", score: float | None = 0.5, actions=()) -> dict:
    return {
        "overall_score": score,
        "measured": score is not None,
        "initial_state_hash": "start",
        "run_dir": run_dir,
        "paired_config_hash": "config",
        "candidate_policy_hash": "policy",
        "evaluator_hash": "evaluator",
        "compiled_program_hash": program_hash,
        "action_sequence": list(actions),
        "blocking_issue_count": 0,
        "pair_stop_reason": "",
    }


def _choice(verdict: str):
    return lambda off, on: {"judgement_kind": "pairwise", "pairwise": {"choice": verdict}}


EXPERIENCE = {"experience_id": "e1", "apply_at": "ppt_compilation", "applicability": "common", "action": {"type": "layout_constraint"}}


class CompileOnlyProtocolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.generations: list[str] = []

    def _runner(self):
        def run(case, enabled, record):
            self.generations.append(f"{case.case_id}:{'on' if enabled else 'off'}")
            return _arm(str(self.root / case.case_id / ("on" if enabled else "off")))

        return run

    def _compile_only(self, *, program_hash: str = "shared"):
        def build(case, control, record):
            arm = _arm(str(self.root / case.case_id / "on"), program_hash=program_hash, score=None, actions=["layout_constraint"])
            arm["memory_enabled"] = True
            return arm

        return build

    def _cases(self):
        return [_case("a"), _case("b"), _case("c", split="holdout", group="cross_domain_same_structure")]

    def test_the_control_arm_is_the_only_one_generated(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(),
            pairwise=_choice("B"),
            compile_only=self._compile_only(),
            case_dir=self.root / "cases",
        )
        self.assertEqual(["a:off", "b:off", "c:off"], self.generations)
        self.assertEqual(3, report["summary"]["cases"])
        self.assertEqual(3, report["summary"]["measured_cases"])
        self.assertTrue(all(item["passed"] for item in report["results"]))

    def test_every_case_says_which_path_decided_it(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(),
            pairwise=_choice("B"),
            compile_only=self._compile_only(),
            case_dir=self.root / "cases",
        )
        self.assertEqual(COMPILE_ONLY_PATH, report["path"])
        self.assertTrue(all(item["path"] == COMPILE_ONLY_PATH for item in report["results"]))

    def test_the_two_arms_are_shown_to_be_the_same_figure_by_construction(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(),
            pairwise=_choice("B"),
            compile_only=self._compile_only(),
            case_dir=self.root / "cases",
        )
        for item in report["results"]:
            self.assertTrue(item["program_identity"]["checked"])
            self.assertTrue(item["program_identity"]["equal"])
            self.assertEqual("shared", item["program_identity"]["hash"])

    def test_a_pair_that_compiled_different_figures_is_not_evidence(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(),
            pairwise=_choice("B"),
            compile_only=self._compile_only(program_hash="different"),
            case_dir=self.root / "cases",
        )
        self.assertFalse(any(item["passed"] for item in report["results"]))
        self.assertTrue(all(item["program_identity"]["equal"] is False for item in report["results"]))

    def test_a_comparison_that_preferred_the_control_arm_is_not_an_improvement(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(),
            pairwise=_choice("A"),
            compile_only=self._compile_only(),
            case_dir=self.root / "cases",
        )
        self.assertTrue(all(item["measured"] for item in report["results"]))
        self.assertFalse(any(item["passed"] for item in report["results"]))

    def test_without_the_compile_only_path_both_arms_are_generated(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(),
            pairwise=_choice("B"),
            case_dir=self.root / "cases",
        )
        self.assertEqual(["a:off", "a:on", "b:off", "b:on", "c:off", "c:on"], self.generations)
        self.assertEqual(PAIRED_RUN_PATH, report["path"])
        self.assertEqual({}, report["results"][0].get("program_identity", {}))

    def test_the_control_arm_keeps_the_score_it_measured(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(),
            pairwise=_choice("B"),
            compile_only=self._compile_only(),
            case_dir=self.root / "cases",
        )
        first = report["results"][0]
        self.assertEqual(0.5, first["off"]["overall_score"])
        # The compiled arm carries no score of its own: nobody scored it, so the pair is
        # decided by the comparison and the record says so instead of inventing a number.
        self.assertIsNone(first["on"]["overall_score"])
        self.assertIsNone(first["target_delta"])


if __name__ == "__main__":
    unittest.main()
