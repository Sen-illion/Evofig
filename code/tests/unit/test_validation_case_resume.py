"""A case set that was interrupted continues where it stopped.

A validation set is hours of generation, and the loop restarts every hour or two, so a set that
could not be continued was never finished at all: 35 rows and 0 promotions was not a judgement on
the experiences, it was a measurement that kept losing its own work. These tests pin the
continuation -- the cases already decided are read back rather than paid for twice, and the verdict
is the same one an uninterrupted pass gives.
"""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import ValidationCase, validate_experience_cases


CASES = [
    ValidationCase("a", "ai_method", "same_domain", "same_domain_same_structure"),
    ValidationCase("b", "ai_method", "same_domain", "same_domain_same_structure"),
    ValidationCase("c", "system", "holdout", "cross_domain_same_structure"),
    ValidationCase("d", "ai_method", "boundary", "boundary"),
]

EXPERIENCE = {"experience_id": "e1", "applicability": "common"}


def _arm(case_id, enabled):
    delta = 0.0 if case_id == "d" else (0.03 if enabled else 0.0)
    return {
        "overall_score": 0.8 + delta,
        "blocking_issue_count": 0,
        "paired_config_hash": "cfg",
        "initial_state_hash": case_id,
        "candidate_policy_hash": "candidates",
        "evaluator_hash": "judge",
    }


def improving_runner(calls):
    """A deterministic stand-in: the memory arm gains 0.03, except on the boundary case."""

    def runner(case, enabled, experience):
        calls.append((case.case_id, enabled))
        return _arm(case.case_id, enabled)

    return runner


def crashing_runner(calls, *, decided_through):
    """The same stand-in, but the call dies the moment a later case starts."""

    def runner(case, enabled, experience):
        calls.append((case.case_id, enabled))
        if case.case_id != decided_through:
            raise RuntimeError("the call ended here")
        return _arm(case.case_id, enabled)

    return runner


class ResumableCaseSetTests(unittest.TestCase):
    def test_every_decided_case_is_written_beside_the_run(self):
        with tempfile.TemporaryDirectory() as temp:
            case_dir = Path(temp) / "cases"
            calls: list[tuple[str, bool]] = []
            result = validate_experience_cases(EXPERIENCE, CASES, improving_runner(calls), case_dir=case_dir)
            written = sorted(path.stem for path in case_dir.glob("*.json"))
            on_disk = json.loads((case_dir / "a.json").read_text(encoding="utf-8"))

        self.assertEqual(["a", "b", "c", "d"], written)
        self.assertEqual(4, result["summary"]["cases"])
        self.assertEqual(0, result["resumed_cases"])
        self.assertEqual("a", on_disk["case_id"])
        self.assertTrue(on_disk["measured"])

    def test_a_set_that_was_interrupted_runs_only_the_cases_it_still_owes(self):
        with tempfile.TemporaryDirectory() as temp:
            case_dir = Path(temp) / "cases"
            first_calls: list[tuple[str, bool]] = []
            with self.assertRaises(RuntimeError):
                validate_experience_cases(EXPERIENCE, CASES, crashing_runner(first_calls, decided_through="a"), case_dir=case_dir)
            decided_after_crash = sorted(path.stem for path in case_dir.glob("*.json"))

            second_calls: list[tuple[str, bool]] = []
            resumed = validate_experience_cases(EXPERIENCE, CASES, improving_runner(second_calls), case_dir=case_dir)

            clean_calls: list[tuple[str, bool]] = []
            clean = validate_experience_cases(EXPERIENCE, CASES, improving_runner(clean_calls))

        # The interruption landed on the second case, so the first one is the only one kept.
        self.assertEqual(["a"], decided_after_crash)
        self.assertEqual({"a", "b"}, {case_id for case_id, _ in first_calls})
        # The restart pays for the remaining cases only: never the one already on disk.
        self.assertEqual({"b", "c", "d"}, {case_id for case_id, _ in second_calls})
        self.assertEqual(1, resumed["resumed_cases"])
        self.assertEqual(4, resumed["summary"]["cases"])
        # A set walked over what the previous call kept plus what this one decided is the same
        # answer as a single uninterrupted pass -- that is what makes continuing safe.
        self.assertEqual(clean["status"], resumed["status"])
        self.assertEqual(clean["required_shape"], resumed["required_shape"])
        self.assertEqual(clean["summary"]["passed_cases"], resumed["summary"]["passed_cases"])
        self.assertEqual([item["case_id"] for item in clean["results"]], [item["case_id"] for item in resumed["results"]])

    def test_a_case_already_on_disk_is_not_put_to_the_judge_again(self):
        with tempfile.TemporaryDirectory() as temp:
            case_dir = Path(temp) / "cases"
            validate_experience_cases(EXPERIENCE, CASES, improving_runner([]), case_dir=case_dir)
            again: list[tuple[str, bool]] = []
            resumed = validate_experience_cases(EXPERIENCE, CASES, improving_runner(again), case_dir=case_dir)

        self.assertEqual([], again)
        self.assertEqual(4, resumed["resumed_cases"])
        self.assertEqual(4, resumed["summary"]["cases"])


if __name__ == "__main__":
    unittest.main()
