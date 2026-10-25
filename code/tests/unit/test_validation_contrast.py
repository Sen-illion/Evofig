"""The contrast between two arms is read off what they produced, and it never spends a row.

The ruler used to be the sequence of action names the two arms ran. That read a memory which
changed the figure without changing the move's name as "no contrast", and it turned a pair the
Judge could very well decide into `inconclusive`. It is now a record -- the products the two arms
wrote and the deterministic readings of their figures -- and a pair with no contrast is reported
as an answer of "none" (which draws another set) rather than as a failure.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct import ValidationCase
from rfs.evolution.direct.validation import COMPILE_ONLY_PATH, PAIRED_RUN_PATH, validate_experience_cases
from rfs.workflows.direct_experience_validation import RESELECT_REASONS, _compile_only_reason


def _case(case_id: str, split: str = "same_domain", group: str = "same_domain_same_structure") -> ValidationCase:
    return ValidationCase(case_id, "ai_method", split, group)


def _arm(run_dir: str, *, products: dict | None = None, readings: dict | None = None, score: float | None = 0.5, **extra) -> dict:
    arm = {
        "overall_score": score,
        "measured": score is not None,
        "initial_state_hash": "start",
        "run_dir": run_dir,
        "paired_config_hash": "config",
        "candidate_policy_hash": "policy",
        "evaluator_hash": "evaluator",
        "compiled_program_hash": "program",
        "action_sequence": [],
        "blocking_issue_count": 0,
        "pair_stop_reason": "",
        "product_hashes": dict(products if products is not None else {"figure_program.json": "one"}),
        "readings": dict(readings if readings is not None else {"crossings": 3}),
    }
    arm.update(extra)
    return arm


def _choice(verdict: str):
    return lambda off, on: {"judgement_kind": "pairwise", "pairwise": {"choice": verdict}}


EXPERIENCE = {"experience_id": "e1", "apply_at": "ppt_compilation", "applicability": "common", "action": {"type": "layout_constraint"}}


class ContrastRulerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.generations: list[str] = []

    def _cases(self):
        return [_case("a"), _case("b"), _case("c", split="holdout", group="cross_domain_same_structure")]

    def _runner(self, *, off_products, on_products, off_readings=None, on_readings=None):
        def run(case, enabled, record):
            self.generations.append(f"{case.case_id}:{'on' if enabled else 'off'}")
            return _arm(
                str(self.root / case.case_id / ("on" if enabled else "off")),
                products=on_products if enabled else off_products,
                readings=on_readings if enabled else off_readings,
            )

        return run

    def test_two_arms_that_produced_different_products_is_contrast(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(off_products={"figure_program.json": "off"}, on_products={"figure_program.json": "on"}),
            pairwise=_choice("B"),
            case_dir=self.root / "cases",
        )
        self.assertTrue(all(item["products_diverged"] for item in report["results"]))

    def test_the_same_products_with_the_same_readings_is_no_contrast(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(off_products={"figure_program.json": "same"}, on_products={"figure_program.json": "same"}),
            pairwise=_choice("B"),
            case_dir=self.root / "cases",
        )
        self.assertTrue(all(item["products_diverged"] is False for item in report["results"]))

    def test_the_same_products_with_different_readings_is_contrast(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(
                off_products={"figure_program.json": "same"},
                on_products={"figure_program.json": "same"},
                off_readings={"crossings": 4},
                on_readings={"crossings": 1},
            ),
            pairwise=_choice("B"),
            case_dir=self.root / "cases",
        )
        self.assertTrue(all(item["products_diverged"] for item in report["results"]))

    def test_a_set_without_contrast_is_an_answer_of_none_not_a_refusal(self):
        """Every case put the same figure in front of the Judge, so the set answered nothing.

        That is recorded as its own outcome, and the reason is a reason to draw another set, rather
        than the row being written off as failed. The Judge's own answer is still on every case, so
        a reader can see what it said about a pair nobody made to differ.
        """

        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(off_products={"figure_program.json": "same"}, on_products={"figure_program.json": "same"}),
            pairwise=_choice("B"),
            case_dir=self.root / "cases",
        )
        self.assertEqual("inconclusive", report["status"])
        self.assertEqual("no_contrast_between_on_and_off", report["ladder"]["stopped_because"])
        self.assertNotIn(report["status"], {"overfit_or_failed", "refuted"})
        self.assertIn("no_contrast_between_on_and_off", RESELECT_REASONS, "the row is not spent; another set is drawn")
        self.assertTrue(all(item["pairwise_verdict"] == "after" for item in report["results"]))
        self.assertTrue(all(item["products_diverged"] is False for item in report["results"]))


class CompileOnlyEscalationTests(unittest.TestCase):
    """A rule the compiler could not honour is measured by the paired run instead of being spent."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.generations: list[str] = []

    def _cases(self):
        return [_case("a"), _case("b"), _case("c", split="holdout", group="cross_domain_same_structure")]

    def _runner(self):
        def run(case, enabled, record):
            self.generations.append(f"{case.case_id}:{'on' if enabled else 'off'}")
            return _arm(
                str(self.root / case.case_id / ("on" if enabled else "off")),
                products={"figure_program.json": f"{case.case_id}:{'on' if enabled else 'off'}"},
            )

        return run

    def _compile_only(self, *, products):
        def build(case, control, record):
            return _arm(
                str(self.root / case.case_id / "on"),
                products=products,
                score=None,
                memory_enabled=True,
                action_sequence=["layout_constraint"],
            )

        return build

    def test_a_rule_that_changed_nothing_is_escalated_to_the_paired_run(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(),
            pairwise=_choice("B"),
            compile_only=self._compile_only(products={"figure_program.json": "a:off"}),
            case_dir=self.root / "cases",
        )
        self.assertIn("a:on", self.generations)
        self.assertEqual(PAIRED_RUN_PATH, report["results"][0]["path"])
        self.assertEqual(COMPILE_ONLY_PATH, report["results"][0]["escalated_from"])

    def test_a_rule_that_changed_the_figure_keeps_the_cheap_pair(self):
        report = validate_experience_cases(
            EXPERIENCE,
            self._cases(),
            self._runner(),
            pairwise=_choice("B"),
            compile_only=self._compile_only(products={"figure_program.json": "changed"}),
            case_dir=self.root / "cases",
        )
        self.assertNotIn("a:on", self.generations)
        self.assertEqual(COMPILE_ONLY_PATH, report["results"][0]["path"])
        self.assertEqual("", report["results"][0]["escalated_from"])


class CompileOnlyRuleTests(unittest.TestCase):
    """A routing rule with parameters is exactly the class the cheap path exists for."""

    def test_a_router_rule_is_measured_by_compiling_one_figure_twice(self):
        router = {"experience_id": "e1", "action": {"type": "router_constraint", "shared_vertical_channel": True}}
        self.assertEqual("", _compile_only_reason(router))

    def test_a_row_with_no_parameters_is_refused_by_the_cheap_path(self):
        bare = {"experience_id": "e1", "action": {"type": "router_constraint"}}
        self.assertTrue(_compile_only_reason(bare))

    def test_a_row_that_plans_its_own_figure_is_not_a_compile_time_rule(self):
        planning = {"experience_id": "e1", "action": {"type": "resegment"}}
        self.assertTrue(_compile_only_reason(planning))


if __name__ == "__main__":
    unittest.main()
