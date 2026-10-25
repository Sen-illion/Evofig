"""End-to-end wiring of the two Judge roles into the evolution loop.

The loop asks two questions about every candidate: the advisory role names the defect and the
bounded repair, the acceptance role measures the figure. These tests pin down what that split
has to do through the real round loop -- each candidate is measured once by the scoring role,
the advisory answer never votes on acceptance, a gate verdict is never mistaken for a baseline,
and the row written to ``judge_experiences.jsonl`` describes the candidate the round chose.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.contracts.evolution import normalize_judge_result
from rfs.workflows import paper_evolve_editable
from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."


def write_paper(root: Path) -> Path:
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Measured Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


class _ScriptedJudge:
    """One role's real interface, answering from a script instead of from a model."""

    def __init__(self, role: str, answers: list[dict]):
        self.role = role
        self.answers = [dict(item) for item in answers]
        self.calls: list[object] = []

    def _answer(self) -> dict:
        item = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        answer = {
            "contract": "rfs.judge_result",
            "judge_role": self.role,
            "status": "fault",
            "overall_score": 0.5,
            "score_measured": True,
            "dimension_scores": {},
            "stage_feedback": [],
            "blocking_issues": [],
            "blocking_issue_count": 0,
            "confidence": 0.5,
        }
        answer.update(item)
        return answer

    def evaluate(self, judge_input) -> dict:
        self.calls.append(judge_input)
        # The real adapters run every answer through the contract, so this one does too: what the
        # loop sees is the normalized answer of one role, never the raw model output.
        return normalize_judge_result(self._answer(), role=self.role)

    def reflect(self, outcome) -> dict:
        return {"outcome": "inconclusive", "next_policy": "keep"}


def _advisory(score: float) -> dict:
    """One advisory answer: a defect with a bounded repair, and a score nobody asked for.

    The repair has to be one that moves the figure. A reroute of this heuristic layout clears
    two connector paths and the router re-derives exactly the same ones, so the round is
    refused as ``figure_unchanged`` before any baseline is read -- and the baseline is what
    these tests are about.
    """

    return {
        "overall_score": score,
        "stage_feedback": [{
            "scope_type": "global",
            "problem": "the panels crowd each other and the figure does not read",
            "evidence": ["n1"],
            "requested_actions": [{"action": "relayout_global", "factor": 1.05, "dx": 0.02, "dy": 0.0}],
            "severity": "major",
        }],
    }


def run_workflow(root: Path, library: Path, *, scripted: tuple[_ScriptedJudge, _ScriptedJudge], max_rounds: int = 2) -> Path:
    out = root / "run"

    with patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)), patch(
        "rfs.workflows.paper_evolve_editable.build_direct_judges", lambda **kwargs: scripted
    ):
        run_paper_evolve_editable(
            paper=root / "paper.pdf",
            out=out,
            planner_mode="heuristic",
            method_mode="heuristic",
            structure_mode="heuristic",
            text_plan_mode="heuristic",
            ppt_plan_mode="heuristic",
            color_theme_mode="fallback",
            icon_mode="placeholder",
            visual_review_mode="heuristic",
            judge_mode="heuristic",
            max_rounds=max_rounds,
            candidates_per_round=2,
            experience_enabled=True,
            experience_library=library,
            fixed_rounds=True,
        )
    return out


class JudgeRoleSplitWiringTests(unittest.TestCase):
    def test_each_candidate_is_measured_once_and_only_a_measured_one_can_win(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            advisory = _ScriptedJudge("online_bootstrap", [_advisory(0.95), _advisory(0.05), _advisory(0.5), _advisory(0.5)])
            acceptance = _ScriptedJudge("frozen_bootstrap", [
                {"overall_score": 0.2}, {"overall_score": 0.95},
                {"overall_score": 0.5}, {"overall_score": 0.5},
            ])
            out = run_workflow(root, root / "library", scripted=(advisory, acceptance))
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]

            self.assertEqual(4, len(acceptance.calls))
            # The advisory answer scores the first candidate high and the second one low; the
            # measurement says the opposite, and the measurement is what selects.
            self.assertEqual(1, rounds[0]["selected_variant"])
            self.assertEqual(0.95, rounds[0]["acceptance_judgement"]["overall_score"])
            self.assertIsNone(rounds[0]["online_judgement"]["overall_score"])
            # The number the advisory answer volunteered is kept as a reported value and decides
            # nothing: the round was decided by the 0.95 measurement, not by the 0.05 opinion.
            self.assertEqual(0.05, rounds[0]["online_judgement"]["reported_score"])
            row = json.loads((out / "judge_experiences.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(rounds[0]["selected_variant"], row["candidate_index"])
            self.assertEqual(rounds[0]["acceptance_judgement"]["overall_score"], row["acceptance"]["overall_score"])
            self.assertIsNone(row["judgement"]["overall_score"])

    def test_a_gate_score_is_remeasured_before_it_becomes_a_baseline(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            advisory = _ScriptedJudge("online_bootstrap", [_advisory(0.5)] * 8)
            acceptance = _ScriptedJudge("frozen_bootstrap", [
                {"overall_score": 0.75, "gate_only": True},
                {"overall_score": 0.75, "gate_only": True},
                {"overall_score": 0.55},
                {"overall_score": 0.6}, {"overall_score": 0.6},
            ])
            out = run_workflow(root, root / "library", scripted=(advisory, acceptance))
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            events = [json.loads(line) for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual("remeasured_after_gate_only", rounds[0]["baseline_source"])
            self.assertEqual(0.55, rounds[0]["baseline_score"])
            remeasured = next(item for item in events if item["event"] == "baseline_remeasured")
            self.assertEqual(0.75, remeasured["gate_score"])
            self.assertEqual(0.55, remeasured["score"])
            # The next round is compared against the remeasured baseline, not against the gate's
            # number: 0.6 over 0.55 is an improvement, 0.6 against 0.75 would not have been.
            self.assertEqual(0.55, rounds[1]["repair_decision"]["before_score"])
            self.assertTrue(rounds[1]["accepted"])

    def test_an_unmeasured_round_cannot_become_a_baseline(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            unmeasured = {"overall_score": None, "score_measured": False, "status": "unmeasured"}
            advisory = _ScriptedJudge("online_bootstrap", [_advisory(0.5)] * 8)
            acceptance = _ScriptedJudge("frozen_bootstrap", [dict(unmeasured)])
            out = run_workflow(root, root / "library", scripted=(advisory, acceptance))
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            state = json.loads((out / "evolution_state.json").read_text(encoding="utf-8"))

            self.assertIsNone(rounds[0]["baseline_score"])
            self.assertFalse(rounds[0]["baseline_ready"])
            # Nothing was measured, so the next round is refused with the reason that names the
            # missing baseline instead of being read as a regression against a placeholder zero.
            self.assertFalse(rounds[1]["accepted"])
            self.assertEqual("baseline_unavailable", rounds[1]["repair_decision"]["reason"])
            self.assertIsNone(rounds[1]["repair_decision"]["score_delta"])
            self.assertIsNone(state["best_score"])


if __name__ == "__main__":
    unittest.main()
