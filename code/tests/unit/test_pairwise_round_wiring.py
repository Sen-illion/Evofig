"""The round loop asks the acceptance role to compare two states, and records the answer.

``compare_candidate`` is unit-tested on its own; this pins that the loop actually hands it a
comparison, that the comparison goes into the round record and the experience row, and that a
judge which cannot answer the question leaves the round on the measured-score path.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.workflows import paper_evolve_editable
from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."


def write_paper(root: Path) -> Path:
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Comparable Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


def _answer(prompt: str, images: list[str]) -> dict:
    """Answer whatever question the loop asked, the way each role's own contract asks for it."""

    # The advisory role is asked about a pair too, so "the first is A" no longer identifies this
    # question; the tail that names the second image as the state just produced does.
    if "the state it has just produced" in prompt:
        return {"preferred_candidate": "B", "margin": "clear", "confidence": 0.7, "blocking_issues": []}
    if "why_changed" in prompt or "reflection" in prompt.casefold():
        return {"outcome": "improved", "failure_type": "none", "why_changed": "the repair helped", "scope_assessment": "appropriate", "next_policy": "keep"}
    if "stage_feedback" in prompt:
        return {
            "status": "fault",
            "stage_feedback": [{
                "scope_type": "global",
                "problem": "the layout is too sparse to read as a method figure",
                "requested_actions": [{"action": "relayout_global", "factor": 1.03, "dx": 0.01, "dy": 0.0}],
            }],
            "blocking_issues": [],
        }
    return {"status": "pass", "overall_score": 0.6, "dimension_scores": {}, "blocking_issues": []}


def run_workflow(root: Path, *, judge_mode: str, pairwise: bool = True) -> Path:
    out = root / "run"
    calls: list[dict] = []

    def fake(prompt, images=None, **kwargs):
        calls.append({"prompt": str(prompt), "images": [str(item) for item in (images or [])]})
        return _answer(str(prompt), list(images or []))

    progress = patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None))
    vlm = patch("rfs.evaluation.direct_judges.call_vlm_json", side_effect=fake) if pairwise else contextlib.nullcontext()
    with progress, vlm:
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
            judge_mode=judge_mode,
            max_rounds=2,
            candidates_per_round=1,
            experience_enabled=False,
        )
    if pairwise:
        (root / "calls.json").write_text(json.dumps(calls), encoding="utf-8")
    return out


class PairwiseRoundWiringTests(unittest.TestCase):
    def test_the_second_round_decides_by_comparing_and_records_the_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            out = run_workflow(root, judge_mode="bootstrap")
            decision = json.loads((out / "round_01" / "repair_decision.json").read_text(encoding="utf-8"))
            self.assertEqual("pairwise", decision["comparison"])
            self.assertEqual("after", decision["pairwise_verdict"])
            self.assertTrue(decision["accepted"])
            self.assertEqual("pairwise_preferred", decision["reason"])
            self.assertEqual("B", decision["pairwise"]["preferred_candidate"])
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            self.assertIsNone(rounds[0]["acceptance_pairwise"])
            self.assertEqual("B", rounds[1]["acceptance_pairwise"]["pairwise"]["choice"])
            self.assertEqual("frozen_bootstrap", rounds[1]["acceptance_pairwise"]["judge_role"])
            row = [json.loads(line) for line in (out / "judge_experiences.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][1]
            self.assertEqual("B", row["acceptance_pairwise"]["pairwise"]["choice"])
            questions = [call for call in json.loads((root / "calls.json").read_text(encoding="utf-8")) if "the state it has just produced" in call["prompt"]]
            self.assertEqual(1, len(questions))
            # Both states were in the one call, so the model saw what it was comparing.
            self.assertEqual(2, len(questions[0]["images"]))

    def test_an_offline_judge_keeps_the_measured_score_path(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            out = run_workflow(root, judge_mode="heuristic", pairwise=False)
            decision = json.loads((out / "round_01" / "repair_decision.json").read_text(encoding="utf-8"))
            self.assertEqual("compared", decision["comparison"])
            self.assertNotIn("acceptance_pairwise", decision)


if __name__ == "__main__":
    unittest.main()
