"""The first round of a run is scored against the figure the run was handed.

A continuous iteration is given the previous iteration directory, so its opening round is a
comparison against a real earlier figure instead of an unmeasured start. A run that was handed
nothing says so in the record rather than reading a score into an empty comparison.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."


def write_paper(root: Path) -> Path:
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Baseline Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


def _choice_answer(record: dict) -> str:
    """A scripted acceptance seat that always calls the first figure the wrong one."""

    return json.dumps({"wrong_figure": "A", "defect": "overlap"})


def _events(run: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run(root: Path, out: Path, **kwargs) -> None:
    with patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)):
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
            max_rounds=1,
            candidates_per_round=2,
            curated_standard="",
            **kwargs,
        )


class BaselineScoringTests(unittest.TestCase):
    def test_a_first_round_against_a_supplied_baseline_is_scored(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            baseline = root / "baseline"
            _run(root, baseline)
            out = root / "scored"
            _run(root, out, baseline_from=str(baseline), acceptance_generate=_choice_answer)

            names = [event["event"] for event in _events(out)]
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            selection = json.loads((out / "round_00" / "round_selection.json").read_text(encoding="utf-8"))

        self.assertIn("baseline_measured", names)
        self.assertIn("acceptance_compared", names)
        # The first round is measured rather than accepted for free.
        self.assertEqual("pairwise", rounds[0]["repair_decision"]["comparison"])
        self.assertIsNotNone(rounds[0]["acceptance_judgement"]["overall_score"])
        self.assertGreater(rounds[0]["baseline_score"], 0)

    def test_the_candidate_scores_the_tournament_wrote_are_recorded(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            baseline = root / "baseline"
            _run(root, baseline)
            out = root / "tournament"
            _run(root, out, baseline_from=str(baseline), acceptance_generate=_choice_answer)

            selection = json.loads((out / "round_00" / "round_selection.json").read_text(encoding="utf-8"))

        self.assertEqual("pairwise_tournament", selection["method"])
        measured = [entry["acceptance_score"] for entry in selection["candidates"]]
        self.assertTrue(all(score is not None for score in measured), measured)
        # The comparison's own numbers are looked up in the defect table, so they are
        # recorded under their own name and never as a measurement; the measurement every
        # candidate carries is the deterministic gate's reading, the same ruler for each.
        derived = [entry["derived_score"] for entry in selection["candidates"]]
        self.assertTrue(all(score is not None for score in derived), derived)
        self.assertGreater(len(set(derived)), 1)

    def test_a_run_without_a_baseline_records_it_instead_of_scoring(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            out = root / "nobaseline"
            _run(root, out, acceptance_generate=_choice_answer)

            events = _events(out)
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]

        unmeasured = [event for event in events if event["event"] == "baseline_unavailable"]
        self.assertTrue(unmeasured, [event["event"] for event in events])
        # The round is recorded as the run's opening candidate: it is not compared against a
        # state that does not exist, so it carries no delta -- a fabricated one is exactly
        # what this record exists to refuse.
        self.assertEqual("initial_candidate", rounds[0]["repair_decision"]["reason"])
        self.assertEqual("none", rounds[0]["repair_decision"]["comparison"])
        self.assertIsNone(rounds[0]["repair_decision"]["score_delta"])


if __name__ == "__main__":
    unittest.main()
