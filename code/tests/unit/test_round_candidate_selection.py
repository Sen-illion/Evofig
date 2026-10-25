"""One round may draw several figures; the candidates settle it against each other first."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.contracts.evolution import normalize_pairwise_result
from rfs.workflows.paper_evolve_editable import (
    CANDIDATE_ROUND_LIMIT,
    run_paper_evolve_editable,
    select_round_candidate,
)


def candidate(index, *, score, visual=0.0):
    return {
        "candidate_index": index,
        "variant": index,
        "dir": "/tmp/candidate_%02d" % index,
        "visual_score": visual,
        "acceptance": {"overall_score": score, "score_measured": score is not None} if score is not None else {},
    }


def prefers(choice):
    def compare(incumbent, challenger):
        return normalize_pairwise_result(
            {"preferred_candidate": choice},
            role="acceptance",
            candidate_a=str(incumbent["candidate_index"]),
            candidate_b=str(challenger["candidate_index"]),
        )

    return compare


class RoundCandidateSelectionTests(unittest.TestCase):
    def test_the_ceiling_is_four(self):
        self.assertEqual(CANDIDATE_ROUND_LIMIT, 4)

    def test_one_candidate_is_the_round_and_nothing_is_compared(self):
        calls = []

        def compare(incumbent, challenger):  # pragma: no cover - must never run
            calls.append((incumbent, challenger))
            return None

        winner, report = select_round_candidate([candidate(0, score=0.6)], compare=compare)
        self.assertEqual(winner["candidate_index"], 0)
        self.assertEqual(report["method"], "single_candidate")
        self.assertEqual(report["comparisons"], [])
        self.assertEqual(calls, [])

    def test_without_a_comparison_the_measured_order_decides(self):
        winner, report = select_round_candidate(
            [candidate(0, score=0.5), candidate(1, score=0.9), candidate(2, score=0.7)]
        )
        self.assertEqual(winner["candidate_index"], 1)
        self.assertEqual(report["method"], "measured_score_rank")
        self.assertEqual(report["winner_rank_by_score"], 0)

    def test_a_tournament_lets_the_role_pick_a_candidate_the_score_ranked_last(self):
        winner, report = select_round_candidate(
            [candidate(0, score=0.9), candidate(1, score=0.5), candidate(2, score=0.4)],
            compare=prefers("B"),
        )
        self.assertEqual(winner["candidate_index"], 2)
        self.assertEqual(report["method"], "pairwise_tournament")
        self.assertEqual([entry["verdict"] for entry in report["comparisons"]], ["after", "after"])
        self.assertEqual(report["winner_rank_by_score"], 2)

    def test_an_unreadable_comparison_keeps_the_incumbent(self):
        winner, report = select_round_candidate(
            [candidate(0, score=0.9), candidate(1, score=0.4)],
            compare=lambda incumbent, challenger: {},
        )
        self.assertEqual(winner["candidate_index"], 0)
        self.assertEqual([entry["verdict"] for entry in report["comparisons"]], [""])

    def test_a_tie_keeps_the_incumbent(self):
        winner, _ = select_round_candidate(
            [candidate(0, score=0.9), candidate(1, score=0.4)], compare=prefers("TIE")
        )
        self.assertEqual(winner["candidate_index"], 0)

    def test_the_comparison_scores_are_recorded_but_never_decide(self):
        """The choice model's numbers come from the defect table, so they are kept as derived.

        The comparison wrote them onto the candidate it blamed, and the same table row lands
        on whichever figure is blamed, so they are recorded beside the round and are not
        allowed to rank the candidates or to stand in for a measurement.
        """

        def with_scores(choice):
            def compare(incumbent, challenger):
                return {
                    **normalize_pairwise_result(
                        {"preferred_candidate": choice, "margin": "clear"},
                        role="acceptance",
                        candidate_a=str(incumbent["candidate_index"]),
                        candidate_b=str(challenger["candidate_index"]),
                    ),
                    "choice_scores": {
                        "before": {"overall_score": 0.1, "dimension_scores": {"scientific_fidelity": 0.1}, "score_source": "derived_from_defect"},
                        "after": {"overall_score": 0.95, "dimension_scores": {"scientific_fidelity": 0.95}, "score_source": "derived_from_defect"},
                    },
                }

            return compare

        winner, report = select_round_candidate(
            [candidate(0, score=0.4), candidate(1, score=None)], compare=with_scores("B")
        )
        # The preference decided the round.
        self.assertEqual(winner["candidate_index"], 1)
        records = report["candidates"]
        # The figure that carried a measurement keeps it; the other carries no measurement,
        # only the defect-table number the comparison wrote.
        self.assertEqual(records[0]["acceptance_score"], 0.4)
        self.assertIsNone(records[1]["acceptance_score"])
        self.assertIsNone(records[0]["derived_score"])
        self.assertEqual(records[1]["derived_score"], 0.95)
        # On the measurements, the figure the comparison preferred ranks below the one it did
        # not, and the record says so: the derived number sits off the scale.
        self.assertEqual(report["winner_rank_by_score"], 1)


def write_paper(root):
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Candidate Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), "We propose a frozen encoder that turns a paper into a symbolic figure plan.", fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


class RoundCandidateWiringTests(unittest.TestCase):
    def test_three_candidates_reach_the_round_and_the_choice_is_recorded(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            out = root / "run"
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
                    candidates_per_round=3,
                    curated_standard="",
                )
            round_dir = out / "round_00"
            selection = json.loads((round_dir / "round_selection.json").read_text(encoding="utf-8"))
            dirs = sorted(path.name for path in round_dir.glob("candidate_*") if path.is_dir())

        self.assertEqual(dirs, ["candidate_00", "candidate_01", "candidate_02"])
        self.assertEqual(selection["candidate_count"], 3)
        self.assertEqual(len(selection["candidates"]), 3)
        self.assertIn(selection["method"], {"pairwise_tournament", "measured_score_rank"})
        self.assertIsInstance(selection["winner"], int)
        self.assertTrue(Path(selection["winner_dir"]).name.startswith("candidate_"))


if __name__ == "__main__":
    unittest.main()
