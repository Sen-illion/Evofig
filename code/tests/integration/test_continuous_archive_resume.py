"""A real iteration is archived, and a stalled loop restarts from it.

The unit tests drive the archive with seeded directories. This one lets the real offline pipeline
draw the first iteration, so the rows come from artifacts a run actually wrote, and then lets the
loop stall and insists the round after the stall inherits the directory the real run left.
"""

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from rfs.utils import write_json
from rfs.workflows.continuous_evolution import (
    CONTINUOUS_EVENTS,
    CONTINUOUS_MARKDOWN,
    CONTINUOUS_STATE,
    read_continuous_config,
    run_continuous_evolution,
)
from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable

PARAGRAPH = (
    "The proposed pipeline first reads the paper, then plans the panels, then compiles an "
    "editable figure. Each stage is validated before the next one starts, and every repair is "
    "recorded so that a later paper can recall it."
)

OFFLINE = {
    "planner_mode": "heuristic",
    "method_mode": "heuristic",
    "structure_mode": "heuristic",
    "text_plan_mode": "heuristic",
    "ppt_plan_mode": "heuristic",
    "color_theme_mode": "fallback",
    "icon_mode": "placeholder",
    "visual_review_mode": "heuristic",
    "judge_mode": "heuristic",
    "candidates_per_round": 1,
}


def _write_paper(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Continuous Archive Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 260), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 300, 550, 360), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(path)
    document.close()


NO_REVIEW = lambda *_a: {"ok": False, "reason": "", "stage_feedback": []}  # noqa: E731 - a review that asks for nothing


class ContinuousArchiveResumeIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _config(self) -> dict:
        return {"continuous": {"base_papers": [{"paper": str(self.root / "paper.pdf"), "paper_id": "one", "domain": "ai_method"}], "budget_hours": 24.0, "iteration_rounds": 1, "evolve": OFFLINE}}

    def _stalled_runner(self, calls: list):
        """An iteration that drew a figure, tried to repair it, kept nothing, and says so on disk.

        The ledger is the one a real run writes: round 0 adopts the figure it drew, and the
        repair rounds after it are the ones that rolled back. An iteration whose only round is
        the initial one has not stalled -- it never tried -- so it is not the state a restart
        from an archived variant is for.
        """

        def run(**options):
            calls.append(dict(options))
            iter_dir = Path(options["out"])
            number = int(iter_dir.name.split("_")[-1])
            candidates = []
            for index, score in enumerate((0.4, 0.35)):
                candidate = iter_dir / "round_00" / f"candidate_{index:02d}"
                candidate.mkdir(parents=True, exist_ok=True)
                write_json(candidate / "figure_program.json", {"panels": [], "semantic_nodes": [], "arrows": []})
                candidates.append({"candidate_index": index, "dir": str(candidate), "acceptance_score": score, "visual_score": 0.0})
            write_json(iter_dir / "round_00" / "round_selection.json", {"contract": "rfs.round_selection", "version": 1, "candidate_count": 2, "candidates": candidates, "winner": 0, "winner_dir": candidates[0]["dir"]})
            write_json(
                iter_dir / "evolution_rounds.json",
                {
                    "rounds": [
                        {"round": 0, "accepted": True, "repair_decision": {"accepted": True, "reason": "initial_candidate"}},
                        {"round": 1, "accepted": False, "repair_decision": {"accepted": False, "reason": "pairwise_not_preferred"}},
                    ]
                },
            )
            write_json(iter_dir / "evolution_state.json", {"best_candidate_dir": candidates[0]["dir"], "best_score": 0.4})
            if options.get("reuse_upstream_from"):
                write_json(iter_dir / "stage_reuse_report.json", {"source": str(options["reuse_upstream_from"]), "fork_before": str(options.get("fork_before") or ""), "reused_stages": ["design_seed"]})
            return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": 1}

        return run

    def test_the_first_iteration_is_archived_from_what_the_run_wrote(self):
        _write_paper(self.root / "paper.pdf")
        out = self.root / "out"

        def real_runner(**options):
            return run_paper_evolve_editable(**{**OFFLINE, **options})

        report = run_continuous_evolution(self._config(), out, evolve_runner=real_runner, review_runner=NO_REVIEW, clock=lambda: 0.0, max_waves=1)

        iter_dir = out / "papers" / "one" / "iter_01"
        archive = json.loads((iter_dir / "archive.json").read_text(encoding="utf-8"))
        selection = json.loads((iter_dir / "round_00" / "round_selection.json").read_text(encoding="utf-8"))

        self.assertTrue(archive["rows"])
        self.assertEqual([item["candidate_index"] for item in selection["candidates"]], [row["candidate"] for row in archive["rows"]])
        self.assertEqual(1, sum(1 for row in archive["rows"] if row["selected"]))
        for row in archive["rows"]:
            self.assertTrue(row["dir"] and Path(row["dir"]).is_dir(), row)
            self.assertEqual(str(iter_dir), row["source"])
        state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
        self.assertEqual(len(archive["rows"]), len(state["papers"]["one"]["archive"]))
        self.assertGreater(report["papers"][0]["archive_size"], 0)
        self.assertNotIn("restart_from_archive", (out / CONTINUOUS_MARKDOWN).read_text(encoding="utf-8"))

    def test_a_loop_that_stalls_after_a_real_iteration_restarts_from_its_archive(self):
        _write_paper(self.root / "paper.pdf")
        out = self.root / "out"

        def real_runner(**options):
            return run_paper_evolve_editable(**{**OFFLINE, **options})

        run_continuous_evolution(self._config(), out, evolve_runner=real_runner, review_runner=NO_REVIEW, clock=lambda: 0.0, max_waves=1)
        iter01 = out / "papers" / "one" / "iter_01"

        calls: list[dict] = []
        report = run_continuous_evolution(self._config(), out, evolve_runner=self._stalled_runner(calls), review_runner=NO_REVIEW, clock=lambda: 0.0, max_waves=2)

        events = [json.loads(line) for line in (out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines() if line.strip()]
        restarts = [item for item in events if item["event"] == "restart_from_archive"]
        self.assertEqual(1, len(restarts))
        self.assertTrue(restarts[0]["reuse_from"].endswith("iter_01"))
        self.assertEqual(str(iter01), restarts[0]["reuse_from"])
        self.assertEqual(1, restarts[0]["source_iteration"])

        # The round after the stall actually asked to inherit the archived directory.
        third = next(item for item in calls if Path(item["out"]).name == "iter_03")
        self.assertEqual(str(iter01), third["reuse_upstream_from"])
        self.assertTrue(Path(third["reuse_upstream_from"]).is_dir())
        self.assertTrue(report["papers"][0]["iterations"][-1]["restart"])
        self.assertIn("started from the archive", (out / "continuous_report.md").read_text(encoding="utf-8"))

        # And the archive still holds what the real run left, alongside the stalled iteration.
        state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
        rows = state["papers"]["one"]["archive"]
        self.assertTrue(any(row["iter"] == 1 and Path(row["dir"]).is_dir() for row in rows))
        self.assertTrue(any(row["iter"] == 2 for row in rows))


if __name__ == "__main__":
    unittest.main()
