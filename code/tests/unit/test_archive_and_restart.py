"""The archive keeps what a round did not publish, and a stalled loop starts from it.

A round draws a figure, keeps it or not, and moves on; the variant it did not keep used to be
gone with the rollout. These tests pin the other behaviour: every variant is recorded with what
it scored and why the round passed on it, the version the loop published is never offered back
as a starting point, and an iteration that keeps nothing is followed by one that starts from an
archived variant instead of from the same version again.

Two scores are recorded per variant and the restart reads the right one. The measured score is
relative -- a variant the comparator defaulted on carries the default -- so ordering on it
returned one old drawing for ever. The absolute score is the variant's own geometry-and-density
reading, and the restart is confined to the recent window before it orders on that.
"""

import json
import tempfile
import unittest
from pathlib import Path

from rfs.utils import write_json, write_text
from rfs.workflows.continuous_evolution import (
    CONTINUOUS_EVENTS,
    RESTART_WINDOW_ITERATIONS,
    repair_rounds,
    CONTINUOUS_STATE,
    absolute_score_for,
    accepted_rounds,
    archive_rows,
    read_continuous_config,
    restart_from_archive,
    run_continuous_evolution,
)


def _paper(name: str) -> dict:
    return {"paper": f"/papers/{name}.pdf", "paper_id": name, "domain": "biology"}


def _config(**overrides) -> dict:
    block = {"base_papers": [_paper("paper_one")], "budget_hours": 24.0, "iteration_rounds": 1}
    block.update(overrides)
    return {"continuous": block}


def _seed_iteration(out: Path, name: str, number: int, *, accepted: bool, scores=(0.4, 0.55)) -> Path:
    """One finished iteration with the two files the archive is read from.

    The program goes at the iteration's own root as well as in each candidate, because that is
    where a real run leaves it and a variant is only inherit-able when the directory it points at
    still holds a figure: a restart offered from an empty directory would be a round with nothing
    to look at and nothing to measure.
    """

    iter_dir = out / "papers" / name / f"iter_{number:02d}"
    candidates = []
    for index, score in enumerate(scores):
        candidate = iter_dir / "round_01" / f"candidate_{index:02d}"
        candidate.mkdir(parents=True, exist_ok=True)
        write_text(candidate / "figure_program.json", "{}")
        candidates.append({"candidate_index": index, "dir": str(candidate), "acceptance_score": score, "visual_score": 0.0})
    winner = len(candidates) - 1
    write_json(iter_dir / "round_01" / "round_selection.json", {"contract": "rfs.round_selection", "version": 1, "candidate_count": len(candidates), "candidates": candidates, "winner": winner, "winner_dir": candidates[winner]["dir"]})
    # Round 0 is the initial candidate every real run writes: it adopts the figure and was
    # compared against nothing. The round under test is the repair that follows it, because
    # whether a repair stuck -- not whether a starting point was adopted -- is what decides if
    # the iteration moved the published version.
    write_json(
        iter_dir / "evolution_rounds.json",
        {
            "rounds": [
                {"round": 0, "accepted": True, "repair_decision": {"accepted": True, "reason": "initial_candidate"}},
                {
                    "round": 1,
                    "accepted": accepted,
                    "repair_decision": {
                        "accepted": accepted,
                        "reason": "pairwise_preferred_without_regression" if accepted else "pairwise_not_preferred",
                    },
                },
            ]
        },
    )
    write_json(iter_dir / "evolution_state.json", {"best_candidate_dir": candidates[winner]["dir"], "best_score": scores[winner]})
    write_text(iter_dir / "final_draft_preview.png", "png")
    write_text(iter_dir / "figure_program.json", "{}")
    return iter_dir


class ArchiveRowTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def test_every_variant_of_every_round_is_recorded_with_its_score_and_reason(self):
        iter_dir = _seed_iteration(self.out, "paper_one", 2, accepted=True, scores=(0.4, 0.55))
        rows = archive_rows(iter_dir, 2)

        self.assertEqual([0, 1], [row["candidate"] for row in rows])
        self.assertEqual([2, 2], [row["iter"] for row in rows])
        self.assertEqual([1, 1], [row["round"] for row in rows])
        self.assertEqual([0.4, 0.55], [row["score"] for row in rows])
        self.assertEqual([False, True], [row["selected"] for row in rows])
        self.assertEqual("pairwise_preferred_without_regression", rows[1]["decision_reason"])
        self.assertIn("candidate 01", rows[0]["decision_reason"])
        for row in rows:
            self.assertEqual(str(iter_dir), row["source"])
            self.assertTrue(Path(row["dir"]).is_dir())

    def test_an_iteration_that_kept_nothing_is_read_as_moving_nothing(self):
        kept = _seed_iteration(self.out, "paper_one", 1, accepted=True)
        stalled = _seed_iteration(self.out, "paper_one", 2, accepted=False, scores=(0.3, 0.2))
        self.assertEqual(1, accepted_rounds(kept))
        self.assertEqual(0, accepted_rounds(stalled))

    def test_an_iteration_that_never_tried_to_repair_has_not_stalled(self):
        """Only a repair that did not stick is a stall; a run with nothing to repair is not."""

        iter_dir = self.out / "papers" / "paper_one" / "iter_04"
        iter_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            iter_dir / "evolution_rounds.json",
            {"rounds": [{"round": 0, "accepted": True, "repair_decision": {"accepted": True, "reason": "initial_candidate"}}]},
        )
        self.assertEqual(0, repair_rounds(iter_dir))
        self.assertEqual(0, accepted_rounds(iter_dir))

    def test_adopting_the_starting_figure_is_not_the_published_version_moving(self):
        """Round 0 is accepted by every run, so counting it would make the restart unreachable."""

        iter_dir = self.out / "papers" / "paper_one" / "iter_03"
        iter_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            iter_dir / "evolution_rounds.json",
            {
                "rounds": [
                    {"round": 0, "accepted": True, "repair_decision": {"accepted": True, "reason": "initial_candidate"}},
                    {"round": 1, "accepted": False, "repair_decision": {"accepted": False, "reason": "new_blocker"}},
                    {"round": 2, "accepted": False, "repair_decision": {"accepted": False, "reason": "pairwise_not_preferred"}},
                ]
            },
        )
        self.assertEqual(0, accepted_rounds(iter_dir))


class RestartChoiceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def test_the_published_version_is_never_the_thing_it_restarts_from(self):
        first = _seed_iteration(self.out, "paper_one", 1, accepted=False, scores=(0.4,))
        second = _seed_iteration(self.out, "paper_one", 2, accepted=False, scores=(0.6,))
        archive = archive_rows(first, 1) + archive_rows(second, 2)

        chosen = restart_from_archive(archive, exclude=[str(second)])

        self.assertEqual(str(first), chosen["source"])
        self.assertEqual(1, chosen["iter"])
        self.assertEqual(0.4, chosen["score"])
        # The version the loop published, and the one it already tried, are both off the menu.
        self.assertEqual(str(second), restart_from_archive(archive, exclude=[str(first)])["source"])
        self.assertEqual({}, restart_from_archive(archive, exclude=[str(first), str(second)]))

    def test_the_highest_score_wins_and_ties_go_to_the_more_recent_drawing(self):
        first = _seed_iteration(self.out, "paper_one", 1, accepted=False, scores=(0.9,))
        second = _seed_iteration(self.out, "paper_one", 2, accepted=False, scores=(0.5,))
        archive = archive_rows(first, 1) + archive_rows(second, 2)

        self.assertEqual(str(first), restart_from_archive(archive, exclude=[])["source"])

        tied = _seed_iteration(self.out, "paper_one", 3, accepted=False, scores=(0.9,))
        self.assertEqual(str(tied), restart_from_archive(archive_rows(first, 1) + archive_rows(tied, 3), exclude=[])["source"])

    def test_a_variant_whose_artifacts_are_gone_is_not_offered(self):
        iter_dir = _seed_iteration(self.out, "paper_one", 1, accepted=False, scores=(0.4,))
        archive = archive_rows(iter_dir, 1)
        for row in archive:
            row["source"] = str(self.out / "papers" / "paper_one" / "iter_99")

        self.assertEqual({}, restart_from_archive(archive, exclude=[]))


class StalledLoopTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def _runner(self, calls: list, *, accepted: bool):
        def run(**options):
            calls.append(dict(options))
            iter_dir = _seed_iteration(self.out, Path(options["out"]).parent.name, int(Path(options["out"]).name.split("_")[-1]), accepted=accepted)
            if options.get("reuse_upstream_from"):
                write_json(iter_dir / "stage_reuse_report.json", {"source": str(options["reuse_upstream_from"]), "fork_before": str(options.get("fork_before") or ""), "reused_stages": ["design_seed"]})
            return {"ok": True, "out_dir": str(iter_dir), "rounds_completed": 1}

        return run

    def _events(self) -> list[dict]:
        return [json.loads(line) for line in (self.out / CONTINUOUS_EVENTS).read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_a_loop_that_keeps_nothing_starts_the_next_round_from_an_archived_variant(self):
        calls: list[dict] = []
        # The review answers and asks for nothing: what this test is about is the restart, and a
        # review that did not answer forks somewhere else on purpose (see SilentReviewTests).
        report = run_continuous_evolution(_config(), self.out, evolve_runner=self._runner(calls, accepted=False), review_runner=lambda *_a: {"ok": True, "reason": "", "stage_feedback": []}, clock=lambda: 0.0, max_waves=3)

        restarts = [item for item in self._events() if item["event"] == "restart_from_archive"]
        self.assertEqual(1, len(restarts))
        self.assertEqual("restart_from_archive", restarts[0]["action"])
        self.assertEqual(1, restarts[0]["source_iteration"])
        self.assertTrue(restarts[0]["reuse_from"].endswith("iter_01"))

        third = next(item for item in calls if Path(item["out"]).name == "iter_03")
        self.assertTrue(third["reuse_upstream_from"].endswith("iter_01"))
        self.assertEqual("segmentation", third["fork_before"])
        paper = report["papers"][0]
        self.assertEqual(3, paper["iteration"])
        self.assertEqual(1, paper["iterations"][-1]["restart"]["iter"])
        self.assertGreaterEqual(paper["archive_size"], 3)
        self.assertIn("started from the archive", (self.out / "continuous_report.md").read_text(encoding="utf-8"))

    def test_it_never_restarts_twice_in_a_row(self):
        calls: list[dict] = []
        run_continuous_evolution(_config(), self.out, evolve_runner=self._runner(calls, accepted=False), review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []}, clock=lambda: 0.0, max_waves=4)

        restarts = [item for item in self._events() if item["event"] == "restart_from_archive"]
        self.assertEqual([3], [item["iteration"] for item in restarts])
        fourth = next(item for item in calls if Path(item["out"]).name == "iter_04")
        self.assertTrue(fourth["reuse_upstream_from"].endswith("iter_03"))

    def test_an_iteration_that_keeps_something_continues_from_the_published_line(self):
        calls: list[dict] = []
        run_continuous_evolution(_config(), self.out, evolve_runner=self._runner(calls, accepted=True), review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []}, clock=lambda: 0.0, max_waves=4)

        self.assertEqual([], [item for item in self._events() if item["event"] == "restart_from_archive"])
        third = next(item for item in calls if Path(item["out"]).name == "iter_03")
        self.assertTrue(third["reuse_upstream_from"].endswith("iter_02"))

    def test_a_restarted_loop_still_has_the_variants_the_previous_calls_archived(self):
        calls: list[dict] = []
        run_continuous_evolution(_config(), self.out, evolve_runner=self._runner(calls, accepted=False), review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []}, clock=lambda: 0.0, max_waves=2)
        state = json.loads((self.out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
        archived = state["papers"]["paper_one"]["archive"]
        self.assertTrue(archived)
        self.assertEqual(0, state["papers"]["paper_one"]["best_moved_at"])

        second_calls: list[dict] = []
        run_continuous_evolution(_config(), self.out, evolve_runner=self._runner(second_calls, accepted=False), review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []}, clock=lambda: 0.0, max_waves=1)

        resumed = json.loads((self.out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
        kept = resumed["papers"]["paper_one"]["archive"]
        self.assertGreaterEqual(len(kept), len(archived))
        keys = lambda rows: {(row["iter"], row["round"], row["candidate"]) for row in rows}
        self.assertTrue(keys(archived) <= keys(kept))
        third = next(item for item in second_calls if Path(item["out"]).name == "iter_03")
        self.assertTrue(third["reuse_upstream_from"].endswith("iter_01"))


class AbsoluteScoreTests(unittest.TestCase):
    """The restart column is the variant's own reading, not what the comparator defaulted to."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)

    def _read_reports_into(self, candidate: Path, *, crossings: int) -> None:
        write_json(
            candidate / "geometry_quality_report.json",
            {
                "status": "pass" if crossings == 0 else "fail",
                "icon_text_overlap_count": 0,
                "node_overlap_count": 0,
                "connector_label_overlap_count": 0,
                "text_pass_count": 0,
                "box_pass_count": crossings,
                "node_boundary_violation_count": 0,
            },
        )
        write_json(
            candidate / "density_quality_report.json",
            {
                "status": "pass",
                "canvas_occupancy": 1.0,
                "targets": {"minimum_occupancy": 1.0},
                "panel_count": 1,
                "sparse_panel_count": 0,
                "editable_text_area": 1.0,
            },
        )

    def test_a_variant_carries_its_own_reading_beside_the_comparison_score(self):
        iter_dir = _seed_iteration(self.out, "paper_one", 2, accepted=False, scores=(0.3, 0.9))
        self._read_reports_into(iter_dir / "round_01" / "candidate_00", crossings=4)
        self._read_reports_into(iter_dir / "round_01" / "candidate_01", crossings=0)

        rows = archive_rows(iter_dir, 2)

        self.assertEqual([0.3, 0.9], [row["choice_score"] for row in rows])
        # Four crossings put the geometry dimension at 1/(1+4); the other four stay at 1.0.
        self.assertEqual([0.84, 1.0], [row["absolute_score"] for row in rows])
        # The column the restart orders on follows the reading of the figure, not the comparison.
        self.assertEqual([0.84, 1.0], [row["score"] for row in rows])

    def test_a_variant_nothing_measured_keeps_the_comparison_score(self):
        iter_dir = _seed_iteration(self.out, "paper_one", 2, accepted=False, scores=(0.55,))

        row = archive_rows(iter_dir, 2)[0]

        self.assertIsNone(row["absolute_score"])
        self.assertEqual(0.55, row["choice_score"])
        self.assertEqual(0.55, row["score"])
        self.assertIsNone(absolute_score_for(iter_dir / "round_01" / "candidate_00"))

    def test_the_restart_prefers_the_variant_with_the_better_reading_of_its_own(self):
        first = _seed_iteration(self.out, "paper_one", 1, accepted=False, scores=(0.9,))
        second = _seed_iteration(self.out, "paper_one", 2, accepted=False, scores=(0.4,))
        self._read_reports_into(first / "round_01" / "candidate_00", crossings=6)
        self._read_reports_into(second / "round_01" / "candidate_00", crossings=0)

        chosen = restart_from_archive(archive_rows(first, 1) + archive_rows(second, 2), exclude=[])

        self.assertEqual(str(second), chosen["source"])
        self.assertEqual(1.0, chosen["absolute_score"])
        self.assertEqual(0.4, chosen["choice_score"])

    def test_the_restart_does_not_reach_past_the_recent_window(self):
        old = _seed_iteration(self.out, "paper_one", 1, accepted=False, scores=(0.1,))
        self._read_reports_into(old / "round_01" / "candidate_00", crossings=0)
        archive = archive_rows(old, 1)
        recent_iter = 1 + RESTART_WINDOW_ITERATIONS
        for number in range(2, recent_iter + 1):
            recent = _seed_iteration(self.out, "paper_one", number, accepted=False, scores=(0.2,))
            self._read_reports_into(recent / "round_01" / "candidate_00", crossings=8)
            archive += archive_rows(recent, number)

        chosen = restart_from_archive(archive, exclude=[])

        self.assertNotEqual(str(old), chosen["source"])
        self.assertEqual(recent_iter, chosen["iter"])
        self.assertEqual(recent_iter - RESTART_WINDOW_ITERATIONS + 1, chosen["window_start"])

    def test_the_restart_says_why_it_did_not_continue_from_the_published_line(self):
        iter_dir = _seed_iteration(self.out, "paper_one", 1, accepted=False, scores=(0.4,))
        self._read_reports_into(iter_dir / "round_01" / "candidate_00", crossings=0)

        chosen = restart_from_archive(archive_rows(iter_dir, 1), exclude=[])

        self.assertIn("kept nothing", chosen["restart_reason"])
        self.assertIn("geometry-and-density", chosen["restart_reason"])


if __name__ == "__main__":
    unittest.main()
