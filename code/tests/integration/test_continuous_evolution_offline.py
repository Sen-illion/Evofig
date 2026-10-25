"""The continuous loop against the real workflow, offline: the fork stage decides what is inherited.

Everything here runs with deterministic modes, so it costs no API call and no reference. What it
proves is the wiring the plan asked for, end to end: the loop hands the real run a reuse source and
a fork stage, the run accepts them, and the amount of the previous iteration it re-plans follows the
review's own words -- a theme fork keeps the design, a semantics fork re-plans it.

The offline structure planner draws nothing and the run has no reference library, so between waves
the previous iteration's design seed and reference profiles are replaced with the artifacts a
model-assisted run would have produced. Both are unrelated preconditions of the loader; the point
of the test is what the *next* iteration does with the source once the loader accepts it.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from rfs.utils import write_json

from rfs.workflows.continuous_evolution import run_continuous_evolution

PARAGRAPH = (
    "The proposed pipeline first reads the paper, then plans the panels, then compiles an "
    "editable figure. Each stage is validated before the next one starts, and every repair is "
    "recorded so that a later paper can recall it."
)

DESIGN = {
    "canvas": {"aspect_ratio": "16:9", "margin": 0.02},
    "topology": "unknown",
    "panels": [
        {"id": "stage_one", "title": "Phase One", "role": "stage", "bbox_percent": {"x": 0.04, "y": 0.15, "w": 0.3, "h": 0.75}},
        {"id": "stage_two", "title": "Phase Two", "role": "stage", "bbox_percent": {"x": 0.38, "y": 0.15, "w": 0.3, "h": 0.75}},
    ],
    "nodes": [
        {"id": "node_alpha", "label": "Alpha", "semantic_source_id": "obj_alpha", "panel_id": "stage_one", "bbox_percent": {"x": 0.08, "y": 0.3, "w": 0.12, "h": 0.2}},
        {"id": "node_beta", "label": "Beta", "semantic_source_id": "obj_beta", "panel_id": "stage_two", "bbox_percent": {"x": 0.42, "y": 0.3, "w": 0.12, "h": 0.2}},
    ],
    "connectors": [],
}

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
    page.insert_text((50, 60), "A Continuous Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 260), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 300, 550, 360), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(path)
    document.close()


def _config(paper: Path) -> dict:
    return {
        "continuous": {
            "base_papers": [{"paper": str(paper), "paper_id": "paper_one", "domain": "biology"}],
            "iteration_rounds": 1,
            "parallel": 1,
            "budget_hours": 0,
            "evolve": dict(OFFLINE),
        }
    }


class ContinuousOfflineTest(unittest.TestCase):
    def _wave(self, out: Path, paper: Path, review_actions) -> dict:
        def review(_previous: Path, _spec: dict) -> dict:
            return {"ok": True, "reason": "", "stage_feedback": [{"requested_actions": list(review_actions)}]}

        with patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)):
            return run_continuous_evolution(_config(paper), out, evolve_runner=None, review_runner=review, clock=lambda: 0.0, max_waves=1)

    def _make_inheritable(self, iteration: Path) -> None:
        """Give an offline iteration the two artifacts the replay loader insists on."""

        write_json(iteration / "design_plan_raw.json", DESIGN)
        for name in ("reference_style_profile.json", "reference_layout_profile.json"):
            write_json(iteration / name, {"status": "complete"})

    def test_the_fork_stage_the_review_names_is_what_the_next_iteration_inherits(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = root / "paper.pdf"
            _write_paper(paper)
            out = root / "continuous"
            papers = out / "papers" / "paper_one"

            self.assertEqual(self._wave(out, paper, [])["stop_reason"], "wave_limit")
            first = papers / "iter_01"
            self.assertTrue((first / "figure_program.json").is_file(), "the first iteration compiled nothing")
            self._make_inheritable(first)

            self._wave(out, paper, ["replan_theme"])
            second = papers / "iter_02"
            theme = json.loads((second / "stage_reuse_report.json").read_text(encoding="utf-8"))
            self.assertFalse(theme.get("declined_reason"), theme.get("declined_reason"))
            self.assertEqual(Path(theme["source"]).name, "iter_01")
            self.assertEqual(theme["fork_before"], "theme_planning")
            self.assertIn("design_seed", theme["reused_stages"], "a theme fork must inherit the design")
            self.assertTrue(theme["layout_frozen"], "a theme fork must keep the frozen layout")
            self.assertNotIn("color_theme", theme["reused_stages"], "a theme fork must re-plan the theme")
            self.assertNotIn("icons", theme["reused_stages"], "a theme fork must re-plan the icons")
            program = json.loads((second / "figure_program.json").read_text(encoding="utf-8"))
            self.assertTrue(program["semantic_nodes"], "the second iteration compiled no figure")

            self._make_inheritable(second)
            self._wave(out, paper, ["reanalyze_semantics"])
            third = papers / "iter_03"
            semantics = json.loads((third / "stage_reuse_report.json").read_text(encoding="utf-8"))
            self.assertFalse(semantics.get("declined_reason"), semantics.get("declined_reason"))
            self.assertEqual(Path(semantics["source"]).name, "iter_02")
            self.assertEqual(semantics["fork_before"], "global_planning")
            self.assertNotIn("design_seed", semantics["reused_stages"], "a semantics fork must re-plan the design")
            self.assertFalse(semantics["layout_frozen"], "a semantics fork must not keep the frozen layout")

            events = [json.loads(line) for line in (out / "continuous_events.jsonl").read_text(encoding="utf-8").splitlines()]
            started = {item["iteration"]: item for item in events if item["event"] == "iteration_started"}
            self.assertEqual(started[2]["fork_before"], "theme_planning")
            self.assertTrue(started[2]["reuse_from"].endswith("iter_01"))
            self.assertEqual(started[3]["fork_before"], "global_planning")
            self.assertTrue(started[3]["reuse_from"].endswith("iter_02"))

    def test_a_review_that_named_a_bounded_repair_becomes_the_first_rounds_move(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = root / "paper.pdf"
            _write_paper(paper)
            out = root / "continuous"
            papers = out / "papers" / "paper_one"

            self.assertEqual(self._wave(out, paper, [])["stop_reason"], "wave_limit")
            first = papers / "iter_01"
            self._make_inheritable(first)

            # A repair that acts on the figure the run already has, so the fork it decides is
            # the one where nothing upstream of the compile is planned again.
            repair = {
                "type": "resize_nodes",
                "scope_type": "slot",
                "slot_ids": ["module_01"],
                "factor": 1.08,
                "dx": 0.0,
                "dy": 0.0,
                "problem": "one box is too small to read",
            }
            self._wave(out, paper, [repair])
            second = papers / "iter_02"
            reuse = json.loads((second / "stage_reuse_report.json").read_text(encoding="utf-8"))
            moves = json.loads((second / "round_00" / "round_moves.json").read_text(encoding="utf-8"))
            cost = json.loads((second / "round_00" / "round_cost.json").read_text(encoding="utf-8"))
            executed = json.loads((second / "round_00" / "action_execution.json").read_text(encoding="utf-8"))
            events = [json.loads(line) for line in (out / "continuous_events.jsonl").read_text(encoding="utf-8").splitlines()]

        # Everything the earlier iteration drew is inherited, so the figure the round acts on is
        # the very figure the review was written about.
        self.assertEqual(reuse["fork_before"], "ppt_compilation")
        self.assertIn("design_seed", reuse["reused_stages"])
        self.assertIn("icons", reuse["reused_stages"])
        self.assertNotIn("design_seed", reuse["replanned_stages"])
        # And the change that review named is what round 0 runs, instead of planning it all again.
        self.assertEqual(1, moves["count"])
        self.assertEqual("resize_nodes", moves["moves"][0]["type"])
        self.assertEqual("iteration_review", moves["moves"][0]["source"])
        self.assertTrue(executed["applied"], executed)
        self.assertEqual(["module_01"], executed["changed_node_ids"])
        self.assertEqual(0, cost["figure_generations"], "the opening move costs no figure generation")
        self.assertEqual(0, cost["stage_reruns"], "the opening move re-runs no stage")
        # The round also carries its own model-call ledger, so "where did the minutes go" is
        # answerable from the round. Offline there is nothing to bill and that is written too.
        self.assertEqual([], cost["model_calls"])
        self.assertEqual(0, cost["model_seconds"])
        started = {item["iteration"]: item for item in events if item["event"] == "iteration_started"}
        self.assertEqual(started[2]["first_round_action"]["type"], "resize_nodes")

    def test_a_restart_after_a_wave_continues_instead_of_repeating_it(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = root / "paper.pdf"
            _write_paper(paper)
            out = root / "continuous"

            self._wave(out, paper, [])
            report = self._wave(out, paper, [])

            self.assertEqual(report["papers"][0]["iteration"], 2)
            self.assertEqual(report["waves"], 2)
            self.assertEqual(report["waves_this_call"], 1)
            self.assertTrue((out / "papers" / "paper_one" / "iter_02" / "figure_program.json").is_file())

    def test_the_one_off_reread_reads_the_paper_again_and_is_spent_once(self):
        """A switch turned on at the reading has to be visible in the reading.

        Inheritance keeps the previous iteration's method contract, so a run that continues from
        the last version re-plans the boxes and leaves the reading untouched: the switch would
        look like a change that had no effect. The iteration the one-off re-read lands on inherits
        nothing upstream, and the comparison against the last published figure is kept, because
        that is a measurement rather than an inheritance.
        """

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = root / "paper.pdf"
            _write_paper(paper)
            out = root / "continuous"
            papers = out / "papers" / "paper_one"

            def review(_previous: Path, _spec: dict) -> dict:
                return {"ok": True, "reason": "", "stage_feedback": [{"requested_actions": ["replan_theme"]}]}

            def wave(config: dict) -> dict:
                with patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)):
                    return run_continuous_evolution(config, out, evolve_runner=None, review_runner=review, clock=lambda: 0.0, max_waves=1)

            config = _config(paper)
            wave(config)
            self._make_inheritable(papers / "iter_01")

            config["continuous"]["reread_once"] = True
            wave(config)
            second = papers / "iter_02"
            reuse = json.loads((second / "stage_reuse_report.json").read_text(encoding="utf-8"))
            # Nothing was inherited -- no source, no reused stage -- so the method contract this
            # figure was drawn from is the one this iteration read for itself.
            self.assertEqual("", str(reuse.get("source") or ""))
            self.assertEqual([], list(reuse.get("reused_stages") or []))
            events = [json.loads(line) for line in (out / "continuous_events.jsonl").read_text(encoding="utf-8").splitlines()]
            forced = [item for item in events if item["event"] == "reread_forced"]
            self.assertEqual(1, len(forced), "the switch is spent once")
            self.assertEqual("iter_01", Path(forced[0]["baseline_from"]).name, "the comparison against the last published figure is kept")
            self.assertEqual("theme_planning", forced[0]["would_have_forked_at"])
            started = {item["iteration"]: item for item in events if item["event"] == "iteration_started"}
            self.assertEqual("global_planning", started[2]["fork_before"])
            self.assertEqual("", started[2]["reuse_from"])

            # And it is spent: the next iteration inherits the version this one published.
            self._make_inheritable(second)
            wave(config)
            third = papers / "iter_03"
            reuse = json.loads((third / "stage_reuse_report.json").read_text(encoding="utf-8"))
            self.assertEqual("iter_02", Path(reuse["source"]).name)
            self.assertEqual("theme_planning", reuse["fork_before"])
            events = [json.loads(line) for line in (out / "continuous_events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(1, len([item for item in events if item["event"] == "reread_forced"]), "the switch must not fire twice")

    def test_a_source_with_no_reference_at_all_still_lends_its_design(self):
        """The all-or-nothing loader refused this source; per-stage inheritance keeps the design.

        The offline run has no reference library, so its reference analysis is ``not_found``. That
        one gap used to void the whole replay; it is now carried over as the same absence and the
        design seed is still inherited, which is the difference between "continue from the last
        version" and "draw it again from the paper".
        """

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = root / "paper.pdf"
            _write_paper(paper)
            out = root / "continuous"
            papers = out / "papers" / "paper_one"

            self._wave(out, paper, [])
            first = papers / "iter_01"
            write_json(first / "design_plan_raw.json", DESIGN)  # a design worth inheriting; no reference profiles

            self._wave(out, paper, ["replan_theme"])
            second = papers / "iter_02"
            reuse = json.loads((second / "stage_reuse_report.json").read_text(encoding="utf-8"))
            self.assertFalse(reuse.get("declined_reason"), reuse.get("declined_reason"))
            self.assertIn("design_seed", reuse["reused_stages"], "a theme fork must inherit the design")
            self.assertIn("reference_analysis", reuse["reused_stages"])
            self.assertNotIn("reference_analysis", reuse["replanned_stages"])
            self.assertTrue(reuse["layout_frozen"])
            gaps = {item["stage"]: item["reason"] for item in reuse["gaps"]}
            self.assertNotIn("reference_analysis", gaps)
            program = json.loads((second / "figure_program.json").read_text(encoding="utf-8"))
            self.assertTrue(program["semantic_nodes"], "the second iteration compiled no figure")

    def test_the_first_wave_says_what_the_next_iteration_will_inherit(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = root / "paper.pdf"
            _write_paper(paper)
            out = root / "continuous"

            self._wave(out, paper, [])
            write_json(out / "papers" / "paper_one" / "iter_01" / "design_plan_raw.json", DESIGN)

            report = self._wave(out, paper, ["replan_theme"])

            forecast = report["papers"][0]["last_reuse_forecast"]
            self.assertTrue(forecast["requested"])
            self.assertIn("design_seed", forecast["reused_stages"])
            self.assertIn("reference_analysis", forecast["reused_stages"])
            events = [json.loads(line) for line in (out / "continuous_events.jsonl").read_text(encoding="utf-8").splitlines()]
            started = {item["iteration"]: item for item in events if item["event"] == "iteration_started"}
            self.assertTrue(started[2]["reuse_forecast"]["accepted"])
            markdown = (out / "continuous_report.md").read_text(encoding="utf-8")
            self.assertIn("will inherit", markdown)

    def test_a_source_the_loader_refuses_is_reported_as_declined(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            paper = root / "paper.pdf"
            _write_paper(paper)
            out = root / "continuous"
            papers = out / "papers" / "paper_one"

            self._wave(out, paper, [])
            # The source claims a different paper, which is the one thing a replay cannot accept.
            write_json(papers / "iter_01" / "document_model.json", {"source_sha256": "0" * 64})

            report = self._wave(out, paper, [])

            self.assertEqual(report["papers"][0]["reuse_declined"], [2])
            self.assertFalse(report["papers"][0]["iterations"][1]["reuse"]["accepted"])
            self.assertFalse(report["papers"][0]["iterations"][1]["reuse_forecast"]["accepted"])
            markdown = (out / "continuous_report.md").read_text(encoding="utf-8")
            self.assertIn("re-planned everything", markdown)
            self.assertIn("cannot continue from the previous version", markdown)


if __name__ == "__main__":
    unittest.main()
