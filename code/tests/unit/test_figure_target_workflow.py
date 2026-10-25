"""A run aimed at the paper's own figure records the target and how much it hit.

The product draws its own overview of a method. A public benchmark scores the figure
the paper printed, down to its panel granularity, so the benchmark row is aimed at
that figure instead. Which figure was aimed at, and how much of it the delivered
figure carries, is written down; it is never a gate.

The same file covers the product parser's second paired-run switch, ``--fork-before``: a
controlled experiment has to say which stage it plans for itself, and a stage outside the
pipeline is refused at parse time rather than discovered mid-run.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.cli import build_parser
from rfs.workflows.stage_replay import READING_FORK_STAGES, STAGE_ORDER
from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."


def write_paper(root: Path) -> Path:
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Figure Target Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Method", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    document.save(target)
    document.close()
    return target


def run_workflow(root: Path, *, figure_target: str):
    out = root / figure_target
    with patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)):
        report = run_paper_evolve_editable(
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
            candidates_per_round=1,
            experience_enabled=False,
            figure_target=figure_target,
        )
    return out, report


class FigureTargetWorkflowTests(unittest.TestCase):
    def test_the_overview_run_says_it_was_not_aimed_at_the_paper(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            write_paper(root)
            out, report = run_workflow(root, figure_target="overview")

            target_report = json.loads((out / "figure_target_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["figure_target"], "overview")
            self.assertEqual(target_report["figure_target"], "overview")
            self.assertFalse(target_report["panel_target"]["available"])
            self.assertEqual(target_report["panel_target"]["reason"], "figure_target_overview")
            self.assertFalse(target_report["coverage"]["enabled"])
            self.assertFalse(report["panel_target"]["available"])

    def test_a_paper_run_records_the_target_it_could_not_find(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            write_paper(root)
            out, report = run_workflow(root, figure_target="paper")

            target_report = json.loads((out / "figure_target_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["figure_target"], "paper")
            self.assertEqual(target_report["figure_target"], "paper")
            # The synthetic paper declares no panel structure, so the run falls back
            # to the overview path and says why, rather than refusing to deliver.
            self.assertFalse(target_report["panel_target"]["available"])
            self.assertEqual(target_report["panel_target"]["reason"], "no_panel_graphs")
            self.assertFalse(target_report["coverage"]["enabled"])

    def test_the_two_targets_write_the_same_artifact_set(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            write_paper(root)
            overview_out, _ = run_workflow(root, figure_target="overview")
            paper_out, _ = run_workflow(root, figure_target="paper")

            overview_files = {path.name for path in overview_out.iterdir() if path.is_file()}
            paper_files = {path.name for path in paper_out.iterdir() if path.is_file()}
            self.assertIn("figure_target_report.json", overview_files)
            self.assertIn("figure_target_report.json", paper_files)
            self.assertEqual(overview_files, paper_files)


class FigureTargetCliTests(unittest.TestCase):
    def test_the_switch_defaults_to_overview_and_takes_paper(self):
        parser = build_parser()
        args = parser.parse_args(["paper-evolve-editable", "--paper", "p.pdf", "--out", "o"])
        self.assertEqual(args.figure_target, "overview")
        args = parser.parse_args(["paper-evolve-editable", "--paper", "p.pdf", "--out", "o", "--figure-target", "paper"])
        self.assertEqual(args.figure_target, "paper")


class ForkBeforeCliTests(unittest.TestCase):
    """The paired-experiment switch: which stage a replayed run plans for itself."""

    def test_it_defaults_to_empty_and_takes_every_stage(self):
        parser = build_parser()
        args = parser.parse_args(["paper-evolve-editable", "--paper", "p.pdf", "--out", "o"])
        self.assertEqual(args.fork_before, "")
        for stage in (stage for stage in STAGE_ORDER if stage not in READING_FORK_STAGES):
            args = parser.parse_args(
                [
                    "paper-evolve-editable", "--paper", "p.pdf", "--out", "o",
                    "--replay-upstream-from", "prior", "--fork-before", stage,
                ]
            )
            self.assertEqual(args.fork_before, stage)
            self.assertEqual(args.replay_upstream_from, "prior")

    def test_a_reading_stage_is_refused_as_a_fork_point(self):
        # A fork before the reading inherits nothing, so the switch cannot name one:
        # the two reading stages belong to the attribution vocabulary, not to the set
        # of points a replayed run can plan for itself.
        parser = build_parser()
        for stage in READING_FORK_STAGES:
            with self.assertRaises(SystemExit):
                parser.parse_args(
                    [
                        "paper-evolve-editable", "--paper", "p.pdf", "--out", "o",
                        "--replay-upstream-from", "prior", "--fork-before", stage,
                    ]
                )

    def test_a_stage_outside_the_pipeline_is_refused(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["paper-evolve-editable", "--paper", "p.pdf", "--out", "o", "--fork-before", "somewhere"]
            )

    def test_replayed_icons_are_kept_unless_the_run_says_otherwise(self):
        parser = build_parser()
        args = parser.parse_args(["paper-evolve-editable", "--paper", "p.pdf", "--out", "o"])
        self.assertEqual(args.reuse_replayed_icons, "on")
        args = parser.parse_args(
            [
                "paper-evolve-editable", "--paper", "p.pdf", "--out", "o",
                "--reuse-replayed-icons", "off",
            ]
        )
        self.assertEqual(args.reuse_replayed_icons, "off")


if __name__ == "__main__":
    unittest.main()
