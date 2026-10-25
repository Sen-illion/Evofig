"""The graph switch changes what planning reads, never which artifacts a run writes."""

from __future__ import annotations

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
    page.insert_text((50, 60), "A Graph Context Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Method", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    document.save(target)
    document.close()
    return target


def run_workflow(root: Path, *, graph_context: bool):
    out = root / ("on" if graph_context else "off")
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
            graph_context=graph_context,
        )
    return out, report


class GraphContextWorkflowTests(unittest.TestCase):
    def test_both_positions_write_the_same_artifact_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_paper(root)
            on_out, on_report = run_workflow(root, graph_context=True)
            off_out, off_report = run_workflow(root, graph_context=False)

            self.assertEqual(on_report["graph_context"], "on")
            self.assertEqual(off_report["graph_context"], "off")
            for out in (on_out, off_out):
                self.assertTrue((out / "scientific_subgraphs.json").exists())
                self.assertTrue((out / "scientific_subgraphs_report.json").exists())
            self.assertEqual(
                {path.name for path in on_out.glob("*.json")},
                {path.name for path in off_out.glob("*.json")},
            )


if __name__ == "__main__":
    unittest.main()
