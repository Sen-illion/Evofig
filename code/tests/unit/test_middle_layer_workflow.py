"""The middle-layer row plans without the paper's explicit structure and says so.

The full configuration turns a paper into a scientific graph, its subgraphs and a
scientific hierarchy plan, and planning reads them. The supplementary ablation row
turns that off: nothing is built while the figure is planned, and the graph is
rebuilt once the design exists, purely as the measurement the Grounding Gate and
the alignment reports are computed against.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.cli import build_parser
from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."

#: The two artifacts only the full row writes. Everything else a run emits has to
#: be present in both, or the two rows would differ in more than one switch.
HIERARCHY_ARTIFACTS = {"scientific_hierarchy_plan.json", "scientific_hierarchy_validation.json"}


def write_paper(root: Path) -> Path:
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Middle Layer Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Method", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    document.save(target)
    document.close()
    return target


def run_workflow(root: Path, *, middle_layer: str):
    out = root / middle_layer
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
            middle_layer=middle_layer,
        )
    return out, report


class MiddleLayerWorkflowTests(unittest.TestCase):
    def test_the_row_writes_no_hierarchy_and_stamps_the_graph_post_hoc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_paper(root)
            out, report = run_workflow(root, middle_layer="off")

            for name in sorted(HIERARCHY_ARTIFACTS):
                self.assertFalse((out / name).exists(), name)
            self.assertEqual(report["middle_layer"], "off")
            self.assertEqual(report["scientific_hierarchy"], "skipped")
            self.assertEqual(report["graph_context"], "not_applicable")
            self.assertEqual(report["scientific_graph_provenance"], "post_hoc_measurement")

            # The graph is still there, measured the same way, and it says in its
            # own report that it was built after the design instead of before it.
            import json

            graph_report = json.loads((out / "scientific_graph_report.json").read_text(encoding="utf-8"))
            self.assertEqual(graph_report["provenance"], "post_hoc_measurement")
            self.assertTrue((out / "scientific_graph.json").is_file())
            self.assertTrue((out / "scientific_subgraphs.json").is_file())

    def test_the_full_row_still_plans_its_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_paper(root)
            out, report = run_workflow(root, middle_layer="on")

            for name in sorted(HIERARCHY_ARTIFACTS):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual(report["middle_layer"], "on")
            self.assertEqual(report["scientific_hierarchy"], "planned")
            self.assertEqual(report["graph_context"], "on")
            self.assertEqual(report["scientific_graph_provenance"], "planned")

    def test_the_two_rows_write_the_same_artifact_set_but_for_the_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_paper(root)
            on_out, _ = run_workflow(root, middle_layer="on")
            off_out, _ = run_workflow(root, middle_layer="off")

            on_files = {path.name for path in on_out.iterdir() if path.is_file()}
            off_files = {path.name for path in off_out.iterdir() if path.is_file()}
            self.assertEqual(on_files - off_files, HIERARCHY_ARTIFACTS)
            self.assertEqual(off_files - on_files, set())


class MiddleLayerNoPlanningTests(unittest.TestCase):
    def test_the_row_pays_for_no_hierarchy_call_even_in_vlm_structure_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_paper(root)

            def refuse(*_args, **_kwargs):
                raise AssertionError("the middle-layer row planned a scientific hierarchy")

            # The structure planner itself is stubbed: this test is about which
            # stages the row pays for, not about what a provider answers.
            with patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)), patch(
                "rfs.workflows.paper_evolve_editable.plan_scientific_hierarchy", refuse
            ), patch(
                "rfs.workflows.paper_evolve_editable.plan_structure_vlm", lambda *a, **k: ({}, {"mode": "stub"})
            ):
                run_paper_evolve_editable(
                    paper=root / "paper.pdf",
                    out=root / "off_vlm_structure",
                    planner_mode="heuristic",
                    method_mode="heuristic",
                    structure_mode="vlm",
                    text_plan_mode="heuristic",
                    ppt_plan_mode="heuristic",
                    color_theme_mode="fallback",
                    icon_mode="placeholder",
                    visual_review_mode="heuristic",
                    judge_mode="heuristic",
                    max_rounds=1,
                    candidates_per_round=1,
                    experience_enabled=False,
                    middle_layer="off",
                )


class MiddleLayerCliTests(unittest.TestCase):
    def test_the_switch_defaults_to_on_and_takes_off(self):
        parser = build_parser()
        args = parser.parse_args(["paper-evolve-editable", "--paper", "p.pdf", "--out", "o"])
        self.assertEqual(args.middle_layer, "on")
        args = parser.parse_args(["paper-evolve-editable", "--paper", "p.pdf", "--out", "o", "--middle-layer", "off"])
        self.assertEqual(args.middle_layer, "off")


if __name__ == "__main__":
    unittest.main()
