"""A rule-class row is validated end to end by compiling one figure twice.

The row under test is a ``layout_constraint``: a rule about how a figure is drawn, carried by a
Memory row and honoured at compile time. The harness draws the control arm once, compiles that
same figure with the rule applied, renders both, and puts the two to the comparison Judge. The
test runs the real offline pipeline and the real workflow, and stands in only for the Judge.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.evolution.direct import ExperienceStore
from rfs.evolution.direct.experience import experience_library_record
from rfs.workflows.direct_experience_validation import run_direct_experience_validation

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
    "max_rounds": 1,
}

# The row a run writes when a layout rule was measured: the rule travels, the boxes it was
# measured on stay behind in the binding, and the measurement it earned travels with it.
MEASURED = {
    "repair_gain": 0.2,
    "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.7},
}

RULE = {
    "experience_id": "rule-align-grid",
    "status": "current_case_validated",
    "domain": "direct_editable",
    "scope": "global",
    "memory_type": "repair",
    "apply_at": "ppt_compilation",
    "applicability": "common",
    "hypothesis": "Snapping the boxes onto one grid keeps a figure from drifting out of alignment.",
    "guidance": "Read every box's edge off one shared grid so a row of panels cannot drift apart.",
    "action": {"type": "layout_constraint", "scope_type": "global", "params": {"align_grid": 0.02}},
    # The shape the rule was learned under. A figure that answers topology differently is one the
    # rule makes no claim about -- the negative control the exam needs to reach the top rung.
    "condition": {"topology": "linear"},
    "graph_scope": {},
    **MEASURED,
}


class _ComparisonJudge:
    """A stand-in for the acceptance seat: it prefers whichever arm is named."""

    role = "acceptance"

    def __init__(self, choice: str = "B"):
        self.choice = choice

    def evaluate_pairwise(self, request):
        return {
            "judgement_kind": "pairwise",
            "pairwise": {"choice": self.choice, "reason": "the compiled figure reads better"},
            "role": self.role,
        }


def _write_paper(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Compile-Only Validation Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 260), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 300, 550, 360), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(path)
    document.close()


def _record(paper_id: str, domain: str, motif=None, topology: str = "linear") -> dict:
    return {"paper_id": paper_id, "domain": domain, "topology": topology, "features": {"node_count": 4, "edge_count": 3, "motif_histogram": motif or {"linear": 2}}}


def _suite(papers: dict[str, Path]) -> dict:
    cases = []
    for case_id, domain, motif, topology in (
        ("a", "ai_method", None, "linear"),
        ("b", "ai_method", None, "linear"),
        ("c", "biology", None, "linear"),
        ("d", "ai_method", {"cyclic": 2}, "cyclic"),
    ):
        record = _record(case_id, domain, motif, topology)
        cases.append({"case_id": case_id, "paper_id": case_id, "paper": str(papers[case_id]), "domain": domain, "graph_record": record})
    return {
        "query_graph_record": _record("query", "ai_method"),
        "selector": {"quota": {"same_domain_same_structure": 2, "cross_domain_same_structure": 1, "same_domain_cross_structure": 0, "boundary": 1}},
        "cases": cases,
    }


class CompileOnlyValidationIntegrationTest(unittest.TestCase):
    def setUp(self):
        # The run's arms are read back after the call returns, so the workspace has to outlive it.
        self._workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self._workspace.cleanup)
        self.root = Path(self._workspace.name)

    def _run_validation(self, *, choice: str):
        from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable

        root = self.root
        papers = {case_id: root / f"{case_id}.pdf" for case_id in ("a", "b", "c", "d")}
        for path in papers.values():
            _write_paper(path)
        library = root / "library"
        binding = root / "rule_binding.json"
        binding.write_text(json.dumps({"graph_scope": {}, "note": "the boxes this rule was measured on"}), encoding="utf-8")
        store = ExperienceStore(library)
        stored = store.append_result(experience_library_record({**RULE}, binding_ref=str(binding)))
        self.assertTrue(stored["ok"], stored)
        suite = root / "suite.json"
        suite.write_text(json.dumps(_suite(papers)), encoding="utf-8")
        out = root / "out"

        generated: list[str] = []

        def runner(**options):
            generated.append(str(options["out"]))
            return run_paper_evolve_editable(**options)

        with patch("rfs.workflows.direct_experience_validation._pairwise_judge", lambda options: _ComparisonJudge(choice)):
            report = run_direct_experience_validation(suite, library, "rule-align-grid", out, runner=runner, run_options=OFFLINE)
        return report, out, generated

    def test_the_rule_is_measured_by_compiling_one_figure_twice(self):
        report, _out, generated = self._run_validation(choice="B")
        self.assertEqual("compile_only", report["path"])
        self.assertEqual(4, report["summary"]["cases"])
        self.assertGreater(report["summary"]["measured_cases"], 0)
        self.assertTrue(all(item["path"] == "compile_only" for item in report["results"]))
        # One generation per case and not two: the compiled arm is the control arm's own figure.
        self.assertEqual(4, len(generated))
        for item in report["results"]:
            self.assertTrue(item["program_identity"]["equal"], item["case_id"])
            on_dir = Path(item["on"]["run_dir"])
            self.assertTrue((on_dir / "figure_program.json").is_file())
            self.assertTrue((on_dir / "final_draft_preview.png").is_file(), item["case_id"])
            self.assertEqual({"align_grid": 0.02}, item["on"]["compile_report"]["layout"]["rules"])

    def test_the_compiled_arm_still_loses_when_the_comparison_prefers_the_control(self):
        report, _out, _generated = self._run_validation(choice="A")
        self.assertEqual("compile_only", report["path"])
        self.assertGreater(report["summary"]["measured_cases"], 0)
        self.assertEqual(0, report["summary"]["passed_cases"])


if __name__ == "__main__":
    unittest.main()
