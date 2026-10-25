"""A case set interrupted against the real pipeline finishes on the next call.

The unit tests pin the continuation with a stand-in runner. This one runs the actual offline
generation pipeline for both arms of a small four-group set, kills the call in the middle of the
set, restarts it, and insists the second call reaches a real ladder rung with cases actually
measured. It is the wiring that made "35 rows, 0 promoted, measured_cases 0" possible: a set that
never finished, reported as if it had been judged.
"""

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from rfs.evolution.direct import ExperienceStore
from rfs.utils import write_json
from rfs.workflows.direct_experience_validation import run_direct_experience_validation

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
    "max_rounds": 1,
}

MEASURED = {
    "repair_gain": 0.25,
    "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75},
}

EXPERIENCE = {
    "experience_id": "e1",
    "status": "current_case_validated",
    "domain": "direct_editable",
    "scope": "slot",
    "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
    "hypothesis": "a single reading path reads better",
    "action": {"type": "regen_assets", "slot_ids": ["icon_n"]},
    "graph_scope": {"slot_ids": ["icon_n"]},
    "applicability": "common",
    **MEASURED,
}


def _write_paper(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Resumable Validation Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 260), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 300, 550, 360), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(path)
    document.close()


def _record(paper_id, domain, motif=None):
    return {"paper_id": paper_id, "domain": domain, "features": {"node_count": 4, "edge_count": 3, "motif_histogram": motif or {"linear": 2}}}


def _suite(root: Path, papers: dict[str, Path]) -> dict:
    cases = []
    for case_id, domain, motif in (
        ("a", "ai_method", None),
        ("b", "ai_method", None),
        ("c", "biology", None),
        ("d", "ai_method", {"cyclic": 2}),
    ):
        record = _record(case_id, domain, motif)
        if case_id == "d":
            # The negative control: a figure holding none of the objects the row is scoped to,
            # which is the boundary counter-example the ladder asks for.
            record["slot_ids"] = ["slot_other"]
        cases.append(
            {
                "case_id": case_id,
                "paper_id": case_id,
                "paper": str(papers[case_id]),
                "domain": domain,
                "graph_record": record,
            }
        )
    return {
        "query_graph_record": _record("query", "ai_method"),
        "selector": {"quota": {"same_domain_same_structure": 2, "cross_domain_same_structure": 1, "same_domain_cross_structure": 0, "boundary": 1}},
        "cases": cases,
    }


class InterruptedCaseSetIntegrationTest(unittest.TestCase):
    def test_an_interrupted_set_resumes_and_finishes_with_real_cases(self):
        from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            papers = {case_id: root / f"{case_id}.pdf" for case_id in ("a", "b", "c", "d")}
            for path in papers.values():
                _write_paper(path)
            library = root / "library"
            store = ExperienceStore(library)
            self.assertTrue(store.append({**EXPERIENCE}))
            suite = root / "suite.json"
            suite.write_text(json.dumps(_suite(root, papers)), encoding="utf-8")
            out = root / "out"

            calls: list[str] = []

            def arm(options):
                """Run one arm, then give the control arm the artifacts a replay insists on.

                The offline structure planner draws nothing, so its own design plan does not
                validate and the memory arm's strict replay of it is refused. Supplying the seed
                the loader asks for is the same accommodation the continuous-loop integration
                test makes; the subject here is the case set that resumes, not how the offline
                planner draws.
                """

                result = run_paper_evolve_editable(**options)
                if options.get("seed_experience") is None:
                    arm_dir = Path(options["out"])
                    write_json(arm_dir / "design_plan_raw.json", DESIGN)
                    for name in ("reference_style_profile.json", "reference_layout_profile.json"):
                        write_json(arm_dir / name, {"status": "complete"})
                return result

            def interrupting(**options):
                calls.append(str(options["out"]))
                if len(calls) > 2:
                    # The call dies part-way into the second case, after the first case's two arms
                    # were generated and decided and the second case's control arm began.
                    raise RuntimeError("the call ended here")
                return arm(options)

            with self.assertRaises(RuntimeError):
                run_direct_experience_validation(suite, library, "e1", out, runner=interrupting, run_options=OFFLINE)

            decided_after_crash = sorted((out / "cases").glob("*.json"))
            self.assertEqual(1, len(decided_after_crash))
            self.assertTrue((out / "validation_case_selection.json").is_file())

            report = run_direct_experience_validation(suite, library, "e1", out, runner=lambda **options: arm(options), run_options=OFFLINE)

        self.assertEqual(4, report["summary"]["cases"])
        self.assertGreater(report["summary"]["measured_cases"], 0)
        self.assertNotIn(report["status"], {"candidate", "unverifiable"})
        self.assertNotEqual(report["ladder"]["stopped_because"], "pair_not_measured")
        self.assertEqual(1, report["resumed_cases"])
        self.assertEqual(["a", "b", "c", "d"], [item["case_id"] for item in report["results"]])


if __name__ == "__main__":
    unittest.main()
