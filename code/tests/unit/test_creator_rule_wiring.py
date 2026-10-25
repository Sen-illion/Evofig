"""End-to-end wiring of the aggregated Creator rules into the evolution loop."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.contracts import CURATED_STANDARD_RULES
from rfs.evolution.direct import ExperienceStore
from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."

# One validated Memory row with two supporting cases, which is the smallest evidence a
# creator rule aggregates from.
MEASURED = {
    "experience_id": "direct-exp-rule",
    "domain": "direct_editable",
    "scope": "slot",
    "memory_type": "segmentation",
    "apply_at": "segmentation",
    "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
    "hypothesis": "resegmenting when connectors cross a panel keeps the figure decomposable",
    "action": {"type": "resegment", "scope_type": "slot"},
    "condition": {"metric": "cross_panel_connectors", "op": "<=", "value": 2},
    "accepted": True,
    "repair_gain": 0.25,
    "validated_cases": ["a.pdf", "b.pdf"],
    "graph_scope": {"slot_ids": ["slot_1"]},
    "status": "cross_domain_validated",
    "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75},
}


def write_paper(root: Path) -> Path:
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Rule Guided Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


def run_workflow(root: Path, library: Path, *, max_rounds: int, curated_standard: str | None = "") -> Path:
    # The authored standard is off by default here: these tests are about the rule the round
    # earned, and a second, always-present rule list would only blur what they assert.
    out = root / "run"
    with patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)):
        run_paper_evolve_editable(
            paper=root / "paper.pdf",
            out=out,
            curated_standard=curated_standard,
            planner_mode="heuristic",
            method_mode="heuristic",
            structure_mode="heuristic",
            text_plan_mode="heuristic",
            ppt_plan_mode="heuristic",
            color_theme_mode="fallback",
            icon_mode="placeholder",
            visual_review_mode="heuristic",
            judge_mode="heuristic",
            max_rounds=max_rounds,
            candidates_per_round=1,
            experience_enabled=True,
            experience_library=library,
        )
    return out


def ledger_rows(library: Path) -> list[dict]:
    path = library / "creator_rule_events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class CreatorRuleWiringTests(unittest.TestCase):
    def test_a_validated_experience_becomes_a_rule_the_round_runs_under(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            library = root / "library"
            self.assertTrue(ExperienceStore(library).append_result(MEASURED)["ok"])

            out = run_workflow(root, library, max_rounds=1)
            report = json.loads((out / "paper_evolve_editable_report.json").read_text(encoding="utf-8"))
            brief = json.loads((out / "exemplar_style_brief.json").read_text(encoding="utf-8"))
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            state = json.loads((out / "evolution_state.json").read_text(encoding="utf-8"))
            rows = ledger_rows(library)
            trace = json.loads((out / "memory_application_trace.json").read_text(encoding="utf-8"))

        creator_rules = report["creator_rules"]
        self.assertEqual(creator_rules["contract"], "rfs.creator_rule_manifest")
        self.assertEqual(len(creator_rules["rules"]), 1)
        rule = creator_rules["rules"][0]
        self.assertEqual(rule["apply_at"], "segmentation")
        self.assertEqual(rule["support"], 2)
        self.assertEqual(rule["source_experience_ids"], ["direct-exp-rule"])
        self.assertEqual(creator_rules["buckets"], {"segmentation": [rule["rule_id"]]})
        self.assertEqual(creator_rules["injected"], {"segmentation": [rule["rule_id"]]})
        self.assertEqual(rounds[0]["applied_rules"], {"segmentation": [rule["rule_id"]]})
        self.assertEqual(state["creator_rules"]["rules"], [rule])
        self.assertEqual(brief["stage4_manifest"]["composition_patterns"], [rule["directive"]])
        self.assertEqual(brief["creator_rules"]["buckets"], {"segmentation": [rule["rule_id"]]})
        self.assertEqual([(item["rule_id"], item["event"]) for item in rows], [(rule["rule_id"], "creator_rule_confirmed")])
        self.assertEqual(trace["contract"], "rfs.memory_application_trace")
        self.assertEqual(report["memory_application_trace"], trace)
        self.assertEqual(state["memory_application_trace"], trace)
        self.assertEqual([item["round"] for item in trace["rounds"]], [item["round"] for item in rounds])
        self.assertEqual(trace["rounds"][0]["applied_rules"], rounds[0]["applied_rules"])
        self.assertEqual(trace["rounds"][0]["applied_prompt_patches"], rounds[0]["applied_prompt_patches"])
        self.assertEqual(trace["rules"]["applied"], {"segmentation": [rule["rule_id"]]})
        self.assertEqual(trace["totals"]["rules_applied"], [rule["rule_id"]])
        self.assertEqual(trace["totals"]["rounds"], 1)
        self.assertEqual(trace["libraries"]["experience"], str(library.resolve()))

    def test_the_authored_standard_travels_with_every_round(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            library = root / "library"
            ExperienceStore(library).append_result(MEASURED)

            out = run_workflow(root, library, max_rounds=1, curated_standard=None)
            report = json.loads((out / "paper_evolve_editable_report.json").read_text(encoding="utf-8"))
            brief = json.loads((out / "exemplar_style_brief.json").read_text(encoding="utf-8"))

        curated = report["creator_rules"]["curated"]
        self.assertEqual(curated["source"], "curated")
        self.assertEqual(curated["count"], len(CURATED_STANDARD_RULES))
        self.assertEqual(len(curated["rule_ids"]), len(CURATED_STANDARD_RULES))
        # The earned rule and the authored ones travel together, and the authored ones are
        # still marked as authored in the brief the planners read.
        sources = {rule["provenance"]["source"] for rule in brief["creator_rules"]["rules"]}
        self.assertIn("curated", sources)
        self.assertEqual(len(sources - {"curated"}), 1)
        from rfs.contracts.vlm_context import advisory_rule_rows
        rows = advisory_rule_rows(brief)
        self.assertTrue(any(row["source"] == "curated" for row in rows))

    def test_a_guided_round_that_rolls_back_refutes_the_rule_behind_it(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            library = root / "library"
            store = ExperienceStore(library)
            store.append_result(MEASURED)

            out = run_workflow(root, library, max_rounds=2)
            report = json.loads((out / "paper_evolve_editable_report.json").read_text(encoding="utf-8"))
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            event_names = [json.loads(line)["event"] for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            rows = ledger_rows(library)
            refuted_status = store.lifecycle_status("direct-exp-rule")

        rule_id = report["creator_rules"]["rules"][0]["rule_id"]
        refutations = [item for item in rows if item["event"] == "creator_rule_refuted"]
        self.assertEqual([item["accepted"] for item in rounds], [True, False])
        self.assertIn("creator_rule_refuted", event_names)
        self.assertEqual(len(refutations), 1)
        self.assertEqual(refutations[0]["rule_id"], rule_id)
        self.assertEqual(refutations[0]["refutations"], 1)
        self.assertFalse(refutations[0]["disabled"])
        self.assertEqual(refutations[0]["source_ids"], ["direct-exp-rule"])
        self.assertEqual(report["creator_rules"]["refutations"], {rule_id: 1})
        self.assertEqual(refuted_status, "refuted")


if __name__ == "__main__":
    unittest.main()
