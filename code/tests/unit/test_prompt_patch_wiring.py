"""End-to-end wiring of prompt patches into the evolution loop and the ON/OFF harness."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.evolution.direct import ExperienceStore, PromptPatchStore
from rfs.evolution.direct.prompt_patch import build_prompt_patch
from rfs.workflows.direct_experience_validation import run_direct_experience_validation
from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."

MEASURED = {
    "experience_id": "direct-exp-1",
    "action": {"type": "relayout_slots", "scope_type": "slot", "factor": 0.9},
    "condition": {"topology": "pipeline"},
    "accepted": True,
    "repair_gain": 0.2,
    "validated_cases": ["a.pdf"],
    "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True},
}


def write_paper(root: Path) -> Path:
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Patchable Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


START_STATE = {
    "layout": {
        "panels": [{"id": "p1", "bbox_percent": {"x": 0.05, "y": 0.1, "w": 0.9, "h": 0.8}}],
        "nodes": [
            {"id": "n1", "panel_id": "p1", "bbox_percent": {"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.2}},
            {"id": "n2", "panel_id": "p1", "bbox_percent": {"x": 0.4, "y": 0.2, "w": 0.2, "h": 0.2}},
        ],
        "connectors": [{"id": "c1", "source": "n1", "target": "n2"}],
    }
}


def start_state(options: dict) -> None:
    """A real run leaves the state it started from; a stand-in has to leave one too."""

    target = Path(options["out"]) / "round_00" / "state_before.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"state_hash": "start", "state": START_STATE}), encoding="utf-8")


def run_workflow(root: Path, *, library: Path | None = None, max_rounds: int = 1) -> Path:
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
            max_rounds=max_rounds,
            candidates_per_round=1,
            experience_enabled=library is not None,
            experience_library=library,
        )
    return out


def events(out: Path) -> list[dict]:
    path = out / "evolution_events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class PromptPatchWiringTests(unittest.TestCase):
    def test_a_promoted_prohibition_blocks_the_candidate_and_rolls_the_round_back(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            library = root / "library"
            patch = build_prompt_patch(
                experience={**MEASURED, "accepted": False, "repair_gain": -0.05, "apply_at": "ppt_compilation"},
                condition={"metric": "unbound_scientific_nodes", "op": "<=", "value": -1},
            )
            store = PromptPatchStore(library)
            self.assertTrue(store.append_result({**patch, "status": "promoted"})["ok"])

            out = run_workflow(root, library=library)
            report = json.loads((out / "paper_evolve_editable_report.json").read_text(encoding="utf-8"))
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            decision = json.loads((out / "round_00" / "repair_decision.json").read_text(encoding="utf-8"))
            check = json.loads((out / "round_00" / "candidate_00" / "prompt_patch_check.json").read_text(encoding="utf-8"))
            recorded = [item["event"] for item in events(out)]

        manifest = report["prompt_patch_manifest"]
        self.assertEqual(manifest["contract"], "rfs.prompt_patch")
        self.assertEqual(manifest["stages"]["ppt_compilation"]["selected"], [patch["patch_id"]])
        self.assertEqual(manifest["applied"]["ppt_compilation"], [patch["patch_id"]])
        self.assertEqual(manifest["applied"]["theme_planning"], [])
        self.assertFalse(check["ok"])
        self.assertEqual(decision["reason"], "prompt_patch_prohibition_violated")
        self.assertFalse(rounds[0]["accepted"])
        self.assertIn("prompt_patch_prohibition_blocked", recorded)
        self.assertIn("candidate_rolled_back", recorded)

    def test_a_promoted_advisory_patch_is_recorded_without_blocking_anything(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            library = root / "library"
            patch = build_prompt_patch(experience={**MEASURED, "apply_at": "global_planning"})
            store = PromptPatchStore(library)
            self.assertTrue(store.append_result({**patch, "status": "promoted"})["ok"])

            out = run_workflow(root, library=library)
            report = json.loads((out / "paper_evolve_editable_report.json").read_text(encoding="utf-8"))
            check = json.loads((out / "round_00" / "candidate_00" / "prompt_patch_check.json").read_text(encoding="utf-8"))
            recorded = [item["event"] for item in events(out)]

        self.assertEqual(report["prompt_patch_manifest"]["applied"]["global_planning"], [patch["patch_id"]])
        self.assertTrue(check["ok"])
        self.assertEqual(check["violations"], [])
        self.assertNotIn("prompt_patch_prohibition_blocked", recorded)

    def test_a_patch_that_applies_at_the_reading_reaches_the_reading(self):
        """The reading and the content plan state their instructions in prose, not in a block.

        Both stage names were accepted by the contract and neither was ever assembled, so a patch
        a round earned at the reading could be stored, permitted and injected nowhere.
        """

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            library = root / "library"
            reading = build_prompt_patch(experience={**MEASURED, "apply_at": "paper_reading"})
            planning = build_prompt_patch(experience={**MEASURED, "apply_at": "content_planning"})
            store = PromptPatchStore(library)
            for patch in (reading, planning):
                self.assertTrue(store.append_result({**patch, "status": "promoted"})["ok"])

            out = run_workflow(root, library=library)
            manifest = json.loads((out / "paper_evolve_editable_report.json").read_text(encoding="utf-8"))["prompt_patch_manifest"]
            injected = [item for item in events(out) if item["event"] == "prompt_patch_injected"]

        self.assertEqual(manifest["applied"]["paper_reading"], [reading["patch_id"]])
        self.assertEqual(manifest["applied"]["content_planning"], [planning["patch_id"]])
        stages = {str(item.get("stage") or "") for item in injected}
        self.assertIn("paper_reading", stages)
        self.assertIn("content_planning", stages)
        self.assertEqual(manifest["revisions"]["paper_reading"]["base_revision"], "base")
        self.assertNotEqual(manifest["revisions"]["paper_reading"]["revision"], "base")
        self.assertFalse(manifest["revisions"]["paper_reading"]["trial"])

    def test_a_patch_never_reaches_a_stage_it_does_not_apply_at(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            library = root / "library"
            patch = build_prompt_patch(experience={**MEASURED, "apply_at": "repair"})
            PromptPatchStore(library).append_result({**patch, "status": "promoted"})

            out = run_workflow(root, library=library)
            manifest = json.loads((out / "paper_evolve_editable_report.json").read_text(encoding="utf-8"))["prompt_patch_manifest"]

        self.assertEqual(manifest["applied"]["repair"], [patch["patch_id"]])
        self.assertEqual(manifest["applied"]["global_planning"], [])


class PromptPatchValidationTests(unittest.TestCase):
    def _suite(self, root: Path) -> Path:
        def record(paper_id, domain):
            return {"paper_id": paper_id, "domain": domain, "features": {"node_count": 4, "edge_count": 3, "motif_histogram": {"linear": 2}}}

        suite = root / "suite.json"
        suite.write_text(
            json.dumps({
                "query_graph_record": record("query", "ai_method"),
                "selector": {"quota": {"same_domain_same_structure": 2, "cross_domain_same_structure": 1, "boundary": 1}},
                "cases": [
                    {"case_id": "a", "paper_id": "a", "paper": "a.pdf", "domain": "ai_method", "graph_record": record("a", "ai_method")},
                    {"case_id": "b", "paper_id": "b", "paper": "b.pdf", "domain": "ai_method", "graph_record": record("b", "ai_method")},
                    {"case_id": "c", "paper_id": "c", "paper": "c.pdf", "domain": "system-platform", "graph_record": record("c", "system-platform")},
                    {"case_id": "d", "paper_id": "d", "paper": "d.pdf", "domain": "ai_method", "graph_record": {**record("d", "ai_method"), "slot_ids": ["slot_other"]}},
                ],
            }),
            encoding="utf-8",
        )
        return suite

    def test_a_patch_is_validated_by_the_same_harness_and_lands_in_the_config_hash(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            library = root / "library"
            # A stage a pair can actually fork at: a patch recorded at a stage with no shareable
            # prefix is retired as unverifiable instead, which its own test pins.
            patch = build_prompt_patch(experience={**MEASURED, "apply_at": "repair", "graph_scope": {"slot_ids": ["icon_1"]}})
            PromptPatchStore(library).append_result({**patch, "status": "candidate"})
            ExperienceStore(library).append({"experience_id": "e1", "applicability": "common", "status": "current_case_validated", "domain": "direct_editable", "scope": "slot", "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
 "hypothesis": "targeted repair", "action": {"type": "regen_assets", "slot_ids": ["icon_1"]}, "graph_scope": {"slot_ids": ["icon_1"]}, "repair_gain": 0.25, "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75}})
            suite = self._suite(root)
            seen: list[dict] = []

            def runner(**kwargs):
                seen.append(kwargs)
                enabled = kwargs.get("seed_experience") is not None or kwargs.get("seed_prompt_patch") is not None
                start_state(kwargs)
                return {"ok": True, "judgement": {"overall_score": 0.84 if enabled else 0.8, "blocking_issue_count": 0}}

            patch_report = run_direct_experience_validation(suite, library, out=root / "patch_out", prompt_patch_id=patch["patch_id"], runner=runner)
            patch_hashes = {item["off"]["paired_config_hash"] for item in patch_report["results"]}
            patch_pairs_isolated = [item["pair_audit"]["ok"] for item in patch_report["results"]]
            on_patches = [item["seed_prompt_patch"] for item in seen if item.get("seed_prompt_patch")]
            off_patches = [item["seed_prompt_patch"] for item in seen if item["seed_prompt_patch"] is None]
            seen.clear()
            experience_report = run_direct_experience_validation(suite, library, "e1", root / "exp_out", runner=runner)

            patch_store = PromptPatchStore(library)
            patch_status = patch_store.lifecycle_status(patch["patch_id"])
        experience_hashes = {item["off"]["paired_config_hash"] for item in experience_report["results"]}

        self.assertEqual(patch_report["prompt_patch_id"], patch["patch_id"])
        self.assertEqual(experience_report["prompt_patch_id"], None)
        self.assertTrue(on_patches and all(item["patch_id"] == patch["patch_id"] for item in on_patches))
        self.assertTrue(off_patches)
        self.assertTrue(all(patch_pairs_isolated))
        self.assertFalse(patch_hashes & experience_hashes)
        self.assertEqual(patch_status, patch_report["status"])
        self.assertEqual(patch_status, "promoted")


if __name__ == "__main__":
    unittest.main()
