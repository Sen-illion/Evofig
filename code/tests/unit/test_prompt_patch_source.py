"""Rendering Prompt Patches from measured Memory rows, and backfilling a library.

A patch is a second view of a measurement the row already carries, so these tests pin down
what a row can become: an advisory or executable-geometry patch when it measured a gain, an
enforceable prohibition when a round refused it and recorded the break, and nothing at all
when the row carries no bound a machine could check.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch as mock_patch

from rfs.contracts.prompt_patch import is_enforceable_prohibition, validate_prompt_patch
from rfs.evolution.direct import ExperienceStore, PromptPatchStore, build_patches_for_experience
from rfs.evolution.direct.prompt_patch import prohibition_condition_for_experience
from rfs.workflows.patch_backfill import backfill_prompt_patches


def measured(**overrides) -> dict:
    row = {
        "experience_id": "direct-exp-source-01",
        "domain": "direct_editable",
        "scope": "slot",
        "memory_type": "layout",
        "apply_at": "ppt_compilation",
        "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
        "hypothesis": "shrinking the slot keeps the label inside its box",
        "action": {"type": "relayout_slots", "scope_type": "slot", "factor": 0.9},
        "condition": {"topology": "pipeline", "node_count_bucket": "medium", "connector_count_bucket": "low"},
        "accepted": True,
        "repair_gain": 0.12,
        "validated_cases": ["a.pdf"],
        "graph_scope": {"slot_ids": ["slot_1"]},
        "status": "candidate",
        "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True},
        "observed_effect": {
            "overall_score_delta": 0.12,
            "decomposability_violations": [{"metric": "cross_panel_connectors", "kind": "soft", "value": 4, "threshold": 2}],
        },
    }
    return {**row, **overrides}


class PromptPatchSourceTests(unittest.TestCase):
    def test_a_measured_gain_becomes_a_patch_the_library_accepts(self):
        with tempfile.TemporaryDirectory() as name:
            rendering = build_patches_for_experience(measured())

            self.assertEqual("rendered", rendering["reason"])
            patch = rendering["patch"]
            self.assertEqual("executable_geometry", patch["class"])
            self.assertEqual("ppt_compilation", patch["apply_at"])
            self.assertTrue(validate_prompt_patch(patch)["ok"])
            stored = PromptPatchStore(Path(name) / "library").append_result(patch)
            self.assertTrue(stored["ok"])

    def test_a_refusal_with_a_measured_break_becomes_an_enforceable_prohibition(self):
        rendering = build_patches_for_experience(
            measured(accepted=False, repair_gain=-0.05, action={"type": "relayout_slots", "scope_type": "slot"})
        )

        patch = rendering["patch"]
        self.assertEqual("prohibited", patch["class"])
        self.assertEqual({"metric": "cross_panel_connectors", "op": "<=", "value": 2.0}, patch["condition"])
        self.assertTrue(is_enforceable_prohibition(patch["condition"]))
        self.assertEqual("Only apply relayout slots when cross_panel_connectors <= 2.", patch["directive"])

    def test_a_floor_metric_is_bound_the_other_way_round(self):
        condition = prohibition_condition_for_experience(
            measured(observed_effect={"decomposability_violations": [{"metric": "min_panel_occupancy", "kind": "soft", "value": 0.1, "threshold": 0.25}]})
        )

        self.assertEqual({"metric": "min_panel_occupancy", "op": ">=", "value": 0.25}, condition)

    def test_a_refusal_with_no_measured_break_becomes_no_patch(self):
        rendering = build_patches_for_experience(
            measured(accepted=False, repair_gain=-0.05, observed_effect={"overall_score_delta": -0.05})
        )

        self.assertIsNone(rendering["patch"])
        self.assertEqual("prohibition_condition_unavailable", rendering["reason"])

    def test_a_pairwise_accepted_geometry_repair_is_not_a_prohibition(self):
        rendering = build_patches_for_experience(
            measured(accepted=True, repair_gain=0.0, comparison="pairwise", pairwise_verdict="after")
        )

        self.assertEqual("rendered", rendering["reason"])
        self.assertEqual("executable_geometry", rendering["patch"]["class"])

    def test_a_row_without_an_id_cannot_become_a_patch(self):
        rendering = build_patches_for_experience({"apply_at": "repair"})

        self.assertIsNone(rendering["patch"])
        self.assertIn("no experience_id", rendering["reason"])


class PromptPatchBackfillTests(unittest.TestCase):
    def _library(self, name: str, rows: list[dict]) -> Path:
        root = Path(name) / "library"
        store = ExperienceStore(root)
        for row in rows:
            self.assertTrue(store.append_result(row)["ok"])
        return root

    def test_backfill_writes_the_missing_patch_once_and_reports_every_row(self):
        with tempfile.TemporaryDirectory() as name:
            root = self._library(name, [measured(), measured(accepted=False, repair_gain=-0.05, experience_id="direct-exp-source-02", observed_effect={})])

            first = backfill_prompt_patches(root)
            second = backfill_prompt_patches(root)

            self.assertEqual(1, first["written"])
            self.assertEqual(1, first["rejected"])
            self.assertEqual(0, first["skipped"])
            self.assertEqual(0, second["written"])
            self.assertEqual(1, second["skipped"])
            written = PromptPatchStore(root).records()
            self.assertEqual(1, len(written))
            self.assertEqual(["direct-exp-source-01"], written[0]["source_experience_ids"])
            report = json.loads((root / "prompt_patch_backfill.json").read_text(encoding="utf-8"))
            self.assertEqual(second["written"], report["written"])
            self.assertEqual(second["skipped"], report["skipped"])

    def test_backfill_does_not_duplicate_a_patch_the_loop_already_wrote(self):
        with tempfile.TemporaryDirectory() as name:
            root = self._library(name, [measured()])
            patch_store = PromptPatchStore(root)
            self.assertTrue(patch_store.append_result(build_patches_for_experience(measured())["patch"]))

            report = backfill_prompt_patches(root)

            self.assertEqual(0, report["written"])
            self.assertEqual(1, report["skipped"])
            self.assertEqual(1, len(patch_store.records()))
class _ReflectingJudge:
    """The real advisory judge, with a stateable reflection instead of whatever the model said."""

    def __init__(self, judge, reflection: dict):
        self._judge = judge
        self._reflection = dict(reflection)

    def __getattr__(self, name):
        return getattr(self._judge, name)

    def reflect(self, outcome):
        return dict(self._reflection)


class PromptPatchLoopWiringTests(unittest.TestCase):
    """The round loop writes the patch its own measurement supports, or records why it cannot.

    A patch rides in the next round that plans the stage it applies at, so whether a run injects
    one is a question about how many rounds follow the round that permitted it. These tests fix
    the two-round reading: the row is written, and the round's own ``applied_prompt_patches`` says
    what that round ran under.
    """

    def _run(self, root: Path, library: Path, *, reflection: dict) -> Path:
        import fitz
        from rfs.workflows import paper_evolve_editable
        from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable

        target = root / "paper.pdf"
        document = fitz.open()
        page = document.new_page(width=600, height=800)
        page.insert_text((50, 60), "A Patchable Study", fontsize=14)
        page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
        page.insert_textbox((50, 160, 550, 240), "We propose a frozen encoder that turns a paper into a symbolic figure plan.", fontsize=10)
        page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
        document.save(target)
        document.close()
        store = ExperienceStore(library)
        self.assertTrue(store.append_result({
            "experience_id": "direct-exp-reflect",
            "domain": "direct_editable",
            "scope": "connector",
            "memory_type": "connector",
            "apply_at": "ppt_compilation",
            "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
            "hypothesis": "rerouting a crossed connector keeps the panel readable",
            "action": {"type": "reroute", "scope_type": "connector"},
            "condition": {"topology": "pipeline"},
            "accepted": True,
            "repair_gain": 0.25,
            "validated_cases": ["a.pdf"],
            "graph_scope": {"slot_ids": ["slot_1"]},
            "status": "promoted",
            "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True},
        })["ok"])
        out = root / "run"
        real_build = paper_evolve_editable.build_direct_judges

        def build(**kwargs):
            online, frozen = real_build(**kwargs)
            return _ReflectingJudge(online, reflection), frozen

        with mock_patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)), mock_patch(
            "rfs.workflows.paper_evolve_editable.build_direct_judges", build
        ):
            run_paper_evolve_editable(
                paper=target,
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
                max_rounds=2,
                candidates_per_round=1,
                experience_enabled=True,
                experience_library=library,
                seed_experience={"experience_id": "direct-exp-reflect", "action": {"type": "reroute", "scope_type": "connector", "edge_ids": ["an-edge-that-is-not-here"]}},
                fixed_rounds=True,
            )
        return out

    def test_a_measured_round_writes_its_patch_or_records_why_it_cannot(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            library = root / "library"
            out = self._run(root, library, reflection={"next_policy": "keep"})
            events = [json.loads(line) for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            names = [event["event"] for event in events]
            self.assertTrue({"prompt_patch_recorded", "prompt_patch_rejected"} & set(names), names)
            round_dir = out / "round_01"
            recorded = round_dir / "prompt_patch_candidate.json"
            rejected = round_dir / "prompt_patch_rejected.json"
            self.assertTrue(recorded.exists() or rejected.exists())
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            patches = PromptPatchStore(library).records()
            if recorded.exists():
                self.assertEqual([str(recorded and json.loads(recorded.read_text(encoding="utf-8"))["patch_id"])], [item["patch_id"] for item in patches])
                self.assertEqual([json.loads(recorded.read_text(encoding="utf-8"))["patch_id"]], [rounds[1]["prompt_patch_recorded"]])
            else:
                self.assertEqual([], patches)
                self.assertNotIn("prompt_patch_recorded", names)
                self.assertIn("experience_refuted", names)
            # The patch is permitted at the end of round one and this run has no round two,
            # so nothing follows it here; what round one itself ran under is still empty.
            self.assertEqual(
                [],
                [patch_id for ids in rounds[1]["applied_prompt_patches"].values() for patch_id in ids],
            )


if __name__ == "__main__":
    unittest.main()
