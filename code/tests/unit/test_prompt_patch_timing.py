"""When a prompt patch is permitted, and when a round it guided counts against it.

Batch C of the fifth repair batch made two changes to a patch's timing. A patch a round permits
now rides in the very next round of the same iteration that plans the stage it applies at,
instead of waiting for the paper's next iteration, and the record of which prompt version each
round ran under is written down. A round that ran under a patch is also a measurement of it, so
a patch that keeps losing is retired after a run of losses rather than sitting in the prompt
forever. These tests pin both down: the streak and the retirement at the store and at the
charging step, and the permission-to-injection path end to end through the real round loop.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch as mock_patch

from rfs.contracts.prompt_patch import PROMPT_PATCH_REFUTATION_LIMIT
from rfs.evolution.direct import ExperienceStore, PromptPatchStore
from rfs.evolution.direct.prompt_patch import build_patches_for_experience
from rfs.workflows.paper_evolve_editable import _charge_injected_patches


def lifecycle_events(store: PromptPatchStore, patch_id: str) -> list[str]:
    return [str(item.get("event") or "") for item in store.events(patch_id)]


class PromptPatchRefutationStreakTests(unittest.TestCase):
    """The streak is read from the patch's own record, and a win clears it."""

    def test_losses_count_up_and_a_win_clears_the_streak(self):
        with tempfile.TemporaryDirectory() as name:
            store = PromptPatchStore(Path(name) / "library")

            self.assertEqual(0, store.refutation_streak("p1"))
            store.append_event("p1", "refuted", round=1)
            self.assertEqual(1, store.refutation_streak("p1"))
            store.append_event("p1", "refuted", round=2)
            self.assertEqual(2, store.refutation_streak("p1"))
            store.append_event("p1", "upheld", round=3)
            self.assertEqual(0, store.refutation_streak("p1"), "a win clears the streak rather than adding to a tally")
            store.append_event("p1", "refuted", round=4)
            self.assertEqual(1, store.refutation_streak("p1"))

    def test_one_patch_streak_is_not_another_patch_streak(self):
        with tempfile.TemporaryDirectory() as name:
            store = PromptPatchStore(Path(name) / "library")

            store.append_event("p1", "refuted", round=1)
            store.append_event("p1", "refuted", round=2)
            store.append_event("p2", "refuted", round=2)

            self.assertEqual(2, store.refutation_streak("p1"))
            self.assertEqual(1, store.refutation_streak("p2"))


class ChargeInjectedPatchesTests(unittest.TestCase):
    """A round is a measurement of the prompts it ran under, and of nothing else."""

    def _store(self, name: str) -> tuple[PromptPatchStore, Path]:
        root = Path(name)
        return PromptPatchStore(root / "library"), root / "events.jsonl"

    def _names(self, event_path: Path) -> list[str]:
        if not event_path.exists():
            return []
        return [json.loads(line)["event"] for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_a_win_is_written_as_upheld_and_clears_the_streak(self):
        with tempfile.TemporaryDirectory() as name:
            store, event_path = self._store(name)
            store.append_event("p1", "refuted", round=1)

            charged = _charge_injected_patches(patch_store=store, event_path=event_path, round_index=2, injected=["p1"], reason="pairwise_preferred")

            self.assertEqual(["p1"], charged)
            self.assertEqual(0, store.refutation_streak("p1"))
            self.assertIn("prompt_patch_upheld", self._names(event_path))
            self.assertNotIn("disabled", lifecycle_events(store, "p1"))

    def test_two_losses_in_a_row_retire_the_patch(self):
        with tempfile.TemporaryDirectory() as name:
            store, event_path = self._store(name)
            store.append_event("p1", "refuted", round=1)

            charged = _charge_injected_patches(patch_store=store, event_path=event_path, round_index=2, injected=["p1"], reason="no_significant_improvement")

            self.assertEqual(["p1"], charged)
            self.assertEqual(PROMPT_PATCH_REFUTATION_LIMIT, store.refutation_streak("p1"))
            self.assertIn("disabled", lifecycle_events(store, "p1"))
            written = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            refuted = [item for item in written if item["event"] == "prompt_patch_refuted"][-1]
            self.assertTrue(refuted["disabled"], refuted)
            self.assertEqual(PROMPT_PATCH_REFUTATION_LIMIT, refuted["refutations"])

    def test_one_loss_does_not_retire_a_patch_that_was_winning(self):
        with tempfile.TemporaryDirectory() as name:
            store, event_path = self._store(name)
            store.append_event("p1", "upheld", round=1)

            _charge_injected_patches(patch_store=store, event_path=event_path, round_index=2, injected=["p1"], reason="no_significant_improvement")

            self.assertEqual(1, store.refutation_streak("p1"))
            self.assertNotIn("disabled", lifecycle_events(store, "p1"))

    def test_a_round_that_could_not_be_measured_charges_nothing(self):
        with tempfile.TemporaryDirectory() as name:
            store, event_path = self._store(name)

            charged = _charge_injected_patches(patch_store=store, event_path=event_path, round_index=1, injected=["p1"], reason="figure_unchanged")

            self.assertEqual([], charged)
            self.assertEqual(0, store.refutation_streak("p1"))
            self.assertEqual([], self._names(event_path))

    def test_no_injected_patch_means_nothing_to_charge(self):
        with tempfile.TemporaryDirectory() as name:
            store, event_path = self._store(name)

            charged = _charge_injected_patches(patch_store=store, event_path=event_path, round_index=1, injected=[], reason="pairwise_not_preferred")

            self.assertEqual([], charged)


class SameIterationInjectionTests(unittest.TestCase):
    """A patch this iteration permitted is injected by the next round that plans its stage."""

    def _paper(self, target: Path) -> None:
        import fitz

        document = fitz.open()
        page = document.new_page(width=600, height=800)
        page.insert_text((50, 60), "A Patchable Study", fontsize=14)
        page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
        page.insert_textbox((50, 160, 550, 240), "We propose a frozen encoder that turns a paper into a symbolic figure plan.", fontsize=10)
        page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
        document.save(target)
        document.close()

    def _rendered_patch(self) -> dict:
        """A real, contract-valid patch, rendered from a row the library already accepts."""

        row = {
            "experience_id": "direct-exp-timing-01",
            "domain": "direct_editable",
            "scope": "slot",
            "memory_type": "layout",
            "apply_at": "repair",
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
        }
        rendered = build_patches_for_experience(row, authored={"apply_at": "repair", "condition": {}, "directive": row["guidance"]})
        self.assertEqual("rendered", rendered["reason"])
        return rendered["patch"]

    def _run(self, root: Path, library: Path) -> Path:
        from rfs.workflows import paper_evolve_editable
        from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable

        target = root / "paper.pdf"
        self._paper(target)
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
        patch = self._rendered_patch()

        class _ReflectingJudge:
            def __init__(self, judge):
                self._judge = judge

            def __getattr__(self, name):
                return getattr(self._judge, name)

            def reflect(self, outcome):
                return {"next_policy": "keep"}

        real_build = paper_evolve_editable.build_direct_judges

        def build(**kwargs):
            online, frozen = real_build(**kwargs)
            return _ReflectingJudge(online), frozen

        def render(*_args, **_kwargs):
            # The sentence a row becomes is measured elsewhere; what is under test here is where
            # the patch it becomes is injected, so the rendering is fixed to one permitted patch.
            return {
                "patch": dict(patch),
                "reason": "rendered",
                "authored_by": "model",
                "authored_discarded": "",
                "derived_from": "",
                "permit": {"granted": True, "status": "current_case_validated", "reason": "the round measured this repair improving the figure", "repair_gain": 0.2},
            }

        with mock_patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)), mock_patch(
            "rfs.workflows.paper_evolve_editable.build_direct_judges", build
        ), mock_patch("rfs.workflows.paper_evolve_editable.build_patches_for_experience", render):
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
                max_rounds=3,
                candidates_per_round=1,
                experience_enabled=True,
                experience_library=library,
                seed_experience={"experience_id": "direct-exp-reflect", "action": {"type": "reroute", "scope_type": "connector", "edge_ids": ["an-edge-that-is-not-here"]}},
                fixed_rounds=True,
            )
        return out

    def test_a_permitted_patch_is_injected_by_a_later_round_of_the_same_iteration(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            out = self._run(root, root / "library")
            events = [json.loads(line) for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            permitted = [event for event in events if event["event"] == "prompt_patch_permitted"]
            self.assertTrue(permitted, [event["event"] for event in events])
            patch_id = str(permitted[0]["patch_id"])
            permitted_round = int(permitted[0]["round"])

            later = [
                event
                for event in events
                if event["event"] == "prompt_patch_injected"
                and int(event.get("round") or 0) > permitted_round
                and patch_id in (event.get("patch_ids") or [])
            ]
            self.assertTrue(later, "the round that permitted the patch is not the only round that can inject it")
            self.assertTrue(later[0]["same_iteration_injection"], later[0])
            self.assertTrue(str(later[0]["stage"]), later[0])

    def test_the_timeline_says_which_version_each_round_ran_under(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            out = self._run(root, root / "library")
            timeline = json.loads((out / "prompt_patch_timeline.json").read_text(encoding="utf-8"))["timeline"]
            report = json.loads((out / "paper_evolve_editable_report.json").read_text(encoding="utf-8"))

            self.assertTrue(timeline)
            self.assertEqual(timeline, report["prompt_patch_timeline"])
            for reading in timeline:
                self.assertIn("round", reading)
                self.assertIn("stage", reading)
                self.assertIn("revision", reading)
                self.assertIn("generation", reading)
                self.assertIn("patch_ids", reading)
            rounds = sorted({int(reading["round"]) for reading in timeline})
            self.assertGreater(len(rounds), 1, "a cached round's prompt version is written down too")
            carrying = [reading for reading in timeline if reading["patch_ids"]]
            self.assertTrue(carrying, "the version a patch rode on has to be readable")
            self.assertTrue(all(int(reading["generation"]) > 0 for reading in carrying), carrying)
            # A stage resolved twice under one prompt is one reading, not two.
            seen = [(reading["round"], reading["stage"], reading["generation"], tuple(reading["patch_ids"])) for reading in timeline]
            self.assertEqual(len(seen), len(set(seen)))


if __name__ == "__main__":
    unittest.main()
