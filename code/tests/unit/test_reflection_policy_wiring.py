"""End-to-end wiring of the reflection's next policy into the evolution loop.

The reflection used to be a note nothing read. These tests pin down the four things it can
now do through the real round loop: keep a row out for the condition a round measured,
disable the row outright, hand a legal revision to the next round, and record a revision
that is not one the loop may run instead of running it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.evolution.direct import ExperienceStore
from rfs.workflows import paper_evolve_editable
from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."

# One promoted row, so retrieval can offer it and a round can run under it.
MEASURED = {
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
    "provenance": {"source": "direct_evolution_process", "run": "a.pdf", "round": 1, "measured": True, "before_score": 0.5, "after_score": 0.75},
}


class _ReflectingJudge:
    """The real judge, with a stateable reflection instead of whatever the model said."""

    def __init__(self, judge, reflection: dict):
        self._judge = judge
        self._reflection = dict(reflection)

    def __getattr__(self, name):
        return getattr(self._judge, name)

    def reflect(self, outcome):
        return dict(self._reflection)


class _PassingJudge:
    """The real acceptance judge, with every answer reporting pass.

    The loop stops by itself once an accepted round passes, so reaching that decision means
    handing it a figure it passes.
    """

    def __init__(self, judge):
        self._judge = judge

    def __getattr__(self, name):
        return getattr(self._judge, name)

    def evaluate(self, judge_input):
        result = self._judge.evaluate(judge_input)
        return {**result, "status": "pass"} if isinstance(result, dict) else result


def write_paper(root: Path) -> Path:
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Reflective Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


def run_workflow(root: Path, library: Path, *, max_rounds: int, reflection: dict | None = None, seed: dict | None = None, fixed_rounds: bool = False, passing_acceptance: bool = False) -> Path:
    out = root / "run"
    real_build = paper_evolve_editable.build_direct_judges

    def build(**kwargs):
        online, frozen = real_build(**kwargs)
        if reflection:
            online = _ReflectingJudge(online, reflection)
        if passing_acceptance:
            frozen = _PassingJudge(frozen)
        return online, frozen

    with patch("rfs.workflows.paper_evolve_editable.build_progress_reporter", lambda *a, **k: (lambda event: None)), patch(
        "rfs.workflows.paper_evolve_editable.build_direct_judges", build
    ):
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
            experience_enabled=True,
            experience_library=library,
            seed_experience=seed,
            fixed_rounds=fixed_rounds,
        )
    return out


# An action that reaches an object this figure does not have: it executes, changes nothing,
# and therefore cannot be accepted, which is the rollback the reflection then comments on.
NO_OP_SEED = {"experience_id": "direct-exp-reflect", "action": {"type": "reroute", "scope_type": "connector", "edge_ids": ["an-edge-that-is-not-here"]}}


class ReflectionPolicyWiringTests(unittest.TestCase):
    def _prepared(self, name: str) -> tuple[Path, Path, ExperienceStore]:
        root = Path(name)
        write_paper(root)
        library = root / "library"
        store = ExperienceStore(library)
        self.assertTrue(store.append_result(dict(MEASURED))["ok"])
        return root, library, store

    def test_narrow_scope_appends_the_measured_condition_and_keeps_the_row_out(self):
        with tempfile.TemporaryDirectory() as name:
            root, library, store = self._prepared(name)
            out = run_workflow(root, library, max_rounds=2, reflection={"next_policy": "narrow_scope"}, seed=dict(NO_OP_SEED))
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            event_names = [json.loads(line)["event"] for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            added = [event for event in store.events("direct-exp-reflect") if event.get("event") == "experience_condition_excluded"]
            self.assertEqual([True, False], [item["accepted"] for item in rounds])
            self.assertEqual("narrow_scope", rounds[1]["reflection_policy"]["next_policy"])
            self.assertIn("reflection_policy_applied", event_names)
            self.assertEqual(1, len(added))
            condition = added[0]["condition"]
            self.assertEqual("reflection_next_policy_narrow_scope", condition["reason"])
            # The row itself is never rewritten: the condition lives in the event log, and
            # the appended condition is what retrieval now honours.
            self.assertEqual("promoted", store.records()[0]["status"])
            counts = {"low": 2, "medium": 6, "high": 12}
            matching = {
                "topology": condition["topology"],
                "node_count": counts[condition["node_count_bucket"]],
                "connector_count": counts[condition["connector_count_bucket"]],
            }
            # The round the row guided also went back to the Gate, so its lifecycle moved on;
            # asking for every status shows the exclusion is what keeps it out, not the rung.
            result = store.retrieve_with_report(context=matching, statuses={"promoted", "refuted"})
            self.assertEqual([], [row["evidence"]["experience_id"] for row in result["rows"]])
            # The rolled-back round wrote a refuted row of its own, which is excluded here too:
            # both rows name this measured condition, so both are kept out.
            self.assertIn("direct-exp-reflect", [row["experience_id"] for row in result["report"]["excluded_rows"]])
            # A size the row was not excluded at still gets it. The rolled-back round appended a
            # refuted row of its own, which is offered here too, so only the guided row is asked for.
            other = next(size for size in ("low", "high") if size != condition["node_count_bucket"])
            elsewhere = store.retrieve(context={**matching, "node_count": counts[other]}, statuses={"promoted", "refuted"})
            self.assertIn("direct-exp-reflect", [row["evidence"]["experience_id"] for row in elsewhere])

    def test_disable_takes_the_row_out_of_retrieval(self):
        with tempfile.TemporaryDirectory() as name:
            root, library, store = self._prepared(name)
            out = run_workflow(root, library, max_rounds=2, reflection={"next_policy": "disable"}, seed=dict(NO_OP_SEED))
            out_events = [json.loads(line)["event"] for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            disabled = [event for event in store.events("direct-exp-reflect") if event.get("status") == "disabled"]
            rows = store.retrieve(context={"topology": "pipeline", "node_count": 4, "connector_count": 3})
            self.assertIn("reflection_policy_applied", out_events)
            self.assertEqual(1, len(disabled))
            self.assertEqual("reflection_next_policy_disable", disabled[0]["reason"])
            self.assertEqual([], [row["evidence"]["experience_id"] for row in rows])
            self.assertNotEqual("promoted", store.lifecycle_status("direct-exp-reflect"))

    def test_a_legal_revision_is_the_next_rounds_action(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            library = root / "library"
            out = run_workflow(
                root,
                library,
                max_rounds=3,
                reflection={"next_policy": "revise", "revised_action": {"type": "resize_nodes", "scope_type": "slot", "slot_ids": ["icon_n"], "factor": 1.08, "dx": 0.0, "dy": 0.0}},
                fixed_rounds=True,
            )
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            self.assertEqual("resize_nodes", rounds[1]["reflection_policy"]["revised_action"]["type"])
            self.assertEqual("resize_nodes", rounds[2]["repair_action"]["type"])

    def test_an_unsupported_revision_is_rejected_instead_of_run(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            library = root / "library"
            out = run_workflow(root, library, max_rounds=3, reflection={"next_policy": "revise", "revised_action": {"type": "redraw_everything"}}, fixed_rounds=True)
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            event_names = [json.loads(line)["event"] for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertIn("revised_action_rejected", event_names)
        self.assertIsNone(rounds[1]["reflection_policy"]["revised_action"])
        # The rejected name is never executed: once the revision is dropped nothing else is
        # proposed, so the loop stops instead of shaping the revision into a real repair.
        self.assertNotIn("redraw_everything", [str((item.get("repair_action") or {}).get("type") or "") for item in rounds])

    def test_an_accepted_round_that_passes_ends_the_run(self):
        with tempfile.TemporaryDirectory() as name:
            root, library, _ = self._prepared(name)
            out = run_workflow(root, library, max_rounds=3, passing_acceptance=True)
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            self.assertEqual(1, len(rounds))
            self.assertTrue(rounds[0]["accepted"])

    def test_a_run_with_nothing_left_to_repair_stops_at_repair_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root, library, _ = self._prepared(name)
            out = run_workflow(root, library, max_rounds=3)
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            event_names = [json.loads(line)["event"] for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        # Nothing is left to prescribe after the first rollback, so the loop ends early rather
        # than spending the remaining rounds on a nudge nobody asked for.
        self.assertLess(len(rounds), 3)
        self.assertIn("repair_refused", event_names)


if __name__ == "__main__":
    unittest.main()
