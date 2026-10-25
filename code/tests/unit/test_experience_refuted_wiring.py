"""The rollback branch of the evolution loop writes down what it measured.

A repair the Gate refused used to leave no trace at all: the Builder could already produce a
``refuted`` row, but only the accept branch ever called it, so the library filled up with
successes and the loop could only learn that every repair works. These tests pin the other
half down through the real round loop -- one measured round, at most one row -- and check the
row says out loud which condition it must not be reused under.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.contracts.evolution import validate_experience
from rfs.evolution.direct import ExperienceStore
from rfs.workflows.paper_evolve_editable import _record_experience_build_refusal, run_paper_evolve_editable


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."

# One promoted row, so retrieval offers it and a round can run under it.
MEASURED = {
    "experience_id": "direct-exp-refuted",
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

# An action that reaches an object this figure does not have: it executes, changes nothing,
# and therefore cannot be accepted, which is the rollback that must leave a row behind.
NO_OP_SEED = {"experience_id": "direct-exp-refuted", "action": {"type": "reroute", "scope_type": "connector", "edge_ids": ["an-edge-that-is-not-here"]}}


def write_paper(root: Path) -> Path:
    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Measured Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


def run_workflow(root: Path, library: Path, *, max_rounds: int, seed: dict | None = None) -> Path:
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
            experience_enabled=True,
            experience_library=library,
            seed_experience=seed,
        )
    return out


def read_events(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


class RefutedExperienceWiringTests(unittest.TestCase):
    def _prepared(self, name: str) -> tuple[Path, Path, ExperienceStore]:
        root = Path(name)
        write_paper(root)
        library = root / "library"
        store = ExperienceStore(library)
        self.assertTrue(store.append_result(dict(MEASURED))["ok"])
        return root, library, store

    def test_a_rolled_back_round_records_the_measurement_it_made(self):
        with tempfile.TemporaryDirectory() as name:
            root, library, store = self._prepared(name)
            out = run_workflow(root, library, max_rounds=2, seed=dict(NO_OP_SEED))
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            self.assertEqual([True, False], [item["accepted"] for item in rounds])

            written = out / "round_01" / "experience_refuted.json"
            self.assertTrue(written.exists(), "the rolled-back round left no Memory row")
            row = json.loads(written.read_text(encoding="utf-8"))

            self.assertEqual("refuted", row["status"])
            self.assertFalse(row["accepted"])
            self.assertLessEqual(float(row["repair_gain"]), 0.0)
            self.assertEqual(1, row["source_round"])
            self.assertTrue(validate_experience(row)["ok"], validate_experience(row)["errors"])

            # The row names the condition it may not be reused under, and says which measurement
            # produced it instead of repeating the explanation the model gave.
            self.assertTrue(row["excluded_conditions"])
            self.assertEqual("repair_gain_not_positive", row["excluded_conditions"][0]["reason"])
            self.assertIsInstance(row["condition_evidence"]["changed_object_ids"], list)
            self.assertTrue(row["condition_evidence"]["measured_by"])
            self.assertFalse(row["reported_reflection"]["verified"])
            self.assertTrue(row["provenance"]["measured"])
            self.assertEqual(1, row["provenance"]["round"])
            # The action reached nothing, and the row says so: the request is kept apart from
            # what the executor touched.
            self.assertEqual(0, row["observed_effect"]["changed_object_count"])
            self.assertEqual(["an-edge-that-is-not-here"], row["observed_effect"]["requested_object_ids"])

    def test_the_first_round_is_refused_for_having_nothing_to_compare_against(self):
        with tempfile.TemporaryDirectory() as name:
            root, library, store = self._prepared(name)
            out = run_workflow(root, library, max_rounds=1)
            rejected = json.loads((out / "round_00" / "experience_rejected.json").read_text(encoding="utf-8"))
            # The round is accepted because it is the starting point, and that is exactly why it
            # cannot become a row: no earlier accepted state means no gain to measure.
            self.assertEqual("no_previous_accepted_round", rejected["reason"])
            self.assertIsNone(rejected["before_score"])
            self.assertEqual([], [item for item in store.records() if item["status"] == "current_case_validated"])

    def test_the_row_is_stored_as_refuted_and_announced(self):
        with tempfile.TemporaryDirectory() as name:
            root, library, store = self._prepared(name)
            out = run_workflow(root, library, max_rounds=2, seed=dict(NO_OP_SEED))
            row = json.loads((out / "round_01" / "experience_refuted.json").read_text(encoding="utf-8"))
            stored = [item for item in store.records() if item["experience_id"] == row["experience_id"]]
            announced = [event for event in read_events(out) if event.get("event") == "experience_refuted"]
            self.assertEqual(["refuted"], [item["status"] for item in stored])
            self.assertEqual([row["experience_id"]], [event["experience_id"] for event in announced])
            self.assertEqual([row["repair_gain"]], [event["gain"] for event in announced])
            # The seeded row itself is not rewritten by this: a new row is appended, so the row
            # the round ran under stays readable exactly as it was written.
            self.assertIn("direct-exp-refuted", [item["experience_id"] for item in store.records()])

    def test_the_refuted_row_is_kept_out_of_the_state_it_was_measured_in(self):
        with tempfile.TemporaryDirectory() as name:
            root, library, store = self._prepared(name)
            out = run_workflow(root, library, max_rounds=2, seed=dict(NO_OP_SEED))
            row = json.loads((out / "round_01" / "experience_refuted.json").read_text(encoding="utf-8"))
            condition = row["excluded_conditions"][0]
            counts = {"low": 2, "medium": 6, "high": 12}
            matching = {
                "topology": condition["topology"],
                "node_count": counts[condition["node_count_bucket"]],
                "connector_count": counts[condition["connector_count_bucket"]],
            }
            result = store.retrieve_with_report(context=matching, statuses={"refuted"})
            self.assertNotIn(row["experience_id"], [item["evidence"]["experience_id"] for item in result["rows"]])
            self.assertIn(row["experience_id"], [item["experience_id"] for item in result["report"]["excluded_rows"]])
            # The exclusion is a condition, not a ban: a figure of another size still gets it.
            other = next(size for size in ("low", "high") if size != condition["node_count_bucket"])
            elsewhere = store.retrieve(context={**matching, "node_count": counts[other]}, statuses={"refuted"})
            self.assertIn(row["experience_id"], [item["evidence"]["experience_id"] for item in elsewhere])

    def test_the_loop_does_not_run_out_of_retries_recording_its_own_rows(self):
        with tempfile.TemporaryDirectory() as name:
            root, library, store = self._prepared(name)
            out = run_workflow(root, library, max_rounds=3, seed=dict(NO_OP_SEED))
            # ``action_retry_index`` counts the guidance a round ran under, so writing a new row
            # must not spend it: the seeded row would otherwise look exhausted after half as many
            # failed rounds and the loop would stop early.
            self.assertNotIn("experience_failed", [event["event"] for event in read_events(out)])
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            self.assertEqual(3, len(rounds))


class NotARepairTests(unittest.TestCase):
    """A round that won by drawing the figure again is not a refusal.

    Every accepted redraw used to be written up as ``experience_rejected``, because the move it won
    with is not in the catalogue and the store refuses a row naming one. The round kept moving
    forward and the only thing it left behind was a refusal notice, which buries the refusals that
    do mean something. The library is still left alone -- there is no move to hand another round --
    so what is pinned here is which of the two the record says happened.
    """

    def test_a_redraw_leaves_a_note_rather_than_a_refusal(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            events = root / "events.jsonl"
            _record_experience_build_refusal(
                event_path=events,
                round_dir=root,
                round_index=2,
                build={"reason": "not_a_catalogue_repair", "action": "initial", "experience": None},
                action={"type": "initial"},
            )
            note = json.loads((root / "experience_not_a_repair.json").read_text(encoding="utf-8"))
            self.assertEqual("initial", note["action"])
            self.assertFalse((root / "experience_rejected.json").exists())
            rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(["experience_not_a_repair"], [row["event"] for row in rows])
            self.assertEqual("initial", rows[0]["action"])

    def test_a_round_that_could_not_be_measured_is_still_written_as_a_refusal(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            events = root / "events.jsonl"
            _record_experience_build_refusal(
                event_path=events,
                round_dir=root,
                round_index=1,
                build={"reason": "no_after_measurement", "experience": None},
                action={"type": "reroute"},
            )
            self.assertFalse((root / "experience_not_a_repair.json").exists())
            rejected = json.loads((root / "experience_rejected.json").read_text(encoding="utf-8"))
            self.assertEqual("no_after_measurement", rejected["reason"])
            rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(["experience_rejected"], [row["event"] for row in rows])


if __name__ == "__main__":
    unittest.main()
