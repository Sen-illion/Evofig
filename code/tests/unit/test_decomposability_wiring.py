"""End-to-end wiring of the decomposability measurement into the evolution loop."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable


PARAGRAPH = "We propose a frozen encoder that turns a paper into a symbolic figure plan."


def write_paper(root: Path) -> Path:
    """Write one small paper the heuristic stages can plan from."""

    target = root / "paper.pdf"
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 60), "A Decomposed Study", fontsize=14)
    page.insert_textbox((50, 90, 550, 150), "1 Introduction", fontsize=11)
    page.insert_textbox((50, 160, 550, 240), PARAGRAPH, fontsize=10)
    page.insert_textbox((50, 280, 550, 340), "Figure 1: Overview of the proposed framework.", fontsize=10)
    document.save(target)
    document.close()
    return target


def run_workflow(root: Path, *, max_rounds: int, evolution_scope: str = "joint") -> Path:
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
            experience_enabled=False,
            evolution_scope=evolution_scope,
        )
    return out


def events(out: Path) -> list[dict]:
    path = out / "evolution_events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def soft_report(round_index: int, *, streak: int) -> dict:
    """Return the smallest report that makes the policy ask for a reflow."""

    return {
        "contract": "rfs.decomposability_report",
        "version": 1,
        "round": round_index,
        "status": "soft_violations",
        "metrics": {
            "unbound_scientific_nodes": 0,
            "slot_boundary_overflows": 0,
            "orphan_nodes": 0,
            "cross_panel_connectors": 3,
            "min_panel_occupancy": 0.2,
            "empty_region_ratio": 0.4,
            "max_node_slot_fanout": 2,
        },
        "thresholds": {"min_panel_occupancy": 0.45, "max_cross_panel_connectors": 2, "max_node_slot_fanout": 3},
        "violations": [{
            "metric": "cross_panel_connectors",
            "value": 3,
            "threshold": 2,
            "kind": "soft",
            "reason": "too many connectors cross a panel boundary",
            "object_ids": ["edge_01"],
            "target_connector_ids": ["edge_01"],
            "target_node_ids": ["node_01"],
        }],
        "repeat": {"soft_streak": streak},
        "provenance": {"panel_source": "synthesized"},
    }


class DecomposabilityWiringTests(unittest.TestCase):
    def test_a_run_records_a_report_per_candidate_and_a_decision_per_round(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            out = run_workflow(root, max_rounds=1)
            report = json.loads((out / "round_00" / "candidate_00" / "decomposability_report.json").read_text(encoding="utf-8"))
            decision = json.loads((out / "round_00" / "decomposability_action.json").read_text(encoding="utf-8"))
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]
            state = json.loads((out / "evolution_state.json").read_text(encoding="utf-8"))

        self.assertEqual(report["contract"], "rfs.decomposability_report")
        self.assertEqual(sorted(report["metrics"]), sorted([
            "unbound_scientific_nodes", "slot_boundary_overflows", "orphan_nodes",
            "cross_panel_connectors", "min_panel_occupancy", "empty_region_ratio",
            "max_node_slot_fanout",
        ]))
        self.assertEqual(decision["contract"], "rfs.decomposition_action")
        self.assertEqual(rounds[0]["decomposability_report"]["metrics"], report["metrics"])
        self.assertEqual(rounds[0]["decomposition_action"]["required"], decision["required"])
        self.assertEqual(state["decomposability_report"]["status"], report["status"])

    def test_a_clean_candidate_asks_for_nothing(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            out = run_workflow(root, max_rounds=2)
            decisions = [
                json.loads((out / f"round_{index:02d}" / "decomposability_action.json").read_text(encoding="utf-8"))
                for index in range(2)
            ]
            recorded = [item["event"] for item in events(out) if item["event"].startswith("decomposability")]

        self.assertTrue(all(not item["required"] for item in decisions), decisions)
        self.assertEqual(recorded, [])

    def test_a_soft_connector_break_lets_the_judge_choose_and_records_the_policy(self):
        """The Judge names the repair; the mechanism records what its own measurement wanted.

        A connector-only soft break is exactly what the policy would route by itself, and its
        opinion stays in the round's record. The move that actually runs is the Judge's, because
        the advisory Judge is what names repairs on this figure. What the mechanism keeps is the
        ladder: a soft break that repeats is escalated, and the escalation re-plans the structure
        even while the Judge is still asking only for a layout move.
        """

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            calls = {"count": 0}

            def stub(*args, **kwargs):
                streak = calls["count"] + 1
                calls["count"] += 1
                return soft_report(kwargs.get("round_index", -1), streak=streak)

            with patch("rfs.workflows.paper_evolve_editable.build_decomposability_report", side_effect=stub):
                out = run_workflow(root, max_rounds=3)

            decisions = [
                json.loads((out / f"round_{index:02d}" / "decomposability_action.json").read_text(encoding="utf-8"))
                for index in range(3)
            ]
            next_actions = [item["repair_action"].get("type") for item in json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]]
            rows = [json.loads(line) for line in (out / "judge_experiences.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            judge_asked = [
                str(action.get("type") or action.get("action") or "")
                for item in ((rows[0].get("judgement") or {}).get("stage_feedback") or [])
                for action in list(item.get("requested_actions") or []) + list(item.get("repair") or [])
                if isinstance(action, dict)
            ]
            recorded = [item["event"] for item in events(out) if item["event"].startswith("decomposability")]

        self.assertEqual(decisions[0]["action"], "reroute")
        self.assertFalse(decisions[0]["escalation"])
        self.assertTrue(judge_asked, "the advisory Judge named no repair to act on")
        self.assertEqual(next_actions[1], judge_asked[0], "the Judge's own repair is what the round ran")
        self.assertEqual(decisions[1]["action"], "replan_structure")
        self.assertTrue(decisions[1]["escalation"])
        self.assertEqual(next_actions[2], "replan_structure", "a hard escalation takes the turn ahead of the Judge")
        self.assertIn("decomposability_blocked", recorded)
        self.assertIn("decomposability_escalated", recorded)

    def test_the_mechanism_routes_when_the_judge_names_nothing(self):
        """The mechanism's connector repair is the fallback, so it runs when the Judge is silent."""

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            calls = {"count": 0}

            def stub(*args, **kwargs):
                streak = calls["count"] + 1
                calls["count"] += 1
                return soft_report(kwargs.get("round_index", -1), streak=streak)

            with patch("rfs.workflows.paper_evolve_editable.build_decomposability_report", side_effect=stub), patch(
                "rfs.evolution.direct.controller.select_bounded_action", return_value=None
            ):
                out = run_workflow(root, max_rounds=2)

            next_actions = [item["repair_action"].get("type") for item in json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]]
            first = json.loads((out / "round_00" / "decomposability_action.json").read_text(encoding="utf-8"))

        self.assertEqual(first["action"], "reroute")
        self.assertFalse(first["escalation"])
        self.assertEqual(next_actions[1], "reroute", "with no repair from the Judge, the connector is routed")

    def test_block_scope_keeps_every_repair_local(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            calls = {"count": 0}

            def stub(*args, **kwargs):
                calls["count"] += 1
                return soft_report(kwargs.get("round_index", -1), streak=calls["count"])

            with patch("rfs.workflows.paper_evolve_editable.build_decomposability_report", side_effect=stub):
                out = run_workflow(root, max_rounds=2, evolution_scope="block")

            decisions = [
                json.loads((out / f"round_{index:02d}" / "decomposability_action.json").read_text(encoding="utf-8"))
                for index in range(2)
            ]
            recorded = [item["event"] for item in events(out) if item["event"].startswith("decomposability")]

        # A connector repair is local, so block scope keeps it; only the global escalation
        # is downgraded, which is what the second round asks for.
        self.assertEqual([item["action"] for item in decisions], ["reroute", "relayout_slots"])
        self.assertTrue(all(not item["escalation"] for item in decisions), decisions)
        self.assertNotIn("decomposability_escalated", recorded)


if __name__ == "__main__":
    unittest.main()
