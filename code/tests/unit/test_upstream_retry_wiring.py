"""Wiring of the upstream re-run into a real round: a hard break really re-plans the structure."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from rfs.workflows import paper_evolve_editable
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


def hard_report(round_index: int) -> dict:
    """The smallest report whose violation is hard, so the policy escalates."""

    return {
        "contract": "rfs.decomposability_report",
        "version": 1,
        "round": round_index,
        "status": "hard_violations",
        "metrics": {
            "unbound_scientific_nodes": 1,
            "slot_boundary_overflows": 0,
            "orphan_nodes": 1,
            "cross_panel_connectors": 0,
            "min_panel_occupancy": 0.5,
            "empty_region_ratio": 0.2,
            "max_node_slot_fanout": 1,
        },
        "thresholds": {"min_panel_occupancy": 0.45, "max_cross_panel_connectors": 2, "max_node_slot_fanout": 3},
        "violations": [{
            "metric": "orphan_nodes",
            "value": 1,
            "threshold": 0,
            "kind": "hard",
            "reason": "a scientific node is not bound to any panel",
            "object_ids": ["node_01"],
            "target_node_ids": ["node_01"],
            "target_connector_ids": [],
        }],
        "repeat": {"soft_streak": 0},
        "provenance": {"panel_source": "synthesized"},
    }


def _moved_design(layout: dict) -> dict:
    """A design that is the accepted layout with one node moved, so the re-run really differs."""

    nodes = [dict(node) for node in layout.get("nodes", [])]
    moved = dict(nodes[0])
    bbox = dict(moved.get("bbox_percent") or {})
    bbox["x"] = float(bbox.get("x") or 0.1) + 0.05
    moved["bbox_percent"] = bbox
    return {
        "nodes": [moved, *nodes[1:]],
        "panels": [dict(panel) for panel in layout.get("panels", [])],
        "connectors": [dict(edge) for edge in layout.get("connectors", [])],
    }


class UpstreamRetryWiringTests(unittest.TestCase):
    def _run(self, root: Path, *, upstream_retry_enabled: bool = True) -> dict:
        captured: dict = {}
        calls = {"plan": 0, "retry": 0}
        real_structure = paper_evolve_editable.plan_paper_figure_structure

        def record_structure(*args, **kwargs):
            result = real_structure(*args, **kwargs)
            captured["structure"] = result
            return result

        def fake_plan_structure(method, reference_profile, model_reference=None, **kwargs):
            if str(kwargs.get("replan_feedback") or "").strip():
                calls["retry"] += 1
                return _moved_design(captured["structure"]["figure_layout_plan"]), {"mode": "vlm", "retry": True}
            calls["plan"] += 1
            return {}, {"mode": "vlm"}

        with patch.object(paper_evolve_editable, "build_progress_reporter", lambda *a, **k: (lambda event: None)), \
                patch.object(paper_evolve_editable, "plan_scientific_hierarchy", lambda *a, **k: ({}, {"mode": "vlm"})), \
                patch.object(paper_evolve_editable, "plan_paper_figure_structure", side_effect=record_structure), \
                patch.object(paper_evolve_editable, "plan_structure_vlm", side_effect=fake_plan_structure), \
                patch.object(paper_evolve_editable, "build_decomposability_report", side_effect=lambda *a, **k: hard_report(k.get("round_index", 0))):
            out = root / "run"
            run_paper_evolve_editable(
                paper=root / "paper.pdf",
                out=out,
                planner_mode="heuristic",
                method_mode="heuristic",
                structure_mode="vlm",
                text_plan_mode="heuristic",
                ppt_plan_mode="heuristic",
                color_theme_mode="fallback",
                icon_mode="placeholder",
                visual_review_mode="heuristic",
                judge_mode="heuristic",
                max_rounds=2,
                candidates_per_round=1,
                experience_enabled=False,
                upstream_retry_enabled=upstream_retry_enabled,
            )
        return {"out": out, "calls": calls}

    def test_a_hard_break_re_runs_the_structure_planning_and_the_figure_really_moves(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            result = self._run(root)
            out = result["out"]
            events = [json.loads(line)["event"] for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            retry_report = json.loads((out / "round_01" / "upstream_retry.json").read_text(encoding="utf-8"))
            execution = json.loads((out / "round_01" / "action_execution.json").read_text(encoding="utf-8"))
            figure_diff = json.loads((out / "round_01" / "candidate_00" / "figure_object_diff.json").read_text(encoding="utf-8"))
            rounds = json.loads((out / "evolution_rounds.json").read_text(encoding="utf-8"))["rounds"]

        # The structure stage is asked twice: once for the first draft, once to answer the break.
        self.assertEqual(result["calls"], {"plan": 1, "retry": 1})
        self.assertEqual(retry_report["kind"], "structure")
        self.assertEqual(retry_report["action_type"], "resegment")
        self.assertTrue(retry_report["attempted"])
        self.assertTrue(retry_report["applied"])
        self.assertTrue(retry_report["changed"])
        self.assertNotEqual(retry_report["before_hash"], retry_report["after_hash"])
        self.assertIn("orphan_nodes", retry_report["feedback"])
        self.assertIn("upstream_retry_started", events)
        self.assertIn("upstream_retry_completed", events)
        # The record says the round moved a real figure, not that it merely compiled one.
        self.assertTrue(execution["applied"])
        self.assertEqual(execution["upstream_retry"], "structure")
        self.assertTrue(execution["changed_node_ids"])
        self.assertTrue(figure_diff["changed"])
        self.assertTrue(figure_diff["changed_object_ids"])
        self.assertTrue(rounds[1]["upstream_retry"]["applied"])
        self.assertEqual(rounds[1]["repair_action"]["type"], "resegment")

    def test_the_switch_records_a_refusal_and_keeps_the_bounded_action(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            write_paper(root)
            result = self._run(root, upstream_retry_enabled=False)
            out = result["out"]
            events = [json.loads(line) for line in (out / "evolution_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            retry_report = json.loads((out / "round_01" / "upstream_retry.json").read_text(encoding="utf-8"))
            execution = json.loads((out / "round_01" / "action_execution.json").read_text(encoding="utf-8"))

        self.assertEqual(result["calls"], {"plan": 1, "retry": 0})
        self.assertEqual(retry_report["kind"], "structure")
        self.assertFalse(retry_report["attempted"])
        self.assertIn("switched off", retry_report["declined_reason"])
        declined = [item for item in events if item["event"] == "upstream_retry_declined"]
        self.assertTrue(declined)
        self.assertFalse(execution["applied"])
        self.assertIn("switched off", execution.get("declined_reason") or "")


if __name__ == "__main__":
    unittest.main()
