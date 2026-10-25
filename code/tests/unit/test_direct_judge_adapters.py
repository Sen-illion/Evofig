"""Tests for replaceable direct-evolution Judge adapters."""

import dataclasses
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rfs.evaluation.direct_judges import DirectJudgeInput, HeuristicDirectJudge, build_direct_judges
from rfs.evolution.direct import ExperienceStore


def _valid_input() -> DirectJudgeInput:
    return DirectJudgeInput(
        method={"modules": [{"id": "n"}], "relations": []},
        layout={"panels": [{"id": "p", "bbox_percent": {"x": 0, "y": 0, "w": 1, "h": 1}}], "nodes": [{"id": "n", "panel_id": "p", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6}}], "connectors": []},
        icons={"icons": [{"asset_id": "icon_n", "semantic_node_ids": ["n"], "prompt": "n"}]},
        program={"panels": [{"id": "p", "bbox_percent": {"x": 0, "y": 0, "w": 1, "h": 1}}], "semantic_nodes": [{"id": "n", "panel_id": "p", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6}}], "slots": [{"id": "slot_n", "asset_id": "icon_n", "bbox_percent": {"x": 0.15, "y": 0.15, "w": 0.3, "h": 0.3}}], "text_program": {"items": [{"id": "t", "text": "n"}]}, "arrows": []},
    )


class DirectJudgeAdapterTests(unittest.TestCase):
    def test_reflect_reads_a_new_blocker_count_without_raising(self) -> None:
        online, _ = build_direct_judges(mode="heuristic")
        outcome = {"before": {"overall_score": 0.5}, "after": {"overall_score": 0.75}, "new_blockers": 3}
        reflection = online.reflect(outcome)
        self.assertEqual(reflection["outcome"], "regressed")
        self.assertEqual(reflection["failure_type"], "new_blocker")

    def test_grounding_gate_blocks_before_heuristic_judge(self) -> None:
        online, _ = build_direct_judges(mode="heuristic")
        result = online.evaluate(DirectJudgeInput(
            method={}, layout={}, icons={}, program={},
            grounding_gate={"status": "blocked", "blocking_issues": ["missing required node"]},
        ))
        self.assertEqual("blocked", result["status"])
        self.assertTrue(result["gate_only"])
        # The advisory role is not asked for a score, so it does not measure one: the wording of
        # the gate stays in the answer as a reported value and never becomes the score the loop
        # compares rounds with. Offline heuristic still has to name a repair, because there is
        # no model; VLM advisory does not take this hardcoded action.
        self.assertIsNone(result["overall_score"])
        self.assertFalse(result["score_measured"])
        self.assertEqual("reanalyze_semantics", result["repair"][0]["action"])

    def test_the_scoring_role_is_never_short_circuited_by_the_gate(self) -> None:
        _, frozen = build_direct_judges(mode="bootstrap", online_model="ordinary-vlm", frozen_model="ordinary-vlm")
        gate = {"status": "fault", "blocking_issues": ["canvas occupancy is too low"], "dimension_scores": {"grounding_gate": 0.2}}
        judge_input = dataclasses.replace(_valid_input(), grounding_gate=gate)
        with patch("rfs.evaluation.direct_judges.call_vlm_json", return_value={"overall_score": 0.62, "status": "fault", "dimension_scores": {"layout": 0.62}, "blocking_issues": ["canvas occupancy is too low"]}) as model:
            result = frozen.evaluate(judge_input)
        self.assertTrue(model.called)
        self.assertEqual(0.62, result["overall_score"])
        self.assertTrue(result["score_measured"])
        self.assertFalse(result["gate_only"])
        # The gate's own finding still counts: it is merged into the blockers, not dropped.
        self.assertIn("canvas occupancy is too low", result["blocking_issues"])

    def test_the_advisory_role_still_answers_when_the_model_volunteers_a_score(self) -> None:
        online, _ = build_direct_judges(mode="bootstrap", online_model="ordinary-vlm")
        passing_gate = {"status": "pass", "blocking_issues": [], "stage_feedback": [], "dimension_scores": {}}
        with patch.object(HeuristicDirectJudge, "evaluate", return_value=passing_gate), patch("rfs.evaluation.direct_judges.call_vlm_json", return_value={"overall_score": 0.9, "status": "fault", "stage_feedback": [{"scope_type": "slot", "slot_ids": ["slot_n"], "problem": "icon is cropped", "requested_actions": [{"action": "regen_assets"}], "severity": "major"}]}):
            result = online.evaluate(_valid_input())
        self.assertIsNone(result["overall_score"])
        self.assertFalse(result["score_measured"])
        self.assertEqual(0.9, result["reported_score"])
        self.assertEqual("regen_assets", result["stage_feedback"][0]["requested_actions"][0]["action"])
        self.assertEqual(["slot_n"], result["stage_feedback"][0]["requested_actions"][0]["slot_ids"])

    def test_the_gate_names_the_defects_it_found_so_two_answers_can_be_compared(self):
        """The gate's half of the blocker list is the only half a rerun reproduces."""

        _, frozen = build_direct_judges(mode="heuristic")
        broken = DirectJudgeInput(
            method={"modules": [{"id": "n"}], "relations": []},
            layout={"panels": [], "nodes": [], "connectors": []},
            icons={"icons": [{"asset_id": "icon_n", "semantic_node_ids": ["n"], "prompt": "n"}]},
            program={"panels": [], "semantic_nodes": [], "slots": [], "text_program": {"items": []}, "arrows": []},
        )

        result = frozen.evaluate(broken)

        self.assertTrue(result["deterministic_blockers"])
        # A defect is named by where it was found and what repairs it, never by the sentence
        # the gate wrote, because that sentence carries the counts it measured.
        for identity in result["deterministic_blockers"]:
            self.assertRegex(identity, r"^[a-z_]*:[a-z_]*$")
        self.assertEqual(result["deterministic_blockers"], frozen.evaluate(broken)["deterministic_blockers"])

    def test_a_model_answer_carries_the_gate_defects_and_not_its_own_prose(self):
        _, frozen = build_direct_judges(mode="bootstrap", frozen_model="ordinary-vlm")
        gate = {"status": "fault", "blocking_issues": ["canvas occupancy is too low"], "deterministic_blockers": ["interface:relayout_slots"], "dimension_scores": {}}
        with patch.object(HeuristicDirectJudge, "evaluate", return_value=gate), patch(
            "rfs.evaluation.direct_judges.call_vlm_json",
            return_value={"overall_score": 0.5, "status": "fault", "blocking_issues": ["the icons look alike", "the background is checkered"]},
        ):
            result = frozen.evaluate(_valid_input())

        self.assertEqual(["interface:relayout_slots"], result["deterministic_blockers"])
        self.assertEqual(3, result["blocking_issue_count"])

    def test_a_grounding_refusal_is_a_named_defect_too(self):
        online, _ = build_direct_judges(mode="heuristic")

        result = online.evaluate(DirectJudgeInput(
            method={}, layout={}, icons={}, program={},
            grounding_gate={"status": "blocked", "blocking_issues": ["missing required node"]},
        ))

        self.assertEqual(["grounding_gate"], result["deterministic_blockers"])

    def test_heuristic_pair_is_independent_by_role(self):
        online, frozen = build_direct_judges(mode="heuristic")
        self.assertNotEqual(online.role, frozen.role)

    def test_bootstrap_vlm_uses_stable_result_contract(self):
        online, _ = build_direct_judges(mode="bootstrap", online_model="ordinary-vlm")
        with patch("rfs.evaluation.direct_judges.call_vlm_json", return_value={"overall_score": 0.8, "status": "pass", "dimension_scores": {"layout": 0.8}, "stage_feedback": []}):
            result = online.evaluate(_valid_input())
        self.assertEqual(result["contract"], "rfs.judge_result")
        self.assertFalse(result["trained"])

    def test_experience_store_tracks_lifecycle_without_rewriting_record(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ExperienceStore(Path(temp))
            record = {"experience_id": "e1", "status": "candidate", "domain": "direct_editable", "scope": "slot", "condition": {"topology": "linear"}, "guidance": "Keep the repaired part on the same reading path as the rest of the figure so the flow stays single-direction.",
 "hypothesis": "Targeted icon repair helps.", "action": {"type": "regen_assets"}, "graph_scope": {"slot_ids": ["slot_n"]}, "outcome": {"after_score": 0.8}, "repair_gain": 0.25, "provenance": {"source": "direct_evolution_process", "run": "paper.pdf", "round": 1, "measured": True, "before_score": 0.55, "after_score": 0.8}}
            self.assertTrue(store.append(record))
            store.transition("e1", "promoted", reason="cross-paper pass")
            self.assertEqual(store.lifecycle_status("e1"), "promoted")
            found = store.retrieve(domain="direct_editable", context={"topology": "linear"}, statuses={"promoted"})
            self.assertEqual(len(found), 1)


if __name__ == "__main__":
    unittest.main()
