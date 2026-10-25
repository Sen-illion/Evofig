"""Tests for compact post-draft repair planning."""

import unittest
from unittest.mock import patch

from rfs.planning.feedback_repair_planner import plan_feedback_repairs_vlm


class FeedbackRepairPlannerTests(unittest.TestCase):
    @patch("rfs.planning.feedback_repair_planner.call_vlm_json")
    def test_sends_only_affected_panel_and_returns_bounded_actions(self, call):
        call.return_value = {"repair_scope": "panel", "repair_intents": [{"action": "distribute_vertical", "panel_id": "p1", "object_ids": ["a", "b"], "reason": "reduce whitespace"}]}
        program = {"panels": [{"id": "p1", "bbox_percent": {"x": 0, "y": 0, "w": .5, "h": 1}}, {"id": "p2", "bbox_percent": {"x": .5, "y": 0, "w": .5, "h": 1}}], "semantic_nodes": [{"id": "a", "panel_id": "p1", "bbox_percent": {"x": .1, "y": .1, "w": .1, "h": .1}}, {"id": "b", "panel_id": "p1", "bbox_percent": {"x": .1, "y": .7, "w": .1, "h": .1}}, {"id": "c", "panel_id": "p2", "bbox_percent": {"x": .6, "y": .2, "w": .1, "h": .1}}], "arrows": [], "text_program": {"items": []}}
        plan, report = plan_feedback_repairs_vlm(program, {"feedback": [{"issue_type": "excess_whitespace", "panel_id": "p1", "object_ids": ["a", "b"]}]})
        self.assertTrue(report["ok"])
        self.assertEqual(plan["repair_scope"], "panel")
        prompt = call.call_args.args[0]
        self.assertIn('"id":"a"', prompt)
        self.assertNotIn('"id":"c"', prompt)
        self.assertLess(report["prompt_chars"], 5000)

    @patch("rfs.planning.feedback_repair_planner.call_vlm_json")
    def test_the_call_reports_its_own_timing(self, call):
        def answer(prompt, images, **kwargs):
            kwargs["call_metadata"].update({"success": True, "elapsed_seconds": 27.0, "completion_tokens": 300})
            return {"repair_scope": "local", "repair_intents": []}

        call.side_effect = answer
        metadata = {}
        _, report = plan_feedback_repairs_vlm({"panels": [], "semantic_nodes": [], "arrows": [], "text_program": {"items": []}}, {"feedback": []}, call_metadata=metadata)
        self.assertEqual(report["provider"]["elapsed_seconds"], 27.0)


if __name__ == "__main__":
    unittest.main()
