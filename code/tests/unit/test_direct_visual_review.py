"""Tests for direct draft visual review contracts and deterministic repairs."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rfs.composition import apply_direct_layout_repairs
from rfs.contracts.visual_review import normalize_visual_review, validate_visual_review
from rfs.evaluation.direct_visual_review import review_direct_draft
from rfs.composition.direct_connectors import resolve_direct_connectors
from rfs.workflows.paper_evolve_editable import _prioritize_repair_intents


class DirectVisualReviewTests(unittest.TestCase):
    def _program(self):
        return {
            "panels": [{"id": "panel", "title": "Panel", "bbox_percent": {"x": 0.0, "y": 0.0, "w": 0.8, "h": 0.5}}],
            "semantic_nodes": [
                {"id": "a", "panel_id": "panel", "label": "A", "bbox_percent": {"x": 0.35, "y": 0.1, "w": 0.1, "h": 0.1}},
                {"id": "b", "panel_id": "panel", "label": "B", "bbox_percent": {"x": 0.55, "y": 0.1, "w": 0.1, "h": 0.1}},
            ],
            "arrows": [], "slots": [], "text_program": {"items": []},
        }

    def test_heuristic_review_detects_horizontal_imbalance(self):
        review, validation = review_direct_draft(self._program(), "unused.png", mode="heuristic")
        self.assertTrue(validation["ok"])
        self.assertEqual(review["feedback"][0]["issue_type"], "horizontal_imbalance")
        self.assertEqual(review["repair_intents"][0]["action"], "rebalance_panel")

    def test_rebalance_moves_group_without_changing_relative_spacing(self):
        program = self._program()
        review, _ = review_direct_draft(program, "unused.png", mode="heuristic")
        repaired, report = apply_direct_layout_repairs(program, review)
        before = [x["bbox_percent"]["x"] for x in program["semantic_nodes"]]
        after = [x["bbox_percent"]["x"] for x in repaired["semantic_nodes"]]
        self.assertGreaterEqual(report["applied_count"], 1)
        self.assertAlmostEqual(before[1] - before[0], after[1] - after[0])
        self.assertLess(after[0], before[0])

    def test_invalid_object_reference_is_reported(self):
        review = normalize_visual_review({"feedback": [{"issue_type": "crowding", "object_ids": ["missing"]}]})
        report = validate_visual_review(review, valid_object_ids={"a"}, valid_panel_ids={"panel"})
        self.assertFalse(report["ok"])

    def test_horizontal_distribution_uses_panel_width(self):
        program = self._program()
        review = {"repair_intents": [{"action": "distribute_horizontal", "panel_id": "panel", "object_ids": ["a", "b"]}]}
        repaired, report = apply_direct_layout_repairs(program, review)
        xs = [x["bbox_percent"]["x"] for x in repaired["semantic_nodes"]]
        self.assertEqual(report["applied_count"], 1)
        self.assertLess(xs[0], 0.1)
        self.assertGreater(xs[1], 0.6)

    def test_distribution_skips_when_panel_cannot_fit_members(self):
        program = self._program()
        program["semantic_nodes"][0]["bbox_percent"]["w"] = 0.39
        program["semantic_nodes"][1]["bbox_percent"]["w"] = 0.39
        repaired, report = apply_direct_layout_repairs(program, {"repair_intents": [{"action": "distribute_horizontal", "panel_id": "panel", "object_ids": ["a", "b"]}]})
        self.assertEqual(report["applied_count"], 0)
        self.assertEqual(repaired["semantic_nodes"], program["semantic_nodes"])

    def test_translate_group_preserves_relative_spacing_at_boundary(self):
        program = self._program()
        before = [node["bbox_percent"]["x"] for node in program["semantic_nodes"]]
        repaired, _ = apply_direct_layout_repairs(program, {"repair_intents": [{"action": "translate_group", "panel_id": "panel", "object_ids": ["a", "b"], "axis": "horizontal", "amount": 0.15}]})
        after = [node["bbox_percent"]["x"] for node in repaired["semantic_nodes"]]
        self.assertAlmostEqual(before[1] - before[0], after[1] - after[0])

    def test_connector_lane_action_changes_actual_path(self):
        program = self._program()
        program["arrows"] = [{"id": "edge", "path_percent": [[0.1, 0.2], [0.5, 0.2]]}]
        repaired, report = apply_direct_layout_repairs(program, {"repair_intents": [{"action": "reserve_connector_lane", "object_ids": ["edge"], "axis": "vertical", "amount": 0.03}]})
        self.assertEqual(report["applied_count"], 1)
        self.assertEqual(repaired["arrows"][0]["path_percent"], [])
        self.assertEqual(report["requested_connector_ids"], ["edge"])

    def test_connector_lane_offset_survives_authoritative_reroute(self):
        program = self._program()
        program["arrows"] = [{"id": "edge", "source": "a", "target": "b", "route_lane_offset_percent": {"x": 0.0, "y": 0.03}}]
        resolved, _ = resolve_direct_connectors(program)
        self.assertAlmostEqual(resolved["arrows"][0]["path_percent"][0][1], 0.18)

    def test_repair_lane_invalidates_obsolete_path_for_reroute(self):
        program = self._program()
        program["arrows"] = [{"id": "edge", "source": "a", "target": "b", "path_percent": [[0.1, 0.1], [0.7, 0.7]]}]
        repaired, report = apply_direct_layout_repairs(program, {"repair_intents": [{"action": "reserve_connector_lane", "object_ids": ["edge"], "axis": "vertical", "amount": 0.03}]})
        self.assertEqual(report["requested_connector_ids"], ["edge"])
        self.assertEqual(repaired["arrows"][0]["path_percent"], [])

    def test_mixed_intents_have_explicit_priority(self):
        review = _prioritize_repair_intents({"repair_intents": [{"action": "rebalance_panel"}, {"action": "replan_structure"}, {"action": "resize_node"}]})
        self.assertEqual([item["action"] for item in review["repair_intents"]], ["replan_structure", "resize_node", "rebalance_panel"])
        self.assertIn("takes precedence", review["repair_policy"])


class VisualReviewCallLedgerTests(unittest.TestCase):
    """The review hands back what its own call cost, so a round can price its minutes."""

    def _program(self):
        return {"panels": [{"id": "panel"}], "semantic_nodes": [], "arrows": [], "slots": [], "text_program": {"items": []}}

    def test_the_call_reports_its_own_timing(self):
        def answer(prompt, images, **kwargs):
            kwargs["call_metadata"].update({"success": True, "elapsed_seconds": 61.5, "prompt_tokens": 900, "completion_tokens": 120})
            return {"status": "pass", "feedback": [], "repair_intents": []}

        with tempfile.TemporaryDirectory() as temp:
            preview = Path(temp) / "preview.png"
            preview.write_bytes(b"png")
            metadata = {}
            with mock.patch("rfs.evaluation.direct_visual_review.call_vlm_json", side_effect=answer):
                _, validation = review_direct_draft(self._program(), preview, call_metadata=metadata)
        self.assertEqual(validation["provider"]["elapsed_seconds"], 61.5)
        self.assertEqual(validation["provider"]["completion_tokens"], 120)

    def test_a_repeat_review_says_it_was_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            preview = Path(temp) / "preview.png"
            preview.write_bytes(b"png")
            metadata = {}
            with mock.patch("rfs.evaluation.direct_visual_review.call_vlm_json", return_value={"status": "pass", "feedback": [], "repair_intents": []}) as call:
                review_direct_draft(self._program(), preview)
                _, validation = review_direct_draft(self._program(), preview, call_metadata=metadata)
            call.assert_called_once()
        self.assertTrue(validation["provider"]["cache_hit"])
        self.assertEqual(validation["provider"]["elapsed_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
