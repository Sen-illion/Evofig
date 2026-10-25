"""Tests for stage-specific VLM context compaction."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rfs.contracts.vlm_context import compact_document_index, compact_evidence, compact_method, compact_review
from rfs.planning.ppt_composition_planner import plan_ppt_composition_vlm


class VlmContextTests(unittest.TestCase):
    def test_document_index_drops_internal_fields(self):
        result = compact_document_index({"sections": [{"title": "Method", "page": 2, "block_id": "secret"}], "formulas": [{"text": "x"}]})
        self.assertEqual(result["sections"], [{"title": "Method", "page": 2}])
        self.assertNotIn("formulas", result)

    def test_evidence_is_bounded_and_keeps_ids(self):
        result = compact_evidence({"evidence": [{"id": "E1", "page": 1, "section_hint": "Method", "text": "important " * 1000}, {"id": "E2", "page": 2, "text": "other"}]}, max_chars=500)
        self.assertLessEqual(len(result), 500)
        self.assertIn("E1", result)

    def test_method_and_review_are_compact(self):
        review = {"paper_identity": {"title": "T"}, "modules": [{"id": "m", "statement": "x"}], "results": [{"statement": "drop"}]}
        compact = compact_review(review)
        self.assertIn("modules", compact)
        self.assertNotIn("results", compact)
        method = compact_method({"figure_goal": "goal", "modules": [{"id": "m", "name": "Module", "evidence_ids": ["E1"], "extra": "drop"}]})
        self.assertEqual(method["modules"][0]["id"], "m")
        self.assertNotIn("extra", method["modules"][0])

    def test_ppt_planner_sends_reference_but_not_icon_images(self):
        with TemporaryDirectory() as temp:
            icon = Path(temp) / "icon.png"
            reference = Path(temp) / "reference.png"
            from PIL import Image
            Image.new("RGB", (8, 8), "white").save(icon)
            Image.new("RGB", (8, 8), "white").save(reference)
            report = {"assets": [{"asset_id": "icon_n1", "node_id": "n1", "path": str(icon)}]}
            raw = {"panels": [], "nodes": [], "text_items": [], "icon_placements": [{"asset_id": "icon_n1", "node_id": "n1", "bbox_percent": {"x": .1, "y": .1, "w": .2, "h": .2}}], "connectors": []}
            with patch("rfs.planning.ppt_composition_planner.call_vlm_json", return_value=raw) as call:
                plan_ppt_composition_vlm({}, {"items": []}, report, reference)
            self.assertEqual(call.call_args.args[1], [reference])


if __name__ == "__main__":
    unittest.main()
