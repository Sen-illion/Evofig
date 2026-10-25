import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from rfs.evaluation.direct_visual_review import review_direct_draft


class VisualReviewNormalizationTests(unittest.TestCase):
    @patch("rfs.evaluation.direct_visual_review.call_vlm_json", return_value={"status": "needs_repair", "feedback": [{"issue_type": "crowding", "severity": "major", "panel_id": "p", "object_ids": ["title"], "evidence": "overlap"}], "repair_intents": [{"action": "translate_group", "panel_id": "p", "object_ids": ["title"], "axis": "vertical", "amount": -0.02, "reason": "move title"}]})
    def test_non_executable_text_translation_remains_feedback_only(self, mocked):
        program = {"panels": [{"id": "p"}], "semantic_nodes": [], "arrows": [], "slots": [], "text_program": {"items": [{"id": "title", "role": "panel_title"}]}}
        with tempfile.TemporaryDirectory() as temp:
            preview = Path(temp) / "preview.png"
            Image.new("RGB", (20, 20), "white").save(preview)
            review, validation = review_direct_draft(program, preview, mode="vlm")
        self.assertTrue(validation["ok"], validation["errors"])
        self.assertEqual([], review["repair_intents"])
        self.assertTrue(any("translate_group:title" in item for item in review["normalization_warnings"]))

    @patch("rfs.evaluation.direct_visual_review.call_vlm_json")
    def test_filters_hallucinated_panel_and_mixed_lane_targets(self, call):
        call.return_value = {
            "status": "needs_repair",
            "feedback": [{"issue_type": "connector_congestion", "panel_id": "missing", "object_ids": ["edge", "node"], "severity": "major"}],
            "repair_intents": [{"action": "reserve_connector_lane", "panel_id": "missing", "object_ids": ["edge", "node"]}],
        }
        program = {
            "panels": [{"id": "panel", "bbox_percent": {"x": 0, "y": 0, "w": 1, "h": 1}}],
            "semantic_nodes": [{"id": "node", "bbox_percent": {"x": .1, "y": .1, "w": .2, "h": .2}}],
            "arrows": [{"id": "edge", "source": "node", "target": "node", "path_percent": [[.1, .1], [.2, .2]]}],
            "slots": [], "text_program": {"items": []},
        }
        with tempfile.TemporaryDirectory() as temp:
            preview = Path(temp) / "preview.png"
            Image.new("RGB", (20, 20), "white").save(preview)
            review, validation = review_direct_draft(program, preview, mode="vlm")
        self.assertTrue(validation["ok"])
        self.assertEqual(review["repair_intents"][0]["panel_id"], "")
        self.assertEqual(review["repair_intents"][0]["object_ids"], ["edge"])
        self.assertTrue(review["normalization_warnings"])


if __name__ == "__main__":
    unittest.main()
