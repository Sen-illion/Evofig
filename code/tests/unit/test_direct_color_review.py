import unittest
from unittest.mock import patch

from rfs.evaluation.direct_color_review import disabled_color_repair_status, review_color_draft


class DirectColorReviewTests(unittest.TestCase):
    def test_repair_request_is_reported_but_never_applied(self):
        status = disabled_color_repair_status({"repair_required": True})
        self.assertFalse(status["enabled"])
        self.assertTrue(status["repair_requested"])
        self.assertFalse(status["applied"])

    @patch("rfs.evaluation.direct_color_review.call_vlm_json", return_value={"status": "needs_repair", "repair_required": True, "feedback": [{"target_role": "panel.p.border", "severity": "major", "reason": "too saturated", "suggested_direction": "use muted blue"}]})
    def test_review_returns_color_only_feedback(self, mocked):
        review, validation = review_color_draft({"style": {"color_theme": {}}}, "missing.png", {"status": "pass"})
        self.assertTrue(validation["ok"])
        self.assertTrue(review["repair_required"])
        self.assertEqual(review["feedback"][0]["target_role"], "panel.p.border")
        mocked.assert_called_once()

    @patch("rfs.evaluation.direct_color_review.call_vlm_json", side_effect=TimeoutError("color review timeout"))
    def test_provider_failure_is_reported_without_requesting_repair(self, _mocked):
        review, validation = review_color_draft({"style": {"color_theme": {}}}, "missing.png", {"status": "pass"})
        self.assertFalse(validation["ok"])
        self.assertEqual(review["status"], "failed")
        self.assertFalse(review["repair_required"])
        self.assertIn("timeout", review["provider_error"])

    @patch("rfs.evaluation.direct_color_review.call_vlm_json")
    def test_the_call_reports_its_own_timing(self, call):
        def answer(prompt, images, **kwargs):
            kwargs["call_metadata"].update({"success": True, "elapsed_seconds": 18.5})
            return {"status": "pass", "repair_required": False, "feedback": []}

        call.side_effect = answer
        metadata = {}
        _, validation = review_color_draft({"style": {"color_theme": {}}}, "missing.png", {"status": "pass"}, call_metadata=metadata)
        self.assertEqual(validation["provider"]["elapsed_seconds"], 18.5)
