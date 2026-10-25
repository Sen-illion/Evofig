import unittest
from unittest.mock import patch

from rfs.evaluation.direct_connector_review import review_connector_routes


class DirectConnectorReviewTests(unittest.TestCase):
    def test_only_collision_free_candidates_are_sent_and_accepted(self):
        program = {"arrows": [{"id": "e", "source": "a", "target": "b", "route_selection": {"candidates": [{"candidate_id": "direct", "obstacle_hits": 0, "score": 1.0, "path": [[0, 0], [1, 1]]}, {"candidate_id": "side", "obstacle_hits": 0, "score": 2.0, "path": [[0, 0], [0, 1], [1, 1]]}, {"candidate_id": "bad", "obstacle_hits": 1, "score": 0.1, "path": [[0, 0], [1, 1]]}]}}]}
        with patch("rfs.evaluation.direct_connector_review.call_vlm_json", return_value={"selections": [{"arrow_id": "e", "candidate_id": "direct"}]}):
            review, validation = review_connector_routes(program, "missing.png")
        self.assertTrue(validation["ok"])
        self.assertEqual(review["selections"][0]["candidate_id"], "direct")

    @patch("rfs.evaluation.direct_connector_review.call_vlm_json", side_effect=TimeoutError("route timeout"))
    def test_provider_failure_is_non_blocking(self, _mocked):
        program = {"arrows": [{"id": "e", "route_selection": {"candidates": [{"candidate_id": "direct", "obstacle_hits": 0, "score": 1.0, "path": [[0, 0], [1, 1]]}, {"candidate_id": "side", "obstacle_hits": 0, "score": 2.0, "path": [[0, 0], [0, 1], [1, 1]]}]}}]}
        review, validation = review_connector_routes(program, "missing.png")
        self.assertFalse(validation["ok"])
        self.assertEqual(review["status"], "failed")
        self.assertEqual(review["selections"], [])

    def test_the_call_reports_its_own_timing(self):
        program = {"arrows": [{"id": "e", "route_selection": {"candidates": [{"candidate_id": "direct", "obstacle_hits": 0, "score": 1.0}, {"candidate_id": "side", "obstacle_hits": 0, "score": 2.0}]}}]}

        def answer(prompt, images, **kwargs):
            kwargs["call_metadata"].update({"success": True, "elapsed_seconds": 42.0, "image_count": len(images)})
            return {"selections": [{"arrow_id": "e", "candidate_id": "direct"}]}

        metadata = {}
        with patch("rfs.evaluation.direct_connector_review.call_vlm_json", side_effect=answer):
            _, validation = review_connector_routes(program, "missing.png", call_metadata=metadata)
        self.assertEqual(validation["provider"]["elapsed_seconds"], 42.0)
        self.assertEqual(validation["provider"]["image_count"], 1)
