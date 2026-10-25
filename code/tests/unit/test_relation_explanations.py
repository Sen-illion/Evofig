"""Tests for evidence-grounded relation explanation planning."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from rfs.planning.relation_explanations import plan_relation_explanations_vlm


class RelationExplanationTests(unittest.TestCase):
    @patch("rfs.planning.relation_explanations.call_vlm_json")
    def test_accepts_only_labels_bound_to_existing_evidence(self, call):
        call.return_value = {"relations": [
            {"id": "r1", "label": "aggregates telemetry", "evidence_ids": ["E1"], "confidence": .9},
            {"id": "r2", "label": "invented claim", "evidence_ids": ["OTHER"], "confidence": .9},
        ]}
        method = {"nodes": [], "relations": [
            {"id": "r1", "source": "a", "target": "b", "type": "data_flow", "label": "", "evidence_ids": ["E1"]},
            {"id": "r2", "source": "b", "target": "c", "type": "prediction", "label": "", "evidence_ids": ["E2"]},
        ]}
        updated, report = plan_relation_explanations_vlm(method)
        self.assertEqual(updated["relations"][0]["label"], "aggregates telemetry")
        self.assertEqual(updated["relations"][1]["label"], "")
        self.assertEqual(report["generated_count"], 1)

    def test_does_not_call_provider_when_no_evidence_backed_relation_is_eligible(self):
        method = {"relations": [{"id": "r1", "label": "", "evidence_ids": []}]}
        updated, report = plan_relation_explanations_vlm(method)
        self.assertEqual(updated, method)
        self.assertEqual(report["eligible_count"], 0)


if __name__ == "__main__":
    unittest.main()

