"""Regression tests for the semantics a composed connector must keep.

The composition pass places and routes connectors; it does not restate what they mean, and the
model that writes it returns id/source/target as often as not. On the 2026-10-08 own-set row 84 of
273 composed connectors arrived with no type, the delivered arrows lost their relation labels with
it, and the edges could no longer be bound back to the scientific graph -- which failed the
Grounding Gate and left every repair round of the continuous probe refused as a new blocker.
"""

from __future__ import annotations

import unittest

from rfs.workflows.paper_evolve_editable import _restore_connector_semantics


class ConnectorSemanticsMergeTests(unittest.TestCase):
    def test_a_composed_connector_gets_its_relation_back_by_id(self):
        planned = [{"id": "c1", "source": "a", "target": "b", "type": "conditioning", "label": "conditions on", "evidence_ids": ["E_1"]}]
        composed = [{"id": "c1", "source": "a", "target": "b", "path_percent": [[0.1, 0.1], [0.9, 0.9]]}]
        merged = _restore_connector_semantics(planned, composed)
        self.assertEqual(merged[0]["type"], "conditioning")
        self.assertEqual(merged[0]["label"], "conditions on")
        self.assertEqual(merged[0]["evidence_ids"], ["E_1"])
        self.assertEqual(merged[0]["path_percent"], [[0.1, 0.1], [0.9, 0.9]])

    def test_a_renumbered_connector_is_matched_by_its_endpoints(self):
        planned = [{"id": "c7", "source": "sampler", "target": "policy", "type": "feedback_loop", "label": "revised policy"}]
        composed = [{"id": "composition_edge_01", "source": "sampler", "target": "policy"}]
        merged = _restore_connector_semantics(planned, composed)
        self.assertEqual(merged[0]["id"], "composition_edge_01")
        self.assertEqual(merged[0]["type"], "feedback_loop")
        self.assertEqual(merged[0]["label"], "revised policy")

    def test_a_stated_relation_is_kept_and_an_endpointless_connector_is_dropped(self):
        planned = [{"id": "c1", "source": "a", "target": "b", "type": "conditioning", "label": "conditions on"}]
        composed = [
            {"id": "c1", "source": "a", "target": "b", "type": "prediction", "label": "predicts"},
            {"id": "c2", "source": "b"},
        ]
        merged = _restore_connector_semantics(planned, composed)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["type"], "prediction")
        self.assertEqual(merged[0]["label"], "predicts")

    def test_a_structure_plan_with_nothing_to_offer_leaves_the_connector_alone(self):
        merged = _restore_connector_semantics([], [{"source": "a", "target": "b"}])
        self.assertEqual(merged, [{"id": "composition_edge_01", "source": "a", "target": "b"}])


if __name__ == "__main__":
    unittest.main()
