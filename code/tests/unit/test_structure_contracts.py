"""Tests for the shared global structure and segmentation contracts."""

import unittest

from rfs.contracts import (
    normalize_global_structure_plan,
    normalize_segmentation_plan,
    validate_global_structure_plan,
    validate_segmentation_plan,
)


class StructureContractTests(unittest.TestCase):
    def test_global_structure_normalizes_and_validates_endpoints(self):
        plan = normalize_global_structure_plan({
            "topology": "branch",
            "nodes": [{"id": "source"}, {"id": "head"}],
            "relations": [{"source": "source", "target": "head"}],
        })
        self.assertEqual(plan["contract"], "rfs.global_structure_plan")
        self.assertTrue(validate_global_structure_plan(plan)["ok"])

    def test_global_structure_rejects_unknown_endpoint(self):
        result = validate_global_structure_plan({
            "nodes": [{"id": "source"}],
            "relations": [{"source": "source", "target": "missing"}],
        })
        self.assertFalse(result["ok"])
        self.assertIn("endpoint", result["errors"][0])

    def test_segmentation_requires_explicit_native_ownership(self):
        plan = normalize_segmentation_plan({
            "slots": [{"slot_id": "encoder", "semantic_node_ids": ["node_encoder"]}],
        })
        result = validate_segmentation_plan(plan)
        self.assertTrue(result["ok"])
        self.assertEqual(plan["slots"][0]["asset_policy"], "visual_only_no_text_no_arrows")
        self.assertEqual(plan["slots"][0]["text_policy"], "native_ppt")

    def test_segmentation_rejects_duplicate_slots_and_non_native_assets(self):
        result = validate_segmentation_plan({
            "slots": [
                {"slot_id": "same", "semantic_node_ids": ["a"]},
                {"slot_id": "same", "semantic_node_ids": ["b"], "asset_policy": "legacy"},
            ],
        })
        self.assertFalse(result["ok"])
        self.assertGreaterEqual(len(result["errors"]), 2)


if __name__ == "__main__":
    unittest.main()
