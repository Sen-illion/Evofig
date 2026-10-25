"""Tests for evidence-grounded scientific hierarchy planning."""

import unittest
from unittest.mock import patch

from rfs.planning.scientific_hierarchy import plan_scientific_hierarchy, validate_scientific_hierarchy, normalize_scientific_hierarchy
from rfs.planning.scientific_hierarchy import repair_scientific_hierarchy


class ScientificHierarchyTests(unittest.TestCase):
    def setUp(self):
        self.method = {
            "inputs": [{"id": "input", "name": "Input", "evidence_ids": ["e1"]}],
            "modules": [{"id": "core", "name": "Core", "evidence_ids": ["e2"]}],
            "stages": [{"id": "stage", "name": "Stage", "evidence_ids": ["e3"]}],
            "outputs": [{"id": "output", "name": "Output", "evidence_ids": ["e4"]}],
            "relations": [],
        }

    def test_section_count_is_not_fixed_to_three(self):
        plan = {
            "sections": [{"id": "s1", "title": "Input", "evidence_ids": ["e1"]}, {"id": "s2", "title": "Core", "evidence_ids": ["e2"]}],
            "submodules": [],
        }
        report = validate_scientific_hierarchy(plan, self.method)
        self.assertTrue(report["ok"])
        self.assertEqual(report["section_count"], 2)

    def test_vlm_plan_requires_evidence_and_known_entities(self):
        bad = {"sections": [{"id": "s1", "title": "Bad", "evidence_ids": []}], "submodules": [{"id": "m1", "parent_section_id": "s1", "title": "Invented", "entity_ids": ["unknown"], "evidence_ids": ["e9"]}]}
        report = validate_scientific_hierarchy(bad, self.method)
        self.assertFalse(report["ok"])
        self.assertTrue(any("evidence" in error for error in report["errors"]))
        self.assertTrue(any("absent" in error for error in report["errors"]))

    def test_vlm_failure_uses_heuristic_fallback(self):
        with patch("rfs.planning.scientific_hierarchy.call_vlm_json", side_effect=TimeoutError("timeout")):
            plan, report = plan_scientific_hierarchy(self.method)
        self.assertTrue(report["fallback_used"])
        self.assertGreaterEqual(len(plan["sections"]), 2)
        self.assertTrue(validate_scientific_hierarchy(plan, self.method)["ok"])

    def test_normalizes_nested_vlm_sections(self):
        plan = normalize_scientific_hierarchy({"sections": [{"id": "s1", "name": "Core", "evidence_ids": ["e2"], "submodules": [{"id": "core", "name": "Core module", "evidence_ids": ["e2"]}]}]})
        self.assertEqual(plan["sections"][0]["title"], "Core")
        self.assertEqual(plan["sections"][0]["submodule_ids"], ["core"])
        self.assertEqual(plan["submodules"][0]["parent_section_id"], "s1")

    def test_normalizes_section_id_alias(self):
        plan = normalize_scientific_hierarchy({"sections": [{"id": "s1", "title": "Core", "evidence_ids": ["e2"]}], "submodules": [{"id": "m1", "section_id": "s1", "name": "Core module", "evidence_ids": ["e2"]}]})
        self.assertEqual(plan["submodules"][0]["parent_section_id"], "s1")

    def test_a_parent_the_model_named_wrongly_is_adopted_not_discarded(self):
        # The model listed the submodule under section s1 and also spelled the
        # parent as "core". Rejecting the whole plan for that replaced the
        # paper's own hierarchy with generic buckets.
        raw = {
            "sections": [{"id": "s1", "title": "Core Method", "submodule_ids": ["core"], "evidence_ids": ["e2"]}],
            "submodules": [{"id": "core", "parent_section_id": "core", "title": "Core", "entity_ids": ["core"], "evidence_ids": ["e2"]}],
        }
        plan = repair_scientific_hierarchy(normalize_scientific_hierarchy(raw), self.method)
        self.assertTrue(plan["repaired"])
        self.assertEqual(plan["submodules"][0]["parent_section_id"], "s1")
        self.assertTrue(validate_scientific_hierarchy(plan, self.method)["ok"])

    def test_unknown_entities_are_dropped_and_the_known_one_is_kept(self):
        raw = {
            "sections": [{"id": "s1", "title": "Core Method", "evidence_ids": ["e2"]}],
            "submodules": [{"id": "m1", "parent_section_id": "s1", "title": "Core", "entity_ids": ["core", "invented"], "evidence_ids": ["e2"]}],
        }
        plan = repair_scientific_hierarchy(normalize_scientific_hierarchy(raw), self.method)
        self.assertEqual(plan["submodules"][0]["entity_ids"], ["core"])
        self.assertTrue(any(item["kind"] == "unknown_entities_dropped" for item in plan["repairs"]))
        self.assertTrue(validate_scientific_hierarchy(plan, self.method)["ok"])

    def test_missing_evidence_is_inherited_from_the_entity_it_cites(self):
        raw = {
            "sections": [{"id": "s1", "title": "Core Method", "evidence_ids": ["e2"]}],
            "submodules": [{"id": "m1", "parent_section_id": "s1", "title": "Core", "entity_ids": ["core"], "evidence_ids": []}],
        }
        plan = repair_scientific_hierarchy(normalize_scientific_hierarchy(raw), self.method)
        self.assertEqual(plan["submodules"][0]["evidence_ids"], ["e2"])

    def test_the_errors_are_handed_back_to_the_model_before_fallback(self):
        broken = {"sections": [{"id": "s1", "title": "Core", "evidence_ids": []}], "submodules": []}
        fixed = {"sections": [{"id": "s1", "title": "Core", "evidence_ids": ["e2"]}], "submodules": []}
        with patch("rfs.planning.scientific_hierarchy.call_vlm_json", side_effect=[broken, fixed]) as call:
            plan, report = plan_scientific_hierarchy(self.method)
        self.assertFalse(report["fallback_used"])
        self.assertEqual(report["repair_rounds"], 1)
        self.assertEqual(call.call_count, 2)
        self.assertIn("requires evidence_ids", call.call_args_list[1].args[0])
        self.assertTrue(validate_scientific_hierarchy(plan, self.method)["ok"])

    def test_a_plan_that_stays_ungrounded_keeps_the_model_structure(self):
        broken = {"sections": [{"id": "s1", "title": "Core", "evidence_ids": []}], "submodules": []}
        with patch("rfs.planning.scientific_hierarchy.call_vlm_json", side_effect=[broken, broken]):
            plan, report = plan_scientific_hierarchy(self.method)
        self.assertFalse(report["fallback_used"])
        self.assertTrue(report["continued_with_invalid_plan"])
        self.assertEqual([section["title"] for section in plan["sections"]], ["Core"])


if __name__ == "__main__":
    unittest.main()
