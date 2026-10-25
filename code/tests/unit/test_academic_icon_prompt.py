"""Regression tests for the concise academic framework-icon contract."""

import unittest

from rfs.contracts.icon_generation import normalize_icon_generation_plan, validate_icon_generation_plan
from rfs.generation.icon_assets import build_icon_prompt
from rfs.planning.icon_style import reference_icon_rules, resolve_primary_object


class AcademicIconPromptTests(unittest.TestCase):
    def test_standard_icon_prompt_is_simple_and_academic(self):
        prompt = build_icon_prompt({
            "asset_id": "icon_tools",
            "subject": "Tools",
            "primary_object": "one compact crossed wrench-and-gear symbol",
            "complexity_kind": "standard_icon",
            "icon_grammar": "academic_flat_symbol",
            "supporting_object_limit": 1,
            "background_policy": "transparent_panel_inherit",
        }).lower()
        self.assertIn("academic research framework-diagram icon", prompt)
        self.assertIn("exactly one dominant symbolic object", prompt)
        self.assertIn("multi-object scene", prompt)
        self.assertIn("3d render", prompt)
        self.assertIn("transparent background", prompt)

    def test_hero_icon_remains_bounded(self):
        prompt = build_icon_prompt({
            "asset_id": "icon_core",
            "subject": "Core method",
            "primary_object": "a compact method mechanism",
            "complexity_kind": "hero_visual",
            "icon_grammar": "bounded_mechanism_visual",
            "supporting_object_limit": 3,
        }).lower()
        self.assertIn("bounded mechanism visual", prompt)
        self.assertIn("no more than 3 compact supporting objects", prompt)
        self.assertIn("never a scene", prompt)

    def test_contract_defaults_to_standard_academic_icon(self):
        plan = normalize_icon_generation_plan({"icons": [{"asset_id": "icon_tools", "semantic_node_ids": ["tools"], "subject": "Tools"}]})
        icon = plan["icons"][0]
        self.assertEqual(icon["icon_grammar"], "academic_flat_symbol")
        self.assertEqual(icon["complexity_kind"], "standard_icon")
        self.assertTrue(validate_icon_generation_plan(plan)["ok"])

    def test_tools_gets_stable_simple_metaphor(self):
        self.assertIn("wrench", resolve_primary_object({"role": "tool"}, "Tools"))

    def test_reference_rules_keep_canonical_user_grammar(self):
        rules = reference_icon_rules({"icon_rules": ["Use flat icon symbols"]})
        self.assertTrue(any("one dominant" in item for item in rules))
        self.assertIn("Use flat icon symbols", rules)


if __name__ == "__main__":
    unittest.main()
