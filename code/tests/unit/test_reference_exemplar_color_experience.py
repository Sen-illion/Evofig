"""Tests for structured color experience in exemplar briefs."""

from __future__ import annotations

import unittest

from rfs.analysis.reference.exemplars import build_exemplar_style_brief


class ReferenceExemplarColorExperienceTests(unittest.TestCase):
    def test_hex_values_are_kept_in_recipes_not_flat_rules(self):
        selected = []
        for index in range(4):
            selected.append({"record": {
                "id": f"e{index}",
                "color": {"palette": ["#123456"], "contrast_policy": "dark text on pale panels"},
                "color_recipe": {
                    "recipe_id": f"recipe-{index}",
                    "background_color_id": "c01",
                    "colors": [{"color_id": "c01", "hex": "#F8FAFC", "area_ratio": 1.0, "used_for": ["canvas"], "semantic_roles": ["background"]}],
                    "relationships": {"harmony": "analogous"},
                    "transfer_rules": ["Preserve a pale canvas."],
                    "analysis": {"semantic_mode": "vlm"},
                },
            }})
        brief = build_exemplar_style_brief(selected, [], "missing-library.jsonl")
        self.assertNotIn("#123456", brief["color_rules"])
        self.assertEqual(len(brief["color_experience"]["selected_recipes"]), 3)
        self.assertEqual(brief["color_experience"]["selected_recipes"][0]["colors"][0]["hex"], "#F8FAFC")


if __name__ == "__main__":
    unittest.main()
