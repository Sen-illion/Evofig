import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from rfs.analysis.reference.color_recipe import MAX_REPRESENTATIVES, analyze_reference_color_recipe, render_palette_card
from rfs.contracts.reference_color_recipe import validate_reference_color_recipe


class ReferenceColorRecipeTests(unittest.TestCase):
    def test_extracts_area_background_and_palette_card(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (300, 180), "#FFFFFF")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 140, 160), fill="#E3F2FD", outline="#2563EB", width=5)
            draw.rectangle((160, 20, 280, 160), fill="#FFF4E6", outline="#F59E0B", width=5)
            source = root / "reference.png"
            card = root / "palette.png"
            image.save(source)
            recipe = analyze_reference_color_recipe(source, "recipe.test", palette_card_path=card)
            validation = validate_reference_color_recipe(recipe)
            self.assertTrue(validation["ok"], validation["errors"])
            self.assertAlmostEqual(sum(item["area_ratio"] for item in recipe["colors"]), 1.0, places=3)
            self.assertEqual(recipe["colors"][0]["used_for"], ["canvas"])
            self.assertTrue(card.is_file())
            deterministic_card = card.read_bytes()
            recipe["colors"][-1]["used_for"] = ["node_fill"]
            recipe["colors"][-1]["semantic_roles"] = ["processing"]
            render_palette_card(recipe, card)
            self.assertNotEqual(card.read_bytes(), deterministic_card)

    def test_a_small_saturated_accent_survives_as_its_own_color(self):
        # A warm block covering well under half a percent of the figure used to be
        # folded into its nearest large neighbour, which is how the stored library
        # ended up without any accent colour at all.
        accent = "#E1C8B4"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (400, 260), "#FFFFFF")
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 10, 300, 250), fill="#EAF1F7")
            draw.rectangle((330, 100, 348, 118), fill=accent)
            source = root / "reference.png"
            image.save(source)
            recipe = analyze_reference_color_recipe(source, "recipe.accent")
            validation = validate_reference_color_recipe(recipe)
            self.assertTrue(validation["ok"], validation["errors"])
            self.assertLessEqual(len(recipe["colors"]), MAX_REPRESENTATIVES)
            accent_rgb = tuple(int(accent[index:index + 2], 16) for index in (1, 3, 5))
            nearest = min(
                sum((item["rgb"][channel] - accent_rgb[channel]) ** 2 for channel in range(3)) ** 0.5
                for item in recipe["colors"]
            )
            self.assertLess(nearest, 24.0, [item["hex"] for item in recipe["colors"]])

    def test_semantic_validation_rejects_unassigned_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "reference.png"
            image = Image.new("RGB", (80, 80), "#FFFFFF")
            ImageDraw.Draw(image).rectangle((20, 20, 60, 60), fill="#2563EB")
            image.save(source)
            recipe = analyze_reference_color_recipe(source, "recipe.test")
            validation = validate_reference_color_recipe(recipe, require_semantics=True)
            self.assertFalse(validation["ok"])


if __name__ == "__main__":
    unittest.main()
