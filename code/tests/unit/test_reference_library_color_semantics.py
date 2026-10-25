import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from rfs.analysis.reference.color_recipe import EXTRACTOR_ID
from rfs.analysis.reference.library_learning import analyze_reference_figures


def _adapter(prompt, images, model, timeout, retries):
    import json
    marker = "COLOR_RECIPE:\n"
    recipe = json.loads(prompt.split(marker, 1)[1])
    return {
        "color_recipe_semantics": {
            "assignments": [
                {
                    "color_id": item["color_id"],
                    "used_for": item["used_for"] if item["used_for"] != ["unassigned"] else ["node_fill"],
                    "semantic_roles": ["background" if item["used_for"] == ["canvas"] else "processing"],
                    "confidence": 0.9,
                }
                for item in recipe["colors"]
            ],
            "relationships": {"harmony": "analogous"},
            "transfer_rules": ["Preserve lightness hierarchy."],
            "avoid_copying": ["paper labels"],
        }
    }


class ReferenceLibraryColorSemanticsTests(unittest.TestCase):
    def test_vlm_semantics_are_bound_to_deterministic_color_ids_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refs = root / "refs" / "excellent"
            refs.mkdir(parents=True)
            image = Image.new("RGB", (240, 140), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((15, 15, 110, 125), fill="#E3F2FD", outline="#2563EB", width=4)
            draw.rectangle((130, 15, 225, 125), fill="#FFF4E6", outline="#F59E0B", width=4)
            image.save(refs / "sample.png")
            out = root / "out"
            first = analyze_reference_figures(refs.parent, out, mode="vlm", vlm_adapter=_adapter)
            second = analyze_reference_figures(refs.parent, out, mode="vlm", vlm_adapter=_adapter)
            self.assertTrue(first["ok"])
            self.assertEqual(second["reused_count"], 1)
            record = __import__("json").loads((out / "reference_library.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(record["color_recipe"]["analysis"]["semantic_mode"], "vlm")
            self.assertTrue(all(item["used_for"] != ["unassigned"] for item in record["color_recipe"]["colors"]))

    def test_records_measured_by_an_older_extractor_are_re_measured(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            refs = root / "refs" / "excellent"
            refs.mkdir(parents=True)
            image = Image.new("RGB", (240, 140), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((15, 15, 110, 125), fill="#E3F2FD", outline="#2563EB", width=4)
            draw.rectangle((130, 15, 225, 125), fill="#FFF4E6", outline="#F59E0B", width=4)
            image.save(refs / "sample.png")
            out = root / "out"
            analyze_reference_figures(refs.parent, out, mode="vlm", vlm_adapter=_adapter)
            library = out / "reference_library.jsonl"
            library.write_text(library.read_text(encoding="utf-8").replace(EXTRACTOR_ID, "deterministic.v1"), encoding="utf-8")
            second = analyze_reference_figures(refs.parent, out, mode="vlm", vlm_adapter=_adapter)
            self.assertEqual(second["reused_count"], 0)
            record = json.loads(library.read_text(encoding="utf-8").strip())
            self.assertEqual(record["color_recipe"]["analysis"]["extractor"], EXTRACTOR_ID)


if __name__ == "__main__":
    unittest.main()
