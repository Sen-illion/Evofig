"""Delivered icon assets are trimmed to the mark that reaches the slide."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rfs.generation.asset_metrics import measure_ink_bbox, trim_asset_to_glyph, trim_program_icon_assets


def _padded_icon(directory: Path, name: str = "padded.png") -> Path:
    image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    for x in range(60, 140):
        for y in range(80, 120):
            image.putpixel((x, y), (12, 24, 36, 255))
    path = directory / name
    image.save(path)
    return path


def _tight_icon(directory: Path) -> Path:
    image = Image.new("RGBA", (64, 64), (12, 24, 36, 255))
    path = directory / "tight.png"
    image.save(path)
    return path


class IconTrimTests(unittest.TestCase):
    def test_padding_is_removed_from_the_delivered_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _padded_icon(root)
            record = trim_asset_to_glyph(source)
            self.assertEqual(record["status"], "trimmed")
            trimmed = Path(record["trimmed_path"])
            self.assertTrue(trimmed.is_file())
            self.assertLess(trimmed.stat().st_size, source.stat().st_size)
            with Image.open(trimmed) as image:
                self.assertEqual(image.size, (84, 44))
            measured = measure_ink_bbox(trimmed)
            self.assertGreater(measured["width_ratio"], 0.9)
            self.assertGreater(measured["height_ratio"], 0.8)

    def test_the_original_selection_stays_on_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _padded_icon(root)
            trim_asset_to_glyph(source)
            self.assertTrue(source.is_file())

    def test_an_already_tight_asset_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = trim_asset_to_glyph(_tight_icon(root))
            self.assertEqual(record["status"], "already_tight")
            self.assertIsNone(record["trimmed_path"])

    def test_program_assets_point_at_the_trimmed_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _padded_icon(root)
            program = {"assets": [{"id": "icon_a", "path": str(source)}]}
            updated, report = trim_program_icon_assets(program)
            self.assertEqual(report["trimmed_count"], 1)
            self.assertNotEqual(updated["assets"][0]["path"], str(source))
            self.assertTrue(Path(updated["assets"][0]["path"]).is_file())
            self.assertEqual(program["assets"][0]["path"], str(source))


if __name__ == "__main__":
    unittest.main()
