"""Tests for icon ink-bbox measurement used by slot compensation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rfs.generation.asset_metrics import measure_ink_bbox, measure_program_assets


def _write_icon(path: Path, box: tuple[int, int, int, int]) -> None:
    image = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for x in range(box[0], box[2]):
        for y in range(box[1], box[3]):
            image.putpixel((x, y), (70, 74, 92, 255))
    image.save(path)


class AssetMetricsTests(unittest.TestCase):
    def test_padded_glyph_reports_smaller_frame_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            tight = Path(tmp) / "tight.png"
            padded = Path(tmp) / "padded.png"
            _write_icon(tight, (0, 0, 100, 100))
            _write_icon(padded, (30, 30, 70, 70))
            tight_metrics = measure_ink_bbox(tight)
            padded_metrics = measure_ink_bbox(padded)
            self.assertEqual(tight_metrics["width_ratio"], 1.0)
            self.assertAlmostEqual(padded_metrics["width_ratio"], 0.4, places=2)
            self.assertAlmostEqual(padded_metrics["height_ratio"], 0.4, places=2)

    def test_missing_asset_reports_explicit_fallback_source(self):
        metrics = measure_ink_bbox(Path(tempfile.gettempdir()) / "paperimage_missing_icon.png")
        self.assertEqual(metrics["source"], "missing_file")
        self.assertEqual(metrics["width_ratio"], 1.0)

    def test_opaque_assets_are_measured_against_their_corner_color(self):
        with tempfile.TemporaryDirectory() as tmp:
            icon = Path(tmp) / "opaque.png"
            image = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
            for x in range(20, 60):
                for y in range(20, 60):
                    image.putpixel((x, y), (40, 40, 40, 255))
            image.save(icon)
            metrics = measure_ink_bbox(icon)
            self.assertEqual(metrics["source"], "corner_color_bbox")
            self.assertAlmostEqual(metrics["width_ratio"], 0.4, places=2)

    def test_program_assets_are_keyed_by_asset_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            icon = Path(tmp) / "icon.png"
            _write_icon(icon, (25, 25, 75, 75))
            measured = measure_program_assets([{"id": "icon_a", "path": str(icon)}])
            self.assertEqual(set(measured), {"icon_a"})
            self.assertAlmostEqual(measured["icon_a"]["width_ratio"], 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
