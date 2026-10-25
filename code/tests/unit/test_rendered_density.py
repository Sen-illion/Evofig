"""Tests for rendered-ink density measurement."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rfs.evaluation.rendered_density import (
    MAX_BODY_EMPTY_CELL_RATIO,
    MIN_INK_COVERAGE,
    measure_rendered_density,
)


def _canvas(path: Path, *, filled: bool) -> None:
    image = Image.new("RGB", (320, 180), (255, 255, 255))
    if filled:
        for row in range(20, 160):
            for column in range(20, 300):
                image.putpixel((column, row), (70, 74, 92))
    image.save(path)


class RenderedDensityTests(unittest.TestCase):
    def test_blank_canvas_is_reported_as_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blank.png"
            _canvas(path, filled=False)
            report = measure_rendered_density(path)
            self.assertEqual(report["status"], "warning")
            self.assertEqual(report["ink_coverage"], 0.0)
            self.assertEqual(report["body_empty_cell_ratio"], 1.0)
            self.assertTrue(report["issues"])

    def test_filled_canvas_passes_both_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "filled.png"
            _canvas(path, filled=True)
            report = measure_rendered_density(path)
            self.assertEqual(report["status"], "pass")
            self.assertGreaterEqual(report["ink_coverage"], MIN_INK_COVERAGE)
            self.assertLessEqual(report["body_empty_cell_ratio"], MAX_BODY_EMPTY_CELL_RATIO)

    def test_missing_preview_is_reported_without_raising(self):
        report = measure_rendered_density(Path(tempfile.gettempdir()) / "definitely-missing-preview.png")
        self.assertEqual(report["status"], "not_measured")
        self.assertEqual(report["reason"], "preview_missing")


if __name__ == "__main__":
    unittest.main()
