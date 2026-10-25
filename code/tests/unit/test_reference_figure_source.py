"""Tests for reading a reference figure asset before it is profiled."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rfs.analysis.reference.figure_source import reference_image_for_analysis, reference_image_path
from rfs.analysis.reference.style_analyzer import analyze_reference_image


def _write_pdf(path: Path, *, width: int = 240, height: int = 120) -> Path:
    """Write a one-page PDF shaped like a small framework figure."""
    import fitz

    document = fitz.open()
    try:
        page = document.new_page(width=width, height=height)
        page.draw_rect(fitz.Rect(10, 30, 70, 90), color=(0, 0, 0), fill=(0.2, 0.4, 0.8))
        page.draw_rect(fitz.Rect(100, 30, 160, 90), color=(0, 0, 0), fill=(0.9, 0.6, 0.1))
        page.insert_text((12, 20), "Module A feeds Module B")
        document.save(str(path))
    finally:
        document.close()
    return path


class ReferenceFigureSourceTests(unittest.TestCase):
    def test_raster_reference_is_used_as_it_is(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "reference.png"
            Image.new("RGB", (80, 60), "white").save(image)
            asset = reference_image_for_analysis(image, work_dir=root / "renders")
            self.assertEqual(asset["status"], "ok")
            self.assertFalse(asset["rendered"])
            self.assertEqual(Path(asset["image"]), image.resolve())

    def test_pdf_reference_is_rendered_once_and_reused_afterwards(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = _write_pdf(root / "framework.pdf")
            work = root / "renders"
            first = reference_image_for_analysis(pdf, work_dir=work)
            self.assertEqual(first["status"], "ok")
            self.assertTrue(first["rendered"])
            self.assertFalse(first["cached"])
            rendered = Path(first["image"])
            self.assertTrue(rendered.is_file())
            self.assertEqual(rendered.suffix, ".png")
            self.assertEqual(rendered.parent, work.resolve())
            self.assertGreater(first["width"], 0)
            self.assertGreater(first["height"], 0)
            second = reference_image_for_analysis(pdf, work_dir=work)
            self.assertTrue(second["cached"])
            self.assertEqual(Path(second["image"]), rendered)
            self.assertEqual(len(list(work.glob("*.png"))), 1)
            with Image.open(rendered) as opened:
                self.assertEqual(opened.mode, "RGB")

    def test_missing_and_unsupported_assets_report_a_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            missing = reference_image_for_analysis(root / "nope.pdf", work_dir=root / "renders")
            self.assertEqual(missing["status"], "missing")
            other = root / "framework.docx"
            other.write_text("not a figure", encoding="utf-8")
            unsupported = reference_image_for_analysis(other, work_dir=root / "renders")
            self.assertEqual(unsupported["status"], "unsupported")
            self.assertIn(".docx", unsupported["error"])


class ReferenceAnalysisWithPdfAssetTests(unittest.TestCase):
    def test_pdf_reference_is_profiled_instead_of_failing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = _write_pdf(root / "framework.pdf")
            profile = analyze_reference_image(pdf, method={"topology": "linear"}, mode="heuristic", work_dir=root / "inputs")
            self.assertEqual(profile["status"], "fallback")
            self.assertTrue(profile["reference_asset"]["rendered"])
            rendered = Path(profile["reference_asset"]["image"])
            self.assertTrue(rendered.is_file())
            self.assertEqual(rendered.parent, (root / "inputs").resolve())
            self.assertNotIn("reference_analysis_failed", profile["influence"]["rejected"])
            self.assertTrue(profile["layout"]["aspect_ratio"] > 0)

    def test_a_missing_reference_is_still_reported_as_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = analyze_reference_image(root / "gone.png", method={}, work_dir=root)
            self.assertEqual(profile["status"], "not_found")
            self.assertIn("reference_missing", profile["influence"]["rejected"])


class VlmReferenceImagePathTests(unittest.TestCase):
    """A vision call must never be handed the PDF the snapshot stores."""

    def test_a_pdf_reference_becomes_a_renderable_png_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pdf = _write_pdf(root / "framework.pdf")
            image, report = reference_image_path(pdf, work_dir=root / "inputs")
            self.assertIsNotNone(image)
            self.assertEqual(Path(image).suffix, ".png")
            self.assertEqual(Path(image).parent, (root / "inputs").resolve())
            self.assertTrue(report["rendered"])
            with Image.open(image) as opened:
                self.assertGreater(opened.size[0], 0)

    def test_a_raster_reference_is_returned_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raster = root / "framework.png"
            Image.new("RGB", (40, 30), "white").save(raster)
            image, report = reference_image_path(raster, work_dir=root / "inputs")
            self.assertEqual(Path(image), raster.resolve())
            self.assertEqual(report["status"], "ok")
            self.assertFalse(report["rendered"])

    def test_an_unreadable_reference_returns_no_image_and_a_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, report = reference_image_path(root / "missing.pdf", work_dir=root / "inputs")
            self.assertIsNone(image)
            self.assertEqual(report["status"], "missing")
            other = root / "framework.docx"
            other.write_text("not a figure", encoding="utf-8")
            image, report = reference_image_path(other, work_dir=root / "inputs")
            self.assertIsNone(image)
            self.assertEqual(report["status"], "unsupported")
