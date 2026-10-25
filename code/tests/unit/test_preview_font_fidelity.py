"""The preview, the measurer and the compiled deck must use the same fonts."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from rfs.composition import fonts
from rfs.composition.preview import render_rebuild_preview
from rfs.composition.pptx import compile_ppt
from rfs.composition.text_layout import fit_title_block


def _panel_program(**style: object) -> dict:
    return {
        "canvas": {"width_px": 1600, "height_px": 900, "width_in": 13.333, "height_in": 7.5, "background": "#FFFFFF"},
        "style": {"palette": ["#FFFFFF", "#4A90C2"], **style},
        "title_block": {"title": "Paper Method", "subtitle": "Editable paper framework", "bbox_percent": {"x": .04, "y": .015, "w": .92, "h": .08}},
        "panels": [{"id": "panel_a", "title": "Pipeline Stage", "bbox_percent": {"x": .05, "y": .2, "w": .9, "h": .6}}],
        "cards": [],
        "slots": [],
        "semantic_nodes": [{"id": "node_a", "label": "Encoder", "bbox_percent": {"x": .1, "y": .3, "w": .2, "h": .1}}],
        "arrows": [],
        "text_program": {"items": [{
            "id": "text_node_a", "text": "Encoder", "text_role": "node_label", "role": "node_label",
            "target_id": "node_a", "panel_id": "panel_a", "visible": True, "font_size_pt": 12,
            "bbox_percent": {"x": .1, "y": .3, "w": .2, "h": .1},
        }]},
    }


class FontResolutionTests(unittest.TestCase):
    def test_arial_and_calibri_resolve_to_metric_compatible_files(self):
        for family, expected in (("Arial", ("arial", "liberationsans", "arimo")), ("Calibri", ("calibri", "carlito"))):
            for bold in (False, True):
                resolved = fonts.resolve_font(family, bold=bold)
                self.assertIsNotNone(resolved, family)
                self.assertTrue(resolved.metric_faithful, f"{family} must not be substituted")
                self.assertTrue(Path(resolved.path).name.casefold().startswith(expected), resolved.path)
                self.assertEqual(resolved.bold, bold)

    def test_unknown_family_falls_back_to_the_body_family(self):
        resolved = fonts.resolve_font("Comic Sans MS", bold=False)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.family, fonts.DEFAULT_FAMILY)

    def test_missing_family_file_is_reported_instead_of_hidden(self):
        with patch.object(fonts, "font_index", return_value={"dejavusans.ttf": "/tmp/DejaVuSans.ttf"}), \
                patch.object(fonts, "_SUBSTITUTIONS", {}), \
                patch.object(fonts, "FALLBACK_FILES", {False: ("DejaVuSans.ttf",), True: ("DejaVuSans.ttf",)}):
            stream = io.StringIO()
            with redirect_stderr(stream):
                resolved = fonts.resolve_font("Calibri", bold=False)
            reported = fonts.font_substitutions()
        self.assertFalse(resolved.metric_faithful)
        self.assertEqual(reported, [{"family": "calibri", "bold": False, "drawn_with": "DejaVuSans.ttf", "metric_faithful": False}])
        self.assertIn("font substitution", stream.getvalue())

    def test_heading_and_body_families_follow_the_program(self):
        style = {"font_family": "Arial", "heading_font_family": "Georgia"}
        self.assertEqual(fonts.heading_font_family(style), "Georgia")
        self.assertEqual(fonts.body_font_family(style), "Arial")
        self.assertEqual(fonts.heading_font_family({}), fonts.THEME_FONT_FAMILY)
        self.assertEqual(fonts.body_font_family({}), "Arial")


class PreviewFontFidelityTests(unittest.TestCase):
    def test_each_role_is_drawn_with_its_own_family(self):
        used: list[tuple[str, str | None]] = []
        real = fonts.load_font

        def record(size_px, family=None, *, bold=False):
            used.append((str(family), str(bold)))
            return real(size_px, family, bold=bold)

        with tempfile.TemporaryDirectory() as tmp, patch("rfs.composition.preview._load_font", side_effect=record):
            render_rebuild_preview(_panel_program(font_family="Arial", heading_font_family="Georgia"), tmp, Path(tmp) / "preview.png")

        families = {family for family, _ in used}
        self.assertEqual(families, {"Georgia", "Arial"}, used)
        self.assertIn(("Georgia", "True"), used)  # title and panel heading
        self.assertIn(("Arial", "False"), used)   # node label

    def test_heading_family_defaults_to_the_deck_theme_font(self):
        used = []
        real = fonts.load_font

        def record(size_px, family=None, *, bold=False):
            used.append(str(family))
            return real(size_px, family, bold=bold)

        with tempfile.TemporaryDirectory() as tmp, patch("rfs.composition.preview._load_font", side_effect=record):
            render_rebuild_preview(_panel_program(font_family="Arial"), tmp, Path(tmp) / "preview.png")

        self.assertIn(fonts.THEME_FONT_FAMILY, used)

    def test_title_sizing_measures_the_heading_font(self):
        measured: list[str | None] = []
        real = fonts.load_font

        def record(size_px, family=None, *, bold=False):
            measured.append(family)
            return real(size_px, family, bold=bold)

        with patch("rfs.composition.text_layout.load_font", side_effect=record):
            fit_title_block(_panel_program(font_family="Arial", heading_font_family="Georgia"), width_in=13.333, height_in=7.5)

        self.assertTrue(measured)
        self.assertEqual(set(measured), {"Georgia"})


class CompilerFontFidelityTests(unittest.TestCase):
    def test_every_run_names_its_family(self):
        program = _panel_program(font_family="Arial", heading_font_family="Calibri")
        program["canvas"] = {"width_in": 13.333, "height_in": 7.5, "ratio": 16 / 9, "background": "#FFFFFF"}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            compile_ppt(program, out)
            with zipfile.ZipFile(out / "editable_composition.pptx") as deck:
                xml = deck.read("ppt/slides/slide1.xml").decode("utf-8")
        # No run may rely on the theme font: a run without a typeface is what
        # made PowerPoint and the preview disagree about the title.
        runs = xml.split("<a:r>")[1:]
        self.assertTrue(runs)
        for run in runs:
            self.assertIn("typeface=", run, run[:200])
        self.assertIn('typeface="Calibri"', xml)
        self.assertIn('typeface="Arial"', xml)


if __name__ == "__main__":
    unittest.main()
