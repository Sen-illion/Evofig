"""Tests that the local preview mirrors PPT panel color layers."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from rfs.composition.preview import render_rebuild_preview


class PreviewColorThemeTests(unittest.TestCase):
    def test_panel_header_color_is_rendered(self):
        program = {
            "canvas": {"width_px": 100, "height_px": 100, "height_in": 10, "background": "#FFFFFF"},
            "panels": [{"id": "p1", "title": "Panel", "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}}],
            "style": {"panel_styles": {"p1": {"fill_color": "#CCFFCC", "stroke_color": "#008800", "header_color": "#880000"}}},
            "cards": [], "slots": [], "semantic_nodes": [], "arrows": [], "text_program": {"items": []},
        }
        with TemporaryDirectory() as temp:
            preview = render_rebuild_preview(program, temp, Path(temp) / "preview.png")
            image = Image.open(preview).convert("RGB")
            self.assertEqual(image.getpixel((20, 11)), (136, 0, 0))
            self.assertEqual(image.getpixel((50, 50)), (204, 255, 204))

    def test_panel_title_is_painted_inside_the_header_band_only(self):
        """A panel heading belongs in the band, exactly as the PPTX compiler paints it."""

        program = {
            "canvas": {"width_px": 200, "height_px": 200, "height_in": 10, "background": "#FFFFFF"},
            "panels": [{"id": "p1", "title": "Stage One", "bbox_percent": {"x": 0.05, "y": 0.2, "w": 0.9, "h": 0.5}}],
            "style": {"panel_styles": {"p1": {"fill_color": "#101820", "stroke_color": "#101820", "header_color": "#880000", "header_text_color": "#FFFFFF", "text_color": "#E2E8F0"}}},
            "cards": [], "slots": [], "semantic_nodes": [], "arrows": [],
            "text_program": {
                "items": [{
                    "id": "title_p1", "text": "Stage One", "text_role": "panel_title", "role": "panel_title",
                    "panel_id": "p1", "target_id": "p1", "visible": True, "font_size_pt": 10,
                    "color_hex": "#FFFFFF",
                    # The planner pins the heading just below the band; the
                    # preview must ignore this box and paint the band instead.
                    "bbox_percent": {"x": 0.05, "y": 0.30, "w": 0.9, "h": 0.05},
                }]
            },
        }
        with TemporaryDirectory() as temp:
            preview = render_rebuild_preview(program, temp, Path(temp) / "preview.png")
            image = Image.open(preview).convert("RGB")

        band_rows = range(41, 47)
        in_band = [image.getpixel((x, y)) for x in range(30, 170, 2) for y in band_rows]
        self.assertTrue(
            any(sum(pixel) > 400 for pixel in in_band),
            "the panel heading must be painted inside the header band",
        )
        below_band = [image.getpixel((x, y)) for x in range(30, 170, 2) for y in range(52, 120)]
        self.assertFalse(
            any(sum(pixel) > 400 for pixel in below_band),
            "the standalone title box must not paint a second copy below the band",
        )

    def test_transparent_icon_and_separate_label_match_ppt_layers(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            assets = root / "assets"
            assets.mkdir()
            icon = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
            for x in range(7, 13):
                for y in range(7, 13):
                    icon.putpixel((x, y), (255, 0, 0, 255))
            icon.save(assets / "icon.png")
            program = {
                "canvas": {"width_px": 100, "height_px": 100, "background": "#FFFFFF"},
                "panels": [{"id": "p", "bbox_percent": {"x": 0, "y": 0, "w": 1, "h": 1}}],
                "style": {"panel_styles": {"p": {"fill_color": "#CCFFCC", "header_color": "#CCFFCC"}}},
                "assets": [{"id": "icon", "path": str(assets / "icon.png")}],
                "slots": [{"id": "slot", "asset_id": "icon", "bbox_percent": {"x": .45, "y": .45, "w": .1, "h": .1}}],
                "semantic_nodes": [{"id": "node", "fill_color": "#E3F2FD", "stroke_color": "#2563EB", "bbox_percent": {"x": .4, "y": .4, "w": .3, "h": .3}, "label_bbox_percent": {"x": .4, "y": .6, "w": .3, "h": .1}}],
                "cards": [], "arrows": [], "text_program": {"items": []},
            }
            preview = render_rebuild_preview(program, root, root / "preview.png")
            rendered = Image.open(preview).convert("RGB")
            self.assertEqual(rendered.getpixel((11, 11)), (204, 255, 204))
            self.assertEqual(rendered.getpixel((65, 50)), (227, 242, 253))
            self.assertEqual(rendered.getpixel((50, 50)), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
