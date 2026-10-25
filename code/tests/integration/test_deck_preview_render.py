"""The delivered preview must come from the deck a real Office engine renders."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rfs.composition.pptx import compile_ppt
from rfs.composition.pptx_render import available_backends, render_pptx_page


def _program() -> dict:
    return {
        "canvas": {"width_px": 1600, "height_px": 900, "width_in": 13.333, "height_in": 7.5, "background": "#FFFFFF"},
        "style": {"palette": ["#FFFFFF", "#4A90C2"]},
        "title_block": {"title": "Paper Method", "subtitle": "Editable paper framework", "bbox_percent": {"x": .04, "y": .015, "w": .92, "h": .08}},
        "panels": [{"id": "panel_a", "title": "Pipeline Stage", "bbox_percent": {"x": .05, "y": .2, "w": .9, "h": .6}}],
        "cards": [],
        "semantic_nodes": [
            {"id": "node_a", "label": "Encoder", "panel_id": "panel_a", "bbox_percent": {"x": .18, "y": .38, "w": .22, "h": .16}},
            {"id": "node_b", "label": "Decoder", "panel_id": "panel_a", "bbox_percent": {"x": .58, "y": .38, "w": .22, "h": .16}},
        ],
        "slots": [],
        "assets": [],
        "arrows": [{"id": "arrow_ab", "source": "node_a", "target": "node_b"}],
    }


@unittest.skipUnless(available_backends(), "no Office engine on this host can render a deck")
class DeckPreviewRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        compile_ppt(_program(), self.root)
        self.pptx = self.root / "editable_composition.pptx"

    def test_the_engine_returns_a_non_blank_page_at_the_requested_size(self) -> None:
        out = self.root / "page.png"
        info = render_pptx_page(self.pptx, out, width_px=1600, height_px=900)
        self.assertIn(info["backend"], available_backends())
        with Image.open(out) as image:
            self.assertEqual(image.size, (1600, 900))
            pixels = np.asarray(image.convert("RGB"))
        background = np.bincount(pixels.reshape(-1, 3)[:, 0]).argmax()
        ink = float((np.abs(pixels[:, :, 0].astype(int) - int(background)) > 12).mean())
        self.assertGreater(ink, 0.01, "a rendered deck must not come back blank")


if __name__ == "__main__":
    unittest.main()