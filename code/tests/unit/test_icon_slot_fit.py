"""Icon slots hug the delivered glyph and stay grouped with their label."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rfs.composition.node_layout import arrange_node_contents


def _wide_glyph(directory: Path) -> str:
    image = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    for x in range(20, 180):
        for y in range(80, 120):
            image.putpixel((x, y), (20, 30, 40, 255))
    path = directory / "wide_icon.png"
    image.save(path)
    return str(path)


def _program(asset_path: str) -> dict:
    return {
        "canvas": {"width_in": 13.333, "height_in": 7.5},
        "assets": [{"id": "icon_planner", "path": asset_path}],
        "semantic_nodes": [
            {"id": "module_planner", "label": "Planner", "bbox_percent": {"x": .18, "y": .3, "w": .12, "h": .45}},
        ],
        "slots": [{"id": "slot_module_planner", "asset_id": "icon_planner", "bbox_percent": {"x": .18, "y": .3, "w": .12, "h": .45}}],
    }


class IconSlotFitTests(unittest.TestCase):
    def test_tall_card_slot_shrinks_to_the_visible_glyph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            updated, _ = arrange_node_contents(_program(_wide_glyph(Path(tmp))))
        slot = updated["slots"][0]["bbox_percent"]
        node = updated["semantic_nodes"][0]
        label = node["label_bbox_percent"]
        card = node["bbox_percent"]
        expected_height = slot["w"] * (40 / 160) * (13.333 / 7.5)
        self.assertAlmostEqual(slot["h"], expected_height, places=3)
        self.assertLess(slot["h"], card["h"] * 0.3)
        self.assertGreater(label["y"], slot["y"] + slot["h"] - 0.005)
        self.assertLess(label["y"] - (slot["y"] + slot["h"]), card["h"] * 0.05)
        group_top, group_bottom = slot["y"], label["y"] + label["h"]
        self.assertAlmostEqual(group_top - card["y"], card["y"] + card["h"] - group_bottom, places=2)

    def test_slot_stays_inside_the_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            updated, _ = arrange_node_contents(_program(_wide_glyph(Path(tmp))))
        slot = updated["slots"][0]["bbox_percent"]
        label = updated["semantic_nodes"][0]["label_bbox_percent"]
        card = updated["semantic_nodes"][0]["bbox_percent"]
        self.assertGreaterEqual(slot["y"], card["y"])
        self.assertLessEqual(label["y"] + label["h"], card["y"] + card["h"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
