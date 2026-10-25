"""An edit to the figure program reaches the deck and the preview, with no model in the loop."""

from __future__ import annotations

import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from PIL import Image

from rfs.composition.direct_connectors import apply_memory_routing_constraints
from rfs.composition.figure_edit import apply_constraints, run_figure_edit
from rfs.composition.layout_repairs import apply_layout_constraints
from rfs.utils import read_json, write_json


def _program():
    return {
        "canvas": {"width_px": 1200, "height_px": 675, "width_in": 13.333, "height_in": 7.5, "background": "#FFFFFF"},
        "style": {"palette": ["#FFFFFF", "#4A90C2"], "arrow_weight_pt": 1.7, "font_family": "Arial", "heading_font_family": "Calibri"},
        "title_block": {"title": "Paper Method", "subtitle": "Editable paper framework", "bbox_percent": {"x": 0.04, "y": 0.015, "w": 0.92, "h": 0.08}},
        "panels": [{"id": "panel_a", "title": "Pipeline", "bbox_percent": {"x": 0.05, "y": 0.2, "w": 0.9, "h": 0.6}}],
        "cards": [],
        "slots": [],
        "assets": [],
        "semantic_nodes": [
            {"id": "node_a", "label": "Encoder", "panel_id": "panel_a", "bbox_percent": {"x": 0.15, "y": 0.35, "w": 0.2, "h": 0.15}},
            {"id": "node_b", "label": "Decoder", "panel_id": "panel_a", "bbox_percent": {"x": 0.6, "y": 0.35, "w": 0.2, "h": 0.15}},
        ],
        "labels": [],
        "arrows": [{"id": "arrow_ab", "source": "node_a", "target": "node_b", "label": ""}],
        "text_program": {
            "items": [
                {"id": "text_node_a", "text": "Encoder", "text_role": "node_label", "role": "node_label", "owner_node_id": "node_a", "target_id": "node_a", "bbox_percent": {"x": 0.17, "y": 0.51, "w": 0.16, "h": 0.05}, "visible": True, "font_size_pt": 12.0},
                {"id": "text_node_b", "text": "Decoder", "text_role": "node_label", "role": "node_label", "owner_node_id": "node_b", "target_id": "node_b", "bbox_percent": {"x": 0.62, "y": 0.51, "w": 0.16, "h": 0.05}, "visible": True, "font_size_pt": 12.0},
            ]
        },
    }


def _differing_pixels(left: Path, right: Path) -> int:
    with Image.open(left) as first, Image.open(right) as second:
        a = first.convert("RGB").tobytes()
        b = second.convert("RGB").tobytes()
    if len(a) != len(b):
        return -1
    return sum(1 for index in range(0, len(a), 3) if a[index:index + 3] != b[index:index + 3])


class EditRecompileRenderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.program_path = self.root / "figure_program.json"
        write_json(self.program_path, _program())
        # Nothing in this test may reach a provider. Point the environment at a
        # dead address: a command that called out would hang or fail instead of
        # returning, so a fast green run is the proof that it did not.
        self._env = {key: os.environ.get(key) for key in ("API_BASE", "API_KEY")}
        os.environ["API_BASE"] = "http://127.0.0.1:1/v1"
        os.environ["API_KEY"] = "not-a-key"
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_a_moved_node_reaches_the_deck_and_the_preview(self):
        before_dir = self.root / "before"
        run_figure_edit("recompile", program_path=self.program_path, out_dir=before_dir)
        run_figure_edit("render", program_path=self.program_path, out_dir=before_dir)

        original = read_json(self.program_path)
        run_figure_edit("move-node", program_path=self.program_path, object_id="node_b", x=0.1, y=0.25)
        run_figure_edit("set-label", program_path=self.program_path, object_id="node_b", text="Second stage")
        edited = read_json(self.program_path)
        self.assertNotEqual(original, edited)
        self.assertEqual(0.1, edited["semantic_nodes"][1]["bbox_percent"]["x"])
        self.assertEqual("Second stage", edited["semantic_nodes"][1]["label"])

        after_dir = self.root / "after"
        compiled = run_figure_edit("recompile", program_path=self.program_path, out_dir=after_dir)
        rendered = run_figure_edit("render", program_path=self.program_path, out_dir=after_dir)

        self.assertTrue(Path(compiled["pptx"]).is_file())
        self.assertTrue(Path(rendered["preview"]).is_file())
        self.assertGreater(_differing_pixels(before_dir / "figure_preview.png", after_dir / "figure_preview.png"), 500)

    def test_the_command_applies_the_very_transforms_the_run_uses(self):
        program = _program()
        layout = {"align_grid": 0.05}
        routing = {"selections": [{"arrow_id": "arrow_ab", "path_percent": [[0.35, 0.42], [0.5, 0.3], [0.6, 0.42]]}]}
        from_command, report = apply_constraints(program, layout=layout, routing=routing)
        expected, _ = apply_layout_constraints(deepcopy(program), layout)
        expected, routed = apply_memory_routing_constraints(expected, routing)
        self.assertEqual(expected, from_command)
        self.assertEqual(["arrow_ab"], routed["applied_arrow_ids"])
        self.assertIn("layout", report)

    def test_the_program_is_the_only_thing_the_edit_changed(self):
        run_figure_edit("move-node", program_path=self.program_path, object_id="node_a", dx=0.05)
        self.assertTrue((self.root / "figure_edits.jsonl").is_file())
        self.assertFalse((self.root / "editable_composition.pptx").exists())
        self.assertEqual(0.2, read_json(self.program_path)["semantic_nodes"][0]["bbox_percent"]["x"])


if __name__ == "__main__":
    unittest.main()
