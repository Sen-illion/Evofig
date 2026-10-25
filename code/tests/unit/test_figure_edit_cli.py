"""Every bounded figure edit is JSON in, JSON out, and leaves a readable trail."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import build_parser
from rfs.composition import figure_edit as figure_edit_module
from rfs.composition.figure_edit import (
    FIGURE_EDIT_COMMANDS,
    FIGURE_EDIT_TRAIL,
    MUTATING_COMMANDS,
    FigureEditError,
    apply_constraints,
    attach_endpoint,
    figure_summary,
    move_object,
    read_edit_trail,
    reroute_arrow,
    resize_object,
    run_figure_edit,
    set_arrow_waypoints,
    set_label,
    set_panel_bounds,
)
from rfs.utils import read_json, write_json

MODULE_PATH = Path(figure_edit_module.__file__)


def _node(node_id, label, x, *, with_label_box=False):
    node = {
        "id": node_id,
        "label": label,
        "panel_id": "panel_a",
        "bbox_percent": {"x": x, "y": 0.35, "w": 0.2, "h": 0.15},
    }
    if with_label_box:
        node["label_bbox_percent"] = {"x": x + 0.02, "y": 0.51, "w": 0.16, "h": 0.05}
    return node


def _program():
    return {
        "canvas": {"width_px": 800, "height_px": 450, "width_in": 13.333, "height_in": 7.5, "background": "#FFFFFF"},
        "style": {"palette": ["#FFFFFF", "#4A90C2"], "arrow_weight_pt": 1.7},
        "panels": [{"id": "panel_a", "title": "Pipeline", "bbox_percent": {"x": 0.05, "y": 0.2, "w": 0.9, "h": 0.6}}],
        "cards": [],
        "slots": [],
        "semantic_nodes": [_node("node_a", "Encoder", 0.15, with_label_box=True), _node("node_b", "Decoder", 0.6)],
        "labels": [
            {
                "id": "label_arrow_ab",
                "text": "flows to",
                "target_id": "arrow_ab",
                "role": "relation_label",
                "bbox_percent": {"x": 0.42, "y": 0.4, "w": 0.1, "h": 0.04},
            }
        ],
        "arrows": [
            {
                "id": "arrow_ab",
                "source": "node_a",
                "target": "node_b",
                "label": "flows to",
                "path_percent": [[0.35, 0.42], [0.6, 0.42]],
            }
        ],
        "text_program": {
            "items": [
                {
                    "id": "text_node_a",
                    "text": "Encoder",
                    "text_role": "node_label",
                    "role": "node_label",
                    "owner_node_id": "node_a",
                    "target_id": "node_a",
                    "bbox_percent": {"x": 0.17, "y": 0.51, "w": 0.16, "h": 0.05},
                    "visible": True,
                    "measured_lines": ["Encoder"],
                    "measured_line_count": 1,
                }
            ]
        },
    }


class FigureEditSurfaceTests(unittest.TestCase):
    def test_the_cli_registers_exactly_the_command_vocabulary(self):
        parser = build_parser()
        figure = parser._subparsers._group_actions[0].choices["figure-edit"]
        self.assertEqual(tuple(figure._subparsers._group_actions[0].choices), FIGURE_EDIT_COMMANDS)

    def test_reading_commands_are_not_mutations(self):
        self.assertEqual(("move-node",), MUTATING_COMMANDS[:1])
        self.assertNotIn("show", MUTATING_COMMANDS)
        self.assertNotIn("recompile", MUTATING_COMMANDS)
        self.assertNotIn("render", MUTATING_COMMANDS)

    def test_the_module_never_imports_a_model_or_touches_a_deck_object(self):
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level:
                imported.add(node.module or "")
        for name in imported:
            self.assertFalse(name.startswith("rfs.providers"), name)
            self.assertFalse(name.startswith("pptx"), name)
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("Presentation(", source)

    def test_a_moved_node_stays_on_the_canvas(self):
        moved, change = move_object(_program(), "node_a", dx=0.9, dy=-0.9)
        box = moved["semantic_nodes"][0]["bbox_percent"]
        self.assertEqual(round(1.0 - 0.2, 4), box["x"])
        self.assertEqual(0.0, box["y"])
        self.assertEqual({"x": 0.15, "y": 0.35}, change["before"])

    def test_resizing_carries_the_label_box_with_its_node(self):
        resized, change = resize_object(_program(), "node_a", scale=2.0)
        node = resized["semantic_nodes"][0]
        self.assertEqual(0.4, node["bbox_percent"]["w"])
        self.assertEqual(0.3, node["bbox_percent"]["h"])
        self.assertEqual(0.32, node["label_bbox_percent"]["w"])
        self.assertEqual("label_bbox_percent", change["fields"][1])

    def test_resizing_refuses_a_zero_or_negative_scale(self):
        with self.assertRaises(FigureEditError):
            resize_object(_program(), "node_a", scale=0.0)

    def test_a_rewritten_label_is_rewritten_everywhere_it_is_kept(self):
        updated, change = set_label(_program(), "node_a", "Input encoder")
        self.assertEqual("Input encoder", updated["semantic_nodes"][0]["label"])
        self.assertEqual("Input encoder", updated["text_program"]["items"][0]["text"])
        self.assertNotIn("measured_lines", updated["text_program"]["items"][0])
        self.assertIn("node_a.label", change["changed"])
        self.assertIn("text_node_a.measured_lines", change["dropped"])

    def test_an_untouched_label_keeps_the_measurement_it_still_has(self):
        program = _program()
        program["text_program"]["items"].append(
            {
                "id": "text_node_b",
                "text": "Decoder",
                "text_role": "node_label",
                "role": "node_label",
                "owner_node_id": "node_b",
                "target_id": "node_b",
                "bbox_percent": {"x": 0.62, "y": 0.51, "w": 0.16, "h": 0.05},
                "visible": True,
                "measured_lines": ["Decoder"],
                "measured_line_count": 1,
            }
        )
        updated, change = set_label(program, "node_a", "Input encoder")
        neighbours = {item["id"]: item for item in updated["text_program"]["items"]}
        self.assertEqual(["Decoder"], neighbours["text_node_b"]["measured_lines"])
        self.assertNotIn("text_node_b.measured_lines", change["dropped"])

    def test_a_rewritten_relation_label_reaches_the_arrow_and_its_caption(self):
        updated, change = set_label(_program(), "arrow_ab", "feeds")
        self.assertEqual("feeds", updated["arrows"][0]["label"])
        self.assertEqual("feeds", updated["labels"][0]["text"])
        self.assertEqual("Encoder", updated["text_program"]["items"][0]["text"])
        self.assertIn("arrow_ab.label", change["changed"])

    def test_an_object_with_no_text_cannot_be_relabelled(self):
        with self.assertRaises(FigureEditError):
            set_label(_program(), "panel_a", "Nope")

    def test_waypoints_must_be_a_polyline_of_pairs(self):
        updated, change = set_arrow_waypoints(_program(), "arrow_ab", "[[0.1,0.2],[0.3,0.2]]")
        self.assertEqual([[0.1, 0.2], [0.3, 0.2]], updated["arrows"][0]["path_percent"])
        self.assertEqual("edited_waypoints", updated["arrows"][0]["route_generation_status"])
        self.assertEqual(2, change["after_point_count"])
        with self.assertRaises(FigureEditError):
            set_arrow_waypoints(_program(), "arrow_ab", [[0.1, 0.2]])
        with self.assertRaises(FigureEditError):
            set_arrow_waypoints(_program(), "arrow_ab", [[0.1, 0.2, 0.3], [0.4, 0.5]])

    def test_attaching_an_endpoint_drops_a_path_that_no_longer_fits(self):
        updated, change = attach_endpoint(_program(), "arrow_ab", target="node_a")
        self.assertEqual("node_a", updated["arrows"][0]["target"])
        self.assertNotIn("path_percent", updated["arrows"][0])
        self.assertTrue(change["dropped_path"])
        with self.assertRaises(FigureEditError):
            attach_endpoint(_program(), "arrow_ab", target="nowhere")

    def test_attaching_needs_at_least_one_endpoint(self):
        with self.assertRaises(FigureEditError):
            attach_endpoint(_program(), "arrow_ab")

    def test_panel_bounds_are_set_and_clamped(self):
        updated, change = set_panel_bounds(_program(), "panel_a", x=0.2, w=0.95)
        panel = updated["panels"][0]["bbox_percent"]
        self.assertEqual(0.2, panel["x"])
        self.assertAlmostEqual(0.8, panel["w"], places=4)
        self.assertEqual(0.05, change["before"]["x"])
        with self.assertRaises(FigureEditError):
            set_panel_bounds(_program(), "node_a", x=0.1)

    def test_constraints_run_the_two_compile_time_transforms(self):
        updated, change = apply_constraints(_program(), layout={"align_grid": 0.05})
        self.assertIn("layout", change)
        self.assertIn("rules", change["layout"])
        self.assertIn("arrows", updated)
        with self.assertRaises(FigureEditError):
            apply_constraints(_program())

    def test_rerouting_reports_which_arrows_moved(self):
        updated, change = reroute_arrow(_program(), "arrow_ab", route_style="rounded_elbow")
        self.assertEqual("rounded_elbow", updated["arrows"][0]["route_style"])
        self.assertIn("moved_arrows", change)
        self.assertIn("router_report", change)

    def test_show_lists_the_objects_and_describes_one_of_them(self):
        listing = figure_summary(_program())
        ids = {row["id"] for row in listing["objects"]}
        self.assertTrue({"panel_a", "node_a", "node_b", "arrow_ab", "label_arrow_ab", "text_node_a"} <= ids)
        self.assertEqual("arrow", next(row["kind"] for row in listing["objects"] if row["id"] == "arrow_ab"))
        described = figure_summary(_program(), "node_a")
        self.assertEqual("Encoder", described["object"]["record"]["label"])


class FigureEditRunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.program_path = self.root / "figure_program.json"
        write_json(self.program_path, _program())

    def test_a_refused_edit_leaves_the_program_and_the_trail_alone(self):
        before = self.program_path.read_bytes()
        with self.assertRaises(FigureEditError):
            run_figure_edit("move-node", program_path=self.program_path, object_id="nowhere", dx=0.1)
        self.assertEqual(before, self.program_path.read_bytes())
        self.assertEqual([], read_edit_trail(self.program_path))

    def test_an_unknown_command_is_refused(self):
        with self.assertRaises(FigureEditError):
            run_figure_edit("draw-something", program_path=self.program_path)

    def test_a_mutation_writes_the_program_once_and_appends_one_trail_line(self):
        result = run_figure_edit("move-node", program_path=self.program_path, object_id="node_b", x=0.1)
        self.assertTrue(result["ok"])
        self.assertEqual(0.1, read_json(self.program_path)["semantic_nodes"][1]["bbox_percent"]["x"])
        trail = read_edit_trail(self.program_path)
        self.assertEqual(1, len(trail))
        self.assertEqual("move-node", trail[0]["command"])
        self.assertEqual("node_b", trail[0]["arguments"]["object_id"])
        self.assertEqual(result["change"], trail[0]["changed"])

    def test_the_trail_keeps_every_edit_in_the_order_it_was_made(self):
        run_figure_edit("move-node", program_path=self.program_path, object_id="node_b", x=0.1)
        run_figure_edit("set-label", program_path=self.program_path, object_id="node_b", text="Head")
        run_figure_edit("resize-node", program_path=self.program_path, object_id="node_b", scale=1.2)
        trail = read_edit_trail(self.program_path)
        self.assertEqual(["move-node", "set-label", "resize-node"], [record["command"] for record in trail])
        self.assertTrue((self.root / FIGURE_EDIT_TRAIL).is_file())

    def test_show_writes_its_reading_beside_the_program(self):
        result = run_figure_edit("show", program_path=self.program_path, object_id="arrow_ab")
        self.assertTrue(Path(result["figure_show"]).is_file())
        self.assertEqual("arrow", json.loads(Path(result["figure_show"]).read_text(encoding="utf-8"))["object"]["kind"])

    def test_recompile_and_render_write_outputs_but_never_the_program(self):
        before = self.program_path.read_bytes()
        compiled = run_figure_edit("recompile", program_path=self.program_path)
        rendered = run_figure_edit("render", program_path=self.program_path)
        self.assertTrue(Path(compiled["pptx"]).is_file())
        self.assertTrue(Path(rendered["preview"]).is_file())
        self.assertEqual(before, self.program_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
