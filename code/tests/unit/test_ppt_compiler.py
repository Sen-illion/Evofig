import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from rfs.composition.preview import render_rebuild_preview
from rfs.composition.pptx import compile_ppt


class PptCompilerArrowTests(unittest.TestCase):
    def test_preview_includes_editable_title_block(self):
        with tempfile.TemporaryDirectory() as tmp, patch("rfs.composition.preview._draw_text") as draw_text:
            render_rebuild_preview(
                {
                    "canvas": {"width_px": 1600, "height_px": 900, "height_in": 7.5, "background": "#FFFFFF"},
                    "title_block": {
                        "title": "Paper Method",
                        "subtitle": "Editable paper framework",
                        "bbox_percent": {"x": .04, "y": .015, "w": .92, "h": .08},
                    },
                    "panels": [], "cards": [], "semantic_nodes": [], "arrows": [], "assets": [], "slots": [],
                    "text_program": {"items": []},
                },
                tmp,
                Path(tmp) / "preview.png",
            )
        rendered_text = [call.args[1]["text"] for call in draw_text.call_args_list]
        self.assertEqual(rendered_text, ["Paper Method", "Editable paper framework"])

    def test_panel_title_is_rendered_once_when_owned_by_text_program(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            program = {
                "canvas": {"width_in": 10.0, "height_in": 5.0, "background": "#FFFFFF"},
                "style": {"palette": ["#FFFFFF", "#4A90C2"]},
                "panels": [{"id": "panel_a", "title": "Conversation Generation Pipeline", "bbox_percent": {"x": .05, "y": .05, "w": .9, "h": .8}}],
                "cards": [], "slots": [], "arrows": [], "labels": [],
                "text_program": {"items": [{
                    "id": "panel_title", "text_role": "panel_title", "panel_id": "panel_a",
                    "text": "Conversation Generation Pipeline", "bbox_percent": {"x": .05, "y": .05, "w": .9, "h": .05}, "font_size_pt": 16,
                }]},
                "export_targets": [{"type": "pptx", "path": "editable_composition.pptx"}],
            }
            pptx = compile_ppt(program, out)
            with zipfile.ZipFile(pptx) as archive:
                slide_xml = archive.read("ppt/slides/slide1.xml").decode("utf-8")
            self.assertEqual(slide_xml.count("Conversation Generation Pipeline"), 1)

    def test_text_program_owns_node_label_without_duplicate_shape_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            program = {
                "canvas": {"width_in": 10.0, "height_in": 5.0, "background": "#FFFFFF"},
                "style": {"palette": ["#FFFFFF", "#4A90C2"]},
                "panels": [], "cards": [], "slots": [], "arrows": [], "labels": [],
                "semantic_nodes": [{"id": "n1", "label": "Node", "bbox_percent": {"x": .1, "y": .1, "w": .3, "h": .3}, "label_bbox_percent": {"x": .15, "y": .32, "w": .2, "h": .05}}],
                "text_program": {"items": [{"id": "node_text", "role": "node_label", "target_id": "n1", "text": "Node", "bbox_percent": {"x": .15, "y": .32, "w": .2, "h": .05}}]},
                "export_targets": [{"type": "pptx", "path": "editable_composition.pptx"}],
            }
            compile_ppt(program, out)
            report = json.loads((out / "composition_quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["text_rendering"]["text_program_owned_node_count"], 1)
            self.assertEqual(report["text_rendering"]["duplicate_text_owner_count"], 0)

    def test_invisible_text_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            program = {"canvas": {"width_in": 5.0, "height_in": 3.0}, "style": {"palette": ["#FFF"]}, "panels": [], "cards": [], "slots": [], "arrows": [], "labels": [], "text_program": {"items": [{"id": "hidden", "text": "hidden", "visible": False, "bbox_percent": {"x": .1, "y": .1, "w": .2, "h": .1}}]}, "export_targets": [{"type": "pptx", "path": "editable_composition.pptx"}]}
            compile_ppt(program, out)
            report = json.loads((out / "composition_quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["text_rendering"]["rendered_text_count"], 0)
            self.assertEqual(report["text_rendering"]["skipped_invisible_text_count"], 1)
    def test_textbox_defaults_to_center_alignment_and_reports_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            program = {
                "canvas": {"width_in": 10.0, "height_in": 5.0, "background": "#FFFFFF"},
                "style": {"palette": ["#FFFFFF", "#4A90C2"]},
                "panels": [],
                "cards": [],
                "slots": [],
                "arrows": [],
                "labels": [],
                "groups": [],
                "text_program": {
                    "items": [{
                        "id": "ocr_left_label",
                        "text": "Left in source",
                        "role": "label",
                        "target_id": "slot_a",
                        "bbox_percent": {"x": 0.20, "y": 0.20, "w": 0.30, "h": 0.10},
                        "align": "left",
                        "font_size_pt": 12,
                    }]
                },
                "export_targets": [{"type": "pptx", "path": "editable_composition.pptx"}],
            }

            pptx = compile_ppt(program, out)
            self.assertTrue(pptx.exists())
            report = json.loads((out / "composition_quality_report.json").read_text(encoding="utf-8"))
            text_item = report["text"][0]
            self.assertEqual(text_item["source_align"], "left")
            self.assertEqual(text_item["rendered_align"], "center")
            self.assertEqual(text_item["text_anchor_policy"], "center_text_in_textbox")

    def test_multisegment_and_dashed_arrows_are_rendered_as_ppt_connectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            program = {
                "canvas": {"width_in": 10.0, "height_in": 5.0, "background": "#FFFFFF"},
                "style": {
                    "palette": ["#FFFFFF", "#E17721", "#6B57C8", "#1B9A94"],
                    "reference_palette": ["#FFFFFF", "#E17721", "#6B57C8", "#1B9A94"],
                    "color_tokens": [{
                        "token_id": "arrow_orange_001",
                        "hex": "#E17721",
                        "usage": "arrow_or_connector_stroke",
                    }],
                    "arrow_weight_pt": 2.25,
                },
                "panels": [{
                    "id": "panel_a",
                    "title": "Panel A",
                    "bbox_percent": {"x": 0.05, "y": 0.05, "w": 0.90, "h": 0.80},
                    "editable_in": "pptx",
                }],
                "cards": [{
                    "id": "card_a",
                    "semantic_role": "outer_group_boundary",
                    "bbox_percent": {"x": 0.08, "y": 0.16, "w": 0.76, "h": 0.54},
                    "shape_kind": "rect",
                    "stroke_color": "#59AFCB",
                    "stroke_width_pt": 1.5,
                    "dash_style": "dash",
                    "fill_transparency": 1.0,
                    "z_index": 12,
                }],
                "slots": [
                    {
                        "id": "slot_a",
                        "asset_id": "asset_a",
                        "paper_concept": "A",
                        "bbox_percent": {"x": 0.10, "y": 0.20, "w": 0.10, "h": 0.10},
                        "composition_type": "full_frame_icon",
                    },
                    {
                        "id": "slot_b",
                        "asset_id": "asset_b",
                        "paper_concept": "B",
                        "bbox_percent": {"x": 0.70, "y": 0.55, "w": 0.10, "h": 0.10},
                        "composition_type": "full_frame_icon",
                    },
                ],
                "arrows": [
                    {
                        "id": "multi_a",
                        "source": "slot_a",
                        "target": "slot_b",
                        "source_id": "slot_a",
                        "target_id": "slot_b",
                        "source_anchor": "right_mid",
                        "target_anchor": "left_mid",
                        "control_kind": "elbow_connector",
                        "path_percent": [[0.20, 0.25], [0.45, 0.25], [0.45, 0.60], [0.70, 0.60]],
                        "style_token_id": "arrow_orange_001",
                        "editable_in": "pptx",
                        "render_policy": "ppt_shape_not_image_asset",
                    },
                    {
                        "id": "loop_a",
                        "source": "slot_a",
                        "target": "slot_b",
                        "source_id": "slot_a",
                        "target_id": "slot_b",
                        "source_anchor": "top_mid",
                        "target_anchor": "bottom_mid",
                        "control_kind": "dashed_loop",
                        "path_percent": [[0.30, 0.30], [0.40, 0.20], [0.50, 0.30], [0.40, 0.40], [0.30, 0.30]],
                        "style_token_id": "arrow_orange_001",
                        "editable_in": "pptx",
                        "render_policy": "ppt_shape_not_image_asset",
                    },
                ],
                "labels": [],
                "groups": [],
                "export_targets": [{"type": "pptx", "path": "editable_composition.pptx"}],
            }

            pptx = compile_ppt(program, out)
            self.assertTrue(pptx.exists())

            report = json.loads((out / "composition_quality_report.json").read_text(encoding="utf-8"))
            by_id = {item["arrow_id"]: item for item in report["arrows"]}
            self.assertEqual(by_id["multi_a"]["segment_count"], 3)
            self.assertEqual(by_id["loop_a"]["segment_count"], 4)
            self.assertEqual(report["cards"][0]["card_id"], "card_a")
            self.assertEqual(report["cards"][0]["render_policy"], "ppt_shape_not_image_asset")

            with zipfile.ZipFile(pptx) as archive:
                slide_xml = archive.read("ppt/slides/slide1.xml").decode("utf-8")
            self.assertGreaterEqual(slide_xml.count("<p:cxnSp>"), 7)
            self.assertGreaterEqual(slide_xml.count("tailEnd"), 2)
            self.assertIn("prstDash", slide_xml)

    def test_oval_card_renders_as_ppt_ellipse_and_preview_ellipse(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            program = {
                "canvas": {"width_px": 100, "height_px": 100, "width_in": 5.0, "height_in": 5.0, "background": "#FFFFFF"},
                "style": {"palette": ["#FFFFFF", "#D94141"]},
                "panels": [],
                "cards": [{
                    "id": "tree_node",
                    "bbox_percent": {"x": 0.20, "y": 0.20, "w": 0.60, "h": 0.60},
                    "shape_kind": "oval",
                    "fill_color": "#D94141",
                    "stroke_color": "#1E2A33",
                    "stroke_width_pt": 1.0,
                    "fill_transparency": 0.0,
                }],
                "slots": [],
                "arrows": [],
                "labels": [],
                "groups": [],
                "export_targets": [{"type": "pptx", "path": "editable_composition.pptx"}],
            }

            pptx = compile_ppt(program, out)
            self.assertTrue(pptx.exists())
            with zipfile.ZipFile(pptx) as archive:
                slide_xml = archive.read("ppt/slides/slide1.xml").decode("utf-8")
            self.assertIn('prst="ellipse"', slide_xml)

            preview = render_rebuild_preview(program, out, out / "preview.png")
            image = Image.open(preview).convert("RGB")
            self.assertEqual(image.getpixel((50, 50)), (217, 65, 65))
            self.assertEqual(image.getpixel((21, 21)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
