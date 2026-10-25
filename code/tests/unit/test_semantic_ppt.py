import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from rfs.composition.semantic_figure import build_semantic_figure_program, compile_semantic_ppt


class SemanticPptTests(unittest.TestCase):
    def test_compiles_contract_to_native_editable_shapes_text_and_connectors(self):
        specification = {
            "topology": "linear",
            "required_labels": ["Input Image", "Patch Encoder", "Transformer", "Prediction"],
            "inputs": [{"id": "image", "name": "Input Image"}],
            "modules": [
                {"id": "patch", "name": "Patch Encoder"},
                {"id": "transformer", "name": "Transformer"},
            ],
            "outputs": [{"id": "prediction", "name": "Prediction"}],
            "relations": [
                {"source": "image", "target": "patch", "type": "data_flow"},
                {"source": "patch", "target": "transformer", "type": "feature_flow"},
                {"source": "transformer", "target": "prediction", "type": "data_flow"},
            ],
        }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = compile_semantic_ppt(specification, root, aspect_ratio="16:9")

            self.assertTrue(report["ok"])
            self.assertEqual(report["node_count"], 4)
            self.assertEqual(report["connector_count"], 3)
            self.assertEqual(report["editable_layers"], ["native_shapes", "native_text", "native_connectors"])
            with zipfile.ZipFile(report["pptx"]) as archive:
                slide_xml = archive.read("ppt/slides/slide1.xml").decode("utf-8")
            for label in specification["required_labels"]:
                self.assertIn(label, slide_xml)
            self.assertGreaterEqual(slide_xml.count("<p:cxnSp>"), 3)
            self.assertGreaterEqual(slide_xml.count("RFS Semantic Node"), 4)
            self.assertGreaterEqual(slide_xml.count("RFS Connector"), 3)

            quality = json.loads((root / "composition_quality_report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(quality["semantic_nodes"]), 4)
            self.assertTrue(all(item["editable_in"] == "pptx" for item in quality["semantic_nodes"]))

    def test_feedback_contract_uses_external_dashed_editable_loop(self):
        specification = {
            "topology": "feedback",
            "required_labels": ["Sample", "Measure", "Update", "Estimate"],
            "inputs": [{"id": "sample", "name": "Sample"}],
            "modules": [{"id": "measure", "name": "Measure"}, {"id": "update", "name": "Update"}],
            "outputs": [{"id": "estimate", "name": "Estimate"}],
            "relations": [
                {"source": "sample", "target": "measure", "type": "data_flow"},
                {"source": "measure", "target": "update", "type": "data_flow"},
                {"source": "update", "target": "estimate", "type": "data_flow"},
                {"source": "estimate", "target": "measure", "type": "feedback_loop", "label": "revise"},
            ],
        }

        program = build_semantic_figure_program(specification, aspect_ratio="3:2")
        loop = next(item for item in program["arrows"] if item["type"] == "feedback_loop")
        self.assertEqual(loop["line_pattern"], "dash")
        self.assertEqual(loop["route_style"], "outer_feedback")
        self.assertEqual(program["labels"][0]["text"], "revise")

    def test_preserves_planned_panel_binding_for_downstream_style_resolution(self):
        specification = {
            "topology": "linear",
            "required_labels": ["Agent"],
            "modules": [{"id": "agent", "name": "Agent"}],
            "relations": [],
        }
        semantic_plan = {
            "applied": True,
            "topology": "linear",
            "nodes": [{
                "id": "agent",
                "label": "Agent",
                "field": "modules",
                "panel_id": "panel_agent",
                "bbox_percent": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.12},
            }],
            "connectors": [],
        }

        program = build_semantic_figure_program(specification, semantic_plan=semantic_plan)

        self.assertEqual(program["semantic_nodes"][0]["panel_id"], "panel_agent")

    def test_long_words_and_relation_labels_are_compiled_for_readable_rendering(self):
        specification = {
            "topology": "linear",
            "required_labels": ["Input", "Encoder", "contextualized embeddings"],
            "inputs": [{"id": "input", "name": "Input"}],
            "modules": [{"id": "encoder", "name": "Encoder"}],
            "outputs": [{"id": "output", "name": "contextualized embeddings"}],
            "relations": [
                {"source": "input", "target": "encoder", "type": "data_flow", "label": "serialize"},
                {"source": "encoder", "target": "output", "type": "data_flow"},
            ],
        }

        program = build_semantic_figure_program(specification)
        output = next(item for item in program["semantic_nodes"] if item["id"] == "output")
        relation_label = program["labels"][0]

        self.assertLessEqual(output["font_size_pt"], 10.5)
        self.assertLess(relation_label["bbox_percent"]["w"], 0.13)
        self.assertLess(relation_label["bbox_percent"]["y"], 0.5)

if __name__ == "__main__":
    unittest.main()
