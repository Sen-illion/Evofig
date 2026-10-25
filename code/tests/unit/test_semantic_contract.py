import tempfile
import unittest
from pathlib import Path

from rfs.contracts.semantic import apply_paper_semantic_contract


class SemanticContractTests(unittest.TestCase):
    def test_card_ids_and_ocr_geometry_drive_semantic_mapping_and_hidden_innovation_is_skipped(self):
        program = {
            "slots": [
                {"id": "slot_micro_left", "bbox_percent": {"x": 0.08, "y": 0.3, "w": 0.08, "h": 0.1}},
                {"id": "slot_micro_middle", "bbox_percent": {"x": 0.43, "y": 0.3, "w": 0.08, "h": 0.1}},
                {"id": "slot_micro_right", "bbox_percent": {"x": 0.78, "y": 0.3, "w": 0.08, "h": 0.1}},
            ],
            "cards": [
                {"id": "card_exact_input", "bbox_percent": {"x": 0.05, "y": 0.2, "w": 0.2, "h": 0.35}},
                {"id": "card_paper_method", "bbox_percent": {"x": 0.4, "y": 0.2, "w": 0.2, "h": 0.35}},
                {"id": "card_exact_output", "bbox_percent": {"x": 0.75, "y": 0.2, "w": 0.2, "h": 0.35}},
            ],
            "panels": [],
            "arrows": [
                {"id": "cross_card_guess", "source_id": "slot_micro_left", "target_id": "slot_micro_middle", "path_percent": [[0.16, 0.35], [0.43, 0.35]]},
                {"id": "local_method_detail", "source_id": "slot_micro_middle", "target_id": "slot_micro_middle", "path_percent": [[0.44, 0.32], [0.5, 0.38]]},
            ],
            "text_program": {"items": [
                {"id": "ocr_input", "text": "Exact Input", "bbox_percent": {"x": 0.08, "y": 0.22, "w": 0.1, "h": 0.04}},
                {"id": "ocr_method", "text": "Paper Method", "bbox_percent": {"x": 0.44, "y": 0.22, "w": 0.1, "h": 0.04}},
                {"id": "ocr_output", "text": "Exact Output", "bbox_percent": {"x": 0.79, "y": 0.22, "w": 0.1, "h": 0.04}},
            ]},
        }
        contract = {"figure_specification": {
            "required_labels": ["Exact Input", "Paper Method", "Exact Output"],
            "inputs": [{"id": "input", "name": "Exact Input", "evidence_ids": ["E1"]}],
            "modules": [{"id": "method", "name": "Paper Method", "evidence_ids": ["E2"]}],
            "outputs": [{"id": "output", "name": "Exact Output", "evidence_ids": ["E3"]}],
            "innovations": [{"id": "hidden", "name": "Background Innovation", "evidence_ids": ["E4"]}],
            "relations": [],
        }}
        with tempfile.TemporaryDirectory() as tmp:
            updated, report = apply_paper_semantic_contract(program, contract, Path(tmp))
            updated, second_report = apply_paper_semantic_contract(updated, contract, Path(tmp))

        mapped = {item["entity_id"]: item["object_id"] for item in report["mappings"]}
        self.assertEqual(mapped, {
            "input": "card_exact_input",
            "method": "card_paper_method",
            "output": "card_exact_output",
        })
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["low_confidence_mappings"], [])
        self.assertIn("local_method_detail", {item["id"] for item in updated["arrows"]})
        self.assertNotIn("cross_card_guess", {item["id"] for item in updated["arrows"]})
        self.assertEqual(second_report["status"], "pass")
        self.assertEqual({item["entity_id"]: item["object_id"] for item in second_report["mappings"]}, mapped)
        self.assertEqual(len([item for item in updated["text_program"]["items"] if item.get("semantic_entity_id")]), 3)
        self.assertNotIn("Background Innovation", {item["text"] for item in updated["text_program"]["items"]})

    def test_exact_paper_labels_and_relations_override_image_interpretation(self):
        program = {
            "slots": [
                {"id": "left", "bbox_percent": {"x": 0.05, "y": 0.25, "w": 0.2, "h": 0.3}},
                {"id": "middle", "bbox_percent": {"x": 0.4, "y": 0.25, "w": 0.2, "h": 0.3}},
                {"id": "right", "bbox_percent": {"x": 0.75, "y": 0.25, "w": 0.2, "h": 0.3}},
            ],
            "cards": [],
            "panels": [],
            "arrows": [{"id": "image_guess", "path_percent": [[0.1, 0.1], [0.9, 0.9]]}],
            "text_program": {"items": [{"id": "ocr_wrong", "text": "lnput", "bbox_percent": {"x": 0.08, "y": 0.3, "w": 0.1, "h": 0.04}}]},
        }
        contract = {"figure_specification": {
            "inputs": [{"id": "input", "name": "Exact Input", "evidence_ids": ["E1"]}],
            "modules": [{"id": "method", "name": "Paper Method", "evidence_ids": ["E2"]}],
            "outputs": [{"id": "output", "name": "Exact Output", "evidence_ids": ["E3"]}],
            "innovations": [],
            "relations": [
                {"source": "input", "target": "method", "type": "data_flow", "evidence_ids": ["E4"]},
                {"source": "method", "target": "output", "type": "data_flow", "evidence_ids": ["E5"]},
            ],
        }}
        with tempfile.TemporaryDirectory() as tmp:
            updated, report = apply_paper_semantic_contract(program, contract, Path(tmp))

            labels = {item["text"] for item in updated["text_program"]["items"]}
            self.assertEqual(labels, {"lnput"})
            self.assertEqual(report["mapped_relation_count"], 0)
            self.assertEqual(len(report["rejected_mappings"]), 3)
            self.assertEqual(updated["arrows"][0]["id"], "image_guess")
            self.assertTrue((Path(tmp) / "semantic_binding_report.json").exists())


if __name__ == "__main__":
    unittest.main()
