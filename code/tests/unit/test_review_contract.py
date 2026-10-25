from __future__ import annotations

import unittest

from rfs.contracts.review_ground_truth import build_repair_plan, compile_review_ground_truth


class ReviewContractTests(unittest.TestCase):
    def test_ground_truth_embeds_page_section_and_original_excerpt(self):
        plan = {
            "figure_specification": {
                "central_claim": {"text": "Evidence-constrained reasoning reduces unsupported conclusions.", "evidence_ids": ["E1"]},
                "modules": [{"id": "parser", "name": "Document Parser", "evidence_ids": ["E2"]}],
                "relations": [{"source": "parser", "target": "answer", "type": "data_flow", "evidence_ids": ["E2"]}],
                "outputs": [{"id": "answer", "name": "Evidence-linked Answer", "evidence_ids": ["E2"]}],
                "training_flow": ["Training loss"],
                "must_show": ["Training loss", "Document Parser"],
                "forbidden_inventions": ["External memory bank"],
                "uncertainties": ["The internal decoder architecture is not specified"],
            }
        }
        evidence = [
            {"id": "E1", "page": 1, "section_hint": "Abstract", "text": "The method reduces unsupported conclusions.", "confidence": 1.0, "source": "pymupdf"},
            {"id": "E2", "page": 3, "section_hint": "Method", "text": "The Document Parser creates page-aware passages used for the final answer.", "confidence": 0.99, "source": "pymupdf"},
            {"id": "E3", "page": 4, "section_hint": "Training", "text": "Training loss is optimized over synthetic tasks.", "confidence": 0.98, "source": "pymupdf"},
        ]

        contract = compile_review_ground_truth(plan, evidence)

        parser = next(item for item in contract["priorities"] if item["label"] == "Document Parser")
        self.assertEqual(parser["evidence"][0]["page"], 3)
        self.assertEqual(parser["evidence"][0]["section"], "Method")
        self.assertIn("page-aware passages", parser["evidence"][0]["text"])
        self.assertTrue(contract["strict_evidence_required"])
        self.assertEqual(contract["forbidden"][0]["label"], "External memory bank")
        self.assertIn("not specified", contract["unknown"][0]["label"])
        training = next(item for item in contract["priorities"] if item["label"] == "Training loss")
        self.assertEqual(training["evidence_link_method"], "deterministic_text_match")
        self.assertEqual(training["evidence"][0]["evidence_id"], "E3")
        self.assertEqual(sum(item["label"] == "Training loss" for item in contract["priorities"]), 1)
        self.assertEqual(sum(item["label"] == "Document Parser" for item in contract["priorities"]), 1)
        self.assertTrue(training["required"], "must_show must keep an otherwise optional training item required")

    def test_training_and_inference_context_are_optional_unless_explicitly_visible(self):
        plan = {
            "figure_specification": {
                "training_flow": [
                    {"id": "train_context", "step": "Optimize auxiliary loss", "evidence_ids": ["E1"]},
                    {"id": "train_visible", "step": "Supervised fine-tuning", "required": True, "evidence_ids": ["E1"]},
                ],
                "inference_flow": [{"id": "infer_context", "step": "Run classifier", "evidence_ids": ["E1"]}],
            }
        }
        evidence = [{"id": "E1", "page": 2, "section_hint": "Method", "text": "The model uses an auxiliary loss, supervised fine-tuning, and a classifier.", "confidence": 1.0, "source": "pymupdf"}]

        contract = compile_review_ground_truth(plan, evidence)
        priorities = {item["id"]: item for item in contract["priorities"]}

        self.assertFalse(priorities["train_context"]["required"])
        self.assertTrue(priorities["train_visible"]["required"])
        self.assertFalse(priorities["infer_context"]["required"])

    def test_repair_plan_prioritizes_paper_conflicts_and_carries_evidence_ids(self):
        review = {
            "preserve": ["correct left panel"],
            "paper_alignment": {
                "missing_priorities": [{"id": "GT_MODULE_parser", "label": "Document Parser", "reason": "missing", "evidence_ids": ["E2"]}],
                "evidence_conflicts": [],
                "unsupported_visual_claims": ["Unsupported memory bank"],
            },
            "scientific": {"missing_modules": [], "missing_relations": [], "reversed_relations": [], "invented_items": [], "role_mismatches": [], "containment_mismatches": []},
            "topology": {"missing_relations": [], "reversed_relations": [], "bypassed_relations": [], "invented_relations": []},
            "ocr": {"missing_labels": ["Document Parser"], "misspelled_labels": [], "duplicate_labels": [], "forbidden_labels_found": [], "unexpected_labels": ["Supported Detail"], "unsupported_unexpected_labels": []},
            "template": {"passed": True},
            "aesthetic": {"passed": True},
            "basic": {"visual_enrichment_passed": True},
            "repair": [],
            "hard_errors": [],
        }

        repair = build_repair_plan(review, "candidate_01", 1)

        self.assertEqual(repair["items"][0]["issue_type"], "paper_missing_priority")
        self.assertEqual(repair["items"][0]["severity"], "critical")
        self.assertEqual(repair["items"][0]["evidence_ids"], ["E2"])
        self.assertEqual(repair["items"][0]["preserve"], ["correct left panel"])
        ocr_item = next(item for item in repair["items"] if item["issue_type"] == "ocr_missing_labels")
        self.assertEqual(ocr_item["owner"], "overlay_compiler")
        self.assertEqual(repair["items"][0]["owner"], "image2_visual_substrate")


if __name__ == "__main__":
    unittest.main()
