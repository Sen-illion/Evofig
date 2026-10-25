"""Tests for raw evidence extraction contracts and D0 selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training.d0_selection import select_d0
from rfs.judge_training.evidence_schema import normalize_structured_evidence, repair_unique_evidence_pages, validate_raw_evidence, validate_structured_evidence
from rfs.judge_training.structured_evidence import build_source_catalog, materialize_source_spans
from rfs.judge_training.structured_evidence import build_prompt
from rfs.judge_training.evidence_reviewer import review_one_evidence_pack
from rfs.judge_training.corruption_generator import CORRUPTION_TYPES, GATE_CORRUPTION_TYPES, plan_corruptions
from rfs.judge_training.corruption_runner import run_corruptions
from rfs.judge_training.corruption_verifier import _normalize_verdict
from rfs.judge_training.local_corruptions import apply_local_corruption, generate_region_candidates


class JudgeEvidenceTests(unittest.TestCase):
    def test_validates_raw_and_grounded_structured_evidence(self):
        raw = {"paper_id": "p1", "pdf_sha256": "abc", "status": "success", "pages": [{"page": 1, "text": "Encoder extracts features."}]}
        structured = {"schema_version": "judge-paper-evidence-v001", "paper_id": "p1", "entities": [{"entity_id": "encoder", "evidence_ids": ["e1"]}], "relations": [], "core_contributions": [], "training_flow": [], "inference_flow": [], "evidence_spans": [{"evidence_id": "e1", "page": 1, "quote": "Encoder extracts features."}]}
        self.assertEqual(validate_raw_evidence(raw), [])
        self.assertEqual(validate_structured_evidence(structured, raw), [])

    def test_normalizes_provider_field_aliases(self):
        normalized = normalize_structured_evidence({
            "evidence_spans": [{"evidence_id": "e1", "text": "quote"}],
            "relations": [{"source_entity_id": "a", "target_entity_id": "b"}],
        })
        self.assertEqual(normalized["evidence_spans"][0]["quote"], "quote")
        self.assertEqual(normalized["relations"][0]["source"], "a")
        self.assertEqual(normalized["relations"][0]["target"], "b")

    def test_repairs_unique_quote_page(self):
        record = {"evidence_spans": [{"evidence_id": "e1", "page": 1, "quote": "A sentence."}]}
        raw = {"pages": [{"page": 2, "text": "A sentence."}]}
        self.assertEqual(repair_unique_evidence_pages(record, raw), ["e1: page 1 -> 2"])
        self.assertEqual(record["evidence_spans"][0]["page"], 2)

    def test_materializes_exact_source_span(self):
        raw = {"pages": [{"page": 3, "text": "First exact paragraph.\n\nSecond exact paragraph."}]}
        catalog = build_source_catalog(raw)
        record = {"evidence_spans": [{"evidence_id": "e1", "source_id": catalog[1]["source_id"], "page": 99, "quote": "paraphrase"}]}
        self.assertEqual(materialize_source_spans(record, catalog), [])
        self.assertEqual(record["evidence_spans"][0]["page"], 3)
        self.assertEqual(record["evidence_spans"][0]["quote"], "Second exact paragraph.")

    def test_semantic_reviewer_applies_and_validates_correction(self):
        raw = {"paper_id": "p1", "pages": [{"page": 1, "text": "Encoder produces features."}]}
        current = {"schema_version": "judge-paper-evidence-v001", "paper_id": "p1", "entities": [], "relations": [], "core_contributions": [], "training_flow": [], "inference_flow": [], "evidence_spans": []}
        source_id = build_source_catalog(raw)[0]["source_id"]
        corrected = {**current, "entities": [{"entity_id": "encoder", "evidence_ids": ["e1"]}], "evidence_spans": [{"evidence_id": "e1", "source_id": source_id, "page": None, "quote": ""}]}
        def adapter(*_args, **_kwargs):
            return {"decision": "revise", "summary": "missing encoder", "issues": [], "corrected_contract": corrected}
        audit, result, errors = review_one_evidence_pack(raw, current, model="test", adapter=adapter)
        self.assertEqual(audit["decision"], "revise")
        self.assertEqual(errors, [])
        self.assertEqual(result["evidence_spans"][0]["quote"], "Encoder produces features.")

    def test_source_catalog_materialization_is_thread_independent(self):
        raw = {"pages": [{"page": 1, "text": "Exact source."}]}
        first = build_source_catalog(raw)
        second = build_source_catalog(raw)
        self.assertEqual(first, second)

    def test_prompt_requires_typed_contributions(self):
        prompt = build_prompt({"paper_id": "p1", "pages": [{"page": 1, "text": "Method."}]})
        self.assertIn('"contribution": "supported contribution statement"', prompt)
        self.assertIn("never output contribution strings", prompt)

    def test_plans_one_deterministic_corruption_per_class(self):
        contract = {"entities": [{"entity_id": "encoder"}, {"entity_id": "head"}], "relations": [{"source": "encoder", "target": "head", "relation_type": "feeds"}], "training_flow": [{"step_description": "train the encoder"}, {"step_description": "freeze the backbone"}], "inference_flow": [{"step_description": "decode the answer"}], "expected_figure_content": {"required_entities": ["encoder", "head"], "must_not_invent": ["fake_decoder"]}}
        first = plan_corruptions("p1", "f1", "figure.png", contract, "train")
        second = plan_corruptions("p1", "f1", "figure.png", contract, "train")
        self.assertEqual(first, second)
        self.assertEqual({item["corruption_type"] for item in first}, set(CORRUPTION_TYPES))
        planned = {item["corruption_type"]: item for item in first}
        for kind in GATE_CORRUPTION_TYPES:
            self.assertIn(kind, planned)
        for kind in ("module_order_error", "overlap"):
            targets = planned[kind]["target"]
            self.assertEqual(len(targets), 2)
            self.assertNotEqual(targets[0], targets[1])

    def test_local_corruption_injects_the_order_and_overlap_gate_defects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from PIL import Image, ImageDraw
            source = root / "source.png"
            image = Image.new("RGB", (200, 100), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 30, 79, 69), fill="blue")
            draw.rectangle((120, 30, 179, 69), fill="red")
            image.save(source)
            base = {"candidate_id": "c1", "paper_id": "p1", "figure_id": "f1", "split": "train", "final_label": "好（科研）", "source_image": str(source), "target_id": "encoder", "bbox_percent": {"x": 0.1, "y": 0.3, "w": 0.3, "h": 0.4}, "secondary_bbox_percent": {"x": 0.6, "y": 0.3, "w": 0.3, "h": 0.4}, "secondary_target_id": "decoder", "swap_axis": "horizontal", "review_status": "approved"}
            swapped = apply_local_corruption({**base, "corruption_type": "module_order_error"}, root / "order.png")
            moved = apply_local_corruption({**base, "corruption_type": "overlap"}, root / "overlap.png")
            self.assertEqual((swapped["status"], swapped["operation"]), ("generated", "swap_module_positions"))
            self.assertEqual((moved["status"], moved["operation"]), ("generated", "shift_module_into_neighbour"))
            self.assertTrue(swapped["automatic_gate"]["locality_passed"])
            self.assertTrue(moved["automatic_gate"]["locality_passed"])
            with Image.open(root / "order.png") as order_image:
                self.assertEqual(order_image.size, (200, 100))
                self.assertEqual(order_image.getpixel((50, 50)), (255, 0, 0))
                self.assertEqual(order_image.getpixel((150, 50)), (0, 0, 255))
            with Image.open(root / "overlap.png") as overlap_image:
                self.assertEqual(overlap_image.getpixel((100, 50)), (0, 0, 255))

    def test_region_candidates_emit_order_and_overlap_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from PIL import Image
            source = root / "source.png"
            Image.new("RGB", (200, 100), "white").save(source)
            figures = root / "figures.jsonl"
            figures.write_text(json.dumps({"figure_id": "f1", "paper_id": "p1", "figure_path": str(source), "split": "train", "final_label": "好（科研）"}, ensure_ascii=False) + "\n", encoding="utf-8")
            structured = root / "structured"
            structured.mkdir()
            contract = {"entities": [{"entity_id": "encoder"}, {"entity_id": "decoder"}], "relations": [], "training_flow": [], "inference_flow": [], "expected_figure_content": {"required_entities": ["encoder", "decoder"]}}
            (structured / "p1.json").write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")

            def fake_ocr(_path, _lang):
                return [
                    {"text": "encoder", "confidence": 0.99, "quad": [[20, 30], [80, 30], [80, 70], [20, 70]]},
                    {"text": "decoder", "confidence": 0.99, "quad": [[120, 30], [180, 30], [180, 70], [120, 70]]},
                ]
            report = generate_region_candidates(figures, structured, root / "candidates.jsonl", per_type=1, ocr_function=fake_ocr)
            self.assertEqual(report["counts"]["module_order_error"], 1)
            self.assertEqual(report["counts"]["overlap"], 1)
            rows = {row["corruption_type"]: row for row in (json.loads(line) for line in (root / "candidates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
            self.assertEqual(rows["module_order_error"]["target_id"], "encoder<->decoder")
            self.assertEqual(rows["module_order_error"]["secondary_target_id"], "decoder")
            self.assertEqual(rows["module_order_error"]["swap_axis"], "horizontal")
            self.assertEqual(rows["overlap"]["target_id"], "encoder")

    def test_corruption_runner_resumes_and_records_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            from PIL import Image
            Image.new("RGB", (16, 16), "white").save(source)
            requests_path = root / "requests.jsonl"
            requests_path.write_text(json.dumps({"corruption_id": "f1__missing", "paper_id": "p1", "figure_id": "f1", "split": "train", "source_image": str(source), "corruption_type": "missing_core_entity", "edit_instruction": "remove", "expected_error": "missing_core_entity", "status": "planned"}) + "\n", encoding="utf-8")

            def fake_edit(source_path, prompt, output, **kwargs):
                Image.open(source_path).save(output)
                return {"model": "fake", "endpoint": "local", "attempts": 1}

            result_path = root / "results.jsonl"
            first = run_corruptions(requests_path, result_path, root / "images", edit_function=fake_edit)
            second = run_corruptions(requests_path, result_path, root / "images", edit_function=fake_edit)
            row = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(first["generated"], 1)
            self.assertEqual(second["completed_this_run"], 0)
            self.assertEqual(len(row["output_sha256"]), 64)

    def test_corruption_verdict_requires_exact_visible_error(self):
        raw = {"target_error_visible": True, "detected_error_type": "missing_core_entity", "scientific_content_preserved_except_target": True, "source_identity_preserved": True, "legible": True, "confidence": 0.9, "evidence": ["module absent"], "unintended_changes": [], "reason": "visible"}
        self.assertTrue(_normalize_verdict(raw, "missing_core_entity")["passed"])
        raw["unintended_changes"] = ["another arrow changed"]
        self.assertFalse(_normalize_verdict(raw, "missing_core_entity")["passed"])
        raw.update({"detected_error_type": "correct_but_ugly", "unintended_changes": [], "legible": False})
        self.assertTrue(_normalize_verdict(raw, "correct_but_ugly")["passed"])

    def test_local_corruption_is_deterministic_and_size_preserving(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from PIL import Image
            source = root / "source.png"
            image = Image.new("RGB", (100, 80), "white")
            from PIL import ImageDraw
            ImageDraw.Draw(image).rectangle((20, 16, 40, 32), fill="blue")
            image.save(source)
            review = {"candidate_id": "c1", "paper_id": "p1", "figure_id": "f1", "split": "train", "final_label": "好（科研）", "source_image": str(source), "corruption_type": "missing_core_entity", "target_id": "encoder", "bbox_percent": {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2}, "review_status": "approved"}
            result = apply_local_corruption(review, root / "out.png")
            self.assertEqual(result["status"], "generated")
            self.assertEqual(result["automatic_gate"]["size_preserved"], True)
            with Image.open(root / "out.png") as output_image:
                self.assertEqual(output_image.size, (100, 80))

    def test_region_candidates_use_ocr_and_limit_per_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from PIL import Image
            source = root / "source.png"
            Image.new("RGB", (100, 80), "white").save(source)
            figures = root / "figures.jsonl"
            figures.write_text(json.dumps({"figure_id": "f1", "paper_id": "p1", "figure_path": str(source), "split": "train", "final_label": "好（科研）"}, ensure_ascii=False) + "\n", encoding="utf-8")
            structured = root / "structured"
            structured.mkdir()
            (structured / "p1.json").write_text(json.dumps({"entities": [{"entity_id": "encoder"}], "relations": [], "training_flow": [], "inference_flow": [], "expected_figure_content": {"required_entities": ["encoder"]}}, ensure_ascii=False), encoding="utf-8")
            def fake_ocr(_path, _lang):
                return [{"text": "encoder", "confidence": 0.99, "quad": [[20, 20], [50, 20], [50, 30], [20, 30]]}]
            report = generate_region_candidates(figures, structured, root / "candidates.jsonl", per_type=1, ocr_function=fake_ocr)
            self.assertEqual(report["counts"]["terminology_corruption"], 1)
            self.assertEqual(report["counts"]["missing_core_entity"], 1)

    def test_d0_selection_is_deterministic_and_balanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{"arxiv_id": f"p{i:02d}", "split": "validation", "final_label": label} for i, label in enumerate(["好（科研）", "好（生动）", "坏"] * 5)]
            source = root / "papers.jsonl"
            source.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8")
            a = select_d0(source, root / "a.json", size=9)
            b = select_d0(source, root / "b.json", size=9)
            self.assertEqual(a, b)
            self.assertEqual(a["size"], 9)
            self.assertEqual(set(a["label_counts"]), {"好（科研）", "好（生动）", "坏"})


if __name__ == "__main__":
    unittest.main()
