import json
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training import build_suite_manifest, compare_shadow_predictions, evaluate_predictions, register_checkpoint, resolve_checkpoint


class JudgeTrainingRegistryTests(unittest.TestCase):
    def test_register_requires_existing_checkpoint_and_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "judge.bin"
            checkpoint.write_bytes(b"checkpoint")
            record = register_checkpoint(checkpoint, {
                "model_id": "judge-v1-rc1",
                "dataset_version": "judge-dataset-v001",
                "roles": ["shadow"],
                "git_commit": "abcdef0",
            }, root / "registry.json")
            self.assertEqual(record["registry_status"], "registered")
            self.assertEqual(resolve_checkpoint("judge-v1-rc1", root / "registry.json")["checkpoint"], str(checkpoint.resolve()))

    def test_online_or_frozen_requires_promotion_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "judge.bin"
            checkpoint.write_bytes(b"checkpoint")
            with self.assertRaises(ValueError):
                register_checkpoint(checkpoint, {"model_id": "judge-v1", "dataset_version": "d1", "roles": ["online"], "git_commit": "abcdef0"}, root / "registry.json")

    def test_evaluation_and_shadow_comparison_use_prediction_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps({"example_id": "x", "labels": {"quality_scores": {"scientific_fidelity": 0.9}}}) + "\n", encoding="utf-8")
            trained = root / "trained.jsonl"
            trained.write_text(json.dumps({"example_id": "x", "prediction": {"quality_scores": {"scientific_fidelity": 0.8}}}) + "\n", encoding="utf-8")
            bootstrap = root / "bootstrap.jsonl"
            bootstrap.write_text(json.dumps({"example_id": "x", "prediction": {"quality_scores": {"scientific_fidelity": 0.7}}}) + "\n", encoding="utf-8")
            evaluation = evaluate_predictions(dataset, trained, suite="hidden_test")
            self.assertTrue(evaluation["ok"])
            self.assertEqual(evaluation["scientific_mae"], 0.1)
            shadow = compare_shadow_predictions(dataset, trained, bootstrap)
            self.assertTrue(shadow["ok"])
            self.assertEqual(shadow["disagreement_count"], 1)

    def test_evaluation_computes_scientific_and_pairwise_gates_and_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.jsonl"
            diagnosis = [{"type": "module_order_error", "target": "stage_2", "severity": "high", "evidence": "stage 2 precedes stage 1"}]
            grounding = {"grounded_entities": ["encoder"], "ungrounded_entities": ["invented"]}
            rows = [
                {"example_id": "bad", "task_type": "image_scientific", "labels": {"diagnosis": diagnosis, "grounding": grounding}},
                {"example_id": "pair", "task_type": "image_pairwise", "labels": {"preferred_candidate": "A"}},
            ]
            dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            manifest = root / "suite.json"
            build_suite_manifest(dataset, manifest, suite="adversarial", thresholds={
                "beautiful_but_wrong_block_rate": None, "required_entity_recall": 0.9,
                "invented_entity_recall": 0.9, "relation_direction_accuracy": None,
                "pairwise_accuracy": 0.8, "aesthetic_preference_accuracy": None,
                "repair_effectiveness": None, "repeat_score_std": None,
                "cross_domain_pairwise_accuracy": None,
            })
            predictions = root / "predictions.jsonl"
            predictions.write_text("\n".join([
                json.dumps({"example_id": "bad", "prediction": {"diagnosis": diagnosis, "grounding": grounding}}),
                json.dumps({"example_id": "pair", "prediction": {"preferred_candidate": "A"}}),
            ]) + "\n", encoding="utf-8")
            result = evaluate_predictions(dataset, predictions, suite_manifest=manifest, report_out=root / "report")
            self.assertTrue(result["gate_pass"])
            self.assertEqual(result["metrics"]["scientific_error_recall"]["value"], 1.0)
            self.assertTrue(Path(result["report_json"]).exists())
            self.assertTrue(Path(result["report_markdown"]).exists())

    def test_manifest_hash_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); dataset = root / "dataset.jsonl"; dataset.write_text(json.dumps({"example_id": "x", "labels": {}}) + "\n", encoding="utf-8")
            manifest = root / "suite.json"; manifest.write_text(json.dumps({"dataset_sha256": "wrong"}), encoding="utf-8")
            predictions = root / "predictions.jsonl"; predictions.write_text(json.dumps({"example_id": "x", "prediction": {}}) + "\n", encoding="utf-8")
            result = evaluate_predictions(dataset, predictions, suite_manifest=manifest)
            self.assertFalse(result["ok"])
            self.assertFalse(result["suite_manifest"]["dataset_integrity"])


if __name__ == "__main__":
    unittest.main()
