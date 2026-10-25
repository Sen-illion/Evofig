import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rfs.judge_training.baseline import run_baseline
from rfs.judge_training.training_export import export_training_data


class JudgeTrainingToolsTests(unittest.TestCase):
    def test_baseline_resumes_and_records_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); image = root / "x.png"; image.write_bytes(b"x")
            row = {"example_id": "e1", "task_type": "scientific_quality", "paper": {"title": "t"}, "ground_truth": {}, "candidate_a": str(image), "labels": {}}
            data = root / "validation.jsonl"; data.write_text(json.dumps(row) + "\n", encoding="utf-8")
            judgement = {"preferred_candidate": "A", "quality_scores": {"scientific_fidelity": 1.0, "logical_flow": 1.0, "layout_quality": 1.0, "readability": 1.0, "domain_fit": 1.0}, "diagnosis": [], "confidence": 0.9}
            def fake(prompt, images, **kwargs): return dict(judgement)
            result = run_baseline(data, root / "out", adapter=fake)
            self.assertEqual(result["prediction_records"], 1)
            result = run_baseline(data, root / "out", adapter=fake)
            self.assertEqual(result["prediction_records"], 1)
            saved = json.loads((root / "out" / "predictions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(saved["prediction"], judgement)

    def test_export_writes_all_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); image = root / "x.png"; image.write_bytes(b"x")
            for split in ("train", "validation", "hidden_test"):
                row = {"example_id": split, "task_type": "scientific_quality", "paper": {"title": "t"}, "ground_truth": {}, "candidate_a": str(image), "labels": {"quality_scores": {"scientific_fidelity": 1.0}, "diagnosis": [], "confidence": 1.0}, "dataset_version": "d", "provenance": {"source": "test"}}
                (root / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            out = root / "training"; result = export_training_data(root, out)
            self.assertEqual(result["records"], 3)
            self.assertTrue((out / "export_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
