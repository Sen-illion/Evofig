"""Tests for the frozen human-selected framework manifest."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training.framework_manifest import build_framework_manifest


class JudgeFrameworkManifestTests(unittest.TestCase):
    def test_builds_selected_figures_with_only_final_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "papers"
            (corpus / "classified").mkdir(parents=True)
            (corpus / "papers.json").write_text(json.dumps([{"arxiv_id": "2501.00001", "title": "Paper", "abstract": "Method"}]), encoding="utf-8")
            with (corpus / "classified" / "分类清单.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["分类", "风格", "arxiv_id", "标题"])
                writer.writeheader()
                writer.writerow({"分类": "好", "风格": "科研风", "arxiv_id": "2501.00001", "标题": "Paper"})
            reviews = root / "reviews.jsonl"
            reviews.write_text(json.dumps({"arxiv_id": "2501.00001", "decision": "selected", "selected_ranks": [1, 2], "selected_candidate_source": "initial", "replacement_label": "坏"}, ensure_ascii=False) + "\n", encoding="utf-8")
            out = root / "dataset"
            manifest = build_framework_manifest(corpus, reviews, out)
            figures = [json.loads(line) for line in (out / "figures.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(manifest["paper_records"], 1)
            self.assertEqual(manifest["figure_records"], 2)
            self.assertEqual({item["final_label"] for item in figures}, {"坏"})
            self.assertTrue(all("quality_label" not in item and "replacement_label" not in item for item in figures))
            self.assertEqual(len({item["split"] for item in figures}), 1)


if __name__ == "__main__":
    unittest.main()
