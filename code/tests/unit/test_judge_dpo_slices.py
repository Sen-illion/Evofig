"""Tests for family-balanced nested DPO slices."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import build_parser
from rfs.judge_training.dpo_slices import pair_family, scale_quotas, slice_mixed_dpo


def _row(example_id: str, *, paper: str, split: str, task: str) -> dict:
    return {
        "example_id": example_id,
        "paper_id": paper,
        "split": split,
        "task_type": task,
        "chosen": "{}",
        "rejected": "{}",
        "images": ["a.png", "b.png"],
    }


def _pool() -> list[dict]:
    rows = []
    for index in range(40):
        paper = f"rag-paper-{index}"
        rows.append(_row(f"rag-subgraph-{index:04d}-aaaa__wrong_pick", paper=paper, split="train", task="pairwise_preference"))
        rows.append(_row(f"rag-subgraph-{index:04d}-aaaa__missed_defect", paper=paper, split="train", task="pairwise_preference"))
        rows.append(_row(f"rag-subgraph-{index:04d}-bbbb__wrong_pick", paper=f"rag-val-{index}", split="validation", task="pairwise_preference"))
    for index in range(20):
        paper = f"corr-paper-{index}"
        val_paper = f"corr-val-{index}"
        for task, tag in (
            ("error_audit", "error_audit"),
            ("pairwise_preference", "pairwise"),
            ("scientific_quality", "scientific_quality"),
            ("style_preference", "style_preference"),
        ):
            rows.append(_row(f"{tag}-{paper}__t{index}", paper=paper, split="train", task=task))
            rows.append(_row(f"{tag}-{val_paper}__v{index}", paper=val_paper, split="validation", task=task))
    rows.append(_row("real-vs-generated-leak__r00", paper="real-paper", split="train", task="pairwise_preference"))
    rows.append(_row("error_audit-exam-paper__x", paper="exam-paper", split="train", task="error_audit"))
    rows.append(_row("rag-subgraph-exam-aaaa__wrong_pick", paper="exam-paper", split="train", task="pairwise_preference"))
    return rows


class PairFamilyTests(unittest.TestCase):
    def test_rag_corruption_and_real_pairs_are_named_from_the_example_id(self):
        self.assertEqual(pair_family({"example_id": "rag-subgraph-00-aa__wrong_pick"}), "rag")
        self.assertEqual(pair_family({"example_id": "error_audit-10.1__x"}), "corruption")
        self.assertEqual(pair_family({"example_id": "run-real-vs-generated-r00-c00"}), "real_vs_generated")

    def test_quotas_sum_to_the_requested_size(self):
        for size in (20, 64, 200, 1000):
            quotas = scale_quotas(size)
            self.assertEqual(sum(quotas.values()), size)
            self.assertGreater(quotas[("rag", "pairwise_preference")], 0)
            self.assertGreater(quotas[("corruption", "error_audit")], 0)


class MixedSliceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.dataset = self.root / "dpo.jsonl"
        with self.dataset.open("w", encoding="utf-8") as handle:
            for row in _pool():
                handle.write(json.dumps(row) + "\n")
        self.exam = self.root / "exam.jsonl"
        self.exam.write_text(json.dumps({"example_id": "exam-1", "paper_id": "exam-paper"}) + "\n", encoding="utf-8")

    def test_nested_slices_mix_families_and_keep_the_exam_out(self):
        report = slice_mixed_dpo(self.dataset, self.root / "slices", exam=self.exam, sizes=(8, 24, 48), val_size=12, seed=0)
        self.assertTrue(report["ok"])
        train8 = {row["example_id"] for row in _read(self.root / "slices" / "train-8" / "dpo.jsonl")}
        train24 = {row["example_id"] for row in _read(self.root / "slices" / "train-24" / "dpo.jsonl")}
        train48 = {row["example_id"] for row in _read(self.root / "slices" / "train-48" / "dpo.jsonl")}
        val = _read(self.root / "slices" / "val-12" / "dpo.jsonl")
        self.assertTrue(train8 <= train24 <= train48)
        self.assertEqual(len(train8), 8)
        self.assertEqual(len(train48), 48)
        self.assertTrue(all("exam-paper" not in row and "real-vs-generated" not in row for row in train48))
        families = {pair_family({"example_id": example_id}) for example_id in train48}
        self.assertEqual(families, {"rag", "corruption"})
        tasks = {row["task_type"] for row in _read(self.root / "slices" / "train-48" / "dpo.jsonl")}
        self.assertGreaterEqual(len(tasks), 3)
        train_papers = {row["paper_id"] for row in _read(self.root / "slices" / "train-48" / "dpo.jsonl")}
        val_papers = {row["paper_id"] for row in val}
        self.assertFalse(train_papers & val_papers)
        self.assertNotIn("exam-paper", val_papers)
        self.assertTrue(any(pair_family(row) == "rag" for row in val))
        self.assertTrue(any(pair_family(row) == "corruption" for row in val))
        self.assertEqual(report["slices"]["train-48"]["exam_paper_overlap"], 0)

    def test_the_cli_registers_slice_dpo(self):
        parser = build_parser()
        args = parser.parse_args([
            "judge-data", "slice-dpo",
            "--dataset", "dpo.jsonl",
            "--out", "slices-mixed",
            "--exam", "exam.jsonl",
            "--sizes", "20,200,1000",
            "--val-size", "64",
            "--json",
        ])
        self.assertEqual(args.judge_data_action, "slice-dpo")
        self.assertEqual(args.sizes, "20,200,1000")
        self.assertEqual(args.val_size, 64)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
