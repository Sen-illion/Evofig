"""Tests for stage-1 grounded SFT, generation eval and on-policy pair collection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import build_parser
from rfs.judge_training.generate_eval import score_generated_judgement
from rfs.judge_training.grounded_judgements import ground_dpo_export, unique_ratio
from rfs.judge_training.onpolicy_pairs import collect_onpolicy_pairs, generation_is_correct
from rfs.judge_training.training import REFERENCE_CACHE, TrainingConfig, train_dpo


def _judgement(*, pick="A", kind="wrong_arrow_target", target="edge-1", evidence="motif over 6 nodes"):
    return {
        "preferred_candidate": pick,
        "quality_scores": {"scientific_fidelity": 0.0, "logical_flow": 0.1, "layout_quality": 0.9, "readability": 0.9, "domain_fit": 0.3},
        "diagnosis": [{"type": kind, "target": target, "severity": "high", "evidence": evidence, "figure": "B"}],
        "confidence": 0.9,
    }


def _record(example_id, *, paper, title, pick="A", target="edge-1"):
    chosen = _judgement(pick=pick, target=target)
    rejected = _judgement(pick="B" if pick == "A" else "A", target=target)
    prompt = json.dumps({"title": title, "task_type": "pairwise_preference"}, ensure_ascii=False)
    return {
        "contract": "rfs.judge_preference_pair",
        "example_id": example_id,
        "paper_id": paper,
        "pair_group": f"{paper}|pairwise_preference",
        "split": "train",
        "domain": "cv",
        "task_type": "pairwise_preference",
        "images": [f"images/{example_id}-a.png", f"images/{example_id}-b.png"],
        "messages": [{"role": "user", "content": [{"type": "text", "text": f"Reference contract:{prompt}"}, {"type": "image", "image": f"images/{example_id}-a.png"}]}],
        "chosen": json.dumps(chosen, ensure_ascii=False, separators=(",", ":")),
        "rejected": json.dumps(rejected, ensure_ascii=False, separators=(",", ":")),
        "preference_basis": {"known_defects": [{"diagnosis_type": "wrong_arrow_target", "target": target, "gating": True}], "human_audited": False},
    }


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


class GroundedExportTests(unittest.TestCase):
    def test_rewrites_shared_templates_into_pair_specific_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = _jsonl(root / "dpo.jsonl", [
                _record("p1", paper="arxiv-1", title="First method", target="module"),
                _record("p2", paper="arxiv-2", title="Second method", target="module"),
            ])
            self.assertLessEqual(unique_ratio([json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]), 0.5)
            report = ground_dpo_export(source, root / "grounded")
            rows = [json.loads(line) for line in (root / "grounded" / "dpo.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertTrue(report["ok"])
        self.assertGreaterEqual(report["unique_chosen_ratio"], 0.99)
        first = json.loads(rows[0]["chosen"])
        second = json.loads(rows[1]["chosen"])
        self.assertIn("module", first["diagnosis"][0]["evidence"])
        self.assertIn("First method", first["diagnosis"][0]["evidence"])
        self.assertIn("Second method", second["diagnosis"][0]["evidence"])
        self.assertNotEqual(rows[0]["chosen"], rows[1]["chosen"])
        self.assertNotIn("p1", first["diagnosis"][0]["evidence"])

    def test_the_cli_parses_ground_dpo(self):
        args = build_parser().parse_args(["judge-data", "ground-dpo", "--dataset", "dpo.jsonl", "--out", "grounded"])
        self.assertEqual(args.judge_data_action, "ground-dpo")
        self.assertEqual(args.min_unique_ratio, 0.5)


class GenerationEvalTests(unittest.TestCase):
    def test_a_correct_pick_and_defect_count_as_a_hit(self):
        record = _record("p1", paper="arxiv-1", title="First method", target="encoder")
        text = json.dumps(_judgement(pick="A", target="encoder", evidence="the arrow from encoder is reversed"), ensure_ascii=False)
        item = score_generated_judgement(text, record)
        self.assertTrue(item["submitted"])
        self.assertTrue(item["pairwise_ok"])
        self.assertTrue(item["diagnosis_hit"])
        self.assertTrue(generation_is_correct(item))

    def test_a_wrong_pick_is_not_correct(self):
        record = _record("p1", paper="arxiv-1", title="First method", target="encoder")
        text = json.dumps(_judgement(pick="B", target="encoder"), ensure_ascii=False)
        self.assertFalse(generation_is_correct(score_generated_judgement(text, record)))


class ScriptedSftBackend:
    name = "scripted-sft-test-backend"

    def __init__(self, answers):
        self.answers = [str(item) for item in answers]
        self.sft_calls = 0
        self.dpo_calls = 0

    def score(self, record, completion) -> float:
        return 1.0 if str(completion) == str(record["chosen"]) else 0.0

    def step(self, record, chosen, rejected, *, beta, reference_margin):
        self.dpo_calls += 1
        raise AssertionError("SFT must not take a DPO step")

    def sft_step(self, record, chosen):
        from rfs.judge_training.training import StepOutcome
        self.sft_calls += 1
        return StepOutcome(loss=0.1, margin=0.0)

    def generate(self, record, *, max_new_tokens=1024):
        return self.answers[0]

    def flush(self):
        return None

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)

    def load(self, directory):
        return None

    def state(self):
        return {"name": self.name, "sft_calls": self.sft_calls}


class SftLoopTests(unittest.TestCase):
    def test_sft_fits_chosen_json_and_selects_by_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = _jsonl(root / "train.jsonl", [
                _record("p1", paper="arxiv-1", title="First method", target="encoder"),
                _record("p2", paper="arxiv-2", title="Second method", target="decoder"),
            ])
            val = _jsonl(root / "val.jsonl", [_record("v1", paper="arxiv-v", title="Held out", target="pool")])
            answer = json.dumps(_judgement(pick="A", target="pool", evidence="held-out arrow"), ensure_ascii=False)
            backend = ScriptedSftBackend([answer])
            report = train_dpo(
                TrainingConfig(dataset=str(train), out_dir=str(root / "run"), backend="synthetic", objective="sft",
                               epochs=1, save_every=2, validation_dataset=str(val), generate_eval_limit=1, smoke_limit=1),
                backend=backend,
            )
            best = json.loads((root / "run" / "best_checkpoint.json").read_text(encoding="utf-8"))

        self.assertTrue(report["ok"])
        self.assertEqual(backend.sft_calls, 2)
        self.assertEqual(backend.dpo_calls, 0)
        self.assertFalse((root / "run" / REFERENCE_CACHE).exists())
        self.assertEqual(best["metrics"]["selection"], "generate")
        self.assertEqual(best["metrics"]["pairwise_accuracy"], 1.0)

    def test_the_train_command_carries_the_sft_switches(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "train", "--dataset", "d.jsonl", "--out", "run", "--objective", "sft", "--generate-eval-limit", "32"])
        self.assertEqual(args.objective, "sft")
        self.assertEqual(args.generate_eval_limit, 32)


class OnPolicyTests(unittest.TestCase):
    def test_pairs_a_correct_writing_with_a_wrong_one(self):
        record = _record("p1", paper="arxiv-1", title="First method", target="encoder")
        correct = json.dumps(_judgement(pick="A", target="encoder"), ensure_ascii=False)
        wrong = json.dumps(_judgement(pick="B", target="encoder"), ensure_ascii=False)

        class Sampler:
            def __init__(self):
                self.n = 0

            def generate(self, row, *, max_new_tokens=1024):
                self.n += 1
                return correct if self.n == 1 else wrong

        with tempfile.TemporaryDirectory() as tmp:
            report = collect_onpolicy_pairs(Sampler(), [record], Path(tmp) / "pairs", samples=2)

        self.assertTrue(report["ok"])
        self.assertEqual(report["pairs"], 1)
        self.assertEqual(report["used_gold_chosen"], 0)


if __name__ == "__main__":
    unittest.main()
