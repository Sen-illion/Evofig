"""Unit tests for the pilot training config template and the memory switch it turns on."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.cli import build_parser
from rfs.judge_training.training import (
    DEFAULT_LORA_TARGET_MODULES,
    TransformersVlmBackend,
    TrainingConfig,
    version_lock,
)


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "configs" / "judge" / "train-pilot.json"
ROUND2 = ROOT / "configs" / "judge" / "train-35b-round2.json"
ROUND2_9B = ROOT / "configs" / "judge" / "train-9b-round2.json"
STAGE1_SFT = ROOT / "configs" / "judge" / "train-9b-stage1-sft.json"


class PilotTemplateTests(unittest.TestCase):
    def test_the_template_parses_and_names_the_fields_a_real_run_needs(self):
        config = TrainingConfig.from_file(TEMPLATE, dataset="pilot.jsonl", out_dir="run-1")
        self.assertEqual(config.backend, "transformers")
        self.assertEqual(config.base_model, "<DATA_ROOT>/models/Qwen3.5-35B-A3B")
        self.assertTrue(config.load_in_4bit)
        self.assertTrue(config.gradient_checkpointing)
        self.assertEqual((config.max_steps, config.save_every), (8, 4))
        self.assertEqual(config.dataset, "pilot.jsonl")
        self.assertEqual(config.out_dir, "run-1")

    def test_every_key_in_the_template_is_a_field_the_dataclass_knows(self):
        payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        known = set(TrainingConfig.__dataclass_fields__)
        self.assertEqual(sorted(set(payload) - known), [])
        for absent in ("dataset", "out_dir"):
            self.assertNotIn(absent, payload)

    def test_the_fingerprint_is_stable_and_ignores_the_paths(self):
        first = TrainingConfig.from_file(TEMPLATE, dataset="a.jsonl", out_dir="run-1")
        second = TrainingConfig.from_file(TEMPLATE, dataset="b.jsonl", out_dir="run-2")
        self.assertEqual(first.fingerprint(), first.fingerprint())
        self.assertEqual(first.fingerprint(), second.fingerprint())


class Round2TemplateTests(unittest.TestCase):
    def test_the_round2_template_is_one_epoch_with_smoke_and_collapse_stops(self):
        config = TrainingConfig.from_file(ROUND2, dataset="dpo.jsonl", out_dir="run-2")
        self.assertEqual(config.epochs, 1)
        self.assertEqual(config.max_steps, 0)
        self.assertEqual(config.save_every, 250)
        self.assertEqual(config.smoke_limit, 4)
        self.assertEqual(config.smoke_max_new_tokens, 1024)
        self.assertEqual(config.pairwise_patience, 2)
        self.assertEqual(config.collapse_margin, 8.0)
        self.assertEqual(config.early_stop_margin, 0)
        self.assertEqual(config.learning_rate, 2e-05)
        self.assertEqual(config.gradient_accumulation, 16)
        self.assertEqual(config.image_max_pixels, 250000)
        self.assertTrue(config.shard_across_gpus)

    def test_every_key_in_the_round2_template_is_a_field_the_dataclass_knows(self):
        payload = json.loads(ROUND2.read_text(encoding="utf-8"))
        known = set(TrainingConfig.__dataclass_fields__)
        self.assertEqual(sorted(set(payload) - known), [])
        for absent in ("dataset", "out_dir", "validation_dataset"):
            self.assertNotIn(absent, payload)


class Round2NineBTemplateTests(unittest.TestCase):
    def test_the_9b_template_is_single_gpu_one_epoch_with_tighter_updates(self):
        config = TrainingConfig.from_file(ROUND2_9B, dataset="dpo.jsonl", out_dir="run-9b")
        self.assertEqual(config.base_model, "<DATA_ROOT>/models/Qwen3.5-9B")
        self.assertEqual(config.epochs, 1)
        self.assertEqual(config.max_steps, 0)
        self.assertEqual(config.save_every, 100)
        self.assertEqual(config.eval_limit, 32)
        self.assertEqual(config.smoke_limit, 4)
        self.assertEqual(config.smoke_max_new_tokens, 1024)
        self.assertEqual(config.smoke_pass_rate, 0.5)
        self.assertEqual(config.pairwise_patience, 2)
        self.assertEqual(config.collapse_margin, 8.0)
        self.assertEqual(config.early_stop_margin, 0)
        self.assertEqual(config.learning_rate, 1e-05)
        self.assertEqual(config.gradient_accumulation, 8)
        self.assertEqual(config.image_max_pixels, 250000)
        self.assertFalse(config.shard_across_gpus)

    def test_every_key_in_the_9b_template_is_a_field_the_dataclass_knows(self):
        payload = json.loads(ROUND2_9B.read_text(encoding="utf-8"))
        known = set(TrainingConfig.__dataclass_fields__)
        self.assertEqual(sorted(set(payload) - known), [])
        for absent in ("dataset", "out_dir", "validation_dataset"):
            self.assertNotIn(absent, payload)

    def test_the_9b_template_does_not_reuse_the_35b_round2_fingerprint(self):
        nine = TrainingConfig.from_file(ROUND2_9B, dataset="dpo.jsonl", out_dir="run-9b")
        thirty_five = TrainingConfig.from_file(ROUND2, dataset="dpo.jsonl", out_dir="run-2")
        self.assertNotEqual(nine.fingerprint(), thirty_five.fingerprint())

    def test_the_stage1_sft_template_fits_chosen_json_and_selects_by_generation(self):
        config = TrainingConfig.from_file(STAGE1_SFT, dataset="dpo.jsonl", out_dir="run-sft")
        self.assertEqual(config.objective, "sft")
        self.assertEqual(config.generate_eval_limit, 32)
        self.assertEqual(config.collapse_margin, 0)
        payload = json.loads(STAGE1_SFT.read_text(encoding="utf-8"))
        self.assertEqual(sorted(set(payload) - set(TrainingConfig.__dataclass_fields__)), [])
        self.assertNotEqual(
            config.fingerprint(),
            TrainingConfig.from_file(ROUND2_9B, dataset="dpo.jsonl", out_dir="run-9b").fingerprint(),
        )
    def test_the_template_names_the_projections_peft_must_attach_to(self):
        payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        self.assertEqual(tuple(payload["lora_target_modules"]), DEFAULT_LORA_TARGET_MODULES)
        config = TrainingConfig.from_file(TEMPLATE, dataset="d", out_dir="o")
        self.assertEqual(tuple(config.lora_target_modules), DEFAULT_LORA_TARGET_MODULES)

    def test_the_backend_hands_the_projections_to_peft_instead_of_letting_it_infer(self):
        backend = TransformersVlmBackend.__new__(TransformersVlmBackend)
        backend.config = TrainingConfig(dataset="d", out_dir="o")
        kwargs = backend._lora_kwargs()
        self.assertEqual(kwargs["target_modules"], list(DEFAULT_LORA_TARGET_MODULES))
        self.assertEqual(kwargs["task_type"], "CAUSAL_LM")
        self.assertEqual(kwargs["r"], 16)

    def test_an_empty_list_leaves_the_choice_to_peft(self):
        backend = TransformersVlmBackend.__new__(TransformersVlmBackend)
        backend.config = TrainingConfig(dataset="d", out_dir="o", lora_target_modules=())
        self.assertNotIn("target_modules", backend._lora_kwargs())

    def test_the_version_lock_records_which_projections_were_trained(self):
        backend = TransformersVlmBackend.__new__(TransformersVlmBackend)
        backend.config = TrainingConfig(dataset="d", out_dir="o")
        lock = version_lock(backend.config, backend)
        self.assertEqual(lock["lora"]["target_modules"], list(DEFAULT_LORA_TARGET_MODULES))


class MemorySwitchTests(unittest.TestCase):
    def test_the_switch_is_off_unless_a_config_or_a_flag_turns_it_on(self):
        self.assertFalse(TrainingConfig(dataset="d", out_dir="o").gradient_checkpointing)

    def test_turning_it_on_changes_the_recorded_config_and_its_fingerprint(self):
        off = TrainingConfig(dataset="d", out_dir="o")
        on = TrainingConfig(dataset="d", out_dir="o", gradient_checkpointing=True)
        self.assertTrue(on.gradient_checkpointing)
        self.assertTrue(on.to_dict()["gradient_checkpointing"])
        self.assertNotEqual(off.fingerprint(), on.fingerprint())

    def test_an_unknown_key_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"backend": "synthetic", "gradient_checkpointingg": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                TrainingConfig.from_file(path, dataset="d", out_dir="o")

    def test_the_train_command_carries_the_switch(self):
        parser = build_parser()
        flag = parser.parse_args(["judge", "train", "--dataset", "d.jsonl", "--out", "run", "--gradient-checkpointing"])
        self.assertTrue(flag.gradient_checkpointing)
        plain = parser.parse_args(["judge", "train", "--dataset", "d.jsonl", "--out", "run"])
        self.assertIsNone(plain.gradient_checkpointing)

    def test_the_train_command_carries_the_held_out_selection_switches(self):
        parser = build_parser()
        args = parser.parse_args(["judge", "train", "--dataset", "d.jsonl", "--out", "run",
                                  "--validation-dataset", "v.jsonl", "--eval-limit", "60", "--early-stop-margin", "20"])
        self.assertEqual(args.validation_dataset, "v.jsonl")
        self.assertEqual(args.eval_limit, 60)
        self.assertEqual(args.early_stop_margin, 20.0)
        config = TrainingConfig(dataset="d.jsonl", out_dir="run", validation_dataset=args.validation_dataset,
                                eval_limit=args.eval_limit, early_stop_margin=args.early_stop_margin)
        self.assertEqual(config.to_dict()["validation_dataset"], "v.jsonl")
        plain = parser.parse_args(["judge", "train", "--dataset", "d.jsonl", "--out", "run"])
        self.assertIsNone(plain.validation_dataset)
        self.assertIsNone(plain.eval_limit)
        self.assertIsNone(plain.early_stop_margin)


if __name__ == "__main__":
    unittest.main()
