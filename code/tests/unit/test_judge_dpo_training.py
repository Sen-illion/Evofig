"""Tests for the DPO training loop, its reference cache and its resume path."""

import json
import math
import types
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training.generation_smoke import (
    GENERATION_SMOKE_FILE,
    has_obvious_repetition,
    run_generation_smoke,
    score_generation,
)
from rfs.judge_training.training import (
    BEST_CHECKPOINT_FILE,
    COMPLETION_LOGPROB_REDUCTION,
    DATA_REPORT_FILE,
    LOCK_FILE,
    METRICS_FILE,
    REPORT_FILE,
    STATE_FILE,
    VALIDATION_METRICS_FILE,
    SyntheticPolicyBackend,
    StepOutcome,
    TransformersVlmBackend,
    TrainingConfig,
    TrainingUnavailable,
    apply_vision_budget,
    build_backend,
    completion_logprob,
    dpo_gradient,
    dpo_loss,
    evaluate_preference_pairs,
    load_training_records,
    parameters_by_device,
    preference_logprob_term,
    reference_key,
    reference_scoring,
    resolve_vision_auto_class,
    train_dpo,
)


def _record(example_id, *, chosen, rejected, domain="cv", defect="wrong_arrow_target"):
    return {
        "contract": "rfs.judge_preference_pair",
        "example_id": example_id,
        "pair_group": f"arxiv-2401.00001|pairwise_preference",
        "split": "train",
        "domain": domain,
        "task_type": "pairwise_preference",
        "images": [f"images/{example_id}-a.png", f"images/{example_id}-b.png"],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "judge the pair"}, {"type": "image", "image": f"images/{example_id}-a.png"}, {"type": "image", "image": f"images/{example_id}-b.png"}]}],
        "chosen": chosen,
        "rejected": rejected,
        "preference_basis": {"known_defects": [{"diagnosis_type": defect, "gating": defect in {"module_order_error", "wrong_arrow_target", "overlap"}}], "human_audited": False},
    }


def _dataset(root, count=3):
    rows = [
        _record(f"p{index}", chosen=json.dumps({"preferred_candidate": "A", "note": f"pair {index}"}), rejected=json.dumps({"preferred_candidate": "B", "note": f"pair {index}"}), domain=("cv" if index % 2 else "biology"), defect=("overlap" if index == 2 else "wrong_arrow_target"))
        for index in range(count)
    ]
    path = root / "dpo.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")
    return path


def _config(dataset, out_dir, **overrides):
    values = {"dataset": str(dataset), "out_dir": str(out_dir), "backend": "synthetic", "learning_rate": 0.4, "epochs": 2, "save_every": 2, "seed": 7}
    values.update(overrides)
    return TrainingConfig(**values)


class _ControlledMarginBackend:
    """A policy whose held-out margin follows a script, so the stop point is exact.

    ``schedule[n]`` is the margin the held-out pass sees once ``n`` steps have run; the chosen
    completion scores it and the rejected one scores zero, so every pair reports the same margin.
    """

    name = "controlled-margin-test-backend"

    def __init__(self, schedule):
        self.schedule = [float(value) for value in schedule]
        self.steps = 0

    def margin(self) -> float:
        return self.schedule[min(self.steps, len(self.schedule) - 1)]

    def score(self, record, completion) -> float:
        return self.margin() if str(completion) == str(record["chosen"]) else 0.0

    def step(self, record, chosen, rejected, *, beta, reference_margin):
        self.steps += 1
        margin = self.margin()
        return StepOutcome(loss=dpo_loss(margin, beta), margin=margin)

    def flush(self) -> None:
        return None

    def save(self, directory) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)

    def load(self, directory) -> None:
        return None

    def state(self):
        return {"name": self.name, "steps": self.steps}


class _ScriptedGenerateBackend(_ControlledMarginBackend):
    """A margin script that also writes scripted completions, so smoke can fail on cue."""

    name = "scripted-generate-test-backend"

    def __init__(self, schedule, answers):
        super().__init__(schedule)
        self.answers = [str(item) for item in answers]

    def generate(self, record, *, max_new_tokens=1024):
        if not self.answers:
            return ""
        index = 0 if self.steps < 4 else min(len(self.answers) - 1, 1)
        return self.answers[index]


class ObjectiveTests(unittest.TestCase):
    def test_loss_and_gradient_match_the_closed_form(self):
        self.assertAlmostEqual(dpo_loss(0.0, 0.1), math.log(2.0), places=9)
        self.assertAlmostEqual(dpo_gradient(0.0, 0.1), 0.05, places=9)
        self.assertLess(dpo_loss(50.0, 0.1), 0.01)
        self.assertGreater(dpo_loss(-50.0, 0.1), 4.9)
        self.assertLess(dpo_gradient(50.0, 0.1), dpo_gradient(-50.0, 0.1))

    def test_the_log_ratio_is_measured_against_the_reference(self):
        self.assertAlmostEqual(preference_logprob_term(-2.0, -3.0, 0.0), 1.0, places=9)
        self.assertAlmostEqual(preference_logprob_term(-2.0, -3.0, 2.0), -1.0, places=9)
        self.assertLess(dpo_loss(preference_logprob_term(-1.0, -3.0, 0.0), 0.1), math.log(2.0))
        self.assertGreater(dpo_loss(preference_logprob_term(-4.0, -3.0, 0.0), 0.1), math.log(2.0))


class ConfigTests(unittest.TestCase):
    def test_round_trips_a_config_file_and_rejects_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "config.json"
            path.write_text(json.dumps({"dataset": "d.jsonl", "out_dir": "run", "epochs": 3, "beta": 0.2}), encoding="utf-8")
            config = TrainingConfig.from_file(path)
            self.assertEqual((config.epochs, config.beta), (3, 0.2))
            overridden = TrainingConfig.from_file(path, epochs=5, out_dir="other")
            self.assertEqual((overridden.epochs, overridden.out_dir), (5, "other"))
            (root / "bad.json").write_text(json.dumps({"dataset": "d", "out_dir": "o", "warp": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                TrainingConfig.from_file(root / "bad.json")

    def test_the_fingerprint_ignores_the_paths(self):
        first = TrainingConfig(dataset="a/dpo.jsonl", out_dir="a/run", backend="synthetic")
        second = TrainingConfig(dataset="b/dpo.jsonl", out_dir="b/run", backend="synthetic")
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_the_vision_budget_is_configurable_and_changes_the_fingerprint(self):
        default = TrainingConfig(dataset="d", out_dir="o", backend="synthetic")
        self.assertEqual(default.image_max_pixels, 0)
        self.assertFalse(default.shard_across_gpus)
        capped = TrainingConfig.from_mapping({"dataset": "d", "out_dir": "o", "backend": "synthetic", "image_max_pixels": 1000000})
        self.assertEqual(capped.image_max_pixels, 1000000)
        self.assertNotEqual(capped.fingerprint(), default.fingerprint())
        sharded = TrainingConfig.from_mapping({"dataset": "d", "out_dir": "o", "backend": "synthetic", "shard_across_gpus": True})
        self.assertTrue(sharded.shard_across_gpus)
        self.assertNotEqual(sharded.fingerprint(), default.fingerprint())
        with self.assertRaises(ValueError):
            TrainingConfig(dataset="d", out_dir="o", image_max_pixels=-1)


class ParameterGroupingTests(unittest.TestCase):
    """A sharded model clips gradients one device at a time."""

    def _parameter(self, device, *, requires_grad=True, grad=True):
        return types.SimpleNamespace(device=device, requires_grad=requires_grad, grad=("g" if grad else None))

    def test_only_trainable_parameters_with_a_gradient_are_grouped(self):
        first = self._parameter("cuda:0")
        second = self._parameter("cuda:0")
        third = self._parameter("cuda:1")
        frozen = self._parameter("cuda:1", requires_grad=False)
        empty = self._parameter("cuda:0", grad=False)
        groups = parameters_by_device([first, second, third, frozen, empty])
        self.assertEqual({device: len(items) for device, items in groups.items()}, {"cuda:0": 2, "cuda:1": 1})
        self.assertIn(first, groups["cuda:0"])
        self.assertIn(third, groups["cuda:1"])


class VisionBudgetTests(unittest.TestCase):
    def _processor(self, **size):
        return types.SimpleNamespace(image_processor=types.SimpleNamespace(size=size))

    def test_the_cap_only_lowers_the_longest_edge(self):
        processor = self._processor(longest_edge=16777216, shortest_edge=65536)
        applied = apply_vision_budget(processor, 1000000)
        self.assertEqual(applied, {"longest_edge": 1000000, "shortest_edge": 65536})
        self.assertEqual(processor.image_processor.size["longest_edge"], 1000000)
        self.assertEqual(processor.image_processor.size["shortest_edge"], 65536)

    def test_zero_leaves_the_budget_untouched(self):
        processor = self._processor(longest_edge=16777216, shortest_edge=65536)
        self.assertEqual(apply_vision_budget(processor, 0), {})
        self.assertEqual(processor.image_processor.size["longest_edge"], 16777216)

    def test_a_max_pixels_style_processor_is_supported(self):
        image_processor = types.SimpleNamespace(max_pixels=16777216)
        applied = apply_vision_budget(types.SimpleNamespace(image_processor=image_processor), 1000000)
        self.assertEqual(applied["longest_edge"], 1000000)
        self.assertEqual(image_processor.max_pixels, 1000000)

    def test_a_size_dict_style_processor_is_supported(self):
        # transformers loads the config's mapping back as an attribute object, not as a dict.
        size = types.SimpleNamespace(longest_edge=16777216, shortest_edge=65536)
        applied = apply_vision_budget(types.SimpleNamespace(image_processor=types.SimpleNamespace(size=size)), 1000000)
        self.assertEqual(applied, {"longest_edge": 1000000, "shortest_edge": 65536})
        self.assertEqual(size.longest_edge, 1000000)

    def test_a_budget_that_cannot_be_changed_fails_loudly(self):
        class Locked:
            longest_edge = 16777216
            shortest_edge = 65536

            def __setattr__(self, name, value):
                raise AttributeError("read only")

        processor = types.SimpleNamespace(image_processor=types.SimpleNamespace(size=Locked()))
        with self.assertRaises(TrainingUnavailable):
            apply_vision_budget(processor, 1000000)

    def test_a_cap_below_the_floor_or_without_a_hook_fails_loudly(self):
        with self.assertRaises(TrainingUnavailable):
            apply_vision_budget(self._processor(longest_edge=16777216, shortest_edge=2000000), 1000000)
        with self.assertRaises(TrainingUnavailable):
            apply_vision_budget(types.SimpleNamespace(), 1000000)


class DatasetTests(unittest.TestCase):
    def test_records_must_carry_the_pair_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = root / "dpo.jsonl"
            broken.write_text(json.dumps({"example_id": "p1", "messages": [{}], "chosen": "{}"}, ensure_ascii=False) + chr(10), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_training_records(broken)
            empty = root / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_training_records(empty)

    def test_unknown_backends_are_refused(self):
        with self.assertRaises(TrainingUnavailable):
            build_backend(TrainingConfig(dataset="d", out_dir="o", backend="chat"))

    def test_the_transformers_backend_fails_loudly_without_the_stack(self):
        config = TrainingConfig(dataset="d", out_dir="o", backend="transformers", base_model="Qwen/Qwen2.5-VL-3B-Instruct")
        with self.assertRaises(TrainingUnavailable):
            build_backend(config)


class TrainingLoopTests(unittest.TestCase):
    def test_the_synthetic_backend_learns_the_certified_judgement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root)
            report = train_dpo(_config(dataset, root / "run"))
            metrics = [json.loads(line) for line in (root / "run" / METRICS_FILE).read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertTrue(report["ok"])
        self.assertEqual(report["metrics"]["steps"], 6)
        self.assertEqual(metrics[0]["margin"], 0.0)
        self.assertFalse(metrics[0]["preferred"])
        self.assertEqual(report["metrics"]["pairwise_accuracy"], round(sum(1 for item in metrics if item["preferred"]) / len(metrics), 6))
        self.assertEqual(report["metrics"]["pairwise_accuracy"], 0.833333)
        self.assertEqual(report["metrics"]["train_memorization"], 0.833333)
        self.assertTrue(metrics[-1]["preferred"])
        self.assertGreater(metrics[-1]["margin"], metrics[0]["margin"])
        self.assertLess(metrics[-1]["loss"], metrics[0]["loss"])
        self.assertEqual(report["metrics"]["per_defect_accuracy"]["overlap"], 1.0)
        self.assertEqual(sorted(report["metrics"]["per_domain_accuracy"]), ["biology", "cv"])
        self.assertEqual(report["metrics"]["gated_recall"]["overlap"], 1.0)
        self.assertIsNone(report["metrics"]["gated_recall"]["module_order_error"])

    def test_training_writes_checkpoints_metrics_data_report_and_the_version_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root)
            report = train_dpo(_config(dataset, root / "run"))
            run = root / "run"
            lock = json.loads((run / LOCK_FILE).read_text(encoding="utf-8"))
            data_report = json.loads((run / DATA_REPORT_FILE).read_text(encoding="utf-8"))
            written = json.loads((run / REPORT_FILE).read_text(encoding="utf-8"))
            checkpoints = sorted((run / "checkpoints").iterdir())
            state = json.loads((checkpoints[-1] / STATE_FILE).read_text(encoding="utf-8"))
            cache_lines = (checkpoints[-1] / "reference_logprobs.jsonl").read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual([item.name for item in checkpoints], ["step-000002", "step-000004", "step-000006"])
        self.assertEqual(state["step"], 6)
        self.assertEqual(len(cache_lines), 3)
        self.assertEqual(lock["backend"], "synthetic-hashed-preference-v1")
        self.assertEqual(lock["judge_contract"], "judge-judgement-v1")
        self.assertEqual(lock["pair_schema_version"], "judge-preference-pair-v1")
        self.assertEqual(lock["logprob_reduction"], COMPLETION_LOGPROB_REDUCTION)
        self.assertEqual(lock["image_max_pixels"], 0)
        self.assertEqual(lock["vision_budget"], {})
        self.assertEqual(lock["device_map"], "single-device")
        self.assertEqual(len(lock["dataset_sha256"]), 64)
        self.assertEqual(data_report["json_valid_rate"], 1.0)
        self.assertEqual(data_report["templated_ratio"], 0.0)
        self.assertEqual(report["checkpoints"], written["checkpoints"])
        self.assertEqual(report["reference_cache"]["computed"], 3)

    def test_the_loop_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root)
            train_dpo(_config(dataset, root / "one"))
            train_dpo(_config(dataset, root / "two"))
            first = (root / "one" / METRICS_FILE).read_text(encoding="utf-8")
            second = (root / "two" / METRICS_FILE).read_text(encoding="utf-8")

        self.assertEqual(first, second)

    def test_resume_continues_from_a_checkpoint_and_reuses_the_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root)
            train_dpo(_config(dataset, root / "run", epochs=2, max_steps=6))
            resumed = train_dpo(_config(dataset, root / "resumed", epochs=3, max_steps=9, resume_from=str(root / "run" / "checkpoints" / "step-000004")))
            metrics = [json.loads(line) for line in (root / "resumed" / METRICS_FILE).read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(resumed["started_at_step"], 4)
        self.assertEqual([item["step"] for item in metrics], [5, 6, 7, 8, 9])
        self.assertEqual(resumed["reference_cache"]["reused"], 3)
        self.assertEqual(resumed["reference_cache"]["computed"], 0)
        self.assertEqual(resumed["metrics"]["steps"], 5)

    def test_init_from_loads_weights_restarts_the_step_and_recomputes_the_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root)
            train_dpo(_config(dataset, root / "source", epochs=2, max_steps=4, save_every=4))
            checkpoint = root / "source" / "checkpoints" / "step-000004"
            started = train_dpo(_config(dataset, root / "fresh", epochs=1, max_steps=2, save_every=2, init_from=str(checkpoint)))
            cold = train_dpo(_config(dataset, root / "cold", epochs=1, max_steps=2, save_every=2))
            fresh_metrics = (root / "fresh" / METRICS_FILE).read_text(encoding="utf-8")
            with self.assertRaises(ValueError):
                _config(dataset, root / "both", resume_from=str(checkpoint), init_from=str(checkpoint))

        self.assertEqual(started["started_at_step"], 0)
        self.assertEqual(started["initialized_from"], str(checkpoint))
        self.assertEqual(started["resumed_from"], "")
        self.assertEqual(started["reference_cache"]["computed"], 3)
        self.assertEqual(started["reference_cache"]["reused"], 0)
        self.assertEqual([json.loads(line)["step"] for line in fresh_metrics.splitlines() if line.strip()], [1, 2])
        self.assertEqual(started["backend"]["steps"], 6)
        self.assertGreater(started["backend"]["weights_l1"], cold["backend"]["weights_l1"])

    def test_a_resume_without_a_usable_reference_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root)
            train_dpo(_config(dataset, root / "run", epochs=1, max_steps=3))
            other = _dataset(root, count=4)
            with self.assertRaises(ValueError):
                train_dpo(_config(other, root / "resumed", epochs=1, max_steps=6, resume_from=str(root / "run" / "checkpoints" / "step-000002")))

    def test_the_reference_key_moves_with_the_vision_budget(self):
        records = [_record("p0", chosen="a", rejected="b")]
        default = _config("d", "o")
        capped = _config("d", "o", image_max_pixels=250000)

        self.assertNotEqual(reference_key(default, records), reference_key(capped, records))
        self.assertEqual(reference_key(default, records), reference_key(_config("d", "o"), records))

    def test_the_reference_key_names_the_sequence_logprob_sum(self):
        scoring = reference_scoring(_config("d", "o"))
        self.assertEqual(scoring["logprob_reduction"], "sum_completion_tokens")
        self.assertIn("logprob_reduction", scoring)

    def test_the_synthetic_backend_state_is_saved_in_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root)
            report = train_dpo(_config(dataset, root / "run", max_steps=2, save_every=2))
            backend = SyntheticPolicyBackend(learning_rate=0.4)
            backend.load(root / "run" / "checkpoints" / "step-000002")

        self.assertEqual(backend.steps, 2)
        self.assertGreater(sum(abs(item) for item in backend.weights), 0.0)
        self.assertEqual(report["backend"]["steps"], 2)


def _logsigmoid(value):
    """A pure-python stand-in for torch.logsigmoid, so the objective is testable without torch."""

    return -math.log1p(math.exp(-value)) if value >= 0 else value - math.log1p(math.exp(value))


class _StubLossModel:
    """A model whose per-completion losses are fixed numbers, so the objective can be read off."""

    def __init__(self, losses):
        self.losses = dict(losses)

    def __call__(self, token, **kwargs):
        return types.SimpleNamespace(loss=self.losses[token])


def _completion_labels(valid_tokens, *, prompt_tokens=3):
    """A labels row whose unmasked count is ``valid_tokens``."""

    return [-100] * int(prompt_tokens) + list(range(int(valid_tokens)))


class PreferenceObjectiveTests(unittest.TestCase):
    """The real backend optimizes log probabilities against the reference, not raw model loss."""

    def _backend(self, losses, counts=None):
        backend = TransformersVlmBackend.__new__(TransformersVlmBackend)
        backend.torch = types.SimpleNamespace(nn=types.SimpleNamespace(functional=types.SimpleNamespace(logsigmoid=_logsigmoid)))
        backend.model = _StubLossModel(losses)
        if counts is None:
            backend._batch = lambda record, completion: {"token": completion}
        else:
            backend._batch = lambda record, completion: {"token": completion, "labels": _completion_labels(counts[completion])}
        return backend

    def test_the_loss_matches_the_reference_relative_closed_form(self):
        backend = self._backend({"chosen": 2.0, "rejected": 3.0})
        for reference_margin, beta in ((0.0, 0.1), (2.0, 0.5)):
            expected_margin = preference_logprob_term(-2.0, -3.0, reference_margin)
            expected = dpo_loss(expected_margin, beta)
            loss, margin = backend._loss({}, "chosen", "rejected", beta=beta, reference_margin=reference_margin)
            self.assertAlmostEqual(loss, expected, places=9)
            self.assertAlmostEqual(margin, expected_margin, places=9)

    def test_favouring_the_rejected_judgement_is_penalised(self):
        backend = self._backend({"chosen": 3.0, "rejected": 2.0})
        loss, margin = backend._loss({}, "chosen", "rejected", beta=0.1, reference_margin=0.0)
        self.assertGreater(loss, math.log(2.0))
        self.assertLess(margin, 0.0)

    def test_equal_length_completions_keep_the_closed_form_after_summing(self):
        backend = self._backend({"chosen": 2.0, "rejected": 3.0}, counts={"chosen": 8, "rejected": 8})
        expected_margin = preference_logprob_term(-16.0, -24.0, 0.0)
        expected = dpo_loss(expected_margin, 0.1)
        loss, margin = backend._loss({}, "chosen", "rejected", beta=0.1, reference_margin=0.0)
        self.assertAlmostEqual(loss, expected, places=9)
        self.assertAlmostEqual(margin, expected_margin, places=9)

    def test_a_short_rejected_with_the_same_mean_nll_is_not_tied_with_the_long_chosen(self):
        backend = self._backend({"chosen": 0.5, "rejected": 0.5}, counts={"chosen": 20, "rejected": 4})
        _, margin = backend._loss({}, "chosen", "rejected", beta=0.1, reference_margin=0.0)
        self.assertAlmostEqual(margin, -8.0, places=9)
        self.assertNotEqual(margin, 0.0)


class CompletionLogprobTests(unittest.TestCase):
    """Sequence logprob is the mean NLL times the number of unmasked tokens."""

    def test_a_batch_without_labels_keeps_the_negated_mean(self):
        self.assertEqual(completion_logprob(types.SimpleNamespace(loss=2.0), {"token": "chosen"}), -2.0)

    def test_prompt_tokens_are_not_counted(self):
        labels = _completion_labels(5, prompt_tokens=7)
        self.assertEqual(completion_logprob(types.SimpleNamespace(loss=0.4), {"labels": labels}), -2.0)

    def test_the_same_mean_nll_does_not_give_a_short_json_the_same_score_as_a_long_audit(self):
        short = completion_logprob(types.SimpleNamespace(loss=0.25), {"labels": _completion_labels(4)})
        long = completion_logprob(types.SimpleNamespace(loss=0.25), {"labels": _completion_labels(40)})
        self.assertAlmostEqual(short, -1.0, places=9)
        self.assertAlmostEqual(long, -10.0, places=9)
        self.assertNotEqual(short, long)


class VisionAutoClassTests(unittest.TestCase):
    """The trainer accepts the class name each transformers side exposes."""

    def test_the_renamed_and_the_legacy_auto_class_are_both_accepted(self):
        class Marker:
            pass

        renamed = types.SimpleNamespace(AutoModelForImageTextToText=Marker)
        legacy = types.SimpleNamespace(AutoModelForVision2Seq=Marker)
        self.assertIs(resolve_vision_auto_class(renamed), Marker)
        self.assertIs(resolve_vision_auto_class(legacy), Marker)

    def test_the_renamed_class_wins_when_a_stack_exposes_both(self):
        renamed, legacy = object(), object()
        both = types.SimpleNamespace(AutoModelForImageTextToText=renamed, AutoModelForVision2Seq=legacy)
        self.assertIs(resolve_vision_auto_class(both), renamed)

    def test_a_stack_with_neither_name_is_refused(self):
        with self.assertRaises(TrainingUnavailable):
            resolve_vision_auto_class(types.SimpleNamespace())


class _StubModel:
    """A model with the adapter parameters a resume is expected to fill."""

    def __init__(self, count=2):
        self._names = [f"base_model.model.model.layers.0.q_proj.lora_{'AB'[i % 2]}.weight" for i in range(count)]

    def named_parameters(self):
        return [(name, None) for name in self._names]


class _StubPeft:
    """A peft stand-in that records how an adapter was applied."""

    def __init__(self, result=None, keys=None):
        self.result = result if result is not None else types.SimpleNamespace(unexpected_keys=[], missing_keys=["base_model.model.lm_head.weight"])
        self.keys = list(keys) if keys is not None else ["lora_A", "lora_B"]
        self.applied = []
        self.wrapped = 0

    def load_peft_weights(self, path):
        self.loaded_from = path
        return {key: "w" for key in self.keys}

    def set_peft_model_state_dict(self, model, weights):
        self.applied.append((model, weights))
        return self.result

    def PeftModel(self):  # pragma: no cover - the nesting path must stay unused
        self.wrapped += 1
        raise AssertionError("resume must not wrap an existing PeftModel again")


class ResumeAdapterTests(unittest.TestCase):
    """A resume that cannot place the adapter must fail instead of training from scratch."""

    def _backend(self, peft, adapter_dir):
        backend = TransformersVlmBackend.__new__(TransformersVlmBackend)
        backend.peft = peft
        backend.model = _StubModel()
        (adapter_dir / "adapter").mkdir(parents=True, exist_ok=True)
        return backend

    def test_the_saved_adapter_is_applied_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            peft = _StubPeft()
            backend = self._backend(peft, Path(tmp))
            backend.load(tmp)
        self.assertEqual(len(peft.applied), 1)
        self.assertEqual(peft.applied[0][0], backend.model)
        self.assertEqual(peft.wrapped, 0)

    def test_frozen_base_keys_in_missing_keys_are_not_a_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            peft = _StubPeft(result=types.SimpleNamespace(unexpected_keys=[], missing_keys=["base_model.model.lm_head.weight"] * 714))
            backend = self._backend(peft, Path(tmp))
            backend.load(tmp)
        self.assertEqual(len(peft.applied), 1)

    def test_adapter_keys_that_do_not_land_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            peft = _StubPeft(result=types.SimpleNamespace(unexpected_keys=["base_model.model.base_model.model.model.layers.0.q_proj.lora_A.weight"], missing_keys=[]))
            backend = self._backend(peft, Path(tmp))
            with self.assertRaises(ValueError) as caught:
                backend.load(tmp)
        self.assertIn("silently train from scratch", str(caught.exception))

    def test_a_partial_adapter_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            peft = _StubPeft(keys=["lora_A"])
            backend = self._backend(peft, Path(tmp))
            with self.assertRaises(ValueError) as caught:
                backend.load(tmp)
        self.assertIn("loaded=1", str(caught.exception))

    def test_a_missing_adapter_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = self._backend(_StubPeft(), Path(tmp))
            with self.assertRaises(ValueError):
                backend.load(str(Path(tmp) / "absent"))


class HeldOutSelectionTests(unittest.TestCase):
    def _held_out(self, root, count=2):
        path = root / "validation.jsonl"
        rows = [_record(f"v{index}", chosen=json.dumps({"preferred_candidate": "A"}), rejected=json.dumps({"preferred_candidate": "B"})) for index in range(count)]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows), encoding="utf-8")
        return path

    def test_the_held_out_pass_reports_accuracy_and_the_margin_distribution(self):
        records = [_record(f"v{index}", chosen=json.dumps({"preferred_candidate": "A"}), rejected=json.dumps({"preferred_candidate": "B"})) for index in range(3)]
        policy = _ControlledMarginBackend([2.0])
        scored = evaluate_preference_pairs(policy, records, limit=2)
        self.assertEqual(scored["pairs"], 2)
        self.assertEqual(scored["pairwise_accuracy"], 1.0)
        self.assertEqual(scored["median_margin"], 2.0)
        self.assertEqual(scored["min_margin"], 2.0)
        self.assertEqual(scored["max_margin"], 2.0)
        self.assertEqual(evaluate_preference_pairs(policy, records)["pairs"], 3)
        with self.assertRaises(ValueError):
            evaluate_preference_pairs(policy, [])

    def test_a_held_out_split_needs_checkpoints_and_a_sane_stop_margin(self):
        with self.assertRaises(ValueError):
            TrainingConfig(dataset="d", out_dir="o", validation_dataset="v.jsonl", save_every=0)
        with self.assertRaises(ValueError):
            TrainingConfig(dataset="d", out_dir="o", eval_limit=-1)
        with self.assertRaises(ValueError):
            TrainingConfig(dataset="d", out_dir="o", early_stop_margin=-0.5)
        with self.assertRaises(ValueError):
            TrainingConfig(dataset="d", out_dir="o", smoke_limit=4, save_every=0)
        with self.assertRaises(ValueError):
            TrainingConfig(dataset="d", out_dir="o", collapse_margin=-1)
        with self.assertRaises(ValueError):
            TrainingConfig(dataset="d", out_dir="o", smoke_pass_rate=0)
        self.assertEqual(TrainingConfig(dataset="d", out_dir="o").validation_dataset, "")

    def test_the_best_checkpoint_is_chosen_by_held_out_score_not_by_arrival(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root, count=4)
            held_out = self._held_out(root, count=2)
            run = root / "run"
            report = train_dpo(_config(dataset, run, validation_dataset=str(held_out)),
                               backend=_ControlledMarginBackend([0.0, 0.0, 4.0, 9.0]))
            rows = [json.loads(line) for line in (run / VALIDATION_METRICS_FILE).read_text(encoding="utf-8").splitlines() if line.strip()]
            best = json.loads((run / BEST_CHECKPOINT_FILE).read_text(encoding="utf-8"))

        self.assertEqual([row["step"] for row in rows], [2, 4, 6, 8])
        self.assertEqual(report["validation"]["evaluations"], rows)
        self.assertFalse(report["stopped_early"])
        self.assertEqual(report["metrics"]["steps"], 8)
        # Every checkpoint from step 2 on ranks every held-out pair correctly, so the tie is
        # broken by the earlier step: never the last one, even when a later margin is smaller.
        self.assertEqual([row["pairwise_accuracy"] for row in rows], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(report["validation"]["best"]["step"], 2)
        self.assertEqual(best["step"], 2)
        self.assertEqual(best["metrics"]["median_margin"], 4.0)
        self.assertEqual(report["validation"]["best_checkpoint"], report["checkpoints"][0])
        self.assertNotEqual(report["validation"]["best_checkpoint"], report["checkpoints"][-1])
        self.assertEqual(report["stopped_early_reason"], "")

    def test_a_later_checkpoint_with_a_smaller_margin_does_not_replace_an_earlier_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root, count=4)
            held_out = self._held_out(root, count=2)
            run = root / "run"
            report = train_dpo(_config(dataset, run, validation_dataset=str(held_out)),
                               backend=_ControlledMarginBackend([0.0, 0.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        self.assertEqual(report["validation"]["best"]["step"], 2)
        self.assertEqual(report["validation"]["best"]["median_margin"], 9.0)

    def test_training_stops_once_the_held_out_margin_reaches_the_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root, count=4)
            held_out = self._held_out(root, count=2)
            run = root / "run"
            report = train_dpo(_config(dataset, run, validation_dataset=str(held_out), early_stop_margin=1.0),
                               backend=_ControlledMarginBackend([0.0, 0.0, 4.0, 9.0]))
            rows = [json.loads(line) for line in (run / VALIDATION_METRICS_FILE).read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertTrue(report["stopped_early"])
        self.assertEqual(report["metrics"]["steps"], 2)
        self.assertEqual(report["target_steps"], 8)
        self.assertEqual([row["step"] for row in rows], [2])
        self.assertEqual(report["validation"]["best"]["step"], 2)
        self.assertEqual(report["stopped_early_reason"], "early_stop_margin")

    def test_training_stops_when_generation_smoke_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root, count=4)
            held_out = self._held_out(root, count=2)
            run = root / "run"
            report = train_dpo(_config(dataset, run, validation_dataset=str(held_out), smoke_limit=2, smoke_pass_rate=0.5),
                               backend=_ScriptedGenerateBackend([0.0, 0.0, 1.0, 1.0, 4.0, 4.0, 9.0, 9.0],
                                                                ['{"preferred_candidate":"A"}', ""]))
            smokes = [json.loads(line) for line in (run / GENERATION_SMOKE_FILE).read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertTrue(report["stopped_early"])
        self.assertEqual(report["stopped_early_reason"], "generation_smoke")
        self.assertEqual(report["metrics"]["steps"], 4)
        self.assertEqual(report["validation"]["best"]["step"], 2)
        self.assertEqual([row["passed"] for row in smokes], [True, False])
        self.assertEqual(report["generation_smoke"]["rows"], smokes)

    def test_training_stops_when_held_out_pairwise_stalls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root, count=4)
            held_out = self._held_out(root, count=2)
            run = root / "run"
            report = train_dpo(_config(dataset, run, validation_dataset=str(held_out), pairwise_patience=2),
                               backend=_ControlledMarginBackend([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

        self.assertTrue(report["stopped_early"])
        self.assertEqual(report["stopped_early_reason"], "pairwise_patience")
        self.assertEqual(report["metrics"]["steps"], 8)
        self.assertEqual(report["validation"]["best"]["step"], 4)

    def test_training_stops_when_a_saturated_margin_crosses_the_collapse_ceiling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root, count=4)
            held_out = self._held_out(root, count=2)
            run = root / "run"
            report = train_dpo(_config(dataset, run, validation_dataset=str(held_out), collapse_margin=8.0),
                               backend=_ControlledMarginBackend([0.0, 0.0, 9.0, 9.0, 20.0, 20.0]))

        self.assertTrue(report["stopped_early"])
        self.assertEqual(report["stopped_early_reason"], "collapse_margin")
        self.assertEqual(report["metrics"]["steps"], 2)

    def test_a_synthetic_backend_skips_generation_smoke_without_stopping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root, count=4)
            held_out = self._held_out(root, count=2)
            run = root / "run"
            report = train_dpo(_config(dataset, run, validation_dataset=str(held_out), smoke_limit=2))

        self.assertFalse(report["stopped_early"])
        self.assertEqual(report["metrics"]["steps"], 8)
        self.assertTrue(all(row.get("skipped") for row in report["generation_smoke"]["rows"]))
        self.assertTrue(all(row.get("passed") for row in report["generation_smoke"]["rows"]))

    def test_a_run_without_a_held_out_split_scores_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = _dataset(root, count=3)
            run = root / "run"
            report = train_dpo(_config(dataset, run))

        self.assertFalse(report["stopped_early"])
        self.assertEqual(report["validation"]["evaluations"], [])
        self.assertIsNone(report["validation"]["best"])
        self.assertEqual(report["validation"]["best_checkpoint"], "")
        self.assertEqual(report["validation"]["metrics_path"], "")
        self.assertFalse((run / VALIDATION_METRICS_FILE).exists())
        self.assertFalse((run / BEST_CHECKPOINT_FILE).exists())
        self.assertEqual(report["generation_smoke"]["rows"], [])
        self.assertEqual(report["stopped_early_reason"], "")


class GenerationSmokeTests(unittest.TestCase):
    def test_a_repeated_syllable_is_an_obvious_collapse(self):
        self.assertFalse(has_obvious_repetition(""))
        self.assertFalse(has_obvious_repetition('{"preferred_candidate":"A"}'))
        self.assertTrue(has_obvious_repetition("d" * 8))
        self.assertTrue(has_obvious_repetition("di" * 8))
        self.assertFalse(score_generation("")["passed"])
        self.assertTrue(score_generation('{"preferred_candidate":"A"}')["passed"])
        self.assertFalse(score_generation("di" * 20)["passed"])

    def test_a_backend_without_generate_is_skipped_not_failed(self):
        records = [_record("v0", chosen="{}", rejected="{}")]
        smoke = run_generation_smoke(_ControlledMarginBackend([1.0]), records, limit=1)
        self.assertTrue(smoke["skipped"])
        self.assertTrue(smoke["passed"])


if __name__ == "__main__":
    unittest.main()
