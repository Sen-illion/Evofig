"""Tests for binding a registered Judge checkpoint to the run it measures."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rfs.cli import build_parser
from rfs.judge_training.registry import checkpoint_identity, register_checkpoint, resolve_serving_checkpoint
from rfs.workflows import resolve_judge_binding


APPROVED = {"approved": True, "hidden_test_passed": True, "adversarial_passed": True, "shadow_stable": True}


def _registry(root: Path, *, model_id: str = "judge-v1", roles: list[str] | None = None, dataset_version: str = "judge-dataset-v003", git_commit: str = "cafebabe") -> Path:
    """Register one checkpoint and return the registry it went into."""

    checkpoint = root / f"{model_id}.bin"
    checkpoint.write_bytes(b"checkpoint")
    registry = root / "registry.json"
    manifest = {"model_id": model_id, "dataset_version": dataset_version, "roles": roles or ["online", "frozen"], "git_commit": git_commit}
    if any(role in {"online", "frozen"} for role in manifest["roles"]):
        manifest["promotion"] = dict(APPROVED)
    register_checkpoint(checkpoint, manifest, registry)
    return registry


class CheckpointIdentityTests(unittest.TestCase):
    def test_the_identity_separates_two_trainings_of_one_model_id(self):
        first = {"model_id": "judge-v1", "dataset_version": "ds-v3", "git_commit": "aaa"}
        second = {"model_id": "judge-v1", "dataset_version": "ds-v4", "git_commit": "bbb"}

        self.assertEqual(checkpoint_identity(first), "judge-v1@ds-v3+aaa")
        self.assertNotEqual(checkpoint_identity(first), checkpoint_identity(second))

    def test_a_manifest_without_a_model_id_names_nothing(self):
        self.assertEqual(checkpoint_identity({"dataset_version": "ds-v3", "git_commit": "aaa"}), "")


class ServingResolutionTests(unittest.TestCase):
    def test_a_checkpoint_registered_for_the_serving_roles_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(Path(tmp))

            record = resolve_serving_checkpoint("judge-v1", registry)

            self.assertEqual(record["model_id"], "judge-v1")
            self.assertEqual(sorted(record["roles"]), ["frozen", "online"])

    def test_an_experimental_checkpoint_may_be_read_but_not_served(self):
        """Registering a Judge that did not pass is allowed; being measured by it is not."""

        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(Path(tmp), model_id="judge-rc", roles=["experimental"])

            with self.assertRaises(ValueError) as caught:
                resolve_serving_checkpoint("judge-rc", registry)

            self.assertIn("did not pass the promotion gate", str(caught.exception))

    def test_a_shadow_only_checkpoint_is_refused_for_the_frozen_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(Path(tmp), model_id="judge-shadow", roles=["shadow", "online"])

            with self.assertRaises(ValueError):
                resolve_serving_checkpoint("judge-shadow", registry)

    def test_an_unregistered_model_id_is_not_guessed_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(Path(tmp))

            with self.assertRaises(KeyError):
                resolve_serving_checkpoint("judge-v2", registry)


class JudgeBindingTests(unittest.TestCase):
    def test_a_run_that_names_no_registry_stays_unbound(self):
        self.assertEqual(resolve_judge_binding(judge_mode="heuristic"), {})
        self.assertEqual(resolve_judge_binding(judge_mode="vlm", registry=None, model_id=""), {})

    def test_half_a_binding_is_refused_rather_than_guessed_at(self):
        with self.assertRaises(ValueError):
            resolve_judge_binding(judge_mode="trained", model_id="judge-v1")
        with self.assertRaises(ValueError):
            resolve_judge_binding(judge_mode="trained", registry="registry.json")

    def test_a_checkpoint_the_run_will_not_call_is_refused(self):
        """Only trained mode serves a checkpoint, so only trained mode may be stamped with one."""

        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(Path(tmp))

            with self.assertRaises(ValueError) as caught:
                resolve_judge_binding(judge_mode="vlm", registry=registry, model_id="judge-v1")

            self.assertIn("only --judge-mode trained serves", str(caught.exception))

    def test_a_bound_run_carries_the_checkpoint_identity_and_the_served_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = _registry(Path(tmp))

            binding = resolve_judge_binding(judge_mode="trained", registry=registry, model_id="judge-v1")

            self.assertEqual(binding["identity"], "judge-v1@judge-dataset-v003+cafebabe")
            self.assertEqual(binding["model_id"], "judge-v1")
            self.assertEqual(binding["dataset_version"], "judge-dataset-v003")
            self.assertEqual(binding["registry"], str(Path(registry).resolve()))


class BindingCommandLineTests(unittest.TestCase):
    def test_the_product_command_accepts_a_registry_and_a_model_id(self):
        args = build_parser().parse_args([
            "paper-evolve-editable", "--paper", "p.pdf", "--out", "out",
            "--judge-mode", "trained", "--registry", "registry.json", "--model-id", "judge-v1",
        ])

        self.assertEqual(args.registry, "registry.json")
        self.assertEqual(args.model_id, "judge-v1")

    def test_a_run_without_them_leaves_both_unset(self):
        args = build_parser().parse_args(["paper-evolve-editable", "--paper", "p.pdf", "--out", "out"])

        self.assertIsNone(args.registry)
        self.assertIsNone(args.model_id)

    def test_advisory_and_acceptance_model_flags_alias_the_legacy_dest(self):
        args = build_parser().parse_args([
            "paper-evolve-editable", "--paper", "p.pdf", "--out", "out",
            "--advisory-judge-model", "advisory-endpoint",
            "--acceptance-judge-model", "acceptance-endpoint",
        ])

        self.assertEqual(args.online_judge_model, "advisory-endpoint")
        self.assertEqual(args.frozen_judge_model, "acceptance-endpoint")

    def test_legacy_online_and_frozen_model_flags_still_set_the_same_dest(self):
        args = build_parser().parse_args([
            "paper-evolve-editable", "--paper", "p.pdf", "--out", "out",
            "--online-judge-model", "legacy-online",
            "--frozen-judge-model", "legacy-frozen",
        ])

        self.assertEqual(args.online_judge_model, "legacy-online")
        self.assertEqual(args.frozen_judge_model, "legacy-frozen")


if __name__ == "__main__":
    unittest.main()
