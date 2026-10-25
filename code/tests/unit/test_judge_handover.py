"""Tests for the one-command Judge handover: train, predict, gate, register, sweep."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rfs.cli import _run_judge, build_parser, main
from rfs.contracts.evolution import judge_fingerprint_identity
from rfs.contracts.judge_handover import (
    HANDOVER_BLOCK_REASONS,
    HANDOVER_STEPS,
    JUDGE_HANDOVER_CONTRACT,
    JUDGE_HANDOVER_VERSION,
    validate_judge_handover_report,
)
from rfs.evolution.direct import ExperienceStore
from rfs.judge_training.evaluator import build_suite_manifest
from rfs.judge_training.promotion import promotion_decision
from rfs.judge_training.training import train_dpo
from rfs.utils import write_json
from rfs.workflows import run_judge_handover
from rfs.workflows.judge_handover import handover_fingerprint
from tests.unit.test_judge_dpo_training import _dataset
from tests.unit.test_judge_promotion import PREDICTIONS, SUITE, _write_rows
from tests.unit.test_memory_sweep import _improving_runner
from tests.unit.test_memory_sweep import _row as _experience_row
from tests.unit.test_memory_sweep import _suite as _memory_suite


NO_ENDPOINT = {"API_BASE": "", "API_KEY": "", "GEMINI_API_KEY": ""}
FAKE_ENDPOINT = {"API_BASE": "http://127.0.0.1:9/v1", "API_KEY": "test-key", "GEMINI_API_KEY": ""}
JUDGEMENTS = {str(row["example_id"]): row["prediction"] for row in PREDICTIONS}


def _save_every_step(config):
    """Save one checkpoint per step; the synthetic loop saves every fifty by default."""

    return train_dpo(replace(config, save_every=1))


def _fake_predictor(prompt, images, **_options):
    """Answer the way the fixture's perfect Judge does, keyed off the figure it is shown."""

    identifier = Path(str(images[0])).name[: -len("-a.png")]
    return json.loads(json.dumps(JUDGEMENTS[identifier]))


def _hidden_suite(root: Path) -> tuple[Path, Path]:
    """Write the gold set the promotion fixture passes, with its figures in place."""

    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in SUITE:
        entry = json.loads(json.dumps(row))
        for suffix in ("a", "b"):
            figure = images / f"{entry['example_id']}-{suffix}.png"
            figure.write_bytes(b"\x89PNG\r\n\x1a\n")
            entry[f"candidate_{suffix}"] = str(figure)
        rows.append(entry)
    dataset = _write_rows(root / "judge_examples.jsonl", rows)
    manifest = root / "suite_manifest.json"
    build_suite_manifest(dataset, manifest, suite="hidden_test", dataset_version="judge-dataset-v003")
    return dataset, manifest


def _seeded_library(root: Path) -> Path:
    """Seed one row of Memory an untrained stand-in Judge wrote."""

    library = root / "library"
    store = ExperienceStore(library)
    store.append_result(_experience_row("direct-exp-stale"))
    suite = _memory_suite(root / "memory_suite.json")
    report = write_json(root / "cross_paper_validation.json", {"status": "cross_domain_validated", "suite": str(suite)})
    store.transition("direct-exp-stale", "cross_domain_validated", validation_report=str(report))
    return library


def _handover_options(root: Path, **overrides) -> dict:
    """Every input a complete handover needs, so a test only spells out what it varies."""

    _dataset(root)
    _, manifest = _hidden_suite(root)
    values = {
        "out": root / "handover",
        "dataset": root / "dpo.jsonl",
        "suite_manifest": manifest,
        "model_id": "synthetic-judge-v1",
        "dataset_version": "judge-dataset-v003",
        "registry": root / "registry.json",
        "bootstrap_predictions": _write_rows(root / "bootstrap.jsonl", PREDICTIONS),
        "judge_model": "synthetic-judge-v1",
        "experience_library": _seeded_library(root),
        "training_runner": _save_every_step,
        "predict_adapter": _fake_predictor,
        "sweep_runner": _improving_runner,
        "git_commit": "cafebabe",
    }
    values.update(overrides)
    return values


def _staged_options(root: Path, out: Path, promotion: dict, **overrides) -> dict:
    """Options that arrive at the register step with every earlier artifact already written."""

    checkpoint = root / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    _dataset(root)
    _write_rows(out / "predictions" / "predictions.jsonl", PREDICTIONS)
    write_json(out / "promotion" / "promotion_report.json", {"promotion": promotion})
    values = {
        "out": out,
        "dataset": root / "dpo.jsonl",
        "checkpoint": str(checkpoint),
        "model_id": "synthetic-judge-v1",
        "dataset_version": "judge-dataset-v003",
        "registry": root / "registry.json",
        "git_commit": "cafebabe",
    }
    values.update(overrides)
    return values


class HandoverContractTests(unittest.TestCase):
    def _report(self, **overrides) -> dict:
        report = {
            "summary": "a handover",
            "ok": True,
            "dry_run": False,
            "force": False,
            "out_dir": "out",
            "model_id": "synthetic-judge-v1",
            "dataset_version": "judge-dataset-v003",
            "suite_id": "hidden_test",
            "suite_manifest": "suite.json",
            "checkpoint": "checkpoints/step-000001",
            "registry": "registry.json",
            "judge_fingerprint": {"role": "frozen", "mode": "trained", "trained": True, "model": "synthetic-judge-v1", "checkpoint": "synthetic-judge-v1@judge-dataset-v003+cafebabe"},
            "predictions_source": "run",
            "roles": ["experimental", "frozen"],
            "promotion": {"approved": True},
            "sweep": {"quarantined": 1, "restored": 1},
            "steps": [
                {
                    "step": name,
                    "status": "completed",
                    "detail": "done",
                    "artifacts": {"registry_entry": "registry.json"} if name == "register" else {},
                }
                for name in HANDOVER_STEPS
            ],
            "blockers": [],
        }
        report.update(overrides)
        return report

    def test_a_completed_chain_validates(self):
        checked = validate_judge_handover_report(self._report())

        self.assertTrue(checked["ok"], checked["errors"])
        self.assertEqual(checked["report"]["contract"], JUDGE_HANDOVER_CONTRACT)
        self.assertEqual(checked["report"]["version"], JUDGE_HANDOVER_VERSION)
        self.assertEqual([item["step"] for item in checked["report"]["steps"]], list(HANDOVER_STEPS))

    def test_steps_must_be_present_once_and_in_order(self):
        report = self._report(steps=list(reversed(self._report()["steps"])))

        checked = validate_judge_handover_report(report)

        self.assertFalse(checked["ok"])
        self.assertIn("exactly once and in order", checked["errors"][0])

    def test_a_blocked_step_has_to_name_a_reason_it_knows(self):
        steps = self._report()["steps"]
        steps[1] = {**steps[1], "status": "blocked", "reason": "no_coffee"}

        checked = validate_judge_handover_report(self._report(ok=False, steps=steps))

        self.assertFalse(checked["ok"])
        self.assertIn("unknown reason", " ".join(checked["errors"]))
        self.assertIn("no_coffee", " ".join(checked["errors"]))

    def test_every_known_blocker_reason_is_accepted(self):
        for reason in HANDOVER_BLOCK_REASONS:
            steps = self._report()["steps"]
            steps[3] = {**steps[3], "status": "blocked", "reason": reason}

            checked = validate_judge_handover_report(self._report(ok=False, roles=[], promotion={}, steps=steps))

            self.assertNotIn("unknown reason", " ".join(checked["errors"]), reason)

    def test_a_dry_run_may_not_claim_it_ran_anything(self):
        checked = validate_judge_handover_report(self._report(dry_run=True, roles=[], promotion={}, sweep={}))

        self.assertFalse(checked["ok"])
        self.assertIn("a dry run cannot have carried out these steps", " ".join(checked["errors"]))

    def test_failure_without_a_blocked_step_is_rejected(self):
        checked = validate_judge_handover_report(self._report(ok=False))

        self.assertFalse(checked["ok"])
        self.assertIn("without naming a blocked or failed step", " ".join(checked["errors"]))

    def test_a_registration_has_to_name_its_promotion_and_its_roles(self):
        checked = validate_judge_handover_report(self._report(promotion={}, roles=[]))

        self.assertFalse(checked["ok"])
        self.assertIn("the promotion block", " ".join(checked["errors"]))
        self.assertIn("the roles", " ".join(checked["errors"]))


class FingerprintTests(unittest.TestCase):
    def test_the_fingerprint_names_the_strongest_role_earned(self):
        fingerprint = handover_fingerprint(roles=["experimental", "shadow", "online"], judge_mode="trained", judge_model="qwen3-35b-a3b")

        self.assertEqual(fingerprint, {"role": "online", "mode": "trained", "trained": True, "model": "qwen3-35b-a3b", "checkpoint": ""})

    def test_the_fingerprint_carries_the_registered_checkpoint(self):
        """The sweep has to quarantine under the identity the evolution loop will stamp."""

        fingerprint = handover_fingerprint(roles=["frozen"], judge_mode="trained", judge_model="qwen3-35b-a3b", checkpoint="judge-v2@ds-v4+abc123")

        self.assertEqual(fingerprint["checkpoint"], "judge-v2@ds-v4+abc123")
        self.assertEqual(judge_fingerprint_identity(fingerprint), "trained:judge-v2@ds-v4+abc123")

    def test_a_heuristic_judge_is_not_recorded_as_trained(self):
        fingerprint = handover_fingerprint(roles=["experimental"], judge_mode="heuristic")

        self.assertFalse(fingerprint["trained"])
        self.assertEqual(fingerprint["role"], "experimental")

    def test_a_chain_that_registered_nothing_names_no_role(self):
        fingerprint = handover_fingerprint(roles=[], judge_mode="trained", model_id="synthetic-judge-v1")

        self.assertEqual(fingerprint["role"], "")
        self.assertEqual(fingerprint["model"], "synthetic-judge-v1")


class HandoverChainTests(unittest.TestCase):
    def test_the_whole_chain_runs_and_registers_a_frozen_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = _handover_options(root)

            with mock.patch.dict(os.environ, FAKE_ENDPOINT, clear=False):
                report = run_judge_handover(**options)

            self.assertTrue(report["ok"], report["validation"]["errors"])
            self.assertEqual([item["step"] for item in report["steps"]], list(HANDOVER_STEPS))
            self.assertEqual([item["status"] for item in report["steps"]], ["completed"] * 5)
            self.assertEqual(report["predictions_source"], "run")
            self.assertEqual(report["roles"], ["experimental", "shadow", "online", "frozen"])
            self.assertEqual(report["judge_fingerprint"], {"role": "frozen", "mode": "trained", "trained": True, "model": "synthetic-judge-v1", "checkpoint": "synthetic-judge-v1@judge-dataset-v003+cafebabe"})
            self.assertEqual(report["sweep"]["quarantined"], 1)
            self.assertEqual(report["sweep"]["restored"], 1)
            registry = json.loads((root / "registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["models"][0]["model_id"], "synthetic-judge-v1")
            self.assertIn("frozen", registry["models"][0]["roles"])
            self.assertEqual(registry["models"][0]["promotion"]["shadow_stable"], True)
            self.assertTrue((options["out"] / "judge_handover_report.json").is_file())
            self.assertIn("| Step | Status | Detail |", (options["out"] / "judge_handover_report.md").read_text(encoding="utf-8"))
            self.assertTrue(validate_judge_handover_report(report)["ok"])

    def test_a_second_run_reuses_every_artifact_instead_of_repeating_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = _handover_options(root)
            with mock.patch.dict(os.environ, FAKE_ENDPOINT, clear=False):
                run_judge_handover(**options)
                again = run_judge_handover(**options)

            self.assertEqual([item["status"] for item in again["steps"]], ["reused"] * 5)
            self.assertTrue(again["ok"], again["validation"]["errors"])
            self.assertEqual(again["roles"], ["experimental", "shadow", "online", "frozen"])
            self.assertEqual(again["sweep"]["restored"], 1)

    def test_force_re_runs_the_steps_whose_artifacts_are_already_there(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = _handover_options(root)
            with mock.patch.dict(os.environ, FAKE_ENDPOINT, clear=False):
                run_judge_handover(**options)
                forced = run_judge_handover(**options, force=True)

            self.assertEqual([item["status"] for item in forced["steps"]], ["completed"] * 5)
            self.assertTrue(forced["ok"], forced["validation"]["errors"])

    def test_a_dry_run_plans_the_chain_without_writing_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            options = _handover_options(root, out=root / "planned")

            report = run_judge_handover(**options, dry_run=True)

            self.assertTrue(report["ok"], report["validation"]["errors"])
            self.assertEqual([item["status"] for item in report["steps"]], ["pending"] * 5)
            self.assertFalse((root / "planned").exists())


class HandoverBlockerTests(unittest.TestCase):
    def test_a_suite_that_is_not_there_stops_the_chain_before_it_trains(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _dataset(root)

            report = run_judge_handover(out=root / "out", dataset=root / "dpo.jsonl", suite_manifest=root / "absent.json", git_commit="cafebabe")

            self.assertEqual([item["status"] for item in report["steps"]], ["blocked"] + ["skipped"] * 4)
            self.assertEqual(report["blockers"][0]["reason"], "no_suite")
            self.assertFalse(report["ok"])
            self.assertTrue((root / "out" / "judge_handover_report.json").is_file())

    def test_nothing_to_train_on_is_named_as_such(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_judge_handover(out=Path(tmp) / "out", git_commit="cafebabe")

            self.assertEqual(report["blockers"][0]["reason"], "no_dataset")
            self.assertFalse(report["ok"])

    def test_training_that_saved_no_checkpoint_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            report = run_judge_handover(out=root / "out", dataset=_dataset(root), training_runner=train_dpo, git_commit="cafebabe")

            self.assertEqual(report["steps"][0]["status"], "blocked")
            self.assertEqual(report["blockers"][0]["reason"], "no_checkpoint")
            self.assertFalse(report["ok"])

    def test_a_missing_bootstrap_stops_at_the_shadow_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            _dataset(root)
            out = root / "out"
            _write_rows(out / "predictions" / "predictions.jsonl", PREDICTIONS)

            report = run_judge_handover(out=out, dataset=root / "dpo.jsonl", checkpoint=str(checkpoint), git_commit="cafebabe")

            self.assertEqual([item["status"] for item in report["steps"]], ["reused", "reused", "blocked", "skipped", "skipped"])
            self.assertEqual(report["blockers"][0]["reason"], "no_bootstrap_predictions")
            self.assertFalse(report["ok"])

    def test_a_gate_that_did_not_approve_keeps_the_checkpoint_out_of_the_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            unapproved = promotion_decision(hidden_test_passed=True, class_gates_passed=False, adversarial_passed=True, shadow_stable=False)

            blocked = run_judge_handover(**_staged_options(root, root / "refused", unapproved))

            self.assertEqual(blocked["blockers"][0]["reason"], "promotion_not_approved")
            self.assertEqual(blocked["roles"], [])
            self.assertFalse(blocked["ok"])
            self.assertFalse((root / "registry.json").exists())

            allowed = run_judge_handover(**_staged_options(root, root / "experimental", unapproved, allow_experimental=True))

            self.assertTrue(allowed["ok"], allowed["validation"]["errors"])
            self.assertEqual(allowed["roles"], ["experimental"])
            self.assertTrue(allowed["registered_as_experimental"])
            registry = json.loads((root / "registry.json").read_text(encoding="utf-8"))
            self.assertEqual(registry["models"][0]["roles"], ["experimental"])

    def test_the_roles_come_from_the_gate_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            promotions = (
                promotion_decision(hidden_test_passed=True, class_gates_passed=True, adversarial_passed=True, shadow_stable=True),
                promotion_decision(hidden_test_passed=True, class_gates_passed=True, adversarial_passed=True, shadow_stable=False),
            )

            for index, promotion in enumerate(promotions):
                report = run_judge_handover(**_staged_options(root, root / f"run{index}", promotion))

                self.assertTrue(report["ok"], report["validation"]["errors"])
                self.assertEqual(report["roles"], promotion["roles"])

            self.assertEqual(promotions[0]["roles"], ["experimental", "shadow", "online", "frozen"])
            self.assertEqual(promotions[1]["roles"], ["experimental", "shadow", "online"])

    def test_a_registration_without_a_commit_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approved = promotion_decision(hidden_test_passed=True, class_gates_passed=True, adversarial_passed=True, shadow_stable=True)

            report = run_judge_handover(**_staged_options(root, root / "out", approved, git_commit=""), git_reader=lambda _project: "")

            self.assertEqual(report["blockers"][0]["reason"], "no_git_commit")
            self.assertFalse(report["ok"])

    def test_a_registration_without_a_dataset_version_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approved = promotion_decision(hidden_test_passed=True, class_gates_passed=True, adversarial_passed=True, shadow_stable=True)

            report = run_judge_handover(**_staged_options(root, root / "out", approved, dataset_version=""))

            self.assertEqual(report["blockers"][0]["reason"], "no_dataset_version")
            self.assertFalse(report["ok"])

    def test_a_handover_without_an_experience_library_skips_the_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approved = promotion_decision(hidden_test_passed=True, class_gates_passed=True, adversarial_passed=True, shadow_stable=True)

            report = run_judge_handover(**_staged_options(root, root / "out", approved))

            self.assertTrue(report["ok"], report["validation"]["errors"])
            self.assertEqual(report["steps"][-1]["status"], "skipped")
            self.assertEqual(report["steps"][-1]["reason"], "no_experience_library")

    def test_a_predict_step_without_an_endpoint_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            _dataset(root)

            with mock.patch.dict(os.environ, NO_ENDPOINT, clear=False):
                report = run_judge_handover(out=root / "out", dataset=root / "dpo.jsonl", checkpoint=str(checkpoint), git_commit="cafebabe")

            self.assertEqual([item["status"] for item in report["steps"]], ["reused", "blocked", "skipped", "skipped", "skipped"])
            self.assertEqual(report["blockers"][0]["reason"], "no_endpoint")
            self.assertFalse((root / "out" / "predictions" / "predictions.jsonl").exists())

    def test_handing_the_rows_over_marks_them_as_supplied(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approved = promotion_decision(hidden_test_passed=True, class_gates_passed=True, adversarial_passed=True, shadow_stable=True)
            supplied = _write_rows(root / "supplied.jsonl", PREDICTIONS)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(os.environ, NO_ENDPOINT, clear=False):
                report = run_judge_handover(
                    out=root / "out",
                    dataset=_dataset(root),
                    checkpoint=str(checkpoint),
                    predict_from=supplied,
                    model_id="synthetic-judge-v1",
                    dataset_version="judge-dataset-v003",
                    registry=root / "registry.json",
                    git_commit="cafebabe",
                )

            self.assertEqual(report["steps"][1]["status"], "reused")
            self.assertEqual(report["predictions_source"], "supplied")


class HandoverCommandTests(unittest.TestCase):
    def test_the_parser_exposes_judge_handover(self):
        args = build_parser().parse_args(["judge", "handover", "--out", "out", "--dry-run"])

        self.assertEqual(args.judge_action, "handover")
        self.assertEqual(args.judge_mode, "trained")
        self.assertEqual(args.backend, "synthetic")
        self.assertEqual(args.sweep_selection, "mismatched")
        self.assertEqual(args.sweep_limit, 0)
        self.assertTrue(args.dry_run)

    def test_a_dry_run_lists_the_five_steps_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = build_parser().parse_args(["judge", "handover", "--out", str(root / "out"), "--dataset", "dpo.jsonl", "--dry-run"])

            report = _run_judge(args)

            self.assertTrue(report["ok"], report["validation"]["errors"])
            self.assertEqual([item["step"] for item in report["steps"]], list(HANDOVER_STEPS))
            self.assertEqual([item["status"] for item in report["steps"]], ["pending"] * 4 + ["skipped"])
            self.assertEqual(report["steps"][2]["would_block"], "no_bootstrap_predictions")
            self.assertFalse((root / "out").exists())

    def test_a_blocked_chain_exits_non_zero_and_names_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            _dataset(root)
            argv = ["judge", "handover", "--out", str(root / "out"), "--dataset", str(root / "dpo.jsonl"), "--checkpoint", str(checkpoint), "--json"]
            buffer = io.StringIO()

            with mock.patch.dict(os.environ, NO_ENDPOINT, clear=False), contextlib.redirect_stdout(buffer):
                code = main(argv)

            self.assertEqual(code, 1)
            payload = json.loads(buffer.getvalue())
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["blockers"][0]["reason"], "no_endpoint")


if __name__ == "__main__":
    unittest.main()
