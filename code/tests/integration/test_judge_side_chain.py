"""One Judge walked from a trained checkpoint to a registered Judge, without an endpoint.

This is the chain the Judge side has always had on paper and never run in one piece: train
a checkpoint, turn it into predicted rows, gate those rows, compare them against the
bootstrap, and register the result. The only seam replaced here is the provider call -- the
baseline takes a fake adapter instead of posting to an endpoint, because a machine without a
served checkpoint must report ``no_endpoint`` rather than invent a Judge of its own. Every
other step runs its real code and reads the artifact the previous step wrote.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rfs.cli import main
from rfs.judge_training.baseline import run_baseline
from rfs.workflows import run_judge_handover
from tests.unit.test_judge_handover import _fake_predictor, _hidden_suite
from tests.unit.test_judge_dpo_training import _dataset
from tests.unit.test_judge_promotion import PREDICTIONS, SUITE, _write_rows


NO_ENDPOINT = {"API_BASE": "", "API_KEY": "", "GEMINI_API_KEY": ""}


def run_cli(argv: list[str]) -> int:
    """Run one command the way a caller does, without its report landing in the test log."""

    with contextlib.redirect_stdout(io.StringIO()):
        return main(list(argv))


def _bootstrap(root: Path) -> Path:
    """Write the untrained stand-in's rows, which the shadow comparison holds the run against."""

    return _write_rows(root / "bootstrap.jsonl", PREDICTIONS)


def _steps(report: dict) -> dict[str, dict]:
    return {str(step["step"]): step for step in report["steps"]}


class JudgeSideChainTests(unittest.TestCase):
    def test_the_chain_runs_from_a_trained_checkpoint_to_a_registered_judge(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, NO_ENDPOINT, clear=False):
            root = Path(tmp)
            dataset = _dataset(root)
            run = root / "training"

            self.assertEqual(run_cli(["judge", "train", "--dataset", str(dataset), "--out", str(run), "--backend", "synthetic", "--epochs", "1", "--max-steps", "3", "--save-every", "1"]), 0)
            training = json.loads((run / "training_report.json").read_text(encoding="utf-8"))
            self.assertEqual(training["backend"]["name"], "synthetic-hashed-preference-v1")
            self.assertEqual(training["records"], 3)
            self.assertEqual(len(training["checkpoints"]), 3)
            self.assertTrue(all(Path(item).is_dir() for item in training["checkpoints"]))
            self.assertTrue((run / "version_lock.json").is_file())
            self.assertEqual(training["version_lock"]["backend"], "synthetic-hashed-preference-v1")

            measured, manifest = _hidden_suite(root)
            baseline = run_baseline(measured, root / "baseline", adapter=_fake_predictor)
            predictions = root / "baseline" / "predictions.jsonl"
            self.assertFalse(baseline["api_configured"])
            self.assertEqual(baseline["prediction_records"], len(SUITE))
            self.assertEqual(baseline["failure_records"], 0)
            self.assertTrue(predictions.is_file())
            self.assertEqual(len(predictions.read_text(encoding="utf-8").strip().splitlines()), len(SUITE))

            bootstrap = _bootstrap(root)
            self.assertEqual(run_cli(["judge", "evaluate", "--dataset", str(measured), "--predictions", str(predictions), "--checkpoint", "synthetic-judge-v1", "--suite", "hidden_test", "--suite-manifest", str(manifest), "--report-out", str(root / "gate")]), 0)
            self.assertTrue((root / "gate" / "judge_evaluation_report.json").is_file())
            self.assertEqual(run_cli(["judge", "shadow", "--dataset", str(measured), "--predictions", str(predictions), "--bootstrap-predictions", str(bootstrap), "--suite-manifest", str(manifest), "--report-out", str(root / "shadow")]), 0)
            self.assertTrue((root / "shadow" / "judge_evaluation_report.json").is_file())

            handover = root / "handover"
            self.assertEqual(
                run_cli([
                    "judge", "handover",
                    "--out", str(handover),
                    "--dataset", str(dataset),
                    "--suite-manifest", str(manifest),
                    "--model-id", "synthetic-judge-v1",
                    "--dataset-version", "judge-dataset-v003",
                    "--registry", str(root / "registry.json"),
                    "--training-report", str(run / "training_report.json"),
                    "--predict-from", str(predictions),
                    "--bootstrap-predictions", str(bootstrap),
                    "--judge-mode", "trained",
                    "--judge-model", "synthetic-judge-v1",
                    "--backend", "synthetic",
                    "--git-commit", "cafebabe",
                ]),
                0,
            )
            report = json.loads((handover / "judge_handover_report.json").read_text(encoding="utf-8"))
            steps = _steps(report)
            self.assertTrue(report["validation"]["ok"], report["validation"]["errors"])
            self.assertEqual(steps["train"]["status"], "reused")
            self.assertEqual(Path(steps["train"]["artifacts"]["checkpoint"]).name, "step-000003")
            self.assertEqual(steps["predict"]["status"], "reused")
            self.assertEqual(Path(steps["predict"]["artifacts"]["predictions"]), predictions)
            self.assertEqual(steps["promote"]["status"], "completed")
            self.assertTrue(report["promotion"]["approved"], report["promotion"].get("reasons"))
            self.assertEqual(steps["register"]["status"], "completed")
            self.assertTrue({"online", "shadow", "frozen"} <= set(report["roles"]), report["roles"])
            manifest_written = json.loads((handover / "registration" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest_written["roles"], report["roles"])
            self.assertEqual(steps["sweep"]["status"], "skipped")
            self.assertEqual(steps["sweep"]["reason"], "no_experience_library")
            self.assertTrue((handover / "registration" / "manifest.json").is_file())
            self.assertTrue((root / "registry.json").is_file())

    def test_a_handover_with_no_endpoint_stops_at_the_named_blocker(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, NO_ENDPOINT, clear=False):
            root = Path(tmp)
            dataset = _dataset(root)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            report = run_judge_handover(
                out=root / "handover",
                dataset=dataset,
                checkpoint=checkpoint,
                model_id="synthetic-judge-v1",
                dataset_version="judge-dataset-v003",
                registry=root / "registry.json",
                backend="synthetic",
                git_commit="cafebabe",
            )
            steps = _steps(report)
            self.assertFalse(report["ok"])
            self.assertEqual(steps["predict"]["reason"], "no_endpoint")
            self.assertIn("set API_BASE and API_KEY", steps["predict"]["detail"])
            for later in ("promote", "register", "sweep"):
                self.assertEqual(steps[later]["status"], "skipped")
                self.assertEqual(steps[later]["reason"], "earlier_step_blocked")
            self.assertEqual([item["step"] for item in report["blockers"]], ["predict"])


if __name__ == "__main__":
    unittest.main()
