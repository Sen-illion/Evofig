"""Tests for the training export a wall-clock session runs at the end of its budget.

The session already carries one station's artifacts into the next; what it also has at its end
is the material the next training round consumes. These tests pin that the three steps run, that
they land under ``<out>/judge_dataset/``, that the switch can turn them off, and that a failed
export is reported without pretending the session itself failed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rfs.cli import build_parser
from rfs.contracts.evolution_session import (
    EVOLUTION_SESSION_VERSION,
    normalize_evolution_session_report,
    validate_evolution_session_report,
)
from rfs.utils import write_json
from rfs.workflows import run_evolution_session
from tests.unit.test_judge_training_dataset import _make_run


def _paper(root: Path, name: str) -> Path:
    path = root / name
    path.write_bytes(b"%PDF-1.4\n")
    return path


def _config(root: Path, stations: list[dict], **overrides) -> Path:
    return write_json(root / "session.json", {"session_id": "session-export", "stations": stations, **overrides})


def _split_ledger(root: Path, rows: list[dict]) -> Path:
    """Write a frozen split ledger the way the ledger writer does: one JSON row per line."""

    path = root / "split_ledger.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


class _Clock:
    """A wall clock the test drives so no budget is really spent."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeEvolve:
    """A run that leaves a whole process-supervision round on disk for the exporter to read."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **options) -> dict:
        self.calls.append(dict(options))
        out = Path(str(options["out"]))
        out.mkdir(parents=True, exist_ok=True)
        _make_run(out)
        write_json(out / "memory_application_trace.json", {"contract": "rfs.memory_application_trace", "version": 1, "totals": {}, "rounds": []})
        return {"summary": "a run that returned", "ok": True, "out_dir": str(out), "rounds_completed": 1, "experience_library": str(options.get("experience_library") or "")}


class SessionTrainingExportTests(unittest.TestCase):
    def _run(self, root: Path, out: Path, **overrides) -> dict:
        station = {"kind": "evolve", "paper": str(_paper(root, "paper-a.pdf")), "paper_id": "paper-a", "max_rounds": 3, "candidates_per_round": 2}
        config = _config(root, [station])
        report = run_evolution_session(config, out, evolve_runner=_FakeEvolve(), clock=_Clock(), sleeper=lambda _s: None, **overrides)
        self.assertTrue(validate_evolution_session_report(report)["ok"], validate_evolution_session_report(report)["errors"])
        return report

    def test_the_session_stages_the_next_training_round_before_it_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "session"
            report = self._run(root, out)

            dataset = out / "judge_dataset"
            self.assertTrue((dataset / "judge_examples.jsonl").is_file())
            self.assertTrue((dataset / "preference_pairs.jsonl").is_file())
            self.assertTrue((dataset / "review" / "review_queue.jsonl").is_file())
            record = json.loads((out / "training_export.json").read_text(encoding="utf-8"))
            self.assertTrue(record["ok"])
            self.assertGreater(record["records"], 0)
            export = report["dataset_export"]
            self.assertTrue(export["ok"])
            self.assertTrue(export["attempted"])
            self.assertEqual("", export["reason"])
            self.assertEqual(record["records"], export["records"])

    def test_the_export_can_be_switched_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "session"
            report = self._run(root, out, training_export=False)

            self.assertFalse((out / "judge_dataset").exists())
            self.assertFalse((out / "training_export.json").exists())
            self.assertNotIn("dataset_export", report)

    def test_a_failed_export_is_reported_and_does_not_fail_the_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "session"
            with mock.patch("rfs.judge_training.dataset_builder.build_dataset", side_effect=RuntimeError("disk went away")):
                report = self._run(root, out)

            self.assertTrue(report["ok"])
            self.assertEqual(["completed"], [item["status"] for item in report["stations"]])
            export = report["dataset_export"]
            self.assertFalse(export["ok"])
            self.assertEqual("export_failed", export["reason"])
            self.assertIn("disk went away", export["detail"])
            record = json.loads((out / "training_export.json").read_text(encoding="utf-8"))
            self.assertFalse(record["ok"])
            events = [json.loads(line) for line in (out / "evolution_session_events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("dataset_export_failed", [item["event"] for item in events])

    def test_the_frozen_split_keeps_the_held_out_papers_out_of_the_training_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "session"
            ledger = _split_ledger(root, [{"paper_id": "p", "split": "hidden_test", "identity_keys": {"paper_id": ["p"]}}])
            report = self._run(root, out, export_split=str(ledger))

            self.assertTrue(report["dataset_export"]["ok"])
            self.assertEqual(str(ledger), report["dataset_export"]["split"])
            self.assertTrue((out / "judge_dataset" / "hidden_test.jsonl").is_file())


class SessionExportContractTests(unittest.TestCase):
    def _report(self, **overrides) -> dict:
        report = {
            "session_id": "session-x",
            "summary": "a session",
            "ok": True,
            "out_dir": "out",
            "budget": {"hours": 5.0, "used_seconds": 60.0, "exhausted": False},
            "stations": [{"kind": "evolve", "subject": "paper-a", "status": "completed", "attempts": [{"attempt": 1, "status": "ok"}]}],
            "blockers": [],
        }
        report.update(overrides)
        return report

    def test_a_version_one_report_is_still_written_and_read(self):
        written = normalize_evolution_session_report(self._report())
        self.assertEqual(EVOLUTION_SESSION_VERSION, written["version"])
        legacy = self._report(version=1)
        self.assertEqual(1, normalize_evolution_session_report(legacy)["version"])
        self.assertTrue(validate_evolution_session_report(legacy)["ok"])

    def test_an_unknown_version_is_rejected(self):
        check = validate_evolution_session_report(self._report(version=3))
        self.assertFalse(check["ok"])
        self.assertIn("unknown session report version: 3", check["errors"])

    def test_a_successful_export_block_validates_and_one_without_detail_does_not(self):
        good = self._report(dataset_export={"attempted": True, "ok": True, "reason": "", "detail": "", "records": 4})
        self.assertTrue(validate_evolution_session_report(good)["ok"])
        bad = self._report(dataset_export={"attempted": True, "ok": False, "reason": "export_failed", "detail": ""})
        check = validate_evolution_session_report(bad)
        self.assertFalse(check["ok"])
        self.assertIn("the dataset export failed without saying what stopped it", check["errors"])

    def test_the_cli_exposes_the_export_switches(self):
        args = build_parser().parse_args(["paper-evolve-session", "--config", "c.json", "--out", "o", "--no-training-export", "--export-split", "s.json"])
        self.assertTrue(args.no_training_export)
        self.assertEqual("s.json", args.export_split)
        default = build_parser().parse_args(["paper-evolve-session", "--config", "c.json", "--out", "o"])
        self.assertFalse(default.no_training_export)
        self.assertIsNone(default.export_split)


if __name__ == "__main__":
    unittest.main()
