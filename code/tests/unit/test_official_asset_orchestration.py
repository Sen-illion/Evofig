"""Tests for host-aware resumable official asset orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfs.judge_training.official_asset_orchestration import (
    _annotate_attempt,
    _terminalize_exhausted,
    orchestrate_official_assets,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _candidate(paper_id: str, *, arxiv_id: str | None = None) -> dict:
    return {
        "paper_id": paper_id,
        "title": paper_id,
        "year": 2026,
        "paper_type": "full",
        "official_record_status": "verified",
        "official_url": f"https://publisher.test/{paper_id}",
        "arxiv_id": arxiv_id,
    }


class OfficialAssetOrchestrationTests(unittest.TestCase):
    def test_circuit_open_does_not_consume_network_attempt(self):
        previous = {"attempt_count": 2, "status": "retryable_failure"}
        saved = _annotate_attempt(
            {
                "record_id": "retry-me",
                "official_url": "https://publisher.test/retry-me",
                "status": "retryable_circuit_open",
            },
            previous,
        )
        self.assertEqual(saved["attempt_count"], 2)
        self.assertIn("next_retry_at", saved)

    def test_third_network_failure_becomes_terminal(self):
        saved = _annotate_attempt(
            {
                "record_id": "retry-me",
                "official_url": "https://publisher.test/retry-me",
                "status": "retryable_failure",
            },
            {"attempt_count": 2, "status": "retryable_failure"},
        )
        self.assertEqual(saved["attempt_count"], 3)
        self.assertEqual(saved["status"], "terminal_retry_exhausted")
        self.assertEqual(saved["retry_exhausted_status"], "retryable_failure")
        self.assertNotIn("next_retry_at", saved)

    def test_legacy_exhausted_retry_is_terminalized(self):
        saved = _terminalize_exhausted(
            {
                "record_id": "retry-me",
                "status": "retryable_failure",
                "attempt_count": 3,
                "next_retry_at": "2026-09-16T00:00:00+00:00",
            }
        )
        self.assertEqual(saved["status"], "terminal_retry_exhausted")
        self.assertEqual(saved["retry_exhausted_status"], "retryable_failure")
        self.assertNotIn("next_retry_at", saved)

    def test_legacy_exhausted_retry_does_not_remain_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            row = _candidate("retry-me")
            _write_jsonl(batch / "candidates" / "agent" / "papers_arxiv_matched.jsonl", [row])
            _write_jsonl(
                batch / "manifests" / "official_asset_rows_agent.jsonl",
                [
                    {
                        "record_id": "retry-me",
                        "status": "retryable_failure",
                        "attempt_count": 3,
                    }
                ],
            )
            with patch("rfs.judge_training.official_assets.bind_official_asset") as binder:
                report = orchestrate_official_assets(batch, domains=["agent"], paper_workers=1)
            binder.assert_not_called()
            self.assertEqual(report["domains"]["agent"]["pending"], 0)
            self.assertTrue(report["domains"]["agent"]["complete"])

    def test_existing_a_binding_is_excluded_from_b(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            row = _candidate("paper-a", arxiv_id="2601.00001")
            _write_jsonl(batch / "candidates" / "agent" / "papers_arxiv_matched.jsonl", [row])
            archive = batch / "assets" / "agent" / "2601.00001" / "arxiv_source.src"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"source")
            source = batch / "manifests" / "source_probe_agent.json"
            source.parent.mkdir(parents=True)
            source.write_text(
                json.dumps(
                    {
                        "domains": {
                            "agent": {
                                "complete": True,
                                "rows": [{"arxiv_id": "2601.00001", "framework_source_bound": True}],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch("rfs.judge_training.official_assets.bind_official_asset") as binder:
                report = orchestrate_official_assets(batch, domains=["agent"], paper_workers=1)
            binder.assert_not_called()
            self.assertEqual(report["domains"]["agent"]["excluded_a"], 1)
            self.assertTrue(report["domains"]["agent"]["complete"])

    def test_no_arxiv_rows_are_processed_first(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            rows = [_candidate("has-arxiv", arxiv_id="2601.00002"), _candidate("needs-b")]
            _write_jsonl(batch / "candidates" / "agent" / "papers_arxiv_matched.jsonl", rows)
            order: list[str] = []

            def bind(_batch, _domain, row, **_kwargs):
                order.append(row["paper_id"])
                return {"record_id": row["paper_id"], "official_url": row["official_url"], "status": "no_eligible_standalone_figure"}

            with patch("rfs.judge_training.official_assets.bind_official_asset", side_effect=bind):
                orchestrate_official_assets(batch, domains=["agent"], paper_workers=1)
            self.assertEqual(order, ["needs-b", "has-arxiv"])

    def test_legacy_probe_failure_is_retried_and_checkpointed(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            row = _candidate("retry-me")
            _write_jsonl(batch / "candidates" / "agent" / "papers_arxiv_matched.jsonl", [row])
            output = batch / "manifests" / "official_asset_rows_agent.jsonl"
            _write_jsonl(output, [{"record_id": "retry-me", "status": "probe_failed"}])

            def bind(_batch, _domain, value, **_kwargs):
                return {"record_id": value["paper_id"], "official_url": value["official_url"], "status": "retryable_failure"}

            with patch("rfs.judge_training.official_assets.bind_official_asset", side_effect=bind):
                report = orchestrate_official_assets(batch, domains=["agent"], paper_workers=1)
            saved = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(saved["attempt_count"], 1)
            self.assertIn("next_retry_at", saved)
            self.assertEqual(report["domains"]["agent"]["retry_count"], 1)
            self.assertTrue((batch / "retry" / "official_asset_agent.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
