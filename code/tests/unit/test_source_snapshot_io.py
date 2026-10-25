"""Snapshot I/O must fail closed on corruption, aliases and concurrent writers."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfs.contracts.source_snapshot import (
    SOURCE_SNAPSHOT_VERSION, snapshot_relative_path, validate_source_manifest,
)
from rfs.judge_training.literature_guard import (
    LiteratureBusyError, active_legacy_writers, literature_batch_guard,
)
from rfs.judge_training.source_snapshot_io import copy_verified, strict_jsonl


class SourceSnapshotIOTests(unittest.TestCase):
    def test_source_snapshot_integrity_is_independent_of_quota(self):
        manifest = {
            "schema_version": SOURCE_SNAPSHOT_VERSION, "snapshot_id": "paper-d0-v001",
            "training_readiness": "source_only", "snapshot_integrity_ok": True,
            "source_split_is_authoritative": False, "quota_met": False,
            "strict_unique_papers": 2,
            **{k: "a" * 64 for k in ("file_hashes_sha256", "paper_set_sha256", "input_set_sha256")},
        }
        self.assertEqual([], validate_source_manifest(manifest))
        manifest["source_split_is_authoritative"] = True
        self.assertIn("source_split_must_not_be_authoritative", validate_source_manifest(manifest))

    def test_jsonl_does_not_discard_malformed_or_nonobject_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            for content in ('{"paper_id":"a"}\n{broken}\n', '{"paper_id":"a"}\n[]\n'):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    strict_jsonl(path)

    def test_paths_are_portable_and_cannot_escape(self):
        for value in ("../a", "a/../b", "<PATH>", "/a", "a\\b", "a//b", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                snapshot_relative_path(value)
        self.assertEqual("assets/paper/figure.pdf", snapshot_relative_path("assets/paper/figure.pdf"))

    def test_unicode_separators_inside_json_strings_are_not_record_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.jsonl"
            rows = [{"page": 1, "text": "a\u0085b\u2028c\u2029d"}, {"page": 2, "text": "next"}]
            path.write_bytes(("\r\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\r\n").encode("utf-8"))
            self.assertEqual(rows, strict_jsonl(path))

    def test_copy_is_independent_and_refuses_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / "source", Path(folder) / "copy"
            source.write_bytes(b"original")
            digest = hashlib.sha256(b"original").hexdigest()
            copy_verified(source, target, digest)
            source.write_bytes(b"changed")
            self.assertEqual(b"original", target.read_bytes())
            with self.assertRaises(RuntimeError):
                copy_verified(source, target.with_name("second"), digest)
            self.assertFalse(target.with_name("second").exists())
            with self.assertRaises(ValueError):
                copy_verified(source, target, hashlib.sha256(b"changed").hexdigest())

    def test_guard_blocks_competing_handle_and_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            with literature_batch_guard(folder, purpose="test"):
                with self.assertRaises(LiteratureBusyError):
                    with literature_batch_guard(folder, purpose="competing"):
                        self.fail("competing writer acquired lock")
            with literature_batch_guard(folder, purpose="after"):
                pass

    def test_legacy_writer_report_contains_no_command_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            command = f'python -m rfs.cli judge-data collect-papers --batch-root "{folder}" --private-value SECRET'
            with patch("rfs.judge_training.literature_guard._python_processes", return_value=[
                {"ProcessId": 98765, "CommandLine": command},
            ]):
                result = active_legacy_writers(folder)
            self.assertEqual([{"pid": 98765, "worker": "collect-papers"}], result)
            self.assertNotIn("SECRET", json.dumps(result))
