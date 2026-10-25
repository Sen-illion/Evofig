"""Unit tests for the LF-only contract of the shared artifact writers."""

import tempfile
import unittest
from pathlib import Path

from rfs.utils import write_json, write_text


class WriterNewlineTests(unittest.TestCase):
    def test_json_artifacts_are_written_with_lf_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = write_json(Path(tmp) / "nested" / "report.json", {"a": [1, 2], "b": "中文"})
            raw = target.read_bytes()
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(raw.count(b"\n"), 6)

    def test_text_artifacts_are_written_with_lf_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = write_text(Path(tmp) / "nested" / "ledger.jsonl", "one\ntwo\n")
            raw = target.read_bytes()
            self.assertEqual(raw, b"one\ntwo\n")


if __name__ == "__main__":
    unittest.main()