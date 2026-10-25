"""Tests for atomic literature checkpoints."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from rfs.judge_training.literature_io import atomic_write_text


class LiteratureIoTests(unittest.TestCase):
    def test_atomic_write_replaces_content_without_temp_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.jsonl"
            atomic_write_text(path, "first\n")
            atomic_write_text(path, "second\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "second\n")
            self.assertEqual(list(path.parent.glob(".checkpoint.jsonl.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
