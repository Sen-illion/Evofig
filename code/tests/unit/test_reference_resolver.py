"""Tests for canonical curated-reference resolution."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from rfs.analysis.paper import normalize_quality_label, resolve_positive_reference


class ReferenceResolverTests(unittest.TestCase):
    def test_positive_metadata_is_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "2505.20824"
            folder.mkdir()
            Image.new("RGB", (10, 10)).save(folder / "framework.png")
            (folder / "meta.json").write_text(json.dumps({"quality": "优"}, ensure_ascii=False), encoding="utf-8")
            result = resolve_positive_reference("2505.20824.pdf", dataset_root=temp)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(normalize_quality_label("优秀"), "positive")

    def test_real_chinese_positive_label_and_multiple_images(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "2505.20824"
            folder.mkdir()
            Image.new("RGB", (10, 10)).save(folder / "other.png")
            Image.new("RGB", (10, 10)).save(folder / "framework.jpg")
            (folder / "meta.json").write_text(json.dumps({"quality": "优"}, ensure_ascii=False), encoding="utf-8")
            result = resolve_positive_reference("2505.20824.pdf", dataset_root=temp)
        self.assertEqual(result["status"], "selected")
        self.assertTrue(result["image"].endswith("framework.jpg"))
        self.assertEqual(len(result["candidates"]), 2)

    def test_negative_metadata_is_never_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "2505.20824"
            folder.mkdir()
            Image.new("RGB", (10, 10)).save(folder / "framework.png")
            (folder / "meta.json").write_text(json.dumps({"quality": "差"}, ensure_ascii=False), encoding="utf-8")
            result = resolve_positive_reference("2505.20824.pdf", dataset_root=temp)
        self.assertEqual(result["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
