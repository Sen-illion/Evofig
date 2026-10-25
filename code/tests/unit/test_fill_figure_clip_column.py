"""Unit tests for scripts/fill_figure_clip_column.py.

The point of the script is that it recomputes *one* column and leaves the rest alone, because the
other two columns cost an API call each and are already paid for. So what is pinned here is that
the other keys survive, that a filled column is left alone without --force, and that a folder it
cannot read is reported rather than half-written.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fill_figure_clip_column", ROOT / "scripts" / "fill_figure_clip_column.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


clip = load_module()


class FakeEmbedder:
    model_name = "BAAI/AltCLIP"

    def embed_image(self, path):
        return [1.0, 0.0]

    def embed_text(self, text):
        return [0.5, 0.5]


class FillFigureClipColumnTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name) / "runs"
        self.paper = self.root / "10.1_a"
        self.paper.mkdir(parents=True)
        (self.paper / "preview.png").write_bytes(b"png")
        (self.paper / "method.txt").write_text("the method", encoding="utf-8")
        self.body = {
            "paper_id": "10.1/a",
            "vlm_nine": {"available": True, "mean": 3.5},
            "paper_quiz": {"available": True, "accuracy": 0.25},
            "paper_figure_clip": {"available": False, "cosine": None, "reason": "embedder_unavailable"},
        }
        self._write(self.body)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, body):
        (self.paper / "metrics.json").write_text(json.dumps(body), encoding="utf-8")

    def _run(self, argv=()):
        code = clip.main(["--runs", str(self.root), "--clip-model", str(self.root / "model"), *argv])
        return code, json.loads((self.root / "_clip_fill_report.json").read_text(encoding="utf-8"))

    def test_the_clip_column_is_filled_and_the_other_columns_survive(self):
        original = clip.LocalAltCLIP
        clip.LocalAltCLIP = lambda directory: FakeEmbedder()
        try:
            code, report = self._run()
        finally:
            clip.LocalAltCLIP = original
        self.assertEqual(code, 0)
        self.assertEqual(report["filled"], 1)
        body = json.loads((self.paper / "metrics.json").read_text(encoding="utf-8"))
        self.assertTrue(body["paper_figure_clip"]["available"])
        self.assertAlmostEqual(body["paper_figure_clip"]["cosine"], 2 ** -0.5, places=6)
        self.assertEqual(body["vlm_nine"]["mean"], 3.5)
        self.assertEqual(body["paper_quiz"]["accuracy"], 0.25)

    def test_a_filled_column_is_left_alone_without_force(self):
        self.body["paper_figure_clip"] = {"available": True, "cosine": 0.31}
        self._write(self.body)
        called = []

        class Recorder(FakeEmbedder):
            def __init__(self, directory):
                called.append(directory)

        original = clip.LocalAltCLIP
        clip.LocalAltCLIP = Recorder
        try:
            _, report = self._run()
        finally:
            clip.LocalAltCLIP = original
        self.assertEqual(report["already_filled"], 1)
        self.assertEqual(report["attempted"], 0)
        self.assertEqual([], called)
        body = json.loads((self.paper / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(body["paper_figure_clip"]["cosine"], 0.31)

    def test_a_missing_method_section_leaves_the_column_null_with_a_reason(self):
        (self.paper / "method.txt").unlink()
        original = clip.LocalAltCLIP
        clip.LocalAltCLIP = lambda directory: FakeEmbedder()
        try:
            _, report = self._run()
        finally:
            clip.LocalAltCLIP = original
        self.assertEqual(report["filled"], 0)
        self.assertEqual(report["failed"][0]["reason"], "missing method section")
        body = json.loads((self.paper / "metrics.json").read_text(encoding="utf-8"))
        self.assertFalse(body["paper_figure_clip"]["available"])


if __name__ == "__main__":
    unittest.main()