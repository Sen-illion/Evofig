"""Unit tests for scripts/build_public_lists.py.

The public benchmarks do not hand the system a list of papers; the script turns each of them
into the one row shape the arm driver reads and stages the inputs beside the list. These tests
cover the resolution order (live source, archived copy, OCR text) and the refusal to claim an
input that is not there. The parquet reader is injected, so no parquet file is needed here.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_public_lists", ROOT / "scripts" / "build_public_lists.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module()


class SciFlowTests(unittest.TestCase):
    def build_tree(self, tmp):
        root = pathlib.Path(tmp)
        index = root / "index.json"
        index.write_text(
            json.dumps({"cv_2025_autobench": ["1111.0001v1", "1111.0002v1"], "ml_2025_autobench": ["2222.0003v1"]}),
            encoding="utf-8",
        )
        autobench = root / "autobench"
        (autobench / "cv_2025_autobench" / "1111.0001v1").mkdir(parents=True)
        (autobench / "cv_2025_autobench" / "1111.0001v1" / "paper.pdf").write_bytes(b"%PDF-1.4 live")
        runs = root / "runs"
        archived = runs / "cv_2025_autobench__1111.0002v1" / "inputs"
        archived.mkdir(parents=True)
        (archived / "paper.pdf").write_bytes(b"%PDF-1.4 archived")
        ocr = runs / "ml_2025_autobench__2222.0003v1"
        ocr.mkdir(parents=True)
        (ocr / "source_text.md").write_text("# paper as OCR markdown\n", encoding="utf-8")
        return index, autobench, runs

    def test_each_paper_keeps_its_index_order_and_names_where_it_came_from(self):
        with tempfile.TemporaryDirectory() as tmp:
            index, autobench, runs = self.build_tree(tmp)
            rows, report = builder.sciflow_rows(index, autobench, runs, pathlib.Path(tmp) / "staged")
            self.assertEqual([row["paper_id"] for row in rows], [
                "cv_2025_autobench/1111.0001v1",
                "cv_2025_autobench/1111.0002v1",
                "ml_2025_autobench/2222.0003v1",
            ])
            self.assertEqual(
                [row["input_kind"] for row in rows],
                ["benchmark_pdf", "archived_run_pdf", "ocr_markdown"],
            )
            self.assertEqual(report["input_kinds"]["benchmark_pdf"], 1)
            for row in rows:
                self.assertTrue(pathlib.Path(row["pdf_path"]).is_file(), row["pdf_path"])
            self.assertIn("source_text.md", rows[2]["pdf_path"])

    def test_a_paper_with_no_source_is_reported_and_not_claimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            index, autobench, runs = self.build_tree(tmp)
            index.write_text(json.dumps({"cv_2025_autobench": ["9999.9999v9"]}), encoding="utf-8")
            rows, report = builder.sciflow_rows(index, autobench, runs, pathlib.Path(tmp) / "staged")
            self.assertEqual(rows, [])
            self.assertEqual(report["missing"], ["cv_2025_autobench/9999.9999v9"])

    def test_ocr_text_is_found_under_inputs_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            index, autobench, runs = self.build_tree(tmp)
            key = "ml_2025_autobench__2222.0003v1"
            (runs / key / "source_text.md").unlink()
            (runs / key / "inputs").mkdir()
            (runs / key / "inputs" / "source_text.md").write_text("# elsewhere\n", encoding="utf-8")
            rows, report = builder.sciflow_rows(index, autobench, runs, pathlib.Path(tmp) / "staged")
            self.assertEqual(report["input_kinds"].get("ocr_markdown"), 1)
            self.assertIn("source_text.md", rows[-1]["pdf_path"])


class FigureBenchTests(unittest.TestCase):
    def test_rows_are_named_by_position_and_the_text_is_staged(self):
        with tempfile.TemporaryDirectory() as tmp:
            staged = pathlib.Path(tmp) / "staged"
            reader = lambda path: (["first paper", "second paper"], ["cv", "nlp"])
            rows, report = builder.figurebench_rows(pathlib.Path(tmp) / "test.parquet", staged, reader)
            self.assertEqual([row["paper_id"] for row in rows], ["test_0000", "test_0001"])
            self.assertEqual([row["domain"] for row in rows], ["cv", "nlp"])
            self.assertEqual(report["rows"], 2)
            self.assertEqual((staged / "figurebench" / "test_0000.txt").read_text(encoding="utf-8"), "first paper")
            self.assertTrue(pathlib.Path(rows[0]["pdf_path"]).is_file())

    def test_a_list_and_its_report_land_where_the_protocol_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [{"domain": "cv", "paper_id": "test_0000", "pdf_path": "/x.txt", "input_kind": "text"}]
            builder.write_list(
                rows, pathlib.Path(tmp) / "list.jsonl", {"rows": 1}, pathlib.Path(tmp) / "report.json"
            )
            written = (pathlib.Path(tmp) / "list.jsonl").read_text(encoding="utf-8")
            self.assertEqual(json.loads(written.strip())["paper_id"], "test_0000")
            self.assertEqual(json.loads((pathlib.Path(tmp) / "report.json").read_text(encoding="utf-8"))["rows"], 1)

    def test_staging_reuses_an_input_that_is_already_the_right_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = pathlib.Path(tmp) / "a" / "paper.pdf"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"same bytes")
            target = pathlib.Path(tmp) / "b" / "paper.pdf"
            pathlib.Path(builder.stage(source, target)).read_bytes()
            self.assertEqual(builder.stage(source, target), str(target))
            self.assertEqual(target.read_bytes(), b"same bytes")


if __name__ == "__main__":
    unittest.main()
