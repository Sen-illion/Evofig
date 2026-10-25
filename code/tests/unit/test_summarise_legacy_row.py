"""Unit tests for scripts/summarise_legacy_row.py.

The row the user chose to read out of the 2026-10-04 batch has to count the same papers as every
other row, so the paper that also sits in the Judge holdout has to leave the row. These tests
cover that exclusion, the id spellings it has to see through (a DOI with slashes in the list, the
same DOI with escaped underscores as a folder name), and the paired difference.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "summarise_legacy_row", ROOT / "scripts" / "summarise_legacy_row.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


row = load_module()


def write_paper(root, folder, paper_id, nine, editable, objects=32):
    target = pathlib.Path(root) / folder
    target.mkdir(parents=True)
    body = {
        "paper_id": paper_id,
        "vlm_nine": dict(nine, available=True, model="gemini-3-pro-preview", prompt_version="nine-v001"),
        "editability": {
            "status": "editable" if editable else "raster",
            "editable": editable,
            "raster_area_ratio": 0.05,
            "native_object_count": objects,
        },
    }
    (target / "metrics.json").write_text(json.dumps(body), encoding="utf-8")


class SummariseTests(unittest.TestCase):
    def build(self, tmp):
        final, round00 = pathlib.Path(tmp) / "final", pathlib.Path(tmp) / "round00"
        write_paper(final, "10.1038__s41467-026-72707-y", "10.1038/s41467-026-72707-y", {"accuracy": 5, "completeness": 5}, True)
        write_paper(final, "iclr_2026_10010667", "iclr:2026:10010667", {"accuracy": 3, "completeness": 1}, True)
        write_paper(final, "iclr_2026_10009799", "iclr:2026:10009799", {"accuracy": 1, "completeness": None}, False, objects=8)
        write_paper(round00, "10.1038__s41467-026-72707-y", "10.1038/s41467-026-72707-y", {"accuracy": 5, "completeness": 5}, True)
        write_paper(round00, "iclr_2026_10010667", "iclr:2026:10010667", {"accuracy": 1, "completeness": 1}, True)
        write_paper(round00, "iclr_2026_10009799", "iclr:2026:10009799", {"accuracy": 1, "completeness": 1}, False, objects=8)
        holdout = pathlib.Path(tmp) / "holdout_overlap.txt"
        holdout.write_text(
            "# comment\n\n10.1038/s41467-026-72707-y\n", encoding="utf-8"
        )
        return final, round00, holdout

    def test_a_holdout_paper_leaves_the_row_even_when_the_two_spellings_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            final, round00, holdout = self.build(tmp)
            blocked = row.held_out(holdout)
            self.assertEqual(blocked, {"10_1038_s41467_026_72707_y"})
            kept = {key: value for key, value in row.read_rows(final).items() if key not in blocked}
            self.assertEqual(sorted(kept), ["iclr_2026_10009799", "iclr_2026_10010667"])

    def test_a_holdout_entry_reads_from_a_judge_list_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "exam.jsonl"
            path.write_text(
                json.dumps({"paper_id": "10.1038/s41467-026-72707-y", "example_id": "x"}) + "\n",
                encoding="utf-8",
            )
            self.assertIn("10_1038_s41467_026_72707_y", row.held_out(path))

    def test_the_row_reports_the_nine_items_editability_and_the_paired_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            final, round00, holdout = self.build(tmp)
            blocked = row.held_out(holdout)
            final_rows = {k: v for k, v in row.read_rows(final).items() if k not in blocked}
            round_rows = {k: v for k, v in row.read_rows(round00).items() if k not in blocked}
            summary = row.summarise(final_rows)
            self.assertEqual(summary["papers"], 2)
            self.assertEqual(summary["nine"]["accuracy"], 2.0)
            self.assertEqual(summary["nine"]["completeness"], 1.0)
            self.assertEqual(summary["editability"]["editable_rate"], 0.5)
            self.assertEqual(summary["editability"]["native_object_count_mean"], 20.0)
            delta = row.paired(final_rows, round_rows, 1729)
            self.assertEqual(delta["accuracy"]["n"], 2)
            self.assertEqual(round(delta["accuracy"]["mean"], 6), 1.0)
            self.assertLessEqual(delta["accuracy"]["lo"], delta["accuracy"]["mean"])
            self.assertGreaterEqual(delta["accuracy"]["hi"], delta["accuracy"]["mean"])

    def test_a_missing_holdout_file_excludes_nothing_instead_of_failing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(row.held_out(pathlib.Path(tmp) / "nope.jsonl"), set())


if __name__ == "__main__":
    unittest.main()
