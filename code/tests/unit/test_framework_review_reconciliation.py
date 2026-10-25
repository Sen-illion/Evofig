"""Reconcile the framework-figure review clicks into a ledger the export can read.

The review sites store an index into the draft screen, not a paper. These tests
pin the mapping, the last-click-wins rule, and the rule that the draft screen
alone never becomes a human decision.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

from rfs.evaluation.benchmark_manifest import (
    export_public_benchmark_manifest,
    framework_figure_label,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "reconcile_framework_reviews",
        ROOT / "benchmarks" / "editable-figure-bench" / "review" / "reconcile_framework_reviews.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reconcile_module = load_module()


def _jsonl(path: pathlib.Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_manifest_dir(root: pathlib.Path) -> pathlib.Path:
    manifest = root / "manifest"
    figure_a = "figs/a.png"
    figure_b = "figs/b.png"
    figure_gone = "figs/gone.png"
    for name in (figure_a, figure_b):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"png")
    _jsonl(
        manifest / "framework_screen_draft.jsonl",
        [
            {"n": 0, "paper_id": "10.1/a", "arxiv_id": "1111.00001", "figure": figure_a, "label": "unknown"},
            {"n": 1, "paper_id": "10.1/b", "arxiv_id": "", "figure": figure_b, "label": "unknown"},
            {"n": 2, "paper_id": "10.1/c", "arxiv_id": "2222.00002", "figure": figure_gone, "label": "framework_like"},
        ],
    )
    _jsonl(
        manifest / "framework_rescreen_review.jsonl",
        [
            {"n": 0, "decision": "framework_like", "at": "2026-10-01T07:00:00+00:00"},
            {"n": 0, "decision": "not_framework", "at": "2026-10-01T08:00:00+00:00"},
        ],
    )
    _jsonl(
        manifest / "framework_paper_captures.jsonl",
        [{"n": 1, "label": "framework_like", "capture": "crops/0001_p02.png"}],
    )
    _jsonl(
        manifest / "hidden_test_manifest.jsonl",
        [
            {"paper_id": "10.1/a", "arxiv_id": "1111.00001"},
            {"paper_id": "10.1/b", "arxiv_id": ""},
            {"paper_id": "10.1/c", "arxiv_id": "2222.00002"},
        ],
    )
    return manifest


class ReconcileTests(unittest.TestCase):
    def test_last_click_wins_and_a_label_counts_as_a_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = _write_manifest_dir(root)
            report = reconcile_module.reconcile(manifest_dir=manifest, repo=root)
            rows = {row["paper_id"]: row for row in reconcile_module._rows(manifest / "framework_human_reviews.jsonl")}
        self.assertEqual({"10.1/a", "10.1/b"}, set(rows))
        self.assertEqual("none", rows["10.1/a"]["decision"])
        self.assertEqual("framework_rescreen_review.jsonl", rows["10.1/a"]["review_log"])
        self.assertEqual("selected", rows["10.1/b"]["decision"])
        self.assertEqual("framework_paper_captures.jsonl", rows["10.1/b"]["review_log"])
        self.assertEqual(2, report["ledger_rows"])

    def test_the_draft_screen_alone_is_not_a_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = _write_manifest_dir(root)
            report = reconcile_module.reconcile(manifest_dir=manifest, repo=root)
            rows = reconcile_module._rows(manifest / "framework_human_reviews.jsonl")
        self.assertNotIn("10.1/c", {row["paper_id"] for row in rows})
        self.assertEqual(1, report["papers_without_a_human_decision"])
        self.assertEqual({"framework_like": 1}, report["papers_without_a_human_decision_by_draft_label"])

    def test_a_paper_without_an_arxiv_id_keeps_its_paper_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = _write_manifest_dir(root)
            report = reconcile_module.reconcile(manifest_dir=manifest, repo=root)
            rows = reconcile_module._rows(manifest / "framework_human_reviews.jsonl")
        row = next(row for row in rows if row["paper_id"] == "10.1/b")
        self.assertEqual("", row["arxiv_id"])
        self.assertEqual([1], report["ledger_rows_without_arxiv_id"])

    def test_an_index_outside_the_draft_is_reported_not_dropped_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            manifest = _write_manifest_dir(root)
            _jsonl(
                manifest / "framework_unknown_review.jsonl",
                [{"n": 99, "decision": "framework_like", "at": "2026-10-01T09:00:00+00:00"}],
            )
            report = reconcile_module.reconcile(manifest_dir=manifest, repo=root)
        self.assertEqual([{"log": "framework_unknown_review.jsonl", "n": 99}], report["unresolved_indexes"])


class LedgerJoinTests(unittest.TestCase):
    def test_a_review_can_join_on_the_paper_id(self):
        reviews = {"10.1/a": {"decision": "selected"}}
        self.assertEqual("yes", framework_figure_label(["10.1/a"], reviews))
        self.assertEqual("unknown", framework_figure_label(["1111.00001"], reviews))

    def test_a_later_ledger_wins_and_the_export_reads_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            _jsonl(
                root / "ledger.jsonl",
                [
                    {
                        "paper_id": "p",
                        "split": "hidden_test",
                        "primary_domain": "rag",
                        "figure_asset_tier": "A_source_verified",
                        "identity_keys": {"arxiv_id": ["2305.1"]},
                    }
                ],
            )
            _jsonl(root / "reviews.jsonl", [{"arxiv_id": "2305.1", "decision": "none"}])
            _jsonl(root / "extra.jsonl", [{"arxiv_id": "2305.1", "decision": "selected"}])
            _jsonl(root / "framework.jsonl", [])
            (root / "snapshot").mkdir()
            report = export_public_benchmark_manifest(
                ledger_path=root / "ledger.jsonl",
                snapshot=root / "snapshot",
                reviews_path=[root / "reviews.jsonl", root / "extra.jsonl"],
                framework_papers_path=root / "framework.jsonl",
                out_dir=root / "out",
            )
        self.assertEqual(1, report["framework_figure_counts"]["yes"])
        self.assertEqual(0, report["framework_figure_counts"]["no"])


if __name__ == "__main__":
    unittest.main()
