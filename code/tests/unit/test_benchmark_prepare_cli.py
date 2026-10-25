"""Tests for the benchmark prepare command line: which splits a freeze may draw from."""

from __future__ import annotations

import unittest
from argparse import Namespace
from unittest import mock

from rfs.cli import _run_benchmark_prepare


def prepare_args(**overrides):
    values = {
        "out": "gold",
        "verify": False,
        "papers": None,
        "figures": None,
        "reviews": None,
        "contracts": None,
        "pdf_root": None,
        "categories_csv": None,
        "splits": "",
        "target": 5,
        "force": False,
        "label_groups": "good,bad",
    }
    values.update(overrides)
    return Namespace(**values)


class BenchmarkPrepareSplitsTests(unittest.TestCase):
    """The released freeze draws on held-out splits unless all is asked for."""

    def _call(self, **overrides):
        args = prepare_args(**overrides)
        with mock.patch("rfs.cli.prepare_gold_instances", return_value={"ok": True}) as prepared:
            report = _run_benchmark_prepare(args)
        return report, prepared.call_args

    def test_the_default_stays_on_the_held_out_splits(self):
        _, call = self._call()
        self.assertEqual(("test", "validation"), call.kwargs["include_splits"])

    def test_all_expands_to_every_split(self):
        _, call = self._call(splits="all")
        self.assertEqual(("train", "test", "validation"), call.kwargs["include_splits"])

    def test_explicit_splits_are_forwarded(self):
        _, call = self._call(splits="train, validation")
        self.assertEqual(("train", "validation"), call.kwargs["include_splits"])

    def test_an_unknown_split_is_refused(self):
        with self.assertRaises(ValueError):
            self._call(splits="train,holdout")

    def test_an_unknown_label_group_is_refused(self):
        with self.assertRaises(ValueError):
            self._call(label_groups="good,excellent")

    def test_category_metadata_is_forwarded_only_when_given(self):
        _, call = self._call(categories_csv="papers.csv")
        self.assertEqual("papers.csv", call.kwargs["categories_path"])
        _, default_call = self._call()
        self.assertNotIn("categories_path", default_call.kwargs)

    def test_verify_does_not_rebuild_the_freeze(self):
        with mock.patch("rfs.cli.verify_gold_instances", return_value={"ok": True}) as verify:
            with mock.patch("rfs.cli.prepare_gold_instances") as prepared:
                _run_benchmark_prepare(prepare_args(verify=True))
        verify.assert_called_once_with("gold")
        self.assertFalse(prepared.called)


if __name__ == "__main__":
    unittest.main()
