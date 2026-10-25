"""Tests for the corpus command line: what it refuses to pay for.

A preview that starts paid workers is not a preview.  ``--sample-only`` used to
write the subset and then launch the whole batch anyway whenever ``--workers``
was above one, so asking to look at the paper list first bought the run.
"""

from __future__ import annotations

import unittest
from argparse import Namespace
from unittest.mock import patch

from rfs.cli import _run_corpus_shards


def corpus_args(**overrides):
    values = {
        "snapshot": "/snapshot",
        "split": "/split",
        "out": "/out",
        "pdf_roots": ["/papers"],
        "archive_root": None,
        "subset": "/out/subset",
        "subset_id": "subset",
        "per_domain": 20,
        "label_cache": None,
        "exclude_drawn_root": None,
        "sample_only": True,
        "limit": None,
        "resume": False,
        "workers": 3,
        "icon_mode": "image2",
        "vlm_timeout": 240,
        "vlm_retries": 2,
        "reference_library": None,
        "shard": None,
    }
    values.update(overrides)
    return Namespace(**values)


class CorpusCliPreviewTests(unittest.TestCase):
    def test_workers_do_not_turn_a_preview_into_a_paid_run(self):
        prepared = {"subset_id": "subset", "subset_dir": "/out/subset", "papers": 15, "sample_only": True}
        with patch("rfs.workflows.supervised_batch.run_corpus_batch", return_value=prepared) as batch, patch("rfs.cli.subprocess.Popen") as popen:
            result = _run_corpus_shards(corpus_args())
        popen.assert_not_called()
        self.assertTrue(result["sample_only"])
        self.assertTrue(batch.call_args.kwargs["sample_only"])

    def test_one_process_per_shard_when_the_caller_really_wants_the_run(self):
        prepared = {"subset_id": "subset", "subset_dir": "/out/subset", "papers": 15}
        with patch("rfs.workflows.supervised_batch.run_corpus_batch", return_value=prepared) as batch, patch("rfs.workflows.supervised_batch.merge_shard_ledgers", return_value={"batch_id": "subset"}), patch("rfs.cli.subprocess.Popen") as popen:
            popen.return_value.wait.return_value = 0
            result = _run_corpus_shards(corpus_args(sample_only=False))
        started = [call.args[0] for call in popen.call_args_list]
        self.assertEqual([argv[argv.index("--shard") + 1] for argv in started], ["0/3", "1/3", "2/3"])
        self.assertTrue(all(argv[argv.index("--workers") + 1] == "1" for argv in started))
        self.assertEqual(result["worker_exit_codes"], [0, 0, 0])
        self.assertTrue(batch.call_args.kwargs["sample_only"])

    def test_the_archive_root_reaches_every_worker(self):
        # The workers are what resolve each paper to its file, so a root the
        # parent knows about but the children never hear about is a root that
        # does nothing.
        prepared = {"subset_id": "subset", "subset_dir": "/out/subset", "papers": 15}
        with patch("rfs.workflows.supervised_batch.run_corpus_batch", return_value=prepared), patch("rfs.workflows.supervised_batch.merge_shard_ledgers", return_value={"batch_id": "subset"}), patch("rfs.cli.subprocess.Popen") as popen:
            popen.return_value.wait.return_value = 0
            _run_corpus_shards(corpus_args(sample_only=False, archive_root="/archive"))
        started = [call.args[0] for call in popen.call_args_list]
        self.assertTrue(started)
        self.assertTrue(all(argv[argv.index("--archive-root") + 1] == "/archive" for argv in started))


    def test_the_roots_that_already_hold_deliveries_reach_parent_and_children(self):
        # The parent resolves the paper list and the children re-resolve it; a
        # root only the parent knows about would let every worker buy the papers
        # the parent had just decided to skip.
        prepared = {"subset_id": "subset", "subset_dir": "/out/subset", "papers": 15, "already_drawn": 4}
        with patch("rfs.workflows.supervised_batch.run_corpus_batch", return_value=prepared) as batch, patch("rfs.workflows.supervised_batch.merge_shard_ledgers", return_value={"batch_id": "subset"}), patch("rfs.cli.subprocess.Popen") as popen:
            popen.return_value.wait.return_value = 0
            _run_corpus_shards(corpus_args(sample_only=False, workers=2, exclude_drawn_root=["/old", "/older"]))
        self.assertEqual(list(batch.call_args.kwargs["exclude_roots"]), ["/old", "/older"])
        started = [call.args[0] for call in popen.call_args_list]
        self.assertTrue(started)
        for argv in started:
            self.assertEqual([argv[index + 1] for index, item in enumerate(argv) if item == "--exclude-drawn-root"], ["/old", "/older"])


if __name__ == "__main__":
    unittest.main()
