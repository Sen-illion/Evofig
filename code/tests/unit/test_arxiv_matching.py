"""Tests for cache-first canonical arXiv candidate matching."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfs.judge_training.arxiv_matching import match_arxiv_candidates
from rfs.providers.literature import ProviderRequestError


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ArxivMatchingTests(unittest.TestCase):
    def test_cache_union_matches_exact_and_preserves_durable_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            manifests = batch / "manifests"
            manifests.mkdir(parents=True)
            (manifests / "arxiv_harvest_agent_2026_2026_20.json").write_text(
                json.dumps(
                    [
                        {"arxiv_id": "2601.00001", "title": "Cached Exact"},
                        {"arxiv_id": "2601.00002", "title": "Ambiguous"},
                    ]
                ),
                encoding="utf-8",
            )
            (manifests / "arxiv_harvest_cv_2026_2026_20.json").write_text(
                json.dumps(
                    [
                        {"arxiv_id": "2601.00003", "title": "Cross Domain Cache"},
                        {"arxiv_id": "2601.00004", "title": "Ambiguous"},
                    ]
                ),
                encoding="utf-8",
            )
            candidates = [
                {"paper_id": "one", "title": "Cached Exact", "year": 2026},
                {"paper_id": "two", "title": "Cross Domain Cache", "year": 2026},
                {"paper_id": "three", "title": "Ambiguous", "year": 2026},
                {"paper_id": "durable", "title": "Not In Cache", "year": 2026},
            ]
            _write_jsonl(batch / "candidates" / "agent" / "papers.jsonl", candidates)
            _write_jsonl(
                batch / "verified" / "agent" / "papers.jsonl",
                [
                    {
                        **candidates[-1],
                        "arxiv_match_status": "matched",
                        "arxiv_id": "2509.99999",
                        "source_url": "https://export.arxiv.org/e-print/2509.99999",
                    }
                ],
            )

            report = match_arxiv_candidates(
                batch,
                year=2026,
                domains=["agent"],
                direct_tail_per_domain=0,
            )

            rows = [
                json.loads(line)
                for line in (batch / "candidates" / "agent" / "papers_arxiv_matched.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            by_id = {row["paper_id"]: row for row in rows}
            self.assertTrue(report["ok"])
            self.assertEqual(report["cache_unique_arxiv_ids"], 4)
            self.assertEqual(report["domains"]["agent"]["matched"], 3)
            self.assertEqual(by_id["two"]["arxiv_id"], "2601.00003")
            self.assertEqual(by_id["durable"]["arxiv_id"], "2509.99999")
            self.assertEqual(by_id["three"]["arxiv_match_method"], "ambiguous_title_cache")

    def test_refused_export_api_hands_the_tail_to_the_live_resolver(self):
        """A refused export query must not abandon the tail; the resolver carries it."""
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            manifests = batch / "manifests"
            manifests.mkdir(parents=True)
            (manifests / "arxiv_harvest_agent_2026_2026_20.json").write_text(
                json.dumps([{"arxiv_id": "2601.00001", "title": "Cached Exact"}]),
                encoding="utf-8",
            )
            _write_jsonl(
                batch / "candidates" / "agent" / "papers.jsonl",
                [
                    {"paper_id": "cached", "title": "Cached Exact", "year": 2026},
                    {
                        "paper_id": "live",
                        "title": "Live Resolved",
                        "year": 2026,
                        "official_record_status": "verified",
                    },
                ],
            )

            def refused(*args, **kwargs):
                raise ProviderRequestError("arxiv", "http_406", "406 Client Error: Not Acceptable")

            record = {
                "arxiv_id": "2502.10835",
                "title": "Live Resolved",
                "title_norm": "live resolved",
                "arxiv_url": "https://arxiv.org/abs/2502.10835",
                "source_url": "https://export.arxiv.org/e-print/2502.10835",
            }
            with patch("rfs.judge_training.arxiv_matching.batch_match_exact", refused), patch(
                "rfs.judge_training.arxiv_matching.ArxivTitleResolver"
            ) as resolver:
                resolver.return_value.resolve.return_value = (record, "exact_title_semantic_scholar")
                report = match_arxiv_candidates(
                    batch,
                    year=2026,
                    domains=["agent"],
                    direct_tail_per_domain=10,
                )

            rows = [
                json.loads(line)
                for line in (batch / "candidates" / "agent" / "papers_arxiv_matched.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            by_id = {row["paper_id"]: row for row in rows}
            self.assertEqual(by_id["live"]["arxiv_id"], "2502.10835")
            self.assertEqual(by_id["live"]["arxiv_match_method"], "exact_title_semantic_scholar")
            self.assertEqual(by_id["cached"]["arxiv_match_method"], "exact_title_cache")
            self.assertEqual(report["domains"]["agent"]["live_matched"], 1)
            self.assertEqual(report["domains"]["agent"]["live_methods"], {"exact_title_semantic_scholar": 1})
            self.assertEqual(report["domains"]["agent"]["matched"], 2)
            self.assertIn("406", report["domains"]["agent"]["direct_error"])
            resolver.return_value.resolve.assert_called_once_with("Live Resolved")

    def test_rate_limited_titles_do_not_end_the_pass(self):
        """Bursts of refusals must not abandon titles that no channel answered yet."""
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            manifests = batch / "manifests"
            manifests.mkdir(parents=True)
            (manifests / "arxiv_harvest_agent_2026_2026_20.json").write_text(
                json.dumps([{"arxiv_id": "2601.00001", "title": "Cached Exact"}]),
                encoding="utf-8",
            )
            _write_jsonl(
                batch / "candidates" / "agent" / "papers.jsonl",
                [
                    {"paper_id": "a", "title": "Refused One", "year": 2026, "official_record_status": "verified"},
                    {"paper_id": "b", "title": "Refused Two", "year": 2026, "official_record_status": "verified"},
                    {"paper_id": "c", "title": "Refused Three", "year": 2026, "official_record_status": "verified"},
                ],
            )

            def refused(*args, **kwargs):
                raise ProviderRequestError("arxiv", "http_406", "406 Client Error: Not Acceptable")

            with patch("rfs.judge_training.arxiv_matching.batch_match_exact", refused), patch(
                "rfs.judge_training.arxiv_matching.ArxivTitleResolver"
            ) as resolver:
                resolver.return_value.resolve.side_effect = ProviderRequestError(
                    "arxiv_title_resolver", "http_429", "too many requests"
                )
                report = match_arxiv_candidates(
                    batch,
                    year=2026,
                    domains=["agent"],
                    direct_tail_per_domain=10,
                )

            domain_report = report["domains"]["agent"]
            self.assertEqual(domain_report["live_attempted"], 3)
            self.assertEqual(domain_report["live_matched"], 0)
            self.assertFalse(domain_report["live_blocked"])
            self.assertEqual(domain_report["live_cooldowns"], 0)
            self.assertEqual(domain_report["direct_error"], "406 Client Error: Not Acceptable")

    def test_missing_cache_fails_without_touching_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            candidate = batch / "candidates" / "agent" / "papers.jsonl"
            _write_jsonl(candidate, [{"paper_id": "one", "title": "Paper", "year": 2026}])

            report = match_arxiv_candidates(batch, year=2026, domains=["agent"])

            self.assertFalse(report["ok"])
            self.assertFalse((candidate.parent / "papers_arxiv_matched.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
