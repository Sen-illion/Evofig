"""Tests for the OpenAlex arXiv exact-title harvest helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "extra-dataset" / "scripts" / "harvest_openalex_arxiv_exact.py"
SPEC = importlib.util.spec_from_file_location("openalex_arxiv_harvest", SCRIPT)
harvester = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(harvester)


class OpenAlexArxivHarvestTests(unittest.TestCase):
    def test_request_budget_preserves_server_reserve(self):
        budget = harvester.RequestBudget(max_requests=5, reserve_requests=10)
        budget.claim()
        budget.observe(SimpleNamespace(headers={"x-ratelimit-remaining": "11"}))
        budget.claim()
        budget.observe(SimpleNamespace(headers={"x-ratelimit-remaining": "10"}))

        with self.assertRaises(harvester.RequestBudgetExhausted):
            budget.claim()

        self.assertEqual(budget.snapshot()["requests_used"], 2)
        self.assertEqual(budget.snapshot()["stop_reason"], "server_quota_reserve_reached")

    def test_build_exact_records_rejects_ambiguous_titles(self):
        works = [
            {
                "title": "Exact Framework",
                "locations": [{"landing_page_url": "https://arxiv.org/abs/2601.00001"}],
            },
            {
                "title": "Exact Framework",
                "locations": [{"landing_page_url": "https://arxiv.org/abs/2601.00002"}],
            },
            {
                "title": "Unique Workflow",
                "doi": "https://doi.org/10.48550/arxiv.2602.00003",
                "locations": [],
            },
        ]
        records, ambiguous = harvester.build_exact_records(works)
        self.assertEqual(ambiguous, 1)
        self.assertEqual([record["arxiv_id"] for record in records], ["2602.00003"])

    def test_merge_exact_records_preserves_existing_and_rejects_conflicts(self):
        existing = [
            {"title": "Existing Paper", "title_norm": "existing paper", "arxiv_id": "2601.00001"},
            {"title": "Conflict", "title_norm": "conflict", "arxiv_id": "2601.00002"},
        ]
        discovered = [
            {"title": "New Paper", "title_norm": "new paper", "arxiv_id": "2602.00003"},
            {"title": "Conflict", "title_norm": "conflict", "arxiv_id": "2602.00004"},
        ]

        records, ambiguous = harvester.merge_exact_records(existing, discovered)

        self.assertEqual(ambiguous, 1)
        self.assertEqual(
            [record["arxiv_id"] for record in records],
            ["2601.00001", "2602.00003"],
        )


if __name__ == "__main__":
    unittest.main()
