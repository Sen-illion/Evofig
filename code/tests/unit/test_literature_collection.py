"""Tests for the canonical strict literature collection command."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rfs.cli import build_parser
from rfs.judge_training.literature_collection import (
    DBLP_VENUES,
    collect_literature_papers,
    merge_staged_inventory,
    prepare_canary_batch,
)
from rfs.judge_training.source_snapshot_boundary import collection_boundary


class LiteratureCollectionTests(unittest.TestCase):
    def test_collect_papers_cli_contract(self):
        args = build_parser().parse_args(
            [
                "judge-data", "collect-papers", "--batch-root", "batch", "--year", "2026",
                "--stage", "freeze", "--domains", "agent", "rag", "--resume",
            ]
        )
        self.assertEqual(args.judge_data_action, "collect-papers")
        self.assertEqual(args.domains, ["agent", "rag"])
        self.assertTrue(args.resume)

    def test_staging_merge_is_append_only_and_canary_limited(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder) / "batch"
            staging = Path(folder) / "staging"
            main = batch / "candidates" / "agent" / "papers.jsonl"
            staged = staging / "candidates" / "agent" / "papers.jsonl"
            main.parent.mkdir(parents=True)
            staged.parent.mkdir(parents=True)
            main.write_text(json.dumps({"paper_id": "old", "title": "Old", "year": 2026}) + "\n")
            rows = [
                {"paper_id": f"new-{index}", "title": f"New {index}", "year": 2026, "official_source": "acl_anthology"}
                for index in range(3)
            ]
            rows.append(
                {
                    "paper_id": "future",
                    "title": "Future paper",
                    "year": 2026,
                    "publication_date": "2026-12-31",
                    "official_source": "crossref_journal",
                }
            )
            staged.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = merge_staged_inventory(batch, staging, year=2026, domains=["agent"], canary_per_source=2)
            self.assertEqual(report["domains"]["agent"]["added"], 2)
            self.assertEqual(len(main.read_text().splitlines()), 3)

    def test_freeze_stage_routes_to_strict_freezer(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "rfs.judge_training.literature_collection.freeze_strict_candidates",
            return_value={"ok": True, "strict_training": {"strict_training_total": 4}},
        ) as freeze:
            result = collect_literature_papers(folder, year=2026, stage="freeze", domains=["agent"])
        self.assertTrue(result["ok"])
        freeze.assert_called_once()

    def test_collection_rejects_year_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                collect_literature_papers(folder, year=2025, stage="freeze")

    def test_freeze_below_quota_completes_batch_boundary(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "rfs.judge_training.literature_collection.freeze_strict_candidates",
            return_value={"ok": True, "strict_training": {"ok": False, "strict_training_total": 4}},
        ):
            batch = Path(folder)
            events = batch / "manifests/multisource_collection_events.jsonl"
            events.parent.mkdir(parents=True)
            events.write_text("".join(json.dumps(event) + "\n" for event in (
                {"stage": "asset-bind", "status": "started"},
                {"stage": "asset-bind", "status": "completed"},
                {"stage": "fulltext", "status": "completed"},
            )), encoding="utf-8")
            result = collect_literature_papers(batch, year=2026, stage="freeze")
            self.assertTrue(result["ok"])
            self.assertFalse(result["stages"]["freeze"]["strict_training"]["ok"])
            self.assertEqual("completed", json.loads(events.read_text(encoding="utf-8").splitlines()[-1])["status"])
            self.assertTrue(collection_boundary(batch)["ready"])

    def test_freeze_execution_failure_remains_failed(self):
        for failure in ({"ok": False}, RuntimeError("strict publication failed")):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder, patch(
                "rfs.judge_training.literature_collection.freeze_strict_candidates",
            ) as freeze:
                if isinstance(failure, Exception):
                    freeze.side_effect = failure
                else:
                    freeze.return_value = failure
                result = collect_literature_papers(folder, year=2026, stage="freeze")
                self.assertFalse(result["ok"])
                events = Path(folder) / "manifests/multisource_collection_events.jsonl"
                self.assertEqual("failed", json.loads(events.read_text(encoding="utf-8").splitlines()[-1])["status"])

    def test_all_overlaps_post_match_stages(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "rfs.judge_training.literature_collection._run_worker",
            return_value={"ok": True},
        ), patch(
            "rfs.judge_training.literature_collection._collect_dblp",
            return_value={"status": "complete", "records": 0, "errors": []},
        ), patch(
            "rfs.judge_training.literature_collection.match_arxiv_candidates",
            return_value={"ok": True, "domains": {}},
        ), patch(
            "rfs.judge_training.literature_collection.bind_official_assets",
            return_value={"ok": True, "domains": {}},
        ), patch(
            "rfs.judge_training.literature_collection.freeze_strict_candidates",
            return_value={"ok": True},
        ):
            result = collect_literature_papers(folder, year=2026, stage="all", domains=["agent"])

        self.assertTrue(result["ok"])
        self.assertIn("asset-bind", result["stages"])
        self.assertIn("fulltext", result["stages"])

    def test_post_inventory_canary_uses_isolated_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder) / "batch"
            raw = batch / "raw" / "acl_anthology" / "agent" / "2026" / "papers.jsonl"
            raw.parent.mkdir(parents=True)
            raw.write_text(
                "".join(
                    json.dumps({"paper_id": f"p-{index}", "title": f"Paper {index}", "year": 2026}) + "\n"
                    for index in range(3)
                ),
                encoding="utf-8",
            )
            cache = batch / "manifests" / "arxiv_harvest_agent_2026_2026_20.json"
            cache.parent.mkdir(parents=True)
            cache.write_text('[{"arxiv_id":"2601.00001"}]', encoding="utf-8")

            canary = prepare_canary_batch(batch, year=2026, domains=["agent"], per_source=2)

            rows = (canary / "candidates" / "agent" / "papers.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertFalse((batch / "candidates" / "agent" / "papers.jsonl").exists())
            self.assertEqual((canary / "manifests" / cache.name).read_text(), cache.read_text())

    def test_arxiv_canary_uses_bounded_direct_tail(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "rfs.judge_training.literature_collection.match_arxiv_candidates",
            return_value={"ok": True},
        ) as matcher:
            batch = Path(folder)
            raw = batch / "raw" / "acl_anthology" / "agent" / "2026" / "papers.jsonl"
            raw.parent.mkdir(parents=True)
            raw.write_text(json.dumps({"paper_id": "p", "title": "Paper", "year": 2026}) + "\n")

            result = collect_literature_papers(
                batch,
                year=2026,
                stage="arxiv-match",
                domains=["agent"],
                canary_per_source=10,
            )

        self.assertTrue(result["canary_isolated"])
        self.assertEqual(matcher.call_args.kwargs["direct_tail_per_domain"], 10)
        self.assertIn("canary_2026", str(matcher.call_args.args[0]))

    def test_asset_canary_does_not_apply_a_second_domain_limit(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "rfs.judge_training.literature_collection._run_worker",
            return_value={"ok": True},
        ), patch(
            "rfs.judge_training.literature_collection.bind_official_assets",
            return_value={"ok": True, "domains": {}},
        ) as binder:
            batch = Path(folder)
            raw = batch / "raw" / "acl_anthology" / "agent" / "2026" / "papers.jsonl"
            raw.parent.mkdir(parents=True)
            raw.write_text(json.dumps({"paper_id": "p", "title": "Paper", "year": 2026}) + "\n")

            collect_literature_papers(
                batch,
                year=2026,
                stage="asset-bind",
                domains=["agent"],
                canary_per_source=10,
            )

        self.assertEqual(binder.call_args.kwargs["limit_per_domain"], 0)

    def test_resume_defers_recent_failed_dblp(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "rfs.judge_training.literature_collection._collect_dblp",
        ) as dblp:
            batch = Path(folder)
            manifest = batch / "work" / "official_inventory_2026" / "manifests" / "official_inventory.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"domains":{}}', encoding="utf-8")
            coverage = batch / "manifests" / "multisource_sources.json"
            coverage.parent.mkdir(parents=True)
            coverage.write_text(
                json.dumps(
                    {
                        "generated_at": "2999-01-01T00:00:00Z",
                        "sources": {"dblp": {"status": "failed"}},
                    }
                ),
                encoding="utf-8",
            )

            result = collect_literature_papers(
                batch,
                year=2026,
                stage="inventory",
                domains=["agent"],
                resume=True,
            )

        dblp.assert_not_called()
        self.assertTrue(result["stages"]["inventory"]["dblp"]["cooldown"])

    def test_dblp_venue_list_covers_the_acl_family(self):
        for domain in ("agent", "rag"):
            for venue in ("ACL", "EMNLP", "NAACL", "EACL", "IJCAI"):
                self.assertIn(venue, DBLP_VENUES[domain])


if __name__ == "__main__":
    unittest.main()
