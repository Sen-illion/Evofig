"""Tests for the non-destructive strict literature training view."""

from __future__ import annotations

import json
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from rfs.judge_training.literature_freeze import build_strict_training_view
from rfs.judge_training.strict_candidate_freeze import freeze_strict_candidates


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class LiteratureFreezeTests(unittest.TestCase):
    def _paper(self, batch: Path, domain: str, paper_id: str, tier: str) -> dict:
        figure_path = batch / "assets" / domain / paper_id / "figure.png"
        text_path = batch / "assets" / domain / paper_id / "text" / "full.txt"
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        figure_path.write_bytes(b"png")
        text_path.write_text("paper text", encoding="utf-8")
        tex_path = figure_path.parent / "main.tex"
        tex_path.write_text("\\includegraphics{figure.png}", encoding="utf-8")
        figure = {
            "caption": "Overview of the proposed workflow",
            "image_paths": [str(figure_path.relative_to(batch))],
            "binding_status": "verified",
        }
        if tier == "A_source_verified":
            figure.update(
                {
                    "tex_file": str(tex_path.relative_to(batch)),
                    "source_files": [str(figure_path.relative_to(batch))],
                    "pdf_page": 2,
                }
            )
        else:
            figure["asset_source_url"] = "https://publisher.test/figure.png"
        return {
            "domain": domain,
            "paper_id": paper_id,
            "title": f"Paper {paper_id}",
            "official_url": f"https://publisher.test/{paper_id}",
            "official_record_status": "verified",
            "paper_type": "research_article" if domain in {"biology", "materials"} else "full",
            "figure_asset_tier": tier,
            "standalone_asset": tier == "B_official_standalone",
            "figure": figure,
            "full_text_path": str(text_path.relative_to(batch)),
            "record_origin": "incremental_v002",
        }

    def test_strict_view_accepts_a_and_b_but_excludes_c(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            a = self._paper(batch, "agent", "a", "A_source_verified")
            b = self._paper(batch, "biology", "b", "B_official_standalone")
            c = self._paper(batch, "materials", "c", "C_pdf_object_extracted")
            for domain, rows in {"agent": [a], "biology": [b], "materials": [c]}.items():
                write_rows(batch / "verified" / domain / "papers.jsonl", rows)

            report = build_strict_training_view(batch, target_per_domain=1)

            strict = [json.loads(line) for line in (batch / "verified" / "strict_training_papers.jsonl").read_text().splitlines()]
            retry = [json.loads(line) for line in (batch / "retry" / "c_asset_revalidation.jsonl").read_text().splitlines()]
            self.assertEqual({row["paper_id"] for row in strict}, {"a", "b"})
            self.assertEqual([row["paper_id"] for row in retry], ["c"])
            self.assertFalse(report["new_c_tier_allowed"])

    def test_cross_domain_paper_counts_once_and_keeps_its_membership(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            write_rows(
                batch / "verified" / "agent" / "papers.jsonl",
                [self._paper(batch, "agent", "shared", "A_source_verified")],
            )
            write_rows(
                batch / "verified" / "rag" / "papers.jsonl",
                [self._paper(batch, "rag", "shared", "A_source_verified")],
            )

            report = build_strict_training_view(batch, target_per_domain=1)

            self.assertEqual(report["strict_training_total"], 1)
            self.assertEqual(report["strict_membership_total"], 2)
            self.assertEqual(report["cross_domain_papers"], 1)
            self.assertEqual(report["domains"]["agent"]["strict"], 1)
            self.assertEqual(report["domains"]["agent"]["strict_membership"], 1)
            self.assertEqual(report["domains"]["agent"]["asset_tiers"], {"A_source_verified": 1})
            self.assertEqual(report["domains"]["rag"]["strict"], 0)
            self.assertEqual(report["domains"]["rag"]["strict_membership"], 1)
            self.assertEqual(report["domains"]["rag"]["duplicates_merged"], 1)
            self.assertEqual(report["domains"]["rag"]["shortfall"], 1)
            self.assertFalse(report["ok"])
            self.assertEqual(
                sum(value["strict"] for value in report["domains"].values()),
                report["strict_training_total"],
            )

    def test_subscription_preview_is_excluded_until_complete_text_is_available(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            row = self._paper(batch, "materials", "preview", "B_official_standalone")
            text = batch / row["full_text_path"]
            text.write_text("Abstract and references. " * 500 + "This is a preview of subscription content.", encoding="utf-8")
            ledger = batch / "verified/materials/papers.jsonl"
            write_rows(ledger, [row])
            original = ledger.read_bytes()
            result = build_strict_training_view(batch)
            self.assertEqual(0, result["strict_training_total"])
            self.assertEqual(1, result["failure_reasons"]["fulltext_preview_only"])
            self.assertEqual(original, ledger.read_bytes())
            text.write_text("Complete paper with methods and results.", encoding="utf-8")
            self.assertEqual(1, build_strict_training_view(batch)["strict_training_total"])

    def test_duplicate_paper_becomes_one_multidomain_record(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            agent = self._paper(batch, "agent", "shared", "A_source_verified")
            rag = self._paper(batch, "rag", "shared", "A_source_verified")
            write_rows(batch / "verified" / "agent" / "papers.jsonl", [agent])
            write_rows(batch / "verified" / "rag" / "papers.jsonl", [rag])

            build_strict_training_view(batch, target_per_domain=1)

            rows = [json.loads(line) for line in (batch / "verified" / "strict_training_papers.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["domain_labels"], ["agent", "rag"])

    def test_domain_ledger_backups_do_not_overwrite_each_other(self):
        from rfs.judge_training.literature_freeze import _publish_with_backup

        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            backup = batch / "backups" / "stamp"
            for domain in ("agent", "biology"):
                ledger = batch / "verified" / domain / "papers.jsonl"
                ledger.parent.mkdir(parents=True, exist_ok=True)
                ledger.write_text(f"{domain}-before\n", encoding="utf-8")
                _publish_with_backup(ledger, f"{domain}-after\n", backup)
            saved = sorted(path.name for path in backup.iterdir())
            self.assertEqual(saved, ["agent.papers.jsonl", "biology.papers.jsonl"])
            self.assertEqual(
                (backup / "agent.papers.jsonl").read_text(encoding="utf-8"), "agent-before\n"
            )

    def test_second_run_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            paper = self._paper(batch, "agent", "a", "A_source_verified")
            write_rows(batch / "verified" / "agent" / "papers.jsonl", [paper])
            build_strict_training_view(batch, target_per_domain=1)
            second = build_strict_training_view(batch, target_per_domain=1)
            self.assertFalse(second["outputs_changed"])

    def _acm_paper(self, batch: Path, paper_id: str) -> dict:
        """Return an A-tier ACM record whose own identifiers imply Crossref evidence."""
        paper = self._paper(batch, "agent", paper_id, "A_source_verified")
        paper.update(
            {
                "venue": "WWW",
                "doi": "10.1145/3774904.3792371",
                "official_url": "https://doi.org/10.1145/3774904.3792371",
                "official_source": "acm_proceedings",
                "paper_type": "full",
                "paper_type_evidence": {
                    "crossref_type": "proceedings-article",
                    "crossref_container": "Proceedings of the ACM Web Conference 2026",
                    "page_span": 12,
                },
                "official_record_evidence": {
                    "method": "crossref_acm_main_proceedings",
                    "doi": "10.1145/3774904.3792371",
                    "container": "Proceedings of the ACM Web Conference 2026",
                    "page_span": 12,
                    "checked_at": "2026-09-14T14:10:34Z",
                },
            }
        )
        return paper

    def test_acm_evidence_rebuilt_from_the_row_identifiers_is_official_proof(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            write_rows(batch / "verified" / "agent" / "papers.jsonl", [self._acm_paper(batch, "acm-main")])

            report = build_strict_training_view(batch, target_per_domain=1)

            self.assertEqual(report["strict_training_total"], 1)
            self.assertNotIn("official_publisher_proof_missing", report["failure_reasons"])

    def test_acm_evidence_must_match_the_row_identifiers_exactly(self):
        complete = {
            "method": "crossref_acm_main_proceedings",
            "doi": "10.1145/3774904.3792371",
            "container": "Proceedings of the ACM Web Conference 2026",
            "page_span": 12,
        }
        forged = {
            "method only": {"method": "crossref_acm_main_proceedings"},
            "wrong container": {**complete, "container": "Proceedings of the ACM Web Conference 2025"},
            "short page span": {**complete, "page_span": 3},
            "wrong doi": {**complete, "doi": "10.1145/0000000.0000000"},
            "unrelated method": {**complete, "method": "publisher_landing_page"},
        }
        for label, recorded in forged.items():
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as folder:
                    batch = Path(folder)
                    paper = self._acm_paper(batch, "acm-forged")
                    paper["official_record_evidence"] = recorded
                    write_rows(batch / "verified" / "agent" / "papers.jsonl", [paper])

                    report = build_strict_training_view(batch, target_per_domain=1)

                    self.assertEqual(report["strict_training_total"], 0)
                    self.assertEqual(
                        report["domains"]["agent"]["failure_reasons"]["official_publisher_proof_missing"], 1
                    )

    def test_acm_evidence_is_refused_outside_the_main_proceedings_whitelist(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            paper = self._acm_paper(batch, "acm-companion")
            paper["paper_type_evidence"]["crossref_container"] = "Companion Proceedings of the ACM Web Conference 2026"
            paper["official_record_evidence"]["container"] = "Companion Proceedings of the ACM Web Conference 2026"
            write_rows(batch / "verified" / "agent" / "papers.jsonl", [paper])

            report = build_strict_training_view(batch, target_per_domain=1)

            self.assertEqual(report["strict_training_total"], 0)
            self.assertEqual(report["domains"]["agent"]["failure_reasons"]["official_publisher_proof_missing"], 1)

    def test_public_asset_layout_supplies_the_complete_text(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            paper = self._paper(batch, "agent", "public-layout", "A_source_verified")
            paper["doi"] = "10.1145/3770854.3785678"
            paper.pop("full_text_path")
            public = batch / "assets/agent/public/10.1145_3770854.3785678/text/full.txt"
            public.parent.mkdir(parents=True)
            public.write_text("Complete paper text", encoding="utf-8")
            write_rows(batch / "verified" / "agent" / "papers.jsonl", [paper])

            report = build_strict_training_view(batch, target_per_domain=1)

            self.assertEqual(report["strict_training_total"], 1)
            self.assertNotIn("fulltext_missing", report["failure_reasons"])

    def test_missing_text_is_still_reported_when_no_layout_has_it(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            paper = self._paper(batch, "agent", "no-text", "A_source_verified")
            paper["full_text_path"] = "assets/agent/no-text/text/absent.txt"
            write_rows(batch / "verified" / "agent" / "papers.jsonl", [paper])

            report = build_strict_training_view(batch, target_per_domain=1)

            self.assertEqual(report["strict_training_total"], 0)
            self.assertEqual(report["domains"]["agent"]["failure_reasons"]["fulltext_missing"], 1)

    def test_candidate_freeze_appends_once_with_text_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            paper = {
                "domain": "agent",
                "paper_id": "paper-2026",
                "arxiv_id": "2601.00001",
                "title": "Agent framework",
                "year": 2026,
                "official_url": "https://aclanthology.org/2026.agent-long.1/",
                "official_source": "acl_anthology",
                "official_record_status": "verified",
                "paper_type": "long",
            }
            write_rows(batch / "candidates" / "agent" / "papers_arxiv_matched.jsonl", [paper])
            asset_root = batch / "assets" / "agent" / "2601.00001"
            (asset_root / "source").mkdir(parents=True)
            (asset_root / "figures").mkdir(parents=True)
            (asset_root / "text").mkdir(parents=True)
            image = asset_root / "figures" / "framework.png"
            text = asset_root / "text" / "full.txt"
            image.write_bytes(b"png")
            text.write_text("full paper text", encoding="utf-8")
            archive_path = asset_root / "arxiv_source.src"
            with tarfile.open(archive_path, mode="w:gz") as archive:
                for name, content in {
                    "main.tex": b"\\includegraphics{figures/framework.png}",
                    "figures/framework.png": b"source png",
                }.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))
            source_manifest = {
                "domains": {
                    "agent": {
                        "rows": [
                            {
                                "arxiv_id": "2601.00001",
                                "framework_candidates": [
                                    {
                                        "caption": "Overview of the agent framework",
                                        "tex_file": "main.tex",
                                        "resolved_files": ["figures/framework.png"],
                                        "pdf_page": 2,
                                        "pdf_caption_bound": True,
                                    }
                                ],
                                "extracted_figure_paths": [str(image.relative_to(batch))],
                            }
                        ]
                    }
                }
            }
            (batch / "manifests").mkdir(parents=True)
            (batch / "manifests" / "source_probe.json").write_text(json.dumps(source_manifest), encoding="utf-8")
            (batch / "manifests" / "fulltext_probe.json").write_text(
                json.dumps({"domains": {"agent": {"rows": [{"arxiv_id": "2601.00001", "text_status": "extracted"}]}}}),
                encoding="utf-8",
            )

            first = freeze_strict_candidates(batch, domains=["agent"], target_per_domain=1)
            second = freeze_strict_candidates(batch, domains=["agent"], target_per_domain=1)
            rows = [json.loads(line) for line in (batch / "verified" / "agent" / "papers.jsonl").read_text().splitlines()]

            self.assertEqual(first["added_by_domain"]["agent"], 1)
            self.assertEqual(second["added_by_domain"]["agent"], 0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["fulltext_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
