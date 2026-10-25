"""Unit tests for the isolated independent-paper collection gates."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "extra-dataset" / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


collector = load("fresh_collector", "collect_official_fresh.py")
matcher = load("fresh_matcher", "match_fresh_arxiv.py")
probe = load("fresh_probe", "probe_fresh_arxiv_sources.py")
freeze = load("fresh_freeze", "freeze_fresh_verified.py")
verifier = load("fresh_verifier", "verify_fresh_official_landings.py")
monitor = load("fresh_monitor", "monitor_independent_10k.py")
incremental_common = load("incremental_common", "incremental_common.py")
sys.modules["incremental_common"] = incremental_common
public_probe = load("public_probe", "probe_public_figure_assets.py")
incremental_freeze = load("incremental_freeze", "freeze_incremental_10k.py")


class FreshCollectionTests(unittest.TestCase):
    def test_dedupe_and_month_priority(self):
        rows = [
            {"doi": "10/x", "title": "Old", "publication_date": "2025-12-01"},
            {"doi": "10/x", "title": "Duplicate", "publication_date": "2026-01-01"},
            {"doi": "10/y", "title": "New", "publication_date": "2026-02-01"},
        ]
        unique = collector.dedupe(rows)
        ordered = collector.prioritized(unique)
        self.assertEqual(len(unique), 2)
        self.assertEqual(ordered[0]["doi"], "10/y")

    def test_biology_query_excludes_cs_lg(self):
        self.assertNotIn("cs.LG", matcher.ARXIV_QUERY["biology"])
        self.assertIn("q-bio", matcher.ARXIV_QUERY["biology"])

    def test_safe_archive_member_paths(self):
        self.assertIsNone(probe.safe_name("../escape.tex"))
        self.assertIsNone(probe.safe_name("/absolute.tex"))
        self.assertEqual(probe.safe_name("fig/main.pdf"), "fig/main.pdf")

    def test_figure_caption_filters_and_binding(self):
        members = {
            "main.tex": br"\\begin{figure}\\caption{Overall framework pipeline}\\includegraphics{fig.pdf}\\end{figure}",
            "bad.tex": br"\\begin{figure}\\caption{ROC curve result comparison}\\includegraphics{roc.pdf}\\end{figure}",
            "fig.pdf": b"pdf",
            "roc.pdf": b"pdf",
        }
        result = probe.inspect_source(members)
        self.assertEqual(len(result["framework_candidates"]), 1)
        self.assertEqual(result["framework_candidates"][0]["resolved_files"], ["fig.pdf"])

    def test_source_probe_accepts_nested_illustration_caption(self):
        members = {
            "main.tex": br"\begin{figure}\caption{An illustration of \textbf{our proposed agent flow}}\includegraphics{flow.pdf}\end{figure}",
            "flow.pdf": b"pdf",
        }
        result = probe.inspect_source(members)
        self.assertEqual(len(result["framework_candidates"]), 1)
        self.assertIn("our proposed agent flow", result["framework_candidates"][0]["caption"])

    def test_source_probe_can_reinspect_cached_rows(self):
        rows = [{"arxiv_id": "2601.00001"}, {"arxiv_id": "2601.00002"}]
        previous = [{"arxiv_id": "2601.00001", "download_status": "cached"}]
        self.assertEqual(probe.pending_source_rows(rows, previous, False), [rows[1]])
        self.assertEqual(probe.pending_source_rows(rows, previous, True), rows)

    def test_deterministic_split(self):
        self.assertEqual(freeze.split("abc"), freeze.split("abc"))
        self.assertIn(freeze.split("abc"), {"train", "validation", "hidden_test"})

    def test_title_evidence(self):
        html = "<title>Deep retrieval framework for scientific agents</title>"
        self.assertTrue(verifier.title_evidence("Deep retrieval framework for scientific agents", html))

    def test_official_pdf_can_verify_blocked_landing(self):
        class PdfResponse:
            url = "https://publisher.example/paper.pdf"
            headers = {"content-type": "application/pdf"}
            content = b"%PDF fake"
            status_code = 200

            def raise_for_status(self):
                return None

        class LandingResponse:
            def raise_for_status(self):
                raise verifier.requests.HTTPError("403")

        class Session:
            def get(self, url, **kwargs):
                return PdfResponse() if url.endswith(".pdf") else LandingResponse()

        class Page:
            def get_text(self, kind):
                return "Zwitterionic polymer coatings for marine anti-fouling Materials Horizons"

        class Document(list):
            def close(self):
                return None

        paper = {
            "title": "Zwitterionic polymer coatings for marine anti-fouling",
            "venue": "Materials Horizons",
            "official_url": "https://doi.org/10.1000/example",
            "official_pdf_url": "https://publisher.example/paper.pdf",
        }
        with patch.object(verifier.fitz, "open", return_value=Document([Page()])):
            item = verifier.verify(Session(), paper)
        self.assertEqual(item["official_record_status"], "verified")
        self.assertEqual(item["official_pdf_evidence"]["method"], "official_pdf_text")

    def test_crossref_acm_main_proceedings_verifies_full_paper_without_network(self):
        paper = {
            "title": "A Full Retrieval Paper",
            "venue": "SIGIR",
            "paper_type": "full",
            "doi": "10.1145/123.456",
            "official_url": "https://doi.org/10.1145/123.456",
            "official_source": "acm_proceedings",
            "paper_type_evidence": {
                "crossref_type": "proceedings-article",
                "crossref_container": "Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval",
                "page_span": 10,
            },
        }
        session = Mock()
        item = verifier.verify(session, paper)
        self.assertEqual(item["official_record_status"], "verified")
        self.assertEqual(item["official_record_evidence"]["method"], "crossref_acm_main_proceedings")
        session.get.assert_not_called()

    def test_crossref_acm_fallback_rejects_companion_and_ictir(self):
        base = {
            "paper_type": "full",
            "doi": "10.1145/123.456",
            "official_url": "https://doi.org/10.1145/123.456",
            "official_source": "acm_proceedings",
            "paper_type_evidence": {"crossref_type": "proceedings-article", "page_span": 10},
        }
        companion = {
            **base,
            "venue": "WWW",
            "paper_type_evidence": {
                **base["paper_type_evidence"],
                "crossref_container": "Companion Proceedings of the ACM Web Conference 2026",
            },
        }
        ictir = {
            **base,
            "venue": "SIGIR",
            "paper_type_evidence": {
                **base["paper_type_evidence"],
                "crossref_container": "Proceedings of the 2026 International ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR)",
            },
        }
        self.assertIsNone(verifier.crossref_acm_main_proceedings_evidence(companion))
        self.assertIsNone(verifier.crossref_acm_main_proceedings_evidence(ictir))

    def test_crossref_acm_fallback_rejects_short_page_span(self):
        paper = {
            "venue": "KDD",
            "paper_type": "full",
            "doi": "10.1145/123.456",
            "official_url": "https://doi.org/10.1145/123.456",
            "official_source": "acm_proceedings",
            "paper_type_evidence": {
                "crossref_type": "proceedings-article",
                "crossref_container": "Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2",
                "page_span": 4,
            },
        }
        self.assertIsNone(verifier.crossref_acm_main_proceedings_evidence(paper))

    def test_registry_only_verification_skips_nonmatching_pending_rows(self):
        papers = [
            {
                "year": 2026,
                "official_record_status": "publisher_landing_pending",
                "venue": "WWW",
                "paper_type": "full",
                "doi": "10.1145/123.456",
                "official_url": "https://doi.org/10.1145/123.456",
                "official_source": "acm_proceedings",
                "paper_type_evidence": {
                    "crossref_type": "proceedings-article",
                    "crossref_container": "Proceedings of the ACM Web Conference 2026",
                    "page_span": 9,
                },
            },
            {"year": 2026, "official_record_status": "publisher_landing_pending", "venue": "AAAI"},
        ]
        selected = [
            index
            for index in verifier.scoped_pending_indexes(papers, 2026)[1]
            if verifier.crossref_acm_main_proceedings_evidence(papers[index]) is not None
        ]
        self.assertEqual(selected, [0])

    def test_official_verifier_year_scope_leaves_other_year_untouched(self):
        papers = [
            {"year": 2025, "official_record_status": "publisher_landing_pending"},
            {"year": 2026, "official_record_status": "publisher_landing_pending"},
            {"year": 2026, "official_record_status": "verified"},
        ]
        scoped, pending = verifier.scoped_pending_indexes(papers, 2026)
        self.assertEqual(scoped, [1, 2])
        self.assertEqual(pending, [1])

    def test_pmlr_is_official_conference_source(self):
        item = collector.row(domain="agent", title="Agent Planning", year=2025, venue="ICML", official_url="https://proceedings.mlr.press/v1/a.html", pdf_url="https://example/a.pdf", paper_id="pmlr:v1:a", source="pmlr", paper_type="full")
        self.assertEqual(item["venue_type"], "conference")
        self.assertEqual(item["official_record_status"], "verified")

    def test_aamas_official_research_track_is_verified(self):
        item = collector.row(domain="agent", title="A Multi-Agent Paper", year=2026, venue="AAMAS", official_url="https://www.ifaamas.org/Proceedings/aamas2026/pdfs/ABCD1234.pdf", pdf_url="https://www.ifaamas.org/Proceedings/aamas2026/pdfs/ABCD1234.pdf", paper_id="aamas:2026:ABCD1234", source="aamas_proceedings", paper_type="full")
        self.assertEqual(item["venue_type"], "conference")
        self.assertEqual(item["official_record_status"], "verified")

    def test_expanded_conference_sources_remain_pending_until_landing_check(self):
        item = collector.row(domain="rag", title="Retrieval planning for agents", year=2026, venue="SIGIR", official_url="https://doi.org/10.1145/example", pdf_url=None, paper_id="10.1145/example", source="acm_proceedings", paper_type="full")
        self.assertEqual(item["venue_type"], "conference")
        self.assertEqual(item["official_record_status"], "publisher_landing_pending")

    def test_short_markers_are_rejected_by_expanded_filter(self):
        self.assertIn("short paper", collector.NON_FULL_MARKERS)
        self.assertGreaterEqual(collector.page_span("123-131"), 7)
        self.assertLess(collector.page_span("123-125"), 7)

    def test_materials_harvests_its_nature_family_titles_through_europe_pmc(self):
        self.assertIn("materials", collector.EUROPE_PMC_DOMAINS)
        self.assertIn("Scientific Reports", collector.JOURNALS["materials"])
        self.assertIn("Communications Engineering", collector.JOURNALS["materials"])
        self.assertEqual(collector.MATERIALS_ISSN["Scientific Reports"], "2045-2322")

    def test_multidisciplinary_materials_title_still_needs_a_materials_term(self):
        self.assertTrue(collector.topic_selected("materials", "Graphene composite battery electrodes", "Scientific reports"))
        self.assertFalse(collector.topic_selected("materials", "Dietary fibre and gut transit time", "Scientific reports"))
        self.assertTrue(collector.topic_selected("materials", "Anything at all", "Nature Materials"))

    def test_acl_volume_type_reads_the_track_off_the_volume_name(self):
        self.assertEqual(collector.acl_volume_type("2026.eacl-long.123"), "long")
        self.assertEqual(collector.acl_volume_type("2026.naacl-long.7"), "long")
        self.assertEqual(collector.acl_volume_type("2026.emnlp-main.5"), "full")
        self.assertIsNone(collector.acl_volume_type("2026.acl-short.9"))
        self.assertIsNone(collector.acl_volume_type("2026.findings-acl.1"))

    def test_crossref_skips_the_europe_pmc_only_titles(self):
        requested: list[str] = []

        def fake_pages(_session, journal, _lower, _upper, _limit):
            requested.append(journal)
            return iter(())

        with patch.object(collector, "_crossref_pages", side_effect=fake_pages):
            collector.crossref(Mock(), "materials", 2026, 0)

        self.assertNotIn("Scientific Reports", requested)
        self.assertNotIn("Communications Engineering", requested)
        self.assertIn("Nature Communications", requested)

    def test_monitor_gives_new_stage_a_restart_grace_period(self):
        event = {"stage": "freeze", "status": "started", "at": datetime.now(timezone.utc).isoformat()}
        self.assertTrue(monitor.recent_started_stage([event]))

    def test_archive_member_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tar"
            with tarfile.open(path, "w") as archive:
                payload = b"x"
                info = tarfile.TarInfo("ok.tex")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            self.assertIn("ok.tex", probe.archive_members(path))

    def test_match_one_preserves_strict_title_gate(self):
        candidate = {"title": "A Retrieval Framework"}
        arxiv = [{"arxiv_id": "1234.5678", "title_norm": "a retrieval framework", "title": "A Retrieval Framework"}]
        exact, postings = matcher.build_title_index(arxiv)
        result, matched = matcher.match_one(candidate, arxiv, exact, postings, delay=0, skip_direct_search=True)
        self.assertTrue(matched)
        self.assertEqual(result["arxiv_match_status"], "matched")
        self.assertEqual(result["arxiv_id"], "1234.5678")

    def test_direct_title_search_many_batches_titles_and_reuses_strict_matcher(self):
        class Response:
            text = """
            <feed xmlns='http://www.w3.org/2005/Atom'>
              <entry><id>http://arxiv.org/abs/1234.5678v1</id><title>First Retrieval Method</title></entry>
              <entry><id>http://arxiv.org/abs/2345.6789v1</id><title>Second Retrieval Method</title></entry>
            </feed>
            """

        class Session:
            def __init__(self):
                self.headers = {}

            def get(self, url, params=None, timeout=None):
                self.params = params
                return Response()

        session = Session()
        with patch.object(matcher, "fetch", side_effect=lambda s, url, params: session.get(url, params=params)):
            result = matcher.direct_title_search_many(session, ["First Retrieval Method", "Second Retrieval Method"])
        self.assertEqual(result["first retrieval method"][0]["arxiv_id"], "1234.5678")
        self.assertEqual(result["second retrieval method"][0]["arxiv_id"], "2345.6789")

    def test_direct_exact_title_search_many_rejects_fuzzy_title(self):
        class Response:
            text = """
            <feed xmlns='http://www.w3.org/2005/Atom'>
              <entry><id>http://arxiv.org/abs/1234.5678v1</id><title>Retrieval Framework for Agents</title></entry>
            </feed>
            """

        with patch.object(matcher, "fetch", return_value=Response()):
            result = matcher.direct_exact_title_search_many(object(), ["Retrieval Framework"])
        self.assertIsNone(result["retrieval framework"])

    def test_direct_search_slot_waits_for_global_interval(self):
        previous = matcher._DIRECT_SEARCH_LAST_AT
        matcher._DIRECT_SEARCH_LAST_AT = 10.0
        try:
            with patch.object(matcher.time, "monotonic", side_effect=[11.0, 13.0]), patch.object(
                matcher.time, "sleep"
            ) as sleep:
                matcher.wait_for_direct_search_slot(3.0)
        finally:
            matcher._DIRECT_SEARCH_LAST_AT = previous
        sleep.assert_called_once_with(2.0)

    def test_openalex_exact_title_search_requires_exact_title_and_arxiv_id(self):
        class Response:
            def json(self):
                return {
                    "results": [
                        {
                            "title": "Thermodynamic assessment of machine learning models",
                            "doi": "https://doi.org/10.48550/arxiv.2602.04075",
                            "ids": {},
                            "locations": [],
                        },
                        {
                            "title": "Thermodynamic assessment of machine learning models extended",
                            "doi": "https://doi.org/10.48550/arxiv.2602.99999",
                            "ids": {},
                            "locations": [],
                        },
                    ]
                }

        with patch.object(matcher, "fetch", return_value=Response()), patch.object(
            matcher, "wait_for_openalex_slot"
        ):
            result = matcher.openalex_exact_title_search(
                "Thermodynamic assessment of machine learning models"
            )
        self.assertEqual(result["arxiv_id"], "2602.04075")
        self.assertEqual(result["source_url"], "https://export.arxiv.org/e-print/2602.04075")

    def test_match_one_does_not_accept_ambiguous_fuzzy_match(self):
        candidate = {"title": "Retrieval Framework"}
        arxiv = [
            {"arxiv_id": "1", "title_norm": "retrieval framework for agents", "title": "Retrieval Framework for Agents"},
            {"arxiv_id": "2", "title_norm": "retrieval framework for rag", "title": "Retrieval Framework for RAG"},
        ]
        exact, postings = matcher.build_title_index(arxiv)
        result, matched = matcher.match_one(candidate, arxiv, exact, postings, delay=0, skip_direct_search=True)
        self.assertFalse(matched)
        self.assertEqual(result["arxiv_match_status"], "unmatched")

    def test_arxiv_matcher_scoped_write_preserves_other_years(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "matched.jsonl"
            matcher.write_jsonl(path, [{"paper_id": "old", "year": 2025, "arxiv_match_status": "matched"}])
            matcher.write_scoped_matches(
                path,
                [{"paper_id": "new", "year": 2026, "arxiv_match_status": "matched"}],
                2026,
            )
            rows = matcher.jsonl(path)
            self.assertEqual([row["paper_id"] for row in rows], ["old", "new"])

    def test_arxiv_matcher_runs_domains_concurrently(self):
        active = 0
        maximum_active = 0
        started = 0
        lock = threading.Lock()
        all_started = threading.Event()

        def fake_process_domain(batch, domain, args):
            nonlocal active, maximum_active, started
            with lock:
                active += 1
                started += 1
                maximum_active = max(maximum_active, active)
                if started == 3:
                    all_started.set()
            all_started.wait(timeout=1)
            time.sleep(0.02)
            with lock:
                active -= 1
            return domain, {"candidate_count": 0, "matched": 0}

        with tempfile.TemporaryDirectory() as directory:
            argv = [
                "match_fresh_arxiv.py",
                "--batch-root",
                directory,
                "--domains",
                "agent",
                "rag",
                "cv",
                "--domain-workers",
                "3",
            ]
            with patch.object(matcher, "process_domain", side_effect=fake_process_domain), patch("sys.argv", argv):
                self.assertEqual(matcher.main(), 0)
        self.assertEqual(maximum_active, 3)

    def test_freeze_evaluate_paper_is_side_effect_free(self):
        paper = {
            "title": "A Full Framework",
            "official_url": "https://example.org/paper",
            "paper_type": "full",
            "official_record_status": "verified",
            "official_source": "acl_anthology",
            "arxiv_match_status": "matched",
            "arxiv_id": "1234.5678",
        }
        source = {"download_status": "cached", "framework_source_bound": True, "framework_candidates": [{"resolved_files": ["fig.pdf"], "pdf_caption_bound": True}], "extracted_figure_paths": ["assets/a/fig.pdf"]}
        text = {"text_status": "extracted"}
        result, failures, qualifying, keys, identity = freeze.evaluate_paper("agent", paper, {"1234.5678": source}, {"1234.5678": text})
        self.assertEqual(failures, [])
        self.assertEqual(qualifying[0]["resolved_files"], ["fig.pdf"])
        self.assertIn("1234.5678", keys)
        self.assertEqual(identity, "1234.5678")

    def test_incremental_identity_keys_include_external_versions(self):
        keys = incremental_common.identity_keys({"doi": "10.1000/X", "pmcid": "PMC123", "title": "A Framework"})
        self.assertIn("doi:10.1000/x", keys)
        self.assertIn("pmcid:pmc123", keys)
        self.assertIn("title:a framework", keys)

    def test_public_figure_policy_requires_standalone_embedded_asset(self):
        self.assertTrue(public_probe.FIGURE_TERMS.search("Figure 1. Overview of the proposed workflow"))
        self.assertTrue(public_probe.EXCLUDED_TERMS.search("Figure 2. ROC curve result comparison"))
        self.assertEqual(public_probe.page_captions("Figure 1. Overview of the proposed workflow"), ["Figure 1. Overview of the proposed workflow"])

    def test_public_probe_resolves_official_landing_pdf_links(self):
        class Response:
            url = "https://dl.acm.org/doi/10.1145/123.456"
            headers = {"content-type": "text/html"}
            content = b"<html></html>"
            text = '<meta name="citation_pdf_url" content="/doi/pdf/10.1145/123.456">'

            def raise_for_status(self):
                return None

        class Session:
            def get(self, *args, **kwargs):
                return Response()

        url, method = public_probe.resolve_official_pdf(Session(), {"official_url": "https://doi.org/10.1145/123.456"}, {})
        self.assertEqual(url, "https://dl.acm.org/doi/pdf/10.1145/123.456")
        self.assertEqual(method, "landing_html_pdf_link")

    def test_public_probe_uses_openreview_pdf_fallback(self):
        class Response:
            url = "https://openreview.net/forum?id=abc123"
            headers = {"content-type": "text/html"}
            content = b"<html></html>"
            text = "<html>No citation metadata</html>"

            def raise_for_status(self):
                return None

        class Session:
            def get(self, *args, **kwargs):
                return Response()

        url, method = public_probe.resolve_official_pdf(Session(), {"official_url": "https://openreview.net/forum?id=abc123"}, {})
        self.assertEqual(url, "https://openreview.net/pdf?id=abc123")
        self.assertEqual(method, "landing_html_pdf_link")

    def test_public_probe_finds_ojs_article_galley_pdf(self):
        class Response:
            url = "https://ojs.aaai.org/index.php/AAAI/article/view/40232"
            headers = {"content-type": "text/html"}
            content = b"<html></html>"
            text = '<a href="/index.php/AAAI/article/view/40232/44193">PDF</a>'

            def raise_for_status(self):
                return None

        class Session:
            def get(self, *args, **kwargs):
                return Response()

        url, method = public_probe.resolve_official_pdf(
            Session(), {"official_url": "https://ojs.aaai.org/index.php/AAAI/article/view/40232"}, {}
        )
        self.assertEqual(url, "https://ojs.aaai.org/index.php/AAAI/article/view/40232/44193")
        self.assertEqual(method, "landing_html_pdf_link")

    def test_incremental_freeze_accepts_public_standalone_asset_without_arxiv(self):
        paper = {
            "title": "A Public Workflow",
            "paper_id": "doi:10.1000/public",
            "official_url": "https://example.org/paper",
            "official_record_status": "verified",
            "paper_type": "research_article",
        }
        public = {
            "standalone_asset": True,
            "figure_asset_tier": "C_pdf_object_extracted",
            "framework_candidates": [{"caption": "Figure 1. Overview workflow", "asset_path": "assets/x.png", "pdf_page": 2}],
            "fulltext_status": "extracted",
        }
        result, failures, figure = incremental_freeze.evaluate("materials", paper, {}, public, {})
        self.assertEqual(failures, [])
        self.assertEqual(figure["figure_asset_tier"], "C_pdf_object_extracted")

    def test_incremental_freeze_uses_public_asset_when_arxiv_source_is_unbound(self):
        paper = {
            "title": "A Public Workflow With Arxiv",
            "paper_id": "doi:10.1000/public-arxiv",
            "official_url": "https://example.org/paper",
            "official_record_status": "verified",
            "paper_type": "research_article",
            "arxiv_match_status": "matched",
            "arxiv_id": "2601.00001",
        }
        source = {
            "download_status": "downloaded",
            "framework_candidates": [],
        }
        public = {
            "standalone_asset": True,
            "figure_asset_tier": "C_pdf_object_extracted",
            "framework_candidates": [{"caption": "Figure 1. Experimental workflow", "asset_path": "assets/x.png", "pdf_page": 2}],
            "fulltext_status": "extracted",
        }
        _, failures, figure = incremental_freeze.evaluate("materials", paper, source, public, {})
        self.assertEqual(failures, [])
        self.assertEqual(figure["figure_asset_tier"], "C_pdf_object_extracted")

    def test_incremental_freeze_reads_newer_domain_source_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            manifests = batch / "manifests"
            manifests.mkdir(parents=True)
            (manifests / "source_probe.json").write_text(
                json.dumps({"domains": {"agent": {"rows": [{"arxiv_id": "2601.00001", "framework_source_bound": False}]}}}),
                encoding="utf-8",
            )
            (manifests / "source_probe_agent.json").write_text(
                json.dumps({"domains": {"agent": {"rows": [{"arxiv_id": "2601.00001", "framework_source_bound": True}]}}}),
                encoding="utf-8",
            )
            rows = incremental_freeze.arxiv_source_rows(batch)
            self.assertTrue(rows["2601.00001"]["framework_source_bound"])

    def test_incremental_freeze_builds_deduplicated_multilabel_training_view(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = Path(folder)
            text = batch / "assets" / "agent" / "2601.00001" / "text"
            text.mkdir(parents=True)
            (text / "source_text.txt").write_text("paper text", encoding="utf-8")
            shared = {
                "paper_id": "paper-1",
                "arxiv_id": "2601.00001",
                "record_origin": "incremental_v002",
                "figure_asset_tier": "A_source_verified",
            }
            rows, report = incremental_freeze.build_canonical_rows(
                batch,
                {"agent": [dict(shared)], "rag": [dict(shared)]},
                {},
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["domain_labels"], ["agent", "rag"])
            self.assertEqual(rows[0]["full_text_path"], "assets/agent/2601.00001/text/source_text.txt")
            self.assertEqual(report["raw_verified_total"], 2)
            self.assertEqual(report["unique_verified_total"], 1)
            self.assertEqual(report["cross_domain_duplicate_groups"], 1)


if __name__ == "__main__":
    unittest.main()
