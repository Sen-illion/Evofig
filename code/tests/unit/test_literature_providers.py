"""Tests for independent literature provider lanes and normalized pages."""

from __future__ import annotations

from unittest.mock import Mock, patch
import unittest

import requests

from rfs.providers.aamas import AamasProceedingsProvider
from rfs.providers.literature import (
    ArxivProvider,
    CrossrefProvider,
    DblpProvider,
    LiteratureHttpClient,
    ProviderPage,
    ProviderRequestError,
    reset_literature_circuit_state,
)
from rfs.providers.arxiv_metadata import ArxivMetadataIndex, batch_match_exact


class LiteratureProviderTests(unittest.TestCase):
    def setUp(self):
        reset_literature_circuit_state()

    def test_provider_page_rejects_unknown_status(self):
        with self.assertRaises(ValueError):
            ProviderPage(provider="x", status="unknown")

    def test_http_client_classifies_rate_limit(self):
        response = Mock(status_code=429, headers={"Retry-After": "0"})
        session = Mock()
        session.headers = {}
        session.get.return_value = response
        client = LiteratureHttpClient("fixture", retries=0, min_interval=0, session=session)
        with self.assertRaises(ProviderRequestError) as raised:
            client.request("https://example.test")
        self.assertEqual(raised.exception.status, "rate_limited")

    def test_client_replaces_the_requests_default_user_agent(self):
        session = requests.Session()
        # requests ships this; setdefault would have silently kept it and every
        # publisher would answer 403.
        self.assertTrue(session.headers["User-Agent"].startswith("python-requests"))
        LiteratureHttpClient("fixture", session=session)
        self.assertTrue(session.headers["User-Agent"].startswith("PaperImage-LiteratureCollector"))

    def test_client_keeps_a_caller_supplied_user_agent(self):
        session = requests.Session()
        session.headers["User-Agent"] = "PaperImage-Independent10k-OfficialVerifier/0.1"
        LiteratureHttpClient("fixture", session=session)
        self.assertEqual(session.headers["User-Agent"], "PaperImage-Independent10k-OfficialVerifier/0.1")

    def test_literature_session_never_leaves_the_requests_default(self):
        from rfs.providers.literature import literature_session

        self.assertTrue(
            literature_session().headers["User-Agent"].startswith("PaperImage-LiteratureCollector")
        )

    def test_client_ignores_the_system_proxy_unless_opted_in(self):
        session = requests.Session()
        session.trust_env = True
        with patch.dict("os.environ", {}, clear=False) as environment:
            environment.pop("RFS_LITERATURE_TRUST_ENV", None)
            LiteratureHttpClient("fixture", session=session)
        self.assertFalse(session.trust_env)

    def test_client_honours_an_explicit_proxy_opt_in(self):
        session = requests.Session()
        session.trust_env = False
        with patch.dict("os.environ", {"RFS_LITERATURE_TRUST_ENV": "1"}):
            LiteratureHttpClient("fixture", session=session)
        self.assertTrue(session.trust_env)

    def test_crossref_records_are_candidates_not_official_proof(self):
        client = Mock()
        client.json.return_value = {
            "message": {
                "next-cursor": "next",
                "items": [
                    {
                        "DOI": "10.1000/ABC",
                        "title": ["A Full Retrieval Paper"],
                        "published": {"date-parts": [[2026, 4, 2]]},
                        "container-title": ["Proceedings of Test"],
                        "type": "proceedings-article",
                        "URL": "https://doi.org/10.1000/ABC",
                    }
                ],
            }
        }
        page = CrossrefProvider(client).list_works(year=2026, container="Proceedings of Test")
        self.assertEqual(page.next_cursor, "next")
        self.assertEqual(page.records[0]["doi"], "10.1000/abc")
        self.assertFalse(page.records[0]["official_proof"])
        self.assertNotIn("subtype", client.json.call_args.kwargs["params"]["select"])

    def test_crossref_cursor_iteration_reads_all_pages(self):
        client = Mock()
        client.json.side_effect = [
            {
                "message": {
                    "next-cursor": "second",
                    "total-results": 2,
                    "items": [{"DOI": "10.1/a", "title": ["First Paper"], "container-title": ["Proceedings"], "type": "proceedings-article"}],
                }
            },
            {
                "message": {
                    "next-cursor": "done",
                    "total-results": 2,
                    "items": [{"DOI": "10.1/b", "title": ["Second Paper"], "container-title": ["Proceedings"], "type": "proceedings-article"}],
                }
            },
        ]
        pages = list(CrossrefProvider(client).iter_works(year=2026, container="Proceedings", rows=1))
        self.assertEqual([[row["doi"] for row in page.records] for page in pages], [["10.1/a"], ["10.1/b"]])
        self.assertEqual(client.json.call_args_list[1].kwargs["params"]["cursor"], "second")

    def test_crossref_prefix_is_added_to_year_filter(self):
        client = Mock()
        client.json.return_value = {"message": {"items": [], "total-results": 0}}
        CrossrefProvider(client).list_works(
            year=2026,
            container="Proceedings of Test",
            prefix="10.1609",
        )
        self.assertIn("prefix:10.1609", client.json.call_args.kwargs["params"]["filter"])

    def test_aamas_provider_keeps_only_research_track(self):
        response = Mock()
        response.text = """
        <a name="R" id="R"></a><h2>Research Paper Track</h2>
        <a href="../pdfs/ABCD1234.pdf"><strong>A Full Multi-Agent Paper</strong></a> (Page 4)
        <a href="../pdfs/EFGH5678.pdf"><strong>Another &amp; Better Agent Paper</strong></a> (Page 13)
        <a name="E" id="E"></a><h2>Extended Abstracts</h2>
        <a href="../pdfs/SHORT.pdf"><strong>A Short Paper</strong></a> (Page 20)
        """
        client = Mock()
        client.request.return_value = response
        page = AamasProceedingsProvider(client).list_works(year=2026)
        self.assertEqual([row["official_paper_id"] for row in page.records], ["aamas:2026:ABCD1234", "aamas:2026:EFGH5678"])
        self.assertEqual(page.records[1]["title"], "Another & Better Agent Paper")
        self.assertTrue(all(row["official_proof"] for row in page.records))

    def test_arxiv_requires_one_exact_title(self):
        xml = """<feed xmlns='http://www.w3.org/2005/Atom'>
        <entry><id>https://arxiv.org/abs/2601.00001v2</id><title>Exact Pipeline</title></entry>
        <entry><id>https://arxiv.org/abs/2601.00002v1</id><title>Different Pipeline</title></entry>
        </feed>"""
        response = Mock(text=xml)
        client = Mock()
        client.request.return_value = response
        rows = ArxivProvider(client).match_exact(title="Exact Pipeline")
        self.assertEqual([row["arxiv_id"] for row in rows], ["2601.00001"])

    def test_local_arxiv_index_rejects_ambiguous_title(self):
        index = ArxivMetadataIndex(
            [
                {"arxiv_id": "2601.00001", "title": "Same Title"},
                {"arxiv_id": "2601.00002", "title": "Same Title"},
                {"arxiv_id": "2601.00003", "title": "Unique Title", "doi": "10.1000/x"},
            ]
        )

        ambiguous, ambiguous_method = index.match(title="Same Title")
        exact, exact_method = index.match(title="Unique Title", doi="10.1000/X")

        self.assertIsNone(ambiguous)
        self.assertEqual(ambiguous_method, "ambiguous_title_cache")
        self.assertEqual(exact["arxiv_id"], "2601.00003")
        self.assertEqual(exact_method, "exact_doi_cache")

    def test_bounded_batch_lookup_uses_one_query_for_twenty_titles(self):
        xml = """<feed xmlns='http://www.w3.org/2005/Atom'>
        <entry><id>https://arxiv.org/abs/2601.00001v2</id><title>Exact Pipeline</title></entry>
        </feed>"""
        response = Mock(text=xml)
        client = Mock()
        client.request.return_value = response

        rows = batch_match_exact(ArxivProvider(client), ["Exact Pipeline", "Missing Pipeline"])

        self.assertEqual(rows["exact pipeline"]["arxiv_id"], "2601.00001")
        client.request.assert_called_once()

    def test_provider_lanes_are_independent(self):
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        session = Mock()
        session.headers = {}
        session.get.return_value = response
        client = LiteratureHttpClient("crossref", min_interval=0, session=session)
        with patch("rfs.providers.literature.serialized_request", return_value=(response, 1)) as request:
            client.request("https://example.test")
        self.assertEqual(request.call_args.kwargs["lane"], "literature:crossref")

    def test_http_client_classifies_invalid_json(self):
        response = Mock(status_code=200, url="https://example.test")
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("invalid json")
        session = Mock()
        session.headers = {}
        session.get.return_value = response
        client = LiteratureHttpClient("broken-json", retries=0, min_interval=0, session=session)

        with self.assertRaises(ProviderRequestError) as raised:
            client.json("https://example.test")

        self.assertEqual(raised.exception.status, "failed")

    def test_not_found_is_terminal_without_retry(self):
        response = Mock(status_code=404, headers={})
        session = Mock()
        session.headers = {}
        session.get.return_value = response
        client = LiteratureHttpClient("missing", retries=2, min_interval=0, session=session)

        with self.assertRaises(ProviderRequestError) as raised:
            client.request("https://example.test/missing")

        self.assertEqual(raised.exception.status, "not_found")
        session.get.assert_called_once()

    def test_terminal_blocked_responses_do_not_open_provider_circuit(self):
        response = Mock(status_code=403, headers={})
        session = Mock()
        session.headers = {}
        session.get.return_value = response
        client = LiteratureHttpClient(
            "blocked-host",
            retries=0,
            min_interval=0,
            session=session,
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown=300,
        )
        for expected in ("blocked", "blocked", "blocked"):
            with self.assertRaises(ProviderRequestError) as raised:
                client.request("https://example.test")
            self.assertEqual(raised.exception.status, expected)
        self.assertEqual(session.get.call_count, 3)

    def test_repeated_transport_failures_open_provider_circuit(self):
        session = Mock()
        session.headers = {}
        session.get.side_effect = requests.RequestException("temporary TLS failure")
        client = LiteratureHttpClient(
            "broken-host",
            retries=0,
            min_interval=0,
            session=session,
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown=300,
        )
        for expected in ("failed", "failed", "circuit_open"):
            with self.assertRaises(ProviderRequestError) as raised:
                client.request("https://example.test")
            self.assertEqual(raised.exception.status, expected)
        self.assertEqual(session.get.call_count, 2)

    def test_dblp_is_paced_and_retried_instead_of_dropping_the_venue(self):
        provider = DblpProvider()
        self.assertGreaterEqual(provider.client.min_interval, 1.0)
        self.assertGreaterEqual(provider.client.retries, 4)


if __name__ == "__main__":
    unittest.main()
