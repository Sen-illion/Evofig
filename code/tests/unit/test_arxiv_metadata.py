"""Tests for exact arXiv metadata records and live title resolution."""

from __future__ import annotations

import unittest

import requests
from unittest.mock import patch

from rfs.providers.arxiv_metadata import (
    ARXIV_WEB_SEARCH_URL,
    SEMANTIC_SCHOLAR_MATCH_URL,
    ArxivTitleResolver,
    modern_arxiv_id,
)
from rfs.providers.literature import ProviderRequestError


class StubResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class StubSession:
    """requests.Session stand-in that answers from a per-URL table."""

    def __init__(self, routes: dict) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict]] = []
        self.headers: dict = {}

    def get(self, url: str, params=None, timeout=None) -> StubResponse:
        self.calls.append((url, dict(params or {})))
        handler = self.routes.get(url)
        if handler is None:
            raise AssertionError(f"unexpected request to {url}")
        return handler(dict(params or {}))


def _result_block(arxiv_id: str, title: str) -> str:
    return (
        '<li class="arxiv-result">'
        f'<p class="list-title is-inline-block"><a href="https://arxiv.org/abs/{arxiv_id}">arXiv:{arxiv_id}</a></p>'
        f'<p class="title is-5 mathjax">{title}</p>'
        "</li>"
    )


def _resolver(session: StubSession, **kwargs) -> ArxivTitleResolver:
    settings = {"min_interval": 0.0, "web_min_interval": 0.0, "retries": 0, "backoff": 0.0}
    settings.update(kwargs)
    return ArxivTitleResolver(session=session, **settings)


class ModernArxivIdTests(unittest.TestCase):
    def test_accepts_bare_and_prefixed_identifiers_only(self):
        self.assertEqual(modern_arxiv_id("2502.10835"), "2502.10835")
        self.assertEqual(modern_arxiv_id("arXiv:2502.10835v2"), "2502.10835")
        self.assertEqual(modern_arxiv_id(" 2502.10835 "), "2502.10835")
        self.assertEqual(modern_arxiv_id("Quantum Paper"), "")
        self.assertEqual(modern_arxiv_id(None), "")


class ArxivTitleResolverTests(unittest.TestCase):
    def test_semantic_scholar_exact_title_builds_a_canonical_record(self):
        session = StubSession(
            {
                SEMANTIC_SCHOLAR_MATCH_URL: lambda params: StubResponse(
                    payload={
                        "data": [
                            {
                                "title": "Exact Paper",
                                "externalIds": {
                                    "ArXiv": "2502.10835",
                                    "DOI": "10.48550/arXiv.2502.10835",
                                },
                            }
                        ]
                    }
                )
            }
        )

        record, method = _resolver(session).resolve("Exact Paper")

        self.assertEqual(method, "exact_title_semantic_scholar")
        self.assertEqual(record["arxiv_id"], "2502.10835")
        self.assertEqual(record["title_norm"], "exact paper")
        self.assertEqual(record["source_url"], "https://export.arxiv.org/e-print/2502.10835")
        self.assertIsNone(record["doi"])

    def test_near_miss_title_falls_through_to_the_search_page(self):
        session = StubSession(
            {
                SEMANTIC_SCHOLAR_MATCH_URL: lambda params: StubResponse(
                    payload={"data": [{"title": "Exact Paper Extended", "externalIds": {"ArXiv": "2502.99999"}}]}
                ),
                ARXIV_WEB_SEARCH_URL: lambda params: StubResponse(
                    text="".join(
                        [
                            _result_block("2502.11111", "Exact Paper Extended"),
                            _result_block("2502.10835", "Exact Paper"),
                        ]
                    )
                ),
            }
        )

        record, method = _resolver(session).resolve("Exact Paper")

        self.assertEqual(method, "exact_title_arxiv_web_search")
        self.assertEqual(record["arxiv_id"], "2502.10835")
        self.assertEqual([url for url, _ in session.calls], [SEMANTIC_SCHOLAR_MATCH_URL, ARXIV_WEB_SEARCH_URL])

    def test_search_page_never_accepts_a_fuzzy_title(self):
        session = StubSession(
            {
                SEMANTIC_SCHOLAR_MATCH_URL: lambda params: StubResponse(payload={"data": []}),
                ARXIV_WEB_SEARCH_URL: lambda params: StubResponse(
                    text=_result_block("2502.11111", "Exact Paper Extended")
                ),
            }
        )

        record, method = _resolver(session).resolve("Exact Paper")

        self.assertIsNone(record)
        self.assertEqual(method, "no_live_title_match")

    def test_exact_title_without_a_preprint_falls_through(self):
        session = StubSession(
            {
                SEMANTIC_SCHOLAR_MATCH_URL: lambda params: StubResponse(
                    payload={"data": [{"title": "Exact Paper", "externalIds": {"DOI": "10.1000/xyz"}}]}
                ),
                ARXIV_WEB_SEARCH_URL: lambda params: StubResponse(text=""),
            }
        )

        record, method = _resolver(session).resolve("Exact Paper")

        self.assertIsNone(record)
        self.assertEqual(method, "no_live_title_match")

    def test_missing_entry_returns_no_match_without_raising(self):
        session = StubSession({SEMANTIC_SCHOLAR_MATCH_URL: lambda params: StubResponse(status_code=404)})

        record, method = _resolver(session, web_search=False).resolve("Exact Paper")

        self.assertIsNone(record)
        self.assertEqual(method, "no_live_title_match")

    def test_persistent_rate_limit_raises_a_classified_error(self):
        session = StubSession({SEMANTIC_SCHOLAR_MATCH_URL: lambda params: StubResponse(status_code=429)})

        with self.assertRaises(ProviderRequestError) as failure:
            _resolver(session, web_search=False).resolve("Exact Paper")

        self.assertEqual(failure.exception.provider, "arxiv_title_resolver")
        self.assertEqual(failure.exception.status, "http_429")

    def test_empty_title_never_reaches_the_network(self):
        session = StubSession({})

        record, method = _resolver(session).resolve("   ")

        self.assertIsNone(record)
        self.assertEqual(method, "empty_title")
        self.assertEqual(session.calls, [])


if __name__ == "__main__":
    unittest.main()