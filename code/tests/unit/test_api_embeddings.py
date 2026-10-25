"""Tests for the paid API embedding provider used by soft semantic scoring."""

import json
import os
import unittest
from unittest.mock import patch

from rfs.providers.api_embeddings import (
    API_EMBEDDING_KIND,
    DEFAULT_API_EMBEDDING_DIMENSION,
    ApiEmbedder,
    api_embedder,
    api_model_name,
    is_api_spec,
)
from rfs.providers.embeddings import EmbeddingUnavailable

CREDENTIALS = {"API_BASE": "https://example.test/v1", "API_KEY": "test-key"}


class _Response:
    """Minimal stand-in for a requests response."""

    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


def _body(vectors, *, tokens=12, with_index=False):
    rows = [{"embedding": list(vector), "index": position} if with_index else {"embedding": list(vector)} for position, vector in enumerate(vectors)]
    return {"data": rows, "usage": {"total_tokens": tokens}}


class ApiEmbedderTests(unittest.TestCase):
    def embedder(self, **options):
        return ApiEmbedder("text-embedding-3-small", **options)

    def test_a_missing_credential_fails_loudly_without_calling_the_endpoint(self):
        with patch.dict(os.environ, {"API_BASE": "", "API_KEY": "", "GEMINI_API_KEY": ""}, clear=False):
            with patch("rfs.providers.api_embeddings.requests.post") as post:
                with self.assertRaises(EmbeddingUnavailable):
                    self.embedder().encode(["encoder"])

        self.assertFalse(post.called)

    def test_texts_are_batched_and_the_usage_is_counted(self):
        calls = []

        def fake_post(url, headers=None, data=None, timeout=None):
            calls.append(json.loads(data))
            return _Response(200, _body([[1.0, 0.0], [0.0, 1.0]], tokens=30))

        with patch.dict(os.environ, CREDENTIALS, clear=False):
            with patch("rfs.providers.api_embeddings.requests.post", fake_post):
                embedder = self.embedder(batch_size=2)
                vectors = embedder.encode(["first", "second"])

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["input"], ["first", "second"])
        self.assertEqual(calls[0]["model"], "text-embedding-3-small")
        self.assertEqual(embedder.usage()["calls"], 1)
        self.assertEqual(embedder.usage()["texts"], 2)
        self.assertEqual(embedder.usage()["total_tokens"], 30)
        self.assertEqual(embedder.dimension, 2)

    def test_a_repeated_text_is_never_asked_for_twice(self):
        seen = []

        def fake_post(url, headers=None, data=None, timeout=None):
            batch = json.loads(data)["input"]
            seen.extend(batch)
            return _Response(200, _body([[0.5, 0.5] for _ in batch]))

        with patch.dict(os.environ, CREDENTIALS, clear=False):
            with patch("rfs.providers.api_embeddings.requests.post", fake_post):
                embedder = self.embedder()
                first = embedder.encode(["transformer", "encoder"])
                second = embedder.encode(["encoder", "transformer"])

        self.assertEqual(seen, ["transformer", "encoder"])
        self.assertEqual(second, [first[1], first[0]])

    def test_a_transient_status_is_retried_once_and_then_succeeds(self):
        statuses = [503, 200]

        def fake_post(url, headers=None, data=None, timeout=None):
            batch = json.loads(data)["input"]
            return _Response(statuses.pop(0), _body([[1.0] for _ in batch]))

        with patch.dict(os.environ, CREDENTIALS, clear=False):
            with patch("rfs.providers.api_embeddings.requests.post", fake_post):
                embedder = self.embedder(retries=1)
                vectors = embedder.encode(["encoder"])

        self.assertEqual(vectors, [[1.0]])
        self.assertEqual(embedder.usage()["calls"], 2)

    def test_a_permanent_status_is_reported_with_its_code(self):
        with patch.dict(os.environ, CREDENTIALS, clear=False):
            with patch("rfs.providers.api_embeddings.requests.post", lambda *args, **kwargs: _Response(400, None, "bad request")):
                with self.assertRaises(EmbeddingUnavailable) as caught:
                    self.embedder(retries=0).encode(["encoder"])

        self.assertIn("400", str(caught.exception))

    def test_an_unreadable_or_mismatched_payload_is_rejected(self):
        with patch.dict(os.environ, CREDENTIALS, clear=False):
            with patch("rfs.providers.api_embeddings.requests.post", lambda *args, **kwargs: _Response(200, _body([[1.0], [2.0]]))):
                with self.assertRaises(EmbeddingUnavailable):
                    self.embedder().encode(["only one text"])

            with patch("rfs.providers.api_embeddings.requests.post", lambda *args, **kwargs: _Response(200, None)):
                with self.assertRaises(EmbeddingUnavailable):
                    self.embedder().encode(["text"])

    def test_an_out_of_order_payload_is_put_back_in_request_order(self):
        def fake_post(url, headers=None, data=None, timeout=None):
            return _Response(200, _body([[1.0], [2.0]], with_index=True))

        with patch.dict(os.environ, CREDENTIALS, clear=False):
            with patch("rfs.providers.api_embeddings.requests.post", fake_post):
                vectors = self.embedder().encode(["first", "second"])

        self.assertEqual(vectors, [[1.0], [2.0]])

    def test_blank_text_stays_a_zero_vector_without_a_call(self):
        with patch.dict(os.environ, CREDENTIALS, clear=False):
            with patch("rfs.providers.api_embeddings.requests.post") as post:
                vectors = self.embedder().encode(["  "])

        self.assertFalse(post.called)
        self.assertEqual(vectors, [[0.0] * DEFAULT_API_EMBEDDING_DIMENSION])

    def test_the_fingerprint_names_the_endpoint_model(self):
        fingerprint = self.embedder().fingerprint()

        self.assertEqual(fingerprint["kind"], API_EMBEDDING_KIND)
        self.assertEqual(fingerprint["model"], "text-embedding-3-small")
        self.assertEqual(fingerprint["dimension"], DEFAULT_API_EMBEDDING_DIMENSION)
        self.assertEqual(fingerprint["contract"], "rfs.text_embedding")

    def test_one_model_is_shared_so_its_cache_survives_the_run(self):
        self.assertIs(api_embedder("text-embedding-3-small"), api_embedder("text-embedding-3-small"))
        self.assertIsNot(api_embedder("text-embedding-3-large"), api_embedder("text-embedding-3-small"))

    def test_a_spec_is_read_as_this_provider(self):
        self.assertTrue(is_api_spec("api"))
        self.assertTrue(is_api_spec("api:text-embedding-3-large"))
        self.assertFalse(is_api_spec("hashing"))
        self.assertFalse(is_api_spec("st:models/local"))
        self.assertEqual(api_model_name("api:text-embedding-3-large"), "text-embedding-3-large")
        self.assertEqual(api_model_name("api"), "text-embedding-3-small")
        self.assertEqual(api_model_name(None), "text-embedding-3-small")


if __name__ == "__main__":
    unittest.main()
