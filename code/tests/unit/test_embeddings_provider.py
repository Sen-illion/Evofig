"""Tests for the local embedding providers used by graph retrieval."""

import json
from pathlib import Path
import tempfile
import unittest

from rfs.providers.embeddings import (
    DEFAULT_MAX_SEQ_LENGTH,
    DEFAULT_DIMENSION,
    EmbeddingUnavailable,
    HashingNgramEmbedder,
    LocalSentenceTransformerEmbedder,
    pinned_revision,
    resolve_embedder,
)


def _cosine(left, right):
    return sum(a * b for a, b in zip(left, right))


class HashingEmbedderTests(unittest.TestCase):
    def test_vectors_are_deterministic_and_unit_length(self):
        encoder = HashingNgramEmbedder()
        first = encoder.encode(["encoder decoder chain"])[0]
        second = encoder.encode(["encoder decoder chain"])[0]

        self.assertEqual(first, second)
        self.assertEqual(len(first), DEFAULT_DIMENSION)
        self.assertAlmostEqual(_cosine(first, first), 1.0, places=4)

    def test_identical_text_is_more_similar_than_unrelated_text(self):
        encoder = HashingNgramEmbedder()
        query, same, other = encoder.encode(["attention encoder decoder", "attention encoder decoder", "gradient boosted decision trees"])

        self.assertAlmostEqual(_cosine(query, same), 1.0, places=6)
        self.assertLess(_cosine(query, other), 0.9)

    def test_fingerprint_names_the_model_and_revision(self):
        fingerprint = HashingNgramEmbedder().fingerprint()

        self.assertEqual(fingerprint["kind"], "hashing_ngram")
        self.assertEqual(fingerprint["model"], "rfs-hashing-ngram-v1")
        self.assertEqual(fingerprint["dimension"], DEFAULT_DIMENSION)

    def test_empty_text_stays_a_zero_vector(self):
        self.assertEqual(set(HashingNgramEmbedder().encode([""])[0]), {0.0})

    def test_resolve_embedder_defaults_to_the_offline_encoder(self):
        self.assertIsInstance(resolve_embedder(), HashingNgramEmbedder)
        self.assertIsInstance(resolve_embedder("offline"), HashingNgramEmbedder)


class LocalModelEmbedderTests(unittest.TestCase):
    def test_a_missing_model_directory_fails_loudly(self):
        with self.assertRaises(EmbeddingUnavailable):
            LocalSentenceTransformerEmbedder(Path("tmp/definitely-not-here")).encode(["text"])

    def test_a_named_path_is_read_as_a_local_model(self):
        embedder = resolve_embedder("st:tmp/definitely-not-here")

        self.assertIsInstance(embedder, LocalSentenceTransformerEmbedder)
        with self.assertRaises(EmbeddingUnavailable):
            embedder.fingerprint()


class PinnedRevisionTests(unittest.TestCase):
    def test_a_missing_pin_file_reports_no_revision(self):
        with tempfile.TemporaryDirectory() as name:
            self.assertEqual("", pinned_revision(Path(name)))

    def test_the_pin_file_is_read_from_inside_and_next_to_the_model(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            model = root / "bge-small-en-v1.5"
            model.mkdir()
            inside = model / "MODEL_REVISION_PIN.json"
            inside.write_text(json.dumps({"resolved_commit": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"}), encoding="utf-8")
            self.assertEqual("5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", pinned_revision(model))
            self.assertEqual("5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", LocalSentenceTransformerEmbedder(model).revision)

            inside.unlink()
            (root / "bge-small-en-v1.5.MODEL_REVISION_PIN.json").write_text(json.dumps({"commit": "abc123"}), encoding="utf-8")
            self.assertEqual("abc123", pinned_revision(model))

    def test_a_broken_pin_file_is_ignored_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as name:
            model = Path(name) / "model"
            model.mkdir()
            (model / "MODEL_REVISION_PIN.json").write_text("{not json", encoding="utf-8")
            self.assertEqual("", pinned_revision(model))

    def test_the_fingerprint_records_how_a_local_model_is_encoded(self):
        embedder = LocalSentenceTransformerEmbedder(Path("tmp/definitely-not-here"), revision="deadbeef")

        self.assertEqual(DEFAULT_MAX_SEQ_LENGTH, embedder.max_seq_length)
        self.assertEqual("deadbeef", embedder.revision)
        with self.assertRaises(EmbeddingUnavailable):
            embedder.fingerprint()


if __name__ == "__main__":
    unittest.main()
