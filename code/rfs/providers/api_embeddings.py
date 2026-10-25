"""Paid OpenAI-compatible text embeddings for soft semantic scoring.

The offline encoders in :mod:`rfs.providers.embeddings` are what a frozen index must use:
they are deterministic, need no network and produce byte identical artifacts. This module
is the opposite trade: a real embedding model behind an HTTP endpoint, used where two
artifacts are compared inside one run and a paraphrase has to score above a shared word.

The canonical caller is the Pilot cross-layer binder, which ranks the drawn blocks a
concept read off the paper's own figure could belong to. Nothing frozen may depend on
this module: a stored index or a sealed artifact has to stay reproducible without a paid
endpoint.

Texts are embedded once per process. The endpoint is asked only for texts it has not seen
yet, so scoring every label of every paper costs one request per batch rather than one
request per comparison. A missing credential or a failed request raises
:class:`~rfs.providers.embeddings.EmbeddingUnavailable` instead of returning a zero
vector: a caller that cannot embed has to say so.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

import requests

from .embeddings import EMBEDDING_CONTRACT, EmbeddingUnavailable

API_EMBEDDING_KIND = "api_embeddings"
DEFAULT_API_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_API_EMBEDDING_DIMENSION = 1536
DEFAULT_API_EMBEDDING_BATCH = 128
DEFAULT_API_EMBEDDING_TIMEOUT = 60
API_RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
API_SPEC_PREFIX = "api:"
_LOADED: dict[str, "ApiEmbedder"] = {}


class ApiEmbedder:
    """One OpenAI-compatible embedding endpoint with a per-process text cache."""

    def __init__(
        self,
        model: str = DEFAULT_API_EMBEDDING_MODEL,
        *,
        batch_size: int = DEFAULT_API_EMBEDDING_BATCH,
        timeout: int = DEFAULT_API_EMBEDDING_TIMEOUT,
        retries: int = 1,
    ) -> None:
        self.name = str(model or DEFAULT_API_EMBEDDING_MODEL)
        self.revision = ""
        self.dimension = DEFAULT_API_EMBEDDING_DIMENSION
        self.batch_size = max(1, int(batch_size))
        self.timeout = max(1, int(timeout))
        self.retries = max(0, int(retries))
        self.calls = 0
        self.texts = 0
        self.total_tokens = 0
        self.last_error = ""
        self._cache: dict[str, list[float]] = {}

    def _credentials(self) -> tuple[str, str]:
        api_base = os.getenv("API_BASE", "").strip().rstrip("/")
        api_key = (os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_base or not api_key:
            raise EmbeddingUnavailable("API embeddings require API_BASE and API_KEY/GEMINI_API_KEY environment variables")
        return api_base, api_key

    def _vectors(self, batch: Sequence[str]) -> list[list[float]]:
        """Embed one batch, retrying only a transient failure."""
        api_base, api_key = self._credentials()
        payload = {"model": self.name, "input": list(batch)}
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(
                    api_base + "/embeddings",
                    headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = "transport_error: " + str(exc)
                continue
            self.calls += 1
            if response.status_code in API_RETRY_STATUSES and attempt < self.retries:
                last_error = "http_" + str(response.status_code)
                continue
            if response.status_code != 200:
                raise EmbeddingUnavailable("API embeddings returned HTTP " + str(response.status_code) + ": " + str(response.text)[:200])
            try:
                body = response.json()
            except ValueError as exc:
                raise EmbeddingUnavailable("API embeddings returned unreadable JSON: " + str(exc)) from exc
            if not isinstance(body, dict):
                raise EmbeddingUnavailable("API embeddings returned a payload that is not an object")
            usage = body.get("usage")
            if isinstance(usage, dict):
                self.total_tokens += int(usage.get("total_tokens") or 0)
            rows = body.get("data")
            if not isinstance(rows, list) or len(rows) != len(batch):
                raise EmbeddingUnavailable("API embeddings returned a different number of vectors than requested")
            ordered = rows
            if all(isinstance(item, dict) and isinstance(item.get("index"), int) for item in rows):
                ordered = [item for _, item in sorted(((item["index"], item) for item in rows), key=lambda pair: pair[0])]
            vectors: list[list[float]] = []
            for item in ordered:
                vector = item.get("embedding") if isinstance(item, dict) else None
                if not isinstance(vector, list) or not vector:
                    raise EmbeddingUnavailable("API embeddings returned an empty vector")
                vectors.append([round(float(value), 8) for value in vector])
            self.dimension = len(vectors[0])
            return vectors
        raise EmbeddingUnavailable("API embeddings failed after " + str(self.retries + 1) + " attempt(s): " + (last_error or "unknown error"))

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, asking the endpoint only for new texts."""
        wanted = [str(text) for text in texts]
        for text in wanted:
            if not text.strip():
                self._cache.setdefault(text, [0.0] * self.dimension)
        missing = list(dict.fromkeys(text for text in wanted if text not in self._cache))
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            try:
                vectors = self._vectors(batch)
            except EmbeddingUnavailable as exc:
                self.last_error = str(exc)
                raise
            for text, vector in zip(batch, vectors):
                self._cache[text] = vector
            self.texts += len(batch)
        return [list(self._cache[text]) for text in wanted]

    def fingerprint(self) -> dict[str, Any]:
        """Return the auditable identity of this provider."""
        return {"contract": EMBEDDING_CONTRACT, "kind": API_EMBEDDING_KIND, "model": self.name, "revision": self.revision or "unpinned", "dimension": int(self.dimension)}

    def usage(self) -> dict[str, Any]:
        """Return the provider usage this embedder caused, for the caller's ledger."""
        return {"calls": int(self.calls), "texts": int(self.texts), "total_tokens": int(self.total_tokens), "error": self.last_error or ""}


def is_api_spec(spec: str | None) -> bool:
    """Say whether an embedder spec names this provider."""
    folded = str(spec or "").strip().casefold()
    return folded == "api" or folded.startswith(API_SPEC_PREFIX)


def api_model_name(spec: str | None = None) -> str:
    """Return the model a spec names, or the default for a bare ``api``."""
    text = str(spec or "").strip()
    if text.casefold().startswith(API_SPEC_PREFIX):
        return text[len(API_SPEC_PREFIX):].strip() or DEFAULT_API_EMBEDDING_MODEL
    return DEFAULT_API_EMBEDDING_MODEL


def api_embedder(model: str = DEFAULT_API_EMBEDDING_MODEL, **options: Any) -> ApiEmbedder:
    """Return the shared embedder of one model, so its cache survives the whole run."""
    key = API_SPEC_PREFIX + str(model)
    if key not in _LOADED:
        _LOADED[key] = ApiEmbedder(model=model, **options)
    return _LOADED[key]


__all__ = [
    "API_EMBEDDING_KIND",
    "API_SPEC_PREFIX",
    "ApiEmbedder",
    "DEFAULT_API_EMBEDDING_BATCH",
    "DEFAULT_API_EMBEDDING_DIMENSION",
    "DEFAULT_API_EMBEDDING_MODEL",
    "DEFAULT_API_EMBEDDING_TIMEOUT",
    "api_embedder",
    "api_model_name",
    "is_api_spec",
]
