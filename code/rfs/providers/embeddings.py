"""Local, offline text embeddings for graph retrieval.

The Judge retrieval index stores one embedding per graph record and per subgraph
locator. Embeddings must be deterministic and offline: two runs over the same version
have to write byte identical artifacts, and no paid endpoint may sit inside a frozen
training index.

Two providers are offered:

* :class:`HashingNgramEmbedder` - the default. Word tokens and character n-grams are
  hashed into a fixed width vector with a stable sign, so it needs no model file, no GPU
  and no network, and it cannot drift between machines.
* :class:`LocalSentenceTransformerEmbedder` - a pinned local sentence-transformers model
  directory. It is used only when a caller names it explicitly and raises
  :class:`EmbeddingUnavailable` when the package or the directory is absent, instead of
  quietly falling back to another provider.

Every provider reports a fingerprint (kind, model, revision, dimension and, for a local
model, the pooling, sequence length and device it encoded with). The index and the
retrieval probe store that fingerprint, so a stored index can always be traced back to
the encoder that produced it.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


EMBEDDING_CONTRACT = "rfs.text_embedding"
HASHING_MODEL = "rfs-hashing-ngram-v1"
HASHING_REVISION = "v1"
DEFAULT_DIMENSION = 256
DEFAULT_NGRAM_SIZES = (3, 4)
DEFAULT_MAX_SEQ_LENGTH = 512
PIN_FILE_NAME = "MODEL_REVISION_PIN.json"
PIN_FILE_SUFFIX = ".MODEL_REVISION_PIN.json"
PIN_REVISION_KEYS = ("resolved_commit", "revision", "commit", "resolved_revision", "sha")
PADDING_POLICY = "one_text_per_forward_pass"
_WORD = re.compile(r"[a-z0-9]+")
_LOADED: dict[str, Any] = {}


class EmbeddingUnavailable(RuntimeError):
    """Raised when a requested embedding provider cannot be loaded."""


class TextEmbedder(Protocol):
    """Minimal interface shared by every embedding provider."""

    name: str
    revision: str
    dimension: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one dense vector per input text."""

    def fingerprint(self) -> dict[str, Any]:
        """Return the auditable identity of this provider."""


def _grams(text: str, sizes: Sequence[int]) -> list[str]:
    words = _WORD.findall(str(text).casefold())
    joined = " ".join(words)
    features = list(words)
    for size in sizes:
        if len(joined) < size:
            continue
        features.extend(joined[index:index + size] for index in range(len(joined) - size + 1))
    return features


def _normalize(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    return [round(value / norm, 8) for value in vector] if norm else [0.0 for _ in vector]


@dataclass(frozen=True)
class HashingNgramEmbedder:
    """Deterministic, dependency-free encoder used as the default provider."""

    dimension: int = DEFAULT_DIMENSION
    ngram_sizes: tuple[int, ...] = DEFAULT_NGRAM_SIZES
    name: str = HASHING_MODEL
    revision: str = HASHING_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", max(1, int(self.dimension)))

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.encode_text(text) for text in texts]

    def encode_text(self, text: str) -> list[float]:
        vector = [0.0] * max(1, int(self.dimension))
        for feature in _grams(text, self.ngram_sizes):
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % len(vector)
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return _normalize(vector)

    def fingerprint(self) -> dict[str, Any]:
        return {"contract": EMBEDDING_CONTRACT, "kind": "hashing_ngram", "model": self.name, "revision": self.revision, "dimension": int(self.dimension)}


def pinned_revision(model_path: str | Path) -> str:
    """Return the revision a local model directory was pinned to, or an empty string.

    A pinned model is what makes an index traceable. The pin file is looked up inside
    the model directory and next to it, because both shapes are in use.
    """

    path = Path(model_path)
    for candidate in (path / PIN_FILE_NAME, path.with_name(path.name + PIN_FILE_SUFFIX)):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        for key in PIN_REVISION_KEYS:
            value = str(payload.get(key) or "").strip()
            if value:
                return value
    return ""


def _transformer_module(model: Any) -> Any:
    """Return the transformer module of a loaded sentence-transformers model."""

    first = model[0]
    return getattr(first, "auto_model", first)


@dataclass(frozen=True)
class LocalSentenceTransformerEmbedder:
    """Pinned local sentence-transformers model, encoded one text at a time.

    The encoder deliberately avoids ``SentenceTransformer.encode``: that entry point
    pads a whole batch, so the same text encodes differently depending on which other
    texts share its batch, and a stored index drifts as the corpus grows. Padding one
    text per forward pass, pooling the first token under ``no_grad`` and normalizing
    makes the bytes of every vector a function of that text and the pinned model alone.
    """

    model_path: Path
    revision: str = ""
    name: str = ""
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    device: str = ""

    def __post_init__(self) -> None:
        path = Path(self.model_path)
        object.__setattr__(self, "model_path", path)
        object.__setattr__(self, "name", self.name or path.name)
        object.__setattr__(self, "max_seq_length", max(8, int(self.max_seq_length)))
        if not str(self.revision).strip():
            object.__setattr__(self, "revision", pinned_revision(path))

    def _model(self) -> Any:
        key = str(self.model_path)
        if key in _LOADED:
            return _LOADED[key]
        if not self.model_path.is_dir():
            raise EmbeddingUnavailable(f"local embedding model directory is missing: {self.model_path}")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailable("sentence-transformers is not installed; install it or use the offline hashing embedder") from exc
        _LOADED[key] = SentenceTransformer(str(self.model_path), revision=self.revision or None)
        return _LOADED[key]

    def _device(self) -> str:
        if str(self.device).strip():
            return str(self.device).strip()
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        items = [str(text) for text in texts]
        if not items:
            return []
        # The pinned model is loaded before the tensor library is touched, so a missing
        # directory or a missing sentence-transformers install raises EmbeddingUnavailable
        # and never a bare import error from a dependency of a dependency.
        model = self._model()
        import torch

        model = model.to(self._device())
        transformer = _transformer_module(model)
        transformer.eval()
        tokenizer = model.tokenizer
        vectors: list[list[float]] = []
        with torch.no_grad():
            for text in items:
                tokens = tokenizer([text], padding=True, truncation=True, max_length=int(self.max_seq_length), return_tensors="pt")
                fed = {key: value.to(self._device()) for key, value in tokens.items()}
                hidden = transformer(**fed).last_hidden_state[:, 0]
                pooled = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
                vectors.append([round(float(value), 8) for value in pooled[0].tolist()])
        return vectors

    @property
    def dimension(self) -> int:
        return int(self._model().get_sentence_embedding_dimension())

    def fingerprint(self) -> dict[str, Any]:
        return {
            "contract": EMBEDDING_CONTRACT,
            "kind": "sentence_transformers",
            "model": self.name,
            "revision": self.revision or "unpinned",
            "dimension": self.dimension,
            "pooling": "cls_token",
            "max_seq_length": int(self.max_seq_length),
            "device": str(self.device).strip() or "auto",
            "padding": PADDING_POLICY,
        }


def resolve_embedder(spec: str | Path | None = None) -> TextEmbedder:
    """Return the embedder named by spec; the offline hashing encoder by default."""

    text = "" if spec is None else str(spec).strip()
    if text.casefold() in {"", "hashing", "offline", "hashing-ngram"}:
        return HashingNgramEmbedder()
    if text.casefold().startswith("st:"):
        return LocalSentenceTransformerEmbedder(Path(text[3:].strip()))
    return LocalSentenceTransformerEmbedder(Path(text))


__all__ = [
    "DEFAULT_DIMENSION",
    "DEFAULT_MAX_SEQ_LENGTH",
    "EMBEDDING_CONTRACT",
    "EmbeddingUnavailable",
    "HashingNgramEmbedder",
    "LocalSentenceTransformerEmbedder",
    "PADDING_POLICY",
    "TextEmbedder",
    "pinned_revision",
    "resolve_embedder",
]