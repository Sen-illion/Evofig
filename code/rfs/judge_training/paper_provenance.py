"""Locate a figure's own wording in the paper the figure belongs to.

The bind stage gives every object of the author's figure a conclusion. Most of
them answer to a concept of the paper's method list, and some the reading itself
declared as content outside this paper's method. What remains is often still the
paper's own text -- a dataset name, a phrase the figure prints, a term the method
list does not carry -- it is simply not part of the method list. This module
answers one question about such an object: does the paper carry this phrase, and
where.

Matching is deliberately strict. A phrase matches only when the frozen snapshot
carries it as one contiguous run of text, after folding PDF line-break
hyphenation (``distilla-\\ntion``) and collapsing whitespace. A phrase whose
words merely occur somewhere in the article, far apart, does not match:
bag-of-words matching would "find" almost any phrase and turn the provenance
bucket into noise. Pages come from the snapshot's own ``text_locations.jsonl``;
the answer is the page number and the text around the hit, and nothing is
inferred beyond that.

The lookup is local, deterministic and calls no provider.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Iterable

from .frozen_document import paper_directory

PAPER_PROVENANCE_CONTRACT = "rfs.paper_provenance"
PAPER_PROVENANCE_VERSION = 1
MIN_PHRASE_CHARS = 5
SNIPPET_CHARS = 90
MAX_MATCH_PAGES = 40
_HYPHEN_BREAK = re.compile(r"-\s*\n\s*")
_WHITESPACE = re.compile(r"\s+")


def normalize_phrase(value: Any) -> str:
    """Fold one phrase the way the paper text is folded, for contiguous matching."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00ad", "")
    text = _HYPHEN_BREAK.sub("", text)
    return _WHITESPACE.sub(" ", text.casefold()).strip()


def is_searchable_phrase(value: Any) -> bool:
    """Say whether a phrase is specific enough to look for in a paper.

    A single glyph or a two-letter token would match half the article and mean
    nothing; a phrase needs some length and at least one letter.
    """
    normalized = normalize_phrase(value)
    if len(normalized) < MIN_PHRASE_CHARS:
        return False
    return any(character.isalpha() for character in normalized)


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                records.append(item)
    return records


class PaperText:
    """One paper's page texts, normalized once and searched contiguously."""

    def __init__(self, pages: Iterable[tuple[int, str]]):
        self.pages = [(int(page), text, normalize_phrase(text)) for page, text in pages]

    @classmethod
    def from_directory(cls, directory: str | Path | None) -> "PaperText | None":
        """Load the snapshot's own page texts; None when the paper carries none."""
        if directory is None:
            return None
        root = Path(directory)
        locations = root / "text_locations.jsonl"
        pages: list[tuple[int, str]] = []
        if locations.is_file():
            for record in _records(locations):
                text = str(record.get("text") or "")
                if text.strip():
                    pages.append((int(record.get("page") or 0), text))
        if not pages:
            flat = root / "full_text.txt"
            if flat.is_file():
                pages = [(0, flat.read_text(encoding="utf-8"))]
        return cls(pages) if pages else None

    def find(self, phrase: Any) -> dict[str, Any] | None:
        """Return where the paper carries this phrase as contiguous text, if it does."""
        if not is_searchable_phrase(phrase):
            return None
        needle = normalize_phrase(phrase)
        matches: list[dict[str, Any]] = []
        for page, _text, normalized in self.pages:
            index = normalized.find(needle)
            if index < 0:
                continue
            start = max(0, index - SNIPPET_CHARS)
            end = min(len(normalized), index + len(needle) + SNIPPET_CHARS)
            matches.append({"page": page, "snippet": normalized[start:end]})
            if len(matches) >= MAX_MATCH_PAGES:
                break
        if not matches:
            return None
        return {
            "contract": PAPER_PROVENANCE_CONTRACT,
            "version": PAPER_PROVENANCE_VERSION,
            "phrase": needle,
            "page": matches[0]["page"],
            "snippet": matches[0]["snippet"],
            "matches": len(matches),
        }


class PaperProvenanceLookup:
    """Resolve each Pilot paper's snapshot directory once, then search it lazily.

    The lookup is the only piece that knows where a paper's text lives; the bind
    layer receives just a callable per paper, so a missing snapshot degrades to
    "no provenance" instead of failing the stage.
    """

    def __init__(self, snapshot: str | Path, papers: Iterable[dict[str, Any]]):
        self.snapshot = Path(snapshot)
        self._directories: dict[str, Path] = {}
        for paper in papers:
            paper_id = str((paper or {}).get("paper_id") or "")
            if not paper_id or paper_id in self._directories:
                continue
            try:
                self._directories[paper_id] = paper_directory(self.snapshot, paper)
            except (KeyError, ValueError, OSError):
                continue
        self._texts: dict[str, PaperText | None] = {}

    def text(self, paper_id: str) -> PaperText | None:
        """Return one paper's page texts, loading them at most once."""
        key = str(paper_id)
        if key not in self._texts:
            directory = self._directories.get(key)
            try:
                self._texts[key] = PaperText.from_directory(directory)
            except (OSError, ValueError, json.JSONDecodeError):
                self._texts[key] = None
        return self._texts[key]

    def finder(self, paper_id: str) -> Callable[[Any], dict[str, Any] | None]:
        """Return the per-paper lookup the bind layer calls once per phrase."""

        def find(phrase: Any) -> dict[str, Any] | None:
            text = self.text(paper_id)
            return text.find(phrase) if text is not None else None

        return find

    def usage_report(self) -> dict[str, Any]:
        """Say how many papers this lookup can search, without reading any of them."""
        return {
            "contract": PAPER_PROVENANCE_CONTRACT,
            "version": PAPER_PROVENANCE_VERSION,
            "papers": len(self._directories),
        }


__all__ = [
    "MAX_MATCH_PAGES",
    "MIN_PHRASE_CHARS",
    "PAPER_PROVENANCE_CONTRACT",
    "PAPER_PROVENANCE_VERSION",
    "PaperProvenanceLookup",
    "PaperText",
    "is_searchable_phrase",
    "normalize_phrase",
]
