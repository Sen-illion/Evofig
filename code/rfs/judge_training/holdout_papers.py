"""Paper ids that must not be used to evolve a figure.

The four lists are the choice-model holdout, the exam, and the arrow and
overlap special sets. A missing file refuses the run. An empty path list is
not treated as "nothing is excluded".
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .dpo_slices import exam_paper_ids


def load_blocked_paper_ids(paths: Iterable[str | Path]) -> set[str]:
    """Union the paper ids in every list. A missing file is an error."""

    listed = [Path(item) for item in paths]
    if not listed:
        raise FileNotFoundError("judge holdout lists were not given")
    missing = [str(path) for path in listed if not path.is_file()]
    if missing:
        raise FileNotFoundError("judge holdout list is missing: " + ", ".join(missing))
    papers: set[str] = set()
    for path in listed:
        papers |= exam_paper_ids(path)
    return papers


def assert_paper_not_held_out(paper_id: str, paths: Iterable[str | Path]) -> set[str]:
    """Refuse an evolution run whose paper sits in one of the lists."""

    blocked = load_blocked_paper_ids(paths)
    identity = str(paper_id or "").strip()
    if identity and identity in blocked:
        raise ValueError(f"paper {identity} is in the judge holdout and is excluded from evolution")
    return blocked
