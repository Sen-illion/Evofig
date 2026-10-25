"""One reading of the words a model writes where a 0-1 priority belongs.

A paper review, a structure plan and the card layout all put a value on an object, and all
three are sometimes handed that value as a word. The words are the same words in all three
places, and each place used to carry its own copy of the table -- one of them with no
fallback at all, where ``importance: "critical"`` was read with a bare ``float()`` and ended
the review that read it. The table lives here once, so a word is read the same way wherever
it appears.
"""

from __future__ import annotations

from typing import Any

#: The words a priority is written as, against the 0-1 scale they stand for. These four are
#: the ones the model writes in practice; anything else falls to the default rather than
#: being guessed at.
PRIORITY_WORDS: dict[str, float] = {
    "critical": 0.95,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.3,
}

#: What a priority that was never stated is worth, and what a word the table does not know is
#: worth: the middle of the scale, which claims nothing.
DEFAULT_PRIORITY = 0.5


def _clamped(value: float) -> float:
    """A priority outside the scale is read as the end of the scale it passed."""

    return max(0.0, min(1.0, value))


def priority_number(value: Any, *, default: float = DEFAULT_PRIORITY) -> float:
    """Read one priority as a number on the 0-1 scale.

    A number is taken as itself, clamped to the scale. A word is read through
    :data:`PRIORITY_WORDS`, and a string that spells a number is read as that number.
    Anything else -- a word the table does not know, a missing value, a mapping -- is
    ``default``. Callers that coerced with ``float()`` raised on exactly the words the model
    writes, and a priority is not worth failing a run over.
    """

    if isinstance(value, str):
        text = value.strip().casefold()
        if text in PRIORITY_WORDS:
            return PRIORITY_WORDS[text]
        try:
            return _clamped(float(text))
        except ValueError:
            return float(default)
    if isinstance(value, (int, float)):
        return _clamped(float(value))
    return float(default)


__all__ = ["DEFAULT_PRIORITY", "PRIORITY_WORDS", "priority_number"]
