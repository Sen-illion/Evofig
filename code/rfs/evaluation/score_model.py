"""Which vendor model our own scoring calls read a paper or a figure with.

Scoring is a ruler. The nine-item scale and the PaperQuiz answer step both read
material with a model we did not train, and the number a paper prints is only
readable if that model is named. Reading ``MODEL_VLM`` instead would let a
generation-side model swap move the scores without anything saying so, and would
let the two scoring paths drift onto different models. One name, one default.
"""

from __future__ import annotations

import os

SCORE_MODEL_ENV_VAR = "RFS_SCORE_MODEL"

#: gemini-3.1-flash-lite: the cheap text/vision model the protocol names for scoring.
DEFAULT_SCORE_MODEL = "gemini-3.1-flash-lite"


def resolve_score_model() -> str:
    """Return the model a scoring call reads with."""

    return os.getenv(SCORE_MODEL_ENV_VAR, "").strip() or DEFAULT_SCORE_MODEL
