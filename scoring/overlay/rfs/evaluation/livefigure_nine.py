"""Nine VLM scores for one rendered framework figure.

The nine names follow LiveFigure: aesthetic, expressiveness, polish, clarity,
flow, legibility, accuracy, completeness, appropriateness. An illegal value
becomes null for that item only. Scoring stays off unless
``RFS_NINE_JUDGE=api``. The ruler reads the figure with its own model
(``RFS_SCORE_MODEL``, shared with the other scoring path, default
``gemini-3.1-flash-lite``) instead of ``MODEL_VLM``: sharing that variable with
generation would let a generation-side model swap move the scores without
anything saying so.

One prompt version is in force. ``nine-v003`` names the nine items and gives the
three fidelity items (accuracy, completeness, appropriateness) their definitions
and 1/3/5 anchors, graded against the paper's method text; the six visual items
are read from the figure itself. It replaced two earlier prompts --
``nine-v001``, which graded the figure alone and left the fidelity items empty,
and ``nine-v002``, the same rubric without the definitions or the anchors. Both
were retired on 2026-10-11: two rulers inside one table leave rows that differ by
the ruler and read as if they differed by the system. A retired name is refused
rather than answered under the current one, and so is a figure offered without
method text, because the fidelity items would otherwise be answered from the
drawing alone.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable

from .score_model import resolve_score_model

NINE_ENV_VAR = "RFS_NINE_JUDGE"
PROMPT_VERSION_ENV_VAR = "RFS_NINE_PROMPT_VERSION"
RETIRED_PROMPT_VERSIONS = ("nine-v001", "nine-v002")
DEFINED_RUBRIC_PROMPT_VERSION = "nine-v003"
PROMPT_VERSION = DEFINED_RUBRIC_PROMPT_VERSION
METHOD_TEXT_MARKER = "SOURCE METHOD TEXT"
NINE_SCORES = (
    "aesthetic",
    "expressiveness",
    "polish",
    "clarity",
    "flow",
    "legibility",
    "accuracy",
    "completeness",
    "appropriateness",
)

_PROMPT_DIR = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "editable-figure-bench"
    / "prompts"
)
_PROMPT_PATHS = {
    DEFINED_RUBRIC_PROMPT_VERSION: _PROMPT_DIR / "nine-v003.md",
}


def _prompt(version: str = PROMPT_VERSION) -> tuple[str, str]:
    text = _PROMPT_PATHS[version].read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _compose_prompt(version: str, method_text: str | None) -> tuple[str, str, bool]:
    """Return the prompt text, its hash, and whether it is grounded in method text."""
    text, text_sha = _prompt(version)
    body = str(method_text or "").strip()
    if not body:
        return text, text_sha, False
    material = f"{text}\n\n{METHOD_TEXT_MARKER}:\n{body}"
    return material, hashlib.sha256(material.encode("utf-8")).hexdigest(), True


def _integer_score(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 1 <= value <= 5:
        return value
    return None


def parse_nine_scores(payload: Any) -> dict[str, int | None]:
    """Keep a legal 1–5 integer and null every other item."""
    source = payload if isinstance(payload, dict) else {}
    return {name: _integer_score(source.get(name)) for name in NINE_SCORES}


def _blank(
    prompt_sha: str,
    reason: str,
    model: str | None = None,
    version: str = PROMPT_VERSION,
    grounded: bool = False,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "summary": "Nine VLM scores of one preview. Not a human preference.",
        "available": False,
        "prompt_version": version,
        "prompt_sha256": prompt_sha,
        "method_grounded": grounded,
        "model": model,
        "reason": reason,
        "mean": None,
    }
    for name in NINE_SCORES:
        report[name] = None
    return report


def score_nine(
    preview_path: str | Path | None,
    *,
    method_text: str | None = None,
    spec: str | None = None,
    call: Callable[..., dict[str, Any]] | None = None,
    render_error: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    """Score one preview with the one nine-item ruler.

    ``call`` replaces the provider, for tests. ``method_text`` is required:
    ``nine-v003`` grades the three fidelity items against the paper's own words,
    so a figure handed over without them is refused rather than graded as if the
    drawing alone answered for accuracy, completeness and appropriateness.

    ``version`` (or ``RFS_NINE_PROMPT_VERSION``) may name the ruler, and only
    ``nine-v003`` is accepted. The two prompts it replaced were retired on
    2026-10-11 and raise rather than being answered under the new name.
    """
    requested = str(version if version is not None else os.getenv(PROMPT_VERSION_ENV_VAR, "")).strip()
    if requested in RETIRED_PROMPT_VERSIONS:
        raise ValueError(
            f"{requested} was retired on 2026-10-11; the nine-item ruler is "
            f"{DEFINED_RUBRIC_PROMPT_VERSION} only"
        )
    if requested and requested not in _PROMPT_PATHS:
        raise ValueError(f"unknown nine-item prompt version: {requested}")
    if not str(method_text or "").strip():
        _, refused_sha, _ = _compose_prompt(PROMPT_VERSION, None)
        return _blank(
            refused_sha,
            f"{PROMPT_VERSION} grades the fidelity items against the method text, which was not passed",
            version=PROMPT_VERSION,
            grounded=True,
        )
    version = PROMPT_VERSION
    prompt, prompt_sha, grounded = _compose_prompt(version, method_text)
    if render_error:
        return _blank(prompt_sha, "render_failed: " + render_error, version=version, grounded=grounded)
    resolved = str(spec if spec is not None else os.getenv(NINE_ENV_VAR, "")).strip().lower()
    if resolved in {"", "off", "0", "false"}:
        return _blank(prompt_sha, "disabled", version=version, grounded=grounded)
    if resolved != "api":
        return _blank(prompt_sha, "unsupported_spec", version=version, grounded=grounded)
    path = Path(preview_path) if preview_path else Path()
    if not path.is_file():
        return _blank(prompt_sha, "missing preview.png", version=version, grounded=grounded)
    from ..providers.client import call_vlm_json, vlm_credentials_available

    model = resolve_score_model()
    if call is None and not vlm_credentials_available():
        return _blank(prompt_sha, "missing API_BASE or API_KEY", model, version=version, grounded=grounded)
    invoke = call or call_vlm_json
    try:
        payload = invoke(prompt, [path], model)
    except Exception as exc:
        return _blank(prompt_sha, f"{type(exc).__name__}: {exc}"[:200], model, version=version, grounded=grounded)
    parsed = parse_nine_scores(payload)
    report = _blank(prompt_sha, "", model, version=version, grounded=grounded)
    report.update(parsed)
    present = [value for value in parsed.values() if isinstance(value, int)]
    report["available"] = bool(present)
    report["mean"] = round(sum(present) / len(present), 4) if present else None
    if not present:
        report["reason"] = "response has no integer from 1 to 5"
    return report
