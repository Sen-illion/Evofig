"""Nine VLM scores for one rendered framework figure.

The nine names follow LiveFigure: aesthetic, expressiveness, polish, clarity,
flow, legibility, accuracy, completeness, appropriateness. An illegal value
becomes null for that item only. Scoring stays off unless
``RFS_NINE_JUDGE=api``. The ruler reads the figure with its own model
(``RFS_SCORE_MODEL``, shared with the other scoring path, default
``gemini-3.1-flash-lite``) instead of ``MODEL_VLM``: sharing that variable with
generation would let a generation-side model swap move the scores without
anything saying so.

Two prompt versions exist. ``nine-v001`` grades from the figure alone and is
kept for reproducibility. ``nine-v002`` also receives the paper's method text
and is used whenever method text is passed, matching FigureBench's AutoFigure
judge, which grades the three fidelity items (accuracy, completeness,
appropriateness) against the source text. The prompt is otherwise fixed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable

from .score_model import resolve_score_model

NINE_ENV_VAR = "RFS_NINE_JUDGE"
IMAGE_ONLY_PROMPT_VERSION = "nine-v001"
PAPER_GROUNDED_PROMPT_VERSION = "nine-v002"
PROMPT_VERSION = IMAGE_ONLY_PROMPT_VERSION
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
    IMAGE_ONLY_PROMPT_VERSION: _PROMPT_DIR / "nine-v001.md",
    PAPER_GROUNDED_PROMPT_VERSION: _PROMPT_DIR / "nine-v002.md",
}


def _prompt(version: str = IMAGE_ONLY_PROMPT_VERSION) -> tuple[str, str]:
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
) -> dict[str, Any]:
    """Score one preview. ``call`` replaces the provider, for tests.

    Passing ``method_text`` grounds the three fidelity items in the source
    method section and switches the prompt to ``nine-v002``, the AutoFigure
    convention. Without it the scorer stays image-only (``nine-v001``).
    """
    version = PAPER_GROUNDED_PROMPT_VERSION if str(method_text or "").strip() else IMAGE_ONLY_PROMPT_VERSION
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
