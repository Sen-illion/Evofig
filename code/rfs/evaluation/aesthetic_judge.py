"""Paper2Poster aesthetic scores for one delivered framework-figure preview.

The three scores are Element Quality, Layout Balance, and Engagement, on the
1–5 scales defined by Paper2Poster. The prompt is fixed at
``benchmarks/editable-figure-bench/prompts/aesthetic-v001.md`` and is adapted
so the judge sees only the delivered preview, never ``reference.png``.

Scoring stays off unless ``RFS_AESTHETIC_JUDGE=api``. A disabled, credential-less,
or unparseable call records null scores and a reason. It does not invent a
number, and it is not a human preference.
"""

from __future__ import annotations

import hashlib
import os
import statistics
from pathlib import Path
from typing import Any, Callable, Sequence

AESTHETIC_ENV_VAR = "RFS_AESTHETIC_JUDGE"
PROMPT_VERSION = "aesthetic-v001"
AESTHETIC_SCORES = ("element_quality", "layout_balance", "engagement")

_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "editable-figure-bench"
    / "prompts"
    / "aesthetic-v001.md"
)


def _prompt() -> tuple[str, str]:
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _blank(prompt_sha: str, reason: str, model: str | None = None) -> dict[str, Any]:
    return {
        "summary": "Paper2Poster aesthetic scores for one delivered preview.",
        "available": False,
        "element_quality": None,
        "layout_balance": None,
        "engagement": None,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha,
        "model": model,
        "env_var": AESTHETIC_ENV_VAR,
        "reason": reason,
    }


def _integer_score(value: Any) -> int | None:
    """Accept only real integers from 1 to 5. Booleans and floats are rejected."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if 1 <= value <= 5:
        return value
    return None


def parse_aesthetic_scores(payload: Any) -> dict[str, int] | None:
    """Return the three scores, or None when any score is missing or not 1–5."""
    if not isinstance(payload, dict):
        return None
    parsed: dict[str, int] = {}
    for name in AESTHETIC_SCORES:
        score = _integer_score(payload.get(name))
        if score is None:
            return None
        parsed[name] = score
    return parsed


def judge_aesthetics(
    preview_path: str | Path,
    *,
    spec: str | None = None,
    call: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score one preview. ``call`` replaces the provider, for tests.

    The default reads ``RFS_AESTHETIC_JUDGE``. Only the value ``api`` places a
    request. Anything else, including an empty value, stays disabled.
    """
    prompt, prompt_sha = _prompt()
    resolved = str(spec if spec is not None else os.getenv(AESTHETIC_ENV_VAR, "")).strip().lower()
    if resolved in {"", "off", "0", "false"}:
        return _blank(prompt_sha, "disabled")
    if resolved != "api":
        return _blank(prompt_sha, "unsupported_spec")
    path = Path(preview_path)
    if not path.is_file():
        return _blank(prompt_sha, "missing preview.png")
    from ..providers.client import call_vlm_json, resolve_vlm_model, vlm_credentials_available

    model = resolve_vlm_model()
    # An injected caller is a test double. The live path still refuses to send
    # a request when the process has no API credentials.
    if call is None and not vlm_credentials_available():
        return _blank(prompt_sha, "missing API_BASE or API_KEY", model)
    invoke = call or call_vlm_json
    try:
        payload = invoke(prompt, [path], model)
    except Exception as exc:
        return _blank(prompt_sha, f"{type(exc).__name__}: {exc}"[:200], model)
    parsed = parse_aesthetic_scores(payload)
    if parsed is None:
        return _blank(prompt_sha, "response is not three integers from 1 to 5", model)
    report = _blank(prompt_sha, "", model)
    report["available"] = True
    report.update(parsed)
    return report


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor
        while end + 1 < len(order) and values[order[end + 1]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end + 1) / 2
        for position in range(cursor, end + 1):
            ranks[order[position]] = average
        cursor = end + 1
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3:
        return None
    ranked_left = _ranks(left)
    ranked_right = _ranks(right)
    mean_left = statistics.fmean(ranked_left)
    mean_right = statistics.fmean(ranked_right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(ranked_left, ranked_right))
    scale_left = sum((x - mean_left) ** 2 for x in ranked_left) ** 0.5
    scale_right = sum((y - mean_right) ** 2 for y in ranked_right) ** 0.5
    if scale_left == 0 or scale_right == 0:
        return None
    return numerator / (scale_left * scale_right)


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("paper_id") or ""), str(row.get("system_id") or ""))


def agreement_report(automatic: Sequence[dict[str, Any]], human: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare automatic scores with a human JSONL on paired papers.

    A metric enters the denominator only when both sides recorded an integer
    from 1 to 5 for that paper and system. Exact agreement is the share of
    those pairs that chose the same integer. Spearman rho needs at least three
    pairs and is null otherwise.
    """
    human_by_key = {_key(row): row for row in human}
    per_score: dict[str, Any] = {}
    for name in AESTHETIC_SCORES:
        pairs: list[tuple[int, int]] = []
        for row in automatic:
            other = human_by_key.get(_key(row))
            if other is None:
                continue
            auto_score = _integer_score(row.get(name))
            human_score = _integer_score(other.get(name))
            if auto_score is None or human_score is None:
                continue
            pairs.append((auto_score, human_score))
        count = len(pairs)
        rho = _spearman([item[0] for item in pairs], [item[1] for item in pairs])
        per_score[name] = {
            "count": count,
            "exact_agreement": round(sum(1 for auto, person in pairs if auto == person) / count, 4) if count else None,
            "spearman_rho": None if rho is None else round(rho, 4),
        }
    return {
        "summary": "Agreement between the aesthetic judge and human scores. This is not a human preference result.",
        "prompt_version": PROMPT_VERSION,
        "scores": per_score,
    }
