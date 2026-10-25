"""Checkpoint generation smoke: does the policy still emit parseable Judge JSON?

DPO never calls ``generate``. A LoRA that has collapsed still ranks preference pairs
while refusing to write a judgement. Each checkpoint therefore has to produce a few
answers with ``enable_thinking`` off; empty text, unparseable JSON or an obvious
repeated fragment fails the smoke and stops the run on the previous healthy archive.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from rfs.providers.client import extract_json


GENERATION_SMOKE_FILE = "generation_smoke.jsonl"
DEFAULT_SMOKE_LIMIT = 4
DEFAULT_SMOKE_MAX_NEW_TOKENS = 1024
DEFAULT_SMOKE_PASS_RATE = 0.5
MIN_REPEATED_RUN = 8


def has_obvious_repetition(text: str, *, min_repeats: int = MIN_REPEATED_RUN) -> bool:
    """Return True when the completion is a short fragment copied back to back.

    The 6000-step adapter degenerated into a single repeating syllable after
    ``skip_special_tokens``. A real judgement is never eight copies of the same 1–8
    character chunk, so that pattern is enough to fail the smoke without a second model.
    """

    body = str(text or "")
    if len(body) < int(min_repeats):
        return False
    run = 1
    for index in range(1, len(body)):
        if body[index] == body[index - 1]:
            run += 1
            if run >= int(min_repeats):
                return True
        else:
            run = 1
    for size in range(1, 9):
        repeats = 1
        previous = body[:size]
        index = size
        while index + size <= len(body):
            chunk = body[index:index + size]
            if chunk == previous:
                repeats += 1
                if repeats >= int(min_repeats):
                    return True
            else:
                previous = chunk
                repeats = 1
            index += size
    return False


def score_generation(text: str) -> dict[str, Any]:
    """Judge one generated string: non-empty, parseable JSON, not a repeated fragment."""

    body = str(text or "").strip()
    empty = not body
    repeated = has_obvious_repetition(body) if body else False
    json_ok = False
    if body and not repeated:
        try:
            parsed = extract_json(body)
            json_ok = isinstance(parsed, dict)
        except (TypeError, ValueError):
            json_ok = False
    return {
        "empty": empty,
        "json_ok": json_ok,
        "repeated": repeated,
        "passed": bool(body) and json_ok and not repeated,
        "chars": len(body),
    }


def run_generation_smoke(
    policy: Any,
    records: Sequence[dict[str, Any]],
    *,
    limit: int,
    max_new_tokens: int = DEFAULT_SMOKE_MAX_NEW_TOKENS,
    pass_rate: float = DEFAULT_SMOKE_PASS_RATE,
) -> dict[str, Any]:
    """Generate on the first ``limit`` records and report whether the policy still writes JSON.

    A backend without ``generate`` is skipped, not failed: the synthetic trainer has no
    decoder, and failing it would block every machine that cannot load the VLM stack.
    """

    generate: Callable[..., str] | None = getattr(policy, "generate", None)
    sample = list(records)[: max(0, int(limit))]
    if not callable(generate):
        return {
            "pairs": len(sample),
            "scored": 0,
            "passed": True,
            "skipped": True,
            "reason": "backend_cannot_generate",
            "pass_rate": None,
            "items": [],
        }
    if not sample:
        raise ValueError("generation smoke needs at least one preference pair")
    items: list[dict[str, Any]] = []
    for record in sample:
        text = generate(record, max_new_tokens=int(max_new_tokens))
        item = {"example_id": str(record.get("example_id") or ""), **score_generation(text)}
        items.append(item)
    scored = len(items)
    wins = sum(1 for item in items if item["passed"])
    rate = wins / scored if scored else 0.0
    return {
        "pairs": scored,
        "scored": scored,
        "passed": rate >= float(pass_rate),
        "skipped": False,
        "reason": "",
        "pass_rate": round(rate, 6),
        "items": items,
    }
