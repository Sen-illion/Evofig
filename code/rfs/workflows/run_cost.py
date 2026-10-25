"""Per-run cost ledger for a paper-evolve-editable directory.

Token counts come from provider usage blocks already written under the run.
Dollars use the public list prices recorded in ``PRICE_SNAPSHOT``. Image output
for ``gemini-3.1-flash-lite-image`` is one 1K image at the published per-image price.
Scoring calls are a separate line: ``PRICE_SNAPSHOT`` also carries the nine-item
ruler's price so its tokens can be priced from the same snapshot and date, but
nothing here adds them to ``generation_usd``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..utils import read_json, write_json

PRICE_SNAPSHOT = {
    "as_of": "2026-10-07",
    "text_model": "gemini-3.1-flash-lite",
    "text_input_usd_per_million": 0.25,
    "text_output_usd_per_million": 1.50,
    "image_model": "gemini-3.1-flash-lite-image",
    "image_usd_per_image": 0.0336,
    "image_output_tokens_per_image": 1120,
    # The nine-item ruler runs after the paper is delivered, so its price is reference data for
    # the scoring line and never part of ``generation_usd``. Its model is pinned in
    # ``rfs.evaluation.livefigure_nine`` and priced at the same published rates as the text model;
    # before 2026-10-07 the ruler read with ``gemini-3-pro-preview`` ($2.00 / $12.00 per million),
    # which the vendor shut down on 2026-03-09, so those scores are a different ruler.
    "judge_model": "gemini-3.1-flash-lite",
    "judge_input_usd_per_million": 0.25,
    "judge_output_usd_per_million": 1.50,
    "sources": [
        "https://ai.google.dev/gemini-api/docs/pricing",
        "https://web.archive.org/web/20251120213749/https://ai.google.dev/gemini-api/docs/pricing",
    ],
}


#: Per-image list prices, by icon model. The icon module resolves its model from
#: ``RFS_IMAGE_MODEL`` then ``IMAGE_MODEL``, and a row driver exports the first one per arm, so
#: the ledger has to price whatever that names. It used to carry a single price -- the one model
#: it was written against -- which billed a row drawn under the cheap model at several times
#: what it cost.
IMAGE_PRICES = {
    "gemini-3.1-flash-lite-image": {
        "image_usd_per_image": 0.0336,
        "sources": [
            "https://ai.google.dev/gemini-api/docs/pricing",
            "https://web.archive.org/web/20251120213749/https://ai.google.dev/gemini-api/docs/pricing",
        ],
    },
    "z-image-turbo": {
        # The icon calls are served by the credential file's API_BASE host, so its price list is
        # the one that bills them; the same table names gemini-3.1-flash-lite-image at 0.12923.
        "image_usd_per_image": 0.0075,
        "sources": ["provider price list at the API_BASE host (<API_HOST>)"],
    },
}


def image_rates(model: str = "") -> dict[str, Any]:
    """The per-image price of the icon model this run actually used.

    An unset or unrecognised name keeps the snapshot's own model and says so, rather than
    silently pricing another model's icons: a paper's icons are its largest cost line, so a
    guessed price is worse than a stated fallback.
    """

    name = str(model or os.getenv("RFS_IMAGE_MODEL") or os.getenv("IMAGE_MODEL") or "").strip()
    entry = IMAGE_PRICES.get(name)
    if entry is None:
        return {
            "image_model": PRICE_SNAPSHOT["image_model"],
            "image_usd_per_image": PRICE_SNAPSHOT["image_usd_per_image"],
            "image_price_basis": "no icon model was named; priced as %s" % PRICE_SNAPSHOT["image_model"],
        }
    return {
        "image_model": name,
        "image_usd_per_image": entry["image_usd_per_image"],
        "image_price_basis": "priced as the icon model this run named",
        "sources": list(PRICE_SNAPSHOT["sources"]) + list(entry["sources"]),
    }


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _usages(node: Any, found: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        usage = node.get("usage")
        if isinstance(usage, dict) and any(key in usage for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            found.append(usage)
        for key, value in node.items():
            if key != "usage":
                _usages(value, found)
    elif isinstance(node, list):
        for item in node:
            _usages(item, found)


def _add(usages: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "prompt_tokens": sum(_int(item.get("prompt_tokens")) for item in usages),
        "completion_tokens": sum(_int(item.get("completion_tokens")) for item in usages),
        "total_tokens": sum(_int(item.get("total_tokens")) for item in usages),
        "image_count": sum(_int(item.get("image_count")) for item in usages),
    }


def judge_usd(prompt_tokens: int, completion_tokens: int) -> float:
    """Price one scoring call at the recorded list prices.

    The nine-item ruler is billed apart from generation, so this number feeds the
    cost table's scoring line instead of ``generation_usd``.
    """

    rates = PRICE_SNAPSHOT
    return (
        _int(prompt_tokens) * rates["judge_input_usd_per_million"]
        + _int(completion_tokens) * rates["judge_output_usd_per_million"]
    ) / 1_000_000


def summarize_run_cost(root: str | Path, *, elapsed_seconds: float | None = None) -> dict[str, Any]:
    """Sum text and image usage under ``root`` and price the generation.

    The run root keeps a copy of the icon report the deliverable came from, and each round
    keeps its own. Both are the same document, so pricing every file named
    ``asset_generation_report.json`` charged that round twice: a paper whose 13 icons were
    drawn in three rounds reported 52 calls instead of 39, and every cost experiment
    inherited the inflation. The copies filed under a round are the ones that count; a
    directory that carries only the run-root copy (an archived round, or a directory written
    before this fix) still counts that one.
    """

    root = Path(root)
    text_usages: list[dict[str, Any]] = []
    image_usages: list[dict[str, Any]] = []
    image_calls = 0
    asset_reports: list[Path] = []
    counted_reports: set[Path] = set()
    if root.is_dir():
        asset_reports = sorted(root.rglob("asset_generation_report.json"))
        counted_reports = set([path for path in asset_reports if path.parent != root] or asset_reports)
        for path in root.rglob("*.json"):
            try:
                payload = read_json(path)
            except (OSError, ValueError):
                continue
            found: list[dict[str, Any]] = []
            _usages(payload, found)
            if path.name == "asset_generation_report.json":
                if path not in counted_reports:
                    continue
                image_usages.extend(found)
                image_calls += _int((payload or {}).get("api_requests_attempted") if isinstance(payload, dict) else 0)
            else:
                text_usages.extend(found)
    text = _add(text_usages)
    image = _add(image_usages)
    image_count = image["image_count"] or image_calls
    rates = {**PRICE_SNAPSHOT, **image_rates()}
    text_usd = (
        text["prompt_tokens"] * rates["text_input_usd_per_million"]
        + text["completion_tokens"] * rates["text_output_usd_per_million"]
    ) / 1_000_000
    image_usd = image_count * float(rates["image_usd_per_image"])
    return {
        "summary": "Generation cost for one paper. Scoring calls are not included.",
        "elapsed_seconds": None if elapsed_seconds is None else round(float(elapsed_seconds), 3),
        "text": text,
        "image": {
            **image,
            "api_requests_attempted": image_calls,
            "images_billed": image_count,
            "asset_reports_counted": len(counted_reports),
            "duplicate_asset_reports_skipped": max(0, len(asset_reports) - len(counted_reports)),
        },
        "price_snapshot": rates,
        "generation_usd": round(text_usd + image_usd, 6),
        "text_usd": round(text_usd, 6),
        "image_usd": round(image_usd, 6),
    }


def write_run_cost_report(root: str | Path, *, elapsed_seconds: float | None = None) -> dict[str, Any]:
    """Write ``cost_report.json`` beside the other run artifacts."""

    report = summarize_run_cost(root, elapsed_seconds=elapsed_seconds)
    write_json(Path(root) / "cost_report.json", report)
    return report

# The per-round ledger below is the same idea as the run ledger above, one level down: a round
# that only edits the figure program costs no generation call, and saying so out loud is what
# makes "the cheap layer was tried first" checkable instead of a claim in a design document.
ROUND_COST_CONTRACT = "rfs.round_cost"
ROUND_COST_SUMMARY_CONTRACT = "rfs.round_cost_summary"
ROUND_COST_VERSION = 1
ROUND_COST_FILE = "round_cost.json"

#: The contract of one move's own cost line. A round that ran two moves leaves two of
#: these beside its own total, so "what did the coach's suggestion cost" is readable
#: without inferring it by subtracting one action's number from another's.
MOVE_COST_CONTRACT = "rfs.move_cost"

#: The progress statuses that close a stage. A stage that never closed -- the run was stopped
#: inside it -- contributes nothing rather than its lifetime so far.
_TERMINAL_PROGRESS_STATUSES = ("completed", "failed", "skipped")

_COUNTED_COST_KEYS = ("candidate_count", "figure_generations", "compile_calls", "render_calls", "judge_calls", "stage_reruns")

#: What one model call reports about itself. A stage clock says how long the stage held; these
#: say how much of that was spent waiting on a model, which is the part a round can cut.
_MODEL_CALL_KEYS = ("elapsed_seconds", "prompt_tokens", "completion_tokens", "total_tokens", "image_count", "attempts", "retries_used")


def stage_seconds_from_progress(root: str | Path) -> dict[str, float]:
    """Sum the wall clock each stage has held so far, read from the run's own progress stream.

    A stage announces itself with a ``started`` event and closes with a terminal one, both
    stamped against the same clock by the reporter that wrote them.
    """

    path = Path(root) / "progress.jsonl"
    if not path.is_file():
        return {}
    seconds: dict[str, float] = {}
    opened: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        stage = str(event.get("stage") or "")
        status = str(event.get("status") or "")
        stamp = event.get("elapsed_seconds")
        if not stage or isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
            continue
        if status == "started":
            opened[stage] = float(stamp)
        elif status in _TERMINAL_PROGRESS_STATUSES and stage in opened:
            seconds[stage] = round(seconds.get(stage, 0.0) + (float(stamp) - opened.pop(stage)), 3)
    return seconds


def stage_seconds_between(before: dict[str, float] | None, after: dict[str, float] | None) -> dict[str, float]:
    """What one round added to each stage's clock, which is how a round reports its own time."""

    start = before if isinstance(before, dict) else {}
    end = after if isinstance(after, dict) else {}
    return {
        stage: round(float(end[stage]) - float(start.get(stage) or 0.0), 3)
        for stage in sorted(end)
        if round(float(end[stage]) - float(start.get(stage) or 0.0), 3) > 0
    }


def model_calls_from_progress(root: str | Path) -> list[dict[str, Any]]:
    """Every model call the run's progress stream has reported, in the order they finished.

    A call announces itself with one terminal event carrying the provider's own timing and
    token counts. The stage clock already says how long a stage held; this says how much of that
    was a model, so a round can name its minutes instead of only counting its calls.

    The carrier event is deliberately never paired with a ``started``: it adds a call to the
    ledger without closing the stage the call ran inside.
    """

    path = Path(root) / "progress.jsonl"
    if not path.is_file():
        return []
    calls: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        provider = event.get("provider")
        if not isinstance(provider, dict) or not provider:
            continue
        record: dict[str, Any] = {
            "call": str(event.get("call") or event.get("stage") or ""),
            "round": event.get("round"),
            "success": bool(provider.get("success")),
        }
        for key in _MODEL_CALL_KEYS:
            if provider.get(key) is not None:
                record[key] = provider[key]
        if provider.get("cache_hit"):
            record["cache_hit"] = True
        if provider.get("failure_categories"):
            record["failure_categories"] = list(provider["failure_categories"])
        calls.append(record)
    return calls


def move_cost_record(
    *,
    index: int,
    action: dict[str, Any] | None = None,
    action_class: str = "",
    source: str = "",
    figure_generations: int = 0,
    compile_calls: int = 0,
    render_calls: int = 0,
    judge_calls: int = 0,
    stage_reruns: int = 0,
    measured: bool = False,
) -> dict[str, Any]:
    """What one move of a round spent, so a two-move round bills each of them.

    A round that runs the mechanism's move and then the coach's suggestion spends twice in
    one round. Billing only the round would make "the cheap compile-only move costs
    nothing" and "the suggestion cost a full stage re-run" read as the same number.
    """

    counts = {
        "figure_generations": _int(figure_generations),
        "compile_calls": _int(compile_calls),
        "render_calls": _int(render_calls),
        "judge_calls": _int(judge_calls),
        "stage_reruns": _int(stage_reruns),
    }
    return {
        "contract": MOVE_COST_CONTRACT,
        "version": ROUND_COST_VERSION,
        "index": _int(index),
        "action": str((action or {}).get("type") or ""),
        "action_class": str(action_class or ""),
        "source": str(source or ""),
        **counts,
        "measured": bool(measured),
        "summary": (
            f"move {_int(index) + 1}: {counts['figure_generations']} generation call(s), "
            f"{counts['compile_calls']} compile(s), {counts['judge_calls']} judge answer(s), "
            f"{counts['stage_reruns']} stage re-run(s)"
        ),
    }


def round_cost_record(
    *,
    round_index: int,
    action: dict[str, Any] | None = None,
    action_class: str = "",
    candidate_count: int = 0,
    figure_generations: int = 0,
    compile_calls: int = 0,
    render_calls: int = 0,
    judge_calls: int = 0,
    stage_reruns: int = 0,
    stage_seconds: dict[str, float] | None = None,
    elapsed_seconds: float | None = None,
    moves: list[dict[str, Any]] | None = None,
    model_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """What one round spent: generation calls, compiles, renders, judge answers, stage re-runs.

    A round that ran more than one move carries each move's own line, so the total and its
    parts are written together instead of one being inferred from the other.
    """

    counts = {
        "candidate_count": _int(candidate_count),
        "figure_generations": _int(figure_generations),
        "compile_calls": _int(compile_calls),
        "render_calls": _int(render_calls),
        "judge_calls": _int(judge_calls),
        "stage_reruns": _int(stage_reruns),
    }
    calls = [dict(item) for item in (model_calls or []) if isinstance(item, dict)]
    return {
        "contract": ROUND_COST_CONTRACT,
        "version": ROUND_COST_VERSION,
        "round": int(round_index),
        "action": str((action or {}).get("type") or ""),
        "action_class": str(action_class or ""),
        **counts,
        "stage_seconds": dict(stage_seconds or {}),
        "elapsed_seconds": None if elapsed_seconds is None else round(float(elapsed_seconds), 3),
        "move_count": len([item for item in (moves or []) if isinstance(item, dict)]),
        "moves": [dict(item) for item in (moves or []) if isinstance(item, dict)],
        "model_calls": calls,
        "model_call_count": len(calls),
        "model_seconds": round(sum(float(item.get("elapsed_seconds") or 0.0) for item in calls), 3),
        "summary": (
            f"{counts['figure_generations']} generation call(s), {counts['compile_calls']} compile(s), "
            f"{counts['judge_calls']} judge answer(s), {counts['stage_reruns']} stage re-run(s)"
        ),
    }


def write_round_cost(round_dir: str | Path, **kwargs: Any) -> dict[str, Any]:
    """Write ``round_cost.json`` into one round directory and return the record."""

    record = round_cost_record(**kwargs)
    write_json(Path(round_dir) / ROUND_COST_FILE, record)
    return record


def summarize_round_costs(records: Any) -> dict[str, Any]:
    """Add the rounds up for the run report, so a wave's spend is one line and not a directory."""

    rounds = [item for item in records or [] if isinstance(item, dict)]
    totals = {key: sum(_int(item.get(key)) for item in rounds) for key in _COUNTED_COST_KEYS}
    return {
        "contract": ROUND_COST_SUMMARY_CONTRACT,
        "version": ROUND_COST_VERSION,
        "rounds": len(rounds),
        "totals": totals,
        "elapsed_seconds": round(sum(float(item.get("elapsed_seconds") or 0.0) for item in rounds), 3),
        "summary": (
            f"{len(rounds)} round(s): {totals['figure_generations']} generation call(s), "
            f"{totals['compile_calls']} compile(s), {totals['judge_calls']} judge answer(s), "
            f"{totals['stage_reruns']} stage re-run(s)"
        ),
        "records": rounds,
    }
