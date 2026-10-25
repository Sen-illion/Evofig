"""Score a checkpoint by generating Judge JSON, not by ranking gold completions.

The product Judge writes an answer. Ranking two already-written JSON strings can
saturate while generation on the same papers gets worse. Each save therefore has
the policy write judgements for a small held-out slice and compares the pick and
the named defects against the certified labels.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Sequence

from rfs.providers.client import extract_json

from .generation_smoke import score_generation
from .judge_schema import GATE_DIAGNOSIS_TYPES, diagnosis_types, pairwise_choice


GENERATION_VAL_FILE = "generation_val.jsonl"
NON_DEFECT = frozenset({"assumed_worse", "correct_but_ugly"})


def parse_judgement(text: str) -> dict[str, Any]:
    """Return the JSON object a generated completion carries, or an empty dict."""

    body = str(text or "").strip()
    if not body:
        return {}
    try:
        value = extract_json(body)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def gold_judgement(record: dict[str, Any]) -> dict[str, Any]:
    """Return the certified judgement stored as the DPO chosen completion."""

    try:
        value = json.loads(str(record.get("chosen") or ""))
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def expected_defects(record: dict[str, Any]) -> list[str]:
    """Return injected diagnosis types the generated answer is supposed to name."""

    basis = record.get("preference_basis") if isinstance(record.get("preference_basis"), dict) else {}
    named = []
    for item in basis.get("known_defects") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("diagnosis_type") or "").strip()
        if kind and kind not in NON_DEFECT:
            named.append(kind)
    if named:
        return sorted(set(named))
    return sorted({item for item in diagnosis_types(gold_judgement(record)) if item and item not in NON_DEFECT})


def score_generated_judgement(text: str, record: dict[str, Any]) -> dict[str, Any]:
    """Compare one generated string with the certified pick and defects of ``record``."""

    generation = score_generation(text)
    parsed = parse_judgement(text) if generation["json_ok"] else {}
    gold = gold_judgement(record)
    gold_pick = pairwise_choice(gold)
    pred_pick = pairwise_choice(parsed)
    pairwise_ok = bool(gold_pick) and pred_pick == gold_pick
    expected = expected_defects(record)
    predicted = set(diagnosis_types(parsed))
    recalled = [item for item in expected if item in predicted]
    return {
        "example_id": str(record.get("example_id") or ""),
        "submitted": bool(generation["passed"]),
        "pairwise_ok": pairwise_ok,
        "has_pairwise": bool(gold_pick),
        "gold_pick": gold_pick,
        "pred_pick": pred_pick,
        "expected_defects": expected,
        "recalled_defects": recalled,
        "diagnosis_hit": (not expected) or bool(recalled),
        **generation,
    }


def evaluate_generated_judgements(
    policy: Any,
    records: Sequence[dict[str, Any]],
    *,
    limit: int = 0,
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    """Generate on a held-out slice and report submit rate, pairwise and defect recall.

    A backend without ``generate`` is skipped, not failed, matching generation smoke.
    """

    generate: Callable[..., str] | None = getattr(policy, "generate", None)
    sample = list(records)
    if int(limit) > 0:
        sample = sample[: int(limit)]
    if not callable(generate):
        return {
            "pairs": len(sample),
            "scored": 0,
            "skipped": True,
            "reason": "backend_cannot_generate",
            "submit_rate": None,
            "pairwise_accuracy": 0.0,
            "diagnosis_recall": None,
            "items": [],
        }
    if not sample:
        raise ValueError("generation eval needs at least one record")
    items = []
    for record in sample:
        text = generate(record, max_new_tokens=int(max_new_tokens))
        items.append(score_generated_judgement(text, record))
    submitted = [item for item in items if item["submitted"]]
    pairwise = [item for item in items if item["has_pairwise"] and item["submitted"]]
    defect_items = [item for item in items if item["expected_defects"] and item["submitted"]]
    hits = sum(len(item["recalled_defects"]) for item in defect_items)
    expected = sum(len(item["expected_defects"]) for item in defect_items)
    return {
        "pairs": len(items),
        "scored": len(items),
        "skipped": False,
        "reason": "",
        "submit_rate": round(len(submitted) / len(items), 6) if items else 0.0,
        "pairwise_accuracy": round(sum(1 for item in pairwise if item["pairwise_ok"]) / len(pairwise), 6) if pairwise else 0.0,
        "diagnosis_recall": round(hits / expected, 6) if expected else None,
        "items": items,
    }
