"""Build DPO pairs from a policy's own generated judgements.

After SFT, preference learning has to rank the strings the model actually writes.
This module samples a few completions per prompt, labels them with the certified
pick and injected defects, and emits a DPO record when a correct writing can be
paired with an incorrect one. All-wrong prompts fall back to the grounded gold
JSON as chosen. All-correct prompts are dropped: they carry no preference signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

from ..utils import write_json
from .generate_eval import gold_judgement, score_generated_judgement
from .training import load_training_records


ONPOLICY_CONTRACT = "rfs.judge_onpolicy_dpo"


def _dump(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def generation_is_correct(item: dict[str, Any]) -> bool:
    """Return whether a scored generation matches the certified pick and defects."""

    if not item.get("submitted"):
        return False
    if item.get("has_pairwise") and not item.get("pairwise_ok"):
        return False
    if item.get("expected_defects") and not item.get("diagnosis_hit"):
        return False
    return True


def collect_onpolicy_pairs(
    policy: Any,
    dataset: str | Path | Sequence[dict[str, Any]],
    out: str | Path,
    *,
    samples: int = 2,
    max_new_tokens: int = 1024,
    limit: int = 0,
) -> dict[str, Any]:
    """Sample completions and write a DPO JSONL of model-written chosen/rejected pairs."""

    generate: Callable[..., str] | None = getattr(policy, "generate", None)
    if not callable(generate):
        raise ValueError("on-policy collection needs a backend that can generate")
    records = list(dataset) if isinstance(dataset, Sequence) and not isinstance(dataset, (str, Path)) else load_training_records(dataset)
    if int(limit) > 0:
        records = records[: int(limit)]
    pairs: list[dict[str, Any]] = []
    skipped_all_correct = 0
    used_gold_chosen = 0
    for record in records:
        scored = []
        for _ in range(max(1, int(samples))):
            text = generate(record, max_new_tokens=int(max_new_tokens))
            item = score_generated_judgement(text, record)
            item["text"] = text
            scored.append(item)
        correct = [item for item in scored if generation_is_correct(item)]
        wrong = [item for item in scored if item.get("submitted") and not generation_is_correct(item)]
        chosen_text = ""
        rejected_text = ""
        source = ""
        if correct and wrong:
            chosen_text = str(correct[0]["text"])
            rejected_text = str(wrong[0]["text"])
            source = "onpolicy"
        elif wrong:
            gold = gold_judgement(record)
            if not gold:
                continue
            chosen_text = str(record["chosen"])
            rejected_text = str(wrong[0]["text"])
            source = "gold_vs_onpolicy"
            used_gold_chosen += 1
        else:
            skipped_all_correct += 1
            continue
        if chosen_text == rejected_text:
            continue
        pairs.append({**record, "chosen": chosen_text, "rejected": rejected_text, "pair_source": source})
    dest = Path(out)
    target = dest if dest.suffix.lower() == ".jsonl" else dest / "dpo.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in pairs), encoding="utf-8")
    report = {
        "contract": ONPOLICY_CONTRACT,
        "out": str(target),
        "prompts": len(records),
        "pairs": len(pairs),
        "used_gold_chosen": used_gold_chosen,
        "skipped_all_correct": skipped_all_correct,
        "samples": int(samples),
        "ok": bool(pairs),
    }
    write_json(target.parent / "onpolicy_report.json", report)
    return report
