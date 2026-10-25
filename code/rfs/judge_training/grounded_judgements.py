"""Rewrite DPO chosen/rejected JSON so the completion names this pair's evidence.

Stage-1 SFT has to supervise generation. The exported DPO completions were filled from a
defect table and a handful of structure phrases, so ranking them taught the policy to
prefer a template rather than to look at the figure. This module keeps the certified
pick and diagnosis type, and replaces the evidence sentence with tokens that belong to
this paper, this defect target and this figure letter. It never writes ``example_id``
into the completion: that token is absent at exam time.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

from ..utils import write_json
from .judge_schema import diagnosis_types, pairwise_choice
from .training import _completion_stats, load_training_records


GROUND_CONTRACT = "rfs.judge_grounded_dpo"
DEFAULT_MIN_UNIQUE_RATIO = 0.5
_TITLE_RE = re.compile(r'"title"\s*:\s*"((?:\\.|[^"\\])*)"')


def _parse_judgement(text: Any) -> dict[str, Any]:
    if isinstance(text, dict):
        return dict(text)
    try:
        value = json.loads(str(text or ""))
    except (TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _prompt_text(record: dict[str, Any]) -> str:
    for message in record.get("messages") or []:
        if not isinstance(message, dict):
            continue
        for item in message.get("content") or []:
            if isinstance(item, dict) and item.get("type") == "text":
                return str(item.get("text") or "")
    return str(record.get("prompt") or "")


def paper_title(record: dict[str, Any]) -> str:
    """Return the paper title already present in the Judge prompt, if any."""

    match = _TITLE_RE.search(_prompt_text(record))
    if not match:
        return str(record.get("paper_id") or "").strip()
    try:
        title = json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        title = match.group(1)
    return str(title or "").strip()


def defect_anchor(record: dict[str, Any]) -> dict[str, str]:
    """Return the certified defect type and target carried on the pair."""

    basis = record.get("preference_basis") if isinstance(record.get("preference_basis"), dict) else {}
    defects = [item for item in (basis.get("known_defects") or []) if isinstance(item, dict)]
    judgement = _parse_judgement(record.get("chosen"))
    types = [str(item.get("diagnosis_type") or "").strip() for item in defects if str(item.get("diagnosis_type") or "").strip()]
    named = diagnosis_types(judgement)
    diagnosis_type = next((item for item in types if item), "") or next((item for item in named if item), "")
    targets = [str(item.get("target") or "").strip() for item in defects if str(item.get("target") or "").strip()]
    diagnosis = judgement.get("diagnosis") if isinstance(judgement.get("diagnosis"), list) else []
    if not targets and diagnosis and isinstance(diagnosis[0], dict):
        targets = [str(diagnosis[0].get("target") or "").strip()]
    return {
        "diagnosis_type": diagnosis_type,
        "target": next((item for item in targets if item), "global_scope"),
    }


def _evidence_sentence(record: dict[str, Any], judgement: dict[str, Any], *, figure: str) -> str:
    anchor = defect_anchor(record)
    title = paper_title(record)
    kind = str(anchor["diagnosis_type"] or "quality difference")
    target = str(anchor["target"] or "global_scope")
    pick = pairwise_choice(judgement)
    parts = [f"{kind} on figure {figure or 'B'}", f"target {target}"]
    if pick in {"A", "B"}:
        parts.append(f"prefer {pick}")
    if title:
        parts.append(f"paper {title[:160]}")
    return "; ".join(parts)


def ground_judgement(record: dict[str, Any], judgement: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``judgement`` whose diagnosis evidence names this pair."""

    grounded = dict(judgement)
    diagnosis = [dict(item) for item in (judgement.get("diagnosis") or []) if isinstance(item, dict)]
    pick = pairwise_choice(judgement)
    worse = {"A": "B", "B": "A"}.get(pick, "")
    if not diagnosis:
        anchor = defect_anchor(record)
        diagnosis = [{
            "type": str(anchor["diagnosis_type"] or "assumed_worse"),
            "target": str(anchor["target"] or "global_scope"),
            "severity": "medium",
            "evidence": "",
        }]
        if worse in {"A", "B"}:
            diagnosis[0]["figure"] = worse
    for entry in diagnosis:
        figure = str(entry.get("figure") or worse or "")
        if figure in {"A", "B"}:
            entry["figure"] = figure
        if not str(entry.get("target") or "").strip():
            entry["target"] = defect_anchor(record)["target"]
        entry["evidence"] = _evidence_sentence(record, grounded, figure=figure)
    grounded["diagnosis"] = diagnosis
    return grounded


def _dump(judgement: dict[str, Any]) -> str:
    return json.dumps(judgement, ensure_ascii=False, separators=(",", ":"))


def ground_record(record: dict[str, Any]) -> dict[str, Any]:
    """Copy one DPO record with grounded chosen and rejected completions."""

    chosen = ground_judgement(record, _parse_judgement(record.get("chosen")))
    rejected = ground_judgement(record, _parse_judgement(record.get("rejected")))
    if _dump(chosen) == _dump(rejected):
        rejected = {**rejected, "preferred_candidate": {"A": "B", "B": "A"}.get(pairwise_choice(chosen), pairwise_choice(rejected))}
    return {**record, "chosen": _dump(chosen), "rejected": _dump(rejected)}


def unique_ratio(records: Sequence[dict[str, Any]], field: str = "chosen") -> float:
    """Return how many completions are unique, as a fraction of the slice."""

    texts = [str(record.get(field) or "") for record in records]
    if not texts:
        return 0.0
    return round(len(set(texts)) / len(texts), 6)


def ground_dpo_export(
    dataset: str | Path,
    out: str | Path,
    *,
    min_unique_ratio: float = DEFAULT_MIN_UNIQUE_RATIO,
) -> dict[str, Any]:
    """Rewrite one DPO JSONL so SFT supervises pair-specific evidence sentences."""

    records = [ground_record(row) for row in load_training_records(dataset)]
    ratio = unique_ratio(records)
    if ratio < float(min_unique_ratio):
        raise ValueError(
            f"grounded chosen unique ratio {ratio} is below {min_unique_ratio}: "
            "the completions are still templates"
        )
    dest = Path(out)
    target = dest if dest.suffix.lower() == ".jsonl" else dest / "dpo.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records),
        encoding="utf-8",
    )
    report = {
        "contract": GROUND_CONTRACT,
        "source": str(Path(dataset)),
        "out": str(target),
        "records": len(records),
        "unique_chosen_ratio": ratio,
        "min_unique_ratio": float(min_unique_ratio),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "data_report": _completion_stats(records),
        "ok": True,
    }
    write_json(target.parent / "ground_report.json", report)
    return report
