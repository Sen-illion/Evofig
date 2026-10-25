"""PaperQuiz, following Paper2Poster's released protocol.

The examiner reads the paper and writes two multiple-choice banks: 50 questions
about method structure and 50 about how that structure fits together. The
questions do not ask for the title, authors, or affiliations. Options are
shuffled and the answer
key is stored locally. Each reader then sees only the figure and the questions,
and returns a letter plus a region reference. ``NA`` is wrong. The score is the
fraction of letters that match the key. Scoring stays off unless
``RFS_PAPER_QUIZ=api``.

Paper2Poster averages six reader models and then applies a density multiplier
anchored to the median length of author-designed posters. This scorer records
one reader unless ``reader_models`` is passed. The density multiplier is left
empty: this benchmark has no author-poster length median.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable

from ..utils import write_json
from .score_model import resolve_score_model

PAPER_QUIZ_ENV_VAR = "RFS_PAPER_QUIZ"
PROMPT_VERSION = "paperquiz-paper2poster"
ANSWER_PROMPT_VERSION = "paperquiz-answer-v001"
QUESTION_TARGET = 50
SOURCE = "Paper2Poster"
SOURCE_READERS = (
    "LLaVA-OneVision-Qwen2-7B-ov-hf",
    "Phi-4-multimodal-instruct",
    "Llama-4-Scout-17B-16E-Instruct",
    "o3",
    "gpt-4o-mini",
    "gemini-2.0-flash",
)

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "editable-figure-bench" / "prompts"
_DETAIL_PROMPT_PATH = _PROMPT_DIR / "paperquiz-detail-v001.md"
_UNDERSTANDING_PROMPT_PATH = _PROMPT_DIR / "paperquiz-understanding-v001.md"
_ANSWER_PROMPT_PATH = _PROMPT_DIR / "paperquiz-answer-v001.md"
_BANKS = ("detail", "understanding")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _blank(prompt_sha: str, reason: str, model: str | None = None) -> dict[str, Any]:
    return {
        "summary": "PaperQuiz: multiple-choice questions from the paper, letters graded from the figure alone.",
        "source": SOURCE,
        "available": False,
        "accuracy": None,
        "detail_accuracy": None,
        "understanding_accuracy": None,
        "correct": None,
        "total": None,
        "density_augmented": None,
        "question_target": QUESTION_TARGET,
        "prompt_version": PROMPT_VERSION,
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "prompt_sha256": prompt_sha,
        "model": model,
        "source_readers": list(SOURCE_READERS),
        "reason": reason,
    }


def option_letter(text: str) -> str:
    """Return the leading A-D letter, or an empty string."""
    head = str(text).strip().split(".", 1)[0].strip().upper()
    if len(head) == 1 and head in "ABCD":
        return head
    return ""


def normalize_question(item: Any) -> dict[str, str] | None:
    """Keep one four-option item whose answer letter matches an option."""
    if not isinstance(item, dict):
        return None
    question = str(item.get("question") or "").strip()
    raw_options = item.get("options")
    if not question or not isinstance(raw_options, list) or len(raw_options) != 4:
        return None
    options: list[str] = []
    letters: list[str] = []
    for option in raw_options:
        text = str(option).strip()
        letter = option_letter(text)
        if not letter:
            return None
        body = text.split(". ", 1)[1] if ". " in text else text
        options.append(f"{letter}. {body}" if ". " not in text else text)
        letters.append(letter)
    if letters != ["A", "B", "C", "D"]:
        return None
    answer = str(item.get("answer") or "").strip()
    if option_letter(answer) not in letters:
        return None
    return {
        "aspect": str(item.get("aspect") or "").strip()[:1].upper(),
        "question": question,
        "options": options,
        "answer": answer,
    }


def parse_question_bank(payload: Any, *, limit: int = QUESTION_TARGET) -> dict[str, dict[str, Any]]:
    """Take up to ``limit`` valid questions, numbered Question 1 onward."""
    if isinstance(payload, dict) and isinstance(payload.get("questions"), dict):
        payload = payload["questions"]
    if not isinstance(payload, dict):
        return {}
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    for key, value in payload.items():
        item = normalize_question(value)
        if item is None:
            continue
        match = re.search(r"(\d+)", str(key))
        ranked.append((int(match.group(1)) if match else 10**9, str(key), item))
    ranked.sort(key=lambda row: (row[0], row[1]))
    return {f"Question {index}": item for index, (_, _, item) in enumerate(ranked[:limit], start=1)}


def shuffle_question_options(
    bank: dict[str, dict[str, Any]],
    rng: random.Random | None = None,
) -> dict[str, dict[str, Any]]:
    """Shuffle option text and move the answer letter with the correct text."""
    chooser = rng or random.Random()
    shuffled: dict[str, dict[str, Any]] = {}
    for key, item in bank.items():
        answer = str(item["answer"])
        answer_text = answer.split(". ", 1)[1] if ". " in answer else answer
        texts = [str(option).split(". ", 1)[1] if ". " in str(option) else str(option) for option in item["options"]]
        chooser.shuffle(texts)
        options: list[str] = []
        updated = answer
        for index, text in enumerate(texts):
            letter = "ABCD"[index]
            option = f"{letter}. {text}"
            options.append(option)
            if text == answer_text:
                updated = option
        shuffled[key] = {**item, "options": options, "answer": updated}
    return shuffled


def reader_questions(bank: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Questions sent with the figure. The answer key stays out."""
    return {
        key: {"question": item["question"], "options": list(item["options"])}
        for key, item in bank.items()
    }


def score_answer_bank(predicted: Any, bank: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Grade letters. A missing question, ``NA``, or an ``NA`` reference is wrong."""
    predicted_map = predicted if isinstance(predicted, dict) else {}
    correct = 0
    aspects: dict[str, dict[str, int]] = {}
    for key, gold in bank.items():
        aspect = str(gold.get("aspect") or "Unknown")
        bucket = aspects.setdefault(aspect, {"total": 0, "correct": 0})
        bucket["total"] += 1
        pred = predicted_map.get(key)
        if not isinstance(pred, dict):
            continue
        letter = str(pred.get("answer") or "").strip()
        reference = str(pred.get("reference") or "").strip()
        if not letter or letter.upper() == "NA" or reference.upper() == "NA":
            continue
        if letter[0].upper() == option_letter(str(gold.get("answer") or "")):
            correct += 1
            bucket["correct"] += 1
    total = len(bank)
    for bucket in aspects.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["total"], 4) if bucket["total"] else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else None,
        "aspects": aspects,
    }


def density_augmented_score(raw_accuracy: float, token_count: int, median_tokens: int) -> float | None:
    """Paper2Poster's length multiplier. ``median_tokens`` is their author-poster median."""
    if median_tokens <= 0:
        return None
    return round(float(raw_accuracy) * (1 + 1 / max(1, token_count / median_tokens)), 4)


def parse_paper_quiz(payload: Any) -> tuple[int, int] | None:
    """Accept a non-negative correct count that does not exceed total."""
    if not isinstance(payload, dict):
        return None
    correct = payload.get("correct")
    total = payload.get("total")
    if isinstance(correct, bool) or isinstance(total, bool):
        return None
    if not isinstance(correct, int) or not isinstance(total, int):
        return None
    if total <= 0 or correct < 0 or correct > total:
        return None
    return correct, total


def _question_file(preview: Path, questions_path: str | Path | None) -> Path:
    if questions_path is not None:
        return Path(questions_path)
    return preview.parent / "questions.json"


def load_quiz(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Read a frozen quiz. A missing file or the old string-list format is empty."""
    empty = {name: {} for name in _BANKS}
    if path is None or not Path(path).is_file():
        return empty
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(payload, dict):
        return empty
    loaded = {}
    for name in _BANKS:
        section = payload.get(name)
        loaded[name] = parse_question_bank(section if isinstance(section, dict) else {})
    return loaded


def _freeze_quiz(path: Path, banks: dict[str, dict[str, Any]]) -> None:
    write_json(
        path,
        {
            "source": SOURCE,
            "prompt_version": PROMPT_VERSION,
            **banks,
        },
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def write_paper_quiz(
    paper_text: str | None,
    questions_path: str | Path,
    *,
    spec: str | None = "api",
    call: Callable[..., dict[str, Any]] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Write the two question banks from the paper alone. No figure is read."""
    question_path = Path(questions_path)
    banks = load_quiz(question_path)
    if all(banks.values()):
        return {
            "ok": True,
            "skipped": True,
            "reason": "",
            "questions_path": str(question_path),
            "question_count": {name: len(banks[name]) for name in _BANKS},
        }
    resolved = str(spec if spec is not None else os.getenv(PAPER_QUIZ_ENV_VAR, "")).strip().lower()
    if resolved in {"", "off", "0", "false"}:
        return {"ok": False, "skipped": False, "reason": "disabled", "questions_path": str(question_path)}
    if resolved != "api":
        return {"ok": False, "skipped": False, "reason": "unsupported_spec", "questions_path": str(question_path)}
    if not str(paper_text or "").strip():
        return {"ok": False, "skipped": False, "reason": "missing paper text", "questions_path": str(question_path)}
    from ..providers.client import call_vlm_json, vlm_credentials_available

    chosen = model or resolve_score_model()
    if call is None and not vlm_credentials_available():
        return {"ok": False, "skipped": False, "reason": "missing API_BASE or API_KEY", "questions_path": str(question_path)}
    invoke = call or call_vlm_json
    prompts = {
        "detail": _DETAIL_PROMPT_PATH.read_text(encoding="utf-8"),
        "understanding": _UNDERSTANDING_PROMPT_PATH.read_text(encoding="utf-8"),
    }
    for name in _BANKS:
        if banks[name]:
            continue
        try:
            asked = invoke(prompts[name] + "\n\nPAPER:\n" + str(paper_text), [], chosen)
        except Exception as exc:
            return {
                "ok": False,
                "skipped": False,
                "reason": f"{type(exc).__name__}: {exc}"[:200],
                "questions_path": str(question_path),
            }
        bank = shuffle_question_options(parse_question_bank(asked))
        if not bank:
            return {
                "ok": False,
                "skipped": False,
                "reason": f"{name} questions were not returned",
                "questions_path": str(question_path),
            }
        banks[name] = bank
    _freeze_quiz(question_path, banks)
    return {
        "ok": True,
        "skipped": False,
        "reason": "",
        "questions_path": str(question_path),
        "question_count": {name: len(banks[name]) for name in _BANKS},
    }


def score_paper_quiz(
    preview_path: str | Path | None,
    paper_text: str | None,
    *,
    spec: str | None = None,
    call: Callable[..., dict[str, Any]] | None = None,
    render_error: str | None = None,
    questions_path: str | Path | None = None,
    reader_models: list[str] | None = None,
    allow_question_generation: bool = True,
) -> dict[str, Any]:
    """Score one preview. ``call`` replaces the provider, for tests.

    Question calls see only the paper. Answer calls see only the preview and
    the questions, never the answer key. A frozen quiz skips the paper calls.
    """
    detail_prompt = _DETAIL_PROMPT_PATH.read_text(encoding="utf-8")
    prompt_sha = _sha(detail_prompt + _UNDERSTANDING_PROMPT_PATH.read_text(encoding="utf-8"))
    if render_error:
        return _blank(prompt_sha, "render_failed: " + render_error)
    resolved = str(spec if spec is not None else os.getenv(PAPER_QUIZ_ENV_VAR, "")).strip().lower()
    if resolved in {"", "off", "0", "false"}:
        return _blank(prompt_sha, "disabled")
    if resolved != "api":
        return _blank(prompt_sha, "unsupported_spec")
    path = Path(preview_path) if preview_path else Path()
    if not path.is_file():
        return _blank(prompt_sha, "missing preview.png")
    question_path = _question_file(path, questions_path)
    banks = load_quiz(question_path)
    from ..providers.client import call_vlm_json, vlm_credentials_available

    model = resolve_score_model()
    if call is None and not vlm_credentials_available():
        return _blank(prompt_sha, "missing API_BASE or API_KEY", model)
    invoke = call or call_vlm_json
    if not any(banks.values()):
        if not allow_question_generation:
            return _blank(prompt_sha, "questions were not returned", model)
        written = write_paper_quiz(
            paper_text,
            question_path,
            spec="api",
            call=invoke,
            model=model,
        )
        if not written.get("ok"):
            return _blank(prompt_sha, str(written.get("reason") or "questions were not returned"), model)
        banks = load_quiz(question_path)
    readers = [item for item in (reader_models or [model]) if str(item).strip()]
    answer_prompt = _ANSWER_PROMPT_PATH.read_text(encoding="utf-8")
    per_reader: dict[str, dict[str, Any]] = {}
    for reader in readers:
        per_reader[reader] = {}
        for name in _BANKS:
            if not banks[name]:
                continue
            visible = json.dumps(reader_questions(banks[name]), ensure_ascii=False)
            try:
                answered = invoke(answer_prompt + "\n\nQUESTIONS:\n" + visible, [path], reader)
            except Exception as exc:
                return _blank(prompt_sha, f"{type(exc).__name__}: {exc}"[:200], reader)
            if not isinstance(answered, dict):
                return _blank(prompt_sha, "response is not a letter for each question", reader)
            per_reader[reader][name] = score_answer_bank(answered, banks[name])
    detail_scores = [item["detail"]["accuracy"] for item in per_reader.values() if item.get("detail")]
    understanding_scores = [
        item["understanding"]["accuracy"] for item in per_reader.values() if item.get("understanding")
    ]
    detail_accuracy = _mean([value for value in detail_scores if value is not None])
    understanding_accuracy = _mean([value for value in understanding_scores if value is not None])
    correct = sum(item[name]["correct"] for item in per_reader.values() for name in _BANKS if name in item)
    total = sum(item[name]["total"] for item in per_reader.values() for name in _BANKS if name in item)
    present = [value for value in (detail_accuracy, understanding_accuracy) if value is not None]
    report = _blank(prompt_sha, "", model)
    report["available"] = bool(present)
    report["detail_accuracy"] = detail_accuracy
    report["understanding_accuracy"] = understanding_accuracy
    report["accuracy"] = _mean(present)
    report["correct"] = correct
    report["total"] = total
    report["question_count"] = {name: len(banks[name]) for name in _BANKS}
    report["questions_path"] = str(question_path)
    report["readers"] = readers
    report["per_reader"] = per_reader
    if not present:
        report["reason"] = "response is not a letter for each question"
    return report
