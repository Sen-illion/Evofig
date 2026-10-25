"""The judgement points a person is asked at, and the card each one leaves behind.

The loop already asks a seat which of two figures is better, and already asks it whether a layer is
finished. When a person is sitting in that seat, those questions are not answered by the model and
taken as read: they are written down as a card, the chain stops at that card, and the answer decides.

Two files are written for every question and neither is the other's copy. ``question.json`` is what a
person is shown: the two figures, the readings, the progress sheet -- and deliberately not the model's
own answer, so the answer a person gives is theirs. ``model_answer.json`` is what the model would have
said, kept as a shadow record whether or not a person ever replies; the pair of them is what makes
"how far apart were the model and the person" a number rather than an impression.

``answer.json`` is written by whoever answers, in the same shape the review channel already uses. A
question nobody answers inside ``hold_hours`` is not an error: the model's own answer is used, the
card says so, and the round's own account of itself records that the person was not there. The chain
is never blocked by a silence, and it never pretends a silence was a verdict.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ...utils import ensure_dir, read_json, write_json

#: The folder under a run root that holds every card, one directory per question.
HUMAN_QUEUE_DIR = "human_queue"
#: What a person is shown, what the model would have said, and what was actually answered.
QUESTION_FILE = "question.json"
MODEL_ANSWER_FILE = "model_answer.json"
ANSWER_FILE = "answer.json"
#: An answer left behind by an earlier attempt at the same card. A chain that restarts inside an
#: iteration poses the same questions again, and the answer a person gave about the figure they were
#: shown then must not be read as a verdict about the figure they are shown now.
STALE_ANSWER_PREFIX = "answer.stale."
QUESTION_CONTRACT = "rfs.human_question"
MODEL_ANSWER_CONTRACT = "rfs.human_model_answer"
ANSWER_CONTRACT = "rfs.human_answer"

#: The judgement points a person may sit at. They are the seats the loop already consults, named the
#: same way the review channel names its own kinds, so one answer shape serves all of them.
SEAT_KINDS = ("figure_choice", "layer_settled", "direction", "validation")
#: What happens when nobody answers. ``model`` uses the model's own answer and writes down that the
#: person was not there; ``hold`` keeps waiting, which no caller should ask for in a live chain.
TIMEOUT_POLICIES = ("model", "hold")

#: The keys an answer always has. Everything else a person writes lands in ``HumanAnswer.fields``,
#: so an answer can carry the extra questions one kind of card asks without this module having to
#: know what they are.
ANSWER_RESERVED_KEYS = ("contract", "version", "kind", "choice", "note", "answered_by", "at")


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class HumanAnswer:
    """What came back from one card: who answered, what they said, and how long it took."""

    answered: bool
    by: str
    choice: str = ""
    note: str = ""
    card: str = ""
    answer_path: str = ""
    waited_seconds: float = 0.0
    reason: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "answered": bool(self.answered),
            "by": str(self.by),
            "choice": str(self.choice),
            "note": str(self.note),
            "card": str(self.card),
            "answer_path": str(self.answer_path),
            "waited_seconds": round(float(self.waited_seconds), 3),
            "reason": str(self.reason),
            "fields": dict(self.fields),
        }


@dataclass
class HumanQueue:
    """The cards one run files, and the answers it reads back.

    ``on_question`` is called the moment a card is written, which is how a notification is sent: a
    caller that wants an email sends one from there, and a caller that wants nothing passes nothing.
    ``clock`` and ``sleeper`` are injected so a test can hold a card without waiting an hour for it.
    """

    root: str | Path
    seats: tuple[str, ...] = ("figure_choice",)
    hold_hours: float = 8.0
    on_timeout: str = "model"
    poll_seconds: float = 20.0
    enabled: bool = True
    on_question: Callable[[dict[str, Any]], None] | None = None
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep
    seen: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.root = Path(str(self.root or ""))
        self.seats = tuple(str(item) for item in (self.seats or ()) if str(item))
        self.on_timeout = str(self.on_timeout or "model").strip().casefold() or "model"
        self.hold_hours = max(0.0, _as_float(self.hold_hours, 8.0))
        self.poll_seconds = max(0.05, _as_float(self.poll_seconds, 20.0))

    def sits_at(self, kind: str) -> bool:
        """Whether a person is sitting at this judgement point."""

        return bool(self.enabled) and str(kind or "") in self.seats

    def card_dir(self, *, paper: str, iteration: int, round_index: int, kind: str) -> Path:
        """Where one question's card lives, keyed so two papers never share one."""

        return self.root / HUMAN_QUEUE_DIR / str(paper or "paper") / f"iter_{int(iteration):02d}" / f"round_{int(round_index):02d}" / str(kind or "question")

    def open_cards(self) -> list[dict[str, Any]]:
        """Every card that has a question and no answer yet, newest first."""

        open_ones: list[dict[str, Any]] = []
        for path in sorted(self.root.glob(f"{HUMAN_QUEUE_DIR}/*/iter_*/round_*/*/{QUESTION_FILE}")):
            if (path.parent / ANSWER_FILE).is_file():
                continue
            question = read_json(path) if path.is_file() else {}
            if isinstance(question, dict) and question:
                open_ones.append({"card": str(path.parent), **question})
        return list(reversed(open_ones))

    def answer(self, card: str | Path, *, choice: str, note: str = "", answered_by: str = "human", extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """File one answer into a card, refusing to overwrite one that is already there."""

        directory = Path(str(card))
        if not (directory / QUESTION_FILE).is_file():
            return {"ok": False, "reason": "no question at this card"}
        target = directory / ANSWER_FILE
        if target.is_file():
            existing = read_json(target)
            return {"ok": False, "reason": "this card is already answered", "answer": existing if isinstance(existing, dict) else {}}
        payload = {
            "contract": ANSWER_CONTRACT,
            "version": 1,
            "kind": str((read_json(directory / QUESTION_FILE) or {}).get("kind") or ""),
            "choice": str(choice or ""),
            "note": str(note or ""),
            "answered_by": str(answered_by or "human"),
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if extra:
            payload.update({str(key): value for key, value in extra.items()})
        write_json(target, payload)
        return {"ok": True, "answer": payload, "path": str(target)}

    def _rotate_stale_answer(self, directory: Path) -> str:
        """Set aside an answer left behind by an earlier attempt at this same card.

        A card is keyed by paper, iteration and round, and a chain that restarts inside an iteration
        poses those same questions again. The figures on the new card are not the figures the person
        answered about -- the candidate is rebuilt -- so reading the old answer would let a verdict
        about one figure decide a round about another, and a round decided that way would be filed as
        a judgement the person never made. The old answer is moved, never deleted: it stays beside the
        card as evidence of what the person said about the attempt that was abandoned.
        """

        target = directory / ANSWER_FILE
        if not target.is_file():
            return ""
        stamp = time.strftime("%Y%m%dT%H%M%S")
        stale = directory / f"{STALE_ANSWER_PREFIX}{stamp}.json"
        suffix = 1
        while stale.exists():
            suffix += 1
            stale = directory / f"{STALE_ANSWER_PREFIX}{stamp}_{suffix}.json"
        try:
            target.replace(stale)
        except OSError:  # noqa: BLE001 - a card that cannot be cleared must not stop the round
            return ""
        return str(stale)

    def ask(
        self,
        *,
        kind: str,
        paper: str,
        iteration: int,
        round_index: int,
        question: str,
        model_answer: dict[str, Any],
        stage: str = "",
        figures: dict[str, Any] | None = None,
        readings: dict[str, Any] | None = None,
        progress_sheet: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> HumanAnswer:
        """Write one card, wait for it, and say who answered.

        A question nobody is sitting at costs nothing: the model's answer is returned straight away
        and no card is written, which is what keeps a run with no person at the seat byte-identical
        to one from before this channel existed.
        """

        if not self.sits_at(kind):
            return HumanAnswer(answered=False, by="model", choice=str(model_answer.get("choice") or ""), reason="no person sits at this judgement point")

        directory = ensure_dir(self.card_dir(paper=paper, iteration=iteration, round_index=round_index, kind=kind))
        rotated = self._rotate_stale_answer(directory)
        shown = {
            "contract": QUESTION_CONTRACT,
            "version": 1,
            "kind": str(kind),
            "paper": str(paper),
            "iteration": int(iteration),
            "round": int(round_index),
            "stage": str(stage or ""),
            "question": str(question or ""),
            "figures": dict(figures or {}),
            "readings": dict(readings or {}),
            "progress_sheet": dict(progress_sheet or {}),
            "hold_hours": float(self.hold_hours),
        }
        if rotated:
            shown["rotated_previous_answer"] = rotated
        if extra:
            shown.update({str(key): value for key, value in extra.items()})
        write_json(directory / QUESTION_FILE, shown)
        shadow = {
            "contract": MODEL_ANSWER_CONTRACT,
            "version": 1,
            "kind": str(kind),
            "paper": str(paper),
            "iteration": int(iteration),
            "round": int(round_index),
            "choice": str(model_answer.get("choice") or ""),
            "reason": str(model_answer.get("reason") or ""),
            "answer": dict(model_answer),
        }
        write_json(directory / MODEL_ANSWER_FILE, shadow)
        self.seen.append(str(directory))
        if self.on_question is not None:
            try:
                self.on_question({"card": str(directory), **shown})
            except Exception:  # noqa: BLE001 - a notification that failed is not a question that failed
                pass

        answer_path = directory / ANSWER_FILE
        started = self.clock()
        deadline = started + (self.hold_hours * 3600.0)
        while True:
            if answer_path.is_file():
                payload = read_json(answer_path)
                if isinstance(payload, dict) and payload:
                    return HumanAnswer(
                        answered=True,
                        by=str(payload.get("answered_by") or "human"),
                        choice=str(payload.get("choice") or ""),
                        note=str(payload.get("note") or ""),
                        card=str(directory),
                        answer_path=str(answer_path),
                        waited_seconds=self.clock() - started,
                        fields={str(key): value for key, value in payload.items() if str(key) not in ANSWER_RESERVED_KEYS},
                    )
            now = self.clock()
            if self.on_timeout != "hold" and now >= deadline:
                return HumanAnswer(
                    answered=False,
                    by="timeout",
                    choice=str(model_answer.get("choice") or ""),
                    card=str(directory),
                    waited_seconds=now - started,
                    reason="nobody answered inside the hold",
                )
            self.sleeper(min(self.poll_seconds, max(0.05, deadline - now) if self.on_timeout != "hold" else self.poll_seconds))


__all__ = [
    "ANSWER_CONTRACT",
    "ANSWER_FILE",
    "ANSWER_RESERVED_KEYS",
    "HumanAnswer",
    "HumanQueue",
    "HUMAN_QUEUE_DIR",
    "MODEL_ANSWER_CONTRACT",
    "MODEL_ANSWER_FILE",
    "QUESTION_CONTRACT",
    "QUESTION_FILE",
    "SEAT_KINDS",
    "STALE_ANSWER_PREFIX",
    "TIMEOUT_POLICIES",
]
