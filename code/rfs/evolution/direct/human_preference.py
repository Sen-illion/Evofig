"""The human-preference channel: what a person says about one comparison, kept as a test item.

The trained seats answer which of two figures is wrong and what is wrong with it.  Nothing in
the loop can say whether that answer was *right*: the deterministic gate reads a different
ruler, and the reflection only ever restates the measurement that was already taken.  A person
can.  This module keeps one person's verdict on one comparison -- which figure is the worse
one, what the defect is called, what the seat had answered -- and renders it in the same shape
the choice seat is trained on, so a review becomes training material instead of a note in a
chat.

Three rules hold everywhere here.

* A review never rewrites the round it is about.  The round keeps the decision it made; the
  review is filed beside it, and a disagreement stays a disagreement.
* A review names the two images it judged by content hash, not only by path.  The path is what
  the page served; the hash is what the person actually looked at, and the two can drift when a
  run is re-rendered.
* A review adds no reader to the loop.  It changes no prompt, no patch and no decision; the
  only thing it can become is a training row.

The queue this module builds is read off the run tree itself: every round that asked the two
figure question is one reviewable item, carrying the two images that were compared and the
answer the seat gave.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...contracts.evolution import STAGE_ORDER
from ...utils import read_json, write_json
from .file_lock import append_jsonl, exclusive_lock

HUMAN_PREFERENCE_CONTRACT = "rfs.human_preference"
HUMAN_PREFERENCE_VERSION = 1

#: What a review may answer.  A preference says which figure is better; the two extra answers
#: exist because a person looking at two figures is often sure that neither is good, and a
#: question with only A and B would force that judgement into a lie.
HUMAN_CHOICES = ("A", "B", "tie", "both_bad")
#: The answers that name a better figure, and so can become a training row.
PREFERENCE_CHOICES = ("A", "B")
#: Which figure the seat itself picked, in the round's own words.
MODEL_CHOICES = ("A", "B")

#: The three things a person may file through this channel. A figure choice answers the seat's own
#: question and becomes a training row; the two others are instructions about the work rather than
#: about one comparison -- a layer that is finished with, and a preference that should steer the
#: rounds that follow.
REVIEW_KINDS = ("figure_choice", "layer_settled", "preference")
DEFAULT_REVIEW_KIND = "figure_choice"

#: Where a run keeps the verdicts a person filed about it, relative to the run root. The loop reads
#: this folder at the start of every round; whoever answers a person writes it. Nothing else reads
#: it, and the loop reads it exactly once per verdict.
HUMAN_REVIEW_DIR = "human_review"
#: What a preference entry is. It has the same shape as a suggestion rule and a prompt patch -- the
#: stage it applies at, the condition it applies under, and the sentence it wants written -- because
#: it goes to the same place they do: a later round's own question.
HUMAN_PREFERENCE_ENTRY_CONTRACT = "rfs.human_preference_entry"
HUMAN_PREFERENCE_ENTRY_VERSION = 1
ENTRIES_FILE = "human_preference_entries.json"
#: An entry is a trial until a round has been measured under it, and a standing instruction after
#: that. Only standing entries are handed to every round; a trial is handed to the round that
#: follows the person's verdict, and that round is what decides whether it stands.
PREFERENCE_STATUSES = ("candidate", "resident")

REVIEW_FILE = "human_preferences.jsonl"
STATE_FILE = "human_preference_state.json"
ITEMS_FILE = "items.json"
IMAGE_DIR = "images"
#: The renders a candidate directory may hold, most final first.  The same order the run uses,
#: so the review shows the figure the round was judged on.
PREVIEW_NAMES = ("final_draft_preview.png", "draft_preview.png", "preview.png")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail_number(name: str, prefix: str) -> int | None:
    """Read the ordinal out of ``iter_07`` / ``round_12``; anything else is not one of those."""

    head, _, tail = str(name).partition("_")
    if head != prefix or not tail.isdigit():
        return None
    return int(tail)


def preview_of(directory: Path | None) -> Path | None:
    """The render one candidate directory holds, or ``None`` when it holds none."""

    if directory is None:
        return None
    for name in PREVIEW_NAMES:
        candidate = Path(directory) / name
        if candidate.exists():
            return candidate
    return None


def first_candidate_dir(round_dir: Path) -> Path | None:
    """The one candidate directory a round compiled, whichever number it carries."""

    candidates = sorted(
        (item for item in Path(round_dir).glob("candidate_*") if item.is_dir()),
        key=lambda item: _tail_number(item.name, "candidate") or 0,
    )
    return candidates[0] if candidates else None


def _iteration_dirs(root: Path) -> list[Path]:
    found: list[tuple[int, Path]] = []
    for path in sorted(Path(root).glob("**/iter_*")):
        if not path.is_dir():
            continue
        number = _tail_number(path.name, "iter")
        if number is not None:
            found.append((number, path))
    found.sort(key=lambda item: (item[0], str(item[1])))
    return [path for _, path in found]


def _round_dirs(iteration: Path) -> list[Path]:
    found: list[tuple[int, Path]] = []
    for path in sorted(iteration.glob("round_*")):
        if not path.is_dir():
            continue
        number = _tail_number(path.name, "round")
        if number is not None:
            found.append((number, path))
    found.sort(key=lambda item: item[0])
    return [path for _, path in found]


def _paper_of(iteration: Path) -> str:
    return iteration.parent.name


def _decision_of(round_dir: Path) -> dict[str, Any]:
    path = round_dir / "repair_decision.json"
    data = read_json(path) if path.exists() else {}
    return data if isinstance(data, dict) else {}


def _pairwise_of(decision: dict[str, Any]) -> dict[str, Any]:
    pairwise = decision.get("pairwise")
    return pairwise if isinstance(pairwise, dict) else {}


def pairwise_choice(pairwise: dict[str, Any]) -> str:
    """Which figure the seat preferred, whichever of its two homes the record carries.

    The round's decision keeps the comparison under one key and the seat's own answer under a
    nested one, and both were written at different times, so a reader that knows only one of them
    reads half the rounds as having answered nothing.
    """

    source = pairwise if isinstance(pairwise, dict) else {}
    nested = source.get("pairwise") if isinstance(source.get("pairwise"), dict) else {}
    for value in (source.get("preferred_candidate"), nested.get("choice"), source.get("choice")):
        text = str(value or "").strip()
        if text in MODEL_CHOICES:
            return text
    return ""


def collect_review_items(root: str | Path) -> list[dict[str, Any]]:
    """Every round in a run tree that asked the two-figure question, oldest first.

    The two images are the ones the comparison itself named: the state the loop had already
    accepted, and the state the round produced.  A round that asked no such question -- a first
    round with nothing to compare against, a round whose seat was unavailable -- is not in the
    queue, because there is no question for a person to answer either.

    Which round a baseline belongs to is read from the tree rather than assumed: a round that
    was accepted becomes the next round's "already accepted" figure, and the last accepted
    figure of an iteration carries into the next one.  Rounds carry the source they were read
    from, so a page can say where the left-hand figure came from instead of presenting a guess
    as a fact.
    """

    items: list[dict[str, Any]] = []
    carried: dict[str, Any] = {}
    for iteration in _iteration_dirs(Path(root)):
        carried = {"preview": carried.get("preview"), "source": carried.get("source") or "", "score": carried.get("score")}
        accepted_dir: Path | None = None
        accepted_score: Any = carried.get("score")
        for round_dir in _round_dirs(iteration):
            decision = _decision_of(round_dir)
            pairwise = _pairwise_of(decision)
            after_dir = first_candidate_dir(round_dir)
            after_preview = preview_of(after_dir)
            if accepted_dir is not None:
                before_preview = preview_of(accepted_dir)
                before_source = "the round before this one, accepted"
            else:
                before_preview = carried.get("preview")
                before_source = str(carried.get("source") or "")
            # The figure this round produced becomes the next round's "already accepted" one as
            # soon as it was accepted, whether or not this round is itself reviewable.  A round
            # with nothing to compare against still moves the baseline, and losing that is how a
            # whole queue behind one unanswered round goes empty.
            if bool(decision.get("accepted")) and after_dir is not None:
                accepted_dir = after_dir
                accepted_score = decision.get("after_score")
            if not pairwise or after_preview is None or before_preview is None:
                continue
            item = {
                "item_id": f"{_paper_of(iteration)}__{iteration.name}__{round_dir.name}",
                "paper": _paper_of(iteration),
                "iteration": _tail_number(iteration.name, "iter"),
                "round": _tail_number(round_dir.name, "round"),
                "accepted": bool(decision.get("accepted")),
                "decision_reason": str(decision.get("reason") or ""),
                "comparison": str(decision.get("comparison") or ""),
                "model_choice": pairwise_choice(pairwise),
                "model_margin": str(pairwise.get("margin") or ""),
                "model_defect": str(pairwise.get("defect") or ""),
                "model_wrong_figure": str(pairwise.get("wrong_figure") or ""),
                "candidate_a": str(pairwise.get("candidate_a") or ""),
                "candidate_b": str(pairwise.get("candidate_b") or ""),
                "score_before": decision.get("before_score"),
                "score_after": decision.get("after_score"),
                "new_blocking_defects": list(decision.get("new_blocking_defects") or []),
                "images": {"A": str(before_preview), "B": str(after_preview)},
                "image_hashes": {"A": _sha256(Path(before_preview)), "B": _sha256(Path(after_preview))},
                "before_source": before_source,
            }
            items.append(item)
        carried = {
            "preview": preview_of(accepted_dir) if accepted_dir is not None else carried.get("preview"),
            "source": "the last round of the previous iteration, accepted" if accepted_dir is not None else str(carried.get("source") or ""),
            "score": accepted_score,
        }
    return items


def write_review_bundle(bundle: str | Path, root: str | Path, *, limit: int = 0) -> dict[str, Any]:
    """Stage the queue and its images under one directory a page can serve.

    The images are copied rather than pointed at: a review is often done on a different machine
    from the one that ran the pipeline, and a queue whose pictures live on a machine the reader
    is not on is a queue nobody reads.  ``limit`` keeps the newest N items, which is what makes
    a first pass cheap on a run with hundreds of rounds behind it.
    """

    directory = Path(bundle)
    items = collect_review_items(root)
    if limit and limit > 0:
        items = items[-int(limit):]
    images_dir = directory / IMAGE_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    staged: list[dict[str, Any]] = []
    for item in items:
        target_dir = images_dir / item["item_id"]
        target_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for name in ("A", "B"):
            source = Path(item["images"][name])
            destination = target_dir / f"{name}.png"
            if not destination.exists() or _sha256(destination) != item["image_hashes"][name]:
                destination.write_bytes(source.read_bytes())
            paths[name] = f"{IMAGE_DIR}/{item['item_id']}/{name}.png"
        staged.append({**item, "images": paths})
    payload = {
        "contract": HUMAN_PREFERENCE_CONTRACT,
        "version": HUMAN_PREFERENCE_VERSION,
        "run_root": str(Path(root).resolve()),
        "items": staged,
    }
    write_json(directory / ITEMS_FILE, payload)
    return payload


def review_kind(review: Any) -> str:
    """Which of the three things a row is. A row written before the kinds existed is a choice."""

    source = review if isinstance(review, dict) else {}
    return str(source.get("kind") or DEFAULT_REVIEW_KIND).strip() or DEFAULT_REVIEW_KIND


def validate_human_review(review: dict[str, Any], *, defect_menu: Any = ()) -> dict[str, Any]:
    """Check one review before it is written.  Nothing is repaired here; it is refused."""

    errors: list[str] = []
    if not isinstance(review, dict):
        return {"ok": False, "errors": ["a review is an object"]}
    kind = review_kind(review)
    if kind not in REVIEW_KINDS:
        return {"ok": False, "errors": ["kind must be one of " + ", ".join(REVIEW_KINDS)]}
    reviewer = str(review.get("reviewer") or "").strip()
    if not reviewer:
        errors.append("a review must name who made it")
    if kind == "figure_choice":
        if not str(review.get("item_id") or "").strip():
            errors.append("a review must name the item it is about")
        choice = str(review.get("human_choice") or "").strip()
        if choice not in HUMAN_CHOICES:
            errors.append("human_choice must be one of " + ", ".join(HUMAN_CHOICES))
        defect = str(review.get("human_defect") or "").strip()
        menu = {str(item) for item in defect_menu} if defect_menu else set()
        if defect and menu and defect not in menu:
            errors.append("human_defect must be one of " + ", ".join(sorted(menu)))
        if choice in PREFERENCE_CHOICES and not defect and not str(review.get("human_note") or "").strip():
            errors.append("a preference must say what is wrong with the worse figure, or say it in a note")
    elif kind == "layer_settled":
        if not str(review.get("paper") or "").strip():
            errors.append("a verdict about a layer must say which paper it is about")
        stage = str(review.get("stage") or "").strip()
        if stage not in STAGE_ORDER:
            errors.append("a settled layer must name a stage of the pipeline: " + ", ".join(STAGE_ORDER))
        if not str(review.get("human_note") or "").strip():
            errors.append("a settled layer must say, in a note, what makes it finished with")
    else:
        if not str(review.get("paper") or "").strip():
            errors.append("a preference must say which paper it is about")
        apply_at = str(review.get("apply_at") or "").strip()
        if apply_at and apply_at not in STAGE_ORDER:
            errors.append("apply_at must name a stage of the pipeline when it names one: " + ", ".join(STAGE_ORDER))
        if not str(review.get("directive") or "").strip():
            errors.append("a preference must carry the sentence it wants written")
        condition = review.get("condition")
        if condition is not None and not isinstance(condition, dict):
            errors.append("condition is an object when there is one")
        dimensions = review.get("dimensions")
        if dimensions is not None and not isinstance(dimensions, dict):
            errors.append("dimensions name one reading at a time, as name: choice")
        else:
            for name, choice in sorted((dimensions or {}).items()):
                if str(choice) not in HUMAN_CHOICES:
                    errors.append(f"dimension {name} must be answered " + ", ".join(HUMAN_CHOICES))
    return {"ok": not errors, "errors": errors}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def review_id_for(review: dict[str, Any]) -> str:
    """A stable id for one review, so the same verdict filed twice is the same row."""

    material = "|".join(
        str(review.get(key) or "")
        for key in (
            "kind",
            "paper",
            "item_id",
            "human_choice",
            "human_defect",
            "stage",
            "apply_at",
            "directive",
            "reviewer",
            "created_at",
        )
    )
    return "human-review-" + hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def entry_id_for(entry: dict[str, Any]) -> str:
    """A stable id for one preference, so the same sentence filed twice is one entry."""

    material = "|".join(
        str(entry.get(key) or "")
        for key in ("source", "apply_at", "directive", "condition", "reviewer", "created_at")
    )
    return "human-pref-" + hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def preference_entry(review: dict[str, Any], *, status: str = "candidate") -> dict[str, Any]:
    """Render one person's preference in the shape a suggestion rule and a patch share.

    The shape is deliberate: apply_at says which stage it is about, condition says when it
    applies, directive is the sentence that stage's prompt should have carried. What differs from
    a patch is only where it came from and how it is paid for -- a person wrote it, so it costs
    nothing to try and is tried as soon as it is filed.
    """

    source = review if isinstance(review, dict) else {}
    condition = source.get("condition") if isinstance(source.get("condition"), dict) else {}
    entry = {
        "contract": HUMAN_PREFERENCE_ENTRY_CONTRACT,
        "version": HUMAN_PREFERENCE_ENTRY_VERSION,
        "entry_id": "",
        "source": "human",
        "kind": "preference",
        "status": str(status) if str(status) in PREFERENCE_STATUSES else "candidate",
        "apply_at": str(source.get("apply_at") or "").strip(),
        "condition": dict(condition),
        "directive": str(source.get("directive") or "").strip(),
        "dimensions": {str(key): str(value) for key, value in (source.get("dimensions") or {}).items()},
        "paper": str(source.get("paper") or ""),
        "review_id": str(source.get("review_id") or ""),
        "reviewer": str(source.get("reviewer") or ""),
        "created_at": str(source.get("created_at") or ""),
        "promoted_at": "",
        "promoted_by": "",
        "promote_because": "",
    }
    entry["entry_id"] = entry_id_for(entry)
    return entry


def validate_preference_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Refuse an entry that names no stage, writes no sentence, or is not one of the two states."""

    errors: list[str] = []
    if str(entry.get("apply_at") or "") not in STAGE_ORDER:
        errors.append("apply_at must name a stage of the pipeline: " + ", ".join(STAGE_ORDER))
    if not str(entry.get("directive") or "").strip():
        errors.append("a preference entry must carry the sentence it wants written")
    if str(entry.get("status") or "") not in PREFERENCE_STATUSES:
        errors.append("status must be one of " + ", ".join(PREFERENCE_STATUSES))
    if not isinstance(entry.get("condition"), dict):
        errors.append("condition is an object")
    return {"ok": not errors, "errors": errors}


def unconsumed_verdicts(rows: Any, seen: Any, *, paper: str = "") -> list[dict[str, Any]]:
    """The instructions one paper has not read yet, oldest first.

    A verdict is read once and once only: a person who says "this layer is finished" has said it
    once, and a loop that re-reads the same sentence every round would turn one answer into a
    standing order nobody gave. Oldest first, so two answers filed in one wave are read in the
    order they were written.

    A run root holds every paper the loop is drawing, so a verdict that names no paper -- a
    comparison, which is about one round and carries the round's own id -- is not one of these. A
    verdict that names a paper is only ever read by that paper: one person saying "the structure is
    finished" about one figure must not settle the same layer in the two beside it.
    """

    known = {str(item) for item in (seen or [])}
    wanted = str(paper or "")
    out: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, (list, tuple)) else []:
        if review_kind(row) == DEFAULT_REVIEW_KIND:
            continue
        if wanted and str(row.get("paper") or "") != wanted:
            continue
        review_id = str(row.get("review_id") or "")
        if not review_id or review_id in known:
            continue
        out.append(dict(row))
    return out


class HumanReviewStore:
    """The reviews one queue holds, as an append-only log plus the latest verdict per item.

    Append-only is the point: a person who changes their mind has changed it, and the earlier
    verdict is evidence about how hard the question is rather than a mistake to erase.  The
    state file is what a page reads to show what has already been answered.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.log_path = self.directory / REVIEW_FILE
        self.state_path = self.directory / STATE_FILE
        self.entries_path = self.directory / ENTRIES_FILE

    def rows(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                out.append(row)
        return out

    def state(self) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        for row in self.rows():
            latest[str(row.get("item_id") or "")] = row
        return {
            "contract": HUMAN_PREFERENCE_CONTRACT,
            "version": HUMAN_PREFERENCE_VERSION,
            "reviewed": len(latest),
            "items": latest,
            # How many standing instructions and how many trials this channel is holding. A round
            # reads this to say what it ran under, not to decide anything with it.
            "entries": len(self.entries()),
            "resident": len([row for row in self.entries() if str(row.get("status") or "") == "resident"]),
        }

    def record(self, review: dict[str, Any], *, defect_menu: Any = ()) -> dict[str, Any]:
        payload = dict(review) if isinstance(review, dict) else {}
        payload.setdefault("created_at", _now())
        payload.setdefault("reviewer", "")
        payload.setdefault("kind", DEFAULT_REVIEW_KIND)
        resident = bool(payload.pop("resident", False))
        validation = validate_human_review(payload, defect_menu=defect_menu)
        if not validation["ok"]:
            return {"ok": False, "reason": "contract_validation_failed", "errors": list(validation["errors"])}
        payload["contract"] = HUMAN_PREFERENCE_CONTRACT
        payload["version"] = HUMAN_PREFERENCE_VERSION
        payload["review_id"] = review_id_for(payload)
        self.directory.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(self.log_path):
            append_jsonl(self.log_path, payload)
        entry = {}
        if review_kind(payload) == "preference":
            # The log keeps what the person said; the entries file keeps what is in force. A
            # preference starts as a trial unless the person said it stands.
            entry = self.install_entry(payload, status="resident" if resident else "candidate")
        write_json(self.state_path, self.state())
        return {"ok": True, "review": payload, **({"entry": entry} if entry else {})}

    def verdicts(self) -> list[dict[str, Any]]:
        """Every row that is an instruction about the work rather than a comparison of two figures."""

        return [row for row in self.rows() if review_kind(row) != DEFAULT_REVIEW_KIND]

    def entries(self) -> list[dict[str, Any]]:
        """The preference entries this channel holds, in the order they were first filed."""

        payload = read_json(self.entries_path) if self.entries_path.exists() else {}
        rows = payload.get("entries") if isinstance(payload, dict) else None
        return [dict(row) for row in (rows or []) if isinstance(row, dict)]

    def install_entry(self, review: dict[str, Any], *, status: str = "candidate") -> dict[str, Any]:
        """Write one preference into the entries file, or return the one already there."""

        entry = preference_entry(review, status=status)
        validation = validate_preference_entry(entry)
        if not validation["ok"]:
            return {}
        entries = self.entries()
        for index, row in enumerate(entries):
            if str(row.get("entry_id") or "") == entry["entry_id"]:
                # The same sentence filed twice is the same entry: the person changed nothing, so
                # nothing here changes either.
                return entries[index]
        entries.append(entry)
        self._write_entries(entries)
        return entry

    def promote(self, entry_id: str, *, by: str = "human", because: str = "") -> dict[str, Any]:
        """Make a trial entry stand, or say it already did."""

        entries = self.entries()
        for index, row in enumerate(entries):
            if str(row.get("entry_id") or "") != str(entry_id):
                continue
            if str(row.get("status") or "") == "resident":
                return {"ok": True, "changed": False, "entry": dict(row)}
            entries[index] = {
                **row,
                "status": "resident",
                "promoted_at": _now(),
                "promoted_by": str(by),
                "promote_because": str(because or ""),
            }
            self._write_entries(entries)
            return {"ok": True, "changed": True, "entry": dict(entries[index])}
        return {"ok": False, "changed": False, "reason": "no entry with that id"}

    def _write_entries(self, entries: list[dict[str, Any]]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(
            self.entries_path,
            {
                "contract": HUMAN_PREFERENCE_ENTRY_CONTRACT,
                "version": HUMAN_PREFERENCE_ENTRY_VERSION,
                "entries": entries,
            },
        )


def review_to_choice_item(item: dict[str, Any], review: dict[str, Any], *, defect_menu: Any) -> dict[str, Any] | None:
    """Render one review as a choice item, or ``None`` when it is not one.

    The seat's question is "which figure is wrong, and what is wrong with it", so a review turns
    into a training row only when it answers both: a preference between the two figures, and a
    named defect.  "Two good figures" and "two bad figures" are real answers and stay in the
    review log, but they are not answers to this question and are not written here as if they
    were.
    """

    choice = str(review.get("human_choice") or "")
    defect = str(review.get("human_defect") or "").strip()
    menu = [str(entry) for entry in defect_menu]
    if choice not in PREFERENCE_CHOICES or not defect or not menu:
        return None
    if defect not in menu:
        return None
    # The person named the better figure; the question asks for the worse one.
    wrong = "B" if choice == "A" else "A"
    images = [str(item["images"]["A"]), str(item["images"]["B"])]
    prompt = (
        "Look at figure A and figure B. Choose which figure is wrong and which defect it has. "
        "wrong_figure must be A or B. "
        "defect must be one of: " + ", ".join(menu) + ". "
        'Reply with only {"wrong_figure":"A or B","defect":"one listed option"}. '
        f"Paper: {item.get('paper')}\n"
        f"Round: iteration {item.get('iteration')} round {item.get('round')}"
    )
    distractor = next((entry for entry in menu if entry != defect), defect)
    other = "A" if wrong == "B" else "B"
    return {
        "schema_version": "rfs.judge_choice_item",
        "example_id": f"human__{item['item_id']}",
        "pair_id": f"human__{item['item_id']}",
        "paper_id": str(item.get("paper") or ""),
        "domain": "",
        "family": "human_review",
        "image_kind": "gold_in_A" if wrong == "B" else "gold_in_B",
        "defect": defect,
        "defect_menu": menu,
        "gold_figure": wrong,
        "gold_defect": defect,
        "images": images,
        "chosen": json.dumps({"wrong_figure": wrong, "defect": defect}, ensure_ascii=False, separators=(",", ":")),
        "rejected": json.dumps({"wrong_figure": other, "defect": distractor}, ensure_ascii=False, separators=(",", ":")),
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": images[0]}, {"type": "image", "image": images[1]}]}],
        "provenance": {
            "source": "human_preference",
            "review_id": str(review.get("review_id") or ""),
            "reviewer": str(review.get("reviewer") or ""),
            "reviewed_at": str(review.get("created_at") or ""),
            "model_choice": str(item.get("model_choice") or ""),
            "model_defect": str(item.get("model_defect") or ""),
            "image_hashes": dict(item.get("image_hashes") or {}),
            "note": str(review.get("human_note") or ""),
        },
    }


def export_choice_items(items: Any, rows: Any, *, defect_menu: Any) -> dict[str, Any]:
    """Render every review that answers the question, plus what could not be rendered."""

    by_id = {str(item.get("item_id") or ""): item for item in items if isinstance(item, dict)}
    chosen: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, (list, tuple)) else []:
        item = by_id.get(str(row.get("item_id") or ""))
        if item is None:
            skipped.append({"review_id": row.get("review_id"), "item_id": row.get("item_id"), "reason": "the queue no longer holds this item"})
            continue
        rendered = review_to_choice_item(item, row, defect_menu=defect_menu)
        if rendered is None:
            skipped.append({"review_id": row.get("review_id"), "item_id": row.get("item_id"), "reason": "this answer is not a figure/defect choice"})
            continue
        chosen.append(rendered)
    return {"items": chosen, "skipped": skipped}


def human_review_stats(items: Any, rows: Any) -> dict[str, Any]:
    """How much of the queue is reviewed, and how often the person and the seat disagreed.

    The disagreement list is the point of the whole channel: every row in it is a comparison the
    seat answered one way and a person answered the other, which is exactly the material a
    later training round needs and the loop cannot produce on its own.
    """

    by_id = {str(item.get("item_id") or ""): item for item in items if isinstance(item, dict)}
    latest: dict[str, dict[str, Any]] = {}
    instructions = 0
    for row in rows if isinstance(rows, (list, tuple)) else []:
        if review_kind(row) != DEFAULT_REVIEW_KIND:
            # A preference and a settled layer are about the work, not about one comparison, so
            # they are not answers to the queue this report counts.
            instructions += 1
            continue
        latest[str(row.get("item_id") or "")] = row
    agreed: list[str] = []
    disagreed: list[dict[str, Any]] = []
    ties: list[str] = []
    both_bad: list[str] = []
    for item_id, row in latest.items():
        item = by_id.get(item_id) or {}
        choice = str(row.get("human_choice") or "")
        if choice == "tie":
            ties.append(item_id)
            continue
        if choice == "both_bad":
            both_bad.append(item_id)
            continue
        model_choice = str(item.get("model_choice") or "")
        if not model_choice:
            continue
        if choice == model_choice:
            agreed.append(item_id)
        else:
            disagreed.append(
                {
                    "item_id": item_id,
                    "model_choice": model_choice,
                    "model_defect": str(item.get("model_defect") or ""),
                    "human_choice": choice,
                    "human_defect": str(row.get("human_defect") or ""),
                    "human_note": str(row.get("human_note") or ""),
                    "images": dict(item.get("images") or {}),
                }
            )
    decided = len(agreed) + len(disagreed)
    return {
        "contract": "rfs.human_preference_stats",
        "version": HUMAN_PREFERENCE_VERSION,
        "queue": len(by_id),
        "reviewed": len(latest),
        "unreviewed": max(0, len(by_id) - len(latest)),
        "decided": decided,
        "agreed": len(agreed),
        "disagreed": len(disagreed),
        "agreement_rate": round(len(agreed) / decided, 4) if decided else None,
        "ties": len(ties),
        "both_bad": len(both_bad),
        "instructions": instructions,
        "model_wrong": disagreed,
    }


__all__ = [
    "DEFAULT_REVIEW_KIND",
    "ENTRIES_FILE",
    "HUMAN_CHOICES",
    "HUMAN_PREFERENCE_CONTRACT",
    "HUMAN_PREFERENCE_ENTRY_CONTRACT",
    "HUMAN_PREFERENCE_ENTRY_VERSION",
    "HUMAN_PREFERENCE_VERSION",
    "HUMAN_REVIEW_DIR",
    "IMAGE_DIR",
    "ITEMS_FILE",
    "PREFERENCE_CHOICES",
    "PREFERENCE_STATUSES",
    "REVIEW_FILE",
    "REVIEW_KINDS",
    "STATE_FILE",
    "HumanReviewStore",
    "collect_review_items",
    "entry_id_for",
    "export_choice_items",
    "human_review_stats",
    "pairwise_choice",
    "preference_entry",
    "review_id_for",
    "review_kind",
    "review_to_choice_item",
    "unconsumed_verdicts",
    "validate_preference_entry",
    "validate_human_review",
    "write_review_bundle",
]
