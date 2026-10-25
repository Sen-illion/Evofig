"""Keep a fixed set of base papers evolving: inherit the best draft, never stop.

The session runner walks a fixed queue once and exits when the queue is spent. That leaves the
loop the project actually wants unbuilt: a few base papers that stay put while the version of them
gets better and better. This module is that loop.

One wave is one iteration of every paper that is not paused. A paper's first iteration is a
complete run. Every later iteration starts from the previous one's directory and inherits
everything up to the stage an advisory review of the previous draft asked to change -- the review
names the repair it wants, the repair names its stage, and the earliest stage named is where the
frozen prefix ends. A review that asks for nothing falls back to re-cutting the structure and the
layout, which is the one change that can still help a figure nobody complained about.

Nothing here draws a figure, moves a box or scores a draft: the runner, the reviewer and the
clock are all injected, so the loop itself can be driven offline in a test. Two limits are worth
naming because the state records them rather than hiding them. The reuse source is the previous
iteration's directory, which is where the contracts, the reference analysis, the design seed and
the frozen Icon assets live; the best candidate of that iteration is what the run's own rounds
already chained from, and its path is recorded as ``best`` for audit. And the only stop this loop
has is the caller's: a stop file, the wall-clock budget, the disk floor, or a paper that keeps
failing.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Iterable

from dataclasses import replace

from ..contracts.evolution import STAGE_ORDER, normalize_layer_settled
from ..evaluation.direct_judges import (
    ADVISORY_LOCALISATION_RETRY,
    DirectJudgeInput,
    SEAT_NO_BACKEND,
    SEAT_RAISED,
    build_direct_judges,
)
from ..evaluation.interactive_judgement import deterministic_dimension_scores
from ..evolution.direct import (
    CURRENT_CASE_STATUS,
    HUMAN_REVIEW_DIR,
    PROMPT_PATCH_REFUTATION_LIMIT,
    RESTART_FROM_ARCHIVE,
    ExperienceStore,
    HumanReviewStore,
    PromptPatchStore,
    action_is_executable,
    aliased_action_name,
    append_event,
    patch_sentence_key,
    review_kind,
    stage_for_action,
    unconsumed_verdicts,
)
from ..evolution.direct.file_lock import append_jsonl
from ..evolution.direct.human_queue import HumanQueue
from ..evolution.direct.metrics import layer_walk, round_metrics
from ..evolution.direct.store import DEFAULT_RETRIEVAL_STATUSES
#: The ladder outcomes that mean the paired run actually measured a prompt patch doing worse. This is
#: the same ruler the round-level path charges with (``CREATOR_RULE_REFUTING_REASONS`` only names
#: ``pairwise_not_preferred`` and ``no_significant_improvement``): only a measured loss is a loss. A
#: set nobody measured (``overfit_or_failed``) or one whose two arms produced the same figure
#: (``inconclusive``) is an answer of "none", so it is recorded as neutral, does not count against the
#: patch and does not lower its rung.
PATCH_REFUTING_STATUSES = frozenset({"current_case_validated", "refuted"})

#: The verdicts that say a measurement moved a row up the ladder. Read only to decide whether the
#: validation seat has an entry decision to make at all -- a measurement that moved nothing has
#: nothing to admit or hold -- and to write the shadow answer on that card. What a verdict actually
#: earns is still settled by the ladder itself, so this is a reading of the verdict, not a second
#: ruler for it.
ROW_MOVING_VERDICTS = frozenset(DEFAULT_RETRIEVAL_STATUSES) | {"quick_validated"}

from ..utils import ensure_dir, read_json, sanitize_surrogates, write_json, write_text
from .direct_experience_validation import (
    ROW_UNDER_TEST_CONTRACT,
    ROW_UNDER_TEST_FILE,
    kept_figure_judge_input,
    run_direct_experience_validation,
)
from .evolution_session import EVOLVE_OPTION_KEYS, station_slug
from .paper_evolve_editable import run_paper_evolve_editable
from .stage_replay import READING_FORK_STAGES, reuse_forecast
from .stage_trace import attribution_verdict
from .session_evidence import evolve_evidence

CONTINUOUS_STATE = "continuous_state.json"
CONTINUOUS_EVENTS = "continuous_events.jsonl"
CONTINUOUS_REPORT = "continuous_report.json"
CONTINUOUS_MARKDOWN = "continuous_report.md"
CONTINUOUS_STATE_SCHEMA = "rfs.continuous_evolution_state"

#: One paper's own account of the round it just finished, written beside that paper.
#: The wave's state file is written once per wave, so an interrupted wave used to take every
#: round it had already finished down with it, and the restart paid for all of them again.
PAPER_RECORD_FILE = "entry_state.json"
PAPER_RECORD_SCHEMA = "rfs.continuous_evolution_paper_record"

#: The keys the ``continuous`` block of a config may carry. Anything else is ignored, so a typo
#: cannot quietly become a run option, and the evolve options come from the session's own list.
CONTINUOUS_KEYS = ("base_papers", "parallel", "budget_hours", "stop_file", "iteration_rounds", "evolve", "disk_floor", "disk_floor_gb", "validation", "reread_once", "reopen_after_stalled_rounds", "cross_round_every", "human_seat", "target_iterations")

#: How the acceptance seat's own answer about the round's layer is treated. ``off`` never puts the
#: extra question to it -- the seat is asked exactly what it was trained on. ``record`` asks and
#: files the answer without acting on it. ``authority`` lets that answer move the work face.
#: The floor is ``record``: nothing about the direction changes until a person has watched the seat
#: answer it and said the answers are worth following.
DEFAULT_DIRECTION_AUTHORITY = "record"
DIRECTION_AUTHORITIES = ("off", "record", "authority")

DEFAULT_ITERATION_ROUNDS = 5

#: How many iterations a call lets a paper run before it is left out of the waves, or zero for
#: "no target". A sweep that reads a fixed round needs it: papers interrupted at different
#: rounds drift apart, and the waves that bring the slowest up to the target would otherwise
#: carry every other paper past it -- rounds that are paid for and never read.
DEFAULT_TARGET_ITERATIONS = 0

#: One validation worker. Each one runs a full paired ablation, so more workers buy contention on
#: the same GPU and the same provider quota rather than more verdicts per hour.
DEFAULT_VALIDATION_WORKERS = 1
#: How many times the loop hands the same unmeasured candidate back to the queue before it
#: stops asking. A candidate nothing can measure -- a row written before an applicability was
#: required of it, say -- would otherwise be re-queued on every call for ever.
MAX_UNMEASURED_REQUEUES = 3

DEFAULT_PARALLEL = 2

#: How many rounds in a row may be spent below the work face without moving the figure before the
#: review is told what the loop has watched. It is a hint and not an action: the loop never reopens
#: a layer itself, it only says how long the layer it is working on has been stuck, and the model
#: decides whether the fault is really one layer up.
DEFAULT_REOPEN_AFTER_STALLED_ROUNDS = 4
#: How often the current version is put beside the best version this paper has reached, in
#: rounds. A settled layer asks for one on its own: that is the moment the figure changed
#: level, and the comparison is what says whether it changed level upward.
DEFAULT_CROSS_ROUND_EVERY = 3
DEFAULT_BUDGET_HOURS = 24.0

#: A wave refuses to start when the volume holding the output has less than this share free.
#: A run that fills the disk reports a worse figure on the next paper for reasons nobody asked
#: about, and a truncated artifact reads as a measurement.
DISK_FLOOR = 0.10

#: A config may replace that share with an absolute floor (``disk_floor_gb``), and may lower the
#: share itself (``disk_floor``). A long loop on a volume that is already almost full is a
#: legitimate thing to ask for when the data on it is not ours to delete; what is not legitimate
#: is lowering the guard silently, so an absolute floor only counts when the caller wrote the
#: number down, and the run's own report says which floor it applied and what it measured.
DEFAULT_DISK_FLOOR_GB = 0.0

#: How many times one paper may fail before the loop parks it. A failing iteration is retried by
#: the next wave, because most failures are transient; a paper that fails this many times in a row
#: is not going to succeed on the next try either, and parking it keeps the other papers moving.
MAX_CONSECUTIVE_FAILURES = 3

#: The first stage in the pipeline's own order, and the fork that inherits nothing upstream of
#: it. A round forked here plans the method contract itself, which is the same thing as reading
#: the paper again.
REREAD_FORK_STAGE = "global_planning"

#: How long a call waits for the measurement worker to finish the case it was in when the loop
#: ended. The worker checks the stop between cases, so this waits for the case in flight and no
#: longer; a worker still busy after this is written down rather than waited on forever, because
#: a hung network call must not hold the process open.
VALIDATION_JOIN_SECONDS = 600.0

#: How far back a restart may look for a variant to start from. A variant drawn many iterations
#: ago was drawn from a version the loop has since replaced, so inheriting it starts the round
#: from a figure nobody is comparing against any more -- which is exactly the loop the restart is
#: meant to break out of. The window is counted in iterations, the newest eligible drawing last.
RESTART_WINDOW_ITERATIONS = 8

#: Where the run is allowed to re-plan when the review asked for nothing.
DEFAULT_FORK_STAGE = "segmentation"
#: The stage a review forks at when every repair it named acts on the figure the run
#: already has. Nothing upstream of it is re-planned, so the iteration opens from the
#: inherited figure and can run one of those repairs instead of redrawing it.
INCREMENTAL_FORK_STAGE = "ppt_compilation"

#: Where an iteration forks when the review did not answer at all. A review that failed, or
#: that had no figure to read, has said nothing about the figure: forking at the default stage
#: would re-plan the whole draft on the strength of a crash, which is an hour of drawing spent
#: learning nothing. These iterations keep the figure they inherited, exactly as an
#: incremental repair does, and let the round find its own change.
SILENT_REVIEW_FORK_STAGE = INCREMENTAL_FORK_STAGE

#: How many times the boundary review is asked again when the seat itself failed to answer. One
#: retry costs a minute next to the hour the round spends, and a seat that raised once or that
#: answered past its budget is routinely a transient. A checkpoint that is not on the disk is not
#: retried -- it will not appear on the second call either -- and is written down straight away.
DEFAULT_REVIEW_RETRY_LIMIT = 1

#: What the boundary review did, written on the review and on the round. The three silences are
#: not one thing: the seat gave no answer at all, the seat answered without saying where the fault
#: entered, and the seat said the figure has no fault. Only the last is a statement about the
#: figure. The first two say the round learned nothing about it, and reading either of them as "no
#: fault" is what used to re-cut the structure on the strength of a crash.
REVIEW_ERROR = "review_error"
REVIEW_UNDEFINED = "review_undefined"
REVIEW_CLEAN = "review_clean"

#: The one file a round leaves for the boundary review it was handed. The review decides where the
#: round forks and the events only carry its exit reason, so the question, the answer, the model
#: and whether the seat was asked again are written down beside the round they decided.
BOUNDARY_REVIEW_FILE = "boundary_review.json"
BOUNDARY_REVIEW_CONTRACT = "rfs.boundary_review"
BOUNDARY_REVIEW_QUESTION = (
    "which stage's own requirement went unwritten, which requirement that was, and the number "
    "from this figure that says so"
)

#: The one file an iteration leaves for the layers it worked on: what each layer did, the
#: reading that layer's own gate took, and whether a model said it had finished with it. It is
#: assembled from artifacts the iteration already wrote, so it costs nothing and settles
#: nothing -- a layer is closed by an answer, never by a reading or by this file.
LAYER_LEDGER_FILE = "layer_ledger.json"
LAYER_LEDGER_CONTRACT = "rfs.layer_ledger"
#: What the acceptance seat said about the round's own layer, written by the run beside the round
#: it was asked in. The loop reads it back for the ledger, and acts on it only when the config says
#: that seat holds the direction.
ACCEPTANCE_DIRECTION_FILE = "acceptance_direction.json"

#: One round's own figure put beside the best version this paper had reached before it. The
#: answer is evidence -- for the layer ledger and for the trend -- and it decides nothing: a
#: round is accepted or rolled back by its own comparison, never by this one.
CROSS_ROUND_FILE = "cross_round_comparison.json"
CROSS_ROUND_CONTRACT = "rfs.cross_round_comparison"
CROSS_ROUND_CHOICES = {"A": "history_better", "B": "current_better", "TIE": "tie"}

#: What one round is asked to know before it answers: the goal, the order the layers are built
#: in, where each of them stands, and what the round before this one changed. It is written
#: beside the round and handed to the seat that decides where the round starts, so the answer
#: is a step along a line rather than a guess about a figure with no history.
#: What a person filed about this paper, and which of it this round acted on. A verdict is read
#: once: the file is the receipt for the one round that read it, so "a person said this and nothing
#: changed" can be told apart from "a person said this and it was applied".
HUMAN_VERDICTS_FILE = "human_verdicts.json"
HUMAN_VERDICTS_CONTRACT = "rfs.human_verdicts"

PROGRESS_SHEET_FILE = "progress_sheet.json"
PROGRESS_SHEET_LATEST = "progress_sheet_latest.json"
PROGRESS_SHEET_CONTRACT = "rfs.progress_sheet"
PROGRESS_SHEET_GOAL = (
    "draw this paper as one editable framework figure, one layer at a time from the top down: "
    "read the paper, choose its content, plan the structure, choose the theme, draw the icons, "
    "build the composition, then repair what the readings show"
)

#: What a person standing at the direction seat is asked. The seat answers one thing -- which layer
#: the iteration that is about to run works on -- and it is the only question whose answer chooses a
#: fork rather than a figure, so it says what the answer does to the run. It never mentions the
#: model's own answer: the card carries the question and the progress sheet, and the seat's answer is
#: kept beside the model's afterwards so the two can be compared without either being anchored.
DIRECTION_QUESTION = (
    "Where should this iteration work? Name one layer -- the layer the fault is in, or the open "
    "layer you want planned again, or, when everything above it is good, the next layer down. "
    "Answer with the layer's own name, one of: paper_reading, content_planning, global_planning, "
    "segmentation, theme_planning, icon_generation, ppt_compilation, repair. Everything before the "
    "layer you name is planned again from the paper it came from, which is why a layer above the "
    "one already open costs a whole re-read. Leave the card unanswered and the run continues on "
    "the model's own answer."
)

#: What a person standing at the validation seat is asked. The seat answers one thing -- whether the
#: measurement that just finished may move the row up the ladder -- and it is asked only where the
#: measurement would move it, because a reading that moved nothing leaves no entry to decide. The
#: two answers are "hold", which leaves the row where it was, and anything else, which lets the
#: ladder read the measurement exactly as it would have. A person cannot write a row into the
#: library: entry is a claim about a measurement, and only a measurement can make it.
VALIDATION_QUESTION = (
    "May this measurement move the row up the ladder? The paired run has finished and the verdict it "
    "reached would carry the row forward; the readings are below. Answer \"hold\" to keep the row "
    "where it is -- the measurement is recorded and nothing about the row changes -- or anything "
    "else to let the verdict stand. Leave the card unanswered and the verdict stands."
)

#: The words that mean "hold" on an answered validation card. Everything else an answered card says
#: is read as "let the verdict stand", so a person cannot lose a measurement by phrasing it their
#: own way -- and the words that stop a row are the ones worth spelling out.
VALIDATION_HOLD_WORDS = frozenset({"hold", "no", "stop", "wait", "not yet", "hold it", "不要", "先不要"})

#: How many reviews in a row have to place the fault at the reading before an iteration gives
#: up the figure it inherits. The attribution is read *instead of* the repair the same call
#: named, so one answer would turn a single verdict into a whole redraw -- and re-reading was
#: measured to fire on half the iterations that way, every one of them a drawing from scratch.
#: Two iterations that each placed the fault at the reading is a statement about the reading;
#: one on its own is a statement about one call.
READING_ATTRIBUTION_CONFIRMATIONS = 2

#: The stage order the fork stage is read from. It is the attribution vocabulary in
#: ``rfs.contracts.evolution`` itself -- one tuple for the fork, the progress sheet and the
#: layer ledger -- including its two leading entries: a review that places the fault at the
#: paper's reading is asking for the paper to be read again, and a fork that could not land
#: there would answer "the content is wrong" by re-planning a figure built from the reading
#: being complained about. ``STAGE_ORDER`` is imported above rather than written again here.


class ContinuousError(RuntimeError):
    """A continuous run that cannot continue, carrying the reason the report will name."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _floor_ratio(value: Any) -> float:
    """Read the share of the volume the run must leave free."""

    return min(0.9, max(0.0, _number(value, DISK_FLOOR)))


def _floor_gb(value: Any) -> float:
    """Read the absolute free space the run must leave, in GB. Zero means "not asked for"."""

    return max(0.0, _number(value, DEFAULT_DISK_FLOOR_GB))


def _validation_spec(raw: Any) -> dict[str, Any]:
    """Read the validation block: which case set, whether to run it, and how many at once.

    A new experience is worth nothing until another paper has measured it, and measuring it means
    running the same pair of arms the experience was earned on. That costs hours, so the loop
    hands the work to a background worker and keeps evolving: the wave that produced a candidate
    does not wait for its verdict, and the verdict arrives in the library's own status.
    """

    if isinstance(raw, str):
        suite = raw.strip()
        return {"enabled": bool(suite), "suite": suite, "profile": "", "workers": DEFAULT_VALIDATION_WORKERS, "reason": "" if suite else "the validation block names no suite"}
    if not isinstance(raw, dict):
        return {"enabled": False, "suite": "", "profile": "", "workers": DEFAULT_VALIDATION_WORKERS, "reason": "the config has no validation block, so candidates are recorded and not measured"}
    suite = str(raw.get("suite") or "").strip()
    return {
        "enabled": bool(raw.get("enabled", bool(suite))) and bool(suite),
        "suite": suite,
        # Which bar this measurement runs at. Left blank the suite's own word decides and the full
        # four-group ladder is the default; a config names "quick" to ask for the single matched
        # paper the caller already agreed to, and the station writes that word into its record so
        # a verdict can never be read as one the full ladder gave.
        "profile": str(raw.get("validation_profile") or "").strip().casefold(),
        "workers": max(1, int(_number(raw.get("workers"), DEFAULT_VALIDATION_WORKERS))),
        "reason": "" if suite else "the validation block names no suite",
    }


#: The judgement points a person may stand at, and the name each is filed under. The seat is named
#: after what is being decided -- the acceptance seat decides this round, the direction seat says
#: where the round goes -- and the card uses the same word as the review channel's own kinds, so an
#: answer in the channel and an answer on a card are the same act written two ways.
HUMAN_SEAT_KINDS = {
    "acceptance": "figure_choice",
    "figure_choice": "figure_choice",
    "direction": "direction",
    "layer_settled": "layer_settled",
    "validation": "validation",
}


def _human_seat_spec(raw: Any) -> dict[str, Any]:
    """Read the human-seat block: whether a person stands in a seat, and what happens without them.

    Nothing here is required. A config with no block behaves exactly as it did before: no seat is
    held for a person, no question is queued, and the acceptance seat is asked about the round's
    layer only because the direction floor says so. Where a person does stand, the loop waits for
    them and falls back to the model's own answer after the hold, and the record says which of the
    two decided -- so a run can never report a person's answer it did not get.
    """

    body = raw if isinstance(raw, dict) else {}
    authority = str(body.get("direction_authority") or DEFAULT_DIRECTION_AUTHORITY).strip().casefold()
    seats = [str(item).strip().casefold() for item in (body.get("seats") or []) if str(item).strip()]
    return {
        "enabled": bool(body.get("enabled", bool(seats))),
        "seats": seats or (["acceptance", "direction", "validation"] if body.get("enabled") else []),
        "dir": str(body.get("dir") or "").strip(),
        # How long a queued question waits for a person before the model's own answer carries the
        # round. A hold of zero means "ask and move on", which is what a smoke test wants.
        "hold_hours": max(0.0, _number(body.get("hold_hours"), 8.0)),
        "on_timeout": str(body.get("on_timeout") or "model").strip().casefold(),
        "advisory_mode": str(body.get("advisory_mode") or "override").strip().casefold(),
        "direction_authority": authority if authority in DIRECTION_AUTHORITIES else DEFAULT_DIRECTION_AUTHORITY,
        # Set by this loop at run time from the seats above: which papers are held for a person.
        "papers": [str(item) for item in (body.get("papers") or []) if str(item).strip()],

        # The judgement points themselves, under the names the cards are filed with. A block that
        # turned the seat on without naming any is read as the three the plan asks for.
        "kinds": [HUMAN_SEAT_KINDS.get(name, name) for name in (seats or (["acceptance", "direction", "validation"] if body.get("enabled") else []))],
    }


def _base_papers(raw: Any) -> list[dict[str, Any]]:
    """Read the fixed paper list, refusing an entry that names no paper."""

    if not isinstance(raw, list) or not raw:
        raise ValueError("a continuous config needs a non-empty base_papers list")
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            entry: dict[str, Any] = {"paper": item}
        elif isinstance(item, dict):
            entry = dict(item)
        else:
            raise ValueError(f"a base paper must be a path or an object: {item!r}")
        paper = str(entry.get("paper") or "").strip()
        if not paper:
            raise ValueError("a base paper needs a paper path")
        name = station_slug(str(entry.get("paper_id") or "").strip() or Path(paper).stem)
        if name in seen:
            raise ValueError(f"two base papers resolve to the same directory name: {name}")
        seen.add(name)
        papers.append(
            {
                "name": name,
                "paper": paper,
                "paper_id": str(entry.get("paper_id") or "").strip(),
                "domain": str(entry.get("domain") or "").strip(),
                # Where this paper's first iteration continues from, and what that round is
                # measured against. Absent means the paper starts from nothing, which is how every
                # paper behaved before either key existed. The two are separate because resuming a
                # paper whose published line is not the version being resumed is exactly the case a
                # run needs: inherit the good draft, be measured against the one before it.
                "start_from": str(entry.get("start_from") or "").strip(),
                "start_baseline_from": str(entry.get("start_baseline_from") or "").strip(),
            }
        )
    return papers


def read_continuous_config(config: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Normalise a continuous config, wherever the block sits in the file."""

    if isinstance(config, dict):
        body, source = dict(config), ""
    else:
        path = Path(config)
        body = read_json(path)
        if not isinstance(body, dict):
            raise ValueError(f"a continuous config must be a JSON object: {path}")
        source = str(path)
    block = body.get("continuous") if isinstance(body.get("continuous"), dict) else body
    evolve: dict[str, Any] = {}
    for holder in (block.get("evolve") if isinstance(block.get("evolve"), dict) else {}, body.get("evolve") if isinstance(body.get("evolve"), dict) else {}):
        for key, value in holder.items():
            if key in EVOLVE_OPTION_KEYS:
                evolve.setdefault(key, value)
    for holder in (block, body):
        for key in EVOLVE_OPTION_KEYS:
            if key in holder:
                evolve.setdefault(key, holder[key])
    return {
        "config_source": source,
        "papers": _base_papers(block.get("base_papers")),
        "parallel": max(1, int(_number(block.get("parallel"), DEFAULT_PARALLEL))),
        "budget_hours": max(0.0, _number(block.get("budget_hours"), DEFAULT_BUDGET_HOURS)),
        "disk_floor": _floor_ratio(block.get("disk_floor")),
        "disk_floor_gb": _floor_gb(block.get("disk_floor_gb")),
        # A one-off switch, read as a boolean: the next iteration reads the paper again instead
        # of inheriting the reading its figure was built from.
        "reread_once": bool(block.get("reread_once")),
        "stop_file": str(block.get("stop_file") or "").strip(),
        "iteration_rounds": max(1, int(_number(block.get("iteration_rounds"), DEFAULT_ITERATION_ROUNDS))),
        "target_iterations": max(0, int(_number(block.get("target_iterations"), DEFAULT_TARGET_ITERATIONS))),
        "reopen_after_stalled_rounds": max(1, int(_number(block.get("reopen_after_stalled_rounds"), DEFAULT_REOPEN_AFTER_STALLED_ROUNDS))),
        "cross_round_every": max(1, int(_number(block.get("cross_round_every"), DEFAULT_CROSS_ROUND_EVERY))),
        "evolve": evolve,
        "validation": _validation_spec(block.get("validation")),
        "human_seat": _human_seat_spec(block.get("human_seat")),
    }


def _empty_state(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": CONTINUOUS_STATE_SCHEMA,
        "version": 1,
        "config_source": str(config.get("config_source") or ""),
        "papers": {
            spec["name"]: {
                "paper": spec["paper"],
                "paper_id": spec["paper_id"],
                "domain": spec["domain"],
                "iteration": 0,
                "best": "",
                "best_score": None,
                "reuse_from": "",
                "last_reuse": {},
                "last_reuse_forecast": {},
                "failures": 0,
                "paused": False,
                # The iteration the one-off re-read was spent on, or 0 while it is still unspent.
                "reread_done_at": 0,
                "best_moved_at": 0,
                "repair_rounds": 0,
                "archive": [],
                "last_restart": 0,
                # The layers of this paper's figure, by the pipeline's own stage names. A stage
                # with no entry has not been settled; the loop reads this to know which layer the
                # round is working on, and never to decide that a layer is good.
                "layers": {},
                "work_face": "",
                # How many rounds in a row have been spent without the figure moving forward. It
                # only ever buys a sentence in the review's question; the loop does not act on it.
                "stalled_streak": 0,
                "last_defect_ids": [],
                # The human verdicts this paper has already acted on, by their own ids. Kept in the
                # state rather than recomputed, so a verdict filed once is read once and a loop that
                # restarts does not read the same sentence again.
                "human_verdicts_seen": [],
                "history": [],
            }
            for spec in config["papers"]
        },
        "validations": {},
        "waves": 0,
        "elapsed_seconds": 0.0,
    }


def load_continuous_state(out: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Read the state a previous run left, so a restart continues instead of starting over.

    A paper the config still names keeps its iteration counter and its best version. A paper the
    config has dropped disappears; a paper the config added starts at its first iteration.
    """

    if "papers" not in config:
        config = read_continuous_config(config)
    path = Path(out) / CONTINUOUS_STATE
    stored = read_json(path) if path.is_file() else None
    if not isinstance(stored, dict) or not isinstance(stored.get("papers"), dict):
        return _empty_state(config)
    state = dict(stored)
    state["schema"] = CONTINUOUS_STATE_SCHEMA
    state["papers"] = {
        spec["name"]: {
            **(_empty_state(config)["papers"][spec["name"]]),
            **(stored["papers"].get(spec["name"]) if isinstance(stored["papers"].get(spec["name"]), dict) else {}),
        }
        for spec in config["papers"]
    }
    state.setdefault("waves", 0)
    state.setdefault("elapsed_seconds", 0.0)
    return state


def requested_action_types(feedback: Any) -> list[str]:
    """Read the bounded repairs an advisory review asked for, in its own vocabulary."""

    found: list[str] = []
    for item in feedback or []:
        if not isinstance(item, dict):
            continue
        for action in list(item.get("requested_actions") or []) + list(item.get("repair") or []):
            record = dict(action) if isinstance(action, dict) else {"type": str(action)}
            if not record.get("scope_type") and item.get("scope_type"):
                record["scope_type"] = item.get("scope_type")
            name, _raw = aliased_action_name(record)
            if name and name not in found:
                found.append(name)
    return found


def requested_actions(feedback: Any) -> list[dict[str, Any]]:
    """Read the bounded repairs an advisory review asked for, parameters included.

    ``requested_action_types`` answers "which names", which is what the fork stage needs; a round
    that goes on to run one of them needs the whole record, because a repair stripped of the
    objects and parameters it was stated with is not one the loop may run.
    """

    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in feedback or []:
        if not isinstance(item, dict):
            continue
        for action in list(item.get("requested_actions") or []) + list(item.get("repair") or []):
            record = dict(action) if isinstance(action, dict) else {"type": str(action)}
            if not record.get("scope_type") and item.get("scope_type"):
                record["scope_type"] = item.get("scope_type")
            name, _raw = aliased_action_name(record)
            if not name or name in seen:
                continue
            seen.add(name)
            found.append({**record, "type": name})
    return found


def opening_move_from_feedback(feedback: Any, fork_stage: str) -> dict[str, Any]:
    """The one change an iteration that inherits a finished figure opens by running.

    The review of the inherited figure named what is wrong with it, so the next iteration may
    execute that change instead of planning the same stages all over again. Only a bounded repair
    qualifies: one that names a stage is a redraw, which is a different round and is what the fork
    stage already decides. ``fork_stage`` is the earliest stage the review named, so a review that
    named any stage at all leaves the fork before the point where a repair could be applied to the
    inherited figure -- there, nothing is returned and the iteration opens the way it did before.
    """

    if fork_stage != INCREMENTAL_FORK_STAGE:
        return {}
    for record in requested_actions(feedback):
        if stage_for_action(record.get("type")) != INCREMENTAL_FORK_STAGE:
            continue
        if not action_is_executable(record):
            continue
        return record
    return {}


def fork_before_from_feedback(feedback: Any) -> tuple[str, str]:
    """The earliest stage the review asked to redraw, and the sentence that says why.

    A repair is planned at one of the contract's stages, so "what did the review want changed"
    answers "how much of the previous draft is still worth keeping" directly: everything before
    the earliest stage named is inherited, and that stage and everything after it is planned
    again. A review that named no repair inherits the contracts and re-cuts the structure, which
    is the one change that can help a figure nobody complained about.
    """

    stages = [stage for stage in (stage_for_action(name) for name in requested_action_types(feedback)) if stage in STAGE_ORDER]
    if not stages:
        return DEFAULT_FORK_STAGE, "the review asked for no repair, so the structure and the layout are planned again"
    earliest = min(stages, key=STAGE_ORDER.index)
    return earliest, f"the review asked to change at {earliest}, so everything before that stage is inherited"


#: What the review is told when it asks to go back over a layer the model already called
#: finished, and has not said why. The rule is stated rather than enforced silently, because the
#: seat is the one that owns the answer: it gets one chance to say what it saw before the round is
#: held where it is. Overlap, out-of-bounds boxes and more crossings are named as *not* the
#: argument on purpose -- those are things to repair on the figure the round already has.
UPSTREAM_EVIDENCE_RETRY = (
    "\n\nYour previous answer moved the round back to a layer the run has already finished with. "
    "Going back over a finished layer has to be argued: name the layer, the requirement that went "
    "unwritten there, and the reading from the figure in front of you that shows it. Overlap, "
    "out-of-bounds objects and more crossings are not that argument -- they are things to repair "
    "on the figure you have. Without the argument, answer for the layer the run is working on."
)


def work_face(layers: Any) -> str:
    """The earliest layer the model has not settled: where this round is allowed to work.

    The pipeline builds a figure in one order, so the layer still open is the earliest one that has
    not been called finished. Working below it is always free -- a settled structure does not stop
    the connectors from being repaired -- and working above it means going back over a layer the
    model has already declared done, which is the argument the clamp below insists on.
    """

    record = layers if isinstance(layers, dict) else {}
    for stage in STAGE_ORDER:
        entry = record.get(stage)
        if isinstance(entry, dict) and entry.get("settled_at"):
            continue
        return stage
    return STAGE_ORDER[-1]


def stage_index(stage: Any) -> int:
    """Where a stage sits in the pipeline's order, or -1 for a name that is not one."""

    try:
        return STAGE_ORDER.index(str(stage or ""))
    except ValueError:
        return -1


def is_upstream(fork: Any, work: Any) -> bool:
    """Whether a fork would go back over the layer this round is working on."""

    return stage_index(fork) < stage_index(work)


def upstream_case(review: Any, *, stages: Any = STAGE_ORDER) -> dict[str, Any]:
    """The argument a review has to carry to go back over a layer that is finished with.

    Going back over a settled layer is a claim about that layer's own output, so it costs the
    argument every other claim costs: the layer, the requirement that went unwritten there, and a
    reading the seat was actually shown. The check is the round's own attribution validator with
    the layer order handed to it, so a layer the pipeline does not have, or a number no block put in
    front of the seat, is discarded here exactly as it is there -- an invented citation cannot buy a
    round of re-planning. Overlap, out-of-bounds objects and more crossings are not arguments: they
    are things to repair on the figure the round already has.
    """

    record = review if isinstance(review, dict) else {}
    judgement = record.get("judgement") if isinstance(record.get("judgement"), dict) else {}
    case = judgement.get("upstream_case") if isinstance(judgement.get("upstream_case"), dict) else {}
    verdict = attribution_verdict(
        {
            "fault_stage": case.get("stage") or judgement.get("fault_stage") or record.get("fault_stage"),
            "fault_requirement": case.get("requirement") or judgement.get("fault_requirement"),
            "fault_evidence": case.get("evidence") or judgement.get("fault_evidence"),
        },
        None,
        shown=record.get("shown") or judgement.get("shown") or judgement.get("progress_sheet"),
        shown_fields=judgement.get("advisory_shown_fields"),
        stages=stages,
    )
    claim = verdict.get("attribution") if isinstance(verdict.get("attribution"), dict) else {}
    if not verdict.get("ok") or not str(claim.get("fault_stage") or ""):
        return {"ok": False, "reason": str(verdict.get("reason") or "no_stage"), "case": {}}
    return {"ok": True, "reason": "", "case": claim}


def upstream_allowed(*, fork: str, work: str, attributed: str, streak: int, review: Any, case: Any = None) -> bool:
    """Whether a fork before the work face may stand.

    Three things answer for it, and nothing else does. A reading the loop itself confirmed -- the
    same verdict on two iterations in a row -- is already an argument, and it is the argument the
    re-read mechanism was built on. A case the review carried for that very layer, with a reading
    the seat was shown behind it, is the other: the seat said what it saw, and the layer it names
    is the one it is asking to go back over. Anything else is a claim without a reading behind it,
    and the round is held at the work face instead, with the refusal recorded.
    """

    if not is_upstream(fork, work):
        return True
    if attributed and fork == attributed and int(streak or 0) >= READING_ATTRIBUTION_CONFIRMATIONS:
        return True
    claim = case.get("case") if isinstance(case, dict) else {}
    return bool(case and case.get("ok") and str((claim or {}).get("fault_stage") or "") == str(fork or ""))


def deadlock_hint(stalled: int, limit: int, work: str) -> str:
    """The line the review is given when the layer it is working on has not moved for a while.

    It does not act. Reopening a layer the model called finished is a judgement about the figure,
    and the loop does not make it -- it reports how long it has watched and leaves the choice
    where it belongs. An empty hint renders the prompt exactly as it did before.
    """

    if int(stalled or 0) < max(1, int(limit or 0)):
        return ""
    # The fact, and nothing that reads like an instruction to go back up. The loop watched a layer
    # stand still for a while; whether that means an earlier layer was sealed wrongly is a
    # judgement about the figure, and the seat that made it is the seat that has to change it.
    return f"For {int(stalled)} rounds in a row the figure has not moved forward from {work}."


def attribution_fork_stage(review: Any) -> str:
    """The reading stage a review placed the fault at, or ``""`` when it placed it elsewhere.

    The fork is normally read from the repairs a review asked for, and for every stage a repair
    can act at that is the right question: the repair names the earliest artifact that has to be
    planned again. The two reading stages are the exception, and the reason is that no repair
    reaches them. A seat that places the fault at ``paper_reading`` or ``content_planning`` is
    saying the paper was read wrongly, and every repair it can name acts below the reading, on a
    figure built from it -- so forking at the repair's own stage would inherit the reading being
    complained about and answer "the content is wrong" by redrawing the same content. Only those
    two stages read the attribution; every later one is already named by its own repair.
    """

    record = review if isinstance(review, dict) else {}
    judgement = record.get("judgement") if isinstance(record.get("judgement"), dict) else {}
    stage = str(judgement.get("fault_stage") or record.get("fault_stage") or "").strip()
    return stage if stage in READING_FORK_STAGES else ""


def fork_from_review(review: Any, *, attribution_streak: int = 0) -> tuple[str, str]:
    """Where an iteration forks, reading what the review asked for before what it blamed.

    ``fork_before_from_feedback`` answers "what did the review ask for", and it cannot tell a
    review that asked for nothing from one that never ran: both hand it an empty list of
    repairs. The two answers cost very different amounts. "Asked for nothing" is a statement
    about the figure, and is answered by planning the structure again; a review that failed
    says nothing about the figure, and forking at the default stage for it redrew the whole
    draft because a call raised. Only a review that ran and asked for nothing keeps that
    behaviour.

    A fault the review placed *in the reading* is the one case a repair cannot reach: every action
    in the catalogue acts below the reading, so running one would answer "the content is wrong" by
    drawing the same content again. That is why the attribution used to outrank the repair the same
    call named -- and the ranking is expensive, so it was being spent on a single verdict, on the
    iterations that named both. Two things guard it now. ``attribution_streak`` counts how many
    iterations in a row have placed the fault at that same reading stage, and it is read against
    :data:`READING_ATTRIBUTION_CONFIRMATIONS`; below that, a review that also named a repair is
    answered by the repair and keeps the reading the inherited figure was built on, while one that
    named no repair re-cuts the structure instead of going back to the paper. A streak is counted
    by the caller against the iteration before it, so a verdict on its own never sends the run back
    to the paper.
    """

    record = review if isinstance(review, dict) else {}
    if record and not record.get("ok"):
        reason = str(record.get("reason") or "the review did not answer")
        exit_reason = str(record.get("exit") or "")
        # The exit reason travels into the sentence, so a round that stayed in place because the
        # seat was unreadable can be told, afterwards, from one that stayed because the answer
        # never said where the fault entered: both are "the review did not answer", and only the
        # first is a broken seat.
        label = f"{reason}; recorded as {exit_reason}" if exit_reason else reason
        return (
            SILENT_REVIEW_FORK_STAGE,
            f"the review did not answer ({label}), so this iteration keeps the figure it inherits "
            "and repairs it in place instead of planning the structure again",
        )
    if str(record.get("exit") or "") == REVIEW_CLEAN:
        # The seat read the figure and found nothing to repair. Planning the structure again on the
        # strength of that is the redraw nobody asked for, so the iteration keeps the figure and
        # looks for its own change.
        return (
            SILENT_REVIEW_FORK_STAGE,
            "the review read the figure and found nothing to repair, so this iteration keeps it "
            "and repairs it in place instead of planning the structure again",
        )
    attributed = attribution_fork_stage(record)
    streak = int(attribution_streak or 0) if attributed else 0
    if attributed and streak >= READING_ATTRIBUTION_CONFIRMATIONS:
        return (
            attributed,
            f"{streak} reviews in a row placed the fault at {attributed}, "
            "so this iteration reads the paper again rather than plan from the reading the figure "
            "was built on",
        )
    named = [
        stage
        for stage in (stage_for_action(name) for name in requested_action_types(record.get("stage_feedback")))
        if stage in STAGE_ORDER
    ]
    if attributed and named:
        fork, reason = fork_before_from_feedback(record.get("stage_feedback"))
        return (
            fork,
            f"the review placed the fault at {attributed} but named a repair that acts on the figure "
            f"this iteration inherits, so that repair is run and the paper is not read again ({reason})",
        )
    if attributed:
        return (
            DEFAULT_FORK_STAGE,
            f"one review placed the fault at {attributed} and named no repair, so the structure and the "
            "layout are planned again: a single attribution does not send the iteration back to the paper",
        )
    return fork_before_from_feedback(record.get("stage_feedback"))


def _is_number(value: Any) -> bool:
    """Whether a score column actually holds a number. A boolean is not a score."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_number(value: Any) -> float:
    """A score column in the form the ordering can compare: a number, or the floor of it."""

    return float(value) if _is_number(value) else -1.0


#: The files that make a run's kept figure showable to a Judge: the preview it rendered and the
#: program it compiled. A run that rolled every repair back wrote neither, and a round that
#: inherits that directory has no figure to look at and nothing to be measured against.
KEPT_FIGURE_PROGRAM = "figure_program.json"
KEPT_FIGURE_PREVIEWS = ("final_draft_preview.png", "draft_preview.png")


def iteration_number(name: Any) -> int | None:
    """The iteration number a run directory's name carries, or ``None`` for anything else."""

    text = str(name or "")
    if not text.startswith("iter_"):
        return None
    try:
        return int(text[len("iter_") :])
    except ValueError:
        return None


def holds_kept_figure(run_dir: Any) -> bool:
    """Whether a finished run left a figure on disk that a Judge can be shown.

    The same two artifacts the Judge input reader insists on, checked here rather than by asking
    that reader, because this is asked about every iteration while walking back and the reader
    would parse the whole program to answer it.
    """

    root = Path(str(run_dir or ""))
    if not str(run_dir or "") or not root.is_dir():
        return False
    if not (root / KEPT_FIGURE_PROGRAM).is_file():
        return False
    return any((root / name).is_file() for name in KEPT_FIGURE_PREVIEWS)


def newest_kept_figure_dir(paper_dir: Path, before: int) -> Path | None:
    """The newest iteration below ``before`` that left a figure, whatever it decided.

    An iteration whose repairs all rolled back leaves no figure of its own -- the version it
    started from is still the published one -- and a round that inherits that empty attempt cannot
    measure anything, which is how a long run of rounds ended with no "after" reading at all. The
    published line is then whichever earlier iteration last put a figure on disk, so that is what
    the next round inherits and is measured against.
    """

    newest: Path | None = None
    newest_number = -1
    if not paper_dir.is_dir():
        return None
    for candidate in sorted(paper_dir.glob("iter_*")):
        number = iteration_number(candidate.name)
        if number is None or number >= int(before) or number <= newest_number:
            continue
        if holds_kept_figure(candidate):
            newest = candidate
            newest_number = number
    return newest


def absolute_score_for(candidate_dir: Any) -> float | None:
    """One candidate's own reading, recomputed from the reports it left behind.

    The comparison score answers "is this better than the thing it was put next to". A restart
    has to answer a different question -- "which of these variants is the better figure" -- and a
    relative score cannot be asked that, because two variants that were compared against
    different things are not on the same scale. The absolute reading is the mean of the five
    deterministic dimensions the geometry and density gates already expose, so it costs no model
    call and two variants can be put side by side. ``None`` when the reports are not there, which
    is the honest answer for a candidate that was never measured this way.
    """

    root = Path(str(candidate_dir or ""))
    if not str(candidate_dir or ""):
        return None
    geometry = _read_json_or_none(root / "geometry_quality_report.json")
    density = _read_json_or_none(root / "density_quality_report.json")
    if not isinstance(geometry, dict) or not isinstance(density, dict):
        return None
    values = [float(value) for value in deterministic_dimension_scores(geometry, density).values() if _is_number(value)]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def archive_rows(iter_dir: Path, iteration: int) -> list[dict[str, Any]]:
    """Every variant one iteration drew, and what the round decided about it.

    A round keeps one candidate and drops the rest, and a round that lost the comparison keeps
    nothing at all -- the figure it drew was thrown away with the rollout. That is the wrong way
    round: a variant the loop did not publish is still a figure somebody drew, and a later round
    may be better served starting from the iteration that drew it than from the version the loop
    settled on. So every candidate of every round is recorded here, with the score it measured,
    where it can be inherited from, and the reason the round gave.

    The rows are read off the run directory rather than off the live objects, so a restart after a
    process died sees exactly what the run left behind.
    """

    rounds: list[dict[str, Any]] = []
    ledger = read_json(iter_dir / "evolution_rounds.json") if (iter_dir / "evolution_rounds.json").is_file() else None
    if isinstance(ledger, dict):
        rounds = [item for item in ledger.get("rounds") or [] if isinstance(item, dict)]
    rows: list[dict[str, Any]] = []
    for record in rounds:
        index = int(record.get("round") or 0)
        selection = read_json(iter_dir / f"round_{index:02d}" / "round_selection.json") if (iter_dir / f"round_{index:02d}" / "round_selection.json").is_file() else None
        if not isinstance(selection, dict):
            continue
        selected = int(selection.get("winner") if selection.get("winner") is not None else -1)
        decision = record.get("repair_decision") if isinstance(record.get("repair_decision"), dict) else {}
        reason = str(decision.get("reason") or "")
        for candidate in selection.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            number = int(candidate.get("candidate_index") or 0)
            score = candidate.get("acceptance_score")
            if score is None:
                score = candidate.get("visual_score")
            absolute = absolute_score_for(candidate.get("dir"))
            rows.append(
                {
                    "iter": int(iteration),
                    "round": index,
                    "candidate": number,
                    "dir": str(candidate.get("dir") or ""),
                    # What a later round can actually inherit from is the iteration's own
                    # contracts and plans, which every candidate of it shares, so the reusable
                    # artifact is the iteration directory that drew this variant.
                    "source": str(iter_dir),
                    # Two scores because two questions are asked of them. The measured score is
                    # what the round's own comparison said about this variant, and it is a
                    # relative reading: a candidate the comparator defaulted on carries the
                    # default rather than a quality. The absolute score is read off the
                    # variant's own geometry and density reports, so any two of them can be put
                    # side by side. Ordering a restart on the relative column is what pinned
                    # every restart to one old drawing.
                    "absolute_score": absolute,
                    "choice_score": score,
                    "score": absolute if absolute is not None else score,
                    "selected": number == selected,
                    "decision_reason": reason if number == selected else f"the round preferred candidate {selected:02d}",
                }
            )
    return rows


def _repair_ledger(iter_dir: Path) -> list[tuple[bool, bool]]:
    """The repairs a finished iteration attempted, and whether each one stuck.

    Round 0 is left out on purpose. It adopts the figure the run drew from its inherited upstream
    and is compared against nothing, so it is the starting point rather than an attempt to move
    it. Reading it as a repair is what made every iteration look like it had moved the published
    version forward, which left the round that starts from an archived variant unreachable.
    """

    ledger = read_json(iter_dir / "evolution_rounds.json") if (iter_dir / "evolution_rounds.json").is_file() else None
    if not isinstance(ledger, dict):
        return []
    repairs: list[tuple[bool, bool]] = []
    for item in ledger.get("rounds") or []:
        if not isinstance(item, dict):
            continue
        decision = item.get("repair_decision") if isinstance(item.get("repair_decision"), dict) else {}
        if int(item.get("round") or 0) == 0 or str(decision.get("reason") or "") == "initial_candidate":
            continue
        repairs.append((True, bool(item.get("accepted"))))
    return repairs


def accepted_rounds(iter_dir: Path) -> int:
    """How many of this iteration's repairs kept the figure they drew.

    An iteration whose repairs all rolled back moved nothing: the version the loop published is
    the version it started from. That is the state in which trying a different starting point is
    worth a round, and it is read here rather than inferred from a score.
    """

    return sum(1 for _attempted, stuck in _repair_ledger(iter_dir) if stuck)


def repair_rounds(iter_dir: Path) -> int:
    """How many rounds of this iteration tried to repair the figure at all.

    An iteration that never attempted a repair did not fail to repair -- there is nothing it
    settled on that the loop might have wanted to look past -- so it is not the state a restart
    from an archived variant is for. Only an attempt that did not stick is.
    """

    return len(_repair_ledger(iter_dir))


def iteration_defects(iter_dir: Path) -> list[str]:
    """The defects this iteration's rounds carried, as the round decisions named them.

    An iteration whose repairs all rolled back and an iteration that kept introducing the same
    blocker are two ways of saying the same thing -- the figure is not moving -- and the loop wants
    both for the same sentence. This is the second one: the round ledger already records what each
    repair decision introduced, and comparing that between two iterations answers "is this the same
    defect that would not go away" without measuring anything again.
    """

    ledger = read_json(iter_dir / "evolution_rounds.json") if (iter_dir / "evolution_rounds.json").is_file() else None
    if not isinstance(ledger, dict):
        return []
    found: list[str] = []
    for item in ledger.get("rounds") or []:
        if not isinstance(item, dict):
            continue
        decision = item.get("repair_decision") if isinstance(item.get("repair_decision"), dict) else {}
        for key in decision.get("new_blocking_defects") or []:
            name = str(key or "")
            if name and name not in found:
                found.append(name)
    return sorted(found)


def stage_gate_rows(iter_dir: Path) -> list[dict[str, Any]]:
    """Every round's stage gate this iteration wrote, in round order.

    The gate is the round's own account of which stage changed and what that stage read. It is read
    back off the directory rather than off the live run, so a restart sees exactly what the
    iteration left behind.
    """

    rows: list[dict[str, Any]] = []
    for path in sorted(iter_dir.glob("round_*/stage_gate.json")):
        value = _read_json_or_none(path)
        if isinstance(value, dict) and value:
            rows.append(value)
    return rows


def stage_moves(iter_dir: Path) -> dict[str, dict[str, Any]]:
    """Which stages this iteration touched, and the reading each stage's own gate took.

    A stage counts as moved when any round of the iteration changed something there. The reading is
    the last one that stage reported, whole. Both are records of what happened -- neither says a
    stage is finished, which is an answer nobody but the seat or a person can give.
    """

    moved: dict[str, dict[str, Any]] = {}
    for row in stage_gate_rows(iter_dir):
        round_index = int(row.get("round") or 0)
        for item in row.get("stages") or []:
            if not isinstance(item, dict):
                continue
            stage = str(item.get("stage") or "")
            if stage not in STAGE_ORDER:
                continue
            record = moved.setdefault(stage, {"moved": False, "rounds": [], "reasons": [], "reading": {}})
            if item.get("changed"):
                record["moved"] = True
                if round_index not in record["rounds"]:
                    record["rounds"].append(round_index)
            for reason in item.get("reasons") or []:
                text = str(reason)
                if text and text not in record["reasons"]:
                    record["reasons"].append(text)
            if isinstance(item.get("reading"), dict) and item["reading"]:
                record["reading"] = {str(key): value for key, value in item["reading"].items()}
    return moved


def iteration_layer_events(iter_dir: Path) -> list[dict[str, Any]]:
    """Every answer this iteration's rounds gave about a layer, in round order.

    Only the seat's own answer is read. A round that named no layer adds nothing: silence about a
    layer is not a statement about it, the same rule the boundary review follows about the figure.
    The whole list is kept rather than one answer per layer, because two rounds may speak about the
    same layer and a later round of a later iteration has to be able to see that the paper went
    back over it rather than only that it was once called finished.
    """

    events: list[dict[str, Any]] = []
    ledger = _read_json_or_none(iter_dir / "evolution_rounds.json")
    if not isinstance(ledger, dict):
        return events
    for item in ledger.get("rounds") or []:
        if not isinstance(item, dict):
            continue
        reflection = item.get("reflection") if isinstance(item.get("reflection"), dict) else {}
        settled = normalize_layer_settled(reflection.get("layer_settled"))
        stage = str(settled.get("stage") or "")
        if not stage:
            continue
        events.append(
            {
                "stage": stage,
                "round": int(item.get("round") or 0),
                "by": "model",
                "because": str(settled.get("because") or ""),
            }
        )
    return events


def iteration_layer_settled(iter_dir: Path) -> dict[str, str]:
    """The layers this iteration's reflections said they had finished with, and why.

    Only the last answer per layer survives: two rounds that settle the same layer say the same
    thing once. A layer no round named is not settled.
    """

    found: dict[str, str] = {}
    for event in iteration_layer_events(iter_dir):
        found[str(event["stage"])] = str(event.get("because") or "")
    return found


def iteration_human_answers(iter_dir: Path) -> list[dict[str, Any]]:
    """What a person answered at this iteration's own judgement points, one row per round.

    Three of the four things a person is asked are about the direction and are read here: the layer
    they called finished with, the layer they said the fault had moved to, and the layer they
    answered about and left open. The figure choice is the fourth and is not read here at all -- it
    decided the round already, inside the round, and re-reading it would let a second reader
    overrule the first.

    A round nobody answered writes a file with nothing in it, and it is skipped rather than read
    as "the person said no": the two are different facts and only one of them is an answer.
    """

    rows: list[dict[str, Any]] = []
    for path in sorted(Path(iter_dir).glob("round_*/human_verdict.json")):
        body = _read_json_or_none(path)
        if not isinstance(body, dict):
            continue
        answer = body.get("answer") if isinstance(body.get("answer"), dict) else {}
        if not answer.get("answered"):
            continue
        rows.append(
            {
                "round": int(body.get("round") or 0),
                "stage": str(body.get("stage") or ""),
                "answered_by": str(answer.get("by") or "human"),
                # Whether the person left this layer open rather than calling it finished with. The
                # card asks about the layer the round worked on, so "not finished with" and "the
                # layer still has work in it" are the same sentence, and it is read where the next
                # iteration's fork is decided: the answer the person already gave moves the round,
                # so a second card does not have to ask the same thing again.
                "left_open": bool(str(body.get("stage") or "") and not str(body.get("settled_layer") or "")),
                "settled_layer": str(body.get("settled_layer") or ""),
                "settled_because": str(body.get("settled_because") or ""),
                "direction_layer": str(body.get("direction_layer") or ""),
                "choice": str(answer.get("choice") or ""),
                "by": str(answer.get("by") or "human"),
                "note": str(answer.get("note") or ""),
            }
        )
    return rows


def person_open_layer(iter_dir: Path) -> dict[str, Any]:
    """The layer a person answered about and left open, from one iteration's own cards.

    A person is asked about the layer the round worked on and can answer three ways. "Finished
    with" is a seal and is read where seals are read. Naming a layer above the one they were asked
    about is a move back up. Leaving it open is the third, and it was the one answer nothing acted
    on: the round read it as "do not seal this", and the layer the next iteration worked on was
    taken from the seat's own review instead -- so a person who said "this layer is not good
    enough yet" watched the next iteration go and polish the structure below it.

    This is the loop's half of "the person's answer comes first": the layer they left open is the
    layer the next iteration works on, and their word is the argument for going back over it, so
    nothing has to be confirmed twice. The last round of the iteration that says it wins, because
    that is the most recent thing the person saw: a round that left a layer open and a later round
    that called the same layer finished are two answers about one layer, and only the later one is
    what the person thinks now.
    """

    latest: dict[str, Any] = {}
    for row in iteration_human_answers(iter_dir):
        if str(row.get("stage") or "") in STAGE_ORDER:
            latest = row
    if not latest or not latest.get("left_open"):
        return {}
    return {
        "stage": str(latest["stage"]),
        "round": int(latest.get("round") or 0),
        "by": str(latest.get("answered_by") or "human"),
        "note": str(latest.get("note") or ""),
        "because": "the person answered about this layer and left it open, so this iteration plans it again",
    }


def moved_iterations(paper_dir: Path, *, upto: int = 0) -> list[int]:
    """The iterations of one paper in which some layer of the pipeline actually changed.

    Read off each iteration's own stage gates rather than off the live state, so the rounds a
    paper worked in can be counted again after a restart and come out the same.
    """

    numbers: list[int] = []
    if not paper_dir.is_dir():
        return numbers
    for candidate in sorted(paper_dir.glob("iter_*")):
        number = iteration_number(candidate.name)
        if number is None or (upto and number > int(upto)):
            continue
        if any(item.get("moved") for item in stage_moves(candidate).values()):
            numbers.append(int(number))
    return numbers


def layer_answer_history(paper_dir: Path, *, upto: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every settle answer this paper's rounds left, up to an iteration, and every reopen.

    The two lists are what a layer walk needs and nothing else: which round each layer was
    finished with, who said so, and which layers were opened again afterwards. Both are read from
    the records rather than from the live state, so recomputing the walk for an earlier iteration
    cannot pick up anything that happened after it.
    """

    settled: list[dict[str, Any]] = []
    if paper_dir.is_dir():
        for candidate in sorted(paper_dir.glob("iter_*")):
            number = iteration_number(candidate.name)
            if number is None or (upto and number > int(upto)):
                continue
            for event in iteration_layer_events(candidate):
                settled.append({**event, "iteration": int(number)})
    return settled, layer_reopen_events(paper_dir, upto=int(upto))


def layer_reopen_events(paper_dir: Path, *, upto: int = 0) -> list[dict[str, Any]]:
    """The reopen answers the loop recorded for one paper's rounds.

    A reopen is the loop's own act -- it writes the layer open and says who asked for it -- so it
    is read from the loop's event log rather than from a round, where nothing would have been
    written had the answer not moved a layer.
    """

    events: list[dict[str, Any]] = []
    path = paper_dir.parent.parent / CONTINUOUS_EVENTS
    if not path.is_file():
        return events
    paper = str(paper_dir.name)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or str(row.get("event") or "") != "layer_reopened":
            continue
        if str(row.get("paper") or "") != paper:
            continue
        number = int(row.get("iteration") or 0)
        if not number or (upto and number > int(upto)):
            continue
        events.append(
            {
                "stage": str(row.get("stage") or ""),
                "iteration": number,
                "by": str(row.get("by") or ""),
                "because": str(row.get("why") or ""),
            }
        )
    return events


#: The metrics file the run already writes beside itself, which also carries the loop's own
#: progress block. One file rather than two, because both sets of numbers describe the same
#: iteration and a reader looking for "how did this round go" should not have to know which of two
#: files to open. The block is added under its own key, so the run's five numbers are untouched.
ROUND_METRICS_FILE = "evolution_metrics.json"
#: The loop's running trend, one line per iteration, beside the library rather than in a run: the
#: per-run metrics file is deleted with the run, and a trend that can only be read by opening every
#: run is not a trend. The run appends its own five-number line here too.
ROUND_METRICS_LEDGER = "evolution_metrics.jsonl"


def iteration_cross_rounds(iter_dir: Path) -> list[dict[str, Any]]:
    """The cross-round comparison this iteration made, as a one-item list of evidence.

    A list rather than the record alone because the caller counts comparisons, and a round that
    made none must contribute zero items rather than one item that says "nothing happened".
    """

    payload = _read_json_or_none(iter_dir / CROSS_ROUND_FILE)
    return [payload] if isinstance(payload, dict) else []


def iteration_reflections(iter_dir: Path) -> list[dict[str, Any]]:
    """Every reflection this iteration's rounds wrote, as the seat answered them."""

    found: list[dict[str, Any]] = []
    for path in sorted(iter_dir.glob("round_*/reflection*.json")):
        value = _read_json_or_none(path)
        if isinstance(value, dict):
            found.append(value)
    return found


def iteration_catalogue_gap(iter_dir: Path) -> dict[str, Any]:
    """A round that moved the figure with a move the catalogue cannot hand another round.

    Not a refusal: the round won, and there is simply nothing to file, because the move it won
    with is not one another round can be given. What the record is for is being counted -- the
    library cannot grow along a move the catalogue does not hold, and how often that happens is
    what says whether the catalogue is the thing to widen.
    """

    payload = _read_json_or_none(iter_dir / "experience_not_a_repair.json")
    if not isinstance(payload, dict):
        return {"count": 0, "moves": []}
    move = str(payload.get("action") or "")
    return {"count": 1, "moves": [move] if move else []}


def iteration_refusals(iter_dir: Path) -> dict[str, int]:
    """Why this iteration's rounds did not become experiences, by reason."""

    counts: dict[str, int] = {}
    rejected = _read_json_or_none(iter_dir / "experience_rejected.json")
    if isinstance(rejected, dict):
        reason = str(rejected.get("reason") or "unspecified")
        counts[reason] = counts.get(reason, 0) + 1
    if (iter_dir / "mechanism_self_repair.json").is_file():
        counts["mechanism_self_repair"] = counts.get("mechanism_self_repair", 0) + 1
    return counts


def layers_from_ledger(ledger: Any, *, first_moves: Any = (), readings: Any = ()) -> dict[str, dict[str, Any]]:
    """The layer record one iteration's ledger carries, in the shape the state keeps it.

    The ledger is written from the state's own layer record, so reading it back says what the round
    knew about every layer: which of them an answer settled, who answered and when, how often each
    was opened again, and where each last moved. The first move and the last reading are not in the
    ledger -- it carries one iteration's moves -- so they are read off the iterations' own gates. A
    stage with nothing to say at all is left out, which is the shape the state uses.
    """

    firsts = first_moves if isinstance(first_moves, dict) else {}
    lasts = readings if isinstance(readings, dict) else {}
    layers: dict[str, dict[str, Any]] = {}
    for row in (ledger.get("layers") if isinstance(ledger, dict) else None) or []:
        if not isinstance(row, dict):
            continue
        stage = str(row.get("stage") or "")
        if stage not in STAGE_ORDER:
            continue
        settled_at = int(row.get("settled_at") or 0)
        last_move_at = int(row.get("last_move_at") or 0)
        reopen_count = int(row.get("reopen_count") or 0)
        if not (settled_at or last_move_at or reopen_count):
            continue
        layers[stage] = {
            "settled_at": settled_at or None,
            "settled_by": str(row.get("settled_by") or ""),
            "because": str(row.get("because") or ""),
            "first_move_at": int(firsts.get(stage) or last_move_at),
            "last_move_at": last_move_at or None,
            "last_reading": dict(lasts.get(stage) or {}),
            "reopen_count": reopen_count,
        }
    return layers


def layer_move_stamps(paper_dir: Path, *, upto: int) -> dict[str, dict[str, Any]]:
    """Every layer's first move and most recent reading, across the iterations up to one.

    Read off each iteration's own stage gates, which is the source ``note_layer_moves`` stamps a
    round from, so a layer that had already moved reads back with the iteration it first moved in:
    that is the iteration the walk measures the cost of settling the layer from.
    """

    first: dict[str, int] = {}
    last: dict[str, dict[str, Any]] = {}
    if not paper_dir.is_dir():
        return {"first": first, "last": last}
    for candidate in sorted(paper_dir.glob("iter_*")):
        number = iteration_number(candidate.name)
        if number is None or (upto and number > int(upto)):
            continue
        for stage, taken in stage_moves(candidate).items():
            if not taken.get("moved"):
                continue
            first.setdefault(stage, int(number))
            last[stage] = dict(taken.get("reading") or {})
    return {"first": first, "last": last}


def layer_ledger(
    iter_dir: Path,
    *,
    layers: Any = None,
    iteration: int = 0,
    settle_events: Any = (),
    reopen_events: Any = (),
    worked_rounds: Any = (),
) -> dict[str, Any]:
    """One iteration's account of every layer: what moved, what it read, what a model settled.

    The ledger is assembled from the round gates the iteration already wrote and the settle answers
    the seats already gave, so it adds no measurement and decides nothing. A layer reads as settled
    only when an answer said so -- a person's or the model's, recorded the same way -- and the
    reading rides along as the evidence behind that answer, never as the answer itself.

    The answers and the rounds the paper worked in are carried in the file as well as folded into
    the walk, because the walk is the number a reader wants and the answers are the record behind
    it: a reader that finds the cost of a layer surprising has to be able to see which round said
    so without going back to the state file that the run rewrites every time it moves.
    """

    state = layers if isinstance(layers, dict) else {}
    moves = stage_moves(iter_dir)
    rows: list[dict[str, Any]] = []
    for order, stage in enumerate(STAGE_ORDER):
        record = state.get(stage) if isinstance(state.get(stage), dict) else {}
        taken = moves.get(stage) or {}
        rows.append(
            {
                "stage": stage,
                "order": order,
                "moved": bool(taken.get("moved")),
                "moved_rounds": list(taken.get("rounds") or []),
                "reasons": list(taken.get("reasons") or []),
                "reading": dict(taken.get("reading") or {}),
                "settled": bool(record.get("settled_at")),
                "settled_at": int(record.get("settled_at") or 0),
                "settled_by": str(record.get("settled_by") or ""),
                "because": str(record.get("because") or ""),
                "last_move_at": int(record.get("last_move_at") or 0),
                "reopen_count": int(record.get("reopen_count") or 0),
            }
        )
    gates = stage_gate_rows(iter_dir)
    metrics = (gates[-1].get("figure_metrics") if gates else {}) or {}
    walked = layer_walk(
        state,
        stage_order=STAGE_ORDER,
        settle_events=settle_events,
        reopen_events=reopen_events,
        worked_rounds=worked_rounds,
    )
    return {
        "contract": LAYER_LEDGER_CONTRACT,
        "version": 1,
        "iteration": int(iteration),
        "work_face": work_face(state),
        "settled": [row["stage"] for row in rows if row["settled"]],
        "figure_metrics": {"after": dict(metrics.get("after") or {}), "delta": dict(metrics.get("delta") or {})},
        "layers": rows,
        "walk": walked,
        "settle_events": [dict(event) for event in settle_events or [] if isinstance(event, dict)],
        "reopen_events": [dict(event) for event in reopen_events or [] if isinstance(event, dict)],
        "reopened_layers": sorted({str(event.get("stage") or "") for event in reopen_events or [] if isinstance(event, dict) and str(event.get("stage") or "")}),
        "worked_rounds": [int(item) for item in worked_rounds or []],
        "directions": acceptance_directions(iter_dir),
        "note": "recorded only: a layer is settled by an answer, never by a reading or by the loop",
    }


def acceptance_directions(iter_dir: Path) -> list[dict[str, Any]]:
    """What the acceptance seat answered about the round's own layer, one row per round.

    A round the seat was not asked leaves no file and adds no row, so the list says exactly how
    many comparisons the seat was given the direction question on and what it answered. It is a
    record: nothing in the loop reads it to decide anything unless the config gave that seat the
    direction.
    """

    rows: list[dict[str, Any]] = []
    for path in sorted(Path(iter_dir).glob(f"round_*/{ACCEPTANCE_DIRECTION_FILE}")):
        body = _read_json_or_none(path)
        if not isinstance(body, dict):
            continue
        rows.append(
            {
                "round": int(body.get("round") or 0),
                "authority": str(body.get("authority") or ""),
                "asked_about": str(body.get("asked_about") or ""),
                "answer": dict(body.get("answer") or {}),
                "problem": str(body.get("problem") or ""),
            }
        )
    return rows


def previous_round_summary(entry: dict[str, Any]) -> dict[str, Any]:
    """What the round before this one changed, and how it turned out.

    The seat that says where the next round should start is otherwise looking at one figure with no
    history at all. This is the cheapest form of that history: the last iteration's own record, the
    defects it introduced, and how many of its repairs stuck.
    """

    history = entry.get("history") if isinstance(entry.get("history"), list) else []
    last = history[-1] if history and isinstance(history[-1], dict) else {}
    if not last:
        return {}
    reuse = last.get("reuse") if isinstance(last.get("reuse"), dict) else {}
    return {
        "iteration": int(last.get("iteration") or 0),
        "status": str(last.get("status") or ""),
        "fork_before": str(last.get("fork_before") or ""),
        "accepted_rounds": int(last.get("accepted_rounds") or 0),
        "best_score": last.get("best_score"),
        "inherited": bool(reuse.get("accepted")),
        "defects_this_round": [str(item) for item in entry.get("last_defect_ids") or []],
    }


def layer_plan(entry: dict[str, Any]) -> dict[str, Any]:
    """The direction this round is on, in the shape the run's own reflection is handed it.

    The sheet travels to the seat that chooses where the round forks. The answer the direction
    turns on is not that seat's: it is the reflection's, which is written inside the run, after the
    round has moved and been measured. So the same record travels into the run -- the layer the round
    is on, the order, which layers are already finished with and by whom, and the question itself --
    and a reflection that says a layer is finished with is what moves the next round one layer down.
    Nothing here decides anything: it is the sheet's own content, cut down to what a reflection can
    act on.
    """

    layers = entry.get("layers") if isinstance(entry.get("layers"), dict) else {}
    work = work_face(layers)
    if not work:
        return {}
    settled: list[dict[str, Any]] = []
    still_open: list[str] = []
    for stage in STAGE_ORDER:
        record = layers.get(stage) if isinstance(layers.get(stage), dict) else {}
        if not record.get("settled_at"):
            still_open.append(stage)
            continue
        settled.append(
            {
                "stage": stage,
                "by": str(record.get("settled_by") or ""),
                "at": int(record.get("settled_at") or 0),
                "because": str(record.get("because") or ""),
            }
        )
    return {
        "working_on": work,
        "layer_order": list(STAGE_ORDER),
        "settled": settled,
        "still_open": still_open,
        "ask": progress_ask(work),
        "note": (
            "The figure is built one layer at a time from the top down: a layer is planned again "
            "only while it is not yet good, and once it is good the next round works on the layer "
            f"below it. This round is on {work}. Say in layer_settled whether it is good enough to "
            "leave alone -- and name the reading that says so -- and the next round moves down; "
            "leave it empty and say what is still wrong with it, and the round stays here. Working "
            "below an open layer needs no reason."
        ),
    }


def progress_ask(work: str) -> str:
    """The question this round has to answer about the layer it is working on.

    Rendered onto the sheet rather than left implicit. Whether a layer is finished with is the
    model's judgement and nothing in the loop may make it, so the question has to be on the paper in
    front of the seat that answers it: a round that is never asked whether the layer it is on is done
    with keeps working on whatever looks worst, which is how a paper spends iteration after iteration
    re-doing a layer it had already got right. The sentence names the layer, names the field the
    answer goes in, says in plain words that a layer which is already good is what carries the round
    down to the layer below (so the judgement reads as progress, not as a verdict to withhold), and
    repeats the one rule that is easy to get backwards -- working below an open layer needs no
    reason, going back over a finished one does.
    """

    if not work:
        return ""
    return (
        f"The layer this run is working on is {work}. Judge it, then answer for it: is this layer "
        f"already good -- planning it again would not make the figure better, and no reading below "
        f"shows a defect in it? If it is, say so, and the next round moves down to the layer below; "
        f"that is how the figure gets built, one good layer at a time. Say it by filling "
        f'layer_settled with {{"stage": "{work}", "because": "<the reading that says it is good>"}}. '
        f"If it is not good yet, leave layer_settled empty and say what is still wrong with it, and "
        f"this round stays on it. Working below an open layer needs no reason; going back over a "
        f"layer already finished with does."
    )


def _same_dir(left: Any, right: Any) -> bool:
    """Whether two recorded directories name the same place, spelled either way."""

    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except OSError:
        return str(left).rstrip("/\\") == str(right).rstrip("/\\")


def _iteration_of_dir(path: Any) -> int | None:
    """The iteration a recorded directory sits in, however deep inside it the path points.

    A kept figure is named by its round and its candidate -- ``iter_129/round_00/candidate_00`` --
    so the iteration is never the last part of the path. Reading it off the deepest part that parses
    is what lets the sheet say which iteration the best version came from.
    """

    for part in reversed(Path(str(path or "")).parts):
        number = iteration_number(part)
        if number is not None:
            return number
    return None


def progress_history(entry: dict[str, Any], *, inherited: str = "") -> dict[str, Any]:
    """Where the figure this round inherits stands against the best this paper ever reached.

    A judgement that a layer is finished with is a judgement about a figure, so the figure has to
    arrive with its own standing: the best directory the paper ever published, what it read, how many
    iterations it took, and whether this round inherits exactly that. The last of those is the fact
    the seat needs most. A round working on a figure that is already the best the paper has reached is
    spending its move on polish, and one working below that figure has ground to make up -- and
    neither can be told from a directory listing.
    """

    best = str(entry.get("best") or "")
    reading = entry.get("best_score")
    archive = entry.get("archive") if isinstance(entry.get("archive"), list) else []
    best_iteration = _iteration_of_dir(best) if best else None
    measured = float(reading) if _is_number(reading) else None
    inherits = bool(best) and bool(inherited) and _same_dir(best, inherited)
    if best_iteration and measured is not None:
        note = (
            f"the figure this round inherits is the best version this paper has reached so far "
            f"(iteration {best_iteration:02d}, reading {measured:.4f})"
            if inherits
            else f"the best version this paper has reached so far is iteration {best_iteration:02d} "
            f"(reading {measured:.4f}); this round inherits something other than that"
        )
    else:
        note = "no version of this paper has been measured against another one yet"
    return {
        "iterations_run": int(entry.get("iteration") or 0),
        "best_dir": best,
        "best_iteration": best_iteration,
        "best_reading": measured,
        "archived_versions": len(archive),
        "inherits_the_best": inherits,
        "note": note,
    }


#: How many earlier ledgers the sheet reads back when the state carries no reading for a layer.
READING_LOOKBACK = 8


def older_layer_readings(paper_dir: Path, *, before: int) -> dict[str, dict[str, Any]]:
    """The readings the newest ledgers below ``before`` carry, keyed by stage.

    The state keeps a layer's reading only while the layer keeps moving, so a layer last worked on
    several iterations ago reaches the sheet as an empty cell -- and an empty cell is not "nothing to
    see", it is a question asked with no evidence beside it. The ledgers are the same record and they
    are on disk, so the sheet reads back a few of them and carries each reading forward with the
    iteration it came from: an old number labelled as old is evidence, the same number unlabelled is
    a claim.
    """

    history: list[tuple[int, Path]] = []
    if paper_dir.is_dir():
        for candidate in paper_dir.glob("iter_*"):
            number = iteration_number(candidate.name)
            if number is not None and number < int(before):
                history.append((number, candidate))
    carried: dict[str, dict[str, Any]] = {}
    for number, candidate in sorted(history, reverse=True)[:READING_LOOKBACK]:
        ledger = _read_json_or_none(candidate / "layer_ledger.json")
        rows = ledger.get("rows") if isinstance(ledger, dict) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            stage = str(row.get("stage") or "")
            reading = row.get("reading")
            if stage and stage not in carried and isinstance(reading, dict) and reading:
                carried[stage] = {"reading": dict(reading), "iteration": int(number)}
        if len(carried) >= len(STAGE_ORDER):
            break
    return carried


def progress_sheet(
    entry: dict[str, Any],
    spec: dict[str, Any],
    *,
    iteration: int,
    readings: Any = None,
    inherited: str = "",
) -> dict[str, Any]:
    """Where this paper stands in the pipeline, written for the seat that decides the next step.

    The sheet names the goal and the order the layers are built in, then says, layer by layer, what
    a model has finished with, what that layer last read and when it read it, and when the layer was
    last worked on -- followed by where this figure stands against the best the paper ever reached,
    the question this round has to answer, and the round before this one and what it changed.
    Everything on it is a record: the readings describe the figure and never say a layer is done
    with, which only an answer says.
    """

    layers = entry.get("layers") if isinstance(entry.get("layers"), dict) else {}
    earlier = readings if isinstance(readings, dict) else {}
    rows: list[dict[str, Any]] = []
    for order, stage in enumerate(STAGE_ORDER):
        record = layers.get(stage) if isinstance(layers.get(stage), dict) else {}
        reading = record.get("last_reading") if isinstance(record.get("last_reading"), dict) else {}
        reading_from = int(record.get("last_move_at") or 0)
        if not reading:
            carried = earlier.get(stage) if isinstance(earlier.get(stage), dict) else {}
            carried_reading = carried.get("reading") if isinstance(carried.get("reading"), dict) else {}
            if carried_reading:
                reading = dict(carried_reading)
                reading_from = int(carried.get("iteration") or reading_from)
        rows.append(
            {
                "order": order,
                "stage": stage,
                "settled": bool(record.get("settled_at")),
                "settled_at": int(record.get("settled_at") or 0),
                "settled_by": str(record.get("settled_by") or ""),
                "because": str(record.get("because") or ""),
                "last_move_at": int(record.get("last_move_at") or 0),
                "last_reading": dict(reading),
                "reading_from": reading_from,
            }
        )
    work = work_face(layers)
    return {
        "contract": PROGRESS_SHEET_CONTRACT,
        "version": 1,
        "paper": str(spec.get("name") or ""),
        "iteration": int(iteration),
        "goal": PROGRESS_SHEET_GOAL,
        "layer_order": list(STAGE_ORDER),
        "now_working_on": work,
        "ask": progress_ask(work),
        "stalled_streak": int(entry.get("stalled_streak") or 0),
        "layers": rows,
        "history": progress_history(entry, inherited=inherited),
        "last_round": previous_round_summary(entry),
        "note": (
            "the readings describe the figure and never say a layer is finished with. Say which layer "
            "you are finished with and what this round should change; the layer you name moves the "
            "next round down the order. Working below the open layer needs no reason. Going back over a "
            "layer that is finished with does: name that layer, the requirement that went unwritten "
            "there, and a reading from the report in front of you that shows it. Overlap, out-of-bounds "
            "objects and more crossings are not that reason -- they are things to repair on the figure "
            "you have."
        ),
    }


def cross_round_due(entry: dict[str, Any], number: int, every: int) -> bool:
    """Whether this round's figure should be put beside the best version the paper had reached.

    Two things ask for it: a layer the model finished with this round -- the figure changed level,
    and the question is whether it changed level upward -- and the cadence itself, so a paper that
    settles nothing still gets a global reading every few rounds instead of never.
    """

    if not int(number or 0):
        return False
    settled_now = any(int((record or {}).get("settled_at") or 0) == int(number) for record in (entry.get("layers") or {}).values() if isinstance(record, dict))
    return bool(settled_now or int(number) % max(1, int(every or 1)) == 0)


def cross_round_baseline(archive: Any, *, before: int, window: int = RESTART_WINDOW_ITERATIONS) -> dict[str, Any]:
    """The best version this paper had reached before a round, read off its own archive.

    The comparison needs a reference that is not the version the round just replaced, or the answer
    would only ever say "the newest drawing is the newest drawing". The absolute score is the one
    reading two variants can be put side by side on, so the reference is the archived variant with
    the highest absolute score among the recent rounds, and a version with no absolute reading
    cannot be the reference at all.
    """

    rows = [row for row in (archive or []) if isinstance(row, dict)]
    eligible = [
        row
        for row in rows
        if int(row.get("iter") or 0) < int(before or 0)
        and abs(int(before or 0) - int(row.get("iter") or 0)) <= max(1, int(window or 1))
        and _is_number(row.get("absolute_score"))
    ]
    if not eligible:
        return {}
    best = max(eligible, key=lambda row: (float(row["absolute_score"]), int(row.get("iter") or 0), int(row.get("round") or 0), int(row.get("candidate") or 0)))
    return {
        "dir": str(best.get("dir") or ""),
        "source": str(best.get("source") or ""),
        "iter": int(best.get("iter") or 0),
        "round": int(best.get("round") or 0),
        "candidate": int(best.get("candidate") or 0),
        "absolute_score": float(best["absolute_score"]),
    }


def cross_round_record(*, paper: str, iteration: int, current: dict[str, Any], baseline: dict[str, Any], answer: Any = None, reason: str = "") -> dict[str, Any]:
    """One cross-round comparison, whether it was made or not, and never a verdict on the round."""

    record = answer if isinstance(answer, dict) else {}
    choice = str(record.get("preferred_candidate") or record.get("pairwise_choice") or "").strip().upper()
    if choice in {"SAME", "EQUAL", "EQUIVALENT"}:
        choice = "TIE"
    if choice not in CROSS_ROUND_CHOICES:
        choice = ""
    payload = {
        "contract": CROSS_ROUND_CONTRACT,
        "version": 1,
        "paper": str(paper),
        "iteration": int(iteration),
        "current": dict(current),
        "history_best": dict(baseline),
        "preference": choice,
        "verdict": CROSS_ROUND_CHOICES.get(choice, "not_compared"),
        "reason": str(reason or ("" if choice else "no comparison was made")),
        "answer": dict(record) if record else {},
        "gate": False,
        "note": (
            "recorded only: a cross-round comparison is evidence for the layer ledger and for the "
            "trend. It never accepts and never rolls back the round it describes."
        ),
    }
    return payload


def restart_from_archive(archive: list[dict[str, Any]], *, exclude: Iterable[str] = (), window: int = RESTART_WINDOW_ITERATIONS) -> dict[str, Any]:
    """The archived variant a later round should start from, or nothing when none is eligible.

    A source the caller excludes never wins -- the version the loop already settled on is not an
    exploration -- and neither does one whose artifacts have since gone, because the next run has
    to inherit something that exists.

    Ordering on the measured score is ordering on a relative reading, and the relative reading of
    a variant the comparator defaulted on is the default, not a quality. Ordering on it therefore
    returned the same old drawing every time. So the candidates are cut down to the recent window
    first, and within the window the variants carrying an absolute reading of their own are
    preferred; the choice is then made on that reading, ties going to the more recent drawing. A
    window in which nothing carries an absolute reading falls back to the measured score rather
    than offering nothing, and the round is told why it is not simply continuing from the line it
    published, so a restart that keeps choosing the same variant is visible.
    """

    blocked = {str(item or "") for item in exclude}
    eligible = [
        row
        for row in archive
        if isinstance(row, dict) and str(row.get("source") or "") and str(row["source"]) not in blocked and Path(str(row["source"])).is_dir()
    ]
    if not eligible:
        return {}
    latest = max(int(row.get("iter") or 0) for row in eligible)
    window_start = max(1, latest - max(1, int(window)) + 1)
    recent = [row for row in eligible if int(row.get("iter") or 0) >= window_start]
    pool = recent or eligible
    scored = [row for row in pool if _is_number(row.get("absolute_score"))]
    if scored:
        ranked = scored
        column = "absolute_score"
        restart_reason = f"the published line kept nothing, so the best variant of the last {len(pool)} archived attempt(s) by its own geometry-and-density reading is inherited"
    else:
        ranked = pool
        column = "choice_score"
        restart_reason = f"the published line kept nothing and no variant of the last {len(pool)} archived attempt(s) carries an absolute reading, so the best measured comparison score is inherited"
    chosen = max(
        ranked,
        key=lambda row: (
            _as_number(row.get(column)),
            int(row.get("iter") or 0),
            int(row.get("round") or 0),
            int(row.get("candidate") or 0),
        ),
    )
    return {
        "source": str(chosen["source"]),
        "dir": str(chosen.get("dir") or ""),
        "iter": int(chosen.get("iter") or 0),
        "round": int(chosen.get("round") or 0),
        "candidate": int(chosen.get("candidate") or 0),
        "score": chosen.get("score"),
        "absolute_score": chosen.get("absolute_score"),
        "choice_score": chosen.get("choice_score"),
        "decision_reason": str(chosen.get("decision_reason") or ""),
        "restart_reason": restart_reason,
        "window_start": window_start,
    }


def _forecast_sentence(forecast: dict[str, Any]) -> str:
    """One line for the report: what the next iteration will inherit and what it will re-plan."""

    if not forecast.get("requested"):
        return "no inheritance was requested"
    if not forecast.get("accepted"):
        return f"cannot continue from the previous version: {forecast.get('declined_reason') or 'no reason given'}"
    reused = ", ".join(forecast.get("reused_stages") or []) or "nothing"
    replanned = ", ".join(forecast.get("replanned_stages") or [])
    if not replanned:
        return f"will inherit {reused}"
    return f"will inherit {reused}; will re-plan {replanned}"


def _disk_free(path: Path) -> dict[str, float] | None:
    """Both readings of the volume holding the output: the share free and the GB free.

    Two numbers because two callers asked: the shipped guard is a share, and a caller who
    configures an absolute floor has to be answered in the unit it wrote down.
    """

    target = path if path.exists() else path.parent
    try:
        usage = shutil.disk_usage(str(target))
    except OSError:
        return None
    if not usage.total:
        return None
    return {"ratio": usage.free / usage.total, "free_gb": usage.free / (1024 ** 3), "total_gb": usage.total / (1024 ** 3)}


def _free_record(value: Any) -> dict[str, float | None] | None:
    """Read one disk probe answer: the default record, a bare share, or nothing."""

    if value is None:
        return None
    if isinstance(value, dict):
        ratio = value.get("ratio")
        free_gb = value.get("free_gb")
        if ratio is None and free_gb is None:
            return None
        return {
            "ratio": float(ratio) if ratio is not None else None,
            "free_gb": float(free_gb) if free_gb is not None else None,
        }
    return {"ratio": float(value), "free_gb": None}


def _stopped(stop_file: str | Path) -> bool:
    return bool(str(stop_file or "").strip()) and Path(str(stop_file)).exists()


def _json_body(data: Any) -> str:
    """The bytes a report is written as, with lone surrogates replaced so it can be encoded."""

    return json.dumps(sanitize_surrogates(data), indent=2, ensure_ascii=False)


def _write_text_atomic(path: Path, body: str) -> Path:
    """Write a report so a reader sees either the whole old one or the whole new one.

    A state file is rewritten while the loop keeps working: the wave workers and the validation
    worker all write it, from their own threads. A process killed in the middle of a plain write
    leaves a truncated file, and the next call then reads a state that is neither the one before
    nor the one after. The body lands beside the target and is moved onto it.
    """

    ensure_dir(path.parent)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(body, encoding="utf-8", newline="\n")
    os.replace(temp, path)
    return path


def _read_json_or_none(path: Path) -> Any:
    """Read a report the loop is only asking about, and answer "not there" when it is not.

    A run that was interrupted can leave a report missing or half-written. A loop that died
    because it could not read a file it merely hoped for would lose every wave it had already
    paid for, and the wave it was about to run.
    """

    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


class _Loop:
    """One continuous run: the state, the clock, the counters and the writers."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        out: Path,
        library: str,
        stop_file: str,
        evolve_runner: Callable[..., dict[str, Any]],
        review_runner: Callable[[Path, dict[str, Any]], dict[str, Any]],
        clock: Callable[[], float],
        disk_usage: Callable[[Path], Any],
        max_waves: int,
        validate_runner: Callable[..., dict[str, Any]] = run_direct_experience_validation,
        cross_round_runner: Callable[[Path, Path, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.config = config
        self.out = out
        self.library = library
        self.stop_file = stop_file
        self.evolve_runner = evolve_runner
        self.review_runner = review_runner
        self.clock = clock
        self.disk_usage = disk_usage
        self.disk_floor = float(config.get("disk_floor") or DISK_FLOOR)
        self.disk_floor_gb = float(config.get("disk_floor_gb") or DEFAULT_DISK_FLOOR_GB)
        self.max_waves = int(max_waves)
        self.validate_runner = validate_runner
        self.cross_round_runner = cross_round_runner
        self.cross_round_every = max(1, int(config.get("cross_round_every") or DEFAULT_CROSS_ROUND_EVERY))
        #: How many iterations this call lets a paper run before it is left out of the waves.
        #: Zero is "no target", which is how every run behaved before the key existed.
        self.target_iterations = max(0, int(config.get("target_iterations") or 0))
        self.validation = dict(config.get("validation") or {})
        self.reopen_after = max(1, int(config.get("reopen_after_stalled_rounds") or DEFAULT_REOPEN_AFTER_STALLED_ROUNDS))
        #: Whether a person stands in a seat for some papers, and what the loop does without one.
        #: An absent block reads as "no seat is held", which is how every run behaved before it.
        self.human_seat = dict(config.get("human_seat") or {})
        #: The cards this run files. A question nobody sits at costs nothing: the queue answers with
        #: the model's own answer and writes no card, so a run with no person at the seat is
        #: unchanged. The queue is built once because the hold has to outlive a single round.
        self.human_queue = HumanQueue(
            root=str(self.human_seat.get("dir") or out),
            seats=tuple(self.human_seat.get("kinds") or ()),
            hold_hours=float(self.human_seat.get("hold_hours") or 0.0),
            on_timeout=str(self.human_seat.get("on_timeout") or "model"),
            enabled=bool(self.human_seat.get("enabled")),
        )
        #: The last reading the loop took, kept so the report can carry the number the stop was
        #: actually decided on rather than the one the caller hoped for.
        self.disk_reading: dict[str, float | None] | None = None
        self.started = float(clock())
        #: Waves this call ran, as opposed to ``state['waves']``, which counts every wave the
        #: output directory has ever run. Bounding a restart by the lifetime counter would stop
        #: it before its first wave, because the count it reads is the one it inherited.
        self.waves_this_call = 0
        self.state = load_continuous_state(out, config)
        #: Every candidate the loop has ever handed to the validation queue, kept in the state so a
        #: restart does not pay for the same measurement twice.
        self.validations: dict[str, Any] = self.state.setdefault("validations", {})
        self._validation_seen: set[str] = set(self.validations)
        self._validation_queue: Queue[dict[str, Any]] = Queue()
        self._validation_worker: threading.Thread | None = None
        self._validation_stop = threading.Event()
        #: Two papers of one wave finish at the same time, so the hand-over and the state write
        #: they both cause are serialised rather than interleaved.
        self._validation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.events_path = out / CONTINUOUS_EVENTS
        self.stop_reason = ""
        self.stop_detail = ""

    # -- reporting ---------------------------------------------------------------------------

    def event(self, name: str, **payload: Any) -> None:
        append_event(self.events_path, name, **payload)

    def spent_seconds(self) -> float:
        return max(0.0, float(self.clock()) - self.started)

    def budget_seconds(self) -> float:
        return float(self.config.get("budget_hours") or 0.0) * 3600.0

    def save_state(self) -> None:
        self.state["waves"] = int(self.state.get("waves") or 0)
        self.state["elapsed_seconds"] = round(self.spent_seconds(), 3)
        # The snapshot is taken inside the lock and serialised there: other threads are adding
        # waves, paper entries and validation rows to this structure while the loop saves, and a
        # walk that runs beside those writes is how a state file comes out half of one and half of
        # the other. Only the finished text is written, and it lands in one move.
        with self._state_lock:
            body = _json_body(self.state)
        _write_text_atomic(self.out / CONTINUOUS_STATE, body)

    def persist_paper_record(self, spec: dict[str, Any], entry: dict[str, Any]) -> None:
        """Write this paper's own account of the round that just ended.

        The wave's state file is written once per wave, so an interrupted wave used to take every
        round it had already finished down with it: the rounds were on disk, the account was not,
        and the restart paid for all of them a second time. This is the same entry, written beside
        the paper the moment its own round ends -- for a round that failed as well as one that
        finished, because the counters a failure moves are what the next attempt decides on.
        """

        payload = {
            "schema": PAPER_RECORD_SCHEMA,
            "version": 1,
            "paper": str(spec.get("name") or ""),
            "iteration": int(entry.get("iteration") or 0),
            "entry": entry,
        }
        with self._state_lock:
            body = _json_body(payload)
        _write_text_atomic(self.out / "papers" / str(spec.get("name") or "") / PAPER_RECORD_FILE, body)

    def rounds_done(self, spec: dict[str, Any]) -> bool:
        """Whether this paper has already run the iterations the caller asked for.

        A paper that has reached the target is left out of the waves rather than run again: the
        sweep reads a fixed round, and a paper carried past it by the waves that bring a slower
        paper up to the target is a round that is paid for and never read.
        """

        if not self.target_iterations:
            return False
        return int(self.state["papers"][spec["name"]].get("iteration") or 0) >= self.target_iterations

    def claim_finished_round(self, spec: dict[str, Any], entry: dict[str, Any], iter_dir: Path, number: int) -> bool:
        """Take a round that finished but was never accounted for off its own output.

        What an interrupted wave loses is the loop's *account* of a round, never the round: the
        figure it kept and what it scored, the layers it settled and where the work stands, every
        variant it drew, how many repairs it made and what came back are all written beside the
        round as it runs. So the account is rebuilt here from those records, with the same readers
        the round itself used, and the round is not paid for a second time.

        A field whose record is missing is not guessed: the round is left to be run again and the
        caller says so in the events. Half an account would decide the next round's fork on a
        premise that was never anybody's answer.
        """

        ledger = _read_json_or_none(iter_dir / LAYER_LEDGER_FILE)
        written = _read_json_or_none(iter_dir / "evolution_state.json")
        state_record = written if isinstance(written, dict) else {}
        if not isinstance(ledger, dict) or int(ledger.get("iteration") or 0) != int(number):
            return False
        stamps = layer_move_stamps(iter_dir.parent, upto=int(number))
        layers = layers_from_ledger(ledger, first_moves=stamps["first"], readings=stamps["last"])
        # The ledger carries the work face the round itself read off this record. A rebuilt record
        # that disagrees with it was rebuilt wrong, and a claim nobody can check is worse than the
        # round being run again.
        recorded_face = str(ledger.get("work_face") or "")
        if recorded_face and work_face(layers) != recorded_face:
            return False
        entry["iteration"] = int(number)
        entry["reuse_from"] = str(iter_dir)
        entry["layers"] = layers
        entry["work_face"] = work_face(layers)
        # The version this round published, with the same fallback the round's own accounting uses:
        # a round whose repairs all rolled back published no best candidate, and the first candidate
        # is then the version it drew.
        best_dir = str(state_record.get("best_candidate_dir") or "")
        if not best_dir:
            candidate = iter_dir / "round_00" / "candidate_00"
            best_dir = str(candidate) if candidate.is_dir() else ""
        if best_dir:
            entry["best"] = best_dir
        entry["best_score"] = state_record.get("best_score")
        archive = entry.setdefault("archive", [])
        known = {
            (int(item.get("iter") or 0), int(item.get("round") or 0), int(item.get("candidate") or 0))
            for item in archive
            if isinstance(item, dict)
        }
        payload = _read_json_or_none(iter_dir / "archive.json")
        for row in (payload.get("rows") if isinstance(payload, dict) else None) or []:
            if not isinstance(row, dict):
                continue
            key = (int(row.get("iter") or 0), int(row.get("round") or 0), int(row.get("candidate") or 0))
            if key in known:
                continue
            known.add(key)
            archive.append(row)
        entry["repair_rounds"] = repair_rounds(iter_dir)
        kept = accepted_rounds(iter_dir)
        if kept:
            # The published version moved forward in this round, so the next one continues from it.
            entry["best_moved_at"] = int(number)
        defects = iteration_defects(iter_dir)
        stalled = not kept or (bool(defects) and defects == [str(item) for item in entry.get("last_defect_ids") or []])
        entry["stalled_streak"] = int(entry.get("stalled_streak") or 0) + 1 if stalled else 0
        entry["last_defect_ids"] = defects
        entry.setdefault("history", []).append(
            {
                "iteration": int(number),
                "status": "claimed_from_output",
                "accepted_rounds": int(kept),
                "source": str(iter_dir),
                "note": "this round finished before the loop wrote its account; the account was read back off the round's own records",
            }
        )
        return True

    def reconcile_paper_records(self) -> list[str]:
        """Start from the accounts the papers wrote for themselves, not from the wave's.

        Two records can say a round is finished when the state file does not: the account the paper
        itself wrote the moment its round ended, and the round's own output. The first is what the
        loop wrote down, so it is taken as it stands. The second is only claimed when the round left
        everything a claim needs; a round that left less is run again, with a line in the events
        saying so rather than being inherited half-known.
        """

        claimed: list[str] = []
        for spec in self.config["papers"]:
            name = str(spec["name"])
            entry = self.state["papers"][name]
            before = int(entry.get("iteration") or 0)
            path = self.out / "papers" / name / PAPER_RECORD_FILE
            record = _read_json_or_none(path)
            if (
                isinstance(record, dict)
                and str(record.get("paper") or "") == name
                and isinstance(record.get("entry"), dict)
                and int(record.get("iteration") or 0) > before
            ):
                # Merged rather than replaced: a record is one round's account, and anything the
                # loop wrote about this paper after that round is not the record's to drop.
                entry.update(dict(record["entry"]))
                claimed.append(name)
                self.event("paper_record_recovered", paper=name, was=before, now=int(record.get("iteration") or 0), source=str(path))
                continue
            number = before + 1
            iter_dir = self.out / "papers" / name / ("iter_%02d" % number)
            if not (iter_dir / "judgement.json").is_file():
                continue
            if self.claim_finished_round(spec, entry, iter_dir, number):
                claimed.append(name)
                self.event("round_claimed_from_output", paper=name, iteration=int(number), out=str(iter_dir), was=before)
            else:
                self.event(
                    "round_on_disk_without_record",
                    paper=name,
                    iteration=int(number),
                    out=str(iter_dir),
                    detail="this round finished but left no record a claim could be built from, so it is run again",
                )
        if claimed:
            self.save_state()
        return claimed

    def stop(self, reason: str, detail: str) -> None:
        self.stop_reason = str(reason)
        self.stop_detail = str(detail)

    # -- the loop ----------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        ensure_dir(self.out)
        self.event(
            "continuous_started",
            papers=[spec["name"] for spec in self.config["papers"]],
            parallel=int(self.config["parallel"]),
            budget_hours=float(self.config["budget_hours"]),
            iteration_rounds=int(self.config["iteration_rounds"]),
            stop_file=self.stop_file,
            disk_floor=self.floor_record(),
        )
        self.reconcile_paper_records()
        self.resume_parked_papers()
        # Before the interrupted ones: a row the library's rotation threw away is not in the middle
        # of anything, it is waiting for a measurement it never got, and the two lists overlap on
        # nothing -- this one only reads entries that stopped, that one only entries still moving.
        self.recover_rotated_rows()
        self.resume_interrupted_validations()
        self.resume_unmeasured_candidates()
        self.resume_cross_domain_follow_ups()
        # After the follow-ups that were already owed: this is the same queue, read from the library
        # instead of from the state, so a row that passed its first measurement is handed the paper
        # that tests whether it travels.
        self.resume_rows_owed_a_second_paper()
        self.start_validation_worker()
        self.save_state()
        while not self.stop_reason:
            if _stopped(self.stop_file):
                self.stop("stop_requested", f"the stop file appeared: {self.stop_file}")
                break
            if self.budget_seconds() and self.spent_seconds() >= self.budget_seconds():
                self.stop("budget_exhausted", f"the {self.config['budget_hours']}h budget is spent")
                break
            free = _free_record(self.disk_usage(self.out))
            self.disk_reading = free
            if free is not None and self.below_disk_floor(free):
                measured = (
                    f"{free['free_gb']:.1f} GB free"
                    if free.get("free_gb") is not None
                    else f"{float(free['ratio']):.1%} of the volume free"
                )
                self.stop("disk_below_floor", f"only {measured}, below the {self.floor_text()} floor")
                break
            if self.max_waves and self.waves_this_call >= self.max_waves:
                self.stop("wave_limit", f"the caller asked for at most {self.max_waves} waves")
                break
            pending = [spec for spec in self.config["papers"] if not self.state["papers"][spec["name"]].get("paused")]
            if not pending:
                self.stop("all_papers_paused", "every paper has failed enough times to be parked")
                break
            if self.target_iterations and all(self.rounds_done(spec) for spec in pending):
                self.stop("rounds_complete", "every paper has reached the " + str(self.target_iterations) + " iterations this call was asked for")
                break
            pending = [spec for spec in pending if not self.rounds_done(spec)]
            self.run_wave(pending)
        self._validation_stop.set()
        worker, self._validation_worker = self._validation_worker, None
        if worker is not None:
            # The worker measures into this run's own output directory, so a call that returns
            # while it is still working leaves a thread writing files nobody is waiting for; on
            # Windows that write lands after the caller has removed the directory, which is how a
            # finished call turned into a permission error. The stop is read between cases, so
            # this waits for the case in flight -- the same case the keeper already refuses to
            # kill -- and no longer.
            worker.join(timeout=VALIDATION_JOIN_SECONDS)
            if worker.is_alive():
                self.event(
                    "validation_worker_outlived_the_run",
                    detail=f"the measurement worker was still on its case after {VALIDATION_JOIN_SECONDS:.0f}s; the work stays owed and is requeued by the next call",
                )
        self.save_state()
        self.event("continuous_stopped", reason=self.stop_reason, detail=self.stop_detail, elapsed_seconds=round(self.spent_seconds(), 3))
        return self.report()

    def floor_text(self) -> str:
        """How the floor reads in a sentence, in the unit the caller configured."""

        if self.disk_floor_gb > 0:
            return f"{self.disk_floor_gb:.0f} GB free"
        return f"{self.disk_floor:.1%} of the volume free"

    def below_disk_floor(self, free: dict[str, float | None]) -> bool:
        """Whether this reading is under the floor the caller set.

        An absolute floor is what the caller asked to be stopped by, so it is what decides,
        and the share only stands in when the probe could not measure the volume in bytes.
        """

        if self.disk_floor_gb > 0 and free.get("free_gb") is not None:
            return float(free["free_gb"]) < self.disk_floor_gb
        return free.get("ratio") is not None and float(free["ratio"]) < self.disk_floor

    def floor_record(self) -> dict[str, Any]:
        """The floor in force, and the last reading, for the report and the start event."""

        return {
            "ratio": self.disk_floor,
            "gb": self.disk_floor_gb,
            "reading": dict(self.disk_reading) if self.disk_reading else {},
            "text": self.floor_text(),
        }

    def resume_parked_papers(self) -> list[str]:
        """Give every parked paper a fresh chance at the start of a call.

        Parking is what happens to a paper that failed the same way three times in a row, and
        most of those failures are the provider being unavailable for a while. Leaving the flag
        on across a restart would mean a run the caller relaunched on purpose quietly skips a
        paper for ever, which is the kind of silence this loop exists not to have. So the flag
        is scoped to one call: it is honoured inside the call that set it and cleared by the
        next one, and the clearing is written down.
        """

        resumed = [spec["name"] for spec in self.config["papers"] if self.state["papers"][spec["name"]].get("paused")]
        for name in resumed:
            self.state["papers"][name]["paused"] = False
            self.event("paper_resumed", paper=name, detail="a new call gave this paper another chance")
        return resumed

    def run_wave(self, pending: list[dict[str, Any]]) -> None:
        wave = int(self.state.get("waves") or 0) + 1
        self.state["waves"] = wave
        self.waves_this_call += 1
        self.event("wave_started", wave=wave, papers=[spec["name"] for spec in pending])
        parallel = max(1, min(int(self.config["parallel"]), len(pending)))
        if parallel == 1:
            outcomes = [self.run_iteration(spec, wave=wave) for spec in pending]
        else:
            # The papers of one wave are independent: each has its own directory, its own state
            # entry and its own run. They share the experience library, which is written through
            # the store's own lock, so a wave can move them at the same time.
            with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="continuous") as pool:
                outcomes = list(pool.map(lambda spec: self.run_iteration(spec, wave=wave), pending))
        self.event("wave_finished", wave=wave, outcomes=[{"paper": item.get("paper"), "status": item.get("status")} for item in outcomes])
        self.queue_validations(outcomes)
        self.save_state()

    def run_iteration(self, spec: dict[str, Any], *, wave: int) -> dict[str, Any]:
        entry = self.state["papers"][spec["name"]]
        number = int(entry.get("iteration") or 0) + 1
        # Where the paper's running counts stood when this round began. The counters themselves
        # are cumulative -- the report wants "how often has this ever happened" -- but a round's
        # own metrics want only its own moves, and subtracting the reading taken here is what
        # keeps a quiet round from inheriting every reach of the rounds before it.
        entry["upstream_base"] = {
            "requests": int(entry.get("upstream_requests") or 0),
            "passed": int(entry.get("upstream_passed") or 0),
        }
        # The layer this round is allowed to work on, derived from what the model has settled. It is
        # recorded rather than remembered, so a state written before the layers existed reads the
        # same as one written after.
        entry["work_face"] = work_face(entry.get("layers"))
        iter_dir = self.out / "papers" / spec["name"] / f"iter_{number:02d}"
        # A person's own answer about this paper is read before the round decides anything: a layer
        # a person called finished is finished, so the round starts one layer further down, and a
        # preference a person filed is handed to the seat that plans the round. It is read here
        # rather than after the sheet because the sheet has to say where the work now stands.
        human_verdicts = self.consume_human_verdicts(entry, spec, number, iter_dir)
        human_preferences = self.standing_preferences()
        review: dict[str, Any] = {"ok": False, "reason": "first_iteration", "stage_feedback": []}
        fork_stage = ""
        attribution = ""
        fork_reason = "the first iteration has nothing to inherit"
        opening_action: dict[str, Any] = {}
        stalled_hint = ""
        review_reask: dict[str, Any] | None = None
        reuse = ""
        restart: dict[str, Any] = {}
        # The figure a re-read of the paper is still measured against, when the fork is at the
        # reading and there is therefore nothing to inherit.
        reading_baseline = ""
        # A paper the config seeded continues from the figure it names, so its first iteration is
        # run the way every later one is: reviewed, forked against the work face, and measured
        # against a version that exists. Without a seed the first iteration still inherits nothing,
        # which is what every run did before this key existed.
        seed_from = str(spec.get("start_from") or "")
        seed_baseline = str(spec.get("start_baseline_from") or "")
        if number > 1 or seed_from or seed_baseline:
            published = Path(seed_from or seed_baseline) if number == 1 else Path(str(entry.get("reuse_from") or entry.get("outputs") or ""))
            if not published.is_dir():
                published = self.out / "papers" / spec["name"] / f"iter_{number - 1:02d}"
            previous = published

            # An iteration whose repairs all rolled back left the version it started from
            # standing. Rather than continue from that same version again, this round may begin
            # from a variant the archive kept -- a figure somebody drew and the loop chose not to
            # publish. An iteration that attempted no repair at all is not that state: it did not
            # try and fail, so its published line is still the one to continue from. A restart
            # never happens twice in a row: one that also stalls goes back to the published line.
            settled = int(entry.get("best_moved_at") or 0) == number - 1
            stalled = bool(int(entry.get("repair_rounds") or 0)) and not settled
            if stalled and int(entry.get("last_restart") or 0) != number - 1:
                restart = restart_from_archive(entry.get("archive") or [], exclude=[str(previous)])
            if restart:
                previous = Path(restart["source"])
                if not holds_kept_figure(previous):
                    # A variant the archive kept that left no figure of its own cannot be looked at
                    # or measured against, so the restart is dropped rather than inherited and the
                    # round goes back to the published line.
                    self.event(
                        "restart_dropped_no_figure",
                        paper=spec["name"],
                        wave=wave,
                        iteration=number,
                        source_dir=str(previous),
                        reason="the archived variant left no figure to show, so the round continues from the published line",
                    )
                    restart = {}
                    previous = published
            # An iteration whose repairs all rolled back left no figure of its own: the version it
            # started from is still the published one, and a round that inherits the empty attempt
            # has nothing to look at and nothing to be measured against. That is how a long run of
            # rounds ended with no reading in either direction -- the record says the baseline run
            # left no kept figure to show, and the round was then accepted unmeasured, so its move
            # was never filed. The published line is the newest earlier iteration that did leave a
            # figure, so the round inherits that one and says which directory stood in for which.
            if not holds_kept_figure(previous):
                walked_back = newest_kept_figure_dir(self.out / "papers" / spec["name"], number)
                if walked_back is not None:
                    self.event(
                        "baseline_walked_back",
                        paper=spec["name"],
                        wave=wave,
                        iteration=number,
                        from_dir=str(previous),
                        to_dir=str(walked_back),
                        reason="the previous iteration kept no figure of its own, so this round inherits the newest one that did",
                    )
                    previous = walked_back
            if previous.is_dir():
                # The layer this round is working on, and how long it has been stuck there. The
                # second half only ever reaches the seat as a sentence: the loop reports what it
                # watched and leaves the decision to reopen an earlier layer to the model.
                stalled_hint = deadlock_hint(
                    int(entry.get("stalled_streak") or 0), self.reopen_after, work_face(entry.get("layers"))
                )
                # The seat that decides where this round forks is handed the paper's own progress
                # sheet, so its answer is a step along the pipeline rather than a guess about a
                # figure with no history. The sheet is written before the call it informs.
                sheet = self.write_progress_sheet(iter_dir, spec, number, entry, inherited=str(previous))
                review_spec = {**spec, "progress_sheet": sheet}
                if stalled_hint:
                    review_spec["review_instruction"] = stalled_hint
                review = self.review_runner(previous, review_spec)
                attribution = attribution_fork_stage(review)
                # The run of attributions is counted against the previous iteration's, never
                # against an earlier one: the stamp carries the iteration it was made for, so an
                # iteration that ran no review -- or ran one that blamed no reading -- breaks the
                # run instead of extending it.
                stamped = entry.get("last_reading_attribution")
                previous_stage = ""
                previous_streak = 0
                if isinstance(stamped, dict) and int(stamped.get("iteration") or 0) == number - 1:
                    previous_stage = str(stamped.get("stage") or "")
                    previous_streak = int(stamped.get("streak") or 0)
                streak = previous_streak + 1 if (attribution and previous_stage == attribution) else 1
                fork_stage, fork_reason = fork_from_review(review, attribution_streak=streak if attribution else 0)
                # The one decision the seat made alone. A person standing in the direction seat
                # answers it instead, and their answer is used as given: the layer they name is the
                # case for planning that layer again, so the evidence a review would have to carry
                # is their word itself. Only the seat's own answer goes past the work-face check.
                person_fork = self.person_direction(spec, number, sheet=sheet, model_stage=fork_stage, model_reason=fork_reason)
                if str(person_fork.get("by") or "") == "person" and str(person_fork.get("stage") or ""):
                    fork_stage = str(person_fork["stage"])
                    fork_reason = str(person_fork.get("because") or "a person named the layer this iteration works on")
                else:
                    review, fork_stage, fork_reason, review_reask = self.cap_to_work_face(
                        entry,
                        previous,
                        spec,
                        review,
                        fork_stage,
                        fork_reason,
                        attributed=attribution,
                        streak=streak if attribution else 0,
                        number=number,
                        sheet=sheet,
                    )
                self.write_boundary_review(iter_dir, spec, number, review, reask=review_reask)
                if attribution:
                    entry["last_reading_attribution"] = {"iteration": number, "stage": attribution, "streak": streak}
                else:
                    entry.pop("last_reading_attribution", None)
                # Read before the run is spent: the review already named the change this
                # iteration should make, so the first round runs it against the inherited
                # figure rather than planning the stages that produced that figure again.
                opening_action = opening_move_from_feedback(review.get("stage_feedback"), fork_stage)
                reuse = str(previous)
                if fork_stage in READING_FORK_STAGES:
                    # A review that placed the fault at the reading cannot be answered from the
                    # reading this figure was built on: inheriting it would inherit the thing the
                    # review says is wrong. This iteration inherits nothing and reads the paper
                    # again, and the figure it continues from is still the one it is measured
                    # against -- a re-read has a previous version like every other round.
                    reading_baseline = str(previous)
                    reuse = ""
            else:
                fork_reason = "the previous iteration left no directory, so this one starts fresh"
        if not (iter_dir / PROGRESS_SHEET_FILE).is_file():
            # The first round inherits nothing and is reviewed by nobody, but it still gets a
            # sheet: the seat that decides the round after it has to be able to read where the
            # work started, and a sheet that only exists from the second round on cannot say that.
            self.write_progress_sheet(iter_dir, spec, number, entry)
        # Ask the replay loader what this source would actually give, before the run spends its
        # hours. A forecast that says "nothing can be inherited" belongs in the first wave's
        # report, not in a decline message found after the iteration has already run.
        forecast: dict[str, Any] = {}
        if reuse:
            reference = self.config["evolve"].get("reference")
            forecast = reuse_forecast(
                reuse,
                paper=spec["paper"],
                reference=str(reference) if reference else None,
                fork_before=fork_stage,
            )
            fork_reason = f"{fork_reason}; {_forecast_sentence(forecast)}"
            entry["last_reuse_forecast"] = forecast
        if restart:
            fork_reason = f"this round starts from the archived version of iteration {restart['iter']:02d} (round {restart['round']:02d}, candidate {restart['candidate']:02d}, score {restart['score']}); {fork_reason}"
        # One switch, spent on the first iteration that runs after it is set: the iteration reads
        # the paper again instead of inheriting the reading its figure was built from. Inheritance
        # is what makes a change to the reading invisible -- a run continuing from an earlier
        # iteration keeps that iteration's method contract, and only the stages below the fork are
        # planned again -- so without this a switch turned on at the reading would look like a
        # change that had no effect. Nothing upstream is inherited here; the comparison against the
        # last published figure is kept, because that is a measurement rather than an inheritance.
        reread_baseline = reuse or reading_baseline
        if number == 1 and seed_baseline:
            # The config named a different version for the first round to be measured against than
            # the one it inherits. Resuming a paper's good draft while scoring against the version
            # before it is the whole reason there are two keys rather than one.
            reread_baseline = seed_baseline
        if bool(self.config.get("reread_once")) and not int(entry.get("reread_done_at") or 0):
            inherited_fork = str(fork_stage or "")
            entry["reread_done_at"] = number
            fork_stage = REREAD_FORK_STAGE
            fork_reason = (
                f"the config asked for one re-read of the paper, so this iteration inherits nothing before "
                f"{REREAD_FORK_STAGE}; it would otherwise have forked at {inherited_fork or 'nothing'}"
            )
            restart = {}
            opening_action = {}
            reuse = ""
            self.event(
                "reread_forced",
                paper=spec["name"],
                wave=wave,
                iteration=number,
                fork_before=fork_stage,
                would_have_forked_at=inherited_fork,
                baseline_from=str(reread_baseline or ""),
            )
        options: dict[str, Any] = {**self.config["evolve"], "paper": spec["paper"], "out": str(iter_dir)}
        if spec.get("paper_id"):
            # The paper's identifier is known here and nowhere inside the run: a paper the
            # middle layer indexed under a path keeps a title but no identifier, and a run left
            # to read the input path names it ``paper`` -- one name for every paper, which
            # detached the run's artifacts from the paper and made the reference-gap probe read
            # the library folder ``paperimage`` as this paper's own output.
            options["paper_id"] = spec["paper_id"]
        options["max_rounds"] = int(self.config["iteration_rounds"])
        # The acceptance seat is the seat that also says where the round goes next, and how far its
        # answer is trusted is the config's call: off keeps the question it was trained on, record
        # files the answer, authority lets it move the work face. Passed through always, because the
        # seat's own reader is what decides whether anything is asked.
        options["acceptance_direction"] = str(self.human_seat.get("direction_authority") or DEFAULT_DIRECTION_AUTHORITY)
        # A person at the round's own judgement point. Only where one is standing: a paper the seat is
        # not held for passes no key at all, and a round nobody sits at never calls the gate.
        if self.human_queue.sits_at("figure_choice") and self.human_paper(spec):
            options["human_gate"] = self.human_gate_for(spec, number)
        # The between-round guards are only read between waves, so a round that spends two
        # moves would sail past a stop the operator left mid-round and past the volume it was
        # told to keep free. Handing the same guards to the round means the second move is
        # asked about them itself, and a move they stop is pinned as the next round's first
        # move instead of dropped. An empty budget is passed as no deadline, not as zero.
        budget = self.budget_seconds()
        options["move_guard"] = {
            "stop_file": self.stop_file,
            "disk_path": str(self.out),
            "disk_floor_gb": self.disk_floor_gb,
            "disk_floor_ratio": self.disk_floor,
            "budget_deadline": self.started + budget if budget else 0.0,
        }
        if self.library:
            options["experience_library"] = self.library
        if reuse:
            options["reuse_upstream_from"] = reuse
        if reread_baseline:
            # The figure this iteration's first round is measured against is the kept figure of
            # the run it is continuing from. It is handed over even when nothing is inherited --
            # a re-read still has a previous version to be measured against -- because without it
            # the first round is accepted unmeasured, which is what left every round-0 experience
            # without an "after" reading.
            options["baseline_from"] = reread_baseline
        if opening_action:
            # The change the review of the inherited figure asked for, run as the first
            # round's own move instead of re-planning what that figure already is.
            options["first_round_action"] = dict(opening_action)
        if fork_stage:
            options["fork_before"] = fork_stage
        if human_preferences:
            # What a person asked for, in the same shape a rule and a patch arrive in, so the round
            # is told how this figure should be drawn by the person who looked at it.
            options["human_preferences"] = [dict(item) for item in human_preferences]
        # The folder those instructions are filed in travels with the run as well, because the round
        # reads it again at the top of every round: a sentence filed while this iteration is running
        # has to reach the round after it, not the iteration after it. The list above is still handed
        # over as the fallback for a folder nobody can read, and the event records the list the
        # iteration started from, so an audit can tell which sentences a round actually ran under.
        options["human_preferences_dir"] = str(self.human_store().directory)
        # The layer this round is allowed to work on, and the question that decides it, travel into
        # the run for the same reason a person's instruction does: the seat that can say whether a
        # layer is finished with is the reflection, and the reflection is inside the run. Without
        # this the question is asked of the seat that chooses the fork and answered by nobody, which
        # is how the work face stayed on the first layer for a hundred iterations. A round with no
        # work face passes no key, so every other prompt is byte-identical.
        if work_face(entry.get("layers")):
            options["layer_plan"] = layer_plan(entry)
            self.event(
                "layer_plan_handed_down",
                paper=spec["name"],
                wave=wave,
                iteration=number,
                working_on=str(options["layer_plan"].get("working_on") or ""),
                settled=list(options["layer_plan"].get("settled") or []),
            )
        options.setdefault("experience_enabled", True)
        options.setdefault("evolution_enabled", True)
        self.event(
            "iteration_started",
            paper=spec["name"],
            wave=wave,
            iteration=number,
            out=str(iter_dir),
            reuse_from=reuse,
            fork_before=fork_stage,
            fork_reason=fork_reason,
            reuse_forecast=forecast,
            review_ok=bool(review.get("ok")),
            review_exit=str(review.get("exit") or ""),
            review_reason=str(review.get("reason") or ""),
            review_reask=bool(review.get("reask")),
            work_face=str(entry.get("work_face") or ""),
            stalled_hint=bool(stalled_hint),
            progress_sheet=bool((iter_dir / PROGRESS_SHEET_FILE).is_file()),
            review_actions=requested_action_types(review.get("stage_feedback")),
            reading_attribution=attribution,
            reading_attribution_streak=(entry.get("last_reading_attribution") or {}).get("streak", 0) if attribution else 0,
            first_round_action=dict(opening_action) or None,
        )
        if restart:
            # A round that begins somewhere other than the published line says so, with the exact
            # variant it chose and what that variant scored, so the choice can be audited later.
            entry["last_restart"] = number
            self.event(
                "restart_from_archive",
                paper=spec["name"],
                wave=wave,
                iteration=number,
                action=RESTART_FROM_ARCHIVE,
                reuse_from=str(restart["source"]),
                source_iteration=int(restart["iter"]),
                source_round=int(restart["round"]),
                source_candidate=int(restart["candidate"]),
                source_dir=str(restart.get("dir") or ""),
                source_score=restart.get("score"),
                source_absolute_score=restart.get("absolute_score"),
                source_choice_score=restart.get("choice_score"),
                source_decision_reason=str(restart.get("decision_reason") or ""),
                restart_reason=str(restart.get("restart_reason") or ""),
            )
        record: dict[str, Any] = {
            "paper": spec["name"],
            "iteration": number,
            "wave": wave,
            "out": str(iter_dir),
            "reuse_from": reuse,
            "fork_before": fork_stage,
            "fork_reason": fork_reason,
            "reuse_forecast": forecast,
            "restart": dict(restart),
            "review": {
                "ok": bool(review.get("ok")),
                "exit": str(review.get("exit") or ""),
                "reason": str(review.get("reason") or ""),
                "retries": int(review.get("retries") or 0),
                "reask": bool(review.get("reask")),
                "capped": bool(review.get("capped")),
                "actions": requested_action_types(review.get("stage_feedback")),
                "reading_attribution": attribution,
                "upstream_case": dict(review.get("upstream_case") or {}),
                "upstream_case_rejected": str(review.get("upstream_case_rejected") or ""),
            },
            "first_round_action": dict(opening_action),
            "work_face": str(entry.get("work_face") or ""),
            "stalled_hint": bool(stalled_hint),
            "human_verdicts": list(human_verdicts.get("verdicts") or []),
            "human_preferences": [
                {"entry_id": str(item.get("entry_id") or ""), "status": str(item.get("status") or ""), "apply_at": str(item.get("apply_at") or "")}
                for item in human_preferences
            ],
            "progress_sheet": str(iter_dir / PROGRESS_SHEET_FILE) if (iter_dir / PROGRESS_SHEET_FILE).is_file() else "",
        }
        try:
            result = self.evolve_runner(**options)
            if not isinstance(result, dict) or not result:
                raise ContinuousError("run_failed", f"the run returned no report: {iter_dir}")
        except Exception as exc:  # noqa: BLE001 - the loop records why instead of guessing
            if int(entry.get("reread_done_at") or 0) == number:
                # This iteration spent the one-off re-read and then failed before anything was
                # read, so the switch stays unspent rather than being lost to a transient error.
                entry["reread_done_at"] = 0
            entry["failures"] = int(entry.get("failures") or 0) + 1
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            entry.setdefault("history", []).append(record)
            self.event("iteration_failed", paper=spec["name"], iteration=number, error=record["error"], failures=entry["failures"])
            if int(entry["failures"]) >= MAX_CONSECUTIVE_FAILURES:
                entry["paused"] = True
                self.event("paper_paused", paper=spec["name"], failures=entry["failures"], detail="this paper failed the same way too many times in a row")
            # A failed round is written down too: it moved the counters the next attempt decides on.
            self.persist_paper_record(spec, entry)
            return record
        entry["failures"] = 0
        entry["iteration"] = number
        entry["reuse_from"] = str(iter_dir)
        outcome = self._record_outcome(spec, entry, iter_dir, result, record)
        # This paper's account is on disk before the round is handed on: an interrupted wave must
        # not take the rounds that finished with it, and the record is what a restart reads
        # instead of paying for them again.
        self.persist_paper_record(spec, entry)
        # The candidate is handed over the moment its own paper finishes, not when the whole
        # wave does: the slowest paper of a wave is hours long and must not hold another
        # paper's measurement back.
        self.queue_validations([outcome])
        return outcome

    def write_boundary_review(self, iter_dir: Path, spec: dict[str, Any], number: int, review: dict[str, Any], *, reask: dict[str, Any] | None = None) -> None:
        """Write down what the boundary review was asked, answered and cost.

        The review decides where the round forks and the only trace of it afterwards was the exit
        reason on an event, which cannot say what the seat was shown or what it said back. The
        answer is kept whole next to the round it decided, so "the seat was unreadable", "the seat
        answered without a location" and "the seat found nothing to repair" can be told apart from
        the run's own output instead of from the loop's sentence about it. The case a round went
        back up on travels with it, and so does the reason a case that did not hold was refused.
        """

        record = review if isinstance(review, dict) else {}

        def block(source: dict[str, Any]) -> dict[str, Any]:
            """One answer, with everything needed to read it later."""

            return {
                "answer": dict(source.get("judgement") or {}),
                "model": str(source.get("model") or ""),
                "exit_reason": str(source.get("exit") or ""),
                "reason": str(source.get("reason") or ""),
                "retried": bool(source.get("retried")),
                "retries": int(source.get("retries") or 0),
                "localisation_retry": bool(source.get("localisation_retry")),
            }

        answer_block = block(record)
        exit_reason = answer_block["exit_reason"]
        payload = {
            "contract": BOUNDARY_REVIEW_CONTRACT,
            "version": 1,
            "paper": str(spec.get("name") or ""),
            "iteration": int(number),
            "question": BOUNDARY_REVIEW_QUESTION,
            "capped": bool(record.get("capped")),
            **answer_block,
        }
        if isinstance(record.get("upstream_case"), dict) and record["upstream_case"]:
            # The argument the round went back up on, kept whole: a layer that was reopened has to be
            # readable afterwards as reopened *for a reason*, not as a layer that drifted open.
            payload["upstream_case"] = dict(record["upstream_case"])
        if str(record.get("upstream_case_rejected") or ""):
            payload["upstream_case_rejected"] = str(record["upstream_case_rejected"])
        if isinstance(reask, dict) and reask:
            # The round was asked again because it wanted to work on a layer the model had already
            # called finished. Both answers are kept: the one it was held on, and the one that came
            # back when the rule was spelled out.
            payload["reask"] = block(reask)
        write_json(iter_dir / BOUNDARY_REVIEW_FILE, payload)
        if exit_reason in {REVIEW_ERROR, REVIEW_UNDEFINED}:
            self.event(
                exit_reason,
                paper=str(spec.get("name") or ""),
                iteration=int(number),
                reason=str(record.get("reason") or ""),
                retries=int(record.get("retries") or 0),
                localisation_retry=bool(record.get("localisation_retry")),
            )

    def human_store(self) -> HumanReviewStore:
        """The folder a person files verdicts about this run in."""

        return HumanReviewStore(self.out / HUMAN_REVIEW_DIR)

    def human_paper(self, spec: dict[str, Any]) -> bool:
        """Whether this paper is the one a person is standing at.

        An empty list of papers means the seat is held for every paper in the run; a list means the
        named papers only, so one config can hold a seat on two arms and leave the third to the
        model. The two spellings differ on purpose: "a person is here for everything" and "a person
        is here for these two" are different claims and the second one is the experiment.
        """

        return self.human_holds(str(spec.get("name") or ""))

    def human_holds(self, paper: str) -> bool:
        """The same question asked by name, for the calls that hold a paper and not a spec.

        The validation worker is handed the paper the measurement is about rather than the run's own
        spec for it, and it has to ask the same question: is a person standing here.
        """

        held = [str(item) for item in (self.human_seat.get("papers") or []) if str(item)]
        return not held or str(paper or "") in held

    def person_direction(self, spec: dict[str, Any], number: int, *, sheet: Any = None, model_stage: str = "", model_reason: str = "") -> dict[str, Any]:
        """Which layer the iteration that is about to run works on, as the person said it.

        The fork is the one decision the round's own review used to make alone, and it is the
        decision a person is in no position to make from the figure card: the card asks about the
        layer the round just worked on, and the fork is about the layer the next one works on. So
        the answer is taken in two places, in that order.

        The answer the person already gave -- "this layer is not finished with" -- is their word
        about where the work still is, so it needs no second card and no confirmation: it *is* the
        case for planning that layer again. When the seat is on and no such answer stands, the
        question is written as a card and the run waits on it exactly as it waits on the figure
        card, and a card nobody answers leaves the model's own stage standing.

        Whoever answered is carried back with the stage, and the other answer is kept: the card
        writes the model's stage into its shadow file before the person answers it, so the two
        readings of the same figure sit beside each other on disk.
        """

        held = (
            person_open_layer(self.out / "papers" / str(spec.get("name") or "") / f"iter_{int(number) - 1:02d}")
            if int(number) > 1
            else {}
        )
        if held:
            # The person has already answered this: "this layer is not finished with" is their word
            # about where the work still is, so it is the case for planning that layer again and no
            # second card asks the same thing twice.
            self.event(
                "human_direction_carried",
                paper=str(spec.get("name") or ""),
                iteration=int(number),
                stage=str(held.get("stage") or ""),
                round=int(held.get("round") or 0),
                model_stage=str(model_stage or ""),
            )
            return {
                "stage": str(held.get("stage") or ""),
                "by": "person",
                "because": str(held.get("because") or ""),
                "round": int(held.get("round") or 0),
                "note": str(held.get("note") or ""),
            }
        fallback = {"stage": str(model_stage or ""), "by": "model", "because": str(model_reason or "")}
        if not (self.human_queue.sits_at("direction") and self.human_paper(spec)):
            return fallback
        answer = self.human_queue.ask(
            kind="direction",
            paper=str(spec.get("name") or ""),
            iteration=int(number),
            round_index=0,
            question=DIRECTION_QUESTION,
            model_answer={
                "choice": str(model_stage or ""),
                "reason": str(model_reason or ""),
                "source": "the seat's own review",
            },
            stage=str(model_stage or ""),
            progress_sheet=dict(sheet) if isinstance(sheet, dict) else {},
        )
        chosen = str(answer.choice or "").strip()
        if answer.answered and chosen in STAGE_ORDER:
            self.event(
                "human_direction_applied",
                paper=str(spec.get("name") or ""),
                iteration=int(number),
                stage=chosen,
                by=str(answer.by),
                card=str(answer.card or ""),
                model_stage=str(model_stage or ""),
            )
            return {
                "stage": chosen,
                "by": "person",
                "because": "a person named the layer this iteration works on",
                "card": str(answer.card or ""),
                "note": str(answer.note or ""),
            }
        if answer.answered:
            self.event(
                "human_direction_unreadable",
                paper=str(spec.get("name") or ""),
                iteration=int(number),
                answer=chosen,
                card=str(answer.card or ""),
                reason="the answer named no layer, so the run continues on the answer it had",
            )
        elif str(answer.by) == "timeout":
            # The card existed and nobody answered it inside the hold. Written down rather than
            # dropped: "the run carried on without a person" and "there was nobody to ask" are
            # different facts, and only the second one is a seat that was never stood at.
            self.event(
                "human_direction_unanswered",
                paper=str(spec.get("name") or ""),
                iteration=int(number),
                card=str(answer.card or ""),
                waited_seconds=float(answer.waited_seconds),
                model_stage=str(model_stage or ""),
            )
        return fallback

    def person_validation(self, record: dict[str, Any]) -> dict[str, Any]:
        """What a person says about one finished measurement, before the ladder reads it.

        Entry to the library is decided here, and it is the last of the three seats: the acceptance
        seat answers about a round of drawing, the direction seat about which layer runs next, and
        this one about whether a measurement may carry a row up the ladder. It is asked in the one
        place that question exists -- after the paired run has ended, and only where the verdict it
        reached would move the row -- so a measurement that moved nothing is put to nobody.

        Two answers. "Hold" keeps the row exactly where it was: the measurement is recorded and
        nothing about the row changes, which is what stops a reading a person does not believe from
        spending a row -- and a hold stops the whole reading, including the charge a losing patch
        would have taken. Anything else, including a card nobody answered, lets the ladder read the
        measurement the way it reads every other one. A person cannot write a row into the library:
        entry is a claim about a measurement, and only a measurement can make it. What they can do
        is refuse one.

        Where the seat is off, or held for other papers, no card is written and this returns nothing,
        so a run with nobody at the seat is byte-identical to one from before it existed.
        """

        paper = str(record.get("paper") or "")
        experience_id = str(record.get("experience_id") or "")
        verdict = str(record.get("status") or "")
        if verdict not in ROW_MOVING_VERDICTS:
            # Nothing to admit or hold: the ladder would leave the row where it is, and asking about
            # a measurement that moved nothing would only spend a person's attention.
            return {}
        if not (self.human_queue.sits_at("validation") and self.human_holds(paper)):
            return {}
        answer = self.human_queue.ask(
            kind="validation",
            paper=paper,
            iteration=int(record.get("iteration") or 0),
            round_index=0,
            question=f"{VALIDATION_QUESTION}\n\nThe measurement ended: {verdict}.",
            model_answer={
                "choice": "leave it standing",
                "reason": f"the measurement ended {verdict}, which the ladder reads as moving the row",
                "source": "read off the verdict, not a decision of its own",
            },
            readings=dict(record.get("summary") or {}),
            extra={
                "experience_id": experience_id,
                # Not ``kind``: the card's own ``kind`` is which seat it was asked at, and writing the
                # row's kind over it would file the answer under the wrong question.
                "row_kind": str(record.get("kind") or ""),
                "verdict": verdict,
                "stopped_because": str(record.get("stopped_because") or ""),
                "starting_status": str(record.get("starting_status") or ""),
                "out": str(record.get("out") or ""),
            },
        )
        if not answer.answered:
            self.event(
                "human_validation_unanswered",
                experience_id=experience_id,
                paper=paper,
                verdict=verdict,
                by=str(answer.by),
                card=str(answer.card or ""),
                waited_seconds=float(answer.waited_seconds),
            )
            return {}
        said = str(answer.choice or "").strip()
        decision = "hold" if said.casefold() in VALIDATION_HOLD_WORDS else "admit"
        self.event(
            "human_validation_answered",
            experience_id=experience_id,
            paper=paper,
            verdict=verdict,
            answer=said,
            decision=decision,
            by=str(answer.by),
            card=str(answer.card or ""),
        )
        return {
            "decision": decision,
            "by": str(answer.by),
            "note": str(answer.note or ""),
            "card": str(answer.card or ""),
            "answer": said,
        }

    def human_gate_for(self, spec: dict[str, Any], number: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
        """The callable the round asks at each judgement point: write a card, wait, return the answer."""

        sheet_path = self.out / "papers" / str(spec.get("name") or "") / f"iter_{int(number):02d}" / PROGRESS_SHEET_FILE

        def gate(payload: dict[str, Any]) -> dict[str, Any]:
            figures = {str(key): str(value or "") for key, value in (payload.get("figures") or {}).items()}
            sheet = _read_json_or_none(sheet_path)
            answer = self.human_queue.ask(
                kind=str(payload.get("kind") or "figure_choice"),
                paper=str(payload.get("paper") or spec.get("name") or ""),
                iteration=int(number),
                round_index=int(payload.get("round") or 0),
                question=str(payload.get("question") or ""),
                model_answer=dict(payload.get("model_answer") or {}),
                stage=str(payload.get("stage") or ""),
                figures=figures,
                readings=dict(payload.get("readings") or {}),
                progress_sheet=dict(sheet) if isinstance(sheet, dict) else {},
            )
            record = answer.as_record()
            if record["answered"] or record["by"] == "timeout":
                self.event(
                    "human_question_settled",
                    paper=str(spec.get("name") or ""),
                    iteration=int(number),
                    kind=str(payload.get("kind") or ""),
                    by=str(record["by"]),
                    choice=str(record["choice"]),
                    card=str(record["card"]),
                    waited_seconds=float(record["waited_seconds"]),
                    model_choice=str((payload.get("model_answer") or {}).get("choice") or ""),
                )
            return record

        return gate

    def standing_preferences(self) -> list[dict[str, Any]]:
        """The preferences a person has filed about this run, trials and standing ones alike.

        Both are handed to the round: a trial is what a person asked for and no round has been
        measured under yet, and a standing one is the same sentence after a round kept the change
        it asked for. The entry says which of the two it is; the loop never quietly drops one.
        """

        store = self.human_store()
        return [dict(row) for row in store.entries()] if store.entries_path.exists() else []

    def consume_human_verdicts(self, entry: dict[str, Any], spec: dict[str, Any], number: int, iter_dir: Path) -> dict[str, Any]:
        """Read what a person filed since the last round, act on it, and mark it read.

        Three things can arrive here and each is a different act. A layer a person called finished
        is finished -- the same call the model's own answer goes through, with the person's answer
        given priority and its source written down. A preference becomes an entry in the channel's
        own file, which the round that follows is handed. A figure choice is a comparison, and
        comparisons are training rows: they are read here so they are not read twice, and nothing
        in the round changes because of one.

        Read once is the whole point. The state keeps the ids, so a restart does not re-apply an
        answer that was already applied, and two verdicts filed in one wave are read in the order
        they were written.
        """

        paper = str(spec.get("name") or "")
        store = self.human_store()
        pending = (
            unconsumed_verdicts(store.rows(), entry.get("human_verdicts_seen"), paper=paper)
            if store.log_path.exists()
            else []
        )
        consumed: list[dict[str, Any]] = []
        for row in pending:
            kind = review_kind(row)
            review_id = str(row.get("review_id") or "")
            applied = ""
            entry_id = ""
            if kind == "layer_settled":
                stage = str(row.get("stage") or "")
                settled = self.settle_layer(
                    entry,
                    stage,
                    paper=paper,
                    number=number,
                    because=str(row.get("human_note") or ""),
                    by="human",
                )
                applied = "layer_settled" if settled else "already_settled"
            elif kind == "preference":
                installed = store.install_entry(row, status="candidate")
                entry_id = str(installed.get("entry_id") or "")
                applied = "preference" if installed else "preference_refused"
            else:
                applied = "figure_choice"
            consumed.append(
                {
                    "review_id": review_id,
                    "kind": kind,
                    "applied": applied,
                    "entry_id": entry_id,
                    "stage": str(row.get("stage") or ""),
                    "apply_at": str(row.get("apply_at") or ""),
                    "directive": str(row.get("directive") or ""),
                    "note": str(row.get("human_note") or ""),
                    "reviewer": str(row.get("reviewer") or ""),
                }
            )
            self.event(
                "human_verdict_consumed",
                paper=paper,
                iteration=int(number),
                review_id=review_id,
                kind=kind,
                applied=applied,
                stage=str(row.get("stage") or ""),
                reviewer=str(row.get("reviewer") or ""),
            )
        if consumed:
            seen = [str(item) for item in entry.get("human_verdicts_seen") or []]
            seen.extend(item["review_id"] for item in consumed if item["review_id"])
            entry["human_verdicts_seen"] = seen
        payload = {
            "contract": HUMAN_VERDICTS_CONTRACT,
            "version": 1,
            "paper": paper,
            "iteration": int(number),
            "verdicts": consumed,
        }
        write_json(iter_dir / HUMAN_VERDICTS_FILE, payload)
        return payload

    def promote_measured_preferences(self, entry: dict[str, Any], iter_dir: Path, paper: str, number: int, *, kept: bool) -> list[str]:
        """Make the trials this round ran under stand, once the round kept the change they asked for.

        A preference is filed as a trial: a person asked for it, and no round has been measured
        under it yet. The round that follows is that measurement, and its own comparison is the
        ruler -- a round that kept nothing is not evidence that the sentence was right. Not the
        only ruler there could be (a paired run on another paper is the stronger one the patch
        channel uses), and the entry says which one promoted it.
        """

        if not kept:
            return []
        payload = _read_json_or_none(iter_dir / HUMAN_VERDICTS_FILE) or {}
        store = self.human_store()
        promoted: list[str] = []
        for verdict in payload.get("verdicts") if isinstance(payload.get("verdicts"), list) else []:
            entry_id = str((verdict or {}).get("entry_id") or "")
            if not entry_id or str((verdict or {}).get("applied") or "") != "preference":
                continue
            result = store.promote(entry_id, by="loop", because=f"iteration {number} kept the change this preference asked for")
            if result.get("changed"):
                promoted.append(entry_id)
                self.event("human_preference_promoted", paper=paper, iteration=int(number), entry_id=entry_id, because=f"iteration {number} kept the change this preference asked for")
        return promoted

    def reopen_layer(self, entry: dict[str, Any], stage: str, *, paper: str, number: int, why: str, by: str = "model", case: dict[str, Any] | None = None) -> bool:
        """Undo a layer's settle so a round may work there, and say who asked for it.

        Only a layer that was settled can be reopened, so this is a no-op for the work face itself
        and for every layer below it. Nothing about the figure changes here: the layer is simply
        open again, which moves the work face back to it.

        A layer a person sealed is theirs to open. The model may not walk over a person's answer on
        its own word, however many rounds it has been stuck: it has to bring the case a person's
        answer is worth reopening the layer for -- the requirement that went unwritten there and a
        reading behind it -- and a case that does not hold leaves the layer sealed, with the
        refusal recorded rather than dropped. The person's own answer opens it without any case,
        because their answer is the thing the case would have been evidence for.
        """

        layers = entry.setdefault("layers", {})
        record = layers.get(stage) if isinstance(layers.get(stage), dict) else {}
        if not record.get("settled_at"):
            return False
        if str(record.get("settled_by") or "") == "human" and str(by) != "human":
            claim = case.get("case") if isinstance(case, dict) and isinstance(case.get("case"), dict) else {}
            held = bool(
                isinstance(case, dict)
                and case.get("ok")
                and str(claim.get("fault_requirement") or "").strip()
                and claim.get("fault_evidence")
            )
            if not held:
                self.event(
                    "layer_reopen_refused",
                    paper=str(paper),
                    iteration=int(number),
                    stage=str(stage),
                    by=str(by),
                    settled_by="human",
                    settled_at=int(record.get("settled_at") or 0),
                    why=str(why),
                    reason=(
                        "a person sealed this layer; reopening it needs that person, or a case naming "
                        "the requirement that went unwritten there and a reading the seat was shown"
                    ),
                )
                return False
        record = {**record, "settled_at": 0, "settled_by": "", "reopen_count": int(record.get("reopen_count") or 0) + 1}
        layers[stage] = record
        entry["work_face"] = work_face(layers)
        self.event(
            "layer_reopened",
            paper=paper,
            iteration=int(number),
            stage=str(stage),
            by=str(by),
            why=str(why),
            reopen_count=int(record["reopen_count"]),
        )
        return True

    def settle_layer(self, entry: dict[str, Any], stage: str, *, paper: str, number: int, because: str = "", by: str = "model") -> bool:
        """Record that a layer is finished with, on the model's word or a person's.

        The loop does not decide this and nothing in it does: a reading describes the figure, and a
        layer is finished with only when the seat that looked at it says so. A person's answer is
        written by the same call and outranks the model's, so both read back the same way. A layer
        already settled is left as it is -- the first answer stands until somebody opens the layer
        again.

        A layer is sealed when the round reaches it. The model may seal the layer it is working on
        and no other: stamping a layer further down says the layers between were finished with when
        nobody said so, and that is how a run came to report progress it had not made. A person may
        seal any layer, because the ordering the loop keeps is a rule about the loop.
        """

        layers = entry.setdefault("layers", {})
        record = layers.get(stage) if isinstance(layers.get(stage), dict) else {}
        if record.get("settled_at"):
            return False
        work = work_face(layers)
        if str(by) != "human" and work and str(stage) != work:
            self.event(
                "layer_settle_refused",
                paper=str(paper),
                iteration=int(number),
                stage=str(stage),
                work_face=str(work),
                by=str(by),
                reason="a layer is sealed when the round reaches it, and this is not the layer the round is on",
            )
            return False
        layers[stage] = {
            **record,
            "settled_at": int(number),
            "settled_by": str(by),
            "because": str(because or ""),
            "reopen_count": int(record.get("reopen_count") or 0),
        }
        entry["work_face"] = work_face(layers)
        self.event(
            "layer_settled",
            paper=str(paper),
            iteration=int(number),
            stage=str(stage),
            by=str(by),
            because=str(because or ""),
            work_face=str(entry.get("work_face") or ""),
        )
        return True

    def note_layer_moves(self, entry: dict[str, Any], iter_dir: Path, *, number: int) -> None:
        """Mark the layers this iteration touched, so the ledger can tell a settled layer from an idle one."""

        layers = entry.setdefault("layers", {})
        for stage, taken in stage_moves(iter_dir).items():
            if not taken.get("moved"):
                continue
            record = layers.get(stage) if isinstance(layers.get(stage), dict) else {}
            layers[stage] = {
                **record,
                # The first iteration that moved this layer is what the cost of settling it is
                # measured from, so it is stamped once and never overwritten.
                "first_move_at": int(record.get("first_move_at") or number),
                "last_move_at": int(number),
                "last_reading": dict(taken.get("reading") or {}),
            }

    def write_layer_ledger(self, iter_dir: Path, spec: dict[str, Any], number: int, entry: dict[str, Any]) -> dict[str, Any]:
        """Write this iteration's layer ledger, and return it."""

        settle_events, reopen_events = layer_answer_history(iter_dir.parent, upto=int(number))
        worked_rounds = moved_iterations(iter_dir.parent, upto=int(number))
        entry["layer_settle_events"] = settle_events
        entry["worked_rounds"] = worked_rounds
        payload = layer_ledger(
            iter_dir,
            layers=entry.get("layers"),
            iteration=number,
            settle_events=settle_events,
            reopen_events=reopen_events,
            worked_rounds=worked_rounds,
        )
        payload["paper"] = str(spec.get("name") or "")
        write_json(iter_dir / LAYER_LEDGER_FILE, payload)
        return payload

    def write_progress_sheet(
        self,
        iter_dir: Path,
        spec: dict[str, Any],
        number: int,
        entry: dict[str, Any],
        *,
        inherited: str = "",
    ) -> dict[str, Any]:
        """Write this round's progress sheet, beside the round and as the paper's running copy.

        The per-layer readings are read back off the newest ledgers, because a layer the state has
        stopped moving carries no reading and a sheet that asks a judgement about a layer has to
        carry the evidence for it. The directory this round inherits is named too, so the sheet can
        say whether it is working on the best version the paper has reached.
        """

        sheet = progress_sheet(
            entry,
            spec,
            iteration=number,
            readings=older_layer_readings(iter_dir.parent, before=int(number)),
            inherited=inherited,
        )
        write_json(iter_dir / PROGRESS_SHEET_FILE, sheet)
        write_json(iter_dir.parent / PROGRESS_SHEET_LATEST, sheet)
        return sheet

    def write_round_metrics(self, iter_dir: Path, spec: dict[str, Any], number: int, entry: dict[str, Any], *, wave: int = 0) -> dict[str, Any]:
        """This iteration's own progress numbers, added to the metrics the run already wrote.

        The run writes what it can see from inside one iteration -- the five numbers about its own
        rounds. The walk down the layers is only visible from here, because it is the record of
        every iteration this paper has run, so it is added to the same file under its own key
        rather than written into a second file a reader would have to know about. Nothing the run
        wrote is replaced: the file is read first and only the progress key is set.

        The same block then goes to the loop's trend ledger, so the cost of settling a layer and the
        verdict of each cross-round comparison can be plotted across iterations without opening
        each iteration's file.
        """

        reopened = [
            item
            for item in layer_reopen_events(iter_dir.parent, upto=int(number))
            if int(item.get("iteration") or 0) == int(number)
        ]
        payload = round_metrics(
            paper=str(spec.get("name") or ""),
            iteration=int(number),
            wave=int(wave),
            layers=entry.get("layers"),
            stage_order=STAGE_ORDER,
            settle_events=entry.get("layer_settle_events") or [],
            reopen_events=reopened,
            worked_rounds=entry.get("worked_rounds") or [],
            reopened_layers=[str(item.get("stage") or "") for item in reopened],
            work_face=str(entry.get("work_face") or ""),
            upstream={
                "requested": int(entry.get("upstream_requests") or 0) - int((entry.get("upstream_base") or {}).get("requests") or 0),
                "passed": int(entry.get("upstream_passed") or 0) - int((entry.get("upstream_base") or {}).get("passed") or 0),
            },
            reflections=iteration_reflections(iter_dir),
            catalogue=iteration_catalogue_gap(iter_dir),
            refusals=iteration_refusals(iter_dir),
            cross_round=iteration_cross_rounds(iter_dir),
        )
        entry["last_upstream_requests"] = int((payload.get("upstream") or {}).get("requested") or 0)
        entry["last_upstream_passed"] = int((payload.get("upstream") or {}).get("passed") or 0)
        existing = _read_json_or_none(iter_dir / ROUND_METRICS_FILE)
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged["progress"] = payload
        write_json(iter_dir / ROUND_METRICS_FILE, merged)
        # The run's answer to "how many of this round's sentences were later checked" is read off
        # the file it just wrote, so the report can carry it without opening every iteration again.
        if isinstance(merged.get("claims"), dict):
            entry["last_claims"] = dict(merged["claims"])
        try:
            append_jsonl(
                self.out / ROUND_METRICS_LEDGER,
                {
                    "paper": str(spec.get("name") or ""),
                    "iteration": int(number),
                    "work_face": str(payload.get("work_face") or ""),
                    "progress": {
                        "layers_settled": int((payload.get("layers") or {}).get("settled") or 0),
                        "layers_total": int((payload.get("layers") or {}).get("total") or 0),
                        "all_settled": bool((payload.get("layers") or {}).get("all_settled")),
                        "rounds_to_settle_all": (payload.get("layers") or {}).get("rounds_to_settle_all"),
                        "reopen_events": int((payload.get("layers") or {}).get("reopen_events") or 0),
                        "upstream": dict(payload.get("upstream") or {}),
                        "settled_this_round": list(payload.get("settled_this_round") or []),
                        "reopened_this_round": list(payload.get("reopened_this_round") or []),
                        "cross_round": dict(payload.get("cross_round") or {}),
                        "reflection": dict(payload.get("reflection") or {}),
                    },
                },
            )
        except Exception as exc:  # noqa: BLE001 - the trend line must never stop an iteration
            self.event("round_metrics_unwritten", paper=str(spec.get("name") or ""), iteration=int(number), error=f"{type(exc).__name__}: {exc}")
        return payload

    def compare_rounds(self, iter_dir: Path, number: int, entry: dict[str, Any], spec: dict[str, Any], *, settled: bool = False) -> dict[str, Any]:
        """Put this round's figure beside the best version the paper had reached before it.

        The comparison is evidence, not a gate: nothing here accepts a round, rolls one back or
        moves a layer. It answers the one question the round-by-round record cannot -- whether the
        paper is better off than it was a few rounds ago -- and that answer is only worth having
        when there is something to compare against and a seat to ask. Every reason there is not
        one is written down beside the answer, so "no comparison" never reads as "no change".
        """

        if not cross_round_due(entry, number, self.cross_round_every) and not settled:
            return {}
        current = str(entry.get("best") or "")
        baseline = cross_round_baseline(entry.get("archive"), before=number)
        if not current or not baseline.get("dir"):
            record = cross_round_record(
                paper=str(spec.get("name") or ""),
                iteration=number,
                current={"dir": current, "absolute_score": absolute_score_for(current) if current else None},
                baseline=baseline,
                reason="the paper has no earlier version with an absolute reading to compare against",
            )
        elif self.cross_round_runner is None:
            record = cross_round_record(
                paper=str(spec.get("name") or ""),
                iteration=number,
                current={"dir": current, "absolute_score": absolute_score_for(current)},
                baseline=baseline,
                reason="no seat is configured to compare two rounds",
            )
        else:
            answer = self.cross_round_runner(Path(current), Path(str(baseline["dir"])), spec)
            record = cross_round_record(
                paper=str(spec.get("name") or ""),
                iteration=number,
                current={"dir": current, "absolute_score": absolute_score_for(current)},
                baseline=baseline,
                answer=answer,
            )
        write_json(iter_dir / CROSS_ROUND_FILE, record)
        entry.setdefault("cross_round", []).append(
            {
                "iteration": int(number),
                "verdict": str(record["verdict"]),
                "preference": str(record["preference"]),
                "reason": str(record["reason"]),
                "baseline_iter": int((record["history_best"] or {}).get("iter") or 0),
                "baseline_absolute_score": (record["history_best"] or {}).get("absolute_score"),
                "current_absolute_score": (record["current"] or {}).get("absolute_score"),
            }
        )
        self.event(
            "cross_round_compared",
            paper=str(spec.get("name") or ""),
            iteration=int(number),
            verdict=str(record["verdict"]),
            preference=str(record["preference"]),
            reason=str(record["reason"]),
        )
        return record

    def cap_to_work_face(self, entry, previous, spec, review, fork_stage, fork_reason, *, attributed="", streak=0, number=0, sheet=None):
        """Hold the fork to the layer this round is working on, unless the review argues past it.

        A round may work on the open layer or on anything below it for free, and there is nothing to
        argue: a settled structure does not stop the connectors from being repaired. Going back over
        a layer the model has already called finished is a claim about that layer's own output, so it
        has to be argued: a case naming that layer and citing a reading the seat was shown is one
        answer, and a seat that gave neither is answered once more with the rule spelled out -- it
        gets one chance to say what it saw -- and only then is the round held where it was. Nothing
        here decides that a layer is good: that is the model's call, and the loop only keeps the
        round's feet on the layer it is actually working on.
        """

        work = work_face(entry.get("layers"))
        if not is_upstream(fork_stage, work):
            return review, fork_stage, fork_reason, None
        # Counted here rather than read back out of the event log: the report asks how often the
        # seat reached for a layer above the one it is on, and how often the loop let it, and both
        # are facts about the moment the decision was made.
        entry["upstream_requests"] = int(entry.get("upstream_requests") or 0) + 1
        case = upstream_case(review)
        if upstream_allowed(fork=fork_stage, work=work, attributed=attributed, streak=streak, review=review, case=case):
            entry["upstream_passed"] = int(entry.get("upstream_passed") or 0) + 1
            self.reopen_layer(entry, fork_stage, paper=str(spec.get("name") or ""), number=number, why=fork_reason, case=case)
            return {**review, "upstream_case": dict(case.get("case") or {})}, fork_stage, fork_reason, None
        second_spec = {**spec, "review_instruction": UPSTREAM_EVIDENCE_RETRY}
        if isinstance(sheet, dict) and sheet:
            second_spec["progress_sheet"] = sheet
        second = self.review_runner(previous, second_spec)
        second_attributed = attribution_fork_stage(second)
        second_fork, second_reason = fork_from_review(second, attribution_streak=streak if second_attributed else 0)
        second_case = upstream_case(second)
        if not is_upstream(second_fork, work) or upstream_allowed(
            fork=second_fork, work=work, attributed=second_attributed, streak=streak, review=second, case=second_case
        ):
            entry["upstream_passed"] = int(entry.get("upstream_passed") or 0) + 1
            self.reopen_layer(entry, second_fork, paper=str(spec.get("name") or ""), number=number, why=second_reason, case=second_case)
            return {**second, "reask": True, "upstream_case": dict(second_case.get("case") or {})}, second_fork, second_reason, second
        self.event(
            "fork_capped",
            paper=str(spec.get("name") or ""),
            iteration=int(number),
            requested=str(fork_stage),
            held_at=str(work),
            work_face=str(work),
            upstream_case_rejected=str(case.get("reason") or ""),
            upstream_case_answered=str(second_case.get("reason") or ""),
        )
        return (
            {**review, "reask": True, "capped": True, "upstream_case_rejected": str(case.get("reason") or "")},
            work,
            f"the review asked to go back to {fork_stage}, before the layer this round is working on "
            f"({work}), and gave no reading of that layer that holds, so the fork is held at {work}",
            second,
        )

    def _record_outcome(self, spec: dict[str, Any], entry: dict[str, Any], iter_dir: Path, result: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        """Read where this iteration landed and make it the next iteration's starting point."""

        try:
            evidence = evolve_evidence(iter_dir, result)
        except Exception as exc:  # noqa: BLE001 - a thin evidence record must not lose the state
            evidence = {"read_error": f"{type(exc).__name__}: {exc}"}
        reuse = self._reuse_verdict(iter_dir, record)
        # Every variant this iteration drew joins the paper's archive before the loop decides
        # anything with it. A variant nobody kept is still a figure somebody drew, and a round
        # that begins from the published line again and again is a round that never explores.
        rows = archive_rows(iter_dir, int(record.get("iteration") or 0))
        archive = entry.setdefault("archive", [])
        known = {(int(item.get("iter") or 0), int(item.get("round") or 0), int(item.get("candidate") or 0)) for item in archive if isinstance(item, dict)}
        for row in rows:
            key = (row["iter"], row["round"], row["candidate"])
            if key in known:
                continue
            known.add(key)
            archive.append(row)
        write_json(iter_dir / "archive.json", {"contract": "rfs.paper_archive", "version": 1, "iteration": int(record.get("iteration") or 0), "rows": rows})
        kept_rounds = accepted_rounds(iter_dir)
        entry["repair_rounds"] = repair_rounds(iter_dir)
        # Two ways of watching the same thing: the round kept nothing it drew, or it kept drawing
        # the same defect. Either one is a round in which the figure did not move forward, and the
        # count is what buys the review a sentence about the layer above -- never an action.
        defects = iteration_defects(iter_dir)
        stalled = not kept_rounds or (bool(defects) and defects == [str(item) for item in entry.get("last_defect_ids") or []])
        entry["last_defect_ids"] = defects
        entry["stalled_streak"] = int(entry.get("stalled_streak") or 0) + 1 if stalled else 0
        # The layers this iteration worked on, and the ones its own reflections called
        # finished. Both are records: the loop keeps the next round's feet where the model
        # left them and never reads a measurement as a verdict about a layer.
        finished_at = int(record.get("iteration") or 0)
        paper_name = str(spec.get("name") or "")
        self.note_layer_moves(entry, iter_dir, number=finished_at)
        # A person's answer is read before the model's, and that order is the whole point. A layer is
        # sealed once, by whoever gets there first, so reading the model's own reflection first left
        # a person's seal recorded as the model's -- and a layer the model had sealed is one the
        # model may open again on its own word. The person answered about a figure they were shown;
        # their word is the one that stands.
        # What a person answered at this iteration's own judgement points. Their seal is a seal,
        # and the layer they name is where the work goes: naming an earlier layer opens it, which
        # is the only way the work face moves back up, and it needs no case because a person asking
        # is the thing the case would have been evidence for. A layer below the work face is left
        # to the walk -- sealing above it is what moves the work face down, not this line.
        for row in iteration_human_answers(iter_dir):
            sealed = str(row.get("settled_layer") or "")
            if sealed:
                self.settle_layer(
                    entry,
                    sealed,
                    paper=paper_name,
                    number=finished_at,
                    because=str(row.get("settled_because") or ""),
                    by="human",
                )
            move_to = str(row.get("direction_layer") or "")
            current = work_face(entry.get("layers"))
            if move_to in STAGE_ORDER and current in STAGE_ORDER and STAGE_ORDER.index(move_to) < STAGE_ORDER.index(current):
                self.reopen_layer(
                    entry,
                    move_to,
                    paper=paper_name,
                    number=finished_at,
                    why="a person said the fault is still in this layer",
                    by="human",
                )
        # The layers this iteration's own reflections called finished. Read after the person's
        # answers, so a model that seals a layer a person already sealed is refused rather than
        # overwriting who sealed it.
        for stage, because in iteration_layer_settled(iter_dir).items():
            self.settle_layer(entry, stage, paper=paper_name, number=finished_at, because=because)
        entry["work_face"] = work_face(entry.get("layers"))
        self.write_layer_ledger(iter_dir, spec, finished_at, entry)
        # The trials a person filed, now that a round has been measured under them: one that kept
        # the change it asked for stands from the next round on. A round that kept nothing is not
        # evidence about the sentence, so the trial stays a trial.
        entry["human_preferences_promoted"] = self.promote_measured_preferences(
            entry, iter_dir, paper_name, finished_at, kept=bool(kept_rounds)
        )
        if kept_rounds:
            # The published version moved forward here, so the next round continues from this line.
            entry["best_moved_at"] = int(record.get("iteration") or 0)
        state_record = _read_json_or_none(iter_dir / "evolution_state.json")
        best_dir = str((state_record or {}).get("best_candidate_dir") or "")
        best_score = (state_record or {}).get("best_score")
        if not best_dir:
            candidate = iter_dir / "round_00" / "candidate_00"
            best_dir = str(candidate) if candidate.is_dir() else ""
        if best_dir:
            entry["best"] = best_dir
        entry["best_score"] = best_score
        # Where this version stands against the best this paper had reached. It is written
        # down here and read by nothing: the round it describes was already decided. The
        # record keeps the verdict and the two versions, not the seat's whole answer, which
        # is beside the round in ``cross_round_comparison.json``.
        comparison = self.compare_rounds(iter_dir, finished_at, entry, spec)
        if comparison:
            record["cross_round"] = {
                "verdict": str(comparison.get("verdict") or ""),
                "preference": str(comparison.get("preference") or ""),
                "reason": str(comparison.get("reason") or ""),
                "history_best": dict(comparison.get("history_best") or {}),
                "current": dict(comparison.get("current") or {}),
            }
        # The same walk, as numbers: the cost of settling each layer, how often the seat reached
        # above its layer and how often it was let, and what its reflection left behind. Written
        # beside the ledger and appended to the trend, because "how many rounds to a layer" is the
        # number the whole direction change is meant to move. It is counted after the comparison
        # above and not before it: the comparison is one of the five numbers, and a trend written
        # first would count every round as one nobody had compared yet.
        self.write_round_metrics(iter_dir, spec, finished_at, entry, wave=int(record.get("wave") or 0))
        record.update(
            {
                "status": "completed",
                "best": str(entry.get("best") or ""),
                "best_score": best_score,
                "figure": str(evidence.get("figure") or ""),
                "rounds": int(evidence.get("rounds") or 0),
                "experiences_retrieved": list(evidence.get("experiences_retrieved") or []),
                "experiences_injected": list(evidence.get("experiences_injected") or []),
                "candidate_experience_ids": list(evidence.get("candidate_experience_ids") or []),
                "prompt_patches_earned": list(evidence.get("prompt_patches_earned") or []),
                "patches_applied": list(evidence.get("patches_applied") or []),
                "rules_applied": list(evidence.get("rules_applied") or []),
                "round_cost": dict(result.get("round_cost") or {}),
                "reuse": reuse,
                "accepted_rounds": kept_rounds,
                "archive_size": len(archive),
            }
        )
        entry["last_reuse"] = reuse
        entry.setdefault("history", []).append(record)
        self.event(
            "iteration_finished",
            paper=spec["name"],
            iteration=record["iteration"],
            best=str(record.get("best") or ""),
            best_score=best_score,
            figure=str(record.get("figure") or ""),
            experiences_injected=list(record.get("experiences_injected") or []),
            candidate_experience_ids=list(record.get("candidate_experience_ids") or []),
            prompt_patches_earned=list(record.get("prompt_patches_earned") or []),
            reuse_accepted=bool(reuse.get("accepted")),
            reuse_declined_reason=str(reuse.get("declined_reason") or ""),
            accepted_rounds=kept_rounds,
            archive_size=len(archive),
        )
        return record

    def _reuse_verdict(self, iter_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
        """Read whether the run actually inherited the previous iteration it was pointed at.

        Asking for a reuse and being declined is not the same run: the second one re-plans the
        whole figure, so its cost and its meaning differ. The loop must never let a wave claim to
        have continued from the previous version when the run refused it, so the verdict is read
        back from the run's own report and recorded on the iteration either way.
        """

        if not record.get("reuse_from"):
            return {"requested": False, "accepted": False, "declined_reason": ""}
        report = _read_json_or_none(iter_dir / "stage_reuse_report.json")
        declined = str((report or {}).get("declined_reason") or "")
        accepted = bool(report) and not declined
        if not accepted:
            self.event(
                "reuse_declined",
                paper=record.get("paper"),
                iteration=record.get("iteration"),
                reuse_from=str(record.get("reuse_from") or ""),
                reason=declined or "the run wrote no reuse report",
            )
        return {"requested": True, "accepted": accepted, "declined_reason": declined}

    # -- validation --------------------------------------------------------------------------

    def queue_validations(self, records: list[dict[str, Any]]) -> list[str]:
        """Hand earned candidates to the validation queue, once each.

        The loop calls this as each paper finishes its iteration -- the slowest paper of a wave
        must not hold another paper's measurement back -- and again over the wave's outcomes as a
        net. An experience the state already knows -- queued, running or decided -- is not queued
        again: the point of the queue is that every candidate is measured exactly once, not that
        the loop pays for the same measurement twice. With validation switched off the candidate
        is still named, so the report says what was left unmeasured instead of staying silent.

        A prompt patch earned this round travels by the same queue and reaches another paper only
        here: the state records which patches have been handed over, so a patch is measured once,
        and a patch this loop already retired is not resurrected by the paper that wrote it.
        """

        queued: list[str] = []
        with self._validation_lock:
            for record in records:
                paper = str(record.get("paper") or "")
                for raw_id in list(record.get("candidate_experience_ids") or []):
                    name = str(raw_id or "").strip()
                    if not name or name in self._validation_seen:
                        continue
                    self._validation_seen.add(name)
                    if not self.validation.get("enabled"):
                        self.event(
                            "validation_skipped",
                            experience_id=name,
                            paper=record.get("paper"),
                            reason=str(self.validation.get("reason") or "validation is not enabled in this config"),
                        )
                        continue
                    item = {"experience_id": name, "kind": "experience", "paper": paper, "iteration": record.get("iteration")}
                    self.validations[name] = {**item, "status": "queued", "wave": int(self.state.get("waves") or 0), "suite": str(self.validation.get("suite") or ""), "out": ""}
                    self._validation_queue.put(item)
                    queued.append(name)
                    self.event("validation_queued", paper=item["paper"], iteration=item["iteration"], experience_id=name)
                for raw_id in list(record.get("prompt_patches_earned") or []):
                    name = str(raw_id or "").strip()
                    if not name or name in self._validation_seen:
                        continue
                    # A patch is handed to the paired ON/OFF only once it holds its own-case
                    # licence: a patch still in candidate has nothing measurement-shaped behind it,
                    # and one this loop already retired is not brought back by the paper that wrote
                    # it. This is the only path by which a patch ever reaches another paper.
                    if not self._patch_holds_own_case(name):
                        continue
                    self._validation_seen.add(name)
                    if not self.validation.get("enabled"):
                        self.event(
                            "validation_skipped",
                            experience_id=name,
                            paper=paper,
                            reason=str(self.validation.get("reason") or "validation is not enabled in this config"),
                        )
                        continue
                    item = {"experience_id": name, "kind": "patch", "prompt_patch": True, "prompt_patch_id": name, "paper": paper, "iteration": record.get("iteration")}
                    self.validations[name] = {**item, "status": "queued", "wave": int(self.state.get("waves") or 0), "suite": str(self.validation.get("suite") or ""), "out": ""}
                    self._validation_queue.put(item)
                    queued.append(name)
                    self.event("validation_queued", paper=paper, iteration=item["iteration"], experience_id=name, kind="patch")
        if queued:
            self.save_state()
        return queued

    def recover_rotated_rows(self) -> list[str]:
        """Put back the measurements the library's own rotation threw away.

        Rotating the library moved rows into an archive directory beside it and left every queue
        entry that pointed at one reading "dropped" or "unknown experience". Neither is a verdict
        about the row: the row was never measured, and the absence of a measurement is not a
        measurement of failure. The archive still holds it, so the measurement goes back on the
        queue -- once, and counted, so a row no archive holds is still allowed to stop.
        """

        if not self.validation.get("enabled"):
            return []
        recovered: list[str] = []
        counters = self.state.setdefault("unmeasured_requeues", {})
        for name, record in list(self.validations.items()):
            status = str(record.get("status") or "")
            if status not in {"dropped", "failed"}:
                continue
            # A row dropped for any other reason is a real refusal and is left alone. The two
            # readings that mean "the file moved" are the drop the loop writes for it and the
            # failure the validator raises when it opens a library that no longer lists the id.
            if status == "dropped" and str(record.get("blocked_because") or "") not in {"", "row_no_longer_in_library"}:
                continue
            if status == "failed" and "Unknown experience" not in str(record.get("error") or ""):
                continue
            kind = "patch" if str(record.get("kind") or "") == "patch" else "experience"
            key = "patch_id" if kind == "patch" else "experience_id"
            if not self._archived_row(key, name):
                continue
            attempts = int(counters.get(name) or 0)
            if attempts >= MAX_UNMEASURED_REQUEUES:
                continue
            counters[name] = attempts + 1
            record["status"] = "requeued"
            record["requeue_reason"] = "the library was rotated while this row waited; the row is read back from the archive"
            record["recovered_from_archive"] = True
            record.pop("error", None)
            item = {"experience_id": name, "paper": str(record.get("paper") or ""), "iteration": record.get("iteration"), "kind": kind}
            if kind == "patch":
                item["prompt_patch"] = True
                item["prompt_patch_id"] = name
            self._validation_queue.put(item)
            recovered.append(name)
            self.event(
                "validation_row_recovered",
                experience_id=name,
                kind=kind,
                previous_status=status,
                reason="the row was rotated into an archive, not retired",
            )
        if recovered:
            self.event("validation_rows_recovered", experience_ids=recovered, reason="the library rotation had thrown these measurements away")
            self.save_state()
        return recovered

    def resume_interrupted_validations(self) -> list[str]:
        """Put back the candidates whose measurement a previous call ended in the middle of.

        A validation is hours long, so a call that stops for any reason will usually stop inside
        one. Leaving that entry marked "running" for ever would mean the candidate is never
        measured and never warned about, which is the silence this loop exists not to have.
        """

        if not self.validation.get("enabled"):
            return []
        resumed: list[str] = []
        dropped: list[str] = []
        counters = self.state.setdefault("unmeasured_requeues", {})
        held = self._library_row_ids()
        held_patches = self._patch_row_ids()
        for name, record in list(self.validations.items()):
            # ``requeued`` is here on purpose: a call that ended between putting a row back on the
            # queue and the worker reaching it leaves that status on disk, and the queue itself
            # dies with the process. Reading only ``queued`` and ``running`` orphaned those rows
            # for ever -- the four that had passed qualification sat at ``requeued`` for a day.
            if str(record.get("status") or "") not in {"queued", "running", "requeued"}:
                continue
            ids_held = held_patches if str(record.get("kind") or "") == "patch" else held
            if ids_held is not None and name not in ids_held:
                # The library no longer holds the row this measurement was about -- it was
                # archived, retired or rewritten while the attempt ran. Putting it back on the
                # queue buys a wave that can only end in "unknown experience", so it is dropped
                # once and named instead of being retried on every restart.
                record["status"] = "dropped"
                record["blocked_because"] = "row_no_longer_in_library"
                record["requeue_reason"] = "the library no longer holds this row"
                dropped.append(name)
                self.event("validation_dropped", experience_id=name, reason="row_no_longer_in_library", wave=record.get("wave"))
                continue
            # An interrupted attempt that had already decided cases is continuing, not failing
            # again: it must not be charged to the budget, or a long set would run out of tries
            # for the only reason that it is long.
            if self._decided_cases(record) <= 0:
                attempts = int(counters.get(name) or 0)
                if attempts >= MAX_UNMEASURED_REQUEUES:
                    record["status"] = "unverifiable"
                    record["blocked_because"] = "requeue_budget_exhausted"
                    record["requeue_reason"] = "the candidate was interrupted before any case was decided this many times"
                    self.event("validation_parked", experience_id=name, reason="requeue_budget_exhausted", attempts=attempts)
                    continue
                counters[name] = attempts + 1
            record["status"] = "requeued"
            record["requeue_reason"] = "the call that was measuring this candidate ended before it finished"
            requeue_item = {"experience_id": name, "paper": str(record.get("paper") or ""), "iteration": record.get("iteration"), "kind": str(record.get("kind") or "experience")}
            if requeue_item["kind"] == "patch":
                requeue_item["prompt_patch"] = True
                requeue_item["prompt_patch_id"] = name
            self._validation_queue.put(requeue_item)
            resumed.append(name)
        if resumed:
            self.event("validation_requeued", experience_ids=resumed, reason="the previous call ended mid-validation")
        # Saved whether or not anything resumed: a row parked for exhausting its requeue budget
        # changed the queue too, and leaving that in memory would re-park it on the next call.
        self.save_state()
        return resumed

    def _library_row(self, experience_id: str) -> dict[str, Any]:
        """The row under test as the library holds it now, or nothing when it cannot be read.

        The rung a validation *starts* from decides what its verdict may mean: a row that already
        won on the paper that produced it is measuring a second paper, and a row that was never
        measured there is not.
        """

        try:
            rows = ExperienceStore(self.library).records_with_status()
        except Exception as exc:  # noqa: BLE001 - an unreadable library must not stop the loop
            self.event("validation_library_unreadable", error=f"{type(exc).__name__}: {exc}")
            return {}
        for row in rows:
            if str((row or {}).get("experience_id") or "") == str(experience_id):
                return dict(row)
        return self._archived_row("experience_id", experience_id)

    def _library_row_ids(self) -> set[str] | None:
        """The ids the library holds right now, or ``None`` when it could not be read.

        ``None`` is not the empty set: a library that will not open is a reason to leave the queue
        alone, while a library that opens and does not hold the row is a reason to stop retrying
        a measurement that can only end in "unknown experience".
        """

        try:
            rows = ExperienceStore(self.library).records_with_status()
        except Exception as exc:  # noqa: BLE001 - an unreadable library must not stop the loop
            self.event("validation_library_unreadable", error=f"{type(exc).__name__}: {exc}")
            return None
        return {str((row or {}).get("experience_id") or "") for row in rows} | self._archived_ids("experience_id")

    def _archive_roots(self) -> list[Path]:
        """Where the libraries this one was rotated out of live.

        Rotating the library used to be a delete as far as a queued measurement was concerned: the
        row it was about moved into an archive directory beside the library and the queue entry
        turned into "unknown experience". The archives are read here for the same reason a row is
        read at all -- the measurement is owed an answer about the row that was queued, not about
        whichever file happens to hold the id today.
        """

        root = Path(str(self.library or ""))
        if not root.name:
            return []
        found: list[Path] = []
        for candidate in sorted(root.parent.glob(root.name + "_archived*")):
            if candidate.is_dir():
                found.append(candidate)
        return found

    def _archived_ids(self, key: str) -> set[str]:
        """Every id the archive directories hold under ``key``, whatever they are rows of."""

        identifiers: set[str] = set()
        for row in self._archived_rows():
            identifier = str(row.get(key) or "")
            if identifier:
                identifiers.add(identifier)
        return identifiers

    def _archived_rows(self) -> list[dict[str, Any]]:
        """The rows every archive directory holds, as plain dictionaries."""

        rows: list[dict[str, Any]] = []
        for root in self._archive_roots():
            for name in sorted(path.name for path in root.glob("*.jsonl")):
                path = root / name
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(value, dict):
                            rows.append(value)
                except OSError:  # noqa: PERF203 - an unreadable archive is skipped, not fatal
                    continue
        return rows

    def _archived_row(self, key: str, identifier: str) -> dict[str, Any]:
        """One archived row by id, or nothing when no archive holds it either."""

        want = str(identifier or "")
        for row in self._archived_rows():
            if str(row.get(key) or "") == want:
                return dict(row)
        return {}

    def _patch_row(self, patch_id: str) -> dict[str, Any]:
        """The prompt patch a measurement is about, as the patch library holds it now.

        A patch is read from its own store for the same reason an experience is: the rung it holds
        when the measurement starts decides what the verdict may mean, and a patch retired or
        rewritten between the queue and the run must not be read as the row that was queued.
        """

        try:
            rows = PromptPatchStore(self.library).records_with_status()
        except Exception as exc:  # noqa: BLE001 - an unreadable library must not stop the loop
            self.event("validation_library_unreadable", error=f"{type(exc).__name__}: {exc}")
            return {}
        for row in rows:
            if str((row or {}).get("patch_id") or "") == str(patch_id):
                return dict(row)
        return self._archived_row("patch_id", patch_id)

    def _patch_row_ids(self) -> set[str] | None:
        """The patch ids the library holds right now, or ``None`` when it could not be read."""

        try:
            rows = PromptPatchStore(self.library).records_with_status()
        except Exception as exc:  # noqa: BLE001 - an unreadable library must not stop the loop
            self.event("validation_library_unreadable", error=f"{type(exc).__name__}: {exc}")
            return None
        return {str((row or {}).get("patch_id") or "") for row in rows} | self._archived_ids("patch_id")

    def _live_row_ids(self, key: str) -> set[str] | None:
        """The ids the live library holds, before any archive is read.

        ``_library_row_ids`` and ``_patch_row_ids`` count the archives, because a queued measurement
        may point at a row the library rotated away and that row must stay queued. This answers the
        narrower question the validator asks when it opens the library -- does the file the
        measurement will read hold this row -- which is what decides whether the archived row has
        to be pinned for that reader.
        """

        store = PromptPatchStore(self.library) if key == "patch_id" else ExperienceStore(self.library)
        try:
            rows = store.records_with_status()
        except Exception as exc:  # noqa: BLE001 - an unreadable library must not stop the loop
            self.event("validation_library_unreadable", error=f"{type(exc).__name__}: {exc}")
            return None
        return {str((row or {}).get(key) or "") for row in rows}

    def _pin_archived_row(self, kind: str, experience_id: str, out: Any) -> bool:
        """Pin the row a measurement is about when only an archive still holds it.

        A queued measurement points at the row the round earned, and the library can rotate that row
        away long before the queue reaches it. ``_library_row_ids`` keeps such a row queued, but the
        validator pins the row from the live library, so without this the measurement raises
        "unknown experience" on every attempt and the requeue this loop performs can never succeed.
        The archived row is written in the shape the validator pins, so the run measures the row the
        queue was about rather than one the library happens to hold now.
        """

        key = "patch_id" if kind == "patch" else "experience_id"
        live = self._live_row_ids(key)
        if live is None or str(experience_id) in live:
            return False
        row = self._archived_row(key, experience_id)
        if not row:
            return False
        payload: dict[str, Any] = {
            "contract": ROW_UNDER_TEST_CONTRACT,
            "version": 1,
            "validation_profile": "",
            "experience_id": "",
            "prompt_patch_id": "",
            "experience": None,
            "prompt_patch": None,
        }
        if kind == "patch":
            payload["prompt_patch_id"] = str(row.get("patch_id") or experience_id)
            payload["prompt_patch"] = row
        else:
            payload["experience_id"] = str(row.get("experience_id") or experience_id)
            payload["experience"] = row
        write_json(ensure_dir(out) / ROW_UNDER_TEST_FILE, payload)
        self.event("validation_row_pinned_from_archive", experience_id=str(experience_id), kind=kind, out=str(out))
        return True

    def _patch_holds_own_case(self, patch_id: str) -> bool:
        """Whether a patch holds the rung that makes it worth asking another paper about.

        The own-case rung is a patch measured on the paper that wrote it. A patch below it has no
        measurement behind it yet and a patch at a rung above it has already travelled, so neither
        is queued: the queue exists to ask one question, once, of the sentence that is ready for it.
        """

        return str(self._patch_row(patch_id).get("status") or "") == CURRENT_CASE_STATUS

    def resume_cross_domain_follow_ups(self) -> list[str]:
        """Queue again the other-domain paper a call was handed but did not get to measure.

        A follow-up is queued from inside a measurement, so it can land while the call is on its
        way out; the worker is a daemon and the loop does not wait hours for it. The row is
        marked in the state instead, so the work is owed rather than lost, and the next call
        queues it before anything else.
        """

        if not self.validation.get("enabled"):
            return []
        resumed: list[str] = []
        follow_ups = self.state.setdefault("cross_domain_follow_ups", {})
        for experience_id, marker in list(follow_ups.items()):
            if str(marker) != "queued":
                continue
            record = dict(self.validations.get(experience_id) or {})
            self.validations[experience_id] = {**record, "status": "queued"}
            self._validation_seen.add(str(experience_id))
            self._validation_queue.put(
                {
                    "experience_id": str(experience_id),
                    "kind": str(record.get("kind") or "experience"),
                    "paper": str(record.get("paper") or ""),
                    "iteration": record.get("iteration"),
                    "purpose": "cross_domain",
                    "profile": "quick",
                }
            )
            resumed.append(str(experience_id))
            self.event("cross_domain_follow_up_requeued", experience_id=str(experience_id))
        if resumed:
            self.save_state()
        return resumed

    def resume_unmeasured_candidates(self) -> list[str]:
        """Queue every candidate in the library that has no measurement to its name.

        A validation that finished with zero cases measured nothing: the case set could not be
        drawn, or the exam was refused before a paper was run. It is not a measurement, so the
        candidate is not a decided row, and treating it as one is how a candidate nothing could
        measure stayed unmeasured for ever while the report said it had been tried. The library,
        not the state, is what says which candidates are waiting, so a row this run never queued
        at all is picked up here too.
        """

        if not self.validation.get("enabled"):
            return []
        try:
            rows = ExperienceStore(self.library).records_with_status()
        except Exception as exc:  # noqa: BLE001 - a library that will not open must not stop the loop
            self.event("validation_library_unreadable", error=f"{type(exc).__name__}: {exc}")
            return []
        counters = self.state.setdefault("unmeasured_requeues", {})
        queued: list[str] = []
        with self._validation_lock:
            for row in rows:
                name = str((row or {}).get("experience_id") or "").strip()
                if not name or str((row or {}).get("status") or "") != "candidate":
                    continue
                decided = self.validations.get(name) or {}
                if int((decided.get("summary") or {}).get("measured_cases") or 0) > 0:
                    continue
                if str(decided.get("status") or "") in {"queued", "running", "requeued"}:
                    continue
                # The budget is spent by an attempt that measured nothing and decided nothing.
                # A row whose set is part-way through is continuing, and charging it here is how
                # a resumable validation used to run out of tries while it was making progress.
                if self._decided_cases(decided) <= 0:
                    attempts = int(counters.get(name) or 0)
                    if attempts >= MAX_UNMEASURED_REQUEUES:
                        self.validations[name] = {**decided, "status": "unverifiable", "blocked_because": "requeue_budget_exhausted"}
                        continue
                    counters[name] = attempts + 1
                self._validation_seen.add(name)
                item = {"experience_id": name, "paper": str(decided.get("paper") or ""), "iteration": decided.get("iteration")}
                self.validations[name] = {
                    **decided,
                    **item,
                    "status": "queued",
                    "wave": int(self.state.get("waves") or 0),
                    "suite": str(self.validation.get("suite") or ""),
                    "out": "",
                }
                self._validation_queue.put(item)
                queued.append(name)
                self.event(
                    "validation_queued",
                    experience_id=name,
                    paper=item["paper"],
                    iteration=item["iteration"],
                    because="the library still holds this candidate and no case was ever measured for it",
                )
        if queued:
            self.save_state()
        return queued

    def resume_rows_owed_a_second_paper(self) -> list[str]:
        """Hand the next paper to every row that has already passed its first measurement.

        The ladder asks two questions of a row: does it reproduce where it came from, and does it
        travel to a paper that did not write it. Only a row born on a paper that had already
        measured it was ever asked the second one, so a row measured for the first time -- most of
        the library -- was written down and left with nothing behind it: its verdict
        ``quick_validated`` sat in the library and ``cross_domain_follow_ups`` stayed empty. The
        rows that have passed the first question name themselves, so they are read from the library
        rather than from the round that wrote them; that is also what picks up a row measured
        before this ran for the first time. Each row is queued once: one already queued, already
        travelling, already promoted or already refused is left where it is.
        """

        if not self.validation.get("enabled"):
            return []
        follow_ups = self.state.setdefault("cross_domain_follow_ups", {})
        queued: list[str] = []
        for row in self._rows_that_passed_the_first_measurement():
            name = str(row.get("id") or "").strip()
            if not name:
                continue
            if str(follow_ups.get(name) or "") in {"queued", "running", "done", "refused"}:
                continue
            decided = dict(self.validations.get(name) or {})
            if str(decided.get("status") or "") in {"queued", "running", "requeued"}:
                continue
            record = {
                "experience_id": name,
                "kind": str(row.get("kind") or "experience"),
                "paper": str(decided.get("paper") or ""),
                "iteration": decided.get("iteration"),
                "out": str(decided.get("out") or ""),
            }
            if self.queue_cross_domain_follow_up(
                record,
                reason="the row passed its first measurement and still owes the paper that tests travel",
            ):
                queued.append(name)
        if queued:
            self.save_state()
        return queued

    def _rows_that_passed_the_first_measurement(self) -> list[dict[str, Any]]:
        """Every row whose first measurement is behind it, so the travel question is still open.

        A row that won on a paper other than its own (``quick_validated``) and a row that won the
        paired ON/OFF on the paper that wrote it (``current_case_validated``) have both answered the
        first question and neither has answered the second. A row that already travelled, that
        lost, or that this loop retired is not one of these and is not named.
        """

        rows: list[dict[str, Any]] = []
        try:
            for row in ExperienceStore(self.library).records_with_status():
                if str((row or {}).get("status") or "") in {CURRENT_CASE_STATUS, "quick_validated"}:
                    rows.append({"id": str((row or {}).get("experience_id") or ""), "kind": "experience"})
        except Exception as exc:  # noqa: BLE001 - a library that will not open must not stop the loop
            self.event("validation_library_unreadable", error=f"{type(exc).__name__}: {exc}")
        try:
            for row in PromptPatchStore(self.library).records_with_status():
                if str((row or {}).get("status") or "") == CURRENT_CASE_STATUS:
                    rows.append({"id": str((row or {}).get("patch_id") or ""), "kind": "patch"})
        except Exception as exc:  # noqa: BLE001 - a library that will not open must not stop the loop
            self.event("prompt_patch_library_unreadable", error=f"{type(exc).__name__}: {exc}")
        return [row for row in rows if row["id"]]

    @staticmethod
    def _decided_cases(record: dict[str, Any] | None) -> int:
        """How many cases of this candidate a call has already put to the Judge.

        Progress, not success: a case that came back noisy still cost its generation and still
        counts as the set moving forward, which is what keeps a resumed set from being charged to
        the retry budget every time the process is restarted.
        """

        body = record if isinstance(record, dict) else {}
        summary = body.get("summary") if isinstance(body.get("summary"), dict) else {}
        return int(summary.get("cases") or body.get("cases_decided") or 0)

    def validation_attempt_dir(self, experience_id: str) -> Path:
        """The directory this measurement writes into, continuing an unfinished one.

        A validation is hours of generation, so a call that stops inside one leaves a partly
        measured set on disk. Starting the next call in a fresh directory would throw that work
        away and pay for it again; the newest directory that holds no finished report is
        continued instead, and only when every attempt has a report is a new one opened.
        """

        base = self.out / "validation" / station_slug(experience_id)
        unfinished = [path for path in sorted(base.glob("wave_*")) if not (path / "cross_paper_validation.json").is_file()]
        if unfinished:
            return unfinished[-1]
        numbers = [int(path.name.split("_")[-1]) for path in base.glob("wave_*") if path.name.split("_")[-1].isdigit()]
        wave = int(self.state.get("waves") or 0)
        return base / f"wave_{max([*numbers, wave - 1]) + 1:02d}"

    def start_validation_worker(self) -> None:
        """Start the background worker that measures candidates while the loop keeps evolving."""

        if not self.validation.get("enabled") or self._validation_worker is not None:
            return
        self._validation_worker = threading.Thread(target=self._validation_loop, name="continuous-validation", daemon=True)
        self._validation_worker.start()
        self.event("validation_worker_started", suite=str(self.validation.get("suite") or ""), workers=1)

    def _validation_loop(self) -> None:
        while not self._validation_stop.is_set():
            try:
                item = self._validation_queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self.run_validation(item)
            except Exception as exc:  # noqa: BLE001 - the worker must survive one bad candidate
                self.event("validation_worker_error", experience_id=str(item.get("experience_id") or ""), error=f"{type(exc).__name__}: {exc}")
            finally:
                self._validation_queue.task_done()

    def run_validation(self, item: dict[str, Any]) -> dict[str, Any]:
        """Measure one candidate on the four-group case set and write down what the ladder said."""

        experience_id = str(item.get("experience_id") or "")
        kind = "patch" if (str(item.get("kind") or "") == "patch" or item.get("prompt_patch")) else "experience"
        patch_id = experience_id if kind == "patch" else ""
        wave = int(self.state.get("waves") or 0)
        out = self.validation_attempt_dir(experience_id)
        # A measurement that has not pinned its row yet needs one to measure. Asking the library
        # before the run starts is what turns "unknown experience" -- a wave of work that dies on
        # its first step -- into a queue entry that is dropped where it stands. An attempt that
        # already pinned its row is measured from the pin, whatever the library says now.
        if not (out / ROW_UNDER_TEST_FILE).is_file():
            held = self._patch_row_ids() if kind == "patch" else self._library_row_ids()
            if held is not None and experience_id not in held:
                dropped_record = {
                    "paper": str(item.get("paper") or ""),
                    "iteration": item.get("iteration"),
                    "wave": wave,
                    "suite": str(self.validation.get("suite") or ""),
                    "out": str(out),
                    "status": "dropped",
                    "blocked_because": "row_no_longer_in_library",
                    "note": "the library no longer holds this row, so the measurement was dropped before it was paid for",
                    "finished_at": round(float(self.clock()), 3),
                }
                self.validations[experience_id] = dropped_record
                self.save_state()
                self.event("validation_dropped", experience_id=experience_id, reason="row_no_longer_in_library", out=str(out))
                return dropped_record
            self._pin_archived_row(kind, experience_id, out)
        previous = self.validations.get(experience_id) or {}
        # The rung the row holds before this measurement is what says whether the case set is
        # asking "does this travel" or "does this reproduce": a candidate has never been measured
        # anywhere, and a row that won on its own paper is being asked about a second one.
        starting_row = self._patch_row(experience_id) if kind == "patch" else self._library_row(experience_id)
        starting_status = str(starting_row.get("status") or "")
        starting_applicability = str(starting_row.get("applicability") or "")
        record: dict[str, Any] = {
            "experience_id": experience_id,
            "kind": kind,
            "paper": str(item.get("paper") or ""),
            "iteration": item.get("iteration"),
            "wave": wave,
            "suite": str(self.validation.get("suite") or ""),
            "purpose": str(item.get("purpose") or "candidate"),
            "starting_status": starting_status,
            "starting_applicability": starting_applicability,
            "validation_profile": str(item.get("profile") or self.validation.get("profile") or ""),
            "out": str(out),
            "started_at": round(float(self.clock()), 3),
        }
        self.validations[experience_id] = {**record, "status": "running"}
        self.save_state()
        self.event("validation_started", experience_id=experience_id, paper=record["paper"], iteration=record["iteration"], suite=record["suite"], validation_profile=record["validation_profile"], out=record["out"], continuing_from=int(self._decided_cases(previous)))

        def on_progress(payload: dict[str, Any]) -> None:
            self.event("validation_progress", experience_id=experience_id, out=str(out), **{str(key): value for key, value in (payload or {}).items()})

        try:
            report = self.validate_runner(
                suite=record["suite"],
                experience_library=self.library,
                experience_id=experience_id,
                prompt_patch_id=patch_id,
                out=str(out),
                run_options=dict(self.config.get("evolve") or {}),
                on_progress=on_progress,
                validation_profile=str(record.get("validation_profile") or ""),
            )
        except Exception as exc:  # noqa: BLE001 - a measurement that failed is a recorded outcome
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        else:
            body = report if isinstance(report, dict) else {}
            ladder = body.get("ladder") if isinstance(body.get("ladder"), dict) else {}
            record.update(
                {
                    "status": str(body.get("status") or "unknown"),
                    "validation_profile": str(body.get("validation_profile") or record.get("validation_profile") or ""),
                    "stopped_because": str(ladder.get("stopped_because") or ""),
                    "summary": {str(key): value for key, value in (body.get("summary") or {}).items()},
                    "cases_decided": int((body.get("summary") or {}).get("cases") or 0),
                    "blocked_because": str(body.get("blocked_because") or ""),
                    "all_pairs_isolated": bool(body.get("all_pairs_isolated")),
                    "all_cases_measured": bool(body.get("all_cases_measured")),
                }
            )
        record["finished_at"] = round(float(self.clock()), 3)
        # A person at the validation seat answers before the ladder does, and only where the ladder
        # would move the row: that is the entry decision, and it is asked about a measurement that
        # exists. Their "hold" keeps the row where it was; anything else leaves the ladder reading it.
        person = self.person_validation(record)
        if str(person.get("decision") or "") == "hold":
            record["rung_awarded"] = str(record.get("starting_status") or "")
            record["held_by_person"] = True
            record["person_note"] = str(person.get("note") or "")
            self.event(
                "human_validation_held",
                experience_id=experience_id,
                verdict=str(record.get("status") or ""),
                rung_kept=str(record.get("starting_status") or ""),
                by=str(person.get("by") or ""),
                card=str(person.get("card") or ""),
            )
        else:
            record = self.settle_measurement_rung(record)
        self.validations[experience_id] = record
        self.save_state()
        self.event(
            "validation_finished",
            experience_id=experience_id,
            status=record["status"],
            rung_awarded=str(record.get("rung_awarded") or ""),
            stopped_because=str(record.get("stopped_because") or ""),
            error=str(record.get("error") or ""),
            summary=dict(record.get("summary") or {}),
        )
        return record

    # -- the rung a measurement earns, and the paper that has to measure it next ---------------

    def settle_measurement_rung(self, record: dict[str, Any]) -> dict[str, Any]:
        """Turn a verdict into the rung it earns, and queue the paper it still has to travel to.

        Two readings the loop used to leave unread:

        *A single matched paper, once the row's own paper has measured it, is the cross-domain
        rung.* The ``quick`` profile asks for exactly one case, and for a row that claims to
        travel that case is drawn from another domain. A row that already won on the paper that
        produced it, and then won on that paper, has shown the two things the ladder asks for one
        at a time -- so it is promoted to ``cross_domain_validated`` rather than left at the
        profile's own rung, where nothing but a full four-group run could ever move it.

        *A row that stops at the same-domain rung is owed one more paper.* The ladder says the
        row reproduced on its own domain and did not show that it travels; the cheap way to ask
        that question is one matched paper in another domain, so it is queued here automatically
        rather than waiting for an operator to notice.
        """

        experience_id = str(record.get("experience_id") or "")
        verdict = str(record.get("status") or "")
        purpose = str(record.get("purpose") or "candidate")
        if str(record.get("kind") or "") == "patch":
            return self._settle_patch_rung(record, verdict=verdict)
        starting = str(record.get("starting_status") or "")
        applicability = str(record.get("starting_applicability") or "")
        passed_the_single_paper = verdict in {"quick_validated", "cross_domain_validated"} and applicability != "domain"
        record["rung_awarded"] = ""
        if purpose == "cross_domain":
            # This run *is* the other-domain paper, so its verdict settles the follow-up whatever
            # it says: a preference earns the cross-domain rung, and anything else is written down
            # as the refusal it is, so the same paper is not paid for twice.
            if passed_the_single_paper and self.promote_cross_domain(experience_id, record):
                record["rung_awarded"] = "cross_domain_validated"
                record["status"] = "cross_domain_validated"
            else:
                follow_ups = self.state.setdefault("cross_domain_follow_ups", {})
                if str(follow_ups.get(experience_id) or "") == "queued":
                    follow_ups[experience_id] = "refused"
                record["rung_awarded"] = starting
                self.event(
                    "cross_domain_not_promoted",
                    experience_id=experience_id,
                    verdict=verdict,
                    stopped_because=str(record.get("stopped_because") or ""),
                )
        elif passed_the_single_paper and starting == CURRENT_CASE_STATUS:
            # A quick run of a row that has already won on the paper that produced it is the same
            # measurement as a follow-up: for a row that claims to travel, the single matched case
            # is drawn from another domain. The two things the ladder asks for have now been shown
            # one at a time, so the row is promoted rather than left at the profile's own rung.
            if self.promote_cross_domain(experience_id, record):
                record["rung_awarded"] = "cross_domain_validated"
                record["status"] = "cross_domain_validated"
        elif verdict == "same_domain_validated":
            # Reproduced on its own domain and did not show that it travels: the cheap way to ask
            # that question is one matched paper in another domain, so it is queued here.
            self.queue_cross_domain_follow_up(record, reason="same_domain_generalization_unshown")
        elif passed_the_single_paper:
            # The single matched paper preferred the row, but the row was never measured on the
            # paper it came from, so this is the first measurement rather than the second. It
            # keeps the profile's own rung instead of borrowing the cross-domain one -- and the
            # ladder's second question, whether the row travels, is queued right here. A row that
            # stopped at this rung used to have nothing put behind it: it stalled one rung short of
            # travel for ever, which is why ``cross_domain_follow_ups`` stayed empty.
            record["rung_awarded"] = "quick_validated"
            self.queue_cross_domain_follow_up(record, reason="the first measurement preferred the row, so the paper that tests travel is owed")
        return record

    def _settle_patch_rung(self, record: dict[str, Any], *, verdict: str) -> dict[str, Any]:
        """Settle what one cross-paper measurement of a prompt patch earned.

        A patch travels as a sentence, so the paired ON/OFF that just finished is the whole
        question: did the sentence help on a paper that did not write it. The ladder has already
        written the rung it awarded onto the patch itself, so this only records that reading and
        charges a patch that lost -- a patch that has now lost ``PROMPT_PATCH_REFUTATION_LIMIT``
        measurements in a row stops being injected, exactly as one that loses that many rounds.

        Only a measurement that came out worse is charged. A set nobody measured, or one whose two
        arms produced the same figure, answered nothing about the patch, so it is recorded as
        neutral and the patch keeps the rung it already held -- a "could not measure" that spent a
        patch would write off a sentence that was never actually asked a question.
        """

        patch_id = str(record.get("experience_id") or "")
        record["rung_awarded"] = ""
        if verdict in DEFAULT_RETRIEVAL_STATUSES:
            record["rung_awarded"] = verdict
            self.event(
                "prompt_patch_cross_domain",
                patch_id=patch_id,
                verdict=verdict,
                out=str(record.get("out") or ""),
                measured_cases=int((record.get("summary") or {}).get("measured_cases") or 0),
            )
        elif verdict in PATCH_REFUTING_STATUSES:
            store = PromptPatchStore(self.library)
            store.append_event(patch_id, "refuted", reason="the patch did not travel: " + verdict)
            streak = store.refutation_streak(patch_id)
            disabled = int(streak) >= PROMPT_PATCH_REFUTATION_LIMIT
            if disabled:
                store.disable(patch_id, reason="prompt_patch_refutations_reached")
            self.event("prompt_patch_refuted", patch_id=patch_id, verdict=verdict, refutations=int(streak), disabled=bool(disabled))
        else:
            self.event("prompt_patch_not_promoted", patch_id=patch_id, verdict=verdict)
        return record

    def promote_cross_domain(self, experience_id: str, record: dict[str, Any]) -> bool:
        """Move a row that travelled to ``cross_domain_validated``, and say what measured it."""

        if not experience_id:
            return False
        try:
            ExperienceStore(self.library).transition(
                experience_id,
                "cross_domain_validated",
                validation_report=str(Path(str(record.get("out") or "")) / "cross_paper_validation.json"),
                summary=dict(record.get("summary") or {}),
                matched_case=str(record.get("validation_profile") or ""),
            )
        except Exception as exc:  # noqa: BLE001 - a promotion that failed is a recorded outcome
            self.event("cross_domain_promotion_failed", experience_id=experience_id, error=f"{type(exc).__name__}: {exc}")
            return False
        if str((self.state.get("cross_domain_follow_ups") or {}).get(experience_id) or "") == "queued":
            self.state.setdefault("cross_domain_follow_ups", {})[experience_id] = "done"
        self.event(
            "cross_domain_validated",
            experience_id=experience_id,
            paper=str(record.get("paper") or ""),
            matched_profile=str(record.get("validation_profile") or ""),
            measured_cases=int((record.get("summary") or {}).get("measured_cases") or 0),
            from_rung=str(record.get("starting_status") or ""),
        )
        return True

    def queue_cross_domain_follow_up(self, record: dict[str, Any], *, reason: str = "") -> bool:
        """Queue the one other-domain paper a same-domain row still has to face.

        The follow-up is queued once per row and runs through the same single worker as every
        other measurement, so it never adds a second heavy API job beside a running one. When a
        measurement is already in flight or waiting, that is written down beside the queue entry
        rather than worked around.
        """

        experience_id = str(record.get("experience_id") or "")
        if not experience_id or not self.validation.get("enabled"):
            return False
        follow_ups = self.state.setdefault("cross_domain_follow_ups", {})
        if str(follow_ups.get(experience_id) or "") in {"queued", "running", "done"}:
            return False
        in_flight = sum(
            1 for item in self.validations.values() if str((item or {}).get("status") or "") in {"queued", "running", "requeued"}
        )
        # A patch travels as a sentence, so its follow-up has to be measured as a patch: the worker
        # reads the kind off the queue item to pick which store the row comes from.
        kind = "patch" if (str(record.get("kind") or "") == "patch" or str(record.get("prompt_patch_id") or "")) else "experience"
        item = {
            "experience_id": experience_id,
            "kind": kind,
            "paper": str(record.get("paper") or ""),
            "iteration": record.get("iteration"),
            "purpose": "cross_domain",
            "profile": "quick",
        }
        if kind == "patch":
            item["prompt_patch"] = True
            item["prompt_patch_id"] = experience_id
        follow_ups[experience_id] = "queued"
        prior_record = dict(self.validations.get(experience_id) or {})
        self.validations[experience_id] = {
            "experience_id": experience_id,
            "kind": kind,
            "prior": prior_record,
            "paper": item["paper"],
            "iteration": item["iteration"],
            "wave": int(self.state.get("waves") or 0),
            "suite": str(self.validation.get("suite") or ""),
            "purpose": "cross_domain",
            "profile": "quick",
            "status": "queued",
            "out": str(Path(str(record.get("out") or "")).parent / "cross_domain_follow_up"),
        }
        self._validation_seen.add(experience_id)
        self._validation_queue.put(item)
        self.save_state()
        self.event(
            "cross_domain_follow_up_queued",
            experience_id=experience_id,
            reason=str(reason or ""),
            in_flight=int(in_flight),
            profile="quick",
        )
        return True

    # -- output ------------------------------------------------------------------------------

    def patch_yield_report(self) -> dict[str, Any]:
        """How many different sentences the patch library holds, and how often one was asked for.

        Two numbers, because one alone flatters the library. ``patches`` counts rows and a row is
        one sentence however many rounds asked for it; ``asked`` counts the rounds behind those
        rows, so a library with three rows and one sentence reads as "one sentence, asked for
        eleven times" rather than as progress. ``novelty`` is the share of rows that are a distinct
        sentence, which is the number that says whether the seat is writing anything new; the
        ``licensed`` count is how many rows got far enough to be injected somewhere.
        """

        try:
            rows = PromptPatchStore(self.library).records_with_status()
        except Exception as exc:  # noqa: BLE001 - an unreadable library is reported, never fatal
            return {"library_error": f"{type(exc).__name__}: {exc}", "patches": 0, "distinct_sentences": 0, "novelty": None, "asked": 0}
        sentences = {patch_sentence_key(row) for row in rows if isinstance(row, dict)}
        asked = sum(max(1, int(row.get("support") or 1)) for row in rows if isinstance(row, dict))
        licensed = [
            row
            for row in rows
            if isinstance(row, dict) and str(row.get("status") or "") in {"current_case_validated", "same_domain_validated", "cross_domain_validated", "promoted"}
        ]
        waves = int(self.state.get("waves") or 0)
        return {
            "patches": len(rows),
            "distinct_sentences": len(sentences),
            "novelty": (round(len(sentences) / len(rows), 4) if rows else None),
            "asked": asked,
            "yield_per_round": (round(asked / waves, 4) if waves else None),
            "licensed": len(licensed),
        }

    def measurement_report(self) -> dict[str, Any]:
        """How far the library has actually got, so a stall cannot read as a failure.

        The loop's queue says what it tried; the library says what exists. Neither alone can tell
        "measured and refused" apart from "never finished", which is the state that made 35 rows
        and 0 promotions look like a judgement on the experiences. Every reading here is taken
        from the two sources directly rather than from a counter the loop kept.
        """

        rows: list[dict[str, Any]] = []
        library_error = ""
        try:
            rows = ExperienceStore(self.library).records_with_status()
        except Exception as exc:  # noqa: BLE001 - an unreadable library is reported, not fatal
            library_error = f"{type(exc).__name__}: {exc}"
        by_status: dict[str, int] = {}
        for row in rows:
            status = str((row or {}).get("status") or "")
            by_status[status] = by_status.get(status, 0) + 1
        counters = self.state.get("unmeasured_requeues") if isinstance(self.state.get("unmeasured_requeues"), dict) else {}
        queue = [{**record, "experience_id": name} for name, record in sorted(self.validations.items())]
        return {
            "library_rows": len(rows),
            "library_error": library_error,
            "by_status": dict(sorted(by_status.items())),
            "candidates": int(by_status.get("candidate") or 0),
            "unverifiable": int(by_status.get("unverifiable") or 0),
            "retrievable": sum(1 for row in rows if str((row or {}).get("status") or "") in DEFAULT_RETRIEVAL_STATUSES),
            "promoted": int(by_status.get("promoted") or 0),
            "in_flight": sum(1 for record in queue if str(record.get("status") or "") in {"queued", "running", "requeued"}),
            "building": sum(1 for record in queue if self._decided_cases(record) > 0),
            "parked": [
                {"experience_id": record["experience_id"], "blocked_because": str(record.get("blocked_because") or "")}
                for record in queue
                if str(record.get("status") or "") in {"unverifiable", "failed", "dropped"}
            ],
            "rows": [
                {
                    "experience_id": record["experience_id"],
                    "status": str(record.get("status") or ""),
                    "cases_decided": self._decided_cases(record),
                    "measured_cases": int((record.get("summary") or {}).get("measured_cases") or 0),
                    "last_verdict": str(record.get("stopped_because") or record.get("error") or record.get("status") or ""),
                    "requeues": int(counters.get(record["experience_id"]) or 0),
                    "blocked_because": str(record.get("blocked_because") or ""),
                    "out": str(record.get("out") or ""),
                }
                for record in queue
            ],
        }

    def cost_report(self) -> dict[str, Any]:
        """What the waves spent, added up from each iteration's own round ledger."""

        totals = {"rounds": 0, "figure_generations": 0, "compile_calls": 0, "render_calls": 0, "judge_calls": 0, "stage_reruns": 0}
        elapsed = 0.0
        for paper in (self.state.get("papers") or {}).values():
            for item in paper.get("history") or []:
                cost = item.get("round_cost") if isinstance(item.get("round_cost"), dict) else {}
                if not cost:
                    continue
                totals["rounds"] += int(cost.get("rounds") or 0)
                for key in ("figure_generations", "compile_calls", "render_calls", "judge_calls", "stage_reruns"):
                    totals[key] += int((cost.get("totals") or {}).get(key) or 0)
                elapsed += float(cost.get("elapsed_seconds") or 0.0)
        return {
            "contract": "rfs.continuous_cost",
            "version": 1,
            **totals,
            "elapsed_seconds": round(elapsed, 3),
            "summary": (
                f"{totals['rounds']} round(s) across the waves: {totals['figure_generations']} generation call(s), "
                f"{totals['compile_calls']} compile(s), {totals['judge_calls']} judge answer(s), {totals['stage_reruns']} stage re-run(s)"
            ),
        }

    def paper_walk(self, name: str) -> dict[str, Any]:
        """How far down the layers a paper has walked, as that paper's own last ledger wrote it.

        Read from the ledger rather than recomputed off the live state, because the state keeps
        moving after the round it describes and a report that recomputed the walk would disagree
        with the per-iteration metric it is supposed to summarise. A paper that has not finished an
        iteration yet has no ledger, and for it the state is all there is.
        """

        entry = self.state["papers"].get(name) or {}
        paper_dir = self.out / "papers" / str(name)
        newest: Path | None = None
        for candidate in sorted(paper_dir.glob("iter_*")):
            if iteration_number(candidate.name) is None:
                continue
            if (candidate / LAYER_LEDGER_FILE).is_file():
                newest = candidate
        if newest is not None:
            ledger = _read_json_or_none(newest / LAYER_LEDGER_FILE)
            walk = ledger.get("walk") if isinstance(ledger, dict) else None
            if isinstance(walk, dict) and walk:
                return dict(walk)
        return layer_walk(entry.get("layers"), stage_order=STAGE_ORDER)

    def report(self) -> dict[str, Any]:
        papers = [
            {
                "name": spec["name"],
                "paper": spec["paper"],
                "domain": spec["domain"],
                "iteration": int(self.state["papers"][spec["name"]].get("iteration") or 0),
                "best": str(self.state["papers"][spec["name"]].get("best") or ""),
                "best_score": self.state["papers"][spec["name"]].get("best_score"),
                "paused": bool(self.state["papers"][spec["name"]].get("paused")),
                "archive_size": len(self.state["papers"][spec["name"]].get("archive") or []),
                "best_moved_at": int(self.state["papers"][spec["name"]].get("best_moved_at") or 0),
                # The direction state: which layer this paper is working on, what the model has
                # settled, and how long the figure has gone without moving forward.
                "work_face": str(self.state["papers"][spec["name"]].get("work_face") or ""),
                "stalled_streak": int(self.state["papers"][spec["name"]].get("stalled_streak") or 0),
                "layers": dict(self.state["papers"][spec["name"]].get("layers") or {}),
                # How far down the layers this paper has walked and what each layer cost: the
                # number the whole direction change is judged on, read from the ledger the last
                # round wrote rather than recomputed off the live state, which keeps moving after
                # the round it describes.
                "layer_walk": self.paper_walk(spec["name"]),
                "upstream": {
                    "requested": int(self.state["papers"][spec["name"]].get("upstream_requests") or 0),
                    "passed": int(self.state["papers"][spec["name"]].get("upstream_passed") or 0),
                },
                "claims": dict(self.state["papers"][spec["name"]].get("last_claims") or {}),
                "last_reuse_forecast": dict(self.state["papers"][spec["name"]].get("last_reuse_forecast") or {}),
                "reuse_declined": [
                    int(item.get("iteration") or 0)
                    for item in self.state["papers"][spec["name"]].get("history") or []
                    if (item.get("reuse") or {}).get("requested") and not (item.get("reuse") or {}).get("accepted")
                ],
                "iterations": list(self.state["papers"][spec["name"]].get("history") or []),
            }
            for spec in self.config["papers"]
        ]
        report = {
            "contract": "rfs.continuous_evolution_report",
            "version": 1,
            # A call that ran the iterations it was asked for succeeded: the sweep stops by
            # having finished, and reading that as a failure would mark every finished batch.
            "ok": not self.stop_reason
            or self.stop_reason in {"stop_requested", "wave_limit", "budget_exhausted", "rounds_complete"},
            "stop_reason": self.stop_reason,
            "stop_detail": self.stop_detail,
            "disk_floor": self.floor_record(),
            "waves": int(self.state.get("waves") or 0),
            "waves_this_call": int(self.waves_this_call),
            "elapsed_seconds": round(self.spent_seconds(), 3),
            "parallel": int(self.config["parallel"]),
            "iteration_rounds": int(self.config["iteration_rounds"]),
            "experience_library": self.library,
            "validation": {
                "enabled": bool(self.validation.get("enabled")),
                "suite": str(self.validation.get("suite") or ""),
                "workers": int(self.validation.get("workers") or 0),
                "reason": str(self.validation.get("reason") or ""),
            },
            "measurement": self.measurement_report(),
            "patch_yield": self.patch_yield_report(),
            "cost": self.cost_report(),
            "validations": [{**record, "experience_id": name} for name, record in sorted(self.validations.items())],
            "state": str(self.out / CONTINUOUS_STATE),
            "events": str(self.events_path),
            "papers": papers,
        }
        self.save_state()
        write_json(self.out / CONTINUOUS_REPORT, report)
        write_text(self.out / CONTINUOUS_MARKDOWN, _markdown(report))
        return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Continuous self-evolution",
        "",
        f"- Stopped: `{report['stop_reason'] or 'still running'}` {report['stop_detail']}",
        f"- Waves: {report['waves']} in {report['elapsed_seconds']}s, {report['parallel']} paper(s) at a time",
        f"- Experience library: `{report['experience_library']}`",
        f"- Cost: {report.get('cost', {}).get('summary', 'nothing spent yet')}",
        "",
        "| Paper | Iteration | Best score | Best version | Archived variants | Paused |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for paper in report["papers"]:
        lines.append(
            f"| {paper['name']} | {paper['iteration']} | {paper['best_score']} | `{paper['best']}` | "
            f"{paper.get('archive_size', 0)} | {'yes' if paper['paused'] else 'no'} |"
        )
    lines.append("")
    for paper in report["papers"]:
        lines.append(f"## {paper['name']}")
        for item in paper["iterations"]:
            reuse = item.get("reuse") or {}
            if not reuse.get("requested"):
                inherited = "started fresh"
            elif reuse.get("accepted"):
                inherited = "inherited the previous version"
            else:
                inherited = f"re-planned everything, the reuse was declined: {reuse.get('declined_reason') or 'no reason given'}"
            detail = item.get("error") or f"best={item.get('best_score')} fork_before={item.get('fork_before') or '-'} injected={len(item.get('experiences_injected') or [])} ({inherited})"
            lines.append(f"- iter {item.get('iteration')}: {item.get('status')} — {detail}")
            restart = item.get("restart") or {}
            if restart:
                lines.append(
                    f"  - started from the archive: iteration {restart['iter']:02d} round {restart['round']:02d} "
                    f"candidate {restart['candidate']:02d} (score {restart['score']}), not the published version"
                )
            forecast = item.get("reuse_forecast") or {}
            if forecast.get("requested") and (forecast.get("replanned_stages") or not forecast.get("accepted")):
                lines.append(f"  - before the run: {_forecast_sentence(forecast)}")
        lines.append("")
    lines.append("## Layers")
    lines.append("| Paper | Work face | Settled | Rounds to settle all | Reopens | Upstream asked / allowed | Sentences checked |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for paper in report["papers"]:
        walk = paper.get("layer_walk") or {}
        upstream = paper.get("upstream") or {}
        claims = paper.get("claims") or {}
        claims_text = f"{claims.get('rate')}" if claims else "-"
        lines.append(
            f"| {paper['name']} | {paper.get('work_face') or '-'} | {walk.get('settled', 0)}/{walk.get('total', 0)} | "
            f"{walk.get('rounds_to_settle_all') if walk.get('rounds_to_settle_all') is not None else '-'} | "
            f"{walk.get('reopen_events', 0)} | {upstream.get('requested', 0)} / {upstream.get('passed', 0)} | {claims_text} |"
        )
    for paper in report["papers"]:
        walk = paper.get("layer_walk") or {}
        for stage, row in (walk.get("layers") or {}).items():
            lines.append(
                f"- {paper['name']} · {stage}: first moved at {row.get('first_move_at') or '-'}, settled at "
                f"{row.get('settled_at') or '-'} ({row.get('settled_by') or 'nobody'}), cost "
                f"{row.get('rounds_to_settle') if row.get('rounds_to_settle') is not None else '-'} round(s), reopened {row.get('reopen_count') or 0}x"
            )
    patch_yield = report.get("patch_yield") or {}
    if patch_yield:
        lines.append("")
        lines.append("## Prompt patches")
        if patch_yield.get("library_error"):
            lines.append(f"- The patch library could not be read: {patch_yield['library_error']}")
        else:
            lines.append(
                f"- {patch_yield.get('patches')} row(s), {patch_yield.get('distinct_sentences')} distinct sentence(s), "
                f"novelty {patch_yield.get('novelty')}, asked for {patch_yield.get('asked')} time(s) "
                f"({patch_yield.get('yield_per_round')} per round), licensed {patch_yield.get('licensed')}"
            )
    lines.append("")
    measurement = report.get("measurement") or {}
    if measurement:
        lines.append("## Experience measurement")
        lines.append(
            f"- Library: {measurement.get('library_rows')} row(s); "
            f"retrievable {measurement.get('retrievable')}, promoted {measurement.get('promoted')}, "
            f"candidates {measurement.get('candidates')}, retired as unverifiable {measurement.get('unverifiable')}"
        )
        lines.append(f"- In flight: {measurement.get('in_flight')}; sets already part-measured: {measurement.get('building')}")
        if measurement.get("library_error"):
            lines.append(f"- The library could not be read: {measurement['library_error']}")
        lines.append("")
        lines.append("| Experience | Status | Cases decided | Measured | Last verdict | Requeues | Blocked because |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in measurement.get("rows") or []:
            lines.append(
                f"| `{row['experience_id']}` | {row['status']} | {row['cases_decided']} | {row['measured_cases']} | "
                f"{row['last_verdict']} | {row['requeues']} | {row['blocked_because']} |"
            )
        lines.append("")
    lines.append("## Validations")
    if not report["validations"]:
        lines.append("- none queued yet")
    for item in report["validations"]:
        lines.append(f"- `{item['experience_id']}`: {item.get('status')} {item.get('stopped_because') or ''}".rstrip())
    lines.append("")
    return "\n".join(lines)


def review_localised(result: Any) -> bool:
    """Whether a boundary review named where the fault entered, with a number behind it.

    The loop reads one thing out of the answer: the stage whose own requirement went unwritten. A
    stage with at least one cited reading is that answer; a stage on its own, or a reading on its
    own, is half of it, and the half is what the second call is for. The sentence about the
    requirement is *not* required here: the round's own gate keeps a claim that names a stage and a
    readable reading even when that sentence was left unsaid, and holding the boundary to a
    stricter contract than the round is measured by would throw away a location the coach did give.
    """

    record = result if isinstance(result, dict) else {}
    if not str(record.get("fault_stage") or "").strip():
        return False
    for item in record.get("fault_evidence") or []:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        return True
    return False


def review_complained(result: Any) -> bool:
    """Whether the seat said something was wrong, even when it did not say where.

    This is what separates an answer that named a defect without localising it -- which has to be
    asked for again -- from one that read the figure and found nothing to repair, which has already
    answered the question the round asked.
    """

    record = result if isinstance(result, dict) else {}
    if record.get("stage_feedback") or record.get("blocking_issues"):
        return True
    return str(record.get("status") or "") not in {"", "pass"}


def _default_cross_round_comparer(config: dict[str, Any]) -> Callable[[Path, Path, dict[str, Any]], dict[str, Any]] | None:
    """Build the advisory seat's cross-round comparison, or nothing when no seat is configured.

    The question is the one the advisory seat is already asked about a pair: which of the two a
    reader of this paper is better served by. It is asked here about two rounds rather than about
    two candidates of one round -- image A is the best version the paper had reached, image B is
    this round's -- and the answer is filed as evidence for the ledger and the trend. A round is
    never accepted, rolled back or moved by this call.
    """

    options = dict(config.get("evolve") or {})
    mode = str(options.get("judge_mode") or "heuristic")
    if mode in {"", "heuristic", "off"}:
        return None
    advisory, _frozen = build_direct_judges(
        mode="bootstrap" if mode == "vlm" else mode,
        online_model=options.get("online_judge_model") or options.get("judge_model"),
        timeout=int(options.get("vlm_timeout") or 180),
        retries=int(options.get("vlm_retries") or 1),
        # The same seat that draws the boundary, so the comparison and the review cannot be two
        # different models answering about the same figure.
        advisory_checkpoint=str(options.get("online_judge_checkpoint") or ""),
        advisory_base_model=str(options.get("online_judge_base_model") or ""),
    )
    if advisory is None:
        return None

    def compare(current: Path, history: Path, spec: dict[str, Any]) -> dict[str, Any]:
        """Ask the seat which of the two versions a reader of this paper is better served by."""

        domain = str(spec.get("domain") or "")
        scope = {"scope": "cross_round_comparison"}
        after = kept_figure_judge_input(current, domain=domain, allowed_scope=scope)
        before = kept_figure_judge_input(history, domain=domain, allowed_scope=scope)
        if after is None or before is None:
            return {}
        try:
            result = advisory.evaluate(replace(after, comparison_preview=str(before.preview or "")))
        except Exception:  # noqa: BLE001 - a comparison that fails must not cost the round
            return {}
        return dict(result or {})

    return compare


def _default_review_runner(config: dict[str, Any]) -> Callable[[Path, dict[str, Any]], dict[str, Any]]:
    """Build the advisory-only boundary reviewer the config asks for.

    The reviewer is deliberately built once and reused: it answers the same question about a
    figure the run has just finished, from the same prompt, and never scores anything.
    """

    options = dict(config.get("evolve") or {})
    mode = str(options.get("judge_mode") or "heuristic")
    advisory: Any = None
    if mode not in {"", "heuristic", "off"}:
        advisory, _frozen = build_direct_judges(
            mode="bootstrap" if mode == "vlm" else mode,
            online_model=options.get("online_judge_model") or options.get("judge_model"),
            timeout=int(options.get("vlm_timeout") or 180),
            retries=int(options.get("vlm_retries") or 1),
            # The reviewer that decides which stage an iteration inherits from is the advisory
            # seat, so it reads the same checkpoint the iterations do; otherwise the boundary
            # would be drawn by one model and the round decided by another.
            advisory_checkpoint=str(options.get("online_judge_checkpoint") or ""),
            advisory_base_model=str(options.get("online_judge_base_model") or ""),
        )

    model = str(getattr(advisory, "model", "") or "")

    def review(run_dir: Path, spec: dict[str, Any]) -> dict[str, Any]:
        """Ask the advisory seat where the fault in the inherited figure entered.

        The answer is read for one thing: the stage whose own requirement went unwritten. A seat
        that did not say it leaves the round with nothing to aim at, and the three ways that
        happens are kept apart here because the round answers them differently -- and because "the
        seat was not there" must never read as "the seat found no fault". A seat with no weights, an
        answer that could not be parsed and a call that raised all arrived as an empty list of
        repairs, which the fork read as a figure nobody had complained about and answered by
        re-cutting the structure: an hour of drawing spent on a crash.
        """

        retries = 0

        def ask(instruction: str) -> tuple[dict[str, Any] | None, str]:
            """One call, asked again once when the seat itself failed. ``(answer, problem)``."""

            nonlocal retries
            while True:
                try:
                    result = advisory.evaluate(judge_input, extra_instruction=instruction)
                except Exception as exc:  # noqa: BLE001 - a failed review must not stop the loop
                    problem = f"review_failed: {type(exc).__name__}: {exc}"
                    kind = SEAT_RAISED
                else:
                    kind = str((result or {}).get("advisory_unavailable_kind") or "")
                    problem = str((result or {}).get("advisory_unavailable") or "")
                    if result and not problem and not kind:
                        return result, ""
                if retries >= DEFAULT_REVIEW_RETRY_LIMIT or kind == SEAT_NO_BACKEND:
                    return None, problem or "the advisory seat did not answer"
                retries += 1

        def answer(result: dict[str, Any] | None, exit_reason: str, reason: str, *, localisation_retry: bool = False) -> dict[str, Any]:
            return {
                "ok": exit_reason in {"", REVIEW_CLEAN},
                "reason": reason,
                "stage_feedback": list((result or {}).get("stage_feedback") or []),
                "judgement": dict(result or {}),
                "exit": exit_reason,
                "retries": retries,
                "retried": bool(retries),
                "localisation_retry": bool(localisation_retry),
                "model": model,
            }

        if advisory is None:
            return answer(None, REVIEW_ERROR, "no_advisory_judge")
        judge_input = kept_figure_judge_input(
            run_dir,
            domain=str(spec.get("domain") or ""),
            allowed_scope={"scope": "continuous_boundary_review"},
            progress_sheet=spec.get("progress_sheet") if isinstance(spec.get("progress_sheet"), dict) else None,
        )
        if judge_input is None:
            return answer(None, REVIEW_ERROR, "no_kept_figure")
        result, problem = ask(str(spec.get("review_instruction") or ""))
        if result is None:
            return answer(None, REVIEW_ERROR, problem)
        localisation_retry = False
        if review_localised(result):
            return answer(result, "", "")
        if not review_complained(result):
            # The seat read the figure and said nothing was wrong with it. Re-asking for a location
            # here would be asking the seat to invent one, so this answer is taken as it stands and
            # kept apart from the two silences above.
            return answer(result, REVIEW_CLEAN, "")
        # The seat named a defect without saying where it entered, which is the half the round
        # needs. It is asked once more with the three fields spelled out, and a second answer that
        # still says nothing is recorded as undefined rather than filled in for the model.
        localisation_retry = True
        second, second_problem = ask(ADVISORY_LOCALISATION_RETRY)
        if second is not None:
            result = second
        else:
            problem = second_problem
        if review_localised(result):
            return answer(result, "", "", localisation_retry=localisation_retry)
        return answer(result, REVIEW_UNDEFINED, problem or "the review named no stage for its fault", localisation_retry=localisation_retry)

    return review


def run_continuous_evolution(
    config: str | Path | dict[str, Any],
    out: str | Path,
    *,
    experience_library: str | Path = "",
    stop_file: str | Path = "",
    evolve_runner: Callable[..., dict[str, Any]] | None = None,
    review_runner: Callable[[Path, dict[str, Any]], dict[str, Any]] | None = None,
    clock: Callable[[], float] = time.time,
    disk_usage: Callable[[Path], Any] | None = None,
    max_waves: int = 0,
    validate_runner: Callable[..., dict[str, Any]] | None = None,
    cross_round_runner: Callable[[Path, Path, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the continuous loop: fixed base papers, each iteration starting from the last one.

    ``stop_file`` appears -> the wave in flight finishes and the loop stops. The wall clock and
    the disk floor stop it the same way. ``max_waves`` bounds the loop for a caller that wants one
    measured wave; zero means no bound, and the loop only ends when something above ends it.
    """

    normalised = read_continuous_config(config)
    root = Path(out)
    runner = evolve_runner or run_paper_evolve_editable
    reviewer = review_runner or _default_review_runner(normalised)
    probe = disk_usage or _disk_free
    loop = _Loop(
        config=normalised,
        out=root,
        library=str(experience_library or ""),
        stop_file=str(stop_file or normalised.get("stop_file") or ""),
        evolve_runner=runner,
        review_runner=reviewer,
        clock=clock,
        disk_usage=probe,
        max_waves=max_waves,
        validate_runner=validate_runner or run_direct_experience_validation,
        cross_round_runner=cross_round_runner if cross_round_runner is not None else _default_cross_round_comparer(normalised),
    )
    return loop.run()


__all__ = [
    "BOUNDARY_REVIEW_CONTRACT",
    "BOUNDARY_REVIEW_FILE",
    "BOUNDARY_REVIEW_QUESTION",
    "CONTINUOUS_EVENTS",
    "CONTINUOUS_KEYS",
    "CONTINUOUS_MARKDOWN",
    "CONTINUOUS_REPORT",
    "CONTINUOUS_STATE",
    "ContinuousError",
    "DEFAULT_DISK_FLOOR_GB",
    "DEFAULT_FORK_STAGE",
    "DEFAULT_REOPEN_AFTER_STALLED_ROUNDS",
    "DEFAULT_REVIEW_RETRY_LIMIT",
    "INCREMENTAL_FORK_STAGE",
    "CROSS_ROUND_CHOICES",
    "CROSS_ROUND_CONTRACT",
    "CROSS_ROUND_FILE",
    "DEFAULT_CROSS_ROUND_EVERY",
    "LAYER_LEDGER_CONTRACT",
    "LAYER_LEDGER_FILE",
    "PROGRESS_SHEET_CONTRACT",
    "PROGRESS_SHEET_FILE",
    "PROGRESS_SHEET_GOAL",
    "PROGRESS_SHEET_LATEST",
    "REVIEW_CLEAN",
    "REVIEW_ERROR",
    "REVIEW_UNDEFINED",
    "SILENT_REVIEW_FORK_STAGE",
    "attribution_fork_stage",
    "fork_from_review",
    "review_complained",
    "review_localised",
    "opening_move_from_feedback",
    "requested_actions",
    "DEFAULT_ITERATION_ROUNDS",
    "DEFAULT_PARALLEL",
    "DEFAULT_VALIDATION_WORKERS",
    "DISK_FLOOR",
    "MAX_CONSECUTIVE_FAILURES",
    "STAGE_ORDER",
    "UPSTREAM_EVIDENCE_RETRY",
    "deadlock_hint",
    "fork_before_from_feedback",
    "holds_kept_figure",
    "is_upstream",
    "iteration_defects",
    "iteration_number",
    "iteration_reflections",
    "iteration_catalogue_gap",
    "iteration_refusals",
    "iteration_cross_rounds",
    "iteration_layer_settled",
    "iteration_layer_events",
    "layer_answer_history",
    "layer_plan",
    "layer_reopen_events",
    "moved_iterations",
    "ROUND_METRICS_FILE",
    "ROUND_METRICS_LEDGER",
    "layer_ledger",
    "acceptance_directions",
    "cross_round_baseline",
    "cross_round_due",
    "cross_round_record",
    "previous_round_summary",
    "older_layer_readings",
    "progress_ask",
    "progress_history",
    "progress_sheet",
    "stage_gate_rows",
    "stage_moves",
    "load_continuous_state",
    "newest_kept_figure_dir",
    "stage_index",
    "upstream_allowed",
    "upstream_case",
    "work_face",
    "read_continuous_config",
    "requested_action_types",
    "run_continuous_evolution",
]
