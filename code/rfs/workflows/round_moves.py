"""The moves one round runs: the mechanism's policy first, then the model's suggestion.

A round used to run exactly one action. The mechanism's own policy proposal and the
coach's suggestion were two entries in one ordered list, so whichever came first ate the
whole round: a round that had a decomposition or colour move never got to run the repair
the coach asked for, and the pairing "this suggestion -> how it turned out" was therefore
never collected. This module keeps a round's decision to a *list* of moves instead. The
mechanism move runs first because it is the cheap one, the model's suggestion runs second
on top of it, and each is measured. Everything here is pure: it decides what a round may
run and never runs it.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ..evolution.direct import action_execution_fingerprint

MOVE_CONTRACT = "rfs.round_moves"
MOVE_VERSION = 1

#: The keys an action names the objects it touches under. Kept in one place so that "the
#: two moves touch the same object" is one definition rather than three spellings of it.
OBJECT_ID_KEYS = ("object_ids", "semantic_node_ids", "slot_ids", "node_ids", "edge_ids")

#: The one mechanism source that is the run's always-on repair: the decomposability policy
#: moves the figure whether or not anything was learned, so a round it led is the shipped
#: default rather than a finding. A colour re-plan is a demand the round's own measured review
#: made, so it is not this, and a round it led keeps its own credit.
MECHANISM_DEFAULT_SOURCE = "decomposition_policy"

#: The actions that rewrite the whole plan rather than nudge an object inside it. A second
#: move whose object bindings were chosen against the old plan cannot survive one of these,
#: so a plan rewrite and an object-level edit are not run in the same round.
PLAN_REWRITE_ACTIONS = frozenset({"reanalyze_semantics", "replan_structure", "resegment"})


def move_source(action: dict[str, Any] | None, *, mechanism_sources: frozenset[str] | set[str]) -> str:
    """Name who proposed a move: its own ``source`` tag, else ``mechanism``/``model``.

    An action carries a ``source`` when the loop's own policy proposed it. A member of
    ``mechanism_sources`` is the mechanism; anything else that has a type is the model.
    """

    source = action if isinstance(action, dict) else {}
    tag = str(source.get("source") or "")
    if tag:
        return tag
    return "model" if str(source.get("type") or "") else ""


def is_mechanism_move(action: dict[str, Any] | None, *, mechanism_sources: frozenset[str] | set[str]) -> bool:
    """Whether the mechanism's own policy proposed this move rather than the coach."""

    return bool(action) and str((action or {}).get("source") or "") in {str(item) for item in mechanism_sources}


def move_target_ids(action: dict[str, Any] | None) -> set[str]:
    """The objects a move names as its targets, in whichever key its action type uses."""

    source = action if isinstance(action, dict) else {}
    targets: set[str] = set()
    for key in OBJECT_ID_KEYS:
        value = source.get(key)
        if isinstance(value, (list, tuple, set)):
            targets |= {str(item) for item in value if str(item)}
        elif isinstance(value, str) and value:
            targets.add(value)
    return targets


def _identical(one: dict[str, Any], other: dict[str, Any]) -> bool:
    return action_execution_fingerprint(one) == action_execution_fingerprint(other)


def plan_round_moves(
    ordered: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    chosen: dict[str, Any] | None,
    *,
    mechanism_sources: frozenset[str] | set[str],
    limit: int = 2,
) -> list[dict[str, Any]]:
    """Return the moves this round runs, the mechanism's first and the model's second.

    ``chosen`` is the action the loop already decided to run. ``ordered`` is the same ordered
    proposal list it was chosen from, so the move the choice displaced is recoverable without
    asking the controller a second time.

    A round runs both halves whenever both were proposed: the mechanism's own move, which is the
    cheap one and runs first, and the seat's suggestion, which runs on top of it and is measured
    with it. Which of the two the loop *picked* is not what decides this -- when the selection
    preferred the seat's executable suggestion over the mechanism's fallback, the round still
    runs the mechanism move first, because the pairing "this suggestion -> how it turned out" is
    the datum the loop is here to collect and a round that runs only one half never collects it.
    Each half is still checked for conflicts by the caller; a pair that cannot both run is
    reported rather than silently halved.

    A soft mechanism fallback that is *not* in the proposal list is still not invented: only a
    move the caller proposed can be run.
    """

    moves: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(candidate: dict[str, Any] | None) -> None:
        spec = dict(candidate) if isinstance(candidate, dict) else {}
        if not str(spec.get("type") or ""):
            return
        fingerprint = action_execution_fingerprint(spec)
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        moves.append(spec)

    _add(chosen)
    budget = max(1, int(limit))
    # Fill whichever half is missing, mechanism first. The loop over the two halves is what makes
    # "both moves ran" independent of which one the selector preferred: a round whose chosen move
    # was the seat's suggestion used to drop the mechanism's proposal on the floor, so the two
    # halves were collected in one order and not the other.
    for want_mechanism in (True, False):
        if len(moves) >= budget:
            break
        if any(is_mechanism_move(item, mechanism_sources=mechanism_sources) is want_mechanism for item in moves):
            continue
        for candidate in ordered or ():
            spec = dict(candidate) if isinstance(candidate, dict) else {}
            if not str(spec.get("type") or ""):
                continue
            if is_mechanism_move(spec, mechanism_sources=mechanism_sources) is not want_mechanism:
                continue
            _add(spec)
            break
    moves.sort(key=lambda item: 0 if is_mechanism_move(item, mechanism_sources=mechanism_sources) else 1)
    return moves[:budget]


def move_conflict_reason(primary: dict[str, Any] | None, secondary: dict[str, Any] | None) -> str:
    """Why the second move may not run after the first, or ``""`` when it may.

    Two rules, both deterministic and both readable off the actions alone:

    * ``duplicate_move`` -- the same action under the same budget is one move, not two.
    * ``plan_rewritten`` -- the first move re-plans the figure, so the object ids the
      second move was chosen against no longer describe what is there. Running it would
      edit whatever took the id's place. A plan rewrite always wins and the second move
      waits; it is not rerouted onto the new plan, because the plan is what it objected to.

    A contradiction that is only visible in the rendered figure (one move enlarges the box
    the other shrinks) is not guessed at here: the second move runs and its own measurement
    is what says whether the pair held together.
    """

    first = primary if isinstance(primary, dict) else {}
    second = secondary if isinstance(secondary, dict) else {}
    if not first.get("type") or not second.get("type"):
        return ""
    if _identical(first, second):
        return "duplicate_move"
    if str(first.get("type") or "") in PLAN_REWRITE_ACTIONS:
        return "plan_rewritten"
    if str(second.get("type") or "") in PLAN_REWRITE_ACTIONS and move_target_ids(first) & move_target_ids(second):
        return "plan_rewritten"
    return ""


def move_unexecutable_reason(action: dict[str, Any] | None, state: dict[str, Any] | None) -> str:
    """Why a move has nothing left to act on in ``state``, or ``""`` when it still does.

    An earlier move may have deleted the very object this one was chosen against -- a
    ``resegment`` that dropped a panel, a repair that removed a node. The move is recorded
    as unexecutable rather than silently dropped, because "the round ran two moves" is a
    claim the run has to be able to check.
    """

    targets = move_target_ids(action)
    if not targets:
        return ""
    source = state if isinstance(state, dict) else {}
    layout = source.get("layout") if isinstance(source.get("layout"), dict) else {}
    icons = source.get("icons") if isinstance(source.get("icons"), dict) else {}
    known = {str(item.get("id")) for item in layout.get("nodes", []) if isinstance(item, dict) and item.get("id")}
    known |= {str(item.get("id")) for item in layout.get("panels", []) if isinstance(item, dict) and item.get("id")}
    known |= {str(item.get("id")) for item in layout.get("connectors", []) if isinstance(item, dict) and item.get("id")}
    known |= {str(item.get("asset_id")) for item in (icons or {}).get("icons", []) if isinstance(item, dict) and item.get("asset_id")}
    known |= {value.removeprefix("slot_") for value in known}
    missing = {item for item in targets if item not in known}
    if not missing:
        return ""
    return "target_missing_after_earlier_move"


def guard_reason(
    *,
    stop_file: str | Path | None = None,
    disk_path: str | Path | None = None,
    disk_floor_gb: float = 0.0,
    disk_floor_ratio: float = 0.0,
    budget_deadline: float = 0.0,
    clock: Callable[[], float] = time.time,
) -> str:
    """Why the round must stop between its moves, or ``""`` when it may carry on.

    The loop between rounds is guarded by the caller; inside a round there was nothing to
    ask. A round that is about to spend a second move asks here: a stop the operator left
    while the first move was running, a budget the round would overrun, or a disk that
    dropped under the floor. The floor is read the two ways the loop reads it -- an
    absolute number of free gigabytes, or a share of the volume -- so a caller that only
    states one of them cannot leave the guard quietly unarmed.
    """

    if stop_file and Path(str(stop_file)).exists():
        return "stop_requested_between_moves"
    deadline = float(budget_deadline or 0.0)
    if deadline and float(clock()) >= deadline:
        return "budget_spent_between_moves"
    floor_gb = float(disk_floor_gb or 0.0)
    floor_ratio = float(disk_floor_ratio or 0.0)
    if disk_path and (floor_gb > 0 or floor_ratio > 0):
        try:
            usage = shutil.disk_usage(str(disk_path))
        except OSError:
            return ""
        if floor_gb > 0:
            # An absolute floor wins over a share, exactly as the loop's own
            # ``below_disk_floor`` reads them: a run that stated 40 GB did not also mean
            # ten percent of a volume that happens to be larger than 400 GB.
            if usage.free / (1024 ** 3) < floor_gb:
                return "disk_floor_between_moves"
        elif floor_ratio > 0 and usage.total and usage.free / usage.total < floor_ratio:
            return "disk_floor_between_moves"
    return ""


def round_moves_record(round_index: int, moves: list[dict[str, Any]] | None, *, source: str = "") -> dict[str, Any]:
    """The round's move list as it lands in ``round_moves.json``."""

    entries = [dict(item) for item in (moves or []) if isinstance(item, dict)]
    return {
        "contract": MOVE_CONTRACT,
        "version": MOVE_VERSION,
        "round": int(round_index),
        "count": len(entries),
        "mechanism_move": next((str(item.get("source") or "") for item in entries if item.get("is_mechanism")), ""),
        "order": [str(i) for i in range(len(entries))],
        "moves": entries,
        "source": str(source or ""),
    }
