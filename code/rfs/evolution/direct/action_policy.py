"""What layer a repair works at, what that layer costs, and which voice asked for it.

One round can answer a defect in three places, and they cost wildly different amounts. Editing the
figure program is deterministic and calls no model. A compile-time rule calls no model either; it
is read while the figure is compiled. Re-running a planning stage is the expensive one: it asks a
model to do the work again and it takes minutes to hours.

This table used to be a gate. A fault the coach placed at the compiled figure could only be answered
by the cheap layers, and the expensive layer had to be earned -- by an attribution that pointed
upstream, by two rounds in a row that moved nothing, or by one of the loop's own readings. It is not
a gate any more. The layer a repair works at is a fact about the repair, and whether a re-plan is
worth its minutes is the seat's decision rather than the mechanism's: the prices below travel to the
seat, which names the layer it chose and says why. Nothing here refuses a move, and the one hard
line against spending is the budget and the clock the round is already guarded by between its moves.

What the coach's attribution and the loop's own measurements said is still recorded, because "who
asked for the expensive layer" is what makes a round readable afterwards. The loop takes two
readings of its own: the decomposability report, which measures that the figure cannot be read in
pieces however its boxes are moved, and the content coverage report, which measures that the figure
carries less of the paper than the paper's own budget asked for. Either one is evidence that a fault
is upstream of the figure, and a measurement the loop took is not overruled by the coach's
attribution of some other defect. The coverage report's two readings say different things and are
kept apart: a paper that yielded four modules and a figure that drew two of them are not the same
finding -- one is answered by drawing what was already read, the other only by reading again.
"""

from __future__ import annotations

from typing import Any

#: Editing the figure program: deterministic, no model call, seconds.
FIGURE_PROGRAM_CLASS = "figure_program"
#: A rule the compiler reads: no model call, one compile.
COMPILE_RULE_CLASS = "compile_rule"
#: Re-running a planning stage: a generation call, minutes to hours.
STAGE_RERUN_CLASS = "stage_rerun"

#: The layers, cheapest first.
CLASS_ORDER = (FIGURE_PROGRAM_CLASS, COMPILE_RULE_CLASS, STAGE_RERUN_CLASS)

#: Which bounded action works at which layer. ``STAGE_RERUN_CLASS`` is exactly the set
#: ``upstream_retry`` reads as "a local repair cannot clear this", so the gate here and the
#: re-run there cannot disagree about which actions are expensive.
ACTION_CLASSES = {
    FIGURE_PROGRAM_CLASS: ("relayout_global", "relayout_slots", "repair_interfaces", "reroute", "resize_nodes"),
    COMPILE_RULE_CLASS: ("layout_constraint", "router_constraint"),
    STAGE_RERUN_CLASS: ("reanalyze_semantics", "replan_structure", "replan_theme", "resegment"),
}

_ACTION_CLASS_OF = {name: klass for klass, names in ACTION_CLASSES.items() for name in names}

#: The layers whose work is a change to the drawing itself: the figure program and the compile
#: rule. A move at either one edits a sentence or a number the compiler reads, so the paid
#: icons are still the figure's answer and must not be re-bought. Only a stage re-run, or an
#: action that explicitly names assets, may ask for art again.
CHEAP_LAYERS = (FIGURE_PROGRAM_CLASS, COMPILE_RULE_CLASS)

#: What each layer spends, in the words the seat that picks a repair is shown. The price is the
#: point of this table now that the layered refusal is gone: a seat told what a stage re-run costs
#: can weigh it against a nudge, and the round record can be read back as the price it paid.
LAYER_COSTS = {
    FIGURE_PROGRAM_CLASS: (
        "edits the figure program: seconds, no model call, and the icons already paid for are kept"
    ),
    COMPILE_RULE_CLASS: (
        "adds a rule the compiler reads: one compile, no model call, and the icons already paid for are kept"
    ),
    STAGE_RERUN_CLASS: (
        "re-runs a planning stage: a model call, minutes to hours, and the icons may be drawn again"
    ),
}

#: How many rounds in a row may move nothing before the loop reads the fault as one upstream of
#: the figure. It names a reading that travels with the round, not a door it opens.
STALLED_ROUNDS_BEFORE_ESCALATION = 2

#: The coverage readings that unlock something, and what each of them unlocks. The figure drawing
#: less than the paper's budget allows a stage re-run -- the content plan or the structure that
#: dropped the modules can be redone. A paper budget that is itself thin unlocks the reading and
#: nothing else, because the answer is to read the paper again rather than re-plan on thin input.
FIGURE_BELOW_PAPER_BUDGET = "figure_below_paper_budget"
PAPER_BUDGET_THIN = "paper_budget_thin"

#: The ``source`` a proposal carries when the loop's own measurement asked for it rather than the
#: coach. A move asked for by a measurement is not deferred by an attribution of something else.
DECOMPOSITION_POLICY_SOURCE = "decomposition_policy"

#: The one action that moves a round's *starting point* rather than the figure: a later round may
#: begin from a variant the loop kept in its archive instead of from the version it published.
#: It is deliberately not a repair the Judge can name and not an action a Memory row may carry --
#: it changes where a round starts, not what the figure becomes -- so it is a name the loop
#: records rather than a move the contract offers.
RESTART_FROM_ARCHIVE = "restart_from_archive"


def action_class(action_type: Any) -> str:
    """Which layer this action works at, or ``""`` when the loop does not run that name."""

    return _ACTION_CLASS_OF.get(str(action_type or "").strip(), "")


def action_reuses_assets(action_type: Any) -> bool:
    """Whether a move at this layer must leave the already-paid icon assets alone.

    True only for a name the loop classifies at the figure-program or compile-rule layer. An
    unknown name is not forced to reuse: the loop does not know what it does, and the guard is
    a shield for the moves it does understand rather than a guess about the rest.
    """

    return action_class(action_type) in CHEAP_LAYERS


def measurement_request(action: Any) -> bool:
    """Whether the loop's own measurement asked for this move, rather than the coach.

    The decomposability report proposes a stage re-run when it measures that the figure cannot be
    read in pieces. That proposal is a reading of the figure, not an attribution of a defect to
    the compiled layer, and an attribution of something else must not defer it -- which is what
    happened while the two were carried the same way: every round measured the break, proposed the
    structure re-run, and had it refused on the coach's ``ppt_compilation`` answer.
    """

    source = action if isinstance(action, dict) else {}
    return str(source.get("source") or "").strip() == DECOMPOSITION_POLICY_SOURCE


def content_gap_signal(report: dict[str, Any] | None) -> str:
    """Which reading of one coverage report unlocks something in the next round, or none.

    The two readings are ordered rather than joined, because they are not the same finding. A
    figure that drew less than the paper declared is the one a re-plan can act on, so it is read
    first; a source that yielded little on its own only says to read it again. A report that was
    never measured unlocks nothing at all -- ``not_measured`` is the absence of a reading, not a
    clean bill of health.
    """

    advisories = (report or {}).get("advisories") if isinstance(report, dict) else None
    codes = [str(item.get("code") or "") for item in advisories or [] if isinstance(item, dict)]
    if FIGURE_BELOW_PAPER_BUDGET in codes:
        return FIGURE_BELOW_PAPER_BUDGET
    if PAPER_BUDGET_THIN in codes:
        return PAPER_BUDGET_THIN
    return ""


def action_class_policy(
    *,
    coach_fault_stage: Any = "",
    stalled_rounds: int = 0,
    measured_escalation: bool = False,
    content_gap: Any = "",
) -> dict[str, Any]:
    """What each layer of a repair costs, and which of the round's voices asked for the costly one.

    Nothing here refuses a move. ``allowed`` names every layer because every layer is open: what
    bounds a round's spending is its budget and its clock, not this table. ``costs`` states what
    each layer spends, so the seat that picks a repair can weigh a stage re-run against a nudge and
    say why the one it named is worth the price. ``signal`` names the one voice that asked for the
    expensive layer and ``measured`` says whether it was a reading the loop took rather than the
    coach's attribution, so a round that spent an hour re-planning reads back as one somebody asked
    an hour for. ``stalled_rounds`` is recorded whether or not it fired, because "the last repair
    changed nothing" is worth knowing about the round either way.
    """

    stage = str(coach_fault_stage or "").strip()
    stalled = max(0, int(stalled_rounds))
    gap = str(content_gap or "").strip()
    voices: list[str] = []
    if stage:
        voices.append(f"the coach placed the fault at {stage}")
    signal = ""
    measured = False
    if stalled >= STALLED_ROUNDS_BEFORE_ESCALATION:
        voices.append(f"{stalled} rounds in a row moved nothing")
        signal, measured = "stalled_rounds", False
    if measured_escalation:
        voices.append("the figure measured as one that cannot be read in pieces")
        signal, measured = DECOMPOSITION_POLICY_SOURCE, True
    if gap == FIGURE_BELOW_PAPER_BUDGET:
        voices.append("the figure drew less than the paper's own budget asked for")
        signal, measured = FIGURE_BELOW_PAPER_BUDGET, True
    elif gap == PAPER_BUDGET_THIN:
        voices.append("the paper itself yielded little, so it may have to be read again")
        signal, measured = PAPER_BUDGET_THIN, True
    return {
        "allowed": list(CLASS_ORDER),
        "costs": dict(LAYER_COSTS),
        "reason": "; ".join(voices) or "no voice named a layer",
        "escalated": bool(signal),
        "signal": signal,
        "measured": measured,
        "fault_stage": stage,
        "stalled_rounds": stalled,
    }


__all__ = [
    "ACTION_CLASSES",
    "CHEAP_LAYERS",
    "CLASS_ORDER",
    "action_reuses_assets",
    "COMPILE_RULE_CLASS",
    "DECOMPOSITION_POLICY_SOURCE",
    "FIGURE_BELOW_PAPER_BUDGET",
    "FIGURE_PROGRAM_CLASS",
    "LAYER_COSTS",
    "PAPER_BUDGET_THIN",
    "RESTART_FROM_ARCHIVE",
    "STAGE_RERUN_CLASS",
    "STALLED_ROUNDS_BEFORE_ESCALATION",
    "action_class",
    "action_class_policy",
    "content_gap_signal",
    "measurement_request",
]
