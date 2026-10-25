"""Which stage a figure defect entered at, and which requirement went unwritten there.

A defect the seat reports is worth something only when it says where the defect came from, and
the answer has to be actionable at the place it names: "the layout is bad" is not a finding,
"the requirement that the main flow be drawn large in the middle was not written for the
structure planner" is. This module is that mapping, written once so the prompt that asks the
question, the gate that checks the answer and the ledger that scores it cannot disagree about
what the possible answers are.

Two decisions are baked into the table rather than left to the seat.

*Location prefers the latest, cheapest stage.* A tangle of connectors is first a routing
defect: the drawing stage can move a line without re-planning anything, so that is what the
table names. Only when the crossings are a symptom of boxes that sit in the wrong panels does
the answer rise to the structure planner. The same rule sends mismatched colours to the palette
planner instead of back through the paper.

*Cost is stated, not implied.* Every row carries what re-running that stage actually costs, in
the same three words the prompt uses, so a seat that can answer cheaply says so and a seat that
names the expensive stage has said something the round can weigh.

The last row is deliberately a dead end -- the icon generator is not where the loop's effort
goes, because the image model behind it is cheap and its reach is limited -- and it says so,
rather than being left out and inviting the seat to file icon complaints under a stage that can
act on them.
"""

from __future__ import annotations

from typing import Any, Iterable

from .evolution import STAGE_ORDER


#: What re-running the named stage costs. Kept to three words plus one that means "do not".
COST_EXPENSIVE = "expensive"
COST_MODERATE = "moderate"
COST_CHEAP = "cheap"
COST_DEFERRED = "deferred"
ATTRIBUTION_COSTS = (COST_CHEAP, COST_MODERATE, COST_EXPENSIVE, COST_DEFERRED)

#: Symptom to stage, one row per complaint the seat is known to raise.
#:
#: ``stage`` is the stage the table sends the complaint to by default, ``requirement`` is the
#: requirement that stage should have written and did not, and ``alternatives`` names the other
#: stage the same symptom sometimes belongs to with the condition that moves it there. A row is
#: readable on its own: the symptom names something measurable or visible, and the requirement
#: names a sentence a planning prompt could carry.
FAULT_ATTRIBUTION: tuple[dict[str, Any], ...] = (
    {
        "symptom": "the drawing is thin: panels are nearly empty and the modules the paper states are not all in it",
        "stage": "content_planning",
        "requirement": "the content plan must list every module the paper states, each with the evidence it rests on",
        "cost": COST_EXPENSIVE,
        "alternatives": (
            {
                "stage": "paper_reading",
                "requirement": "the reading must extract every module the paper states, not only the ones a heading happens to name",
                "when": "the reading itself came back with only a handful of modules, so there was nothing for the content plan to choose from",
            },
        ),
        "note": "",
    },
    {
        "symptom": "the layout does not read well: the centre of gravity is off, the gaps are uneven, nothing leads the eye",
        "stage": "segmentation",
        "requirement": "the main flow is drawn large in the middle and the inputs and the outputs are moved out to the sides",
        "cost": COST_MODERATE,
        "alternatives": (),
        "note": "",
    },
    {
        "symptom": "the connectors are a tangle: they cross, they take long detours, they wander across panels",
        "stage": "ppt_compilation",
        "requirement": "a connector leaves its node on the side that faces its target and runs down a channel shared with its neighbours instead of across a panel",
        "cost": COST_CHEAP,
        "alternatives": (
            {
                "stage": "segmentation",
                "requirement": "two nodes a connector joins are placed near each other, and in the same panel when the paper says they belong together",
                "when": "the crossings are a symptom of boxes that sit in the wrong panels, so the layout rather than the routing is what has to change",
            },
        ),
        "note": "prefer changing the drawing alone: reroute, repair the interfaces, or state a routing constraint before anything upstream is planned again",
    },
    {
        "symptom": "the lettering is crowded: a panel title is overlapped by the box under it, or a label runs into the box beside it",
        "stage": "ppt_compilation",
        "requirement": "a panel title sits in a band of its own with nothing drawn over it, and every label keeps a margin inside the box it belongs to",
        "cost": COST_CHEAP,
        "alternatives": (
            {
                "stage": "segmentation",
                "requirement": "two things that must not touch are given regions that do not overlap, so a title band and the content under it never share space",
                "when": "the title band and the content under it were given overlapping space by the structure plan, so no amount of drawing can separate them",
            },
        ),
        "note": "prefer changing the drawing alone: the composition stage can open a band, move a label or shrink it inside its box, and this is what a figure that is otherwise already good is usually waiting on -- it is not a reason to plan the structure again",
    },
    {
        "symptom": "the colours do not sit together: too many of them, or the eye is pulled somewhere that carries no meaning",
        "stage": "theme_planning",
        "requirement": "at most three base colours, and the accent colour kept for the state labels",
        "cost": COST_CHEAP,
        "alternatives": (),
        "note": "the palette planner is the stage that decides the whole figure's colours; the paper is not re-read for this",
    },
    {
        "symptom": "the icons do not look like what they stand for, or they are not in one style",
        "stage": "icon_generation",
        "requirement": "flat single-stroke line icons, drawn at one stroke width",
        "cost": COST_DEFERRED,
        "alternatives": (),
        "note": "not where this loop's effort goes: the image model behind the icons is cheap and its reach is limited, so a complaint here is recorded and not chased",
    },
)


def _normalise(text: Any) -> str:
    """One spelling for a requirement, so a menu match does not turn on punctuation or case."""

    return " ".join(str(text or "").casefold().split()).strip(" .;,:")


def attribution_stages() -> tuple[str, ...]:
    """Every stage the table may name, in the pipeline's own order."""

    named = {str(row.get("stage") or "") for row in FAULT_ATTRIBUTION}
    for row in FAULT_ATTRIBUTION:
        named |= {str(item.get("stage") or "") for item in row.get("alternatives") or () if isinstance(item, dict)}
    return tuple(stage for stage in STAGE_ORDER if stage in named)


def stage_requirements() -> dict[str, tuple[str, ...]]:
    """The requirement menu each stage is offered, primary rows before alternates."""

    menu: dict[str, list[str]] = {}
    for row in FAULT_ATTRIBUTION:
        entries = [{"stage": row.get("stage"), "requirement": row.get("requirement")}] + [
            dict(item) for item in row.get("alternatives") or () if isinstance(item, dict)
        ]
        for entry in entries:
            stage = str(entry.get("stage") or "")
            requirement = str(entry.get("requirement") or "")
            if not stage or not requirement:
                continue
            bucket = menu.setdefault(stage, [])
            if requirement not in bucket:
                bucket.append(requirement)
    return {stage: tuple(items) for stage, items in menu.items()}


def row_for_stage(stage: Any) -> dict[str, Any]:
    """The table row a stage is the answer to, primary rows first."""

    name = str(stage or "").strip()
    for row in FAULT_ATTRIBUTION:
        if str(row.get("stage") or "") == name:
            return dict(row)
    for row in FAULT_ATTRIBUTION:
        if any(str(item.get("stage") or "") == name for item in row.get("alternatives") or () if isinstance(item, dict)):
            return dict(row)
    return {}


def requirement_is_on_menu(stage: Any, requirement: Any) -> bool:
    """Whether a stated requirement is one of the sentences that stage was offered.

    The comparison is deliberately loose about spelling and exact about meaning: a seat that
    writes the menu's sentence back with different punctuation has answered the question, and a
    seat that invents a requirement has written something the ledger should be able to see it
    invented -- which is why an off-menu answer is kept and flagged rather than thrown away.
    """

    wanted = _normalise(requirement)
    if not wanted:
        return False
    for sentence in stage_requirements().get(str(stage or "").strip(), ()):
        known = _normalise(sentence)
        if wanted == known or wanted in known or known in wanted:
            return True
    return False


def render_fault_attribution_for_prompt() -> str:
    """The table as the advisory seat reads it, cheapest stage first within each row.

    Rendered from the same rows the gate checks against, so the menu a served answer is written
    from and the menu it is judged against are one text.
    """

    lines = [
        "Where a defect entered, and which requirement went unwritten there.",
        "Answer fault_stage with one of these stage names, fault_requirement with the requirement "
        "that stage failed to write, and fault_evidence with the numbers you read it off.",
        "Prefer the latest and cheapest stage that can carry the fix: a complaint about the "
        "connectors is answered at the drawing if the drawing can move a line, and rises to the "
        "structure only when the boxes themselves are in the wrong panels.",
    ]
    for row in FAULT_ATTRIBUTION:
        lines.append(
            "- symptom: {symptom}\n  fault_stage: {stage}\n  fault_requirement: {requirement}\n  cost of re-running it: {cost}"
            .format(
                symptom=str(row.get("symptom") or ""),
                stage=str(row.get("stage") or ""),
                requirement=str(row.get("requirement") or ""),
                cost=str(row.get("cost") or ""),
            )
        )
        for item in row.get("alternatives") or ():
            if not isinstance(item, dict):
                continue
            lines.append(
                "  instead {stage}, when {when}: {requirement}".format(
                    stage=str(item.get("stage") or ""),
                    when=str(item.get("when") or ""),
                    requirement=str(item.get("requirement") or ""),
                )
            )
        note = str(row.get("note") or "").strip()
        if note:
            lines.append(f"  note: {note}")
    return "\n".join(lines)


def validate_fault_attribution(rows: Iterable[dict[str, Any]] | None = None) -> list[str]:
    """Check the table is readable, that its rows are distinct, and that its stages exist.

    A row whose stage is not in the pipeline's own order cannot be answered, and two rows that
    send the same symptom to the same requirement are one row written twice -- both make the
    menu ambiguous in a way the seat cannot resolve, so both are errors rather than warnings.
    """

    table = tuple(rows) if rows is not None else FAULT_ATTRIBUTION
    errors: list[str] = []
    seen_symptoms: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(table):
        if not isinstance(row, dict):
            errors.append(f"row {index} is not an object")
            continue
        symptom = str(row.get("symptom") or "").strip()
        stage = str(row.get("stage") or "").strip()
        requirement = str(row.get("requirement") or "").strip()
        cost = str(row.get("cost") or "").strip()
        if not symptom:
            errors.append(f"row {index} names no symptom")
        if not requirement:
            errors.append(f"row {index} names no requirement")
        if stage not in STAGE_ORDER:
            errors.append(f"row {index} names a stage outside the pipeline order: {stage}")
        if cost not in ATTRIBUTION_COSTS:
            errors.append(f"row {index} names a cost outside {ATTRIBUTION_COSTS}: {cost}")
        key = _normalise(symptom)
        if key in seen_symptoms:
            errors.append(f"row {index} repeats an earlier symptom: {symptom}")
        seen_symptoms.add(key)
        pair = (stage, _normalise(requirement))
        if pair in seen_pairs:
            errors.append(f"row {index} repeats an earlier stage and requirement")
        seen_pairs.add(pair)
        for item in row.get("alternatives") or ():
            if not isinstance(item, dict):
                errors.append(f"row {index} has an alternative that is not an object")
                continue
            alt_stage = str(item.get("stage") or "").strip()
            if alt_stage not in STAGE_ORDER:
                errors.append(f"row {index} names an alternative stage outside the pipeline order: {alt_stage}")
            if not str(item.get("requirement") or "").strip():
                errors.append(f"row {index} names an alternative with no requirement")
            if not str(item.get("when") or "").strip():
                errors.append(f"row {index} names an alternative with no condition")
    return errors


__all__ = [
    "ATTRIBUTION_COSTS",
    "COST_CHEAP",
    "COST_DEFERRED",
    "COST_EXPENSIVE",
    "COST_MODERATE",
    "FAULT_ATTRIBUTION",
    "attribution_stages",
    "render_fault_attribution_for_prompt",
    "requirement_is_on_menu",
    "row_for_stage",
    "stage_requirements",
    "validate_fault_attribution",
]
