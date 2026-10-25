"""Ask the advisory seat where our figure falls short of a good one.

The loop only ever showed the advisory seat one figure and asked what was wrong with it, so the
answer it could give was about that figure: a label drifting, a text box overflowing. A seat that
cannot see a good figure has no way to say the panels are arranged worse than they should be, and
that is why every round of advice has been local. This module holds the other question -- a good
figure from the library beside ours, and "which class of gap separates them, which way must ours
move" -- together with its schema and the reading of the answer.

The question is deliberately not part of the measurement path. A measurement answers from the
figure alone; the second image here would be a preference travelling into a score. So the answer
is filed beside the round, carried into the reflection and reported, and it never decides whether
a figure is accepted.

An answer that names no class, or that asks for no action, is the question going unanswered
rather than a gap that was found. It is counted as such: the seat is expected to answer badly
before it is trained on this question, and a silent answer recorded as a finding is how a dead
channel reads as a working one.
"""

from __future__ import annotations

import json
from typing import Any

from ..contracts.repair_actions import action_semantics

REFERENCE_GAP_CONTRACT = "rfs.reference_gap"
REFERENCE_GAP_VERSION = 1

#: The classes a gap between two figures may be filed under. They are the axes a reader would
#: name if asked "why does this one read worse", and they are deliberately coarse: a class the
#: loop cannot act on is not worth asking for.
GAP_CLASSES = (
    "layout_structure",
    "information_organisation",
    "visual_hierarchy",
    "colour_system",
    "visual_metaphor",
)

#: The actions that change how the figure is planned or drawn rather than nudging an object
#: inside it. A gap is not closed by moving one box, so these are the only actions this question
#: may ask for.
MACRO_ACTIONS = ("replan_structure", "resegment", "replan_theme", "relayout_global")

GAP_IDENTITY = (
    "You are a strict scientific framework-figure reviewer. Two images are supplied after "
    "this text, in this order: the first is A, a figure of this kind that a journal published "
    "and that a reader found easy to read; the second is B, our own figure for a different "
    "paper, drawn by a generator that can only change what you name. Scientific correctness "
    "and grounding take priority over aesthetics. Do not invent facts. Your only job is to say "
    "which class of gap separates B from A and which way B must move to close it. "
    "'No change is needed' is not an answer: if you can see no difference at all, name the "
    "smallest one you can defend from the images."
)

GAP_RULE = (
    "gap_class must come from this menu and must be exactly one of them: "
    "layout_structure (the panels, columns, rows or reading path are arranged worse); "
    "information_organisation (what is grouped, ordered or given its own panel is wrong); "
    "visual_hierarchy (what a reader sees first, and how emphasis and scale are distributed, "
    "is wrong); colour_system (the palette, its contrast or what colour is used to mean is "
    "wrong); visual_metaphor (A explains the mechanism as a picture while B lists boxes). "
    "direction is one or two sentences saying where B should move, written as an instruction "
    "to a draughtsman: name no node label, no panel title, no object id and no coordinate. "
    "evidence is what in the two images shows the gap; it is a visible fact, not a restatement "
    "of the class. requested_actions must contain at least one action from the menu below, "
    "each with expected_effect saying what B must look like afterwards. Choose the smallest "
    "action that can close the gap you named, and choose more than one only when one cannot."
)


def macro_menu() -> str:
    """Render the macro actions from the catalog the compiler already reads.

    The wording is taken from ACTION_SEMANTICS rather than written again here, so a question
    asked of a seat cannot describe an action the loop has since changed.
    """

    lines = []
    for action in MACRO_ACTIONS:
        row = action_semantics(action)
        lines.append(f"- {action}: {row.get('does', '')}. It cannot: {row.get('cannot', '')}.")
    return "\n".join(lines)


def gap_schema() -> dict[str, Any]:
    """The object the seat must return, in the shape the seat's other answers take."""

    return {
        "gap_class": "|".join(GAP_CLASSES),
        "direction": "one or two sentences: where B must move; names no label, id or coordinate",
        "evidence": ["what in the two images shows the gap"],
        "confidence": "number 0..1",
        "requested_actions": [
            {
                "action": "|".join(MACRO_ACTIONS),
                "expected_effect": "what B must look like once this action has run",
            }
        ],
    }


def build_reference_gap_prompt(reference_id: str, reference_note: str, ours_note: str) -> str:
    """Render the one question this module asks, in the shape the seat already reads."""

    payload = {
        "a_identity": str(reference_id or ""),
        "a_is": "a published figure of this kind, kept in the reference library as a good example",
        "a_analysis": str(reference_note or ""),
        "b_is": str(ours_note or ""),
    }
    return (
        f"{GAP_IDENTITY}\n"
        "Return only one valid JSON object, with no markdown.\n"
        "Judge role: advisory.\n"
        "Task: reference_gap.\n"
        f"{GAP_RULE}\n"
        "Macro action menu:\n"
        f"{macro_menu()}\n"
        f"Reference contract:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
        f"Required output schema (types are descriptive):\n{json.dumps(gap_schema(), ensure_ascii=False)}"
    )


def reference_gap_conformance(answer: Any) -> list[str]:
    """Return why an answer cannot be used, or an empty list when it can.

    The question forbids "nothing to change", so an answer that names no class or asks for no
    action is not a mild answer to file away: it is the question going unanswered, and it is
    counted as such rather than as a gap that was found.
    """

    problems: list[str] = []
    if not isinstance(answer, dict) or not answer:
        return ["the answer was not an object"]
    gap = str(answer.get("gap_class") or "")
    if gap not in GAP_CLASSES:
        problems.append(f"gap_class is not one of the menu: {gap!r}")
    if not str(answer.get("direction") or "").strip():
        problems.append("no direction was named")
    actions = answer.get("requested_actions")
    if not isinstance(actions, list) or not actions:
        problems.append("no requested_actions at all")
    else:
        for item in actions:
            name = str((item or {}).get("action") or "") if isinstance(item, dict) else str(item)
            if name not in MACRO_ACTIONS:
                problems.append(f"requested action is not a macro action: {name!r}")
    return problems


def normalize_reference_gap(answer: Any, *, problem: str = "", retried: bool = False) -> dict[str, Any]:
    """Read one answer into the record a round files.

    readable is about the seat having spoken at all and conforms about the answer being one this
    question can use. They are separate on purpose: a round needs to tell "the seat said nothing"
    apart from "the seat named a class the loop cannot act on", because the first is a broken
    channel and the second is a seat that needs the question taught.
    """

    raw = answer if isinstance(answer, dict) else {}
    problems = [problem] if problem else reference_gap_conformance(raw)
    return {
        "contract": REFERENCE_GAP_CONTRACT,
        "version": REFERENCE_GAP_VERSION,
        "gap_class": str(raw.get("gap_class") or ""),
        "direction": str(raw.get("direction") or ""),
        "evidence": [str(item) for item in (raw.get("evidence") or []) if str(item).strip()][:6],
        "confidence": raw.get("confidence"),
        "requested_actions": [
            dict(item) if isinstance(item, dict) else {"action": str(item)}
            for item in (raw.get("requested_actions") or [])
        ],
        "readable": not problem,
        "conforms": not problems,
        "problems": problems,
        "retried": bool(retried),
    }


def reference_gap_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count what the question got, so a channel that never answers cannot read as a quiet one."""

    classes: dict[str, int] = {}
    actions: dict[str, int] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        answer = record.get("answer") if isinstance(record.get("answer"), dict) else record
        name = str(answer.get("gap_class") or "(none)")
        classes[name] = classes.get(name, 0) + 1
        for item in answer.get("requested_actions") or []:
            action = str((item or {}).get("action") or "(none)") if isinstance(item, dict) else str(item)
            actions[action] = actions.get(action, 0) + 1
    return {
        "contract": REFERENCE_GAP_CONTRACT,
        "version": REFERENCE_GAP_VERSION,
        "asked": len(records),
        "readable": sum(1 for record in records if record.get("readable")),
        "conforming": sum(1 for record in records if record.get("conforms")),
        "gap_classes": dict(sorted(classes.items())),
        "requested_actions": dict(sorted(actions.items())),
    }


__all__ = [
    "GAP_CLASSES",
    "GAP_IDENTITY",
    "GAP_RULE",
    "MACRO_ACTIONS",
    "REFERENCE_GAP_CONTRACT",
    "REFERENCE_GAP_VERSION",
    "build_reference_gap_prompt",
    "gap_schema",
    "macro_menu",
    "normalize_reference_gap",
    "reference_gap_conformance",
    "reference_gap_summary",
]