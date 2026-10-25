"""Re-run the upstream stage a measured break asks for.

The spec (§6.2) lets the loop go back to the method extraction and the framework structure
planning, but only when a local repair cannot help: a global topology break, a scientific
error, or a figure that cannot be decomposed reasonably. The loop used to record that
intent without acting on it -- ``reanalyze_semantics`` appended an ``analysis_requests``
entry no module read, and ``resegment`` / ``replan_structure`` fell through to the same
bounded node nudge as ``relayout_global`` -- so the escalation happened in the record and
not in the figure.

This module performs the re-run. Both upstream stages are injected, so the orchestration,
its refusals and its record are testable with no provider, and the caller keeps ownership
of its own state: the report carries what the re-run produced and the workflow decides
what to adopt. The second attempt is never told to invent anything; it is asked to make the
objects the first attempt misplaced separable, under the same paper evidence.

The theme planner is the third stage of the same family: a colour system the acceptance role
or the colour review asked to change is re-planned by the model that owns colour, against the
colour feedback the previous round measured. Nothing here writes a palette itself.
"""

from __future__ import annotations

from typing import Any, Callable

from ..contracts.memory_parameters import replayable_theme
from ..evolution.direct.snapshots import state_hash


UPSTREAM_RETRY_CONTRACT = "rfs.upstream_retry"
UPSTREAM_RETRY_VERSION = 1

#: The bounded actions that mean "a local repair cannot clear this".
UPSTREAM_RETRY_KINDS = {
    "resegment": "structure",
    "replan_structure": "structure",
    "reanalyze_semantics": "semantics",
    "replan_theme": "theme",
}


def upstream_retry_kind(action: dict[str, Any] | None) -> str:
    """Name the upstream stage this action asks to re-run, or nothing at all."""

    return UPSTREAM_RETRY_KINDS.get(str((action or {}).get("type") or ""), "")


#: What the re-run is asked to *do*, read off the metric the measurement named.
#:
#: The note named the metrics and the objects and then stopped ("re-plan those objects so each
#: sits inside exactly one panel"), which is one sentence whichever break was measured -- and it
#: is close to a restatement of the prompt the first attempt had already run, so the second
#: attempt had no reason to come out different from the first. A direction is what makes it a
#: different question: it says which way the objects move, and the metric that was measured is
#: the same one the re-run is checked against afterwards.
RETRY_DIRECTIONS: dict[str, str] = {
    "cross_panel_connectors": (
        "a connector that joins two panels must leave its node on the side that faces its target "
        "and run down a channel shared with its neighbours, instead of cutting across the panel band"
    ),
    "slot_boundary_overflows": (
        "a slot must sit wholly inside the panel it belongs to, with a visible gap to the panel edge"
    ),
    "unbound_scientific_nodes": (
        "every node the paper states must be attached to the module it belongs to, and no node may "
        "be left belonging to nothing"
    ),
    "orphan_nodes": (
        "no node may be drawn outside every panel: each one joins the module the paper puts it in"
    ),
    "min_panel_occupancy": (
        "each panel must be filled by its own group rather than left mostly empty, and the groups "
        "are widened to use the room they have"
    ),
    "empty_region_ratio": (
        "the canvas must not carry a large empty region: the groups are spread so the leftover "
        "space sits between the margins instead of pooling in one corner"
    ),
    "max_node_slot_fanout": (
        "regroup the labels under fewer, wider nodes, so a reader can tell which label belongs to "
        "which node"
    ),
}

#: What to do when the measurement named a break this table does not know: a direction is still
#: owed, so the note falls back to the one instruction that is true of every break here.
RETRY_DIRECTION_FALLBACK = (
    "make the objects the measurement named separable from their neighbours, under the same paper "
    "evidence the first attempt had"
)


def retry_directions(policy: dict[str, Any] | None) -> list[str]:
    """The direction to ask for, one per measured break, in the order the breaks were reported."""

    source = policy if isinstance(policy, dict) else {}
    seen: list[str] = []
    for item in source.get("violations") or []:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric") or "").strip()
        text = RETRY_DIRECTIONS.get(metric)
        if text is None:
            continue
        if text not in seen:
            seen.append(text)
    return seen or [RETRY_DIRECTION_FALLBACK]


def describe_retry_feedback(policy: dict[str, Any] | None) -> str:
    """Turn a measured break into the note the re-run has to answer.

    The note names the metrics and the typed targets the measurement already produced, then
    says which way those targets have to move. The direction is read off the metric the
    measurement named, so the second attempt is aimed at the same objects the first attempt got
    wrong and told what to do about them rather than asked again. The note closes by naming the
    numbers the re-run will be checked against, so the round that runs it has a reading to be
    verified against rather than a claim to be believed.
    """

    source = policy if isinstance(policy, dict) else {}
    violations = [item for item in source.get("violations") or [] if isinstance(item, dict)]
    metrics = sorted({str(item.get("metric") or "") for item in violations if str(item.get("metric") or "")})
    targets: list[str] = []
    for item in violations:
        for key in ("target_node_ids", "target_connector_ids"):
            for value in item.get(key) or []:
                text = str(value).strip()
                if text and text not in targets:
                    targets.append(text)
    parts = [
        "A previous attempt left a decomposability break that a local repair could not clear"
        + (": " + ", ".join(metrics) + "." if metrics else ".")
    ]
    if targets:
        parts.append("The measured objects are " + ", ".join(targets) + ".")
    reason = str(source.get("reason") or "").strip()
    if reason:
        parts.append(reason if reason.endswith(".") else reason + ".")
    parts.append("Change this, and change only this: " + "; ".join(retry_directions(source)) + ".")
    parts.append(
        "Keep every scientific claim and every evidenced label, and never add an object the paper "
        "does not state."
    )
    if metrics:
        parts.append("The re-run is measured against the same numbers: " + ", ".join(metrics) + ".")
    return " ".join(parts)


def _pair(value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read a stage answer, which a provider-backed stage returns with its own report."""

    if isinstance(value, tuple) and len(value) == 2:
        first, second = value
        return (
            first if isinstance(first, dict) else {},
            second if isinstance(second, dict) else {},
        )
    return (value if isinstance(value, dict) else {}), {}


def _theme_identity(theme: dict[str, Any] | None) -> dict[str, Any]:
    """The palette itself, without the bookkeeping the caller hangs off it.

    The workflow stamps ``provenance`` onto the theme it adopts, and a re-planned theme comes
    back without one. Comparing the whole dict would report "the palette changed" whenever
    only that stamp differed, so a re-run that returned the same colours would look like it
    had re-coloured the figure.
    """

    source = theme if isinstance(theme, dict) else {}
    return {key: value for key, value in source.items() if key != "provenance"}


def _record(action: dict[str, Any] | None, *, kind: str, feedback: str, retry_index: int) -> dict[str, Any]:
    return {
        "contract": UPSTREAM_RETRY_CONTRACT,
        "version": UPSTREAM_RETRY_VERSION,
        "action_type": str((action or {}).get("type") or ""),
        "kind": kind,
        "retry_index": int(retry_index),
        "feedback": str(feedback or ""),
        "attempted": False,
        "applied": False,
        "declined_reason": "",
        "provider_error": "",
        "before_hash": "",
        "after_hash": "",
        "changed": False,
        "method": {},
        "method_report": {},
        "design": {},
        "structure": {},
        "structure_report": {},
        "theme": {},
        "theme_report": {},
    }


def run_upstream_retry(
    action: dict[str, Any] | None,
    *,
    enabled: bool = True,
    planner_mode: str = "vlm",
    feedback: str = "",
    retry_index: int = 0,
    method: dict[str, Any] | None = None,
    design: dict[str, Any] | None = None,
    theme: dict[str, Any] | None = None,
    color_feedback: dict[str, Any] | None = None,
    structure_planner: Callable[[dict[str, Any], str], Any] | None = None,
    structure_builder: Callable[[dict[str, Any]], Any] | None = None,
    method_extractor: Callable[[str], Any] | None = None,
    theme_planner: Callable[[dict[str, Any]], Any] | None = None,
    theme_applier: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Re-run the stage the escalation names, or say why it could not be re-run.

    ``structure_planner`` takes the method and the feedback note, ``method_extractor`` takes
    the feedback note, and ``structure_builder`` turns a design into the layout/icon
    contracts the round runs on. A semantic escalation re-extracts the method first and
    plans the structure against what came back, because a layout whose node ids belong to
    the previous contract is not a layout of the new one. A refusal is recorded as a
    reason, never as a silent skip, and a provider that fails leaves the state untouched.

    ``theme_planner`` takes the colour feedback and returns the re-planned theme with its own
    report; ``theme_applier`` rebuilds the same design's structure under that theme, so the
    round compiles the panel and role styles the new palette produces.
    """

    kind = upstream_retry_kind(action)
    report = _record(action, kind=kind, feedback=feedback, retry_index=retry_index)
    stored_theme = replayable_theme((action or {}).get("color_theme")) if kind == "theme" else {}
    if not kind:
        return {**report, "declined_reason": "the action does not ask for an upstream re-run"}
    if not enabled:
        return {**report, "declined_reason": "upstream retry is switched off for this run"}
    mode = str(planner_mode or "").strip().casefold()
    # A Memory row that already carries a palette is applied, not re-sampled. Heuristic
    # mode may still replay it; only a missing palette still needs the VLM planner.
    if mode != "vlm" and not stored_theme:
        return {**report, "declined_reason": f"planner_mode_{mode or 'unknown'}"}
    if kind == "structure" and (structure_planner is None or structure_builder is None):
        return {**report, "declined_reason": "no structure re-planning is available on this run"}
    if kind == "semantics" and (method_extractor is None or structure_planner is None or structure_builder is None):
        return {**report, "declined_reason": "a semantic re-run needs both the method extractor and the structure planner"}
    if kind == "theme" and theme_applier is None:
        return {**report, "declined_reason": "re-planning the theme needs both the theme planner and the builder that applies it"}
    if kind == "theme" and not stored_theme and theme_planner is None:
        return {**report, "declined_reason": "re-planning the theme needs both the theme planner and the builder that applies it"}

    current_method = method if isinstance(method, dict) else {}
    current_design = design if isinstance(design, dict) else {}
    current_theme = theme if isinstance(theme, dict) else {}
    before_hash = state_hash({"method": current_method, "design": current_design, "theme": _theme_identity(current_theme)})
    attempted = {**report, "attempted": True}
    try:
        if kind == "theme":
            new_method, method_report = current_method, {}
            new_design, structure_report = current_design, {}
            if stored_theme:
                new_theme, theme_report = stored_theme, {"ok": True, "mode": "memory", "fallback_used": False}
            else:
                new_theme, theme_report = _pair(theme_planner(color_feedback if isinstance(color_feedback, dict) else {}))
            if not new_theme:
                return {**attempted, "declined_reason": "the theme re-plan returned no theme"}
        elif kind == "semantics":
            new_theme, theme_report = {}, {}
            new_method, method_report = _pair(method_extractor(feedback))
            if not new_method:
                return {**attempted, "declined_reason": "the method re-extraction returned no contract"}
            design_answer = structure_planner(new_method, feedback)
            new_design, structure_report = _pair(design_answer)
        else:
            new_theme, theme_report = {}, {}
            new_method, method_report = current_method, {}
            new_design, structure_report = _pair(structure_planner(current_method, feedback))
    except Exception as exc:  # noqa: BLE001 - a failed re-run is a recorded outcome, not a crash
        return {
            **attempted,
            "declined_reason": "provider_error",
            "provider_error": f"{type(exc).__name__}: {exc}",
        }
    # A theme re-run keeps the design it was given; only the palette is new, so there is no
    # design to be missing and the layout check below is what decides whether it worked.
    if not new_design and kind != "theme":
        return {**attempted, "declined_reason": "the re-plan returned no design"}
    structure = theme_applier(new_theme) if kind == "theme" else (structure_builder(new_design) if structure_builder is not None else {})
    structure = structure if isinstance(structure, dict) else {}
    layout = structure.get("figure_layout_plan") if isinstance(structure.get("figure_layout_plan"), dict) else {}
    if not layout:
        return {
            **attempted,
            "method": new_method,
            "method_report": method_report,
            "design": new_design,
            "structure_report": structure_report,
            "theme": new_theme,
            "theme_report": theme_report,
            "declined_reason": "the re-plan produced no layout",
        }
    after_hash = state_hash({"method": new_method, "design": new_design, "theme": _theme_identity(new_theme)})
    return {
        **attempted,
        "applied": True,
        "before_hash": before_hash,
        "after_hash": after_hash,
        "changed": after_hash != before_hash,
        "method": new_method,
        "method_report": method_report,
        "design": new_design,
        "structure": structure,
        "structure_report": structure_report,
        "theme": new_theme,
        "theme_report": theme_report,
    }


__all__ = [
    "UPSTREAM_RETRY_CONTRACT",
    "UPSTREAM_RETRY_KINDS",
    "UPSTREAM_RETRY_VERSION",
    "describe_retry_feedback",
    "run_upstream_retry",
    "upstream_retry_kind",
]
