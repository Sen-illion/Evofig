"""Build the online Judge's prompt row and render it through the shared prompt contract.

The prompt text, its version and its output schema live with the Judge training package,
because that is what the trained model is trained on. This module only assembles the input
row from a :class:`~rfs.evaluation.direct_judges.DirectJudgeInput` and hands it to the
canonical renderer, so the product loop can never drift onto a prompt of its own.

The renderer is imported on first use. ``rfs.judge_training`` is a heavy package holding PDF
and dataset tooling, and importing ``rfs.evaluation`` has to stay cheap for the product path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..contracts.evolution import ADVISORY_ROLE_FAMILY, judge_role_family
from ..contracts.suggestion_rule import normalize_suggestion_rule, suggestion_rule_usable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .direct_judges import DirectJudgeInput


ONLINE_TASK_TYPE = "repair"
EVIDENCE_LIMIT = 12

#: How many advice rules one prompt may carry. Advice competes with the paper contract for the
#: model's attention, so a larger library is trimmed to the first ones rather than handed over.
SUGGESTION_RULE_LIMIT = 6

#: How many of a person's own instructions one prompt may carry. Capped for the same reason the
#: advice is: the sentence competes with the paper contract for the model's attention, so a long
#: standing list is trimmed to the newest ones rather than handed over whole.
HUMAN_PREFERENCE_LIMIT = 6


def _text(value: Any) -> str:
    return str(value or "").strip()


def _labels_of(node: dict[str, Any]) -> str:
    return _text(node.get("label")) or _text(node.get("name")) or _text(node.get("id"))


def method_contract(method: dict[str, Any]) -> dict[str, Any]:
    """Read the required entities and relations out of the paper's method contract."""

    modules = [item for item in (method or {}).get("modules") or [] if isinstance(item, dict)]
    relations = [item for item in (method or {}).get("relations") or [] if isinstance(item, dict)]
    by_id = {_text(item.get("id")): item for item in modules if _text(item.get("id"))}
    required = [item for item in modules if item.get("must_show")]
    entities = [_labels_of(item) for item in (required or modules)]
    links = []
    for item in relations:
        source = _labels_of(by_id.get(_text(item.get("source")), {})) or _text(item.get("source"))
        target = _labels_of(by_id.get(_text(item.get("target")), {})) or _text(item.get("target"))
        if source and target:
            kind = _text(item.get("type")) or _text(item.get("label"))
            links.append(f"{source} -> {target}" + (f" ({kind})" if kind else ""))
    return {
        "method_summary": _text((method or {}).get("figure_goal")),
        "required_entities": [item for item in entities if item],
        "required_relations": links,
    }


def injectable_suggestion_rules(rules: Any, *, role: str = "", pairwise: bool = False) -> list[dict[str, Any]]:
    """Return the advice rules a prompt may carry -- which, for a measurement role, is none.

    Advice belongs to the role that proposes the next move. The acceptance role measures what the
    figure is worth and the pairwise role chooses between two figures; telling either of them what
    to ask for would let a preference leak into a measurement. A rule that has not been stated
    twice, or validated, is dropped here as well, so one round cannot brief the next paper.
    """

    if pairwise or judge_role_family(role) != ADVISORY_ROLE_FAMILY:
        return []
    kept: list[dict[str, Any]] = []
    for item in rules or []:
        normalized = normalize_suggestion_rule(item if isinstance(item, dict) else {})
        if not suggestion_rule_usable(normalized):
            continue
        kept.append({key: normalized[key] for key in ("rule_id", "apply_at", "defect_pattern", "rule_text", "rule", "support")})
        if len(kept) >= SUGGESTION_RULE_LIMIT:
            break
    return kept


def human_preferences_for_prompt(preferences: Any, *, role: str = "", pairwise: bool = False) -> list[dict[str, Any]]:
    """Return the person's own instructions a prompt may carry -- which, for a measurement role, is none.

    The same boundary the advice rules are held to, for the same reason: the advisory seat proposes
    the next move and may be told what a person asked for, while the acceptance and pairwise seats
    measure what a figure is worth, and a preference in their mouth would stop being a measurement.
    Only entries that actually carry a sentence are kept, so a half-written verdict adds no key.
    """

    if pairwise or judge_role_family(role) != ADVISORY_ROLE_FAMILY:
        return []
    kept: list[dict[str, Any]] = []
    for item in preferences or []:
        row = item if isinstance(item, dict) else {}
        directive = _text(row.get("directive"))
        if not directive:
            continue
        kept.append(
            {
                "apply_at": _text(row.get("apply_at")),
                "directive": directive,
                "condition": dict(row.get("condition") or {}) if isinstance(row.get("condition"), dict) else {},
                "source": _text(row.get("source")) or "human",
            }
        )
        if len(kept) >= HUMAN_PREFERENCE_LIMIT:
            break
    return kept


def online_judge_row(judge_input: "DirectJudgeInput", *, role: str = "", pairwise: bool = False, comparison: bool = False) -> dict[str, Any]:
    """Return the shared-contract row for one compiled candidate.

    Only what the Judge can know at call time enters the row: the paper's method contract,
    the candidate itself and the bounded scope it may speak about. Consequences of a repair
    belong to the example's ``observed`` block, never to a prompt. ``pairwise`` marks the row as
    one half of a comparison, the other question advice never enters.
    """

    method = judge_input.method if isinstance(judge_input.method, dict) else {}
    process_context = {
        "grounding_gate": judge_input.grounding_gate or {},
        "allowed_scope": judge_input.allowed_scope or {},
        "prior_feedback": list(judge_input.prior_feedback or [])[:EVIDENCE_LIMIT],
        "validated_experiences": list(judge_input.experiences or []),
    }
    # An empty list would still change the rendered prompt, so it is only carried when a
    # patch actually applies to this stage.
    if judge_input.prompt_patches:
        process_context["prompt_patches"] = list(judge_input.prompt_patches)
    # The measured colour problem the previous round found travels to the advisory role as
    # context, so it can ask for the theme to be re-planned. An empty report adds no key.
    if judge_input.color_report:
        process_context["color_report"] = dict(judge_input.color_report)
    # Advice the library has earned travels as context too, and the reader of that context is the
    # advisory role alone: a measurement role told what to ask for is no longer a measurement. An
    # empty library adds no key, so a run with no earned advice renders exactly as before.
    advice = injectable_suggestion_rules(judge_input.suggestion_rules, role=role, pairwise=pairwise)
    if advice:
        process_context["suggestion_rules"] = advice
    # A person's own instruction travels the way the library's earned advice does, and to the same
    # seat alone. It names the layer it is about and the sentence that layer's prompt should have
    # carried, which is a thing the seat that proposes the next move can act on and a measurement
    # cannot be told. A round that carries none adds no key, so it renders exactly as before.
    preferences = human_preferences_for_prompt(judge_input.human_preferences, role=role, pairwise=pairwise)
    if preferences:
        process_context["human_preferences"] = preferences
        process_context["human_preference_note"] = (
            "a person's own standing instruction about this figure, kept as they wrote it; where it "
            "applies to what you are about to propose, follow it"
        )
    # The version the loop has already accepted may travel beside this round's candidate, so the
    # advisory role answers about a pair instead of about one figure. The Judge was trained on two
    # figures at a time and a lone figure gives it nothing to compare, so it answers "no defect"
    # however the figure looks. Only the advisory role may be handed the second image: a
    # measurement told what to prefer is no longer measuring. A row without one adds no key and
    # renders exactly as it did before.
    # The round's stage trace travels to the advisory seat alone: it is what lets that seat say
    # where a defect entered rather than only that a figure looks worse. A round that changed
    # nothing carries an empty trace and adds no key, so the prompt renders exactly as before.
    if judge_input.stage_trace and not pairwise and judge_role_family(role) == ADVISORY_ROLE_FAMILY:
        process_context["stage_trace"] = judge_input.stage_trace
    # What each layer of a repair costs travels to the advisory seat alone, so the seat that picks
    # a repair can weigh the minutes a stage re-run spends instead of only seeing the defect. The
    # executor refuses nothing for the layer a repair works at, so this is a price list rather
    # than a list of what is forbidden. A round that carries none adds no key and renders exactly
    # as it did before.
    if judge_input.layer_costs and not pairwise and judge_role_family(role) == ADVISORY_ROLE_FAMILY:
        process_context["layer_costs"] = dict(judge_input.layer_costs)
    # Where the figure stands in the pipeline travels to the advisory seat alone: the seat that
    # says which layer to work on is the seat that has to know which layers a model has already
    # finished with. A question asked without one adds no key, so every other prompt renders
    # exactly as it did before.
    if judge_input.progress_sheet and not pairwise and judge_role_family(role) == ADVISORY_ROLE_FAMILY:
        process_context["progress_sheet"] = dict(judge_input.progress_sheet)
    if comparison and not pairwise and judge_role_family(role) == ADVISORY_ROLE_FAMILY:
        process_context["comparison"] = {
            "image_1": "A: the version the loop has already accepted",
            "image_2": "B: this round's candidate",
            "question": "which of the two a reader of this paper is better served by, and what the one you did not prefer shows",
        }
    row = {
        "task_type": ONLINE_TASK_TYPE,
        "judge_role": _text(role),
        "domain": _text((judge_input.domain_profile or {}).get("domain")),
        "domain_rubric": judge_input.domain_profile or {},
        "paper": {"evidence_spans": []},
        "ground_truth": method_contract(method),
        "labels": {},
        "candidate": {
            "method": method,
            "layout": judge_input.layout or {},
            "icons": judge_input.icons or {},
            "program": judge_input.program or {},
        },
        "process_context": process_context,
    }
    if advice:
        # The ids of the advice a judgement was made under travel outside the rendered contract,
        # so an audit can name them without changing one byte of the prompt the model saw.
        row["judge_metadata"] = {"suggestion_rule_ids": [str(item["rule_id"]) for item in advice]}
    if preferences:
        # How many of the person's instructions this judgement was made under, recorded the same
        # way and for the same reason: the count is auditable without entering the prompt.
        row.setdefault("judge_metadata", {})["human_preference_count"] = len(preferences)
    return row


def canonical_prompt(row: dict[str, Any]) -> str:
    """Render one row through the canonical renderer shared with Judge training."""

    from ..judge_training.prompts import build_prompt

    return build_prompt(row)


def canonical_pairwise_prompt(row: dict[str, Any]) -> str:
    """Render the two-state question through the canonical renderer.

    The task name lives in the prompts module, so it is read from there rather than repeated
    here: a served prompt and a trained one may not drift apart.
    """

    from ..judge_training.prompts import PAIRWISE_TASK_TYPE, build_prompt

    return build_prompt({**row, "task_type": PAIRWISE_TASK_TYPE})


def canonical_reflection_prompt(outcome: dict[str, Any]) -> str:
    """Render the reflection prompt through the canonical renderer."""

    from ..judge_training.prompts import build_reflection_prompt

    return build_reflection_prompt(outcome)


def prompt_version() -> str:
    """Return the version of the served prompt, for auditing every judgement."""

    from ..judge_training.prompts import PROMPT_VERSION

    return PROMPT_VERSION


__all__ = [
    "ONLINE_TASK_TYPE",
    "canonical_pairwise_prompt",
    "canonical_prompt",
    "canonical_reflection_prompt",
    "method_contract",
    "online_judge_row",
    "prompt_version",
]
