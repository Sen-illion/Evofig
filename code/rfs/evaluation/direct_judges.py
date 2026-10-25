"""Replaceable Judge adapters for direct paper-to-PPTX evolution."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from ..contracts.evolution import ADVISORY_ROLE_FAMILY, DERIVED_SCORE_SOURCE, citable_fields, deterministic_blocker_counts, deterministic_blocker_ids, judge_role_family, normalize_judge_result, normalize_pairwise_result, normalize_reflection, reflection_conflict
from ..providers import call_vlm_json, resolve_vlm_model
from ..providers.client import extract_json
from .interactive_judgement import judge_direct_run
from .judge_prompt import canonical_pairwise_prompt, canonical_prompt, canonical_reflection_prompt, online_judge_row, prompt_version
from .reference_gap import build_reference_gap_prompt, normalize_reference_gap


@dataclass(frozen=True)
class DirectJudgeInput:
    """Model-independent Judge input for one compiled candidate."""

    method: dict[str, Any]
    layout: dict[str, Any]
    icons: dict[str, Any]
    program: dict[str, Any]
    preview: str | Path | None = None
    domain_profile: dict[str, Any] | None = None
    prior_feedback: list[dict[str, Any]] | None = None
    allowed_scope: dict[str, Any] | None = None
    experiences: list[dict[str, Any]] | None = None
    grounding_gate: dict[str, Any] | None = None
    prompt_patches: list[dict[str, Any]] | None = None
    color_report: dict[str, Any] | None = None
    suggestion_rules: list[dict[str, Any]] | None = None
    #: What a person standing at this run's judgement points has asked for, in the shape a
    #: suggestion rule and a prompt patch share: the layer it is about, and the sentence that
    #: layer's prompt should have carried. It travels to the advisory seat alone -- the seat
    #: allowed to say what to change next -- because a measurement told what to prefer is no
    #: longer measuring. It is handed over fresh for every round, so a rule a person files while
    #: the iteration is running reaches the round that follows it. A round carrying none adds no
    #: key, so every other prompt renders exactly as it did before.
    human_preferences: list[dict[str, Any]] | None = None
    #: The render of the state the loop has already accepted, when the advisory role is being
    #: asked about a pair rather than about one figure. Only the advisory seat ever reads it.
    comparison_preview: str | Path | None = None
    #: The round's own stage trace, compacted. It is what lets the advisory seat say *where* a
    #: defect entered instead of only that the figure looks worse, so only that seat reads it.
    stage_trace: dict[str, Any] | None = None
    #: What each layer of a repair costs, and the reading the loop took of this figure. The seat
    #: that picks a repair spends the round's minutes, so it is the seat that has to weigh a stage
    #: re-run against a nudge -- the executor refuses nothing for the layer a repair works at, and
    #: this is a price list rather than a list of what is forbidden. Only the advisory seat ever
    #: reads it.
    layer_costs: dict[str, Any] | None = None
    #: Where the figure stands in the pipeline: the goal, the layer order, what each layer last
    #: read and which of them a model has finished with, and what the round before this one
    #: changed. It travels to the advisory seat alone, because the seat that says where to start
    #: is the seat that has to know where the work already stands. A question asked without one
    #: adds no key and renders exactly as it did before.
    progress_sheet: dict[str, Any] | None = None
    #: The layer this round is working on, the order the layers are built in and which of them a
    #: model has already finished with. The acceptance seat is the seat that also says where the
    #: round goes next, and this is the record it needs to say it. A question asked without one
    #: adds no key and renders exactly as it did before.
    layer_plan: dict[str, Any] | None = None
    #: The sentence this comparison is measuring, when the comparison is an ON/OFF pair run for
    #: one experience or patch. The seat is told what is under test so its answer is about that
    #: sentence; a comparison that measures nothing in particular leaves it empty.
    experience_under_test: str = ""


@dataclass(frozen=True)
class DirectPairJudgeInput:
    """Model-independent Judge input for one comparison between two compiled states.

    ``before`` is the state the loop had already accepted and ``after`` is the state it has just
    produced; the two names travel with the question so the answer cannot be read against the
    wrong state.
    """

    before: DirectJudgeInput
    after: DirectJudgeInput
    candidate_a: str = "the state the loop had already accepted"
    candidate_b: str = "the state this round produced"
@dataclass(frozen=True)
class ReferenceGapInput:
    """Model-independent input for the question "where does ours fall short of a good figure".

    The reference is one figure from the library that is already known to be the kind of figure
    this project draws, and the two paths travel in the order the question names them: the good
    figure first, ours second. Only the advisory seat is ever handed this input. A measurement
    role told what a good figure looks like is no longer measuring, so the answer is filed beside
    the round and never folded into a score.
    """

    reference_path: str | Path
    ours_path: str | Path
    reference_id: str = ""
    reference_note: str = ""
    ours_note: str = ""
    domain_profile: dict[str, Any] | None = None


def _reference_gap_unavailable(seat: Any, reason: str) -> dict[str, Any]:
    """The record a seat files when it cannot answer about a reference figure.

    Silence is a state the round has to tell apart from "the two figures are alike", so the reason
    travels on the record and the round writes down that the question went unanswered.
    """

    return {
        **normalize_reference_gap({}, problem=reason),
        "judge_role": str(getattr(seat, "role", "") or ""),
        "model": str(getattr(seat, "model", "") or ""),
        "trained": bool(getattr(seat, "trained", False)),
        "prompt_version": prompt_version(),
    }


def _judge_images(judge_input: "DirectJudgeInput", *, role: str) -> tuple[list[Any], bool]:
    """Return the images one question travels with, and whether they form a comparison.

    The advisory role may be shown the version the loop already accepted beside this round's
    candidate. The Judge was trained on two figures at a time, and a lone figure gives it nothing
    to compare, so it answers "no defect" however the figure looks. Image one is A, the accepted
    version, and image two is B, the candidate -- the same order and the same names the pairwise
    question already uses, so the two questions cannot disagree about which image they mean.

    The acceptance and pairwise roles measure one state, so they are never handed a second image.
    """

    images: list[Any] = []
    if judge_input.preview and Path(judge_input.preview).exists():
        images.append(judge_input.preview)
    accepted = judge_input.comparison_preview
    if judge_role_family(role) == ADVISORY_ROLE_FAMILY and accepted and images and Path(accepted).exists():
        return [accepted, images[0]], True
    return images, False


def _new_blocker_count(value: Any) -> int:
    """Read a new-blocker tally that upstream reports as either a count or a list."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, list):
        return len(value)
    return 0


_STATUS_RANK = {"pass": 0, "fault": 1, "blocked": 2}


def _gate_findings(*sources: dict[str, Any] | None) -> tuple[list[str], dict[str, float], str]:
    """Collect what the deterministic gate measured about one candidate.

    The gate is an instrument, not a second judge: its measured blockers, dimensions and
    verdict are folded into the scoring answer, so a block the gate already found still
    counts and is still visible, without the gate ever producing the score itself.
    """

    issues: list[str] = []
    dimensions: dict[str, float] = {}
    status = "pass"
    for source in sources:
        if not isinstance(source, dict):
            continue
        for issue in source.get("blocking_issues") or []:
            text = str(issue)
            if text and text not in issues:
                issues.append(text)
        for item in source.get("stage_feedback") or []:
            if isinstance(item, dict) and str(item.get("severity") or "") == "blocker":
                text = str(item.get("problem") or "")
                if text and text not in issues:
                    issues.append(text)
        for key, value in (source.get("dimension_scores") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                dimensions[str(key)] = float(value)
        candidate = str(source.get("status") or "pass")
        if _STATUS_RANK.get(candidate, 0) > _STATUS_RANK.get(status, 0):
            status = candidate
    return issues, dimensions, status


def _with_gate_findings(result: dict[str, Any], gate: dict[str, Any] | None, grounding_block: dict[str, Any] | None, *, role: str) -> dict[str, Any]:
    """Fold the deterministic findings into a scored answer, keeping the model score.

    The merged ``blocking_issues`` list mixes two instruments: the model rewrites its half on
    every call, the gate's half is code reading one figure. ``deterministic_blockers`` keeps
    the gate's half addressable, so a later comparison between two answers can ask which
    defects are new instead of whether the sentence count went up.
    """

    issues, dimensions, status = _gate_findings(gate, grounding_block)
    deterministic = list(dict.fromkeys([
        *[str(item) for item in (gate or {}).get("deterministic_blockers") or []],
        *[str(item) for item in (grounding_block or {}).get("deterministic_blockers") or []],
    ]))
    existing = [str(item) for item in result.get("blocking_issues") or []]
    merged = list(dict.fromkeys([*existing, *issues]))
    verdict = max([str(result.get("status") or "pass"), status], key=lambda item: _STATUS_RANK.get(item, 0))
    dimensions = {**dimensions, **(result.get("dimension_scores") or {})}
    counts: dict[str, int] = {}
    for block in (gate, grounding_block):
        for key, value in ((block or {}).get("deterministic_blocker_counts") or {}).items():
            try:
                counts[str(key)] = counts.get(str(key), 0) + int(value)
            except (TypeError, ValueError):
                continue
    return normalize_judge_result({**result, "status": verdict, "blocking_issues": merged, "dimension_scores": dimensions, "deterministic_blockers": deterministic, "deterministic_blocker_counts": counts}, role=role)


def _grounding_block(judge_input: DirectJudgeInput, role: str) -> dict[str, Any] | None:
    """Name what the Grounding Gate measured, without choosing a repair.

    Heuristic judges use this as their whole answer, because they have no model. VLM judges
    fold it into the model reply: the missing binding stays a named defect, but the repair
    comes from the preview, not from a hardcoded ``reanalyze_semantics``.
    """

    gate = judge_input.grounding_gate
    if not isinstance(gate, dict) or gate.get("status") == "pass":
        return None
    issues = list(gate.get("blocking_issues") or ["Grounding Gate did not pass."])
    return normalize_judge_result({
        "overall_score": 0.0,
        "status": "blocked",
        "confidence": 1.0,
        "judge_role": role,
        "gate_only": True,
        "blocking_issues": issues,
        # The gate is code reading the contract, so this finding is reproducible and belongs
        # with the other defects a comparison may hold one answer to.
        "deterministic_blockers": ["grounding_gate"],
        "dimension_scores": {"grounding_gate": 0.0},
        "stage_feedback": [{
            "scope_type": "semantic",
            "problem": "Graph Grounding Gate found unbound scientific objects.",
            "evidence": issues,
            "requested_actions": [],
        }],
    }, role=role)


class DirectJudgeAdapter(Protocol):
    role: str

    def evaluate(self, judge_input: DirectJudgeInput) -> dict[str, Any]: ...

    def reflect(self, outcome: dict[str, Any]) -> dict[str, Any]: ...

    def reference_gap(self, gap_input: "ReferenceGapInput") -> dict[str, Any]: ...


class HeuristicDirectJudge:
    """Deterministic hard gate used before every model-assisted judgement."""

    def __init__(self, *, frozen: bool = False):
        self.frozen = bool(frozen)
        self.role = "frozen_heuristic" if frozen else "online_heuristic"

    def evaluate(self, judge_input: DirectJudgeInput) -> dict[str, Any]:
        blocked = _grounding_block(judge_input, self.role)
        if blocked is not None:
            # No model to ask, so the named defect is the whole answer and the only
            # repair this role can propose is to re-read the paper. VLM judges do not
            # take this branch: they merge the finding and let the model choose.
            feedback = [{**item, "requested_actions": ["reanalyze_semantics"]} if isinstance(item, dict) else item for item in (blocked.get("stage_feedback") or [])]
            return normalize_judge_result({**blocked, "stage_feedback": feedback}, role=self.role)
        result = judge_direct_run(
            judge_input.method,
            judge_input.layout,
            judge_input.icons,
            program=judge_input.program,
            mode="heuristic",
            frozen=self.frozen,
        )
        dimensions = result.get("dimension_scores") if isinstance(result.get("dimension_scores"), dict) else {}
        # Every blocker this role finds is one a rerun would find again, so all of them are
        # addressable. The ids are taken before normalization, which strips the per-entry
        # repair action from an acceptance answer.
        feedback = result.get("stage_feedback")
        dimensions = {**dimensions, "deterministic_gate": result.get("overall_score", 0.0)}
        # The absolute reading of this figure, on the one ruler every state is measured by: the
        # mean of the gate's own dimensions. The seat that answers the pair question produces no
        # score of its own, and without this a round decided by a comparison carried nothing
        # comparable to the round before it.
        measured = [
            float(value)
            for key, value in dimensions.items()
            if key != "deterministic_gate" and isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        result.update({
            "judge_role": self.role,
            "trained": False,
            "dimension_scores": dimensions,
            "absolute_score": round(sum(measured) / len(measured), 4) if measured else None,
            "deterministic_blockers": deterministic_blocker_ids(feedback),
            "deterministic_blocker_counts": deterministic_blocker_counts(feedback),
        })
        return normalize_judge_result(result, role=self.role)

    def reflect(self, outcome: dict[str, Any]) -> dict[str, Any]:
        before = float((outcome.get("before") or {}).get("overall_score") or 0.0)
        after = float((outcome.get("after") or {}).get("overall_score") or 0.0)
        blockers = _new_blocker_count(outcome.get("new_blockers"))
        improved = after > before and not blockers
        return normalize_reflection({
            "outcome": "improved" if improved else "regressed",
            "failure_type": "new_blocker" if blockers else "no_target_improvement" if not improved else "none",
            "why_changed": "Target metrics improved without deterministic regression." if improved else "The repair did not improve the target score or introduced a deterministic blocker.",
            "scope_assessment": "too_broad" if outcome.get("preserve_failures") else "appropriate",
            "next_policy": "keep" if improved else "narrow_scope",
        })


    def reference_gap(self, gap_input: ReferenceGapInput) -> dict[str, Any]:
        """A seat with no model has nothing to say about a good figure, and says so."""

        return _reference_gap_unavailable(self, "this seat has no model to ask")
#: One process holds at most one copy of each Judge checkpoint. Both local seats read the same
#: weights, and a second copy is not a second opinion: it is the same answer for twice the
#: memory, which on a shared GPU is the difference between a measured round and an out-of-memory
#: round. The key is the pair that decides what gets loaded, so two different checkpoints still
#: get one copy each and a seat that names no checkpoint still loads nothing.
_JUDGE_BACKEND_CACHE: dict[tuple[str, str], Any] = {}
_JUDGE_BACKEND_LOAD_LOCK = threading.RLock()

#: Generation is serialized across seats and across the papers of a wave. A backend holds one set
#: of weights and one cache; two threads stepping it at once would interleave their forwards, and
#: the paper running beside another would pay for the other's tokens in memory it does not have.
#: The lock is reentrant so a seat that reflects while it evaluates cannot deadlock itself.
_JUDGE_GENERATE_LOCK = threading.RLock()


def _judge_backend_key(checkpoint: str, base_model: str) -> tuple[str, str]:
    """The pair that decides which weights a seat reads, compared as the caller wrote it."""

    return (str(base_model or ""), str(checkpoint or ""))


def load_local_judge_backend(checkpoint: str, base_model: str, *, seat: str = "acceptance") -> tuple[Any, str]:
    """Load a Judge checkpoint in process, returning the backend and any load error.

    Both local seats read the same weights the same way, so the way they load cannot drift
    apart: 4-bit, gradient checkpointing on, one device. The failure text names the seat the
    caller asked about, because "no acceptance checkpoint was given" is not what an advisory
    seat should say about itself.

    A checkpoint this process already holds is handed back instead of loaded again: the two
    seats name the same weights in the shipped configuration, so a second load would spend the
    memory of a whole extra model to answer the same question with the same weights.
    """

    if not str(checkpoint or "").strip():
        return None, f"no {seat} checkpoint was given"
    if not str(base_model or "").strip():
        return None, f"no base model was given for the {seat} checkpoint"
    key = _judge_backend_key(checkpoint, base_model)
    with _JUDGE_BACKEND_LOAD_LOCK:
        cached = _JUDGE_BACKEND_CACHE.get(key)
        if cached is not None:
            return cached, ""
        try:
            from ..judge_training.training import TrainingConfig, TransformersVlmBackend

            # Gradient checkpointing is what puts this 4-bit model on GPU 0.
            # The default config leaves the device map empty and the load stays on CPU.
            config = TrainingConfig(
                dataset="",
                out_dir="",
                base_model=str(base_model),
                objective="sft",
                max_steps=0,
                gradient_checkpointing=True,
            )
            backend = TransformersVlmBackend(config)
            backend.load(str(checkpoint))
        except Exception as exc:  # noqa: BLE001 - a load failure is a recorded reason, not a crash
            return None, str(exc)
        _JUDGE_BACKEND_CACHE[key] = backend
        return backend, ""


#: How long an advisory answer may be. The pair question asks for a preference, a diagnosis and
#: the stage feedback that follows from them, and a checkpoint that names several panels can pass
#: the ordinary generation budget while it is still inside its object -- measured on the live run
#: twice, as "Unterminated string starting at line 1 column 4510" and as "Expecting ',' delimiter
#: at line 1 column 6002". The budget is raised here rather than at every call site so the two
#: seats cannot drift apart.
ADVISORY_MAX_NEW_TOKENS = 3072

#: What the seat is told when its own answer was too long to finish. A cut-off answer is not a
#: silent seat: it had something to say and said too much, so the retry asks for the same object
#: with one finding instead of several.
ADVISORY_BREVITY_RETRY = (
    "\n\nYour previous answer was cut off before it finished. Return one complete JSON object and "
    "nothing else, with at most one entry in each list: the single most important finding, stated "
    "once, in the same schema."
)

#: What the seat is told when its own reflection was too long to finish. Same reasoning as
#: ``ADVISORY_BREVITY_RETRY``: an answer that ran past its budget is not a silent seat, so it is
#: asked once more for the same object with a single finding rather than being recorded as empty.
REFLECTION_BREVITY_RETRY = (
    "\n\nYour previous answer was cut off before it finished. Return one complete JSON object and "
    "nothing else, in the same schema, with only the single most important finding stated once."
)

#: What the seat is told when it named a fault but not where the fault entered. The boundary
#: review's answer is read for one thing above all: the stage whose requirement went unwritten,
#: with the number that says so. An answer that names the defect and stops leaves the loop with
#: nothing to aim at, and the round it decides then does the thing that helps least -- it draws
#: the figure again. One more call costs a minute next to the hour the round spends, so a seat
#: that answered without saying where is asked once more with the three fields spelled out, and a
#: second answer that still says nothing is recorded as undefined rather than filled in for the
#: model.
ADVISORY_LOCALISATION_RETRY = (
    "\n\nYour previous answer named a defect but not where it entered. Answer the same question "
    "again and fill in all three fields: fault_stage is the stage whose own requirement went "
    "unwritten, fault_requirement is that requirement in one sentence, and fault_evidence cites "
    "at least one number this round put in front of you, spelled as the block you read it in "
    "spells it. A defect that entered nowhere in particular is not an answer here: name the stage "
    "the figure was built wrongly at, even when you can only say which one it was."
)

#: Why a seat had nothing to say. The loop reads the kind rather than the sentence, because the
#: two kinds want opposite treatment: an answer that could not be read is worth asking for once
#: more, while a checkpoint that is not there will not arrive on a second call.
SEAT_NO_BACKEND = "no_backend"
SEAT_UNREADABLE = "unreadable"
SEAT_RAISED = "raised"

#: What the seat is told when its own reflection contradicts itself. It is not told which of
#: the two answers to keep -- choosing for it would put the pipeline's guess in the model's
#: mouth -- only that the two cannot both be true of one round.
REFLECTION_CONFLICT_RETRY = (
    "\n\nYour previous answer contradicted itself: it named an improvement and also named "
    "something that regressed, and one round cannot have done both. Answer the same question "
    "again and keep outcome, defects and failure_type consistent with each other."
)

#: The other self-contradiction a reflection can state: an improvement, plus a doubt about the very
#: measurement that improvement rests on. The seat is told the same thing -- one of the two cannot
#: stand beside the other -- and again is not told which of the two to keep.
REFLECTION_NOISE_CONFLICT_RETRY = (
    "\n\nYour previous answer contradicted itself: it named an improvement and also said the "
    "verdict that improvement rests on cannot be trusted, and one round cannot have done both. "
    "Answer the same question again and keep outcome, defects, failure_type and "
    "measurement_noise_risk consistent with each other."
)


#: What the seat is told when it explained a round and stated no rule. The question asks for a
#: rule whenever the round taught something about how figures are drawn, and a round that drew or
#: moved a figure has at least one such thing to teach, so an empty rule is a missing answer
#: rather than a finding. The seat is asked once more with what a rule looks like spelled out; a
#: second empty answer is kept and recorded, never filled in on the model's behalf.
REFLECTION_RULE_MISSING_RETRY = (
    "\n\nYour previous answer stated no rule. This round drew or moved a figure, so say what it "
    "taught about how such a figure is drawn: set rule to {\"type\": \"router_constraint\", "
    "\"constraints\": {\"shared_vertical_channel\": true, \"forbid_crossing\": true}} or to "
    "{\"type\": \"layout_constraint\", \"params\": {\"align_grid\": 0.05, \"gutter_min\": 0.02}}, "
    "carrying no object id and no coordinate, and set next_policy to match. Keep every other field "
    "of the answer as it is."
)


#: What the seat is told when it explained the round but wrote no patch for it. A patch is the half
#: of a lesson a stage prompt can carry, and the seat that named which stage's requirement went
#: unwritten -- the same answer, the same evidence -- has already written it. The seat is asked once
#: more with the three fields spelled out; a second empty patch is kept and recorded as
#: ``prompt_patch_missing``, never filled in on the model's behalf.
REFLECTION_PATCH_MISSING_RETRY = (
    "\n\nYour previous answer named what went wrong but wrote no prompt_patch. A stage prompt "
    "cannot carry a rule's parameters, so say the sentence it should have carried: set "
    "prompt_patch to {\"apply_at\": the stage whose requirement went unwritten, "
    "\"directive\": that requirement in one sentence, carrying no object id and no coordinate, "
    "\"condition\": the defect patterns it is about}. Keep every other field of the answer as it is."
)


def reflection_states_no_rule(reflection: dict[str, Any] | None) -> bool:
    """Whether a reflection stated no rule a later paper could be drawn by."""

    rule = (reflection or {}).get("rule")
    return not (isinstance(rule, dict) and rule)


def reflection_states_no_patch(reflection: dict[str, Any] | None) -> bool:
    """Whether a reflection wrote no patch sentence a later prompt could carry."""

    patch = (reflection or {}).get("prompt_patch")
    if not isinstance(patch, dict):
        return True
    return not (str(patch.get("apply_at") or "").strip() and str(patch.get("directive") or "").strip())


def reflection_conflict_retry(reflection: dict[str, Any]) -> str:
    """Pick the sentence that names this row's own contradiction."""

    if str(reflection.get("outcome") or "").strip() == "improved" and bool(reflection.get("measurement_noise_risk")):
        return REFLECTION_NOISE_CONFLICT_RETRY
    return REFLECTION_CONFLICT_RETRY


def generate_local_judge(backend: Any, record: dict[str, Any], *, max_new_tokens: int = 1024) -> str:
    """Run one generation on a shared local backend, one call at a time.

    The backend is process-wide, so this is where "one model, one generation" is enforced. A
    caller that injected its own generator keeps its own behaviour: only the local arm passes
    through here, and the acceptance seat's pair comparison and the advisory seat's answer queue
    behind the same lock rather than through two copies of the same weights.
    """

    with _JUDGE_GENERATE_LOCK:
        return backend.generate(record, max_new_tokens=int(max_new_tokens))


def local_judge_generator(backend: Any, *, max_new_tokens: int = 1024) -> Callable[[str, list[Any]], dict[str, Any]]:
    """Turn an in-process backend into the ``(prompt, images) -> parsed JSON`` a seat asks through.

    A served endpoint and a local checkpoint answer the same question, so the seats call both
    the same way; this is the local arm of :func:`call_vlm_json`. It answers with the parsed
    object rather than the raw completion, because that is what the reader of an answer wants
    and what the served arm already hands back.
    """

    def ask(prompt: str, images: list[Any]) -> dict[str, Any]:
        paths = [str(Path(item)) for item in images if item and Path(item).exists()]
        content: list[dict[str, Any]] = [{"type": "text", "text": str(prompt)}]
        content.extend({"type": "image", "image": path} for path in paths)
        record = {"messages": [{"role": "user", "content": content}], "images": paths}
        return extract_json(str(generate_local_judge(backend, record, max_new_tokens=int(max_new_tokens))))

    return ask


class BootstrapVLMDirectJudge:
    """Ordinary VLM placeholder implementing the future trained-Judge contract."""

    def __init__(self, *, model: str | None = None, frozen: bool = False, timeout: int = 180, retries: int = 1, trained: bool = False, ask_json: Any = None):
        self.model = resolve_vlm_model("RFS_FROZEN_JUDGE_MODEL" if frozen else "RFS_ONLINE_JUDGE_MODEL", explicit_model=model)
        self.frozen = bool(frozen)
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.trained = bool(trained)
        self.role = ("frozen_" if frozen else "online_") + ("trained" if trained else "bootstrap")
        self.gate = HeuristicDirectJudge(frozen=frozen)
        #: Where the answer comes from. ``None`` is the served endpoint, which is what every
        #: run did before a checkpoint could be seated locally; a caller that seats a local
        #: checkpoint passes its own ``(prompt, images) -> parsed JSON`` here instead.
        self._ask_json = ask_json

    def _ask(self, prompt: str, images: list[Any]) -> dict[str, Any]:
        """Ask whichever model backs this seat: the served endpoint, or the caller's own."""

        if self._ask_json is None:
            return call_vlm_json(prompt, images, model=self.model, timeout=self.timeout, retries=self.retries)
        return self._ask_json(prompt, images)

    def evaluate(self, judge_input: DirectJudgeInput) -> dict[str, Any]:
        blocked = _grounding_block(judge_input, self.role)
        gate = self.gate.evaluate(judge_input)
        # Grounding Gate used to return here for the advisory role with a hardcoded
        # ``reanalyze_semantics``. That stole the repair: the model never saw the preview, so
        # arrow-routing defects could not become actions, and the loop spent its rounds re-reading
        # the paper. The gate still runs and its findings are still merged into the answer; it
        # no longer decides whether either role gets to speak. Heuristic judges keep the
        # short-circuit, because they have no model to ask.
        images, comparison = _judge_images(judge_input, role=self.role)
        prompt = canonical_prompt(online_judge_row(judge_input, role=self.role, comparison=comparison))
        raw = self._ask(prompt, images)
        result = normalize_judge_result(raw, role=self.role)
        result = _with_gate_findings(result, gate, blocked, role=self.role)
        result.update({"judge_role": self.role, "model": self.model, "trained": self.trained, "gate_only": False})
        result.update({"prompt_version": prompt_version()})
        return result

    def evaluate_pairwise(self, judge_input: DirectPairJudgeInput) -> dict[str, Any]:
        """Ask which of two states is better instead of asking twice for a number.

        Both previews go into one call, because "which of these two is better" is a question
        the model answers consistently while "what number out of 1 is this" is not: across 74
        answers the served model used only ten distinct values, so subtracting two of them
        measured its rounding rather than the difference between the figures. The deterministic
        gate still runs and its blockers are still folded in; it just no longer answers this
        question, because the rounds where the figure is worst are the ones a comparison is for.

        The grounding gate still folds its blockers into the answer. It does not refuse the
        comparison: "the gate has a finding" is not "the two figures are indistinguishable",
        and a missing binding must not hide a routing defect the previews already show.
        """

        before, after = judge_input.before, judge_input.after
        blocked = _grounding_block(after, self.role)
        gate = self.gate.evaluate(after)
        gate_issues, _, gate_status = _gate_findings(gate, blocked)
        images = [str(path) for path in (before.preview, after.preview) if path and Path(path).exists()]
        if len(images) < 2:
            # Nothing was compared, so nothing is claimed: the caller keeps its measured
            # comparison and the record says why this answer is missing.
            reason = "a preview of one of the two states is missing"
            issues = list(gate_issues)
            return {
                **normalize_pairwise_result({}, role=self.role, candidate_a=judge_input.candidate_a, candidate_b=judge_input.candidate_b),
                "judge_role": self.role,
                "model": self.model,
                "trained": self.trained,
                "gate_only": True,
                "pairwise_unavailable": reason,
                "blocking_issues": issues,
                "blocking_issue_count": len(issues),
                "deterministic_status": gate_status,
                "prompt_version": prompt_version(),
            }
        prompt = canonical_pairwise_prompt(online_judge_row(after, role=self.role, pairwise=True))
        raw = self._ask(prompt, images)
        result = normalize_pairwise_result(raw, role=self.role, candidate_a=judge_input.candidate_a, candidate_b=judge_input.candidate_b)
        model_issues = [str(item) for item in result.get("blocking_issues") or []]
        issues = list(dict.fromkeys([*model_issues, *gate_issues]))
        return {
            **result,
            "judge_role": self.role,
            "model": self.model,
            "trained": self.trained,
            "gate_only": False,
            "deterministic_status": gate_status,
            "blocking_issues": issues,
            "blocking_issue_count": len(issues),
            "prompt_version": prompt_version(),
        }

    def reference_gap(self, gap_input: ReferenceGapInput) -> dict[str, Any]:
        """Ask where our figure falls short of a good one, when this seat is the advisory one.

        This is the only question this seat asks about two figures that are not two versions of
        the same paper: a published figure of this kind, and ours. It is refused for every other
        seat, because the answer is a preference and a preference inside a measurement is how a
        score stops being a measurement.
        """

        if judge_role_family(self.role) != ADVISORY_ROLE_FAMILY:
            return _reference_gap_unavailable(self, "this seat measures; it is not shown a reference figure")
        prompt = build_reference_gap_prompt(gap_input.reference_id, gap_input.reference_note, gap_input.ours_note or "our own output")
        raw = self._ask(prompt, [str(gap_input.reference_path), str(gap_input.ours_path)])
        return {
            **normalize_reference_gap(raw),
            "judge_role": self.role,
            "model": self.model,
            "trained": self.trained,
            "reference_id": str(gap_input.reference_id or ""),
            "prompt_version": prompt_version(),
        }
    def reflect(self, outcome: dict[str, Any]) -> dict[str, Any]:
        raw = self._ask(canonical_reflection_prompt(outcome), [])
        return normalize_reflection(raw)


class TrainedDirectJudge(BootstrapVLMDirectJudge):
    """Adapter for a trained Judge served through the same JSON API contract."""

    def __init__(self, **kwargs: Any):
        super().__init__(trained=True, **kwargs)


_CLEAN_DIMENSIONS = {
    "scientific_fidelity": 1.0,
    "logical_flow": 1.0,
    "layout_quality": 1.0,
    "readability": 1.0,
    "domain_fit": 1.0,
}


def choice_pair_judgement(text: str) -> dict[str, Any]:
    """Map a two-field short answer onto a preference and defect-table scores.

    The model names the wrong figure and one defect. The program writes that
    defect's scores onto the wrong figure. The other figure does not carry the
    defect. ``no_defect`` uses the ``assumed_worse`` row already in the table.

    Every number here is looked up in the defect table, so it is labelled
    ``derived_from_defect``: it says which defect was named, not how good the figure
    is. It is kept for the report and never decides an accept, a rollback or a ranking.

    The same answer may also say where the round stands. That half is read here and carried out
    under ``direction``, but it decides nothing: the comparison it came in with is scored exactly
    as it was, and a direction half that is missing or unreadable leaves the pair answer alone.
    """

    from ..judge_training.choice_items import CHOICE_DEFECTS, parse_choice_answer
    from ..judge_training.judge_schema import DEFECT_LABELS

    parsed = parse_choice_answer(text, CHOICE_DEFECTS)
    if not parsed:
        return {"pairwise_unavailable": "the choice answer was not a two-field short answer"}
    wrong, defect = str(parsed["wrong_figure"]), str(parsed["defect"])
    label = DEFECT_LABELS["assumed_worse"] if defect == "no_defect" else DEFECT_LABELS.get(defect)
    if label is None:
        return {"pairwise_unavailable": f"the defect {defect} is not in the defect table"}
    wrong_scores = {str(key): float(value) for key, value in dict(label["quality_scores"]).items()}
    clean = dict(_CLEAN_DIMENSIONS)
    before_scores = wrong_scores if wrong == "A" else clean
    after_scores = wrong_scores if wrong == "B" else clean

    def side(scores: dict[str, float]) -> dict[str, Any]:
        return {
            "overall_score": scores["scientific_fidelity"],
            "dimension_scores": scores,
            "score_source": DERIVED_SCORE_SOURCE,
        }

    return {
        "preferred_candidate": "B" if wrong == "A" else "A",
        "margin": "clear",
        "wrong_figure": wrong,
        "defect": defect,
        "choice_scores": {"before": side(before_scores), "after": side(after_scores)},
        # What the seat said, in words, so a reader never has to take the looked-up numbers as a
        # measurement. The scores above say which defect was named; this says that is what they
        # are, and no accept, rollback or ranking is allowed to read them.
        "verdict_reason": (
            f"the acceptance seat called {wrong} the wrong figure and named the defect class "
            f"'{defect}'; nothing was measured, so no score was taken"
        ),
        "direction": dict(parsed.get("direction") or {}),
        "direction_problem": str(parsed.get("direction_problem") or ""),
    }


#: The defect classes that say something was lost from the figure: a module or a stage the paper
#: has and the figure does not, a module it never had, a term that came back changed. Every one of
#: them is a claim about what the figure draws, and every one of them can be checked against what
#: the two figures draw -- which is what the re-ask below does.
CONTENT_LOSS_DEFECTS = frozenset({"missing_core_entity", "missing_core_stage", "invented_module", "terminology_corruption", "training_inference_confusion"})


def drawn_content(program: dict[str, Any] | None) -> dict[str, Any]:
    """What one figure draws, as a reader would list it: its module labels and its connections.

    Labels rather than ids, because a rebuild rewrites ids and what a reader sees does not move.
    """

    body = program if isinstance(program, dict) else {}
    labels = {str(item.get("label") or item.get("id") or "").strip() for item in body.get("semantic_nodes") or [] if isinstance(item, dict)}
    pairs = set()
    for arrow in body.get("arrows") or []:
        if isinstance(arrow, dict):
            pairs.add((str(arrow.get("source") or ""), str(arrow.get("target") or "")))
    labels.discard("")
    return {"labels": sorted(labels), "connections": sorted(pairs)}


def content_loss_contradicted(before: "DirectJudgeInput | None", after: "DirectJudgeInput | None", defect: str) -> str:
    """The evidence sentence when a content-loss answer cannot be true of two figures, or ``""``.

    A seat that says a module is missing while both figures draw the same modules, and the same
    connections between them, has contradicted something the loop can see. The sentence it is
    handed names what both figures draw, so the second answer is about the figures rather than
    about the first answer.
    """

    if str(defect or "") not in CONTENT_LOSS_DEFECTS:
        return ""
    left = drawn_content(getattr(before, "program", None))
    right = drawn_content(getattr(after, "program", None))
    if not left["labels"] and not right["labels"]:
        return ""
    if left != right:
        return ""
    modules = ", ".join(right["labels"])
    return (
        f"Check this before you answer again. Both figures draw the same {len(right['labels'])} "
        f"modules ({modules}) and the same {len(right['connections'])} connections between them: "
        "nothing was added to one of them or dropped from it. A defect that says something is "
        "missing or was invented cannot separate these two figures, so name the defect that does."
    )


#: The two states' own readings, in the order the question shows them. They are the loop's own
#: measurement of each figure, and the seat that judges the pair is shown them as evidence: a
#: figure that carries more crossings than the other is a fact about the drawing, and the seat
#: weighs it rather than obeying it.
def pair_readings(before: "DirectJudgeInput | None", after: "DirectJudgeInput | None") -> str:
    """What the loop measured on each of the two states, on one line each."""

    rows = [(label, _state_readings(side)) for label, side in (("A", before), ("B", after))]
    return ". ".join(f"{label}: " + ", ".join(f"{key}={value}" for key, value in reading.items()) for label, reading in rows if reading)


def _state_readings(judge_input: "DirectJudgeInput | None") -> dict[str, Any]:
    """The deterministic reading of one state, taken from its own figure program.

    Read here rather than passed in, so the two sides of every comparison are measured by the same
    code on the same numbers, and a caller that has no program to show simply adds nothing.
    """

    program = judge_input.program if isinstance(getattr(judge_input, "program", None), dict) else {}
    if not program:
        return {}
    return program_readings(program)


def program_readings(program: "dict[str, Any] | None") -> dict[str, Any]:
    """The deterministic reading of one figure program, or ``{}`` when there is none to read.

    Public because the validation harness measures its two arms the same way the loop measures
    the two states of a round: the numbers a reader is shown have to come from one piece of code,
    or two comparisons of the same figure can disagree about how many lines cross.
    """

    if not isinstance(program, dict) or not program:
        return {}
    from ..composition.arrow_routing import count_connector_crossings
    from .geometry_quality import evaluate_geometry_quality

    geometry = evaluate_geometry_quality(program) or {}
    readings = {
        "crossings": int(count_connector_crossings(program) or 0),
        "box_passes": int(geometry.get("box_pass_count") or 0),
        "boundary_violations": int(geometry.get("node_boundary_violation_count") or 0),
        "icon_text_overlaps": int(geometry.get("icon_text_overlap_count") or 0),
        "node_overlaps": int(geometry.get("node_overlap_count") or 0),
        "occupancy": round(float(geometry.get("canvas_occupancy") or 0.0), 4),
    }
    return readings


class ChoiceAcceptanceJudge:
    """Acceptance-only judge backed by a local choice checkpoint.

    It answers which of two figures is wrong and which defect that figure has.
    It does not propose a repair. A load failure or an illegal answer is an
    unavailable comparison, not a fall back to the long Judge JSON prompt.
    """

    def __init__(self, *, checkpoint: str = "", base_model: str = "", generate: Any = None, direction: str = "off"):
        self.checkpoint = str(checkpoint or "")
        self.base_model = str(base_model or "")
        self.role = "frozen_choice"
        self.trained = True
        self.model = self.base_model
        self._generate = generate
        self._backend: Any = None
        self._load_error = ""
        chosen = str(direction or "").strip().casefold()
        self.direction = chosen if chosen in self.DIRECTIONS else "off"
        if generate is None:
            self._load_error = self._load()

    #: How the acceptance seat is asked about the round's own layer. ``off`` keeps the two-field
    #: question it was trained on; ``record`` asks the extra question and files the answer without
    #: acting on it; ``authority`` lets that answer move the work face. The two-field half of the
    #: answer is read and scored the same way in all three.
    DIRECTIONS = ("off", "record", "authority")

    def _load(self) -> str:
        self._backend, error = load_local_judge_backend(self.checkpoint, self.base_model, seat="acceptance")
        return error

    def evaluate(self, judge_input: DirectJudgeInput) -> dict[str, Any]:
        """Answer no score of its own, and keep the deterministic gate's measurement beside it.

        This seat is asked the two-state question, so its ``overall_score`` stays empty and no
        round is ever decided by a number this seat did not produce. It does run the
        deterministic gate first -- that is where its status and its blockers come from -- and
        the gate's own score is kept under ``gate_score``. Without it, a round whose pair
        question could not be asked (a first round with nothing to compare against) carried no
        measurement at all: the gate had measured the figure, the score was thrown away, and the
        round could never become Memory however well it had been measured. The gate's per-kind
        counts ride along with its names for the same reason the score does: they are the same
        measurement, and dropping them made the comparison read this seat's answer as one that
        had found nothing.
        """

        gate = HeuristicDirectJudge(frozen=True).evaluate(judge_input)
        blockers = list(gate.get("deterministic_blockers") or [])
        # The gate separates its defects into counts, and the seat has to pass that half on: the
        # comparison between two rounds reads how many times each kind was found, and a seat that
        # forwarded only the names left the other side of the comparison with nothing to weigh --
        # "no count reported" then read as "no defect introduced".
        blocker_counts = {
            str(key): int(count)
            for key, count in (gate.get("deterministic_blocker_counts") or {}).items()
            if isinstance(count, (int, float)) and not isinstance(count, bool)
        }
        gate_score = gate.get("overall_score")
        if isinstance(gate_score, bool) or not isinstance(gate_score, (int, float)):
            gate_score = None
        return normalize_judge_result(
            {
                "status": gate.get("status") or "fault",
                "blocking_issues": list(gate.get("blocking_issues") or []),
                "stage_feedback": [],
                "deterministic_blockers": blockers,
                "deterministic_blocker_counts": blocker_counts,
                "overall_score": None,
                # The gate's measurement of this same figure, labelled as the gate's. The pair
                # comparison overwrites ``overall_score`` when it runs; this stays whatever
                # happened, so the round always carries the cheap deterministic reading of the
                # figure it kept.
                "gate_score": gate_score,
                "gate_dimension_scores": dict(gate.get("dimension_scores") or {}),
                # The same deterministic reading, under the name every decision path reads a
                # measurement by: the pair comparison overwrites no score of the model's, and
                # this one is code reading the figure.
                "absolute_score": gate.get("absolute_score"),
                "judge_role": self.role,
                "trained": True,
                "model": self.model,
                "checkpoint": self.checkpoint,
            },
            role=self.role,
        )

    def evaluate_pairwise(self, judge_input: DirectPairJudgeInput) -> dict[str, Any]:
        before, after = judge_input.before, judge_input.after
        images = [str(path) for path in (before.preview, after.preview) if path and Path(path).exists()]
        unavailable = self._load_error
        if len(images) < 2:
            unavailable = unavailable or "a preview of one of the two states is missing"
        raw: dict[str, Any] = {}
        reask = ""
        if not unavailable:
            from ..judge_training.choice_items import CHOICE_DEFECTS, choice_prompt
            from ..judge_training.prompts import WHOLE_FIRST_RULE

            title = str((after.method or {}).get("title") or "figure")
            nodes = (after.layout or {}).get("nodes") or []
            evidence = ", ".join(
                str(item.get("label") or item.get("id") or "")
                for item in nodes
                if isinstance(item, dict)
            )[:500]
            # The rule, the readings and the round's own layer reach the seat as the question it
            # was asked before they are allowed to change anything it answers: the two fields it
            # has always been scored on are asked in the same words, and the rest is added around
            # them. The rule is the same constant the advisory seat is told, so the two seats
            # cannot answer one pair by two different rules.
            asks_direction = self.direction in {"record", "authority"}
            plan = after.layer_plan if isinstance(after.layer_plan, dict) else {}
            question = choice_prompt(
                title,
                evidence,
                CHOICE_DEFECTS,
                rule=WHOLE_FIRST_RULE,
                readings=pair_readings(before, after),
                layer=str(plan.get("working_on") or ""),
                stages=tuple(str(item) for item in (plan.get("layer_order") or ())),
                direction=asks_direction,
            )
            under_test = str(after.experience_under_test or "").strip()
            if under_test:
                question += f"\nThe change under test in this comparison is: {under_test}"

            def ask(extra: str = "") -> str:
                """Put the question, with anything the loop has to add, and read one answer."""

                content = [
                    {"type": "text", "text": question + (f"\n{extra}" if extra else "")},
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                ]
                record = {"messages": [{"role": "user", "content": content}], "images": images}
                budget = 192 if asks_direction else 64
                return self._generate(record) if self._generate is not None else generate_local_judge(self._backend, record, max_new_tokens=budget)

            try:
                text = ask()
            except Exception as exc:
                unavailable = str(exc)
                text = ""
            if not unavailable:
                raw = choice_pair_judgement(str(text))
                unavailable = str(raw.get("pairwise_unavailable") or "")
                # An answer that says something was lost from a figure both figures draw the same
                # modules and connections of is asked once more, with that evidence in the
                # question. The doubt is the seat's own, so the second answer settles it: a repeat
                # stands as the answer and the round says it was asked twice, and a different
                # answer is the answer. Nothing else about the comparison changes.
                evidence_against = "" if unavailable else content_loss_contradicted(before, after, str(raw.get("defect") or ""))
                if evidence_against:
                    first = str(raw.get("wrong_figure") or "") + "/" + str(raw.get("defect") or "")
                    try:
                        second_text = ask(evidence_against)
                    except Exception as exc:  # noqa: BLE001 - a second failure keeps the first answer
                        reask = f"repeated: the seat could not be asked again ({type(exc).__name__})"
                    else:
                        second = choice_pair_judgement(str(second_text))
                        if second.get("pairwise_unavailable") or str(second.get("defect") or "") in CONTENT_LOSS_DEFECTS:
                            reask = f"repeated: the answer came back as {first}"
                        else:
                            raw = second
                            reask = f"changed: from {first} to {str(second.get('wrong_figure') or '')}/{str(second.get('defect') or '')}"
        result = normalize_pairwise_result(
            raw,
            role=self.role,
            candidate_a=judge_input.candidate_a,
            candidate_b=judge_input.candidate_b,
        )
        return {
            **result,
            "judge_role": self.role,
            "model": self.model,
            "trained": True,
            "checkpoint": self.checkpoint,
            "gate_only": False,
            "pairwise_unavailable": unavailable or None,
            "choice_scores": raw.get("choice_scores") or {},
            "wrong_figure": raw.get("wrong_figure") or "",
            "defect": raw.get("defect") or "",
            "verdict_reason": str(raw.get("verdict_reason") or ""),
            "seat_reasked": reask,
            "direction": dict(raw.get("direction") or {}),
            "direction_problem": str(raw.get("direction_problem") or ""),
            "direction_authority": self.direction,
        }

    def reflect(self, outcome: dict[str, Any]) -> dict[str, Any]:
        return normalize_reflection(default_reflection_from_outcome(outcome))


class CheckpointAdvisoryJudge:
    """Advisory seat backed by a local checkpoint, loaded in process.

    Same question, same prompt and the same contract as the served advisory role; what
    differs is where the answer comes from. Seating a checkpoint here is what makes the
    advice a model gives measurable at all: each round records what it advised, what the loop
    did with that advice and what the comparison said afterwards, and those triples are the
    only material from which advice can later be trained. A checkpoint seated here is
    expected to advise badly at first -- its advice is an outcome to record, not a claim.

    A load failure leaves the seat with nothing to say, and it says exactly that. It does not
    fall back to the served model, because a run whose advisory seat quietly became the vendor
    model is not the run its own record describes.
    """

    def __init__(self, *, checkpoint: str = "", base_model: str = "", ask_json: Any = None, timeout: int = 180, retries: int = 1):
        self.checkpoint = str(checkpoint or "")
        self.base_model = str(base_model or "")
        self.model = self.base_model or self.checkpoint
        self.frozen = False
        self.trained = bool(self.checkpoint)
        self.role = "online_trained" if self.trained else "online_bootstrap"
        self.timeout = int(timeout)
        self.retries = int(retries)
        self.gate = HeuristicDirectJudge()
        self._load_error = ""
        self._ask_json = ask_json
        if self._ask_json is None:
            backend, self._load_error = load_local_judge_backend(self.checkpoint, self.base_model, seat="advisory")
            self._ask_json = local_judge_generator(backend, max_new_tokens=ADVISORY_MAX_NEW_TOKENS) if backend is not None else None

    def evaluate(self, judge_input: DirectJudgeInput, *, extra_instruction: str = "") -> dict[str, Any]:
        """Answer one question about one figure, and say why when there is no answer.

        ``extra_instruction`` is appended after the schema, the way every other re-ask in this
        seat is: the caller that read the first answer knows which half of it was missing, and
        the seat is the one that owns the prompt text. Nothing else about the question changes,
        so a call with no instruction renders the prompt byte for byte as it did before.
        """

        blocked = _grounding_block(judge_input, self.role)
        gate = self.gate.evaluate(judge_input)
        if self._ask_json is None:
            # The weights could not be read, so this seat has no opinion. The reason travels on
            # the answer so the round records "the advisory seat was unavailable" instead of a
            # silent zero-feedback judgement that reads like "it looked and found nothing".
            return self._seat_unavailable(self._load_error or "no local advisory backend", gate, blocked, kind=SEAT_NO_BACKEND)
        images, comparison = _judge_images(judge_input, role=self.role)
        row = online_judge_row(judge_input, role=self.role, comparison=comparison)
        prompt = canonical_prompt(row) + str(extra_instruction or "")
        answer, problem, retried = self._ask_readable(prompt, images)
        if answer is None:
            # A local checkpoint that stops in the middle of its object, or that fails on the
            # device, leaves this round without advice. That is exactly the outcome the round
            # exists to record, and the alternative -- letting the parse error end the whole
            # iteration -- throws away the figure, the review and the hours that produced them.
            result = self._seat_unavailable(f"the advisory answer could not be read: {problem}", gate, blocked, comparison=comparison, kind=SEAT_UNREADABLE)
            result["advisory_parse_retry"] = retried
            return result
        result = normalize_judge_result(answer, role=self.role)
        result = _with_gate_findings(result, gate, blocked, role=self.role)
        result.update(
            {
                "judge_role": self.role,
                "model": self.model,
                "trained": self.trained,
                "gate_only": False,
                "comparison_question": comparison,
                "advisory_parse_retry": retried,
                "prompt_version": prompt_version(),
                # The field names this prompt carried. A caller that has to decide whether the
                # answer named something the figure was actually measured on cannot rebuild
                # this list -- only the seat knows which blocks it rendered -- so it travels on
                # the answer instead of being guessed at.
                "advisory_shown_fields": sorted(citable_fields(row)),
            }
        )
        return result

    def reference_gap(self, gap_input: ReferenceGapInput) -> dict[str, Any]:
        """Ask the local checkpoint where our figure falls short of a good one.

        A checkpoint seated here is the seat this question exists for. Its first answers are
        expected to be poor and they are recorded either way: an answer that names no class is the
        question going unanswered, which is the outcome a round has to be able to see, rather than
        a gap quietly filed as found.
        """

        if self._ask_json is None:
            return _reference_gap_unavailable(self, self._load_error or "no local advisory backend")
        prompt = build_reference_gap_prompt(gap_input.reference_id, gap_input.reference_note, gap_input.ours_note or "our own output")
        raw, problem, retried = self._ask_readable(prompt, [str(gap_input.reference_path), str(gap_input.ours_path)])
        record = normalize_reference_gap(raw if isinstance(raw, dict) else {}, problem=problem, retried=retried)
        record.update(
            {
                "judge_role": self.role,
                "model": self.model,
                "trained": self.trained,
                "reference_id": str(gap_input.reference_id or ""),
                "prompt_version": prompt_version(),
            }
        )
        return record
    def _ask_readable(self, prompt: str, images: list[Any]) -> tuple[dict[str, Any] | None, str, bool]:
        """Ask once, and once more if the first answer could not be read.

        A checkpoint that answers past its budget is cut off inside its object and the answer is
        lost even though the seat did have something to say. One retry that asks for the same
        object with a single finding is cheap next to the round it saves; when the second answer
        cannot be read either, both reasons travel on the result and the seat is reported as
        unreadable rather than as having found nothing.
        """

        try:
            return self._ask_json(prompt, images), "", False
        except Exception as exc:  # noqa: BLE001 - the retry below is what handles it
            first = f"{type(exc).__name__}: {exc}"
        try:
            return self._ask_json(prompt + ADVISORY_BREVITY_RETRY, images), "", True
        except Exception as exc:  # noqa: BLE001 - an unreadable answer is a recorded outcome
            return None, f"{first}; retry: {type(exc).__name__}: {exc}", True

    def _seat_unavailable(self, reason: str, gate: dict[str, Any], blocked: dict[str, Any], *, comparison: bool = False, kind: str = "") -> dict[str, Any]:
        """The shape this seat answers with when it has nothing readable to say.

        Silence is a state the round has to be able to tell apart from agreement, so the reason
        is carried on the answer and the deterministic gate findings stay on it. ``kind`` says
        which silence it was, because the caller treats the two differently: an answer that
        could not be read is worth one more call, and a checkpoint that is not there is not.
        """

        raw: dict[str, Any] = {"stage_feedback": [], "blocking_issues": [reason] if reason else []}
        result = normalize_judge_result(raw, role=self.role)
        result = _with_gate_findings(result, gate, blocked, role=self.role)
        result.update(
            {
                "judge_role": self.role,
                "model": self.model,
                "trained": self.trained,
                "gate_only": True,
                "advisory_unavailable": reason,
                "advisory_unavailable_kind": str(kind or ""),
                "comparison_question": comparison,
                "prompt_version": prompt_version(),
            }
        )
        return result

    def reflect(self, outcome: dict[str, Any]) -> dict[str, Any]:
        """Ask for the reflection, and record one that could not be read as a named outcome.

        The reflection is the quietest call this seat makes -- no round is blocked on it -- so a
        failure here used to leave an empty object behind and the loop moved on without recording
        why. The reason now travels on the reflection, and an answer that was cut off gets the same
        second, briefer attempt the answer path already had: same seat, same question, one finding
        instead of several. When both attempts fail, both reasons are kept.
        """

        if self._ask_json is None:
            return normalize_reflection(default_reflection_from_outcome(outcome))
        prompt = canonical_reflection_prompt(outcome)
        raw, problem, retried = self._ask_reflection_readable(prompt)
        first_read = normalize_reflection(raw) if isinstance(raw, dict) else {}
        if reflection_conflict(first_read):
            # A reflection that says the round improved and also names a regression -- or calls
            # the very measurement that improvement rests on noise -- is a contradiction, not a
            # finding, so the seat is asked once more with the contradiction named. Whatever it
            # answers second is kept, and a row that still contradicts itself travels with
            # ``reflection_conflict`` set rather than being quietly rewritten into agreement.
            second, second_problem, _ = self._ask_reflection_readable(prompt + reflection_conflict_retry(first_read))
            retried = True
            if isinstance(second, dict):
                raw, problem = second, second_problem
        if not isinstance(raw, dict):
            return normalize_reflection(
                {**default_reflection_from_outcome(outcome), "provider_error": problem, "reflection_parse_retry": retried}
            )
        # Two parts of a reflection are drawn on later, and the question asks for both whenever the
        # round taught something: the rule, which another paper is drawn by, and the patch, which a
        # later stage prompt carries. A round that drew or moved a figure did teach something, so an
        # answer that states neither is a missing answer, not a finding: it is asked once more, with
        # each missing field spelled out. One retry covers both, so a round that left both out is
        # asked once and not twice. What the second answer says is kept either way, and a field still
        # empty after it travels as ``rule_missing`` or ``prompt_patch_missing`` so the run records
        # it rather than quietly reading the round as having taught nothing.
        rule_missing = reflection_states_no_rule(normalize_reflection(raw))
        patch_missing = reflection_states_no_patch(normalize_reflection(raw))
        if rule_missing or patch_missing:
            nudge = (REFLECTION_RULE_MISSING_RETRY if rule_missing else "") + (
                REFLECTION_PATCH_MISSING_RETRY if patch_missing else ""
            )
            second, second_problem, _ = self._ask_reflection_readable(prompt + nudge)
            retried = True
            if isinstance(second, dict) and second:
                # Only a second answer that arrived replaces the first: an empty object is the seat
                # saying nothing the second time, and a reply that said nothing does not overwrite
                # one that said something.
                raw, problem = second, second_problem
                rule_missing = reflection_states_no_rule(normalize_reflection(raw))
                patch_missing = reflection_states_no_patch(normalize_reflection(raw))
        return normalize_reflection(
            {
                **raw,
                "reflection_parse_retry": retried,
                **({"rule_missing": True} if rule_missing else {}),
                **({"prompt_patch_missing": True} if patch_missing else {}),
            }
        )

    def _ask_reflection_readable(self, prompt: str) -> tuple[dict[str, Any] | None, str, bool]:
        """Ask for the reflection once, and once more if the first answer could not be read."""

        try:
            return self._ask_json(prompt, []), "", False
        except Exception as exc:  # noqa: BLE001 - the retry below is what handles it
            first = f"{type(exc).__name__}: {exc}"
        try:
            return self._ask_json(prompt + REFLECTION_BREVITY_RETRY, []), "", True
        except Exception as exc:  # noqa: BLE001 - an unreadable reflection is a recorded outcome
            return None, f"{first}; retry: {type(exc).__name__}: {exc}", True


def default_reflection_from_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    decision = outcome.get("decision") if isinstance(outcome.get("decision"), dict) else {}
    accepted = bool(decision.get("accepted"))
    return {
        "outcome": "improved" if accepted else "regressed",
        "failure_type": "none" if accepted else str(decision.get("reason") or "not_preferred"),
        "why_changed": "The acceptance checkpoint preferred the revised figure." if accepted else "The acceptance checkpoint did not prefer the revised figure.",
        "scope_assessment": "appropriate",
        "next_policy": "keep" if accepted else "narrow_scope",
    }


def build_direct_judges(*, mode: str, online_model: str | None = None, frozen_model: str | None = None, timeout: int = 180, retries: int = 1, acceptance_checkpoint: str = "", acceptance_base_model: str = "", acceptance_generate: Any = None, advisory_checkpoint: str = "", advisory_base_model: str = "", advisory_ask: Any = None, acceptance_direction: str = "off") -> tuple[DirectJudgeAdapter, DirectJudgeAdapter]:
    """Build the two seats. A seat with a checkpoint of its own no longer calls the endpoint.

    The advisory seat is the one that proposes; the acceptance seat is the one that measures.
    Either can be backed by a local checkpoint instead of the served model, and the two
    decisions are independent on purpose: seating a checkpoint in the advisory seat is how
    the advice itself becomes something a run can record outcomes for.
    """

    if mode == "heuristic":
        advisory: DirectJudgeAdapter = HeuristicDirectJudge()
        frozen: DirectJudgeAdapter = HeuristicDirectJudge(frozen=True)
    else:
        cls = TrainedDirectJudge if mode == "trained" else BootstrapVLMDirectJudge
        advisory = cls(model=online_model, frozen=False, timeout=timeout, retries=retries)
        frozen = cls(model=frozen_model, frozen=True, timeout=timeout, retries=retries)
    if advisory_checkpoint or advisory_ask is not None:
        advisory = CheckpointAdvisoryJudge(checkpoint=advisory_checkpoint, base_model=advisory_base_model, ask_json=advisory_ask, timeout=timeout, retries=retries)
    if acceptance_checkpoint or acceptance_generate is not None:
        frozen = ChoiceAcceptanceJudge(checkpoint=acceptance_checkpoint, base_model=acceptance_base_model, generate=acceptance_generate, direction=acceptance_direction)
    return advisory, frozen
