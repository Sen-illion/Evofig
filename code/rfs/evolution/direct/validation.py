"""Paired ON/OFF validation and promotion rules for direct evolution experience."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable

from ...analysis.graph.retrieval import CASE_GROUPS
from ...contracts.evolution import normalize_experience, pairwise_verdict

#: Which protocol decided a case. A compile-time rule acts on a figure the compiler is about to
#: draw, so it can be measured by drawing one figure and compiling it twice -- with and without
#: the rule. Everything else has to be measured by running both arms, because the stage it acts
#: at has to plan its own figure. The path is written into every case and into the report, so a
#: reader can tell a two-generation measurement from a two-compilation one.
COMPILE_ONLY_PATH = "compile_only"
PAIRED_RUN_PATH = "paired_run"

#: The profile that runs the four-group set. It is the default, so a caller that names none keeps
#: the ladder exactly as it was.
FULL_PROFILE = "full"
#: The profile that runs one structurally matched paper instead of the set. See
#: ``rfs.analysis.graph.retrieval.QUICK_PROFILE`` for what it gives up and why.
QUICK_PROFILE = "quick"
#: What a row is marked as when the single matched case was measured and preferred it. It is not
#: ``promoted``: no boundary counter-example ran and no second same-domain case re-showed
#: reproduction, so a reader has to be able to tell the two apart from the row itself.
QUICK_VALIDATED_STATUS = "quick_validated"


@dataclass(frozen=True)
class ValidationCase:
    """One paper-level validation case with explicit split membership.

    ``group`` is the case group the selector put this case in. It is kept beside ``split`` because
    the two answer different questions: the split says which ladder rung the case feeds, and the
    group says why it was chosen. A case the caller wrote by hand has no group, which is what
    makes an old suite read exactly as it did before.
    """

    case_id: str
    domain: str
    split: str = "same_domain"
    group: str = ""


def _score(result: dict[str, Any]) -> float | None:
    """Read the measurement one paired run reported, or None when it measured nothing.

    A run that carries no score is not a run that scored zero, and a paired difference of two
    such runs must not be reported as an improvement of nothing.
    """

    if result.get("measured") is False:
        return None
    for key in ("overall_score", "quality_score"):
        value = result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _blockers(result: dict[str, Any]) -> int:
    return int(result.get("blocking_issue_count") or len(result.get("blocking_issues") or []))


def _target_score(result: dict[str, Any], target_metric: str) -> float | None:
    dimensions = result.get("dimension_scores") if isinstance(result.get("dimension_scores"), dict) else {}
    value = dimensions.get(target_metric, result.get(target_metric))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return _score(result)


def _majority(count: int) -> int:
    return max(1, (count + 1) // 2)


def _majority_verdict(verdicts: list[str]) -> str:
    """The verdict the repeated judgements agreed on, or empty when they did not agree.

    One judgement of a pair is a sample of a model that rounds, so a single reading is a
    reading of the rounding as much as of the figure. Asking the same question again and
    taking the answer only when it repeats turns a coin flip into an abstention instead of a
    verdict, which is what keeps a noisy pair from being counted as evidence either way.
    """

    counted = [item for item in verdicts if item]
    if not counted or len(set(counted)) > 1:
        return ""
    return counted[0]


def _unmeasured_reason(evidence: dict[str, Any]) -> str:
    """Why a set that cannot support a verdict cannot support one.

    A pair nobody measured, a pair the Judge could not decide twice the same way, and a pair
    whose arms were not comparable all say nothing about the experience. Naming the reason
    keeps a noisy measurement from being read as a regression.

    The name has to be the condition that actually holds, because the three are not the same
    fact. A set from which no case was ever drawn has no pair to isolate and no arms to make
    safe; reporting that as an isolation failure asserts a check that never ran. A pair whose
    arms were isolated but which carried a new defect failed the safety gate, which is a
    refusal to promote and not a failure to measure. Only a pair that ran and whose audit
    disagreed with itself is an isolation failure.
    """

    if evidence.get("noise_cases"):
        return "measurement_noise"
    # An unmeasured case is an unmeasured case wherever it sat. Comparing a count taken over every
    # case with a total taken over the ones that are not boundary cases made a set whose boundary
    # case could not be drawn read as "isolation or safety failed", which claims something else.
    if evidence.get("unmeasured_cases") or evidence["measured_cases"] < evidence["total_cases"]:
        return "pair_not_measured"
    drawn = sum(
        int(evidence.get(key) or 0)
        for key in ("same_domain_cases", "holdout_cases", "boundary_cases")
    )
    if drawn <= 0:
        return "pair_not_measured"
    if not evidence.get("pairs_isolated"):
        return "pair_isolation_or_safety_failed"
    if not evidence.get("all_cases_safe"):
        return "new_defect_not_safe"
    return "pair_not_measured"


def _measured_contrast(off: dict[str, Any], on: dict[str, Any]) -> bool | None:
    """Whether the seeded memory made the ON arm produce anything different from the OFF arm.

    Read off what each arm wrote -- the products it produced and the deterministic readings taken
    from its own figure -- and never off the names of the moves the two arms happened to run. The
    name sequence was the wrong ruler in both directions: two arms that ran the same named move can
    still have drawn two different figures (a memory that reroutes connectors inside the same
    ``resegment`` step was read as "no contrast"), and two arms that ran differently-named moves
    can still land on the same figure. Returns None when neither side recorded anything comparable,
    so "no contrast" stays distinguishable from "not measured".

    This is evidence, never a gate: a pair with no contrast is a pair that answered nothing, and the
    ladder is told that by the caller rather than being refused here.
    """

    compared = False
    for key in ("product_hashes", "readings"):
        left, right = off.get(key), on.get(key)
        if isinstance(left, dict) and isinstance(right, dict) and left and right:
            compared = True
            if left != right:
                return True
    return False if compared else None


def _contrast_was_shown(item: dict[str, Any]) -> bool:
    """Whether one decided case carries a record that says its two arms differ.

    Both spellings are read: ``products_diverged`` is what the current cases write, and
    ``action_diverged`` is what cases decided before the ruler changed carry on disk. A resumed set
    mixes the two, and reading only one of them would call an old case "no contrast" on no evidence.
    """

    for key in ("products_diverged", "action_diverged"):
        if item.get(key) is not None:
            return bool(item.get(key))
    return True


def _compiled_pair_answered(off: dict[str, Any], compiled: dict[str, Any]) -> bool:
    """Whether the second compilation left a pair the Judge can actually be asked about.

    Two things have to hold. Both arms have to have compiled a figure, and both have to have
    compiled the *same* one -- otherwise the comparison would be reading the difference between two
    figures rather than the difference the rule made. And the rule has to have changed something:
    a compiled arm identical to its control is a pair with no contrast, which answers nothing.
    Anything else is escalated to the paired run, which plans its own figure and can be measured.
    """

    if str(compiled.get("pair_stop_reason") or "") == "compile_only_arm_drew_nothing":
        return False
    if not bool(compiled.get("compiled_program_hash")):
        return False
    # Only a pair that is *known* to have produced the same thing is escalated. The absence of a
    # product record is not evidence of no contrast: a caller that reports no products keeps the
    # cheap pair it built, exactly as it did before this escalation existed.
    return _measured_contrast(off, compiled) is not False


def _case_is_safe(item: dict[str, Any], tolerance: float) -> bool:
    """Whether one paired case is evidence the experience did not make things worse.

    A comparison that preferred the OFF arm is the experience failing, not a safety problem; a
    comparison nobody could take is a case with no evidence, and the ladder refuses it on
    ``measured`` before this is read.
    """

    if item.get("new_blockers"):
        return False
    if item.get("comparison") == "pairwise":
        return item.get("pairwise_verdict") != "before"
    delta = item.get("target_delta")
    return delta is not None and float(delta) >= -float(tolerance)


def _is_boundary(item: dict[str, Any]) -> bool:
    """Whether this case is the boundary counter-example rung rather than a generalization one."""

    return str(item.get("split") or "") == "boundary" or str(item.get("group") or "") == "boundary"


def _domain_ladder_status(
    evidence: dict[str, Any],
    same_domain: list[dict[str, Any]],
    same_passed: int,
    boundary: list[dict[str, Any]],
    *,
    pairs_isolated: bool,
) -> dict[str, Any]:
    """Stop a domain experience inside its own domain.

    Missing a cross-domain exam, or doing worse in another domain, is not
    ``overfit_or_failed``. A missing or regressing boundary keeps the row from
    being marked usable.
    """

    if not pairs_isolated:
        return {"status": "overfit_or_failed", "stopped_because": _unmeasured_reason(evidence), **evidence}
    if evidence.get("unmeasured_cases") or evidence["measured_cases"] < evidence["total_cases"]:
        return {"status": "overfit_or_failed", "stopped_because": "pair_not_measured", **evidence}
    if evidence["inconclusive_cases"] and evidence["inconclusive_cases"] == evidence["total_cases"] + evidence["boundary_cases"]:
        return {"status": "inconclusive", "stopped_because": "no_contrast_between_on_and_off", **evidence}
    if len(same_domain) < 2 or same_passed < _majority(len(same_domain)):
        return {"status": "current_case_validated", "stopped_because": "same_domain_reproduction_not_shown", **evidence}
    if not boundary:
        return {"status": "current_case_validated", "stopped_because": "boundary_uncovered", **evidence}
    if any(not item["passed"] for item in boundary):
        return {"status": "current_case_validated", "stopped_because": "boundary_regressed", **evidence}
    return {"status": "domain_validated", "stopped_because": "", **evidence}


def _quick_ladder_status(
    evidence: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    pairs_isolated: bool,
    all_safe: bool,
) -> dict[str, Any]:
    """Decide the single matched case the ``quick`` profile asked for.

    This is not the ladder and is not reported as one. It answers the one question the caller
    named: was the row preferred on a paper whose shape matches, drawn from another domain where
    the row claims to travel? A row that is not preferred there keeps the rung it already had,
    because one matched paper that disagreed is a refusal to promote rather than evidence of
    failure -- the same reading the full ladder gives a set whose reproduction was not shown.
    """

    if not pairs_isolated:
        return {"status": "overfit_or_failed", "stopped_because": _unmeasured_reason(evidence), **evidence}
    measured = [item for item in results if item.get("measured")]
    if not measured:
        return {"status": "overfit_or_failed", "stopped_because": _unmeasured_reason(evidence), **evidence}
    # The two arms produced the same products, so the pair holds no contrast to read and the Judge
    # was handed two runs it could not tell apart. That is read ahead of the safety gate: otherwise
    # a pair nobody made to differ is refused as a failure, under a reason that blames isolation.
    if all(not _contrast_was_shown(item) for item in measured):
        return {"status": "inconclusive", "stopped_because": "no_contrast_between_on_and_off", **evidence}
    if not all_safe:
        return {"status": "overfit_or_failed", "stopped_because": _unmeasured_reason(evidence), **evidence}
    if any(not bool(item.get("passed")) for item in measured):
        return {"status": "current_case_validated", "stopped_because": "matched_case_did_not_improve", **evidence}
    return {"status": QUICK_VALIDATED_STATUS, "stopped_because": "", **evidence}


#: The rungs that let a row reach a paper other than the one that measured it. A row is held at
#: the rung it already had until a validation run shows the defect it introduced does not
#: reproduce, which is exactly what "every case stayed safe" means.
PROMOTING_STATUSES = ("same_domain_validated", "cross_domain_validated", "domain_validated", "quick_validated", "promoted")


def _settle_carried_new_defect(
    decision: dict[str, Any],
    *,
    carried_new_defect: bool,
    run_is_safe: bool,
) -> dict[str, Any]:
    """Hold a row that still carries the defect it introduced, and spend the mark when it clears.

    A round may file a row and still have introduced a defect: the gain was measured, so the row
    is worth keeping, and the defect travels with it. The row may then be used by the paper that
    measured it, but it may not reach another paper while the defect stands. The evidence that
    clears it is the same evidence that already has to hold for any promoting rung -- every case
    in the set stayed safe, meaning nothing the row did introduced a defect anywhere -- so the
    mark is spent at the moment the run proves it, and the row is promoted on that same run
    rather than waiting for a second one. A run that could not show that leaves the mark standing
    and the row at the rung it already held.
    """

    result = dict(decision)
    if not carried_new_defect or str(result.get("status") or "") not in PROMOTING_STATUSES:
        return result
    if not run_is_safe:
        capped = {
            key: value
            for key, value in result.items()
            if key not in {"status", "stopped_because"}
        }
        capped["status"] = "current_case_validated"
        capped["stopped_because"] = "carried_new_defect"
        capped["carried_new_defect_cleared"] = False
        return capped
    result["carried_new_defect_cleared"] = True
    return result


def ladder_status(
    results: list[dict[str, Any]],
    same_domain: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
    *,
    pairs_isolated: bool,
    all_safe: bool,
    expected_groups: Iterable[str] = CASE_GROUPS,
    applicability: str = "common",
    profile: str = FULL_PROFILE,
    carried_new_defect: bool = False,
) -> dict[str, Any]:
    """Walk the promotion ladder one rung at a time and record where it stopped.

    The rungs are cumulative: same-domain reproduction must hold before cross-domain
    generalization counts, and both must hold before an experience is promoted.

    The first two rungs are majority tests, which together already imply an overall
    majority; promotion therefore asks for the strictly stronger evidence that every
    paired case improved.

    A boundary case is not a rung: it only has to stay safe, because the experience was never
    meant to apply there, so it is kept out of the improvement count in both the numerator and
    the denominator. ``expected_groups`` names the groups this set set out to cover. Missing
    ``cross_domain_same_structure`` stops at ``same_domain_validated``; missing ``boundary``
    cannot reach ``promoted``. Other missing groups are reported as uncovered without blocking.
    """
    same_passed = sum(bool(item["passed"]) for item in same_domain)
    holdout_passed = sum(bool(item["passed"]) for item in holdout)
    boundary = [item for item in results if _is_boundary(item)]
    required = [item for item in results if not _is_boundary(item)]
    total_passed = sum(bool(item["passed"]) for item in required)
    coverage: dict[str, int] = {str(name): 0 for name in expected_groups}
    split_group = {
        "same_domain": "same_domain_same_structure",
        "holdout": "cross_domain_same_structure",
        "boundary": "boundary",
    }
    for item in results:
        group = str(item.get("group") or "") or split_group.get(str(item.get("split") or ""), "")
        if group:
            coverage[group] = coverage.get(group, 0) + 1
    evidence = {
        "pairs_isolated": bool(pairs_isolated),
        "all_cases_safe": bool(all_safe),
        "same_domain_cases": len(same_domain),
        "same_domain_passed": same_passed,
        "holdout_cases": len(holdout),
        "holdout_passed": holdout_passed,
        "total_cases": len(required),
        "total_passed": total_passed,
        "measured_cases": sum(1 for item in results if item.get("measured")),
        "unmeasured_cases": sum(1 for item in results if not item.get("measured")),
        "noise_cases": sum(1 for item in results if item.get("noise")),
        "inconclusive_cases": sum(1 for item in results if not _contrast_was_shown(item)),
        "boundary_cases": len(boundary),
        "boundary_passed": sum(bool(item["passed"]) for item in boundary),
        "group_coverage": coverage,
        "group_uncovered": [{"group": name, "reason": "no case in this set belongs to this group"} for name in coverage if not coverage[name]],
    }
    kind = str(applicability or "").strip().casefold()
    if kind not in {"common", "domain"}:
        decision = {"status": "candidate", "stopped_because": "applicability_unspecified", **evidence}
    elif str(profile or FULL_PROFILE).strip().casefold() == QUICK_PROFILE:
        decision = _quick_ladder_status(evidence, results, pairs_isolated=pairs_isolated, all_safe=all_safe)
    else:
        if kind == "domain":
            decision = _domain_ladder_status(evidence, same_domain, same_passed, boundary, pairs_isolated=pairs_isolated)
        elif not pairs_isolated or not all_safe:
            # A pair nobody measured is refused for its own reason: it is not evidence that the
            # experience failed, it is the absence of evidence either way.
            decision = {"status": "overfit_or_failed", "stopped_because": _unmeasured_reason(evidence), **evidence}
        elif evidence["inconclusive_cases"] and evidence["inconclusive_cases"] == len(results):
            decision = {"status": "inconclusive", "stopped_because": "no_contrast_between_on_and_off", **evidence}
        elif len(same_domain) < 2 or same_passed < _majority(len(same_domain)):
            decision = {"status": "current_case_validated", "stopped_because": "same_domain_reproduction_not_shown", **evidence}
        elif not holdout or holdout_passed < _majority(len(holdout)):
            decision = {"status": "same_domain_validated", "stopped_because": "cross_domain_generalization_not_shown", **evidence}
        elif "cross_domain_same_structure" in {str(item.get("group") or "") for item in evidence["group_uncovered"]}:
            decision = {"status": "same_domain_validated", "stopped_because": "cross_domain_same_structure_uncovered", **evidence}
        elif total_passed < len(required):
            decision = {"status": "cross_domain_validated", "stopped_because": "not_every_case_improved", **evidence}
        elif "boundary" in {str(item.get("group") or "") for item in evidence["group_uncovered"]}:
            decision = {"status": "cross_domain_validated", "stopped_because": "boundary_uncovered", **evidence}
        else:
            decision = {"status": "promoted", "stopped_because": "", **evidence}
    return _settle_carried_new_defect(
        decision,
        carried_new_defect=carried_new_defect,
        run_is_safe=bool(pairs_isolated and all_safe),
    )


def paired_configuration_hash(configuration: dict[str, Any]) -> str:
    canonical = json.dumps(configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _pair_audit(off: dict[str, Any], on: dict[str, Any]) -> dict[str, Any]:
    fields = ("paired_config_hash", "initial_state_hash", "candidate_policy_hash", "evaluator_hash")
    checks = {
        field: bool(off.get(field)) and str(off.get(field)) == str(on.get(field))
        for field in fields
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "off": {field: off.get(field) for field in fields},
        "on": {field: on.get(field) for field in fields},
        "allowed_difference": "target_memory_injection_only",
    }


def _case_result_path(case_dir: Path, case_id: str) -> Path:
    """Where one decided case is written, so a later call can reuse it instead of redoing it."""

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(case_id))
    return case_dir / f"{safe}.json"


def read_decided_cases(case_dir: Any) -> dict[str, dict[str, Any]]:
    """Read back the cases a previous call already decided, keyed by case id.

    A case set is hours of generation, and a call that stops in the middle of one used to lose
    every case it had already paid for. Each decided case is written beside the run as it is
    decided and read back here, which is what makes a restart continue the set rather than
    start it again.
    """

    root = Path(str(case_dir)) if case_dir else None
    if root is None or not root.is_dir():
        return {}
    decided: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and str(value.get("case_id") or ""):
            decided[str(value["case_id"])] = value
    return decided


def write_decided_case(case_dir: Any, entry: dict[str, Any]) -> None:
    """Record one decided case, atomically, so a half-written file is never read as a verdict."""

    if not case_dir:
        return
    root = Path(str(case_dir))
    root.mkdir(parents=True, exist_ok=True)
    target = _case_result_path(root, str(entry.get("case_id") or ""))
    scratch = target.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(entry, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    scratch.replace(target)


def validate_experience_cases(
    experience: dict[str, Any],
    cases: list[ValidationCase],
    runner: Callable[[ValidationCase, bool, dict[str, Any]], dict[str, Any]],
    *,
    target_metric: str = "overall_score",
    tolerance: float = 0.01,
    pairwise: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    expected_groups: Iterable[str] = CASE_GROUPS,
    judge_samples: int = 3,
    case_dir: Any = None,
    compile_only: Callable[[ValidationCase, dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None,
    on_case_decided: Callable[[dict[str, Any]], None] | None = None,
    profile: str = FULL_PROFILE,
) -> dict[str, Any]:
    """Run paired OFF/ON experiments and decide whether an experience generalizes.

    ``pairwise`` is the same two-state question the round asks: handed the two arms' own
    records, it answers which arm a reader is better served by. When it is supplied it decides
    the case, because subtracting two numbers the model rounded to a 0.1 grid measured its
    rounding rather than the difference between the runs; both scores are still recorded.

    A boundary case passes on safety alone: the experience was not meant to apply to it, so
    improving there is not evidence and regressing there is. ``expected_groups`` is forwarded to
    the ladder: missing cross-domain isomorphism stops at ``same_domain_validated``, and a
    missing boundary cannot reach ``promoted``.

    ``judge_samples`` is how many times the same pair is put to the pairwise Judge. Asked once,
    the answer moves with the sampling as much as with the figure, and the same pair has read
    ``after`` on one run and ``before`` on the next. The case counts only when the answers
    repeat; when they do not it is recorded as ``noise`` and measured as nothing.

    ``case_dir`` makes the set resumable: every decided case is written there as it is decided and
    read back at the start, so a call that stopped in the middle of a set continues it instead of
    paying for the same generations again. The verdict is then walked over the union of the cases
    the previous calls decided and the ones this call decided, which is the same answer a single
    uninterrupted pass gives. ``on_case_decided`` is called with each newly decided case so a
    caller can report progress while the set is still running.

    ``compile_only`` swaps the second arm for a second compilation of the first arm's own figure.
    It is handed the control arm and returns the arm that carries the rule, which is what makes a
    rule-class row cost one generation instead of two. The two arms cannot have started from
    different figures by construction, so the arms record the hash of the program they compiled
    and the case only passes when those two hashes are the same; the audit is a check on the
    construction rather than a hope about it. Every case then names the path that decided it.

    The contrast between the two arms is read off the products each one wrote and off the
    deterministic readings of its own figure, and it is recorded rather than obeyed: a pair whose two
    arms produced the same thing is a pair that answered nothing about the memory, which the ladder
    reports as its own outcome instead of the case being refused out of hand.
    """

    comparison_path = COMPILE_ONLY_PATH if compile_only is not None else PAIRED_RUN_PATH

    normalized = normalize_experience(experience)
    applicability = str(normalized.get("applicability") or "")
    decided_before = read_decided_cases(case_dir)
    results: list[dict[str, Any]] = []
    for case in cases:
        kept = decided_before.get(case.case_id)
        if kept is not None:
            # A previous call already put this pair to the Judge. Running it again would pay for
            # the same measurement twice and could reach a different verdict than the one on disk.
            results.append(kept)
            continue
        off = runner(case, False, normalized)
        case_path = comparison_path
        escalated_from = ""
        if compile_only is not None:
            # The rule acts at compile time, so the honest question is what it does to the figure
            # the control arm already drew: compiling that one figure twice costs no generation
            # and cannot compare two different figures.
            compiled = compile_only(case, off, normalized)
            # The cheap pair is only worth keeping when it changed something. A rule the compiler
            # could not honour leaves the second compilation identical to the first, so there are
            # no two figures to ask about; rather than spending the case on a measurement that
            # cannot answer, the pair is upgraded to the one that plans its own figure.
            if _compiled_pair_answered(off, compiled):
                on = compiled
            else:
                on = runner(case, True, normalized)
                case_path = PAIRED_RUN_PATH
                escalated_from = COMPILE_ONLY_PATH
        else:
            on = runner(case, True, normalized)
        off_target, on_target = _target_score(off, target_metric), _target_score(on, target_metric)
        scores_measured = off_target is not None and on_target is not None
        delta = round(on_target - off_target, 4) if scores_measured else None
        blockers = _blockers(on) - _blockers(off)
        pair_audit = _pair_audit(off, on)
        comparison = None
        verdict = ""
        noise = False
        if pairwise is not None:
            # The same pair is put to the Judge more than once: it rounds, and one sample of a
            # rounding is not a measurement of the figure.
            samples = [pairwise(off, on) for _ in range(max(1, int(judge_samples)))]
            comparison = samples[0]
            verdicts = [pairwise_verdict(item) for item in samples]
            verdict = _majority_verdict(verdicts)
            noise = bool(any(verdicts)) and len(set(verdicts)) > 1
        # A pair whose arms never showed the same start state has not been measured, whatever the
        # Judge said: the difference it read may be the difference between two starting points.
        arms_measured = bool(off.get("initial_state_hash")) and bool(on.get("initial_state_hash"))
        measured = (bool(verdict) if pairwise is not None else scores_measured) and arms_measured and not noise
        improved = verdict == "after" if pairwise is not None else bool(scores_measured and delta > tolerance)
        diverged = _measured_contrast(off, on)
        entry = {"case_id": case.case_id, "domain": case.domain, "split": case.split, "group": case.group, "path": case_path, "off": off, "on": on, "pair_audit": pair_audit, "target_metric": target_metric, "comparison": "pairwise" if pairwise is not None else "absolute_score", "pairwise": comparison, "pairwise_verdict": verdict, "measured": measured, "noise": noise, "pair_stop_reason": str(off.get("pair_stop_reason") or on.get("pair_stop_reason") or ""), "target_delta": delta, "new_blockers": max(0, blockers), "products_diverged": diverged, "action_diverged": diverged, "off_actions": off.get("action_sequence"), "on_actions": on.get("action_sequence"), "escalated_from": escalated_from}
        if case_path == COMPILE_ONLY_PATH:
            # Both arms compiled a figure; the case is only evidence when both compiled the same
            # one. The hash is read off the program each arm wrote beside itself -- the bytes it
            # compiled from -- not off a claim about the protocol.
            off_program = str(off.get("compiled_program_hash") or "")
            on_program = str(on.get("compiled_program_hash") or "")
            entry["program_identity"] = {"checked": True, "hash": off_program, "equal": bool(off_program) and off_program == on_program}
            if not entry["program_identity"]["equal"]:
                measured = False
                entry["measured"] = False
                entry["pair_stop_reason"] = entry["pair_stop_reason"] or "arms_compiled_different_figures"
        # A boundary case is decided on safety, and only on safety: "which arm a reader is
        # better served by" is a question the experience was never asked to answer here.
        entry["safe"] = bool(measured) and _case_is_safe(entry, tolerance)
        same_figure = bool((entry.get("program_identity") or {}).get("equal", True))
        entry["passed"] = bool(pair_audit["ok"] and same_figure and blockers <= 0 and (entry["safe"] if _is_boundary(entry) else improved))
        results.append(entry)
        # Written as it is decided, not at the end of the set: a set that is interrupted must keep
        # everything it already measured, or a long validation can never finish.
        write_decided_case(case_dir, entry)
        if on_case_decided is not None:
            on_case_decided(entry)
    same_domain = [item for item in results if item["split"] == "same_domain"]
    holdout = [item for item in results if item["split"] == "holdout"]
    # A set that drew no case has no pair to isolate, and ``all`` over an empty list says yes. Read
    # as isolated, such a set reported the isolation check as passed when it never ran.
    all_pairs_isolated = bool(results) and all(item["pair_audit"]["ok"] for item in results)
    # An unmeasured case is not safe evidence either: it can neither support nor refuse the
    # experience, so it keeps the ladder from being climbed on a pair nobody measured.
    all_measured = all(bool(item["measured"]) for item in results)
    all_safe = all_measured and all(_case_is_safe(item, tolerance) for item in results)
    majority_improved = sum(bool(item["passed"]) for item in results) >= _majority(len(results))
    required_shape = len(same_domain) >= 2 and len(holdout) >= 1
    ladder = ladder_status(results, same_domain, holdout, pairs_isolated=all_pairs_isolated, all_safe=all_safe, expected_groups=expected_groups, applicability=applicability, profile=profile, carried_new_defect=bool(normalized.get("carried_new_defect")))
    # A case the side-by-side comparison decided is measured even when an arm left no number,
    # so a missing delta is skipped rather than coerced: the summary averages the differences
    # that exist and reports nothing when none does.
    measured_deltas = [
        float(item["target_delta"])
        for item in results
        if item["measured"] and item["target_delta"] is not None
    ]
    return {"experience_id": normalized.get("experience_id"), "status": ladder["status"], "validation_profile": str(profile or FULL_PROFILE).strip().casefold() or FULL_PROFILE, "path": comparison_path, "ladder": ladder, "comparison": "pairwise" if pairwise is not None else "absolute_score", "required_shape": {"same_domain": len(same_domain), "holdout": len(holdout), "passed": required_shape}, "all_pairs_isolated": all_pairs_isolated, "all_cases_safe": all_safe, "all_cases_measured": all_measured, "majority_improved": majority_improved, "target_metric": target_metric, "results": results, "resumed_cases": len(decided_before), "summary": {"cases": len(results), "measured_cases": sum(1 for item in results if item["measured"]), "passed_cases": sum(bool(item["passed"]) for item in results), "boundary_cases": sum(1 for item in results if _is_boundary(item)), "mean_target_delta": round(sum(measured_deltas) / max(1, len(measured_deltas)), 4) if measured_deltas else None}}
