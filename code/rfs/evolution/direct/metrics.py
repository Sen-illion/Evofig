"""The five numbers that tell self-evolution from iterative repair.

A loop that only repairs whatever the last reading complained about improves one figure and
learns nothing; the difference is measurable, and these are the measurements. Every number here
is a fraction of rounds or rows that the run already recorded -- nothing is scored, and nothing
gates: a low number is a finding about how the loop is learning, not a reason to refuse a round.

The five are:

* location accuracy -- when a round said "the fault is at this stage, because of these numbers",
  did the following round's reading bear it out;
* advice improvement -- when the model proposed the move, how often the figure came out better;
* portability -- of the rows the ladder has decided, how many stand up away from their own case;
* claim settlement -- how much of what a reflection asserted has been checked rather than believed;
* recurrence -- when the same defect comes back, does the count move the right way.

A number whose denominator is zero is reported as null rather than as zero: "nothing was
measured" and "everything failed" must not read the same.
"""

from __future__ import annotations

from typing import Any

from ...contracts.evolution import deterministic_blocker_counts

METRICS_CONTRACT = "rfs.evolution_metrics"
METRICS_VERSION = 1

#: Which way a cross-round verdict counts for the paper. ``current_better`` is this round's version
#: beating the best the paper had reached; ``history_better`` is this round coming out worse than
#: that best, which is the one reading that says the walk is not going anywhere. The short forms are
#: kept because the ledger is read by scripts written against an earlier spelling of the same two
#: facts, and counting them as undecided would hide exactly the number this file exists for.
CROSS_ROUND_VERDICT_BUCKETS = {
    "current_better": "wins",
    "better": "wins",
    "history_better": "losses",
    "worse": "losses",
    "tie": "ties",
}

#: Which way each measured number has to move to count as an improvement. A number that is not
#: classified here is read as having no direction, and a claim about it is not scored -- guessing
#: a direction for a metric nobody classified would make the accuracy number mean nothing.
LOWER_IS_BETTER = (
    # ``crossing_count`` is two connectors crossing each other. ``box_pass_count`` and
    # ``text_pass_count`` are a line crossing a card or its text, and
    # ``route_obstacle_overlap_count`` is the same smell counted on the arrow report's own drawn
    # routes. Four names, four readings, all of them down-is-better.
    "crossing_count",
    "box_pass_count",
    "text_pass_count",
    "route_obstacle_overlap_count",
    "connector_issue_count",
    "sparse_panel_count",
    "wrapped_label_count",
    "orphan_nodes",
    "slot_boundary_overflows",
    "unbound_scientific_nodes",
    "empty_region_ratio",
    "interface:box_pass",
    "interface:reroute",
    "interface:relayout_slots",
    "interface:relayout_global",
)
HIGHER_IS_BETTER = ("node_occupancy", "panel_occupancy", "edge_coverage")

#: The rungs a row has to reach to be reusable away from the case that produced it, and the
#: statuses that mean the ladder decided against the row. Anything else is undecided.
PORTABLE_STATUSES = ("same_domain_validated", "cross_domain_validated", "domain_validated", "promoted")
FAILED_STATUSES = ("overfit_or_failed", "refuted", "disabled")

#: The accept reasons that mean the figure measured better, not merely "accepted".
IMPROVED_REASONS = ("pairwise_preferred", "coverage_improved_over_pairwise", "target_improved")


def _fraction(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 4)


def metric_direction(name: str) -> str:
    """Which way a measured number has to move to be an improvement, or "" when it is unknown."""

    key = str(name or "")
    if key in LOWER_IS_BETTER:
        return "down"
    if key in HIGHER_IS_BETTER:
        return "up"
    return ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _did_not_worsen(name: str, before: Any, after: Any) -> bool | None:
    """Whether a metric held or improved, or None when the direction of the metric is unknown."""

    direction = metric_direction(name)
    if not direction:
        return None
    start, end = _number(before), _number(after)
    if start is None or end is None:
        return None
    return end <= start if direction == "down" else end >= start


def _evidence_metrics(evidence: Any) -> list[str]:
    """The metric names an attribution pointed at: the last path segment of each evidence field."""

    names: list[str] = []
    for item in evidence if isinstance(evidence, (list, tuple)) else []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or item.get("metric") or "").strip()
        if not field:
            continue
        name = field.split(".")[-1].strip()
        if name and name not in names:
            names.append(name)
    return names


def location_accuracy(rounds: Any) -> dict[str, Any]:
    """How often the next round's reading bore out the stage a round placed the fault at.

    A round is checked when it named a fault stage, pointed at at least one measured number, and
    the following round measured that same number. It is borne out when every number it named
    held or improved. A round whose claim was about a number the next round did not measure is
    left out of the denominator: an unmeasured claim was not checked, and counting it either way
    would put a reading nobody took into the accuracy.
    """

    checked = borne_out = 0
    for previous, following in zip(rounds or [], list(rounds or [])[1:]):
        if not isinstance(previous, dict) or not isinstance(following, dict):
            continue
        if not str(previous.get("fault_stage") or "").strip():
            continue
        before = previous.get("round_metrics") if isinstance(previous.get("round_metrics"), dict) else {}
        after = following.get("round_metrics") if isinstance(following.get("round_metrics"), dict) else {}
        trials = []
        for name in _evidence_metrics(previous.get("fault_evidence")):
            verdict = _did_not_worsen(name, before.get(name), after.get(name))
            if verdict is not None:
                trials.append(verdict)
        if not trials:
            continue
        checked += 1
        if all(trials):
            borne_out += 1
    return {"checked": checked, "borne_out": borne_out, "rate": _fraction(borne_out, checked)}


def advice_improvement(rounds: Any, *, mechanism_sources: Any = ()) -> dict[str, Any]:
    """How often a move the model proposed left the figure better, of the rounds it proposed one.

    A move the mechanism's own policy proposed is excluded: the mechanism's default runs whether
    or not anything was learned, so crediting the model for it would measure the shipped default.
    """

    excluded = {str(item) for item in mechanism_sources} if mechanism_sources else set()
    offered = improved = 0
    for record in rounds or []:
        if not isinstance(record, dict):
            continue
        action = record.get("repair_action") if isinstance(record.get("repair_action"), dict) else {}
        if not str(action.get("type") or ""):
            continue
        if str(action.get("source") or "") in excluded:
            continue
        decision = record.get("repair_decision") if isinstance(record.get("repair_decision"), dict) else {}
        offered += 1
        if record.get("accepted") and str(decision.get("reason") or "") in IMPROVED_REASONS:
            improved += 1
    return {"offered": offered, "improved": improved, "rate": _fraction(improved, offered)}


def portable_pass_rate(rows: Any) -> dict[str, Any]:
    """Of the rows the ladder has decided, how many reached a reusable rung."""

    portable = failed = 0
    cross_domain = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        if status in PORTABLE_STATUSES:
            portable += 1
            if status in {"cross_domain_validated", "promoted"}:
                cross_domain += 1
        elif status in FAILED_STATUSES:
            failed += 1
    decided = portable + failed
    return {
        "decided": decided,
        "portable": portable,
        "failed": failed,
        "cross_domain": cross_domain,
        "rate": _fraction(portable, decided),
    }


def claim_settlement_rate(report: Any) -> dict[str, Any]:
    """How much of what the reflections asserted has been settled by a later reading."""

    counts = (report or {}).get("counts") if isinstance(report, dict) else {}
    counts = counts if isinstance(counts, dict) else {}
    registered = int(counts.get("registered") or 0)
    settled = int(counts.get("settled") or 0)
    return {
        "registered": registered,
        "settled": settled,
        "confirmed": int(counts.get("confirmed") or 0),
        "refuted": int(counts.get("refuted") or 0),
        "open": int(counts.get("open") or 0),
        "rate": _fraction(settled, registered),
    }


def recurrence(rounds: Any) -> dict[str, Any]:
    """When the same defect is measured again, whether its count moved the right way."""

    last: dict[str, float] = {}
    pairs = gains = 0
    for record in rounds or []:
        metrics = record.get("round_metrics") if isinstance(record, dict) and isinstance(record.get("round_metrics"), dict) else {}
        for name in LOWER_IS_BETTER:
            if name not in metrics:
                continue
            value = _number(metrics.get(name))
            if value is None:
                continue
            if name in last:
                pairs += 1
                if value <= last[name]:
                    gains += 1
            last[name] = value
    return {"reappearances": pairs, "held_or_better": gains, "rate": _fraction(gains, pairs)}


ROUND_METRICS_CONTRACT = "rfs.evolution_round_metrics"
ROUND_METRICS_VERSION = 1


def layer_walk(
    layers: Any,
    *,
    stage_order: Any = (),
    settle_events: Any = (),
    reopen_events: Any = (),
    worked_rounds: Any = (),
) -> dict[str, Any]:
    """How far down the layers a paper has walked, and what each step cost.

    The order is handed in rather than restated here, because a second copy of it is how a round
    could believe it was working on one layer while the sheet said another. ``rounds_to_settle``
    is the number the plan asks for -- the iteration that settled a layer minus the iteration that
    first moved it, plus one -- and it is null for a layer nobody has settled yet, because "not
    finished" and "finished in no rounds at all" must not read the same.

    Which round settled a layer is read from the answers themselves -- newest answer per layer --
    rather than from the state's single ``settled_at`` slot: a layer that was reopened and settled
    again leaves two answers, and only the record of them says which round the paper really
    finished the layer in. A paper whose rounds predate that record falls back to the state, so an
    old run still reads as far as it truly got.

    A layer opened again after its last answer is not settled, however many times it was called
    finished before: the reopen is the later statement, and a walk that ignored it would say the
    paper had finished every layer while the work face said otherwise.

    ``rounds_to_settle_all`` is the first round by which every layer in the order stood settled,
    counted in rounds the paper actually worked in, so a settle answered in a round that moved
    nothing counts from the next round that did. It is null until every layer has been settled
    once, and a layer that was reopened has no settle to count, which pushes the number out by
    construction rather than by anybody remembering to.
    """

    order = [str(item) for item in (stage_order or [])]
    records = layers if isinstance(layers, dict) else {}
    answers: dict[str, dict[str, Any]] = {}
    opened: dict[str, int] = {}
    for event in reopen_events or []:
        if not isinstance(event, dict):
            continue
        stage = str(event.get("stage") or "")
        number = int(event.get("iteration") or 0)
        if stage not in order or not number:
            continue
        opened[stage] = max(number, int(opened.get(stage) or 0))
    for event in settle_events or []:
        if not isinstance(event, dict):
            continue
        stage = str(event.get("stage") or "")
        number = int(event.get("iteration") or 0)
        if stage not in order or not number:
            continue
        previous = answers.get(stage) or {}
        if number >= int(previous.get("iteration") or 0):
            answers[stage] = {
                "iteration": number,
                "by": str(event.get("by") or ""),
                "because": str(event.get("because") or ""),
            }
    rows: dict[str, dict[str, Any]] = {}
    reopened = 0
    for stage in order:
        record = records.get(stage) if isinstance(records.get(stage), dict) else {}
        answer = answers.get(stage) or {}
        first = int(record.get("first_move_at") or 0)
        settled = int(answer.get("iteration") or 0) or int(record.get("settled_at") or 0)
        settled_by = str(answer.get("by") or record.get("settled_by") or "")
        because = str(answer.get("because") or record.get("because") or "")
        if settled and settled <= int(opened.get(stage) or 0):
            # The layer was opened again at or after this answer, so the later statement is that it
            # is being worked on. A reading cannot unfinish a layer and only an answer can, and an
            # answer to reopen is exactly that.
            settled, settled_by, because = 0, "", ""
        reopen_count = int(record.get("reopen_count") or 0)
        reopened += reopen_count
        rows[stage] = {
            "settled_at": settled,
            "settled_by": settled_by,
            "because": because,
            "first_move_at": first,
            "last_move_at": int(record.get("last_move_at") or 0),
            "reopen_count": reopen_count,
            "rounds_to_settle": (settled - first + 1) if (settled and first and settled >= first) else None,
        }
    settled_stages = [stage for stage in order if rows[stage]["settled_at"]]
    all_settled = bool(order) and len(settled_stages) == len(order)
    rounds_to_settle_all: int | None = None
    if all_settled:
        latest = max(rows[stage]["settled_at"] for stage in settled_stages)
        later = sorted(int(item) for item in (worked_rounds or []) if int(item) >= latest)
        rounds_to_settle_all = later[0] if later else latest
    return {
        "layers": rows,
        "order": order,
        "settled": len(settled_stages),
        "total": len(order),
        "all_settled": all_settled,
        "rounds_to_settle_all": rounds_to_settle_all,
        "settle_events": len([event for event in (settle_events or []) if isinstance(event, dict) and int(event.get("iteration") or 0)]),
        "reopen_events": reopened,
        "rounds_worked": len(list(worked_rounds or [])),
    }


def reflection_yield(reflections: Any) -> dict[str, Any]:
    """Of the rounds that were reflected on, how many left something a later paper can be drawn by.

    A reflection is counted as productive when it named a mechanism-level rule or wrote a
    sentence for a later prompt: those are the two fields the library can carry forward, and a
    round that wrote neither taught it nothing however well the figure came out.
    """

    total = productive = 0
    for entry in reflections or []:
        if not isinstance(entry, dict):
            continue
        total += 1
        rule = entry.get("rule") if isinstance(entry.get("rule"), dict) else {}
        patch = entry.get("prompt_patch") if isinstance(entry.get("prompt_patch"), dict) else {}
        if str(rule.get("type") or "") or str(patch.get("directive") or "").strip() or str(entry.get("guidance") or "").strip():
            productive += 1
    return {"rounds": total, "productive": productive, "yield": _fraction(productive, total)}


def round_metrics(
    *,
    paper: str,
    iteration: int,
    wave: int = 0,
    layers: Any = None,
    stage_order: Any = (),
    settle_events: Any = (),
    reopen_events: Any = (),
    worked_rounds: Any = (),
    reopened_layers: Any = (),
    work_face: str = "",
    upstream: Any = None,
    reflections: Any = (),
    catalogue: Any = None,
    refusals: Any = None,
    cross_round: Any = (),
) -> dict[str, Any]:
    """One iteration's own progress numbers, in the shape the ledger and the report both read.

    Everything the round can only know about itself is asked of this round's own record: which
    layers its answers settled or opened, and how many times it reached above the work face. What
    the round inherits -- the walk so far -- is passed in as the walk's own inputs rather than as
    the walk, so the same numbers recompute identically whenever the record is read again.
    """

    walk = layer_walk(
        layers,
        stage_order=stage_order,
        settle_events=settle_events,
        reopen_events=reopen_events,
        worked_rounds=worked_rounds,
    )
    upstream_counts = upstream if isinstance(upstream, dict) else {}
    requested = int(upstream_counts.get("requested") or 0)
    passed = int(upstream_counts.get("passed") or 0)
    comparisons = [item for item in (cross_round or []) if isinstance(item, dict)]
    # The verdict is the seat's own word, so it is read as the seat writes it: "which version did
    # you prefer" comes back as the version it chose, not as a direction the caller has to infer.
    # A verdict nobody has a bucket for is undecided rather than a tie -- "the two came out equal"
    # and "nobody could say" are different facts and averaging them would flatter the trend.
    verdict_counts = {"wins": 0, "losses": 0, "ties": 0}
    undecided = 0
    for item in comparisons:
        bucket = CROSS_ROUND_VERDICT_BUCKETS.get(str(item.get("verdict") or "").strip().lower())
        if bucket:
            verdict_counts[bucket] += 1
        else:
            undecided += 1
    settled_here = [
        str(event.get("stage") or "")
        for event in (settle_events or [])
        if isinstance(event, dict) and int(event.get("iteration") or 0) == int(iteration) and str(event.get("stage") or "")
    ]
    return {
        "contract": ROUND_METRICS_CONTRACT,
        "version": ROUND_METRICS_VERSION,
        "paper": str(paper or ""),
        "iteration": int(iteration),
        "wave": int(wave),
        "work_face": str(work_face or ""),
        "layers": walk,
        "settled_this_round": sorted(set(settled_here)),
        "reopened_this_round": sorted({str(item) for item in (reopened_layers or []) if str(item)}),
        "reopen_answers": len([item for item in (reopen_events or []) if isinstance(item, dict)]),
        "upstream": {"requested": requested, "passed": passed, "rate": _fraction(passed, requested)},
        "reflection": reflection_yield(reflections),
        "catalogue": dict(catalogue) if isinstance(catalogue, dict) else {"count": 0, "moves": []},
        "refusals": {str(key): int(value) for key, value in (refusals or {}).items()},
        "cross_round": {
            "comparisons": len(comparisons),
            **verdict_counts,
            "undecided": undecided,
        },
    }


def evolution_metrics(
    *,
    rounds: Any = (),
    library_rows: Any = (),
    claim_report: Any = None,
    mechanism_sources: Any = (),
) -> dict[str, Any]:
    """The five numbers, each with the counts it was computed from."""

    parts = {
        "location": location_accuracy(rounds),
        "advice": advice_improvement(rounds, mechanism_sources=mechanism_sources),
        "portability": portable_pass_rate(library_rows),
        "claims": claim_settlement_rate(claim_report),
        "recurrence": recurrence(rounds),
    }
    return {
        "contract": METRICS_CONTRACT,
        "version": METRICS_VERSION,
        "rounds": len([item for item in rounds or [] if isinstance(item, dict)]),
        **parts,
        "headline": {
            "location_accuracy": parts["location"]["rate"],
            "advice_improvement_rate": parts["advice"]["rate"],
            "portable_pass_rate": parts["portability"]["rate"],
            "claim_settlement_rate": parts["claims"]["rate"],
            "recurrence_rate": parts["recurrence"]["rate"],
        },
    }


__all__ = [
    "FAILED_STATUSES",
    "HIGHER_IS_BETTER",
    "IMPROVED_REASONS",
    "LOWER_IS_BETTER",
    "METRICS_CONTRACT",
    "METRICS_VERSION",
    "PORTABLE_STATUSES",
    "ROUND_METRICS_CONTRACT",
    "ROUND_METRICS_VERSION",
    "advice_improvement",
    "claim_settlement_rate",
    "evolution_metrics",
    "layer_walk",
    "location_accuracy",
    "metric_direction",
    "portable_pass_rate",
    "recurrence",
    "reflection_yield",
    "round_metrics",
]
