"""Run the whole evolution loop on one wall-clock budget, without a human in between.

The chain the build plan describes has every piece already: a paper is evolved, an accepted
round writes an experience, the experience is validated on other papers, promotion makes it
retrievable, and a later run retrieves it. What was missing was anything that carried one
step's output into the next -- the artifacts were on disk and the paths were retyped by hand.
This module is that carrier. It walks an ordered queue of stations, gives each one its own
attempt directory so a retry never overwrites what the earlier try wrote, stops a station on
the first failure it cannot retry, and writes down which Memory and which patch each station
actually recalled, injected or refused. It stops on the clock, not on a count: a station that
already started is allowed to finish, and the rest of the queue is dropped as
``budget_exhausted`` rather than cutting a paper in half.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from ..contracts.evolution_session import (
    normalize_evolution_session_report,
    normalize_session_station,
    session_station_id,
)
from ..evolution.direct import ExperienceStore
from ..providers import rate_limited_error
from ..utils import ensure_dir, read_json, write_json, write_text
from .direct_experience_validation import run_direct_experience_validation
from .paper_evolve_editable import run_paper_evolve_editable
from .session_evidence import evolve_evidence, session_markdown, validate_evidence


SESSION_REPORT = "evolution_session_report.json"
SESSION_MARKDOWN = "evolution_session_report.md"
SESSION_EVENTS = "evolution_session_events.jsonl"
JUDGE_DATASET_DIR = "judge_dataset"
TRAINING_EXPORT = "training_export.json"

# The settings a session config may carry per paper. Everything else an evolve station wants
# is the session's own, so a typo in a config cannot quietly become a run option.
EVOLVE_OPTION_KEYS = (
    "reference",
    "planner_mode",
    "read_mode",
    "aspect_ratio",
    "max_rounds",
    "candidates_per_round",
    "icon_mode",
    "judge_mode",
    "judge_model",
    "method_mode",
    "reference_mode",
    "icon_candidates",
    "structure_mode",
    "text_plan_mode",
    "ppt_plan_mode",
    "structure_model",
    "text_plan_model",
    "ppt_plan_model",
    "color_theme_mode",
    "color_theme_model",
    "vlm_timeout",
    "vlm_retries",
    "reference_library",
    "reference_top_k",
    "exemplar_guidance",
    "curated_standard",
    # The three structure switches the two-stage split moves one at a time. They were dropped
    # by this filter, so a config could name middle_layer off and get a round with it on: the
    # round would then be filed under a switch it never ran, and the ablation would read as
    # "the explicit structure changes nothing" for a reason that has nothing to do with it.
    "graph_context",
    "middle_layer",
    "figure_target",
    "visual_review_mode",
    "visual_review_model",
    "domain_profile",
    "online_judge_model",
    "frozen_judge_model",
    "max_retries_per_experience",
    "evolution_scope",
    "replay_upstream_from",
    "reuse_upstream_from",
    "upstream_from",
    "bulk_figures",
    "experience_enabled",
    "retrieval_statuses",
    "evolution_enabled",
    "acceptance_checkpoint",
    "acceptance_base_model",
    "online_judge_checkpoint",
    "online_judge_base_model",
    "judge_holdout_paths",
)
#: Statuses a lifecycle row can already carry that make a validation station pointless: the
#: ladder has nothing left to climb, so the station says so instead of spending two runs.
SETTLED_EXPERIENCE_STATUSES = ("promoted", "disabled")
#: A validation station did its job when the ladder landed on a status later papers may use.
#: ``domain_validated`` is that status for an experience that was only examined inside its domain.
#: ``quick_validated`` is the same word for a station the caller ran at the quick bar: the row was
#: measured on one matched paper and preferred there, and the rung says so on its own name.
VALIDATE_PASSED_STATUSES = ("same_domain_validated", "cross_domain_validated", "domain_validated", "promoted", "quick_validated")
EXAMINABLE_APPLICABILITY = ("common", "domain")


class StationError(RuntimeError):
    """A station that cannot continue, carrying the contract's word for why."""

    def __init__(self, reason: str, detail: str, *, retryable: bool = False) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.retryable = retryable


def _float_setting(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def station_slug(value: Any) -> str:
    """Turn a paper or experience id into one directory name that keeps its identity.

    One definition for every module that has to name a directory after a paper or an
    experience, so the continuous runner and the session cannot file the same paper under
    two names.
    """

    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return text.strip("._-") or "station"


def _read_config(config: str | Path | dict[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(config, dict):
        return dict(config), ""
    path = Path(config)
    body = read_json(path)
    if not isinstance(body, dict):
        raise ValueError(f"a session config must be a JSON object: {path}")
    return body, str(path)


def _raw_stations(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the ordered queue, accepting either an explicit list or the two shorthand lists."""

    stations = [dict(item) for item in body.get("stations") or [] if isinstance(item, dict)]
    if stations:
        return stations
    # Validating first is the order this session is meant to run in: the next paper can only
    # retrieve a row once the ladder has promoted it.
    stations = [dict(item) for item in body.get("validations") or [] if isinstance(item, dict)]
    for item in stations:
        item.setdefault("kind", "validate")
    for item in body.get("papers") or []:
        if isinstance(item, dict):
            stations.append({"kind": "evolve", **dict(item)})
    return stations


def _queued_suite(specs: list[dict[str, Any]]) -> str:
    """The suite an auto-queued validation should use when the session named none.

    A config that hangs a suite off each validate station is naming the session's validation
    suite; the first one is that answer, and an evolve station that earned a row should be
    judged by the same suite its siblings are judged by.
    """

    for spec in specs:
        if spec["kind"] == "validate" and spec.get("suite"):
            return str(spec["suite"])
    return ""


def _holdout_paths(value: Any) -> tuple[str, ...]:
    """Read a holdout list whether the config wrote one path, or several."""

    if value is None or value == "":
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _session_evolve_defaults(
    body: dict[str, Any],
    *,
    acceptance_checkpoint: str,
    acceptance_base_model: str,
    judge_holdout_paths: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    """Settings every evolve station inherits unless that station names its own.

    An empty function argument leaves the config's top-level value in place. A station
    that writes the same key replaces both.
    """

    defaults: dict[str, Any] = {}
    checkpoint = str(acceptance_checkpoint or "").strip() or str(body.get("acceptance_checkpoint") or "").strip()
    base_model = str(acceptance_base_model or "").strip() or str(body.get("acceptance_base_model") or "").strip()
    holdout = _holdout_paths(judge_holdout_paths) or _holdout_paths(body.get("judge_holdout_paths"))
    if checkpoint:
        defaults["acceptance_checkpoint"] = checkpoint
    if base_model:
        defaults["acceptance_base_model"] = base_model
    if holdout:
        defaults["judge_holdout_paths"] = holdout
    return defaults


def _station_spec(raw: dict[str, Any], *, validation_suite: str, evolve_defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    """Turn one config entry into the spec the driver runs, refusing anything ambiguous."""

    kind = str(raw.get("kind") or "evolve").strip().casefold()
    if kind == "evolve":
        paper = str(raw.get("paper") or "").strip()
        if not paper:
            raise ValueError("an evolve station needs a paper")
        subject = str(raw.get("paper_id") or "").strip() or Path(paper).stem
        options = {key: raw[key] for key in EVOLVE_OPTION_KEYS if key in raw}
        for key, value in (evolve_defaults or {}).items():
            options.setdefault(key, value)
        if "judge_holdout_paths" in options:
            options["judge_holdout_paths"] = _holdout_paths(options["judge_holdout_paths"])
        return {
            "kind": "evolve",
            "subject": subject,
            "paper": paper,
            "options": options,
        }
    if kind == "validate":
        experience_id = str(raw.get("experience_id") or "").strip()
        if not experience_id:
            raise ValueError("a validate station needs an experience_id")
        return {
            "kind": "validate",
            "subject": experience_id,
            "experience_id": experience_id,
            "suite": str(raw.get("suite") or "").strip() or str(validation_suite or ""),
            # Which bar this station measures at, when the station names one. Left blank the suite
            # decides, which is what every station written before this field meant.
            "validation_profile": str(raw.get("validation_profile") or "").strip().casefold(),
        }
    raise ValueError(f"unknown station kind: {kind!r}")


def _attempt_dir(root: Path, spec: dict[str, Any], number: int) -> Path:
    base = root / ("papers" if spec["kind"] == "evolve" else "validation") / station_slug(spec["subject"])
    return base / f"attempt_{number:02d}"


def _run_evolve(spec: dict[str, Any], attempt_dir: Path, *, library: str, runner: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    paper = Path(str(spec["paper"]))
    if not paper.is_file():
        raise StationError("paper_unreadable", f"the paper could not be read: {paper}")
    options: dict[str, Any] = {**spec["options"], "paper": str(paper), "out": str(attempt_dir)}
    options.setdefault("experience_enabled", True)
    options.setdefault("evolution_enabled", True)
    if library:
        options["experience_library"] = library
    result = runner(**options)
    if not isinstance(result, dict) or not result:
        raise StationError("run_failed", f"the run returned no report: {attempt_dir}")
    return evolve_evidence(attempt_dir, result)


def _run_validate(spec: dict[str, Any], attempt_dir: Path, *, library: str, runner: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    report = runner(suite=spec["suite"], experience_library=library, experience_id=spec["experience_id"], out=str(attempt_dir), validation_profile=str(spec.get("validation_profile") or ""))
    if not isinstance(report, dict) or not report.get("status"):
        raise StationError("validation_failed", f"the validation returned no status: {attempt_dir}")
    evidence = validate_evidence(attempt_dir, report)
    # Where the ladder stood before and after is the one thing a reader of the session wants
    # from a validation station, and it is not in the validator's own report.
    if library:
        status = str(ExperienceStore(library).lifecycle_status(spec["experience_id"], "") or "")
        evidence["status_before"] = status
        evidence["status_after"] = status
    return evidence


def _settled_reason(library: str, experience_id: str) -> str:
    """Why there is nothing to validate, or "" when the row is a candidate for the ladder."""

    if not library:
        return "no experience library was given to this session, so there is no row to validate"
    store = ExperienceStore(library)
    if not store.get(experience_id):
        return f"the experience library holds no row named {experience_id}"
    status = str(store.lifecycle_status(experience_id, "") or "")
    if status in SETTLED_EXPERIENCE_STATUSES:
        return f"{experience_id} is already {status}"
    return ""


def _failure_reason(exc: BaseException, kind: str) -> str:
    if isinstance(exc, StationError):
        return exc.reason
    if rate_limited_error(exc):
        return "rate_limited"
    return "run_failed" if kind == "evolve" else "validation_failed"


def _worth_retrying(exc: BaseException) -> bool:
    """A shared quota or a dropped connection deserves another try; a real defect does not."""

    if isinstance(exc, StationError):
        return exc.retryable
    if rate_limited_error(exc) or isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.casefold()
    text = str(exc).casefold()
    return "ssl" in name or "ssl" in text or "unexpected_eof" in text


def _station(
    spec: dict[str, Any],
    *,
    status: str,
    reason: str = "",
    detail: str = "",
    attempts: list[dict[str, Any]] | None = None,
    evidence: dict[str, Any] | None = None,
    out_dir: str = "",
    ok: bool = False,
) -> dict[str, Any]:
    rows = list(attempts or [])
    return normalize_session_station(
        {
            "kind": spec["kind"],
            "subject": spec["subject"],
            "status": status,
            "reason": reason,
            "detail": detail,
            "out_dir": out_dir,
            "seconds": round(sum(float(row.get("seconds") or 0.0) for row in rows), 2),
            "ok": ok,
            "attempts": rows,
            "evidence": dict(evidence or {}),
        }
    )


def _run_station(
    spec: dict[str, Any],
    *,
    root: Path,
    library: str,
    retries: int,
    cooldown: float,
    evolve_runner: Callable[..., dict[str, Any]],
    validate_runner: Callable[..., dict[str, Any]],
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
    event: Callable[..., None],
) -> dict[str, Any]:
    """Run one station, retrying a quota refusal into a fresh attempt directory."""

    kind = spec["kind"]
    station_id = session_station_id(kind, spec["subject"])
    attempts: list[dict[str, Any]] = []
    # ``--rate-limit-retries N`` is N retries *after* the first attempt, so a station that
    # never gets through is recorded with N + 1 attempts.
    allowed_retries = max(0, int(retries))
    if kind == "validate":
        settled = _settled_reason(library, spec["experience_id"])
        if settled:
            event("station_skipped", station_id=station_id, reason="no_candidate_experience", detail=settled)
            return _station(spec, status="skipped", reason="no_candidate_experience", detail=settled, out_dir=str(_attempt_dir(root, spec, 1)))
    for number in range(1, allowed_retries + 2):
        attempt_dir = _attempt_dir(root, spec, number)
        started_at = clock()
        event("attempt_started", station_id=station_id, attempt=number, out_dir=str(attempt_dir))
        try:
            evidence = (
                _run_evolve(spec, attempt_dir, library=library, runner=evolve_runner)
                if kind == "evolve"
                else _run_validate(spec, attempt_dir, library=library, runner=validate_runner)
            )
        except Exception as exc:  # a station may not take the whole session down
            detail = str(exc) or type(exc).__name__
            reason = _failure_reason(exc, kind)
            attempts.append(
                {
                    "attempt": number,
                    "status": "failed",
                    "reason": reason,
                    "detail": detail,
                    "out_dir": str(attempt_dir),
                    "seconds": round(max(0.0, clock() - started_at), 2),
                }
            )
            event("attempt_failed", station_id=station_id, attempt=number, reason=reason, detail=detail)
            if _worth_retrying(exc) and number <= allowed_retries:
                event("attempt_backoff", station_id=station_id, attempt=number, seconds=max(0.0, float(cooldown)))
                sleeper(max(0.0, float(cooldown)))
                continue
            return _station(spec, status="failed", reason=reason, detail=detail, attempts=attempts, out_dir=str(attempt_dir))
        attempts.append(
            {
                "attempt": number,
                "status": "ok",
                "out_dir": str(attempt_dir),
                "seconds": round(max(0.0, clock() - started_at), 2),
            }
        )
        landed = evidence.get("status", "") if kind == "validate" else ""
        event("attempt_completed", station_id=station_id, attempt=number, out_dir=str(attempt_dir), status=landed)
        return _station(
            spec,
            status="completed",
            attempts=attempts,
            evidence=evidence,
            out_dir=str(attempt_dir),
            ok=(bool(evidence.get("run_ok", True)) if kind == "evolve" else landed in VALIDATE_PASSED_STATUSES),
        )
    raise AssertionError("the attempt loop must return")


def _summary(stations: list[dict[str, Any]], *, exhausted: bool) -> str:
    counts = {name: sum(1 for item in stations if item["status"] == name) for name in ("completed", "failed", "skipped", "pending")}
    text = (
        f"Evolution session: {counts['completed']} completed, {counts['failed']} failed, "
        f"{counts['skipped']} skipped, {counts['pending']} pending of {len(stations)} stations."
    )
    if exhausted:
        text += " The wall-clock budget ran out and the rest of the queue was dropped."
    return text


def _export_training_data(root: Path, *, split: str | Path, event: Callable[..., None]) -> dict[str, Any]:
    """Turn the runs a session produced into Judge data, preference pairs and a review queue.

    This is the hand-off the build plan expects, done at the end of the session instead of by
    someone retyping three commands later: the examples, the pairs built from them and the
    human review queue all land under ``<out>/judge_dataset/``. A failure here is recorded and
    reported, never raised -- the session's own verdict is about its stations, not the export --
    and it is never silent, because a session whose runs taught nothing must say so.
    """

    from ..judge_training.dataset_builder import build_dataset
    from ..judge_training.preference_pairs import build_preference_pairs, build_review_queue
    from ..judge_training.split_ledger import build_split_index, load_split_ledger

    dataset_dir = root / JUDGE_DATASET_DIR
    review_dir = dataset_dir / "review"
    report_path = root / TRAINING_EXPORT
    result: dict[str, Any] = {
        "attempted": True,
        "ok": False,
        "reason": "",
        "detail": "",
        "dataset_dir": str(dataset_dir),
        "review_dir": str(review_dir),
        "records": 0,
        "pairs": 0,
        "review_pairs": 0,
        "split": str(split or ""),
        "report": str(report_path),
    }
    try:
        split_index = build_split_index(load_split_ledger(split)) if split else None
        dataset = build_dataset(root, dataset_dir, split_index=split_index)
        pairs_path = dataset_dir / "preference_pairs.jsonl"
        pairs = build_preference_pairs(dataset_dir / "judge_examples.jsonl", pairs_path)
        review = build_review_queue(pairs_path, review_dir)
    except Exception as exc:  # noqa: BLE001 - a failed export is a recorded outcome, not a crash
        result["reason"] = "export_failed"
        result["detail"] = f"{type(exc).__name__}: {exc}"
        write_json(report_path, result)
        event("dataset_export_failed", reason=result["reason"], detail=result["detail"])
        return result
    result.update(
        {
            "ok": True,
            "records": int(dataset.get("records") or 0),
            "pairs": int(pairs.get("pairs") or 0),
            "review_pairs": int(review.get("pairs") or 0),
        }
    )
    write_json(report_path, result)
    return result


def run_evolution_session(
    config: str | Path | dict[str, Any],
    out: str | Path,
    *,
    validation_suite: str | Path = "",
    experience_library: str | Path = "",
    hours: float = 5.0,
    rate_limit_cooldown: float = 45.0,
    rate_limit_retries: int = 3,
    validate: bool = True,
    auto_validate: bool = True,
    dry_run: bool = False,
    training_export: bool = True,
    export_split: str | Path = "",
    acceptance_checkpoint: str = "",
    acceptance_base_model: str = "",
    judge_holdout_paths: tuple[str, ...] | list[str] = (),
    evolve_runner: Callable[..., dict[str, Any]] = run_paper_evolve_editable,
    validate_runner: Callable[..., dict[str, Any]] = run_direct_experience_validation,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run an ordered queue of stations until the wall clock runs out.

    ``evolve_runner`` and ``validate_runner`` are the seams the two kinds of station run
    through, so the ordering, the attempts, the retries and the report can be exercised
    without paying for a single call. The clock and the sleeper are seams for the same reason.
    """

    body, config_path = _read_config(config)
    root = Path(out).resolve()
    budget_hours = _float_setting(body.get("hours"), hours)
    cooldown = _float_setting(body.get("rate_limit_cooldown"), rate_limit_cooldown)
    retries = int(_float_setting(body.get("rate_limit_retries"), rate_limit_retries))
    library = str(body.get("experience_library") or experience_library or "")
    suite = str(body.get("validation_suite") or validation_suite or "")
    evolve_defaults = _session_evolve_defaults(
        body,
        acceptance_checkpoint=acceptance_checkpoint,
        acceptance_base_model=acceptance_base_model,
        judge_holdout_paths=judge_holdout_paths,
    )
    specs = [_station_spec(item, validation_suite=suite, evolve_defaults=evolve_defaults) for item in _raw_stations(body)]
    if not specs:
        raise ValueError("a session config needs at least one station")
    if validate:
        without_suite = [spec["subject"] for spec in specs if spec["kind"] == "validate" and not spec["suite"]]
        if without_suite:
            raise ValueError("every validate station needs a suite; these have none: " + ", ".join(without_suite))
    auto_suite = suite or _queued_suite(specs)
    session_id = str(body.get("session_id") or "") or f"session-{root.name}"
    if not dry_run:
        ensure_dir(root)
    started_at = clock()

    def elapsed() -> float:
        return max(0.0, clock() - started_at)

    def event(name: str, **payload: Any) -> None:
        if dry_run:
            return
        row = {"event": name, "at_seconds": round(elapsed(), 2), **payload}
        with (root / SESSION_EVENTS).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    stations: list[dict[str, Any]] = []
    exhausted = False
    dataset_export: dict[str, Any] = {}

    def report_now() -> dict[str, Any]:
        failed = [item for item in stations if item["status"] == "failed"]
        payload: dict[str, Any] = {
            "session_id": session_id,
            "summary": _summary(stations, exhausted=exhausted),
            "ok": not failed,
            "dry_run": dry_run,
            "out_dir": str(root),
            "config": config_path,
            "experience_library": library,
            "validation_suite": suite,
            "validate": bool(validate),
            "budget": {"hours": budget_hours, "used_seconds": elapsed(), "exhausted": exhausted},
            "stations": stations,
            "blockers": [
                {"station_id": item["station_id"], "kind": item["kind"], "reason": item["reason"], "detail": item["detail"]}
                for item in failed
            ],
        }
        if dataset_export:
            payload["dataset_export"] = dataset_export
        report = normalize_evolution_session_report(payload)
        if not dry_run:
            write_json(root / SESSION_REPORT, report)
            write_text(root / SESSION_MARKDOWN, session_markdown(report))
        return report

    queue = list(specs)
    queued = {session_station_id(spec["kind"], spec["subject"]) for spec in queue}
    settled: set[str] = set()
    index = 0
    while index < len(queue):
        spec = queue[index]
        index += 1
        if spec["kind"] == "validate" and spec["subject"] in settled:
            continue
        if spec["kind"] == "validate" and not validate:
            stations.append(
                _station(
                    spec,
                    status="skipped",
                    reason="validation_disabled",
                    detail="this session was started with validation switched off",
                    out_dir=str(_attempt_dir(root, spec, 1)),
                )
            )
            report_now()
            continue
        if budget_hours * 3600.0 - elapsed() <= 0:
            exhausted = True
            stations.append(
                _station(
                    spec,
                    status="skipped",
                    reason="budget_exhausted",
                    detail=f"the session ran out of its {budget_hours} h budget before this station started",
                    out_dir=str(_attempt_dir(root, spec, 1)),
                )
            )
            report_now()
            continue
        if dry_run:
            stations.append(_station(spec, status="pending", detail="planned; nothing has run yet", out_dir=str(_attempt_dir(root, spec, 1))))
            report_now()
            continue
        station_id = session_station_id(spec["kind"], spec["subject"])
        event("station_started", station_id=station_id, kind=spec["kind"])
        record = _run_station(
            spec,
            root=root,
            library=library,
            retries=retries,
            cooldown=cooldown,
            evolve_runner=evolve_runner,
            validate_runner=validate_runner,
            clock=clock,
            sleeper=sleeper,
            event=event,
        )
        stations.append(record)
        event("station_finished", station_id=station_id, status=record["status"], reason=record["reason"])
        if spec["kind"] == "validate":
            settled.add(spec["subject"])
        elif validate and auto_validate and record["status"] == "completed":
            # An accepted round wrote a row the ladder has not judged yet: queue its
            # validation behind the configured stations, on the budget that is left.
            applicability = record["evidence"].get("candidate_applicability") or {}
            for experience_id in record["evidence"].get("candidate_experience_ids") or []:
                candidate = session_station_id("validate", experience_id)
                if not experience_id or experience_id in settled or candidate in queued:
                    continue
                kind_written = str(applicability.get(experience_id) or "").strip().casefold()
                if kind_written not in EXAMINABLE_APPLICABILITY:
                    # A row with no applicability is saved and left there. Queuing it would
                    # spend a validation run that the selector is required to refuse. It is not
                    # left to sit silently at ``candidate`` either: the library is told, in a
                    # status of its own, that this row cannot be examined, so the dead end shows
                    # up in the library and not only in a session log.
                    queued.add(candidate)
                    event(
                        "auto_validate_declined",
                        station_id=candidate,
                        because=station_id,
                        detail="applicability was not written",
                    )
                    store = ExperienceStore(library) if library else None
                    if store is not None:
                        store.transition(str(experience_id), "unverifiable", reason="no_applicability_was_written", because=station_id)
                    event(
                        "experience_unverifiable",
                        station_id=candidate,
                        experience_id=str(experience_id),
                        because=station_id,
                        detail="no applicability was written, so this row cannot be validated",
                    )
                    continue
                if not auto_suite:
                    # Nothing to judge the row on: say so rather than queueing a station that
                    # can only refuse, and never validate against a suite nobody named.
                    event("auto_validate_declined", station_id=candidate, because=station_id, detail="no validation suite was given to this session")
                    continue
                queued.add(candidate)
                queue.append({"kind": "validate", "subject": experience_id, "experience_id": experience_id, "suite": auto_suite})
                event("station_queued", station_id=candidate, kind="validate", because=station_id)
        report_now()
    if training_export and not dry_run:
        # The runs are on disk now; the material the next training round consumes is staged from
        # them before the final report is written, so "what did this session teach" is answered
        # in the report itself instead of by whoever remembers to run the export later.
        dataset_export = _export_training_data(root, split=export_split, event=event)
    return report_now()


__all__ = [
    "EVOLVE_OPTION_KEYS",
    "JUDGE_DATASET_DIR",
    "SESSION_EVENTS",
    "SESSION_MARKDOWN",
    "SESSION_REPORT",
    "StationError",
    "station_slug",
    "TRAINING_EXPORT",
    "run_evolution_session",
]
