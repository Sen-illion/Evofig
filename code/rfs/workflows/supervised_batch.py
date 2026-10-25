"""Run a corpus subset paper by paper and decide after each whether to pay again.

The batch loop is deliberately thin: it hands one paper at a time to the product
workflow, grades the attempt with the deterministic supervision verdict, and
spends another attempt only when the verdict says breakage is worth redoing.
Every attempt lands in ``attempts.jsonl`` as it finishes, so a batch that dies
half way still says exactly which paper and which step died.

Two things are recorded per attempt that the pipeline itself does not write:
how long the attempt took, and how many provider tokens it used.  Both are read
back off the run directory, never asked of the provider, so supervision of a
hundred papers costs nothing beyond the papers themselves.

The token figure is a lower bound: it adds up the usage blocks the run reports
recorded for themselves, and the stages that record none (the visual review, the
Icon calls) contribute nothing to it.  It is there to compare attempts and to
watch a batch, not to replace the provider's own bill.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..evaluation.content_coverage import REPORT_FILE as CONTENT_COVERAGE_FILE
from ..evaluation.run_supervision import judge_attempt, load_attempt_reports
from ..judge_training.pilot_selection import load_figure_index
from ..judge_training.pilot_graphs import safe_paper_directory
from ..judge_training.source_snapshot_io import strict_jsonl
from ..utils import read_json, write_json, write_text
from .corpus_subset import (
    DEFAULT_PER_DOMAIN,
    DEFAULT_SUBSET_ID,
    apply_labels,
    build_subset,
    delivered_paper_directories,
    index_source_pdfs,
    index_snapshot_pdfs,
    load_snapshot_index,
    load_subset,
    resolve_paper_inputs,
    select_undelivered,
    write_subset_version,
)

BATCH_CONTRACT = "rfs.supervised_batch"
BATCH_VERSION = 1
LEDGER_FILE = "attempts.jsonl"
SUMMARY_FILE = "batch_summary.json"
ALREADY_DRAWN_FILE = "already_drawn.jsonl"
FAILURE_CONTRACT = "rfs.supervised_batch_attempt_error"
FAILURE_FILE = "attempt_error.json"

#: Nothing may spend more than this many attempts on one paper, whatever the
#: verdict asks for.
MAX_ATTEMPTS = 3

#: What the ledger records when ``keep_first_draft`` refused to buy another
#: attempt: the paper already delivered a figure, so what the checks disliked is
#: written down and the figure is kept instead of being paid for twice.
WITHHELD_DELIVERED = "the attempt delivered a figure, so the batch kept it"

#: Reports that each own exactly one provider block.  The roll-up run report is
#: deliberately absent: it echoes the same numbers again and would double them.
TOKEN_OWNERS = (
    "run_report.json",
    "method_extraction_report.json",
    "scientific_hierarchy_validation.json",
    "design_plan_validation.json",
    "figure_text_plan_validation.json",
    "ppt_composition_validation.json",
    "color_theme_validation.json",
    "connector_review_validation.json",
    "color_review_validation.json",
    "asset_generation_report.json",
)

TOKEN_GLOBS = ("visual_review_*_validation.json",)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: What a batch says in its summary when it stopped for the account's sake.
ACCOUNT_EXHAUSTED = "provider_account_exhausted"


def attempt_directory(out_root: str | Path, paper_id: str, attempt: int) -> Path:
    """Directory that holds one attempt of one paper."""
    return Path(out_root) / safe_paper_directory(paper_id) / f"attempt_{attempt:02d}"


#: A paper in one of these states already produced the result the run was paid
#: for, so a resumed run must not buy it again.
KEPT_STATUSES = frozenset({"clean"})


def _resume_keeps(row: dict[str, Any], *, keep_first_draft: bool = False) -> bool:
    """True when a resumed batch must not pay for this paper again.

    A clean paper is the obvious case. A bulk run also keeps the first figure it
    already delivered, even when the checks asked for another attempt, and that
    figure is already paid for. A paper that delivered nothing stays in the queue.
    """

    if str(row.get("status") or "") in KEPT_STATUSES | {"record_only"}:
        return True
    if row.get("retry_withheld"):
        return True
    if not keep_first_draft:
        return False
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return str(metrics.get("delivery_status") or "") == "available"

#: When the provider says the account is out of quota, every paper still in the
#: queue would fail the same way within seconds.  A batch that kept walking the
#: queue would spend the whole retry budget of the corpus on questions that were
#: never asked, so it stops and says why instead.
ACCOUNT_EXHAUSTED_SIGNS = (
    "insufficient_quota",
    "user quota is not enough",
    "insufficient balance",
    "exceeded your current quota",
)


def account_exhausted(text: Any) -> bool:
    """True when the failure is about the account rather than about the paper."""

    haystack = str(text or "").lower()
    return any(sign in haystack for sign in ACCOUNT_EXHAUSTED_SIGNS)


#: The small files a stage writes about itself. The verdict reads only some of
#: them, and the account complaint lands in whichever stage asked the provider
#: first, so the check has to look wider than the verdict does.
EXHAUSTION_PROBE_GLOBS = ("*_validation.json", "*_report.json", "progress.jsonl")

#: Saved model answers can be megabytes; the complaint is never in one of them.
EXHAUSTION_PROBE_MAX_BYTES = 8_000_000


def account_exhausted_in(directory: str | Path) -> bool:
    """Look for the provider's account complaint in anything the attempt wrote.

    Seven of the hundred attempts the 2026-09-21 outage touched recorded it
    nowhere the batch was reading, and every one of those kept the queue moving.
    """

    for pattern in EXHAUSTION_PROBE_GLOBS:
        for path in sorted(Path(directory).glob(pattern)):
            try:
                if not path.is_file() or path.stat().st_size > EXHAUSTION_PROBE_MAX_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if account_exhausted(text):
                return True
    return False


def _report_text(reports: dict[str, Any] | None) -> str:
    """Collect what the reports say about themselves, minus the saved answers."""

    chunks: list[str] = []
    for report in (reports or {}).values():
        if not isinstance(report, dict):
            continue
        for key, value in report.items():
            if key == "raw_response":
                continue
            if isinstance(value, (str, int, float)):
                chunks.append(str(value))
            elif isinstance(value, list):
                chunks.extend(str(item) for item in value[:20] if isinstance(item, (str, int, float)))
    return " ".join(chunks)


def load_ledger_rows(out_root: str | Path) -> list[dict[str, Any]]:
    """Read every attempt row any earlier run wrote into the batch directory."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(out_root).glob("attempts*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def previous_paper_state(out_root: str | Path) -> dict[str, dict[str, Any]]:
    """Return each paper's last attempt row, so a resumed run knows where it got to."""
    latest: dict[str, dict[str, Any]] = {}
    for row in load_ledger_rows(out_root):
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            continue
        if int(row.get("attempt") or 0) >= int(latest.get(paper_id, {}).get("attempt") or 0):
            latest[paper_id] = row
    return latest


def next_attempt_number(out_root: str | Path, paper_id: str) -> int:
    """Number the next attempt after the ones on disk, never over writing them."""
    folder = Path(out_root) / safe_paper_directory(paper_id)
    highest = 0
    if folder.is_dir():
        for child in folder.iterdir():
            match = re.fullmatch("attempt_([0-9]+)", child.name)
            if match and child.is_dir():
                highest = max(highest, int(match.group(1)))
    return highest


def previous_attempt_directory(out_root: str | Path, paper_id: str, attempt: int) -> Path | None:
    """The newest earlier attempt on disk, so a retry continues instead of restarting.

    A verdict that says "the layout is wrong" says nothing about the paper
    reading, the contracts or the Icons, and those are already on disk and
    already paid for.  Handing the earlier attempt to the workflow lets it keep
    them and re-do only the part that was complained about.
    """

    folder = Path(out_root) / safe_paper_directory(paper_id)
    if not folder.is_dir():
        return None
    earlier: list[tuple[int, Path]] = []
    for child in folder.iterdir():
        match = re.fullmatch("attempt_([0-9]+)", child.name)
        if match and child.is_dir() and int(match.group(1)) < int(attempt):
            earlier.append((int(match.group(1)), child))
    if not earlier:
        return None
    return max(earlier, key=lambda item: item[0])[1]


def _usage_total(block: dict[str, Any]) -> int:
    total = block.get("total_tokens")
    if isinstance(total, (int, float)) and not isinstance(total, bool):
        return int(total)
    prompt = block.get("prompt_tokens")
    completion = block.get("completion_tokens")
    if isinstance(prompt, (int, float)) and isinstance(completion, (int, float)):
        return int(prompt) + int(completion)
    return 0


def sum_provider_tokens(root: str | Path) -> dict[str, Any]:
    """Read the provider usage each run report recorded for itself."""
    root = Path(root)
    names = [name for name in TOKEN_OWNERS if (root / name).is_file()]
    for pattern in TOKEN_GLOBS:
        names.extend(sorted(path.name for path in root.glob(pattern)))
    by_file: dict[str, int] = {}
    for name in names:
        try:
            report = read_json(root / name)
        except (OSError, ValueError):
            continue
        if not isinstance(report, dict):
            continue
        subtotal = 0
        provider = report.get("provider")
        if isinstance(provider, dict):
            if isinstance(provider.get("usage"), dict):
                subtotal += _usage_total(provider["usage"])
            for value in provider.values():
                if isinstance(value, dict) and isinstance(value.get("usage"), dict):
                    subtotal += _usage_total(value["usage"])
        if subtotal:
            by_file[name] = subtotal
    return {"total": sum(by_file.values()), "by_file": by_file, "source": "run_reports"}


def _delivered_a_figure(verdict: dict[str, Any]) -> bool:
    """True when the attempt reached a delivered figure, however flawed it is.

    ``delivery_status`` is what the workflow itself concluded after looking at
    the compiled deck, so an attempt that crashed, stopped on a contract or never
    wrote a PPTX answers False and stays retryable.
    """
    return str((verdict.get("metrics") or {}).get("delivery_status") or "") == "available"


def run_supervised_batch(
    entries: Sequence[dict[str, Any]],
    *,
    out_root: str | Path,
    runner: Callable[[dict[str, Any], Path], dict[str, Any] | None],
    batch_id: str = "batch",
    shard: tuple[int, int] | None = None,
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
    clock: Callable[[], float] | None = None,
    resume: bool = False,
    reuse_upstream: bool = True,
    keep_first_draft: bool = False,
) -> dict[str, Any]:
    """Run every entry, retrying only where the verdict says it is worth it.

    With ``resume`` the papers an earlier run already kept are skipped and the
    attempt numbering continues from disk, so a second pass pays only for what is
    still missing and the earlier attempts stay readable.

    With ``keep_first_draft`` a paper that delivered a figure is never paid for
    twice: the checks still record what they disliked and only an attempt that
    delivered nothing at all is retried.  Bulk work wants the flawed figure next
    to the note saying why, not a second bill.

    ``shard`` splits the work between processes that run side by side: shard
    ``(1, 4)`` takes every fourth paper and writes its own ledger, so the whole
    corpus run is as parallel as the machine allows without any of the workers
    sharing state.
    """
    import time

    clock = clock or time.monotonic
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if shard is not None:
        index_of_shard, shard_count = int(shard[0]), int(shard[1])
        if shard_count < 1 or not 0 <= index_of_shard < shard_count:
            raise ValueError(f"shard {index_of_shard}/{shard_count} is out of range")
        entries = [entry for position, entry in enumerate(entries) if position % shard_count == index_of_shard]
    ledger_name = LEDGER_FILE if shard is None else f"attempts_shard_{int(shard[0]):02d}.jsonl"
    summary_name = SUMMARY_FILE if shard is None else f"batch_summary_shard_{int(shard[0]):02d}.json"
    ledger: list[dict[str, Any]] = []
    skipped: list[str] = []
    started_at = _utc_now()
    batch_started = clock()

    previous = previous_paper_state(out_root) if resume else {}
    halted_reason = ""
    for index, entry in enumerate(entries, 1):
        paper_id = str(entry.get("paper_id") or "")
        earlier = previous.get(paper_id) or {}
        if resume and _resume_keeps(earlier, keep_first_draft=keep_first_draft):
            # Nothing is written to the ledger: a ledger holds attempts that
            # happened, and this paper was already carried by one of them.
            skipped.append(paper_id)
            if on_attempt:
                on_attempt({"index": index, "of": len(entries), "paper_id": paper_id, "attempt": int(earlier.get("attempt") or 0), "status": "clean", "category": "already_kept", "reasons": [{"kind": "advisory", "code": "already_kept", "detail": "an earlier run kept this paper, so it was not paid for again"}], "retry_budget": 0, "advisories": [], "metrics": {}, "token_total": 0, "token_by_file": {}})
            continue
        attempts = 0
        attempt_number = next_attempt_number(out_root, paper_id) if resume else 0
        verdict: dict[str, Any] = {}
        while True:
            attempts += 1
            attempt_number += 1
            directory = attempt_directory(out_root, paper_id, attempt_number)
            row: dict[str, Any] = {
                "contract": "rfs.supervised_batch_attempt",
                "version": BATCH_VERSION,
                "batch_id": batch_id,
                "paper_id": paper_id,
                "domain": str(entry.get("domain") or ""),
                "split": str(entry.get("split") or ""),
                "attempt": attempt_number,
                "attempt_dir": str(directory),
                "started_at": _utc_now(),
            }
            missing = [str(item) for item in (entry.get("missing") or [])]
            if missing:
                row.update(
                    {
                        "elapsed_seconds": 0.0,
                        "status": "skipped",
                        "category": "unresolved_input",
                        "reasons": [
                            {
                                "kind": "upstream",
                                "code": f"missing:{','.join(missing)}",
                                "detail": "the paper or its framework figure could not be located",
                            }
                        ],
                        "advisories": [],
                        "metrics": {},
                        "token_total": 0,
                    }
                )
                attempts = 0
                ledger.append(row)
                _append(out_root / ledger_name, row)
                if on_attempt:
                    on_attempt({"index": index, "of": len(entries), **row})
                break

            # An earlier attempt of this paper already paid for the reading, the
            # contracts and the Icons; a retry continues from it rather than
            # buying the same work twice.
            reuse_from = previous_attempt_directory(out_root, paper_id, attempt_number) if reuse_upstream else None
            reuse_entry = {**entry, "reuse_upstream_from": str(reuse_from)} if reuse_from else entry
            attempt_started = clock()
            error: str | None = None
            failure: dict[str, Any] | None = None
            result: dict[str, Any] = {}
            try:
                returned = runner(reuse_entry, directory)
                if isinstance(returned, dict):
                    result = returned
                    if returned.get("error"):
                        error = str(returned["error"])
            except Exception as exc:  # noqa: BLE001 - the ledger is the record
                error = f"{type(exc).__name__}: {exc}"
                failure = record_attempt_failure(
                    directory,
                    paper_id=paper_id,
                    attempt=attempt_number,
                    error=exc,
                    elapsed=round(clock() - attempt_started, 3),
                )
            elapsed = round(clock() - attempt_started, 3)
            reports = load_attempt_reports(directory)
            verdict = judge_attempt(reports, error=error, paper_id=paper_id)
            if verdict.get("content_coverage"):
                # Written beside the other attempt reports so the coverage of one
                # paper can be read back without replaying the whole ledger.
                write_json(Path(directory) / CONTENT_COVERAGE_FILE, verdict["content_coverage"])
            exhausted = account_exhausted(error) or account_exhausted(_report_text(reports))
            if not exhausted:
                # A refusal can hide in a stage whose failure the verdict does
                # not read, so the attempt is looked over regardless of how the
                # verdict went.
                exhausted = account_exhausted_in(directory)
            tokens = sum_provider_tokens(directory)
            row.update(
                {
                    "elapsed_seconds": elapsed,
                    "status": verdict["status"],
                    "category": verdict["category"],
                    "categories": verdict["categories"],
                    "retry_budget": verdict["retry_budget"],
                    "reasons": verdict["reasons"],
                    "advisories": verdict["advisories"],
                    "metrics": verdict["metrics"],
                    "token_total": tokens["total"],
                    "token_by_file": tokens["by_file"],
                }
            )
            if reuse_from:
                row["reused_upstream_from"] = str(reuse_from)
            if result.get("metrics"):
                row["runner_metrics"] = result["metrics"]
            retry_wanted = (
                verdict["action"] == "retry"
                and attempts < MAX_ATTEMPTS
                and attempts <= int(verdict["retry_budget"] or 0)
            )
            withheld = bool(keep_first_draft and retry_wanted and _delivered_a_figure(verdict))
            if failure:
                # A crash inside the pipeline is not the same defect as a report
                # that says something is wrong, so say which one this was and
                # where it happened instead of leaving one message to carry it.
                row["failure"] = {
                    "error_type": failure["error_type"],
                    "stage": failure["stage"],
                    "location": failure["location"],
                    "call": failure["call"],
                    "report": FAILURE_FILE,
                }
            if exhausted:
                row["halted_reason"] = ACCOUNT_EXHAUSTED
            if retry_wanted and withheld:
                # The checks asked for a second figure; the batch keeps the one
                # already on disk and writes down why it stopped here.
                row["retry_withheld"] = WITHHELD_DELIVERED
            ledger.append(row)
            _append(out_root / ledger_name, row)
            if on_attempt:
                on_attempt({"index": index, "of": len(entries), **row})
            if exhausted:
                # Every paper left in the queue would fail the same way, so the
                # batch stops here instead of spending the rest of the corpus's
                # retry budget on a question the provider refuses to answer.
                halted_reason = ACCOUNT_EXHAUSTED
                break
            if not retry_wanted or withheld:
                break
        if halted_reason:
            break

    summary = _summarise(batch_id, ledger, elapsed=round(clock() - batch_started, 3), started_at=started_at, skipped=skipped)
    summary["shard"] = None if shard is None else f"{int(shard[0])}/{int(shard[1])}"
    if halted_reason:
        summary["halted_reason"] = halted_reason
        summary["halted_after"] = len(ledger)
    write_json(out_root / summary_name, summary)
    return summary


def _append(path: Path, row: dict[str, Any]) -> None:
    """Append one attempt row so a crashed batch still says where it stopped."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _last_stage(directory: str | Path) -> str:
    """Return the stage the attempt was in when it stopped."""

    path = Path(directory) / "progress.jsonl"
    if not path.is_file():
        return ""
    try:
        lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return ""
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("stage"):
            return str(record["stage"])
    return ""


def record_attempt_failure(
    directory: str | Path,
    *,
    paper_id: str,
    attempt: int,
    error: BaseException,
    elapsed: float | None = None,
) -> dict[str, Any]:
    """Write where an attempt broke, next to the attempt itself.

    The ledger keeps the message, and a message alone is not a fix: a crash
    inside the pipeline reads as one line ("TypeError") while what would end it
    is the frame it happened on.  The stage, the location and the stack are
    written here so the next occurrence can be attributed without paying for a
    replay.
    """

    frames = traceback.extract_tb(error.__traceback__)
    internal = [frame for frame in frames if f"{os.sep}rfs{os.sep}" in frame.filename.replace("/", os.sep)]
    frame = internal[-1] if internal else (frames[-1] if frames else None)
    payload: dict[str, Any] = {
        "contract": FAILURE_CONTRACT,
        "version": BATCH_VERSION,
        "paper_id": paper_id,
        "attempt": int(attempt),
        "stage": _last_stage(directory),
        "error_type": type(error).__name__,
        "error": str(error),
        "location": f"{frame.filename}:{frame.lineno}" if frame else "",
        "call": frame.name if frame else "",
        "source_line": (frame.line or "") if frame else "",
        "elapsed_seconds": elapsed,
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))[-4000:],
    }
    try:
        write_json(Path(directory) / FAILURE_FILE, payload)
    except OSError:
        # Recording a crash must never be the thing that ends the batch; the
        # ledger row still carries the stage and the location.
        payload["report_written"] = False
    return payload


def _summarise(
    batch_id: str,
    ledger: Sequence[dict[str, Any]],
    *,
    elapsed: float,
    started_at: str,
    skipped: Sequence[str] = (),
) -> dict[str, Any]:
    final: dict[str, dict[str, Any]] = {}
    for row in ledger:
        final[str(row.get("paper_id"))] = row
    by_final: dict[str, int] = {}
    by_attempt: dict[str, int] = {}
    for row in ledger:
        by_attempt[str(row.get("category"))] = by_attempt.get(str(row.get("category")), 0) + 1
    for row in final.values():
        by_final[str(row.get("category"))] = by_final.get(str(row.get("category")), 0) + 1
    attention = [
        {
            "paper_id": paper_id,
            "category": row.get("category"),
            "attempts": row.get("attempt"),
            "codes": [item.get("code") for item in (row.get("reasons") or [])],
        }
        for paper_id, row in sorted(final.items())
        if row.get("category") not in ("clean",)
    ]
    kept_clean = sorted([paper_id for paper_id, row in final.items() if row.get("category") == "clean"] + list(skipped))
    kept_first_draft = sorted(
        paper_id for paper_id, row in final.items() if row.get("retry_withheld")
    )
    kept_after_retry = sorted(
        paper_id
        for paper_id, row in final.items()
        if row.get("category") != "clean" and int(row.get("attempt") or 1) > 1 and row.get("status") != "skipped"
    )
    return {
        "contract": BATCH_CONTRACT,
        "version": BATCH_VERSION,
        "batch_id": batch_id,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_seconds": elapsed,
        "papers": len(final) + len(skipped),
        "already_kept": len(skipped),
        "attempts": len(ledger),
        "extra_attempts": len(ledger) - len(final),
        "kept_clean": len(kept_clean),
        "kept_after_retry": len(kept_after_retry),
        "kept_first_draft": len(kept_first_draft),
        "kept_first_draft_paper_ids": kept_first_draft,
        "unresolved": len(final) - len(kept_clean) - len(kept_after_retry),
        "clean_rate": round(len(kept_clean) / max(1, len(final)), 4),
        "by_final_category": dict(sorted(by_final.items())),
        "by_attempt_category": dict(sorted(by_attempt.items())),
        "token_total": sum(int(row.get("token_total") or 0) for row in ledger),
        "token_source": "run_reports",
        "needs_attention": attention,
        "clean_paper_ids": kept_clean,
        "ledger": str(Path(LEDGER_FILE)),
    }


def make_paper_evolve_runner(
    *,
    reference_library: str | Path | None = None,
    icon_mode: str = "image2",
    max_rounds: int = 1,
    vlm_timeout: int = 240,
    vlm_retries: int = 2,
    evolution_enabled: bool = False,
    experience_enabled: bool = False,
    bulk_figures: bool = False,
    contracts_from: str | Path | None = None,
    frozen_reground_min_ratio: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Callable[[dict[str, Any], Path], dict[str, Any]]:
    """Build the runner the batch loop calls for the product workflow.

    ``contracts_from`` points at the middleware's exported contracts, one
    directory per paper.  A paper whose contract is there is drawn from it and
    the reading, the review and the method extraction are not paid for again; a
    paper that is not there falls back to the generation line's own analysis, so
    a batch can run before, during or after the middleware export.
    """
    from .paper_evolve_editable import run_paper_evolve_editable
    from ..judge_training.pilot_graphs import safe_paper_directory

    def run(entry: dict[str, Any], out_dir: Path) -> dict[str, Any]:
        options: dict[str, Any] = {
            "paper": entry["pdf"],
            "reference": entry.get("reference"),
            "out": out_dir,
            "planner_mode": "vlm",
            "method_mode": "vlm",
            "reference_mode": "vlm",
            "structure_mode": "vlm",
            "text_plan_mode": "vlm",
            "ppt_plan_mode": "vlm",
            "color_theme_mode": "vlm",
            "icon_mode": icon_mode,
            "visual_review_mode": "vlm",
            "judge_mode": "heuristic",
            "domain_profile": "auto",
            "evolution_enabled": evolution_enabled,
            "experience_enabled": experience_enabled,
            "max_rounds": max_rounds,
            "candidates_per_round": 1,
            "icon_candidates": 1,
            "vlm_timeout": vlm_timeout,
            "vlm_retries": vlm_retries,
            "reference_top_k": 3,
            "exemplar_guidance": True,
            "bulk_figures": bulk_figures,
        }
        if reference_library:
            options["reference_library"] = str(reference_library)
        reuse = str(entry.get("reuse_upstream_from") or "")
        if reuse:
            options["reuse_upstream_from"] = reuse
        contract_dir = str(entry.get("upstream_contract_dir") or "").strip()
        if not contract_dir and contracts_from:
            candidate = Path(contracts_from) / safe_paper_directory(str(entry.get("paper_id") or ""))
            if candidate.is_dir():
                contract_dir = str(candidate)
        if contract_dir:
            options["upstream_from"] = contract_dir
        if frozen_reground_min_ratio is not None:
            options["frozen_reground_min_ratio"] = float(frozen_reground_min_ratio)
        options.update(extra or {})
        report = run_paper_evolve_editable(**options)
        return {
            "ok": bool(report.get("ok")),
            "error": None,
            "metrics": {
                "delivery_status": report.get("delivery_status"),
                "quality_status": report.get("quality_status"),
                "node_count": report.get("node_count"),
                "connector_count": report.get("connector_count"),
                "upstream_contract": contract_dir,
            },
        }

    return run


def run_corpus_batch(
    *,
    snapshot: str | Path,
    split_dir: str | Path,
    out_root: str | Path,
    pdf_roots: Sequence[str | Path],
    exclude_roots: Sequence[str | Path] = (),
    archive_root: str | Path | None = None,
    subset: str | Path | None = None,
    subset_id: str = DEFAULT_SUBSET_ID,
    per_domain: int = DEFAULT_PER_DOMAIN,
    label_cache: str | Path | None = None,
    sample_only: bool = False,
    limit: int | None = None,
    resume: bool = False,
    shard: tuple[int, int] | None = None,
    runner: Callable[[dict[str, Any], Path], dict[str, Any] | None] | None = None,
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
    reuse_upstream: bool = True,
    keep_first_draft: bool = False,
    contracts_from: str | Path | None = None,
) -> dict[str, Any]:
    """Sample (or load) a corpus subset, resolve its inputs, and run it supervised.

    With ``sample_only`` the subset is written and nothing is paid for, which is
    how the paper list gets reviewed before the first billed attempt.

    ``contracts_from`` is the middleware export root, one directory per paper.
    It is handed to the default runner and never required: the subset can be run
    without it, and the papers it does not cover are analysed as before.

    ``archive_root`` is the snapshot's own archive.  When it is given, a sampled
    paper is resolved to the copy the snapshot froze rather than to the
    collection directory it was collected from, which is what lets a subset run
    after the collection directories have been cleaned up.

    ``exclude_roots`` names the product roots that already hold deliveries.  A
    paper with a run directory in any of them is dropped before a provider is
    called, because it has been paid for once already, and the ids that were
    dropped are written next to the subset so the drop is visible rather than
    silent.
    """

    out_root = Path(out_root).resolve()
    if shard is not None and subset is None:
        raise ValueError("a sharded batch has to be pointed at the subset version the parent already wrote")
    if subset is not None:
        payload = load_subset(subset)
        subset_dir = Path(subset)
        written: dict[str, Any] | None = None
    else:
        payload = build_subset(
            snapshot,
            split_dir,
            subset_id=subset_id,
            per_domain=per_domain,
        )
        subset_dir = out_root / "subset"
        written = write_subset_version(subset_dir, payload)
    if label_cache is not None:
        apply_labels(payload, strict_jsonl(label_cache))

    queued = list(payload.get("papers") or [])
    already_drawn: list[str] = []
    if exclude_roots:
        queued, already_drawn = select_undelivered(queued, delivered_paper_directories(exclude_roots))
        if already_drawn:
            write_text(
                out_root / ALREADY_DRAWN_FILE,
                "".join(json.dumps({"paper_id": paper_id}, ensure_ascii=False, sort_keys=True) + "\n" for paper_id in already_drawn),
            )

    result: dict[str, Any] = {
        "subset_id": payload.get("subset_id"),
        "subset_dir": str(subset_dir),
        "subset_written": written,
        "papers": len(payload.get("papers") or []),
        "queued": len(queued),
        "already_drawn": len(already_drawn),
        "by_domain": payload.get("by_domain"),
        "label_report": payload.get("label_report"),
        "sample_only": bool(sample_only),
    }
    if sample_only:
        return result

    pdf_by_paper = index_snapshot_pdfs(snapshot, archive_root) if archive_root else {}
    pdf_index = index_source_pdfs(pdf_roots)
    papers = load_snapshot_index(snapshot)
    figure_index = load_figure_index(snapshot)
    entries = [
        resolve_paper_inputs(
            snapshot,
            entry,
            pdf_index=pdf_index,
            pdf_by_paper=pdf_by_paper,
            papers=papers,
            figure_index=figure_index,
        )
        for entry in queued
    ]
    unresolved = [entry["paper_id"] for entry in entries if entry["missing"]]
    if limit is not None:
        entries = entries[: max(0, int(limit))]
    result["resolved"] = sum(1 for entry in entries if not entry["missing"])
    result["unresolved_paper_ids"] = unresolved
    result["pdf_index_size"] = len(pdf_index)
    result["archive_pdf_index_size"] = len(pdf_by_paper)
    if contracts_from:
        result["contracts_from"] = str(Path(contracts_from).resolve())
    result["batch"] = run_supervised_batch(
        entries,
        out_root=out_root / "runs",
        runner=runner or make_paper_evolve_runner(contracts_from=contracts_from),
        batch_id=str(payload.get("subset_id") or "batch"),
        shard=shard,
        on_attempt=on_attempt,
        resume=resume,
        reuse_upstream=reuse_upstream,
        keep_first_draft=keep_first_draft,
    )
    return result


def merge_shard_ledgers(out_root: str | Path, *, batch_id: str = "batch") -> dict[str, Any]:
    """Fold the per-shard attempt ledgers back into one batch summary."""
    root = Path(out_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("attempts_shard_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        merged[(str(row.get("paper_id")), int(row.get("attempt") or 0))] = row
    ordered = [merged[key] for key in sorted(merged)]
    write_text(
        root / LEDGER_FILE,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered),
    )
    elapsed = round(sum(float(row.get("elapsed_seconds") or 0.0) for row in ordered), 3)
    summary = _summarise(batch_id, ordered, elapsed=elapsed, started_at=_utc_now())
    summary["shards"] = len(sorted(root.glob("attempts_shard_*.jsonl")))
    summary["shard"] = None
    halted: list[dict[str, Any]] = []
    for path in sorted(root.glob(f"batch_summary_shard_*.json")):
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("halted_reason"):
            halted.append({"shard": payload.get("shard"), "reason": payload.get("halted_reason"), "halted_after": payload.get("halted_after")})
    if halted:
        # The shards stop for the account's sake; the merged summary has to
        # carry that up, or the batch looks like it simply ran out of papers.
        summary["halted_reason"] = str(halted[0].get("reason") or "")
        summary["halted_shards"] = halted
    write_json(root / SUMMARY_FILE, summary)
    return summary


__all__ = [
    "BATCH_CONTRACT",
    "FAILURE_FILE",
    "LEDGER_FILE",
    "MAX_ATTEMPTS",
    "SUMMARY_FILE",
    "WITHHELD_DELIVERED",
    "record_attempt_failure",
    "attempt_directory",
    "make_paper_evolve_runner",
    "merge_shard_ledgers",
    "run_corpus_batch",
    "run_supervised_batch",
    "sum_provider_tokens",
]
