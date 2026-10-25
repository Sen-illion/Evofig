"""Read one station's artifacts into the evidence its slot in the session report carries.

A session report is only worth trusting if each station says what it actually recalled and
what it actually wrote. That reading is mechanical and identical for every station of a kind,
so it lives here rather than inside the driver: the driver decides what runs next, this module
says what the run left behind.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def round_rows(attempt_dir: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(attempt_dir.glob(f"round_*/{name}")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def evolve_evidence(attempt_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Collect what one evolve attempt recalled, injected, refused and recorded."""

    trace = result.get("memory_application_trace") if isinstance(result.get("memory_application_trace"), dict) else {}
    totals = trace.get("totals") if isinstance(trace.get("totals"), dict) else {}
    rounds = [item for item in trace.get("rounds") or [] if isinstance(item, dict)]
    recall = [
        {
            "round": int(item.get("round") or 0),
            "candidates": int((item.get("retrieval") or {}).get("candidates") or 0),
            "reason": str((item.get("retrieval") or {}).get("reason") or ""),
            "injected": str((item.get("injected_experience") or {}).get("experience_id") or ""),
            "status": str((item.get("injected_experience") or {}).get("status") or ""),
        }
        for item in rounds
    ]
    losses = [dict(loss) for item in rounds for loss in item.get("rejected_experiences") or [] if isinstance(loss, dict)]
    figure = attempt_dir / "final_draft_preview.png"
    manifest = result.get("prompt_patch_manifest") if isinstance(result.get("prompt_patch_manifest"), dict) else {}
    candidates = round_rows(attempt_dir, "experience_candidate.json")
    return {
        "run_ok": bool(result.get("ok")),
        "rounds": int(result.get("rounds_completed") or 0),
        "figure": str(figure) if figure.is_file() else "",
        "outputs": str(result.get("out_dir") or attempt_dir),
        "experience_library": str(result.get("experience_library") or ""),
        "experiences_retrieved": [str(item) for item in result.get("retrieved_strategy_ids") or [] if item],
        "experiences_injected": [str(item) for item in result.get("injected_experience_ids") or [] if item],
        "recall": recall,
        "rejected_experiences": losses,
        "candidate_experience_ids": [str(item.get("experience_id") or "") for item in candidates],
        "candidate_applicability": {
            str(item.get("experience_id") or ""): str(item.get("applicability") or "").strip().casefold()
            for item in candidates
            if str(item.get("experience_id") or "")
        },
        "refuted_experience_ids": [str(item.get("experience_id") or "") for item in round_rows(attempt_dir, "experience_refuted.json")],
        # The patches this run permitted, in id order. A patch earned here is a sentence a later
        # prompt may carry, and it is this list -- not the injection set -- that the loop hands to
        # the paired ON/OFF, so a patch is only asked to travel once it holds its own-case licence.
        "prompt_patches_earned": sorted(
            {str((row or {}).get("prompt_patch_recorded") or "") for row in (result.get("rounds") or []) if isinstance(row, dict)} - {""}
        ),
        "patches_applied": [str(item) for item in totals.get("patches_applied") or []],
        "patches_rejected": int(totals.get("patches_rejected") or 0),
        "rules_applied": [str(item) for item in totals.get("rules_applied") or []],
        "rules_withheld": int(totals.get("rules_withheld") or 0),
        "rules_refuted": int(totals.get("rules_refuted") or 0),
        "prompt_patch_manifest": {str(key): value for key, value in manifest.items()},
        "memory_application_trace": str(attempt_dir / "memory_application_trace.json") if (attempt_dir / "memory_application_trace.json").is_file() else "",
    }


def validate_evidence(attempt_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Collect the ladder result of one validation attempt, case by case."""

    cases = []
    for item in report.get("results") or []:
        if not isinstance(item, dict):
            continue
        off = item.get("off") if isinstance(item.get("off"), dict) else {}
        on = item.get("on") if isinstance(item.get("on"), dict) else {}
        cases.append(
            {
                "case_id": str(item.get("case_id") or ""),
                "domain": str(item.get("domain") or ""),
                "split": str(item.get("split") or ""),
                "measured": bool(item.get("measured")),
                "target_delta": item.get("target_delta"),
                "passed": bool(item.get("passed")),
                "new_blockers": int(item.get("new_blockers") or 0),
                "off_score": off.get("overall_score"),
                "on_score": on.get("overall_score"),
                "off_actions": off.get("action_sequence"),
                "on_actions": on.get("action_sequence"),
            }
        )
    return {
        "status": str(report.get("status") or ""),
        "stopped_because": str((report.get("ladder") or {}).get("stopped_because") or ""),
        "summary": {str(key): value for key, value in (report.get("summary") or {}).items()},
        "target_metric": str(report.get("target_metric") or ""),
        "all_pairs_isolated": bool(report.get("all_pairs_isolated")),
        "all_cases_measured": bool(report.get("all_cases_measured")),
        "cases": cases,
        "report": str(attempt_dir / "cross_paper_validation.json") if (attempt_dir / "cross_paper_validation.json").is_file() else "",
    }


def session_markdown(report: dict[str, Any]) -> str:
    budget = report["budget"]
    lines = [
        "# Evolution session report",
        "",
        f"- Session: `{report['session_id'] or 'unspecified'}`",
        f"- Config: `{report['config'] or 'inline'}`",
        f"- Experience library: `{report['experience_library'] or 'none'}`",
        f"- Validation suite: `{report['validation_suite'] or 'none'}`",
        f"- Budget: {budget['hours']} h, used {budget['used_seconds']} s, exhausted: `{budget['exhausted']}`",
        f"- Stations: {report['counts']['stations']} (evolve {report['counts']['evolve']}, validate {report['counts']['validate']})",
        f"- Dry run: `{report['dry_run']}`",
        "",
        "| Station | Kind | Status | Reason | Out | Seconds |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    lines += [
        f"| {item['station_id']} | {item['kind']} | {item['status']} | {item['reason']} | `{item['out_dir']}` | {item['seconds']} |"
        for item in report["stations"]
    ]
    for item in report["stations"]:
        evidence = item.get("evidence") or {}
        if item["kind"] == "evolve" and evidence:
            lines += [
                "",
                f"## {item['station_id']}",
                "",
                f"- Rounds: {evidence.get('rounds')}, figure: `{evidence.get('figure') or 'none'}`",
                f"- Retrieved: {', '.join(evidence.get('experiences_retrieved') or []) or 'none'}",
                f"- Injected: {', '.join(evidence.get('experiences_injected') or []) or 'none'}",
                f"- Patches applied: {', '.join(evidence.get('patches_applied') or []) or 'none'} (rejected {evidence.get('patches_rejected')})",
                f"- Rules applied: {', '.join(evidence.get('rules_applied') or []) or 'none'} (withheld {evidence.get('rules_withheld')}, refuted {evidence.get('rules_refuted')})",
                f"- New experience rows: {', '.join(evidence.get('candidate_experience_ids') or []) or 'none'}",
            ]
            for row in evidence.get("recall") or []:
                lines.append(f"- Round {row['round']} recall: {row['candidates']} candidates, injected `{row['injected'] or 'none'}` at `{row['status'] or 'unknown rung'}` ({row['reason']})")
        if item["kind"] == "validate" and evidence:
            lines += [
                "",
                f"## {item['station_id']}",
                "",
                f"- Ladder: `{evidence.get('status')}` ({evidence.get('stopped_because') or 'no reason given'})",
                f"- Status before: `{evidence.get('status_before') or 'unknown'}`, after: `{evidence.get('status_after') or 'unknown'}`",
                f"- Cases: {len(evidence.get('cases') or [])}",
            ]
            for row in evidence.get("cases") or []:
                lines.append(f"- `{row['case_id']}` ({row['split']}/{row['domain']}): delta {row['target_delta']}, passed {row['passed']}, new blockers {row['new_blockers']}")
    if report["blockers"]:
        lines += ["", "## Blockers", ""] + [f"- `{item['station_id']}`: {item['reason']} - {item['detail']}" for item in report["blockers"]]
    return "\n".join(lines) + "\n"


__all__ = [
    "evolve_evidence",
    "round_rows",
    "session_markdown",
    "validate_evidence",
]
