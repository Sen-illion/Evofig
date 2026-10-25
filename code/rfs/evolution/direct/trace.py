"""Assemble the per-round audit of which Memory, patch and rule the run actually used.

The retrieval report says what could have applied, and the routing report says which row won
its stage; this module answers the question asked afterwards -- for each round, which
experience was injected, which prompt patch every stage carried, which patches lost the
precedence decision and why, which creator rule the round ran under, and whether the
prohibitions held. It reads state the run already produced and calls no provider.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ...contracts.prompt_patch import PROMPT_ASSEMBLY_ORDER

MEMORY_APPLICATION_TRACE_CONTRACT = "rfs.memory_application_trace"
MEMORY_APPLICATION_TRACE_VERSION = 1
TRACE_SOURCE = "run_state"


def _rows(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, (list, tuple)) else []


def _ids(value: Any) -> list[str]:
    if isinstance(value, dict):
        return sorted({str(item) for items in value.values() for item in (items or []) if str(item)})
    if isinstance(value, (list, tuple)):
        return sorted({str(item) for item in value if str(item)})
    return []


def _stage_patch_rows(stage_patches: dict[str, Any] | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split the resolved stage patches into what each stage carries and what lost."""

    stages: dict[str, Any] = {}
    rejected: list[dict[str, Any]] = []
    for stage, record in sorted((stage_patches or {}).items()):
        source = record if isinstance(record, dict) else {}
        losses = [
            {
                "stage": str(stage),
                "patch_id": str(item.get("patch_id") or ""),
                "class": str(item.get("class") or ""),
                "reason": str(item.get("reason") or ""),
            }
            for item in _rows(source.get("rejected"))
        ]
        stages[str(stage)] = {
            "applied": [str(item.get("patch_id") or "") for item in _rows(source.get("patches"))],
            "considered": int(source.get("considered") or 0),
            "rejected": [{key: value for key, value in item.items() if key != "stage"} for item in losses],
            "reason": str(source.get("reason") or ""),
        }
        rejected.extend(losses)
    return stages, rejected


def _rules_section(creator_rules: dict[str, Any] | None) -> dict[str, Any]:
    source = creator_rules if isinstance(creator_rules, dict) else {}
    return {
        "applied": {str(stage): list(ids) for stage, ids in (source.get("injected") or {}).items()},
        "withheld": _rows(source.get("withheld")),
        "refuted": [
            {"rule_id": str(item.get("rule_id") or ""), "status": str(item.get("status") or ""), "refutations": int(item.get("refutations") or 0)}
            for item in _rows(source.get("refuted_rules"))
        ],
        "refutations": {str(key): int(value) for key, value in (source.get("refutations") or {}).items()},
        "min_support": source.get("min_support"),
    }


def _routing_losses(report: dict[str, Any] | None) -> list[dict[str, Any]]:
    source = report if isinstance(report, dict) else {}
    return [
        {
            "experience_id": str(item.get("experience_id") or ""),
            "boundary_tier": int(item.get("boundary_tier") or 0),
            "median_repair_gain": item.get("median_repair_gain"),
            "support": int(item.get("support") or 0),
            "reason": f"lost the precedence decision at {source.get('stage') or 'this stage'}",
        }
        for item in _rows(source.get("conflicts"))
        if not item.get("selected")
    ]


def build_memory_application_trace(
    *,
    rounds: Iterable[dict[str, Any]] = (),
    retrieval_reports: Iterable[dict[str, Any]] = (),
    routing_reports: Iterable[dict[str, Any]] = (),
    stage_patches: dict[str, Any] | None = None,
    creator_rules: dict[str, Any] | None = None,
    experience_library: str | Path = "",
    patch_library: str | Path = "",
) -> dict[str, Any]:
    """Build the per-round trace of the Memory, patches and rules a run actually used."""

    stages, patch_losses = _stage_patch_rows(stage_patches)
    by_round: dict[int, list[dict[str, Any]]] = {}
    for item in _rows(retrieval_reports):
        by_round.setdefault(int(item.get("round") or 0), []).append(item)
    routing_by_round: dict[int, list[dict[str, Any]]] = {}
    for item in _rows(routing_reports):
        routing_by_round.setdefault(int(item.get("round") or 0), []).append(item)
    trace_rounds: list[dict[str, Any]] = []
    for record in _rows(rounds):
        index = int(record.get("round") or 0)
        retrieval = (by_round.get(index) or [{}])[0]
        routing = (routing_by_round.get(index) or [{}])[0]
        check = record.get("prompt_patch_check") if isinstance(record.get("prompt_patch_check"), dict) else None
        trace_rounds.append(
            {
                "round": index,
                "accepted": bool(record.get("accepted")),
                "retrieval": {
                    "mode": str(retrieval.get("mode") or ""),
                    "degraded": bool(retrieval.get("degraded")),
                    "candidates": int(retrieval.get("candidates") or 0),
                    "statuses": [str(item) for item in retrieval.get("statuses") or []],
                    "reason": str(retrieval.get("reason") or ""),
                },
                "routing": {"stage": str(routing.get("stage") or ""), "selected": _ids(routing.get("selected")), "reason": str(routing.get("reason") or "")},
                "rejected_experiences": _routing_losses(routing),
                "injected_experience": record.get("injected_experience"),
                "applied_rules": dict(record.get("applied_rules") or {}),
                "applied_prompt_patches": dict(record.get("applied_prompt_patches") or {}),
                "prohibition_check": {
                    "ok": bool(check.get("ok")),
                    "checked": [str(item.get("patch_id") or "") for item in _rows(check.get("checked"))],
                    "violations": _rows(check.get("violations")),
                    "unchecked": _rows(check.get("unchecked")),
                }
                if check
                else None,
            }
        )
    rules = _rules_section(creator_rules)
    injected = [str((item.get("injected_experience") or {}).get("experience_id") or "") for item in trace_rounds]
    return {
        "contract": MEMORY_APPLICATION_TRACE_CONTRACT,
        "version": MEMORY_APPLICATION_TRACE_VERSION,
        "source": TRACE_SOURCE,
        "assembly_order": list(PROMPT_ASSEMBLY_ORDER),
        "libraries": {"experience": str(experience_library), "prompt_patch": str(patch_library)},
        "prompt_patches": {
            "stages": stages,
            "applied": sorted({item for record in stages.values() for item in record["applied"]}),
            "rejected": patch_losses,
        },
        "rules": rules,
        "rounds": trace_rounds,
        "totals": {
            "rounds": len(trace_rounds),
            "experiences_injected": sorted({item for item in injected if item}),
            "patches_applied": sorted({item for record in stages.values() for item in record["applied"]}),
            "patches_rejected": len(patch_losses),
            "rules_applied": _ids(rules["applied"]),
            "rules_withheld": len(rules["withheld"]),
            "rules_refuted": len(rules["refuted"]),
        },
    }


__all__ = [
    "MEMORY_APPLICATION_TRACE_CONTRACT",
    "MEMORY_APPLICATION_TRACE_VERSION",
    "TRACE_SOURCE",
    "build_memory_application_trace",
]
