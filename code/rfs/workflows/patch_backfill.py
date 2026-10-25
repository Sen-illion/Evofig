"""Render Prompt Patches for the Memory rows an experience library already holds.

The loop renders a patch from the row it has just measured, so a library that predates that
wiring holds rows no patch was ever derived from. This module reads the library and does the
same rendering for the rows that have none, through the same renderer and the same contract
validation the loop uses, so a backfilled patch cannot differ from the one the loop would
have written for the same row.

A row that cannot render a patch is reported with its reason. Nothing is invented and nothing
is dropped silently: the report names every row and what happened to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..evolution.direct import ExperienceStore, PromptPatchStore, build_patches_for_experience
from ..utils import write_json


PATCH_BACKFILL_CONTRACT = "rfs.prompt_patch_backfill"
PATCH_BACKFILL_VERSION = 1
PATCH_BACKFILL_REPORT_FILE = "prompt_patch_backfill.json"


def backfill_prompt_patches(experience_library: str | Path, *, out: str | Path | None = None) -> dict[str, Any]:
    """Write the missing patch for every measured row and report each row's outcome."""

    root = Path(experience_library).resolve()
    store = ExperienceStore(root)
    patch_store = PromptPatchStore(root)
    patches = patch_store.records()
    sourced = {
        str(experience_id)
        for record in patches
        for experience_id in (record.get("source_experience_ids") or [])
        if str(experience_id)
    }
    results: list[dict[str, Any]] = []
    written = 0
    skipped = 0
    for row in store.records():
        experience_id = str(row.get("experience_id") or "")
        if not experience_id:
            results.append({"experience_id": "", "status": "skipped", "reason": "the row carries no experience_id"})
            skipped += 1
            continue
        if experience_id in sourced:
            results.append({"experience_id": experience_id, "status": "skipped", "reason": "a patch already names this row"})
            skipped += 1
            continue
        rendering = build_patches_for_experience(row)
        patch = rendering.get("patch")
        if not isinstance(patch, dict):
            results.append({"experience_id": experience_id, "status": "rejected", "reason": str(rendering.get("reason") or ""), "errors": list(rendering.get("errors") or [])})
            continue
        stored = patch_store.append_result(patch)
        if not stored.get("ok"):
            results.append({"experience_id": experience_id, "status": "rejected", "patch_id": str(patch.get("patch_id") or ""), "reason": str(stored.get("reason") or ""), "errors": list(stored.get("errors") or [])})
            continue
        sourced.add(experience_id)
        written += 1
        results.append({"experience_id": experience_id, "status": "written", "patch_id": str(patch.get("patch_id") or ""), "patch_class": str(patch.get("class") or ""), "apply_at": str(patch.get("apply_at") or "")})
    report = {
        "contract": PATCH_BACKFILL_CONTRACT,
        "version": PATCH_BACKFILL_VERSION,
        "summary": "Prompt patches backfilled from an existing experience library.",
        "ok": True,
        "library": str(root),
        "rows": len(results),
        "written": written,
        "skipped": skipped,
        "rejected": len(results) - written - skipped,
        "results": results,
    }
    write_json(Path(out).resolve() if out else root / PATCH_BACKFILL_REPORT_FILE, report)
    return report


__all__ = ["PATCH_BACKFILL_CONTRACT", "PATCH_BACKFILL_REPORT_FILE", "PATCH_BACKFILL_VERSION", "backfill_prompt_patches"]
