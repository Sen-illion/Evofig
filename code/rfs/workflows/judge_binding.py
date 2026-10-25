"""Bind a registered Judge checkpoint to the run it is about to measure.

Naming a Judge by its served model name is enough to call it and not enough to audit it:
the name says which endpoint answered, not which weights were behind the endpoint, and it
does not say whether those weights ever passed the promotion gate. This module turns a
``--registry`` plus ``--model-id`` pair into the one binding the generation chain will use,
refusing a checkpoint that is not allowed to serve, and hands back the identity that every
Memory row of that run is stamped with.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..judge_training.registry import SERVING_ROLES, checkpoint_identity, resolve_serving_checkpoint


def resolve_judge_binding(*, judge_mode: str, registry: str | Path | None = None, model_id: str | None = None, roles: tuple[str, ...] = SERVING_ROLES) -> dict[str, Any]:
    """Resolve which registered Judge a run is allowed to be measured by.

    A run that names neither a registry nor a model id is unbound and keeps the old
    behaviour, where the served model name is the whole identity. Naming one of the two is
    refused rather than guessed at, and naming both outside ``trained`` mode is refused too:
    a checkpoint that the run will not actually call must not end up stamped on its rows.
    """

    named = str(model_id or "").strip()
    location = str(registry or "").strip()
    if not named and not location:
        return {}
    if not named or not location:
        raise ValueError("--model-id and --registry name one registered Judge together; pass both or neither")
    if str(judge_mode) != "trained":
        raise ValueError(f"--model-id names a registered checkpoint, which only --judge-mode trained serves; this run asked for {judge_mode!r}")
    record = resolve_serving_checkpoint(named, location, roles=roles)
    return {
        "model_id": str(record.get("model_id") or named),
        "dataset_version": str(record.get("dataset_version") or ""),
        "git_commit": str(record.get("git_commit") or ""),
        "checkpoint": str(record.get("checkpoint") or ""),
        "identity": checkpoint_identity(record),
        "roles": [str(role) for role in record.get("roles") or []],
        "registry": str(Path(location).resolve()),
    }


__all__ = ["resolve_judge_binding"]
