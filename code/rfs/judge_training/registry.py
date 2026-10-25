"""Checkpoint manifests and role-safe Judge registration.

The registry stores metadata only; it never loads model weights.  This keeps
model serving/training dependencies optional and makes promotion auditable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


ROLES = {"shadow", "online", "frozen", "experimental"}
SERVING_ROLES = ("online", "frozen")


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def validate_manifest(manifest: dict[str, Any], *, checkpoint: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    for key in ("model_id", "dataset_version"):
        if not str(manifest.get(key) or "").strip():
            errors.append(f"missing {key}")
    roles = manifest.get("roles", [])
    if not isinstance(roles, list) or not roles:
        errors.append("roles must be a non-empty list")
    elif any(str(role) not in ROLES for role in roles):
        errors.append("roles contains unsupported value")
    promotion = manifest.get("promotion") if isinstance(manifest.get("promotion"), dict) else {}
    if any(str(role) in {"online", "frozen"} for role in roles):
        for key in ("approved", "hidden_test_passed", "adversarial_passed"):
            if promotion.get(key) is not True:
                errors.append(f"promotion.{key} must be true for online/frozen")
    if "frozen" in roles and promotion.get("shadow_stable") is not True:
        errors.append("promotion.shadow_stable must be true for frozen")
    if not str(manifest.get("git_commit") or "").strip():
        errors.append("missing git_commit")
    if checkpoint is not None and not Path(checkpoint).exists():
        errors.append("checkpoint path does not exist")
    return errors


def register_checkpoint(checkpoint: str | Path, manifest: dict[str, Any], registry: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    errors = validate_manifest(manifest, checkpoint=checkpoint_path)
    if errors:
        raise ValueError("invalid Judge checkpoint manifest: " + "; ".join(errors))
    record = dict(manifest)
    record.update({"checkpoint": str(checkpoint_path), "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "registry_status": "registered"})
    registry_path = Path(registry).resolve()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    current = _read(registry_path)
    entries = current.get("models") if isinstance(current.get("models"), list) else []
    entries = [item for item in entries if isinstance(item, dict) and item.get("model_id") != record["model_id"]]
    entries.append(record)
    registry_path.write_text(json.dumps({"schema_version": "1.0", "models": entries}, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def resolve_checkpoint(model_id: str, registry: str | Path) -> dict[str, Any]:
    entries = _read(Path(registry)).get("models", [])
    for item in entries if isinstance(entries, list) else []:
        if isinstance(item, dict) and str(item.get("model_id")) == model_id:
            errors = validate_manifest(item, checkpoint=item.get("checkpoint"))
            if errors:
                raise ValueError("registered Judge model is invalid: " + "; ".join(errors))
            return item
    raise KeyError(f"Judge model is not registered: {model_id}")


def checkpoint_identity(manifest: dict[str, Any]) -> str:
    """Name the weights a run was measured under, rather than the endpoint serving them.

    A registration is replaced in place when the same ``model_id`` is trained again, so the
    id alone does not separate two trainings. The dataset version and the commit do, and
    both are already required of every manifest.
    """

    model_id = str(manifest.get("model_id") or "").strip()
    if not model_id:
        return ""
    return f"{model_id}@{str(manifest.get('dataset_version') or '').strip()}+{str(manifest.get('git_commit') or '').strip()}"


def resolve_serving_checkpoint(model_id: str, registry: str | Path, *, roles: tuple[str, ...] = SERVING_ROLES) -> dict[str, Any]:
    """Resolve a checkpoint that is allowed to judge production runs.

    ``resolve_checkpoint`` already refuses a manifest that claims ``online`` or ``frozen``
    without the promotion gates behind it. What it cannot refuse is a checkpoint that never
    claimed those roles: one registered under ``--allow-experimental`` is a valid record of
    a Judge that did not pass, and reading it back is fine. Serving one is not, so the roles
    a caller needs are named here and a checkpoint that lacks them is declined.
    """

    record = resolve_checkpoint(model_id, registry)
    declared = {str(role) for role in record.get("roles") or []}
    missing = [role for role in roles if role not in declared]
    if missing:
        raise ValueError(
            f"Judge checkpoint {model_id} is registered as {', '.join(sorted(declared)) or 'nothing'} "
            f"and may not serve {', '.join(missing)}: it did not pass the promotion gate for those roles"
        )
    return record
