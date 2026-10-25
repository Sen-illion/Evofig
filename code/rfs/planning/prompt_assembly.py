"""Place the promoted prompt patches into the prompt a planning stage sends.

Every stage prompt is built from the same five levels, strongest first: the safety and
pipeline boundary, the paper's evidence and scientific contract, the stage's own output
schema and action whitelist, the promoted patches, and finally the reference-image prior.
Only the fourth level moves here, and it can only ever be additive text: a patch cannot
restate an output schema, widen the action whitelist, or outrank the evidence above it.

The module deliberately knows nothing about the experience library. A caller resolves the
patches elsewhere and hands the chosen ones in, so ``planning`` never depends on
``evolution``.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..contracts.prompt_patch import PATCH_PARAMETER_KEYS, PROMPT_ASSEMBLY_ORDER

PATCH_BLOCK_LABEL = "PROMOTED_PROMPT_PATCHES"
PARAMETER_PATCH_STAGE = "segmentation"
PARAMETER_PATCH_KEYS = tuple(key for key in PATCH_PARAMETER_KEYS if key not in ("factor", "dx", "dy"))
_SAFETY_TIER = {"prohibited": 2, "executable_geometry": 1, "advisory": 1}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def patch_sort_key(patch: dict[str, Any]) -> tuple:
    """Order the patches injected into one prompt the way precedence ordered them."""

    gain = _number(patch.get("median_repair_gain"))
    return (
        -_SAFETY_TIER.get(str(patch.get("class") or ""), 0),
        -(gain if gain is not None else 0.0),
        -int(patch.get("support") or 0),
        str(patch.get("patch_id") or ""),
        -int(patch.get("version") or 0),
    )


def ordered_patches(patches: Any) -> list[dict[str, Any]]:
    """Return the patches in their fixed injection order, dropping anything malformed."""

    items = [item for item in patches if isinstance(item, dict)] if isinstance(patches, Iterable) else []
    return sorted(items, key=patch_sort_key)


def render_patch_block(patches: Any) -> str:
    """Render the patch level as prompt text, or nothing when this stage has no patch."""

    chosen = ordered_patches(patches)
    if not chosen:
        return ""
    lines = [
        "",
        f"{PATCH_BLOCK_LABEL}: measured directives from earlier runs of this pipeline.",
        "They refine the instructions above and never override them, the paper evidence, "
        "the output schema, or the legal action list.",
    ]
    for patch in chosen:
        directive = str(patch.get("directive") or "").strip()
        if not directive:
            continue
        lines.append(f"- [{str(patch.get('class') or 'advisory')}] {directive}")
    return "\n".join(lines) + "\n"


def assemble_prompt(base_prompt: Any, patches: Any = (), *, reference_prior: str = "") -> str:
    """Assemble one stage prompt: everything, then the patches, then the reference prior."""

    return str(base_prompt) + render_patch_block(patches) + (f"\n{reference_prior}\n" if str(reference_prior or "").strip() else "")


def patch_parameters(patches: Any, *, apply_at: str = PARAMETER_PATCH_STAGE) -> dict[str, float]:
    """Collect the deterministic parameters a parameter-only patch stage may override.

    The highest-precedence patch wins each key, so two parameter patches cannot produce a
    value that depends on their storage order.
    """

    overrides: dict[str, float] = {}
    for patch in ordered_patches(patches):
        if str(patch.get("apply_at") or "") != str(apply_at):
            continue
        declared = patch.get("parameters") if isinstance(patch.get("parameters"), dict) else {}
        for key in PARAMETER_PATCH_KEYS:
            value = _number(declared.get(key))
            if value is not None and key not in overrides:
                overrides[key] = value
    return overrides


def effective_thresholds(defaults: dict[str, Any], parameters: dict[str, Any] | None) -> dict[str, Any]:
    """Apply parameter overrides onto the defaults a deterministic check measures against."""

    merged = dict(defaults)
    for key, value in (parameters or {}).items():
        number = _number(value)
        if number is not None and key in merged:
            merged[key] = number
    return merged


def assemble_manifest(*, apply_at: str, selected: Any, rejected: Any = (), reason: str = "") -> dict[str, Any]:
    """Describe which patch a stage used, in the shape the run report keeps."""

    return {
        "order": list(PROMPT_ASSEMBLY_ORDER),
        "apply_at": str(apply_at),
        "selected": [str(item.get("patch_id") or "") for item in ordered_patches(selected)],
        "classes": {str(item.get("patch_id") or ""): str(item.get("class") or "") for item in ordered_patches(selected)},
        "rejected": [dict(item) for item in rejected if isinstance(item, dict)],
        "reason": str(reason),
    }


def describe_patches(patches: Any) -> str:
    """Return a compact JSON rendering, for callers that log what they injected."""

    return json.dumps([{"patch_id": item.get("patch_id"), "class": item.get("class"), "directive": item.get("directive")} for item in ordered_patches(patches)], ensure_ascii=False)


__all__ = [
    "PARAMETER_PATCH_KEYS",
    "PARAMETER_PATCH_STAGE",
    "PATCH_BLOCK_LABEL",
    "PROMPT_ASSEMBLY_ORDER",
    "assemble_manifest",
    "assemble_prompt",
    "describe_patches",
    "effective_thresholds",
    "ordered_patches",
    "patch_parameters",
    "patch_sort_key",
    "render_patch_block",
]
