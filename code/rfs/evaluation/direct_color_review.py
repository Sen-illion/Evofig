"""VLM visual review dedicated to color-only repair decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..providers import call_vlm_json, resolve_vlm_model


def disabled_color_repair_status(review: dict[str, Any] | None) -> dict[str, Any]:
    """Report that color feedback was observed without mutating the draft."""
    return {
        "enabled": False,
        "status": "disabled",
        "repair_requested": bool((review or {}).get("repair_required")),
        "applied": False,
        "reason": "automatic color repair is disabled; preserve initial VLM theme",
    }


def review_color_draft(program: dict[str, Any], preview_path: str | Path, deterministic_report: dict[str, Any], *, reference_path: str | Path | None = None, mode: str = "vlm", model: str | None = None, timeout: int = 180, retries: int = 1, call_metadata: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Review only color harmony; never authorize geometry or semantic changes.

    ``call_metadata`` is an optional caller-owned dict the provider fills with the call's own
    timing, so a round can show what this seat cost instead of only that it ran.
    """
    if mode == "off":
        return {"status": "skipped", "repair_required": False, "feedback": []}, {"ok": True, "mode": "off", "errors": []}
    if mode == "heuristic":
        required = deterministic_report.get("status") != "pass"
        return {"status": "needs_repair" if required else "pass", "repair_required": required, "feedback": deterministic_report.get("issues", [])}, {"ok": True, "mode": "heuristic", "errors": []}
    theme = ((program.get("style") or {}).get("color_theme") or {})
    prompt = "Review ONLY the color system of the current editable scientific PPT draft. Do not propose layout, text, icon, connector-route, node, panel-geometry, or scientific changes. Return JSON only: {status:'pass|needs_repair',repair_required:true|false,feedback:[{target_role:'canvas|panel.<id>.fill|panel.<id>.border|panel.<id>.header|text.primary|connectors.primary|roles.input|roles.module|roles.output',severity:'minor|major',reason:'',suggested_direction:''}],preserve_roles:[]}. Judge harmony, academic restraint, hierarchy, accessibility, panel differentiation, and whether icons feel compatible with their panels. CURRENT_THEME=" + str(theme) + " DETERMINISTIC_REPORT=" + str(deterministic_report)
    images = [preview_path]
    if reference_path and Path(reference_path).exists():
        images.insert(0, reference_path)
    try:
        raw = call_vlm_json(prompt, images, model=resolve_vlm_model("RFS_COLOR_REVIEW_MODEL", "RFS_CRITIC_MODEL", explicit_model=model), timeout=timeout, retries=retries, call_metadata=call_metadata)
    except Exception as exc:
        return (
            {"status": "failed", "repair_required": False, "feedback": [], "provider_error": str(exc)},
            {"ok": False, "mode": "vlm", "errors": [str(exc)]},
        )
    feedback = []
    for item in raw.get("feedback", []) if isinstance(raw.get("feedback"), list) else []:
        if isinstance(item, dict) and str(item.get("target_role") or "").strip():
            feedback.append({"target_role": str(item.get("target_role")), "severity": str(item.get("severity") or "minor"), "reason": str(item.get("reason") or ""), "suggested_direction": str(item.get("suggested_direction") or "")})
    status = str(raw.get("status") or ("needs_repair" if feedback else "pass"))
    review = {"status": status if status in {"pass", "needs_repair"} else "needs_repair", "repair_required": bool(raw.get("repair_required")) or bool(feedback), "feedback": feedback, "preserve_roles": [str(x) for x in raw.get("preserve_roles", []) if isinstance(x, str)]}
    report = {"ok": True, "mode": "vlm", "errors": [], "feedback_count": len(feedback)}
    if call_metadata is not None:
        report["provider"] = dict(call_metadata)
    return review, report
