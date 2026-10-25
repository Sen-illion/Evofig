"""Classify generation delivery independently from visual quality."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile


def _contract_degradation(name: str, validation: dict[str, Any]) -> dict[str, Any]:
    """Describe one contract a deck was delivered with despite its own errors.

    The deck exists, so the paper keeps its figure; what the contract could not
    resolve stays visible on the quality side instead of voiding the run.
    """
    errors = [str(item) for item in (validation.get("errors") or []) if str(item).strip()]
    return {
        "source": f"{name}_contract",
        "severity": "degraded",
        "status": "invalid",
        "detail": "; ".join(errors[:3])[:300] or f"the {name} contract reported itself invalid",
    }


def classify_delivery_status(*, compiled: dict[str, Any] | None, root: str | Path, method_validation: dict[str, Any] | None = None, structure_validation: dict[str, Any] | None = None, asset_report: dict[str, Any] | None = None, quality_reports: dict[str, dict[str, Any]] | None = None, visual_review: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return delivery/quality states; a compiled deck is delivered, flaws and all.

    Delivery answers one question: did this run produce a deck.  A method or
    structure contract reporting itself invalid used to answer it "no", and the
    finished PowerPoint was thrown away with the answer -- 207 of the 228
    A-tier papers that ran and delivered no figure lost it exactly there.  Such
    a contract now degrades the quality verdict instead: the deck is delivered,
    and the errors it could not resolve are named in ``contract_degradations``.
    Ordinary visual defects remain repairable, as before.
    """
    root = Path(root)
    blocking: list[str] = []
    if not compiled:
        blocking.append("compile_result_missing")
    pptx = root / "editable_composition.pptx"
    if not pptx.exists() or pptx.stat().st_size == 0 or not zipfile.is_zipfile(pptx):
        blocking.append("editable_pptx_missing_or_empty")
    if isinstance(asset_report, dict):
        for asset in asset_report.get("assets", []):
            delivered_path = root / "assets" / f"{asset.get('asset_id')}.png" if isinstance(asset, dict) else None
            if not delivered_path or not delivered_path.exists():
                blocking.append(f"asset_missing:{asset.get('asset_id') if isinstance(asset, dict) else 'unknown'}")
    issues: list[dict[str, Any]] = []
    degradations: list[str] = []
    for name, validation in (("method", method_validation), ("structure", structure_validation)):
        if isinstance(validation, dict) and validation.get("ok") is False:
            degradations.append(f"{name}_contract_invalid")
            issues.append(_contract_degradation(name, validation))
    for name, report in (quality_reports or {}).items():
        if isinstance(report, dict) and report.get("status") != "pass":
            issues.append({"source": name, "severity": "repairable", "status": report.get("status")})
    if isinstance(visual_review, dict) and visual_review.get("status") not in {None, "pass"}:
        issues.append({"source": "visual_review", "severity": "repairable", "status": visual_review.get("status")})
    generation_status = "failed" if blocking else "completed"
    quality_status = "passed" if not issues else ("needs_repair" if any(i.get("severity") == "repairable" for i in issues) else "warning")
    return {
        "generation_status": generation_status,
        "quality_status": quality_status,
        "delivery_status": "available" if generation_status == "completed" else "unavailable",
        "quality_passed": quality_status == "passed",
        "repair_recommended": bool(issues),
        "quality_issues": issues,
        "blocking_generation_errors": blocking,
        "contract_degradations": degradations,
        "ok": generation_status == "completed",
    }
