"""Validate deliverables emitted by the sole direct evolution workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts.method_figure import FIGURE_CONTENT_PLAN_FILE, LEGACY_PLAN_FILE

REQUIRED_JSON = (
    FIGURE_CONTENT_PLAN_FILE,
    "scientific_graph.json",
    "scientific_ground_truth.json",
    "visual_graph.json",
    "editable_graph.json",
    "graph_bindings.json",
    "graph_alignment_report.json",
    "grounding_gate.json",
    "figure_layout_plan_final.json",
    "judgement.json",
    "evolution_rounds.json",
    "evolution_state.json",
    "final_quality_gate.json",
    "paper_evolve_editable_report.json",
)
#: Names an artifact may still sit under from a run written before it was renamed.
LEGACY_ARTIFACT_NAMES = {FIGURE_CONTENT_PLAN_FILE: LEGACY_PLAN_FILE}


def _stored_path(root: Path, name: str) -> Path:
    """Return the file carrying an artifact, which may sit under an earlier name."""
    path = root / name
    if path.is_file():
        return path
    legacy = LEGACY_ARTIFACT_NAMES.get(name)
    return root / legacy if legacy else path


def validate_output(out_dir: str | Path) -> dict[str, Any]:
    """Check direct contracts, process evidence, and the editable PPTX."""
    root = Path(out_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        return {"ok": False, "out_dir": str(root), "errors": ["output directory does not exist"], "warnings": []}
    for name in REQUIRED_JSON:
        path = _stored_path(root, name)
        if not path.is_file():
            errors.append(f"missing required artifact: {name}")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                errors.append(f"artifact must contain a JSON object: {name}")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON artifact {name}: {exc}")
    gate_path = root / "grounding_gate.json"
    if gate_path.is_file():
        try:
            gate = json.loads(gate_path.read_text(encoding="utf-8"))
            if gate.get("status") != "pass":
                errors.append("Grounding Gate did not pass")
        except (OSError, json.JSONDecodeError):
            pass
    pptx = root / "editable_composition.pptx"
    if not pptx.is_file() or pptx.stat().st_size == 0:
        errors.append("missing or empty editable_composition.pptx")
    event_path = root / "evolution_events.jsonl"
    if not event_path.is_file():
        errors.append("missing process trajectory: evolution_events.jsonl")
    else:
        valid_events = 0
        for index, line in enumerate(event_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"invalid evolution event JSON at line {index}")
                continue
            if isinstance(event, dict) and event.get("event"):
                valid_events += 1
        if not valid_events:
            errors.append("evolution_events.jsonl contains no valid events")
    return {"ok": not errors, "out_dir": str(root), "errors": errors, "warnings": warnings}

__all__ = ["validate_output"]
