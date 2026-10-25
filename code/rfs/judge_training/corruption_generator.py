"""Plan auditable, contract-driven scientific corruptions for framework figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


# The three gating defect classes come first. Two of them are their own corruption
# type; the arrow class is carried by relation_reversal, which reverses the visible
# direction of one required relation. The remaining classes stay trainable but do not
# enter the section 8 gates, so they are listed after the trio.
CORRUPTION_TYPES = (
    "module_order_error",
    "overlap",
    "relation_reversal",
    "missing_core_entity",
    "invented_module",
    "terminology_corruption",
    "training_inference_confusion",
    "missing_core_stage",
    "beautiful_but_wrong",
    "correct_but_ugly",
)
GATE_CORRUPTION_TYPES = ("module_order_error", "relation_reversal", "overlap")


def _pick(items: list[Any], seed: str) -> Any | None:
    if not items:
        return None
    index = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % len(items)
    return items[index]


def _pick_pair(items: list[Any], seed: str) -> list[Any] | None:
    """Pick two distinct items in a stable order, or None when fewer than two exist."""

    if len(items) < 2:
        return None
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    first = int(digest[:16], 16) % len(items)
    second = int(digest[16:32], 16) % (len(items) - 1)
    return [items[first], items[second + 1 if second >= first else second]]


def _stage_names(training: list[Any], inference: list[Any]) -> list[str]:
    """Return the visible stage descriptions of both flows in reading order."""

    names = []
    for step in training + inference:
        label = step.get("step_description") if isinstance(step, dict) else step
        if label:
            names.append(str(label))
    return names


def plan_corruptions(paper_id: str, figure_id: str, figure_path: str, contract: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Create one deterministic error-edit specification per defect class."""
    expected = contract.get("expected_figure_content") if isinstance(contract.get("expected_figure_content"), dict) else {}
    entities = contract.get("entities", []) if isinstance(contract.get("entities"), list) else []
    relations = contract.get("relations", []) if isinstance(contract.get("relations"), list) else []
    training = contract.get("training_flow", []) if isinstance(contract.get("training_flow"), list) else []
    inference = contract.get("inference_flow", []) if isinstance(contract.get("inference_flow"), list) else []
    required_entities = expected.get("required_entities", []) or [x.get("entity_id") for x in entities if isinstance(x, dict)]
    required_relations = expected.get("required_relations", []) or relations
    entity_names = [str(item.get("entity_id") or "") for item in entities if isinstance(item, dict) and item.get("entity_id")]
    stages = _stage_names(training, inference)
    order_pair = _pick_pair(stages, paper_id + "order") or _pick_pair(entity_names, paper_id + "order")
    overlap_pair = _pick_pair(entity_names, paper_id + "overlap") or _pick_pair(stages, paper_id + "overlap")
    targets = {
        "module_order_error": order_pair,
        "overlap": overlap_pair,
        "missing_core_entity": _pick(required_entities, paper_id + "entity"),
        "invented_module": _pick(expected.get("must_not_invent", []) or ["unsupported_auxiliary_module"], paper_id + "invent"),
        "relation_reversal": _pick(required_relations, paper_id + "relation"),
        "terminology_corruption": _pick(required_entities, paper_id + "term"),
        "training_inference_confusion": {"training": _pick(training, paper_id + "train"), "inference": _pick(inference, paper_id + "infer")},
        "missing_core_stage": _pick(training + inference, paper_id + "stage"),
        "beautiful_but_wrong": _pick(required_relations or required_entities, paper_id + "beautiful"),
        "correct_but_ugly": "readability",
    }
    prompts = {
        "module_order_error": f"Swap the visible positions of the two method stages {order_pair} so the pipeline reads in the wrong order, while every stage, label, arrow style and panel stays identical.",
        "overlap": f"Move the visible module {overlap_pair} so that it partly covers its neighbour, without removing or restyling any content.",
        "missing_core_entity": f"Remove the visible core module corresponding to {targets['missing_core_entity']} while preserving surrounding visual style and layout.",
        "invented_module": f"Add a plausible-looking but unsupported module labelled {targets['invented_module']} into the method flow.",
        "relation_reversal": f"Reverse the visible direction of the key relation {targets['relation_reversal']} without changing other content.",
        "terminology_corruption": f"Replace the visible core term {targets['terminology_corruption']} with a different, scientifically incorrect term.",
        "training_inference_confusion": "Swap one visible training-stage component with one inference-stage component while keeping the diagram polished.",
        "missing_core_stage": f"Remove the visible method stage represented by {targets['missing_core_stage']} and close the layout gap naturally.",
        "beautiful_but_wrong": f"Keep the figure visually polished but alter the scientific fact represented by {targets['beautiful_but_wrong']}.",
        "correct_but_ugly": "Preserve all scientific content but make the figure severely harder to read through poor spacing, tiny text, and weak contrast.",
    }
    preservation = " Preserve the original image pixel-for-pixel outside the smallest local edit region. Do not rewrite, restyle, or regenerate any unrelated text, formula, symbol, arrow, module, panel, color, or layout."
    return [{"corruption_id": f"{figure_id}__{kind}", "paper_id": paper_id, "figure_id": figure_id, "split": split, "source_image": str(Path(figure_path).resolve()), "corruption_type": kind, "target": targets[kind], "edit_instruction": prompts[kind] + preservation, "expected_error": kind, "status": "planned"} for kind in CORRUPTION_TYPES]


def build_corruption_manifest(figures_path: str | Path, structured_dir: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Write deterministic corruption requests without claiming image-generation success."""
    figures = [json.loads(line) for line in Path(figures_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    contracts = Path(structured_dir)
    requests: list[dict[str, Any]] = []
    for figure in figures:
        aid = str(figure["arxiv_id"])
        contract = json.loads((contracts / f"{aid}.json").read_text(encoding="utf-8"))
        requests.extend(plan_corruptions(aid, str(figure["figure_id"]), str(figure["figure_path"]), contract, str(figure["split"])))
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in requests), encoding="utf-8")
    report = {"schema_version": "judge-corruption-manifest-v1", "requests": len(requests), "figures": len(figures), "types": {kind: sum(item["corruption_type"] == kind for item in requests) for kind in CORRUPTION_TYPES}, "status": "planned_not_generated"}
    target.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
