"""Build reference-versus-generated Judge pairs from a run and the middleware truth.

This is the third preference source. The middleware publishes the reference figure of one
paper next to its scientific ground truth; a ``paper-evolve-editable`` run publishes the
candidates it generated for that same paper. The pair to judge is "the published figure
against the candidate the pipeline wants to ship", which is the only pair that measures
real generation quality instead of an injected defect.

Nothing here judges by eye. A candidate enters the dataset only when the run itself
recorded a Grounding Gate verdict for it. A gate-blocked candidate is certified worse
than the reference and is labelled from the failed check. A candidate that passed the
gate is weaker than the paper's own published figure too, and the two ways of saying so
stay apart: one where a deterministic quality report flagged the compiled deck
(``correct_but_ugly``), and one where nothing could fault it at all
(``assumed_worse``, the only uncertified label, which is why it is off by default and
recorded as not human audited when it is on).

The merge check compares contracts, never their derivatives. The run's own
``scientific_ground_truth.json`` is compiled from its figure content plan, so it differs
from the middleware's file by construction; what a run that adopted the middleware's
contract files instead is a verbatim copy of that contract
(:data:`UPSTREAM_CONTRACT_FILE`). A paper whose middleware truth and generation run were
not built against one contract is refused rather than compared across two sciences, and
so is a copy that names another paper or another contract version.

The label reuses the canonical defect vocabulary: every failed gate check is mapped to the
defect key whose :func:`rfs.judge_training.judge_schema.defect_label` vector the Judge has
to produce, so a gate-blocked candidate is labelled exactly like a controlled defect of
the same kind, and its evidence quotes the gate that blocked it.

What the pair judges against is the paper's own method inventory - the list of concepts
and relations the paper states, read from the middleware's frozen specification with the
figure's selection keys stripped (:mod:`.method_inventory`). The middleware's figure
content plan states the same concepts plus which of them the middleware's own untrained
model wanted drawn; handing that over would score the Judge against that model's taste.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ..contracts.semantic import UPSTREAM_GROUND_TRUTH_COPY as UPSTREAM_CONTRACT_FILE
from .dataset_builder import paper_identity
from .judge_schema import defect_label
from .preference_pairs import _write_jsonl, mirror_judgement
from .preference_sources import GenerationRunSource, MiddlewareSource, SourceUnavailable, run_directory_candidates


REAL_PAIR_CONTRACT = "rfs.judge_real_vs_generated"
REAL_PAIR_VERSION = 3
DATASET_VERSION = "judge-dataset-v005"
TASK_TYPE = "pairwise_preference"
CONTRACT_FILE = "scientific_ground_truth.json"
#: A run that adopted the middleware contract keeps a verbatim copy of the contract's own
#: ground truth under this name, and that copy is what the merge check compares.
ADOPTED_CONTRACT_ROUTE = "adopted_upstream_contract"
BYTE_IDENTICAL_ROUTE = "byte_identical"
EVIDENCE_FILE = "evidence_map.json"
GATE_FILE = "grounding_gate.json"
DEFECT_SOURCE = "grounding_gate"
QUALITY_DEFECT_SOURCE = "deterministic_quality_reports"
ASSUMED_DEFECT_SOURCE = "ranking_without_evidence"
DEFECT_SOURCES = {
    "gate_blocked": DEFECT_SOURCE,
    "correct_but_ugly": QUALITY_DEFECT_SOURCE,
    "assumed_worse": ASSUMED_DEFECT_SOURCE,
}
CERTIFIED_CONFIDENCE = 1.0
#: An ``assumed_worse`` pair claims a ranking, not a defect, so it never claims certainty.
ASSUMED_CONFIDENCE = 0.5
EVIDENCE_LIMIT = 12
EXPLORATION_DOMAIN = "unknown"
#: The deterministic reports of one compiled candidate that can fault a figure whose
#: science already passed the gate. They measure the deck, not the paper.
VISUAL_TIER_REPORTS = (
    "geometry_quality_report.json",
    "density_quality_report.json",
    "text_readability_report.json",
    "connector_review.json",
    "connector_route_selection_report.json",
    "arrow_quality_report.json",
    "joint_geometry_report.json",
)
PASSING_REPORT_STATUSES = ("pass", "ok", "applied", "clean", "completed")
TIER_GATE_BLOCKED = "gate_blocked"
TIER_CORRECT_BUT_UGLY = "correct_but_ugly"
TIER_ASSUMED_WORSE = "assumed_worse"
VISUAL_MODULE_KIND = "module"
VISUAL_NOTE_REGION_KIND = "notation_region"
REFERENCE_REPEAT_LIMIT = 10
REFERENCE_REGION_LIMIT = 10

# The gate's own checks, mapped onto the defect key whose canned vector and severity the
# Judge should produce. ``graph_contracts_valid`` is absent on purpose: it describes the
# run's layered graphs, not the figure the Judge looks at.
GATE_CHECK_DEFECTS: dict[str, str] = {
    "required_nodes_grounded": "missing_core_entity",
    "required_edges_grounded": "missing_relation",
    "forbidden_labels_absent": "invented_module",
}
EXCLUDED_REASONS = (
    "candidate_has_no_grounding_gate",
    "candidate_passed_grounding_gate",
    "candidate_has_no_rendered_preview",
    "gate_evidence_does_not_describe_the_figure",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return str(value or "").strip()


def _texts(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [text for text in (_text(item) for item in value) if text]
    text = _text(value)
    return [text] if text else []


def shared_contract(truth: MiddlewareSource, run: GenerationRunSource, folder: str) -> dict[str, Any]:
    """Check that the reference and the run were built against one scientific contract.

    The merge between the middleware line and the generation line has to have happened for
    the comparison to mean anything, so a missing or disagreeing contract is a failure and
    not a warning.
    """

    reference = truth.paper_dir(folder) / CONTRACT_FILE
    if not reference.is_file():
        raise SourceUnavailable(f"the middleware reference has no scientific ground truth: {reference}")
    published = run.root / CONTRACT_FILE
    if not published.is_file():
        raise SourceUnavailable(f"the generation run published no scientific ground truth: {published}")
    adopted = run.root / UPSTREAM_CONTRACT_FILE
    if adopted.is_file():
        # The run holds the contract itself, so the check is one byte comparison against
        # the middleware's own file. A copy that disagrees names a different paper or a
        # different contract version, which is exactly what this has to refuse.
        reference_sha, adopted_sha = _sha256(reference), _sha256(adopted)
        if reference_sha != adopted_sha:
            raise SourceUnavailable(
                "the run adopted a middle layer contract that is not this paper's: "
                f"{folder}: adopted {adopted_sha[:12]} != reference {reference_sha[:12]}"
            )
        return {
            "path": str(reference),
            "run_path": str(published),
            "adopted_path": str(adopted),
            "route": ADOPTED_CONTRACT_ROUTE,
            "sha256": reference_sha,
            "body": _json(reference),
        }
    # A run that planned its own contract has no copy to offer, so its own view has to be
    # the middleware's byte for byte. This is the path that refuses a run planned against
    # another contract.
    reference_sha, run_sha = _sha256(reference), _sha256(published)
    if reference_sha != run_sha:
        raise SourceUnavailable(
            "the run and the middleware reference disagree about the scientific contract of "
            f"{folder}: run {run_sha[:12]} != reference {reference_sha[:12]}"
        )
    return {
        "path": str(reference),
        "run_path": str(published),
        "adopted_path": "",
        "route": BYTE_IDENTICAL_ROUTE,
        "sha256": reference_sha,
        "body": _json(reference),
    }


def gate_verdict(gate: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Turn one candidate's Grounding Gate into the certified judgement of the pair."""

    checks = gate.get("checks") if isinstance(gate.get("checks"), dict) else {}
    failed = [str(name) for name, passed in checks.items() if not passed]
    mapped = [name for name in failed if name in GATE_CHECK_DEFECTS]
    if not mapped:
        return {}, ""
    issues = _texts(gate.get("blocking_issues"))
    coverage = f"node_coverage {gate.get('node_coverage')}, edge_coverage {gate.get('edge_coverage')}"
    diagnosis = []
    for check in mapped:
        defect = defect_label(GATE_CHECK_DEFECTS[check])
        quoted = [item for item in issues if check in item]
        diagnosis.append({
            "figure": "B",
            "type": str(defect["diagnosis_type"]),
            "target": f"global:{check}",
            "severity": str(defect["severity"]),
            "evidence": ("; ".join(quoted) if quoted else f"grounding check failed: {check}") + f" ({coverage})",
        })
    primary = defect_label(GATE_CHECK_DEFECTS[mapped[0]])
    judgement = {
        "preferred_candidate": "A",
        "quality_scores": dict(primary["quality_scores"]),
        "diagnosis": diagnosis,
        "confidence": CERTIFIED_CONFIDENCE,
    }
    return judgement, GATE_CHECK_DEFECTS[mapped[0]]


def _domain(truth: MiddlewareSource, run: GenerationRunSource, paper_id: str) -> str:
    for record in truth.index_records():
        if _text(record.get("paper_id")) == paper_id and _text(record.get("domain")):
            return _text(record.get("domain"))
    for record in run.evolution_rounds():
        if _text(record.get("domain")):
            return _text(record.get("domain"))
    return EXPLORATION_DOMAIN


def _labels_of(node: dict[str, Any]) -> str:
    return _text(node.get("label")) or _text(node.get("id"))


def _ground_truth_block(graph: dict[str, Any], contract: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    nodes = {_text(node.get("id")): node for node in (graph.get("nodes") or []) if isinstance(node, dict)}
    edges = {_text(edge.get("id")): edge for edge in (graph.get("edges") or []) if isinstance(edge, dict)}
    required_nodes = _texts(contract.get("required_node_ids")) or _texts(graph.get("required_node_ids"))
    required_edges = _texts(contract.get("required_edge_ids")) or _texts(graph.get("required_edge_ids"))
    relations = []
    for edge_id in required_edges:
        edge = edges.get(edge_id) or {}
        source, target = nodes.get(_text(edge.get("source")), {}), nodes.get(_text(edge.get("target")), {})
        relations.append(f"{_labels_of(source)} -> {_labels_of(target)}" + (f" ({_text(edge.get('relation_type'))})" if _text(edge.get("relation_type")) else ""))
    return {
        "method_summary": _text(inventory.get("figure_goal")) or _text(graph.get("summary")),
        "required_entities": [_labels_of(nodes.get(node_id, {})) for node_id in required_nodes],
        "required_relations": relations,
        "forbidden_labels": _texts(contract.get("forbidden_labels")) or _texts(graph.get("forbidden_labels")),
        "graph_id": _text(graph.get("graph_id")) or _text(contract.get("graph_id")),
        "contract_path": "",
        "contract_sha256": "",
    }


def _required_evidence(graph: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    """Return the evidence ids that the required nodes of the contract hang on."""

    required = set(_texts(contract.get("required_node_ids")) or _texts(graph.get("required_node_ids")))
    wanted: list[str] = []
    for node in graph.get("nodes") or []:
        if isinstance(node, dict) and _text(node.get("id")) in required:
            wanted.extend(_texts(node.get("evidence_ids")))
    return wanted


def _reference_figure_block(visual_graph: dict[str, Any]) -> dict[str, Any]:
    """Describe the author's own figure the way the reading of it recorded what it draws.

    The pair asks the Judge to compare a candidate against this figure, so the sample has
    to say what the figure itself is: how many objects of each role it carries, which of
    its drawings repeat, and which notation regions it prints about itself. The concepts
    stay where they were, in ``required_entities`` and ``required_relations``, so nothing
    that already reads this dataset has to change shape.

    A visual layer nobody read is reported as unavailable rather than filled in: a figure
    no one read is not a figure that draws nothing.
    """
    nodes = [node for node in (visual_graph.get("nodes") or []) if isinstance(node, dict)]
    roles: dict[str, int] = {}
    repeated: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = []
    for node in nodes:
        properties = node.get("properties") if isinstance(node.get("properties"), dict) else {}
        role = _text(properties.get("role"))
        if role:
            roles[role] = roles.get(role, 0) + 1
        kind = _text(node.get("kind"))
        if kind == VISUAL_MODULE_KIND and _text(properties.get("repeat_group")):
            repeated.append({
                "label": _labels_of(node),
                "group": _text(properties.get("repeat_group")),
                "index": properties.get("repeat_index"),
                "count": properties.get("repeat_count"),
            })
        elif kind == VISUAL_NOTE_REGION_KIND:
            entries = properties.get("entries") if isinstance(properties.get("entries"), list) else []
            regions.append({
                "kind": _text(properties.get("kind")) or "notation",
                "explains": _text(properties.get("explains")),
                "label": _labels_of(node),
                "entries": len(entries),
            })
    return {
        "available": bool(nodes),
        "objects": len(nodes),
        "roles": {role: roles[role] for role in sorted(roles)},
        "repeat_drawn": sorted(repeated, key=lambda item: (item["label"], item["group"]))[:REFERENCE_REPEAT_LIMIT],
        "notation_regions": sorted(regions, key=lambda item: (item["kind"], item["label"]))[:REFERENCE_REGION_LIMIT],
    }


def _paper_block(identity: dict[str, Any], graph: dict[str, Any], evidence_map: dict[str, Any], wanted: Iterable[str]) -> dict[str, Any]:
    spans: list[dict[str, Any]] = []
    index = {_text(item.get("id")): item for item in (evidence_map.get("evidence") or []) if isinstance(item, dict)}
    for evidence_id in wanted:
        item = index.get(_text(evidence_id))
        if item is None:
            continue
        spans.append({"evidence_id": _text(item.get("id")), "quote": _text(item.get("text"))[:600], "page": item.get("page"), "kind": _text(item.get("kind"))})
        if len(spans) >= EVIDENCE_LIMIT:
            break
    if not spans:
        spans = [{"evidence_id": _text(item.get("id")), "quote": _text(item.get("text"))[:600], "page": item.get("page")} for item in (evidence_map.get("evidence") or [])[:EVIDENCE_LIMIT] if isinstance(item, dict)]
    return {
        "title": _text(identity.get("title")) or _text(graph.get("summary")) or _text(graph.get("paper_id")),
        "abstract": _text(graph.get("summary")),
        "evidence_spans": spans,
        "source_path": _text(evidence_map.get("source_path")),
    }


def flagged_quality_reports(folder: Path) -> list[dict[str, str]]:
    """Return the deterministic reports of one candidate that did not pass.

    These reports read the compiled deck rather than the paper, so they are the only
    evidence left once a candidate has satisfied every grounding check: a figure can be
    scientifically complete and still carry overlapping nodes or text below the readable
    size. A report that is absent is not a report that failed, so it is skipped.
    """

    flagged: list[dict[str, str]] = []
    for name in VISUAL_TIER_REPORTS:
        path = folder / name
        if not path.is_file():
            continue
        report = _json(path)
        status = _text(report.get("status"))
        if status and status not in PASSING_REPORT_STATUSES:
            flagged.append({"report": name, "status": status})
        elif report.get("ok") is False:
            flagged.append({"report": name, "status": "ok_false"})
    return flagged


def passed_candidate_judgement(flagged: list[dict[str, str]]) -> tuple[dict[str, Any], str, str]:
    """Return the judgement, the defect key and the tier of a candidate the gate passed.

    A deterministic report that flagged the figure ranks it below the published one on
    measured grounds and says which report and status did it. When nothing flagged the
    figure, no evidence says it is worse, so the pair says only that: the published figure
    keeps the ranking by default, at half confidence, and the sample is marked as ranking
    without evidence instead of dressed up as a defect.
    """

    if flagged:
        defect = defect_label(TIER_CORRECT_BUT_UGLY)
        judgement = {
            "preferred_candidate": "A",
            "quality_scores": dict(defect["quality_scores"]),
            "diagnosis": [{
                "figure": "B",
                "type": str(defect["diagnosis_type"]),
                "target": f"global:{flagged[0]['report']}",
                "severity": str(defect["severity"]),
                "evidence": "; ".join(f"{item['report']} reports {item['status']}" for item in flagged),
            }],
            "confidence": CERTIFIED_CONFIDENCE,
        }
        return judgement, TIER_CORRECT_BUT_UGLY, TIER_CORRECT_BUT_UGLY
    defect = defect_label(TIER_ASSUMED_WORSE)
    judgement = {
        "preferred_candidate": "A",
        "quality_scores": dict(defect["quality_scores"]),
        "diagnosis": [{
            "figure": "B",
            "type": str(defect["diagnosis_type"]),
            "target": "global:no_deterministic_evidence",
            "severity": str(defect["severity"]),
            "evidence": (
                "no grounding or quality report faulted this candidate; the paper's own "
                "published figure keeps the ranking by default"
            ),
        }],
        "confidence": ASSUMED_CONFIDENCE,
    }
    return judgement, TIER_ASSUMED_WORSE, TIER_ASSUMED_WORSE


def _candidate_pair(run: GenerationRunSource, round_index: int, variant: int, *, context: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Return the examples of one candidate, or the single reason it was excluded."""

    folder = run.root / f"round_{round_index:02d}" / f"candidate_{variant:02d}"
    gate_path = folder / GATE_FILE
    if not gate_path.is_file():
        return [], "candidate_has_no_grounding_gate"
    gate = _json(gate_path)
    flagged: list[dict[str, str]] = []
    if _text(gate.get("status")) == "pass" or gate.get("ok") is True:
        # The reference pair can only be built from a candidate the run itself certified,
        # and a candidate that passed every check was not. It is still weaker than the
        # paper's own figure, so it is taken when the caller asks for that tier, under a
        # label that says which evidence carries the verdict.
        if not context["include_passed"]:
            return [], "candidate_passed_grounding_gate"
        flagged = flagged_quality_reports(folder)
        label, defect_key, tier = passed_candidate_judgement(flagged)
    else:
        label, defect_key = gate_verdict(gate)
        tier = TIER_GATE_BLOCKED
        if not label:
            return [], "gate_evidence_does_not_describe_the_figure"
    try:
        preview = run.candidate_preview(round_index, variant)
    except SourceUnavailable:
        return [], "candidate_has_no_rendered_preview"

    example_id = f"{run.root.name}-real-vs-generated-r{round_index:02d}-c{variant:02d}"
    row: dict[str, Any] = {
        "example_id": example_id,
        "dataset_version": DATASET_VERSION,
        "ground_truth_id": context["paper_id"],
        "paper_id": context["paper_id"],
        "domain": context["domain"],
        "split": context["split"],
        "task_type": TASK_TYPE,
        "candidate_a": str(context["reference_figure"]),
        "candidate_b": str(preview),
        "labels": label,
        "pair_group": f"{context['paper_id']}|real_vs_generated|round_{round_index:02d}|candidate_{variant:02d}",
        "paper": context["paper"],
        "ground_truth": context["ground_truth"],
        "provenance": {
            "source": "real_vs_generated",
            "defect_source": DEFECT_SOURCES[tier],
            "corruption_type": defect_key,
            "label_tier": tier,
            # Only the two tiers a deterministic check carried are certified. A ranking
            # nobody could fault stays an assumption and never claims a human looked at it.
            "certified": tier != TIER_ASSUMED_WORSE,
            "human_audited": False,
            "quality_reports_flagged": flagged,
            "target": str(label["diagnosis"][0]["target"]),
            "run_dir": str(run.root),
            "recorded_paper_id": context["recorded_paper_id"],
            "paper_identity_route": context["folder_route"],
            "reference_figure": str(context["reference_figure"]),
            "candidate_dir": str(folder),
            "round": round_index,
            "variant": variant,
            "contract_path": context["contract"]["path"],
            "contract_sha256": context["contract"]["sha256"],
            "contract_route": context["contract"]["route"],
            "contract_adopted_path": context["contract"]["adopted_path"],
            "grounding_gate": {
                "status": _text(gate.get("status")),
                "ok": bool(gate.get("ok")),
                "checks": gate.get("checks") or {},
                "blocking_issues": _texts(gate.get("blocking_issues")),
                "node_coverage": gate.get("node_coverage"),
                "edge_coverage": gate.get("edge_coverage"),
            },
        },
    }
    examples = [row]
    if context["include_swapped"]:
        mirrored = dict(row)
        mirrored["example_id"] = f"{example_id}__swap"
        mirrored["candidate_a"] = row["candidate_b"]
        mirrored["candidate_b"] = row["candidate_a"]
        mirrored["labels"] = mirror_judgement(row["labels"])
        examples.append(mirrored)
    return examples, ""


def build_real_vs_generated_examples(truth: MiddlewareSource, run: GenerationRunSource, out_path: str | Path, *, split: str = "train", include_swapped: bool = True, limit: int = 0, include_passed: bool = False) -> dict[str, Any]:
    """Write the reference-versus-candidate examples of one generation run.

    The builder needs both ports, so it refuses before writing anything when the paper is
    not in the middleware version, when the shared contract is absent or disagrees, or when
    not a single candidate carries a verdict that certifies it as worse than the reference.
    ``include_passed`` widens the second condition to every candidate the run recorded a
    gate for: a candidate that passed is still ranked below the paper's own figure, labelled
    from a deterministic quality report when one flagged it and as a ranking without
    evidence when none did. Nothing is written on an empty result: an empty dataset must not
    look like a finished build.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError("a framework Judge example needs a train, validation or test split")
    identity = paper_identity(run.root)
    recorded = _text(identity.get("paper_id"))
    if not recorded:
        raise SourceUnavailable(f"the generation run does not record which paper it drew: {run.root}")
    paper_id, folder, folder_route = truth.folder_for_run(paper_id=recorded, run_directories=run_directory_candidates(run.root))
    contract = shared_contract(truth, run, folder)
    reference_figure = truth.truth_figure(folder)
    graph = truth.layer_graph(folder, "scientific")
    try:
        visual_graph = truth.layer_graph(folder, "visual")
    except SourceUnavailable:
        # A version whose author figures were never read carries no roles, repeats or
        # notation to hand over; the pair still measures the candidate against the
        # paper's own concepts, which is what the sample carried before this block.
        visual_graph = {}
    inventory = truth.method_inventory(folder)
    evidence_map = _json(run.root / EVIDENCE_FILE) if (run.root / EVIDENCE_FILE).is_file() else {}
    ground_truth = _ground_truth_block(graph, contract["body"], inventory)
    ground_truth["reference_figure"] = _reference_figure_block(visual_graph)
    ground_truth["contract_path"] = contract["path"]
    ground_truth["contract_sha256"] = contract["sha256"]
    wanted = _required_evidence(graph, contract["body"])
    context = {
        "paper_id": paper_id,
        "recorded_paper_id": recorded,
        "folder_route": folder_route,
        "domain": _domain(truth, run, paper_id),
        "split": split,
        "reference_figure": reference_figure,
        "include_swapped": bool(include_swapped),
        "include_passed": bool(include_passed),
        "contract": contract,
        "ground_truth": ground_truth,
        "paper": _paper_block(identity, graph, evidence_map, wanted),
    }
    examples: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    candidates = 0
    for round_dir in run.round_dirs():
        round_index = int(round_dir.name.rsplit("_", 1)[-1])
        for candidate_dir in run.candidate_dirs(round_index):
            if limit and candidates >= int(limit):
                break
            candidates += 1
            variant = int(candidate_dir.name.rsplit("_", 1)[-1])
            rows, reason = _candidate_pair(run, round_index, variant, context=context)
            if reason:
                excluded.append({"round": round_index, "variant": variant, "reason": reason})
                continue
            examples.extend(rows)
    if not examples:
        raise SourceUnavailable(
            "no candidate of this run carries a Grounding Gate verdict that certifies it as worse "
            "than the reference; pass include_passed to rank the candidates that passed below the "
            "paper's own figure as well"
        )
    target = Path(out_path)
    _write_jsonl(target, examples)
    report = {
        "contract": REAL_PAIR_CONTRACT,
        "version": REAL_PAIR_VERSION,
        "summary": "Reference-versus-generated Judge examples of one generation run.",
        "ok": True,
        "paper_id": paper_id,
        "middleware_folder": folder,
        "recorded_paper_id": context["recorded_paper_id"],
        "folder_route": context["folder_route"],
        "run_dir": str(run.root),
        "reference_figure": str(reference_figure),
        "contract": {
            "path": contract["path"],
            "run_path": contract["run_path"],
            "adopted_path": contract["adopted_path"],
            "route": contract["route"],
            "sha256": contract["sha256"],
        },
        "include_passed_candidates": bool(include_passed),
        "counts": {
            "candidates": candidates,
            "examples": len(examples),
            "swapped": sum(1 for row in examples if str(row["example_id"]).endswith("__swap")),
            "excluded": len(excluded),
            "evidence_spans": len(context["paper"]["evidence_spans"]),
        },
        "tier_distribution": _distribution(_text(row["provenance"]["label_tier"]) for row in examples if not str(row["example_id"]).endswith("__swap")),
        "defect_distribution": _distribution(_text(row["provenance"]["corruption_type"]) for row in examples if not str(row["example_id"]).endswith("__swap")),
        "diagnosis_distribution": _distribution(entry["type"] for row in examples if not str(row["example_id"]).endswith("__swap") for entry in row["labels"]["diagnosis"]),
        "exclusions": excluded,
        "out": str(target),
        "split": split,
    }
    reference = ground_truth.get("reference_figure") or {}
    report["reference_figure"] = {
        "available": bool(reference.get("available")),
        "objects": int(reference.get("objects") or 0),
        "roles": dict(reference.get("roles") or {}),
        "repeat_drawn": len(reference.get("repeat_drawn") or []),
        "notation_regions": len(reference.get("notation_regions") or []),
    }
    target.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = _text(value) or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
