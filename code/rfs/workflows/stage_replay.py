"""Validated upstream-artifact replay for controlled layout experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..contracts import normalize_method_figure_content, validate_design_plan
from ..contracts.design_plan import DESIGN_PLAN_SEED_FILE
from ..contracts.evolution import STAGE_ORDER
from ..contracts.method_figure import FIGURE_CONTENT_PLAN_FILE, figure_content_plan_path
from ..utils import read_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_json(root: Path, name: str) -> dict[str, Any]:
    path = root / name
    if not path.is_file():
        raise ValueError(f"stage replay source is missing {name}: {root}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"stage replay artifact must be a JSON object: {path}")
    return value


def _required_plan(root: Path) -> dict[str, Any]:
    """Read the figure content plan of a replay source, under either of its names."""
    path = figure_content_plan_path(root)
    if path is None:
        raise ValueError(f"stage replay source is missing {FIGURE_CONTENT_PLAN_FILE}: {root}")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"stage replay artifact must be a JSON object: {path}")
    return value


def _fold_seed_aliases(design: dict[str, Any], remap: dict[str, str]) -> list[dict[str, Any]]:
    """Drop seed nodes the normalized contract folded away and remap references.

    A replayed seed was planned from the contract as it existed then. Once the
    contract folds an alias node, replaying the seed verbatim would resurrect
    the duplicate card the fold just removed.
    """

    if not remap:
        return []
    folded: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    for node in design.get("nodes", []) or []:
        node_id = str(node.get("id") or "")
        if node_id in remap:
            folded.append({"node_id": node_id, "merged_into": remap[node_id]})
            continue
        parent_id = str(node.get("parent_id") or "")
        if parent_id in remap:
            node["parent_id"] = remap[parent_id]
        nodes.append(node)
    design["nodes"] = nodes
    for panel in design.get("panels", []) or []:
        for key in ("node_ids", "payload_node_ids"):
            if isinstance(panel.get(key), list):
                panel[key] = list(dict.fromkeys(remap.get(str(value), str(value)) for value in panel[key]))
    for group in design.get("groups", []) or []:
        if isinstance(group.get("node_ids"), list):
            group["node_ids"] = list(dict.fromkeys(remap.get(str(value), str(value)) for value in group["node_ids"]))
    connectors: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for edge in design.get("connectors", []) or []:
        for key in ("source", "target"):
            value = str(edge.get(key) or "")
            if value in remap:
                edge[key] = remap[value]
        edge_id = str(edge.get("id") or "")
        if str(edge.get("source") or "") == str(edge.get("target") or ""):
            continue
        if edge_id and edge_id in seen_ids:
            continue
        if edge_id:
            seen_ids.add(edge_id)
        connectors.append(edge)
    design["connectors"] = connectors
    return folded


# This is the pipeline's own list of the stages it can re-run: which artifacts exist and what a
# fork at each of them inherits. It is the attribution vocabulary in ``rfs.contracts.evolution``
# itself, imported above rather than written a second time, down to its first two entries, which
# name where a defect entered rather than a stage this module can replay -- and a fork there is a
# real fork point with a real meaning: "inherit nothing, the reading itself is what is wrong".
# They are refused by :func:`_freeze_plan` below rather than silently downgraded, because the
# honest answer to a caller that asks to fork before the reading is that this source can lend it
# nothing at all.

#: The fork points that re-do the paper's own reading. A fork at either one inherits nothing: the
#: method contract and the content plan are what the reading produces, and every stage below them
#: was planned from that reading. A caller that names one is asking to read the paper again.
READING_FORK_STAGES = ("paper_reading", "content_planning")

#: The stages a replay can inherit, in the order their artifacts become available. The report
#: names every one of them as either reused or re-planned, so a decline is never silent.
REPLAY_STAGES = ("method", "scientific_hierarchy", "relation_explanations", "reference_analysis", "design_seed", "figure_text_plan", "color_theme", "icons")

#: The stages a *controlled* replay has to inherit whole. The paired OFF/ON experiment requires its
#: two arms to start from one byte-identical state, so a stage it would re-plan is a hole in the
#: experiment rather than a gap to record. The continuous loop is the opposite -- it wants whatever
#: the last version can lend -- so it calls with ``strict=False`` and gets the per-stage result.
#: A reference analysis the source never had is not one of these: see :func:`load_stage_replay`.
STRICT_REQUIRED_STAGES = ("relation_explanations", "reference_analysis", "design_seed", "icons")


def _freeze_plan(fork_before: str, freeze_layout: bool) -> dict[str, bool]:
    """Decide which planning artifacts a paired run shares, from the stage it forks at.

    ``fork_before`` names the first stage the run is allowed to plan for itself, in the order
    the contract lists the apply_at stages. Everything before it is the frozen prefix both arms
    of a pair share; the named stage and every stage after it are planned again. An empty value
    keeps the old ``freeze_layout`` behaviour.

    The reading itself is what every stage this module can fork at was planned from, so a fork
    before it inherits nothing at all. That is refused with the reason rather than answered with a
    plan whose every stage says "re-planned": a caller told "nothing can be inherited" can go and
    read the paper again, while a caller handed an empty plan reads it as an inheritance that
    failed for some other reason.
    """

    stage = str(fork_before or "").strip()
    if not stage:
        return {"design": bool(freeze_layout), "text_plan": bool(freeze_layout), "theme": bool(freeze_layout), "icons": True}
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown fork stage: {fork_before}")
    if stage in READING_FORK_STAGES:
        raise ValueError(
            "a fork at the paper reading inherits nothing: the reading and the content plan are "
            "what the run produces from the paper, so there is no replay to load from this source"
        )
    index = STAGE_ORDER.index(stage)
    return {
        "design": index > STAGE_ORDER.index("segmentation"),
        "text_plan": index > STAGE_ORDER.index("theme_planning"),
        "theme": index > STAGE_ORDER.index("theme_planning"),
        "icons": index > STAGE_ORDER.index("icon_generation"),
    }


def _optional_json(root: Path, name: str) -> tuple[dict[str, Any] | None, str]:
    """Read an artifact a replay may do without, and say why when it cannot.

    A file that is absent or unreadable is a gap to record, not a reason to refuse the whole
    source: an earlier iteration can be worth inheriting even when one of its artifacts is gone.
    """

    path = root / name
    if not path.is_file():
        return None, f"{name} is missing from the source"
    try:
        value = read_json(path)
    except (OSError, ValueError) as exc:  # noqa: BLE001 - an unreadable artifact is a gap
        return None, f"{name} could not be read: {exc}"
    if not isinstance(value, dict):
        return None, f"{name} is not a JSON object"
    return value, ""


#: The artifacts a replay may read a design seed from, in the order it trusts them: the response the
#: arm built its figure from, the response it planned first, and last the plan it finished with.
DESIGN_SEED_SOURCES = (DESIGN_PLAN_SEED_FILE, "design_plan_raw.json", "design_plan.json")


def _read_design_seed(root: Path) -> tuple[dict[str, Any] | None, str, str]:
    """Pick the first design seed the contract accepts, and name the artifact it came from.

    A response is preferred to the finished plan because it is the input the source arm's own
    planning ran on, so replaying it reproduces that arm as it was. The finished plan is the last
    resort: it is the boxes the arm drew, which is what a paired replay has to reproduce, and it is
    the only artifact left when the contract has since outgrown the response the arm started from.

    The third value says why a response was passed over. A replay that had to fall back is then
    able to say in its own report which artifact it read its boxes from, and why.
    """

    missing: list[str] = []
    passed_over = ""
    for name in DESIGN_SEED_SOURCES:
        payload, read_gap = _optional_json(root, name)
        if not payload:
            missing.append(read_gap or f"{name} is missing from the source")
            continue
        validation = validate_design_plan(payload)
        if validation.get("ok"):
            return payload, name, passed_over
        passed_over = "the seed no longer validates: " + "; ".join(validation.get("errors") or [])
    return None, "", passed_over or "the source carries no design seed (" + "; ".join(missing) + ")"


def load_stage_replay(
    source: str | Path,
    *,
    paper: str | Path,
    reference: str | Path | None,
    freeze_layout: bool = True,
    fork_before: str = "",
    probe: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Load a prior run stage by stage, keeping every part that is still good.

    ``freeze_layout`` decides how much of the prior run is reused. A controlled
    experiment wants the layout frozen as well, so that stays on by default.
    A retry after a verdict about the layout wants the opposite: keeping the
    reading, the contracts and the reference analysis saves the work the
    verdict had nothing to say about, while the design seed, the text plan and
    the colour theme are planned again, because starting from the same boxes
    redraws the same complaint.

    ``fork_before`` is the paired-run form of the same question. A pair has to share everything
    up to the point the memory acts and differ only from there, so the fork stage is where the
    frozen prefix ends: a memory recorded at ``repair`` is measured against a fully frozen
    figure, and one recorded at ``segmentation`` is allowed to plan its own boxes, because
    freezing the design would leave the memory nothing to change.

    Inheritance is decided one stage at a time. A stage whose artifact is present and valid is
    copied; a stage whose artifact is missing or invalid is planned again by this run, and the
    reason is recorded, instead of the whole source being refused. Only the paper itself, its
    method contract and its scientific hierarchy are all-or-nothing, because everything
    downstream is read against them. The visual reference analysis is deliberately *not* one of
    those: a figure can be produced with no reference at all, so a source that has none records
    a gap and re-plans that stage alone. An invalid design seed is the same kind of gap: the
    contract has moved on, so the boxes are planned again and everything else is kept.

    A source that had no reference to analyse at all -- the analysis it wrote says ``not_found``
    -- is carried over as that same absence even in a strict replay. Analysing a missing reference
    is the same empty answer on every run, so inheriting it leaves the two arms of a pair on one
    starting line; there is no analysis to differ about. Only an analysis that was attempted and
    did not complete is a hole a controlled replay still refuses. Without that distinction a
    deployment with no reference library cannot run a pair at all, which is how every validation
    of every candidate came back as an error instead of a verdict.

    ``probe`` answers "what would this source give me" without building a run from it, which is
    what the continuous loop asks before it commits an iteration to a source.

    ``strict`` is what a controlled experiment needs and a continuous loop does not: with it, a
    stage that would have to be re-planned is refused outright, because the paired arms of an
    experiment must begin from one identical state and a re-planned stage defeats that. Without
    it, the same stage is re-planned and recorded as a gap. The text plan and the colour theme
    stay optional either way: they were never a precondition of a replay.
    """
    root = Path(source).resolve()
    if not root.is_dir():
        raise ValueError(f"stage replay source does not exist: {root}")
    outcomes: dict[str, dict[str, Any]] = {}

    def note(stage: str, status: str, reason: str = "", **extra: Any) -> None:
        record: dict[str, Any] = {"status": status}
        if reason:
            record["reason"] = reason
        record.update(extra)
        outcomes[stage] = record

    def gap(stage: str, reason: str, **extra: Any) -> None:
        """Record a stage this run will plan again, or refuse it in a controlled replay."""

        note(stage, "replanned", reason, **extra)
        if strict and stage in STRICT_REQUIRED_STAGES:
            raise ValueError(reason)

    manifest = _required_json(root, "input_manifest.json")
    document = _required_json(root, "document_model.json")
    paper_sha = _sha256(Path(paper).resolve())
    expected_paper_sha = str(document.get("source_sha256") or "")
    if not expected_paper_sha or paper_sha != expected_paper_sha:
        raise ValueError("stage replay paper SHA-256 does not match the current input")

    expected_references = [Path(item) for item in manifest.get("positive_references", [])]
    reference_sha = None
    reference_gap = ""
    if reference is not None:
        reference_sha = _sha256(Path(reference).resolve())
        if not expected_references or not expected_references[0].is_file():
            reference_gap = "stage replay source has no readable positive reference"
        elif reference_sha != _sha256(expected_references[0]):
            reference_gap = "stage replay reference SHA-256 does not match the current input"

    freeze = _freeze_plan(fork_before, freeze_layout)
    method = normalize_method_figure_content(_required_plan(root))
    method_validation = _required_json(root, "method_extraction_report.json")
    alias_merges = (method.get("provenance") or {}).get("alias_merges") or []
    alias_remap = {str(item.get("alias_id")): str(item.get("merged_into")) for item in alias_merges if isinstance(item, dict) and item.get("alias_id") and item.get("merged_into")}
    # A row that never built the middle layer has no hierarchy to lend, and its
    # absence is the truth about that row rather than a hole in the source. It is
    # recorded as a gap the caller can read; the caller's own switch decides what
    # to do with the empty plan, and a controlled replay still refuses anything it
    # must inherit whole (the hierarchy is not one of those stages).
    hierarchy, hierarchy_gap = _optional_json(root, "scientific_hierarchy_plan.json")
    hierarchy_validation, hierarchy_validation_gap = _optional_json(root, "scientific_hierarchy_validation.json")
    if not method_validation.get("ok") or method_validation.get("fallback_used"):
        raise ValueError("stage replay requires a successful non-fallback method contract")
    if hierarchy_validation and hierarchy_validation.get("fallback_used"):
        raise ValueError("stage replay requires a non-fallback scientific hierarchy")
    note("method", "reused")
    if hierarchy is None or hierarchy_validation is None:
        gap("scientific_hierarchy", hierarchy_gap or hierarchy_validation_gap or "the source carries no scientific hierarchy")
    else:
        note("scientific_hierarchy", "reused")

    relation_explanation_report, relation_gap = _optional_json(root, "relation_explanation_report.json")
    if relation_explanation_report is not None:
        note("relation_explanations", "reused")
    else:
        gap("relation_explanations", relation_gap)

    design: dict[str, Any] = {}
    design_seed: dict[str, Any] | None = None
    folded_seed_nodes: list[dict[str, Any]] = []
    design_validation: dict[str, Any] = {"ok": True, "errors": [], "frozen": False}
    if freeze["design"]:
        # The seed is the response the source arm built its figure from, which is its first response
        # unless a plan repair replaced it. A strict replay must reproduce the arm, and re-deriving
        # from the first response would rebuild the plan the repair rejected; a source written before
        # that artifact existed still has the first response and is replayed from it as it was.
        #
        # Neither response is guaranteed to survive to today: the contract has tightened since, and
        # an arm that repaired its own plan left a first response describing boxes it never drew.
        # Refusing such a source would refuse every pair on it, and inheriting the rejected response
        # would hand the two arms different starting figures. The plan the arm finished with is the
        # one artifact that still says what it drew, so it is read as the seed of last resort; which
        # artifact answered is recorded on the stage, together with why an earlier one was passed
        # over.
        seed, seed_source, seed_problem = _read_design_seed(root)
        if seed is None:
            gap("design_seed", seed_problem)
        else:
            design = seed
            folded_seed_nodes = _fold_seed_aliases(design, alias_remap)
            design_validation = validate_design_plan(design)
            design_seed = design
            note("design_seed", "reused", seed_problem, seed_source=seed_source)
    else:
        note("design_seed", "replanned", "the boxes are planned again from this stage on")

    reference_profile: dict[str, Any] | None = None
    if reference_gap:
        gap("reference_analysis", reference_gap)
    else:
        style_record, style_gap = _optional_json(root, "reference_style_profile.json")
        layout_record, layout_gap = _optional_json(root, "reference_layout_profile.json")
        missing = style_gap or layout_gap
        if missing:
            gap("reference_analysis", missing)
        else:
            status = str(style_record.get("status") or layout_record.get("status") or "complete")
            if status == "not_found":
                # The source was produced with no reference figure to read, and the analysis of a
                # missing reference is the same empty answer on every run, so inheriting it keeps
                # a pair on one starting line. See the note on this function.
                note("reference_analysis", "reused", "the source had no reference figure to analyse (status not_found); the pair proceeds with none")
                reference_profile = {"status": "not_found", "style": {}, "layout": {}, "influence": {"adopted": [], "rejected": ["reference_missing"]}}
            elif status != "complete":
                gap("reference_analysis", f"stage replay requires a complete reference analysis (the source's is {status})")
            else:
                # The influence and template records refine the analysis but do not define it, so a
                # source that has the style and the layout is still worth inheriting from.
                influence, influence_gap = _optional_json(root, "reference_influence_report.json")
                template, template_gap = _optional_json(root, "reference_layout_template.json")
                template_validation, template_validation_gap = _optional_json(root, "reference_layout_template_validation.json")
                notes = [message for message in (influence_gap, template_gap, template_validation_gap) if message]
                reference_profile = {
                    "status": status,
                    "style": {key: value for key, value in style_record.items() if key != "status"},
                    "layout": {key: value for key, value in layout_record.items() if key != "status"},
                    "influence": influence or {},
                    "reference_layout_template": template or {},
                    "reference_layout_template_validation": template_validation or {},
                }
                note("reference_analysis", "reused", notes=notes)

    icon_root = root / "round_00" / "candidate_00" / "assets"
    icon_asset_ids = sorted(path.parent.name for path in icon_root.glob("*/selected.png"))
    icons_reused = bool(icon_asset_ids)
    if not freeze["icons"]:
        note("icons", "replanned", "the Icons are drawn again from this stage on")
    elif icons_reused:
        note("icons", "reused")
    else:
        gap("icons", "stage replay source has no frozen selected Icon assets")

    optional_stages: dict[str, Any] = {}
    # The text plan and the colour theme are keyed to the boxes they were planned for, so a
    # run that plans the boxes again plans these again.
    wanted: list[tuple[str, str, str]] = []
    if freeze["text_plan"]:
        wanted.append(("figure_text_plan.json", "figure_text_plan_validation.json", "figure_text_plan"))
    if freeze["theme"]:
        wanted.append(("color_theme.json", "color_theme_validation.json", "color_theme"))
    for name, validation_name, stage in wanted:
        value, gap = _optional_json(root, name)
        if value is None:
            note(stage, "replanned", gap)
            continue
        optional_stages[stage] = value
        validation, validation_gap = _optional_json(root, validation_name)
        if validation is None:
            note(stage, "reused", notes=[validation_gap])
        else:
            optional_stages[f"{stage}_validation"] = validation
            note(stage, "reused")

    reused_stages = [stage for stage in REPLAY_STAGES if (outcomes.get(stage) or {}).get("status") == "reused"]
    replanned_stages = [stage for stage in REPLAY_STAGES if (outcomes.get(stage) or {}).get("status") == "replanned"]
    gaps = [{"stage": stage, "reason": str((outcomes.get(stage) or {}).get("reason") or "")} for stage in replanned_stages]
    report = {
        "contract": "rfs.stage_replay_report",
        "version": 2,
        "status": "validated",
        "partial": bool(replanned_stages),
        "source": str(root),
        "paper_sha256": paper_sha,
        "reference_sha256": reference_sha,
        "reused_stages": reused_stages,
        "replanned_stages": replanned_stages,
        "stage_outcomes": outcomes,
        "gaps": gaps,
        "layout_frozen": bool(design_seed),
        "fork_before": str(fork_before or ""),
        "frozen_stages": sorted(key for key, value in freeze.items() if value),
        "frozen_icon_asset_ids": icon_asset_ids if icons_reused else [],
        "design_seed_validation": design_validation,
        "folded_contract_alias_ids": sorted(alias_remap),
        "folded_seed_node_ids": folded_seed_nodes,
    }
    if probe:
        return {"probe": True, "accepted": True, "declined_reason": "", "root": root, "report": report}
    return {
        "root": root,
        "method": method,
        "method_validation": method_validation,
        "relation_explanation_report": relation_explanation_report,
        "reference_profile": reference_profile,
        "scientific_hierarchy": hierarchy,
        "hierarchy_validation": hierarchy_validation,
        "design_seed": design_seed,
        "icon_root": icon_root if icons_reused else None,
        "report": report,
        **optional_stages,
    }


def reuse_forecast(
    source: str | Path,
    *,
    paper: str | Path,
    reference: str | Path | None = None,
    freeze_layout: bool = False,
    fork_before: str = "",
) -> dict[str, Any]:
    """Say what an inheritance from ``source`` would keep, before the run commits to it.

    The continuous loop asks this at the top of an iteration so the first report already answers
    "will this wave really continue from the last one, and from which stages", instead of leaving
    the answer to a decline discovered after the run. A source that fails the all-or-nothing core
    (the paper, its method contract, its hierarchy) comes back as ``accepted: false`` with the
    reason; every other gap is a stage that will be re-planned and is listed as such.
    """

    try:
        loaded = load_stage_replay(
            source,
            paper=paper,
            reference=reference,
            freeze_layout=freeze_layout,
            fork_before=fork_before,
            probe=True,
        )
    except (ValueError, OSError) as exc:  # noqa: BLE001 - a declined forecast is a report, not a crash
        return {
            "requested": True,
            "accepted": False,
            "declined_reason": str(exc),
            "fork_before": str(fork_before or ""),
            "reused_stages": [],
            "replanned_stages": [],
            "stage_outcomes": {},
            "gaps": [],
        }
    report = dict(loaded.get("report") or {})
    return {
        "requested": True,
        "accepted": True,
        "declined_reason": "",
        "fork_before": str(fork_before or ""),
        "reused_stages": list(report.get("reused_stages") or []),
        "replanned_stages": list(report.get("replanned_stages") or []),
        "stage_outcomes": dict(report.get("stage_outcomes") or {}),
        "gaps": list(report.get("gaps") or []),
    }


__all__ = ["READING_FORK_STAGES", "REPLAY_STAGES", "STAGE_ORDER", "load_stage_replay", "reuse_forecast"]
