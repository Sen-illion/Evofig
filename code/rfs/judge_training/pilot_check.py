"""Independent figure-to-text correspondence review of one Pilot paper.

The visual stage reads the paper's framework figure into a visual layer, and the
binding stage matches that reading against the scientific layer, so two machine
readings are compared with each other. This stage adds the independent
measurement: a provider looks at the same rendered figure and answers, concept by
concept, whether the figure really shows what the paper contract requires. Every
concept ends in one of five verdicts:

confirmed
    the figure shows it and the stored visual layer also read it.
missed_by_reader
    the figure shows it but the stored visual layer never read it.
hallucinated_block
    the stored visual layer claims it while the figure does not show it.
not_drawn
    neither the figure nor the stored visual layer carries it.
unreadable
    the provider could not answer or could not read the label; it is reported,
    never guessed.

The review only covers the Pilot papers. Every other D0 paper stays unreviewed
until stage 3.5 asks for it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from rfs.judge_training.pilot_bindings import (
    HOWS,
    HOW_AS_BLOCK,
    HOW_INSIDE_OTHER_GRAPHICS,
    HOW_NOT_DRAWN,
    label_score,
)
from rfs.judge_training.pilot_visual import (
    DEFAULT_FIGURE_DPI,
    MAX_LONG_SIDE,
    load_figure_assets,
    render_figure_image,
    resolve_figure_asset,
)
from rfs.judge_training.pilot_selection import load_snapshot_papers
from rfs.judge_training.source_snapshot_io import sha256_file
from rfs.providers import call_vlm_json, resolve_vlm_model, vlm_credentials_available

LEDGER_SCHEMA_VERSION = "pilot-graph-v1"
CHECK_STEP = "check"
CHECK_PROMPT_VERSION = "pilot-figure-check-v2"
CHECK_VERSION = 2
CORRESPONDENCE_CONTRACT = "rfs.figure_text_review"
DEFAULT_CHECK_TIMEOUT_SECONDS = 300
MAX_CONCEPTS = 20
MAX_RELATIONS = 20
CHECK_MATCH_THRESHOLD = 0.6
PANEL_KIND = "panel"
LOW_CONFIDENCE = 0.5

VERDICT_CONFIRMED = "confirmed"
VERDICT_MISSED = "missed_by_reader"
VERDICT_HALLUCINATED = "hallucinated_block"
VERDICT_NOT_DRAWN = "not_drawn"
VERDICT_UNREADABLE = "unreadable"
VERDICTS = (VERDICT_CONFIRMED, VERDICT_MISSED, VERDICT_HALLUCINATED, VERDICT_NOT_DRAWN, VERDICT_UNREADABLE)
COUNTED_VERDICTS = (VERDICT_CONFIRMED, VERDICT_MISSED, VERDICT_HALLUCINATED, VERDICT_NOT_DRAWN, VERDICT_UNREADABLE)

PROMPT_TEMPLATE = """You audit whether a paper's own framework figure really shows the method the paper requires.

The image is the framework figure of the paper.

Required concepts of the method:
{concepts}

Required relations of the method:
{relations}

Answer for every concept and every relation. Read the image; never guess.
- visible: true when the figure carries the concept at all -- a box, a label, a panel name, or the notation the figure prints for it (a formula, a variable, a symbol). "Visible" does not mean "drawn as a block of its own".
- how: "as_block" when the concept has a drawing of its own, "inside_other_graphics" when the figure prints it inside another drawing (a formula in a box, a word under an arrow, a row name), "not_drawn" when the figure carries it nowhere.
- image_label: the text you actually read for it, exactly as printed, or an empty string when the figure carries no readable text.
- where: a short locator such as "panel 2, middle row", or an empty string.
- confidence: 0.0 to 1.0 for your own read of that concept.
- arrow_present: true only when a connector is drawn between the two concepts of that relation.
- direction_matches: true only when that connector runs from the source concept to the target concept.
- region_annotations: the figure's own annotations -- a legend, a panel title, a scale, a colour bar -- each with the text as printed and what it explains.
- unlisted_content: content the figure draws that the list above does not name and that is not an annotation of the figure itself, such as the other methods of a comparison row.
- uncertainties: everything you could not read, or had to guess about.
Answer with the short id printed for each concept and relation (c1, c2 ... and r1, r2 ...); never invent an id.

Return exactly one JSON object, no markdown and no commentary:
{{"image_summary": "one sentence", "concepts": [{{"id": "c1", "visible": true, "how": "as_block", "image_label": "", "where": "", "confidence": 0.0}}], "relations": [{{"id": "r1", "arrow_present": true, "direction_matches": true, "confidence": 0.0}}], "region_annotations": [{{"label": "", "explains": "", "where": ""}}], "unlisted_content": [{{"label": "", "where": ""}}], "uncertainties": []}}
"""


def _records(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _strings(value: Any) -> list[str]:
    return [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else []


def _how(value: Any) -> str:
    """Read the drawing form the review chose; an unknown answer carries no form."""
    text = str(value or "").strip().casefold().replace(" ", "_").replace("-", "_")
    return text if text in HOWS else ""


def _region_annotation_rows(value: Any) -> list[dict[str, str]]:
    """Read the annotations the figure prints about itself: legends, titles, scales."""
    rows: list[dict[str, str]] = []
    for item in _records(value):
        label = str(item.get("label") or item.get("text") or item.get("region") or "").strip()
        explains = str(item.get("explains") or item.get("explanation") or "").strip()
        if label or explains:
            rows.append({
                "label": label,
                "kind": str(item.get("kind") or "").strip().casefold(),
                "explains": explains,
                "where": str(item.get("where") or "").strip(),
            })
    return rows


def _content_rows(value: Any) -> list[dict[str, str]]:
    """Read labelled content the figure draws that the concept list never named."""
    return [
        {"label": str(item.get("label") or item.get("text") or "").strip(), "where": str(item.get("where") or "").strip()}
        for item in _records(value)
        if str(item.get("label") or item.get("text") or "").strip()
    ]


def stored_check_prompt_version(entry: dict[str, Any] | None) -> str:
    """Return the prompt version a stored review was written by, empty when it names none.

    Answers about the figure are only comparable while they answer the same question,
    so a review of an older prompt is read again rather than reused.
    """
    review = (entry or {}).get("figure_text_review")
    version = str((review or {}).get("prompt_version") or "") if isinstance(review, dict) else ""
    if version:
        return version
    report = (entry or {}).get("check_report")
    return str((report or {}).get("prompt_version") or "") if isinstance(report, dict) else ""


def required_concepts(graph: dict[str, Any] | None, *, limit: int = MAX_CONCEPTS) -> list[dict[str, Any]]:
    """Return the concepts the figure is asked about, required ones first."""
    nodes = [item for item in ((graph or {}).get("nodes") or []) if isinstance(item, dict)]
    ordered = sorted(
        nodes,
        key=lambda item: (
            not bool(item.get("required")),
            str(item.get("label") or item.get("id") or ""),
            str(item.get("id") or ""),
        ),
    )
    return ordered[: max(1, int(limit))]


def required_relations(graph: dict[str, Any] | None, *, limit: int = MAX_RELATIONS) -> list[dict[str, Any]]:
    """Return the relations the figure is asked about, required ones first."""
    edges = [item for item in ((graph or {}).get("edges") or []) if isinstance(item, dict)]
    ordered = sorted(
        edges,
        key=lambda item: (
            not bool(item.get("required")),
            str(item.get("id") or ""),
        ),
    )
    return ordered[: max(1, int(limit))]


def check_alias_map(concepts: list[dict[str, Any]], relations: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    """Return scientific id to short check id, for the prompt and for the answers.

    Scientific ids are content hashes, so the prompt asks with c1, c2 and r1, r2
    instead and the answers are mapped back here.
    """
    return (
        {str(item.get("id") or ""): f"c{index}" for index, item in enumerate(concepts, start=1)},
        {str(item.get("id") or ""): f"r{index}" for index, item in enumerate(relations, start=1)},
    )


def build_check_prompt(scientific_graph: dict[str, Any] | None) -> str:
    """Build the one prompt that audits the figure against the paper contract."""
    labels = {
        str(item.get("id") or ""): str(item.get("label") or item.get("id") or "")
        for item in ((scientific_graph or {}).get("nodes") or [])
        if isinstance(item, dict)
    }
    concept_items = required_concepts(scientific_graph)
    relation_items = required_relations(scientific_graph)
    concept_aliases, relation_aliases = check_alias_map(concept_items, relation_items)
    concepts = [
        "- " + concept_aliases[str(item.get("id") or "")] + ' | "' + str(item.get("label") or item.get("id") or "") + '"'
        + " | kind=" + str(item.get("kind") or "unknown")
        + " | required=" + ("yes" if item.get("required") else "no")
        for item in concept_items
    ]
    relations = [
        "- " + relation_aliases[str(item.get("id") or "")] + ' | "' + labels.get(str(item.get("source") or ""), str(item.get("source") or ""))
        + '" -> "' + labels.get(str(item.get("target") or ""), str(item.get("target") or "")) + '"'
        + " | " + str(item.get("relation_type") or "related_to")
        + " | required=" + ("yes" if item.get("required") else "no")
        for item in relation_items
    ]
    return PROMPT_TEMPLATE.format(
        concepts="\n".join(concepts) or "(none declared)",
        relations="\n".join(relations) or "(none declared)",
    )


def visual_blocks(visual_graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the blocks of a stored visual graph, panels excluded."""
    return [
        item
        for item in ((visual_graph or {}).get("nodes") or [])
        if isinstance(item, dict) and str(item.get("kind") or "") != PANEL_KIND
    ]


def visual_targets(visual_graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return every visual node a concept may be bound to: drawn blocks and panel labels.

    A panel is the row or group name of the figure, and on a comparison figure that name
    is where the author states which row is theirs. A concept the figure carries as a row
    label therefore has to be judged against it, or the verdict calls a figure silent
    about something it states in words.
    """
    return [
        item
        for item in ((visual_graph or {}).get("nodes") or [])
        if isinstance(item, dict)
    ]


def best_block(label: Any, blocks: Iterable[dict[str, Any]], *, threshold: float = CHECK_MATCH_THRESHOLD) -> tuple[str, float]:
    """Return the stored target that claims this label, best score first."""
    best_id = ""
    best_score = 0.0
    for block in sorted(blocks, key=lambda item: str(item.get("id") or "")):
        score = label_score(label, block.get("label"))
        if score > best_score:
            best_id = str(block.get("id") or "")
            best_score = score
    return (best_id, best_score) if best_score >= float(threshold) else ("", best_score)


def _finding_index(findings: list[dict[str, Any]], aliases: dict[str, str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Map the provider's answers onto the asked concepts, or report strays."""
    by_scientific_id = {alias: scientific_id for scientific_id, alias in aliases.items()}
    by_id: dict[str, dict[str, Any]] = {}
    strays: list[str] = []
    for item in findings:
        key = str(item.get("id") or "").strip().casefold()
        scientific_id = by_scientific_id.get(key)
        if scientific_id and scientific_id not in by_id:
            by_id[scientific_id] = item
        elif key:
            strays.append(key)
    return by_id, sorted(set(strays))


def normalize_check_payload(payload: dict[str, Any], *, paper_id: str, graph_id: str, row: dict[str, Any], scientific_graph: dict[str, Any] | None, visual_graph: dict[str, Any] | None, image: dict[str, Any] | None = None, prompt_chars: int = 0) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Turn one provider answer into the correspondence review. No provider call."""
    errors: list[str] = []
    warnings: list[str] = []
    source = payload if isinstance(payload, dict) else {}
    concepts = required_concepts(scientific_graph)
    relations = required_relations(scientific_graph)
    blocks = visual_targets(visual_graph)
    concept_aliases, relation_aliases = check_alias_map(concepts, relations)
    if not concepts:
        errors.append("the scientific graph declares no concept to audit")
    findings, concept_strays = _finding_index(_records(source.get("concepts")), concept_aliases)
    relation_findings, relation_strays = _finding_index(_records(source.get("relations")), relation_aliases)
    strays = sorted(set(concept_strays) | set(relation_strays))
    if concept_strays:
        warnings.append(f"{len(concept_strays)} answers name a concept outside the contract")
    if relation_strays:
        warnings.append(f"{len(relation_strays)} answers name a relation outside the contract")
    concept_rows: list[dict[str, Any]] = []
    for concept in concepts:
        concept_id = str(concept.get("id") or "")
        label = str(concept.get("label") or concept_id)
        finding = findings.get(concept_id)
        block_id, block_score = best_block(label, blocks)
        visible = bool(finding.get("visible")) if isinstance(finding, dict) and "visible" in finding else None
        if visible is None:
            verdict = VERDICT_UNREADABLE
        elif visible and block_id:
            verdict = VERDICT_CONFIRMED
        elif visible and not block_id:
            verdict = VERDICT_MISSED
        elif block_id:
            verdict = VERDICT_HALLUCINATED
        else:
            verdict = VERDICT_NOT_DRAWN
        confidence = finding.get("confidence") if isinstance(finding, dict) else None
        concept_rows.append({
            "concept_id": concept_id,
            "check_id": concept_aliases.get(concept_id, ""),
            "label": label,
            "kind": str(concept.get("kind") or "unknown"),
            "required": bool(concept.get("required")),
            "verdict": verdict,
            "visible": visible,
            "reader_how": _how((finding or {}).get("how")),
            "image_label": str((finding or {}).get("image_label") or ""),
            "where": str((finding or {}).get("where") or ""),
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
            "stored_block_id": block_id,
            "stored_block_score": round(float(block_score), 6),
            "evidence_ids": _strings(concept.get("evidence_ids")),
        })
        if verdict == VERDICT_UNREADABLE:
            warnings.append(f"concept {label} could not be judged from the figure")
    relation_rows: list[dict[str, Any]] = []
    for relation in relations:
        relation_id = str(relation.get("id") or "")
        finding = relation_findings.get(relation_id)
        if not isinstance(finding, dict) or "arrow_present" not in finding:
            verdict = VERDICT_UNREADABLE
        elif not finding.get("arrow_present"):
            verdict = VERDICT_NOT_DRAWN
        elif finding.get("direction_matches"):
            verdict = VERDICT_CONFIRMED
        else:
            verdict = "direction_conflict"
        confidence = finding.get("confidence") if isinstance(finding, dict) else None
        relation_rows.append({
            "scientific_edge_id": relation_id,
            "check_id": relation_aliases.get(relation_id, ""),
            "source": str(relation.get("source") or ""),
            "target": str(relation.get("target") or ""),
            "relation_type": str(relation.get("relation_type") or "related_to"),
            "required": bool(relation.get("required")),
            "verdict": verdict,
            "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
            "evidence_ids": _strings(relation.get("evidence_ids")),
        })
        if verdict == VERDICT_UNREADABLE:
            warnings.append(f"relation {relation_id} could not be judged from the figure")
    region_annotations = _region_annotation_rows(source.get("region_annotations"))
    # A review written before the split only carried extra_blocks, and those rows were
    # the content the figure draws outside the concept list rather than the figure's own
    # annotations, so they are read as unlisted content.
    unlisted_content = (
        _content_rows(source.get("unlisted_content"))
        if "unlisted_content" in source
        else _content_rows(source.get("extra_blocks"))
    )
    extra_blocks = list(unlisted_content)
    summary = correspondence_summary(concept_rows, relation_rows, extra_blocks, region_annotations=region_annotations, unlisted_content=unlisted_content)
    review = {
        "contract": CORRESPONDENCE_CONTRACT,
        "version": CHECK_VERSION,
        "prompt_version": CHECK_PROMPT_VERSION,
        "paper_id": paper_id,
        "graph_id": graph_id,
        "split": str(row.get("split") or ""),
        "domain": str(row.get("domain") or ""),
        "image": {
            "source_sha256": (image or {}).get("source_sha256"),
            "png_sha256": (image or {}).get("png_sha256"),
            "width": (image or {}).get("width"),
            "height": (image or {}).get("height"),
        },
        "image_summary": str(source.get("image_summary") or ""),
        "concepts": concept_rows,
        "relations": relation_rows,
        "region_annotations": region_annotations,
        "unlisted_content": unlisted_content,
        "extra_blocks": extra_blocks,
        "uncertainties": _strings(source.get("uncertainties")),
        "findings_outside_contract": strays,
        "summary": summary,
    }
    return review, {
        "paper_id": paper_id,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "prompt_chars": int(prompt_chars),
        "prompt_version": CHECK_PROMPT_VERSION,
        "image": dict(image or {}),
        "summary": summary,
        "findings_outside_contract": strays,
    }


def correspondence_summary(
    concept_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    extra_blocks: list[dict[str, Any]],
    *,
    region_annotations: list[dict[str, Any]] | None = None,
    unlisted_content: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Count the verdicts, overall and over the required concepts only."""
    counts = {verdict: 0 for verdict in COUNTED_VERDICTS}
    required = {verdict: 0 for verdict in COUNTED_VERDICTS}
    for item in concept_rows:
        verdict = str(item.get("verdict") or VERDICT_UNREADABLE)
        counts[verdict] = counts.get(verdict, 0) + 1
        if item.get("required"):
            required[verdict] = required.get(verdict, 0) + 1
    relation_counts: dict[str, int] = {}
    required_relation_counts: dict[str, int] = {}
    for item in relation_rows:
        verdict = str(item.get("verdict") or VERDICT_UNREADABLE)
        relation_counts[verdict] = relation_counts.get(verdict, 0) + 1
        if item.get("required"):
            required_relation_counts[verdict] = required_relation_counts.get(verdict, 0) + 1
    answered_required = sum(required.values()) - required.get(VERDICT_UNREADABLE, 0)
    tried = counts.get(VERDICT_CONFIRMED, 0) + counts.get(VERDICT_MISSED, 0) + counts.get(VERDICT_HALLUCINATED, 0)
    judged = tried + counts.get(VERDICT_NOT_DRAWN, 0)
    hows = {how: sum(1 for item in concept_rows if item.get("reader_how") == how) for how in HOWS}
    return {
        "concepts": len(concept_rows),
        "required_concepts": sum(1 for item in concept_rows if item.get("required")),
        "confirmed": counts.get(VERDICT_CONFIRMED, 0),
        "missed_by_reader": counts.get(VERDICT_MISSED, 0),
        "hallucinated_block": counts.get(VERDICT_HALLUCINATED, 0),
        "not_drawn": counts.get(VERDICT_NOT_DRAWN, 0),
        "unreadable": counts.get(VERDICT_UNREADABLE, 0),
        "required_confirmed": required.get(VERDICT_CONFIRMED, 0),
        "required_missed_by_reader": required.get(VERDICT_MISSED, 0),
        "required_hallucinated_block": required.get(VERDICT_HALLUCINATED, 0),
        "required_not_drawn": required.get(VERDICT_NOT_DRAWN, 0),
        "required_unreadable": required.get(VERDICT_UNREADABLE, 0),
        "required_visible_rate": round((required.get(VERDICT_CONFIRMED, 0) + required.get(VERDICT_MISSED, 0)) / answered_required, 4) if answered_required else 0.0,
        "reader_precision": round(counts.get(VERDICT_CONFIRMED, 0) / tried, 4) if tried else 0.0,
        "reader_recall": round(counts.get(VERDICT_CONFIRMED, 0) / (counts.get(VERDICT_CONFIRMED, 0) + counts.get(VERDICT_MISSED, 0)), 4) if (counts.get(VERDICT_CONFIRMED, 0) + counts.get(VERDICT_MISSED, 0)) else 0.0,
        "judged_concepts": judged,
        "low_confidence_concepts": sum(
            1 for item in concept_rows
            if isinstance(item.get("confidence"), (int, float)) and float(item["confidence"]) < LOW_CONFIDENCE
        ),
        "relations": len(relation_rows),
        "required_relations": sum(1 for item in relation_rows if item.get("required")),
        "relations_confirmed": relation_counts.get(VERDICT_CONFIRMED, 0),
        "direction_conflicts": relation_counts.get("direction_conflict", 0),
        "relations_undrawn": relation_counts.get(VERDICT_NOT_DRAWN, 0),
        "relations_unreadable": relation_counts.get(VERDICT_UNREADABLE, 0),
        "required_direction_conflicts": required_relation_counts.get("direction_conflict", 0),
        "extra_blocks": len(extra_blocks),
        "region_annotations": len(region_annotations if region_annotations is not None else []),
        "unlisted_content": len(unlisted_content if unlisted_content is not None else extra_blocks),
        "reader_how": hows,
        "reader_how_answered": sum(hows.values()),
    }


def _blocked_row(row: dict[str, Any], error: str, started: float) -> dict[str, Any]:
    """Ledger row of a paper that never reached the provider."""
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "paper_id": str(row.get("paper_id") or ""),
        "split": str(row.get("split") or ""),
        "domain": str(row.get("domain") or ""),
        "step": CHECK_STEP,
        "status": "failed",
        "attempts": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_count": 0,
        "image_bytes": 0,
        "prompt_chars": None,
        "estimated_tokens": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "token_source": "not_called",
        "failure_categories": [],
        "error": error,
        "artifact_sha256": {},
        "warnings": [error],
    }


def check_paper(snapshot: str | Path, paper: dict[str, Any], row: dict[str, Any], scientific_graph: dict[str, Any] | None, visual_graph: dict[str, Any] | None, *, figure_dir: str | Path, directory: str = "", assets: dict[str, list[dict[str, Any]]] | None = None, dpi: int = DEFAULT_FIGURE_DPI, max_long_side: int = MAX_LONG_SIDE, model: str | None = None, timeout_seconds: int = DEFAULT_CHECK_TIMEOUT_SECONDS, retries: int = 1, strict: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """Audit one paper's figure against its contract. Returns (entry, ledger row, report)."""
    started = time.monotonic()
    paper_id = str(row.get("paper_id") or "")
    if scientific_graph is None:
        return None, _blocked_row(row, "no stored scientific graph for this paper", started), {
            "paper_id": paper_id, "ok": False, "errors": ["no stored scientific graph for this paper"], "warnings": [],
        }
    if visual_graph is None:
        return None, _blocked_row(row, "no stored visual graph for this paper", started), {
            "paper_id": paper_id, "ok": False, "errors": ["no stored visual graph for this paper"], "warnings": [],
        }
    asset = resolve_figure_asset(snapshot, row, assets)
    if asset is None:
        error = "the frozen snapshot carries no readable figure asset for this paper"
        return None, _blocked_row(row, error, started), {"paper_id": paper_id, "ok": False, "errors": [error], "warnings": []}
    source = Path(snapshot) / asset["relative_path"]
    folder = str(directory or paper_id)
    target = Path(figure_dir) / folder / (Path(asset["relative_path"]).stem + ".png")
    sidecar = target.with_name(target.name + ".render.json")
    digest = sha256_file(source)
    reuse = False
    if target.is_file() and sidecar.is_file():
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        reuse = (
            recorded.get("source_sha256") == digest
            and int(recorded.get("dpi") or 0) == int(dpi)
            and int(recorded.get("max_long_side") or 0) == int(max_long_side)
        )
    if not reuse and (Path(figure_dir).parent / "COMPLETE").is_file():
        error = "refusing to render a figure into a sealed version"
        return None, _blocked_row(row, error, started), {"paper_id": paper_id, "ok": False, "errors": [error], "warnings": []}
    prompt = build_check_prompt(scientific_graph)
    provider: dict[str, Any] = {}
    payload: dict[str, Any] = {}
    image: dict[str, Any] = {}
    error: str | None = None
    try:
        image = render_figure_image(source, target, dpi=dpi, max_long_side=max_long_side)
    except Exception as exc:  # pragma: no cover - defensive, reported not hidden
        error = "figure render failed: " + type(exc).__name__ + ": " + str(exc)
    if error is None:
        try:
            raw = call_vlm_json(
                prompt,
                [target],
                model=resolve_vlm_model("RFS_PAPER_REVIEW_MODEL", explicit_model=model),
                timeout=max(10, int(timeout_seconds)),
                retries=max(0, int(retries)),
                call_metadata=provider,
            )
            payload = raw if isinstance(raw, dict) else {}
        except Exception as exc:  # pragma: no cover - defensive, reported not hidden
            error = type(exc).__name__ + ": " + str(exc)
    review: dict[str, Any] | None = None
    report: dict[str, Any] = {"paper_id": paper_id, "ok": False, "errors": [error] if error else [], "warnings": []}
    if error is None:
        review, report = normalize_check_payload(
            payload,
            paper_id=paper_id,
            graph_id=str((scientific_graph or {}).get("graph_id") or ""),
            row=row,
            scientific_graph=scientific_graph,
            visual_graph=visual_graph,
            image=image,
            prompt_chars=len(prompt),
        )
        report["paper_id"] = paper_id
        report["asset"] = {
            "relative_path": asset["relative_path"],
            "asset_id": asset.get("asset_id"),
            "resolved_by": asset.get("source"),
        }
        report["provider"] = dict(provider)
        report["model"] = resolve_vlm_model("RFS_PAPER_REVIEW_MODEL", explicit_model=model)
        if report["errors"]:
            error = "; ".join(str(item) for item in report["errors"][:3])
        elif strict and report["warnings"]:
            report["ok"] = False
            error = "; ".join(str(item) for item in report["warnings"][:3])
    if not report["ok"]:
        review = None
    entry = None
    if review is not None:
        entry = {
            "paper_id": paper_id,
            "split": str(row.get("split") or ""),
            "domain": str(row.get("domain") or ""),
            "figure_text_review": review,
            "check_report": report,
        }
    summary = dict(report.get("summary") or {})
    ledger_row = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "paper_id": paper_id,
        "split": str(row.get("split") or ""),
        "domain": str(row.get("domain") or ""),
        "step": CHECK_STEP,
        "status": "ok" if entry is not None else "failed",
        "attempts": int(provider.get("attempts") or 0),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_count": int(provider.get("image_count") or 0),
        "image_bytes": int(provider.get("image_bytes") or (image or {}).get("bytes") or 0),
        "prompt_chars": provider.get("prompt_chars") or len(prompt),
        "estimated_tokens": provider.get("estimated_tokens"),
        "prompt_tokens": provider.get("prompt_tokens"),
        "completion_tokens": provider.get("completion_tokens"),
        "total_tokens": provider.get("total_tokens"),
        "token_source": "unavailable" if not provider else str(provider.get("token_source") or "unavailable"),
        "failure_categories": [str(item) for item in (provider.get("failure_categories") or [])],
        "error": error,
        "concepts": int(summary.get("concepts") or 0),
        "confirmed": int(summary.get("confirmed") or 0),
        "missed_by_reader": int(summary.get("missed_by_reader") or 0),
        "hallucinated_block": int(summary.get("hallucinated_block") or 0),
        "not_drawn": int(summary.get("not_drawn") or 0),
        "unreadable": int(summary.get("unreadable") or 0),
        "required_visible_rate": summary.get("required_visible_rate"),
        "direction_conflicts": int(summary.get("direction_conflicts") or 0),
        "extra_blocks": int(summary.get("extra_blocks") or 0),
        "region_annotations": int(summary.get("region_annotations") or 0),
        "unlisted_content": int(summary.get("unlisted_content") or 0),
        "artifact_sha256": {"figure_text_review.json": canonical_review_sha256(review)} if review is not None else {},
        "warnings": [str(item) for item in (report.get("warnings") or [])],
    }
    return entry, ledger_row, report


def canonical_review_sha256(review: dict[str, Any] | None) -> str:
    """Hash the review document exactly as it is written."""
    if not isinstance(review, dict):
        return ""
    payload = json.dumps(review, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_all(snapshot: str | Path, rows: list[dict[str, Any]], *, graphs: dict[str, dict[str, Any]], visuals: dict[str, dict[str, Any]], figure_dir: str | Path, existing_checks: dict[str, dict[str, Any]] | None = None, directories: dict[str, str] | None = None, assets: dict[str, list[dict[str, Any]]] | None = None, dpi: int = DEFAULT_FIGURE_DPI, max_long_side: int = MAX_LONG_SIDE, model: str | None = None, timeout_seconds: int = DEFAULT_CHECK_TIMEOUT_SECONDS, retries: int = 1, workers: int = 4, strict: bool = False) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Audit the figure of every Pilot paper that has no stored review yet."""
    # A stored review written by another prompt answers another question, so it is not a
    # finished result for this version; it stays on disk and is read again instead.
    existing = {
        key: value
        for key, value in dict(existing_checks or {}).items()
        if stored_check_prompt_version(value) == CHECK_PROMPT_VERSION
    }
    folders = {str(key): str(value) for key, value in (directories or {}).items()}
    papers = {str(item.get("paper_id") or ""): item for item in load_snapshot_papers(snapshot)}
    index = assets if assets is not None else load_figure_assets(snapshot)
    pending = [row for row in rows if str(row.get("paper_id") or "") not in existing]
    if pending and not vlm_credentials_available():
        raise RuntimeError("the check stage needs API_BASE and API_KEY (or GEMINI_API_KEY) in the environment")
    results: dict[str, tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {}
            for row in pending:
                paper_id = str(row.get("paper_id") or "")
                paper = papers.get(paper_id)
                if paper is None:
                    raise KeyError("Pilot paper is not part of the snapshot: " + paper_id)
                futures[pool.submit(
                    check_paper,
                    snapshot,
                    paper,
                    row,
                    (graphs.get(paper_id) or {}).get("scientific_graph"),
                    (visuals.get(paper_id) or {}).get("visual_graph"),
                    figure_dir=figure_dir,
                    directory=folders.get(paper_id) or paper_id,
                    assets=index,
                    dpi=dpi,
                    max_long_side=max_long_side,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    strict=strict,
                )] = paper_id
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    entries: dict[str, dict[str, Any]] = dict(existing)
    reports: dict[str, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id in existing:
            continue
        entry, ledger_row, report = results[paper_id]
        ledger.append(ledger_row)
        reports[paper_id] = report
        if entry is not None:
            entries[paper_id] = entry
    return entries, ledger, reports


def check_coverage(checks: dict[str, dict[str, Any]], *, papers: int = 0) -> dict[str, Any]:
    """Summarize the correspondence reviews of a version."""
    totals: dict[str, Any] = {
        "concepts": 0, "required_concepts": 0, "confirmed": 0, "missed_by_reader": 0,
        "hallucinated_block": 0, "not_drawn": 0, "unreadable": 0,
        "required_confirmed": 0, "required_missed_by_reader": 0, "required_hallucinated_block": 0,
        "required_not_drawn": 0, "relations": 0, "relations_confirmed": 0,
        "direction_conflicts": 0, "relations_undrawn": 0, "extra_blocks": 0, "low_confidence_concepts": 0,
        "region_annotations": 0, "unlisted_content": 0, "reader_how_answered": 0,
    }
    rates: list[float] = []
    papers_with_missed: list[str] = []
    papers_with_hallucinated: list[str] = []
    papers_with_conflicts: list[str] = []
    papers_with_unreadable: list[str] = []
    for paper_id, entry in sorted(checks.items()):
        summary = ((entry.get("figure_text_review") or {}).get("summary") or {})
        for key in totals:
            totals[key] += int(summary.get(key) or 0)
        rate = summary.get("required_visible_rate")
        if isinstance(rate, (int, float)):
            rates.append(float(rate))
        if int(summary.get("missed_by_reader") or 0):
            papers_with_missed.append(paper_id)
        if int(summary.get("hallucinated_block") or 0):
            papers_with_hallucinated.append(paper_id)
        if int(summary.get("direction_conflicts") or 0):
            papers_with_conflicts.append(paper_id)
        if int(summary.get("unreadable") or 0):
            papers_with_unreadable.append(paper_id)
    return {
        "reviewed_papers": len(checks),
        "papers": int(papers) if papers else len(checks),
        "unreviewed_papers": max(0, int(papers) - len(checks)) if papers else 0,
        **totals,
        "mean_required_visible_rate": round(sum(rates) / len(rates), 4) if rates else 0.0,
        "papers_with_missed_concepts": papers_with_missed,
        "papers_with_hallucinated_blocks": papers_with_hallucinated,
        "papers_with_direction_conflicts": papers_with_conflicts,
        "papers_with_unreadable_concepts": papers_with_unreadable,
    }


__all__ = [
    "CHECK_MATCH_THRESHOLD",
    "CHECK_PROMPT_VERSION",
    "CHECK_STEP",
    "CHECK_VERSION",
    "CORRESPONDENCE_CONTRACT",
    "DEFAULT_CHECK_TIMEOUT_SECONDS",
    "LEDGER_SCHEMA_VERSION",
    "MAX_CONCEPTS",
    "MAX_RELATIONS",
    "PROMPT_TEMPLATE",
    "VERDICTS",
    "best_block",
    "build_check_prompt",
    "check_alias_map",
    "canonical_review_sha256",
    "check_all",
    "check_coverage",
    "check_paper",
    "correspondence_summary",
    "normalize_check_payload",
    "required_concepts",
    "required_relations",
    "stored_check_prompt_version",
    "visual_blocks",
    "visual_targets",
]
