"""Grade one finished generation attempt from the reports it already wrote.

A corpus run has to answer two questions after every paper: keep what was
produced, or pay for another attempt?  Both answers are already implied by the
reports the pipeline writes, so this module reads those reports and returns one
verdict.  It never calls a provider and never looks at a picture, which is what
makes it affordable to run after every paper.

Where the bar sits
------------------
The bar is breakage, not taste.  Six earlier runs whose figures were accepted by
hand were read back through these same reports, and every one of them carried
one or two thin panels, one or two links with a poor route, and a visual review
that asked for another pass.  Retrying on those would treble the bill and change
nothing, because that is the pipeline's ordinary quality today and improving it
is the Judge's job, not the retry loop's.  So those signals are *recorded* on
every paper and only gross breakage buys another attempt.

Four kinds of bad are kept apart, because they call for opposite responses:

``failure``
    The attempt did not finish: a provider error, a missing Icon asset, an
    unreadable deliverable.  Usually transient, so a retry is worth paying for.

``upstream``
    The contract or its evidence was unusable before any drawing happened
    (``method_contract_invalid`` and friends).  Another attempt runs into the
    same wall, so the paper is recorded instead of paid for again.

``mechanical``
    Gross drawing breakage: cards colliding, an Icon over its own label, a card
    off the canvas, text below the readable minimum, a quarter of the links
    misrouted, required links or modules missing from the drawing.

``degraded``
    A stage fell back to a written-in-advance answer, or a design repair was
    accepted with errors still open.  The figure can still be delivered, but it
    is no longer a faithful record of what the model decided; one retry is
    worth paying for and the paper is flagged either way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import read_json

from .content_coverage import measure_content_coverage

VERDICT_CONTRACT = "rfs.run_supervision_verdict"
VERDICT_VERSION = 1

RUN_REPORT = "paper_evolve_editable_report.json"
METHOD_REPORT = "method_extraction_report.json"
HIERARCHY_VALIDATION = "scientific_hierarchy_validation.json"
DESIGN_VALIDATION = "design_plan_validation.json"
GEOMETRY_REPORT = "geometry_quality_report.json"
DENSITY_REPORT = "density_quality_report.json"
TEXT_REPORT = "text_readability_report.json"
RENDERED_REPORT = "rendered_density_report.json"
COLOR_REPORT = "color_quality_report.json"
CONNECTOR_REPORT = "connector_visibility_report.json"
ALIGNMENT_REPORT = "graph_alignment_report.json"
GATE_REPORT = "final_quality_gate.json"
STYLE_PROFILE = "reference_style_profile.json"
COLOR_THEME_VALIDATION = "color_theme_validation.json"
REFERENCE_IMAGE = "reference_image_report.json"
LAYOUT_PROFILE = "reference_layout_profile.json"
PAPER_REVIEW = "paper_review.json"
VISUAL_GRAPH = "visual_graph.json"
REFERENCE_TEMPLATE = "reference_layout_template.json"

ATTEMPT_REPORTS = (
    RUN_REPORT,
    METHOD_REPORT,
    HIERARCHY_VALIDATION,
    DESIGN_VALIDATION,
    GEOMETRY_REPORT,
    DENSITY_REPORT,
    TEXT_REPORT,
    RENDERED_REPORT,
    COLOR_REPORT,
    CONNECTOR_REPORT,
    ALIGNMENT_REPORT,
    GATE_REPORT,
    STYLE_PROFILE,
    LAYOUT_PROFILE,
    COLOR_THEME_VALIDATION,
    REFERENCE_IMAGE,
    PAPER_REVIEW,
    VISUAL_GRAPH,
    REFERENCE_TEMPLATE,
)

#: Blocking reasons that come from the contract rather than from the drawing.
UPSTREAM_BLOCKERS = ("method_contract_invalid", "structure_contract_invalid")

#: Extra attempts each category is allowed to spend.
RETRY_BUDGET = {"failure": 2, "mechanical": 2, "degraded": 1, "upstream": 0, "clean": 0}

CATEGORY_ORDER = ("upstream", "failure", "mechanical", "degraded")

#: Two cards overlapping by less than this share of the canvas are touching
#: edges, not a collision; the accepted runs sat at 0.0001 to 0.0009.
GROSS_OVERLAP_AREA = 0.005

#: A quarter of the links misrouted is a pattern; one or two is the status quo.
GROSS_CONNECTOR_ISSUE_SHARE = 0.25

#: The same quarter applies to required links and modules that never reached the
#: drawing.  Alignment also fails when a required node or link arrived without
#: its evidence id: nothing is missing from the figure then, and the reason used
#: to claim required links were missing while printing that zero of them were,
#: which is how a bookkeeping gap bought a whole second figure.
GROSS_MISSING_REQUIRED_SHARE = 0.25

#: One line across one label survives the eye; two is a habit worth a retry.
GROSS_TEXT_CROSSINGS = 2

#: A child poking this far out of its own container is a container that was not
#: grown to its payload. It is worth saying, and worth a repair, but the nesting
#: itself is not a collision, so it never buys a retry.
CONTAINER_OVERHANG_AREA = 0.002


def load_attempt_reports(root: str | Path) -> dict[str, Any]:
    """Read every supervision report an attempt wrote, skipping the ones it did not."""
    root = Path(root)
    reports: dict[str, Any] = {}
    for name in ATTEMPT_REPORTS:
        path = root / name
        if not path.is_file():
            continue
        try:
            value = read_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(value, dict):
            reports[name] = value
    return reports


def _reason(kind: str, code: str, detail: str) -> dict[str, str]:
    return {"kind": kind, "code": code, "detail": detail}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def _count(value: Any) -> int:
    return int(_number(value))


def _upstream_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("method_contract_invalid", "structure_contract_invalid", "evidence_ids", "does not describe this paper"))



#: A fallback caused by one of these is a transport accident and may well pass
#: on the next call; anything else is the model's own answer failing a check.
TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "max retries",
    "server error",
    "rate limit",
    "429",
    "502",
    "503",
    "504",
    "invalid control character",
    "invalid json",
    "jsondecode",
    "unterminated",
)


def _is_transient(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in TRANSIENT_MARKERS)


def _fallback_code(prefix: str, reason: str) -> str:
    return f"{prefix}_{'transient' if _is_transient(reason) else 'content'}"


def _gross_overlaps(geometry: dict[str, Any]) -> list[float]:
    areas = [
        _number(item.get("overlap_area"))
        for item in (geometry.get("node_overlaps") or [])
        if isinstance(item, dict)
    ]
    return sorted(area for area in areas if area >= GROSS_OVERLAP_AREA)


def judge_attempt(reports: dict[str, Any], *, error: str | None = None, paper_id: str = "") -> dict[str, Any]:
    """Return the supervision verdict for one attempt.

    ``reports`` is what :func:`load_attempt_reports` returned.  ``error`` is the
    exception text when the attempt never reached the end of the workflow.
    """
    reasons: list[dict[str, str]] = []
    advisories: list[dict[str, str]] = []
    metrics: dict[str, Any] = {}
    run = reports.get(RUN_REPORT) or {}

    if error:
        text = str(error).strip()[:400] or "the attempt raised without a message"
        reasons.append(_reason("upstream" if _upstream_text(text) else "failure", "raised", text))
    if not run and not error:
        reasons.append(_reason("failure", "missing_run_report", f"no {RUN_REPORT} was written"))
    if run:
        metrics["generation_status"] = str(run.get("generation_status") or "")
        metrics["quality_status"] = str(run.get("quality_status") or "")
        metrics["delivery_status"] = str(run.get("delivery_status") or "")
        metrics["node_count"] = _count(run.get("node_count"))
        metrics["connector_count"] = _count(run.get("connector_count"))
        gaps = _count(run.get("paper_contract_gap_count"))
        if gaps:
            metrics["paper_contract_gap_count"] = gaps
            advisories.append(_reason("advisory", "paper_contract_gaps", f"{gaps} objects are drawn without a citation to the paper"))
        if str(run.get("generation_status") or "") != "completed":
            blocking = [str(item) for item in (run.get("blocking_generation_errors") or [])] or ["generation_incomplete"]
            for code in blocking:
                kind = "upstream" if code in UPSTREAM_BLOCKERS or _upstream_text(code) else "failure"
                reasons.append(_reason(kind, f"blocking:{code}", f"generation stopped: {code}"))
        if str(run.get("visual_review_mode") or "") == "vlm":
            review_status = str(run.get("visual_review_status") or "")
            if review_status not in ("", "pass"):
                advisories.append(_reason("advisory", "visual_review", f"the review asks for {review_status}"))
                metrics["visual_review_status"] = review_status

    method = reports.get(METHOD_REPORT) or {}
    if method:
        metrics["method_fallback_used"] = bool(method.get("fallback_used"))
        if method.get("fallback_used"):
            fallback_reason = str(method.get("fallback_reason") or "the method contract fell back")
            reasons.append(_reason("degraded", _fallback_code("method_contract_fallback", fallback_reason), fallback_reason))
        if method.get("ok") is False:
            reasons.append(_reason("upstream", "method_contract_invalid", "; ".join(str(item) for item in (method.get("errors") or []))[:300]))
        elif method.get("repairs"):
            # The contract was salvaged rather than thrown away, so the paper
            # keeps its figure; what was read as something else stays visible.
            advisories.append(_reason("advisory", "method_contract_salvaged", "; ".join(str(item) for item in (method.get("repairs") or []))[:300]))

    hierarchy = reports.get(HIERARCHY_VALIDATION) or {}
    if hierarchy.get("fallback_used"):
        metrics["hierarchy_fallback_used"] = True
        fallback_reason = str(hierarchy.get("fallback_reason") or "the hierarchy plan fell back")
        reasons.append(_reason("degraded", _fallback_code("hierarchy_fallback", fallback_reason), fallback_reason))

    design = reports.get(DESIGN_VALIDATION) or {}
    if design:
        metrics["design_repair_used"] = bool(design.get("plan_repair_used"))
        if design.get("continued_with_invalid_plan"):
            open_errors = [str(item) for item in (design.get("unrepaired_validation_errors") or [])]
            reasons.append(_reason("degraded", "design_plan_continued_invalid", "; ".join(open_errors)[:300] or "the design plan kept unresolved errors"))

    color_theme = reports.get(COLOR_THEME_VALIDATION) or {}
    if color_theme.get("fallback_used") and color_theme.get("mode") == "vlm_then_safe_fallback":
        fallback_reason = str(color_theme.get("failure_reason") or "the colour plan was replaced")
        metrics["color_theme_fallback_used"] = True
        reasons.append(_reason("degraded", _fallback_code("color_theme_fallback", fallback_reason), fallback_reason))

    for name in (STYLE_PROFILE, LAYOUT_PROFILE):
        profile = reports.get(name) or {}
        status = str(profile.get("status") or "")
        if status and status != "complete":
            metrics["reference_analysis_status"] = status
            reasons.append(_reason("degraded", f"reference_analysis:{status}", f"{name} did not complete"))
            break

    # The plan is drawn without the paper own framework figure whenever that
    # asset cannot be turned into pixels, so the run has to say it.
    reference_image = reports.get(REFERENCE_IMAGE) or {}
    if reference_image and str(reference_image.get("status") or "") != "ok":
        detail = str(reference_image.get("error") or reference_image.get("status") or "the reference asset could not be read")
        metrics["reference_image_status"] = str(reference_image.get("status") or "")
        reasons.append(_reason("degraded", _fallback_code("reference_image_unreadable", detail), detail))

    geometry = reports.get(GEOMETRY_REPORT) or {}
    if geometry:
        metrics["node_overlap_count"] = _count(geometry.get("node_overlap_count"))
        metrics["box_pass_count"] = _count(geometry.get("box_pass_count"))
        metrics["canvas_occupancy"] = _number(geometry.get("canvas_occupancy"))
        colliding = _gross_overlaps(geometry)
        if colliding:
            reasons.append(_reason("mechanical", "node_collision", f"{len(colliding)} card pairs overlap by up to {colliding[-1]:.4f} of the canvas"))
        elif _count(geometry.get("node_overlap_count")):
            advisories.append(_reason("advisory", "node_touching", f"{_count(geometry.get('node_overlap_count'))} card pairs touch without a visible collision"))
        for field, code, detail in (
            ("icon_text_overlap_count", "icon_text_overlap", "an Icon covers its own label"),
            ("connector_label_overlap_count", "connector_label_overlap", "a link label covers something"),
            ("node_boundary_violation_count", "node_boundary_violation", "a card or its label left the canvas"),
        ):
            count = _count(geometry.get(field))
            if count:
                reasons.append(_reason("mechanical", code, f"{count} x {detail}"))
        # ``hierarchy_containment_count`` counts the children that sit inside
        # their own container; the geometry gate records them *instead of* a
        # collision, and its own status treats them as a pass. Reading that
        # count as "a child left its parent" bought a second attempt for every
        # correctly nested figure that was otherwise clean, so it is recorded
        # as a metric and never gates.
        metrics["nested_children_count"] = _count(geometry.get("hierarchy_containment_count"))
        overhang = [
            row
            for row in (geometry.get("hierarchy_containments") or [])
            if float(row.get("outside_area") or 0) >= CONTAINER_OVERHANG_AREA
        ]
        if overhang:
            advisories.append(_reason("advisory", "child_overhangs_container", f"{len(overhang)} cards stick out of the container they belong to"))
        text_passes = _count(geometry.get("text_pass_count"))
        if text_passes >= GROSS_TEXT_CROSSINGS:
            reasons.append(_reason("mechanical", "connector_text_pass", f"{text_passes} links run through text"))
        elif text_passes:
            advisories.append(_reason("advisory", "connector_text_pass", f"{text_passes} link runs through text"))

    connectors = reports.get(CONNECTOR_REPORT) or {}
    if connectors:
        issues = _count(connectors.get("issue_count"))
        total = _count(connectors.get("connector_count"))
        share = (issues / total) if total else 0.0
        metrics["connector_issue_count"] = issues
        metrics["connector_issue_share"] = round(share, 4)
        if issues and share >= GROSS_CONNECTOR_ISSUE_SHARE:
            reasons.append(_reason("mechanical", "connector_misrouted", f"{issues} of {total} links cross an unrelated object"))
        elif issues:
            advisories.append(_reason("advisory", "connector_route", f"{issues} of {total} links want another route"))

    density = reports.get(DENSITY_REPORT) or {}
    if density:
        sparse = _count(density.get("sparse_panel_count"))
        metrics["sparse_panel_count"] = sparse
        metrics["empty_region_ratio"] = _number(density.get("empty_region_ratio"))
        metrics["density_status"] = str(density.get("status") or "")
        if sparse:
            advisories.append(_reason("advisory", "sparse_panel", f"{sparse} of {_count(density.get('panel_count'))} panels are nearly empty"))

    rendered = reports.get(RENDERED_REPORT) or {}
    if rendered:
        metrics["ink_coverage"] = _number(rendered.get("ink_coverage"))
        metrics["body_empty_cell_ratio"] = _number(rendered.get("body_empty_cell_ratio"))
        issues = [str(item) for item in (rendered.get("issues") or [])]
        if issues:
            advisories.append(_reason("advisory", "rendered_density", "; ".join(issues)[:240]))

    readability = reports.get(TEXT_REPORT) or {}
    if readability:
        failures = _count(readability.get("failure_count"))
        metrics["text_failure_count"] = failures
        metrics["minimum_font_size_pt"] = _number(readability.get("minimum_font_size_pt"))
        if failures:
            reasons.append(_reason("mechanical", "text_readability", f"{failures} texts are too small or covered"))

    alignment = reports.get(ALIGNMENT_REPORT) or {}
    if alignment:
        metrics["edge_coverage"] = _number(alignment.get("edge_coverage"))
        metrics["node_coverage"] = _number(alignment.get("node_coverage"))
        if alignment.get("ok") is False:
            missing_edges = [str(item) for item in (alignment.get("missing_required_edge_ids") or [])]
            missing_nodes = [str(item) for item in (alignment.get("missing_required_node_ids") or [])]
            edge_share = len(missing_edges) / max(1, _count(alignment.get("required_edge_count")))
            node_share = len(missing_nodes) / max(1, _count(alignment.get("required_node_count")))
            share = max(edge_share, node_share)
            metrics["missing_required_share"] = round(share, 4)
            detail = f"{len(missing_edges)} required links and {len(missing_nodes)} required modules never reached the drawing"
            if share >= GROSS_MISSING_REQUIRED_SHARE:
                reasons.append(_reason("mechanical", "graph_alignment", detail))
            elif missing_edges or missing_nodes:
                advisories.append(_reason("advisory", "graph_alignment_partial", detail))
            else:
                advisories.append(_reason("advisory", "graph_alignment_evidence", "; ".join(str(item) for item in (alignment.get("errors") or []))[:240]))

    coverage = measure_content_coverage(
        paper_review=reports.get(PAPER_REVIEW),
        visual_graph=reports.get(VISUAL_GRAPH),
        reference_layout_template=reports.get(REFERENCE_TEMPLATE),
    )
    if coverage["status"] != "not_measured":
        metrics["paper_budget_modules"] = coverage["paper_budget"]["required"]["modules"]
        metrics["paper_budget_relations"] = coverage["paper_budget"]["required"]["relations"]
        metrics["content_coverage_status"] = coverage["status"]
        for item in coverage["advisories"]:
            advisories.append(_reason("advisory", str(item["code"]), str(item["detail"])))

    gate = reports.get(GATE_REPORT) or {}
    if gate:
        metrics["final_gate_status"] = str(gate.get("status") or "")
        advisories.append(_reason("advisory", "final_gate", f"status={gate.get('status')} provenance={gate.get('provenance')}"))
    color = reports.get(COLOR_REPORT) or {}
    if color:
        metrics["color_status"] = str(color.get("status") or "")

    kinds = {reason["kind"] for reason in reasons}
    categories = [name for name in CATEGORY_ORDER if name in kinds]
    primary = categories[0] if categories else "clean"
    # An unusable contract vetoes every other reason: a better drawing of the
    # wrong content is not worth paying for.
    budget = 0 if "upstream" in categories else max((RETRY_BUDGET[name] for name in categories), default=0)
    status = "clean" if primary == "clean" else ("record_only" if primary == "upstream" else "retry")
    return {
        "contract": VERDICT_CONTRACT,
        "version": VERDICT_VERSION,
        "paper_id": paper_id,
        "status": status,
        "action": {"clean": "keep", "retry": "retry", "record_only": "stop"}[status],
        "category": primary,
        "categories": categories,
        "retry_budget": budget,
        "reasons": reasons,
        "advisories": advisories,
        "metrics": metrics,
        "content_coverage": coverage,
    }


__all__ = [
    "ATTEMPT_REPORTS",
    "CONTAINER_OVERHANG_AREA",
    "GROSS_OVERLAP_AREA",
    "RETRY_BUDGET",
    "VERDICT_CONTRACT",
    "judge_attempt",
    "load_attempt_reports",
]
