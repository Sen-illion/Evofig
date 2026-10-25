"""Measure how much of the paper the delivered figure actually carries.

The alignment report answers a narrower question: did the figure draw everything
its own contract asked for?  A run that asked for little and drew all of it scores
a perfect ``node_coverage`` of 1.0, so that ratio cannot see a thin figure.  Both
the cheap profile and the strong profile print 1.0, while the cheap one read 3
modules out of its paper where the strong one read 8 and turned them into 14
nodes and 11 arrows instead of 9 and 5.

This module measures the figure against the paper side instead of against itself:

``paper_budget``
    what the paper review read out of the paper -- inputs, modules, outputs and
    the relations between them -- counted both as declared and as required.

``delivered``
    what the figure shows, together with how many of its arrows are bound to a
    relation the paper actually declared.  An arrow that carries no
    ``scientific_edge_ids`` is drawn but not claimed by the paper.  Delivered
    content is read against both denominators the budget carries -- what the
    paper declared and what its own review marked as having to appear -- so the
    narrow reading cannot hide a thin figure behind its own small denominator.

``reference``
    what the reference figure was measured to contain, when it was measured at
    all.  It is reported and never used as a bar: the reference pass leaves
    ``relation_patterns`` empty on most papers today.

Nothing here blocks delivery.  It records a number and names it, because a judge
trained on thin pairs would otherwise never learn to see a thin figure as a fault.
"""

from __future__ import annotations

from typing import Any

REPORT_CONTRACT = "rfs.content_coverage_report"
REPORT_VERSION = 1
REPORT_FILE = "content_coverage_report.json"

#: A paper whose own review read out fewer than this is thin at the source: the
#: figure cannot carry content the paper side never handed over.  The observed
#: medians are 3 modules and 4 relations on the cheap profile against 8 and 12 on
#: the strong one, so the bar sits between the two populations.
THIN_MODULES = 4
THIN_RELATIONS = 6

#: Delivered content below the paper's own declared budget. The bar is what the paper declared,
#: not the subset its review marked as having to appear: counting only the must-appear entries
#: leaves a denominator small enough that a figure can pass by drawing a third of the paper. A
#: run measured 7 declared modules against 12 drawn nodes and printed a ratio of 1.0 with no
#: advisory, every round, so neither of the two readings this report exists to produce could fire.
MIN_NODE_RATIO = 1.0
MIN_EDGE_RATIO = 1.0

#: Share of drawn arrows that must be bound to a declared relation before the
#: arrow field counts as answering the paper.  Below this the figure is decorated
#: with flow rather than showing the paper's relations.
MIN_BOUND_EDGE_SHARE = 0.5
BOUND_EDGE_FLOOR = 3


def _items(value: Any) -> list[dict[str, Any]]:
    """Return the dict entries of one review section."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _required(items: list[dict[str, Any]]) -> int:
    """Count the entries the review marked as having to appear in the figure."""
    total = 0
    for item in items:
        if bool(item.get("must_appear_in_figure")) or str(item.get("status") or "").strip() == "required":
            total += 1
    return total


def _content_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the drawn nodes that carry content rather than frame a panel."""
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    return [node for node in nodes if str(node.get("kind") or "").strip() != "panel"]


def _content_edges(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [edge for edge in graph.get("edges") or [] if isinstance(edge, dict)]


def _ratio(part: int, whole: int) -> float | None:
    if whole <= 0:
        return None
    return round(part / whole, 4)


def measure_content_coverage(
    *,
    paper_review: dict[str, Any] | None = None,
    visual_graph: dict[str, Any] | None = None,
    reference_layout_template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the paper-versus-figure content coverage of one attempt."""
    review = paper_review if isinstance(paper_review, dict) else {}
    graph = visual_graph if isinstance(visual_graph, dict) else {}
    template = reference_layout_template if isinstance(reference_layout_template, dict) else {}

    sections = {name: _items(review.get(name)) for name in ("inputs", "modules", "outputs", "relations")}
    required = {name: _required(items) for name, items in sections.items()}
    declared_nodes = sum(len(sections[name]) for name in ("inputs", "modules", "outputs"))
    required_nodes = sum(required[name] for name in ("inputs", "modules", "outputs"))

    nodes = _content_nodes(graph)
    edges = _content_edges(graph)
    bound = [edge for edge in edges if edge.get("scientific_edge_ids")]

    budget = {
        "declared": {"nodes": declared_nodes, "edges": len(sections["relations"]),
                     **{name: len(items) for name, items in sections.items()}},
        "required": {"nodes": required_nodes, "edges": required["relations"],
                     **required},
    }
    delivered = {"nodes": len(nodes), "edges": len(edges), "bound_edges": len(bound)}
    # Two denominators, both stated, because they answer different questions. The ``required``
    # pair reads "did the figure draw what its own review said it had to"; the ``declared`` pair
    # reads "did it draw what the paper has" -- which is the question this report exists to
    # answer, and the one the advisory below is decided on. A reader that wants the narrower
    # reading still has it, named, instead of having to work out which one fired.
    ratios = {
        "required_node_ratio": _ratio(len(nodes), required_nodes),
        "required_edge_ratio": _ratio(len(edges), required["relations"]),
        "declared_node_ratio": _ratio(len(nodes), declared_nodes),
        "declared_edge_ratio": _ratio(len(edges), len(sections["relations"])),
        "bound_edge_share": _ratio(len(bound), len(edges)),
    }

    advisories: list[dict[str, str]] = []
    thin_source = required["modules"] < THIN_MODULES and required["relations"] < THIN_RELATIONS
    if review and thin_source:
        advisories.append({
            "code": "paper_budget_thin",
            "detail": (
                f"the paper side read {required['modules']} required modules and "
                f"{required['relations']} required relations, so the figure has little to carry"
            ),
        })
    if ratios["declared_node_ratio"] is not None and ratios["declared_node_ratio"] < MIN_NODE_RATIO:
        advisories.append({
            "code": "figure_below_paper_budget",
            "detail": (
                f"the figure draws {len(nodes)} nodes against the {declared_nodes} the paper "
                f"declared ({required_nodes} of which its own review marked as having to appear)"
            ),
        })
    elif ratios["declared_edge_ratio"] is not None and ratios["declared_edge_ratio"] < MIN_EDGE_RATIO:
        advisories.append({
            "code": "figure_below_paper_budget",
            "detail": (
                f"the figure draws {len(edges)} arrows against the {len(sections['relations'])} "
                f"relations the paper declared ({required['relations']} required)"
            ),
        })
    if len(edges) >= BOUND_EDGE_FLOOR and (ratios["bound_edge_share"] or 0.0) < MIN_BOUND_EDGE_SHARE:
        advisories.append({
            "code": "arrows_unbound",
            "detail": (
                f"{len(edges) - len(bound)} of {len(edges)} arrows are bound to no relation the paper declared"
            ),
        })

    if not review and not graph:
        status = "not_measured"
    elif advisories:
        status = "warning"
    else:
        status = "pass"

    reference = {
        "object_patterns": len(template.get("object_patterns") or []),
        "relation_patterns": len(template.get("relation_patterns") or []),
        "zones": len(template.get("zones") or []),
    }
    return {
        "contract": REPORT_CONTRACT,
        "version": REPORT_VERSION,
        "summary": "How much of the paper the delivered figure carries.",
        "status": status,
        "paper_budget": budget,
        "delivered": delivered,
        "reference": reference,
        "ratios": ratios,
        "advisories": advisories,
    }


__all__ = [
    "BOUND_EDGE_FLOOR",
    "REPORT_FILE",
    "MIN_BOUND_EDGE_SHARE",
    "REPORT_CONTRACT",
    "REPORT_VERSION",
    "THIN_MODULES",
    "THIN_RELATIONS",
    "measure_content_coverage",
]