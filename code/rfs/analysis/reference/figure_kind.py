"""Which kind of figure a reference record is, read off its own analysis.

The reference library holds published figures of many kinds, and this project draws
exactly one of them: a *framework* figure, built from labelled modules joined by
arrows. A *data* figure -- a band structure, a field map, a micrograph, a materials
cross-section, a Sankey -- is a good figure too, but it is a different question.
Handing one to the comparison seat produces advice nobody can act on ("you draw boxes,
they draw curves"), so a selection that means to offer "a good figure like ours" has
to be able to tell the two apart.

The kind is read from the record's own analysis text, never from the image at
selection time: `analyze-reference-figures` has already written what it saw into
`layout`, `structure`, `density`, `hierarchy`, `text`, `connectors` and `why_good`.
Reading that text keeps the split reproducible and lets a selection say why a record
was left out.

The rule is deliberately biased towards *not* calling a figure a framework figure:

* strong data evidence -- a named plot, image or measured object -- means the record
  is not offered as a framework reference. One such word is not enough on its own: a
  framework figure that carries an inset describes that inset in the same vocabulary,
  so the record is left out only when the analysis names measured content twice, or
  names it once and names no workflow, pipeline, process diagram or connector pathway
  anywhere.
* a framework figure has to *say* it is one: the analysis must name a workflow, a
  pipeline, a process diagram or a connector pathway. Absence of data evidence is not
  enough, because a record whose analysis stays generic should not be guessed into the
  kind the caller was hoping for.

A record may also carry an explicit ``figure_kind`` written by a curator or by a
screening pass. When it is one of the known kinds it wins over the text, and the
result says so in ``source``.
"""

from __future__ import annotations

from typing import Any

FIGURE_KINDS = ("framework", "data_panel", "unknown")

#: A named plot, image or measured object. Any single one of these says the figure
#: shows measured content rather than a diagram, which is enough to leave it out.
DATA_STRONG = (
    "small-multiple",
    "small multiple",
    "plot frame",
    "plot frames",
    "plotting margin",
    "plotting margins",
    "axis label",
    "axis labels",
    "axis typography",
    "aligned axes",
    "shared scale",
    "shared quantitative scale",
    "colormap",
    "colorbar",
    "heatmap",
    "heat map",
    "field map",
    "contour",
    "band structure",
    "energy band",
    "dispersion",
    "spectrum",
    "spectra",
    "spectral",
    "dose-response",
    "dose response",
    "phase diagram",
    "time series",
    "box plot",
    "boxplot",
    "histogram",
    "scatter",
    "violin plot",
    "forest plot",
    "volcano plot",
    "sankey",
    "proportional ribbon",
    "ribbon width",
    "ribbon bundles",
    "micrograph",
    "microscopy",
    "histolog",
    "tomograph",
    "radiolog",
    "mri",
    "ultrasound",
    "atomic structure",
    "crystal structure",
    "unit cell",
    "lattice",
    "molecular structure",
    "gene expression",
    "phylogen",
    "electrode",
    "thin film",
    "wafer",
    "substrate",
    "wavelength",
    "band gap",
    "energy level",
    "quantitative result",
    "quantitative summary",
    "quantitative output",
    "quantitative evidence",
    "measurement evidence",
    "residual",
    "residuals",
    "image view",
    "image views",
    "slice comparison",
)

#: Mentions of measurement in passing. Recorded for the audit trail; a figure is not
#: called a data figure for one of these alone, because framework figures describe
#: their own annotations with the same words.
DATA_WEAK = (
    "axis",
    "axes",
    "curve",
    "curves",
    "plot",
    "plots",
    "chart",
    "charts",
    "measurement",
    "measurements",
    "quantitative annotation",
    "quantitative annotations",
    "scale",
    "scales",
    "mesh",
    "cross section",
    "cross-section",
    "graph",
    "graphs",
    "density map",
)

#: A figure made of labelled modules joined by connectors says so in these words, and
#: a data figure does not.
FRAMEWORK_STRONG = (
    "workflow",
    "pipeline",
    "flowchart",
    "flow chart",
    "block diagram",
    "process flow",
    "process diagram",
    "system overview",
    "process schematic",
    "workflow trunk",
    "connector pathway",
    "sequential workflow",
    "process module",
    "process stage",
    "forward pathway",
    "feedback pathway",
    "forward workflow",
    "process field",
    "input band",
    "framework overview",
    "architecture diagram",
    "module and connector",
    "system logic",
    "process tree",
    "method flow",
    "overview diagram",
)

#: The ordinary vocabulary of both kinds. Never decides a kind on its own.
FRAMEWORK_WEAK = (
    "module",
    "modules",
    "stage",
    "stages",
    "arrow",
    "arrows",
    "connector",
    "connectors",
    "input",
    "output",
    "outputs",
    "feedback",
    "routing",
    "route",
    "branch",
    "branches",
    "box",
    "boxes",
    "container",
    "containers",
    "legend",
    "flow",
    "process",
    "schematic",
    "architecture",
    "tree",
    "iteration",
    "protocol",
)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in (_strings(entry) for entry in value) for item in item]
    if isinstance(value, dict):
        return [item for entry in value.values() for item in _strings(entry)]
    return []


def record_text(record: dict[str, Any] | None) -> str:
    """Every string the analysis wrote about a record, folded to lower case."""

    return " ".join(_strings(record if isinstance(record, dict) else {})).lower()


def _hits(text: str, tokens: tuple[str, ...]) -> list[str]:
    return [token for token in tokens if token in text]


def classify_figure_kind(record: dict[str, Any] | None) -> dict[str, Any]:
    """Return the kind of figure a record describes, with the reasons for it.

    ``framework`` is a figure built from labelled modules and connectors, ``data_panel``
    one built from measured content, and ``unknown`` a record whose analysis does not
    say enough to tell. Unknown is a real answer: a record is not guessed into a kind
    because the caller wanted one, and a selection that needs to be sure can drop it.
    """

    source = record if isinstance(record, dict) else {}
    declared = str(source.get("figure_kind") or "").strip().casefold()
    if declared in FIGURE_KINDS:
        return {
            "kind": declared,
            "source": "record",
            "data_evidence": [],
            "framework_evidence": [],
            "data_context": [],
            "framework_context": [],
        }
    text = record_text(source)
    data_evidence = _hits(text, DATA_STRONG)
    framework_evidence = _hits(text, FRAMEWORK_STRONG)
    # One measured-content word is not enough to leave a record out. A figure of the kind
    # this project draws may carry a single inset -- a micrograph, a contour map, a heat
    # map -- and describe that inset with one of these words while the rest of the
    # analysis names its workflow, so a single hit used to drop figures that are exactly
    # the reference the comparison seat needs. Two of them, or one with no workflow word
    # anywhere in the analysis, still means the record is about measured content.
    if len(data_evidence) >= 2 or (data_evidence and not framework_evidence):
        kind = "data_panel"
    elif framework_evidence:
        kind = "framework"
    else:
        kind = "unknown"
    return {
        "kind": kind,
        "source": "analysis_text",
        "data_evidence": data_evidence,
        "framework_evidence": framework_evidence,
        "data_context": [] if data_evidence else _hits(text, DATA_WEAK),
        "framework_context": [] if framework_evidence else _hits(text, FRAMEWORK_WEAK),
    }


def is_framework_figure(record: dict[str, Any] | None) -> bool:
    """Whether a record describes the kind of figure this project draws."""

    return classify_figure_kind(record)["kind"] == "framework"


def partition_by_kind(records: list[dict[str, Any]] | None) -> dict[str, list[dict[str, Any]]]:
    """Split records into the three kinds, keeping the order they arrived in."""

    buckets: dict[str, list[dict[str, Any]]] = {kind: [] for kind in FIGURE_KINDS}
    for record in records or []:
        if not isinstance(record, dict):
            continue
        buckets[classify_figure_kind(record)["kind"]].append(record)
    return buckets
