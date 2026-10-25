"""Read the real framework figure of a frozen Pilot paper into a visual graph.

Phase 3 builds two graph layers per paper. The scientific layer comes from the
paper review; this module builds the visual layer from the paper's own framework
image, which is the only place where panels, reading direction, branches, merges,
feedback loops and cross-panel connections can actually be observed.

The frozen snapshot stays read-only: the figure asset is rendered once into the
graph version directory and that render is recorded with its reader, DPI, size and
source hash. Everything after the provider answer is deterministic, so one answer
always produces the same graph bytes on any machine.

The Pilot's own stage-2 labels (panel count, node count, structure) are never sent
to the provider. They are only used afterwards to cross-check what was read from
the image; sending them would turn that check into a self-fulfilling answer.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import threading
from typing import Any, Iterable

from PIL import Image

from rfs.analysis.graph.motifs import classify_node_motif
from rfs.contracts import normalize_graph, stable_graph_id, validate_graph
from rfs.utils import write_json

from .source_snapshot_io import sha256_file

VISUAL_PROMPT_VERSION = "pilot-visual-topology-v2"
DEFAULT_FIGURE_DPI = 150
MIN_FIGURE_DPI = 72
MAX_FIGURE_DPI = 400
MAX_LONG_SIDE = 1600
BOX_TOLERANCE = 0.02
NODE_OVERLAP_LIMIT = 0.35
PANEL_OVERLAP_LIMIT = 0.10
READING_BAND = 0.12
MAX_CAPTIONS = 6
PANEL_KIND = "panel"
MODULE_KIND = "module"
NOTE_REGION_KIND = "notation_region"
DEFAULT_BLOCK_ROLE = "module"
MAX_ANNOTATIONS = 20
MAX_REGION_ENTRIES = 40
FEEDBACK_RELATIONS = frozenset({"feedback", "feedback_loop", "iteration", "iterate", "return_flow", "loop"})
KNOWN_RELATIONS = frozenset({"forward", "branch", "merge", "feedback", "cross_panel", "container"})
STRUCTURE_KINDS = ("multi_panel", "feedback_loop", "converging", "branching", "linear")
DEFAULT_READING_DIRECTION = "left_to_right"
READING_DIRECTIONS = ("left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top", "mixed")


def load_figure_assets(snapshot: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Index every figure asset of a snapshot by paper id, read-only."""
    path = Path(snapshot) / "asset_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError("Snapshot is missing asset_manifest.jsonl: " + str(path))
    by_paper: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if str(record.get("role") or "") != "figure":
                continue
            by_paper.setdefault(str(record.get("paper_id") or ""), []).append({
                "asset_id": str(record.get("asset_id") or ""),
                "relative_path": str(record.get("relative_path") or ""),
                "sha256": str(record.get("sha256") or ""),
                "suffix": str(record.get("suffix") or ""),
            })
    for rows in by_paper.values():
        rows.sort(key=lambda row: row["relative_path"])
    return by_paper


def resolve_figure_asset(snapshot: str | Path, row: dict[str, Any], assets: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any] | None:
    """Resolve the framework figure the Pilot selected for one paper.

    Order: the ledger's own override file, then its snapshot-relative path, then its
    asset id, then the paper's first figure asset. The override exists for papers whose
    figure never became an asset of its own (only a full-page render and the source
    companion were frozen) and lives outside the snapshot on purpose; every other
    candidate must exist inside the snapshot. A missing override falls through, so an
    unusable paper still blocks as a visible ledger row instead of reading nothing.
    """
    root = Path(snapshot)
    override = str(row.get("figure_override_path") or "").strip()
    if override and Path(override).is_file():
        return {
            "relative_path": override,
            "asset_id": str(row.get("figure_asset_id") or ""),
            "source": "ledger-override",
        }
    relative = str(row.get("figure_relative_path") or "").strip()
    if relative and (root / relative).is_file():
        return {
            "relative_path": relative,
            "asset_id": str(row.get("figure_asset_id") or ""),
            "source": "pilot-ledger",
        }
    asset_id = str(row.get("figure_asset_id") or "").strip()
    rows = (assets or load_figure_assets(root)).get(str(row.get("paper_id") or ""), [])
    if asset_id:
        for entry in rows:
            if entry["asset_id"] == asset_id and entry["relative_path"] and (root / entry["relative_path"]).is_file():
                return {**entry, "source": "asset-manifest-by-id"}
    for entry in rows:
        if entry["relative_path"] and (root / entry["relative_path"]).is_file():
            return {**entry, "source": "asset-manifest-first"}
    return None


PDFIUM_LOCK = threading.Lock()


def _render_pdf_first_page(path: Path, dpi: int) -> tuple[Image.Image, str]:
    """Render the first PDF page while holding the process-wide PDFium lock.

    PDFium holds mutable state per process, so two worker threads that open or
    render at the same time corrupt its heap and abort the interpreter
    ("corrupted size vs. prev_size", SIGABRT, no Python traceback). Renders are
    therefore serialized here while the provider calls around them stay
    concurrent.
    """
    import pypdfium2

    with PDFIUM_LOCK:
        document = pypdfium2.PdfDocument(str(path))
        try:
            page = document[0]
            try:
                bitmap = page.render(scale=dpi / 72)
                try:
                    return bitmap.to_pil().convert("RGB"), "pypdfium2=" + importlib.metadata.version("pypdfium2")
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            document.close()


def _open_figure(path: Path, dpi: int) -> tuple[Image.Image, str]:
    """Open one figure asset as RGB, rendering the first PDF page when needed."""
    if path.suffix.lower() == ".pdf":
        return _render_pdf_first_page(path, dpi)
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB"), "pillow=" + Image.__version__


def render_figure_image(source: str | Path, target: str | Path, *, dpi: int = DEFAULT_FIGURE_DPI, max_long_side: int = MAX_LONG_SIDE, reuse: bool = True) -> dict[str, Any]:
    """Render one figure asset to PNG and record exactly how it was produced."""
    source_path = Path(source)
    target_path = Path(target)
    if not source_path.is_file():
        raise FileNotFoundError("figure asset does not exist: " + str(source_path))
    if dpi < MIN_FIGURE_DPI or dpi > MAX_FIGURE_DPI:
        raise ValueError("figure DPI must be between " + str(MIN_FIGURE_DPI) + " and " + str(MAX_FIGURE_DPI))
    digest = sha256_file(source_path)
    sidecar = target_path.with_name(target_path.name + ".render.json")
    if reuse and target_path.is_file() and sidecar.is_file():
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        same_source = recorded.get("source_sha256") == digest
        same_settings = int(recorded.get("dpi") or 0) == int(dpi) and int(recorded.get("max_long_side") or 0) == int(max_long_side)
        if same_source and same_settings:
            return {**recorded, "reused": True}
    image, reader = _open_figure(source_path, dpi)
    try:
        source_width, source_height = image.size
        scale = min(1.0, float(max_long_side) / max(source_width, source_height, 1))
        if scale < 1.0:
            resized = image.resize(
                (max(1, int(round(source_width * scale))), max(1, int(round(source_height * scale)))),
                Image.LANCZOS,
            )
            image.close()
            image = resized
        width, height = image.size
        target_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(target_path, format="PNG")
    finally:
        image.close()
    provenance = {
        "reader": reader,
        "dpi": int(dpi),
        "max_long_side": int(max_long_side),
        "source_suffix": source_path.suffix.lower(),
        "source_sha256": digest,
        "source_width": int(source_width),
        "source_height": int(source_height),
        "width": int(width),
        "height": int(height),
        "bytes": int(target_path.stat().st_size),
        "scaled": bool(scale < 1.0),
        "png_sha256": sha256_file(target_path),
        "reused": False,
    }
    write_json(sidecar, provenance)
    return provenance


def build_visual_prompt(*, captions: Iterable[str] = ()) -> str:
    """Build the figure-reading prompt. Pilot labels are deliberately absent."""
    caption_lines = [str(value) for value in list(captions)[:MAX_CAPTIONS]]
    caption_block = "\n".join("- " + value for value in caption_lines) or "- (no caption in the frozen snapshot)"
    return f"""You are reading the framework figure of one scientific paper. Return JSON only.
Read the figure the way its author meant it, then write down what you understood. Describe only what is visible in the image. Never invent a module, a label or an arrow that you cannot see.
- A block is a named step, module, dataset or result of the method.
- An arrow is a relation; a legend, a panel title, a formula or a scale is the figure explaining itself, not a step of the method.
- When the same thing is drawn more than once, it is the same object drawn again (the authors showing that a step repeats) -- record every drawing.

Task:
1. Find every panel: a bordered or clearly separated region, including subfigures. Give each panel an id, a short label and its bounding box.
2. Find every drawn block inside those panels: boxes, rounded rectangles, stacked layers, datasets, losses. Give each block an id, the exact visible text as its label, its panel id, its bounding box, and its role:
   "method_object" (a step, module, dataset or result of this paper's method) | "external_method" (another method's content, e.g. a comparison row) | "stage_annotation" (panel title, row name, legend heading) | "symbol" (a formula, a variable, a Greek letter drawn as its own object) | "decorative" (a scale tick, a number, a single letter, a colour swatch).
3. Find every legend or notation key: record its kind, what it explains, and its entries. When the text says "3x3 conv" next to a small square, the square is not a block -- that meaning is what the entry says.
4. Find every drawn arrow or connector between blocks, including arrows that cross panels, arrows that split into several targets, arrows that merge several sources, and arrows that loop back to an earlier block.
5. Report the reading direction as one of: left_to_right, right_to_left, top_to_bottom, bottom_to_top, mixed.

Geometry rules:
- Coordinates are normalized to the whole image: x and y are the top-left corner, w and h are width and height, all inside 0..1.
- Every block box must lie inside its panel box.
- Use the block's own visible text as its label. When a block carries only an icon or an abbreviation you cannot read, return an empty label and lower text_confidence. Never put the block's id in the label.

Repetition and attachment rules:
- Every drawing of a repeated thing gets its own entry with repeat_group, repeat_index and repeat_count; never merge them and never drop the later ones.
- Say which object an annotation belongs to with belongs_to, and list what sits next to that object in annotations.

Return exactly this shape:
{{
  "summary": "one sentence about what the figure shows",
  "reading_direction": "left_to_right",
  "panels": [{{"id": "p1", "label": "training", "box": {{"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}}}}],
  "blocks": [{{
    "id": "b1", "label": "exact visible text", "panel_id": "p1",
    "box": {{"x": 0.05, "y": 0.1, "w": 0.2, "h": 0.12}},
    "text_confidence": 0.0, "role": "method_object",
    "repeat_group": "", "repeat_index": 0, "repeat_count": 0,
    "belongs_to": "", "annotations": [{{"text": "text printed inside or beside the block", "kind": "formula|note|symbol"}}]
  }}],
  "regions": [{{
    "id": "r1", "kind": "legend|notation_key|colour_bar|scale", "label": "what the region is called",
    "explains": "what it explains", "box": {{"x": 0.0, "y": 0.8, "w": 1.0, "h": 0.2}},
    "entries": [{{"glyph": "the mark as printed", "meaning": "what that mark means"}}]
  }}],
  "connectors": [{{
    "id": "e1", "source_id": "b1", "target_id": "b2",
    "relation_type": "forward|branch|merge|feedback|cross_panel|container",
    "label": "visible arrow label or empty"
  }}],
  "uncertainties": ["what you could not read"]
}}

Figure captions of this paper (context only; a caption may belong to another figure):
{caption_block}
"""


def _box(value: Any) -> dict[str, float] | None:
    """Read one normalized box; return None when the payload is unusable."""
    if not isinstance(value, dict):
        return None
    numbers: dict[str, float] = {}
    for key in ("x", "y", "w", "h"):
        raw = value.get(key)
        if isinstance(raw, bool) or raw is None:
            return None
        try:
            numbers[key] = float(raw)
        except (TypeError, ValueError):
            return None
    if numbers["w"] <= 0 or numbers["h"] <= 0:
        return None
    return numbers


def _clamp_box(box: dict[str, float]) -> tuple[dict[str, float], bool]:
    """Clamp one box into 0..1 without ever making it degenerate."""
    width = min(box["w"], 1.0)
    height = min(box["h"], 1.0)
    x = min(max(box["x"], 0.0), max(0.0, 1.0 - width))
    y = min(max(box["y"], 0.0), max(0.0, 1.0 - height))
    clamped = {
        "x": round(x, 6),
        "y": round(y, 6),
        "w": round(width, 6),
        "h": round(height, 6),
    }
    moved = any(abs(clamped[key] - box[key]) > 1e-6 for key in ("x", "y", "w", "h"))
    return clamped, moved


def _overlap_ratio(first: dict[str, float], second: dict[str, float]) -> float:
    """Return the overlap of two boxes relative to the smaller one."""
    left = max(first["x"], second["x"])
    top = max(first["y"], second["y"])
    right = min(first["x"] + first["w"], second["x"] + second["w"])
    bottom = min(first["y"] + first["h"], second["y"] + second["h"])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    smaller = min(first["w"] * first["h"], second["w"] * second["h"])
    return round(intersection / smaller, 6) if smaller > 0 else 0.0


def _contains(outer: dict[str, float], inner: dict[str, float], *, tolerance: float = BOX_TOLERANCE) -> bool:
    return (
        inner["x"] >= outer["x"] - tolerance
        and inner["y"] >= outer["y"] - tolerance
        and inner["x"] + inner["w"] <= outer["x"] + outer["w"] + tolerance
        and inner["y"] + inner["h"] <= outer["y"] + outer["h"] + tolerance
    )


def _reading_key(box: dict[str, float], panel_index: int) -> tuple[int, int, float]:
    """Deterministic reading key: panel, then row band, then horizontal position."""
    band = int((box["y"] + READING_BAND / 2) / READING_BAND)
    return panel_index, band, box["x"]


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _records(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _positive_int(value: Any) -> int:
    """Read a count the reading gave; anything unusable becomes no count at all."""
    if isinstance(value, bool) or value is None:
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _annotations(value: Any) -> list[dict[str, str]]:
    """Read the notes the reading attached to one object, keeping the wording only."""
    rows: list[dict[str, str]] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str):
            text, kind = item.strip(), ""
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("label") or "").strip()
            kind = str(item.get("kind") or "").strip().casefold()
        else:
            continue
        if text or kind:
            rows.append({"text": text, "kind": kind})
        if len(rows) >= MAX_ANNOTATIONS:
            break
    return rows


def _region_entries(value: Any) -> list[dict[str, str]]:
    """Read the glyph-to-meaning pairs of one legend, notation key or scale."""
    rows: list[dict[str, str]] = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, str):
            glyph, meaning = item.strip(), ""
        elif isinstance(item, dict):
            glyph = str(item.get("glyph") or item.get("mark") or "").strip()
            meaning = str(item.get("meaning") or item.get("text") or "").strip()
        else:
            continue
        if glyph or meaning:
            rows.append({"glyph": glyph, "meaning": meaning})
        if len(rows) >= MAX_REGION_ENTRIES:
            break
    return rows


def derive_structure(panel_count: int, *, converging: bool, branching: bool, feedback_present: bool) -> tuple[str, list[str]]:
    """Derive the figure-level structure signals from the read graph.

    Converging and branching come from the connectors themselves, not from the
    per-node motif names: the shared motif classifier labels a plain source or
    sink block as multi_output or multi_input, which would report a simple chain
    as a merge.
    """
    signals: list[str] = []
    if panel_count > 1:
        signals.append("multi_panel")
    if feedback_present:
        signals.append("feedback_loop")
    if converging:
        signals.append("converging")
    if branching:
        signals.append("branching")
    if not signals:
        signals.append("linear")
    primary = next((kind for kind in STRUCTURE_KINDS if kind in signals), "linear")
    return primary, signals


def _scientific_label_matches(nodes: list[dict[str, Any]], scientific_graph: dict[str, Any] | None) -> int:
    """Count blocks whose visible label also names a scientific node."""
    labels = {
        str(node.get("label") or "").strip().casefold()
        for node in ((scientific_graph or {}).get("nodes") or [])
        if isinstance(node, dict)
    }
    labels.discard("")
    return sum(1 for node in nodes if str(node["label"] or "").strip().casefold() in labels)


def _endpoint_node_id(endpoint: str, *maps: dict[str, str]) -> str:
    """Resolve a connector endpoint to its node, whichever kind of object it reaches."""
    for mapping in maps:
        found = mapping.get(endpoint)
        if found:
            return found
    return ""


def stored_visual_prompt_version(entry: dict[str, Any] | None) -> str:
    """Return the prompt version a stored read was written by, empty when it names none.

    A read made by an older prompt answers the older question, so it cannot stand in
    for a stage that now asks a different one: the version is compared before any
    stored read is reused.
    """
    graph = (entry or {}).get("visual_graph")
    if not isinstance(graph, dict):
        return ""
    return str((graph.get("provenance") or {}).get("prompt_version") or "")


def normalize_visual_payload(payload: dict[str, Any], *, paper_id: str, graph_id: str | None = None, pilot_row: dict[str, Any] | None = None, scientific_graph: dict[str, Any] | None = None, image: dict[str, Any] | None = None, prompt_chars: int = 0) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Turn one provider answer into a validated visual graph plus its audit."""
    source = payload if isinstance(payload, dict) else {}
    errors: list[str] = []
    warnings: list[str] = []
    panels: list[dict[str, Any]] = []
    clamped: list[str] = []
    for index, raw in enumerate(_records(source.get("panels"))):
        panel_id = str(raw.get("id") or "").strip() or "panel_" + str(index)
        box = _box(raw.get("box"))
        if box is None:
            errors.append("panel " + panel_id + " has no usable normalized box")
            continue
        box, moved = _clamp_box(box)
        if moved:
            clamped.append("panel:" + panel_id)
        panels.append({"id": panel_id, "label": str(raw.get("label") or "").strip(), "box": box})
    blocks: list[dict[str, Any]] = []
    seen_blocks: set[str] = set()
    for index, raw in enumerate(_records(source.get("blocks"))):
        block_id = str(raw.get("id") or "").strip() or "block_" + str(index)
        if block_id in seen_blocks:
            errors.append("duplicate block id in the provider answer: " + block_id)
            continue
        seen_blocks.add(block_id)
        box = _box(raw.get("box"))
        if box is None:
            errors.append("block " + block_id + " has no usable normalized box")
            continue
        box, moved = _clamp_box(box)
        if moved:
            clamped.append("block:" + block_id)
        blocks.append({
            "id": block_id,
            "label": str(raw.get("label") or "").strip(),
            "panel_id": str(raw.get("panel_id") or "").strip(),
            "role": str(raw.get("visual_role") or "").strip(),
            "declared_role": str(raw.get("role") or "").strip().casefold(),
            "text_confidence": raw.get("text_confidence"),
            "box": box,
            "repeat_group": str(raw.get("repeat_group") or "").strip(),
            "repeat_index": _positive_int(raw.get("repeat_index")),
            "repeat_count": _positive_int(raw.get("repeat_count")),
            "belongs_to": str(raw.get("belongs_to") or "").strip(),
            "annotations": _annotations(raw.get("annotations")),
        })
    regions: list[dict[str, Any]] = []
    for index, raw in enumerate(_records(source.get("regions"))):
        region_id = str(raw.get("id") or "").strip() or "region_" + str(index)
        box = _box(raw.get("box"))
        if box is not None:
            box, moved = _clamp_box(box)
            if moved:
                clamped.append("region:" + region_id)
        regions.append({
            "id": region_id,
            "kind": str(raw.get("kind") or "").strip().casefold(),
            "label": str(raw.get("label") or "").strip(),
            "explains": str(raw.get("explains") or "").strip(),
            "box": box,
            "entries": _region_entries(raw.get("entries")),
        })
    panel_ids = {panel["id"] for panel in panels}
    unknown_panels = sorted({block["panel_id"] for block in blocks if block["panel_id"] and block["panel_id"] not in panel_ids})
    if unknown_panels:
        warnings.append("blocks reference unknown panels: " + ", ".join(unknown_panels[:5]))
    # A connector may point at a legend as well as at a block: the annotations of a figure
    # are drawn things the arrows reach, so their ids count as declared endpoints too.
    declared = panel_ids | {block["id"] for block in blocks} | {region["id"] for region in regions}
    connectors: list[dict[str, Any]] = []
    removed: list[dict[str, str]] = []
    for index, raw in enumerate(_records(source.get("connectors"))):
        connector_id = str(raw.get("id") or "").strip() or "connector_" + str(index)
        source_id = str(raw.get("source_id") or raw.get("source") or "").strip()
        target_id = str(raw.get("target_id") or raw.get("target") or "").strip()
        if not source_id or not target_id or source_id not in declared or target_id not in declared:
            removed.append({"id": connector_id, "source": source_id, "target": target_id, "reason": "undeclared_endpoint"})
            continue
        connectors.append({
            "id": connector_id,
            "source": source_id,
            "target": target_id,
            "relation_type": str(raw.get("relation_type") or raw.get("type") or "forward").strip().casefold(),
            "label": str(raw.get("label") or "").strip(),
        })
    if removed:
        warnings.append(str(len(removed)) + " connectors referenced an undeclared block or panel")
    # A paper whose first drawing is a plot rather than a diagram reads as no block at
    # all: the answer still names the panel and the notation the plot explains itself
    # with, which is a reading of that figure rather than a failure to read it. Only an
    # answer that carries no shape of any kind is unreadable.
    if not (blocks or panels or regions or connectors):
        errors.append("the provider answer carries no readable block")

    panel_index = {panel["id"]: index for index, panel in enumerate(panels)}
    panel_node_id = {panel["id"]: stable_graph_id("vnode", paper_id, "panel", panel["id"]) for panel in panels}
    block_node_id = {block["id"]: stable_graph_id("vnode", paper_id, "block", block["id"]) for block in blocks}
    region_node_id = {region["id"]: stable_graph_id("vnode", paper_id, "region", region["id"]) for region in regions}
    reading_order = [
        block["id"]
        for block in sorted(blocks, key=lambda item: _reading_key(item["box"], panel_index.get(item["panel_id"], 0)))
    ]
    reading_index = {block_id: index for index, block_id in enumerate(reading_order)}
    graph_nodes: list[dict[str, Any]] = []
    for panel in panels:
        graph_nodes.append({
            "id": panel_node_id[panel["id"]],
            "kind": PANEL_KIND,
            "label": panel["label"],
            "role": "panel",
            "properties": {
                "box": panel["box"],
                "source_id": panel["id"],
                "is_panel": True,
                "reading_index": None,
                "motif": "unknown",
            },
            "provenance": {"stage": "visual", "provider_id": panel["id"], "prompt_version": VISUAL_PROMPT_VERSION},
        })
    for block in blocks:
        properties: dict[str, Any] = {
            "box": block["box"],
            "source_id": block["id"],
            "text_confidence": block["text_confidence"],
            "is_panel": False,
            "reading_index": reading_index.get(block["id"]),
            "motif": "unknown",
        }
        # The role the reading chose for the object itself is kept apart from the older
        # graphic tag: only this one answers what the object is to the paper's method.
        if block["declared_role"]:
            properties["role"] = block["declared_role"]
        if block["repeat_group"]:
            properties["repeat_group"] = block["repeat_group"]
            properties["repeat_index"] = block["repeat_index"] or None
            properties["repeat_count"] = block["repeat_count"] or None
        if block["belongs_to"]:
            properties["belongs_to"] = block["belongs_to"]
        if block["annotations"]:
            properties["annotations"] = block["annotations"]
        graph_nodes.append({
            "id": block_node_id[block["id"]],
            "kind": MODULE_KIND,
            "label": block["label"],
            "role": block["role"] or DEFAULT_BLOCK_ROLE,
            "panel_ids": [panel_node_id[block["panel_id"]]] if block["panel_id"] in panel_node_id else [],
            "properties": properties,
            "provenance": {"stage": "visual", "provider_id": block["id"], "prompt_version": VISUAL_PROMPT_VERSION},
        })
    for region in regions:
        graph_nodes.append({
            "id": region_node_id[region["id"]],
            "kind": NOTE_REGION_KIND,
            "label": region["label"] or region["kind"],
            "role": "annotation",
            "properties": {
                "box": region["box"],
                "source_id": region["id"],
                "kind": region["kind"],
                "explains": region["explains"],
                "entries": region["entries"],
                "is_panel": False,
                "reading_index": None,
                "motif": "unknown",
            },
            "provenance": {"stage": "visual", "provider_id": region["id"], "prompt_version": VISUAL_PROMPT_VERSION},
        })
    graph_edges: list[dict[str, Any]] = []
    for index, connector in enumerate(connectors):
        graph_edges.append({
            "id": stable_graph_id("vedge", paper_id, connector["source"], connector["target"], connector["relation_type"], connector["id"] or index),
            "source": _endpoint_node_id(connector["source"], block_node_id, panel_node_id, region_node_id),
            "target": _endpoint_node_id(connector["target"], block_node_id, panel_node_id, region_node_id),
            "relation_type": connector["relation_type"] or "forward",
            "properties": {
                "label": connector["label"],
                "source_id": connector["id"],
                "provider_source_id": connector["source"],
                "provider_target_id": connector["target"],
                "provider_relation_type": connector["relation_type"],
            },
            "provenance": {"stage": "visual", "provider_id": connector["id"], "prompt_version": VISUAL_PROMPT_VERSION},
        })
    nodes_by_id = {node["id"]: node for node in graph_nodes}
    motif_input = [{"source": edge["source"], "target": edge["target"], "relation_type": edge["relation_type"]} for edge in graph_edges]
    motifs: dict[str, str] = {}
    for node in graph_nodes:
        if node["kind"] in (PANEL_KIND, NOTE_REGION_KIND):
            continue
        motifs[node["id"]] = classify_node_motif(node["id"], motif_input, node=node, nodes_by_id=nodes_by_id)
        node["properties"]["motif"] = motifs[node["id"]]

    panel_of = {block["id"]: block["panel_id"] for block in blocks}
    feedback_edges: list[str] = []
    backwards_edges: list[str] = []
    cross_panel_edges: list[str] = []
    unknown_relations: list[str] = []
    for edge in graph_edges:
        properties = edge["properties"]
        provider_source = str(properties.get("provider_source_id") or "")
        provider_target = str(properties.get("provider_target_id") or "")
        relation = str(edge["relation_type"] or "")
        if relation not in KNOWN_RELATIONS:
            unknown_relations.append(relation)
        if relation in FEEDBACK_RELATIONS:
            feedback_edges.append(edge["id"])
        source_position = reading_index.get(provider_source)
        target_position = reading_index.get(provider_target)
        if source_position is not None and target_position is not None and target_position <= source_position:
            backwards_edges.append(edge["id"])
        source_panel = panel_of.get(provider_source, "")
        target_panel = panel_of.get(provider_target, "")
        if source_panel and target_panel and source_panel != target_panel:
            cross_panel_edges.append(edge["id"])
    not_marked_feedback = [edge_id for edge_id in backwards_edges if edge_id not in feedback_edges]
    if not_marked_feedback:
        warnings.append(str(len(not_marked_feedback)) + " connectors point backwards in reading order without being marked as feedback")
    if unknown_relations:
        warnings.append("unknown connector relation types: " + ", ".join(sorted(set(unknown_relations))[:5]))

    panel_overlaps: list[dict[str, Any]] = []
    for first_index, first in enumerate(panels):
        for second in panels[first_index + 1:]:
            ratio = _overlap_ratio(first["box"], second["box"])
            if ratio > PANEL_OVERLAP_LIMIT:
                panel_overlaps.append({"first": first["id"], "second": second["id"], "overlap": ratio})
    if panel_overlaps:
        warnings.append(str(len(panel_overlaps)) + " panel pairs overlap more than the allowed ratio")
    block_overlaps: list[dict[str, Any]] = []
    for first_index, first in enumerate(blocks):
        for second in blocks[first_index + 1:]:
            ratio = _overlap_ratio(first["box"], second["box"])
            if ratio > NODE_OVERLAP_LIMIT:
                block_overlaps.append({"first": first["id"], "second": second["id"], "overlap": ratio})
    if block_overlaps:
        warnings.append(str(len(block_overlaps)) + " block pairs overlap more than the allowed ratio")
    outside_panel: list[str] = []
    for block in blocks:
        panel = next((item for item in panels if item["id"] == block["panel_id"]), None)
        if panel and not _contains(panel["box"], block["box"]):
            outside_panel.append(block["id"])
    if outside_panel:
        warnings.append(str(len(outside_panel)) + " blocks stick out of their panel")
    if clamped:
        warnings.append(str(len(clamped)) + " boxes were clamped into the normalized range")

    reading_direction = str(source.get("reading_direction") or "").strip().casefold() or DEFAULT_READING_DIRECTION
    if reading_direction not in READING_DIRECTIONS:
        warnings.append("unknown reading direction: " + reading_direction)
        reading_direction = DEFAULT_READING_DIRECTION
    uncertainties = _strings(source.get("uncertainties"))
    source_counts: dict[str, int] = {}
    target_counts: dict[str, int] = {}
    for connector in connectors:
        source_counts[connector["source"]] = source_counts.get(connector["source"], 0) + 1
        target_counts[connector["target"]] = target_counts.get(connector["target"], 0) + 1
    branching_blocks = sorted(block_id for block_id, count in source_counts.items() if count > 1)
    merging_blocks = sorted(block_id for block_id, count in target_counts.items() if count > 1)
    primary_structure, signals = derive_structure(
        len(panels),
        converging=bool(merging_blocks),
        branching=bool(branching_blocks),
        feedback_present=bool(feedback_edges),
    )
    agreement: dict[str, Any] = {}
    if isinstance(pilot_row, dict) and pilot_row:
        pilot_panels = pilot_row.get("panel_count")
        pilot_blocks = pilot_row.get("node_count")
        pilot_structure = pilot_row.get("structure")
        pilot_arrows = pilot_row.get("has_arrows")
        agreement = {
            "panel_count": {"pilot": pilot_panels, "derived": len(panels), "agrees": pilot_panels in (None, len(panels))},
            "node_count": {
                "pilot": pilot_blocks,
                "derived": len(blocks),
                "delta": (len(blocks) - int(pilot_blocks)) if isinstance(pilot_blocks, int) else None,
                "agrees": pilot_blocks in (None, len(blocks)),
            },
            "structure": {
                "pilot": pilot_structure,
                "derived_primary": primary_structure,
                "derived_signals": signals,
                "agrees": (pilot_structure == primary_structure) if pilot_structure else None,
                "agrees_with_signals": (pilot_structure in signals) if pilot_structure else None,
            },
            "has_arrows": {
                "pilot": pilot_arrows,
                "derived": bool(graph_edges),
                "agrees": (bool(pilot_arrows) == bool(graph_edges)) if pilot_arrows is not None else None,
            },
        }
    graph = normalize_graph({
        "graph_id": graph_id or stable_graph_id("visual_graph", paper_id),
        "paper_id": paper_id,
        "kind": "visual",
        "summary": str(source.get("summary") or "Visual topology read from the paper framework figure.").strip(),
        "nodes": graph_nodes,
        "edges": graph_edges,
        "forbidden_labels": [],
        "uncertainties": uncertainties,
        "provenance": {
            "stage": "visual",
            "prompt_version": VISUAL_PROMPT_VERSION,
            "reading_direction": reading_direction,
            "reading_order": reading_order,
            "panel_count": len(panels),
            "block_count": len(blocks),
            "region_count": len(regions),
            "image": dict(image or {}),
            "scientific_graph_id": (scientific_graph or {}).get("graph_id"),
            "scientific_label_matches": _scientific_label_matches(blocks, scientific_graph),
            "pilot_labels_used_in_prompt": False,
        },
    }, kind="visual")
    graph_validation = validate_graph(graph, expected_kind="visual")
    errors.extend(str(item) for item in graph_validation["errors"])
    expresses_flow = bool(graph_edges)
    if not expresses_flow:
        warnings.append(
            "the read carries no connector, so this figure shows no module, step or data flow"
        )
    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "panel_count": len(panels),
        "block_count": len(blocks),
        "region_count": len(regions),
        "edge_count": len(graph_edges),
        "expresses_flow": expresses_flow,
        "reading_direction": reading_direction,
        "reading_order": reading_order,
        "primary_structure": primary_structure,
        "structure_signals": signals,
        "branching_blocks": branching_blocks,
        "merging_blocks": merging_blocks,
        "motif_histogram": {motif: sum(1 for value in motifs.values() if value == motif) for motif in sorted(set(motifs.values()))},
        "clamped_boxes": clamped,
        "overlapping_blocks": block_overlaps[:20],
        "overlapping_panels": panel_overlaps[:20],
        "blocks_outside_panel": outside_panel,
        "removed_connectors": removed,
        "unknown_relation_types": sorted(set(unknown_relations)),
        "feedback_edges": feedback_edges,
        "backwards_edges": backwards_edges,
        "connectors_not_marked_feedback": not_marked_feedback,
        "cross_panel_edges": cross_panel_edges,
        "uncertainties": uncertainties,
        "pilot_label_agreement": agreement,
        "image": dict(image or {}),
        "prompt_chars": int(prompt_chars),
        "graph_id": graph.get("graph_id"),
    }
    if errors:
        return None, report
    return graph, report


__all__ = [
    "DEFAULT_FIGURE_DPI",
    "DEFAULT_READING_DIRECTION",
    "MAX_LONG_SIDE",
    "NOTE_REGION_KIND",
    "VISUAL_PROMPT_VERSION",
    "build_visual_prompt",
    "derive_structure",
    "load_figure_assets",
    "normalize_visual_payload",
    "PDFIUM_LOCK",
    "render_figure_image",
    "resolve_figure_asset",
    "stored_visual_prompt_version",
]
