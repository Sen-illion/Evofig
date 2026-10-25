"""Native-editability introspection for the EditableFigureBench L2 layer.

The L2 layer asks what a human actually receives. Does the delivered
`editable_composition.pptx` carry native connectors, editable text, and grouped
objects, or is the figure a picture of a figure? This module answers that
question from the deck alone, so the numbers describe the artifact a reviewer
would open rather than the intent recorded in intermediate JSON. It never
writes to the deck.
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Iterator

from pptx import Presentation

#: DrawingML preset geometries that describe a straight or bent connector.
CONNECTOR_PRESETS = frozenset(
    {
        "line",
        "straightConnector1",
        "bentConnector2",
        "bentConnector3",
        "bentConnector4",
        "bentConnector5",
        "curvedConnector2",
        "curvedConnector3",
        "curvedConnector4",
        "curvedConnector5",
    }
)

#: Picture part extensions that stay raster on the slide.
RASTER_IMAGE_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp"})

#: A picture covering this share of the canvas is treated as a full-canvas raster.
FULL_CANVAS_PICTURE_RATIO = 0.9
#: A picture covering this much of the canvas, with few native objects, is a
#: flattened paste. This is separate from FULL_CANVAS_PICTURE_RATIO.
FLATTEN_AREA_RATIO = 0.85
FEW_NATIVE_OBJECTS = 2

_DRAWINGML = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _local_name(element: Any) -> str:
    """Return the local XML tag name, for example `cxnSp` for a connector."""
    tag = str(getattr(element, "tag", "") or "")
    return tag.rsplit("}", 1)[-1]


def _preset_geometry(shape: Any) -> str | None:
    geometry = shape._element.find(f".//{_DRAWINGML}prstGeom")
    return geometry.get("prst") if geometry is not None else None


def _picture_extension(shape: Any) -> str:
    try:
        return str(shape.image.ext or "").lower()
    except Exception:
        return ""


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _walk(shapes: Any, nested: bool = False) -> Iterator[tuple[Any, bool, bool]]:
    """Yield `(shape, nested, is_group)` for every shape, descending into groups."""
    for shape in shapes:
        if _local_name(shape._element) == "grpSp":
            yield shape, nested, True
            yield from _walk(shape.shapes, True)
        else:
            yield shape, nested, False


def native_object_count(report: dict[str, Any]) -> int:
    """Count native text frames, shapes, connectors, and groups. Pictures are excluded."""
    return sum(
        int(report.get(name) or 0)
        for name in (
            "text_frame_count",
            "autoshape_count",
            "native_connector_count",
            "line_shape_count",
            "group_count",
        )
    )


def editability_verdict(report: dict[str, Any], *, delivery: str = "pptx") -> dict[str, Any]:
    """Classify one delivery for the main-table editability column.

    A missing or unreadable deck is not editable. A TikZ or raster delivery
    with no OOXML deck is ``non_ooxml``. One raster covering at least 85% of
    the canvas with few native objects is not editable. The value is never
    null for a scored delivery.
    """
    if delivery == "non_ooxml":
        return {"status": "non_ooxml", "editable": False, "reason": "delivery is not OOXML"}
    if not report.get("ok") or not report.get("ooxml_valid"):
        return {
            "status": "not_editable",
            "editable": False,
            "reason": str(report.get("error") or "unreadable powerpoint"),
        }
    raster = float(report.get("raster_area_ratio") or 0.0)
    natives = native_object_count(report)
    if raster >= FLATTEN_AREA_RATIO and natives <= FEW_NATIVE_OBJECTS:
        return {
            "status": "not_editable",
            "editable": False,
            "reason": "flattened_picture",
            "raster_area_ratio": raster,
            "native_object_count": natives,
        }
    return {
        "status": "editable",
        "editable": True,
        "reason": "",
        "raster_area_ratio": raster,
        "native_object_count": natives,
    }


def measure_native_editability(
    pptx_path: str | Path,
    *,
    arrow_bearing_raster_count: int | None = None,
    expected_connector_count: int | None = None,
    include_text: bool = False,
) -> dict[str, Any]:
    """Measure L2 native editability for one delivered deck.

    `arrow_bearing_raster_count` is the number of arrows baked into pictures. It
    needs an external detector, so the native connector ratio stays `None` when
    the caller cannot supply it; the absolute connector count is always reported.
    `expected_connector_count` is the number of relations the gold contract says
    the figure needs, which turns the count into a realization ratio.
    """
    path = Path(pptx_path)
    if not path.is_file():
        return {"ok": False, "ooxml_valid": False, "pptx_path": str(path), "error": "missing pptx"}
    try:
        deck = Presentation(str(path))
        slide_count = len(deck.slides)
        canvas_area = int(deck.slide_width or 0) * int(deck.slide_height or 0)
        counters: collections.Counter[str] = collections.Counter()
        picture_area = 0
        full_canvas_pictures = 0
        editable_chars = 0
        text_frame_count = 0
        shape_count = 0
        group_count = 0
        grouped_shape_count = 0
        texts: list[str] = []
        for slide in deck.slides:
            for shape, nested, is_group in _walk(slide.shapes):
                shape_count += 1
                if is_group:
                    group_count += 1
                    continue
                if nested:
                    grouped_shape_count += 1
                local = _local_name(shape._element)
                if local == "cxnSp":
                    counters["connector"] += 1
                elif local == "pic":
                    counters["picture"] += 1
                    if _picture_extension(shape) in RASTER_IMAGE_EXTENSIONS:
                        counters["raster_picture"] += 1
                        area = int(shape.width or 0) * int(shape.height or 0)
                        picture_area += area
                        if canvas_area and area >= FULL_CANVAS_PICTURE_RATIO * canvas_area:
                            full_canvas_pictures += 1
                    else:
                        counters["vector_picture"] += 1
                elif local == "graphicFrame":
                    counters["graphic_frame"] += 1
                elif local == "sp":
                    if _preset_geometry(shape) in CONNECTOR_PRESETS:
                        counters["line_shape"] += 1
                    else:
                        counters["autoshape"] += 1
                if getattr(shape, "has_text_frame", False):
                    text = shape.text_frame.text or ""
                    text_frame_count += 1
                    editable_chars += len(text)
                    if text:
                        texts.append(text)
    except Exception as exc:
        return {"ok": False, "ooxml_valid": False, "pptx_path": str(path), "error": f"{type(exc).__name__}: {exc}"}

    native_connectors = counters["connector"] + counters["line_shape"]
    raster_ratio = (picture_area / canvas_area) if canvas_area else 0.0
    leaf_count = shape_count - group_count
    native_ratio = None
    if arrow_bearing_raster_count is not None:
        denominator = native_connectors + int(arrow_bearing_raster_count)
        native_ratio = (native_connectors / denominator) if denominator else None
    realization = (native_connectors / int(expected_connector_count)) if expected_connector_count else None
    report: dict[str, Any] = {
        "summary": "Native editability report for one delivered editable_composition.pptx.",
        "ok": True,
        "pptx_path": str(path),
        "ooxml_valid": slide_count > 0,
        "slide_count": slide_count,
        "shape_count": shape_count,
        "leaf_shape_count": leaf_count,
        "native_connector_count": counters["connector"],
        "line_shape_count": counters["line_shape"],
        "arrow_capable_shape_count": native_connectors,
        "native_connector_ratio": _round(native_ratio),
        "connector_realization_ratio": _round(realization),
        "arrow_bearing_raster_count": arrow_bearing_raster_count,
        "text_frame_count": text_frame_count,
        "editable_text_char_count": editable_chars,
        "autoshape_count": counters["autoshape"],
        "graphic_frame_count": counters["graphic_frame"],
        "picture_count": counters["picture"],
        "raster_picture_count": counters["raster_picture"],
        "vector_picture_count": counters["vector_picture"],
        "raster_area_ratio": _round(raster_ratio),
        "no_rasterization_ratio": _round(max(0.0, 1.0 - raster_ratio)),
        "full_canvas_picture_count": full_canvas_pictures,
        "group_count": group_count,
        "grouped_shape_count": grouped_shape_count,
        "grouping_ratio": _round((grouped_shape_count / leaf_count) if leaf_count else 0.0),
        "has_connectors": native_connectors > 0,
    }
    report["editability"] = editability_verdict(report)
    if include_text:
        report["editable_text"] = "\n".join(texts)
    return report


def extract_shape_labels(pptx_path: str | Path) -> list[str]:
    """Return the text label of every text-bearing shape, in document order.

    Systems that ship no semantic program still name their blocks in the deck
    itself, so reading the delivered text frames is the system-agnostic way to
    get comparable entity predictions for the L1 layer. Empty frames, groups and
    whitespace-only frames contribute nothing.
    """
    path = Path(pptx_path)
    if not path.is_file():
        return []
    try:
        deck = Presentation(str(path))
    except Exception:
        return []
    labels: list[str] = []
    for slide in deck.slides:
        for shape, _nested, is_group in _walk(slide.shapes):
            if is_group or not getattr(shape, "has_text_frame", False):
                continue
            try:
                text = str(shape.text_frame.text or "").strip()
            except Exception:
                continue
            if not text:
                continue
            first_line = text.splitlines()[0].strip()
            if first_line:
                labels.append(first_line)
    return labels


def extract_editable_text(pptx_path: str | Path) -> str:
    """Return every editable text frame in the deck, joined by newlines."""
    report = measure_native_editability(pptx_path, include_text=True)
    return str(report.get("editable_text") or "")


__all__ = [
    "CONNECTOR_PRESETS",
    "RASTER_IMAGE_EXTENSIONS",
    "extract_editable_text",
    "measure_native_editability",
]
