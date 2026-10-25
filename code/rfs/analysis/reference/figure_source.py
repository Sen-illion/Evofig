"""Normalise a reference figure asset into a raster image the analyzers can read.

A reference asset is either a raster image or the single-page PDF that the
frozen source snapshot keeps for a paper own framework figure. Pillow cannot
open a PDF, so an analysis that only called `Image.open` failed on every
snapshot figure. This module owns that one conversion, including the render
cache, so the reference analyzers stay free of file-format handling.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from ...utils import ensure_dir

DEFAULT_PDF_DPI = 150
DEFAULT_MAX_LONG_SIDE = 1600
RASTER_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"})

# PDFium holds mutable state per process: two threads that render at the same
# time can corrupt its heap and abort the interpreter without a traceback, so
# every render is serialised here while the provider calls around it stay free
# to run concurrently.
_PDFIUM_LOCK = threading.Lock()


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _cap_long_side(image: Image.Image, max_long_side: int) -> Image.Image:
    """Shrink an oversized page render so it stays cheap to analyse."""
    longest = max(image.size)
    if longest <= max_long_side or max_long_side <= 0:
        return image
    scale = max_long_side / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def _render_pdf_first_page(path: Path, dpi: int, max_long_side: int) -> Image.Image:
    import pypdfium2

    with _PDFIUM_LOCK:
        document = pypdfium2.PdfDocument(str(path))
        try:
            page = document[0]
            try:
                bitmap = page.render(scale=dpi / 72)
                try:
                    rendered = bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()
        finally:
            document.close()
    return _cap_long_side(rendered, max_long_side)


def _pdfium_version() -> str:
    try:
        return "pypdfium2=" + importlib.metadata.version("pypdfium2")
    except Exception:
        return "pypdfium2"


def reference_image_for_analysis(source: str | Path, *, work_dir: str | Path, dpi: int = DEFAULT_PDF_DPI, max_long_side: int = DEFAULT_MAX_LONG_SIDE) -> dict[str, Any]:
    """Return a readable raster for one reference asset, rendering a PDF once.

    The rendered page is cached in `work_dir` under the source digest, so a
    retried attempt reuses the same pixels instead of rasterising the PDF again.
    """
    path = Path(source).resolve()
    if not path.is_file():
        return {"status": "missing", "source": str(path), "error": "reference asset does not exist"}
    suffix = path.suffix.lower()
    if suffix in RASTER_SUFFIXES:
        return {"status": "ok", "source": str(path), "image": str(path), "rendered": False, "format": suffix.lstrip(".")}
    if suffix != ".pdf":
        return {"status": "unsupported", "source": str(path), "error": "unsupported reference asset type: " + (suffix or path.name)}
    target = ensure_dir(work_dir) / ("reference_" + _digest(path)[:16] + "_dpi" + str(dpi) + ".png")
    if target.is_file():
        with Image.open(target) as cached:
            width, height = cached.size
        return {"status": "ok", "source": str(path), "image": str(target), "rendered": True, "renderer": _pdfium_version(), "dpi": dpi, "width": width, "height": height, "cached": True}
    try:
        rendered = _render_pdf_first_page(path, dpi, max_long_side)
    except Exception as exc:
        return {"status": "failed", "source": str(path), "error": type(exc).__name__ + ": " + str(exc)}
    try:
        rendered.save(target, format="PNG")
        width, height = rendered.size
    finally:
        rendered.close()
    return {"status": "ok", "source": str(path), "image": str(target), "rendered": True, "renderer": _pdfium_version(), "dpi": dpi, "width": width, "height": height, "cached": False}


def reference_image_path(source: str | Path, *, work_dir: str | Path, dpi: int = DEFAULT_PDF_DPI, max_long_side: int = DEFAULT_MAX_LONG_SIDE) -> tuple[str | None, dict[str, Any]]:
    """Return the path of a reference asset a vision model can actually read.

    A caller that hands the raw asset to a VLM is handing it a PDF whenever the
    frozen snapshot kept the paper framework figure as a PDF, and the provider
    answers with "unable to process input image" for every attempt. The second
    value is the report from :func:`reference_image_for_analysis`, so a caller
    can record what it sent instead of guessing.
    """
    report = reference_image_for_analysis(source, work_dir=work_dir, dpi=dpi, max_long_side=max_long_side)
    image = report.get("image") if report.get("status") == "ok" else None
    return (str(image) if image else None), report