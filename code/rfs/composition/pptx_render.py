"""Render a compiled editable PPTX to the raster the visual reviewer actually sees.

The preview is the only window the visual reviewer has onto a figure: the reviewer
receives a PNG, never the deck.  Redrawing the figure program with Pillow is a
second implementation of the same layout, and two implementations of one layout
drift.  Rounded corners, panel header bands, shadows and text metrics all differed
slightly between the drawing and the deck, so the reviewer judged a figure that was
not the one being delivered - and nothing in the report said so.

This module renders the compiled deck with a real Office engine instead:

* LibreOffice Impress, driven headless to PDF, on Linux and macOS;
* PowerPoint through COM on Windows.

The page is then rasterised with PyMuPDF at the exact canvas size the figure program
declares, so the preview keeps the pixel geometry the rest of the pipeline expects.

Nothing here falls back silently.  When no engine is available, or an engine fails,
the caller receives a ``PptxRenderError`` and decides what to do; ``render_deck_preview``
below performs that fallback for the pipeline and records in its report which
drawing was used.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .preview import render_rebuild_preview

# How long one deck may take before its engine counts as hung.  A single slide
# converts in a couple of seconds; the budget is wide because a batch host runs
# many conversions at once.
DEFAULT_TIMEOUT_S = 180

# The engines, in the order they are tried.
LIBREOFFICE_BINARIES = ("soffice", "libreoffice")
BACKENDS = ("libreoffice", "powerpoint")

# Picking the drawing engine explicitly is useful when reproducing an older run:
# ``program`` always uses the Pillow renderer, ``auto`` prefers the compiled deck.
RENDER_MODE_ENV = "RFS_PREVIEW_RENDER"
RENDER_MODES = ("auto", "program")


class PptxRenderError(RuntimeError):
    """Raised when the compiled deck could not be rendered by any Office engine."""


def _libreoffice_binary() -> str | None:
    for name in LIBREOFFICE_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _powerpoint_available() -> bool:
    try:
        import win32com.client  # noqa: F401
    except Exception:
        return False
    return True


def available_backends() -> list[str]:
    """Return the Office engines that can render a deck on this host."""

    found: list[str] = []
    if _libreoffice_binary():
        found.append("libreoffice")
    if _powerpoint_available():
        found.append("powerpoint")
    return found


def render_mode() -> str:
    """Return the requested preview drawing mode, defaulting to the compiled deck."""

    requested = str(os.environ.get(RENDER_MODE_ENV) or "auto").strip().casefold()
    return requested if requested in RENDER_MODES else "auto"


def _rasterise_pdf(pdf_path: Path, out_path: Path, width_px: int, height_px: int) -> None:
    """Rasterise page 1 of ``pdf_path`` to ``out_path`` at the requested pixel size."""

    try:
        import pymupdf
    except Exception as error:  # pragma: no cover - a broken install, not a code path
        raise PptxRenderError(f"PyMuPDF is required to rasterise the deck: {error}") from error
    with pymupdf.open(pdf_path) as document:
        if document.page_count < 1:
            raise PptxRenderError(f"{pdf_path.name} has no pages")
        page = document[0]
        zoom = width_px / float(page.rect.width)
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    if image.size != (width_px, height_px):
        # The slide's aspect ratio rounds to a neighbouring pixel row; the canvas
        # size is authoritative because downstream measurements index by it.
        image = image.resize((width_px, height_px), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _convert_with_libreoffice(pptx_path: Path, pdf_path: Path, timeout: int) -> None:
    """Convert ``pptx_path`` to ``pdf_path`` with headless LibreOffice Impress."""

    binary = _libreoffice_binary()
    if binary is None:
        raise PptxRenderError("LibreOffice (soffice) is not installed")
    # Every conversion needs its own profile and its own output directory.
    # LibreOffice names the PDF after the input file, and every deck in the corpus
    # is called editable_composition.pptx, so conversions sharing an output
    # directory overwrite each other and abort.  Separate profiles also keep a
    # batch host's concurrent conversions from fighting over one user profile.
    profile_dir = tempfile.mkdtemp(prefix="rfs_lo_profile_")
    output_dir = tempfile.mkdtemp(prefix="rfs_lo_out_")
    try:
        command = [
            binary,
            "--headless",
            "--norestore",
            "--nolockcheck",
            f"-env:UserInstallation={Path(profile_dir).as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            output_dir,
            str(pptx_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise PptxRenderError(f"LibreOffice timed out after {timeout}s on {pptx_path.name}") from error
        produced = sorted(Path(output_dir).glob("*.pdf"))
        if not produced:
            detail = " ".join(str(completed.stderr or completed.stdout or "").split())[:400]
            raise PptxRenderError(f"LibreOffice produced no PDF for {pptx_path.name}: {detail}")
        shutil.copyfile(produced[0], pdf_path)
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


# A PowerPoint application object costs about twenty seconds to start, so one is
# kept per thread and reused for every deck that thread renders.  The cache is
# per thread because COM apartments are per thread.
_POWERPOINT_LOCAL = threading.local()


def _quit_powerpoint() -> None:
    application = getattr(_POWERPOINT_LOCAL, "application", None)
    _POWERPOINT_LOCAL.application = None
    if application is None:
        return
    try:
        application.Quit()
    except Exception:
        pass


atexit.register(_quit_powerpoint)


def _powerpoint_application() -> Any:
    application = getattr(_POWERPOINT_LOCAL, "application", None)
    if application is not None:
        return application
    import win32com.client

    application = win32com.client.Dispatch("PowerPoint.Application")
    _POWERPOINT_LOCAL.application = application
    return application


def _convert_with_powerpoint(pptx_path: Path, pdf_path: Path, timeout: int) -> None:
    """Export ``pptx_path`` to ``pdf_path`` through PowerPoint COM.

    COM cannot be interrupted safely, so ``timeout`` bounds nothing here; the
    engine either returns or raises.  A failed attempt discards the cached
    application rather than reusing an instance that may have died.
    """

    del timeout
    try:
        import win32com.client  # noqa: F401
    except Exception as error:
        raise PptxRenderError("pywin32 is not installed, so PowerPoint cannot be driven") from error
    last_error: Exception | None = None
    for _ in range(2):
        try:
            application = _powerpoint_application()
        except Exception as error:
            raise PptxRenderError(f"PowerPoint is not available through COM: {error}") from error
        presentation = None
        try:
            presentation = application.Presentations.Open(str(pptx_path), ReadOnly=True, Untitled=False, WithWindow=False)
            presentation.SaveAs(str(pdf_path), 32)  # 32 = ppSaveAsPDF
            return
        except Exception as error:
            last_error = error
            _quit_powerpoint()
        finally:
            if presentation is not None:
                try:
                    presentation.Close()
                except Exception:
                    pass
    raise PptxRenderError(f"PowerPoint could not export {pptx_path.name}: {last_error}")


def render_pptx_page(
    pptx_path: str | Path,
    out_path: str | Path,
    *,
    width_px: int = 1600,
    height_px: int = 900,
    timeout: int = DEFAULT_TIMEOUT_S,
    backend: str | None = None,
) -> dict[str, Any]:
    """Render page 1 of ``pptx_path`` to ``out_path`` and describe how it was done."""

    # Both engines resolve paths against their own working directory, and COM
    # rejects a relative path outright (0x80070003), so hand over absolute paths.
    pptx = Path(pptx_path).resolve()
    if not pptx.exists():
        raise PptxRenderError(f"compiled deck is missing: {pptx}")
    out = Path(out_path).resolve()
    chosen = backend if backend is not None else next(iter(available_backends()), None)
    if chosen is None:
        raise PptxRenderError(
            "no Office engine is available to render the deck: install LibreOffice (soffice) "
            "or, on Windows, pywin32 with PowerPoint"
        )
    if chosen not in BACKENDS:
        raise PptxRenderError(f"unknown render backend: {chosen}")
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="rfs_pptx_page_") as work:
        pdf_path = Path(work) / "page.pdf"
        if chosen == "libreoffice":
            _convert_with_libreoffice(pptx, pdf_path, timeout)
        else:
            _convert_with_powerpoint(pptx, pdf_path, timeout)
        _rasterise_pdf(pdf_path, out, width_px, height_px)
    return {
        "backend": chosen,
        "pptx": str(pptx),
        "preview": str(out),
        "width_px": width_px,
        "height_px": height_px,
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def render_deck_preview(
    program: dict[str, Any],
    pptx_path: str | Path,
    out_dir: str | Path,
    preview_path: str | Path,
    *,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> tuple[Path, dict[str, Any]]:
    """Render the preview from the compiled deck, or report why it could not be.

    Returns the preview path and a report naming the drawing that produced it.
    When the deck cannot be rendered - or when ``RFS_PREVIEW_RENDER=program`` asks
    for the older renderer - the program renderer draws the preview and the report
    carries ``backend: "program"`` with a ``fallback_reason``, so a run that shipped
    an approximation cannot be mistaken for one that shipped the real deck.
    """

    canvas = program.get("canvas") if isinstance(program.get("canvas"), dict) else {}
    width_px = int(canvas.get("width_px") or 1600)
    height_px = int(canvas.get("height_px") or 900)
    preview = Path(preview_path)

    def _draw_from_program(reason: str, *, forced: bool = False) -> tuple[Path, dict[str, Any]]:
        render_rebuild_preview(program, out_dir, preview)
        report: dict[str, Any] = {
            "backend": "program",
            "pptx": str(pptx_path),
            "preview": str(preview),
            "width_px": width_px,
            "height_px": height_px,
        }
        if forced:
            report["forced"] = f"{RENDER_MODE_ENV}={render_mode()}"
        else:
            report["fallback_reason"] = reason
        return preview, report

    mode = render_mode()
    if mode == "program":
        return _draw_from_program("", forced=True)
    try:
        return preview, render_pptx_page(pptx_path, preview, width_px=width_px, height_px=height_px, timeout=timeout)
    except PptxRenderError as error:
        return _draw_from_program(str(error))


__all__ = [
    "BACKENDS",
    "DEFAULT_TIMEOUT_S",
    "PptxRenderError",
    "RENDER_MODE_ENV",
    "available_backends",
    "render_deck_preview",
    "render_mode",
    "render_pptx_page",
]