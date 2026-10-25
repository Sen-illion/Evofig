"""Resolve the exact font files the measurer and the preview must draw with.

PowerPoint renders a run in the family that run declares and falls back to the
deck's theme font when a run declares none.  Our own stack used to guess: on
Windows it silently found Arial for everything, on Linux it silently fell back
to DejaVu for everything, so one figure was measured and previewed with sets of
glyph widths that matched neither each other nor PowerPoint.  Resolving a
declared family to a metric-compatible file here keeps the layout measurement,
the preview and the compiled deck on the same metrics.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from PIL import ImageFont


class ResolvedFont(NamedTuple):
    """A font file picked for a declared family."""

    family: str
    path: str
    bold: bool
    metric_faithful: bool


# Families the compiler can ask for, mapped to the file names PowerPoint uses
# plus metric-compatible clones (Liberation Sans ~ Arial, Carlito ~ Calibri).
FAMILY_FILES: dict[str, dict[bool, tuple[str, ...]]] = {
    "arial": {
        False: ("arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf", "Arimo-Regular.ttf"),
        True: ("arialbd.ttf", "Arial-Bold.ttf", "LiberationSans-Bold.ttf", "Arimo-Bold.ttf"),
    },
    "calibri": {
        False: ("calibri.ttf", "Calibri.ttf", "Carlito-Regular.ttf"),
        True: ("calibrib.ttf", "Calibri-Bold.ttf", "Carlito-Bold.ttf"),
    },
}

# Last resort.  DejaVu is installed almost everywhere but its metrics match
# neither Arial nor Calibri, so anything resolved from here is reported.
FALLBACK_FILES: dict[bool, tuple[str, ...]] = {
    False: ("DejaVuSans.ttf",),
    True: ("DejaVuSans-Bold.ttf",),
}

DEFAULT_FAMILY = "arial"

# python-pptx builds the slide master, and PowerPoint paints a run that names
# no family with the theme's minor font.  Decks we wrote before the compiler
# named its headings rely on exactly that, so the theme font is the honest
# default for them.
THEME_FONT_FAMILY = "Calibri"

_SUBSTITUTIONS: dict[tuple[str, bool], ResolvedFont] = {}


def _font_directories() -> list[Path]:
    directories = []
    configured = os.environ.get("RFS_FONT_DIR")
    if configured:
        directories.extend(Path(part) for part in configured.split(os.pathsep) if part)
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR") or "<PATH>"
        directories.append(Path(windir) / "Fonts")
    directories.extend(
        Path(path).expanduser()
        for path in (
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            "~/.fonts",
            "~/.local/share/fonts",
        )
    )
    try:  # matplotlib ships DejaVu for hosts that have nothing else
        import matplotlib

        directories.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf")
    except Exception:  # pragma: no cover - optional dependency
        pass
    return [path for path in directories if path.is_dir()]


@lru_cache(maxsize=1)
def font_index() -> dict[str, str]:
    """Map every installed font's file name, in lower case, to its path."""

    index: dict[str, str] = {}
    for directory in _font_directories():
        for path in directory.rglob("*"):
            if path.suffix.lower() in {".ttf", ".otf", ".ttc"} and path.name.casefold() not in index:
                index[path.name.casefold()] = str(path)
    return index


def _lookup(names: tuple[str, ...]) -> str | None:
    index = font_index()
    for name in names:
        found = index.get(name.casefold())
        if found:
            return found
    return None


def resolve_font(family: str | None, *, bold: bool) -> ResolvedFont | None:
    """Return the file to draw ``family`` with, or ``None`` when nothing fits."""

    requested = str(family or DEFAULT_FAMILY).strip().casefold() or DEFAULT_FAMILY
    names = FAMILY_FILES.get(requested)
    if names is None:
        requested = DEFAULT_FAMILY
        names = FAMILY_FILES[requested]
    found = _lookup(names[bold])
    if found:
        return ResolvedFont(requested, found, bold, True)
    fallback = _lookup(FALLBACK_FILES[bold])
    if not fallback:
        return None
    resolved = ResolvedFont(requested, fallback, bold, False)
    if _SUBSTITUTIONS.get((requested, bold)) != resolved:
        _SUBSTITUTIONS[(requested, bold)] = resolved
        print(
            f"[rfs] font substitution: {requested}{' bold' if bold else ''} has no "
            f"installed file, drawing with {Path(fallback).name} instead",
            file=sys.stderr,
        )
    return resolved


def load_font(size_px: int, family: str | None = None, *, bold: bool = False) -> ImageFont.ImageFont:
    """Load a font that keeps the declared family's metrics."""

    resolved = resolve_font(family, bold=bold)
    if resolved is None:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(resolved.path, size=max(1, int(size_px)))
    except Exception:  # pragma: no cover - unreadable font file
        return ImageFont.load_default()


def heading_font_family(style: dict | None) -> str:
    """Family the compiler draws titles, subtitles and panel headings with."""

    block = style if isinstance(style, dict) else {}
    return str(block.get("heading_font_family") or "").strip() or THEME_FONT_FAMILY


def body_font_family(style: dict | None) -> str:
    """Family the compiler draws node labels, connector labels and captions with."""

    block = style if isinstance(style, dict) else {}
    return str(block.get("font_family") or "").strip() or "Arial"


def font_substitutions() -> list[dict[str, object]]:
    """Report every family that had to be drawn with different metrics."""

    return [
        {
            "family": font.family,
            "bold": font.bold,
            "drawn_with": Path(font.path).name,
            "metric_faithful": font.metric_faithful,
        }
        for font in sorted(_SUBSTITUTIONS.values(), key=lambda item: (item.family, item.bold))
    ]


__all__ = [
    "DEFAULT_FAMILY",
    "FAMILY_FILES",
    "ResolvedFont",
    "THEME_FONT_FAMILY",
    "body_font_family",
    "font_index",
    "font_substitutions",
    "heading_font_family",
    "load_font",
    "resolve_font",
]
