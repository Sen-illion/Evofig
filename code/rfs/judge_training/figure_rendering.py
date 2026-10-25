"""Supply the missing decodable figure for records bound to source fragments.

Some historical A/B records select a framework figure that only exists as TeX,
TikZ, PGF, pdf_tex, or EPS source.  The canonical strict gate excludes those
records as ``figure_rendering_missing`` and the D0 plan keeps them on the
``pending_figure_rendering`` list until a decodable image exists.  This module
produces that image from the record's own declared paper PDF page and records
exactly how it was produced: reader, DPI, page number, source PDF hash, and the
declared path.

Captions, declared source bindings, and original fragment paths are never
changed, page numbers are never invented, and the historical ledger is
rewritten only after its previous content has been backed up.  The render is
derived rather than copied, so it is recorded as its own asset role instead of
claiming byte identity with a declared source member.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Iterable

from rfs.contracts.source_snapshot import SOURCE_FIGURE_SUFFIXES
from rfs.utils import write_json

from .literature_freeze import (
    DOMAINS,
    STRICT_ASSET_TIERS,
    _artifact_root,
    _clean,
    _existing_path,
    _figure_paths,
    _fulltext_path,
    _jsonl_text,
    _publish_with_backup,
    infer_asset_tier,
    strict_failures,
)
from .source_snapshot_io import sha256_file


RENDER_METHOD = "pdfium_page_render"
FIGURE_PAGE_RENDER_ROLE = "figure_page_render"
FIGURE_ROLES = frozenset({"figure", FIGURE_PAGE_RENDER_ROLE})
DEFAULT_DPI = 150
MIN_DPI = 72
MAX_DPI = 600
_PAGE_NAME = re.compile(r"^source_page_[0-9]{3}\.png$")


def reader_name() -> str:
    """Return the exact PDF reader identity recorded with every render."""
    return "pypdfium2=" + importlib.metadata.version("pypdfium2")


def rendered_page_asset(row: dict[str, Any]) -> str:
    """Return the declared derived-render path of one record, or an empty string.

    Records are written on Windows, so a declared path can carry backslashes.
    The name rule is matched on a portable spelling of the same path.
    """
    render = (row.get("figure") or {}).get("rendered_page") or {}
    if _clean(render.get("method")) != RENDER_METHOD:
        return ""
    value = _clean(render.get("path"))
    name = PurePosixPath(value.replace("\\", "/")).name
    return value if value and _PAGE_NAME.fullmatch(name) else ""


def _cheap_candidate(batch: Path, row: dict[str, Any]) -> bool:
    """Skip records that cannot be render-pending before paying for a full gate run."""
    figure = row.get("figure") or {}
    paths = _figure_paths(row)
    if not figure.get("pdf_page") or not figure.get("caption") or not paths:
        return False
    if infer_asset_tier(row) not in STRICT_ASSET_TIERS:
        return False
    if any(Path(value).suffix.lower() not in SOURCE_FIGURE_SUFFIXES for value in paths):
        return False
    return all(_existing_path(batch, row, value) for value in paths)


def pending_figures(batch: Path, domain: str, row: dict[str, Any]) -> bool:
    """Return whether rendering is the only reason this record is still excluded."""
    return _cheap_candidate(batch, row) and strict_failures(batch, domain, row) == ["figure_rendering_missing"]


def paper_pdf_path(batch: Path, domain: str, row: dict[str, Any]) -> Path | None:
    """Resolve the original paper PDF the same way the source snapshot resolves it."""
    for field in ("paper_pdf_path", "official_pdf_path"):
        if path := _existing_path(batch, row, row.get(field)):
            return path
    root = _artifact_root(batch, row)
    arxiv_id = _clean(row.get("arxiv_id"))
    if arxiv_id:
        candidate = root / "assets" / domain / arxiv_id / "arxiv.pdf"
        if candidate.is_file():
            return candidate
    text = _fulltext_path(batch, domain, row)
    if text is not None:
        for name in ("official.pdf", "arxiv.pdf"):
            candidate = text.parent.parent / name
            if candidate.is_file():
                return candidate
    return None


def _render_target(batch: Path, row: dict[str, Any], page: int) -> tuple[Path, str] | None:
    """Place the render next to the declared fragments so both resolve alike."""
    for value in _figure_paths(row):
        resolved = _existing_path(batch, row, value)
        if resolved is None:
            continue
        target = resolved.parent / f"source_page_{page:03d}.png"
        declared = Path(value)
        relative = target if declared.is_absolute() else declared.parent / target.name
        return target, str(relative)
    return None


def render_page(
    pdf: Path,
    page_number: int,
    target: Path,
    dpi: int,
    source_sha256: str,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Render one declared PDF page with PDFium and return its exact provenance."""
    import pypdfium2

    with pypdfium2.PdfDocument(str(pdf)) as document:
        page_count = len(document)
        if page_number < 1 or page_number > page_count:
            raise ValueError(f"Declared figure page {page_number} is outside {page_count} pages: {pdf}")
        page = document[page_number - 1]
        try:
            bitmap = page.render(scale=dpi / 72)
            try:
                image = bitmap.to_pil()
                try:
                    if write:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        image.save(target, format="PNG")
                finally:
                    image.close()
            finally:
                bitmap.close()
        finally:
            page.close()
    return {
        "method": RENDER_METHOD,
        "reader": reader_name(),
        "page": page_number,
        "page_count": page_count,
        "dpi": dpi,
        "source_pdf": str(pdf),
        "source_pdf_sha256": source_sha256,
        "rendered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def render_row_figure(
    batch: Path,
    domain: str,
    row: dict[str, Any],
    *,
    dpi: int,
    write: bool = True,
) -> dict[str, Any] | None:
    """Return the row with a rendered page figure added, or None when nothing changes."""
    figure = row.get("figure") or {}
    page = int(figure["pdf_page"])
    target = _render_target(batch, row, page)
    pdf = paper_pdf_path(batch, domain, row)
    if target is None or pdf is None:
        return None
    path, relative = target
    recorded = figure.get("rendered_page") or {}
    digest = sha256_file(pdf)
    if (
        recorded.get("path") == relative
        and _clean(recorded.get("method")) == RENDER_METHOD
        and int(recorded.get("page") or 0) == page
        and int(recorded.get("dpi") or 0) == dpi
        and recorded.get("source_pdf_sha256") == digest
        and path.is_file()
    ):
        return None
    provenance = render_page(pdf, page, path, dpi, digest, write=write)
    provenance["path"] = relative
    updated = dict(row)
    updated["figure"] = {**figure, "image_paths": [*_figure_paths(row), relative], "rendered_page": provenance}
    return updated


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def render_pending_source_figures(
    batch_root: str | Path,
    *,
    domains: Iterable[str] | None = None,
    dpi: int = DEFAULT_DPI,
    limit: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Supply the missing decodable figure for every pending record in a batch."""
    batch = Path(batch_root)
    if dpi < MIN_DPI or dpi > MAX_DPI:
        raise ValueError(f"Rendering DPI must be between {MIN_DPI} and {MAX_DPI}: {dpi}")
    selected = tuple(domains or DOMAINS)
    unknown = [domain for domain in selected if domain not in DOMAINS]
    if unknown:
        raise ValueError("Unknown domains: " + ", ".join(unknown))
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = batch / "manifests" / "backups" / f"figure_rendering_{stamp}"
    report: dict[str, Any] = {
        "version": "figure-rendering-v001",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dpi": dpi,
        "reader": reader_name(),
        "dry_run": dry_run,
        "domains": {},
        "rendered": [],
        "skipped": [],
        "outputs_changed": False,
    }
    for domain in selected:
        path = batch / "verified" / domain / "papers.jsonl"
        rows = _load_rows(path)
        changed = 0
        attempted = 0
        for index, row in enumerate(rows):
            if limit and attempted >= limit:
                break
            if not pending_figures(batch, domain, row):
                continue
            attempted += 1
            paper_id = row.get("paper_id")
            try:
                updated = render_row_figure(batch, domain, row, dpi=dpi, write=not dry_run)
            except Exception as exc:  # keep the remaining records moving, report honestly
                report["skipped"].append({"domain": domain, "paper_id": paper_id,
                                          "reason": f"{type(exc).__name__}: {exc}"})
                continue
            if updated is None:
                report["skipped"].append({"domain": domain, "paper_id": paper_id,
                                          "reason": "already_rendered_or_source_pdf_unresolved"})
                continue
            rows[index] = updated
            changed += 1
            report["rendered"].append({"domain": domain, "paper_id": paper_id,
                                       "page": updated["figure"]["rendered_page"]["page"],
                                       "path": updated["figure"]["rendered_page"]["path"]})
        if changed and not dry_run:
            if _publish_with_backup(path, _jsonl_text(rows), backup):
                report["outputs_changed"] = True
        report["domains"][domain] = {"rendered": changed, "attempted": attempted}
    report["rendered_total"] = len(report["rendered"])
    report["skipped_total"] = len(report["skipped"])
    report["backup"] = str(backup.relative_to(batch)) if backup.exists() else None
    if not dry_run:
        write_json(batch / "manifests" / "figure_rendering_dashboard.json", report)
    return report


__all__ = [
    "DEFAULT_DPI",
    "FIGURE_PAGE_RENDER_ROLE",
    "RENDER_METHOD",
    "paper_pdf_path",
    "pending_figures",
    "reader_name",
    "render_page",
    "render_pending_source_figures",
    "render_row_figure",
    "rendered_page_asset",
]
