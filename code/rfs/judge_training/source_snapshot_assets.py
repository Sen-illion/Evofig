"""Inventory and decode the actual source assets referenced by strict papers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rfs.contracts.source_snapshot import SOURCE_FIGURE_SUFFIXES

from .figure_rendering import FIGURE_PAGE_RENDER_ROLE, FIGURE_ROLES, rendered_page_asset
from .literature_freeze import DOMAINS, _artifact_root, _clean, _existing_path, _figure_paths, _fulltext_path
from .source_snapshot_io import canonical_hash, sha256_file, strict_jsonl
from .source_snapshot_bindings import verify_source_members


def _validate_pdf(path: Path, role: str) -> dict[str, Any]:
    """PDFium handles source PDFs that can abort MuPDF's native renderer."""
    import pypdfium2
    with pypdfium2.PdfDocument(str(path)) as document:
        if not len(document):
            raise ValueError(f"Empty PDF document: {path}")
        if role in FIGURE_ROLES:
            for index in range(len(document)):
                page = document[index]
                try:
                    bitmap = page.render(scale=0.15)
                    bitmap.close()
                finally:
                    page.close()
        return {"kind": "document", "pages": len(document), "format": "pdf", "decoder": "pdfium"}


def validate_readable_asset(path: Path, role: str, *, suffix: str | None = None) -> dict[str, Any]:
    """Decode data without judging the scientific meaning or visual style."""
    if not path.stat().st_size:
        raise ValueError(f"Empty {role}: {path}")
    suffix = suffix if suffix is not None else path.suffix.lower()
    if role == "source_original":
        return {"kind": "source_original"}
    if role in {"full_text", "source_text", "source_companion"}:
        if suffix != ".eps":
            text = path.read_text(encoding="utf-8-sig")
            if not text.strip():
                raise ValueError(f"Whitespace-only {role}: {path}")
        return {"kind": "source" if role == "source_companion" else "text"}
    if role == "text_locations":
        values = strict_jsonl(path)
        if not values:
            raise ValueError(f"Empty locations: {path}")
        return {"kind": "jsonl", "records": len(values)}
    if role == "source_archive":
        from .literature_freeze import _archive_members
        if not _archive_members(str(path.resolve())):
            raise ValueError(f"Unreadable source archive: {path}")
        return {"kind": "archive"}
    with path.open("rb") as stream:
        pdf_header = stream.read(5) == b"%PDF-"
    if pdf_header or role == "paper_pdf":
        return _validate_pdf(path, role)
    if suffix == ".svg":
        import pymupdf
        filetype = "svg"
        with pymupdf.open(path, filetype=filetype) as document:
            if not len(document) or document.needs_pass:
                raise ValueError(f"Empty or encrypted document: {path}")
            if role in FIGURE_ROLES:
                for page in document:
                    page.get_pixmap(matrix=pymupdf.Matrix(0.15, 0.15), alpha=False)
            return {"kind": "document", "pages": len(document), "format": filetype}
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(path) as image:
            image.verify()
    except UnidentifiedImageError:
        if suffix == ".pdf":
            return _validate_pdf(path, role)
        raise
    with Image.open(path) as image:
        image.load()
        return {"kind": "image", "width": image.width, "height": image.height,
                "format": str(image.format).lower()}


def _shared_source_pdf(root: Path, arxiv_id: str, source: Path | None) -> tuple[Path | None, dict]:
    """Resolve cross-domain duplicates only when their exact source archive agrees."""
    if source is None or not arxiv_id:
        return None, {}
    source_digest = sha256_file(source)
    candidates = {}
    for domain in DOMAINS:
        folder = root / "assets" / domain / arxiv_id
        pdf, original = folder / "arxiv.pdf", folder / "arxiv_source.src"
        if pdf.is_file() and original.is_file() and sha256_file(original) == source_digest:
            candidates.setdefault(sha256_file(pdf), pdf)
    if len(candidates) > 1:
        raise ValueError("Conflicting PDF versions for the same source archive")
    if not candidates:
        return None, {}
    return next(iter(candidates.values())), {"method": "same_batch_arxiv_id_and_source_archive_sha256",
                                          "source_archive_sha256": source_digest}


def _is_pdf(path: Path) -> bool:
    """Report whether a slot holds the paper's own PDF instead of a source archive."""
    with path.open("rb") as stream:
        return stream.read(5) == b"%PDF-"


def paper_asset_specs(batch: Path, row: dict[str, Any]) -> tuple[list[dict], list[str]]:
    """Resolve source paths once and retain non-image figure companions honestly."""
    domain = str(row.get("primary_domain") or row.get("domain") or "")
    paper_id = str(row.get("paper_id") or "")
    root = _artifact_root(batch, row).resolve()
    if not root.is_relative_to(batch.parent.resolve()):
        raise ValueError(f"Artifact root is outside declared collection roots: {paper_id}")
    figure = row.get("figure") or {}
    entries: list[tuple[str, Path, str]] = []
    issues = []
    text = _fulltext_path(batch, domain, row)
    if text is None:
        raise ValueError(f"No full text: {paper_id}")
    entries.append(("full_text", text, "snapshot"))
    pages = _existing_path(batch, row, row.get("pages_path"))
    if pages is None and (text.parent / "pages.jsonl").is_file():
        pages = text.parent / "pages.jsonl"
    if pages is not None:
        entries.append(("text_locations", pages, "snapshot"))
    else:
        issues.append("original_page_mapping_unavailable")
    render = rendered_page_asset(row)
    for value in _figure_paths(row):
        path = _existing_path(batch, row, value)
        if path is None:
            raise ValueError(f"Missing selected figure: {paper_id}: {value}")
        if render and _clean(value) == render:
            entries.append((FIGURE_PAGE_RENDER_ROLE, path, "snapshot"))
            continue
        companion = path.suffix.lower() in SOURCE_FIGURE_SUFFIXES
        entries.append(("source_companion" if companion else "figure", path, "archive" if companion else "snapshot"))
    if not any(role in FIGURE_ROLES for role, _, _ in entries):
        raise ValueError(f"No decodable figure asset (only source fragments): {paper_id}")
    arxiv_id = str(row.get("arxiv_id") or "")
    source = _existing_path(batch, row, row.get("arxiv_source_path"))
    if source is None and arxiv_id:
        candidate = root / "assets" / domain / arxiv_id / "arxiv_source.src"
        source = candidate if candidate.is_file() else None
    # arXiv answers the e-print request for a PDF-only submission with the paper
    # itself, so the source slot can hold the record's own PDF instead of an archive.
    source_pdf = source is not None and _is_pdf(source)
    if source is not None and not source_pdf:
        entries.append(("source_archive", source, "archive"))
    elif row.get("figure_asset_tier") == "A_source_verified" and not source_pdf:
        for value in [figure.get("tex_file"), *(figure.get("source_files") or [])]:
            path = _existing_path(batch, row, value)
            if path is None:
                raise ValueError(f"Missing source binding original: {paper_id}: {value}")
            entries.append(("source_original", path, "archive"))
    pdf = next((p for field in ("paper_pdf_path", "official_pdf_path")
                if (p := _existing_path(batch, row, row.get(field))) is not None), None)
    if pdf is None:
        options = [root / "assets" / domain / arxiv_id / "arxiv.pdf", text.parent.parent / "official.pdf"]
        pdf = next((p for p in options if p.is_file()), None)
    pdf_resolution = {}
    if pdf is None and source_pdf:
        pdf, pdf_resolution = source, {"method": "arxiv_eprint_returned_the_paper_pdf"}
    if pdf is None:
        pdf, pdf_resolution = _shared_source_pdf(root, arxiv_id, source)
    if pdf is None:
        raise ValueError(f"Original paper PDF is unavailable for provenance: {paper_id}")
    entries.append(("paper_pdf", pdf, "archive"))
    specs = []
    seen = set()
    for role, path, storage in entries:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Asset is outside its declared root: {paper_id}: {path}")
        identity = (role, str(resolved))
        if identity in seen:
            continue
        seen.add(identity)
        try:
            check = {"kind": "source_original"} if role == "source_original" else validate_readable_asset(resolved, role)
        except Exception as exc:
            raise ValueError(f"Unreadable {role}: {resolved}: {exc}") from exc
        if not resolved.stat().st_size:
            raise ValueError(f"Empty source original: {resolved}")
        if role == "paper_pdf" and figure.get("pdf_page"):
            page = int(figure["pdf_page"])
            if page < 1 or page > int(check.get("pages", 0)):
                raise ValueError(f"Figure page is outside paper PDF: {paper_id}")
        digest = sha256_file(resolved)
        specs.append({
            "asset_id": "asset-" + canonical_hash([paper_id, role, digest])[:24],
            "paper_id": paper_id, "role": role, "storage": storage,
            "source_path": str(resolved),
            "suffix": ("." + {"jpeg": "jpg"}.get(check["format"], check["format"]))
                      if check.get("format") else resolved.suffix.lower(),
            "source_suffix": resolved.suffix.lower(),
            "sha256": digest, "size_bytes": resolved.stat().st_size,
            "readability": check,
            **({"source_resolution": pdf_resolution} if role == "paper_pdf" and pdf_resolution else {}),
        })
    if row.get("figure_asset_tier") == "A_source_verified":
        names = [figure.get("tex_file"), *(figure.get("source_files") or [])]
        originals = {name: path for name in names if name and (path := _existing_path(batch, row, name))}
        binding = verify_source_members(None if source_pdf else source, originals, names,
                                        {s["sha256"] for s in specs if s["role"] == "figure"})
        for spec in specs:
            if spec["role"] == "figure":
                spec["source_binding"] = binding
    return specs, issues
