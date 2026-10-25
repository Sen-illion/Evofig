from __future__ import annotations

from pathlib import Path


def rasterize_pdf_as_scan(source: str | Path, target: str | Path, dpi: int = 144, jpeg_quality: int = 78) -> Path:
    """Create an image-only PDF fixture while preserving displayed page sizes."""
    import fitz

    source_path = Path(source).resolve()
    target_path = Path(target).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"PDF does not exist: {source_path}")
    if source_path.suffix.casefold() != ".pdf":
        raise ValueError(f"Rasterized scan input must be a PDF: {source_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.unlink(missing_ok=True)
    document = fitz.open(str(source_path))
    scanned = fitz.open()
    try:
        for page in document:
            pixmap = page.get_pixmap(dpi=max(72, min(300, int(dpi))), alpha=False)
            image = pixmap.tobytes("jpeg", jpg_quality=max(40, min(95, int(jpeg_quality))))
            output_page = scanned.new_page(width=float(page.rect.width), height=float(page.rect.height))
            output_page.insert_image(output_page.rect, stream=image)
        scanned.set_metadata({
            "title": "ResearchFigureStudio rasterized scan benchmark",
            "author": "ResearchFigureStudio",
            "subject": "Deterministic image-only PDF benchmark fixture",
            "keywords": "ResearchFigureStudio,scan,benchmark",
            "creator": "ResearchFigureStudio",
            "producer": "ResearchFigureStudio",
            "creationDate": "",
            "modDate": "",
        })
        scanned.save(str(target_path), garbage=4, deflate=True, no_new_id=True, preserve_metadata=False)
    finally:
        scanned.close()
        document.close()
    return target_path
