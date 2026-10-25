from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ...utils import ensure_dir, read_json, write_json, write_text
from .document import paper_markdown, parse_paper
from .document_cache import DOCUMENT_CACHE_VERSION, read_document_cache, write_document_cache


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_paper(
    paper: str | Path,
    out: str | Path,
    deadline_seconds: int = 180,
    ocr_engine: str = "auto",
    ocr_lang: str = "en_ch",
    ocr_adapter: Callable | None = None,
    archive_input: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    source = Path(paper).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Paper does not exist: {source}")
    root = ensure_dir(out).resolve()
    active_source = source
    if archive_input:
        inputs = ensure_dir(root / "inputs")
        active_source = inputs / f"paper{source.suffix.lower()}"
        if active_source.resolve() != source:
            shutil.copyfile(source, active_source)
    digest = _source_hash(active_source)
    model_path = root / "document_model.json"
    if model_path.exists():
        cached = read_json(model_path)
        if (
            isinstance(cached, dict)
            and cached.get("source_sha256") == digest
            and cached.get("document_cache_version") == DOCUMENT_CACHE_VERSION
        ):
            report = dict(cached.get("extraction_report") or {})
            elapsed = round(time.monotonic() - started, 3)
            return {
                "summary": "PDF inspection reused a matching structured document cache.",
                "ok": report.get("status") != "fail",
                "status": "cached",
                "out_dir": str(root),
                "source_sha256": digest,
                "extraction_report": str(root / "extraction_report.json"),
                "document_model": str(model_path),
                "elapsed_seconds": elapsed,
            }
    deadline = max(30, int(deadline_seconds))
    parsed = None if ocr_adapter else read_document_cache(active_source, ocr_engine=ocr_engine, ocr_lang=ocr_lang)
    document_cache_hit = parsed is not None
    if parsed is None:
        parsed = parse_paper(
            active_source,
            deadline_at=time.monotonic() + deadline,
            ocr_engine=ocr_engine,
            ocr_lang=ocr_lang,
            ocr_adapter=ocr_adapter,
        )
        if not ocr_adapter:
            write_document_cache(active_source, parsed, ocr_engine=ocr_engine, ocr_lang=ocr_lang)
    parsed["document_cache_version"] = DOCUMENT_CACHE_VERSION
    write_json(model_path, parsed)
    write_json(root / "extraction_report.json", parsed["extraction_report"])
    write_json(root / "section_index.json", parsed["document_index"])
    write_text(root / "paper.md", paper_markdown(parsed))
    elapsed = round(time.monotonic() - started, 3)
    return {
        "summary": "Paper inspection completed without model calls.",
        "ok": parsed["extraction_report"].get("status") != "fail",
        "status": parsed["extraction_report"].get("status"),
        "out_dir": str(root),
        "source_sha256": digest,
        "page_count": parsed["page_count"],
        "evidence_count": len(parsed["evidence"]),
        "document_cache_hit": document_cache_hit,
        "extraction_report": str(root / "extraction_report.json"),
        "document_model": str(model_path),
        "elapsed_seconds": elapsed,
    }
