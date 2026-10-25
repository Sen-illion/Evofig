from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ...utils import read_json, write_json


#: Bumped when a stored parse would be reread differently, so the old file is not reused.
DOCUMENT_CACHE_VERSION = 35


def document_model_cacheable(parsed: dict[str, Any]) -> bool:
    report = parsed.get("extraction_report", {})
    return bool(report.get("status") != "fail" and report.get("ocr_run_complete", True))


def _cache_root() -> Path:
    return Path(os.getenv("RFS_CACHE_DIR", "").strip() or (Path.home() / ".cache" / "research-figure-studio"))


def document_cache_path(
    source: str | Path,
    *,
    ocr_engine: str,
    ocr_lang: str,
    max_chars: int,
    max_ocr_pages: int,
) -> Path:
    path = Path(source).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    signature = json.dumps({
        "version": DOCUMENT_CACHE_VERSION,
        "suffix": path.suffix.lower(),
        "ocr_engine": str(ocr_engine),
        "ocr_lang": str(ocr_lang),
        "max_chars": int(max_chars),
        "max_ocr_pages": int(max_ocr_pages),
    }, sort_keys=True).encode("utf-8")
    variant = hashlib.sha256(signature).hexdigest()[:16]
    return _cache_root() / "documents" / digest / variant / "document_model.json"


def read_document_cache(
    source: str | Path,
    *,
    ocr_engine: str,
    ocr_lang: str,
    max_chars: int = 90000,
    max_ocr_pages: int = 6,
) -> dict[str, Any] | None:
    path = document_cache_path(source, ocr_engine=ocr_engine, ocr_lang=ocr_lang, max_chars=max_chars, max_ocr_pages=max_ocr_pages)
    if not path.exists():
        return None
    cached = read_json(path)
    if (
        not isinstance(cached, dict)
        or cached.get("document_cache_version") != DOCUMENT_CACHE_VERSION
        or not document_model_cacheable(cached)
    ):
        return None
    active = Path(source).resolve()
    cached["source_path"] = str(active)
    cached["source_name"] = active.name
    if isinstance(cached.get("extraction_report"), dict):
        cached["extraction_report"]["source_path"] = str(active)
    return cached


def write_document_cache(
    source: str | Path,
    parsed: dict[str, Any],
    *,
    ocr_engine: str,
    ocr_lang: str,
    max_chars: int = 90000,
    max_ocr_pages: int = 6,
) -> Path | None:
    if not document_model_cacheable(parsed):
        return None
    path = document_cache_path(source, ocr_engine=ocr_engine, ocr_lang=ocr_lang, max_chars=max_chars, max_ocr_pages=max_ocr_pages)
    payload = dict(parsed)
    payload["document_cache_version"] = DOCUMENT_CACHE_VERSION
    write_json(path, payload)
    return path
