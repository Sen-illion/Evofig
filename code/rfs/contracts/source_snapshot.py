"""Stable manifest and path contracts for immutable Judge source snapshots."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any


SOURCE_SNAPSHOT_VERSION = "judge-source-snapshot-v1"
SOURCE_FIGURE_SUFFIXES = frozenset({".tex", ".tikz", ".pgf", ".pdf_tex", ".eps"})
# Which audited asset tiers each named publication scope selects. Every scope
# audits the whole strict view; only the published set and its stated coverage differ.
SCOPE_TIERS = {"all": (), "a-source": ("A_source_verified",)}
REQUIRED_SNAPSHOT_FILES = (
    "eligible_papers.jsonl", "asset_manifest.jsonl", "domain_report.json",
    "eligibility_audit.jsonl", "exclusion_summary.json", "review_queue.jsonl",
    "validation_report.json", "schema_versions.json",
)
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def snapshot_relative_path(value: Any) -> str:
    """Reject absolute, drive-qualified, ambiguous, and escaping asset paths."""
    text = str(value or "")
    path = PurePosixPath(text)
    if (
        not text or "\\" in text or ":" in text or "\x00" in text
        or path.is_absolute() or any(part in {"", ".", ".."} for part in text.split("/"))
    ):
        raise ValueError(f"Invalid snapshot relative path: {text!r}")
    return path.as_posix()


def validate_source_manifest(manifest: dict[str, Any]) -> list[str]:
    """Check source-only publication semantics independently of quota success."""
    errors = []
    if manifest.get("schema_version") != SOURCE_SNAPSHOT_VERSION:
        errors.append("unsupported_source_snapshot_schema")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(manifest.get("snapshot_id") or "")):
        errors.append("invalid_snapshot_id")
    if manifest.get("training_readiness") != "source_only":
        errors.append("snapshot_must_be_source_only")
    if manifest.get("source_split_is_authoritative") is not False:
        errors.append("source_split_must_not_be_authoritative")
    if manifest.get("snapshot_integrity_ok") is not True:
        errors.append("snapshot_integrity_not_passed")
    for key in ("file_hashes_sha256", "paper_set_sha256", "input_set_sha256"):
        if not HASH_PATTERN.fullmatch(str(manifest.get(key) or "")):
            errors.append(f"invalid_{key}")
    if not isinstance(manifest.get("strict_unique_papers"), int) or manifest["strict_unique_papers"] <= 0:
        errors.append("empty_or_invalid_paper_count")
    scope = manifest.get("scope")
    if scope is not None:
        counts = [scope.get(key) for key in
                  ("in_scope_papers", "out_of_scope_strict_papers", "strict_view_papers")] if isinstance(scope, dict) else []
        if len(counts) != 3 or not all(isinstance(value, int) and value >= 0 for value in counts):
            errors.append("invalid_scope_counts")
        elif counts[0] + counts[1] != counts[2] or counts[0] != manifest.get("strict_unique_papers"):
            errors.append("inconsistent_scope_counts")
    return errors
