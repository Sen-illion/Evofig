"""Build a strict A/B-only paper training view without mutating history."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import tarfile
import time
from typing import Any, Iterable
import zipfile

from rfs.utils import write_json
from rfs.contracts.source_snapshot import SOURCE_FIGURE_SUFFIXES

from .literature_io import atomic_write_text
from .official_record import publisher_proof_ok


DOMAINS = ("agent", "rag", "cv", "biology", "materials")
ACCEPTED_TYPES = {
    "agent": {"long", "full", "main"},
    "rag": {"long", "full", "main"},
    "cv": {"long", "full", "main"},
    "biology": {"research_article", "original_research"},
    "materials": {"research_article", "original_research"},
}
STRICT_ASSET_TIERS = {"A_source_verified", "B_official_standalone"}
EXPLICIT_ID_FIELDS = ("doi", "paper_id", "official_paper_id", "arxiv_id", "pmid", "pmcid")
AGGREGATOR_SOURCES = {
    "crossref",
    "crossref_journal",
    "acm_proceedings",
    "aaai_ojs",
    "dblp",
    "semantic_scholar",
    "openalex",
    "unpaywall",
}


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean(value).casefold()).strip()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def _portable_path(value: Any) -> Path:
    """Read one recorded path with the separators the capturing host wrote."""
    return Path(_clean(value).replace("\\", "/"))


def _artifact_root(batch: Path, row: dict[str, Any]) -> Path:
    """Resolve one record's artifact root without trusting host-specific paths.

    Records were captured on Windows, so relative asset paths use backslashes
    and ``legacy_batch`` can be an absolute drive-letter path naming an earlier
    collection.  Neither survives a move to another machine, so the declared
    root is honoured only while it is reachable and is otherwise matched by
    name against the batch's sibling directories, then dropped in favour of the
    batch itself, which is where merged artifacts live.
    """
    root = _clean(row.get("legacy_batch") or row.get("artifact_batch_root"))
    if not root or root == ".":
        return batch
    declared = _portable_path(root)
    if declared.is_dir():
        return declared
    sibling = batch.parent / declared.name
    if declared.name and sibling.is_dir():
        return sibling
    return batch


def _existing_path(batch: Path, row: dict[str, Any], value: Any) -> Path | None:
    text = _clean(value)
    if not text:
        return None
    path = _portable_path(text)
    if path.is_absolute():
        return path if path.is_file() else None
    candidate = _artifact_root(batch, row) / path
    return candidate if candidate.is_file() else None


def _figure_paths(row: dict[str, Any]) -> list[str]:
    figure = row.get("figure") or {}
    values = figure.get("image_paths") or row.get("figure_assets") or []
    return [str(value) for value in values if _clean(value)]


@lru_cache(maxsize=1024)
def _archive_members(path_text: str) -> frozenset[str]:
    path = Path(path_text)
    try:
        with tarfile.open(path, mode="r:*") as archive:
            return frozenset(str(PurePosixPath(item.name)).casefold() for item in archive if item.isfile())
    except (OSError, tarfile.TarError):
        pass
    try:
        with zipfile.ZipFile(path) as archive:
            return frozenset(
                str(PurePosixPath(item.filename)).casefold()
                for item in archive.infolist()
                if not item.is_dir()
            )
    except (OSError, zipfile.BadZipFile):
        return frozenset()


def _source_binding_files_exist(batch: Path, domain: str, row: dict[str, Any]) -> bool:
    figure = row.get("figure") or {}
    values = [figure.get("tex_file"), *(figure.get("source_files") or [])]
    if values and all(_existing_path(batch, row, value) for value in values):
        return True
    arxiv_id = _clean(row.get("arxiv_id"))
    if not arxiv_id:
        return False
    archive = _artifact_root(batch, row) / "assets" / domain / arxiv_id / "arxiv_source.src"
    if not archive.is_file():
        return False
    members = _archive_members(str(archive.resolve()))
    expected = {
        str(PurePosixPath(_clean(value).replace("\\", "/"))).casefold()
        for value in values
        if _clean(value)
    }
    return bool(expected) and expected <= members


def infer_asset_tier(row: dict[str, Any]) -> str | None:
    """Infer A only for legacy rows carrying the complete old source binding."""
    tier = _clean(row.get("figure_asset_tier"))
    if tier:
        return tier
    figure = row.get("figure") or {}
    if (
        row.get("record_origin") == "legacy_verified"
        and figure.get("binding_status") == "verified"
        and figure.get("tex_file")
        and figure.get("source_files")
        and figure.get("pdf_page")
        and _figure_paths(row)
    ):
        return "A_source_verified"
    return None


def _public_folder_name(identity: str) -> str:
    """Match the collector's folder naming for one multi-source public asset."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", identity)[:180]


def _public_identities(row: dict[str, Any]) -> list[str]:
    """Return the identity spellings the collector may have used for public assets."""
    values = [_clean(row.get(field)) for field in ("paper_id", "official_paper_id", "doi")]
    return list(dict.fromkeys(value for value in values if value))


def _fulltext_path(batch: Path, domain: str, row: dict[str, Any]) -> Path | None:
    """Locate the complete text of one historical record across its asset layouts."""
    for value in (row.get("full_text_path"), row.get("source_text_path")):
        if path := _existing_path(batch, row, value):
            return path
    arxiv_id = _clean(row.get("arxiv_id"))
    if arxiv_id:
        for relative in (
            f"assets/{domain}/{arxiv_id}/text/full.txt",
            f"assets/{domain}/{arxiv_id}/text/source_text.txt",
        ):
            candidate = batch / relative
            if candidate.is_file() and candidate.stat().st_size:
                return candidate
    for identity in _public_identities(row):
        for folder in (identity, _public_folder_name(identity)):
            candidate = batch / "assets" / domain / "public" / folder / "text" / "full.txt"
            if candidate.is_file() and candidate.stat().st_size:
                return candidate
    return None


def strict_failures(batch: Path, domain: str, row: dict[str, Any]) -> list[str]:
    """Return deterministic hard-gate failures for one historical paper."""
    failures: list[str] = []
    if row.get("official_record_status") != "verified" or not row.get("official_url"):
        failures.append("official_record_unverified")
    source = _clean(row.get("official_source") or row.get("provider")).casefold()
    if source in AGGREGATOR_SOURCES and not publisher_proof_ok(row):
        failures.append("official_publisher_proof_missing")
    if _clean(row.get("paper_type")).casefold() not in ACCEPTED_TYPES[domain]:
        failures.append("not_long_full_or_research_article")
    tier = infer_asset_tier(row)
    if tier not in STRICT_ASSET_TIERS:
        failures.append("strict_ab_asset_missing")
    figure = row.get("figure") or {}
    paths = _figure_paths(row)
    if paths and all(Path(value).suffix.lower() in SOURCE_FIGURE_SUFFIXES for value in paths):
        failures.append("figure_rendering_missing")
    if not figure.get("caption") or not paths or not all(_existing_path(batch, row, value) for value in paths):
        failures.append("figure_file_or_caption_unbound")
    if tier == "A_source_verified" and not (
        figure.get("tex_file")
        and figure.get("source_files")
        and figure.get("pdf_page")
    ):
        failures.append("source_binding_incomplete")
    elif tier == "A_source_verified":
        if not _source_binding_files_exist(batch, domain, row):
            failures.append("source_files_missing")
    if tier == "B_official_standalone" and not (
        row.get("standalone_asset")
        and figure.get("binding_status") == "verified"
        and (figure.get("asset_source_url") or row.get("asset_source_url"))
    ):
        failures.append("official_standalone_binding_incomplete")
    fulltext = _fulltext_path(batch, domain, row)
    if fulltext is None:
        failures.append("fulltext_missing")
    elif tier == "B_official_standalone":
        try:
            content = _clean(fulltext.read_text(encoding="utf-8-sig")).casefold()
        except UnicodeError:
            failures.append("fulltext_unreadable")
        else:
            if "this is a preview of subscription content" in content:
                failures.append("fulltext_preview_only")
    return sorted(set(failures))


def _identity_keys(row: dict[str, Any]) -> set[str]:
    keys = {
        f"{field}:{_clean(row.get(field)).casefold()}"
        for field in EXPLICIT_ID_FIELDS
        if _clean(row.get(field))
    }
    title = _norm(row.get("title"))
    authors = row.get("authors") or []
    if title and isinstance(authors, list) and authors:
        keys.add(f"title_author:{title}|{_norm(authors[0])}")
    elif title and not keys:
        keys.add(f"title:{title}")
    return keys


def canonicalize(rows_by_domain: dict[str, list[dict[str, Any]]], *, include_provenance: bool = False) -> list[dict[str, Any]]:
    """Deduplicate strict rows while retaining deterministic multi-domain labels."""
    flattened = [(domain, row) for domain in DOMAINS for row in rows_by_domain.get(domain, [])]
    parent = list(range(len(flattened)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    owner: dict[str, int] = {}
    for index, (_domain, row) in enumerate(flattened):
        for key in _identity_keys(row):
            if key in owner:
                union(index, owner[key])
            else:
                owner[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(flattened)):
        groups.setdefault(find(index), []).append(index)
    result: list[dict[str, Any]] = []
    for indices in sorted(groups.values(), key=min):
        representative = min(
            indices,
            key=lambda index: (
                infer_asset_tier(flattened[index][1]) != "A_source_verified",
                flattened[index][1].get("record_origin") != "legacy_verified",
                index,
            ),
        )
        primary, source = flattened[representative]
        item = dict(source)
        item["figure_asset_tier"] = infer_asset_tier(item)
        item["primary_domain"] = primary
        item["domain_labels"] = [
            domain for domain in DOMAINS if any(flattened[index][0] == domain for index in indices)
        ]
        item["strict_training_status"] = "verified"
        if include_provenance:
            item["canonical_sources"] = [
                {"domain": flattened[index][0], "record": flattened[index][1],
                 "identity_keys": sorted(_identity_keys(flattened[index][1]))}
                for index in indices
            ]
        result.append(item)
    return result


def _backup_name(path: Path) -> str:
    """Qualify the backup by its parent directory.

    All five domain ledgers are named ``papers.jsonl``. Copying them into one
    backup root by bare filename kept only the last domain written and silently
    destroyed the other four pre-images.
    """
    parent = path.parent.name
    return f"{parent}.{path.name}" if parent else path.name


def _publish_with_backup(path: Path, content: str, backup_root: Path) -> bool:
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    if previous == content:
        return False
    if path.exists():
        backup_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_root / _backup_name(path))
    atomic_write_text(path, content)
    return True


def build_strict_training_view(
    batch_root: str | Path,
    *,
    target_per_domain: int = 2000,
    domain_workers: int = 5,
    paper_workers: int = 16,
) -> dict[str, Any]:
    """Publish the strict view, its revalidation queue, and one single-count report.

    A paper can satisfy several domains, so it is counted once:
    ``domains[domain]["strict"]`` holds the unique papers ``canonicalize``
    attributed to that domain, ``domains[domain]["strict_membership"]`` holds how
    many papers name the domain, and ``duplicates_merged`` is the difference
    between the two. The per-domain ``strict`` counts sum to
    ``strict_training_total``, and ``shortfall`` and ``ok`` follow them, because
    the unique count is what the collector still has to close.
    """
    batch = Path(batch_root).resolve()
    rows_by_domain: dict[str, list[dict[str, Any]]] = {}
    retry_rows: list[dict[str, Any]] = []
    historical_total = 0
    failures: Counter[str] = Counter()
    domain_report: dict[str, Any] = {}
    def evaluate_domain(
        domain: str,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], Counter[str], dict[str, Any]]:
        historical = _jsonl(batch / "verified" / domain / "papers.jsonl")
        strict: list[dict[str, Any]] = []
        domain_retry: list[dict[str, Any]] = []
        domain_failures: Counter[str] = Counter()
        with ThreadPoolExecutor(
            max_workers=max(1, paper_workers),
            thread_name_prefix=f"strict-paper-{domain}",
        ) as paper_pool:
            checked = paper_pool.map(lambda row: (row, strict_failures(batch, domain, row)), historical)
            for row, row_failures in checked:
                if not row_failures:
                    item = dict(row)
                    item["figure_asset_tier"] = infer_asset_tier(item)
                    strict.append(item)
                    continue
                domain_failures.update(row_failures)
                if infer_asset_tier(row) == "C_pdf_object_extracted":
                    domain_retry.append(
                        {
                            "domain": domain,
                            "title": row.get("title"),
                            "doi": row.get("doi"),
                            "paper_id": row.get("paper_id"),
                            "official_paper_id": row.get("official_paper_id"),
                            "arxiv_id": row.get("arxiv_id"),
                            "previous_asset_tier": "C_pdf_object_extracted",
                            "revalidation_targets": ["A_source_verified", "B_official_standalone"],
                            "strict_training_status": "revalidate_excluded",
                        }
                    )
        return domain, strict, domain_retry, domain_failures, {
            "historical": len(historical),
            "target": target_per_domain,
            "failure_reasons": dict(domain_failures),
        }

    with ThreadPoolExecutor(
        max_workers=max(1, min(domain_workers, len(DOMAINS))),
        thread_name_prefix="strict-domain",
    ) as domain_pool:
        evaluated_domains = domain_pool.map(evaluate_domain, DOMAINS)
        for domain, strict, domain_retry, domain_failures, summary in evaluated_domains:
            historical_total += summary["historical"]
            rows_by_domain[domain] = strict
            retry_rows.extend(domain_retry)
            failures.update(domain_failures)
            domain_report[domain] = summary
    canonical = canonicalize(rows_by_domain)
    unique_by_domain: Counter[str] = Counter()
    membership_by_domain: Counter[str] = Counter()
    tiers_by_domain: dict[str, Counter[str]] = {domain: Counter() for domain in DOMAINS}
    cross_domain_papers = 0
    for row in canonical:
        primary = row["primary_domain"]
        unique_by_domain[primary] += 1
        tiers_by_domain[primary][row["figure_asset_tier"]] += 1
        if len(row["domain_labels"]) > 1:
            cross_domain_papers += 1
        for domain in row["domain_labels"]:
            membership_by_domain[domain] += 1
    for domain in DOMAINS:
        summary = domain_report[domain]
        summary["strict"] = unique_by_domain[domain]
        summary["strict_membership"] = membership_by_domain[domain]
        summary["duplicates_merged"] = membership_by_domain[domain] - unique_by_domain[domain]
        summary["shortfall"] = max(0, target_per_domain - unique_by_domain[domain])
        summary["asset_tiers"] = dict(tiers_by_domain[domain])
    retry_unique: list[dict[str, Any]] = []
    seen_retry: set[str] = set()
    for row in retry_rows:
        key = next(iter(sorted(_identity_keys(row))), hashlib.sha256(_jsonl_text([row]).encode()).hexdigest())
        if key not in seen_retry:
            seen_retry.add(key)
            retry_unique.append(row)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = batch / "manifests" / "backups" / f"strict_training_{stamp}"
    strict_path = batch / "verified" / "strict_training_papers.jsonl"
    retry_path = batch / "retry" / "c_asset_revalidation.jsonl"
    strict_changed = _publish_with_backup(strict_path, _jsonl_text(canonical), backup)
    retry_changed = _publish_with_backup(retry_path, _jsonl_text(retry_unique), backup)
    report = {
        "version": "strict-literature-training-v001",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "historical_ledger_path": "verified/{domain}/papers.jsonl",
        "historical_verified_total": historical_total,
        "strict_training_path": "verified/strict_training_papers.jsonl",
        "strict_training_total": len(canonical),
        "strict_membership_total": sum(membership_by_domain.values()),
        "cross_domain_papers": cross_domain_papers,
        "domains": domain_report,
        "strict_asset_tiers": sorted(STRICT_ASSET_TIERS),
        "new_c_tier_allowed": False,
        "c_asset_revalidation_path": "retry/c_asset_revalidation.jsonl",
        "c_asset_revalidation_count": len(retry_unique),
        "failure_reasons": dict(failures),
        "target_per_domain": target_per_domain,
        "target_unique": target_per_domain * len(DOMAINS),
        "ok": len(canonical) >= target_per_domain * len(DOMAINS)
        and all(value["strict"] >= target_per_domain for value in domain_report.values()),
        "outputs_changed": strict_changed or retry_changed,
        "backup": str(backup.relative_to(batch)) if backup.exists() else None,
    }
    write_json(batch / "manifests" / "strict_training_dashboard.json", report)
    return report


__all__ = [
    "ACCEPTED_TYPES",
    "DOMAINS",
    "STRICT_ASSET_TIERS",
    "build_strict_training_view",
    "canonicalize",
    "infer_asset_tier",
    "strict_failures",
]
