"""Orchestrate resumable 2026 multi-source literature collection stages."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

from rfs.providers.literature import DblpProvider, ProviderRequestError
from rfs.utils import write_json

from .arxiv_matching import match_arxiv_candidates
from .literature_freeze import DOMAINS
from .literature_io import atomic_write_text
from .official_assets import bind_official_assets
from .strict_candidate_freeze import freeze_strict_candidates


ROOT = Path(__file__).resolve().parents[2]
STAGES = ("inventory", "official-verify", "arxiv-match", "asset-bind", "fulltext", "freeze", "all")
# A full run resolves every still-unmatched, officially verified title of a domain
# through the live title resolver, one paced network round trip per title. The 2026
# pools measured 2,300 (agent) and 1,959 (rag) eligible rows on 2026-09-25, so this
# covers a whole pass with headroom. Rows a pass cannot resolve stay unmatched and are
# retried by the next pass.
LIVE_TITLE_TAIL_PER_DOMAIN = 3000
EXPECTED_SOURCES = (
    "acl_anthology",
    "cvf_open_access",
    "pmlr",
    "iclr_official",
    "neurips_proceedings",
    "aaai_ojs",
    "aamas_proceedings",
    "acm_proceedings",
    "europe_pmc",
    "crossref_journal",
    "dblp",
    "openalex",
)
DBLP_VENUES = {
    "agent": ("AAAI", "AAMAS", "IJCAI", "ICML", "ICLR", "NeurIPS", "ACL", "EMNLP", "NAACL", "EACL"),
    "rag": ("SIGIR", "KDD", "WWW", "ACL", "ICML", "ICLR", "NeurIPS", "IJCAI", "EMNLP", "NAACL", "EACL"),
    "cv": ("CVPR", "ECCV"),
}
TOPICS = {
    "agent": ("agent", "planning", "reasoning", "tool", "memory", "multi-agent", "robot"),
    "rag": ("retrieval", "search", "grounded", "knowledge", "citation", "rerank", "question answering"),
    "cv": ("vision", "image", "video", "3d", "segmentation", "detection", "multimodal"),
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _identity(row: dict[str, Any]) -> str:
    for field in ("doi", "paper_id", "official_paper_id", "arxiv_id"):
        if value := str(row.get(field) or "").strip():
            return f"{field}:{value.casefold()}"
    return f"title:{_norm(row.get('title'))}"


def _normalize_candidate(row: dict[str, Any], domain: str, year: int) -> dict[str, Any]:
    item = dict(row)
    item.setdefault("record_status", "candidate")
    item["domain"] = domain
    item["year"] = year
    item.setdefault("authors", [])
    item.setdefault("official_paper_id", item.get("paper_id"))
    item.setdefault("license", None)
    item.setdefault("official_record_status", "publisher_landing_pending")
    item.setdefault("official_verification_required", item["official_record_status"] != "verified")
    source = str(item.get("official_source") or item.get("provider") or "unknown")
    item["official_source"] = source
    item.setdefault(
        "provenance",
        [
            {
                "provider": source,
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "official_proof": item["official_record_status"] == "verified",
            }
        ],
    )
    return item


def _published_by_now(row: dict[str, Any], year: int) -> bool:
    if int(row.get("year") or 0) != year:
        return False
    publication_date = str(row.get("publication_date") or "").strip()[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", publication_date):
        return True
    try:
        return date.fromisoformat(publication_date) <= date.today()
    except ValueError:
        return False


def merge_staged_inventory(
    batch: Path,
    staging: Path,
    *,
    year: int,
    domains: Iterable[str],
    canary_per_source: int = 0,
) -> dict[str, Any]:
    """Append a staging inventory without replacing existing candidates."""
    report: dict[str, Any] = {"domains": {}, "source_counts": Counter()}
    for domain in domains:
        main_path = batch / "candidates" / domain / "papers.jsonl"
        existing = _jsonl(main_path)
        staged = [
            _normalize_candidate(row, domain, year)
            for row in _jsonl(staging / "candidates" / domain / "papers.jsonl")
            if _published_by_now(row, year)
        ]
        if canary_per_source:
            limited: list[dict[str, Any]] = []
            counts: Counter[str] = Counter()
            for row in staged:
                source = str(row.get("official_source") or "unknown")
                if counts[source] < canary_per_source:
                    limited.append(row)
                    counts[source] += 1
            staged = limited
        by_id = {_identity(row): row for row in existing}
        added = 0
        for row in staged:
            key = _identity(row)
            if key not in by_id:
                by_id[key] = row
                added += 1
        rows = list(by_id.values())
        main_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(main_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        source_counts = Counter(str(row.get("official_source") or "unknown") for row in staged)
        for source, count in source_counts.items():
            raw_path = batch / "raw" / source / domain / str(year) / "papers.jsonl"
            raw_existing = {_identity(row): row for row in _jsonl(raw_path)}
            raw_existing.update({_identity(row): row for row in staged if row.get("official_source") == source})
            atomic_write_text(raw_path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in raw_existing.values()))
            report["source_counts"][source] += count
        report["domains"][domain] = {"existing_before": len(existing), "staged": len(staged), "added": added, "after": len(rows)}
    report["source_counts"] = dict(report["source_counts"])
    return report


def prepare_canary_batch(
    batch: Path,
    *,
    year: int,
    domains: Iterable[str],
    per_source: int,
) -> Path:
    """Create one persistent isolated batch for post-inventory live canaries."""
    canary = batch / "work" / f"canary_{year}"
    for domain in domains:
        destination = canary / "candidates" / domain / "papers.jsonl"
        if not destination.exists():
            selected: list[dict[str, Any]] = []
            seen: set[str] = set()
            for source in EXPECTED_SOURCES:
                raw_path = batch / "raw" / source / domain / str(year) / "papers.jsonl"
                for row in _jsonl(raw_path)[:per_source]:
                    key = _identity(row)
                    if key not in seen:
                        seen.add(key)
                        selected.append(row)
            atomic_write_text(
                destination,
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
            )
        cache_pattern = f"arxiv_harvest_{domain}_*_{year}_*.json"
        cache_candidates = list((batch / "manifests").glob(cache_pattern))
        if cache_candidates:
            source_cache = max(cache_candidates, key=lambda path: path.stat().st_size)
            target_cache = canary / "manifests" / source_cache.name
            if not target_cache.exists():
                target_cache.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source_cache, target_cache)
                except OSError:
                    shutil.copy2(source_cache, target_cache)
    write_json(
        canary / "manifests" / "canary_contract.json",
        {
            "year": year,
            "per_source": per_source,
            "domains": list(domains),
            "source_batch": str(batch),
            "strict_isolation": True,
        },
    )
    return canary


def _collect_dblp(batch: Path, year: int, domains: Iterable[str], canary: int) -> dict[str, Any]:
    provider = DblpProvider()
    result: dict[str, Any] = {"status": "complete", "records": 0, "errors": []}
    for domain in domains:
        if domain not in DBLP_VENUES:
            continue
        values: list[dict[str, Any]] = []
        for venue in DBLP_VENUES[domain]:
            try:
                page = provider.search(f"{venue} {year}", rows=max(10, canary or 100))
            except ProviderRequestError as exc:
                result["status"] = exc.status
                result["errors"].append(str(exc))
                continue
            for row in page.records:
                if str(row.get("publication_date")) != str(year):
                    continue
                if not any(term in _norm(row.get("title")) for term in TOPICS[domain]):
                    continue
                doi = str(row.get("doi") or "")
                item = _normalize_candidate(
                    {
                        **row,
                        "paper_id": doi or f"dblp:{_norm(row.get('title'))}",
                        "official_paper_id": None,
                        "official_url": f"https://doi.org/{doi}" if doi else row.get("official_url"),
                        "official_source": "dblp",
                        "official_record_status": "publisher_landing_pending",
                        "official_verification_required": True,
                        "paper_type": None,
                    },
                    domain,
                    year,
                )
                values.append(item)
        path = batch / "work" / f"dblp_{year}" / "candidates" / domain / "papers.jsonl"
        atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values))
        result["records"] += len(values)
    return result


def _run_worker(script: str, args: list[str]) -> dict[str, Any]:
    path = ROOT / "extra-dataset" / "scripts" / script
    completed = subprocess.run([sys.executable, str(path), *args], cwd=ROOT, check=False)
    return {"script": script, "returncode": completed.returncode, "ok": completed.returncode == 0}


def _safe_payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _stage_ok(value: Any) -> bool:
    if not isinstance(value, dict):
        return True
    if value.get("ok") is False:
        return False
    return all(_stage_ok(child) for child in value.values() if isinstance(child, dict))


def _provider_on_cooldown(batch: Path, source: str, seconds: int = 3600) -> bool:
    payload = _safe_payload(batch / "manifests" / "multisource_sources.json")
    status = (payload.get("sources", {}).get(source, {}) or {}).get("status")
    if status not in {"failed", "blocked", "rate_limited"}:
        return False
    try:
        generated = datetime.fromisoformat(str(payload.get("generated_at")).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return (datetime.now(timezone.utc) - generated).total_seconds() < seconds


def _append_event(batch: Path, stage: str, status: str, **details: Any) -> None:
    path = batch / "manifests" / "multisource_collection_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": stage, "status": status, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details}, ensure_ascii=False) + "\n")


def _coverage(batch: Path, year: int, inventory: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    counts = Counter(inventory.get("source_counts") or {})
    sources = {}
    for source in EXPECTED_SOURCES:
        domain_counts = {
            domain: len(_jsonl(batch / "raw" / source / domain / str(year) / "papers.jsonl"))
            for domain in DOMAINS
        }
        count = max(int(counts.get(source, 0)), sum(domain_counts.values()))
        source_errors = [error for error in errors if source.replace("_", " ") in error.casefold().replace("_", " ")]
        if count:
            status = "complete"
        elif source == "openalex":
            resume = batch / "manifests" / f"openalex_arxiv_{year}_budgeted_resume.json"
            months = _safe_payload(resume).get("months") or []
            status = "complete" if months and all(item.get("completed") for item in months) else "rate_limited"
        elif source_errors:
            text = " ".join(source_errors).casefold()
            status = "rate_limited" if "429" in text else "blocked" if "403" in text else "failed"
        elif source in {"dblp"}:
            status = "failed"
        else:
            status = "not_yet_published"
        sources[source] = {
            "status": status,
            "candidate_count": count,
            "domain_counts": domain_counts,
            "errors": source_errors[:20],
        }
    payload = {
        "version": "multisource-coverage-v001",
        "year": year,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": sources,
        "unattributed_errors": [
            error
            for error in errors
            if not any(source.replace("_", " ") in error.casefold().replace("_", " ") for source in EXPECTED_SOURCES)
        ][:50],
    }
    write_json(batch / "manifests" / "multisource_sources.json", payload)
    return payload


def collect_literature_papers(
    batch_root: str | Path,
    *,
    year: int,
    stage: str = "all",
    domains: Iterable[str] = DOMAINS,
    resume: bool = False,
    domain_workers: int = 5,
    paper_workers: int = 16,
    asset_policy: str = "strict-ab",
    legacy_c_policy: str = "revalidate-excluded",
    canary_per_source: int = 0,
) -> dict[str, Any]:
    """Run selected stages while preserving the append-only historical ledger."""
    if stage not in STAGES:
        raise ValueError(f"unsupported collection stage: {stage}")
    if asset_policy != "strict-ab" or legacy_c_policy != "revalidate-excluded":
        raise ValueError("only strict-ab with revalidate-excluded is supported")
    if year != 2026:
        raise ValueError("the active collection policy is restricted to 2026")
    requested_domains = list(dict.fromkeys(domains))
    unknown_domains = [domain for domain in requested_domains if domain not in DOMAINS]
    if unknown_domains:
        raise ValueError(f"unsupported domains: {', '.join(unknown_domains)}")
    selected_domains = [domain for domain in requested_domains if domain in DOMAINS]
    if not selected_domains:
        raise ValueError("at least one domain is required")
    if domain_workers < 1 or paper_workers < 1:
        raise ValueError("worker counts must be positive")
    batch = Path(batch_root).resolve()
    batch.mkdir(parents=True, exist_ok=True)
    canary_batch = (
        prepare_canary_batch(
            batch,
            year=year,
            domains=selected_domains,
            per_source=canary_per_source,
        )
        if canary_per_source and stage != "inventory"
        else None
    )
    selected_stages = list(STAGES[:-1]) if stage == "all" else [stage]
    result: dict[str, Any] = {
        "summary": "2026 multi-source literature collection run.",
        "year": year,
        "domains": selected_domains,
        "asset_policy": asset_policy,
        "legacy_c_policy": legacy_c_policy,
        "execution_batch": str(canary_batch or batch),
        "canary_isolated": bool(canary_batch),
        "stages": {},
        "ok": True,
    }
    fulltext_completed = False
    for current in selected_stages:
        if current == "fulltext" and fulltext_completed:
            continue
        _append_event(batch, current, "started")
        active_batch = batch if current == "inventory" else (canary_batch or batch)
        try:
            if current == "inventory":
                staging = batch / "work" / f"official_inventory_{year}"
                manifest = staging / "manifests" / "official_inventory.json"
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="inventory-source") as pool:
                    if resume and manifest.exists():
                        official_future = None
                        worker = {"ok": True, "returncode": 0, "resumed": True}
                    else:
                        official_future = pool.submit(
                            _run_worker,
                            "collect_official_fresh.py",
                            [
                                "--output-root", str(staging), "--start-year", str(year), "--end-year", str(year),
                                "--domains", *selected_domains, "--target-per-domain", "2000", "--limit-per-journal", "0",
                                "--expanded-topic-filter", "--domain-workers", str(domain_workers),
                            ],
                        )
                    if resume and _provider_on_cooldown(batch, "dblp"):
                        dblp_future = None
                        dblp = {
                            "status": "failed",
                            "records": 0,
                            "errors": ["DBLP retry deferred by one-hour provider cooldown"],
                            "cooldown": True,
                        }
                    else:
                        dblp_future = pool.submit(_collect_dblp, batch, year, selected_domains, canary_per_source)
                    if official_future is not None:
                        worker = official_future.result()
                    if dblp_future is not None:
                        dblp = dblp_future.result()
                inventory = merge_staged_inventory(batch, staging, year=year, domains=selected_domains, canary_per_source=canary_per_source)
                dblp_inventory = merge_staged_inventory(
                    batch,
                    batch / "work" / f"dblp_{year}",
                    year=year,
                    domains=selected_domains,
                    canary_per_source=canary_per_source,
                )
                inventory["source_counts"] = dict(Counter(inventory.get("source_counts") or {}) + Counter(dblp_inventory.get("source_counts") or {}))
                errors = []
                if manifest.exists():
                    raw = _safe_payload(manifest)
                    errors = [str(error) for value in raw.get("domains", {}).values() for error in value.get("errors", [])]
                coverage = _coverage(batch, year, inventory, [*errors, *dblp.get("errors", [])])
                value = {"worker": worker, "inventory": inventory, "dblp": dblp, "coverage": coverage}
                if canary_per_source and canary_batch is None:
                    canary_batch = prepare_canary_batch(
                        batch,
                        year=year,
                        domains=selected_domains,
                        per_source=canary_per_source,
                    )
                    result["execution_batch"] = str(canary_batch)
                    result["canary_isolated"] = True
            elif current == "official-verify":
                value = _run_worker(
                    "verify_fresh_official_landings.py",
                    ["--batch-root", str(active_batch), "--domains", *selected_domains, "--year", str(year), "--workers", str(paper_workers), "--domain-workers", str(domain_workers), "--delay", "0.25"],
                )
            elif current == "arxiv-match":
                value = match_arxiv_candidates(
                    active_batch,
                    year=year,
                    domains=selected_domains,
                    domain_workers=domain_workers,
                    direct_tail_per_domain=10 if canary_per_source else LIVE_TITLE_TAIL_PER_DOMAIN,
                    direct_batch_size=20,
                )
            elif current == "asset-bind":
                def run_assets() -> dict[str, Any]:
                    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="asset-tier") as pool:
                        source_future = pool.submit(
                            _run_worker,
                            "probe_fresh_arxiv_sources.py",
                            ["--batch-root", str(active_batch), "--domains", *selected_domains, "--year", str(year), "--workers", str(paper_workers), "--domain-workers", str(domain_workers), "--checkpoint-every", "25", "--delay", "0.2"],
                        )
                        official_future = pool.submit(
                            bind_official_assets,
                            active_batch,
                            year=year,
                            domains=selected_domains,
                            domain_workers=domain_workers,
                            paper_workers=paper_workers,
                            limit_per_domain=0,
                            coordinate_source_probe=True,
                        )
                        source_result = source_future.result()
                        official_result = official_future.result()
                    # Re-read the completed A checkpoints once.  This pass is
                    # normally local-only and picks up rows deliberately
                    # deferred while the source probe was still running.
                    official_final = bind_official_assets(
                        active_batch,
                        year=year,
                        domains=selected_domains,
                        domain_workers=domain_workers,
                        paper_workers=paper_workers,
                        limit_per_domain=0,
                        coordinate_source_probe=False,
                    )
                    return {"source": source_result, "official_b": official_result, "official_b_final": official_final}

                if stage == "all":
                    _append_event(batch, "fulltext", "started", overlapped_with="asset-bind")
                    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="post-match") as pool:
                        asset_future = pool.submit(run_assets)
                        text_future = pool.submit(
                            _run_worker,
                            "extract_fresh_fulltext.py",
                            ["--batch-root", str(active_batch), "--domains", *selected_domains, "--workers", str(paper_workers), "--domain-workers", str(domain_workers), "--checkpoint-every", "25", "--delay", "0.2"],
                        )
                        value = asset_future.result()
                        text_value = text_future.result()
                    result["stages"]["fulltext"] = text_value
                    fulltext_completed = True
                    _append_event(batch, "fulltext", "completed" if _stage_ok(text_value) else "failed", result=text_value)
                else:
                    value = run_assets()
            elif current == "fulltext":
                value = _run_worker(
                    "extract_fresh_fulltext.py",
                    ["--batch-root", str(active_batch), "--domains", *selected_domains, "--workers", str(paper_workers), "--domain-workers", str(domain_workers), "--checkpoint-every", "25", "--delay", "0.2"],
                )
            else:
                value = freeze_strict_candidates(
                    active_batch,
                    year=year,
                    domains=selected_domains,
                    domain_workers=domain_workers,
                    paper_workers=paper_workers,
                )
            result["stages"][current] = value
            # The strict dashboard's nested `ok` measures collection quotas.
            # Freeze execution succeeds independently of that training target.
            failed = value.get("ok") is not True if current == "freeze" else not _stage_ok(value)
            if current == "asset-bind" and fulltext_completed and not _stage_ok(result["stages"]["fulltext"]):
                failed = True
            if failed:
                result["ok"] = False
                _append_event(batch, current, "failed", result=value)
                break
            _append_event(batch, current, "completed", result=value)
        except Exception as exc:
            result["ok"] = False
            result["stages"][current] = {"ok": False, "error": str(exc)}
            _append_event(batch, current, "failed", error=str(exc))
            break
    write_json(batch / "manifests" / "multisource_collection_report.json", result)
    return result


__all__ = [
    "EXPECTED_SOURCES",
    "STAGES",
    "collect_literature_papers",
    "merge_staged_inventory",
    "prepare_canary_batch",
]
