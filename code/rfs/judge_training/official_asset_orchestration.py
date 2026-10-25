"""Resumable host-aware orchestration for official standalone figure assets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterable
from urllib.parse import urlparse

import requests

from rfs.providers.literature import LiteratureHttpClient
from rfs.providers.rate_limit import rate_limit_snapshot
from rfs.utils import write_json

from .literature_freeze import ACCEPTED_TYPES, DOMAINS
from .literature_io import atomic_write_text


TERMINAL_STATUSES = {
    "verified",
    "no_eligible_standalone_figure",
    "terminal_blocked",
    "terminal_not_found",
    "terminal_invalid_document",
    "terminal_retry_exhausted",
}
RETRYABLE_STATUSES = {
    "probe_failed",
    "retryable_failure",
    "retryable_rate_limited",
    "retryable_circuit_open",
}
MAX_RETRY_ATTEMPTS = 3
HIGH_CAPACITY_HOSTS = (
    "openaccess.thecvf.com",
    "www.ebi.ac.uk",
    "ncbi.nlm.nih.gov",
    "europepmc.org",
)
# Publisher sites answer 403 to bursts long before they answer 429, so a
# refused request looks terminal and silently drops an otherwise open-access
# paper. Crawl them single-file and well under one request per second.
POLITE_HOSTS = (
    "nature.com",
    "media.springernature.com",
    "link.springer.com",
    "journals.plos.org",
    "onlinelibrary.wiley.com",
    "pubs.acs.org",
    "dl.acm.org",
    # A redirector, but the load lands on whichever publisher it resolves to,
    # so it must be paced as if it were one.
    "doi.org",
    # A JavaScript shell that answers 403 to scripted clients; pacing it keeps
    # a last-resort route from spending the host's goodwill.
    "europepmc.org",
)
POLITE_MIN_INTERVAL = 1.5
_DASHBOARD_LOCK = threading.Lock()


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _identity(row: dict[str, Any]) -> str:
    for field in ("doi", "paper_id", "official_paper_id", "pmcid", "pmid", "arxiv_id"):
        if value := _clean(row.get(field)):
            return value.casefold()
    return " ".join(part for part in _clean(row.get("title")).casefold().split() if part)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            output.append(value)
    return output


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _host_matches(host: str, values: Iterable[str]) -> bool:
    return any(host == value or host.endswith(f".{value}") for value in values)


def _host_capacity(host: str) -> int:
    if _host_matches(host, POLITE_HOSTS):
        return 1
    return 4 if _host_matches(host, HIGH_CAPACITY_HOSTS) else 2


def _host_min_interval(host: str) -> float:
    return POLITE_MIN_INTERVAL if _host_matches(host, POLITE_HOSTS) else 0.2


class OfficialAssetClientPool:
    """Reuse one requests session per worker and host while sharing host lanes."""

    def __init__(self) -> None:
        self.local = threading.local()

    def for_url(self, url: str) -> LiteratureHttpClient:
        host = urlparse(url).netloc.casefold() or "unknown"
        clients = getattr(self.local, "clients", None)
        if clients is None:
            clients = {}
            self.local.clients = clients
        if host not in clients:
            session = requests.Session()
            clients[host] = LiteratureHttpClient(
                f"official_asset:{host}",
                min_interval=_host_min_interval(host),
                retries=1,
                timeout=15,
                session=session,
                max_in_flight=_host_capacity(host),
                circuit_breaker_threshold=5,
                circuit_breaker_cooldown=300,
            )
        return clients[host]


def _source_state(batch: Path, domain: str) -> tuple[set[str], bool, float]:
    path = batch / "manifests" / f"source_probe_{domain}.json"
    if not path.exists():
        return set(), False, 0.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set(), False, path.stat().st_mtime
    section = (payload.get("domains") or {}).get(domain) or {}
    bound: set[str] = set()
    for row in section.get("rows") or []:
        arxiv_id = _clean(row.get("arxiv_id"))
        if not arxiv_id or not row.get("framework_source_bound"):
            continue
        archive = batch / "assets" / domain / arxiv_id / "arxiv_source.src"
        extracted = [batch / str(value) for value in row.get("extracted_figure_paths") or []]
        if archive.is_file() and (not extracted or any(value.is_file() for value in extracted)):
            bound.add(arxiv_id)
    return bound, bool(section.get("complete")), path.stat().st_mtime


def _retry_due(row: dict[str, Any]) -> bool:
    if str(row.get("status") or "") not in RETRYABLE_STATUSES:
        return False
    if int(row.get("attempt_count") or 0) >= MAX_RETRY_ATTEMPTS:
        return False
    value = _clean(row.get("next_retry_at"))
    if not value:
        return True
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")) <= datetime.now(timezone.utc)
    except ValueError:
        return True


def _annotate_attempt(item: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, Any]:
    output = dict(item)
    previous_attempts = int((previous or {}).get("attempt_count") or 0)
    status = str(output.get("status") or "")
    # A circuit-open result means no HTTP request was made, and a rate-limited
    # one means the host asked us to slow down rather than refusing the paper.
    # Neither says anything about the record, so neither may consume the
    # bounded network-attempt budget: three polite-host 429s in a row would
    # otherwise permanently terminalize a perfectly reachable paper.
    unbudgeted = {"retryable_circuit_open", "retryable_rate_limited"}
    output["attempt_count"] = previous_attempts if status in unbudgeted else previous_attempts + 1
    output["checked_at"] = datetime.now(timezone.utc).isoformat()
    output["host"] = urlparse(str(output.get("official_url") or "")).netloc.casefold()
    if status in RETRYABLE_STATUSES and output["attempt_count"] >= MAX_RETRY_ATTEMPTS:
        output["retry_exhausted_status"] = status
        output["status"] = "terminal_retry_exhausted"
        output.pop("next_retry_at", None)
    elif status in RETRYABLE_STATUSES:
        output["next_retry_at"] = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    else:
        output.pop("next_retry_at", None)
    return output


def _terminalize_exhausted(row: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy exhausted retry rows so they cannot remain pending forever."""
    if (
        str(row.get("status") or "") in RETRYABLE_STATUSES
        and int(row.get("attempt_count") or 0) >= MAX_RETRY_ATTEMPTS
    ):
        output = dict(row)
        output["retry_exhausted_status"] = output["status"]
        output["status"] = "terminal_retry_exhausted"
        output.pop("next_retry_at", None)
        return output
    return row


def _pending_count(
    candidates: Iterable[dict[str, Any]],
    previous: dict[str, dict[str, Any]],
    excluded_a: set[str],
) -> int:
    remaining = 0
    for row in candidates:
        record_id = _identity(row)
        if record_id in excluded_a:
            continue
        old = previous.get(record_id)
        if old and str(old.get("status") or "") in TERMINAL_STATUSES:
            continue
        remaining += 1
    return remaining


def _publish_dashboard(batch: Path, domain_reports: dict[str, dict[str, Any]], started: float) -> None:
    elapsed = max(1.0, time.monotonic() - started)
    processed = sum(int(value.get("processed_this_run") or 0) for value in domain_reports.values())
    remaining = sum(int(value.get("pending") or 0) for value in domain_reports.values())
    throughput = processed / elapsed * 60.0
    runtime = {
        "version": "official-asset-runtime-v003",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "throughput_per_minute": round(throughput, 2),
        "estimated_remaining_minutes": round(remaining / throughput, 1) if throughput else None,
        "domains": domain_reports,
        "host_lanes": rate_limit_snapshot("literature:official_asset:"),
    }
    with _DASHBOARD_LOCK:
        write_json(batch / "manifests" / "official_asset_dashboard.json", runtime)
        strict_path = batch / "manifests" / "strict_training_dashboard.json"
        if strict_path.exists():
            try:
                strict = json.loads(strict_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                strict = {}
            strict["official_asset_runtime"] = runtime
            write_json(strict_path, strict)


def orchestrate_official_assets(
    batch_root: str | Path,
    *,
    year: int = 2026,
    domains: Iterable[str] = DOMAINS,
    domain_workers: int = 5,
    paper_workers: int = 16,
    limit_per_domain: int = 0,
    coordinate_source_probe: bool = False,
) -> dict[str, Any]:
    """Probe B assets in bounded batches while dynamically excluding valid A rows."""
    from .official_assets import bind_official_asset

    batch = Path(batch_root).resolve()
    selected_domains = [value for value in dict.fromkeys(domains) if value in DOMAINS]
    started = time.monotonic()
    started_wall = time.time()
    reports: dict[str, dict[str, Any]] = {}
    reports_lock = threading.Lock()

    def run_domain(domain: str) -> tuple[str, dict[str, Any]]:
        candidate_path = batch / "candidates" / domain / "papers_arxiv_matched.jsonl"
        if not candidate_path.exists():
            candidate_path = batch / "candidates" / domain / "papers.jsonl"
        candidates = [
            row
            for row in _jsonl(candidate_path)
            if int(row.get("year") or 0) == year
            and row.get("official_record_status") == "verified"
            and _clean(row.get("paper_type")).casefold() in ACCEPTED_TYPES[domain]
        ]
        candidates.sort(key=lambda row: (bool(row.get("arxiv_id")), _clean(row.get("publication_date"))))
        if limit_per_domain:
            candidates = candidates[:limit_per_domain]
        output_path = batch / "manifests" / f"official_asset_rows_{domain}.jsonl"
        previous = {
            str(row.get("record_id")): _terminalize_exhausted(row)
            for row in _jsonl(output_path)
            if row.get("record_id")
        }
        client_pool = OfficialAssetClientPool()
        processed_this_run = 0
        excluded_a: set[str] = set()
        deferred_for_a = 0
        unexpected = 0
        position = 0
        chunk_size = 100
        with reports_lock:
            reports[domain] = {
                "eligible": len(candidates),
                "checked": len(previous),
                "processed_this_run": 0,
                "verified_b": sum(row.get("figure_asset_tier") == "B_official_standalone" for row in previous.values()),
                "excluded_a": 0,
                "deferred_for_a": 0,
                "retry_count": sum(str(row.get("status") or "") in RETRYABLE_STATUSES for row in previous.values()),
                "unexpected_failures": 0,
                "pending": _pending_count(candidates, previous, excluded_a),
                "complete": False,
            }

        with ThreadPoolExecutor(
            max_workers=max(1, paper_workers),
            thread_name_prefix=f"official-{domain}",
        ) as pool:
            while position < len(candidates):
                chunk = candidates[position : position + chunk_size]
                position += len(chunk)
                bound_ids, source_complete, source_mtime = _source_state(batch, domain)
                runnable: list[dict[str, Any]] = []
                for row in chunk:
                    record_id = _identity(row)
                    old = previous.get(record_id)
                    arxiv_id = _clean(row.get("arxiv_id"))
                    if arxiv_id and arxiv_id in bound_ids:
                        excluded_a.add(record_id)
                        continue
                    if old and (str(old.get("status") or "") in TERMINAL_STATUSES or not _retry_due(old)):
                        continue
                    if coordinate_source_probe and arxiv_id and not (
                        source_complete and source_mtime >= started_wall
                    ):
                        deferred_for_a += 1
                        continue
                    runnable.append(row)
                futures = {pool.submit(bind_official_asset, batch, domain, row, client=client_pool): row for row in runnable}
                for future in as_completed(futures):
                    row = futures[future]
                    record_id = _identity(row)
                    old = previous.get(record_id)
                    try:
                        item = future.result()
                    except Exception as exc:
                        item = {
                            "record_id": record_id,
                            "title": row.get("title"),
                            "official_url": row.get("official_url"),
                            "status": "retryable_failure",
                            "probe_error": str(exc)[:500],
                        }
                        unexpected += 1
                    previous[record_id] = _annotate_attempt(item, old)
                    processed_this_run += 1
                _write_rows(output_path, previous.values())
                retry_rows = [row for row in previous.values() if str(row.get("status") or "") in RETRYABLE_STATUSES]
                _write_rows(batch / "retry" / f"official_asset_{domain}.jsonl", retry_rows)
                summary = {
                    "eligible": len(candidates),
                    "checked": len(previous),
                    "processed_this_run": processed_this_run,
                    "verified_b": sum(row.get("figure_asset_tier") == "B_official_standalone" for row in previous.values()),
                    "excluded_a": len(excluded_a),
                    "deferred_for_a": deferred_for_a,
                    "retry_count": len(retry_rows),
                    "unexpected_failures": unexpected,
                    "pending": _pending_count(candidates, previous, excluded_a),
                    "complete": False,
                }
                with reports_lock:
                    reports[domain] = summary
                    _publish_dashboard(batch, dict(reports), started)

        final = dict(reports.get(domain) or {})
        final["complete"] = final.get("pending", 0) == 0 and not deferred_for_a
        with reports_lock:
            reports[domain] = final
            _publish_dashboard(batch, dict(reports), started)
        return domain, final

    with ThreadPoolExecutor(
        max_workers=max(1, min(domain_workers, len(selected_domains) or 1)),
        thread_name_prefix="official-domain",
    ) as pool:
        results = dict(pool.map(run_domain, selected_domains))
    report = {
        "version": "official-standalone-assets-v003",
        "domains": results,
        "host_lanes": rate_limit_snapshot("literature:official_asset:"),
        "ok": True,
    }
    _publish_dashboard(batch, results, started)
    return report


__all__ = ["OfficialAssetClientPool", "orchestrate_official_assets"]
