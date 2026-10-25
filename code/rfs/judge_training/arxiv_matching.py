"""Cache-first exact arXiv matching for strict literature candidates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

from rfs.providers.arxiv_metadata import ArxivMetadataIndex, ArxivTitleResolver, batch_match_exact
from rfs.providers.literature import ArxivProvider, ProviderRequestError, normalized_title
from rfs.utils import write_json

from .literature_io import atomic_write_text


ARXIV_FIELDS = (
    "arxiv_id",
    "arxiv_version",
    "arxiv_url",
    "pdf_url",
    "source_url",
    "journal_ref",
)
DIRECT_TAIL_DOMAINS = frozenset({"agent", "rag", "cv"})
# Live title resolution falls back to public metadata services that rate limit in
# bursts (measured 2026-09-25: runs of a dozen refusals, then normal service again).
# A storm is ridden out with a cooling pause instead of ending the pass; only an
# unbroken run this long means no channel is usable at all.
RESOLVER_FAILURE_LIMIT = 100
RESOLVER_COOLDOWN_AFTER = 5
RESOLVER_COOLDOWN_SECONDS = 45
RESOLVER_MAX_COOLDOWN_SECONDS = 300


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


def _identity_keys(row: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for field in ("doi", "paper_id", "official_paper_id", "arxiv_id"):
        value = str(row.get(field) or "").strip().casefold()
        if value:
            values.append(f"{field}:{value}")
    title = normalized_title(row.get("title"))
    if title:
        values.append(f"title:{title}")
    return tuple(values)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _cache_paths(batch: Path) -> list[Path]:
    manifests = batch / "manifests"
    values = list(manifests.glob("arxiv_harvest_*.json"))
    values.extend(
        path
        for path in manifests.glob("openalex_arxiv_*exact.json")
        if not path.name.endswith("_manifest.json")
    )
    return sorted(set(values), key=lambda path: path.name)


def load_cached_arxiv_metadata(batch: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all durable arXiv metadata caches once and describe their provenance."""
    root = Path(batch)
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in _cache_paths(root):
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            sources.append({"path": str(path), "status": "failed"})
            continue
        rows = [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []
        records.extend(rows)
        sources.append(
            {
                "path": str(path),
                "status": "complete",
                "records": len(rows),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return records, sources


def _published_by_now(row: dict[str, Any], year: int) -> bool:
    if int(row.get("year") or 0) != year:
        return False
    value = str(row.get("publication_date") or "").strip()[:10]
    if not value:
        return True
    try:
        return date.fromisoformat(value) <= date.today()
    except ValueError:
        return True


def _copy_match(item: dict[str, Any], record: dict[str, Any], method: str) -> None:
    item.update(
        {
            "arxiv_match_status": "matched",
            "arxiv_match_score": 1.0,
            "arxiv_match_method": method,
        }
    )
    item.update({field: record.get(field) for field in ARXIV_FIELDS if record.get(field) is not None})


def _durable_matches(batch: Path, domain: str, output_path: Path) -> dict[str, dict[str, Any]]:
    prior: dict[str, dict[str, Any]] = {}
    for path in (output_path, batch / "verified" / domain / "papers.jsonl"):
        for row in _jsonl(path):
            if row.get("arxiv_match_status") != "matched" or not row.get("arxiv_id"):
                continue
            for key in _identity_keys(row):
                prior.setdefault(key, row)
    return prior


def _match_domain(
    batch: Path,
    domain: str,
    year: int,
    index: ArxivMetadataIndex,
    *,
    direct_tail_limit: int,
    direct_batch_size: int,
) -> tuple[str, dict[str, Any]]:
    candidate_path = batch / "candidates" / domain / "papers.jsonl"
    output_path = batch / "candidates" / domain / "papers_arxiv_matched.jsonl"
    candidates = [row for row in _jsonl(candidate_path) if int(row.get("year") or 0) == year]
    preserved = [row for row in _jsonl(output_path) if int(row.get("year") or 0) != year]
    prior = _durable_matches(batch, domain, output_path)
    output: list[dict[str, Any]] = []
    unresolved: list[int] = []
    methods: dict[str, int] = {}

    for candidate in candidates:
        item = dict(candidate)
        previous = next((prior[key] for key in _identity_keys(candidate) if key in prior), None)
        if previous:
            _copy_match(item, previous, "durable_checkpoint")
        else:
            record, method = index.match(title=item.get("title"), doi=item.get("doi"))
            if record:
                _copy_match(item, record, method)
            else:
                item.update(
                    {
                        "arxiv_match_status": "unmatched",
                        "arxiv_match_score": 0.0,
                        "arxiv_match_method": method,
                    }
                )
                unresolved.append(len(output))
        method_name = str(item.get("arxiv_match_method") or "unknown")
        methods[method_name] = methods.get(method_name, 0) + 1
        output.append(item)

    _write_jsonl(output_path, [*preserved, *output])

    direct_attempted = 0
    direct_matched = 0
    direct_error: str | None = None
    live_attempted = 0
    live_matched = 0
    live_methods: dict[str, int] = {}
    live_blocked = False
    live_cooldowns = 0
    if domain in DIRECT_TAIL_DOMAINS and direct_tail_limit > 0:
        eligible = [
            index_value
            for index_value in unresolved
            if output[index_value].get("official_record_status") == "verified"
            and _published_by_now(output[index_value], year)
        ]
        eligible.sort(
            key=lambda index_value: str(output[index_value].get("publication_date") or ""),
            reverse=True,
        )
        eligible = eligible[:direct_tail_limit]
        provider = ArxivProvider()
        resolver: ArxivTitleResolver | None = None
        size = max(1, min(int(direct_batch_size), 20))
        api_blocked = False
        resolver_failures = 0
        for offset in range(0, len(eligible), size):
            positions = eligible[offset : offset + size]
            titles = [str(output[position].get("title") or "") for position in positions]
            found: dict[str, dict[str, Any]] = {}
            if not api_blocked:
                direct_attempted += len(titles)
                try:
                    found = batch_match_exact(provider, titles, batch_size=size)
                except (ProviderRequestError, OSError, ValueError) as exc:
                    # One refused batch means this host cannot use the export API's
                    # title query at all (every `ti:`/quoted `search_query` answered
                    # HTTP 406 here), so stop asking it and let the fallback
                    # resolver carry the tail instead of abandoning it.
                    api_blocked = True
                    direct_error = str(exc)[:300]
            for position, title in zip(positions, titles):
                record = found.get(normalized_title(title))
                method = "exact_title_direct_batch"
                if record is None and not live_blocked:
                    if resolver is None:
                        resolver = ArxivTitleResolver()
                    live_attempted += 1
                    try:
                        record, method = resolver.resolve(title)
                    except (ProviderRequestError, OSError, ValueError) as exc:
                        resolver_failures += 1
                        if direct_error is None:
                            direct_error = str(exc)[:300]
                        if resolver_failures >= RESOLVER_FAILURE_LIMIT:
                            # No channel has answered for this many titles in a row:
                            # stop instead of grinding through the rest of the pass.
                            live_blocked = True
                            break
                        if resolver_failures % RESOLVER_COOLDOWN_AFTER == 0:
                            cooldown = min(
                                RESOLVER_COOLDOWN_SECONDS * (resolver_failures // RESOLVER_COOLDOWN_AFTER),
                                RESOLVER_MAX_COOLDOWN_SECONDS,
                            )
                            live_cooldowns += 1
                            time.sleep(cooldown)
                        continue
                    resolver_failures = 0
                    live_methods[method] = live_methods.get(method, 0) + 1
                if not record:
                    continue
                old_method = str(output[position].get("arxiv_match_method") or "unknown")
                methods[old_method] = max(0, methods.get(old_method, 0) - 1)
                _copy_match(output[position], record, method)
                methods[method] = methods.get(method, 0) + 1
                if method == "exact_title_direct_batch":
                    direct_matched += 1
                else:
                    live_matched += 1
            _write_jsonl(output_path, [*preserved, *output])
            if live_blocked:
                break

    matched = sum(row.get("arxiv_match_status") == "matched" for row in output)
    return domain, {
        "candidate_count": len(candidates),
        "matched": matched,
        "unmatched": len(candidates) - matched,
        "match_rate": round(matched / max(1, len(candidates)), 4),
        "methods": {key: value for key, value in sorted(methods.items()) if value},
        "direct_attempted": direct_attempted,
        "direct_matched": direct_matched,
        "direct_error": direct_error,
        "live_attempted": live_attempted,
        "live_matched": live_matched,
        "live_methods": {key: value for key, value in sorted(live_methods.items()) if value},
        "live_blocked": live_blocked,
        "live_cooldowns": live_cooldowns,
        "output_path": str(output_path),
    }


def match_arxiv_candidates(
    batch_root: str | Path,
    *,
    year: int,
    domains: Iterable[str],
    domain_workers: int = 5,
    direct_tail_per_domain: int = 40,
    direct_batch_size: int = 20,
) -> dict[str, Any]:
    """Match five domains from one shared cache, then query only a bounded tail."""
    batch = Path(batch_root).resolve()
    cached, sources = load_cached_arxiv_metadata(batch)
    index = ArxivMetadataIndex(cached)
    selected = list(dict.fromkeys(domains))
    result: dict[str, Any] = {
        "version": "cache-first-arxiv-match-v002",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "year": year,
        "cache_input_records": len(cached),
        "cache_unique_arxiv_ids": len(index.records),
        "cache_sources": sources,
        "direct_tail_per_domain": direct_tail_per_domain,
        "domains": {},
        "ok": bool(index.records),
    }
    if not index.records:
        result["error"] = "no_local_arxiv_metadata_cache"
    else:
        with ThreadPoolExecutor(
            max_workers=max(1, min(domain_workers, len(selected))),
            thread_name_prefix="arxiv-cache-domain",
        ) as pool:
            futures = [
                pool.submit(
                    _match_domain,
                    batch,
                    domain,
                    year,
                    index,
                    direct_tail_limit=direct_tail_per_domain,
                    direct_batch_size=direct_batch_size,
                )
                for domain in selected
            ]
            for future in as_completed(futures):
                domain, domain_result = future.result()
                result["domains"][domain] = domain_result
    write_json(batch / "manifests" / "arxiv_cache_snapshot.json", {"sources": sources})
    write_json(batch / "manifests" / "arxiv_match.json", result)
    return result


__all__ = ["load_cached_arxiv_metadata", "match_arxiv_candidates"]
