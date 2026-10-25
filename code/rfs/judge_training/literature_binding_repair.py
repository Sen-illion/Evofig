"""Repair overbroad A-figure lists using exact source evidence and resumable journals."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from rfs.utils import write_json
from .figure_rendering import rendered_page_asset
from .literature_freeze import (DOMAINS, _artifact_root, _clean, _existing_path, _figure_paths,
                                infer_asset_tier, strict_failures)
from .literature_io import atomic_write_text
from .source_snapshot_bindings import recover_source_member, source_member_name, verify_source_members
from .source_snapshot_io import canonical_hash, contained_path, copy_verified, sha256_file, stable_bytes, strict_jsonl, write_jsonl


def propose_binding_repair(batch: Path, domain: str, row: dict) -> dict | None:
    """Select or recover exact bytes from the record's declared source files.

    This cannot select a new scientific figure or merge conflicting captions.
    Unmatched files remain on disk and in the journal; no paper is discarded.
    """
    failures = strict_failures(batch, domain, row)
    if infer_asset_tier(row) != "A_source_verified" or set(failures) - {"figure_file_or_caption_unbound"}:
        return None
    figure = row["figure"]
    if not figure.get("caption"):
        return None
    names = [figure.get("tex_file"), *(figure.get("source_files") or [])]
    root = _artifact_root(batch, row).resolve()
    if not root.is_relative_to(batch.parent.resolve()):
        raise ValueError("Source repair root is outside the declared collection")
    archive = _existing_path(batch, row, row.get("arxiv_source_path"))
    if archive is None:
        candidate = root / "assets" / domain / str(row.get("arxiv_id") or "") / "arxiv_source.src"
        archive = candidate if candidate.is_file() else None
    originals = {name: p for name in names if name and (p := _existing_path(batch, row, name))}
    for path in [*originals.values(), *([archive] if archive else [])]:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Source repair input is outside its artifact root")
    binding = verify_source_members(archive, originals, names, set())
    allowed = set(binding["members"].values())
    # A declared page render is derived from the record's own PDF, not copied from
    # the source package, so its bytes can never occur among the declared members.
    # It is bound by its own evidence and must survive the narrowing below;
    # dropping it would strip the record of its only decodable figure.
    render = rendered_page_asset(row)
    selected, unselected, selected_hashes = [], [], set()
    for value in _figure_paths(row):
        path = _existing_path(batch, row, value)
        digest = sha256_file(path) if path else None
        if (render and _clean(value) == render) or digest in allowed:
            selected.append(value)
            if digest:
                selected_hashes.add(digest)
        else:
            unselected.append(value)
    recoveries = []
    for name in figure.get("source_files") or []:
        suffix = Path(name).suffix.lower()
        digest = binding["members"][source_member_name(name)]
        if digest in selected_hashes:
            continue
        target = root / "assets" / domain / str(row.get("arxiv_id") or "source-recovery") / "figures" / ("source_bound_" + digest[:24] + suffix)
        if not target.resolve().is_relative_to(root):
            raise ValueError("Recovered source asset path escapes its root")
        relative = target.relative_to(root).as_posix()
        recoveries.append({"source_name": name, "source_archive": str(archive) if archive else None,
                           "source_original": str(originals.get(name)) if name in originals else None,
                           "target": str(target), "sha256": digest})
        selected.append(relative)
        selected_hashes.add(digest)
    if not unselected and not recoveries:
        return None
    if not selected:
        raise ValueError("No selected asset matches declared source; requires re-extraction")
    return {"paper_id": row["paper_id"], "domain": domain, "record_sha256": canonical_hash(row),
            "previous_image_paths": _figure_paths(row), "selected_image_paths": selected,
            "unselected_image_paths": unselected, "source_members": binding["members"],
            "recovered_source_assets": recoveries,
            "method": "exact_declared_source_member_sha256", "semantic_review_status": "unreviewed"}


def _recover_assets(batch: Path, changes: list[dict]) -> None:
    for change in changes:
        for asset in change.get("recovered_source_assets", []):
            target = Path(asset["target"]).resolve()
            if not target.is_relative_to(batch.parent.resolve()):
                raise ValueError("Recovery target is outside collection roots")
            if asset["source_archive"]:
                source = Path(asset["source_archive"]).resolve()
                if not source.is_relative_to(batch.parent.resolve()):
                    raise ValueError("Recovery archive is outside collection roots")
                recover_source_member(source, asset["source_name"], target, asset["sha256"])
            else:
                source = Path(asset["source_original"]).resolve()
                if not source.is_relative_to(batch.parent.resolve()):
                    raise ValueError("Recovery original is outside collection roots")
                copy_verified(source, target, asset["sha256"])


def prepare_binding_repairs(batch_root: str | Path, *, workers: int = 4, progress=None) -> dict:
    """Read all five ledgers strictly; the caller must hold the collection guard."""
    batch = Path(batch_root).resolve()
    ledgers, jobs = {}, []
    for domain in DOMAINS:
        path = batch / "verified" / domain / "papers.jsonl"
        raw = stable_bytes(path)
        rows = strict_jsonl(path)
        ledgers[domain] = {"relative_path": path.relative_to(batch).as_posix(),
                           "sha256": hashlib.sha256(raw).hexdigest(), "rows": rows, "raw": raw}
        jobs.extend((domain, index, row) for index, row in enumerate(rows))
    changes, errors = [], []
    def inspect(job):
        domain, index, row = job
        try:
            proposal = propose_binding_repair(batch, domain, row)
            return {**proposal, "row_index": index} if proposal else None, None
        except Exception as exc:
            return None, {"paper_id": row.get("paper_id"), "domain": domain, "error": str(exc)}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, (change, error) in enumerate(pool.map(inspect, jobs), 1):
            if change:
                changes.append(change)
            if error:
                errors.append(error)
            if progress and (index % 100 == 0 or index == len(jobs)):
                progress({"stage": "binding_repair_plan", "checked": index, "total": len(jobs),
                          "proposals": len(changes), "unresolved": len(errors)})
    for ledger in ledgers.values():
        if sha256_file(batch / ledger["relative_path"]) != ledger["sha256"]:
            raise RuntimeError("Source ledger changed while preparing binding repairs")
    return {"changes": changes, "errors": errors, "ledgers": ledgers}


def _resume(batch: Path, pending: Path) -> dict:
    journal = json.loads(pending.read_text(encoding="utf-8"))
    # Verify the whole transaction before changing any remaining ledger.
    for item in journal["files"]:
        actual = sha256_file(contained_path(batch, item["relative_path"]))
        if actual not in {item["before_sha256"], item["after_sha256"]}:
            raise ValueError(f"Binding repair journal conflicts with current ledger: {item['relative_path']}")
        for key, digest in (("backup", item["before_sha256"]), ("replacement", item["after_sha256"])):
            if sha256_file(contained_path(batch, item[key])) != digest:
                raise ValueError("Binding repair backup/replacement was changed")
    _recover_assets(batch, journal["changes"])
    for item in journal["files"]:
        target = contained_path(batch, item["relative_path"])
        if sha256_file(target) == item["before_sha256"]:
            raw = contained_path(batch, item["replacement"]).read_bytes().decode("utf-8")
            atomic_write_text(target, raw)
        if sha256_file(target) != item["after_sha256"]:
            raise RuntimeError("Binding repair replacement did not preserve its exact bytes")
    journal["status"] = "complete"
    write_json(contained_path(batch, journal["report_path"]), journal)
    pending.unlink()
    return journal


def apply_binding_repairs(batch_root: str | Path, plan: dict | None = None) -> dict:
    """Apply/recover only deterministic proposals, with exact originals retained."""
    batch = Path(batch_root).resolve()
    pending = batch / "manifests/source_binding_repair_pending.json"
    if pending.exists():
        return _resume(batch, pending)
    if plan is None or not plan["changes"]:
        return {"status": "unchanged", "repaired_records": 0}
    identity = canonical_hash(plan["changes"])
    root = batch / "manifests/source_binding_repairs" / identity
    root.mkdir(parents=True, exist_ok=True)
    files = []
    for domain, ledger in plan["ledgers"].items():
        changes = [c for c in plan["changes"] if c["domain"] == domain]
        if not changes:
            continue
        if sha256_file(batch / ledger["relative_path"]) != ledger["sha256"]:
            raise RuntimeError("Ledger changed before binding repair")
        rows = [dict(r) for r in ledger["rows"]]
        for change in changes:
            row = rows[change["row_index"]]
            if canonical_hash(row) != change["record_sha256"]:
                raise ValueError("Binding repair record identity changed")
            row["figure"] = {**row["figure"], "image_paths": change["selected_image_paths"]}
            row["source_binding_repair"] = change
        before, after = root / f"{domain}.before.jsonl", root / f"{domain}.after.jsonl"
        if before.exists() and sha256_file(before) != ledger["sha256"]:
            raise ValueError("Existing binding repair backup conflicts")
        before.write_bytes(ledger["raw"])
        write_jsonl(after, rows)
        files.append({"relative_path": ledger["relative_path"], "before_sha256": ledger["sha256"],
                      "after_sha256": sha256_file(after), "backup": before.relative_to(batch).as_posix(),
                      "replacement": after.relative_to(batch).as_posix()})
    journal = {"status": "prepared", "repaired_records": len(plan["changes"]), "files": files,
               "changes": plan["changes"], "unresolved": plan["errors"],
               "report_path": (root / "report.json").relative_to(batch).as_posix()}
    atomic_write_text(pending, json.dumps(journal, ensure_ascii=False, indent=2))
    return _resume(batch, pending)
