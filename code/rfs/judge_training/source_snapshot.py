"""Coordinate strict rebuilding, immutable source capture and atomic publication."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from rfs.contracts.source_snapshot import SOURCE_SNAPSHOT_VERSION, snapshot_relative_path
from rfs.utils import write_json
from .literature_freeze import DOMAINS, build_strict_training_view
from .literature_guard import LiteratureBusyError, active_legacy_writers, literature_batch_guard
from .literature_binding_repair import apply_binding_repairs, prepare_binding_repairs
from .source_snapshot_content import builder_provenance, estimate_storage, materialize_snapshot
from .source_snapshot_boundary import collection_boundary
from .source_snapshot_inventory import assert_metadata_unchanged, audit_source_generation
from .source_snapshot_io import canonical_hash, sha256_file, strict_jsonl, write_jsonl
from .source_snapshot_validation import validate_source_snapshot


def _seal(stage: Path, manifest: dict) -> None:
    excluded = {"snapshot_manifest.json", "file_hashes.jsonl", "COMPLETE"}
    hashes = [{"relative_path": p.relative_to(stage).as_posix(), "size_bytes": p.stat().st_size,
               "sha256": sha256_file(p)} for p in sorted(stage.rglob("*")) if p.is_file() and p.name not in excluded]
    write_jsonl(stage / "file_hashes.jsonl", hashes)
    manifest["file_hashes_sha256"] = sha256_file(stage / "file_hashes.jsonl")
    write_json(stage / "snapshot_manifest.json", manifest)


def _verify_publish(stage: Path, out: Path, archive: Path, work: Path, progress) -> dict:
    first = validate_source_snapshot(stage, archive, require_complete=False)
    if not first["ok"]:
        raise ValueError(f"Staging verification failed: {first}")
    if not (stage / "COMPLETE").exists():
        write_json(stage / "validation_report.json", first)
        manifest = json.loads((stage / "snapshot_manifest.json").read_text(encoding="utf-8"))
        _seal(stage, manifest)
    progress({"stage": "isolated_verification"})
    # The verification copy contains no link to the mutable collection.
    with tempfile.TemporaryDirectory(prefix="verification-", dir=work) as temporary:
        isolated = Path(temporary) / "snapshot"
        shutil.copytree(stage, isolated)
        second = validate_source_snapshot(isolated, archive, require_complete=False)
    if not second["ok"] or any(first[key] != second[key] for key in (
        "paper_set_sha256", "asset_set_sha256", "domain_report", "strict_unique_papers", "assets"
    )):
        raise ValueError(f"Independent verification failed: {second}")
    write_json(work / "isolated_verification.json", second)
    if not (stage / "COMPLETE").exists():
        write_json(stage / "COMPLETE", {"manifest_sha256": sha256_file(stage / "snapshot_manifest.json")})
    if out.exists():
        raise FileExistsError(f"Snapshot already exists; never overwrite: {out}")
    # Windows rename fails if destination exists; on POSIX a valid nonempty D0
    # cannot be replaced either. The per-output OS guard serializes publishers.
    os.rename(stage, out)
    final = validate_source_snapshot(out, archive)
    write_json(work / "published_verification.json", final)
    if not final["ok"]:
        raise ValueError(f"Published snapshot failed read-only verification: {final}")
    return {**final, "status": "complete", "snapshot": str(out)}


def _capture(batch: Path, out: Path, archive: Path, project: Path, work: Path,
             workers: int, preflight: bool, repair_bindings: bool, progress,
             tiers: tuple[str, ...] = ()) -> dict:
    with literature_batch_guard(batch, purpose="freeze-source"):
        writers = active_legacy_writers(batch)
        if writers:
            return {"ok": False, "status": "waiting_for_batch", "retryable": True, "writers": writers}
        if (batch / "manifests/source_binding_repair_pending.json").exists() and (preflight or not repair_bindings):
            return {"ok": False, "status": "waiting_for_binding_repair_recovery", "retryable": True,
                    "next_action": "freeze-source --repair-source-bindings without --preflight"}
        events = collection_boundary(batch)
        if not events["ready"]:
            return {"ok": False, "status": "waiting_for_batch", "retryable": True, "boundary": events}
        boundary = {"captured_at": datetime.now(timezone.utc).isoformat(),
                    "coordination": "exclusive_batch_guard_and_no_legacy_writers", "legacy_writers": [], "events": events}
        versions, code = builder_provenance(project)
        if repair_bindings:
            if not preflight:
                apply_binding_repairs(batch)  # Recover any interrupted ledger transaction first.
            plan = prepare_binding_repairs(batch, workers=workers, progress=progress)
            write_json(work / "binding_repair_plan.json", {k: plan[k] for k in ("changes", "errors")})
            if preflight:
                return {"ok": not plan["errors"], "status": "binding_repair_preflight", "proposed_repairs": len(plan["changes"]), "unresolved": plan["errors"]}
            repair = apply_binding_repairs(batch, plan)
            write_json(work / "binding_repair_result.json", repair)
        # Fail closed before invoking the legacy reader that tolerates bad lines.
        for domain in DOMAINS:
            for row in strict_jsonl(batch / "verified" / domain / "papers.jsonl"):
                if not row.get("paper_id"):
                    raise ValueError(f"Missing paper ID in {domain} ledger")
        progress({"stage": "strict_rebuild"})
        build_strict_training_view(batch)
        progress({"stage": "audit"})
        audit = audit_source_generation(batch, workers=workers, progress=progress, tiers=tiers)
        write_json(work / "audit_report.json", {k: audit[k] for k in (
            "ok", "errors", "domain_report", "input_files", "input_set_sha256", "paper_set_sha256", "scope"
        )})
        if not audit["ok"]:
            return {"ok": False, "status": "source_audit_failed", "errors": audit["errors"],
                    "report": str(work / "audit_report.json")}
        storage = estimate_storage(audit, archive, out.parent)
        write_json(work / "storage_estimate.json", storage)
        if not storage["ok"]:
            return {"ok": False, "status": "insufficient_disk", "storage": storage}
        if preflight:
            return {"ok": True, "status": "preflight_passed", "storage": storage, "domain_report": audit["domain_report"]}
        request = {"strict_rows_sha256": canonical_hash(audit["papers"]), "runtime_set_sha256": versions["runtime_set_sha256"],
                   "asset_set_sha256": canonical_hash(audit["assets"]), "snapshot_id": out.name}
        stage = out.parent / ("." + out.name + ".staging-" + canonical_hash(request)[:20])
        stage.mkdir(exist_ok=True)
        # An interrupted capture may reuse verified copies; COMPLETE staging is
        # already sealed and must not be edited during recovery.
        if (stage / "COMPLETE").exists():
            existing = validate_source_snapshot(stage, archive)
            if not existing["ok"]:
                raise ValueError(f"Invalid completed staging: {existing}")
        else:
            materialize_snapshot(stage, archive, audit, versions, code, progress)
            assert_metadata_unchanged(batch, audit["input_files"])
            current_versions, _ = builder_provenance(project)
            if current_versions["runtime_set_sha256"] != versions["runtime_set_sha256"]:
                raise RuntimeError("Builder code or rules changed during capture; retry on a stable version")
            manifest = {"schema_version": SOURCE_SNAPSHOT_VERSION, "snapshot_id": out.name,
                "snapshot_integrity_ok": True, "training_readiness": "source_only",
                "source_split_is_authoritative": False, "quota_met": audit["domain_report"]["quota_met"],
                "strict_unique_papers": len(audit["papers"]), "scope": audit["scope"],
                "paper_set_sha256": audit["paper_set_sha256"],
                "input_set_sha256": audit["input_set_sha256"], "source_inputs": audit["input_files"],
                "builder_git_commit": versions["builder_git_commit"], "runtime_set_sha256": versions["runtime_set_sha256"],
                "batch_boundary": boundary, "archive_contract": "explicit_root/sha256/<prefix>/<sha256>",
                "provenance": {"source_batch": str(batch)}, "semantic_review_status": "unreviewed"}
            write_json(stage / "validation_report.json", {"status": "pending_independent_verification"})
            _seal(stage, manifest)
    # All content is now independent: release collection while verifying D0.
    return _verify_publish(stage, out, archive, work, progress)


def freeze_source_snapshot(batch_root: str | Path, out: str | Path, archive_root: str | Path,
                           *, workers: int = 4, preflight: bool = False, repair_bindings: bool = False,
                           project_root: str | Path | None = None,
                           tiers: tuple[str, ...] = ()) -> dict:
    """Never overwrite a published version; preserve reports and resumable staging."""
    batch, out, archive = Path(batch_root).resolve(), Path(out).resolve(), Path(archive_root).resolve()
    snapshot_relative_path(out.name)
    project = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    if out.is_relative_to(batch) or batch.is_relative_to(out) or archive.is_relative_to(out) or out.is_relative_to(archive):
        raise ValueError("Snapshot, source batch and archive must have independent roots")
    work = out.parent / "_work" / out.name
    work.mkdir(parents=True, exist_ok=True)
    def progress(value):
        write_json(work / "freezer_progress.json", {"updated_at": datetime.now(timezone.utc).isoformat(), **value})
    try:
        with literature_batch_guard(work, purpose="source-snapshot-publisher"):
            if out.exists():
                existing = validate_source_snapshot(out, archive)
                if not existing["ok"]:
                    raise ValueError(f"Existing destination is not a valid completed snapshot: {existing}")
                manifest = json.loads((out / "snapshot_manifest.json").read_text(encoding="utf-8"))
                if manifest["snapshot_id"] != out.name or manifest["provenance"]["source_batch"] != str(batch):
                    raise ValueError("Existing version belongs to another request; use a new snapshot ID")
                result = {**existing, "status": "already_complete", "snapshot": str(out)}
            else:
                result = _capture(batch, out, archive, project, work, workers, preflight, repair_bindings,
                                  progress, tiers)
            progress(result)
            return result
    except LiteratureBusyError as exc:
        return {"ok": False, "status": "waiting_for_batch", "retryable": True, "error": str(exc)}
    except Exception as exc:
        result = {"ok": False, "status": "failed", "error": str(exc)}
        write_json(work / ("failure-" + uuid.uuid4().hex + ".json"), result)
        progress(result)
        return result
