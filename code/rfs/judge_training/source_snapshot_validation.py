"""Read-only validation using only a snapshot and its declared original archive."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from rfs.contracts.source_snapshot import REQUIRED_SNAPSHOT_FILES, validate_source_manifest
from .figure_rendering import FIGURE_ROLES
from .literature_freeze import DOMAINS
from .source_snapshot_assets import validate_readable_asset
from .source_snapshot_bindings import verify_source_members
from .source_snapshot_io import archive_path, canonical_hash, contained_path, sha256_file, strict_jsonl


def _json(root: Path, name: str):
    return json.loads(contained_path(root, name).read_text(encoding="utf-8"))


def _check_files(root: Path, manifest: dict, require_complete: bool) -> list[dict]:
    if require_complete:
        marker = _json(root, "COMPLETE")
        if marker.get("manifest_sha256") != sha256_file(root / "snapshot_manifest.json"):
            raise ValueError("COMPLETE does not match manifest")
    if sha256_file(root / "file_hashes.jsonl") != manifest["file_hashes_sha256"]:
        raise ValueError("File hash inventory mismatch")
    rows = strict_jsonl(root / "file_hashes.jsonl")
    names = [row["relative_path"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate file hash inventory entry")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in root.rglob("*")):
        raise ValueError("Snapshot must not contain filesystem links")
    controls = {"COMPLETE", "snapshot_manifest.json", "file_hashes.jsonl"}
    if actual - controls != set(names) or not set(REQUIRED_SNAPSHOT_FILES) <= set(names):
        raise ValueError("Missing or unexpected snapshot payload files")
    for row in rows:
        path = contained_path(root, row["relative_path"])
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Payload corruption: {row['relative_path']}")
    return rows


def _check_assets(root: Path, archive: Path, papers: list[dict], assets: list[dict]) -> dict:
    paper_ids = {r["paper_id"] for r in papers}
    asset_ids, roles, by_id, paper_assets = set(), {}, {}, {}
    for row in assets:
        if row["paper_id"] not in paper_ids or row["asset_id"] in asset_ids:
            raise ValueError("Unknown paper or duplicate asset identity")
        asset_ids.add(row["asset_id"])
        by_id[row["asset_id"]] = row
        paper_assets.setdefault(row["paper_id"], set()).add(row["asset_id"])
        roles.setdefault(row["paper_id"], set()).add(row["role"])
        if row["storage"] == "archive":
            if row["relative_path"] != archive_path(row["sha256"]):
                raise ValueError("Archive object is not content addressed")
            path = contained_path(archive, row["relative_path"])
        elif row["storage"] == "snapshot":
            path = contained_path(root, row["relative_path"])
        else:
            raise ValueError("Unknown asset storage")
        # A collection entry and its content-addressed archive object may share
        # one inode: the same bytes legitimately live under both names, and a
        # shared inode cannot be rewritten in place without corrupting the
        # collection as well.  Links stay refused, because a link lets an asset
        # resolve outside the archive or snapshot root that owns it.
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("Assets must be real files, not links")
        if not path.is_file():
            raise ValueError("Assets must be regular files")
        if sha256_file(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
            raise ValueError(f"Asset corruption: {row['asset_id']}")
        validate_readable_asset(path, row["role"], suffix=row["suffix"])
    for paper in papers:
        required = {"full_text", "text_locations", "paper_pdf"}
        if not required <= roles.get(paper["paper_id"], set()):
            raise ValueError(f"Missing paper asset roles: {paper['paper_id']}")
        if not roles.get(paper["paper_id"], set()) & FIGURE_ROLES:
            raise ValueError(f"Missing decodable figure asset: {paper['paper_id']}")
        if paper["figure_asset_tier"] == "A_source_verified" and not roles[paper["paper_id"]] & {"source_archive", "source_original"}:
            raise ValueError("A source original is missing")
        if paper["figure_asset_tier"] == "A_source_verified":
            owned = [by_id[value] for value in paper_assets[paper["paper_id"]]]
            source = next((contained_path(archive, a["relative_path"]) for a in owned if a["role"] == "source_archive"), None)
            figures = [a for a in owned if a["role"] == "figure"]
            if figures:
                binding = figures[0]["source_binding"]
                originals = {name: contained_path(archive, a["relative_path"]) for name, digest in binding["members"].items()
                             for a in owned if a["role"] == "source_original" and a["sha256"] == digest}
                checked = verify_source_members(source, originals, list(binding["members"]), {a["sha256"] for a in figures})
                if checked != binding or any(a["source_binding"] != binding for a in figures):
                    raise ValueError("Source binding evidence mismatch")
        for asset_id in paper["asset_ids"]:
            if asset_id not in by_id or by_id[asset_id]["paper_id"] != paper["paper_id"]:
                raise ValueError("Broken paper asset binding")
        expected_ids = paper_assets[paper["paper_id"]]
        if set(paper["asset_ids"]) != expected_ids:
            raise ValueError("Incomplete paper asset list")
        captions = _json(root, paper["captions_path"])
        bindings = _json(root, paper["bindings_path"])
        if captions["paper_id"] != paper["paper_id"] or bindings["paper_id"] != paper["paper_id"]:
            raise ValueError("Caption/binding paper mismatch")
        figure_ids = {f["figure_id"] for f in paper["figures"]}
        if len(figure_ids) != len(paper["figures"]) or not figure_ids:
            raise ValueError("Missing or duplicate figure identity")
        if bindings.get("figure_id") not in figure_ids or set(bindings.get("asset_ids", [])) != expected_ids:
            raise ValueError("Bindings file does not match paper assets/figures")
        if bindings.get("semantic_review_status") != "unreviewed":
            raise ValueError("Bindings file claims unperformed semantic review")
        if figure_ids != {f["figure_id"] for f in captions["figures"]}:
            raise ValueError("Caption figure set mismatch")
        if not all(f.get("caption") for f in captions["figures"]):
            raise ValueError("Empty caption")
        original_figure = paper["provenance"]["source_record"]["figure"]
        if any(f["caption"] != original_figure["caption"] for f in captions["figures"]):
            raise ValueError("Caption differs from captured source record")
        if bindings.get("provenance", {}).get("source_figure") != original_figure:
            raise ValueError("Binding provenance differs from captured source figure")
        for figure in paper["figures"]:
            for asset_id in figure["asset_ids"]:
                if (asset_id not in by_id or by_id[asset_id]["role"] not in FIGURE_ROLES
                        or by_id[asset_id]["paper_id"] != paper["paper_id"]):
                    raise ValueError("Broken figure asset binding")
    return {"assets": len(assets), "archive_objects": len({r["sha256"] for r in assets if r["storage"] == "archive"}),
            "asset_set_sha256": canonical_hash(assets)}


def _check_provenance(root: Path, manifest: dict, papers: list[dict], domains: dict) -> None:
    """Cross-check frozen source records and runtime metadata without live paths."""
    from .source_snapshot_inventory import identity_aliases
    source_rows = strict_jsonl(root / "provenance/inputs/verified/strict_training_papers.jsonl")
    # A declared scope must publish exactly its own tiers of the frozen view, and
    # must account for every record it left out; an absent scope means the whole view.
    scope = manifest.get("scope") or {}
    tiers = tuple(scope.get("figure_asset_tiers") or ())
    expected_rows = [r for r in source_rows if not tiers or r.get("figure_asset_tier") in tiers]
    if scope and (scope.get("strict_view_papers") != len(source_rows)
                  or scope.get("in_scope_papers") != len(expected_rows)
                  or scope.get("out_of_scope_strict_papers") != len(source_rows) - len(expected_rows)):
        raise ValueError("Declared scope counts do not match the frozen strict view")
    sources = {r["paper_id"]: r for r in source_rows}
    in_scope = {r["paper_id"]: r for r in expected_rows}
    if len(in_scope) != len(expected_rows) or set(in_scope) != {p["paper_id"] for p in papers}:
        raise ValueError("Frozen strict input differs from snapshot paper set")
    for paper in papers:
        original = sources[paper["paper_id"]]
        if paper["provenance"]["source_record"] != {k: v for k, v in original.items() if k != "split"}:
            raise ValueError("Paper provenance differs from frozen strict input")
        if paper["provenance"].get("source_split") != original.get("split"):
            raise ValueError("Original source split was not preserved as provenance")
        for field in ("title", "authors", "primary_domain", "domain_labels", "figure_asset_tier"):
            if paper.get(field) != original.get(field):
                raise ValueError(f"Paper metadata differs from frozen strict input: {field}")
        labels = paper["domain_labels"]
        if len(labels) != len(set(labels)) or not set(labels) <= set(DOMAINS) or paper["primary_domain"] not in labels:
            raise ValueError("Invalid source domain membership")
    expected, source_index = Counter(), {}
    for domain in DOMAINS:
        for row in strict_jsonl(root / "provenance/inputs/verified" / domain / "papers.jsonl"):
            key = (domain, canonical_hash(row))
            expected[key] += 1
            source_index[key] = row
    audit = strict_jsonl(root / "eligibility_audit.jsonl")
    if Counter((r["domain"], r["record_sha256"]) for r in audit) != expected:
        raise ValueError("Eligibility audit does not cover every frozen ledger record")
    if domains["historical_ledger_records"] != len(audit):
        raise ValueError("Historical record count differs from eligibility audit")
    expected_aliases = {p["paper_id"]: set() for p in papers}
    for item in audit:
        if item["eligible"] != (not item["reasons"]) or (item["eligible"] and item.get("canonical_paper_id") not in sources):
            raise ValueError("Eligibility audit has contradictory status or merge destination")
        # An eligible record outside the published scope still keeps its own
        # verdict; it simply contributes aliases only to a published paper.
        if item["eligible"] and item["canonical_paper_id"] in expected_aliases:
            expected_aliases[item["canonical_paper_id"]].update(identity_aliases(source_index[(item["domain"], item["record_sha256"])]))
    for paper in papers:
        aliases = paper["identity_aliases"]
        if len(aliases) != len(set(aliases)) or set(aliases) != expected_aliases[paper["paper_id"]]:
            raise ValueError("Paper alias mapping differs from captured merge evidence")
        if not set(identity_aliases(sources[paper["paper_id"]])) <= set(aliases):
            raise ValueError("Representative paper aliases are missing")
    exclusions = _json(root, "exclusion_summary.json")
    if exclusions["excluded_ledger_records"] != sum(not r["eligible"] for r in audit):
        raise ValueError("Exclusion count differs from eligibility audit")
    if exclusions["failure_reasons"] != dict(Counter(reason for r in audit for reason in r["reasons"])):
        raise ValueError("Exclusion reasons differ from eligibility audit")
    previews = [{"paper_id": r["paper_id"], "domain": r["domain"], "status": "needs_complete_paper_text"}
                for r in audit if "fulltext_preview_only" in r["reasons"]]
    if exclusions.get("pending_full_text", []) != previews:
        raise ValueError("Pending full text list differs from eligibility audit")
    quota = len(papers) >= domains["target_per_domain"] * len(DOMAINS) and all(
        domains["domain_labels"].get(d, 0) >= domains["target_per_domain"] for d in DOMAINS)
    if domains["quota_met"] != quota or manifest["quota_met"] != quota:
        raise ValueError("Quota claim differs from measured source coverage")
    versions = _json(root, "schema_versions.json")
    runtime = versions["runtime_files"]
    if canonical_hash(runtime) != versions["runtime_set_sha256"] or versions["runtime_set_sha256"] != manifest["runtime_set_sha256"]:
        raise ValueError("Runtime provenance digest mismatch")
    if versions["builder_git_commit"] != manifest["builder_git_commit"] or versions["snapshot"] != manifest["schema_version"]:
        raise ValueError("Builder/schema provenance mismatch")
    for item in runtime:
        path = contained_path(root, "provenance/rules/" + item["relative_path"])
        if sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise ValueError("Archived builder/rule bytes differ from runtime provenance")


def validate_source_snapshot(snapshot: str | Path, archive_root: str | Path, *, require_complete: bool = True) -> dict:
    """Do not read live source paths, import collectors, or modify frozen files."""
    root, archive = Path(snapshot).resolve(), Path(archive_root).resolve()
    try:
        manifest = _json(root, "snapshot_manifest.json")
        errors = validate_source_manifest(manifest)
        if errors:
            raise ValueError(", ".join(errors))
        files = _check_files(root, manifest, require_complete)
        papers, assets = strict_jsonl(root / "eligible_papers.jsonl"), strict_jsonl(root / "asset_manifest.jsonl")
        ids = [p["paper_id"] for p in papers]
        if len(ids) != len(set(ids)) or len(ids) != manifest["strict_unique_papers"]:
            raise ValueError("Paper identities/count mismatch")
        if canonical_hash(sorted(ids)) != manifest["paper_set_sha256"]:
            raise ValueError("Paper set mismatch")
        aliases = set()
        for paper in papers:
            if "split" in paper or paper.get("semantic_review_status") != "unreviewed":
                raise ValueError("Source snapshot claims unperformed review or training split")
            if paper.get("figure_asset_tier") not in {"A_source_verified", "B_official_standalone"}:
                raise ValueError("Non A/B asset in snapshot")
            if aliases.intersection(paper["identity_aliases"]):
                raise ValueError("Overlapping paper aliases")
            aliases.update(paper["identity_aliases"])
        counts = _check_assets(root, archive, papers, assets)
        domains = _json(root, "domain_report.json")
        primary = dict(Counter(p["primary_domain"] for p in papers))
        labels = dict(Counter(d for p in papers for d in p["domain_labels"]))
        if domains["unique_papers"] != len(papers) or domains["primary_domains"] != primary or domains["domain_labels"] != labels:
            raise ValueError("Domain report mismatch")
        _check_provenance(root, manifest, papers, domains)
        review = strict_jsonl(root / "review_queue.jsonl")
        if {r["paper_id"] for r in review} != set(ids) or any(r["semantic_review_status"] != "unreviewed" for r in review):
            raise ValueError("Review queue does not cover unreviewed papers")
        inputs = manifest["source_inputs"]
        if canonical_hash(inputs) != manifest["input_set_sha256"]:
            raise ValueError("Input generation hash mismatch")
        for item in inputs:
            if sha256_file(contained_path(root, "provenance/inputs/" + item["relative_path"])) != item["sha256"]:
                raise ValueError("Source input provenance mismatch")
        return {"ok": True, "snapshot_integrity_ok": True, "training_readiness": "source_only",
                "quota_met": domains["quota_met"], "strict_unique_papers": len(papers), "files": len(files),
                "paper_set_sha256": manifest["paper_set_sha256"], "domain_report": domains, **counts,
                "semantic_review_status": "unreviewed", "live_source_required": False}
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        return {"ok": False, "snapshot_integrity_ok": False, "error": str(exc)}
