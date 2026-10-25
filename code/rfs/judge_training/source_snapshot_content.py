"""Materialize portable source papers, evidence and exact builder provenance."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from rfs.contracts.source_snapshot import SOURCE_SNAPSHOT_VERSION
from rfs.utils import write_json
from .figure_rendering import FIGURE_ROLES
from .source_snapshot_inventory import identity_aliases
from .source_snapshot_io import archive_path, canonical_hash, contained_path, copy_verified, sha256_file, stable_bytes, write_jsonl


DOCUMENTS = (
    "docs/paperimage.docx", "docs/new-branch-workflow-spec.md",
    "docs/new-branch-workflow-flowchart.svg", "docs/judge_model_training_plan.md",
    "docs/judge-source-snapshot-d0-plan.md",
)


def builder_provenance(project: Path) -> tuple[dict, dict[str, bytes]]:
    """Capture actual code bytes as well as HEAD, including uncommitted changes."""
    paths = [project / name for name in DOCUMENTS]
    paths += sorted((project / "rfs").rglob("*.py"))
    contents = {p.relative_to(project).as_posix(): stable_bytes(p) for p in paths}
    git = Path("<PATH>")
    executable = str(git) if git.is_file() else "git"
    commit = subprocess.run([executable, "-C", str(project), "rev-parse", "HEAD"], capture_output=True, check=True, text=True).stdout.strip()
    versions = {name: importlib.metadata.version(name) for name in ("PyMuPDF", "pypdfium2", "Pillow", "python-pptx")}
    files = [{"relative_path": name, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)} for name, raw in sorted(contents.items())]
    return {"snapshot": SOURCE_SNAPSHOT_VERSION, "builder_git_commit": commit,
            "python": sys.version, "dependencies": versions, "runtime_files": files,
            "source_pdf_decoder": "pdfium",
            "runtime_set_sha256": canonical_hash(files),
            "graph_middleware": "not_used", "judge_examples": "not_used", "training_splits": "not_used"}, contents


def estimate_storage(audit: dict, archive: Path, staging_parent: Path) -> dict:
    """Include a second independent verification copy plus conservative headroom."""
    import shutil
    archive.mkdir(parents=True, exist_ok=True)
    staging_parent.mkdir(parents=True, exist_ok=True)
    originals = {a["sha256"]: a for a in audit["assets"] if a["storage"] == "archive"}
    archive_bytes = sum(a["size_bytes"] for h, a in originals.items() if not (archive / archive_path(h)).exists())
    payload_bytes = sum(a["size_bytes"] for a in audit["assets"] if a["storage"] == "snapshot")
    metadata_bytes = sum(len(raw) for raw in audit["input_contents"].values())
    snapshot_required = int(2 * (payload_bytes + metadata_bytes) * 1.15) + 512 * 1024**2
    archive_required = int(archive_bytes * 1.15)
    same_disk = archive.stat().st_dev == staging_parent.stat().st_dev
    snapshot_free, archive_free = shutil.disk_usage(staging_parent).free, shutil.disk_usage(archive).free
    enough = snapshot_free >= snapshot_required + (archive_required if same_disk else 0) and archive_free >= archive_required
    return {"ok": enough, "snapshot_payload_bytes": payload_bytes, "new_archive_bytes": archive_bytes,
            "snapshot_required_bytes": snapshot_required, "archive_required_bytes": archive_required,
            "snapshot_free_bytes": snapshot_free, "archive_free_bytes": archive_free, "same_disk": same_disk}


def _materialize_assets(stage: Path, archive: Path, specs: list[dict], progress) -> list[dict]:
    result, seen = [], set()
    for index, spec in enumerate(specs, 1):
        if spec["asset_id"] in seen:
            continue
        seen.add(spec["asset_id"])
        paper_dir = "assets/paper-" + canonical_hash(spec["paper_id"])[:24]
        if spec["storage"] == "archive":
            relative = archive_path(spec["sha256"])
            root = archive
        else:
            filename = {"full_text": "full_text.txt", "text_locations": "text_locations.jsonl"}.get(spec["role"])
            relative = paper_dir + "/" + (filename or f"figures/{spec['asset_id']}{spec['suffix']}")
            root = stage
        copy_verified(spec["source_path"], contained_path(root, relative), spec["sha256"])
        item = {k: v for k, v in spec.items() if k != "source_path"}
        item["relative_path"] = relative
        item["provenance"] = {"source_path": spec["source_path"]}
        result.append(item)
        if progress and (index % 100 == 0 or index == len(specs)):
            progress({"stage": "copy", "checked": index, "total": len(specs)})
    return result


def _paper_record(stage: Path, row: dict, assets: list[dict]) -> dict:
    paper_id = row["paper_id"]
    directory = "assets/paper-" + canonical_hash(paper_id)[:24]
    figure = row["figure"]
    if not any(a["role"] == "text_locations" for a in assets):
        text = next(a for a in assets if a["role"] == "full_text")
        relative = directory + "/text_locations.jsonl"
        source = contained_path(stage, text["relative_path"]).read_text(encoding="utf-8-sig")
        write_jsonl(stage / relative, [{"kind": "source_document", "text_asset_id": text["asset_id"],
            "char_start": 0, "char_end": len(source), "pdf_page": None,
            "source_location": text["provenance"], "precision": "whole_document_only"}])
        digest = sha256_file(stage / relative)
        assets.append({"asset_id": "asset-" + canonical_hash([paper_id, "text_locations", digest])[:24],
            "paper_id": paper_id, "role": "text_locations", "storage": "snapshot", "relative_path": relative,
            "sha256": digest, "size_bytes": (stage / relative).stat().st_size, "suffix": ".jsonl",
            "provenance": {"method": "document_extent_only", "pdf_page_mapping_available": False}})
    figure_id = "figure-" + canonical_hash([paper_id, figure.get("caption"), figure.get("source_files"), figure.get("pdf_page")])[:24]
    figures = [{"figure_id": figure_id,
                "asset_ids": [a["asset_id"] for a in assets if a["role"] in FIGURE_ROLES],
                "completeness_review_status": "unreviewed"}]
    captions_path, bindings_path = directory + "/captions.json", directory + "/bindings.json"
    write_json(stage / captions_path, {"paper_id": paper_id, "figures": [{"figure_id": figure_id,
        "caption": figure["caption"], "figure_label": figure.get("figure_label"), "pdf_page": figure.get("pdf_page")}]})
    write_json(stage / bindings_path, {"paper_id": paper_id, "figure_id": figure_id,
        "asset_ids": [a["asset_id"] for a in assets], "source_binding_status": "metadata_and_assets_checked",
        "semantic_review_status": "unreviewed", "provenance": {"source_figure": figure,
            "official_url": row.get("official_url"), "source_record_sha256": canonical_hash(row)}})
    return {"paper_id": paper_id, "title": row.get("title"), "authors": row.get("authors"),
            "primary_domain": row["primary_domain"], "domain_labels": row["domain_labels"],
            "identity_aliases": identity_aliases(row), "figure_asset_tier": row["figure_asset_tier"],
            "figures": figures, "asset_ids": [a["asset_id"] for a in assets],
            "captions_path": captions_path, "bindings_path": bindings_path,
            "semantic_review_status": "unreviewed", "asset_integrity_status": "verified",
            "provenance": {"source_split": row.get("split"), "source_record": {k: v for k, v in row.items() if k != "split"}}}


def materialize_snapshot(stage: Path, archive: Path, audit: dict, versions: dict, code: dict, progress=None) -> None:
    assets = _materialize_assets(stage, archive, audit["assets"], progress)
    grouped = defaultdict(list)
    for asset in assets:
        grouped[asset["paper_id"]].append(asset)
    papers = [_paper_record(stage, row, grouped[row["paper_id"]]) for row in audit["papers"]]
    for paper in papers:
        paper["identity_aliases"] = audit["paper_aliases"][paper["paper_id"]]
    assets = [asset for row in audit["papers"] for asset in grouped[row["paper_id"]]]
    write_jsonl(stage / "eligible_papers.jsonl", papers)
    write_jsonl(stage / "asset_manifest.jsonl", assets)
    for name, key in (("eligibility_audit", "audit"), ("review_queue", "review_queue")):
        write_jsonl(stage / (name + ".jsonl"), audit[key])
    for name in ("domain_report", "exclusion_summary"):
        write_json(stage / (name + ".json"), audit[name])
    write_json(stage / "schema_versions.json", versions)
    for prefix, mapping in (("provenance/inputs", audit["input_contents"]), ("provenance/rules", code)):
        for name, raw in mapping.items():
            target = contained_path(stage, prefix + "/" + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
