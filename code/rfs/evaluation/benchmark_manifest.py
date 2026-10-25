"""Export the public EditableFigureBench manifest from the sealed hidden test.

The 1,376-paper hidden test becomes the public benchmark once this manifest
is written. The export reads the frozen split and the source snapshot. It
does not download papers, call a model, or relabel a figure. A framework
flag is ``yes`` only when a human review selected the figure, ``no`` only
when a human review explicitly selected none, and ``unknown`` otherwise.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..utils import write_json, write_text

PUBLIC_BENCHMARK_NOTE = (
    "This hidden test is exported as the public EditableFigureBench. "
    "It is no longer a sealed Judge exam."
)
HIDDEN_TEST_SPLIT = "hidden_test"
FRAMEWORK_YES = "yes"
FRAMEWORK_NO = "no"
FRAMEWORK_UNKNOWN = "unknown"
SELECTED_DECISIONS = frozenset({"selected"})
REJECTED_DECISIONS = frozenset({"none", "rejected", "not_framework"})
NOT_DOWNLOADED = "not_downloaded"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _arxiv_ids(row: dict[str, Any]) -> list[str]:
    keys = row.get("identity_keys") if isinstance(row.get("identity_keys"), dict) else {}
    raw = keys.get("arxiv_id") if isinstance(keys, dict) else None
    if raw is None:
        raw = row.get("arxiv_id")
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = [str(item) for item in raw if str(item).strip()]
    else:
        values = []
    return [item.strip() for item in values if item.strip()]


def framework_figure_label(
    review_keys: Sequence[str],
    reviews_by_key: dict[str, dict[str, Any]],
) -> str:
    """Join existing human reviews by arXiv id or paper id.

    Membership in the draft corpus is not a decision.
    """
    decisions = []
    for key in review_keys:
        review = reviews_by_key.get(key)
        if review is None:
            continue
        decisions.append(str(review.get("decision") or "").strip().casefold())
    if any(decision in SELECTED_DECISIONS for decision in decisions):
        return FRAMEWORK_YES
    if decisions and all(decision in REJECTED_DECISIONS for decision in decisions):
        return FRAMEWORK_NO
    return FRAMEWORK_UNKNOWN


def _existing_asset_path(snapshot: Path, record: dict[str, Any]) -> str:
    relative = str(record.get("relative_path") or "").strip()
    if relative:
        candidate = snapshot / relative
        if candidate.is_file():
            return str(candidate)
    source = ""
    provenance = record.get("provenance")
    if isinstance(provenance, dict):
        source = str(provenance.get("source_path") or "").strip()
    if source and Path(source).is_file():
        return source
    return ""


def index_snapshot_assets(snapshot: str | Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Index paper PDFs and figure assets by paper id. Missing files stay empty."""
    root = Path(snapshot)
    path = root / "asset_manifest.jsonl"
    indexed: dict[str, dict[str, list[dict[str, Any]]]] = {}
    if not path.is_file():
        return indexed
    for record in _read_jsonl(path):
        role = str(record.get("role") or "")
        if role not in {"paper_pdf", "figure"}:
            continue
        paper_id = str(record.get("paper_id") or "")
        if not paper_id:
            continue
        bucket = indexed.setdefault(paper_id, {"paper_pdf": [], "figure": []})
        bucket[role].append(record)
    return indexed


def _pick_figure(records: Sequence[dict[str, Any]], wanted_sha256: Sequence[str]) -> dict[str, Any] | None:
    wanted = {item for item in wanted_sha256 if item}
    if wanted:
        for record in records:
            if str(record.get("sha256") or "") in wanted:
                return record
    return records[0] if records else None


def _load_id_file(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    ids: set[str] = set()
    for line in file_path.read_text(encoding="utf-8-sig").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            ids.add(item)
    return ids


def overlap_with_ids(rows: Sequence[dict[str, Any]], ids: set[str] | None, *, label: str) -> dict[str, Any]:
    if ids is None:
        return {"dataset": label, "status": NOT_DOWNLOADED, "overlap_count": None, "overlap_ids": []}
    hits: list[str] = []
    for row in rows:
        keys = {str(row.get("paper_id") or ""), str(row.get("arxiv_id") or ""), *list(row.get("arxiv_ids") or [])}
        keys.discard("")
        shared = sorted(keys & ids)
        if shared:
            hits.append(str(row.get("paper_id") or shared[0]))
    return {
        "dataset": label,
        "status": "compared",
        "overlap_count": len(hits),
        "overlap_ids": hits,
    }


def build_manifest_rows(
    ledger_rows: Iterable[dict[str, Any]],
    *,
    snapshot: str | Path,
    reviews_by_arxiv: dict[str, dict[str, Any]],
    framework_arxiv_ids: set[str],
    assets: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
    exclude_paper_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """One row per hidden-test paper. Train-split and excluded ids are dropped."""
    root = Path(snapshot)
    asset_index = assets if assets is not None else index_snapshot_assets(root)
    excluded = exclude_paper_ids or set()
    rows: list[dict[str, Any]] = []
    for ledger in ledger_rows:
        if str(ledger.get("split") or "") != HIDDEN_TEST_SPLIT:
            continue
        paper_id = str(ledger.get("paper_id") or "")
        if not paper_id or paper_id in excluded:
            continue
        arxiv_ids = _arxiv_ids(ledger)
        assets_for_paper = asset_index.get(paper_id) or {"paper_pdf": [], "figure": []}
        pdf_record = (assets_for_paper.get("paper_pdf") or [None])[0]
        figure_record = _pick_figure(
            assets_for_paper.get("figure") or [],
            [str(item) for item in (ledger.get("figure_asset_sha256") or [])],
        )
        rows.append(
            {
                "paper_id": paper_id,
                "arxiv_id": arxiv_ids[0] if arxiv_ids else "",
                "arxiv_ids": arxiv_ids,
                "pdf_path": _existing_asset_path(root, pdf_record) if pdf_record else "",
                "official_figure_path": _existing_asset_path(root, figure_record) if figure_record else "",
                "figure_asset_tier": str(ledger.get("figure_asset_tier") or ""),
                "domain": str(ledger.get("primary_domain") or ""),
                "framework_figure": framework_figure_label([*arxiv_ids, paper_id], reviews_by_arxiv),
                "in_framework_corpus": any(arxiv_id in framework_arxiv_ids for arxiv_id in arxiv_ids),
                "public_benchmark_note": PUBLIC_BENCHMARK_NOTE,
            }
        )
    rows.sort(key=lambda row: row["paper_id"])
    return rows


def append_hidden_test_access(path: str | Path, *, split_id: str, paper_count: int) -> None:
    """Append one access event. The sealed ledger itself is not rewritten."""
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event": "public_benchmark_export",
        "split_id": split_id,
        "hidden_test_papers": paper_count,
        "reader": "rfs benchmark export-manifest",
        "note": PUBLIC_BENCHMARK_NOTE,
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _read_review_ledgers(paths: str | Path | Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    """Merge review ledgers into one store keyed by arXiv id and paper id.

    A later ledger wins for the same key, so a corpus-specific ledger can be
    layered on top of an older one.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    reviews: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _read_jsonl(Path(path)):
            for key in (str(row.get("arxiv_id") or ""), str(row.get("paper_id") or "")):
                if key:
                    reviews[key] = row
    return reviews


def export_public_benchmark_manifest(
    *,
    ledger_path: str | Path,
    snapshot: str | Path,
    reviews_path: str | Path | Sequence[str | Path],
    framework_papers_path: str | Path,
    out_dir: str | Path,
    sciflow_ids: str | Path | None = None,
    figurebench_test_ids: str | Path | None = None,
    figurebench_dev_ids: str | Path | None = None,
    access_log: str | Path | None = None,
    exclude_paper_ids: set[str] | None = None,
    split_id: str = "paper-d0-split-v001",
) -> dict[str, Any]:
    """Write the manifest and the overlap report. Does not call a model."""
    reviews = _read_review_ledgers(reviews_path)
    framework_ids = {str(row.get("arxiv_id") or row.get("paper_id") or "") for row in _read_jsonl(Path(framework_papers_path))}
    framework_ids.discard("")
    rows = build_manifest_rows(
        _read_jsonl(Path(ledger_path)),
        snapshot=snapshot,
        reviews_by_arxiv=reviews,
        framework_arxiv_ids=framework_ids,
        exclude_paper_ids=exclude_paper_ids,
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "hidden_test_manifest.jsonl"
    write_text(
        manifest_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )
    overlap = {
        "summary": "Overlap between the public 1,376-paper set and the two external benchmarks.",
        "sciflow": overlap_with_ids(rows, _load_id_file(sciflow_ids), label="SciFlow-Bench"),
        "figurebench_official_test": overlap_with_ids(rows, _load_id_file(figurebench_test_ids), label="FigureBench official test"),
        "figurebench_development": overlap_with_ids(rows, _load_id_file(figurebench_dev_ids), label="FigureBench development"),
    }
    write_json(out / "overlap.json", overlap)
    counts = {label: sum(1 for row in rows if row["framework_figure"] == label) for label in (FRAMEWORK_YES, FRAMEWORK_NO, FRAMEWORK_UNKNOWN)}
    if access_log is not None:
        append_hidden_test_access(access_log, split_id=split_id, paper_count=len(rows))
    report = {
        "summary": PUBLIC_BENCHMARK_NOTE,
        "ok": True,
        "paper_count": len(rows),
        "framework_figure_counts": counts,
        "manifest": str(manifest_path),
        "overlap": str(out / "overlap.json"),
        "pdf_present": sum(1 for row in rows if row["pdf_path"]),
        "official_figure_present": sum(1 for row in rows if row["official_figure_path"]),
    }
    write_json(out / "manifest_report.json", report)
    return report
