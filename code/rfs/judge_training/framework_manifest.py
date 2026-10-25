"""Build a frozen Judge manifest from the human framework-review corpus."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def _json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return default


def _reviews(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("arxiv_id"):
                result[str(item["arxiv_id"])] = item
    return result


def _split(arxiv_id: str) -> str:
    bucket = int(hashlib.sha256(arxiv_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "test" if bucket < 15 else "validation" if bucket < 30 else "train"


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_label(quality: str, style: str, review: dict[str, Any]) -> str:
    replacement = review.get("replacement_label")
    if replacement in {"\u597d\uff08\u79d1\u7814\uff09", "\u597d\uff08\u751f\u52a8\uff09", "\u574f"}:
        return replacement
    if quality == "\u574f":
        return "\u574f"
    if quality == "\u672a\u7b5b":
        return "\u672a\u7b5b"
    return "\u597d\uff08\u751f\u52a8\uff09" if "\u751f\u52a8" in style else "\u597d\uff08\u79d1\u7814\uff09"


def build_framework_manifest(
    corpus_root: str | Path,
    reviews_path: str | Path,
    out_dir: str | Path,
    *,
    dataset_version: str = "judge-framework-v001",
) -> dict[str, Any]:
    """Write paper, figure, split, report, and README artifacts for selected figures."""
    corpus = Path(corpus_root).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    papers = {str(item.get("arxiv_id")): item for item in _json(corpus / "papers.json", []) if isinstance(item, dict) and item.get("arxiv_id")}
    reviews = _reviews(Path(reviews_path))
    labels_path = corpus / "classified" / "分类清单.csv"
    paper_records: list[dict[str, Any]] = []
    figure_records: list[dict[str, Any]] = []
    stats = {"total_labeled": 0, "selected_papers": 0, "selected_figures": 0, "none_papers": 0, "excluded_unreviewed": 0, "excluded_no_figure": 0}
    with labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            aid = str(row.get("arxiv_id") or "").strip()
            if not aid:
                continue
            stats["total_labeled"] += 1
            review = reviews.get(aid) or {}
            decision = str(review.get("decision") or "")
            quality = str(row.get("分类") or "")
            style = str(row.get("风格") or "")
            if quality == "\u672a\u7b5b":
                stats["excluded_unreviewed"] += 1
                continue
            if decision != "selected":
                stats["none_papers"] += decision == "none"
                stats["excluded_no_figure"] += decision == "none"
                continue
            paper = papers.get(aid, {})
            split = _split(aid)
            final_label = _final_label(quality, style, review)
            paper_records.append({"paper_id": aid, "arxiv_id": aid, "title": paper.get("title") or row.get("标题") or "", "abstract": paper.get("abstract") or "", "paper_path": f"experiments/literature_review/papers/pdf/{aid}.pdf", "split": split, "final_label": final_label})
            ranks = review.get("selected_ranks") or ([review["selected_rank"]] if str(review.get("selected_rank", "")).isdigit() else [])
            source = str(review.get("selected_candidate_source") or "initial")
            root_name = "_reselect_good" if source == "reselection_good" else "_reselect" if source == "reselection" else "_stage"
            for rank in sorted({int(value) for value in ranks}):
                figure_rel = f"experiments/literature_review/papers/{root_name}/{aid}/cand_{rank}.png"
                figure_records.append({"figure_id": f"{aid}__{source}_{rank}", "paper_id": aid, "arxiv_id": aid, "figure_path": figure_rel, "image_sha256": _sha256(corpus / root_name / aid / f"cand_{rank}.png"), "candidate_rank": rank, "candidate_source": source, "split": split, "final_label": final_label, "scope": review.get("scope") or "unknown", "vlm_choice": review.get("vlm_choice_at_review")})
    stats["selected_papers"] = len(paper_records)
    stats["selected_figures"] = len(figure_records)
    for name, records in (("papers.jsonl", paper_records), ("figures.jsonl", figure_records)):
        (out / name).write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    manifest = {"dataset_version": dataset_version, "task": "scientific_framework_judge", "label_field": "final_label", "labels": ["\u597d\uff08\u79d1\u7814\uff09", "\u597d\uff08\u751f\u52a8\uff09", "\u574f"], "paper_records": len(paper_records), "figure_records": len(figure_records), "splits": {split: sum(item["split"] == split for item in figure_records) for split in ("train", "validation", "test")}, "source_reviews": str(Path(reviews_path).resolve()), "corpus_root": str(corpus), "stats": stats}
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dataset_report.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "README.md").write_text("# Judge framework dataset\n\n冻结的人工作选框架图 manifest。训练标签统一使用 `final_label`；原始分类与人工纠正过程不写入样本记录。数据按 `arxiv_id` 固定切分。\n", encoding="utf-8")
    return manifest
