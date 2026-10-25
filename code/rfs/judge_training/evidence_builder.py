"""Build paper evidence packs for the first Judge dataset iteration.

The source corpus is read-only.  This builder combines a paper PDF, an optional
classification CSV, and an optional selected framework figure into one
auditable JSONL record.  It intentionally does not infer scientific ground
truth from the paper; that remains a human/contract annotation step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fitz


SECTION_RE = re.compile(
    r"(?im)^(?P<header>(?:\d+(?:\.\d+)*\s+)?"
    r"(?:introduction|background|problem formulation|problem definition|"
    r"method(?:ology|s)?|approach|model|framework|architecture|system|"
    r"implementation|experiments?|conclusion|related work)\b.*?)$"
)
CAPTION_RE = re.compile(r"(?is)\b(?:figure|fig\.|图)\s*\d+[.:]?\s*.*?(?=\n\s*\n|\Z)")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _clean(text: str) -> str:
    text = re.sub(r"([A-Za-z])[-‐]\s*\n\s*([a-z])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[str]:
    with fitz.open(pdf_path) as doc:
        pages = []
        for page in doc:
            blocks = [b for b in page.get_text("blocks") if b[4].strip()]
            blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))
            pages.append(_clean("\n".join(b[4] for b in blocks)))
        return pages


def extract_title_abstract(pages: list[str], metadata: dict[str, Any]) -> tuple[str, str]:
    title = _clean(str(metadata.get("title") or ""))
    first = pages[0] if pages else ""
    if not title:
        title = _clean(first.split("\n", 1)[0]) if first else ""
    match = re.search(r"(?is)\babstract\b\s*[:.]?\s*(.*?)(?=\n\s*\d?\.?\s*(?:introduction|1\s+introduction)\b)", first)
    abstract = _clean(match.group(1)) if match else ""
    return title, abstract


def extract_evidence(pdf_path: Path, metadata: dict[str, Any], pages: list[str] | None = None) -> dict[str, Any]:
    pages = pages if pages is not None else extract_pages(pdf_path)
    title, abstract = extract_title_abstract(pages, metadata)
    full_text = "\n\n".join(f"--- PAGE {i + 1} ---\n{p}" for i, p in enumerate(pages))
    headers = list(SECTION_RE.finditer(full_text))
    sections: dict[str, str] = {}
    for index, match in enumerate(headers):
        start = match.end()
        end = headers[index + 1].start() if index + 1 < len(headers) else len(full_text)
        key = re.sub(r"^\d+(?:\.\d+)*\s+", "", match.group("header").strip().lower())
        if any(token in key for token in ("method", "approach", "model", "framework", "architecture", "system", "problem")):
            sections[key] = _clean(full_text[start:end])
    captions = [_clean(m.group(0)) for m in CAPTION_RE.finditer(full_text)]
    method_text = "\n\n".join(sections.values())
    return {
        "title": title,
        "abstract": abstract,
        "problem_definition": sections.get("problem definition", sections.get("problem formulation", "")),
        "method": method_text,
        "method_sections": sections,
        "figure_captions": captions,
        "page_count": len(pages),
    }


def load_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {str(row.get("arxiv_id") or "").strip(): dict(row) for row in rows if row.get("arxiv_id")}


def load_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("items", payload.get("papers", []))
    result = {}
    for item in items:
        if isinstance(item, dict) and item.get("arxiv_id"):
            result[str(item["arxiv_id"])] = item
    return result


def load_selected_ids(path: Path | None) -> list[str] | None:
    """Read IDs from a D0 selection manifest or a JSON array."""
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return [str(item.get("arxiv_id") or "").strip() for item in records if isinstance(item, dict) and item.get("arxiv_id")]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_raw_manifest_evidence(corpus_root: str | Path, manifest_path: str | Path, out_dir: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    """Extract one auditable, page-addressable raw evidence pack per manifest paper."""
    corpus = Path(corpus_root).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = _jsonl(Path(manifest_path))
    written = 0
    failures: list[dict[str, str]] = []
    for record in manifest[:limit] if limit else manifest:
        aid = str(record.get("arxiv_id") or record.get("paper_id") or "").strip()
        pdf = corpus / "pdf" / f"{aid}.pdf"
        target = out / f"{aid}.json"
        if target.exists():
            written += 1
            continue
        try:
            pages = extract_pages(pdf)
            metadata = {"title": record.get("title") or ""}
            evidence = extract_evidence(pdf, metadata, pages)
            evidence["pages"] = [{"page": index + 1, "text": page} for index, page in enumerate(pages)]
            evidence.update({"paper_id": aid, "arxiv_id": aid, "pdf_path": str(pdf), "pdf_sha256": _sha256(pdf), "split": record.get("split"), "final_label": record.get("final_label"), "status": "success", "parser": "pymupdf", "parser_version": "raw-v001"})
            target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            written += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"arxiv_id": aid, "error": str(exc)})
    report = {"manifest_path": str(Path(manifest_path).resolve()), "out_dir": str(out), "requested": min(limit, len(manifest)) if limit else len(manifest), "written": written, "failures": failures, "status": "success" if not failures else "partial"}
    (out / "_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_records(source_root: Path, out_path: Path, labels_path: Path | None, figures_root: Path | None, limit: int | None, selected_ids_path: Path | None = None) -> dict[str, Any]:
    pdf_dir = source_root / "pdf"
    metadata = load_metadata(source_root / "papers.json")
    labels = load_labels(labels_path)
    selected_ids = load_selected_ids(selected_ids_path)
    pdfs = (
        [pdf_dir / f"{arxiv_id}.pdf" for arxiv_id in selected_ids if (pdf_dir / f"{arxiv_id}.pdf").exists()]
        if selected_ids is not None
        else sorted(pdf_dir.glob("*.pdf"))
    )
    if limit:
        pdfs = pdfs[:limit]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing_labels: list[str] = []
    records = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for pdf in pdfs:
            arxiv_id = pdf.stem
            label = labels.get(arxiv_id)
            if labels_path and label is None:
                missing_labels.append(arxiv_id)
                continue
            figure_path = None
            if figures_root:
                candidate = figures_root / arxiv_id / "framework.png"
                if candidate.exists():
                    figure_path = str(candidate)
            evidence = extract_evidence(pdf, metadata.get(arxiv_id, {}))
            record = {
                "example_id": f"paper-{arxiv_id}",
                "dataset_version": "judge-d0-evidence-v001",
                "ground_truth_id": arxiv_id,
                "task_type": "quality_classification",
                "paper": {"arxiv_id": arxiv_id, "pdf_path": str(pdf), **evidence},
                "figure": {"path": figure_path, "status": "selected" if figure_path else "missing"},
                "human_labels": label or {},
                "scientific_gt": {"entities": [], "relations": [], "must_show": [], "must_not_invent": [], "terminology": [], "annotation_status": "pending"},
                "provenance": {"source_root": str(source_root), "labels_path": str(labels_path) if labels_path else None},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1
    report = {"source_root": str(source_root), "labels_path": str(labels_path) if labels_path else None, "selected_ids_path": str(selected_ids_path) if selected_ids_path else None, "pdf_count_considered": len(pdfs), "records_written": records, "missing_labels": missing_labels, "label_count": len(labels)}
    out_path.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(r"<PATH>"))
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--figures-root", type=Path)
    parser.add_argument("--selected-ids", type=Path, help="D0 selection manifest or JSON array of arXiv IDs.")
    parser.add_argument("--manifest", type=Path, help="Frozen framework manifest papers.jsonl; writes one raw pack per paper.")
    parser.add_argument("--raw-out", type=Path, help="Directory for manifest-driven raw evidence packs.")
    parser.add_argument("--out", type=Path, default=Path("datasets/judge-d0/evidence_packs.jsonl"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.manifest:
        report = build_raw_manifest_evidence(args.source_root, args.manifest, args.raw_out or Path("datasets/judge-framework-v001/evidence/raw"), limit=args.limit)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    report = build_records(args.source_root, args.out, args.labels, args.figures_root, args.limit, args.selected_ids)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
