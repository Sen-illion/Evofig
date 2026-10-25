#!/usr/bin/env python3
"""Build the paper lists the public-benchmark arms read.

SciFlow-Bench and FigureBench hand the system a paper, not a recording of one: SciFlow keeps
one paper per id under ``benchmarks/autobench_2025``, FigureBench keeps the paper text in a
parquet column. Neither is a row the arm driver can read, so this script turns both into the
one row shape the driver does read (``domain`` / ``paper_id`` / ``pdf_path``) and stages every
input under a single directory, so a list keeps working after the run tree it was harvested
from is cleaned up.

SciFlow's source PDFs are no longer under ``benchmarks/autobench_2025`` on the experiment host.
The copies an earlier run archived beside its own output (``inputs/paper.pdf``) are the ones
that are left, and they are what this script stages. A paper that came in as OCR markdown is
staged as ``source_text.md`` and counted in the report, so a list never claims a PDF input
that does not exist.

Usage (on the experiment host; this script talks to the filesystem only):

    <DATA_ROOT>/envs/paperimage/bin/python scripts/build_public_lists.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

AUTOBENCH_ROOT = Path("<DATA_ROOT>/benchmarks/autobench_2025")
SCIFLOW_INDEX = Path("<DATA_ROOT>/benchmarks/SciFlow-Bench/autobench_index.json")
SCIFLOW_RUN_ROOT = Path("<DATA_ROOT>/efb_runs/sciflow")
FIGUREBENCH_PARQUET = Path("<DATA_ROOT>/benchmarks/FigureBench/data/test.parquet")


def stage(source: Path, target: Path) -> str:
    """Put one input where the list points, preferring a link over a second copy.

    Both trees sit on the same disk here, so a link keeps several gigabytes of SciFlow PDFs
    from being duplicated while still surviving a cleanup of the run tree they came from.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size:
        return str(target)
    if target.exists():
        target.unlink()
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)
    return str(target)


def sciflow_rows(index_path: Path, autobench_root: Path, run_root: Path, stage_root: Path) -> tuple[list, dict]:
    """One row per paper of the official SciFlow index, in index order."""

    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    rows: list = []
    kinds: dict = {}
    missing: list = []
    for domain in sorted(index):
        for paper_id in index[domain]:
            key = "%s__%s" % (domain, paper_id)
            candidates = (
                (Path(autobench_root) / domain / paper_id / "paper.pdf", "benchmark_pdf", "paper.pdf"),
                (Path(run_root) / key / "inputs" / "paper.pdf", "archived_run_pdf", "paper.pdf"),
                (Path(run_root) / key / "source_text.md", "ocr_markdown", "source_text.md"),
                (Path(run_root) / key / "inputs" / "source_text.md", "ocr_markdown", "source_text.md"),
            )
            found = next(((path, kind, name) for path, kind, name in candidates if path.is_file()), None)
            if found is None:
                missing.append("%s/%s" % (domain, paper_id))
                continue
            source, kind, name = found
            kinds[kind] = kinds.get(kind, 0) + 1
            rows.append(
                {
                    "domain": domain,
                    "paper_id": "%s/%s" % (domain, paper_id),
                    "pdf_path": stage(source, Path(stage_root) / "sciflow" / key / name),
                    "input_kind": kind,
                }
            )
    report = {"benchmark": "sciflow", "rows": len(rows), "input_kinds": kinds, "missing": missing}
    return rows, report


def read_figurebench_parquet(path: Path) -> tuple[list, list]:
    """The test split's paper text and category, in file order."""

    import pyarrow.parquet as pq

    table = pq.read_table(Path(path), columns=["text", "category"])
    return table.column("text").to_pylist(), table.column("category").to_pylist()


def figurebench_rows(parquet: Path, stage_root: Path, reader=read_figurebench_parquet) -> tuple[list, dict]:
    """One row per FigureBench test item; the row names it after its position in the file.

    FigureBench ids carry no paper identity, so the position is the identity, exactly as the
    split's own scorer addresses them.
    """

    texts, categories = reader(Path(parquet))
    rows: list = []
    for index, (text, category) in enumerate(zip(texts, categories)):
        paper_id = "test_%04d" % index
        body = str(text or "")
        target = Path(stage_root) / "figurebench" / (paper_id + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_text(encoding="utf-8") != body:
            target.write_text(body, encoding="utf-8")
        rows.append(
            {
                "domain": str(category or ""),
                "paper_id": paper_id,
                "pdf_path": str(target),
                "input_kind": "text",
            }
        )
    report = {"benchmark": "figurebench", "rows": len(rows), "input_kinds": {"text": len(rows)}}
    return rows, report


def write_list(rows: list, list_path: Path, report: dict, report_path: Path) -> None:
    Path(list_path).parent.mkdir(parents=True, exist_ok=True)
    Path(list_path).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the public-benchmark paper lists.")
    parser.add_argument("--out-root", default="<DATA_ROOT>/abl/public_inputs")
    parser.add_argument("--list-dir", default="<DATA_ROOT>/abl")
    parser.add_argument("--sciflow-index", default=str(SCIFLOW_INDEX))
    parser.add_argument("--autobench-root", default=str(AUTOBENCH_ROOT))
    parser.add_argument("--sciflow-run-root", default=str(SCIFLOW_RUN_ROOT))
    parser.add_argument("--figurebench-parquet", default=str(FIGUREBENCH_PARQUET))
    args = parser.parse_args(argv)

    out_root = Path(args.out_root)
    list_dir = Path(args.list_dir)
    rows, report = sciflow_rows(
        Path(args.sciflow_index), Path(args.autobench_root), Path(args.sciflow_run_root), out_root
    )
    write_list(rows, list_dir / "sciflow_500.jsonl", report, list_dir / "sciflow_500_report.json")
    print("sciflow rows=%d kinds=%s missing=%d" % (len(rows), report["input_kinds"], len(report["missing"])))

    rows, report = figurebench_rows(Path(args.figurebench_parquet), out_root)
    write_list(rows, list_dir / "figurebench_300.jsonl", report, list_dir / "figurebench_300_report.json")
    print("figurebench rows=%d" % len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
