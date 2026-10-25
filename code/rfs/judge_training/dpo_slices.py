"""Cut nested, family-balanced DPO ladders that match the generation exam mix.

A paper-greedy fill of ``full-002`` produced a 1000-pair train slice that was entirely
``rag-subgraph`` pairwise records. Stage-1 DPO is supposed to mix RAG subgraph pairs with
controlled-corruption / quality tasks in the same proportions as ``validation-screen150``.
This module copies existing DPO rows; it does not rebuild pairs.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..utils import write_json

SLICE_CONTRACT = "rfs.judge_dpo_slice"
DEFAULT_SIZES = (20, 200, 1000)
DEFAULT_VAL_SIZE = 64
#: Counts on the frozen generation exam, used only as a mix prior — never as the train ids.
SCREEN150_MIX: dict[tuple[str, str], int] = {
    ("rag", "pairwise_preference"): 78,
    ("corruption", "error_audit"): 19,
    ("corruption", "pairwise_preference"): 17,
    ("corruption", "scientific_quality"): 18,
    ("corruption", "style_preference"): 18,
}


def pair_family(row: dict[str, Any]) -> str:
    """Name the preference family of one DPO record from its example id."""

    example_id = str(row.get("example_id") or "")
    lowered = example_id.lower()
    if "real-vs-generated" in lowered or "real_vs_generated" in lowered:
        return "real_vs_generated"
    if example_id.startswith("rag-subgraph-") or example_id.startswith("rag-"):
        return "rag"
    return "corruption"


def pair_task(row: dict[str, Any]) -> str:
    """Return the judgement task type, defaulting blank rows to pairwise."""

    return str(row.get("task_type") or "pairwise_preference")


def pair_bucket(row: dict[str, Any]) -> tuple[str, str]:
    """Return ``(family, task)`` used for mix quotas."""

    return pair_family(row), pair_task(row)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exam_paper_ids(path: str | Path | None) -> set[str]:
    """Collect paper ids (and example ids as a fallback) from a generation-exam JSONL."""

    if not path:
        return set()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"exam jsonl not found: {source}")
    papers: set[str] = set()
    for row in _jsonl(source):
        paper_id = str(row.get("paper_id") or "").strip()
        if paper_id:
            papers.add(paper_id)
        example_id = str(row.get("example_id") or "").strip()
        if example_id:
            papers.add(example_id)
    return papers


def scale_quotas(size: int, mix: dict[tuple[str, str], int] | None = None) -> dict[tuple[str, str], int]:
    """Largest-remainder allocation of ``size`` seats onto the exam mix."""

    prior = mix or SCREEN150_MIX
    total = sum(prior.values()) or 1
    raw = {key: size * count / total for key, count in prior.items()}
    floors = {key: int(value) for key, value in raw.items()}
    remain = size - sum(floors.values())
    order = sorted(raw, key=lambda key: (raw[key] - floors[key], key[0], key[1]), reverse=True)
    for key in order[: max(0, remain)]:
        floors[key] += 1
    return floors


def _paper_id(row: dict[str, Any]) -> str:
    return str(row.get("paper_id") or row.get("example_id") or "")


def _round_robin_by_paper(rows: Sequence[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_paper_id(row)].append(row)
    papers = list(grouped)
    rng.shuffle(papers)
    for paper in papers:
        rng.shuffle(grouped[paper])
    out: list[dict[str, Any]] = []
    while any(grouped.values()):
        for paper in papers:
            bucket = grouped[paper]
            if bucket:
                out.append(bucket.pop())
    return out


def _fill_quota(pool: Sequence[dict[str, Any]], quotas: dict[tuple[str, str], int], rng: random.Random) -> list[dict[str, Any]]:
    by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    leftover: list[dict[str, Any]] = []
    for row in pool:
        family = pair_family(row)
        if family == "real_vs_generated":
            continue
        key = pair_bucket(row)
        if key in quotas:
            by_bucket[key].append(row)
        else:
            leftover.append(row)
    selected: dict[tuple[str, str], list[dict[str, Any]]] = {}
    unused: list[dict[str, Any]] = []
    shortfall = 0
    for key, quota in quotas.items():
        ordered = _round_robin_by_paper(by_bucket.get(key) or [], rng)
        selected[key] = ordered[:quota]
        unused.extend(ordered[quota:])
        if len(selected[key]) < quota:
            shortfall += quota - len(selected[key])
    unused.extend(_round_robin_by_paper(leftover, rng))
    if shortfall:
        extra = unused[:shortfall]
        unused = unused[shortfall:]
        selected[("_fill", "any")] = extra
    keys = [key for key in quotas if selected.get(key)] + [key for key in selected if key not in quotas]
    assembled: list[dict[str, Any]] = []
    target = sum(quotas.values())
    while len(assembled) < target:
        progressed = False
        for key in keys:
            bucket = selected.get(key) or []
            if bucket:
                assembled.append(bucket.pop(0))
                progressed = True
                if len(assembled) >= target:
                    break
        if not progressed:
            break
    if len(assembled) < target:
        assembled.extend(unused[: target - len(assembled)])
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for row in assembled:
        example_id = str(row.get("example_id") or "")
        if example_id in seen:
            continue
        seen.add(example_id)
        ordered.append(row)
        if len(ordered) >= target:
            break
    return ordered


def _eligible(rows: Sequence[dict[str, Any]], *, split: str | None, excluded_papers: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if pair_family(row) == "real_vs_generated":
            continue
        paper_id = _paper_id(row)
        if paper_id and paper_id in excluded_papers:
            continue
        example_id = str(row.get("example_id") or "")
        if example_id and example_id in excluded_papers:
            continue
        row_split = str(row.get("split") or "train")
        if split and row_split != split:
            continue
        out.append(row)
    return out


def _counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    family: dict[str, int] = defaultdict(int)
    task: dict[str, int] = defaultdict(int)
    bucket: dict[str, int] = defaultdict(int)
    papers: set[str] = set()
    for row in rows:
        family[pair_family(row)] += 1
        task[pair_task(row)] += 1
        bucket[f"{pair_family(row)}:{pair_task(row)}"] += 1
        paper_id = _paper_id(row)
        if paper_id:
            papers.add(paper_id)
    return {
        "records": len(rows),
        "paper_count": len(papers),
        "family": dict(sorted(family.items())),
        "task_type": dict(sorted(task.items())),
        "bucket": dict(sorted(bucket.items())),
    }


def _manifest(name: str, rows: Sequence[dict[str, Any]], *, sha256: str, seed: int, nested: bool, exam_overlap: int, quotas: dict[tuple[str, str], int]) -> dict[str, Any]:
    return {
        "contract": SLICE_CONTRACT,
        "name": name,
        "nested": nested,
        "seed": seed,
        "sha256": sha256,
        "exam_paper_overlap": exam_overlap,
        "quotas": {f"{family}:{task}": count for (family, task), count in sorted(quotas.items())},
        **_counts(rows),
    }


def slice_mixed_dpo(
    dataset: str | Path,
    out: str | Path,
    *,
    exam: str | Path | None = None,
    sizes: Sequence[int] = DEFAULT_SIZES,
    val_size: int = DEFAULT_VAL_SIZE,
    seed: int = 0,
) -> dict[str, Any]:
    """Write nested mixed train slices and a disjoint mixed val slice."""

    source = Path(dataset)
    source = source / "dpo.jsonl" if source.is_dir() else source
    if not source.is_file():
        raise FileNotFoundError(f"DPO jsonl not found: {source}")
    rows = _jsonl(source)
    excluded = exam_paper_ids(exam)
    train_pool = _eligible(rows, split="train", excluded_papers=excluded)
    val_pool = _eligible(rows, split="validation", excluded_papers=excluded)
    if not train_pool:
        raise ValueError("no eligible train DPO records after dropping exam papers and real-vs-generated")

    sizes = tuple(sorted({int(item) for item in sizes if int(item) > 0}))
    if not sizes:
        raise ValueError("at least one positive slice size is required")
    rng = random.Random(seed)
    largest = sizes[-1]
    largest_quotas = scale_quotas(largest)
    selected = {largest: _fill_quota(train_pool, largest_quotas, rng)}
    if len(selected[largest]) < largest:
        raise ValueError(f"train pool only supplied {len(selected[largest])} of {largest} mixed records")
    parent = selected[largest]
    for size in reversed(sizes[:-1]):
        selected[size] = _fill_quota(parent, scale_quotas(size), random.Random(seed + size))
        parent = selected[size]
        if len(selected[size]) < size:
            raise ValueError(f"could not nest a mixed {size}-record slice inside the larger slice")

    train_papers = {_paper_id(row) for row in selected[largest]}
    val_candidates = [row for row in val_pool if _paper_id(row) not in train_papers]
    if len(val_candidates) < val_size:
        largest_ids = {str(item.get("example_id")) for item in selected[largest]}
        unused_train = [
            row for row in train_pool
            if _paper_id(row) not in train_papers and str(row.get("example_id")) not in largest_ids
        ]
        val_candidates.extend(unused_train)
    val_rows = _fill_quota(val_candidates, scale_quotas(val_size), random.Random(seed + 1)) if val_size else []
    if val_size and len(val_rows) < val_size:
        raise ValueError(f"val pool only supplied {len(val_rows)} of {val_size} mixed records")
    val_papers = {_paper_id(row) for row in val_rows}
    if val_size and (val_papers & excluded or val_papers & train_papers):
        raise ValueError("val slice overlapped the exam or the largest train slice papers")

    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    slices: dict[str, Any] = {}
    previous_ids: set[str] | None = None
    for size in sizes:
        name = f"train-{size}"
        rows_for_size = selected[size]
        ids = {str(row.get("example_id")) for row in rows_for_size}
        if previous_ids is not None and not previous_ids <= ids:
            raise ValueError(f"train-{sizes[sizes.index(size) - 1]} is not nested inside {name}")
        previous_ids = ids
        overlap = len({_paper_id(row) for row in rows_for_size} & excluded)
        if overlap:
            raise ValueError(f"{name} leaked {overlap} exam papers")
        digest = _write_jsonl(dest / name / "dpo.jsonl", rows_for_size)
        manifest = _manifest(name, rows_for_size, sha256=digest, seed=seed, nested=True, exam_overlap=overlap, quotas=scale_quotas(size))
        write_json(dest / name / "slice_manifest.json", manifest)
        slices[name] = manifest

    report: dict[str, Any] = {"ok": True, "contract": SLICE_CONTRACT, "dataset": str(source.resolve()), "out": str(dest.resolve()), "seed": seed, "exam": str(Path(exam).resolve()) if exam else None, "slices": slices}
    if val_size:
        digest = _write_jsonl(dest / f"val-{val_size}" / "dpo.jsonl", val_rows)
        manifest = _manifest(f"val-{val_size}", val_rows, sha256=digest, seed=seed, nested=False, exam_overlap=0, quotas=scale_quotas(val_size))
        manifest["train_paper_overlap"] = 0
        write_json(dest / f"val-{val_size}" / "slice_manifest.json", manifest)
        report["val"] = manifest
    write_json(dest / "slice_report.json", report)
    return report


def parse_sizes(raw: str | Iterable[int] | None) -> tuple[int, ...]:
    """Parse ``20,200,1000`` or an iterable of ints into a sorted tuple."""

    if raw is None:
        return DEFAULT_SIZES
    if isinstance(raw, str):
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
        return tuple(sorted(set(values)))
    return tuple(sorted({int(item) for item in raw}))
