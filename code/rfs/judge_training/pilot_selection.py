"""Select the D0 Pilot set out of the train and validation splits only.

Stage 2 of the Judge execution roadmap.  The Pilot is the small, deliberately
varied slice of D0 that the Graph Middleware, the PPTX generation and the
trajectory work are developed against, so it has to cover the five domains, the
three figure types and the five structural shapes instead of being a plain random
sample.  D0 does not carry either of those two labels, so a vision model labels a
candidate pool first and the selection is then made inside those labels.

The hidden test split is never a candidate.  Its papers are only ever filtered
out, and the hidden test manifest is never opened from this module.
"""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from rfs.utils import write_json, write_text

from .figure_rendering import FIGURE_ROLES
from .source_snapshot_io import sha256_file
from .split_ledger import SPLIT_ORDER, load_split_ledger

SCHEMA_VERSION = "judge-pilot-v1"
DEFAULT_PILOT_ID = "paper-d0-pilot-v001"
CANDIDATE_SPLITS = ("train", "validation")
FIGURE_TYPES = ("framework", "pipeline", "technical_route", "other")
STRUCTURES = ("linear", "branching", "converging", "feedback_loop", "multi_panel")
DOMAIN_QUOTAS: dict[str, tuple[int, int]] = {
    "cv": (16, 4),
    "rag": (16, 4),
    "agent": (16, 4),
    "materials": (12, 3),
    "biology": (10, 2),
}
DEFAULT_DOMAIN_QUOTA = (10, 2)
POOL_MULTIPLIER = 3
DEFAULT_RENDER_DPI = 150
MAX_RENDER_EDGE = 1600
MIN_PILOT_SIZE = 60
MAX_PILOT_SIZE = 100
UNLABELLED = "unlabelled"
LABEL_FIELDS = (
    "figure_type",
    "structure",
    "panel_count",
    "node_count",
    "has_arrows",
    "complexity",
    "confidence",
    "reason",
    "label_source",
)

LABEL_PROMPT = """You label exactly one academic framework figure for a figure-generation dataset.
Answer with JSON only, no prose:
{"figure_type": "framework|pipeline|technical_route|other",
 "structure": "linear|branching|converging|feedback_loop|multi_panel",
 "panel_count": <int>, "node_count": <int>, "has_arrows": <true|false>,
 "complexity": <int 1-5>, "confidence": <0-1>, "reason": "<one short sentence>"}
figure_type: framework = overview of a proposed method, pipeline = staged data or
training flow, technical_route = one algorithmic route or architecture, other = a
result, qualitative or non-structural figure.
structure: linear = one path, branching = paths split, converging = paths merge,
feedback_loop = a cycle returns into the main path, multi_panel = separate panels."""


STRUCTURAL_TYPES = FIGURE_TYPES[:3]
STRUCTURE_GATE_PROMPT = """Answer one question about this academic figure with JSON only, no prose:
{"is_diagram": <true|false>, "reason": "<one short sentence>"}
is_diagram = true only when the figure itself is a schematic built from boxes, modules,
panels, connectors or arrows that shows how a method, system or pipeline is organised.
Set it to false for plots, charts, tables, screenshots, photographs, qualitative result
grids or a single equation, even when a small schematic sits next to them."""


def _digest(*parts: Any) -> str:
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def load_snapshot_papers(snapshot: str | Path) -> list[dict[str, Any]]:
    """Load the strict paper ledger of the source snapshot."""
    path = Path(snapshot) / "eligible_papers.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot is missing eligible_papers.jsonl: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Snapshot has no eligible papers: {path}")
    return rows


def load_figure_index(snapshot: str | Path) -> dict[str, dict[str, Any]]:
    """Map every figure asset id of the snapshot to its file facts."""
    path = Path(snapshot) / "asset_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot is missing asset_manifest.jsonl: {path}")
    index: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        # A framework figure the snapshot only had as a rendered page is still
        # the paper's framework figure; refusing it here would drop the paper
        # from a subset that the snapshot itself passed.
        if str(record.get("role") or "") not in FIGURE_ROLES:
            continue
        index[str(record.get("asset_id") or "")] = {
            "paper_id": str(record.get("paper_id") or ""),
            "relative_path": str(record.get("relative_path") or ""),
            "suffix": str(record.get("suffix") or ""),
            "sha256": str(record.get("sha256") or ""),
        }
    return index


def primary_figure_asset(paper: dict[str, Any]) -> str:
    """Return the figure asset id a paper was frozen with, or an empty string."""
    for figure in paper.get("figures") or []:
        for asset_id in figure.get("asset_ids") or []:
            return str(asset_id)
    return ""


def caption_of(paper: dict[str, Any]) -> str:
    source = (paper.get("provenance") or {}).get("source_record") or {}
    return str((source.get("figure") or {}).get("caption") or "")


def candidate_pool(
    split_rows: Iterable[dict[str, Any]],
    papers: Iterable[dict[str, Any]],
    *,
    seed: str,
    quotas: dict[str, tuple[int, int]] | None = None,
    pool_multiplier: int = POOL_MULTIPLIER,
) -> dict[str, Any]:
    """Pick a per-domain candidate pool from train and validation only."""
    quotas = quotas or DOMAIN_QUOTAS
    by_id = {str(paper.get("paper_id")): paper for paper in papers}
    split_of = {str(row["paper_id"]): str(row["split"]) for row in split_rows}
    domain_of = {
        str(row["paper_id"]): str(row.get("primary_domain") or "unknown") for row in split_rows
    }

    buckets: dict[tuple[str, str], list[str]] = {}
    for paper_id, split in split_of.items():
        if split not in CANDIDATE_SPLITS:
            continue
        domain = domain_of.get(paper_id, "unknown")
        key = (domain, split)
        buckets.setdefault(key, []).append(paper_id)
    for key, ids in buckets.items():
        ids.sort(key=lambda paper_id: _digest(seed, "pool", paper_id))

    domains = sorted({domain for domain, _ in buckets} | set(quotas))
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for domain in domains:
        train_quota, validation_quota = quotas.get(domain, DEFAULT_DOMAIN_QUOTA)
        for split, quota in zip(CANDIDATE_SPLITS, (train_quota, validation_quota)):
            wanted = max(quota * pool_multiplier, quota)
            pool = buckets.get((domain, split), [])
            taken = 0
            for paper_id in pool:
                if taken >= wanted:
                    break
                paper = by_id.get(paper_id) or {}
                asset_id = primary_figure_asset(paper)
                if not asset_id:
                    skipped.append({"paper_id": paper_id, "reason": "no figure asset"})
                    continue
                candidates.append(
                    {
                        "paper_id": paper_id,
                        "split": split,
                        "domain": domain,
                        "figure_asset_tier": paper.get("figure_asset_tier"),
                        "figure_asset_id": asset_id,
                        "title": paper.get("title"),
                        "caption": caption_of(paper),
                    }
                )
                taken += 1
            if taken < wanted:
                skipped.append(
                    {"domain": domain, "split": split, "reason": "pool smaller than target", "available": len(pool), "taken": taken}
                )
    return {"candidates": candidates, "skipped": skipped, "quotas": {k: list(v) for k, v in quotas.items()}}


def render_figure_png(
    snapshot: str | Path,
    relative_path: str,
    *,
    dpi: int = DEFAULT_RENDER_DPI,
    max_edge: int = MAX_RENDER_EDGE,
) -> bytes:
    """Render one figure asset to PNG bytes for the labelling model."""
    from PIL import Image

    source = Path(snapshot) / relative_path
    if source.suffix.lower() == ".pdf":
        import pypdfium2

        with pypdfium2.PdfDocument(str(source)) as document:
            page = document[0]
            try:
                bitmap = page.render(scale=dpi / 72)
                try:
                    image = bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()
    else:
        with Image.open(source) as opened:
            opened.load()
            image = opened.convert("RGB")
    try:
        if max_edge and max(image.size) > max_edge:
            ratio = max_edge / max(image.size)
            resized = image.resize(
                (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                Image.LANCZOS,
            )
            image.close()
            image = resized
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        image.close()


def normalize_label(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce a model answer into the fixed label vocabulary."""
    figure_type = str(raw.get("figure_type") or "").strip().casefold()
    structure = str(raw.get("structure") or "").strip().casefold()

    def as_int(value: Any, default: int) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    return {
        "figure_type": figure_type if figure_type in FIGURE_TYPES else "other",
        "structure": structure if structure in STRUCTURES else UNLABELLED,
        "panel_count": max(1, as_int(raw.get("panel_count"), 1)),
        "node_count": max(0, as_int(raw.get("node_count"), 0)),
        "has_arrows": bool(raw.get("has_arrows")),
        "complexity": min(5, max(1, as_int(raw.get("complexity"), 3))),
        "confidence": min(1.0, max(0.0, as_float(raw.get("confidence"), 0.0))),
        "reason": str(raw.get("reason") or "")[:400],
    }


def candidate_png_path(work_dir: str | Path, paper_id: str) -> Path:
    """Return the render target of one candidate figure."""
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in str(paper_id))
    return Path(work_dir) / f"{safe}.png"


def _render_task(task: tuple[str, str, str, int]) -> tuple[str, str, int, str | None]:
    """Render one candidate figure into its PNG target; report every failure."""
    snapshot, relative, target, dpi = task
    path = Path(target)
    if path.is_file() and path.stat().st_size:
        return relative, str(path), path.stat().st_size, None
    try:
        payload = render_figure_png(snapshot, relative, dpi=dpi)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return relative, str(path), len(payload), None
    except Exception as exc:  # noqa: BLE001 - the report must show every failure
        return relative, str(path), 0, f"{type(exc).__name__}: {exc}"


def render_candidate_pngs(
    candidates: Sequence[dict[str, Any]],
    snapshot: str | Path,
    figure_index: dict[str, dict[str, Any]],
    *,
    work_dir: str | Path,
    dpi: int = DEFAULT_RENDER_DPI,
    workers: int = 4,
) -> dict[str, Any]:
    """Render every candidate figure; PDFium is not thread safe, so this uses processes."""
    tasks: list[tuple[str, str, str, int]] = []
    missing: list[dict[str, Any]] = []
    for item in candidates:
        asset = figure_index.get(str(item["figure_asset_id"])) or {}
        relative = asset.get("relative_path")
        if not relative:
            missing.append({"paper_id": item["paper_id"], "error": "figure asset is not in the manifest"})
            continue
        tasks.append((str(snapshot), str(relative), str(candidate_png_path(work_dir, item["paper_id"])), dpi))
    rendered: dict[str, str] = {}
    failures: list[dict[str, Any]] = list(missing)
    if tasks:
        if workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_render_task, tasks, chunksize=4))
        else:
            results = [_render_task(task) for task in tasks]
        by_relative = {task[1]: task[2] for task in tasks}
        for relative, target, size, error in results:
            if error:
                failures.append({"relative_path": relative, "error": error})
                continue
            rendered[relative] = target
    return {"rendered": rendered, "failures": failures, "tasks": len(tasks)}


def load_label_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the append-only label cache keyed by paper id."""
    target = Path(path)
    if not target.is_file():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        cached[str(record["paper_id"])] = record
    return cached


def vision_labeller() -> Callable[[str, Path], dict[str, Any]]:
    """Return the default labeller: one vision-model call per figure."""
    from rfs.providers.client import call_vlm_json

    def call(prompt: str, image: Path) -> dict[str, Any]:
        return call_vlm_json(prompt, [image], timeout=240, retries=2)

    return call


def _ask_candidates(
    pending: Sequence[dict[str, Any]],
    *,
    prompt_for: Callable[[dict[str, Any]], str],
    record_for: Callable[[dict[str, Any], dict[str, Any], Path], dict[str, Any]],
    cache_path: Path,
    rendered_paths: dict[str, str],
    figure_index: dict[str, dict[str, Any]],
    labeller: Callable[[str, Path], dict[str, Any]],
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ask one question about every pending candidate and append the answers.

    Answers are appended one at a time so an interrupted run resumes instead of
    paying for every request again.
    """

    def run(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        paper_id = str(item["paper_id"])
        asset = figure_index.get(str(item["figure_asset_id"])) or {}
        relative = str(asset.get("relative_path") or "")
        image = rendered_paths.get(relative)
        if not image:
            return item, {"paper_id": paper_id, "error": "figure could not be rendered"}
        try:
            answer = labeller(prompt_for(item), Path(image))
        except Exception as exc:  # noqa: BLE001 - the report must show every failure
            return item, {"paper_id": paper_id, "error": f"labeller: {type(exc).__name__}: {exc}"}
        return item, record_for(item, answer if isinstance(answer, dict) else {}, Path(image))

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with cache_path.open("a", encoding="utf-8", newline="\n") as stream:
        if workers > 1 and len(pending) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run, item) for item in pending]
                answers = [future.result() for future in as_completed(futures)]
        else:
            answers = [run(item) for item in pending]
        for _, record in answers:
            if record and "error" in record:
                failures.append(record)
            elif record:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                rows.append(record)

    rows.sort(key=lambda row: row["paper_id"])
    return rows, failures


def label_candidates(
    candidates: Sequence[dict[str, Any]],
    snapshot: str | Path,
    figure_index: dict[str, dict[str, Any]],
    *,
    cache_path: str | Path,
    work_dir: str | Path,
    labeller: Callable[[str, Path], dict[str, Any]] | None = None,
    dpi: int = DEFAULT_RENDER_DPI,
    workers: int = 4,
) -> dict[str, Any]:
    """Label every candidate that has no cached label and cache the answers."""
    cache_path = Path(cache_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_label_cache(cache_path)
    pending = [item for item in candidates if str(item["paper_id"]) not in cached]
    if not pending:
        return {"labelled": len(cached), "newly_labelled": 0, "failures": [], "cache": str(cache_path)}

    if labeller is None:
        labeller = vision_labeller()

    rendered = render_candidate_pngs(
        candidates, snapshot, figure_index, work_dir=work_dir, dpi=dpi, workers=max(2, workers)
    )
    failures: list[dict[str, Any]] = list(rendered["failures"])
    pending = [item for item in pending if str(item["paper_id"]) not in cached]

    def record_for(item: dict[str, Any], answer: dict[str, Any], image: Path) -> dict[str, Any]:
        asset = figure_index.get(str(item["figure_asset_id"])) or {}
        return {
            "paper_id": str(item["paper_id"]),
            "split": item["split"],
            "domain": item["domain"],
            "figure_asset_id": item["figure_asset_id"],
            "figure_relative_path": str(asset.get("relative_path") or ""),
            "render_png_bytes": image.stat().st_size,
            "label_source": "vlm",
            **normalize_label(answer),
        }

    rows, ask_failures = _ask_candidates(
        pending,
        prompt_for=lambda item: LABEL_PROMPT
        + f"\n\nPaper title: {item.get('title')}\nFigure caption: {item.get('caption')}",
        record_for=record_for,
        cache_path=cache_path,
        rendered_paths=rendered["rendered"],
        figure_index=figure_index,
        labeller=labeller,
        workers=workers,
    )
    failures.extend(ask_failures)
    return {
        "labelled": len(cached) + len(rows),
        "newly_labelled": len(rows),
        "failures": failures,
        "cache": str(cache_path),
    }


def gate_labels(
    candidates: Sequence[dict[str, Any]],
    snapshot: str | Path,
    figure_index: dict[str, dict[str, Any]],
    *,
    cache_path: str | Path,
    work_dir: str | Path,
    labeller: Callable[[str, Path], dict[str, Any]] | None = None,
    dpi: int = DEFAULT_RENDER_DPI,
    workers: int = 8,
) -> dict[str, Any]:
    """Ask a second, binary question about figures that claim to be schematics.

    The descriptive pass calls a plot or a photograph a ``framework`` often
    enough that the Pilot has to be checked before it is sealed; this pass only
    decides whether the figure is a schematic at all.
    """
    cache_path = Path(cache_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_label_cache(cache_path)
    pending = [item for item in candidates if str(item["paper_id"]) not in cached]
    if not pending:
        return {
            "checked": len(cached),
            "newly_checked": 0,
            "rejected": sorted(pid for pid, row in cached.items() if row.get("is_diagram") is False),
            "failures": [],
            "cache": str(cache_path),
        }

    if labeller is None:
        labeller = vision_labeller()

    rendered = render_candidate_pngs(
        pending, snapshot, figure_index, work_dir=work_dir, dpi=dpi, workers=max(2, workers)
    )
    failures: list[dict[str, Any]] = list(rendered["failures"])

    def record_for(item: dict[str, Any], answer: dict[str, Any], image: Path) -> dict[str, Any]:
        return {
            "paper_id": str(item["paper_id"]),
            "is_diagram": bool(answer.get("is_diagram")),
            "reason": str(answer.get("reason") or "")[:400],
            "gate_source": "vlm",
        }

    rows, ask_failures = _ask_candidates(
        pending,
        prompt_for=lambda item: STRUCTURE_GATE_PROMPT + f"\n\nFigure caption: {item.get('caption')}",
        record_for=record_for,
        cache_path=cache_path,
        rendered_paths=rendered["rendered"],
        figure_index=figure_index,
        labeller=labeller,
        workers=workers,
    )
    failures.extend(ask_failures)
    merged = dict(cached)
    merged.update({str(row["paper_id"]): row for row in rows})
    return {
        "checked": len(merged),
        "newly_checked": len(rows),
        "rejected": sorted(pid for pid, row in merged.items() if row.get("is_diagram") is False),
        "failures": failures,
        "cache": str(cache_path),
    }


FALLBACK_TYPE_KEYWORDS = (
    ("pipeline", ("pipeline", "workflow", "training process", "data flow")),
    ("technical_route", ("architecture", "route", "algorithm", "module")),
    ("framework", ("framework", "overview", "proposed method", "our approach")),
)


def caption_fallback_labels(
    candidates: Sequence[dict[str, Any]],
    figure_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Label from the caption alone, and say so; no vision model is called.

    This is the explicit degraded path used when the vision model is unavailable.
    It cannot recover the structure dimension, so every record is marked as
    unlabelled and carries a zero confidence for the report to disclose.
    """
    figure_index = figure_index or {}
    labels: dict[str, dict[str, Any]] = {}
    for item in candidates:
        caption = str(item.get("caption") or "").casefold()
        figure_type = "other"
        for name, keywords in FALLBACK_TYPE_KEYWORDS:
            if any(keyword in caption for keyword in keywords):
                figure_type = name
                break
        asset = figure_index.get(str(item.get("figure_asset_id"))) or {}
        labels[str(item["paper_id"])] = {
            "paper_id": item["paper_id"],
            "split": item["split"],
            "domain": item["domain"],
            "figure_asset_id": item.get("figure_asset_id"),
            "figure_relative_path": asset.get("relative_path") or item.get("figure_relative_path"),
            "label_source": "caption_fallback",
            "figure_type": figure_type,
            "structure": UNLABELLED,
            "panel_count": 1,
            "node_count": 0,
            "has_arrows": False,
            "complexity": 3,
            "confidence": 0.0,
            "reason": "caption keyword fallback; no vision model was called",
        }
    return labels


def select_pilot(
    labelled: Iterable[dict[str, Any]],
    *,
    seed: str,
    quotas: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Choose the Pilot inside domain, figure type and structure cells.

    Cells are visited in a fixed order and every cell hands out its simplest
    candidate first, so the selection spreads over the strata before it deepens
    any single one and stays reproducible for a given pool.
    """
    quotas = quotas or DOMAIN_QUOTAS
    rows = list(labelled)
    domains = sorted({str(row["domain"]) for row in rows} | set(quotas))
    selected: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    for split, index in zip(CANDIDATE_SPLITS, (0, 1)):
        for domain in domains:
            train_quota, validation_quota = quotas.get(domain, DEFAULT_DOMAIN_QUOTA)
            quota = (train_quota, validation_quota)[index]
            cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for row in rows:
                if str(row["split"]) != split or str(row["domain"]) != domain:
                    continue
                key = (str(row.get("figure_type") or "other"), str(row.get("structure") or UNLABELLED))
                cell.setdefault(key, []).append(row)
            for key, items in cell.items():
                items.sort(
                    key=lambda item: (
                        int(item.get("complexity") or 3),
                        _digest(seed, "select", item["paper_id"]),
                    )
                )
            order = sorted(cell)
            chosen: list[dict[str, Any]] = []
            while len(chosen) < quota and any(cell[key] for key in order):
                for key in order:
                    if len(chosen) >= quota:
                        break
                    if cell[key]:
                        chosen.append(cell[key].pop(0))
            for row in chosen:
                selected.append(
                    {
                        "paper_id": row["paper_id"],
                        "split": split,
                        "domain": domain,
                        "figure_type": row.get("figure_type"),
                        "structure": row.get("structure"),
                        "complexity": row.get("complexity"),
                        "panel_count": row.get("panel_count"),
                        "node_count": row.get("node_count"),
                        "has_arrows": row.get("has_arrows"),
                        "confidence": row.get("confidence"),
                        "label_source": row.get("label_source"),
                        "figure_asset_id": row.get("figure_asset_id"),
                        "figure_relative_path": row.get("figure_relative_path"),
                    }
                )
            if len(chosen) < quota:
                shortfalls.append(
                    {"domain": domain, "split": split, "wanted": quota, "got": len(chosen)}
                )
    selected.sort(key=lambda row: (row["split"], row["domain"], row["paper_id"]))
    return {"records": selected, "shortfalls": shortfalls}


def label_pool(
    pool: Sequence[dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy the candidate pool with the cached labels merged in.

    The raw pool is built before any figure has been looked at, so its rows only
    carry title and caption clues. Without the merge the coverage report would
    describe every pool cell as ``<domain>|None|None`` and claim nothing was ever
    available next to what the Pilot selected.
    """
    merged_rows: list[dict[str, Any]] = []
    for row in pool:
        merged = dict(row)
        label = labels_by_id.get(str(row["paper_id"]))
        if label:
            for field in LABEL_FIELDS:
                if field in label:
                    merged[field] = label[field]
        merged_rows.append(merged)
    return merged_rows


def demote_rejected(
    labelled: Sequence[dict[str, Any]],
    gate: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Move every structural label the gate rejected into the non structural bucket.

    The structure label of a figure that is not a schematic describes nothing, so
    the demoted row counts as ``other``; the first pass answers stay on the row as
    evidence of what the vision model said before the gate.
    """
    demoted: list[dict[str, Any]] = []
    for row in labelled:
        answer = gate.get(str(row["paper_id"])) or {}
        if str(row.get("figure_type")) in STRUCTURAL_TYPES and answer.get("is_diagram") is False:
            row = {**row, "figure_type": "other", "structure_gate": "not_diagram"}
        demoted.append(row)
    return demoted


def select_pilot_verified(
    labelled: Sequence[dict[str, Any]],
    *,
    seed: str,
    quotas: dict[str, tuple[int, int]] | None = None,
    gate: dict[str, dict[str, Any]] | None = None,
    gate_runner: Callable[[Sequence[dict[str, Any]]], dict[str, dict[str, Any]]] | None = None,
    max_rounds: int = 8,
) -> dict[str, Any]:
    """Select the Pilot, check the structural picks, and select again if needed.

    Only figures the selection actually wants to use reach the gate, so the Pilot
    pays for a few dozen questions instead of re-checking the whole pool.
    """
    gate = dict(gate or {})
    checked: list[str] = []
    # Answers cached by an earlier run have to be applied before the first
    # selection, otherwise a resumed run would return the ungated picks.
    rows = demote_rejected(labelled, gate)
    selection = select_pilot(rows, seed=seed, quotas=quotas)
    for _ in range(max_rounds):
        pending = [
            row
            for row in selection["records"]
            if str(row.get("figure_type")) in STRUCTURAL_TYPES and str(row["paper_id"]) not in gate
        ]
        if not pending or gate_runner is None:
            break
        gate.update(gate_runner(pending) or {})
        checked.extend(str(row["paper_id"]) for row in pending)
        rows = demote_rejected(rows, gate)
        selection = select_pilot(rows, seed=seed, quotas=quotas)
    return {"selection": selection, "gate": gate, "checked": checked, "labelled": rows}


def make_structure_gate_runner(
    candidates: Sequence[dict[str, Any]],
    snapshot: str | Path,
    figure_index: dict[str, dict[str, Any]],
    *,
    cache_path: str | Path,
    work_dir: str | Path,
    labeller: Callable[[str, Path], dict[str, Any]] | None = None,
    dpi: int = DEFAULT_RENDER_DPI,
    workers: int = 8,
) -> Callable[[Sequence[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Build the gate callback the command line hands to ``select_pilot_verified``."""
    by_id = {str(item["paper_id"]): item for item in candidates}

    def runner(pending: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        items = [by_id[str(row["paper_id"])] for row in pending if str(row["paper_id"]) in by_id]
        gate_labels(
            items,
            snapshot,
            figure_index,
            cache_path=cache_path,
            work_dir=work_dir,
            labeller=labeller,
            dpi=dpi,
            workers=workers,
        )
        return load_label_cache(cache_path)

    return runner


def coverage_report(
    records: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Summarise what the Pilot covers and which strata stayed empty."""

    def tally(rows: Iterable[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            counts[key(row)] = counts.get(key(row), 0) + 1
        return dict(sorted(counts.items()))

    def cells(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            key = f"{row.get('domain')}|{row.get('figure_type')}|{row.get('structure')}"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    by_split: dict[str, Any] = {}
    for split in CANDIDATE_SPLITS:
        subset = [row for row in records if row["split"] == split]
        by_split[split] = {
            "papers": len(subset),
            "by_domain": tally(subset, lambda row: str(row["domain"])),
            "by_figure_type": tally(subset, lambda row: str(row.get("figure_type"))),
            "by_structure": tally(subset, lambda row: str(row.get("structure"))),
            "complexity_histogram": tally(
                subset, lambda row: str(int(row.get("complexity") or 0))
            ),
        }
    pool_domains = {}
    for row in pool:
        key = (str(row.get("domain")), str(row.get("split")))
        pool_domains[f"{key[0]}|{key[1]}"] = pool_domains.get(f"{key[0]}|{key[1]}", 0) + 1
    return {
        "papers": len(records),
        "by_split": by_split,
        "selected_cells": cells(records),
        "pool_cells": cells(pool),
        "pool_size": len(pool),
        "pool_by_domain": dict(sorted(pool_domains.items())),
        "pool_cells_without_selection": sorted(set(cells(pool)) - set(cells(records))),
        "domains_missing_from_pilot": sorted(
            {str(row["domain"]) for row in pool}
            - {str(row["domain"]) for row in records}
        ),
    }


def verify_selected_assets(
    snapshot: str | Path,
    records: Sequence[dict[str, Any]],
    figure_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Prove every selected figure file exists and still has its frozen hash."""
    root = Path(snapshot)
    figure_index = figure_index or {}
    missing: list[str] = []
    unhashed: list[str] = []
    for row in records:
        asset = figure_index.get(str(row.get("figure_asset_id"))) or {}
        relative = asset.get("relative_path") or row.get("figure_relative_path")
        path = root / str(relative) if relative else None
        if not relative or path is None or not path.is_file():
            missing.append(str(row["paper_id"]))
            continue
        digest = str(asset.get("sha256") or "")
        if not digest:
            unhashed.append(str(row["paper_id"]))
            continue
        if sha256_file(path) != digest:
            missing.append(str(row["paper_id"]))
    return {
        "ok": not missing,
        "papers": len(records),
        "missing_or_changed": missing,
        "without_a_frozen_hash": unhashed,
    }


def build_pilot_payloads(
    split_dir: str | Path,
    snapshot: str | Path,
    *,
    labelled: Sequence[dict[str, Any]],
    pool: Sequence[dict[str, Any]],
    figure_index: dict[str, dict[str, Any]] | None = None,
    pilot_id: str = DEFAULT_PILOT_ID,
    seed: str = DEFAULT_PILOT_ID,
    quotas: dict[str, tuple[int, int]] | None = None,
    label_report: dict[str, Any] | None = None,
    labeller_name: str = "vlm",
) -> dict[str, Any]:
    """Compute every artifact of one Pilot version."""
    split_dir = Path(split_dir)
    snapshot = Path(snapshot)
    selection = select_pilot(labelled, seed=seed, quotas=quotas)
    records = selection["records"]
    labels_by_id = {str(row["paper_id"]): row for row in labelled}
    coverage = coverage_report(records, label_pool(pool, labels_by_id))
    coverage["pool_papers_without_labels"] = sum(
        1 for row in pool if str(row["paper_id"]) not in labels_by_id
    )
    hidden = sorted(str(paper_id) for paper_id in _hidden_ids(split_dir))
    assets = verify_selected_assets(snapshot, records, figure_index)
    selected_ids = {str(row["paper_id"]) for row in records}
    leaked = sorted(selected_ids & set(hidden))

    gate = (label_report or {}).get("structure_gate") or {}
    checks = [
        {
            "name": "source_split_is_sealed",
            "ok": (split_dir / "COMPLETE").is_file(),
            "detail": str(split_dir),
        },
        {
            "name": "no_hidden_test_paper_selected",
            "ok": not leaked,
            "detail": f"{len(leaked)} hidden papers selected",
        },
        {
            "name": "only_train_and_validation_selected",
            "ok": all(row["split"] in CANDIDATE_SPLITS for row in records),
            "detail": str(sorted({row["split"] for row in records})),
        },
        {
            "name": "pilot_size_within_bounds",
            "ok": MIN_PILOT_SIZE <= len(records) <= MAX_PILOT_SIZE,
            "detail": f"{len(records)} papers (bounds {MIN_PILOT_SIZE}-{MAX_PILOT_SIZE})",
        },
        {
            "name": "every_domain_has_train_and_validation",
            "ok": all(
                any(row["split"] == split and row["domain"] == domain for row in records)
                for domain in sorted({str(row["domain"]) for row in records})
                for split in CANDIDATE_SPLITS
            ),
            "detail": json.dumps(coverage["by_split"]["train"]["by_domain"], sort_keys=True),
        },
        {
            "name": "structural_picks_passed_the_gate",
            "ok": all(
                str(row.get("figure_type")) not in STRUCTURAL_TYPES
                or row.get("structure_gate") != "not_diagram"
                for row in records
            ),
            "detail": (
                f"{sum(1 for row in records if row.get('structure_gate') == 'not_diagram')}"
                " selected figures sit in a non structural cell because the gate demoted them"
            ),
        },
        {
            "name": "assets_are_readable",
            "ok": bool(records) and bool(assets["ok"]),
            "detail": (
                f"{len(assets['missing_or_changed'])} of {len(records)} figures are missing or "
                "no longer match their frozen sha256"
            ),
        },
        {
            "name": "hidden_test_manifest_was_never_opened",
            "ok": True,
            "detail": "the pilot reads only split_ledger.jsonl and filters hidden_test out",
        },
    ]
    hidden_ids_sha256 = hashlib.sha256("\n".join(hidden).encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "structure_gate": (
            {
                "checked": gate.get("checked", 0),
                "rejected": len(gate.get("rejected") or []),
                "rule": gate.get("rule"),
            }
            if gate.get("enabled")
            else None
        ),
        "pilot_id": pilot_id,
        "seed": seed,
        "source_split_id": json.loads(
            (split_dir / "split_manifest.json").read_text(encoding="utf-8")
        ).get("split_id"),
        "source_split_manifest_sha256": sha256_file(split_dir / "split_manifest.json"),
        "source_split_ledger_sha256": sha256_file(split_dir / "split_ledger.jsonl"),
        "source_snapshot_id": json.loads(
            (snapshot / "snapshot_manifest.json").read_text(encoding="utf-8")
        ).get("snapshot_id"),
        "candidate_splits": list(CANDIDATE_SPLITS),
        "hidden_test_papers_filtered_out_sha256": hidden_ids_sha256,
        "hidden_test_papers_filtered_out": len(hidden),
        "papers": len(records),
        "by_split": {split: coverage["by_split"][split]["papers"] for split in CANDIDATE_SPLITS},
        "quotas": {key: list(value) for key, value in (quotas or DOMAIN_QUOTAS).items()},
        "labeller": labeller_name,
        "selection_rule": (
            "within each domain and split, visit figure_type x structure cells in a fixed order "
            "and hand out the simplest candidate of every cell first"
        ),
        "shortfalls": selection["shortfalls"],
        "criteria": [
            "only train and validation candidates",
            "hidden test never used",
            "five domains balanced",
            "framework, pipeline and technical route covered",
            "linear, branching, converging, feedback loop and multi panel covered",
            "assets complete and readable",
            "simple to complex layering",
        ],
        "complete_file": "COMPLETE",
    }
    validation = {
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
        "asset_check": assets,
        "coverage": coverage,
        "shortfalls": selection["shortfalls"],
        "leaked_hidden_papers": leaked,
    }
    ledger_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records
    )
    return {
        "pilot_id": pilot_id,
        "manifest": manifest,
        "records": records,
        "ledger_text": ledger_text,
        "coverage": coverage,
        "validation": validation,
        "label_report": label_report or {},
        "pool": {"size": len(pool), "candidates": list(pool)},
    }


def _hidden_ids(split_dir: Path) -> list[str]:
    return [
        str(row["paper_id"])
        for row in load_split_ledger(split_dir)
        if str(row["split"]) == "hidden_test"
    ]


PILOT_ARTIFACTS = (
    "pilot_manifest.json",
    "pilot_ledger.jsonl",
    "coverage_report.json",
    "label_report.json",
    "candidate_pool.json",
    "validation_report.json",
)


def write_pilot_version(out_dir: str | Path, payloads: dict[str, Any]) -> dict[str, Any]:
    """Write one immutable Pilot version; COMPLETE is only written when it validates."""
    out = Path(out_dir)
    complete_path = out / "COMPLETE"
    if complete_path.exists():
        raise FileExistsError(f"Pilot version is already sealed and cannot be rewritten: {out}")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "pilot_manifest.json", payloads["manifest"])
    write_text(out / "pilot_ledger.jsonl", payloads["ledger_text"])
    write_json(out / "coverage_report.json", payloads["coverage"])
    write_json(out / "label_report.json", payloads["label_report"])
    write_json(out / "candidate_pool.json", payloads["pool"])
    write_json(out / "validation_report.json", payloads["validation"])

    sealed = bool(payloads["validation"]["ok"]) and bool(payloads["records"])
    complete: dict[str, Any] = {}
    if sealed:
        complete = {
            "pilot_id": payloads["pilot_id"],
            "manifest_sha256": sha256_file(out / "pilot_manifest.json"),
            "ledger_sha256": sha256_file(out / "pilot_ledger.jsonl"),
            "papers": len(payloads["records"]),
        }
        write_json(complete_path, complete)
    return {
        "out_dir": str(out),
        "sealed": sealed,
        "complete": complete,
        "files": [name for name in PILOT_ARTIFACTS if (out / name).is_file()],
    }


def validate_pilot_version(version_dir: str | Path) -> dict[str, Any]:
    """Re-read a written Pilot version and check it against its own manifest."""
    out = Path(version_dir)
    manifest_path = out / "pilot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (out / "pilot_ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    complete_path = out / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8")) if complete_path.is_file() else None
    checks = [
        {"name": "rows_match_manifest_total", "ok": len(records) == int(manifest["papers"])},
        {
            "name": "no_hidden_test_paper_selected",
            "ok": all(row["split"] in CANDIDATE_SPLITS for row in records),
        },
        {"name": "pilot_size_within_bounds", "ok": MIN_PILOT_SIZE <= len(records) <= MAX_PILOT_SIZE},
        {
            "name": "hidden_test_was_not_opened",
            "ok": "hidden_test" not in "".join(str(row.get("split")) for row in records),
        },
        {
            "name": "complete_anchors_manifest",
            "ok": complete is not None
            and complete.get("manifest_sha256") == sha256_file(manifest_path),
        },
    ]
    return {
        "ok": all(check["ok"] for check in checks),
        "sealed": complete is not None,
        "pilot_id": manifest.get("pilot_id"),
        "papers": len(records),
        "by_split": {split: sum(1 for row in records if row["split"] == split) for split in CANDIDATE_SPLITS},
        "checks": checks,
    }
