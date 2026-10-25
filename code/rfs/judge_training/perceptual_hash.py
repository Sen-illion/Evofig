"""Render every figure asset of a snapshot and hash it for leakage detection.

Stage 2 of the Judge execution roadmap.  A split may only be sealed when no figure
is shared by papers that sit in two different splits.  Byte identity alone is not
enough: the same published figure appears both as a PDF fragment in one record and
as a PNG in another, so every figure of the snapshot is rendered and reduced to a
64-bit dHash.

This module writes a pure fact file.  It records which rendered figures are the
same image and nothing else; deciding whether that identity crosses a split
boundary belongs to the split ledger.  The report carries no timing, host, user or
absolute path, so two runs over one snapshot produce identical bytes.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image

from rfs.utils import write_json

from .source_snapshot_io import sha256_file

SCHEMA_VERSION = "judge-perceptual-hash-v1"
HASH_NAME = "dhash64"
DEFAULT_DPI = 72
DEFAULT_NEAR_DISTANCE = 4
DEFAULT_WORKERS = 8
MAX_LISTED_GROUPS = 5000
MAX_LISTED_NEAR_PAIRS = 2000
MAX_SCANNED_NEAR_PAIRS = 200000
_HASH_BYTES = 8
_POPCOUNT = np.array([bin(value).count("1") for value in range(256)], dtype=np.uint8)


def _render_first_representation(path: Path, dpi: int) -> Image.Image:
    """Render the first visual representation of one figure asset as RGB."""
    if path.suffix.lower() == ".pdf":
        import pypdfium2

        with pypdfium2.PdfDocument(str(path)) as document:
            page = document[0]
            try:
                bitmap = page.render(scale=dpi / 72)
                try:
                    return bitmap.to_pil().convert("RGB")
                finally:
                    bitmap.close()
            finally:
                page.close()
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def dhash64(image: Image.Image) -> bytes:
    """Return the 64-bit difference hash of one rendered figure."""
    grey = image.convert("L").resize((9, 8), Image.LANCZOS)
    pixels = list(grey.tobytes())
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return bits.to_bytes(_HASH_BYTES, "big")


def _hash_one(task: tuple[str, str, int]) -> tuple[str, str | None, str | None]:
    """Hash one figure; every failure is returned, never swallowed."""
    root, relative, dpi = task
    try:
        image = _render_first_representation(Path(root) / relative, dpi)
        try:
            return relative, dhash64(image).hex(), None
        finally:
            image.close()
    except Exception as exc:  # noqa: BLE001 - the report must show every failure
        return relative, None, f"{type(exc).__name__}: {exc}"


def figure_assets(snapshot: str | Path) -> list[dict[str, Any]]:
    """Return every figure asset of a snapshot sorted by paper then path."""
    path = Path(snapshot) / "asset_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot is missing asset_manifest.jsonl: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("role") != "figure":
                continue
            rows.append(
                {
                    "paper_id": str(record.get("paper_id") or ""),
                    "asset_id": str(record.get("asset_id") or ""),
                    "relative_path": str(record.get("relative_path") or ""),
                    "sha256": str(record.get("sha256") or ""),
                    "suffix": str(record.get("suffix") or ""),
                }
            )
    rows.sort(key=lambda row: (row["paper_id"], row["relative_path"]))
    return rows


def near_pairs(hashes: Iterable[str], distance: int, *, chunk: int = 128) -> dict[str, Any]:
    """Return every index pair of dHashes within a Hamming distance."""
    values = list(hashes)
    result: dict[str, Any] = {"pairs": [], "total": 0, "truncated": False}
    if distance < 0 or len(values) < 2:
        return result
    packed = np.frombuffer(b"".join(bytes.fromhex(value) for value in values), dtype=np.uint8)
    packed = packed.reshape(-1, _HASH_BYTES)
    for start in range(0, len(packed), chunk):
        block = packed[start : start + chunk]
        xor = np.bitwise_xor(block[:, None, :], packed[None, :, :])
        totals = _POPCOUNT[xor].sum(axis=2)
        rows, columns = np.nonzero(totals <= distance)
        for row, column in zip(rows.tolist(), columns.tolist()):
            index = start + row
            if index >= column:
                continue
            result["total"] += 1
            if len(result["pairs"]) < MAX_LISTED_NEAR_PAIRS:
                result["pairs"].append(
                    {"left": index, "right": column, "distance": int(totals[row, column])}
                )
            elif result["total"] > MAX_SCANNED_NEAR_PAIRS:
                result["truncated"] = True
                return result
    return result


def _asset_facts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "paper_id": row["paper_id"],
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]


def build_perceptual_hash_report(
    snapshot: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    workers: int = DEFAULT_WORKERS,
    near_distance: int = DEFAULT_NEAR_DISTANCE,
    hasher: Callable[[tuple[str, str, int]], tuple[str, str | None, str | None]] | None = None,
) -> dict[str, Any]:
    """Render and hash every figure of a snapshot and return the fact report."""
    root = Path(snapshot)
    assets = figure_assets(root)
    if not assets:
        raise ValueError(f"Snapshot has no figure assets: {root}")
    manifest_path = root / "asset_manifest.jsonl"
    snapshot_manifest_path = root / "snapshot_manifest.json"
    snapshot_id = None
    if snapshot_manifest_path.is_file():
        snapshot_id = json.loads(snapshot_manifest_path.read_text(encoding="utf-8")).get("snapshot_id")

    tasks = [(str(root), row["relative_path"], dpi) for row in assets]
    worker = hasher or _hash_one
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(worker, tasks, chunksize=16))
    else:
        results = [worker(task) for task in tasks]

    by_hash: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, Any]] = []
    for row, (relative, digest, error) in zip(assets, results):
        if error or not digest:
            failures.append({"relative_path": relative, "error": error or "empty hash"})
            continue
        by_hash.setdefault(digest, []).append(row)

    groups = [
        {"dhash": digest, "assets": _asset_facts(members)}
        for digest, members in sorted(by_hash.items())
        if len(members) > 1
    ]
    truncated = len(groups) > MAX_LISTED_GROUPS
    groups = groups[:MAX_LISTED_GROUPS]

    ordered: list[dict[str, Any]] = []
    ordered_hashes: list[str] = []
    for digest, members in sorted(by_hash.items()):
        for row in members:
            ordered.append(row)
            ordered_hashes.append(digest)
    near = near_pairs(ordered_hashes, near_distance)
    near_facts = []
    for pair in near["pairs"]:
        left = ordered[pair["left"]]
        right = ordered[pair["right"]]
        if left["paper_id"] == right["paper_id"]:
            continue
        near_facts.append(
            {
                "distance": pair["distance"],
                "left": {"paper_id": left["paper_id"], "relative_path": left["relative_path"]},
                "right": {"paper_id": right["paper_id"], "relative_path": right["relative_path"]},
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "asset_manifest_sha256": sha256_file(manifest_path),
        "method": {
            "hash": HASH_NAME,
            "description": "greyscale 9x8 render compared column-wise, 64 bits",
            "render": f"first representation at {dpi} dpi (pdf page 1 via pypdfium2, image via Pillow)",
            "dpi": dpi,
            "workers": workers,
            "near_distance": near_distance,
        },
        "figures_total": len(assets),
        "hashed": len(assets) - len(failures),
        "failed": len(failures),
        "failures": failures,
        "distinct_hashes": len(by_hash),
        "exact_groups": groups,
        "exact_groups_total": len(groups),
        "groups_truncated": truncated,
        "near_cross_paper_pairs": near_facts,
        "near_pairs_total": near["total"],
        "near_scan_truncated": near["truncated"],
        "ok": not failures and not truncated,
    }


def audit_perceptual_leakage(
    rows: Iterable[dict[str, Any]],
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Decide whether papers that share a rendered figure sit in two splits."""
    splits = {str(row["paper_id"]): str(row["split"]) for row in rows}

    def crossing(groups: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for group in groups or []:
            papers = sorted(
                {
                    str(asset.get("paper_id") or "")
                    for asset in group.get("assets") or []
                    if asset.get("paper_id")
                }
            )
            observed = sorted({splits[paper] for paper in papers if paper in splits})
            if len(observed) > 1:
                found.append({"papers": papers, "splits": observed, "dhash": group.get("dhash")})
        return found

    if not report:
        return {
            "ok": False,
            "supplied": False,
            "report_ok": None,
            "detail": "no perceptual hash report was supplied; the split cannot be sealed",
            "exact_cross_split_groups": [],
            "near_cross_split_pairs": [],
        }
    exact = crossing(report.get("exact_groups") or [])
    near = []
    for pair in report.get("near_cross_paper_pairs") or []:
        left = str((pair.get("left") or {}).get("paper_id") or "")
        right = str((pair.get("right") or {}).get("paper_id") or "")
        if left in splits and right in splits and splits[left] != splits[right]:
            near.append(pair)
    return {
        "ok": bool(report.get("ok")) and not exact,
        "supplied": True,
        "report_ok": bool(report.get("ok")),
        "detail": f"{len(exact)} rendered-figure groups cross a split boundary",
        "exact_cross_split_groups": exact,
        "near_cross_split_pairs": near,
        "near_cross_split_is_advisory": True,
    }


def write_perceptual_hash_report(path: str | Path, report: dict[str, Any]) -> str:
    """Write the report and return its sha256."""
    target = Path(path)
    write_json(target, report)
    return sha256_file(target)