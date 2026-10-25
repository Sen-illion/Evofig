"""Collect the reference-versus-generated examples of a whole wave of sealed-contract runs.

One paper's pair is built by :mod:`.real_vs_generated`; a wave is five thousand of them, and
at that size the only questions worth asking are which papers produced a sample, which were
refused and why, and whether the refusals cluster into one fixable cause.  A refused paper
is not a failure of the batch - the ground truth of the comparison is the paper's own
published figure, and a candidate the run itself certified carries no defect to name - so
the refusals are accounted for instead of raised.

The subset the wave ran is read from its own sealed files (``subset_manifest.json`` and
``subset_ledger.jsonl``) rather than through the workflow that wrote them: this module sits
in the Judge data package, which only consumes produced artifacts.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..utils import ensure_dir, write_json, write_text
from .preference_sources import SourceUnavailable, open_generation_source, open_middleware_source
from .real_vs_generated import build_real_vs_generated_examples

DELIVERED_DECK = "editable_composition.pptx"
ATTEMPT_PREFIX = "attempt_"
LEDGER_FILE = "subset_ledger.jsonl"
MANIFEST_FILE = "subset_manifest.json"
PAPERS_DIR = "papers"
EXAMPLES_FILE = "examples.jsonl"
MERGED_FILE = "examples_merged.jsonl"
MANIFEST_OUT = "examples_manifest.json"
REPORT_OUT = "wave_report.json"
ADOPTION_FILE = "upstream_contract_report.json"
SWAP_SUFFIX = "__swap"

#: Split labels a pair may carry.  A hidden test paper is drawn like any other - the
#: product has to work on every paper - but its samples may never train a Judge, so the
#: ledger label is mapped onto the one split that keeps them out.
TRAINING_SPLITS = ("train", "validation")
HIDDEN_TEST_SPLIT = "hidden_test"

FAILURE_CODES = (
    ("disagree about the scientific contract", "contract_mismatch"),
    ("carries a Grounding Gate verdict", "no_gate_blocked_candidate"),
    ("no graph folder for this run", "middleware_has_no_folder_for_the_run"),
    ("must hold exactly one rendered figure", "reference_figure_is_ambiguous"),
    ("has no rendered figures", "reference_figure_is_missing"),
    ("has no round_XX/candidate_XX", "run_has_no_candidate"),
    ("does not record which paper", "run_does_not_record_its_paper"),
    ("published no scientific ground truth", "run_has_no_contract"),
)


def delivered_attempt(paper_root: str | Path) -> Path | None:
    """The attempt of one paper that actually delivered a figure.

    A wave can hand a paper the icons an earlier adjudication already paid for by placing
    them in the directory its first attempt would have written, which leaves that directory
    looking like an attempt the paper already spent.  The figure therefore lives in the
    newest attempt that carries one, and reading a fixed attempt name instead finds either
    no run at all or the staged directory the run never used.
    """

    root = Path(paper_root)
    if not root.is_dir():
        return None
    delivered: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith(ATTEMPT_PREFIX):
            continue
        suffix = child.name[len(ATTEMPT_PREFIX) :]
        if suffix.isdigit() and (child / DELIVERED_DECK).is_file():
            delivered.append((int(suffix), child))
    if not delivered:
        return None
    return max(delivered, key=lambda item: item[0])[1]


def training_split(value: Any) -> str:
    """Map a split ledger label onto the split a Judge example may carry."""
    label = str(value or "").strip()
    if label in TRAINING_SPLITS:
        return label
    if label == HIDDEN_TEST_SPLIT:
        return "test"
    if label == "test":
        return "test"
    return "train"


def load_subset_ledger(subset: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a sealed subset version: its manifest and its per paper ledger."""
    source = Path(subset)
    if source.is_file():
        source = source.parent
    manifest_path = source / MANIFEST_FILE
    ledger_path = source / LEDGER_FILE
    if not ledger_path.is_file():
        raise FileNotFoundError(f"subset version is missing {LEDGER_FILE}: {ledger_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return manifest, rows


def failure_reason(exc: BaseException) -> str:
    """Reduce one refusal to a stable code, so the wave's reasons group together."""
    text = str(exc)
    for needle, code in FAILURE_CODES:
        if needle in text:
            return code
    return f"{type(exc).__name__}: {text[:120]}"


def contract_adoption(run_dir: str | Path) -> str:
    """Report whether a run adopted the middleware contract it was pointed at."""
    path = Path(run_dir) / ADOPTION_FILE
    if not path.is_file():
        return "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return "unreadable"
    if not isinstance(payload, dict):
        return "unreadable"
    return "adopted" if payload.get("used") else "declined"


def namespace_examples(name: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefix one paper's example ids with the paper, so a wave cannot collide with itself.

    The builder names an example after the run directory it read, and every run of a wave is
    called ``attempt_01``: without the prefix every paper of the wave would mint the same two
    or three ids, and a merge would keep one paper's sample per id and silently drop the rest.
    """

    named: list[dict[str, Any]] = []
    for row in rows:
        example_id = str(row.get("example_id") or "")
        if not example_id:
            continue
        named.append({**row, "example_id": f"{name}-{example_id}"})
    return named


def _jsonl_line(row: dict[str, Any]) -> str:
    """Encode one example as a single physical line.

    A quote can contain U+2028 or U+2029. ``str.splitlines`` treats those as line
    breaks, so a later reader would see an unterminated JSON string. The escapes
    keep every example on the one ``\\n``-terminated line it was written as.
    """

    return json.dumps(row, ensure_ascii=False).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def write_examples(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write one example JSONL file in the canonical one-object-per-line form."""

    target = Path(path)
    ensure_dir(target.parent)
    write_text(target, "".join(_jsonl_line(row) + "\n" for row in rows))


def read_examples(path: str | Path) -> list[dict[str, Any]]:
    """Read one example JSONL file, tolerating an absent file."""
    source = Path(path)
    if not source.is_file():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").split("\n") if line.strip()]


def merge_examples(base_path: str | Path, new_rows: Iterable[dict[str, Any]], out_path: str | Path) -> dict[str, Any]:
    """Write the base examples plus the new ones, one row per example id.

    The first occurrence of an id wins and the rest are counted, so a merge can never
    silently duplicate a training row or drop one without saying so.
    """
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    base_rows = read_examples(base_path)
    for row in list(base_rows) + list(new_rows):
        example_id = str(row.get("example_id") or "")
        if not example_id:
            continue
        if example_id in merged:
            continue
        merged[example_id] = row
        order.append(example_id)
    target = Path(out_path)
    ensure_dir(target.parent)
    write_text(target, "".join(json.dumps(merged[key], ensure_ascii=False) + "\n" for key in order))
    return {
        "out": str(target),
        "base": str(base_path),
        "base_examples": len(base_rows),
        "new_examples": len(list(new_rows)),
        "merged_examples": len(order),
    }


def collect_reference_pairs(
    subset: str | Path,
    middleware: str | Path,
    wave_root: str | Path,
    out: str | Path,
    *,
    attempt: str | None = None,
    include_swapped: bool = True,
    include_passed: bool = False,
    limit: int = 0,
    merge_into: str | Path | None = None,
) -> dict[str, Any]:
    """Build every paper's reference pair that this wave can carry, and account for the rest.

    ``middleware`` is the sealed version directory itself (the one the runs adopted their
    contract from), not the per paper export: the adopted copy is compared against the
    version's own ``scientific_ground_truth.json``, so the version is the reference.

    ``attempt`` names the directory to read under each run; left empty it is the newest
    attempt that delivered a figure, which is the one a wave actually paid for.
    """
    manifest, entries = load_subset_ledger(subset)
    truth = open_middleware_source(middleware)
    out_dir = ensure_dir(out)
    papers_dir = ensure_dir(Path(out_dir) / PAPERS_DIR)

    examples: list[dict[str, Any]] = []
    accounted: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    adoption: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    defects: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    built = 0
    duplicates = 0
    seen: set[str] = set()
    for entry in entries:
        paper_id = str(entry.get("paper_id") or "")
        if not paper_id:
            continue
        name = str(entry.get("directory") or "") or str(entry.get("paper_id") or "")
        paper_root = Path(wave_root) / "runs" / name
        run_dir = (paper_root / attempt) if attempt else delivered_attempt(paper_root)
        if run_dir is None or not run_dir.is_dir():
            reasons["wave_has_no_run"] += 1
            accounted.append({"paper": name, "examples": 0, "reason": "wave_has_no_run"})
            continue
        adoption[contract_adoption(run_dir)] += 1
        split = training_split(entry.get("split"))
        try:
            report = build_real_vs_generated_examples(
                truth,
                open_generation_source(run_dir),
                papers_dir / f"{name}.jsonl",
                split=split,
                include_swapped=bool(include_swapped),
                include_passed=bool(include_passed),
                limit=int(limit),
            )
        except (SourceUnavailable, ValueError, OSError) as exc:
            reason = failure_reason(exc)
            reasons[reason] += 1
            accounted.append({"paper": name, "examples": 0, "reason": reason})
            continue
        # The per paper file is rewritten with the paper's own namespace, so the wave's two
        # outputs - one file per paper and the merged file - carry the same ids.
        rows = namespace_examples(name, read_examples(papers_dir / f"{name}.jsonl"))
        write_examples(papers_dir / f"{name}.jsonl", rows)
        for row in rows:
            example_id = str(row.get("example_id") or "")
            if example_id in seen:
                duplicates += 1
                continue
            seen.add(example_id)
            examples.append(row)
        built += 1
        splits[split] += 1
        for key, value in (report.get("tier_distribution") or {}).items():
            tiers[str(key)] += int(value)
        for key, value in (report.get("defect_distribution") or {}).items():
            defects[str(key)] += int(value)
        routes[str((report.get("contract") or {}).get("route") or "unknown")] += 1
        accounted.append({"paper": name, "examples": len(rows), "split": split,
                          "tiers": report.get("tier_distribution") or {},
                          "contract_route": (report.get("contract") or {}).get("route")})

    write_examples(Path(out_dir) / EXAMPLES_FILE, examples)
    payload = {
        "contract": "rfs.judge_example_set",
        "summary": "Reference-versus-generated examples collected from one generation wave.",
        "subset_id": str(manifest.get("subset_id") or ""),
        "subset": str(subset),
        "middleware": str(middleware),
        "wave_root": str(wave_root),
        "attempt": str(attempt),
        "include_swapped": bool(include_swapped),
        "include_passed_candidates": bool(include_passed),
        "counts": {
            "papers": len(entries),
            "papers_with_examples": built,
            "examples": len(examples),
            "duplicate_example_ids": duplicates,
        },
        "by_split": dict(sorted(splits.items())),
        "by_tier": dict(sorted(tiers.items())),
        "by_defect": dict(sorted(defects.items())),
        "by_contract_route": dict(sorted(routes.items())),
        "contract_adoption": dict(sorted(adoption.items())),
        "refusals": dict(sorted(reasons.items())),
        "papers": accounted,
        "out": str(out_dir),
    }
    write_json(Path(out_dir) / MANIFEST_OUT, payload)
    merged = merge_examples(merge_into, examples, Path(out_dir) / MERGED_FILE) if merge_into else None
    report = {
        "ok": True,
        "summary": (
            f"{len(examples)} examples from {built} of {len(entries)} papers; "
            f"{sum(reasons.values())} papers were refused."
        ),
        "out": str(out_dir),
        "examples": str(Path(out_dir) / EXAMPLES_FILE),
        "manifest": str(Path(out_dir) / MANIFEST_OUT),
        "papers": len(entries),
        "papers_with_examples": built,
        "examples_written": len(examples),
        "duplicate_example_ids": duplicates,
        "by_split": dict(sorted(splits.items())),
        "by_tier": dict(sorted(tiers.items())),
        "by_defect": dict(sorted(defects.items())),
        "by_contract_route": dict(sorted(routes.items())),
        "contract_adoption": dict(sorted(adoption.items())),
        "refusals": dict(sorted(reasons.items())),
        "merged": merged,
        "sample": accounted[:20],
    }
    write_json(Path(out_dir) / REPORT_OUT, report)
    return report
