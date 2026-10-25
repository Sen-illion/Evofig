"""Adopt decks an earlier batch drew and the old delivery rule discarded.

A corpus batch can finish a deck and still record no delivery, because the
run's own method contract reported itself invalid and delivery used to answer
with that contract instead of with the deck.  The PowerPoint, its Icons and every
report of that attempt are still on disk, so paying a provider to draw the paper
again would buy a second copy of work already finished -- 228 A-tier papers are
in exactly that state.

This module re-reads one stored attempt, re-answers the delivery question with
the rule in force now, and copies the whole attempt into a new batch root with
``adopted_from`` written beside it.  It never calls a provider.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import re
import shutil
import zipfile

from ..evaluation.delivery_status import classify_delivery_status
from ..evaluation.run_supervision import (
    COLOR_REPORT,
    DENSITY_REPORT,
    GATE_REPORT,
    GEOMETRY_REPORT,
    RENDERED_REPORT,
    RUN_REPORT,
    TEXT_REPORT,
)
from ..utils import read_json, write_json
from .corpus_subset import safe_paper_directory

DECK_FILE = "editable_composition.pptx"
ASSET_REPORT = "asset_generation_report.json"
ADOPTION_REPORT = "adoption_report.json"
ADOPTION_CONTRACT = "rfs.deck_adoption"
ADOPTION_VERSION = 1
#: The reports the run itself graded the deck with, read back in the same order.
QUALITY_REPORTS = (GEOMETRY_REPORT, DENSITY_REPORT, TEXT_REPORT, RENDERED_REPORT, COLOR_REPORT)
_ATTEMPT_PATTERN = re.compile(r"attempt_(\d+)$")


def _attempt_directories(root: str | Path, directory_name: str) -> list[Path]:
    """Every numbered attempt an earlier batch left for one paper, oldest first."""
    runs = Path(root) / "runs" / directory_name
    if not runs.is_dir():
        return []
    attempts = [child for child in runs.iterdir() if child.is_dir() and _ATTEMPT_PATTERN.fullmatch(child.name)]
    return sorted(attempts, key=lambda child: int(_ATTEMPT_PATTERN.fullmatch(child.name).group(1)))


def _readable_deck(attempt: Path) -> bool:
    deck = attempt / DECK_FILE
    return deck.is_file() and deck.stat().st_size > 0 and zipfile.is_zipfile(deck)


def _optional_report(attempt: Path, name: str) -> dict[str, Any] | None:
    path = attempt / name
    if not path.is_file():
        return None
    value = read_json(path)
    return value if isinstance(value, dict) else None


def adjudicate_stored_attempt(attempt: str | Path) -> dict[str, Any]:
    """Answer one stored attempt's delivery question from the reports it wrote.

    The inputs are the same ones the run itself graded with -- its own method and
    structure contracts, its asset report, its quality reports and the status
    its visual review returned -- so re-reading an attempt under today's rule
    never invents a verdict the run could not have reached.
    """
    directory = Path(attempt)
    report = _optional_report(directory, RUN_REPORT) or {}
    deck = directory / DECK_FILE
    return classify_delivery_status(
        compiled={"pptx": str(deck)} if _readable_deck(directory) else None,
        root=directory,
        method_validation=report.get("method_validation") if isinstance(report.get("method_validation"), dict) else None,
        structure_validation=report.get("structure_validation") if isinstance(report.get("structure_validation"), dict) else None,
        asset_report=_optional_report(directory, ASSET_REPORT),
        quality_reports={name: value for name in QUALITY_REPORTS if (value := _optional_report(directory, name)) is not None},
        visual_review=({"status": report["visual_review_status"]} if str(report.get("visual_review_status") or "") else None),
    )


def _record_adoption(path: Path, *, delivery: dict[str, Any], source: Path, paper_id: str, adopted_at: str) -> None:
    """Write the adopted verdict over the copied run report, keeping the old one."""
    report = read_json(path) if path.is_file() else {}
    report = dict(report) if isinstance(report, dict) else {}
    report["adopted_from"] = str(source)
    report["adopted_paper_id"] = paper_id
    report["adopted_at"] = adopted_at
    # The copy lives in the new batch, so its own out_dir follows it; the
    # directory it was adopted from stays readable in adopted_from.
    report["out_dir"] = str(path.parent)
    # What the old rule concluded stays readable next to what the new one does.
    report["adopted_verdict"] = {
        "delivery_status": report.get("delivery_status"),
        "quality_status": report.get("quality_status"),
        "blocking_generation_errors": report.get("blocking_generation_errors"),
    }
    report.update(delivery)
    write_json(path, report)


def adopt_undelivered_decks(
    *,
    sources: Sequence[str | Path],
    out_root: str | Path,
    entries: Iterable[dict[str, Any]],
    adopted_at: Callable[[], str] | None = None,
) -> dict[str, Any]:
    """Copy every paper's already-drawn, now-admissible deck into a batch root.

    A paper already delivered in ``out_root`` is left alone, an attempt whose
    deck is unreadable or still inadmissible is skipped with the reason it
    failed, and nothing is ever deleted from a source root.
    """
    stamp = (adopted_at or (lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")))()
    sources = [Path(item) for item in sources]
    target_root = Path(out_root)
    rows: list[dict[str, Any]] = []
    for entry in entries:
        paper_id = str(entry.get("paper_id") or "").strip()
        if not paper_id:
            continue
        directory_name = safe_paper_directory(paper_id)
        target = target_root / "runs" / directory_name / "attempt_01"
        if (target / RUN_REPORT).is_file():
            rows.append({"paper_id": paper_id, "status": "already_in_target", "path": str(target)})
            continue
        admissible: list[tuple[float, Path, dict[str, Any]]] = []
        unreadable = 0
        inadmissible = ""
        for source in sources:
            for attempt in _attempt_directories(source, directory_name):
                if not _readable_deck(attempt):
                    unreadable += 1
                    continue
                delivery = adjudicate_stored_attempt(attempt)
                if delivery["delivery_status"] != "available":
                    inadmissible = ", ".join(delivery["blocking_generation_errors"]) or "not deliverable"
                    continue
                admissible.append((attempt.stat().st_mtime, attempt, delivery))
        if not admissible:
            rows.append({
                "paper_id": paper_id,
                "status": "skipped",
                "reason": inadmissible or (f"{unreadable} attempts carry no readable deck" if unreadable else "no attempt directory in any source root"),
            })
            continue
        _, source_attempt, delivery = max(admissible, key=lambda item: item[0])
        shutil.copytree(source_attempt, target)
        _record_adoption(target / RUN_REPORT, delivery=delivery, source=source_attempt, paper_id=paper_id, adopted_at=stamp)
        gate = target / GATE_REPORT
        if gate.is_file():
            gate_report = read_json(gate)
            if isinstance(gate_report, dict):
                write_json(gate, {**gate_report, **delivery})
        rows.append({
            "paper_id": paper_id,
            "status": "adopted",
            "adopted_from": str(source_attempt),
            "quality_status": delivery["quality_status"],
            "contract_degradations": delivery["contract_degradations"],
        })
    adopted = [row for row in rows if row["status"] == "adopted"]
    report = {
        "contract": ADOPTION_CONTRACT,
        "version": ADOPTION_VERSION,
        "adopted_at": stamp,
        "out_root": str(target_root),
        "sources": [str(item) for item in sources],
        "papers": len(rows),
        "adopted": len(adopted),
        "skipped": sum(1 for row in rows if row["status"] == "skipped"),
        "already_in_target": sum(1 for row in rows if row["status"] == "already_in_target"),
        "degraded": sum(1 for row in adopted if row["quality_status"] != "passed"),
        "rows": rows,
    }
    write_json(target_root / ADOPTION_REPORT, report)
    return report
