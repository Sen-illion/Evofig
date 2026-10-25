"""The commands behind the human-preference channel: build the queue, take a verdict, export it.

The channel's own record lives in ``rfs.evolution.direct.human_preference``; this module is the
thin layer the command line and the review page both go through, so the defect menu the page
offers and the menu the trained seat answers are the same list read from one place.

Nothing here changes a run.  Collecting reads the run tree and copies the renders a person is
about to look at; recording appends one verdict; exporting renders the verdicts in the shape the
choice seat is trained on.  A run that is still going is unaffected by all three.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..evolution.direct import (
    DEFAULT_REVIEW_KIND,
    HUMAN_REVIEW_DIR,
    HumanReviewStore,
    export_choice_items,
    human_review_stats,
    review_kind,
    write_review_bundle,
)
from ..evolution.direct.human_queue import HumanQueue
from ..judge_training.choice_items import CHOICE_DEFECTS
from ..utils import read_json, write_json

ITEMS_FILE = "items.json"
EXPORT_FILE = "human_choice_items.jsonl"
EXPORT_REPORT = "human_choice_export_report.json"


def review_menu() -> list[str]:
    """The defect names a review may pick, which are the names the seat is trained on."""

    return [str(item) for item in CHOICE_DEFECTS]


def _bundle_items(bundle: str | Path) -> list[dict[str, Any]]:
    path = Path(bundle) / ITEMS_FILE
    if not path.exists():
        raise ValueError(f"the review queue is not collected yet: {path} does not exist")
    payload = read_json(path)
    items = payload.get("items") if isinstance(payload, dict) else None
    return [item for item in (items or []) if isinstance(item, dict)]


def review_collect(*, run_root: str | Path, out: str | Path, limit: int = 0) -> dict[str, Any]:
    """Stage the queue a person is about to review, images included."""

    payload = write_review_bundle(out, run_root, limit=limit)
    directory = Path(out).resolve()
    return {
        "ok": True,
        "summary": f"{len(payload['items'])} comparisons are waiting for a human verdict.",
        "bundle": str(directory),
        "queue": len(payload["items"]),
        "items_file": str((directory / ITEMS_FILE)),
        "menu": review_menu(),
    }


def review_bundle_for(bundle: str | Path = "", *, run_root: str | Path = "") -> Path:
    """Where a verdict is filed: a bundle the caller named, or the run's own review folder.

    Naming the run root rather than the bundle is the shorthand worth having: the loop reads one
    folder per run, and a person who has the run's path should not have to know what the folder
    under it is called.
    """

    if str(bundle or "").strip():
        return Path(bundle)
    if not str(run_root or "").strip():
        raise ValueError("a verdict needs either the bundle it belongs to or the run root it is about")
    return Path(run_root) / HUMAN_REVIEW_DIR


def _dimension_pairs(values: Any) -> dict[str, str]:
    """Read ``name=choice`` pairs, which is how a person names one reading of the figure at a time."""

    out: dict[str, str] = {}
    for item in values or []:
        name, _, choice = str(item).partition("=")
        if not name.strip() or not choice.strip():
            raise ValueError("a dimension is written as name=choice, for example layout=A")
        out[name.strip()] = choice.strip()
    return out


def review_record(
    *,
    bundle: str | Path = "",
    item_id: str = "",
    choice: str = "",
    defect: str = "",
    note: str = "",
    reviewer: str = "",
    kind: str = DEFAULT_REVIEW_KIND,
    paper: str = "",
    stage: str = "",
    apply_at: str = "",
    directive: str = "",
    condition: Any = None,
    dimensions: Any = None,
    resident: bool = False,
    run_root: str | Path = "",
) -> dict[str, Any]:
    """File one verdict, without touching the round it is about.

    Three kinds of verdict go through here, and they are not the same act. A figure choice answers
    the question the seat was asked and becomes a training row. ``layer_settled`` says a layer is
    finished with, which is a person's call and outranks the model's. ``preference`` is an
    instruction about how the figure should be drawn -- the stage it is about, when it applies, and
    the sentence that stage's prompt should have carried -- and it steers the rounds that follow.
    """

    directory = review_bundle_for(bundle, run_root=run_root)
    store = HumanReviewStore(directory)
    payload: dict[str, Any] = {"kind": kind, "human_note": note, "reviewer": reviewer}
    if kind == DEFAULT_REVIEW_KIND:
        payload.update({"item_id": item_id, "human_choice": choice, "human_defect": defect})
    elif kind == "layer_settled":
        payload.update({"paper": paper, "stage": stage})
    else:
        payload.update(
            {
                "paper": paper,
                "apply_at": apply_at,
                "directive": directive,
                "condition": dict(condition) if isinstance(condition, dict) else {},
                "dimensions": _dimension_pairs(dimensions),
                "resident": bool(resident),
            }
        )
    result = store.record(payload, defect_menu=review_menu())
    if not result.get("ok"):
        return {"ok": False, "summary": "The verdict was not written.", **result}
    state = store.state()
    return {
        "ok": True,
        "summary": "Recorded.",
        "kind": review_kind(result["review"]),
        "review": result["review"],
        "entry": dict(result.get("entry") or {}),
        "reviewed": state["reviewed"],
        "entries": state["entries"],
        "resident": state["resident"],
        "bundle": str(directory),
    }


def review_queue(*, run_root: str | Path) -> dict[str, Any]:
    """The judgement points a run is holding for a person, newest first.

    A card is a question the chain has stopped on. Listing them is how a person finds out there is
    something to answer without going looking through the run's folders. The model's own answer is
    deliberately not in this listing: a person answers the question, not the model's answer to it.
    """

    queue = HumanQueue(str(Path(str(run_root))))
    cards = queue.open_cards()
    return {"run_root": str(Path(str(run_root))), "open": len(cards), "cards": cards}


def review_answer(*, card: str | Path, choice: str, note: str = "", reviewer: str = "human") -> dict[str, Any]:
    """File one person's answer into one held card, which is the answer the chain is waiting for.

    Written once and never rewritten: two answers to the same question would leave the chain unable
    to say which one decided, and the card already keeps what the model would have said, so the
    distance between the two survives either way.
    """

    directory = Path(str(card))
    root = directory.parents[4] if len(directory.parents) > 4 else directory
    return HumanQueue(str(root)).answer(str(directory), choice=str(choice), note=str(note), answered_by=str(reviewer))


def review_export(*, bundle: str | Path, out: str | Path) -> dict[str, Any]:
    """Write the verdicts that answer the seat's question as training rows."""

    items = _bundle_items(bundle)
    store = HumanReviewStore(bundle)
    rows = store.rows()
    rendered = export_choice_items(items, rows, defect_menu=review_menu())
    destination = Path(out)
    if destination.suffix.lower() == ".jsonl":
        target = destination
        report_path = destination.with_name(EXPORT_REPORT)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / EXPORT_FILE
        report_path = destination / EXPORT_REPORT
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rendered["items"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "contract": "rfs.human_choice_export",
        "version": 1,
        "bundle": str(Path(bundle).resolve()),
        "written": len(rendered["items"]),
        "skipped": rendered["skipped"],
        "output": str(target.resolve()),
    }
    write_json(report_path, report)
    return {"ok": True, "summary": f"{len(rendered['items'])} training rows written.", **report}


def review_stats(*, bundle: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    """How much of the queue is answered, and where the person and the seat disagree."""

    items = _bundle_items(bundle)
    rows = HumanReviewStore(bundle).rows()
    report = human_review_stats(items, rows)
    report["bundle"] = str(Path(bundle).resolve())
    if out:
        write_json(out, report)
    return {"ok": True, "summary": f"{report['reviewed']} of {report['queue']} compared.", **report}


__all__ = ["EXPORT_FILE", "EXPORT_REPORT", "ITEMS_FILE", "review_bundle_for", "review_collect", "review_export", "review_menu", "review_record", "review_stats"]
