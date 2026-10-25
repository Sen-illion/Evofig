#!/usr/bin/env python3
"""Fill the paper-figure CLIP column of own-set metrics that were scored without an embedder.

The own-set scorer takes AltCLIP as an input, and leaves this column null when it is not given one
-- so a run of `benchmark score-own` without `--clip-model` reports an empty column for reasons that
have nothing to do with the figure. Re-running the scorer to fill it would recompute the nine-item
and PaperQuiz calls as well, which are the expensive ones and the ones already paid for.

This recomputes only the CLIP column, in place, and leaves every other key exactly as it was. The
preview and the method section it reads are the same files the scorer read.

    python scripts/fill_figure_clip_column.py \
        --runs <DATA_ROOT>/abl/staging_main/ours_own500 \
        --clip-model <DATA_ROOT>/models/AltCLIP
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rfs.evaluation.own_set_score import LocalAltCLIP  # noqa: E402 - path first
from rfs.evaluation.paper_figure_clip import score_paper_figure_clip  # noqa: E402 - path first
from rfs.utils import write_json  # noqa: E402 - path first

PREVIEWS = ("preview.png", "figure.png")


def one(run_dir: Path, embedder) -> dict:
    """Recompute one folder's CLIP column. Returns what happened, for the report."""

    metrics_path = run_dir / "metrics.json"
    try:
        body = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"folder": run_dir.name, "ok": False, "reason": "unreadable metrics.json"}
    if not isinstance(body, dict):
        return {"folder": run_dir.name, "ok": False, "reason": "metrics.json is not an object"}
    preview = next((run_dir / name for name in PREVIEWS if (run_dir / name).is_file()), None)
    method_path = run_dir / "method.txt"
    method = method_path.read_text(encoding="utf-8") if method_path.is_file() else None
    column = score_paper_figure_clip(preview, method, embedder=embedder)
    body["paper_figure_clip"] = column
    write_json(metrics_path, body)
    return {"folder": run_dir.name, "ok": bool(column.get("available")),
            "cosine": column.get("cosine"), "reason": column.get("reason") or ""}


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", required=True, help="Run root holding <paper>/metrics.json folders.")
    parser.add_argument("--clip-model", required=True, help="Local AltCLIP directory.")
    parser.add_argument("--force", action="store_true", help="Recompute a column that is already filled.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.runs)
    if not root.is_dir():
        raise SystemExit("no such run root: %s" % root)
    folders = sorted(item for item in root.iterdir() if item.is_dir() and (item / "metrics.json").is_file())
    if args.limit is not None:
        folders = folders[: max(0, int(args.limit))]

    todo = []
    skipped = 0
    for folder in folders:
        if not args.force:
            try:
                body = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                body = {}
            if isinstance(body.get("paper_figure_clip"), dict) and body["paper_figure_clip"].get("available"):
                skipped += 1
                continue
        todo.append(folder)

    embedder = LocalAltCLIP(args.clip_model) if todo else None
    results = []
    for index, folder in enumerate(todo, start=1):
        outcome = one(folder, embedder)
        results.append(outcome)
        print("[%d/%d] %s %s" % (index, len(todo), folder.name,
                                 outcome.get("cosine") if outcome["ok"] else outcome["reason"]), flush=True)

    report = {
        "summary": "Fill the paper-figure CLIP column without recomputing the other columns.",
        "runs": str(root),
        "clip_model": str(args.clip_model),
        "folders_with_metrics": len(folders),
        "already_filled": skipped,
        "attempted": len(todo),
        "filled": sum(1 for row in results if row["ok"]),
        "failed": [row for row in results if not row["ok"]][:50],
    }
    write_json(root / "_clip_fill_report.json", report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        values = [row["cosine"] for row in results if row["ok"]]
        if values:
            print("filled=%d mean=%.4f min=%.4f max=%.4f" % (
                len(values), sum(values) / len(values), min(values), max(values)))
        else:
            print("filled=0 of %d attempted" % len(todo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())