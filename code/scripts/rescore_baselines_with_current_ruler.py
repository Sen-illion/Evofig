#!/usr/bin/env python3
"""Re-read the external baselines' figures with this project's current nine-item ruler.

Table 1's baseline columns were written before 2026-10-07, when the rubric was scored with
``gemini-3-pro-preview``. This project's rows score with ``gemini-3.1-flash-lite``. The protocol
records the consequence in its own words: "scores written before 2026-10-07 used
gemini-3-pro-preview and are a different ruler, not a comparable column". Subtracting one column
from the other would measure the two judge models, not the two systems.

So this script reads the baselines' *existing* figures again -- the same preview, the same frozen
PaperQuiz bank, the same method section -- and writes the result into a tree of its own. The
original `metrics.json` files stay where their authors left them. Nothing is generated; the only
calls are the rubric and the quiz.

It defaults to the papers this project's own list covers, because those are the only ones a paired
table needs, and it links the same frozen question bank so both sides answer the same questions.

    python scripts/rescore_baselines_with_current_ruler.py \
        --baseline-root <DATA_ROOT>/efb_main \
        --systems paper2sysarch image autofigure livefigure \
        --manifest <DATA_ROOT>/abl/own_500.jsonl \
        --out-root <DATA_ROOT>/abl/rescore_current_ruler \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rfs.evaluation.own_set_score import safe_paper_id, score_own_delivery  # noqa: E402 - path first
from rfs.utils import write_json  # noqa: E402 - path first

#: A system's figure, in the order the scorer prefers them.
FIGURES = ("preview.png", "figure.png")

#: Where a paper's method section lives when the system's own folder does not carry one.
SHARED_TEXT_SUBDIR = ("own", "text")


def link(origin: Path, dest: Path) -> bool:
    """Point ``dest`` at ``origin``, replacing whatever was there. False if the source is absent."""

    if not origin.is_file():
        return False
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    try:
        os.symlink(origin, dest)
    except OSError:
        try:
            os.link(origin, dest)
        except OSError:
            shutil.copyfile(origin, dest)
    return True


def method_source_for(baseline_root: Path, system_dir: Path, safe: str) -> Path | None:
    """The method section to ground the rubric with: the system's own, else the shared one."""

    own = system_dir / "method.txt"
    if own.is_file():
        return own
    shared = baseline_root.joinpath(*SHARED_TEXT_SUBDIR) / safe / "method.txt"
    return shared if shared.is_file() else None


def prepare(source: Path, dest: Path, method_source: Path | None) -> str | None:
    """Link the figure, the frozen quiz and the method text into a tree of its own.

    Returns ``None`` when the folder is ready to score, otherwise the reason it is not.
    """

    if not (source / "metrics.json").is_file():
        return "system_never_scored_it"
    figure = next((source / name for name in FIGURES if (source / name).is_file()), None)
    if figure is None:
        return "no_figure"
    dest.mkdir(parents=True, exist_ok=True)
    link(figure, dest / "preview.png")
    link(source / "questions.json", dest / "questions.json")
    if method_source is not None:
        link(method_source, dest / "method.txt")
    return None


def one(args) -> dict:
    """Score one paper of one system, and say what happened in a shape the report can count."""

    system, paper_id, safe, source, dest, method_source, force = args
    row = {"system": system, "paper_id": paper_id, "safe": safe}
    if not force and (dest / "metrics.json").is_file():
        return dict(row, ok=True, reason="already_scored")
    reason = prepare(source, dest, method_source)
    if reason:
        return dict(row, ok=False, reason=reason)
    text = method_source.read_text(encoding="utf-8") if method_source is not None else None
    try:
        result = score_own_delivery(
            dest,
            paper_id=paper_id,
            method_text=text,
            embedder=None,
            paper_quiz_spec="api",
            nine_spec="api",
            allow_question_generation=False,
            force=True,
        )
    except Exception as exc:  # noqa: BLE001 - one paper's failure must not end the pass
        return dict(row, ok=False, reason="%s: %s" % (type(exc).__name__, exc))
    metrics = result.get("metrics") or {}
    nine = metrics.get("vlm_nine") or {}
    quiz = metrics.get("paper_quiz") or {}
    return dict(
        row, ok=bool(result.get("ok")), reason=str(result.get("reason") or ""),
        nine_available=bool(nine.get("available")), nine_model=nine.get("model"),
        nine_prompt=nine.get("prompt_version"),
        quiz_available=bool(quiz.get("available")),
    )


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline-root", default="<DATA_ROOT>/efb_main")
    parser.add_argument("--systems", nargs="+",
                        default=["paper2sysarch", "image", "autofigure", "livefigure"])
    parser.add_argument("--subdir", default="own", help="Each system's delivery subdirectory.")
    parser.add_argument("--manifest", required=True, help="The paper list to cover, jsonl.")
    parser.add_argument("--out-root", required=True, help="Where the new metrics tree goes.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent scoring calls.")
    parser.add_argument("--force", action="store_true", help="Re-score a paper that already has metrics.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N papers.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = parser.parse_args(argv)

    baseline_root = Path(args.baseline_root)
    out_root = Path(args.out_root)
    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    papers = [(str(row.get("paper_id") or ""), safe_paper_id(str(row.get("paper_id") or "")))
              for row in rows if row.get("paper_id")]
    if args.limit is not None:
        papers = papers[: max(0, int(args.limit))]

    jobs = []
    for system in args.systems:
        system_dir = baseline_root / system / args.subdir
        for paper_id, safe in papers:
            jobs.append((
                system, paper_id, safe,
                system_dir / safe,
                out_root / system / args.subdir / safe,
                method_source_for(baseline_root, system_dir / safe, safe),
                bool(args.force),
            ))

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        results = list(pool.map(one, jobs))

    counts: dict = {}
    for row in results:
        key = row["reason"] or ("ok" if row.get("ok") else "failed")
        counts[key] = counts.get(key, 0) + 1
    report = {
        "summary": "Re-read the baselines' figures with this project's current nine-item ruler.",
        "baseline_root": str(baseline_root),
        "out_root": str(out_root),
        "systems": args.systems,
        "manifest": str(args.manifest),
        "papers_considered": len(papers),
        "jobs": len(jobs),
        "counts": counts,
        "nine_available": sum(1 for row in results if row.get("nine_available")),
        "quiz_available": sum(1 for row in results if row.get("quiz_available")),
        "outcomes": [row for row in results if not row.get("ok")][:50],
    }
    write_json(out_root / "rescore_report.json", report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("jobs=%d nine_available=%d quiz_available=%d" % (
            len(jobs), report["nine_available"], report["quiz_available"]))
        for key in sorted(counts):
            print("  %-28s %d" % (key, counts[key]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())