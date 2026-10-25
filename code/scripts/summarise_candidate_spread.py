#!/usr/bin/env python3
"""Read what a best-of-N rung got for the candidates it paid for.

A rung's published number is the one the advisory seat picked out of N candidates. That single
number cannot tell two very different worlds apart: an arm whose other N-1 candidates were no
better than the winner spent its extra budget on *sampling* that had nothing to offer, while an
arm whose runner-up was better spent it on *selection* that missed. The two want opposite fixes
-- draft better, or pick better -- and only the losing candidates can separate them, which is why
they have to be scored: ``stage_budget_deliveries.py --candidates`` lays each one out as a tree of
its own and the own-set scorer reads each tree.

This script generates and scores nothing. It reads the per-candidate trees, the delivered tree the
published table was built from, and each run's own record of which variant won
(``rounds[0].selected_variant``), and reports per rung:

  * the pick's nine-item mean and the best-in-row (oracle) mean, side by side;
  * the paired difference oracle minus pick, with a paper-level bootstrap interval;
  * how far apart one paper's N candidates actually were, and how often the pick was the best;
  * a check that the column the record names as the winner scores like the delivered tree, so a
    render-path difference cannot be read as a picking difference.

    python scripts/summarise_candidate_spread.py \
        --list <DATA_ROOT>/abl/de25.jsonl \
        --arm bestofn_4=<DATA_ROOT>/abl/staging_cand/bestofn_4 \
        --delivered bestofn_4=<DATA_ROOT>/abl/staging_d/bestofn_4 \
        --run-root bestofn_4=<DATA_ROOT>/abl/out/bestofn_4 \
        --out <DATA_ROOT>/abl/results/bestofn_oracle.json

The nine-item column is a ruler, and a ruler only compares with itself. Every column this prints
therefore comes from one scoring run over one ruler, and the ruler it read is named in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _entry in (REPO, Path(__file__).resolve().parent):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from stage_budget_deliveries import find_paper_dir  # noqa: E402 - path first
from summarise_budget_arm import bootstrap, key_of, mean, median, read_tree  # noqa: E402 - path first
from rfs.evaluation.own_set_score import safe_paper_id  # noqa: E402 - path first


def _loose_json(path: Path) -> dict:
    """A run's own record, which some writers prefix with a library warning line."""

    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    start = text.find("{")
    if start < 0:
        return {}
    try:
        body = json.loads(text[start:])
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def winner_tree(row: dict, paper_dir: Path | None) -> tuple[str, str]:
    """The candidate tree the run's record says it delivered, and why it could not be read.

    ``selected_variant`` is the index of the candidate directory the loop kept -- the delivered
    deck is byte-identical to that candidate's. When the record is unreadable the paper is
    reported rather than guessed at, because guessing here would silently move the pick's own
    column and with it the whole difference this script exists to measure.
    """

    if paper_dir is None:
        return "", "run_dir_missing"
    rounds = _loose_json(Path(paper_dir) / "run.json").get("rounds") or []
    if not rounds:
        return "", "no_round_record"
    record = rounds[0]
    variant = int(record.get("selected_variant") or 0)
    named = "round_%02d_candidate_%02d" % (int(record.get("round") or 0), variant)
    if named in row:
        return named, ""
    suffix = "candidate_%02d" % variant
    matches = [tree for tree in sorted(row) if tree.endswith(suffix)]
    if len(matches) == 1:
        return matches[0], ""
    return "", "winner_not_staged"


def _pairs(values: list) -> dict:
    out: dict = {}
    for item in values:
        if "=" not in item:
            raise SystemExit("expected name=directory, got %r" % item)
        name, directory = item.split("=", 1)
        out[name.strip()] = Path(directory)
    return out


def paper_dirs(run_root: Path, papers: list) -> dict:
    """safe paper id -> the run directory the driver wrote for it.

    The path is resolved with the same rule the stager uses rather than by matching folder
    names, because the driver and the scorer cut a long paper id at different lengths: on that
    one paper an exact-name map silently drops it from every arm, which would read as a smaller
    sample instead of as the missing row it is.
    """

    out: dict = {}
    for _paper, safe in papers:
        found = find_paper_dir(Path(run_root), safe)
        if found is not None:
            out[safe] = found
    return out


def read_matrix(root: Path) -> tuple[dict, set]:
    """paper -> {tree name: nine-item mean}, plus the ruler the trees were scored under."""

    matrix: dict = {}
    ruler: set = set()
    for tree in sorted(item for item in Path(root).iterdir() if item.is_dir()):
        for paper, record in read_tree(tree).items():
            matrix.setdefault(paper, {})[tree.name] = record.get("nine_mean")
            if record.get("nine_ruler"):
                ruler.add(str(record["nine_ruler"]))
    return matrix, ruler


def summarise_arm(root: Path, delivered_root: Path, run_root: Path, papers: list, seed: int) -> dict:
    matrix, ruler = read_matrix(root)
    delivered = read_tree(delivered_root)
    runs = paper_dirs(run_root, papers)

    picks: list = []
    oracles: list = []
    deltas: list = []
    spreads: list = []
    render_gaps: list = []
    picked_best = 0
    winner_failed: dict = {}
    counts: list = []

    for paper, safe in papers:
        row = {tree: value for tree, value in (matrix.get(paper) or {}).items() if isinstance(value, (int, float))}
        if not row:
            continue
        counts.append(len(row))
        pick_tree, why = winner_tree(row, runs.get(safe))
        oracle = max(row.values())
        if why or pick_tree not in row:
            winner_failed[why or "winner_not_scored"] = winner_failed.get(why or "winner_not_scored", 0) + 1
            pick_tree, pick = "", None
        else:
            pick = row[pick_tree]
        if pick is None:
            continue
        picks.append(pick)
        oracles.append(oracle)
        deltas.append(oracle - pick)
        spreads.append(max(row.values()) - min(row.values()))
        if oracle - pick <= 1e-9:
            picked_best += 1
        published = (delivered.get(paper) or {}).get("nine_mean")
        if isinstance(published, (int, float)):
            render_gaps.append(pick - published)


    return {
        "candidates": max(counts) if counts else 0,
        "papers": len(picks),
        "ruler": sorted(ruler),
        "pick": {"n": len(picks), "mean": mean(picks)},
        "oracle": {"n": len(oracles), "mean": mean(oracles)},
        "delta_oracle_minus_pick": bootstrap(deltas, seed),
        "spread": {"n": len(spreads), "mean": mean(spreads), "median": median(spreads)},
        "pick_is_best": {"count": picked_best, "of": len(picks)},
        "winner_unreadable": winner_failed,
        "render_check": {
            "summary": "pick's own candidate column minus the published delivered column; both from one ruler, so any gap is the render path",
            "n": len(render_gaps),
            "mean": mean(render_gaps),
            "median": median(render_gaps),
            "exact": sum(1 for gap in render_gaps if isinstance(gap, (int, float)) and abs(gap) < 1e-9),
        },
    }


def _fmt(value, digits: int = 4) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("%%.%df" % digits) % value
    return "-"


def _interval(block: dict) -> str:
    if not isinstance(block, dict) or block.get("mean") is None:
        return "-"
    if block.get("lo") is None or block.get("hi") is None:
        return "%s n=%s" % (_fmt(block["mean"]), block.get("n"))
    return "%s [%s, %s] n=%s" % (_fmt(block["mean"]), _fmt(block["lo"]), _fmt(block["hi"]), block.get("n"))


def markdown(report: dict) -> str:
    lines = [
        "# best-of-N：多出来的候选里到底有没有更好的",
        "",
        "「判座挑中」是已发表那一列；「神谕」是同一条候选环里最好的那条。两者的差就是**挑选损失**的规模，",
        "同一篇候选之间的极差就是**候选本身有没有差异**的规模。两者都很小，说明这一轴什么都没买到；",
        "只有挑选损失大，才说明该修的是挑的人，而不是省掉候选。",
        "",
        "| 臂 | 候选数 | 篇数 | 判座挑中 | 神谕（最好那条） | Δ神谕−判座 | 同一篇极差（中位） | 挑中即最好 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, block in sorted(report["arms"].items()):
        lines.append(
            "| %s | %d | %d | %s | %s | %s | %s | %d/%d |"
            % (
                name,
                block["candidates"],
                block["papers"],
                _fmt((block["pick"] or {}).get("mean")),
                _fmt((block["oracle"] or {}).get("mean")),
                _interval(block["delta_oracle_minus_pick"]),
                _fmt((block["spread"] or {}).get("median")),
                block["pick_is_best"]["count"],
                block["pick_is_best"]["of"],
            )
        )
    lines += ["", "读数用哪把尺子：%s" % (", ".join(report["configuration"]["ruler"]) or "-"), ""]
    for name, block in sorted(report["arms"].items()):
        check = block.get("render_check") or {}
        lines.append(
            "- `%s` 渲染核对：判座挑中那条在候选矩阵里的分，与已发表那一列的差 中位 %s（n=%s，完全相等 %s 篇）。"
            % (name, _fmt(check.get("median")), check.get("n"), check.get("exact"))
        )
        if block.get("winner_unreadable"):
            lines.append("- `%s` 有读不出的胜者记录：%s" % (name, json.dumps(block["winner_unreadable"], ensure_ascii=False)))
    lines.append("")
    return "\n".join(lines)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read the candidate matrix under a best-of-N rung.")
    parser.add_argument("--list", required=True, help="The paper list every arm of a family ran.")
    parser.add_argument("--arm", action="append", default=[], required=True, help="name=staging tree holding one directory per round and candidate.")
    parser.add_argument("--delivered", action="append", default=[], required=True, help="name=staging tree the published column was built from.")
    parser.add_argument("--run-root", action="append", default=[], required=True, help="name=the arm's run root, where each paper's own record lives.")
    parser.add_argument("--out", default="", help="Where to write the JSON; the markdown is written beside it.")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in Path(args.list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    papers = [
        (key_of(row.get("paper_id")), safe_paper_id(str(row.get("paper_id") or "")))
        for row in rows
        if row.get("paper_id")
    ]
    arms = _pairs(args.arm)
    delivered = _pairs(args.delivered)
    runs = _pairs(args.run_root)

    report = {
        "row": "experiment-6-candidate-spread",
        "configuration": {
            "list": str(args.list),
            "papers": len(papers),
            "arms": {name: str(path) for name, path in arms.items()},
            "delivered": {name: str(path) for name, path in delivered.items()},
            "run_root": {name: str(path) for name, path in runs.items()},
        },
        "arms": {},
    }
    ruler: set = set()
    for name, root in arms.items():
        if name not in delivered or name not in runs:
            raise SystemExit("arm %s needs --delivered and --run-root too" % name)
        block = summarise_arm(root, delivered[name], runs[name], papers, args.seed)
        report["arms"][name] = block
        ruler |= set(block.get("ruler") or [])
    report["configuration"]["ruler"] = sorted(ruler)

    text = json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
        print("wrote %s and %s" % (out, out.with_suffix(".md")))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())