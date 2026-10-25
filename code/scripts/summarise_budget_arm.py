#!/usr/bin/env python3
"""Summarise the inference-budget arms as one cost-versus-score table per family.

Experiment 6 has two axes, and both ask the same kind of question: what does a run buy by
spending more per paper? One spends it by writing more candidates each round (best-of-N); the
other by looking further into the reference library (retrieval depth k). Neither answer comes
from the run itself -- the run only leaves the delivery, its cost ledger and, once the scorer
has been over it, the nine-item ruler's numbers beside it.

This script generates and scores nothing. It reads trees that already exist, pairs the rungs of
one family paper by paper (every arm of a family runs the same 25 papers, so the pairing is
exact), and reports for each rung:

  * the nine-item mean and every item, with the count that carried each one;
  * the deterministic editability rate and what it was measured from;
  * the median and mean cost of a paper: wall clock, prompt and completion tokens, images billed;
  * the paired difference against the family's baseline, with a paper-level bootstrap interval.

    python scripts/summarise_budget_arm.py \
        --list <DATA_ROOT>/abl/de25.jsonl \
        --baseline bestofn_1=<DATA_ROOT>/abl/score/bestofn_1 \
        --arm bestofn_4=<DATA_ROOT>/abl/score/bestofn_4 \
        --out <DATA_ROOT>/abl/budget_arms.json

An arm is written ``name=directory``. The family of an arm is the leading token of its name, so
``bestofn_4`` is read against ``bestofn_1`` and ``topk_8`` against ``topk_5``: the two axes are
reported side by side, never pooled into one curve, because candidates and library depth are not
the same unit of spend.

The nine-item column is a ruler, and a ruler only compares with itself. Scores taken before
2026-10-07 read with gemini-3-pro-preview; the ones written since read with gemini-3.1-flash-lite
(``RFS_SCORE_MODEL``). This script does not know which ruler a tree was scored under and does not
guess: every column it prints carries the models that scored it, and a comparison across two
different rulers is neither made nor reported as one.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from pathlib import Path

NINE_ITEMS = (
    "accuracy",
    "completeness",
    "clarity",
    "flow",
    "legibility",
    "aesthetic",
    "expressiveness",
    "polish",
    "appropriateness",
)

#: What one paper costs and how long it took, keyed by the field it is read from in
#: ``cost_report.json``. Read by name so a missing block is a blank, never a zero.
COST_FIELDS = (
    ("seconds", ("elapsed_seconds",)),
    ("prompt_tokens", ("text", "prompt_tokens")),
    ("completion_tokens", ("text", "completion_tokens")),
    ("images_billed", ("image", "images_billed")),
    ("usd", ("generation_usd",)),
    ("text_usd", ("text_usd",)),
    ("image_usd", ("image_usd",)),
)


def key_of(value: str) -> str:
    """One spelling for a paper id, so a DOI folder and a DOI list entry compare equal."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")


def family_of(name: str) -> str:
    """The axis an arm belongs to: its name up to the first underscore."""

    return str(name or "").split("_")[0]


def held_out(path: Path) -> set:
    """Every id a holdout file names, JSONL or one id per line."""

    ids: set = set()
    if not path or not Path(path).is_file():
        return ids
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("{"):
            row = json.loads(text)
            for field in ("paper_id", "example_id"):
                if row.get(field):
                    ids.add(key_of(row[field]))
            continue
        ids.add(key_of(text.split()[0]))
    return ids


def sample_ids(path: Path) -> list:
    """The papers an arm was supposed to run, in list order."""

    ids: list = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            ids.append(key_of(json.loads(line).get("paper_id")))
    return ids


def _dig(body: dict, path: tuple):
    node = body
    for name in path:
        if not isinstance(node, dict):
            return None
        node = node.get(name)
    return node


def read_tree(root: Path) -> dict:
    """paper key -> its nine scores, its editability verdict and what the paper cost."""

    rows: dict = {}
    for metrics_path in sorted(Path(root).glob("*/metrics.json")):
        folder = metrics_path.parent
        try:
            body = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        paper = key_of(body.get("paper_id") or folder.name)
        nine = body.get("vlm_nine") or {}
        editable = body.get("editability") or {}
        cost_path = folder / "cost_report.json"
        cost: dict = {}
        if cost_path.is_file():
            try:
                cost = json.loads(cost_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cost = {}
        record = {
            "nine": {item: nine.get(item) for item in NINE_ITEMS} if nine.get("available") else {},
            "nine_mean": nine.get("mean"),
            "nine_ruler": nine.get("model"),
            "editable": editable.get("editable"),
            "raster_area_ratio": editable.get("raster_area_ratio"),
            "native_object_count": editable.get("native_object_count"),
            "delivered": (folder / "delivery.pptx").is_file(),
        }
        for name, path in COST_FIELDS:
            value = _dig(cost, path)
            record[name] = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        rows[paper] = record
    return rows


def mean(values: list):
    clean = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return sum(clean) / len(clean) if clean else None


def median(values: list):
    clean = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return statistics.median(clean) if clean else None


def bootstrap(values: list, seed: int, draws: int = 2000, lo: float = 2.5, hi: float = 97.5):
    """Paper-level percentile interval for the mean of a paired difference."""

    clean = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if len(clean) < 2:
        return {"n": len(clean), "mean": mean(clean), "lo": None, "hi": None}
    rng = random.Random(seed)
    size = len(clean)
    means = sorted(sum(clean[rng.randrange(size)] for _ in range(size)) / size for _ in range(draws))
    return {
        "n": size,
        "mean": sum(clean) / size,
        "lo": means[int(len(means) * lo / 100.0)],
        "hi": means[min(len(means) - 1, int(len(means) * hi / 100.0))],
    }


def paired(base: dict, arm: dict, seed: int) -> dict:
    """arm minus baseline, per column, over the papers scored on both sides."""

    out: dict = {}
    for item in NINE_ITEMS + ("nine_mean", "usd", "seconds", "images_billed"):
        deltas = []
        for paper, row in arm.items():
            other = base.get(paper)
            if not other:
                continue
            if item in NINE_ITEMS:
                before, after = other["nine"].get(item), row["nine"].get(item)
            else:
                before, after = other.get(item), row.get(item)
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                deltas.append(float(after) - float(before))
        out[item] = bootstrap(deltas, seed)
    return out


def summarise(rows: dict, expected: list) -> dict:
    nine = {item: mean([row["nine"].get(item) for row in rows.values()]) for item in NINE_ITEMS}
    editable = [row["editable"] for row in rows.values() if isinstance(row["editable"], bool)]
    rulers = sorted({str(row.get("nine_ruler") or "") for row in rows.values() if row.get("nine_ruler")})
    missing = [paper for paper in expected if paper not in rows]
    return {
        "papers_scored": len(rows),
        "papers_expected": len(expected),
        "coverage": (len(rows) / len(expected)) if expected else None,
        "papers_delivered": sum(1 for row in rows.values() if row["delivered"]),
        "missing_papers": missing,
        "nine_ruler": rulers,
        "nine": nine,
        "nine_mean": mean([row["nine_mean"] for row in rows.values()]),
        "editability": {
            "scored": len(editable),
            "editable_rate": (sum(1 for value in editable if value) / len(editable)) if editable else None,
            "raster_area_ratio_mean": mean([row["raster_area_ratio"] for row in rows.values()]),
            "native_object_count_mean": mean([row["native_object_count"] for row in rows.values()]),
        },
        "cost": {
            name: {
                "mean": mean([row.get(name) for row in rows.values()]),
                "median": median([row.get(name) for row in rows.values()]),
                "n": len([row.get(name) for row in rows.values() if isinstance(row.get(name), (int, float))]),
            }
            for name, _ in COST_FIELDS
        },
    }


def _fmt(value, width=8, digits=4) -> str:
    if value is None:
        return " " * (width - 3) + "n/a"
    if digits == 0:
        return ("%*d" % (width, int(round(value))))
    return ("%*.4f" % (width, value)) if digits == 4 else ("%*.2f" % (width, value))


def markdown(report: dict) -> str:
    """The same numbers as tables, one per family, in the order the axes are read."""

    lines = ["# 推理期预算：分数 vs 成本", ""]
    for family, block in report["families"].items():
        baseline = block["baseline"]
        lines += ["## %s（基准 %s）" % (family, baseline), ""]
        lines += ["| 臂 | 篇数 | 九项均值 | 准确 | 完整 | 可编辑率 | 中位 $/篇 | 中位秒 | 中位图标 |"]
        lines += ["| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, summary in block["arms"].items():
            lines.append("| %s | %d/%d | %s | %s | %s | %s | %s | %s | %s |" % (
                name,
                summary["papers_scored"],
                summary["papers_expected"],
                _fmt(summary["nine_mean"]),
                _fmt(summary["nine"].get("accuracy")),
                _fmt(summary["nine"].get("completeness")),
                _fmt(summary["editability"]["editable_rate"]),
                _fmt(summary["cost"]["usd"]["median"]),
                _fmt(summary["cost"]["seconds"]["median"], digits=0),
                _fmt(summary["cost"]["images_billed"]["median"], digits=0),
            ))
        lines += ["", "相对基准的配对差（论文级 bootstrap 95% 区间）：", ""]
        lines += ["| 臂 | 九项均值 | 准确 | 完整 | $/篇 | 图标数 |"]
        lines += ["| --- | --- | --- | --- | --- | --- |"]
        for name, delta in block["deltas"].items():
            def cell(item):
                value = delta[item]
                if value["mean"] is None:
                    return "n/a"
                interval = ""
                if value["lo"] is not None:
                    interval = " [%.4f, %.4f]" % (value["lo"], value["hi"])
                return "%+.4f%s n=%d" % (value["mean"], interval, value["n"])
            lines.append("| %s | %s | %s | %s | %s | %s |" % (
                name, cell("nine_mean"), cell("accuracy"), cell("completeness"), cell("usd"), cell("images_billed"),
            ))
        lines += ["", "读数用哪把尺子：%s" % (", ".join(block["ruler"]) or "未标注"), ""]
    return "\n".join(lines) + "\n"


def _pairs(values: list) -> dict:
    out: dict = {}
    for item in values:
        name, _, directory = str(item).partition("=")
        if not name or not directory:
            raise SystemExit("--baseline and --arm take name=directory, not %r" % item)
        out[name] = Path(directory)
    return out


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise the inference-budget arms.")
    parser.add_argument("--list", required=True, help="The paper list every arm of a family ran.")
    parser.add_argument("--baseline", action="append", default=[], required=True, help="name=scored tree. Repeat once per family.")
    parser.add_argument("--arm", action="append", default=[], required=True, help="name=scored tree. Repeat for each rung.")
    parser.add_argument("--holdout", default="", help="Optional overlap file; papers it names are dropped from every arm.")
    parser.add_argument("--out", default="", help="Where to write the JSON; the markdown is written beside it.")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)

    expected = sample_ids(Path(args.list))
    blocked = held_out(Path(args.holdout)) if args.holdout else set()
    keep = [paper for paper in expected if paper not in blocked]
    baselines = _pairs(args.baseline)
    arms = _pairs(args.arm)

    families: dict = {}
    for name, directory in list(baselines.items()) + list(arms.items()):
        family = family_of(name)
        block = families.setdefault(family, {"baseline": "", "ruler": [], "arms": {}, "deltas": {}})
        if name in baselines:
            block["baseline"] = name
    for name, directory in arms.items():
        family = family_of(name)
        block = families[family]
        if not block["baseline"]:
            raise SystemExit("family %s has no baseline; pass --baseline %s_...=<tree>" % (family, family))

    read: dict = {}
    for name, directory in list(baselines.items()) + list(arms.items()):
        rows = read_tree(directory)
        read[name] = {paper: row for paper, row in rows.items() if not blocked or paper not in blocked}

    for family, block in families.items():
        base_name = block["baseline"]
        block["arms"] = {base_name: summarise(read[base_name], keep)}
        for name in arms:
            if family_of(name) != family:
                continue
            block["arms"][name] = summarise(read[name], keep)
            block["deltas"][name] = paired(read[base_name], read[name], args.seed)
        block["ruler"] = sorted({
            str(model)
            for summary in block["arms"].values()
            for model in summary["nine_ruler"]
        })

    report = {
        "row": "experiment-6-inference-budget",
        "configuration": {
            "list": str(args.list),
            "papers": len(keep),
            "holdout": str(args.holdout or ""),
            "baselines": {name: str(path) for name, path in baselines.items()},
            "arms": {name: str(path) for name, path in arms.items()},
            "note": "Nine-item and editability columns only compare within one ruler. Cand-ceiling and library depth are different units of spend and are never pooled.",
        },
        "families": families,
    }
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
