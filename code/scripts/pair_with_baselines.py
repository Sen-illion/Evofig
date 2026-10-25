#!/usr/bin/env python3
"""Pair this project's deliveries against the external baselines, paper by paper.

Table 1 of the benchmark compares systems over the whole 1500-paper own set. This project's rows
are drawn from the 500-paper list instead, so the two tables overlap but are not the same set:
the honest comparison is the intersection, read per paper. This script does that arithmetic on
artefacts that already exist. It generates nothing and scores nothing.

Three things it refuses to hide:

* **The two sides may be read with different rulers.** The baseline trees were scored before
  2026-10-07 and their ``vlm_nine`` records name ``gemini-3-pro-preview``; this project's rows
  score with ``gemini-3.1-flash-lite``. The protocol itself calls those "a different ruler, not a
  comparable column". Every block therefore carries the prompt version and model each side read
  with, and ``ruler_match`` says whether the nine-item columns may be subtracted at all.
* **A blank is not a zero.** A paper one side never scored drops that pair and nothing else,
  matching ``configs/experiment_protocol.json``'s ``missing_value: blank``.
* **A paper the other table never had is not a missing score.** Those are counted separately,
  because dropping them silently would turn a list difference into a quality difference.

    python scripts/pair_with_baselines.py \
        --ours-runs <DATA_ROOT>/abl/out/ours_own500 \
        --ours-runs <DATA_ROOT>/abl/staging_main/ours_own500 \
        --baseline-root <DATA_ROOT>/efb_main \
        --systems paper2sysarch image autofigure livefigure \
        --out <DATA_ROOT>/abl/results/pairing
"""

from __future__ import annotations

import argparse
import json
import random
import re
from typing import Any
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


def key_of(value: str) -> str:
    """One spelling for a paper id, so a DOI folder and a DOI list entry compare equal."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().casefold()).strip("_")


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def record_of(body: dict, fallback_key: str) -> dict:
    """One paper's comparable columns, from either scorer's ``metrics.json``."""

    nine = body.get("vlm_nine") or {}
    quiz = body.get("paper_quiz") or {}
    clip = body.get("paper_figure_clip") or {}
    editable = body.get("editability") if isinstance(body.get("editability"), dict) else {}
    return {
        "paper_id": str(body.get("paper_id") or fallback_key),
        "nine_available": bool(nine.get("available")),
        "nine": {item: nine.get(item) for item in NINE_ITEMS},
        "nine_mean": nine.get("mean"),
        "nine_prompt": nine.get("prompt_version"),
        "nine_model": nine.get("model"),
        "quiz_available": bool(quiz.get("available")),
        "quiz_accuracy": quiz.get("accuracy"),
        "quiz_model": quiz.get("model"),
        "clip_available": bool(clip.get("available")),
        "clip_cosine": clip.get("cosine"),
        "editable": editable.get("editable"),
        "editable_status": editable.get("status"),
        "raster_area_ratio": editable.get("raster_area_ratio"),
        "native_object_count": editable.get("native_object_count"),
    }


def collect(roots) -> dict:
    """paper key -> record, for every ``metrics.json`` under one or more run roots.

    Later roots win, so pass the unscored delivery tree first and the scored copy last.
    """

    rows: dict = {}
    for base in [Path(item) for item in roots]:
        if not base.is_dir():
            continue
        for metrics in sorted(base.rglob("metrics.json")):
            body = read_json(metrics)
            if not isinstance(body, dict):
                continue
            key = key_of(body.get("paper_id") or metrics.parent.name)
            if key:
                rows[key] = record_of(body, key)
    return rows


def collect_systems(baseline_root: Path, systems, subdir: str) -> dict:
    """system -> paper key -> record, from ``<root>/<system>/<subdir>/<paper>/metrics.json``."""

    return {system: collect([Path(baseline_root) / system / subdir]) for system in systems}


def mean(values: list):
    clean = [value for value in values if _is_number(value)]
    return sum(clean) / len(clean) if clean else None


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def bootstrap(values: list, seed: int, draws: int = 2000, lo: float = 2.5, hi: float = 97.5) -> dict:
    """Paper-level percentile interval for the mean of a paired difference."""

    clean = [float(value) for value in values if _is_number(value)]
    if len(clean) < 2:
        return {"n": len(clean), "mean": mean(clean), "lo": None, "hi": None}
    rng = random.Random(seed)
    size = len(clean)
    means = []
    for _ in range(draws):
        means.append(sum(clean[rng.randrange(size)] for _ in range(size)) / size)
    means.sort()
    return {
        "n": size,
        "mean": sum(clean) / size,
        "lo": means[int(len(means) * lo / 100.0)],
        "hi": means[min(len(means) - 1, int(len(means) * hi / 100.0))],
    }


def pair_item(ours: dict, others: dict, field: str, seed: int) -> dict:
    """Paired ``ours minus baseline`` for one numeric column, dropping blanks pair by pair."""

    deltas = []
    for key, row in ours.items():
        other = others.get(key)
        if not other:
            continue
        mine, theirs = row.get(field), other.get(field)
        if _is_number(mine) and _is_number(theirs):
            deltas.append(float(mine) - float(theirs))
    return bootstrap(deltas, seed)


def pair_nine_item(ours: dict, others: dict, item: str, seed: int) -> dict:
    """Paired ``ours minus baseline`` for one rubric item, dropping blanks pair by pair."""

    deltas = []
    for key, row in ours.items():
        other = others.get(key)
        if not other:
            continue
        mine = (row.get("nine") or {}).get(item)
        theirs = (other.get("nine") or {}).get(item)
        if _is_number(mine) and _is_number(theirs):
            deltas.append(float(mine) - float(theirs))
    return bootstrap(deltas, seed)


def _shared_nine(ours: dict, others: dict) -> list:
    return [
        key for key, row in ours.items()
        if others.get(key) and _is_number(row.get("nine_mean")) and _is_number(others[key].get("nine_mean"))
    ]


def pair_nine(ours: dict, others: dict, seed: int) -> dict:
    both = _shared_nine(ours, others)
    return {
        "items": {item: pair_nine_item(ours, others, item, seed) for item in NINE_ITEMS},
        "mean": bootstrap([ours[key]["nine_mean"] - others[key]["nine_mean"] for key in both], seed),
        "ours_mean": mean([ours[key]["nine_mean"] for key in both]),
        "baseline_mean": mean([others[key]["nine_mean"] for key in both]),
    }


def rulers(ours: dict, others: dict) -> dict:
    """Which prompt and model each side actually read with, and whether that is the same ruler."""

    def inventory(rows: dict) -> dict:
        prompts, models = {}, {}
        for row in rows.values():
            if not row.get("nine_available"):
                continue
            prompts[str(row.get("nine_prompt"))] = prompts.get(str(row.get("nine_prompt")), 0) + 1
            models[str(row.get("nine_model"))] = models.get(str(row.get("nine_model")), 0) + 1
        return {"prompt_versions": prompts, "models": models}

    mine, theirs = inventory(ours), inventory(others)
    return {
        "ours": mine,
        "baseline": theirs,
        "ruler_match": bool(mine["models"]) and mine["models"] == theirs["models"]
        and mine["prompt_versions"] == theirs["prompt_versions"],
    }


def editable_block(rows: dict, shared) -> dict:
    """The deterministic editability verdict, as a rate plus the label histogram behind it."""

    flags = []
    labels = []
    for key in shared:
        row = rows.get(key)
        if not row:
            continue
        if isinstance(row.get("editable"), bool):
            flags.append(row["editable"])
        if isinstance(row.get("editable_status"), str) and row["editable_status"]:
            labels.append(row["editable_status"])
    return {
        "scored": len(flags),
        "editable": sum(1 for value in flags if value),
        "rate": (sum(1 for value in flags if value) / len(flags)) if flags else None,
        "labels": {label: labels.count(label) for label in sorted(set(labels))},
    }


def _num(value, signed: bool = False) -> str:
    if not _is_number(value):
        return ""
    return ("%+.3f" if signed else "%.3f") % float(value)


def _interval(cell: dict) -> str:
    if cell.get("lo") is None:
        return ""
    return "[%+.3f, %+.3f]" % (cell["lo"], cell["hi"])


def markdown(report: dict) -> str:
    over = report["overlap"]
    lines = [
        "# Paired comparison against the external baselines",
        "",
        "Our scored papers: %d. Baseline list: %d. Joined by paper id: **%d**. "
        "Our papers the baseline table never had: %d."
        % (over["ours_scored"], over["baseline_papers"], over["joined"], over["ours_not_joined"]),
        "",
    ]
    if not report["ruler_match_anywhere"]:
        lines += [
            "> The nine-item columns were **not** read with the same ruler on both sides, so they are "
            "reported for provenance and must not be subtracted. Re-score the baseline previews with the "
            "current ruler before quoting a delta.",
            "",
        ]
    for system, block in report["systems"].items():
        nine, quiz, clip = block["nine"], block["paper_quiz"], block["clip_cosine"]
        lines += [
            "## %s" % system,
            "",
            "Papers in the baseline table: %d. Shared with ours: %d. Scored on both sides (nine): %d."
            % (block["papers_in_baseline"], block["papers_shared_with_ours"], nine["mean"]["n"]),
            "",
            "| column | paired n | ours | baseline | delta | 95% interval |",
            "| --- | --- | --- | --- | --- | --- |",
            "| nine mean | %d | %s | %s | %s | %s |" % (
                nine["mean"]["n"], _num(nine["ours_mean"]), _num(nine["baseline_mean"]),
                _num(nine["mean"]["mean"], signed=True), _interval(nine["mean"])),
            "| PaperQuiz accuracy | %d | %s | %s | %s | %s |" % (
                quiz["accuracy"]["n"], _num(quiz["ours_mean"]), _num(quiz["baseline_mean"]),
                _num(quiz["accuracy"]["mean"], signed=True), _interval(quiz["accuracy"])),
            "| AltCLIP cosine | %d | %s | %s | %s | %s |" % (
                clip["n"], _num(clip["ours_mean"]), _num(clip["baseline_mean"]),
                _num(clip["mean"], signed=True), _interval(clip)),
            "",
            "| nine item | paired n | delta | 95% interval |",
            "| --- | --- | --- | --- |",
        ]
        for item in NINE_ITEMS:
            cell = nine["items"][item]
            lines.append("| %s | %d | %s | %s |" % (item, cell["n"], _num(cell["mean"], signed=True), _interval(cell)))
        lines += [
            "",
            "Editable decks: ours %s of %d, baseline %s of %d."
            % (_num(block["editability"]["ours"]["rate"]), block["editability"]["ours"]["scored"],
               _num(block["editability"]["baseline"]["rate"]), block["editability"]["baseline"]["scored"]),
            "",
            "Ruler on our side: %s. On the baseline side: %s. Same ruler: %s."
            % (json.dumps(block["rulers"]["ours"], ensure_ascii=False),
               json.dumps(block["rulers"]["baseline"], ensure_ascii=False),
               "yes" if block["rulers"]["ruler_match"] else "no"),
            "",
        ]
    return "\n".join(lines) + "\n"


# --- cost --------------------------------------------------------------------------------------
#
# Cost reads the same two trees the scores do: our ``cost_report.json`` against a baseline row's
# ``run.json``. Three rules keep the table honest:
#
# * every row's tokens are quoted on **one** scale, the run-cost snapshot, exactly as every row's
#   icons are drawn on one image model. A row that internally calls a stronger model
#   (Paper2SysArch's ReAct loop calls ``gpt-5.1``) therefore reports token volume at the snapshot
#   rate rather than that model's own list price, and the models that served its calls are listed
#   beside the number so the caveat is visible;
# * the gate is **delivery**, not scoring -- a paper counts once the row says ``ok`` and our side
#   has a cost report -- and a failed paper is counted, never entered as a zero;
# * a paper whose usage the probe missed is priced from what it left on disk and labelled
#   ``estimated``, so an incomplete row cannot pass for a measured one.

COST_FIELDS = (
    "usd",
    "text_usd",
    "image_usd",
    "seconds",
    "seconds_to_figure",
    "image_calls",
    "text_prompt_tokens",
    "text_completion_tokens",
)

_RATES: dict | None = None


def cost_rates() -> dict:
    """The one scale every row's tokens are quoted on, taken from the run-cost snapshot."""

    global _RATES
    if _RATES is None:
        from rfs.workflows.run_cost import IMAGE_PRICES, PRICE_SNAPSHOT

        _RATES = {
            "as_of": PRICE_SNAPSHOT["as_of"],
            "text_model": PRICE_SNAPSHOT["text_model"],
            "input_usd_per_million": PRICE_SNAPSHOT["text_input_usd_per_million"],
            "output_usd_per_million": PRICE_SNAPSHOT["text_output_usd_per_million"],
            "image_model": "z-image-turbo",
            "image_usd_per_image": IMAGE_PRICES["z-image-turbo"]["image_usd_per_image"],
        }
    return _RATES


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def median(values: list) -> float | None:
    clean = sorted(float(value) for value in values if _is_number(value))
    if not clean:
        return None
    middle = len(clean) // 2
    if len(clean) % 2:
        return clean[middle]
    return (clean[middle - 1] + clean[middle]) / 2


def price_tokens(prompt_tokens: int, completion_tokens: int, image_calls: int) -> dict:
    rates = cost_rates()
    text = (
        prompt_tokens * rates["input_usd_per_million"]
        + completion_tokens * rates["output_usd_per_million"]
    ) / 1e6
    image = image_calls * rates["image_usd_per_image"]
    return {"text_usd": text, "image_usd": image, "usd": text + image}


def _jsonl(path: Path) -> list[dict]:
    """Every well-formed object in a JSONL file; a torn line is skipped, not fatal."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            body = json.loads(line)
        except ValueError:
            continue
        if isinstance(body, dict):
            rows.append(body)
    return rows


def to_figure_seconds(paper_dir: Path) -> float | None:
    """Our §5 clock: the paper's own ``ppt_compilation`` completion, judge and repair excluded."""

    for body in reversed(_jsonl(paper_dir / "progress.jsonl")):
        if body.get("stage") == "ppt_compilation" and body.get("status") == "completed":
            value = body.get("elapsed_seconds")
            return float(value) if _is_number(value) else None
    return None


def collect_our_costs(roots) -> dict:
    """paper key -> cost record, from every ``cost_report.json`` under one or more run roots."""

    rows: dict = {}
    for base in [Path(item) for item in roots]:
        if not base.is_dir():
            continue
        for report in sorted(base.rglob("cost_report.json")):
            body = read_json(report)
            if not isinstance(body, dict):
                continue
            text = body.get("text") if isinstance(body.get("text"), dict) else {}
            images = body.get("image") if isinstance(body.get("image"), dict) else {}
            key = key_of(report.parent.name)
            if not key:
                continue
            record = {
                "paper_id": report.parent.name,
                "paper_dir": str(report.parent),
                "usd": body.get("generation_usd"),
                "text_usd": body.get("text_usd"),
                "image_usd": body.get("image_usd"),
                "seconds": body.get("elapsed_seconds"),
                "image_calls": _int(images.get("images_billed")),
                "text_prompt_tokens": _int(text.get("prompt_tokens")),
                "text_completion_tokens": _int(text.get("completion_tokens")),
                "usage_source": "cost_report",
            }
            figure_seconds = to_figure_seconds(report.parent)
            if figure_seconds is not None:
                record["seconds_to_figure"] = figure_seconds
            rows[key] = record
    return rows


def _media_parts(deck: Path) -> int:
    """Images a delivered deck embeds, for rows that never write a figure file of their own."""

    try:
        import zipfile

        with zipfile.ZipFile(deck) as archive:
            return sum(1 for name in archive.namelist() if name.startswith("ppt/media/"))
    except Exception:
        return 0


def estimate_from_disk(paper_dir: Path) -> dict:
    """Rebuild a missed probe log from the paper's leftovers. A floor, not a measurement.

    Input is the method text the row consumed and output the text artefacts it wrote, both at
    four characters per token; image calls are counted from the figure files, or from the media
    parts of a delivered deck when the row embeds its images instead.
    """

    method = paper_dir / "method.txt"
    prompt_chars = len(method.read_text(encoding="utf-8", errors="replace")) if method.is_file() else 0
    bookkeeping = {
        "run.json", "status.json", "inputs.json", "metrics.json", "questions.json",
        "usage.jsonl", "method.txt",
    }
    output_chars = 0
    for path in sorted(paper_dir.glob("*")):
        if not path.is_file() or path.name in bookkeeping:
            continue
        if path.suffix.lower() in {".json", ".py", ".md", ".txt", ".tex"}:
            output_chars += path.stat().st_size
    figures = len(list(paper_dir.glob("fig*.png")))
    if not figures:
        figures = _media_parts(paper_dir / "delivery.pptx")
    if not figures and (paper_dir / "figure.png").is_file():
        figures = 1
    prompt_tokens = prompt_chars // 4
    completion_tokens = output_chars // 4
    return {
        "text_prompt_tokens": prompt_tokens,
        "text_completion_tokens": completion_tokens,
        "image_calls": figures,
        **price_tokens(prompt_tokens, completion_tokens, figures),
    }


def _priced_baseline(paper_dir: Path, usage: dict) -> dict:
    """The cost columns of one delivered baseline paper, measured or, if missed, estimated."""

    source = str(usage.get("usage_source") or "missing")
    prompt = _int(usage.get("text_prompt_tokens"))
    completion = _int(usage.get("text_completion_tokens"))
    images = _int(usage.get("image_calls"))
    if source == "missing" or not (prompt or completion or images):
        return {**estimate_from_disk(paper_dir), "usage_source": "estimated"}
    return {
        "text_prompt_tokens": prompt,
        "text_completion_tokens": completion,
        "image_calls": images,
        "unmeasured_calls": _int(usage.get("unmeasured_calls")),
        "usage_source": source,
        **price_tokens(prompt, completion, images),
    }


def cost_row_dir(root: Path, system: str, subdir: str) -> Path:
    """Where one cost row's papers live.

    The scored rows keep theirs at ``<root>/<system>/<subdir>``. A re-run row is not nested under
    its scored sibling -- the turbo rows were written beside it, at ``<root>/<system>_turbo/own``
    -- so a value may name its own path instead (``image:image_turbo/own``), read relative to the
    root. Naming the path is checked first because that is the only form that can reach a row
    living outside ``<root>/<system>``.
    """

    named = Path(root) / subdir
    return named if named.is_dir() else Path(root) / system / subdir


def read_sample(path: Path) -> set:
    """The paper keys one cost table is allowed to talk about, one jsonl row per paper.

    A sample has to be stated. The rows do not hold the same papers -- the image row on disk
    holds 1401 while the turbo rows were pointed at 25 -- so without it the widest row in the
    table quietly becomes the sample for every other row, and our columns are cut to it.
    """

    keys: set = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            body = json.loads(line)
        except ValueError:
            continue
        if isinstance(body, dict):
            key = key_of(body.get("paper_id") or body.get("id") or "")
            if key:
                keys.add(key)
    return keys


def collect_baseline_costs(root: Path, system: str, subdir: str) -> dict:
    """paper key -> cost record for one baseline row, gated on the row's own ``ok``."""

    base = cost_row_dir(Path(root), system, subdir)
    rows: dict = {}
    if not base.is_dir():
        return rows
    for paper_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        run = read_json(paper_dir / "run.json")
        if not isinstance(run, dict):
            # A paper the row never got to run still leaves its verdict on disk, and a row's
            # failures are part of what it took to get the deliveries it did report. Reading
            # only run.json would drop them from the table and make the row look complete.
            run = read_json(paper_dir / "status.json")
        if not isinstance(run, dict):
            continue
        key = key_of(run.get("paper_id") or paper_dir.name)
        if not key:
            continue
        record = {
            "paper_id": str(run.get("paper_id") or paper_dir.name),
            "paper_dir": str(paper_dir),
            "delivered": bool(run.get("ok")),
            "seconds": run.get("seconds"),
        }
        if record["delivered"]:
            usage = run.get("usage") if isinstance(run.get("usage"), dict) else {}
            record.update(_priced_baseline(paper_dir, usage))
        rows[key] = record
    return rows


def _model_tally(records) -> dict:
    """Which models actually served a row's calls, from the probe logs it left behind."""

    counts: dict = {}
    for record in records:
        for row in _jsonl(Path(record.get("paper_dir", ".")) / "usage.jsonl"):
            if row.get("kind") not in ("text", "image"):
                continue
            name = str(row.get("model") or "").strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _field_stats(records: list, field: str) -> dict:
    values = [record.get(field) for record in records]
    clean = [float(value) for value in values if _is_number(value)]
    return {"n": len(clean), "median": median(clean), "mean": mean(clean)}


def cost_block(ours: dict, systems: dict, seed: int) -> dict:
    """Per row: what one delivery cost, and the paired difference against us on shared papers.

    A cost row is a *sample*: the 25 papers the turbo rows were pointed at, not the 500 the main
    run holds. So our columns are cut to the keys those rows list. Putting our whole run root
    beside a 25-paper row would set a 500-paper median against a 25-paper one and call the gap a
    difference, which is the comparison this table exists not to make.
    """

    sample = {key for rows in systems.values() for key in rows}
    scoped = {key: record for key, record in ours.items() if key in sample}
    report = {
        "rates": cost_rates(),
        "sample_papers": len(sample),
        "ours_papers_in_sample": len(scoped),
        "systems": {},
    }
    for name, rows in systems.items():
        delivered = [record for record in rows.values() if record.get("delivered")]
        shared = [key for key, record in rows.items()
                  if record.get("delivered") and key in scoped]
        differences = [
            float(rows[key]["usd"]) - float(scoped[key]["usd"])
            for key in shared
            if _is_number(rows[key].get("usd")) and _is_number(scoped[key].get("usd"))
        ]
        report["systems"][name] = {
            "papers_listed": len(rows),
            "papers_delivered": len(delivered),
            "papers_failed": len(rows) - len(delivered),
            "papers_paired_with_ours": len(differences),
            "usage_basis": sorted({str(record.get("usage_source", "missing")) for record in delivered}),
            "models": _model_tally(delivered),
            "metrics": {field: _field_stats(delivered, field) for field in COST_FIELDS},
            "ours_metrics": {field: _field_stats([scoped[key] for key in shared], field)
                             for field in COST_FIELDS},
            "paired_ours_usd": _field_stats([scoped[key] for key in shared], "usd"),
            "paired_theirs_usd": _field_stats([rows[key] for key in shared], "usd"),
            "paired_usd_median_difference": median(differences),
            "paired_usd_difference": bootstrap(differences, seed),
        }
    report["ours_metrics"] = {field: _field_stats(list(scoped.values()), field) for field in COST_FIELDS}
    return report


def cost_markdown(block: dict) -> str:
    rates = block["rates"]
    lines = [
        "# Cost of one delivered figure, one image model",
        "",
        "Priced at the run-cost snapshot of %s: text in %.2f / out %.2f USD per million tokens, "
        "`%s` at %.4f USD per image. Every row's tokens are quoted on that one scale, so a row that "
        "internally calls a stronger model shows token volume rather than that model's own list "
        "price; the models actually observed are listed per row."
        % (rates["as_of"], rates["input_usd_per_million"], rates["output_usd_per_million"],
           rates["image_model"], rates["image_usd_per_image"]),
        "",
        "Every row is quoted on the sample it lists (%d papers); our columns are cut to that same "
        "sample, so our row is never measured on more papers than the row it is set beside."
        % (block.get("sample_papers") or 0),
        "",
        "| row | delivered | failed | median USD | text USD | image USD | median s | images | tokens in/out | basis |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    ours = block["ours_metrics"]
    lines.append(
        "| ours | %d | 0 | %s | %s | %s | %s | %s | %d/%d | cost_report |"
        % (ours["usd"]["n"], _num(ours["usd"]["median"]), _num(ours["text_usd"]["median"]),
           _num(ours["image_usd"]["median"]),
           _num(ours["seconds_to_figure"]["median"] if ours["seconds_to_figure"]["n"] else ours["seconds"]["median"]),
           _num(ours["image_calls"]["median"]),
           ours["text_prompt_tokens"]["median"] or 0, ours["text_completion_tokens"]["median"] or 0)
    )
    for name, row in block["systems"].items():
        metrics = row["metrics"]
        lines.append(
            "| %s | %d | %d | %s | %s | %s | %s | %s | %d/%d | %s |"
            % (name, row["papers_delivered"], row["papers_failed"],
               _num(metrics["usd"]["median"]), _num(metrics["text_usd"]["median"]),
               _num(metrics["image_usd"]["median"]), _num(metrics["seconds"]["median"]),
               _num(metrics["image_calls"]["median"]),
               metrics["text_prompt_tokens"]["median"] or 0,
               metrics["text_completion_tokens"]["median"] or 0,
               "+".join(row["usage_basis"]) or "n/a")
        )
    lines += [
        "",
        "## Paired against us, on the papers both sides delivered",
        "",
        "| row | paired n | our median USD | their median USD | median difference | mean difference (95% CI) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in block["systems"].items():
        lines.append(
            "| %s | %d | %s | %s | %s | %s |"
            % (name, row["papers_paired_with_ours"],
               _num(row["paired_ours_usd"].get("median")),
               _num(row["paired_theirs_usd"].get("median")),
               _num(row.get("paired_usd_median_difference"), signed=True),
               _interval(row["paired_usd_difference"]))
        )
    lines += ["", "## Models observed, and how each row was measured", ""]
    for name, row in block["systems"].items():
        lines.append("- **%s**: basis %s; models %s."
                     % (name, "+".join(row["usage_basis"]) or "n/a",
                        json.dumps(row["models"], ensure_ascii=False) or "{}"))
    lines += [
        "",
        "A row whose basis mentions `estimated` had at least one paper priced from its leftovers "
        "rather than from a measured usage log; see `docs/cost-experiment-plan.md` §6.",
        "",
        "Ours seconds are the §5 clock (the paper's own `ppt_compilation` completion, judge and "
        "repair excluded) where the run recorded one; baseline seconds are their whole `run.json`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pair a project row against the external baselines.")
    parser.add_argument("--ours-runs", action="append", default=None,
                        help="run root(s) holding our metrics.json; repeat for more than one layout")
    parser.add_argument("--ours-label", default="ours")
    parser.add_argument("--baseline-root", default="<DATA_ROOT>/efb_main")
    parser.add_argument("--baseline-subdir", default="own")
    parser.add_argument("--systems", nargs="+", default=["paper2sysarch", "image", "autofigure", "livefigure"])
    parser.add_argument("--out", required=True, help="output directory for pairing.json/.jsonl/.md")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cost-row", action="append", default=None,
                        help="system:subdir to price as a cost row, repeatable; subdir may name a "
                             "path under the baseline root, e.g. image:image_turbo/own, for a row "
                             "that does not live under <root>/<system>. Omitting it leaves the "
                             "report exactly as it was before cost existed")
    parser.add_argument("--sample", default=None,
                        help="jsonl of the papers the cost rows were pointed at; only those keys "
                             "are priced, so a wide row on disk cannot widen the sample")
    args = parser.parse_args(argv)

    roots = args.ours_runs or []
    ours = collect(roots)
    systems = collect_systems(Path(args.baseline_root), args.systems, args.baseline_subdir)
    baseline_keys = {key for rows in systems.values() for key in rows}
    missing = sorted(key for key in ours if key not in baseline_keys)

    overlap = {
        "ours_scored": len(ours),
        "baseline_papers": len(baseline_keys),
        "joined": len(ours) - len(missing),
        "ours_not_joined": len(missing),
        "examples_not_joined": missing[:50],
    }

    blocks = {}
    for system, rows in systems.items():
        shared = {key: rows[key] for key in rows if key in ours}
        nine = pair_nine(ours, shared, args.seed)
        clip = pair_item(ours, shared, "clip_cosine", args.seed)
        clip["ours_mean"] = mean([ours[key]["clip_cosine"] for key in shared])
        clip["baseline_mean"] = mean([shared[key]["clip_cosine"] for key in shared])
        blocks[system] = {
            "papers_in_baseline": len(rows),
            "papers_shared_with_ours": len(shared),
            "nine": nine,
            "paper_quiz": {
                "accuracy": pair_item(ours, shared, "quiz_accuracy", args.seed),
                "ours_mean": mean([ours[key]["quiz_accuracy"] for key in shared if ours[key].get("quiz_available")]),
                "baseline_mean": mean([shared[key]["quiz_accuracy"] for key in shared if shared[key].get("quiz_available")]),
            },
            "clip_cosine": clip,
            "editability": {"ours": editable_block(ours, shared), "baseline": editable_block(shared, shared)},
            "rulers": rulers(ours, shared),
        }

    report = {
        "ours_label": args.ours_label,
        "ours_runs": [str(item) for item in roots],
        "baseline_root": str(args.baseline_root),
        "baseline_subdir": args.baseline_subdir,
        "overlap": overlap,
        "ruler_match_anywhere": any(block["rulers"]["ruler_match"] for block in blocks.values()),
        "systems": blocks,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.cost_row:
        sample = read_sample(Path(args.sample)) if args.sample else None
        systems_costs = {}
        for item in args.cost_row:
            system, _, subdir = str(item).partition(":")
            rows = collect_baseline_costs(
                Path(args.baseline_root), system, subdir or "own"
            )
            if sample is not None:
                rows = {key: record for key, record in rows.items() if key in sample}
            systems_costs[system] = rows
        our_costs = collect_our_costs(roots)
        if sample is not None:
            our_costs = {key: record for key, record in our_costs.items() if key in sample}
        cost = cost_block(our_costs, systems_costs, args.seed)
        cost["sample_file"] = str(args.sample or "")
        report["cost"] = cost
        (out / "cost_arms.json").write_text(
            json.dumps(cost, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "cost_arms.md").write_text(cost_markdown(cost), encoding="utf-8")
    (out / "pairing.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "pairing.md").write_text(markdown(report), encoding="utf-8")
    with (out / "pairing.jsonl").open("w", encoding="utf-8") as handle:
        for key in sorted(ours):
            for system, rows in systems.items():
                other = rows.get(key)
                if not other:
                    continue
                handle.write(json.dumps({
                    "paper_key": key,
                    "paper_id": ours[key]["paper_id"],
                    "baseline_system": system,
                    "ours_nine_mean": ours[key]["nine_mean"],
                    "baseline_nine_mean": other["nine_mean"],
                    "ours_quiz_accuracy": ours[key]["quiz_accuracy"],
                    "baseline_quiz_accuracy": other["quiz_accuracy"],
                    "ours_clip_cosine": ours[key]["clip_cosine"],
                    "baseline_clip_cosine": other["clip_cosine"],
                }, ensure_ascii=False) + "\n")
    print(json.dumps({
        "out": str(out),
        "ours_scored": len(ours),
        "joined": overlap["joined"],
        "ruler_match_anywhere": report["ruler_match_anywhere"],
        "papers_scored_both_nine": {name: block["nine"]["mean"]["n"] for name, block in blocks.items()},
        "cost_rows": ({} if "cost" not in report else
                      {name: row["papers_delivered"] for name, row in report["cost"]["systems"].items()}),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
