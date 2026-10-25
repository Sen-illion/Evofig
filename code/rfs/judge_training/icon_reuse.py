"""Reuse a legacy run's icons in a fresh run that draws the sealed middleware contract.

The first preference source compares a paper's published framework figure with one our own
pipeline generated. Only a run drawn against the sealed middleware contract can be paired
with that figure, and every legacy run drew the plan it wrote for itself, so that half has
to be drawn again. Drawing is the expensive half, and most of it is already paid for: the
same paper's legacy run holds one icon per method node, and the middleware contract wants
icons for the same nodes under different identifiers (``icon_user_task`` there,
``icon_input_user_task`` here).

Both sides mint their own node ids and their own evidence ids, so the only bridge between
them is the node label. :func:`lexical_mapping` matches labels and sorts every slot into
``confident`` (identical wording), ``review`` (close wording) or ``missing``;
:func:`adjudicate_records` then hands one paper's whole slot list and icon list to the VLM
once, which keeps, drops or replaces every proposal and re-tiers every close-wording
match it confirms; :func:`stage_reuse` copies the kept PNGs into the asset paths a fresh run
reads, which is what makes the icon free - the icon generator reuses any candidate file
that already exists instead of paying the provider again.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Sequence

from ..providers import call_vlm_json, resolve_vlm_model
from ..utils import ensure_dir, write_json, write_text

PLAN_FILE = "figure_content_plan.json"
ASSET_REPORT_FILE = "asset_generation_report.json"
PRIMARY_BUCKETS = ("inputs", "outputs", "modules")
SECONDARY_BUCKETS = ("stages", "innovations", "optional_details")
ROLE_PREFIXES = frozenset({
    "input", "inputs", "mod", "module", "modules", "output", "outputs", "opt", "inno",
    "stage", "stages", "node", "nodes", "desc", "sec", "detail", "group",
})
FILLER_TOKENS = frozenset({
    "the", "a", "an", "of", "for", "and", "to", "in", "on", "with", "via", "by", "its",
    "into", "path", "module", "system", "step", "process", "based", "using", "as",
})
MIN_ICON_BYTES = 20_000
LEXICAL_REVIEW_THRESHOLD = 0.45
DEFAULT_ATTEMPT = "attempt_01"
DEFAULT_CANDIDATE = "candidate_00"
SHEET_NAMES = ("confident", "review", "missing")
MODEL_ENV_NAMES = ("RFS_ICON_REUSE_MODEL", "RFS_TEXT_PLAN_MODEL")
#: Routes whose PNG a fresh run may be handed. ``confident`` matched the wording exactly and
#: ``adjudicated`` was confirmed by the VLM; ``review`` was never confirmed by it and
#: ``missing`` owns no icon at all, so neither may be pre-staged into a paid run.
STAGING_ROUTES = ("confident", "adjudicated")
STAGE_REPORT_FILE = "icon_reuse_stage_report.json"


def _tokens(text: str) -> list[str]:
    """Lower-case word tokens of a label, with parentheticals and filler words dropped."""
    cleaned = re.sub(r"\([^)]*\)", " ", str(text or ""))
    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", cleaned).lower()
    return [token for token in cleaned.split() if token and token not in FILLER_TOKENS]


def _strip_role(tokens: list[str]) -> list[str]:
    """Drop leading role prefixes so ``input_user_task`` and ``user_task`` compare equal."""
    remaining = list(tokens)
    while remaining and remaining[0] in ROLE_PREFIXES:
        remaining.pop(0)
    return remaining


def _jaccard(left: list[str], right: list[str]) -> float:
    first, second = set(left), set(right)
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def _morphology(left: list[str], right: list[str]) -> float:
    """Catch near-miss word forms such as ``comparison`` against ``comparator``."""
    return difflib.SequenceMatcher(None, " ".join(left), " ".join(right)).ratio()


def _similarity(slot: dict[str, Any], icon: dict[str, Any]) -> tuple[float, str]:
    """Score one slot/icon pair; the second value says why it scores that way."""
    exact_id = bool(slot["id_tokens"]) and set(slot["id_tokens"]) == set(icon["id_tokens"])
    exact_label = bool(slot["label_tokens"]) and set(slot["label_tokens"]) == set(icon["label_tokens"])
    if exact_id or exact_label:
        return 1.0, "identical"
    score = max(
        max(_jaccard(slot["id_tokens"], icon["id_tokens"]), _morphology(slot["id_tokens"], icon["id_tokens"])),
        max(_jaccard(slot["label_tokens"], icon["label_tokens"]), _morphology(slot["label_tokens"], icon["label_tokens"])),
    )
    return score, "similar"


def contract_icon_slots(contract_dir: str | Path) -> list[dict[str, Any]]:
    """Read the icon-worthy nodes of one exported middleware contract."""
    path = Path(contract_dir) / PLAN_FILE
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    slots = []
    for bucket in PRIMARY_BUCKETS + SECONDARY_BUCKETS:
        for item in plan.get(bucket) or []:
            if not isinstance(item, dict):
                continue
            node_id = str(item.get("id") or "").strip()
            if not node_id:
                continue
            label = str(item.get("label") or node_id)
            slots.append(
                {
                    "slot": f"icon_{node_id}",
                    "node_id": node_id,
                    "label": label,
                    "bucket": bucket,
                    "primary": bucket in PRIMARY_BUCKETS,
                    "id_tokens": _strip_role(_tokens(node_id)),
                    "label_tokens": _tokens(label),
                }
            )
    return slots


def legacy_run_icons(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read the icons one legacy run already paid for, with the subject it drew."""
    root = Path(run_dir)
    try:
        report = json.loads((root / ASSET_REPORT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    icons = []
    for item in report.get("assets") or []:
        if not isinstance(item, dict):
            continue
        asset_id = str(item.get("asset_id") or "")
        if not asset_id:
            continue
        subject = asset_id
        try:
            text = (root / "assets" / asset_id / "prompt.txt").read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        found = re.search(r"for:\s*([^.]*)", text)
        if found:
            subject = found.group(1).strip()
        png = root / "assets" / asset_id / "selected.png"
        size = png.stat().st_size if png.is_file() else 0
        key = asset_id[5:] if asset_id.startswith("icon_") else asset_id
        icons.append(
            {
                "asset_id": asset_id,
                "label": subject,
                "png": str(png),
                "bytes": size,
                "usable": size > MIN_ICON_BYTES,
                "id_tokens": _strip_role(_tokens(key)),
                "label_tokens": _tokens(subject),
            }
        )
    return icons


def lexical_mapping(slots: list[dict[str, Any]], icons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair every slot with at most one legacy icon, best score first."""
    usable = [icon for icon in icons if icon.get("usable")]
    proposals: list[tuple[float, int, int]] = []
    for slot_index, slot in enumerate(slots):
        for icon_index, icon in enumerate(usable):
            score, why = _similarity(slot, icon)
            if why == "identical" or score >= LEXICAL_REVIEW_THRESHOLD:
                proposals.append((1.0 if why == "identical" else score, slot_index, icon_index))
    proposals.sort(key=lambda item: -item[0])
    taken_slots: set[int] = set()
    taken_icons: set[int] = set()
    chosen: dict[int, tuple[float, int]] = {}
    for score, slot_index, icon_index in proposals:
        if slot_index in taken_slots or icon_index in taken_icons:
            continue
        taken_slots.add(slot_index)
        taken_icons.add(icon_index)
        chosen[slot_index] = (score, icon_index)

    records = []
    for slot_index, slot in enumerate(slots):
        hit = chosen.get(slot_index)
        record = {
            "slot": slot["slot"],
            "node_id": slot["node_id"],
            "label": slot["label"],
            "bucket": slot["bucket"],
            "primary": slot["primary"],
            "route": "missing",
            "lexical_route": "missing",
            "score": 0.0,
            "legacy_asset": None,
            "legacy_label": None,
            "legacy_png": None,
        }
        if hit is not None:
            score, icon_index = hit
            icon = usable[icon_index]
            record.update(
                {
                    "route": "confident" if score >= 1.0 else "review",
                    "lexical_route": "confident" if score >= 1.0 else "review",
                    "score": round(float(score), 3),
                    "legacy_asset": icon["asset_id"],
                    "legacy_label": icon["label"],
                    "legacy_png": icon["png"],
                }
            )
        records.append(record)
    return records


def _adjudication_prompt(paper: str, records: list[dict[str, Any]], icons: list[dict[str, Any]]) -> str:
    """Ask once, for one paper, which legacy icon each new slot may reuse."""
    lines = [
        "Two runs drew a framework figure for the same paper. The new plan has to be drawn again;",
        "the legacy icons are already paid for. Decide which new slot may reuse which legacy icon.",
        "",
        f"Paper: {paper}",
        "",
        "NEW SLOTS (icon_<node id> :: label):",
    ]
    lines += [f"  {record['slot']} :: {record['label']}" for record in records]
    lines.append("")
    lines.append("LEGACY ICONS (asset id :: label):")
    lines += [f"  {icon['asset_id']} :: {icon['label']}" for icon in icons]
    proposal = {record["slot"]: record["legacy_asset"] for record in records if record.get("legacy_asset")}
    lines += [
        "",
        "A first textual pass proposed these pairs (review them, do not trust them):",
        json.dumps(proposal, ensure_ascii=False),
        "",
        "Rules:",
        "- Reuse only when the legacy icon depicts the same scientific object as the slot.",
        "- At most one legacy icon per slot and at most one slot per legacy icon.",
        "- Use null when nothing fits; never invent an asset id.",
        "- A shared word is not enough: a reverse J-V curve reading is not reverse-bias stress.",
        "",
        'Answer with JSON only: {"mapping": {"<new slot id>": "<legacy asset id>" or null}}',
    ]
    return "\n".join(lines)


def _vlm_call(prompt: str, images: list[str], *, model: str, timeout: int, retries: int) -> dict:
    return call_vlm_json(prompt, images, model=model, timeout=timeout, retries=retries)


def adjudicate_records(
    records: list[dict[str, Any]],
    icons: list[dict[str, Any]],
    *,
    paper: str = "",
    call: Callable[..., dict] | None = None,
    model: str | None = None,
    timeout: int = 180,
    retries: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep, drop or replace every lexical proposal for one paper with one VLM answer."""
    caller = call or _vlm_call
    usable = [icon for icon in icons if icon.get("usable")]
    if not records or not usable:
        return records, {"applied": False, "reason": "no slots or no usable legacy icons"}
    resolved_model = model or resolve_vlm_model(*MODEL_ENV_NAMES)
    answered = caller(
        _adjudication_prompt(paper, records, usable),
        [],
        model=resolved_model,
        timeout=int(timeout),
        retries=int(retries),
    )
    mapping = answered.get("mapping") if isinstance(answered, dict) else None
    if not isinstance(mapping, dict):
        return records, {"applied": False, "reason": "the answer carried no mapping", "model": resolved_model}

    by_asset = {icon["asset_id"]: icon for icon in usable}
    claimed: set[str] = set()
    updated = []
    tally = Counter()
    for record in records:
        if record["slot"] not in mapping:
            updated.append(record)
            continue
        decision = mapping[record["slot"]]
        wanted = None if decision in (None, "", "null") else str(decision)
        if wanted is not None and (wanted not in by_asset or wanted in claimed):
            # An unusable answer must not delete a good lexical match.
            updated.append(record)
            tally["ignored"] += 1
            continue
        if wanted is not None:
            claimed.add(wanted)
        current = record.get("legacy_asset")
        if wanted == current:
            tally["agreed" if current else "unchanged"] += 1
            # A confirmed identical-wording match already sits in the strongest tier; a confirmed
            # close-wording match has now been ruled on, so it stops asking for a human look.
            route = record["route"] if record.get("route") == "confident" else ("adjudicated" if current else "missing")
        elif wanted is None:
            tally["rejected" if current else "unchanged"] += 1
            route = "missing"
        elif current:
            tally["replaced"] += 1
            route = "adjudicated"
        else:
            tally["recovered"] += 1
            route = "adjudicated"
        icon = by_asset.get(wanted) if wanted else None
        updated.append(
            {
                **record,
                "route": route,
                "lexical_route": record.get("lexical_route", record.get("route", "missing")),
                "legacy_asset": wanted,
                "legacy_label": icon["label"] if icon else None,
                "legacy_png": icon["png"] if icon else None,
            }
        )
    return updated, {"applied": True, "model": resolved_model, **dict(tally)}


def map_paper(
    contract_dir: str | Path,
    run_dir: str | Path,
    *,
    adjudicate: bool = False,
    call: Callable[..., dict] | None = None,
    model: str | None = None,
    timeout: int = 180,
    retries: int = 1,
) -> dict[str, Any]:
    """Map one paper's contract slots onto the icons its legacy run already owns."""
    slots = contract_icon_slots(contract_dir)
    icons = legacy_run_icons(run_dir)
    records = lexical_mapping(slots, icons)
    adjudication: dict[str, Any] = {"applied": False}
    if adjudicate and records:
        records, adjudication = adjudicate_records(
            records,
            icons,
            paper=Path(contract_dir).name,
            call=call,
            model=model,
            timeout=timeout,
            retries=retries,
        )
    return {"slots": slots, "icons": icons, "records": records, "adjudication": adjudication}


def _sheet_row(record: dict[str, Any], paper: str) -> dict[str, Any]:
    """Flatten one slot record into the row a reviewer reads."""
    return {
        "paper": paper,
        "slot": record["slot"],
        "label": record["label"],
        "bucket": record["bucket"],
        "route": record["route"],
        "lexical_route": record.get("lexical_route", ""),
        "score": record.get("score", 0.0),
        "legacy_asset": record.get("legacy_asset") or "",
        "legacy_label": record.get("legacy_label") or "",
        "legacy_png": record.get("legacy_png") or "",
    }


def _sheet_name(record: dict[str, Any]) -> str:
    """Which sheet a slot belongs on: a match that survived review, or work still owed."""
    route = str(record.get("route") or "missing")
    if route in SHEET_NAMES:
        return route
    return "confident" if record.get("legacy_asset") else "missing"


def _write_sheets(out_root: Path, sheets: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    written = {}
    for name in SHEET_NAMES:
        path = out_root / f"contact_sheet_{name}.csv"
        rows = sheets.get(name) or []
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["paper", "slot"])
            writer.writeheader()
            writer.writerows(rows)
        written[name] = str(path)
    return written


def _coverage_lines(summary: dict[str, Any], per_paper_matches: list[int]) -> list[str]:
    ordered = sorted(per_paper_matches)
    median = ordered[len(ordered) // 2] if ordered else 0
    lines = [
        "# Icon reuse coverage (legacy runs -> sealed middleware contract)",
        "",
        "Free deterministic mapping"
        + (" plus one VLM adjudication call per paper." if summary.get("adjudication_calls") else "; no API calls."),
        "",
        f"- papers read: {summary['papers']}",
        f"- papers with usable legacy icons: {summary['papers_with_icons']}",
        f"- contract slots wanting an icon (inputs/outputs/modules): {summary['slots']}",
        f"- legacy icons usable: {summary['legacy_icons']}",
        f"- **reusable slots: {summary['matched']} of {summary['slots']} ({summary['coverage']:.1%})**",
        f"  - identical wording (confident): {summary['confident']}",
        f"  - close wording, not confirmed by the VLM (review): {summary['review']}",
        f"  - confirmed or recovered by the VLM: {summary['adjudicated']}",
        f"- no legacy icon, must be drawn: {summary['missing']}",
        f"- papers with at least one reusable icon: {summary['papers_with_matches']}",
        f"- reusable icons per such paper: median {median}, max {ordered[-1] if ordered else 0}",
        "",
        "Sheets: `contact_sheet_confident.csv`, `contact_sheet_review.csv`, `contact_sheet_missing.csv`,",
        "and the machine table `mapping.jsonl`. Staging with `--stage-into <batch root>` copies the kept",
        "PNGs to `<batch root>/runs/<paper>/attempt_01/round_00/candidate_00/assets/<slot>/candidate_00.png`,",
        "where a fresh run reuses them instead of paying the image provider.",
    ]
    return lines


def map_corpus(
    contracts: str | Path,
    legacy_root: str | Path,
    out: str | Path,
    *,
    limit: int = 0,
    adjudicate: bool = False,
    call: Callable[..., dict] | None = None,
    model: str | None = None,
    timeout: int = 180,
    retries: int = 1,
    workers: int = 1,
    stage_into: str | Path | None = None,
    attempt: str = DEFAULT_ATTEMPT,
    round_index: int = 0,
    candidate: str = DEFAULT_CANDIDATE,
) -> dict[str, Any]:
    """Map every exported contract onto its legacy run's icons and report the coverage."""
    contracts_root = Path(contracts)
    legacy = Path(legacy_root)
    out_root = ensure_dir(out)
    papers = sorted(entry.name for entry in contracts_root.iterdir() if entry.is_dir())
    if limit:
        papers = papers[: int(limit)]

    sheets: dict[str, list[dict[str, Any]]] = {name: [] for name in SHEET_NAMES}
    mapping_rows: list[dict[str, Any]] = []
    counts = Counter()
    per_paper_matches: list[int] = []
    def _map_one(paper: str) -> tuple[str, dict[str, Any]]:
        return paper, map_paper(
            contracts_root / paper,
            legacy / paper / attempt,
            adjudicate=adjudicate,
            call=call,
            model=model,
            timeout=timeout,
            retries=retries,
        )

    if adjudicate and int(workers) > 1:
        # The adjudication is one slow network call per paper, so the papers are the unit
        # of parallelism; every other stage of this command is local file reading.
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            mapped_papers = list(pool.map(_map_one, papers))
    else:
        mapped_papers = [_map_one(paper) for paper in papers]

    for paper, mapped in mapped_papers:
        records = mapped["records"]
        counts["papers"] += 1
        counts["legacy_icons"] += len([icon for icon in mapped["icons"] if icon.get("usable")])
        counts["papers_with_icons"] += 1 if any(icon.get("usable") for icon in mapped["icons"]) else 0
        if mapped["adjudication"].get("applied"):
            counts["adjudication_calls"] += 1
        primary = [record for record in records if record.get("primary")]
        counts["slots"] += len(primary)
        matches = 0
        for record in primary:
            route = record["route"]
            if route == "confident":
                counts["confident"] += 1
            elif route == "review":
                counts["review"] += 1
            elif route == "adjudicated":
                counts["adjudicated"] += 1
            counts["missing"] += 1 if route == "missing" else 0
            counts["matched"] += 1 if record.get("legacy_asset") else 0
            matches += 1 if record.get("legacy_asset") else 0
        per_paper_matches.append(matches)
        counts["papers_with_matches"] += 1 if matches else 0
        for record in primary:
            sheets[_sheet_name(record)].append(_sheet_row(record, paper))
        mapping_rows.append({"paper": paper, "records": records, "adjudication": mapped["adjudication"]})

    summary = {
        "papers": counts["papers"],
        "papers_with_icons": counts["papers_with_icons"],
        "slots": counts["slots"],
        "legacy_icons": counts["legacy_icons"],
        "confident": counts["confident"],
        "review": counts["review"],
        "adjudicated": counts["adjudicated"],
        "missing": counts["missing"],
        "matched": counts["matched"],
        "papers_with_matches": counts["papers_with_matches"],
        "adjudication_calls": counts["adjudication_calls"],
    }
    summary["coverage"] = round(summary["matched"] / summary["slots"], 4) if summary["slots"] else 0.0

    write_text(out_root / "mapping.jsonl", "\n".join(json.dumps(row, ensure_ascii=False) for row in mapping_rows) + "\n")
    sheets_written = _write_sheets(out_root, sheets)
    write_text(out_root / "coverage_report.md", "\n".join(_coverage_lines(summary, per_paper_matches)) + "\n")
    write_json(out_root / "analysis.json", {"summary": summary, "per_paper_matches": per_paper_matches})

    payload: dict[str, Any] = {
        "ok": True,
        "summary": f"Reusable icons for {summary['matched']} of {summary['slots']} contract slots ({summary['coverage']:.1%}).",
        "out": str(out_root),
        "sheets": sheets_written,
        "mapping": str(out_root / "mapping.jsonl"),
        **summary,
    }
    if stage_into:
        payload["staging"] = stage_reuse(
            mapping_rows,
            stage_into,
            attempt=attempt,
            round_index=round_index,
            candidate=candidate,
            routes=STAGING_ROUTES,
        )
    return payload


def stage_reuse(
    mapping_rows: list[dict[str, Any]],
    batch_root: str | Path,
    *,
    attempt: str = DEFAULT_ATTEMPT,
    round_index: int = 0,
    candidate: str = DEFAULT_CANDIDATE,
    force: bool = False,
    routes: Sequence[str] | None = STAGING_ROUTES,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Copy each kept legacy PNG where a fresh run reads its icons.

    Only the routes in ``routes`` are copied.  A slot the mapper was not sure about is a
    guess, and an icon staged for a guess ends up inside a delivered figure as if the VLM
    had confirmed it; the two routes that carry a confirmation are the default.  ``dry_run``
    reports the same plan without touching the batch root.
    """
    root = Path(batch_root)
    allowed = None if routes is None else {str(route) for route in routes}
    staged, skipped, filtered, copied = [], 0, 0, 0
    for row in mapping_rows:
        paper = str(row.get("paper") or "")
        if not paper:
            continue
        for record in row.get("records") or []:
            if allowed is not None and str(record.get("route") or "") not in allowed:
                filtered += 1
                continue
            slot = str(record.get("slot") or "")
            source = str(record.get("legacy_png") or "")
            if not slot or not source or not Path(source).is_file():
                skipped += 1
                continue
            target = root / "runs" / paper / attempt / f"round_{int(round_index):02d}" / candidate / "assets" / slot / "candidate_00.png"
            if target.exists() and not force:
                skipped += 1
                continue
            if dry_run:
                staged.append({"paper": paper, "slot": slot, "source": source, "target": str(target)})
                continue
            ensure_dir(target.parent)
            shutil.copyfile(source, target)
            copied += target.stat().st_size
            staged.append({"paper": paper, "slot": slot, "source": source, "target": str(target)})
    return {
        "ok": True,
        "summary": f"Staged {len(staged)} reusable icons into {root}.",
        "staged": len(staged),
        "skipped": skipped,
        "filtered": filtered,
        "bytes": copied,
        "root": str(root),
        "records": staged,
    }


def read_mapping(path: str | Path) -> list[dict[str, Any]]:
    """Read a mapping table written by :func:`map_corpus`, one JSON object per paper."""
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def stage_from_mapping(
    mapping_path: str | Path,
    batch_root: str | Path,
    *,
    routes: Sequence[str] = STAGING_ROUTES,
    attempt: str = DEFAULT_ATTEMPT,
    round_index: int = 0,
    candidate: str = DEFAULT_CANDIDATE,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stage the icons a mapping table already certified, without asking the VLM again.

    A wave that re-draws papers against the sealed contract reuses the mapping an earlier
    adjudication run already paid for: one call per paper.  Reading that table and copying
    the PNGs is free, while re-mapping the corpus would pay for every call a second time.
    """
    rows = read_mapping(mapping_path)
    allowed = [str(route) for route in routes]
    routing = Counter(
        str(record.get("route") or "unrouted")
        for row in rows
        for record in (row.get("records") or [])
    )
    staged = stage_reuse(
        rows,
        batch_root,
        attempt=attempt,
        round_index=round_index,
        candidate=candidate,
        force=force,
        routes=allowed,
        dry_run=dry_run,
    )
    counts = {key: value for key, value in staged.items() if key not in {"ok", "summary", "records"}}
    report: dict[str, Any] = {
        "ok": True,
        "summary": (
            f"{'Would stage' if dry_run else 'Staged'} {counts['staged']} icons into {counts['root']} "
            f"from {len(rows)} mapped papers; {counts['filtered']} slots stayed behind."
        ),
        "mapping": str(mapping_path),
        "routes": allowed,
        "routing": dict(sorted(routing.items())),
        "dry_run": bool(dry_run),
        "records_total": len(staged["records"]),
        "records_sample": staged["records"][:20],
        **counts,
    }
    if not dry_run:
        report_path = Path(batch_root) / STAGE_REPORT_FILE
        write_json(report_path, {**report, "records": staged["records"]})
        report["report"] = str(report_path)
    return report
