"""A reproducible corpus subset and the inputs every paper in it needs.

The first billed batch of the merged run has to be small enough to check and
large enough to predict the full corpus from.  This module fixes that subset in
one place: it samples a set number of papers per domain out of the frozen split
ledger, refuses every split the hidden test owns, resolves each sampled paper
back to the paper file it was collected from and to the framework figure it was
frozen with, and writes a versioned manifest so the same hundred papers can be
run again next week.

Nothing here calls a provider.  The figure labels that report which structures
the sample covers come from the Pilot labeller and are merged in afterwards.

The paper file is reachable two ways.  The snapshot keeps its own copy as an
archive object, addressed by the archive root the caller passes, and that copy
is the one the snapshot was verified against, so it wins whenever it is
indexed.  When no archive root is given the original file is looked up in the
collection directories the snapshot was built from, under either of the two
names the collector gave them (arxiv.pdf next to the arxiv id, official.pdf
next to the doi for publisher-only papers).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from rfs.judge_training.icon_reuse import (
    STAGING_ROUTES,
    contract_icon_slots,
    legacy_run_icons,
    read_mapping,
)
from rfs.judge_training.pilot_graphs import safe_paper_directory
from rfs.judge_training.pilot_selection import (
    candidate_pool,
    load_figure_index,
    load_snapshot_papers,
    primary_figure_asset,
)
from rfs.judge_training.source_snapshot_io import sha256_file
from rfs.judge_training.split_ledger import load_split_ledger
from rfs.utils import write_json, write_text

SUBSET_CONTRACT = "rfs.corpus_subset"
SUBSET_VERSION = 1
DEFAULT_SUBSET_ID = "paper-d0-subset-100"
DEFAULT_PER_DOMAIN = 20

#: Papers per domain, split into (train, validation) the way the Pilot quotas do.
VALIDATION_SHARE = 0.2

SELECTION_RULE = (
    "per domain and per split, take the first candidate in a stable seed-hash order "
    "until the per-domain quota is met; hidden test papers are never sampled"
)

MANIFEST_FILE = "subset_manifest.json"
LEDGER_FILE = "subset_ledger.jsonl"

#: What a paper needs before a reference-versus-generated pair can ever be built from it.
CONTRACT_SUBSET_RULE = (
    "every paper of the frozen snapshot that carries a sealed middleware contract and an "
    "earlier run whose icons the re-draw may reuse; papers a named batch root already "
    "delivered a figure for are dropped"
)
CONTRACT_NAMES_FILE = "paper_names.txt"
DELIVERED_DECK = "editable_composition.pptx"
DEFAULT_LEGACY_ATTEMPT = "attempt_01"

LABEL_FIELDS = ("figure_type", "structure", "node_count", "panel_count", "has_arrows", "complexity")


def quotas_for(domains: Sequence[str], per_domain: int) -> dict[str, tuple[int, int]]:
    """Split a per-domain total into a train part and a validation part."""
    validation = int(round(per_domain * VALIDATION_SHARE))
    validation = min(max(validation, 1), per_domain - 1) if per_domain > 1 else 0
    return {str(domain): (per_domain - validation, validation) for domain in domains}


def build_subset(
    snapshot: str | Path,
    split_dir: str | Path,
    *,
    subset_id: str = DEFAULT_SUBSET_ID,
    per_domain: int = DEFAULT_PER_DOMAIN,
    domains: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Sample a fixed per-domain subset from the frozen snapshot and split ledger."""
    snapshot = Path(snapshot)
    split_rows = load_split_ledger(split_dir)
    papers = load_snapshot_papers(snapshot)
    wanted = sorted({str(domain) for domain in domains}) if domains else sorted(
        {str(row.get("primary_domain") or "unknown") for row in split_rows}
    )
    pool = candidate_pool(
        split_rows,
        papers,
        seed=subset_id,
        quotas=quotas_for(wanted, per_domain),
        pool_multiplier=1,
    )
    figure_index = load_figure_index(snapshot)
    selected: list[dict[str, Any]] = []
    for candidate in sorted(pool["candidates"], key=lambda item: (item["domain"], item["split"], item["paper_id"])):
        asset_id = str(candidate.get("figure_asset_id") or "")
        asset = figure_index.get(asset_id) or {}
        selected.append(
            {
                "paper_id": str(candidate["paper_id"]),
                "split": str(candidate["split"]),
                "domain": str(candidate["domain"]),
                "figure_asset_id": asset_id,
                "figure_relative_path": str(asset.get("relative_path") or ""),
                "title": str(candidate.get("title") or ""),
            }
        )
    by_domain: dict[str, int] = {}
    for entry in selected:
        by_domain[entry["domain"]] = by_domain.get(entry["domain"], 0) + 1
    ledger_path = Path(split_dir) / "split_ledger.jsonl"
    return {
        "contract": SUBSET_CONTRACT,
        "version": SUBSET_VERSION,
        "subset_id": str(subset_id),
        "per_domain": int(per_domain),
        "domains": wanted,
        "quotas": {domain: list(quota) for domain, quota in quotas_for(wanted, per_domain).items()},
        "selection_rule": SELECTION_RULE,
        "source": {
            "snapshot": str(snapshot),
            "snapshot_id": snapshot.name,
            "split": str(split_dir),
            "split_ledger_sha256": sha256_file(ledger_path) if ledger_path.is_file() else None,
        },
        "papers": selected,
        "by_domain": {domain: by_domain.get(domain, 0) for domain in wanted},
        "shortfalls": list(pool["skipped"]),
    }


def apply_labels(subset: dict[str, Any], label_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Attach the Pilot's figure labels and report which structures the sample covers."""
    by_paper = {str(row.get("paper_id")): row for row in label_rows}
    structures: dict[str, int] = {}
    figure_types: dict[str, int] = {}
    labelled = 0
    for entry in subset.get("papers") or []:
        label = by_paper.get(str(entry.get("paper_id"))) or {}
        entry["labelled"] = bool(label)
        for field in LABEL_FIELDS:
            if field in label:
                entry[field] = label[field]
        if not label:
            continue
        labelled += 1
        structure = str(label.get("structure") or "unknown")
        figure_type = str(label.get("figure_type") or "unknown")
        structures[structure] = structures.get(structure, 0) + 1
        figure_types[figure_type] = figure_types.get(figure_type, 0) + 1
    subset["label_report"] = {
        "labelled": labelled,
        "unlabelled": len(subset.get("papers") or []) - labelled,
        "by_structure": dict(sorted(structures.items())),
        "by_figure_type": dict(sorted(figure_types.items())),
        "by_domain_structure": _by_domain_structure(subset),
    }
    return subset


def _by_domain_structure(subset: dict[str, Any]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for entry in subset.get("papers") or []:
        domain = str(entry.get("domain") or "unknown")
        structure = str(entry.get("structure") or ("unlabelled" if not entry.get("labelled") else "unknown"))
        table.setdefault(domain, {})
        table[domain][structure] = table[domain].get(structure, 0) + 1
    return {domain: dict(sorted(counts.items())) for domain, counts in sorted(table.items())}


def write_subset_version(out_dir: str | Path, subset: dict[str, Any]) -> dict[str, Any]:
    """Write a subset version once. An existing sealed version is never overwritten."""
    target = Path(out_dir)
    if (target / "COMPLETE").is_file():
        raise FileExistsError(f"Refusing to overwrite a sealed subset version: {target}")
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / MANIFEST_FILE, subset)
    write_text(
        target / LEDGER_FILE,
        "".join(json_line(entry) for entry in subset.get("papers") or []),
    )
    write_text(
        target / "COMPLETE",
        f"subset_id={subset.get('subset_id')} papers={len(subset.get('papers') or [])}\n",
    )
    return {
        "out_dir": str(target),
        "papers": len(subset.get("papers") or []),
        "files": [MANIFEST_FILE, LEDGER_FILE, "COMPLETE"],
    }


def json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"


def load_subset(path: str | Path) -> dict[str, Any]:
    """Load a subset version from its directory or its manifest file."""
    source = Path(path)
    if source.is_dir():
        source = source / MANIFEST_FILE
    if not source.is_file():
        raise FileNotFoundError(f"Subset version is missing {MANIFEST_FILE}: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != SUBSET_CONTRACT:
        raise ValueError(f"Not a corpus subset manifest: {source}")
    return payload


def arxiv_token(arxiv_id: str) -> str:
    """Key the paper file of an arxiv paper by its id."""
    return f"arxiv:{str(arxiv_id).strip().lower()}"


def doi_token(doi: str) -> str:
    """Key the paper file of a publisher-only paper by its doi."""
    return "doi:" + str(doi).strip().lower().replace("/", "_")


def index_source_pdfs(roots: Iterable[str | Path]) -> dict[tuple[str, str], Path]:
    """Map (domain, file token) to the paper file the snapshot was built from."""
    index: dict[tuple[str, str], Path] = {}
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        for path in base.rglob("*.pdf"):
            name = path.name.lower()
            if name == "arxiv.pdf" and path.parent.parent.parent.name == "assets":
                index.setdefault((path.parent.parent.name, arxiv_token(path.parent.name)), path)
            elif name == "official.pdf" and path.parent.parent.name == "official":
                index.setdefault((path.parent.parent.parent.name, doi_token(path.parent.name)), path)
    return index


def index_snapshot_pdfs(snapshot: str | Path, archive_root: str | Path) -> dict[str, Path]:
    """Map each frozen paper id to the paper file the snapshot archived for it.

    The snapshot addresses its own copies as archive objects, so the file is the
    archive root plus the object's relative path.  Only objects that are really
    on disk are indexed: a paper whose archive object went missing is then
    reported as unresolved instead of being handed out as a broken path.
    """
    manifest = Path(snapshot) / "asset_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Snapshot is missing asset_manifest.jsonl: {manifest}")
    root = Path(archive_root)
    index: dict[str, Path] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if str(record.get("role") or "") != "paper_pdf":
            continue
        paper_id = str(record.get("paper_id") or "").strip()
        relative = str(record.get("relative_path") or "").strip()
        if not paper_id or not relative:
            continue
        candidate = root / relative
        if candidate.is_file():
            index.setdefault(paper_id, candidate)
    return index


def resolve_paper_inputs(
    snapshot: str | Path,
    entry: dict[str, Any],
    *,
    pdf_index: dict[tuple[str, str], Path],
    pdf_by_paper: dict[str, Path] | None = None,
    papers: dict[str, dict[str, Any]] | None = None,
    figure_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve one sampled paper to its paper file and its frozen framework figure.

    The snapshot's own archived copy is preferred when it was indexed, because
    it is the file the snapshot was verified against; the collection
    directories are only the fallback for a paper whose archive object is gone.
    """
    paper_id = str(entry.get("paper_id"))
    paper = (papers or {}).get(paper_id) or {}
    source_record = (paper.get("provenance") or {}).get("source_record") or {}
    arxiv_id = str(source_record.get("arxiv_id") or "").strip()
    doi = str(source_record.get("doi") or paper.get("doi") or "").strip()
    domain = str(paper.get("primary_domain") or entry.get("domain") or "")
    archived = (pdf_by_paper or {}).get(paper_id)
    pdf = Path(archived) if archived is not None else None
    paper_source = "snapshot_archive" if pdf is not None else ""
    if pdf is None and arxiv_id and domain:
        pdf = pdf_index.get((domain, arxiv_token(arxiv_id)))
        paper_source = "arxiv" if pdf is not None else ""
    if pdf is None and doi and domain:
        pdf = pdf_index.get((domain, doi_token(doi)))
        paper_source = "official" if pdf is not None else ""
    asset_id = str(entry.get("figure_asset_id") or primary_figure_asset(paper))
    asset = (figure_index or {}).get(asset_id) or {}
    relative = str(asset.get("relative_path") or entry.get("figure_relative_path") or "")
    reference = Path(snapshot) / relative if relative else None
    missing: list[str] = []
    if not arxiv_id and not doi:
        missing.append("no_paper_identity")
    elif pdf is None:
        missing.append("paper_pdf")
    if reference is None or not reference.is_file():
        missing.append("framework_figure")
    return {
        "paper_id": paper_id,
        "domain": domain,
        "split": str(entry.get("split") or ""),
        "arxiv_id": arxiv_id,
        "doi": doi,
        "paper_source": paper_source,
        "pdf": str(pdf) if pdf is not None else None,
        "reference": str(reference) if reference is not None and reference.is_file() else None,
        "missing": missing,
    }


def load_snapshot_index(snapshot: str | Path) -> dict[str, dict[str, Any]]:
    """Index the frozen paper ledger by paper id so inputs can be resolved by lookup."""
    return {str(row.get("paper_id")): row for row in load_snapshot_papers(snapshot)}


def delivered_paper_directories(roots: Sequence[str | Path]) -> set[str]:
    """Directory names of every paper that already has a run directory on disk.

    A batch writes one paper into ``safe_paper_directory(paper_id)``, so the
    directory name - never the raw paper id - is the form two sources can be
    compared in.  Ids such as ``10.1021/acsnano.5c17401`` or
    ``iclr:2026:10006567`` carry separators that can never appear in that name,
    and a check that compared the two literally matched nothing: it answered
    "not drawn" for most of the corpus and bought the whole list twice.
    """

    found: set[str] = set()
    for root in roots:
        runs = Path(root) / "runs"
        if not runs.is_dir():
            continue
        for child in runs.iterdir():
            if child.is_dir():
                found.add(safe_paper_directory(child.name))
    return found


def select_undelivered(
    papers: Sequence[dict[str, Any]],
    delivered: Iterable[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Split papers into the ones still worth paying for and the ones already drawn.

    Both sides go through ``safe_paper_directory``, so a paper id and the
    directory it owns always agree however the id was written down.  Two ids
    that would share one directory are refused rather than silently merged into
    one paper: that collision is the only way this comparison can lose a paper,
    and losing one without saying so is what let a paid batch run twice.
    """

    owner: dict[str, str] = {}
    for paper in papers:
        paper_id = str(paper.get("paper_id") or "")
        name = safe_paper_directory(paper_id)
        first = owner.setdefault(name, paper_id)
        if first != paper_id:
            raise ValueError(f"paper ids {first!r} and {paper_id!r} both own the run directory {name!r}")
    drawn = {safe_paper_directory(entry) for entry in delivered}
    kept: list[dict[str, Any]] = []
    skipped: list[str] = []
    for paper in papers:
        paper_id = str(paper.get("paper_id") or "")
        if safe_paper_directory(paper_id) in drawn:
            skipped.append(paper_id)
        else:
            kept.append(paper)
    return kept, skipped


def delivered_deck_names(roots: Sequence[str | Path]) -> set[str]:
    """Directory names of the papers a batch root already delivered a figure for.

    Only a real delivery counts.  An attempt that was killed half way owns no
    ``editable_composition.pptx`` and its next attempt continues from the work already paid
    for, so it has to stay in the list - dropping it would pay for that paper twice.
    """

    found: set[str] = set()
    for root in roots:
        runs = Path(root) / "runs"
        if not runs.is_dir():
            continue
        for paper in runs.iterdir():
            if not paper.is_dir():
                continue
            for attempt in paper.iterdir():
                if attempt.is_dir() and (attempt / DELIVERED_DECK).is_file():
                    found.add(safe_paper_directory(paper.name))
                    break
    return found


def reusable_slots_by_paper(mapping: str | Path) -> dict[str, int]:
    """Count, per paper directory, the contract slots a mapping table certified a PNG for.

    Only the routes that carry a confirmation are counted, so the number is the same one a
    staged wave will really reuse instead of a looser "a candidate exists" count.
    """

    allowed = {str(route) for route in STAGING_ROUTES}
    counts: Counter[str] = Counter()
    for row in read_mapping(mapping):
        paper = str(row.get("paper") or "")
        if not paper:
            continue
        counts[paper] += sum(
            1
            for record in (row.get("records") or [])
            if str(record.get("route") or "") in allowed and record.get("legacy_png")
        )
    return dict(counts)


def build_contract_subset(
    snapshot: str | Path,
    *,
    contracts: str | Path,
    legacy_root: str | Path,
    out: str | Path,
    subset_id: str,
    split: str | Path | None = None,
    names_out: str | Path | None = None,
    exclude_drawn: Sequence[str | Path] = (),
    mapping: str | Path | None = None,
    legacy_attempt: str = DEFAULT_LEGACY_ATTEMPT,
    require_legacy_icons: bool = True,
    limit: int = 0,
) -> dict[str, Any]:
    """Seal every snapshot paper that can carry a reference pair and still needs drawing.

    Two halves have to exist on disk for a paper to be worth paying for.  The sealed
    middleware contract is what the re-draw adopts, and the paper's earlier run is where the
    icons it reuses come from; a paper missing either half cannot produce a sample, so the
    batch never spends a provider call on it.  Every paper that is left out is reported by
    reason, because "we drew 4,957 of 13,895" is only meaningful next to why.
    """

    snapshot = Path(snapshot)
    contracts_root = Path(contracts)
    legacy = Path(legacy_root)
    if not contracts_root.is_dir():
        raise FileNotFoundError(f"the contract export root is missing: {contracts_root}")
    if not legacy.is_dir():
        raise FileNotFoundError(f"the legacy runs root is missing: {legacy}")

    split_of = (
        {str(row.get("paper_id") or ""): str(row.get("split") or "") for row in load_split_ledger(split)}
        if split
        else {}
    )
    papers = load_snapshot_papers(snapshot)
    figures = load_figure_index(snapshot)
    reusable = reusable_slots_by_paper(mapping) if mapping else {}
    delivered = delivered_deck_names(exclude_drawn)

    selected: list[dict[str, Any]] = []
    no_contract: list[str] = []
    no_legacy: list[str] = []
    no_icon: list[str] = []
    no_slots: list[str] = []
    no_figure: list[str] = []
    already_drawn: list[str] = []
    duplicates: list[str] = []
    owner: dict[str, str] = {}
    slots_total = 0
    reusable_total = 0
    for row in papers:
        paper_id = str(row.get("paper_id") or "")
        if not paper_id:
            continue
        name = safe_paper_directory(paper_id)
        first = owner.setdefault(name, paper_id)
        if first != paper_id:
            duplicates.append(paper_id)
            continue
        if name in delivered:
            already_drawn.append(paper_id)
            continue
        contract_dir = contracts_root / name
        if not contract_dir.is_dir():
            no_contract.append(paper_id)
            continue
        run_dir = legacy / name / legacy_attempt
        if not run_dir.is_dir():
            no_legacy.append(paper_id)
            continue
        if require_legacy_icons and not any(icon.get("usable") for icon in legacy_run_icons(run_dir)):
            no_icon.append(paper_id)
            continue
        slots = contract_icon_slots(contract_dir)
        if not slots:
            no_slots.append(paper_id)
        asset_id = primary_figure_asset(row)
        asset = figures.get(asset_id) or {}
        if not asset_id:
            no_figure.append(paper_id)
        slots_total += len(slots)
        reusable_total += min(int(reusable.get(name, 0)), len(slots))
        selected.append(
            {
                "paper_id": paper_id,
                "directory": name,
                "split": split_of.get(paper_id) or "train",
                "domain": str(row.get("primary_domain") or "unknown"),
                "figure_asset_id": asset_id,
                "figure_relative_path": str(asset.get("relative_path") or ""),
                "title": str(row.get("title") or ""),
                "contract_slots": len(slots),
            }
        )
        if limit and len(selected) >= int(limit):
            break

    by_domain: dict[str, int] = {}
    for entry in selected:
        by_domain[entry["domain"]] = by_domain.get(entry["domain"], 0) + 1
    ledger_path = Path(split) / "split_ledger.jsonl" if split else None
    payload = {
        "contract": SUBSET_CONTRACT,
        "version": SUBSET_VERSION,
        "subset_id": str(subset_id),
        "per_domain": 0,
        "domains": sorted(by_domain),
        "quotas": {},
        "selection_rule": CONTRACT_SUBSET_RULE,
        "source": {
            "snapshot": str(snapshot),
            "snapshot_id": snapshot.name,
            "split": str(split) if split else None,
            "split_ledger_sha256": sha256_file(ledger_path) if ledger_path and ledger_path.is_file() else None,
            "contracts": str(contracts_root),
            "legacy_root": str(legacy),
            "legacy_attempt": str(legacy_attempt),
            "exclude_drawn": [str(root) for root in exclude_drawn],
            "reuse_mapping": str(mapping) if mapping else None,
        },
        "papers": selected,
        "by_domain": by_domain,
        "requested": len(papers),
        "contract_slots": slots_total,
        "reusable_slots": reusable_total,
        "already_drawn": already_drawn,
        "no_contract": no_contract,
        "no_legacy_run": no_legacy,
        "no_legacy_icon": no_icon,
        "no_slots": no_slots,
        "no_figure": no_figure,
        "duplicate_directories": duplicates,
    }
    written = write_subset_version(out, payload)
    if names_out:
        write_text(Path(names_out), "".join(f"{entry['directory']}\n" for entry in selected))
    return {
        "ok": True,
        "summary": (
            f"{written['papers']} of {len(papers)} snapshot papers can carry a reference pair; "
            f"{len(already_drawn)} were already delivered, {len(no_contract)} have no contract, "
            f"{len(no_legacy)} have no earlier run and {len(no_icon)} own no reusable icon."
        ),
        "out": written["out_dir"],
        "subset_id": str(subset_id),
        "papers": written["papers"],
        "requested": len(papers),
        "by_domain": by_domain,
        "contract_slots": slots_total,
        "reusable_slots": reusable_total,
        "already_drawn": len(already_drawn),
        "no_contract": len(no_contract),
        "no_legacy_run": len(no_legacy),
        "no_legacy_icon": len(no_icon),
        "no_slots": len(no_slots),
        "no_figure": len(no_figure),
        "duplicate_directories": len(duplicates),
        "names_out": str(names_out) if names_out else None,
        "files": written["files"],
    }


__all__ = [
    "CONTRACT_NAMES_FILE",
    "CONTRACT_SUBSET_RULE",
    "DEFAULT_PER_DOMAIN",
    "DEFAULT_SUBSET_ID",
    "MANIFEST_FILE",
    "SUBSET_CONTRACT",
    "apply_labels",
    "build_contract_subset",
    "build_subset",
    "delivered_deck_names",
    "delivered_paper_directories",
    "index_source_pdfs",
    "index_snapshot_pdfs",
    "load_snapshot_index",
    "load_subset",
    "quotas_for",
    "resolve_paper_inputs",
    "reusable_slots_by_paper",
    "select_undelivered",
    "write_subset_version",
]
