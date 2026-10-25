"""Deterministic, domain-stratified split ledger for an immutable paper snapshot.

Stage 2 of the Judge execution roadmap. Every paper of a frozen source snapshot is
assigned to exactly one of ``train`` / ``validation`` / ``hidden_test`` by a stable
hash bucket over ``(seed, paper_id)``. Bucket thresholds are absolute, so a later
batch can only add papers to a split and can never move an already assigned paper
between splits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from rfs.utils import write_json, write_text

from .perceptual_hash import audit_perceptual_leakage
from .source_snapshot_io import sha256_file

SPLIT_ORDER = ("train", "validation", "hidden_test")
DEFAULT_RATIOS = (0.65, 0.15, 0.20)
DEFAULT_SEED = "paper-d0-split-v001"
SCHEMA_VERSION = "judge-split-v1"

ARTIFACT_FILES = (
    "split_manifest.json",
    "split_ledger.jsonl",
    "leakage_report.json",
    "domain_distribution.json",
    "hidden_test_manifest.json",
    "validation_report.json",
)

HIDDEN_TEST_FORBIDDEN_USES = (
    "prompt_tuning",
    "dpo_pair_selection",
    "memory_exam_selection",
    "checkpoint_selection",
)

DERIVED_HIDDEN_TEST_BUCKETS = ("unseen_generator", "ab_swap", "pretty_but_wrong")

INHERITANCE_RULE = (
    "真实图、破坏图、Generator 候选图、Graph、PPTX 与轨迹一律继承源论文的 split"
)

FIGURE_IDENTITY_NAMESPACE = "figure_sha256"
FIGURE_GROUP_REVIEW_LIMIT = 8

_U64 = 1 << 64
_TITLE_NOISE = re.compile(r"[^0-9a-z]+")
_RATIO_SPLIT = re.compile(r"[/,\s]+")


def normalize_title(title: Any) -> str | None:
    """Return the identity-normalised title used for duplicate detection."""
    if not title:
        return None
    collapsed = " ".join(_TITLE_NOISE.sub(" ", str(title).lower()).split())
    return collapsed or None


def parse_ratios(value: str | Sequence[float] | None) -> tuple[float, float, float]:
    """Parse ``65/15/20`` or an explicit fraction triple into three fractions."""
    if value is None:
        return DEFAULT_RATIOS
    if isinstance(value, str):
        parts = [item for item in _RATIO_SPLIT.split(value.strip()) if item]
        if len(parts) != 3:
            raise ValueError(f"Expected three ratios such as 65/15/20, got {value!r}")
        numbers = [float(item) for item in parts]
    else:
        numbers = [float(item) for item in value]
        if len(numbers) != 3:
            raise ValueError(f"Expected three ratios, got {value!r}")
    if any(item <= 0 for item in numbers):
        raise ValueError(f"Ratios must be positive, got {value!r}")
    total = sum(numbers)
    if any(item > 1 for item in numbers):
        if abs(total - 100.0) > 1e-6:
            raise ValueError(f"Percent ratios must sum to 100, got {total}")
        numbers = [item / 100.0 for item in numbers]
    elif abs(total - 1.0) > 1e-9:
        raise ValueError(f"Fraction ratios must sum to 1, got {total}")
    return (numbers[0], numbers[1], numbers[2])


def bucket_u64(paper_id: str, seed: str) -> int:
    """Return the stable 64-bit bucket of one paper; depends only on seed and id."""
    digest = hashlib.sha256(f"{seed}\x1f{paper_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def split_thresholds(ratios: Sequence[float]) -> tuple[int, int]:
    """Return the integer bucket thresholds for train and validation."""
    return int(round(ratios[0] * _U64)), int(round((ratios[0] + ratios[1]) * _U64))


def split_for_bucket(bucket: int, thresholds: tuple[int, int]) -> str:
    """Return the split name for one bucket value."""
    train_below, validation_below = thresholds
    if bucket < train_below:
        return "train"
    if bucket < validation_below:
        return "validation"
    return "hidden_test"


def identity_keys(record: dict[str, Any]) -> dict[str, list[str]]:
    """Return every deduplication key a paper can be recognised by."""
    keys: dict[str, set[str]] = {}

    def add(namespace: str, value: Any) -> None:
        text = str(value or "").strip().lower()
        if text:
            keys.setdefault(namespace, set()).add(text)

    add("paper_id", record.get("paper_id"))
    for alias in record.get("identity_aliases") or []:
        text = str(alias)
        if ":" not in text:
            continue
        namespace, value = text.split(":", 1)
        add(namespace.strip(), value)
    add("title", normalize_title(record.get("title")))
    return {name: sorted(values) for name, values in sorted(keys.items())}


EXPLORATION_SPLIT = "exploration"


def namespaced_identity_keys(keys: dict[str, list[str]]) -> list[str]:
    """Flatten identity namespaces into ``namespace:value`` lookup keys."""
    return [f"{namespace}:{value}" for namespace, values in sorted(keys.items()) for value in values]


def paper_identity_keys(*, paper_id: Any = None, title: Any = None, identity_aliases: Iterable[str] = ()) -> list[str]:
    """Return the lookup keys one produced paper can be recognised by."""
    return namespaced_identity_keys(identity_keys({"paper_id": paper_id, "title": title, "identity_aliases": list(identity_aliases)}))


def build_split_index(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Index every identity key of a split ledger to the split of its paper."""
    index: dict[str, str] = {}
    for row in rows:
        split = str(row.get("split") or "")
        if not split:
            continue
        stored = row.get("identity_keys")
        keys = stored if isinstance(stored, dict) else identity_keys(row)
        flattened = {str(name): [str(item) for item in values] for name, values in keys.items()}
        for key in namespaced_identity_keys(flattened):
            index.setdefault(key.lower(), split)
    return index


def resolve_split(index: dict[str, str], *, paper_id: Any = None, title: Any = None, identity_aliases: Iterable[str] = ()) -> tuple[str | None, dict[str, Any]]:
    """Resolve a produced paper to its frozen split, or report that it is unknown.

    Only papers of the frozen snapshot carry a split. A paper collected after the
    snapshot is deliberately left unresolved so callers can keep it as an
    exploration sample instead of leaking it into training.
    """
    candidates = paper_identity_keys(paper_id=paper_id, title=title, identity_aliases=identity_aliases)
    matched: list[str] = []
    for key in candidates:
        split = index.get(key.lower())
        if not split:
            continue
        matched.append(f"{key}->{split}")
        if not key.lower().startswith("title:"):
            return split, {"source": "split_ledger", "matched_key": key, "matched": matched, "candidates": candidates}
    for key in candidates:
        split = index.get(key.lower())
        if split:
            return split, {"source": "split_ledger", "matched_key": key, "matched": matched, "candidates": candidates}
    return None, {"source": "unresolved", "matched_key": "", "matched": [], "candidates": candidates}


def load_figure_asset_shas(snapshot: str | Path) -> dict[str, str]:
    """Map every figure asset id of a snapshot to the sha256 of its file."""
    path = Path(snapshot) / "asset_manifest.jsonl"
    if not path.is_file():
        return {}
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("role") != "figure":
                continue
            asset_id = str(record.get("asset_id") or "").strip()
            digest = str(record.get("sha256") or "").strip().lower()
            if asset_id and digest:
                mapping[asset_id] = digest
    return mapping


def figure_content_groups(
    papers: Iterable[dict[str, Any]],
    asset_shas: dict[str, str] | None = None,
    rendered_groups: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Group papers that share a figure so one figure cannot cross splits.

    Two independent relations are merged. Papers that own a byte-identical figure
    file are linked by sha256, and papers whose figures render to the same
    perceptual hash are linked by the rendered report; the second relation catches
    the same figure republished as a PDF fragment in one record and as a PNG in
    another. The merged group is bound to the split of its smallest ``paper_id``,
    which stays deterministic and reproducible across reruns.
    """
    asset_shas = asset_shas or {}
    parent: dict[str, str] = {}
    reasons: list[tuple[str, str, str]] = []

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    owners: dict[str, str] = {}
    digests_by_paper: dict[str, set[str]] = {}
    for record in papers:
        paper_id = str(record.get("paper_id") or "").strip()
        if not paper_id:
            continue
        find(paper_id)
        digests: set[str] = set()
        for figure in record.get("figures") or []:
            if not isinstance(figure, dict):
                continue
            for asset_id in figure.get("asset_ids") or []:
                digest = asset_shas.get(str(asset_id))
                if digest:
                    digests.add(digest)
        if not digests:
            continue
        digests_by_paper[paper_id] = digests
        for digest in sorted(digests):
            first = owners.setdefault(digest, paper_id)
            if first != paper_id:
                reasons.append((first, paper_id, "byte_identical_figure"))
                union(first, paper_id)

    for group in rendered_groups or []:
        ids = sorted(
            {
                str(asset.get("paper_id") or "")
                for asset in group.get("assets") or []
                if asset.get("paper_id")
            }
        )
        for other in ids[1:]:
            reasons.append((ids[0], other, "rendered_identical_figure"))
            union(ids[0], other)

    members: dict[str, list[str]] = {}
    for paper_id in parent:
        members.setdefault(find(paper_id), []).append(paper_id)
    shared = {root: sorted(items) for root, items in members.items() if len(items) > 1}
    group_reasons = {root: set() for root in shared}
    for left, right, reason in reasons:
        root = find(left)
        if root in group_reasons:
            group_reasons[root].add(reason)
    return {
        "binding": {paper_id: root for root, items in members.items() for paper_id in items},
        "shared_groups": dict(sorted(shared.items())),
        "group_reasons": {root: sorted(values) for root, values in sorted(group_reasons.items())},
        "digests_by_paper": {
            paper_id: sorted(values) for paper_id, values in digests_by_paper.items()
        },
        "assets_mapped": len(asset_shas),
    }


def load_papers(snapshot: str | Path) -> list[dict[str, Any]]:
    """Load the strict paper ledger of a frozen source snapshot."""
    path = Path(snapshot) / "eligible_papers.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot is missing eligible_papers.jsonl: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Snapshot has no eligible papers: {path}")
    return rows


def build_split_rows(
    papers: Iterable[dict[str, Any]],
    *,
    seed: str,
    ratios: Sequence[float],
    asset_shas: dict[str, str] | None = None,
    rendered_groups: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assign every paper to one split and return ledger rows sorted by paper_id.

    The bucket is computed from the figure group leader, so papers that share one
    figure - by file or by rendered image - always receive the same split while
    every other paper keeps a pure ``hash(seed, paper_id)`` assignment.
    """
    papers = list(papers)
    thresholds = split_thresholds(ratios)
    groups = figure_content_groups(papers, asset_shas, rendered_groups)
    binding = groups["binding"]
    digests_by_paper = groups["digests_by_paper"]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in papers:
        paper_id = str(record.get("paper_id") or "").strip()
        if not paper_id:
            raise ValueError("Snapshot contains a paper without paper_id")
        if paper_id in seen:
            raise ValueError(f"Snapshot repeats paper_id {paper_id!r}")
        seen.add(paper_id)
        bucket_id = binding.get(paper_id, paper_id)
        bucket = bucket_u64(bucket_id, seed)
        keys = identity_keys(record)
        for digest in digests_by_paper.get(paper_id, []):
            keys.setdefault(FIGURE_IDENTITY_NAMESPACE, []).append(digest)
        rows.append(
            {
                "paper_id": paper_id,
                "split": split_for_bucket(bucket, thresholds),
                "bucket_id": bucket_id,
                "bucket_u64": bucket,
                "figure_group": bucket_id if bucket_id != paper_id else None,
                "primary_domain": record.get("primary_domain") or "unknown",
                "domain_labels": [str(item) for item in (record.get("domain_labels") or []) if item],
                "figure_asset_tier": record.get("figure_asset_tier"),
                "figure_asset_sha256": digests_by_paper.get(paper_id, []),
                "identity_keys": {name: sorted(set(values)) for name, values in keys.items()},
            }
        )
    rows.sort(key=lambda row: row["paper_id"])
    return rows


def audit_leakage(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Detect papers that would appear in more than one split."""
    index: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for row in rows:
        for name, values in (row.get("identity_keys") or {}).items():
            for value in values:
                index.setdefault(name, {}).setdefault(value, []).append((row["paper_id"], row["split"]))

    cross_split_duplicates = 0
    cross_examples: list[dict[str, Any]] = []
    within_split_duplicates: dict[str, int] = {}
    for name in sorted(index):
        within = 0
        for value, entries in sorted(index[name].items()):
            splits = {split for _, split in entries}
            papers = sorted({paper for paper, _ in entries})
            if len(splits) > 1:
                cross_split_duplicates += 1
                if len(cross_examples) < 20:
                    cross_examples.append(
                        {"key": name, "value": value, "splits": sorted(splits), "paper_ids": papers}
                    )
            elif len(papers) > 1:
                within += 1
        within_split_duplicates[name] = within

    return {
        "ok": cross_split_duplicates == 0,
        "keys_checked": sorted(index),
        "cross_split_duplicates": cross_split_duplicates,
        "cross_split_examples": cross_examples,
        "within_split_duplicates": within_split_duplicates,
    }


def key_availability(papers: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Count how many papers expose each deduplication namespace."""
    counts: dict[str, int] = {}
    for record in papers:
        for name in identity_keys(record):
            counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def domain_distribution(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return per-domain and overall split counts plus the empty cells."""
    rows = list(rows)
    domains = sorted({str(row.get("primary_domain") or "unknown") for row in rows})
    by_domain: dict[str, Any] = {}
    for domain in domains:
        subset = [row for row in rows if str(row.get("primary_domain") or "unknown") == domain]
        counts = {split: 0 for split in SPLIT_ORDER}
        for row in subset:
            counts[row["split"]] += 1
        total = len(subset)
        by_domain[domain] = {
            "total": total,
            "counts": counts,
            "ratios": {split: (round(counts[split] / total, 6) if total else 0.0) for split in SPLIT_ORDER},
        }
    totals = {split: sum(item["counts"][split] for item in by_domain.values()) for split in SPLIT_ORDER}
    total = len(rows)
    empty_cells = [
        {"domain": domain, "split": split}
        for domain in domains
        for split in SPLIT_ORDER
        if by_domain[domain]["counts"][split] == 0
    ]
    return {
        "by_domain": by_domain,
        "totals": totals,
        "total": total,
        "ratios": {split: (round(totals[split] / total, 6) if total else 0.0) for split in SPLIT_ORDER},
        "empty_cells": empty_cells,
        "small_domains": sorted([domain for domain in domains if by_domain[domain]["total"] < 200]),
    }


def hidden_test_manifest(
    rows: Sequence[dict[str, Any]],
    *,
    split_id: str,
    thresholds: tuple[int, int],
) -> dict[str, Any]:
    """Build the sealed, content-addressed hidden test manifest."""
    hidden = sorted(str(row["paper_id"]) for row in rows if row["split"] == "hidden_test")
    by_domain: dict[str, int] = {}
    for row in rows:
        if row["split"] == "hidden_test":
            domain = str(row.get("primary_domain") or "unknown")
            by_domain[domain] = by_domain.get(domain, 0) + 1
    return {
        "split_id": split_id,
        "seal": "sealed",
        "rule": "split == hidden_test under the recorded bucket thresholds",
        "bucket_thresholds": {"train_below": thresholds[0], "validation_below": thresholds[1]},
        "papers": len(hidden),
        "by_domain": dict(sorted(by_domain.items())),
        "paper_ids_sha256": hashlib.sha256("\n".join(hidden).encode("utf-8")).hexdigest(),
        "derived_buckets": list(DERIVED_HIDDEN_TEST_BUCKETS),
        "access": {
            "reader": "final acceptance script only",
            "log": "hidden_test_access.log",
            "append_only": True,
            "forbidden_uses": list(HIDDEN_TEST_FORBIDDEN_USES),
        },
        "perceptual_hash_participation": (
            "figures are rendered and hashed for leakage detection only; "
            "content is never inspected and no tuning decision may read it"
        ),
    }


def build_split_payloads(
    snapshot: str | Path,
    *,
    seed: str = DEFAULT_SEED,
    ratios: str | Sequence[float] | None = None,
    split_id: str | None = None,
    perceptual_hash_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute every deterministic artifact of one split version.

    ``perceptual_hash_report`` is the rendered-figure fact file of the snapshot.  It
    is part of the payload rather than only of the write step, because a rendered
    figure shared by two splits is a leakage that has to be visible in the ledger
    itself before anything can be sealed.
    """
    snapshot = Path(snapshot)
    parsed = parse_ratios(ratios)
    manifest_path = snapshot / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snapshot is missing snapshot_manifest.json: {manifest_path}")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = load_papers(snapshot)
    asset_shas = load_figure_asset_shas(snapshot)
    rendered_groups = (perceptual_hash_report or {}).get("exact_groups") or []
    rows = build_split_rows(
        papers,
        seed=seed,
        ratios=parsed,
        asset_shas=asset_shas,
        rendered_groups=rendered_groups,
    )
    thresholds = split_thresholds(parsed)
    resolved_split_id = split_id or seed

    leakage = audit_leakage(rows)
    leakage["key_availability"] = key_availability(papers)
    figure_groups = figure_content_groups(papers, asset_shas, rendered_groups)
    shared_figures = figure_groups["shared_groups"]
    group_reasons = figure_groups["group_reasons"]
    leakage["figure_groups"] = {
        "figure_assets_mapped": figure_groups["assets_mapped"],
        "shared_groups": len(shared_figures),
        "binding_rule": (
            "papers sharing a figure file are bound to one group and hashed by the smallest paper_id"
        ),
        "groups": [
            {
                "leader": leader,
                "size": len(items),
                "papers": items,
                "reasons": group_reasons.get(leader, []),
            }
            for leader, items in shared_figures.items()
        ][:50],
        "oversized_groups_for_review": [
            {"leader": leader, "size": len(items), "papers": items}
            for leader, items in shared_figures.items()
            if len(items) > FIGURE_GROUP_REVIEW_LIMIT
        ],
    }
    perceptual = audit_perceptual_leakage(rows, perceptual_hash_report)
    leakage["perceptual_hash"] = perceptual
    seal_gates = {
        "identity_keys_do_not_cross_splits": bool(leakage["ok"]),
        "perceptual_hash_report_ok": bool(perceptual.get("report_ok")),
        "rendered_figures_do_not_cross_splits": perceptual.get("supplied") is True
        and not perceptual.get("exact_cross_split_groups"),
    }
    seal_ready = all(seal_gates.values())

    distribution = domain_distribution(rows)
    hidden = hidden_test_manifest(rows, split_id=resolved_split_id, thresholds=thresholds)

    validation = validate_rows(
        rows,
        seed=seed,
        ratios=parsed,
        distribution=distribution,
        leakage=leakage,
        perceptual=perceptual,
        source_complete=(snapshot / "COMPLETE").is_file(),
        source_strict_unique=source_manifest.get("strict_unique_papers"),
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split_id": resolved_split_id,
        "source_snapshot_id": source_manifest.get("snapshot_id"),
        "source_snapshot_manifest_sha256": sha256_file(manifest_path),
        "source_snapshot_paper_set_sha256": source_manifest.get("paper_set_sha256"),
        "source_snapshot_strict_unique_papers": source_manifest.get("strict_unique_papers"),
        "seed": seed,
        "ratios_target": {
            "train": parsed[0],
            "validation": parsed[1],
            "hidden_test": parsed[2],
        },
        "bucket_rule": {
            "function": "sha256(seed + '\\x1f' + paper_id)[:16] as unsigned 64-bit",
            "train_below": thresholds[0],
            "validation_below": thresholds[1],
            "hidden_test_from": thresholds[1],
        },
        "papers_total": len(rows),
        "split_counts": distribution["totals"],
        "split_ratios_actual": distribution["ratios"],
        "inheritance_rule": INHERITANCE_RULE,
        "figure_identity_rule": (
            "papers that share a figure file or render to the same perceptual hash form one group "
            "and are hashed by the smallest paper_id of that group; an incremental batch must "
            "reuse the leader recorded by the previous split version so that no already assigned "
            "paper can move"
        ),
        "figure_groups_bound": len(shared_figures),
        "figure_assets_mapped": figure_groups["assets_mapped"],
        "leakage_ok": bool(leakage["ok"]),
        "hidden_test_manifest": "hidden_test_manifest.json",
        "hidden_test_paper_ids_sha256": hidden["paper_ids_sha256"],
        "seal_gates": seal_gates,
        "seal_requirements": [
            "leakage_report.json ok",
            "perceptual_hash_report.json supplied and ok",
            "no rendered figure shared across splits",
        ],
        "complete_file": "COMPLETE",
    }

    return {
        "split_id": resolved_split_id,
        "seed": seed,
        "ratios": parsed,
        "thresholds": thresholds,
        "rows": rows,
        "manifest": manifest,
        "leakage": leakage,
        "distribution": distribution,
        "hidden_test_manifest": hidden,
        "validation": validation,
        "perceptual": perceptual,
        "seal_gates": seal_gates,
        "seal_ready": seal_ready,
    }


def validate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    seed: str,
    ratios: Sequence[float],
    distribution: dict[str, Any],
    leakage: dict[str, Any],
    perceptual: dict[str, Any],
    source_complete: bool,
    source_strict_unique: Any,
) -> dict[str, Any]:
    """Self-check a computed ledger before anything is written."""
    thresholds = split_thresholds(ratios)
    reassigned = [
        row["paper_id"]
        for row in rows
        if split_for_bucket(
            bucket_u64(str(row.get("bucket_id") or row["paper_id"]), seed), thresholds
        )
        != row["split"]
    ]
    group_splits: dict[str, set[str]] = {}
    for row in rows:
        group = str(row.get("figure_group") or "")
        if group:
            group_splits.setdefault(group, set()).add(str(row["split"]))
    split_figure_groups = sorted(key for key, values in group_splits.items() if len(values) > 1)
    totals = distribution["totals"]
    total = distribution["total"]
    actual = {split: (totals[split] / total if total else 0.0) for split in SPLIT_ORDER}
    drift = {split: round(abs(actual[split] - float(ratios[index])), 6) for index, split in enumerate(SPLIT_ORDER)}
    checks = [
        {
            "name": "source_snapshot_complete",
            "ok": bool(source_complete),
            "detail": "source snapshot carries COMPLETE",
        },
        {
            "name": "source_paper_count_matches",
            "ok": source_strict_unique in (None, len(rows)),
            "detail": f"strict_unique_papers={source_strict_unique} rows={len(rows)}",
        },
        {
            "name": "every_paper_assigned_once",
            "ok": len({str(row["paper_id"]) for row in rows}) == len(rows),
            "detail": f"{len(rows)} rows",
        },
        {
            "name": "bucket_rule_reproduced",
            "ok": not reassigned,
            "detail": f"{len(reassigned)} rows would land elsewhere",
        },
        {
            "name": "shared_figures_stay_in_one_split",
            "ok": not split_figure_groups,
            "detail": f"{len(split_figure_groups)} shared-figure groups span splits",
        },
        {
            "name": "no_cross_split_leakage",
            "ok": bool(leakage["ok"]),
            "detail": f"{leakage['cross_split_duplicates']} duplicated keys across splits",
        },
        {
            "name": "perceptual_hash_report_ok",
            "ok": bool(perceptual.get("report_ok")),
            "detail": (
                "no perceptual hash report supplied"
                if not perceptual.get("supplied")
                else f"report ok={bool(perceptual.get('report_ok'))}"
            ),
        },
        {
            "name": "no_rendered_figure_crosses_splits",
            "ok": perceptual.get("supplied") is True
            and not perceptual.get("exact_cross_split_groups"),
            "detail": f"{len(perceptual.get('exact_cross_split_groups') or [])} rendered-figure "
            "groups cross a split boundary",
        },
        {
            "name": "hidden_test_non_empty",
            "ok": totals["hidden_test"] > 0,
            "detail": f"hidden_test={totals['hidden_test']}",
        },
        {
            "name": "every_split_non_empty",
            "ok": all(totals[split] > 0 for split in SPLIT_ORDER),
            "detail": str(dict(totals)),
        },
        {
            "name": "totals_ratio_drift_within_2pct",
            "ok": all(value <= 0.02 for value in drift.values()),
            "detail": str(drift),
        },
    ]
    return {
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
        "ratio_drift": drift,
        "empty_cells": distribution["empty_cells"],
        "small_domains": distribution["small_domains"],
        "per_domain_ratio_is_reported_not_enforced": True,
        "reassigned_papers": reassigned[:20],
        "split_figure_groups": split_figure_groups[:20],
        "perceptual": perceptual,
    }


def write_split_version(
    out_dir: str | Path,
    payloads: dict[str, Any],
    *,
    perceptual_hash: dict[str, Any] | None = None,
    allow_existing: bool = False,
) -> dict[str, Any]:
    """Write one immutable split version; COMPLETE is only written when all seals pass."""
    out = Path(out_dir)
    complete_path = out / "COMPLETE"
    if complete_path.exists() and not allow_existing:
        raise FileExistsError(f"Split version is already sealed and cannot be rewritten: {out}")
    out.mkdir(parents=True, exist_ok=True)

    write_json(out / "split_manifest.json", payloads["manifest"])
    write_json(out / "leakage_report.json", payloads["leakage"])
    write_json(out / "domain_distribution.json", payloads["distribution"])
    write_json(out / "hidden_test_manifest.json", payloads["hidden_test_manifest"])
    write_json(out / "validation_report.json", payloads["validation"])

    ledger_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in payloads["rows"]
    )
    write_text(out / "split_ledger.jsonl", ledger_text)

    access_log = out / "hidden_test_access.log"
    if not access_log.exists():
        write_text(
            access_log,
            json.dumps(
                {
                    "event": "seal_created",
                    "split_id": payloads["split_id"],
                    "hidden_test_papers": payloads["distribution"]["totals"]["hidden_test"],
                    "reader": "final acceptance script only",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
        )

    sealed = False
    complete: dict[str, Any] = {}
    if perceptual_hash is not None:
        write_json(out / "perceptual_hash_report.json", perceptual_hash)
        sealed = bool(payloads.get("seal_ready"))
        if sealed:
            complete = {
                "split_id": payloads["split_id"],
                "manifest_sha256": sha256_file(out / "split_manifest.json"),
                "hidden_test_manifest_sha256": sha256_file(out / "hidden_test_manifest.json"),
                "perceptual_hash_report_sha256": sha256_file(out / "perceptual_hash_report.json"),
            }
            write_json(complete_path, complete)
    return {
        "out_dir": str(out),
        "sealed": sealed,
        "seal_gates": payloads.get("seal_gates"),
        "complete": complete,
        "files": [name for name in ARTIFACT_FILES if (out / name).is_file()],
    }


def load_split_ledger(path: str | Path) -> list[dict[str, Any]]:
    """Load a written split ledger."""
    source = Path(path)
    if source.is_dir():
        source = source / "split_ledger.jsonl"
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_split_version(version_dir: str | Path) -> dict[str, Any]:
    """Re-derive a written split version and prove it matches its own manifest."""
    out = Path(version_dir)
    manifest_path = out / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = load_split_ledger(out)
    thresholds = (
        int(manifest["bucket_rule"]["train_below"]),
        int(manifest["bucket_rule"]["validation_below"]),
    )
    seed = str(manifest["seed"])
    mismatched = [
        row["paper_id"]
        for row in rows
        if split_for_bucket(
            bucket_u64(str(row.get("bucket_id") or row["paper_id"]), seed), thresholds
        )
        != row["split"]
    ]
    counts = {split: 0 for split in SPLIT_ORDER}
    for row in rows:
        counts[row["split"]] += 1
    leakage = json.loads((out / "leakage_report.json").read_text(encoding="utf-8"))
    perceptual = leakage.get("perceptual_hash") or {}
    complete_path = out / "COMPLETE"
    complete = json.loads(complete_path.read_text(encoding="utf-8")) if complete_path.is_file() else None
    checks = [
        {"name": "rows_match_manifest_total", "ok": len(rows) == int(manifest["papers_total"])},
        {"name": "counts_match_manifest", "ok": counts == manifest["split_counts"]},
        {"name": "bucket_rule_reproduced", "ok": not mismatched},
        {"name": "leakage_ok", "ok": bool(leakage.get("ok"))},
        {
            "name": "perceptual_hash_report_ok",
            "ok": bool(perceptual.get("report_ok")),
        },
        {
            "name": "no_rendered_figure_crosses_splits",
            "ok": perceptual.get("supplied") is True
            and not perceptual.get("exact_cross_split_groups"),
        },
        {
            "name": "ready_for_training",
            "ok": bool(rows)
            and counts["hidden_test"] > 0
            and bool(leakage.get("ok"))
            and bool(perceptual.get("ok")),
        },
        {
            "name": "complete_anchors_manifest",
            "ok": complete is not None
            and complete.get("manifest_sha256") == sha256_file(manifest_path),
        },
    ]
    return {
        "ok": all(check["ok"] for check in checks),
        "sealed": complete is not None,
        "split_id": manifest.get("split_id"),
        "papers": len(rows),
        "split_counts": counts,
        "checks": checks,
    }
