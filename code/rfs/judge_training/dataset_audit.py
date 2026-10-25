"""Count what a Judge dataset holds before anything is trained on it.

A dataset directory can look complete and still be unusable in two ways a file listing
cannot show: a label that no served model ever produced -- a deterministic gate answered in
its place -- and a quota the training plan requires that the directory never reaches. This
module reads one dataset directory and reports both, next to the distributions a reviewer
needs (task type, role, split, domain, prompt version, source) and the preference-pair
families the DPO export will draw from.

Two limits are deliberate. Nothing here calls a model, so every number is a count over rows
that already exist and a report can be regenerated and compared at any time. And an
acceptance example carries no observation block by design, so the prompt version a round was
answered under is only visible on that round's advisory example; the audit therefore reads
the version per round and lets it stand for every row of that round, instead of reporting an
unanswerable question per row.

That per-round inheritance is why the flag has a third way to fire: two runs that share a
ground-truth id -- the off and on arms of one paper -- share their round keys, so one arm's
advisory answer can hand its version to the other arm's gate-short-circuited row. A row that
recorded ``gate_only`` says outright that the gate answered, so it is flagged on its own
evidence and the inheritance cannot excuse it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..contracts.evolution import ACCEPTANCE_ROLE_FAMILY, ADVISORY_ROLE_FAMILY, judge_role_family
from ..utils import write_json, write_text

AUDIT_CONTRACT = "rfs.judge_dataset_audit"
AUDIT_VERSION = 1
EXAMPLES_FILE = "judge_examples.jsonl"
PAIRS_FILE = "preference_pairs.jsonl"
DPO_FILE = "dpo.jsonl"
REVIEW_DIR = "review"
REVIEW_QUEUE = "review_queue.jsonl"
AUDIT_JSON = "audit.json"
AUDIT_MARKDOWN = "audit.md"
SAMPLE_LIMIT = 20

# The minimums the training plan states before a full run is allowed. They are literals
# because they are a plan decision rather than something to parse out of prose at runtime:
# ``docs/judge_model_training_plan.md`` asks for 10,000 distinct source papers with 2,000
# per domain across five domains.
DEFAULT_QUOTA: dict[str, int] = {"source_papers": 10000, "papers_per_domain": 2000}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL file, skipping lines that are not objects."""

    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    """Count how often each value appears, with blanks named rather than dropped."""

    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "").strip() or "unassigned"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _labels(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("labels")
    return value if isinstance(value, dict) else {}


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("judge_metadata")
    return value if isinstance(value, dict) else {}


def _observed(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("observed")
    return value if isinstance(value, dict) else {}


def _provenance(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("provenance")
    return value if isinstance(value, dict) else {}


def _round_key(row: dict[str, Any]) -> tuple[str, int]:
    """Identify the round a row was read from, so its answer can be looked up once."""

    try:
        index = int(_provenance(row).get("round") or 0)
    except (TypeError, ValueError):
        index = 0
    return str(row.get("ground_truth_id") or ""), index


def round_prompt_versions(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int], str]:
    """Return the prompt version each round was answered under, when it recorded one.

    Only the advisory example carries an observation block, so a round's version is read
    from whichever row of that round has one and applied to every row of the round.
    """

    versions: dict[tuple[str, int], str] = {}
    for row in rows:
        judge = _observed(row).get("judge")
        version = str((judge or {}).get("prompt_version") or "") if isinstance(judge, dict) else ""
        if version:
            versions.setdefault(_round_key(row), version)
    return versions


def _role_of(row: dict[str, Any]) -> str:
    return str(_metadata(row).get("judge_role") or "")


def _is_deterministic_role(role: str) -> bool:
    """Report whether a role name says a gate answered instead of a served model."""

    return "heuristic" in str(role or "").casefold()


def _answered_by_gate(row: dict[str, Any]) -> bool:
    """Report whether the row itself records that the gate answered and no model was called."""

    judge = _observed(row).get("judge")
    return bool((judge or {}).get("gate_only")) if isinstance(judge, dict) else False


def _acceptance_has_score(row: dict[str, Any]) -> bool:
    """Report whether an acceptance label carries the score its question asked for."""

    value = _labels(row).get("overall_score")
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _domain_of(row: dict[str, Any]) -> str:
    """Read the domain a row can be bucketed by, from wherever the row records it."""

    for source in (_labels(row), row.get("paper"), row.get("process_context"), row.get("ground_truth")):
        if isinstance(source, dict):
            value = source.get("domain") or source.get("subdomain")
            if str(value or "").strip():
                return str(value).strip()
    return ""


def audit_examples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise every process example of one dataset."""

    versions = round_prompt_versions(rows)
    roles = [_role_of(row) for row in rows]
    families = [judge_role_family(role) for role in roles]
    deterministic_ids: list[str] = []
    acceptance_rows = [row for row, family in zip(rows, families) if family == ACCEPTANCE_ROLE_FAMILY]
    for row, role in zip(rows, roles):
        version = versions.get(_round_key(row), "")
        if _is_deterministic_role(role) or _answered_by_gate(row) or not version:
            deterministic_ids.append(str(row.get("example_id") or ""))
    domains = [_domain_of(row) for row in rows]
    sources = [str(_provenance(row).get("source") or "") for row in rows]
    ground_truths = {str(row.get("ground_truth_id") or "") for row in rows if str(row.get("ground_truth_id") or "")}
    return {
        "total": len(rows),
        "by_task_type": _distribution(row.get("task_type") for row in rows),
        "by_judge_role": _distribution(roles),
        "by_role_family": _distribution(families),
        "by_split": _distribution(row.get("split") for row in rows),
        "by_domain": _distribution(domains),
        "by_prompt_version": _distribution(versions.get(_round_key(row), "") for row in rows),
        "by_source": _distribution(sources),
        "advisory": sum(1 for family in families if family == ADVISORY_ROLE_FAMILY),
        "acceptance": len(acceptance_rows),
        "acceptance_with_score": sum(1 for row in acceptance_rows if _acceptance_has_score(row)),
        "acceptance_without_score": sum(1 for row in acceptance_rows if not _acceptance_has_score(row)),
        "ground_truths": len(ground_truths),
        "domains_seen": len({domain for domain in domains if domain}),
        "deterministic_only": len(deterministic_ids),
        "deterministic_only_examples": deterministic_ids[:SAMPLE_LIMIT],
    }


def audit_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise the preference pairs a DPO export would draw from."""

    def family(row: dict[str, Any]) -> str:
        rejected = row.get("rejected")
        return str((rejected or {}).get("family") or "") if isinstance(rejected, dict) else ""

    def human_audited(row: dict[str, Any]) -> bool:
        basis = row.get("preference_basis")
        return bool(basis.get("human_audited")) if isinstance(basis, dict) else False

    return {
        "total": len(rows),
        "by_family": _distribution(family(row) for row in rows),
        "by_split": _distribution(row.get("split") for row in rows),
        "by_task_type": _distribution(row.get("task_type") for row in rows),
        "human_audited": sum(1 for row in rows if human_audited(row)),
    }


def audit_assets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count how many examples still point at a figure that exists on this machine."""

    missing: list[str] = []
    for row in rows:
        preview = str(row.get("preview") or "")
        if not preview or not Path(preview).is_file():
            missing.append(str(row.get("example_id") or ""))
    return {
        "with_preview": len(rows) - len(missing),
        "missing_preview": len(missing),
        "missing_preview_examples": missing[:SAMPLE_LIMIT],
    }


def audit_review_queue(dataset_dir: Path) -> dict[str, Any]:
    """Report the human review queue sitting next to a pair dataset, when one was written."""

    queue = dataset_dir / REVIEW_DIR / REVIEW_QUEUE
    rows = _jsonl(queue)
    if not queue.is_file():
        return {"present": False, "total": 0, "pending": 0, "audited": 0}
    return {
        "present": True,
        "total": len(rows),
        "pending": sum(1 for row in rows if str(row.get("status") or "") == "pending"),
        "audited": sum(1 for row in rows if bool(row.get("human_audited"))),
    }


def audit_gaps(examples: dict[str, Any], pairs: dict[str, Any], quota: dict[str, int]) -> list[dict[str, Any]]:
    """List every measurable requirement the dataset has not reached yet."""

    gaps: list[dict[str, Any]] = []

    def add(requirement: str, have: float, need: float) -> None:
        shortfall = need - have
        if shortfall > 0:
            gaps.append({"requirement": requirement, "have": have, "need": need, "shortfall": shortfall})

    add("source_papers", float(examples["ground_truths"]), float(quota["source_papers"]))
    per_domain = float(quota["papers_per_domain"])
    by_domain = examples["by_domain"]
    for domain, count in by_domain.items():
        if domain == "unassigned":
            continue
        add(f"papers_per_domain:{domain}", float(count), per_domain)
    if len(pairs.get("by_family") or {}) == 0:
        gaps.append({"requirement": "preference_pairs", "have": float(pairs["total"]), "need": 1.0, "shortfall": 1.0 - float(pairs["total"])})
    return gaps


def audit_warnings(examples: dict[str, Any], pairs: dict[str, Any], assets: dict[str, Any], review: dict[str, Any]) -> list[str]:
    """Name the things a reader must know before trusting the counts."""

    warnings: list[str] = []
    if examples["deterministic_only"]:
        warnings.append(f"{examples['deterministic_only']} of {examples['total']} examples carry a label no served prompt produced; they teach a gate's fixed answer, not a model's judgement.")
    if "unassigned" in (examples["by_domain"] or {}):
        warnings.append(f"{examples['by_domain']['unassigned']} examples record no domain, so per-domain quotas can only be reported for the rest.")
    if "unassigned" in (examples["by_split"] or {}):
        warnings.append(f"{examples['by_split']['unassigned']} examples carry no split, so no leakage boundary is enforced for them.")
    if examples["acceptance_without_score"]:
        warnings.append(f"{examples['acceptance_without_score']} acceptance examples carry no score; a scoring label without a score cannot teach a ranking.")
    if assets["missing_preview"]:
        warnings.append(f"{assets['missing_preview']} examples point at a preview this machine does not have; training has to run where those images are.")
    if pairs["total"] and not review["present"]:
        warnings.append("no human review queue was written next to the pairs.")
    return warnings


def audit_dataset(dataset: str | Path, *, out: str | Path | None = None, quota: dict[str, int] | None = None) -> dict[str, Any]:
    """Read one dataset directory and report what it holds and what it is missing."""

    dataset_dir = Path(dataset)
    examples_path = dataset_dir / EXAMPLES_FILE
    pairs_path = dataset_dir / PAIRS_FILE
    dpo_path = dataset_dir / DPO_FILE
    example_rows = _jsonl(examples_path)
    pair_rows = _jsonl(pairs_path)
    examples = audit_examples(example_rows)
    pairs = audit_pairs(pair_rows)
    assets = audit_assets(example_rows)
    review = audit_review_queue(dataset_dir)
    limits = {**DEFAULT_QUOTA, **(quota or {})}
    gaps = audit_gaps(examples, pairs, limits)
    warnings = audit_warnings(examples, pairs, assets, review)
    report = {
        "contract": AUDIT_CONTRACT,
        "version": AUDIT_VERSION,
        "dataset_dir": str(dataset_dir),
        "summary": (
            f"{examples['total']} examples from {examples['ground_truths']} ground truths, "
            f"{pairs['total']} preference pairs, {len(gaps)} unmet requirements"
        ),
        "ok": True,
        "clean": not gaps,
        "files": {
            "examples": str(examples_path) if examples_path.is_file() else "",
            "pairs": str(pairs_path) if pairs_path.is_file() else "",
            "dpo": str(dpo_path) if dpo_path.is_file() else "",
            "dpo_rows": len(_jsonl(dpo_path)),
        },
        "examples": examples,
        "pairs": pairs,
        "assets": assets,
        "review": review,
        "quota": limits,
        "gaps": gaps,
        "warnings": warnings,
    }
    target = Path(out) if out else dataset_dir
    report["report"] = str(target / AUDIT_JSON)
    write_json(target / AUDIT_JSON, report)
    write_text(target / AUDIT_MARKDOWN, render_markdown(report))
    return report


def render_markdown(report: dict[str, Any]) -> str:
    """Render one audit report for reading, in the order a reviewer asks the questions."""

    examples = report["examples"]
    pairs = report["pairs"]
    assets = report["assets"]
    lines = [
        "# Judge dataset audit",
        "",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Examples: `{examples['total']}`",
        f"- Ground truths: `{examples['ground_truths']}`",
        f"- Preference pairs: `{pairs['total']}`",
        f"- Unmet requirements: `{len(report['gaps'])}`",
        "",
        "## Examples",
        "",
    ]
    for name in ("by_task_type", "by_role_family", "by_judge_role", "by_split", "by_domain", "by_prompt_version", "by_source"):
        lines.append(f"- `{name}`: " + ", ".join(f"{key}={value}" for key, value in (examples[name] or {}).items()))
    lines.extend([
        f"- advisory examples: `{examples['advisory']}`",
        f"- acceptance examples: `{examples['acceptance']}` (with score `{examples['acceptance_with_score']}`, without `{examples['acceptance_without_score']}`)",
        f"- labels no served prompt produced: `{examples['deterministic_only']}`",
        "",
        "## Preference pairs",
        "",
        f"- by family: " + ", ".join(f"{key}={value}" for key, value in (pairs["by_family"] or {}).items()),
        f"- by split: " + ", ".join(f"{key}={value}" for key, value in (pairs["by_split"] or {}).items()),
        f"- human audited: `{pairs['human_audited']}`",
        f"- review queue: present `{report['review']['present']}`, pending `{report['review']['pending']}`",
        "",
        "## Assets",
        "",
        f"- previews present: `{assets['with_preview']}`",
        f"- previews missing on this machine: `{assets['missing_preview']}`",
        "",
        "## Unmet requirements",
        "",
    ])
    if report["gaps"]:
        lines.extend(["| requirement | have | need | shortfall |", "| --- | --- | --- | --- |"])
        for gap in report["gaps"]:
            lines.append(f"| {gap['requirement']} | {gap['have']} | {gap['need']} | {gap['shortfall']} |")
    else:
        lines.append("None.")
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {item}" for item in report["warnings"]] or ["None."])
    return "\n".join(lines) + "\n"
