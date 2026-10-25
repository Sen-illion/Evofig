"""Write a middleware version's per paper contracts where the generation line reads them.

The middleware decides what a paper's figure has to show and the generation line
draws it. The middleware keeps its half of that answer where its own stages want
it: the review in ``paper_reviews.jsonl`` at the version root, the frozen method
inventory derived from that review, the figure content plan inside each paper's
``graphs/<paper>/`` directory. The generation line reads one directory holding
``paper_review.json`` and ``paper_method_inventory.json`` beside the plan, so this
module writes exactly that directory, for one paper or for a whole version, and
leaves both sides able to point at the same files. A version that stored the two
under their earlier names is read through those names and written under the
current ones.

The two artifact families are the two halves of one answer and are not
interchangeable. The specification is the method inventory: what the paper states,
which is also what a Judge sample is built from (:mod:`.method_inventory`). The
method content is the figure content plan: which of those concepts the middleware's
own model chose to draw. A Judge is never handed the second.

The middleware's own ``scientific_ground_truth.json`` travels with the export. It is the
record of which paper the contract was built for and which nodes and edges it requires,
and the pairing builder accepts a generation run only when its copy of that file is
byte-identical to the middleware's. A run's own ground truth is a derivative of its figure
content plan, so the copy of the original is what lets both sides prove they describe one
paper instead of two look-alike contracts.

Nothing here calls a provider. A version that already wrote the specification
has it copied; a version sealed before that has it re-derived from the stored
review with the same deterministic builder the middleware used, which is what
lets the sealed pilot papers be consumed without paying for the middleware
again. Every export is checked against the hash the middleware recorded for the
same artifact, and an artifact that does not reproduce is reported as a failure
instead of being shipped.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..utils import read_json, write_json
from .frozen_document import build_frozen_document
from .method_inventory import (
    FIGURE_CONTENT_PLAN_FILE,
    LEGACY_INVENTORY_FILE,
    LEGACY_PLAN_FILE,
    METHOD_INVENTORY_FILE,
    inventory_path,
    plan_path,
)
from .pilot_graphs import canonical_sha256, derive_figure_specification, safe_paper_directory
from .pilot_selection import load_snapshot_papers
from .source_snapshot_io import sha256_file, strict_jsonl


EXPORT_CONTRACT = "rfs.middle_layer_contract_export"
EXPORT_VERSION = 1
GRAPH_DIRECTORY = "graphs"
REVIEW_FILE = "paper_review.json"
SPECIFICATION_FILE = METHOD_INVENTORY_FILE
LEGACY_SPECIFICATION_FILE = LEGACY_INVENTORY_FILE
METHOD_CONTENT_FILE = FIGURE_CONTENT_PLAN_FILE
LEGACY_METHOD_CONTENT_FILE = LEGACY_PLAN_FILE
METHOD_REPORT_FILE = "method_extraction_report.json"
GROUND_TRUTH_FILE = "scientific_ground_truth.json"
EXPORT_REPORT_FILE = "contract_export_report.json"
EXPORT_MANIFEST_FILE = "contract_export_manifest.json"
REVIEWS_FILE = "paper_reviews.jsonl"
MANIFEST_FILE = "graph_manifest.json"
PAPER_REPORT_FILE = "graph_report.json"
LEDGER_FILE = "graph_ledger.jsonl"
DEFAULT_SNAPSHOT_ROOT = Path("datasets") / "judge_source_snapshots"
CONTRACT_STEP = "contract"
SIBLING_ARTIFACTS = (METHOD_REPORT_FILE,)


class ContractExportMismatch(RuntimeError):
    """Raised when an exported artifact does not reproduce the middleware's own."""


_LEDGER_CACHE: dict[str, dict[str, dict[str, str]]] = {}
_SNAPSHOT_CACHE: dict[str, dict[str, dict[str, Any]]] = {}


def version_reviews(version_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Index one version's stored reviews by paper id."""
    path = Path(version_dir) / REVIEWS_FILE
    if not path.is_file():
        return {}
    found: dict[str, dict[str, Any]] = {}
    for row in strict_jsonl(path):
        provenance = row.get("graph_provenance") if isinstance(row.get("graph_provenance"), dict) else {}
        paper_id = str(row.get("paper_id") or provenance.get("paper_id") or "").strip()
        if paper_id and paper_id not in found:
            found[paper_id] = row
    return found


def resolve_snapshot(version_dir: str | Path, snapshot: str | Path | None = None) -> Path:
    """Return the frozen snapshot a version was built from, as its manifest names it."""
    if snapshot not in (None, ""):
        path = Path(snapshot).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"frozen snapshot directory is missing: {path}")
        return path
    version = Path(version_dir)
    manifest_path = version / MANIFEST_FILE
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    name = str(((manifest.get("inputs") or {}).get("snapshot")) or "").strip()
    candidate = DEFAULT_SNAPSHOT_ROOT / name if name else Path("")
    if name and candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"cannot find the frozen snapshot {version} was built from; pass snapshot explicitly "
        f"(its manifest names {name or 'nothing'}, looked for {candidate or 'nothing'})"
    )


def snapshot_paper_index(snapshot: str | Path) -> dict[str, dict[str, Any]]:
    """Index a frozen snapshot's paper ledger by paper id, loading each snapshot once."""
    key = str(Path(snapshot).resolve())
    if key not in _SNAPSHOT_CACHE:
        _SNAPSHOT_CACHE[key] = {
            str(item.get("paper_id") or ""): item for item in load_snapshot_papers(snapshot)
        }
    return _SNAPSHOT_CACHE[key]


def recorded_hashes(version_dir: str | Path, paper_id: str) -> dict[str, str]:
    """Return the artifact hashes the middleware recorded for this paper.

    A sealed version states what it wrote, so an export can prove it rebuilt the
    middleware's own specification and method contract instead of a look alike.
    The per paper ledger carries the contract step's artifact hashes; a version
    sealed before the specification became a per paper file also names it in the
    paper's graph report.
    """
    version = Path(version_dir)
    key = str(version.resolve())
    if key not in _LEDGER_CACHE:
        rows: dict[str, dict[str, str]] = {}
        ledger_path = version / LEDGER_FILE
        for row in strict_jsonl(ledger_path) if ledger_path.is_file() else []:
            if str(row.get("step") or "") != CONTRACT_STEP:
                continue
            identifier = str(row.get("paper_id") or "")
            hashes = row.get("artifact_sha256") if isinstance(row.get("artifact_sha256"), dict) else {}
            target = rows.setdefault(identifier, {})
            for name, value in hashes.items():
                if isinstance(value, str) and value:
                    target.setdefault(str(name), value)
        _LEDGER_CACHE[key] = rows
    recorded = dict(_LEDGER_CACHE[key].get(str(paper_id)) or {})
    report_path = version / GRAPH_DIRECTORY / safe_paper_directory(paper_id) / PAPER_REPORT_FILE
    if report_path.is_file():
        report = read_json(report_path)
        specification = str(((report.get("method_contract") or {}).get("specification_sha256")) or "")
        if specification:
            recorded.setdefault(SPECIFICATION_FILE, specification)
    return recorded


def _reproduced(recorded: dict[str, str], names: Sequence[str], value: Any, *, paper_id: str, what: str) -> dict[str, Any]:
    """Compare one exported artifact with the hash the middleware recorded for it.

    A version recorded its artifacts under the names it wrote, so both the current
    name and the earlier one are accepted; a version that recorded neither is not
    compared, because there is nothing to compare it against.
    """

    actual = canonical_sha256(value)
    expected = next((str(recorded.get(name) or "") for name in names if recorded.get(name)), "")
    if expected and expected != actual:
        raise ContractExportMismatch(
            f"the exported {what} of {paper_id} does not reproduce the middleware's own copy: "
            f"the version recorded {expected[:12]}, this export rebuilt {actual[:12]}"
        )
    return {
        "sha256": actual,
        "recorded_sha256": expected,
        "matches_middleware": actual == expected if expected else None,
    }


def _copy_figure_content_plan(paper_directory: Path, target: Path) -> list[str]:
    """Copy the middleware's figure content plan and the report beside it, byte for byte.

    The plan is stored under whichever name the version used and written under the
    current one, so a consumer of the export reads one name either way.
    """

    copied: list[str] = []
    plan = plan_path(paper_directory)
    target.mkdir(parents=True, exist_ok=True)
    if plan is not None:
        shutil.copyfile(plan, target / METHOD_CONTENT_FILE)
        copied.append(METHOD_CONTENT_FILE)
    for name in SIBLING_ARTIFACTS:
        source = paper_directory / name
        if source.is_file():
            shutil.copyfile(source, target / name)
            copied.append(name)
    return copied


def _copy_ground_truth(paper_directory: Path, target: Path) -> dict[str, Any]:
    """Copy the middleware's own scientific ground truth into the export, byte for byte.

    The run that reads this directory publishes its own plan-derived view of the paper,
    which is a different artifact by design, so the middleware's original travels here
    and the run keeps a verbatim copy of it. That copy is what the pairing builder
    compares against the version, which turns "did both sides work from one contract"
    into a byte comparison. A version that never wrote one is reported as absent rather
    than failing the export.
    """

    source = paper_directory / GROUND_TRUTH_FILE
    if not source.is_file():
        return {"present": False, "sha256": "", "source": ""}
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target / GROUND_TRUTH_FILE)
    return {"present": True, "sha256": sha256_file(target / GROUND_TRUTH_FILE), "source": str(source.resolve())}


def _stored_inventory(paper_directory: Path) -> Path:
    """Return the method inventory a version stored, under whichever name it used.

    The middleware's own name for it is authoritative when present; an earlier version
    that never stored one is answered with the path it would have used, so the caller
    derives the specification from the review instead of shipping nothing.
    """

    return inventory_path(paper_directory, plan_fallback=False) or (paper_directory / SPECIFICATION_FILE)


def export_paper_contract(
    version_dir: str | Path,
    paper_id: str,
    out_dir: str | Path,
    *,
    snapshot: str | Path | None = None,
    reviews: dict[str, dict[str, Any]] | None = None,
    papers: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Write one paper's contract into a directory the generation line can read."""
    version = Path(version_dir)
    source_directory = version / GRAPH_DIRECTORY / safe_paper_directory(paper_id)
    target = Path(out_dir)
    stored_specification = _stored_inventory(source_directory)
    stored_review = source_directory / REVIEW_FILE
    if stored_specification.is_file() and stored_review.is_file():
        review = read_json(stored_review)
        specification = read_json(stored_specification)
        specification_source = "version"
        domain_source = "version"
    else:
        review = (reviews if reviews is not None else version_reviews(version)).get(str(paper_id))
        if not isinstance(review, dict) or not review:
            raise FileNotFoundError(
                f"the version carries no stored review for {paper_id}: {version / REVIEWS_FILE}"
            )
        resolved = resolve_snapshot(version, snapshot)
        record = (papers if papers is not None else snapshot_paper_index(resolved)).get(str(paper_id))
        if not isinstance(record, dict) or not record:
            raise FileNotFoundError(f"the frozen snapshot carries no record for {paper_id}: {resolved}")
        report_path = source_directory / PAPER_REPORT_FILE
        report = read_json(report_path) if report_path.is_file() else {}
        domain = str(report.get("domain") or record.get("primary_domain") or "")
        domain_source = "graph_report" if report.get("domain") else "snapshot"
        specification = derive_figure_specification(
            build_frozen_document(resolved, record),
            {"paper_id": str(paper_id), "domain": domain},
            review,
        )
        specification_source = "derived_from_stored_review"
    recorded = recorded_hashes(version, paper_id)
    specification_check = _reproduced(
        recorded,
        (SPECIFICATION_FILE, LEGACY_SPECIFICATION_FILE),
        specification,
        paper_id=paper_id,
        what="method inventory",
    )
    write_json(target / REVIEW_FILE, review)
    write_json(target / SPECIFICATION_FILE, specification)
    copied = _copy_figure_content_plan(source_directory, target)
    ground_truth = _copy_ground_truth(source_directory, target)
    method_check: dict[str, Any] = {"sha256": "", "recorded_sha256": "", "matches_middleware": None}
    if METHOD_CONTENT_FILE in copied:
        method_check = _reproduced(
            recorded,
            (METHOD_CONTENT_FILE, LEGACY_METHOD_CONTENT_FILE),
            read_json(target / METHOD_CONTENT_FILE),
            paper_id=paper_id,
            what="figure content plan",
        )
    record_payload = {
        "contract": EXPORT_CONTRACT,
        "version": EXPORT_VERSION,
        "summary": "One middleware paper's contract, written where the generation line reads it.",
        "paper_id": str(paper_id),
        "source_version": str(version.resolve()),
        "source_directory": str(source_directory.resolve()),
        "out_dir": str(target.resolve()),
        "specification_source": specification_source,
        "domain_source": domain_source,
        "with_figure_content_plan": METHOD_CONTENT_FILE in copied,
        "with_scientific_ground_truth": bool(ground_truth["present"]),
        "scientific_ground_truth": ground_truth,
        "files": sorted([
            REVIEW_FILE,
            SPECIFICATION_FILE,
            *copied,
            *([GROUND_TRUTH_FILE] if ground_truth["present"] else []),
            EXPORT_REPORT_FILE,
        ]),
        "specification": specification_check,
        "method": method_check,
        "review_sha256": sha256_file(target / REVIEW_FILE),
    }
    if isinstance(review.get("graph_provenance"), dict):
        record_payload["paper_id"] = str(review["graph_provenance"].get("paper_id") or paper_id)
    write_json(target / EXPORT_REPORT_FILE, record_payload)
    return record_payload


def export_version_contracts(
    version_dir: str | Path,
    out_root: str | Path,
    *,
    snapshot: str | Path | None = None,
    paper_ids: Iterable[str] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Export one middleware version into an output root, one directory per paper."""
    version = Path(version_dir)
    if not version.is_dir():
        raise FileNotFoundError(f"middleware version directory does not exist: {version}")
    reviews = version_reviews(version)
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    identifiers = requested or sorted(reviews)
    if limit is not None:
        identifiers = identifiers[: max(0, int(limit))]
    if not identifiers:
        raise ValueError(f"the version carries no reviews to export: {version / REVIEWS_FILE}")
    root = Path(out_root)
    papers: dict[str, dict[str, Any]] | None = None
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for paper_id in identifiers:
        try:
            if papers is None and not (
                _stored_inventory(version / GRAPH_DIRECTORY / safe_paper_directory(paper_id)).is_file()
            ):
                papers = snapshot_paper_index(resolve_snapshot(version, snapshot))
            record = export_paper_contract(
                version,
                paper_id,
                root / safe_paper_directory(paper_id),
                snapshot=snapshot,
                reviews=reviews,
                papers=papers,
            )
        except (ContractExportMismatch, FileNotFoundError, ValueError, OSError) as error:
            failures.append({"paper_id": paper_id, "error": str(error)})
            continue
        records.append({
            "paper_id": record["paper_id"],
            "out_dir": record["out_dir"],
            "specification_source": record["specification_source"],
            "with_figure_content_plan": record["with_figure_content_plan"],
            "with_scientific_ground_truth": record["with_scientific_ground_truth"],
            "specification_matches_middleware": record["specification"]["matches_middleware"],
            "method_matches_middleware": record["method"]["matches_middleware"],
        })
    manifest = {
        "contract": EXPORT_CONTRACT,
        "version": EXPORT_VERSION,
        "summary": "The middleware's per paper contracts, written where the generation line reads them.",
        "ok": not failures,
        "source_version": str(version.resolve()),
        "out_root": str(root.resolve()),
        "papers": len(identifiers),
        "exported": len(records),
        "failed": len(failures),
        "with_figure_content_plan": sum(1 for item in records if item["with_figure_content_plan"]),
        "with_scientific_ground_truth": sum(1 for item in records if item["with_scientific_ground_truth"]),
        "inventories_derived": sum(
            1 for item in records if item["specification_source"] == "derived_from_stored_review"
        ),
        "inventories_verified": sum(
            1 for item in records if item["specification_matches_middleware"]
        ),
        "failures": failures[:20],
        "records": records,
    }
    write_json(root / EXPORT_MANIFEST_FILE, manifest)
    return manifest


__all__ = [
    "EXPORT_CONTRACT",
    "EXPORT_VERSION",
    "ContractExportMismatch",
    "export_paper_contract",
    "export_version_contracts",
    "recorded_hashes",
    "resolve_snapshot",
    "snapshot_paper_index",
    "version_reviews",
]
