"""Phase 3 Pilot graph middleware orchestration over the frozen D0 snapshot.

The middleware turns the frozen Pilot set into per-paper graph artifacts. Stage
``prepare`` rebuilds every frozen document and its review request without any
provider call, so the whole input contract can be validated before money is
spent. ``review`` asks the provider for the paper review, ``contract`` asks it for
the grounded method contract, and ``graph`` turns that contract into the
scientific graph without touching the provider. The visual stage reads the
paper's own framework figure into the visual layer. Later stages (bind, index)
build on the same ledger and are registered as they land.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import tempfile
from pathlib import Path
import time
from typing import Any, Iterable

from rfs.analysis.graph.scientific import build_scientific_graph
from rfs.analysis.paper.method_extractor import extract_method_figure_content
from rfs.analysis.paper.review import build_paper_review, detect_domain_profile, validate_review_coverage
from rfs.providers import call_vlm_json, resolve_vlm_model, vlm_credentials_available
from rfs.providers.embeddings import resolve_embedder
from rfs.providers.embeddings import EmbeddingUnavailable, TextEmbedder
from rfs.judge_training.pilot_bindings import (
    BINDING_VERSION,
    build_layer_bindings,
    graph_sha256,
    resolve_binding_embedder,
)
from rfs.judge_training.paper_provenance import PaperProvenanceLookup
from rfs.judge_training.pilot_check import (
    CHECK_PROMPT_VERSION,
    CHECK_STEP,
    check_all,
    check_coverage,
    check_paper,
    stored_check_prompt_version,
)
from rfs.judge_training.pilot_index import (
    DEFAULT_PROBE_LIMIT,
    DEFAULT_PROBE_TOP_K,
    INDEX_ARTIFACTS,
    INDEX_STEP,
    index_all,
    index_coverage,
    index_report,
    load_index_state,
    paper_index_report,
)
from rfs.judge_training.pilot_visual import (
    DEFAULT_FIGURE_DPI,
    MAX_LONG_SIDE,
    VISUAL_PROMPT_VERSION,
    build_visual_prompt,
    load_figure_assets,
    normalize_visual_payload,
    render_figure_image,
    resolve_figure_asset,
    stored_visual_prompt_version,
)
from rfs.judge_training.frozen_document import (
    FROZEN_DOCUMENT_VERSION,
    FROZEN_SPECIFICATION_VERSION,
    build_frozen_document,
    build_frozen_evidence_map,
    build_frozen_figure_specification,
    frozen_document_stats,
)
from rfs.judge_training.pilot_selection import load_snapshot_papers
from rfs.judge_training.source_snapshot_io import sha256_file, strict_jsonl, write_jsonl
from rfs.utils import sanitize_surrogates, write_json, write_text
from rfs.judge_training.method_inventory import (
    FIGURE_CONTENT_PLAN_FILE,
    LEGACY_INVENTORY_FILE,
    LEGACY_PLAN_FILE,
    METHOD_INVENTORY_FILE,
    inventory_path,
    plan_path,
)

SCHEMA_VERSION = "pilot-graph-v1"
DEFAULT_GRAPH_ID = "paper-d0-pilot-graph-v001"
DEFAULT_EVIDENCE_MAX_CHARS = 65000
PREPARE_STEP = "prepare"
REVIEW_STEP = "review"
CONTRACT_STEP = "contract"
GRAPH_STEP = "graph"
VISUAL_STEP = "visual"
BIND_STEP = "bind"
PROJECTION_PAPERS = 5732
CONTRACT_TIMEOUT_SECONDS = 240
# The material the scientific layer is compiled from (see graph_paper). The value
# travels into the graph's provenance, so a stored graph whose source is older
# than this one is rebuilt instead of being resumed as current work. The file it
# names was renamed from figure_specification.json; a graph compiled under the
# older name is rebuilt, which changes no concept because the material did not
# move with the name.
SCIENTIFIC_GRAPH_SOURCE = "paper_method_inventory"
STAGE_ORDER = (PREPARE_STEP, REVIEW_STEP, CONTRACT_STEP, GRAPH_STEP, VISUAL_STEP, CHECK_STEP, BIND_STEP, INDEX_STEP)
IMPLEMENTED_STAGES = (PREPARE_STEP, REVIEW_STEP, CONTRACT_STEP, GRAPH_STEP, VISUAL_STEP, CHECK_STEP, BIND_STEP, INDEX_STEP)
BILLABLE_STEPS = (REVIEW_STEP, CONTRACT_STEP, VISUAL_STEP, CHECK_STEP)
PAPER_ARTIFACTS_V1 = {
    CONTRACT_STEP: ("method_figure_content.json", "method_extraction_report.json"),
    GRAPH_STEP: ("scientific_graph.json", "scientific_ground_truth.json", "scientific_graph_report.json"),
    VISUAL_STEP: ("visual_graph.json", "visual_graph_report.json"),
    BIND_STEP: ("graph_bindings.json",),
    CHECK_STEP: ("figure_text_review.json",),
}
# Schema 2 adds the two per paper files the generation side reads back: the
# frozen figure specification and the review it was grounded in. Schema 3 adds
# the claim audit the bind stage writes beside its bindings, which accounts for
# every object the figure draws instead of only for the concepts it carries. A
# version states the per paper files it must ship and the schema number moves
# with the list, so a version sealed earlier keeps being checked against the
# rules it was sealed with instead of today's list.
PAPER_ARTIFACTS_V2 = {
    REVIEW_STEP: ("paper_review.json",),
    CONTRACT_STEP: PAPER_ARTIFACTS_V1[CONTRACT_STEP] + ("figure_specification.json",),
    GRAPH_STEP: PAPER_ARTIFACTS_V1[GRAPH_STEP],
    VISUAL_STEP: PAPER_ARTIFACTS_V1[VISUAL_STEP],
    BIND_STEP: PAPER_ARTIFACTS_V1[BIND_STEP],
    CHECK_STEP: PAPER_ARTIFACTS_V1[CHECK_STEP],
}
# Schema 3 is schema 2 plus the claim audit: the bind stage accounts for every
# object the figure draws, not only for the concepts it carries.
PAPER_ARTIFACTS_V3 = {
    **PAPER_ARTIFACTS_V2,
    BIND_STEP: PAPER_ARTIFACTS_V2[BIND_STEP] + ("figure_claims.json",),
}
# Schema 4 renames the two contract artifacts: the specification becomes the
# paper's method inventory (what the paper states) and the method content becomes
# the figure content plan (which of it the figure draws). The inventory stays
# required because it is a deterministic projection of the stored review and a
# resume rebuilds it for free. The plan is not required: the extraction report is
# the contract step's proof of work, and the ledger records the plan's hash when
# the step produced one, so a plan that is recorded and then lost is still an
# error while a version that never stored one seals without it.
PAPER_ARTIFACTS_V4 = {
    **PAPER_ARTIFACTS_V3,
    CONTRACT_STEP: (METHOD_INVENTORY_FILE, "method_extraction_report.json"),
}
PAPER_ARTIFACTS = PAPER_ARTIFACTS_V4
PAPER_ARTIFACT_STEPS = (REVIEW_STEP, CONTRACT_STEP, GRAPH_STEP, VISUAL_STEP, BIND_STEP, CHECK_STEP)
ARTIFACT_SCHEMA_VERSION = 4
PAPER_ARTIFACTS_BY_SCHEMA: dict[int, dict[str, tuple[str, ...]]] = {
    1: PAPER_ARTIFACTS_V1,
    2: PAPER_ARTIFACTS_V2,
    3: PAPER_ARTIFACTS_V3,
    ARTIFACT_SCHEMA_VERSION: PAPER_ARTIFACTS_V4,
}
DOMAIN_REVIEW_PROFILE = {
    "cv": "ai-ml-method",
    "rag": "ai-ml-method",
    "agent": "ai-ml-method",
    "materials": "empirical-science",
    "biology": "empirical-science",
}
UNSAFE_ID_CHARS = '<>:"/\\|?*'
DEFAULT_REVIEW_PROFILE = "general"
PILOT_SPLITS = ("train", "validation")
VOLATILE_LEDGER_FIELDS = ("elapsed_seconds", "prompt_tokens", "completion_tokens", "total_tokens")
GRAPH_VERSION_ARTIFACTS = (
    "graph_manifest.json",
    "graph_ledger.jsonl",
    "coverage_report.json",
    "cost_report.json",
    "validation_report.json",
)

LEDGER_ARTIFACT_FILES: dict[str, dict[str, str]] = {
    PREPARE_STEP: {"graph_report.json": "graph_report.json"},
    REVIEW_STEP: {"paper_review.json": "paper_review.json"},
    CONTRACT_STEP: {
        # Both halves, under the current name and the earlier one, so a ledger row
        # written before the rename still resolves to the file it recorded.
        FIGURE_CONTENT_PLAN_FILE: FIGURE_CONTENT_PLAN_FILE,
        METHOD_INVENTORY_FILE: METHOD_INVENTORY_FILE,
        LEGACY_PLAN_FILE: LEGACY_PLAN_FILE,
        LEGACY_INVENTORY_FILE: LEGACY_INVENTORY_FILE,
    },
    GRAPH_STEP: {
        "scientific_graph.json": "scientific_graph.json",
        "scientific_ground_truth.json": "scientific_ground_truth.json",
        "scientific_graph_report.json": "scientific_graph_report.json",
    },
    VISUAL_STEP: {"visual_graph.json": "visual_graph.json"},
    BIND_STEP: {
        "graph_bindings.json": "graph_bindings.json",
        "figure_claims.json": "figure_claims.json",
    },
    CHECK_STEP: {"figure_text_review.json": "figure_text_review.json"},
}
#: Where a recorded artifact name may also live. A resumed version copies the rows
#: of an earlier one, so a row that recorded the file under its earlier name is
#: looked for under the current one as well; a version written before the rename
#: still holds the earlier file, which is why the recorded name is tried first.
LEDGER_ARTIFACT_RENAMES: dict[str, str] = {
    LEGACY_PLAN_FILE: FIGURE_CONTENT_PLAN_FILE,
    LEGACY_INVENTORY_FILE: METHOD_INVENTORY_FILE,
}
REVIEW_ARTIFACT = "review"


def canonical_bytes(value: Any) -> bytes:
    """Stable bytes used for artifact hashes and cross-machine comparisons."""
    return json.dumps(sanitize_surrogates(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def safe_paper_directory(paper_id: str) -> str:
    """Return a cross-platform directory name for a paper id.

    Paper ids such as ``iclr:2026:10006567`` carry characters that Windows
    refuses in a path segment, so every reserved character becomes an
    underscore while the id itself stays readable in the ledger.
    """
    cleaned = "".join("_" if char in UNSAFE_ID_CHARS or ord(char) < 32 else char for char in str(paper_id or "").strip())
    cleaned = cleaned.rstrip(". ")
    return cleaned or "unknown-paper"


def review_profile_for(domain: str) -> str:
    """Map a Pilot domain to the review profile used for its review prompt."""
    return DOMAIN_REVIEW_PROFILE.get(str(domain or "").strip().casefold(), DEFAULT_REVIEW_PROFILE)


def parse_stages(value: str | Iterable[str] | None) -> tuple[str, ...]:
    """Parse a comma separated stage list and reject unknown stages."""
    if value is None:
        return (PREPARE_STEP,)
    items = [item.strip().casefold() for item in (str(value).split(",") if isinstance(value, str) else [str(item) for item in value])]
    stages = tuple(item for item in items if item)
    if not stages:
        return (PREPARE_STEP,)
    if "all" in stages:
        stages = tuple(IMPLEMENTED_STAGES)
    unknown = [item for item in stages if item not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"Unknown Pilot graph stages: {unknown}; known stages are {list(STAGE_ORDER)}")
    unavailable = [item for item in stages if item not in IMPLEMENTED_STAGES]
    if unavailable:
        raise ValueError(f"Pilot graph stages not implemented yet: {unavailable}; implemented stages are {list(IMPLEMENTED_STAGES)}")
    return tuple(dict.fromkeys(stages))


def _blocked_ledger_row(row: dict[str, Any], step: str, error: str, started: float) -> dict[str, Any]:
    """Ledger row of a paper that never reached the provider or its input artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_id": str(row.get("paper_id") or ""),
        "split": str(row.get("split") or ""),
        "domain": str(row.get("domain") or ""),
        "step": step,
        "status": "failed",
        "attempts": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_count": 0,
        "image_bytes": 0,
        "prompt_chars": None,
        "estimated_tokens": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "token_source": "not_called",
        "failure_categories": [],
        "error": error,
        "artifact_sha256": {},
        "warnings": [error],
    }


def _stage_ledger(rows: list[dict[str, Any]], fresh: list[dict[str, Any]], stored: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one ledger row per Pilot paper, preferring rows from this run."""
    fresh_by_paper = {str(row.get("paper_id") or ""): row for row in fresh}
    ledger: list[dict[str, Any]] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id in fresh_by_paper:
            ledger.append(fresh_by_paper[paper_id])
        elif paper_id in stored:
            ledger.append(_with_current_artifact_names(stored[paper_id]))
    return ledger


def _with_current_artifact_names(row: dict[str, Any]) -> dict[str, Any]:
    """Rename a carried ledger row's artifacts to the names this run stores them under.

    A resumed version copies rows that may have been written before an artifact was
    renamed, so such a row names the earlier file while the version being written
    stores the current one. The recorded hash travels with the name because the
    content did not change with it; a row whose content did change is still caught
    by the artifact check, under whichever of the two names it ends up carrying.
    """
    recorded = row.get("artifact_sha256")
    if not isinstance(recorded, dict) or not recorded:
        return row
    hashes = {LEDGER_ARTIFACT_RENAMES.get(name, name): value for name, value in recorded.items()}
    if hashes == recorded:
        return row
    return {**row, "artifact_sha256": hashes}


def _review_paper_id(row: dict[str, Any]) -> str:
    return str((row.get("graph_provenance") or {}).get("paper_id") or "")


def _failed_papers(ledger: list[dict[str, Any]], step: str) -> list[str]:
    return [
        str(row.get("paper_id") or "")
        for row in ledger
        if str(row.get("step") or "") == step and str(row.get("status") or "") != "ok"
    ]


def _unknown_evidence_ids(content: dict[str, Any] | None, parsed: dict[str, Any]) -> list[str]:
    """Evidence ids a method contract cites that the frozen document does not carry."""
    if not isinstance(content, dict):
        return []
    known = {
        str(item.get("id") or "")
        for item in (parsed.get("evidence") or [])
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    referenced: list[str] = []
    for section in ("inputs", "outputs", "modules", "stages", "innovations", "optional_details", "relations"):
        for item in content.get(section) or []:
            if isinstance(item, dict) and isinstance(item.get("evidence_ids"), list):
                referenced.extend(str(value) for value in item["evidence_ids"])
    return sorted({value for value in referenced if value and value not in known})


def _paper_dir(version_dir: str | Path, paper_id: str) -> Path:
    """Directory that holds one paper's graph artifacts."""
    return Path(version_dir) / "graphs" / safe_paper_directory(paper_id)


def load_pilot_ledger(pilot: str | Path) -> list[dict[str, Any]]:
    """Read the sealed Pilot ledger, refusing any split that must stay sealed.

    Phase 2 sealed the hidden test papers and phase 3 never builds them, so a
    Pilot version carries train and validation rows only. A ledger that names
    another split means the caller pointed at the wrong file (the split ledger
    instead of the Pilot ledger, for example); the run is refused here, before a
    single provider call, rather than spending calls on a paper the final
    acceptance has to meet for the first time.
    """
    path = Path(pilot) / "pilot_ledger.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Pilot version is missing pilot_ledger.jsonl: {path}")
    rows = strict_jsonl(path)
    if not rows:
        raise ValueError(f"Pilot ledger is empty: {path}")
    sealed = sorted({
        str(row.get("split") or "")
        for row in rows
        if str(row.get("split") or "") not in PILOT_SPLITS
    })
    if sealed:
        raise ValueError(
            f"Pilot ledger {path} names splits that must stay sealed: {sealed}; "
            f"a Pilot version carries {list(PILOT_SPLITS)} only"
        )
    return rows


def load_pilot_manifest(pilot: str | Path) -> dict[str, Any]:
    path = Path(pilot) / "pilot_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def prepare_paper(snapshot: str | Path, paper: dict[str, Any], row: dict[str, Any], *, evidence_max_chars: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild one frozen Pilot document and the review request it produces."""
    started = time.monotonic()
    parsed = build_frozen_document(snapshot, paper)
    stats = frozen_document_stats(parsed)
    requested_profile = review_profile_for(row.get("domain"))
    profile = detect_domain_profile(parsed, explicit=requested_profile)
    detected = str(detect_domain_profile(parsed).get("id") or "")
    warnings = list(stats.get("warnings") or [])
    context: dict[str, Any] = {}
    failure: str | None = None
    if stats["evidence_count"]:
        try:
            _review, metadata = build_paper_review(parsed, profile, mode="heuristic", evidence_max_chars=evidence_max_chars)
            context = dict(metadata.get("context") or {})
        except Exception as exc:  # pragma: no cover - defensive, reported not hidden
            failure = f"{type(exc).__name__}: {exc}"
    else:
        failure = "no evidence text was extracted from the frozen snapshot"
    if failure:
        warnings.append(failure)
    status = "failed" if failure else ("warning" if warnings else "ok")
    report = {
        "schema_version": SCHEMA_VERSION,
        "paper_id": str(row.get("paper_id") or ""),
        "split": str(row.get("split") or ""),
        "domain": str(row.get("domain") or ""),
        "step": PREPARE_STEP,
        "status": status,
        "document": stats,
        "domain_profile": {
            "requested": requested_profile,
            "detected": detected,
            "source": "pilot-domain-map",
            "agrees_with_detection": requested_profile == detected,
        },
        "review_context": {
            "prompt_chars": int(context.get("prompt_chars") or 0),
            "estimated_tokens": int(context.get("estimated_tokens") or 0),
            "evidence_max_chars": int(evidence_max_chars),
        },
        "warnings": warnings,
        "failure": failure,
    }
    ledger_row = {
        "schema_version": SCHEMA_VERSION,
        "paper_id": report["paper_id"],
        "split": report["split"],
        "domain": report["domain"],
        "step": PREPARE_STEP,
        "status": status,
        "attempts": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_count": 0,
        "image_bytes": 0,
        "prompt_chars": report["review_context"]["prompt_chars"],
        "estimated_tokens": report["review_context"]["estimated_tokens"],
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "token_source": "not_called",
        "evidence_count": stats["evidence_count"],
        "artifact_sha256": {"graph_report.json": canonical_sha256(report)},
        "warnings": warnings,
    }
    return report, ledger_row


def prepare_all(snapshot: str | Path, pilot: str | Path, *, limit: int | None = None, evidence_max_chars: int = DEFAULT_EVIDENCE_MAX_CHARS) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Rebuild every Pilot document. Returns (reports, ledger rows, coverage rows)."""
    snapshot = Path(snapshot)
    rows = load_pilot_ledger(pilot)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    papers = {str(item.get("paper_id") or ""): item for item in load_snapshot_papers(snapshot)}
    reports: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        paper = papers.get(paper_id)
        if paper is None:
            raise KeyError(f"Pilot paper is not part of the snapshot: {paper_id}")
        report, ledger_row = prepare_paper(snapshot, paper, row, evidence_max_chars=evidence_max_chars)
        reports.append(report)
        ledger.append(ledger_row)
    return reports, ledger, rows


def review_paper(snapshot: str | Path, paper: dict[str, Any], row: dict[str, Any], *, model: str | None = None, timeout_seconds: int = 300, retries: int = 1, evidence_max_chars: int = DEFAULT_EVIDENCE_MAX_CHARS, strict: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Ask the provider for one paper review and return (review row, ledger row).

    ``build_paper_review`` silently falls back to a heuristic review when the
    provider fails or credentials are missing. This stage must never report such
    a fallback as a real review, so anything whose mode is not ``vlm`` becomes a
    failed ledger row instead of a stored review.
    """
    started = time.monotonic()
    parsed = build_frozen_document(snapshot, paper)
    stats = frozen_document_stats(parsed)
    requested_profile = review_profile_for(row.get("domain"))
    profile = detect_domain_profile(parsed, explicit=requested_profile)
    detected = str(detect_domain_profile(parsed).get("id") or "")
    metadata: dict[str, Any] = {}
    review: dict[str, Any] | None = None
    coverage: dict[str, Any] = {}
    try:
        review, metadata = build_paper_review(parsed, profile, mode="vlm", model=model, timeout_seconds=timeout_seconds, retries=retries, evidence_max_chars=evidence_max_chars)
    except Exception as exc:  # pragma: no cover - defensive, reported not hidden
        metadata = {"mode": "failed", "warning": f"{type(exc).__name__}: {exc}"}
    provider: dict[str, Any] = dict(metadata.get("provider") or {})
    mode = str(metadata.get("mode") or "")
    error = None if mode == "vlm" else str(metadata.get("warning") or f"review provider returned mode {mode!r}")
    if error:
        review = None
    else:
        coverage = validate_review_coverage(review, parsed, profile, bool(strict))
    review_row = None
    if review is not None:
        review_row = {
            **review,
            "graph_provenance": {
                "paper_id": str(row.get("paper_id") or ""),
                "split": str(row.get("split") or ""),
                "domain": str(row.get("domain") or ""),
                "requested_profile": requested_profile,
                "detected_profile": detected,
                "review_mode": mode,
                "model": metadata.get("model"),
                "prompt_chars": int((metadata.get("context") or {}).get("prompt_chars") or 0),
                "evidence_max_chars": int(evidence_max_chars),
                "evidence_count": stats["evidence_count"],
                "facts_checked": coverage.get("facts_checked"),
                "grounded_facts": coverage.get("grounded_facts"),
                "grounding_rate": coverage.get("grounding_rate"),
                "coverage_warnings": [str(item) for item in (coverage.get("warnings") or [])[:20]],
                "attempts": int(provider.get("attempts") or 0),
                "token_source": provider.get("token_source"),
                "prompt_tokens": provider.get("prompt_tokens"),
                "completion_tokens": provider.get("completion_tokens"),
                "total_tokens": provider.get("total_tokens"),
            },
        }
    warnings: list[str] = []
    if error:
        warnings.append(error)
    warnings.extend(str(item) for item in (coverage.get("warnings") or [])[:5])
    ledger_row = {
        "schema_version": SCHEMA_VERSION,
        "paper_id": str(row.get("paper_id") or ""),
        "split": str(row.get("split") or ""),
        "domain": str(row.get("domain") or ""),
        "step": REVIEW_STEP,
        "status": "ok" if review_row else "failed",
        "attempts": int(provider.get("attempts") or 0),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_count": int(provider.get("image_count") or 0),
        "image_bytes": int(provider.get("image_bytes") or 0),
        "prompt_chars": provider.get("prompt_chars"),
        "estimated_tokens": provider.get("estimated_tokens"),
        "prompt_tokens": provider.get("prompt_tokens"),
        "completion_tokens": provider.get("completion_tokens"),
        "total_tokens": provider.get("total_tokens"),
        "token_source": str(provider.get("token_source") or "unavailable"),
        "failure_categories": [str(item) for item in (provider.get("failure_categories") or [])],
        "error": error,
        "evidence_count": stats["evidence_count"],
        "grounding_rate": coverage.get("grounding_rate"),
        "artifact_sha256": {
            "review": canonical_sha256(review_row),
            "paper_review.json": canonical_sha256(review_row),
        } if review_row else {},
        "warnings": warnings,
    }
    return review_row, ledger_row


def review_all(snapshot: str | Path, rows: list[dict[str, Any]], *, existing_reviews: dict[str, dict[str, Any]] | None = None, model: str | None = None, timeout_seconds: int = 300, retries: int = 1, workers: int = 4, evidence_max_chars: int = DEFAULT_EVIDENCE_MAX_CHARS, strict: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Review every Pilot paper that has no stored review yet."""
    existing = dict(existing_reviews or {})
    papers = {str(item.get("paper_id") or ""): item for item in load_snapshot_papers(snapshot)}
    pending = [row for row in rows if str(row.get("paper_id") or "") not in existing]
    if pending and not vlm_credentials_available():
        raise RuntimeError("the review stage needs API_BASE and API_KEY (or GEMINI_API_KEY) in the environment")
    results: dict[str, tuple[dict[str, Any] | None, dict[str, Any]]] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {}
            for row in pending:
                paper_id = str(row.get("paper_id") or "")
                paper = papers.get(paper_id)
                if paper is None:
                    raise KeyError(f"Pilot paper is not part of the snapshot: {paper_id}")
                futures[pool.submit(review_paper, snapshot, paper, row, model=model, timeout_seconds=timeout_seconds, retries=retries, evidence_max_chars=evidence_max_chars, strict=strict)] = paper_id
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    reviews: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id in existing:
            reviews.append(existing[paper_id])
            continue
        review_row, ledger_row = results[paper_id]
        ledger.append(ledger_row)
        if review_row is not None:
            reviews.append(review_row)
    return reviews, ledger


def derive_figure_specification(parsed: dict[str, Any], row: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one paper's frozen figure specification from its stored review.

    The specification is deterministic, so a resume that only kept the method
    contract rebuilds the exact bytes the earlier run wrote without paying for
    the review and the contract again.
    """
    provenance = review.get("graph_provenance") if isinstance(review.get("graph_provenance"), dict) else {}
    return build_frozen_figure_specification(
        parsed,
        review,
        paper_id=str(row.get("paper_id") or ""),
        domain=str(row.get("domain") or ""),
        review_mode=str(provenance.get("review_mode") or ""),
    )


def backfill_resumed_specifications(
    snapshot: str | Path,
    rows: list[dict[str, Any]],
    *,
    reviews: dict[str, dict[str, Any]] | None = None,
    specifications: dict[str, dict[str, Any]] | None = None,
    papers: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Rebuild the specification of every resumed paper that never stored one.

    A version written before the specification was an artifact resumes with the
    file missing, and a resume whose requested stages exclude the contract stage
    would never derive it again, so the version would declare the current
    artifact schema while shipping a smaller set of files than that schema
    requires. The specification is a deterministic projection of the stored
    review, so it is rebuilt from that review without asking any provider
    anything; a paper with no review is left for the run that reads it.
    """
    reviewed = set(reviews or {})
    rebuilt: dict[str, dict[str, Any]] = dict(specifications or {})
    pending = [
        row
        for row in rows
        if str(row.get("paper_id") or "") in reviewed and str(row.get("paper_id") or "") not in rebuilt
    ]
    if not pending:
        return rebuilt
    catalogue = papers if papers is not None else {
        str(item.get("paper_id") or ""): item for item in load_snapshot_papers(snapshot)
    }
    for row in pending:
        paper_id = str(row.get("paper_id") or "")
        review = (reviews or {}).get(paper_id)
        paper = catalogue.get(paper_id)
        if review is None or paper is None:
            continue
        rebuilt[paper_id] = derive_figure_specification(build_frozen_document(snapshot, paper), row, review)
    return rebuilt


def contract_paper(snapshot: str | Path, paper: dict[str, Any], row: dict[str, Any], review: dict[str, Any] | None, *, model: str | None = None, timeout_seconds: int = CONTRACT_TIMEOUT_SECONDS, retries: int = 1, strict: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Ask the provider for one paper-grounded method contract.

    The extractor silently returns a heuristic seed when the provider fails, so a
    fallback becomes a failed ledger row and is never stored as a contract.
    Returns (method content or None, ledger row, contract report, figure specification).
    """
    started = time.monotonic()
    paper_id = str(row.get("paper_id") or "")
    if review is None:
        error = "no stored review to ground the method contract"
        return None, _blocked_ledger_row(row, CONTRACT_STEP, error, started), {
            "paper_id": paper_id,
            "ok": False,
            "errors": [error],
            "missing_evidence_ids": [],
            "warnings": [error],
        }, None
    parsed = build_frozen_document(snapshot, paper)
    specification = derive_figure_specification(parsed, row, review)
    content, extraction = extract_method_figure_content(
        specification,
        paper_review=review,
        mode="vlm",
        model=model,
        timeout_seconds=timeout_seconds,
        retries=retries,
        provenance={
            "paper_id": paper_id,
            "domain": str(row.get("domain") or ""),
            "specification_source": "pilot-review-seed",
            "specification_version": FROZEN_SPECIFICATION_VERSION,
        },
    )
    provider = dict(extraction.get("provider") or {})
    mode = str(extraction.get("mode") or "")
    fallback = bool(extraction.get("fallback_used"))
    error: str | None = None
    if fallback:
        error = "method contract fell back to the review seed: " + str(extraction.get("fallback_reason"))
    elif mode != "vlm":
        error = "contract provider returned mode " + repr(mode)
    missing = [str(item) for item in (extraction.get("missing_evidence_ids") or [])]
    unknown_evidence = _unknown_evidence_ids(content, parsed) if content is not None else []
    warnings: list[str] = []
    if missing:
        warnings.append(str(len(missing)) + " required method items carry no evidence id")
    if unknown_evidence:
        warnings.append(str(len(unknown_evidence)) + " evidence ids are not part of the frozen document")
    if not extraction.get("ok"):
        warnings.append("method contract validation failed: " + str(list(extraction.get("errors") or [])[:5]))
    if warnings and strict and error is None:
        error = "; ".join(warnings)
    if error:
        content = None
    contract_report = {
        "paper_id": paper_id,
        "split": str(row.get("split") or ""),
        "domain": str(row.get("domain") or ""),
        "specification_sha256": canonical_sha256(specification),
        "seed": dict(specification.get("seed") or {}),
        "ok": bool(extraction.get("ok")) and error is None,
        "mode": mode,
        "fallback_used": fallback,
        "fallback_reason": extraction.get("fallback_reason"),
        "errors": [str(item) for item in (extraction.get("errors") or [])],
        "missing_evidence_ids": missing,
        "unknown_evidence_ids": unknown_evidence,
        "removed_relations": list(extraction.get("removed_relations") or []),
        "removed_relation_count": int(extraction.get("removed_relation_count") or 0),
        "grounding_status": extraction.get("grounding_status"),
        "node_count": int(extraction.get("node_count") or 0),
        "relation_count": int(extraction.get("relation_count") or 0),
        "context": dict(extraction.get("context") or {}),
        "provider": provider,
        "warnings": warnings,
        "error": error,
    }
    ledger_row = {
        "schema_version": SCHEMA_VERSION,
        "paper_id": paper_id,
        "split": contract_report["split"],
        "domain": contract_report["domain"],
        "step": CONTRACT_STEP,
        "status": "ok" if content is not None else "failed",
        "attempts": int(provider.get("attempts") or 0),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_count": int(provider.get("image_count") or 0),
        "image_bytes": int(provider.get("image_bytes") or 0),
        "prompt_chars": provider.get("prompt_chars"),
        "estimated_tokens": provider.get("estimated_tokens"),
        "prompt_tokens": provider.get("prompt_tokens"),
        "completion_tokens": provider.get("completion_tokens"),
        "total_tokens": provider.get("total_tokens"),
        "token_source": str(provider.get("token_source") or "unavailable"),
        "failure_categories": [str(item) for item in (provider.get("failure_categories") or [])],
        "error": error,
        "node_count": contract_report["node_count"],
        "relation_count": contract_report["relation_count"],
        "unknown_evidence_ids": unknown_evidence,
        "artifact_sha256": {
            **({FIGURE_CONTENT_PLAN_FILE: canonical_sha256(content)} if content is not None else {}),
            METHOD_INVENTORY_FILE: contract_report["specification_sha256"],
        },
        "warnings": warnings,
    }
    return content, ledger_row, contract_report, specification


def contract_all(snapshot: str | Path, rows: list[dict[str, Any]], *, reviews: dict[str, dict[str, Any]], existing_contracts: dict[str, dict[str, Any]] | None = None, existing_specifications: dict[str, dict[str, Any]] | None = None, model: str | None = None, timeout_seconds: int = CONTRACT_TIMEOUT_SECONDS, retries: int = 1, workers: int = 4, strict: bool = False) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Extract the method contract and the frozen specification of every Pilot paper that has none."""
    existing = dict(existing_contracts or {})
    papers = {str(item.get("paper_id") or ""): item for item in load_snapshot_papers(snapshot)}
    pending = [row for row in rows if str(row.get("paper_id") or "") not in existing]
    if pending and not vlm_credentials_available():
        raise RuntimeError("the contract stage needs API_BASE and API_KEY (or GEMINI_API_KEY) in the environment")
    results: dict[str, tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {}
            for row in pending:
                paper_id = str(row.get("paper_id") or "")
                paper = papers.get(paper_id)
                if paper is None:
                    raise KeyError("Pilot paper is not part of the snapshot: " + paper_id)
                futures[pool.submit(contract_paper, snapshot, paper, row, reviews.get(paper_id), model=model, timeout_seconds=timeout_seconds, retries=retries, strict=strict)] = paper_id
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    contents: dict[str, dict[str, Any]] = dict(existing)
    specifications = backfill_resumed_specifications(
        snapshot,
        rows,
        reviews=reviews,
        specifications=existing_specifications,
        papers=papers,
    )
    reports: dict[str, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id in existing:
            continue
        content, ledger_row, contract_report, specification = results[paper_id]
        ledger.append(ledger_row)
        reports[paper_id] = contract_report
        if specification is not None:
            specifications[paper_id] = specification
        if content is not None:
            contents[paper_id] = content
    return contents, ledger, reports, specifications


def graph_paper(snapshot: str | Path, paper: dict[str, Any], row: dict[str, Any], review: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Turn one stored paper review into the scientific graph. No provider call.

    The method inventory is the deterministic projection of that same review, so
    the scientific layer is compiled from what the paper review states and never
    needs the generation-side method plan.
    """
    started = time.monotonic()
    paper_id = str(row.get("paper_id") or "")
    if review is None:
        return None, _blocked_ledger_row(row, GRAPH_STEP, "no stored paper review to build the scientific graph", started)
    parsed = build_frozen_document(snapshot, paper)
    specification = derive_figure_specification(parsed, row, review)
    graph, ground_truth, graph_report = build_scientific_graph(
        specification,
        paper_id=paper_id,
        evidence_map=build_frozen_evidence_map(parsed),
        source=SCIENTIFIC_GRAPH_SOURCE,
    )
    errors = [str(item) for item in (graph_report.get("errors") or [])]
    unknown = [str(item) for item in (graph_report.get("unknown_evidence_ids") or [])]
    warnings: list[str] = []
    if unknown:
        warnings.append(str(len(unknown)) + " referenced evidence ids are not part of the frozen document")
    entry = {
        "paper_id": paper_id,
        "split": str(row.get("split") or ""),
        "domain": str(row.get("domain") or ""),
        "specification_sha256": canonical_sha256(specification),
        "scientific_graph": graph,
        "scientific_ground_truth": ground_truth,
        "scientific_graph_report": graph_report,
    }
    ledger_row = {
        "schema_version": SCHEMA_VERSION,
        "paper_id": paper_id,
        "split": entry["split"],
        "domain": entry["domain"],
        "step": GRAPH_STEP,
        "status": "ok" if not errors else "failed",
        "attempts": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_count": 0,
        "image_bytes": 0,
        "prompt_chars": None,
        "estimated_tokens": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "token_source": "not_called",
        "failure_categories": [],
        "error": "; ".join(errors) or None,
        "node_count": int(graph_report.get("node_count") or 0),
        "edge_count": int(graph_report.get("edge_count") or 0),
        "unknown_evidence_ids": unknown,
        "artifact_sha256": {
            "scientific_graph.json": canonical_sha256(graph),
            "scientific_ground_truth.json": canonical_sha256(ground_truth),
            "scientific_graph_report.json": canonical_sha256(graph_report),
        },
        "warnings": warnings,
    }
    return entry, ledger_row


def _stored_graph_is_current(entry: dict[str, Any] | None) -> bool:
    """Say whether a stored scientific graph was compiled from the current source.

    A graph built from the generation-side method plan answers with a different
    concept set than the paper method inventory does, so it stops counting as
    resumed work the moment the source changes; its provenance names the source
    it used.
    """
    graph = (entry or {}).get("scientific_graph") or {}
    provenance = graph.get("provenance") if isinstance(graph.get("provenance"), dict) else {}
    return str(provenance.get("source_contract") or "") == "rfs." + SCIENTIFIC_GRAPH_SOURCE


def graph_all(snapshot: str | Path, rows: list[dict[str, Any]], *, reviews: dict[str, dict[str, Any]], existing_graphs: dict[str, dict[str, Any]] | None = None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Build the scientific graph of every Pilot paper that has a stored review."""
    existing = dict(existing_graphs or {})
    papers = {str(item.get("paper_id") or ""): item for item in load_snapshot_papers(snapshot)}
    entries: dict[str, dict[str, Any]] = dict(existing)
    ledger: list[dict[str, Any]] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id in existing and _stored_graph_is_current(existing[paper_id]):
            continue
        paper = papers.get(paper_id)
        if paper is None:
            raise KeyError("Pilot paper is not part of the snapshot: " + paper_id)
        entry, ledger_row = graph_paper(snapshot, paper, row, reviews.get(paper_id))
        ledger.append(ledger_row)
        if entry is not None:
            entries[paper_id] = entry
    return entries, ledger


def visual_paper(snapshot: str | Path, paper: dict[str, Any], row: dict[str, Any], scientific_graph: dict[str, Any] | None, *, figure_dir: str | Path, assets: dict[str, list[dict[str, Any]]] | None = None, dpi: int = DEFAULT_FIGURE_DPI, max_long_side: int = MAX_LONG_SIDE, model: str | None = None, timeout_seconds: int = 300, retries: int = 1, strict: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """Read one paper's framework figure into a visual graph and audit the read.

    The Pilot's own stage-2 labels stay out of the prompt on purpose; they are
    only used afterwards to cross-check the reading. Returns (entry, ledger row,
    visual report).
    """
    started = time.monotonic()
    paper_id = str(row.get("paper_id") or "")
    asset = resolve_figure_asset(snapshot, row, assets)
    if asset is None:
        error = "the frozen snapshot carries no readable figure asset for this paper"
        return None, _blocked_ledger_row(row, VISUAL_STEP, error, started), {
            "paper_id": paper_id,
            "ok": False,
            "errors": [error],
            "warnings": [],
        }
    source = Path(snapshot) / asset["relative_path"]
    target = Path(figure_dir) / safe_paper_directory(paper_id) / (Path(asset["relative_path"]).stem + ".png")
    parsed = build_frozen_document(snapshot, paper)
    captions = [
        str(item.get("text") or "")
        for item in ((parsed.get("document_index") or {}).get("figures") or [])
        if isinstance(item, dict)
    ]
    captions = [value for value in captions if value]
    prompt = build_visual_prompt(captions=captions)
    provider: dict[str, Any] = {}
    payload: dict[str, Any] = {}
    image: dict[str, Any] = {}
    error: str | None = None
    try:
        image = render_figure_image(source, target, dpi=dpi, max_long_side=max_long_side)
    except Exception as exc:  # pragma: no cover - defensive, reported not hidden
        error = "figure render failed: " + type(exc).__name__ + ": " + str(exc)
    if error is None:
        try:
            raw = call_vlm_json(
                prompt,
                [target],
                model=resolve_vlm_model("RFS_PAPER_REVIEW_MODEL", explicit_model=model),
                timeout=max(10, int(timeout_seconds)),
                retries=max(0, int(retries)),
                call_metadata=provider,
            )
            payload = raw if isinstance(raw, dict) else {}
        except Exception as exc:  # pragma: no cover - defensive, reported not hidden
            error = type(exc).__name__ + ": " + str(exc)
    report: dict[str, Any] = {
        "paper_id": paper_id,
        "ok": False,
        "errors": [error] if error else [],
        "warnings": [],
    }
    graph: dict[str, Any] | None = None
    if error is None:
        graph, report = normalize_visual_payload(
            payload,
            paper_id=paper_id,
            pilot_row=row,
            scientific_graph=scientific_graph,
            image=image,
            prompt_chars=len(prompt),
        )
        report["paper_id"] = paper_id
        report["asset"] = {
            "relative_path": asset["relative_path"],
            "asset_id": asset.get("asset_id"),
            "resolved_by": asset.get("source"),
        }
        report["captions_used"] = len(captions)
        report["provider"] = dict(provider)
        if report["errors"]:
            error = "; ".join(str(item) for item in report["errors"][:3])
        elif strict and report["warnings"]:
            report["ok"] = False
            error = "; ".join(str(item) for item in report["warnings"][:3])
    if not report["ok"]:
        graph = None
    entry = None
    if graph is not None:
        entry = {
            "paper_id": paper_id,
            "split": str(row.get("split") or ""),
            "domain": str(row.get("domain") or ""),
            "visual_graph": graph,
            "visual_graph_report": report,
        }
    ledger_row = {
        "schema_version": SCHEMA_VERSION,
        "paper_id": paper_id,
        "split": str(row.get("split") or ""),
        "domain": str(row.get("domain") or ""),
        "step": VISUAL_STEP,
        "status": "ok" if entry is not None else "failed",
        "attempts": int(provider.get("attempts") or 0),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_count": int(provider.get("image_count") or 0),
        "image_bytes": int(provider.get("image_bytes") or (image or {}).get("bytes") or 0),
        "prompt_chars": provider.get("prompt_chars") or len(prompt),
        "estimated_tokens": provider.get("estimated_tokens"),
        "prompt_tokens": provider.get("prompt_tokens"),
        "completion_tokens": provider.get("completion_tokens"),
        "total_tokens": provider.get("total_tokens"),
        "token_source": "not_called" if not provider else str(provider.get("token_source") or "unavailable"),
        "failure_categories": [str(item) for item in (provider.get("failure_categories") or [])],
        "error": error,
        "panel_count": int(report.get("panel_count") or 0),
        "block_count": int(report.get("block_count") or 0),
        "edge_count": int(report.get("edge_count") or 0),
        "artifact_sha256": {"visual_graph.json": canonical_sha256(graph)} if graph is not None else {},
        "warnings": [str(item) for item in (report.get("warnings") or [])],
    }
    return entry, ledger_row, report


def visual_all(snapshot: str | Path, rows: list[dict[str, Any]], *, graphs: dict[str, dict[str, Any]], figure_dir: str | Path, existing_visuals: dict[str, dict[str, Any]] | None = None, assets: dict[str, list[dict[str, Any]]] | None = None, dpi: int = DEFAULT_FIGURE_DPI, max_long_side: int = MAX_LONG_SIDE, model: str | None = None, timeout_seconds: int = 300, retries: int = 1, workers: int = 4, strict: bool = False) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Read the framework figure of every Pilot paper that has no stored read."""
    # A stored read written by another prompt answers another question, so it is not a
    # finished result for this version; it stays on disk and is read again instead.
    existing = {
        key: value
        for key, value in dict(existing_visuals or {}).items()
        if stored_visual_prompt_version(value) == VISUAL_PROMPT_VERSION
    }
    papers = {str(item.get("paper_id") or ""): item for item in load_snapshot_papers(snapshot)}
    index = assets if assets is not None else load_figure_assets(snapshot)
    pending = [row for row in rows if str(row.get("paper_id") or "") not in existing]
    if pending and not vlm_credentials_available():
        raise RuntimeError("the visual stage needs API_BASE and API_KEY (or GEMINI_API_KEY) in the environment")
    results: dict[str, tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {}
            for row in pending:
                paper_id = str(row.get("paper_id") or "")
                paper = papers.get(paper_id)
                if paper is None:
                    raise KeyError("Pilot paper is not part of the snapshot: " + paper_id)
                futures[pool.submit(
                    visual_paper,
                    snapshot,
                    paper,
                    row,
                    graphs.get(paper_id),
                    figure_dir=figure_dir,
                    assets=index,
                    dpi=dpi,
                    max_long_side=max_long_side,
                    model=model,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    strict=strict,
                )] = paper_id
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    entries: dict[str, dict[str, Any]] = dict(existing)
    reports: dict[str, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id in existing:
            continue
        entry, ledger_row, report = results[paper_id]
        ledger.append(ledger_row)
        reports[paper_id] = report
        if entry is not None:
            entries[paper_id] = entry
    return entries, ledger, reports


def bind_paper(
    row: dict[str, Any],
    graph_entry: dict[str, Any] | None,
    visual_entry: dict[str, Any] | None,
    check_entry: dict[str, Any] | None = None,
    *,
    embedder: TextEmbedder | None = None,
    embedder_status: str = "off",
    embedder_error: str = "",
    provenance: PaperProvenanceLookup | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    """Bind one paper's two stored layers, deterministic unless a semantic layer is on.

    The stored correspondence review, when the check stage already produced one,
    supplies the text the figure carries for each visible concept; without it only
    the direct label match runs. An embedder adds the soft semantic layer to both
    passes; with it absent the stage spends nothing and decides exactly what it
    always did.
    """
    started = time.monotonic()
    paper_id = str(row.get("paper_id") or "")
    scientific = (graph_entry or {}).get("scientific_graph")
    visual = (visual_entry or {}).get("visual_graph")
    blocker = None
    if scientific is None:
        blocker = "no stored scientific graph for this paper"
    elif visual is None:
        blocker = "no stored visual graph for this paper"
    if blocker is not None:
        return None, _blocked_ledger_row(row, BIND_STEP, blocker, started), {
            "paper_id": paper_id,
            "ok": False,
            "errors": [blocker],
            "warnings": [],
        }
    review = (check_entry or {}).get("figure_text_review")
    provenance_finder = provenance.finder(paper_id) if provenance is not None else None
    try:
        bindings, report = build_layer_bindings(
            scientific,
            visual,
            review,
            embedder=embedder,
            embedder_status=embedder_status,
            embedder_error=embedder_error,
            provenance=provenance_finder,
        )
    except EmbeddingUnavailable as exc:
        # A paid embedder must never be able to fail a stage that is free without it: the
        # semantic layer is dropped, the deterministic layers decide, and the reason is
        # kept in the report instead of being raised at the caller.
        bindings, report = build_layer_bindings(
            scientific,
            visual,
            review,
            embedder=None,
            embedder_status="unavailable",
            embedder_error=str(exc),
            provenance=provenance_finder,
        )
    claims = report.get("figure_claims") if isinstance(report.get("figure_claims"), dict) else {}
    entry = {
        "paper_id": paper_id,
        "split": str(row.get("split") or ""),
        "domain": str(row.get("domain") or ""),
        "graph_bindings": bindings,
        "figure_claims": claims,
        "binding_report": report,
    } if report["ok"] else None
    embedding_calls = int(report.get("embedding_calls") or 0)
    embedding_tokens = int(report.get("embedding_tokens") or 0)
    ledger_row = {
        "schema_version": SCHEMA_VERSION,
        "paper_id": paper_id,
        "split": str(row.get("split") or ""),
        "domain": str(row.get("domain") or ""),
        "step": BIND_STEP,
        "status": "ok" if entry is not None else "failed",
        "attempts": embedding_calls,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "image_count": 0,
        "image_bytes": 0,
        "prompt_chars": None,
        "estimated_tokens": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": embedding_tokens if embedding_calls else None,
        "token_source": "provider" if embedding_calls else "not_called",
        "embedding_status": str(report.get("embedding_status") or "off"),
        "embedding_texts": int(report.get("embedding_texts") or 0),
        "embedding_calls": embedding_calls,
        "failure_categories": [],
        "error": None if entry is not None else "; ".join(str(item) for item in report["errors"][:3]),
        "node_count": int(report.get("node_count") or 0),
        "edge_count": int(report.get("edge_count") or 0),
        "bound_nodes": int(report.get("bound_nodes") or 0),
        "bound_edges": int(report.get("bound_edges") or 0),
        "diagnostic_required_node_coverage": report.get("diagnostic_required_node_coverage"),
        "diagnostic_required_edge_coverage": report.get("diagnostic_required_edge_coverage"),
        "claim_coverage": (claims.get("summary") or {}).get("claim_coverage") if entry is not None else None,
        "reversed_edges": len(report.get("reversed_edges") or []),
        "artifact_sha256": {
            "graph_bindings.json": canonical_sha256(bindings),
            "figure_claims.json": canonical_sha256(claims),
        } if entry is not None else {},
        "warnings": [],
    }
    # The claim audit is stored as its own artifact, so the report of the paper keeps
    # only its counts: whoever aggregates a run reads them there, and the objects
    # themselves stay in the one file that carries them.
    report["figure_claims"] = dict(claims.get("summary") or {})
    return entry, ledger_row, report


def _layer_sha256(graph: Any) -> str:
    """Hash one layer graph; a missing layer hashes to nothing so it never matches."""
    return graph_sha256(graph) if isinstance(graph, dict) else ""


def _stored_binding_is_current(entry: dict[str, Any] | None, *, check_available: bool, scientific_sha256: str, visual_sha256: str) -> bool:
    """Say whether a stored binding set may stand as this version's result unchanged.

    Three things make one stale: it was built by an older binding version, it was
    built against another scientific or visual graph, or it was built before its
    check stage existed while the correspondence review is now available. Either way
    the set is rebuilt, because the bind stage is free and a stale set would
    silently ship the old semantics. A graph's id is stable for its paper, so the
    comparison uses the hash of the layer the binding recorded reading: a binding
    compiled from a superseded graph answers about concepts the current graph no
    longer declares.
    """
    provenance = ((entry or {}).get("graph_bindings") or {}).get("provenance")
    if not isinstance(provenance, dict):
        return False
    if str(provenance.get("binding_version") or "") != BINDING_VERSION:
        return False
    if not scientific_sha256 or str(provenance.get("scientific_graph_sha256") or "") != scientific_sha256:
        return False
    if not visual_sha256 or str(provenance.get("visual_graph_sha256") or "") != visual_sha256:
        return False
    return bool(provenance.get("check_supplied")) or not check_available


def bind_all(
    rows: list[dict[str, Any]],
    *,
    graphs: dict[str, dict[str, Any]],
    visuals: dict[str, dict[str, Any]],
    checks: dict[str, dict[str, Any]] | None = None,
    existing_bindings: dict[str, dict[str, Any]] | None = None,
    embedder: TextEmbedder | None = None,
    embedder_status: str = "off",
    embedder_error: str = "",
    provenance: PaperProvenanceLookup | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Bind the stored layers of every Pilot paper that has no current binding set.

    A binding built by an older binder, or one built before its check stage existed, is
    rebuilt once the review is available, so a resumed version carries the same bindings
    as a fresh run. The stage calls no provider unless a semantic embedder was named, so
    redoing it costs nothing.

    The embedder is a fuse as well as a provider: the first paper that reports it
    unavailable drops it for the rest of the run, so a broken endpoint costs one failed
    request instead of one per paper, and every later paper binds deterministically.
    """
    existing = dict(existing_bindings or {})
    stored_checks = dict(checks or {})
    entries: dict[str, dict[str, Any]] = dict(existing)
    reports: dict[str, dict[str, Any]] = {}
    ledger: list[dict[str, Any]] = []
    active = embedder
    status = embedder_status
    error = embedder_error
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        scientific_sha256 = _layer_sha256((graphs.get(paper_id) or {}).get("scientific_graph"))
        visual_sha256 = _layer_sha256((visuals.get(paper_id) or {}).get("visual_graph"))
        if paper_id in existing and _stored_binding_is_current(
            existing[paper_id],
            check_available=paper_id in stored_checks,
            scientific_sha256=scientific_sha256,
            visual_sha256=visual_sha256,
        ):
            continue
        entry, ledger_row, report = bind_paper(
            row,
            graphs.get(paper_id),
            visuals.get(paper_id),
            stored_checks.get(paper_id),
            embedder=active,
            embedder_status=status,
            embedder_error=error,
            provenance=provenance,
        )
        if active is not None and str(report.get("embedding_status") or "") == "unavailable":
            active = None
            status = "unavailable"
            error = str(report.get("embedding_error") or "the semantic embedder was unavailable")
        ledger.append(ledger_row)
        reports[paper_id] = report
        if entry is not None:
            entries[paper_id] = entry
    return entries, ledger, reports


def binding_coverage(bindings: dict[str, dict[str, Any]], *, without_flow: Iterable[str] = ()) -> dict[str, Any]:
    """Summarize how well the two layers of each paper connect.

    Papers whose figure carries no connector are counted separately: a figure that
    draws no flow has no arrows to bind and few blocks to name, so leaving them in
    the headline would blame the binder for a figure that is not a framework figure.
    The author's own omissions are reported as the figure-first reconciliation and as
    selection samples: the figure is the standard, so a concept it leaves out is a
    trade-off worth learning from, never a miss.

    The headline asks a third question: of the objects a reader would call meaningful
    -- blocks, row labels and arrows, never a symbol or a decorative mark -- how many
    ended with a conclusion at all? ``claim_coverage`` uses that denominator, so an
    object the figure draws for its own notation cannot flatter the number and an
    undecided meaningful object cannot hide inside it.
    """
    skipped = {str(value) for value in without_flow}
    nodes = edges = bound_nodes = bound_edges = required_nodes = bound_required = 0
    label_bound = image_label_bound = 0
    fact_bound = fact_override = label_pass_bound = 0
    semantic_weak = structure_conflicts = needs_review = multi_bound = conflicts = 0
    label_required_bound = 0
    flow_required = flow_bound_required = 0
    statuses: dict[str, int] = {}
    embedding_statuses: dict[str, int] = {}
    embedding_fingerprints: dict[str, dict[str, Any]] = {}
    figure_drawn = figure_claimed = figure_declared = figure_unaccounted = 0
    claim_objects = claim_bound = claim_outside = claim_unresolved = 0
    claim_in_paper = claim_annotation = claim_meaningful = claim_resolved_meaningful = 0
    claim_blocks = claim_row_labels = claim_arrows = claim_needs_review = 0
    claim_note_regions = claim_annotation_arrows = claim_repeat_from_reading = claim_how_conflicts = 0
    annotation_attached = annotation_annotatable = 0
    provenance_searched = provenance_found = 0
    repeat_groups = repeat_instances = 0
    role_totals: dict[str, int] = {}
    selection_judged = selection_drawn = selection_not_drawn = 0
    selection_as_block = selection_inside = 0
    reversed_total = 0
    mismatch_total = 0
    scores: list[float] = []
    unbound_required: list[str] = []
    reversed_papers: list[str] = []
    weak_papers: list[str] = []
    for paper_id, entry in sorted(bindings.items()):
        report = entry.get("binding_report") or {}
        nodes += int(report.get("node_count") or 0)
        edges += int(report.get("edge_count") or 0)
        bound_nodes += int(report.get("bound_nodes") or 0)
        bound_edges += int(report.get("bound_edges") or 0)
        required_here = int(report.get("required_nodes") or 0)
        required_nodes += required_here
        bound_required += max(0, required_here - len(report.get("required_nodes_unbound") or []))
        label_bound += int(report.get("label_match_bound_nodes") or 0)
        image_label_bound += int(report.get("image_label_bound_nodes") or 0)
        fact_bound += int(report.get("fact_bound_nodes") or 0)
        fact_override += int(report.get("fact_override_nodes") or 0)
        label_pass_bound += int(report.get("label_pass_bound_nodes") or 0)
        semantic_weak += int(report.get("semantic_weak_nodes") or 0)
        structure_conflicts += int(report.get("structure_conflict_nodes") or 0)
        needs_review += int(report.get("needs_review_nodes") or 0)
        multi_bound += int(report.get("multi_bound_nodes") or 0)
        conflicts += int(report.get("fact_vs_label_conflicts") or 0)
        figure = report.get("figure_first") or {}
        figure_drawn += int(figure.get("drawn_blocks") or 0)
        figure_claimed += int(figure.get("claimed_blocks") or 0)
        figure_declared += int(figure.get("declared_outside_contract") or 0)
        figure_unaccounted += int(figure.get("unaccounted_blocks") or 0)
        claims = report.get("figure_claims") or {}
        claim_objects += int(claims.get("objects") or 0)
        claim_bound += int(claims.get("bound") or 0)
        claim_outside += int(claims.get("outside_paper_method") or 0)
        claim_unresolved += int(claims.get("unresolved") or 0)
        claim_blocks += int(claims.get("blocks") or 0)
        claim_row_labels += int(claims.get("row_labels") or 0)
        claim_arrows += int(claims.get("arrows") or 0)
        claim_needs_review += int(claims.get("needs_review") or 0)
        claim_in_paper += int(claims.get("in_paper_not_in_method") or 0)
        claim_annotation += int(claims.get("figure_annotation") or 0)
        claim_meaningful += int(claims.get("meaningful_objects") or 0)
        claim_resolved_meaningful += int(claims.get("resolved_meaningful") or 0)
        claim_note_regions += int(claims.get("notation_regions") or 0)
        claim_annotation_arrows += int(claims.get("annotation_arrows") or 0)
        claim_repeat_from_reading += int(claims.get("repeat_groups_from_reading") or 0)
        annotation_attached += int(claims.get("annotations_attached") or 0)
        annotation_annotatable += int(claims.get("annotations_attached") or 0) + int(claims.get("annotations_unanchored") or 0)
        provenance_searched += int(claims.get("provenance_searched") or 0)
        provenance_found += int(claims.get("provenance_found") or 0)
        repeat_groups += int(claims.get("repeat_groups") or 0)
        repeat_instances += int(claims.get("repeat_instances") or 0)
        for role, count in (claims.get("roles") or {}).items():
            role_totals[str(role)] = role_totals.get(str(role), 0) + int(count or 0)
        selection = report.get("selection_samples") or {}
        selection_judged += int(selection.get("judged") or 0)
        selection_drawn += int(selection.get("drawn") or 0)
        selection_not_drawn += int(selection.get("not_drawn") or 0)
        selection_as_block += int(selection.get("as_block") or 0)
        selection_inside += int(selection.get("inside_other_graphics") or 0)
        claim_how_conflicts += int(selection.get("reader_how_conflicts") or 0)
        embedding_status = str(report.get("embedding_status") or "off")
        embedding_statuses[embedding_status] = embedding_statuses.get(embedding_status, 0) + 1
        fingerprint = report.get("embedding_fingerprint")
        if isinstance(fingerprint, dict) and fingerprint:
            embedding_fingerprints.setdefault(canonical_sha256(fingerprint), dict(fingerprint))
        label_required_bound += round(float(report.get("diagnostic_label_match_required_coverage") or 0.0) * required_here)
        if paper_id not in skipped:
            flow_required += required_here
            flow_bound_required += max(0, required_here - len(report.get("required_nodes_unbound") or []))
        for status, count in (report.get("node_statuses") or {}).items():
            statuses[str(status)] = statuses.get(str(status), 0) + int(count or 0)
        reversed_here = list(report.get("reversed_edges") or [])
        reversed_total += len(reversed_here)
        if reversed_here:
            reversed_papers.append(paper_id)
        mismatch_total += len(report.get("relation_type_mismatches") or [])
        if report.get("required_nodes_unbound"):
            unbound_required.append(paper_id)
        if (report.get("node_statuses") or {}).get("weak"):
            weak_papers.append(paper_id)
        if report.get("mean_binding_score"):
            scores.append(float(report["mean_binding_score"]))
    return {
        "papers": len(bindings),
        "nodes": nodes,
        "edges": edges,
        "bound_nodes": bound_nodes,
        "bound_edges": bound_edges,
        "node_statuses": statuses,
        "required_nodes": required_nodes,
        "bound_required_nodes": bound_required,
        "diagnostic_required_node_coverage": 1.0 if not required_nodes else round(bound_required / required_nodes, 4),
        "label_match_bound_nodes": label_bound,
        "image_label_bound_nodes": image_label_bound,
        "fact_bound_nodes": fact_bound,
        "label_pass_bound_nodes": label_pass_bound,
        "fact_override_nodes": fact_override,
        "fact_vs_label_conflicts": conflicts,
        "semantic_weak_nodes": semantic_weak,
        "structure_conflict_nodes": structure_conflicts,
        "needs_review_nodes": needs_review,
        "multi_bound_nodes": multi_bound,
        "figure_first": {
            "drawn_blocks": figure_drawn,
            "claimed_blocks": figure_claimed,
            "declared_outside_contract": figure_declared,
            "unaccounted_blocks": figure_unaccounted,
        },
        "claim_objects": claim_objects,
        "claim_blocks": claim_blocks,
        "claim_row_labels": claim_row_labels,
        "claim_arrows": claim_arrows,
        "claim_notation_regions": claim_note_regions,
        "claim_annotation_arrows": claim_annotation_arrows,
        "claim_repeat_groups_from_reading": claim_repeat_from_reading,
        "claim_reader_how_conflicts": claim_how_conflicts,
        "claim_bound": claim_bound,
        "claim_outside_paper_method": claim_outside,
        "claim_in_paper_not_in_method": claim_in_paper,
        "claim_figure_annotation": claim_annotation,
        "claim_unresolved": claim_unresolved,
        "claim_meaningful_objects": claim_meaningful,
        "claim_excluded_from_meaning": claim_objects - claim_meaningful,
        "claim_coverage": 1.0 if not claim_meaningful else round(claim_resolved_meaningful / claim_meaningful, 4),
        "diagnostic_claim_coverage_all_objects": 1.0 if not claim_objects else round((claim_bound + claim_outside) / claim_objects, 4),
        "claim_roles": role_totals,
        "claim_repeat_groups": repeat_groups,
        "claim_repeat_instances": repeat_instances,
        "claim_annotation_coverage": 1.0 if not annotation_annotatable else round(annotation_attached / annotation_annotatable, 4),
        "claim_annotations_attached": annotation_attached,
        "claim_annotations_unanchored": annotation_annotatable - annotation_attached,
        "claim_provenance_searched": provenance_searched,
        "claim_provenance_found": provenance_found,
        "claim_provenance_coverage": round(provenance_found / provenance_searched, 4) if provenance_searched else 0.0,
        "claim_needs_review": claim_needs_review,
        "selection_samples": {
            "judged": selection_judged,
            "drawn": selection_drawn,
            "not_drawn": selection_not_drawn,
            "as_block": selection_as_block,
            "inside_other_graphics": selection_inside,
        },
        "embedding_statuses": embedding_statuses,
        "embedding_fingerprints": list(embedding_fingerprints.values()),
        "diagnostic_label_match_required_coverage": 1.0 if not required_nodes else round(label_required_bound / required_nodes, 4),
        "papers_without_flow": sorted(skipped),
        "required_nodes_with_flow": flow_required,
        "diagnostic_required_node_coverage_with_flow": 1.0 if not flow_required else round(flow_bound_required / flow_required, 4),
        "papers_with_unbound_required_nodes": unbound_required,
        "papers_with_weak_bindings": weak_papers,
        "reversed_edges": reversed_total,
        "papers_with_reversed_edges": reversed_papers,
        "relation_type_mismatches": mismatch_total,
        "mean_binding_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
    }


def load_review_resume_state(resume_dir: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load stored reviews and successful review ledger rows from a partial version."""
    directory = Path(resume_dir)
    reviews: dict[str, dict[str, Any]] = {}
    ledger: dict[str, dict[str, Any]] = {}
    reviews_path = directory / "paper_reviews.jsonl"
    if reviews_path.is_file():
        for row in strict_jsonl(reviews_path):
            paper_id = str((row.get("graph_provenance") or {}).get("paper_id") or "")
            if paper_id:
                reviews[paper_id] = row
    ledger_path = directory / "graph_ledger.jsonl"
    if ledger_path.is_file():
        for row in strict_jsonl(ledger_path):
            if str(row.get("step") or "") == REVIEW_STEP and str(row.get("status") or "") == "ok":
                ledger[str(row.get("paper_id") or "")] = row
    return reviews, ledger


def load_resume_state(resume_dir: str | Path) -> dict[str, Any]:
    """Load every artifact a partial Pilot graph version already holds."""
    directory = Path(resume_dir)
    reviews, review_ledger = load_review_resume_state(directory)
    state: dict[str, Any] = {
        "reviews": reviews,
        "review_ledger": review_ledger,
        "contracts": {},
        "contract_ledger": {},
        "contract_reports": {},
        "specifications": {},
        "graphs": {},
        "graph_ledger": {},
        "visuals": {},
        "visual_reports": {},
        "visual_ledger": {},
        "bindings": {},
        "binding_reports": {},
        "binding_ledger": {},
        "index": None,
        "index_ledger": {},
        "checks": {},
        "check_reports": {},
        "check_ledger": {},
        "stages": [],
    }
    manifest_path = directory / "graph_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state["stages"] = [str(item) for item in (manifest.get("stages_completed") or [])]
    ledger_path = directory / "graph_ledger.jsonl"
    if ledger_path.is_file():
        for row in strict_jsonl(ledger_path):
            step = str(row.get("step") or "")
            if step not in (CONTRACT_STEP, GRAPH_STEP, VISUAL_STEP, BIND_STEP, INDEX_STEP, CHECK_STEP) or str(row.get("status") or "") != "ok":
                continue
            key = {
                CONTRACT_STEP: "contract_ledger",
                GRAPH_STEP: "graph_ledger",
                VISUAL_STEP: "visual_ledger",
                BIND_STEP: "binding_ledger",
                INDEX_STEP: "index_ledger",
                CHECK_STEP: "check_ledger",
            }[step]
            state[key][str(row.get("paper_id") or "")] = row
    # A stage whose stored artifacts were written by another prompt is not this
    # version's completed work, so it drops out of the resumed stage list below.
    # The set is created before the loop: a version nothing has been written to yet
    # holds no paper directory, and the loop body would then never run.
    stale_stages: set[str] = set()
    for paper_directory in sorted((directory / "graphs").glob("*")):
        if not paper_directory.is_dir():
            continue
        report_path = paper_directory / "graph_report.json"
        paper_id = ""
        if report_path.is_file():
            paper_id = str((json.loads(report_path.read_text(encoding="utf-8")) or {}).get("paper_id") or "")
        if not paper_id:
            continue
        # An artifact only counts as resumed work when this version's ledger holds
        # an ok row for that stage: a failed stage leaves its earlier artifacts on
        # disk, and trusting them would silently carry the failure into the next
        # run instead of filling the gap.
        contract_ready = paper_id in state["contract_ledger"]
        graph_ready = paper_id in state["graph_ledger"]
        visual_ready = paper_id in state["visual_ledger"]
        binding_ready = paper_id in state["binding_ledger"]
        check_ready = paper_id in state["check_ledger"]
        content_path = plan_path(paper_directory)
        if contract_ready and content_path is not None:
            state["contracts"][paper_id] = json.loads(content_path.read_text(encoding="utf-8"))
        extraction_path = paper_directory / "method_extraction_report.json"
        if contract_ready and extraction_path.is_file():
            state["contract_reports"][paper_id] = json.loads(extraction_path.read_text(encoding="utf-8"))
        specification_path = inventory_path(paper_directory, plan_fallback=False)
        if contract_ready and specification_path is not None:
            state["specifications"][paper_id] = json.loads(specification_path.read_text(encoding="utf-8"))
        graph_path = paper_directory / "scientific_graph.json"
        if graph_ready and graph_path.is_file():
            entry: dict[str, Any] = {"paper_id": paper_id, "scientific_graph": json.loads(graph_path.read_text(encoding="utf-8"))}
            for name, key in (("scientific_ground_truth.json", "scientific_ground_truth"), ("scientific_graph_report.json", "scientific_graph_report")):
                extra_path = paper_directory / name
                if extra_path.is_file():
                    entry[key] = json.loads(extra_path.read_text(encoding="utf-8"))
            specification_sha256 = (state["contract_reports"].get(paper_id) or {}).get("specification_sha256")
            if specification_sha256:
                entry["specification_sha256"] = specification_sha256
            state["graphs"][paper_id] = entry
        visual_path = paper_directory / "visual_graph.json"
        if visual_ready and visual_path.is_file():
            visual_entry: dict[str, Any] = {"paper_id": paper_id, "visual_graph": json.loads(visual_path.read_text(encoding="utf-8"))}
            visual_report_path = paper_directory / "visual_graph_report.json"
            if visual_report_path.is_file():
                visual_report = json.loads(visual_report_path.read_text(encoding="utf-8"))
                visual_entry["visual_graph_report"] = visual_report
                state["visual_reports"][paper_id] = visual_report
            # The stored read stays available as an input, but a read of an older prompt
            # is not this version's own result: the stage drops out of the resumed list
            # below, so asking for it again reads the figure again.
            state["visuals"][paper_id] = visual_entry
            if stored_visual_prompt_version(visual_entry) != VISUAL_PROMPT_VERSION:
                stale_stages.add(VISUAL_STEP)
        bindings_path = paper_directory / "graph_bindings.json"
        if binding_ready and bindings_path.is_file():
            binding_report: dict[str, Any] = {"paper_id": paper_id, "ok": True, "errors": [], "warnings": []}
            if report_path.is_file():
                stored_report = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(stored_report.get("binding"), dict):
                    binding_report = {"paper_id": paper_id, **stored_report["binding"]}
            binding_entry: dict[str, Any] = {
                "paper_id": paper_id,
                "graph_bindings": json.loads(bindings_path.read_text(encoding="utf-8")),
                "binding_report": binding_report,
            }
            claims_path = paper_directory / "figure_claims.json"
            if claims_path.is_file():
                binding_entry["figure_claims"] = json.loads(claims_path.read_text(encoding="utf-8"))
            state["bindings"][paper_id] = binding_entry
            state["binding_reports"][paper_id] = binding_report
        review_path = paper_directory / "figure_text_review.json"
        if check_ready and review_path.is_file():
            check_report: dict[str, Any] = {"paper_id": paper_id, "ok": True, "errors": [], "warnings": []}
            if report_path.is_file():
                stored_report = json.loads(report_path.read_text(encoding="utf-8"))
                if isinstance(stored_report.get("correspondence"), dict):
                    check_report = {"paper_id": paper_id, **stored_report["correspondence"]}
            review_entry = {
                "paper_id": paper_id,
                "figure_text_review": json.loads(review_path.read_text(encoding="utf-8")),
                "check_report": check_report,
            }
            state["checks"][paper_id] = review_entry
            state["check_reports"][paper_id] = check_report
            if stored_check_prompt_version(review_entry) != CHECK_PROMPT_VERSION:
                stale_stages.add(CHECK_STEP)
    if stale_stages:
        state["stages"] = [item for item in state["stages"] if item not in stale_stages]
    state["index"] = load_index_state(directory) if state["index_ledger"] else None
    return state


def cost_report(graph_id: str, ledger: list[dict[str, Any]], *, projection_papers: int = 0) -> dict[str, Any]:
    """Summarize provider usage, tokens and elapsed time per stage."""
    steps: dict[str, dict[str, Any]] = {}
    for row in ledger:
        step = str(row.get("step") or "unknown")
        bucket = steps.setdefault(step, {"rows": 0, "by_status": {}, "attempts": 0, "elapsed_seconds": 0.0, "image_count": 0, "image_bytes": 0, "papers_with_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "token_sources": {}})
        bucket["rows"] += 1
        status = str(row.get("status") or "unknown")
        bucket["by_status"][status] = bucket["by_status"].get(status, 0) + 1
        bucket["attempts"] += int(row.get("attempts") or 0)
        bucket["elapsed_seconds"] = round(bucket["elapsed_seconds"] + float(row.get("elapsed_seconds") or 0.0), 3)
        bucket["image_count"] += int(row.get("image_count") or 0)
        bucket["image_bytes"] += int(row.get("image_bytes") or 0)
        source = str(row.get("token_source") or "unavailable")
        bucket["token_sources"][source] = bucket["token_sources"].get(source, 0) + 1
        if row.get("total_tokens") is not None:
            bucket["papers_with_tokens"] += 1
            bucket["prompt_tokens"] += int(row.get("prompt_tokens") or 0)
            bucket["completion_tokens"] += int(row.get("completion_tokens") or 0)
            bucket["total_tokens"] += int(row.get("total_tokens") or 0)
    projection: dict[str, Any] = {}
    if projection_papers:
        per_step: dict[str, dict[str, Any]] = {}
        projected_total = 0
        for step in BILLABLE_STEPS:
            bucket = steps.get(step) or {}
            if not bucket.get("papers_with_tokens"):
                continue
            per_paper = bucket["total_tokens"] / max(1, bucket["rows"])
            projected = int(round(per_paper * int(projection_papers)))
            per_step[step] = {"per_paper_tokens": round(per_paper, 1), "total_tokens": projected}
            projected_total += projected
        if per_step:
            projection = {"papers": int(projection_papers), "per_step": per_step, "total_tokens": projected_total}
            if REVIEW_STEP in per_step:
                projection["per_paper_review_tokens"] = per_step[REVIEW_STEP]["per_paper_tokens"]
                projection["review_total_tokens"] = per_step[REVIEW_STEP]["total_tokens"]
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_id": graph_id,
        "steps": steps,
        "projection": projection,
        "notes": [
            "failed attempts still consume tokens but report no usage, so every total is a lower bound",
            "elapsed_seconds and provider token counts are the only volatile ledger fields",
        ],
    }


def validate_stages(reports: list[dict[str, Any]], ledger: list[dict[str, Any]], coverage: dict[str, Any], reviews: list[dict[str, Any]], *, review_expected: bool, contracts: list[dict[str, Any]] | None = None, contract_expected: bool = False, contract_reports: list[dict[str, Any]] | None = None, graphs: list[dict[str, Any]] | None = None, graph_expected: bool = False, visuals: list[dict[str, Any]] | None = None, visual_expected: bool = False, visual_reports: list[dict[str, Any]] | None = None, bindings: list[dict[str, Any]] | None = None, bind_expected: bool = False, binding_reports: list[dict[str, Any]] | None = None, index_payload: dict[str, Any] | None = None, index_expected: bool = False, check_entries: list[dict[str, Any]] | None = None, check_expected: bool = False, check_reports: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Check the document stage and every provider or graph stage that ran."""
    validation = validate_prepare(reports, ledger, coverage)
    failures = list(validation["failures"])
    human_review = list(validation.get("human_review_required") or [])
    expected = [str(report.get("paper_id") or "") for report in reports]
    checks = validation["checks"]
    checks["reviews"] = 0
    checks["contracts"] = 0
    checks["graphs"] = 0
    checks["visuals"] = 0
    checks["bindings"] = 0
    checks["index_records"] = 0
    checks["check_reviews"] = 0
    if review_expected:
        review_papers = {_review_paper_id(row) for row in reviews}
        missing = [paper_id for paper_id in expected if paper_id not in review_papers]
        failed = _failed_papers(ledger, REVIEW_STEP)
        if missing:
            failures.append(f"reviews are missing for {len(missing)} papers")
        if failed:
            failures.append(f"the review stage failed for {len(failed)} papers")
        for row in reviews:
            provenance = row.get("graph_provenance") or {}
            rate = provenance.get("grounding_rate")
            if rate is not None and float(rate) < 0.5:
                human_review.append({"paper_id": str(provenance.get("paper_id") or ""), "reason": f"review grounding rate {rate}"})
        checks["reviews"] = len(reviews)
        checks["missing_reviews"] = missing
        checks["failed_reviews"] = failed
    if contract_expected:
        contract_papers = {str(entry.get("paper_id") or "") for entry in contracts or []}
        missing = [paper_id for paper_id in expected if paper_id not in contract_papers]
        failed = _failed_papers(ledger, CONTRACT_STEP)
        if missing:
            failures.append(f"method contracts are missing for {len(missing)} papers")
        if failed:
            failures.append(f"the contract stage failed for {len(failed)} papers")
        for report in contract_reports or []:
            missing_evidence = list(report.get("missing_evidence_ids") or [])
            if missing_evidence:
                human_review.append({
                    "paper_id": str(report.get("paper_id") or ""),
                    "reason": f"{len(missing_evidence)} required method items carry no evidence id",
                })
        checks["contracts"] = len(contracts or [])
        checks["missing_contracts"] = missing
        checks["failed_contracts"] = failed
    if graph_expected:
        graph_papers = {str(entry.get("paper_id") or "") for entry in graphs or []}
        missing = [paper_id for paper_id in expected if paper_id not in graph_papers]
        failed = _failed_papers(ledger, GRAPH_STEP)
        if missing:
            failures.append(f"scientific graphs are missing for {len(missing)} papers")
        if failed:
            failures.append(f"the graph stage failed for {len(failed)} papers")
        for entry in graphs or []:
            unknown = list((entry.get("scientific_graph_report") or {}).get("unknown_evidence_ids") or [])
            if unknown:
                human_review.append({
                    "paper_id": str(entry.get("paper_id") or ""),
                    "reason": f"{len(unknown)} evidence ids are not part of the frozen document",
                })
        checks["graphs"] = len(graphs or [])
        checks["missing_graphs"] = missing
        checks["failed_graphs"] = failed
    if visual_expected:
        visual_papers = {str(entry.get("paper_id") or "") for entry in visuals or []}
        missing = [paper_id for paper_id in expected if paper_id not in visual_papers]
        failed = _failed_papers(ledger, VISUAL_STEP)
        if missing:
            failures.append(f"visual graphs are missing for {len(missing)} papers")
        if failed:
            failures.append(f"the visual stage failed for {len(failed)} papers")
        for report in visual_reports or []:
            if report.get("errors"):
                human_review.append({
                    "paper_id": str(report.get("paper_id") or ""),
                    "reason": "the visual graph could not be built: " + str((report.get("errors") or [""])[0])[:120],
                })
            if report.get("blocks_outside_panel"):
                human_review.append({
                    "paper_id": str(report.get("paper_id") or ""),
                    "reason": f"{len(report['blocks_outside_panel'])} blocks sit outside their panel",
                })
            if report.get("connectors_not_marked_feedback"):
                human_review.append({
                    "paper_id": str(report.get("paper_id") or ""),
                    "reason": f"{len(report['connectors_not_marked_feedback'])} connectors point backwards without being marked as feedback",
                })
        checks["visuals"] = len(visuals or [])
        checks["missing_visuals"] = missing
        checks["failed_visuals"] = failed
    if bind_expected:
        binding_papers = {str(entry.get("paper_id") or "") for entry in bindings or []}
        missing = [paper_id for paper_id in expected if paper_id not in binding_papers]
        failed = _failed_papers(ledger, BIND_STEP)
        if missing:
            failures.append(f"layer bindings are missing for {len(missing)} papers")
        if failed:
            failures.append(f"the bind stage failed for {len(failed)} papers")
        for report in binding_reports or []:
            paper_id = str(report.get("paper_id") or "")
            if report.get("reversed_edges"):
                human_review.append({
                    "paper_id": paper_id,
                    "reason": (
                        f"{len(report['reversed_edges'])} drawn arrows point the other way from the relation's "
                        "wording; kept for reading review, not counted against the figure"
                    ),
                })
            if report.get("relation_type_mismatches"):
                human_review.append({
                    "paper_id": paper_id,
                    "reason": (
                        f"{len(report['relation_type_mismatches'])} drawn arrows differ on feedback versus forward; "
                        "kept for reading review"
                    ),
                })
            required_here = int(report.get("required_nodes") or 0)
            if required_here and float(report.get("diagnostic_required_node_coverage") or 0.0) < 0.5:
                human_review.append({
                    "paper_id": paper_id,
                    "reason": (
                        "the figure draws fewer than half of the concepts the required list names "
                        f"(diagnostic coverage {report.get('diagnostic_required_node_coverage')}); this is the "
                        "required list asking for more than the figure needs to show, so look at the list, not at "
                        "the figure"
                    ),
                })
        checks["bindings"] = len(bindings or [])
        checks["missing_bindings"] = missing
        checks["failed_bindings"] = failed
    if index_expected:
        index_records = list((index_payload or {}).get("records") or [])
        index_audit = (index_payload or {}).get("report") or {}
        guards = dict(index_audit.get("probe_guards") or {})
        indexed_papers = {str(item.get("paper_id") or "") for item in index_records}
        missing = [paper_id for paper_id in expected if paper_id not in indexed_papers]
        failed = _failed_papers(ledger, INDEX_STEP)
        if not index_records:
            failures.append("the index holds no retrieval record")
        if missing:
            failures.append(f"index records are missing for {len(missing)} papers")
        if failed:
            failures.append(f"the index stage failed for {len(failed)} papers")
        for message in list(index_audit.get("errors") or []):
            failures.append(str(message))
        if int(guards.get("hidden_test_records") or 0):
            failures.append("the index contains hidden test records")
        if int(guards.get("self_hits") or 0):
            failures.append("the retrieval probe returns the query paper itself")
        unexpected = list(guards.get("unexpected_splits") or [])
        if unexpected:
            failures.append(f"the index holds papers outside the Pilot splits: {unexpected}")
        for paper_id in list(guards.get("records_without_motifs") or []):
            human_review.append({
                "paper_id": str(paper_id),
                "reason": "the stored graph holds no structural motif, so retrieval cannot match it",
            })
        for paper_id in list((coverage.get("index") or {}).get("papers_missing_a_layer") or []):
            human_review.append({
                "paper_id": str(paper_id),
                "reason": "one of the two graph layers carries no index record",
            })
        checks["index_records"] = len(index_records)
        checks["missing_index_records"] = missing
        checks["failed_index"] = failed
        checks["index_motifs"] = int(index_audit.get("motifs") or 0)
        checks["index_subgraph_locations"] = int(index_audit.get("subgraph_locations") or 0)
        checks["index_probe_queries"] = int(index_audit.get("probe_queries") or 0)
        checks["index_probe_guards"] = guards
    if check_expected:
        checked_papers = {str(entry.get("paper_id") or "") for entry in check_entries or []}
        missing = [paper_id for paper_id in expected if paper_id not in checked_papers]
        failed = _failed_papers(ledger, CHECK_STEP)
        if missing:
            failures.append(f"figure correspondence reviews are missing for {len(missing)} papers")
        if failed:
            failures.append(f"the check stage failed for {len(failed)} papers")
        for report in check_reports or []:
            paper_id = str(report.get("paper_id") or "")
            summary = dict(report.get("summary") or {})
            if int(summary.get("hallucinated_block") or 0):
                human_review.append({
                    "paper_id": paper_id,
                    "reason": (
                        f"{int(summary.get('hallucinated_block') or 0)} blocks are claimed by the stored visual "
                        "layer while the reading says the concept is not drawn - the two readings disagree"
                    ),
                })
            if int(summary.get("required_missed_by_reader") or 0):
                human_review.append({
                    "paper_id": paper_id,
                    "reason": (
                        f"{int(summary.get('required_missed_by_reader') or 0)} required concepts are drawn in the "
                        "figure but the stored visual layer did not read them"
                    ),
                })
            if int(summary.get("required_not_drawn") or 0):
                human_review.append({
                    "paper_id": paper_id,
                    "reason": (
                        f"{int(summary.get('required_not_drawn') or 0)} required concepts were not drawn in the "
                        "figure; kept as selection samples, not counted against the figure"
                    ),
                })
            if int(summary.get("direction_conflicts") or 0):
                human_review.append({
                    "paper_id": paper_id,
                    "reason": (
                        f"{int(summary.get('direction_conflicts') or 0)} drawn arrows point the other way from the "
                        "wording of the relation; kept for reading review, not counted against the figure"
                    ),
                })
            if int(summary.get("unreadable") or 0):
                human_review.append({
                    "paper_id": paper_id,
                    "reason": f"{int(summary.get('unreadable') or 0)} concepts could not be judged from the figure",
                })
            if report.get("findings_outside_contract"):
                human_review.append({
                    "paper_id": paper_id,
                    "reason": f"{len(report['findings_outside_contract'])} answers name a concept outside the contract",
                })
        checks["check_reviews"] = len(check_entries or [])
        checks["missing_checks"] = missing
        checks["failed_checks"] = failed
        checks["mean_required_visible_rate"] = (coverage.get("correspondence") or {}).get("mean_required_visible_rate")
    validation["ok"] = not failures
    validation["failures"] = failures
    validation["human_review_required"] = human_review
    return validation


def scientific_coverage(graphs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize the scientific graphs a run produced."""
    nodes = edges = required_nodes = required_edges = 0
    without_evidence: list[str] = []
    ungrounded: list[str] = []
    for paper_id, entry in sorted(graphs.items()):
        graph = entry.get("scientific_graph") or {}
        report = entry.get("scientific_graph_report") or {}
        nodes += int(report.get("node_count") or 0)
        edges += int(report.get("edge_count") or 0)
        required_nodes += len(graph.get("required_node_ids") or [])
        required_edges += len(graph.get("required_edge_ids") or [])
        if report.get("unknown_evidence_ids"):
            ungrounded.append(paper_id)
        if any(not node.get("evidence_ids") for node in (graph.get("nodes") or [])):
            without_evidence.append(paper_id)
    return {
        "papers": len(graphs),
        "nodes": nodes,
        "edges": edges,
        "required_nodes": required_nodes,
        "required_edges": required_edges,
        "papers_with_ungrounded_evidence": ungrounded,
        "papers_with_nodes_without_evidence": without_evidence,
    }


def visual_coverage(visuals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize the visual graphs a run produced, including the label agreement."""
    panels = blocks = edges = regions = 0
    structures: dict[str, int] = {}
    motifs: dict[str, int] = {}
    directions: dict[str, int] = {}
    agreement = {
        "papers": 0,
        "panel_count_agrees": 0,
        "node_count_agrees": 0,
        "structure_agrees": 0,
        "structure_in_signals": 0,
        "arrows_agree": 0,
    }
    warned: list[str] = []
    uncertain: list[str] = []
    unreadable_text: list[str] = []
    without_flow: list[str] = []
    for paper_id, entry in sorted(visuals.items()):
        report = entry.get("visual_graph_report") or {}
        panels += int(report.get("panel_count") or 0)
        blocks += int(report.get("block_count") or 0)
        edges += int(report.get("edge_count") or 0)
        regions += int(report.get("region_count") or 0)
        structure = str(report.get("primary_structure") or "")
        structures[structure] = structures.get(structure, 0) + 1
        direction = str(report.get("reading_direction") or "")
        directions[direction] = directions.get(direction, 0) + 1
        for motif, count in (report.get("motif_histogram") or {}).items():
            motifs[str(motif)] = motifs.get(str(motif), 0) + int(count or 0)
        if report.get("warnings"):
            warned.append(paper_id)
        if report.get("uncertainties"):
            uncertain.append(paper_id)
        if report.get("removed_connectors"):
            unreadable_text.append(paper_id)
        stored_flow = report.get("expresses_flow")
        # A version written before this field existed still carries its edge count.
        if not (bool(stored_flow) if isinstance(stored_flow, bool) else int(report.get("edge_count") or 0) > 0):
            without_flow.append(paper_id)
        label = report.get("pilot_label_agreement") or {}
        if not label:
            continue
        agreement["papers"] += 1
        if (label.get("panel_count") or {}).get("agrees"):
            agreement["panel_count_agrees"] += 1
        if (label.get("node_count") or {}).get("agrees"):
            agreement["node_count_agrees"] += 1
        if (label.get("has_arrows") or {}).get("agrees"):
            agreement["arrows_agree"] += 1
        structure_row = label.get("structure") or {}
        if structure_row.get("agrees"):
            agreement["structure_agrees"] += 1
        if structure_row.get("agrees_with_signals"):
            agreement["structure_in_signals"] += 1
    return {
        "papers": len(visuals),
        "panels": panels,
        "blocks": blocks,
        "edges": edges,
        "regions": regions,
        "structures": structures,
        "motif_histogram": motifs,
        "reading_directions": directions,
        "papers_without_flow": without_flow,
        "papers_with_warnings": warned,
        "papers_with_uncertainties": uncertain,
        "papers_with_dropped_connectors": unreadable_text,
        "pilot_label_agreement": agreement,
    }


def merge_paper_report(report: dict[str, Any], review: dict[str, Any] | None, contract_report: dict[str, Any] | None, graph_entry: dict[str, Any] | None, stage_status: dict[str, str], visual_entry: dict[str, Any] | None = None, visual_report: dict[str, Any] | None = None, binding_entry: dict[str, Any] | None = None, binding_report: dict[str, Any] | None = None, index_report: dict[str, Any] | None = None, check_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Add every later stage's audit result to one paper's prepare report."""
    merged = dict(report)
    merged["stages"] = dict(stage_status)
    provenance = (review or {}).get("graph_provenance") or {}
    merged["review"] = {
        "review_mode": provenance.get("review_mode"),
        "model": provenance.get("model"),
        "grounding_rate": provenance.get("grounding_rate"),
        "facts_checked": provenance.get("facts_checked"),
        "grounded_facts": provenance.get("grounded_facts"),
        "coverage_warnings": list(provenance.get("coverage_warnings") or []),
    } if review else None
    merged["method_contract"] = {
        "ok": bool((contract_report or {}).get("ok")),
        "mode": (contract_report or {}).get("mode"),
        "fallback_used": bool((contract_report or {}).get("fallback_used")),
        "grounding_status": (contract_report or {}).get("grounding_status"),
        "node_count": (contract_report or {}).get("node_count"),
        "relation_count": (contract_report or {}).get("relation_count"),
        "missing_evidence_ids": list((contract_report or {}).get("missing_evidence_ids") or []),
        "unknown_evidence_ids": list((contract_report or {}).get("unknown_evidence_ids") or []),
        "removed_relation_count": (contract_report or {}).get("removed_relation_count"),
        "errors": list((contract_report or {}).get("errors") or []),
        "context": dict((contract_report or {}).get("context") or {}),
        "seed": dict((contract_report or {}).get("seed") or {}),
        "specification_sha256": (contract_report or {}).get("specification_sha256"),
    } if contract_report else None
    graph = (graph_entry or {}).get("scientific_graph") or {}
    graph_report = (graph_entry or {}).get("scientific_graph_report") or {}
    merged["scientific"] = {
        "ok": bool(graph_report.get("ok")),
        "node_count": graph_report.get("node_count"),
        "edge_count": graph_report.get("edge_count"),
        "required_nodes": len(graph.get("required_node_ids") or []),
        "required_edges": len(graph.get("required_edge_ids") or []),
        "unknown_evidence_ids": list(graph_report.get("unknown_evidence_ids") or []),
        "errors": list(graph_report.get("errors") or []),
        "specification_sha256": (graph_entry or {}).get("specification_sha256"),
    } if graph_entry else None
    visual = (visual_entry or {}).get("visual_graph") or {}
    visual_audit = visual_report or (visual_entry or {}).get("visual_graph_report") or {}
    merged["visual"] = {
        "ok": bool(visual_audit.get("ok")),
        "panel_count": visual_audit.get("panel_count"),
        "block_count": visual_audit.get("block_count"),
        "edge_count": visual_audit.get("edge_count"),
        "primary_structure": visual_audit.get("primary_structure"),
        "reading_direction": visual_audit.get("reading_direction"),
        "motif_histogram": dict(visual_audit.get("motif_histogram") or {}),
        "cross_panel_edges": len(visual_audit.get("cross_panel_edges") or []),
        "feedback_edges": len(visual_audit.get("feedback_edges") or []),
        "removed_connector_count": len(visual_audit.get("removed_connectors") or []),
        "image": dict(visual_audit.get("image") or {}),
        "warnings": list(visual_audit.get("warnings") or []),
        "errors": list(visual_audit.get("errors") or []),
        "pilot_label_agreement": dict(visual_audit.get("pilot_label_agreement") or {}),
        "graph_id": visual.get("graph_id"),
    } if (visual_entry or visual_report) else None
    binding_audit = binding_report or (binding_entry or {}).get("binding_report") or {}
    # The stored report is the only thing the next run can read back, so every key
    # the binding coverage aggregates has to survive the merge. Listing the keys
    # here silently turned the measurements a later run aggregates into zeroes.
    merged["binding"] = {
        **binding_audit,
        "paper_id": merged["paper_id"],
        "ok": bool(binding_audit.get("ok")),
        "binding_version": binding_audit.get("binding_version") or BINDING_VERSION,
        "provider_calls": 0,
        "binding_set_id": ((binding_entry or {}).get("graph_bindings") or {}).get("binding_set_id"),
    } if (binding_entry or binding_report) else None
    merged["index"] = {
        "paper_id": merged["paper_id"],
        "ok": bool((index_report or {}).get("ok")),
        "record_count": (index_report or {}).get("record_count"),
        "record_ids": list((index_report or {}).get("record_ids") or []),
        "node_count": (index_report or {}).get("node_count"),
        "edge_count": (index_report or {}).get("edge_count"),
        "motif_count": (index_report or {}).get("motif_count"),
        "subgraph_count": (index_report or {}).get("subgraph_count"),
        "provider_calls": 0,
        "errors": list((index_report or {}).get("errors") or []),
    } if index_report else None
    check_summary = dict((check_report or {}).get("summary") or {})
    merged["correspondence"] = {
        "paper_id": merged["paper_id"],
        "ok": bool((check_report or {}).get("ok")),
        "prompt_version": (check_report or {}).get("prompt_version"),
        "concepts": int(check_summary.get("concepts") or 0),
        "required_concepts": int(check_summary.get("required_concepts") or 0),
        "confirmed": int(check_summary.get("confirmed") or 0),
        "missed_by_reader": int(check_summary.get("missed_by_reader") or 0),
        "hallucinated_block": int(check_summary.get("hallucinated_block") or 0),
        "not_drawn": int(check_summary.get("not_drawn") or 0),
        "unreadable": int(check_summary.get("unreadable") or 0),
        "required_visible_rate": check_summary.get("required_visible_rate"),
        "reader_precision": check_summary.get("reader_precision"),
        "reader_recall": check_summary.get("reader_recall"),
        "direction_conflicts": int(check_summary.get("direction_conflicts") or 0),
        "extra_blocks": int(check_summary.get("extra_blocks") or 0),
        "region_annotations": int(check_summary.get("region_annotations") or 0),
        "unlisted_content": int(check_summary.get("unlisted_content") or 0),
        "findings_outside_contract": list((check_report or {}).get("findings_outside_contract") or []),
        "errors": list((check_report or {}).get("errors") or []),
        "warnings": list((check_report or {}).get("warnings") or []),
    } if check_report else None
    return merged


def coverage_report(graph_id: str, reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize what the prepared documents cover."""
    by_split: dict[str, int] = {}
    by_domain: dict[str, dict[str, int]] = {}
    missing_captions: list[str] = []
    empty_evidence: list[str] = []
    warned: list[str] = []
    for report in reports:
        split = str(report.get("split") or "")
        domain = str(report.get("domain") or "")
        by_split[split] = by_split.get(split, 0) + 1
        by_domain.setdefault(domain, {})
        by_domain[domain][split] = by_domain[domain].get(split, 0) + 1
        stats = report.get("document") or {}
        if not stats.get("figure_count"):
            missing_captions.append(report["paper_id"])
        if not stats.get("evidence_count"):
            empty_evidence.append(report["paper_id"])
        if report.get("warnings"):
            warned.append(report["paper_id"])
    return {
        "schema_version": SCHEMA_VERSION,
        "graph_id": graph_id,
        "papers": len(reports),
        "by_split": by_split,
        "by_domain": by_domain,
        "documents_rebuilt": sum(1 for report in reports if (report.get("document") or {}).get("char_count")),
        "missing_captions": missing_captions,
        "empty_evidence": empty_evidence,
        "papers_with_warnings": warned,
        "hidden_test_papers": sum(1 for report in reports if str(report.get("split") or "") not in PILOT_SPLITS),
    }


def validate_prepare(reports: list[dict[str, Any]], ledger: list[dict[str, Any]], coverage: dict[str, Any]) -> dict[str, Any]:
    """Check the prepared documents against the G0 contract."""
    failures: list[str] = []
    human_review: list[dict[str, str]] = []
    prepare_rows = [row for row in ledger if str(row.get("step") or "") == PREPARE_STEP]
    if len(reports) != len(prepare_rows):
        failures.append("report and prepare ledger counts differ")
    if any(report.get("failure") for report in reports):
        failures.append("one or more documents could not produce a review request")
    if coverage.get("hidden_test_papers"):
        failures.append("Pilot graph artifacts contain hidden_test papers")
    duplicate_ids = [report["paper_id"] for report in reports if (report.get("document") or {}).get("duplicate_evidence_ids")]
    if duplicate_ids:
        failures.append(f"duplicate evidence ids: {duplicate_ids}")
    for report in reports:
        stats = report.get("document") or {}
        if not stats.get("figure_count"):
            human_review.append({"paper_id": report["paper_id"], "reason": "no figure caption in the frozen snapshot"})
        if not stats.get("evidence_count"):
            human_review.append({"paper_id": report["paper_id"], "reason": "no evidence text in the frozen snapshot"})
        if not stats.get("heading_count"):
            human_review.append({"paper_id": report["paper_id"], "reason": "no section heading detected"})
    checks = {
        "papers": len(reports),
        "documents_rebuilt": coverage.get("documents_rebuilt"),
        "evidence_ids_unique": not duplicate_ids,
        "prompts_built": sum(1 for report in reports if (report.get("review_context") or {}).get("prompt_chars")),
        "no_hidden_test_papers": not coverage.get("hidden_test_papers"),
        "failures": [],
    }
    checks["failures"] = failures
    return {"ok": not failures, "checks": checks, "failures": failures, "human_review_required": human_review}


def _ordered_values(mapping: dict[str, Any], paper_ids: list[str]) -> list[Any]:
    """Return stored stage payloads in Pilot ledger order."""
    return [mapping[paper_id] for paper_id in paper_ids if paper_id in mapping]


def build_pilot_graph_payloads(snapshot: str | Path, pilot: str | Path, *, graph_id: str = DEFAULT_GRAPH_ID, stages: str | Iterable[str] | None = None, limit: int | None = None, evidence_max_chars: int = DEFAULT_EVIDENCE_MAX_CHARS, model: str | None = None, timeout_seconds: int = 300, retries: int = 1, workers: int = 4, resume_dir: str | Path | None = None, figure_dir: str | Path | None = None, figure_dpi: int = DEFAULT_FIGURE_DPI, max_long_side: int = MAX_LONG_SIDE, probe_top_k: int = DEFAULT_PROBE_TOP_K, probe_limit: int = DEFAULT_PROBE_LIMIT, embedding_model: str = "", binding_embedder: str | None = None, strict: bool = False) -> dict[str, Any]:
    """Run the requested Pilot graph stages and collect every artifact.

    Stage order is the dependency order: review feeds contract, contract feeds
    graph, and the visual stage reads the paper's own framework figure. A resume
    directory keeps the artifacts of stages this run did not request, so
    continuing a version never drops earlier work.
    """
    snapshot = Path(snapshot)
    pilot = Path(pilot)
    requested = parse_stages(stages)
    run_review = REVIEW_STEP in requested
    run_contract = CONTRACT_STEP in requested
    run_graph = GRAPH_STEP in requested
    run_visual = VISUAL_STEP in requested
    run_bind = BIND_STEP in requested
    run_index = INDEX_STEP in requested
    run_check = CHECK_STEP in requested
    reports, prepare_ledger, rows = prepare_all(snapshot, pilot, limit=limit, evidence_max_chars=evidence_max_chars)
    paper_ids = [str(row.get("paper_id") or "") for row in rows]
    resumed = load_resume_state(resume_dir) if resume_dir is not None else {}
    review_map: dict[str, dict[str, Any]] = dict(resumed.get("reviews") or {})
    contract_map: dict[str, dict[str, Any]] = dict(resumed.get("contracts") or {})
    contract_report_map: dict[str, dict[str, Any]] = dict(resumed.get("contract_reports") or {})
    specification_map: dict[str, dict[str, Any]] = dict(resumed.get("specifications") or {})
    # A resumed version is re-sealed under the current artifact schema, so the
    # files that schema declares must be present even when this run does not
    # request the stage that writes them.
    specification_map = backfill_resumed_specifications(
        snapshot, rows, reviews=review_map, specifications=specification_map
    )
    graph_map: dict[str, dict[str, Any]] = dict(resumed.get("graphs") or {})
    visual_map: dict[str, dict[str, Any]] = dict(resumed.get("visuals") or {})
    visual_report_map: dict[str, dict[str, Any]] = dict(resumed.get("visual_reports") or {})
    bind_map: dict[str, dict[str, Any]] = dict(resumed.get("bindings") or {})
    binding_report_map: dict[str, dict[str, Any]] = dict(resumed.get("binding_reports") or {})
    check_map: dict[str, dict[str, Any]] = dict(resumed.get("checks") or {})
    check_report_map: dict[str, dict[str, Any]] = dict(resumed.get("check_reports") or {})
    fresh_review_ledger: list[dict[str, Any]] = []
    fresh_contract_ledger: list[dict[str, Any]] = []
    fresh_graph_ledger: list[dict[str, Any]] = []
    fresh_visual_ledger: list[dict[str, Any]] = []
    fresh_bind_ledger: list[dict[str, Any]] = []
    fresh_index_ledger: list[dict[str, Any]] = []
    fresh_check_ledger: list[dict[str, Any]] = []

    if run_review:
        fresh_reviews, fresh_review_ledger = review_all(
            snapshot,
            rows,
            existing_reviews=review_map,
            model=model,
            timeout_seconds=timeout_seconds,
            retries=retries,
            workers=workers,
            evidence_max_chars=evidence_max_chars,
            strict=strict,
        )
        review_map = {_review_paper_id(row): row for row in fresh_reviews}
    if run_contract:
        contract_map, fresh_contract_ledger, fresh_contract_reports, specification_map = contract_all(
            snapshot,
            rows,
            reviews=review_map,
            existing_contracts=contract_map,
            existing_specifications=specification_map,
            model=model,
            timeout_seconds=int(timeout_seconds),
            retries=retries,
            workers=workers,
            strict=strict,
        )
        contract_report_map.update(fresh_contract_reports)
    if run_graph:
        graph_map, fresh_graph_ledger = graph_all(
            snapshot,
            rows,
            reviews=review_map,
            existing_graphs=graph_map,
        )
    figure_root: Path | None = None
    if run_visual:
        figure_root = (
            Path(figure_dir)
            if figure_dir is not None
            else (Path(resume_dir) / "figures" if resume_dir is not None else Path(tempfile.mkdtemp(prefix="pilot-graph-figures-")))
        )
        if (figure_root.parent / "COMPLETE").is_file():
            raise FileExistsError("Refusing to render figures into a sealed version: " + str(figure_root.parent))
        visual_map, fresh_visual_ledger, fresh_visual_reports = visual_all(
            snapshot,
            rows,
            graphs=graph_map,
            figure_dir=figure_root,
            existing_visuals=visual_map,
            dpi=figure_dpi,
            max_long_side=max_long_side,
            model=model,
            timeout_seconds=timeout_seconds,
            retries=retries,
            workers=workers,
            strict=strict,
        )
        visual_report_map.update(fresh_visual_reports)
    if run_check:
        check_map, fresh_check_ledger, fresh_check_reports = check_all(
            snapshot,
            rows,
            graphs=graph_map,
            visuals=visual_map,
            figure_dir=figure_root if figure_root is not None else (Path(resume_dir) / "figures" if resume_dir is not None else Path(tempfile.mkdtemp(prefix="pilot-check-figures-"))),
            existing_checks=check_map,
            directories={each_paper_id: safe_paper_directory(each_paper_id) for each_paper_id in paper_ids},
            dpi=figure_dpi,
            max_long_side=max_long_side,
            model=model,
            timeout_seconds=timeout_seconds,
            retries=retries,
            workers=workers,
            strict=strict,
        )
        check_report_map.update(fresh_check_reports)
    if run_bind:
        # The bind stage's semantic layer is opt-in: an unset spec keeps the three
        # deterministic layers, and a paid endpoint has to be named on purpose.
        binding_encoder, binding_embedding = resolve_binding_embedder(
            binding_embedder if binding_embedder is not None else os.getenv("RFS_BINDING_EMBEDDER")
        )
        bind_map, fresh_bind_ledger, fresh_binding_reports = bind_all(
            rows,
            graphs=graph_map,
            visuals=visual_map,
            checks=check_map,
            existing_bindings=bind_map,
            embedder=binding_encoder,
            embedder_status=str(binding_embedding.get("status") or "off"),
            embedder_error=str(binding_embedding.get("error") or ""),
            provenance=PaperProvenanceLookup(snapshot, load_snapshot_papers(snapshot)),
        )
        binding_report_map.update(fresh_binding_reports)
    index_payload: dict[str, Any] | None = None
    # The subgraph vectors are part of the index stage, so the encoder is chosen here
    # and recorded in the manifest: a version must never ship vectors whose model it
    # cannot name. An unnamed encoder is the offline hashing one, not a paid endpoint.
    encoder = resolve_embedder(embedding_model) if run_index else None
    if run_index:
        index_payload, fresh_index_ledger = index_all(
            rows,
            graphs=graph_map,
            visuals=visual_map,
            graph_id=graph_id,
            directories={each_paper_id: safe_paper_directory(each_paper_id) for each_paper_id in paper_ids},
            probe_top_k=probe_top_k,
            probe_limit=probe_limit,
            embedder=encoder,
        )
    else:
        stored_index = resumed.get("index")
        if isinstance(stored_index, dict):
            index_payload = stored_index

    ledger = (
        prepare_ledger
        + _stage_ledger(rows, fresh_review_ledger, resumed.get("review_ledger") or {})
        + _stage_ledger(rows, fresh_contract_ledger, resumed.get("contract_ledger") or {})
        + _stage_ledger(rows, fresh_graph_ledger, resumed.get("graph_ledger") or {})
        + _stage_ledger(rows, fresh_visual_ledger, resumed.get("visual_ledger") or {})
        + _stage_ledger(rows, fresh_check_ledger, resumed.get("check_ledger") or {})
        + _stage_ledger(rows, fresh_bind_ledger, resumed.get("binding_ledger") or {})
        + _stage_ledger(rows, fresh_index_ledger, resumed.get("index_ledger") or {})
    )
    completed = [PREPARE_STEP]
    if review_map:
        completed.append(REVIEW_STEP)
    if contract_map:
        completed.append(CONTRACT_STEP)
    if graph_map:
        completed.append(GRAPH_STEP)
    if visual_map:
        completed.append(VISUAL_STEP)
    if check_map:
        completed.append(CHECK_STEP)
    if bind_map:
        completed.append(BIND_STEP)
    if index_payload is not None:
        completed.append(INDEX_STEP)
    reviews = _ordered_values(review_map, paper_ids)
    graphs = _ordered_values(graph_map, paper_ids)
    contract_entries = [
        {
            "paper_id": paper_id,
            "content": contract_map[paper_id],
            "figure_specification": specification_map.get(paper_id),
        }
        for paper_id in paper_ids
        if paper_id in contract_map
    ]
    contract_report_entries = [
        {"paper_id": paper_id, "report": contract_report_map[paper_id]}
        for paper_id in paper_ids
        if paper_id in contract_report_map
    ]
    stage_status: dict[str, dict[str, str]] = {}
    for row in ledger:
        stage_status.setdefault(str(row.get("paper_id") or ""), {})[str(row.get("step") or "")] = str(row.get("status") or "")
    index_report_map = {
        str(row.get("paper_id") or ""): paper_index_report(row)
        for row in ledger
        if str(row.get("step") or "") == INDEX_STEP
    }
    merged_reports = [
        merge_paper_report(
            report,
            review_map.get(report["paper_id"]),
            contract_report_map.get(report["paper_id"]),
            graph_map.get(report["paper_id"]),
            stage_status.get(report["paper_id"], {}),
            visual_map.get(report["paper_id"]),
            visual_report_map.get(report["paper_id"]),
            bind_map.get(report["paper_id"]),
            binding_report_map.get(report["paper_id"]),
            index_report=index_report_map.get(report["paper_id"]),
            check_report=check_report_map.get(report["paper_id"]),
        )
        for report in reports
    ]
    # Every later stage's audit is merged into the prepare report before it is
    # written, so the ledger hash is re-pointed at the report as it is sealed.
    merged_report_sha256 = {report["paper_id"]: canonical_sha256(report) for report in merged_reports}
    for row in prepare_ledger:
        claimed = (row.get("artifact_sha256") or {}).get("graph_report.json")
        refreshed = merged_report_sha256.get(str(row.get("paper_id") or ""))
        if claimed and refreshed:
            row["artifact_sha256"]["graph_report.json"] = refreshed
    visuals = _ordered_values(visual_map, paper_ids)
    bindings = _ordered_values(bind_map, paper_ids)
    check_entries = _ordered_values(check_map, paper_ids)
    coverage = coverage_report(graph_id, merged_reports)
    coverage["reviews"] = len(reviews)
    coverage["contracts"] = len(contract_entries)
    coverage["scientific"] = scientific_coverage(graph_map)
    coverage["visual"] = visual_coverage(visual_map)
    coverage["bindings"] = binding_coverage(bind_map, without_flow=coverage["visual"].get("papers_without_flow") or [])
    coverage["index"] = index_coverage(index_payload)
    coverage["correspondence"] = check_coverage(check_map, papers=len(paper_ids))
    validation = validate_stages(
        merged_reports,
        ledger,
        coverage,
        reviews,
        review_expected=run_review or REVIEW_STEP in completed,
        contracts=contract_entries,
        contract_expected=run_contract or CONTRACT_STEP in completed,
        contract_reports=[entry["report"] for entry in contract_report_entries],
        graphs=graphs,
        graph_expected=run_graph or GRAPH_STEP in completed,
        visuals=visuals,
        visual_expected=run_visual or VISUAL_STEP in completed,
        visual_reports=[entry["visual_graph_report"] for entry in visuals if entry.get("visual_graph_report")],
        bindings=bindings,
        bind_expected=run_bind or BIND_STEP in completed,
        binding_reports=[entry["binding_report"] for entry in bindings if entry.get("binding_report")],
        index_payload=index_payload,
        index_expected=run_index or INDEX_STEP in completed,
        check_entries=check_entries,
        check_expected=run_check or CHECK_STEP in completed,
        check_reports=[entry["check_report"] for entry in check_entries if entry.get("check_report")],
    )
    costs = cost_report(graph_id, ledger, projection_papers=PROJECTION_PAPERS if (review_map or contract_map) else 0)
    snapshot_manifest = snapshot / "snapshot_manifest.json"
    pilot_ledger_path = pilot / "pilot_ledger.jsonl"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "graph_id": graph_id,
        "summary": "Phase 3 Pilot graph middleware artifacts.",
        "stages_requested": list(requested),
        "stages_completed": completed,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "paper_artifacts": {step: list(names) for step, names in PAPER_ARTIFACTS.items()},
        "inputs": {
            "snapshot": snapshot.name,
            "snapshot_manifest_sha256": sha256_file(snapshot_manifest) if snapshot_manifest.is_file() else None,
            "pilot": pilot.name,
            "pilot_ledger_sha256": sha256_file(pilot_ledger_path),
            "pilot_papers": len(rows),
        },
        "documents": {
            "document_version": FROZEN_DOCUMENT_VERSION,
            "specification_version": FROZEN_SPECIFICATION_VERSION,
            "specification_source": "pilot-review-seed",
            "loader": "frozen-snapshot",
            "evidence_max_chars": int(evidence_max_chars),
        },
        "domain_review_profiles": dict(DOMAIN_REVIEW_PROFILE),
        "provider": {
            "model": model,
            "model_env": "RFS_PAPER_REVIEW_MODEL",
            "review_model_env": "RFS_PAPER_REVIEW_MODEL",
            "contract_model_env": "RFS_PAPER_REVIEW_MODEL",
            "calls": sum(1 for row in ledger if str(row.get("step") or "") in BILLABLE_STEPS),
            "timeout_seconds": int(timeout_seconds),
            "retries": int(retries),
            "workers": int(workers),
            "strict": bool(strict),
        },
        "figures": {
            "render_directory": figure_root.name if figure_root is not None else None,
            "render_dpi": int(figure_dpi),
            "max_long_side": int(max_long_side),
            "visual_prompt_version": VISUAL_PROMPT_VERSION,
            "visual_model_env": "RFS_PAPER_REVIEW_MODEL",
            "rendered": len(visual_map),
        },
        "index": {
            "artifacts": list(INDEX_ARTIFACTS),
            "provider_calls": 0,
            "probe_top_k": int(probe_top_k),
            "probe_limit": int(probe_limit),
            "subgraph_policy": "locator index; a subgraph is re-extracted from the stored full graph on demand",
            "embedding_spec": str(embedding_model or ""),
            "embedding": index_report(index_payload).get("embedding") or {},
            "report": index_report(index_payload),
        } if index_payload is not None else None,
        "correspondence": {
            "prompt_version": CHECK_PROMPT_VERSION,
            "reviewed_papers": len(check_map),
            "prompt_scope": "Pilot papers only; every other D0 paper stays unreviewed",
        } if check_map else None,
        "counts": coverage["by_split"],
        "volatile_fields": list(VOLATILE_LEDGER_FIELDS),
    }
    return {
        "manifest": manifest,
        "ledger": ledger,
        "reports": merged_reports,
        "reviews": reviews,
        "contracts": contract_entries,
        "contract_reports": contract_report_entries,
        "graphs": graphs,
        "visuals": visuals,
        "bindings": bindings,
        "index": index_payload,
        "checks": check_entries,
        "coverage": coverage,
        "costs": costs,
        "validation": validation,
        "stages_requested": list(requested),
        "stages_completed": completed,
        "complete": set(completed) >= set(STAGE_ORDER),
    }


def write_pilot_graph_version(out_dir: str | Path, payloads: dict[str, Any]) -> dict[str, Any]:
    """Write a Pilot graph version. A sealed version is never overwritten."""
    target = Path(out_dir)
    if (target / "COMPLETE").is_file():
        raise FileExistsError(f"Refusing to overwrite a sealed version: {target}")
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "graph_manifest.json", payloads["manifest"])
    write_jsonl(target / "graph_ledger.jsonl", payloads["ledger"])
    write_json(target / "coverage_report.json", payloads["coverage"])
    write_json(target / "cost_report.json", payloads["costs"])
    write_json(target / "validation_report.json", payloads["validation"])
    if payloads.get("reviews"):
        write_jsonl(target / "paper_reviews.jsonl", payloads["reviews"])
        for entry in payloads["reviews"]:
            paper_id = _review_paper_id(entry)
            if paper_id:
                write_json(_paper_dir(target, paper_id) / "paper_review.json", entry)
    for report in payloads.get("reports") or []:
        write_json(_paper_dir(target, report["paper_id"]) / "graph_report.json", report)
    for entry in payloads.get("contracts") or []:
        directory = _paper_dir(target, entry["paper_id"])
        write_json(directory / FIGURE_CONTENT_PLAN_FILE, entry["content"])
        if entry.get("figure_specification") is not None:
            write_json(directory / METHOD_INVENTORY_FILE, entry["figure_specification"])
    for entry in payloads.get("contract_reports") or []:
        write_json(_paper_dir(target, entry["paper_id"]) / "method_extraction_report.json", entry["report"])
    for entry in payloads.get("graphs") or []:
        directory = _paper_dir(target, entry["paper_id"])
        for name, key in (
            ("scientific_graph.json", "scientific_graph"),
            ("scientific_ground_truth.json", "scientific_ground_truth"),
            ("scientific_graph_report.json", "scientific_graph_report"),
        ):
            if entry.get(key) is not None:
                write_json(directory / name, entry[key])
    for entry in payloads.get("visuals") or []:
        directory = _paper_dir(target, entry["paper_id"])
        write_json(directory / "visual_graph.json", entry["visual_graph"])
        if entry.get("visual_graph_report") is not None:
            write_json(directory / "visual_graph_report.json", entry["visual_graph_report"])
    for entry in payloads.get("bindings") or []:
        directory = _paper_dir(target, entry["paper_id"])
        write_json(directory / "graph_bindings.json", entry["graph_bindings"])
        if entry.get("figure_claims") is not None:
            write_json(directory / "figure_claims.json", entry["figure_claims"])
    for entry in payloads.get("checks") or []:
        write_json(_paper_dir(target, entry["paper_id"]) / "figure_text_review.json", entry["figure_text_review"])
    index_payload = payloads.get("index") or None
    if index_payload is not None:
        write_jsonl(target / "graph_index.jsonl", index_payload.get("records") or [])
        write_jsonl(target / "motif_index.jsonl", index_payload.get("motifs") or [])
        write_jsonl(target / "subgraph_index.jsonl", index_payload.get("subgraphs") or [])
        write_json(target / "retrieval_probe.json", index_payload.get("probe") or {})
    files = {name: sha256_file(target / name) for name in GRAPH_VERSION_ARTIFACTS if (target / name).is_file()}
    for name in INDEX_ARTIFACTS:
        if (target / name).is_file():
            files[name] = sha256_file(target / name)
    if (target / "paper_reviews.jsonl").is_file():
        files["paper_reviews.jsonl"] = sha256_file(target / "paper_reviews.jsonl")
    sealed = bool(payloads["validation"]["ok"]) and bool(payloads.get("complete"))
    if sealed:
        write_text(target / "COMPLETE", json.dumps({"graph_id": payloads["manifest"]["graph_id"], "validation_report_sha256": files.get("validation_report.json"), "stages": payloads["stages_completed"]}, ensure_ascii=False, sort_keys=True) + "\n")
        files["COMPLETE"] = sha256_file(target / "COMPLETE")
    return {"out_dir": str(target), "files": files, "sealed": sealed, "stages_completed": payloads["stages_completed"]}


def ledger_artifact_mismatches(directory: str | Path) -> list[dict[str, Any]]:
    """Compare every hash the ledger claims with the artifact stored on disk.

    A Pilot graph version is only trustworthy while its own audit trail still
    describes the files it ships, so each ``artifact_sha256`` entry is recomputed
    from the stored artifact. Paper artifacts are hashed from the parsed JSON
    content, which is the canonical form the ledger recorded, so the check does
    not depend on file formatting. Review rows live in ``paper_reviews.jsonl``
    instead of a per paper file and are found through
    ``graph_provenance.paper_id``.
    """
    target = Path(directory)
    ledger_path = target / "graph_ledger.jsonl"
    if not ledger_path.is_file():
        return []
    reviews: dict[str, dict[str, Any]] = {}
    reviews_path = target / "paper_reviews.jsonl"
    if reviews_path.is_file():
        for row in strict_jsonl(reviews_path):
            paper_id = _review_paper_id(row)
            if paper_id:
                reviews[paper_id] = row
    mismatches: list[dict[str, Any]] = []
    for row in strict_jsonl(ledger_path):
        step = str(row.get("step") or "")
        paper_id = str(row.get("paper_id") or "")
        for artifact, declared in sorted((row.get("artifact_sha256") or {}).items()):
            if artifact == REVIEW_ARTIFACT:
                location = "paper_reviews.jsonl"
                stored = reviews.get(paper_id)
                actual = canonical_sha256(stored) if stored is not None else None
                reason = "missing" if actual is None else "content_mismatch"
            else:
                name = (LEDGER_ARTIFACT_FILES.get(step) or {}).get(artifact)
                location = name or artifact
                actual = None
                reason = "unknown_artifact"
                if name:
                    stored_path = _paper_dir(target, paper_id) / name
                    if not stored_path.is_file():
                        renamed = LEDGER_ARTIFACT_RENAMES.get(name)
                        if renamed:
                            alternative = _paper_dir(target, paper_id) / renamed
                            if alternative.is_file():
                                stored_path, location = alternative, renamed
                    reason = "missing"
                    if stored_path.is_file():
                        try:
                            actual = canonical_sha256(json.loads(stored_path.read_text(encoding="utf-8")))
                            reason = "content_mismatch"
                        except (OSError, ValueError) as error:
                            reason = f"unreadable: {error}"
            if actual != declared:
                mismatches.append({
                    "paper_id": paper_id,
                    "step": step,
                    "artifact": artifact,
                    "location": location,
                    "expected": declared,
                    "actual": actual,
                    "reason": reason,
                })
    return mismatches


def figure_render_mismatches(directory: str | Path) -> list[dict[str, Any]]:
    """Check every rendered framework figure against its render ledger.

    The phase 3 acceptance rule is that each real figure ships with a
    reproducible render ledger, so beside every ``figures/<paper>/<stem>.png``
    there must be a ``<stem>.png.render.json`` recording the hash of that exact
    PNG. A missing ledger, an unreadable one, or a recorded ``png_sha256`` that
    no longer matches the file is reported here.
    """
    target = Path(directory)
    figures = target / "figures"
    if not figures.is_dir():
        return []
    mismatches: list[dict[str, Any]] = []
    for png in sorted(figures.glob("*/*.png")):
        relative = png.relative_to(target).as_posix()
        ledger_path = Path(f"{png}.render.json")
        if not ledger_path.is_file():
            mismatches.append({
                "figure": relative,
                "ledger": ledger_path.relative_to(target).as_posix(),
                "reason": "missing_ledger",
                "expected": None,
                "actual": None,
            })
            continue
        recorded = None
        try:
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                recorded = payload.get("png_sha256")
        except (OSError, ValueError) as error:
            mismatches.append({
                "figure": relative,
                "ledger": ledger_path.relative_to(target).as_posix(),
                "reason": f"unreadable_ledger: {error}",
                "expected": None,
                "actual": None,
            })
            continue
        actual = sha256_file(png)
        if recorded != actual:
            mismatches.append({
                "figure": relative,
                "ledger": ledger_path.relative_to(target).as_posix(),
                "reason": "ledger_hash_mismatch",
                "expected": recorded,
                "actual": actual,
            })
    return mismatches


def validate_pilot_graph_seal(target: Path, manifest: dict[str, Any], stages: list[str]) -> dict[str, Any]:
    """Check a ``COMPLETE`` marker against the version it claims to seal.

    The marker promises which graph id was sealed, which stages ran, and the
    hash of the validation report that allowed the seal. All three are
    re-derived here so a stale marker, or a marker whose report was edited
    afterwards, is reported instead of trusted.
    """
    marker = target / "COMPLETE"
    report_path = target / "validation_report.json"
    seal: dict[str, Any] = {
        "present": marker.is_file(),
        "valid": False,
        "graph_id": None,
        "stages": [],
        "validation_report_sha256": None,
        "graph_id_matches": False,
        "stages_match": False,
        "validation_report_sha256_matches": False,
        "validation_report_ok": None,
        "error": None,
    }
    if report_path.is_file():
        try:
            seal["validation_report_ok"] = bool(json.loads(report_path.read_text(encoding="utf-8")).get("ok"))
        except (OSError, ValueError) as error:
            seal["validation_report_ok"] = False
            seal["error"] = f"validation_report.json unreadable: {error}"
    if not seal["present"]:
        return seal
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        seal["error"] = f"COMPLETE unreadable: {error}"
        return seal
    if not isinstance(payload, dict):
        seal["error"] = "COMPLETE is not a JSON object"
        return seal
    seal["graph_id"] = payload.get("graph_id")
    seal["stages"] = [str(item) for item in (payload.get("stages") or [])]
    seal["validation_report_sha256"] = payload.get("validation_report_sha256")
    seal["graph_id_matches"] = seal["graph_id"] == manifest.get("graph_id")
    seal["stages_match"] = seal["stages"] == list(stages)
    seal["validation_report_sha256_matches"] = bool(seal["validation_report_sha256"]) and seal["validation_report_sha256"] == (
        sha256_file(report_path) if report_path.is_file() else None
    )
    seal["valid"] = all((
        seal["graph_id_matches"],
        seal["stages_match"],
        seal["validation_report_sha256_matches"],
        bool(seal["validation_report_ok"]),
    ))
    return seal


def expected_paper_artifacts(manifest: dict[str, Any], stages: Iterable[str]) -> list[str]:
    """Return the per paper files a version must carry for the stages it completed.

    A version states its own list in ``paper_artifacts``. A version written
    before that key existed is read through its ``artifact_schema_version``, and
    a version written before the schema number existed at all falls back to
    schema 1, so a sealed version keeps being checked against the rules it was
    sealed with instead of today's list.
    """
    declared = manifest.get("paper_artifacts")
    if not isinstance(declared, dict):
        try:
            schema = int(manifest.get("artifact_schema_version"))
        except (TypeError, ValueError):
            schema = 1
        declared = PAPER_ARTIFACTS_BY_SCHEMA.get(schema, PAPER_ARTIFACTS)
    completed = {str(item) for item in stages}
    files: list[str] = []
    for step in PAPER_ARTIFACT_STEPS:
        if step in completed:
            files.extend(str(name) for name in (declared.get(step) or ()))
    return files


def validate_pilot_graph_version(version_dir: str | Path) -> dict[str, Any]:
    """Re-read a written Pilot graph version and report what is present."""
    target = Path(version_dir)
    if not target.is_dir():
        raise FileNotFoundError(f"Pilot graph version does not exist: {target}")
    missing = [name for name in GRAPH_VERSION_ARTIFACTS if not (target / name).is_file()]
    manifest_path = target / "graph_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    ledger = strict_jsonl(target / "graph_ledger.jsonl") if (target / "graph_ledger.jsonl").is_file() else []
    paper_ids = [str(row.get("paper_id") or "") for row in ledger]
    reports_missing = [paper_id for paper_id in paper_ids if not (_paper_dir(target, paper_id) / "graph_report.json").is_file()]
    stages = [str(item) for item in (manifest.get("stages_completed") or [])]
    expected_paper_files = expected_paper_artifacts(manifest, stages)
    missing_paper_artifacts = [
        f"{paper_id}/{name}"
        for paper_id in dict.fromkeys(paper_ids)
        for name in expected_paper_files
        if not (_paper_dir(target, paper_id) / name).is_file()
    ]
    reviews_path = target / "paper_reviews.jsonl"
    reviews = strict_jsonl(reviews_path) if reviews_path.is_file() else []
    reviewing = REVIEW_STEP in stages
    failed_reviews = _failed_papers(ledger, REVIEW_STEP)
    failed_contracts = _failed_papers(ledger, CONTRACT_STEP)
    failed_graphs = _failed_papers(ledger, GRAPH_STEP)
    failed_visuals = _failed_papers(ledger, VISUAL_STEP)
    failed_bindings = _failed_papers(ledger, BIND_STEP)
    contract_papers = sorted({
        str(row.get("paper_id") or "")
        for row in ledger
        if row.get("step") == CONTRACT_STEP and str(row.get("status") or "") == "ok"
    })
    graph_papers = sorted({
        str(row.get("paper_id") or "")
        for row in ledger
        if row.get("step") == GRAPH_STEP and str(row.get("status") or "") == "ok"
    })
    visual_papers = sorted({
        str(row.get("paper_id") or "")
        for row in ledger
        if row.get("step") == VISUAL_STEP and str(row.get("status") or "") == "ok"
    })
    binding_papers = sorted({
        str(row.get("paper_id") or "")
        for row in ledger
        if row.get("step") == BIND_STEP and str(row.get("status") or "") == "ok"
    })
    check_papers = sorted({
        str(row.get("paper_id") or "")
        for row in ledger
        if row.get("step") == CHECK_STEP and str(row.get("status") or "") == "ok"
    })
    index_records = strict_jsonl(target / "graph_index.jsonl") if (target / "graph_index.jsonl").is_file() else []
    probe_path = target / "retrieval_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.is_file() else {}
    indexing = INDEX_STEP in stages
    index_papers = {str(row.get("paper_id") or "") for row in index_records}
    index_missing_papers = sorted(paper_id for paper_id in sorted(set(paper_ids)) if paper_id not in index_papers) if indexing else []
    missing_index_artifacts = [name for name in INDEX_ARTIFACTS if not (target / name).is_file()] if indexing else []
    failed_index = _failed_papers(ledger, INDEX_STEP)
    figures_root = target / "figures"
    rendered_figures = {path.parent.name for path in figures_root.glob("*/*.png")} if figures_root.is_dir() else set()
    missing_figures = [paper_id for paper_id in visual_papers if safe_paper_directory(paper_id) not in rendered_figures]
    ledger_mismatches = ledger_artifact_mismatches(target)
    render_mismatches = figure_render_mismatches(target)
    seal = validate_pilot_graph_seal(target, manifest, stages)
    empty_artifacts = [
        name
        for name in GRAPH_VERSION_ARTIFACTS
        if (target / name).is_file() and (target / name).stat().st_size == 0
    ]
    checks = {
        "manifest_present": manifest_path.is_file(),
        "ledger_rows": len(ledger),
        "unique_papers": len(set(paper_ids)),
        "reports_present": not reports_missing,
        "reports_missing": reports_missing,
        "stages_completed": stages,
        "artifact_schema_version": manifest.get("artifact_schema_version"),
        "expected_paper_artifacts": expected_paper_files,
        "review_rows": len(reviews),
        "reviews_present": (not reviewing) or reviews_path.is_file(),
        "failed_reviews": failed_reviews,
        "contracts": len(contract_papers),
        "graphs": len(graph_papers),
        "visuals": len(visual_papers),
        "bindings": len(binding_papers),
        "failed_contracts": failed_contracts,
        "failed_graphs": failed_graphs,
        "failed_visuals": failed_visuals,
        "failed_bindings": failed_bindings,
        "rendered_figures": len(rendered_figures),
        "missing_figures": missing_figures,
        "missing_paper_artifacts": missing_paper_artifacts,
        "complete_marker": (target / "COMPLETE").is_file(),
        "seal": seal,
        "ledger_artifact_mismatches": ledger_mismatches,
        "figure_render_mismatches": render_mismatches,
        "empty_artifacts": empty_artifacts,
        "missing_artifacts": missing,
        "index_records": len(index_records),
        "index_papers": len(index_papers),
        "index_missing_papers": index_missing_papers,
        "index_probe_queries": len(list((probe or {}).get("queries") or [])),
        "index_motifs": len(strict_jsonl(target / "motif_index.jsonl")) if (target / "motif_index.jsonl").is_file() else 0,
        "missing_index_artifacts": missing_index_artifacts,
        "failed_index": failed_index,
        "checks": len(check_papers),
        "failed_checks": _failed_papers(ledger, CHECK_STEP),
    }
    ok = all((
        checks["manifest_present"],
        bool(ledger),
        not missing,
        not reports_missing,
        checks["reviews_present"],
        not missing_paper_artifacts,
        not failed_reviews,
        not failed_contracts,
        not failed_graphs,
        not failed_visuals,
        not failed_bindings,
        not missing_figures,
        not missing_index_artifacts,
        not index_missing_papers,
        not failed_index,
        not _failed_papers(ledger, CHECK_STEP),
        not ledger_mismatches,
        not render_mismatches,
        not empty_artifacts,
        (not seal["present"]) or bool(seal["valid"]),
    ))
    sealed = (
        bool(seal["valid"])
        and set(stages) >= set(STAGE_ORDER)
        and not ledger_mismatches
        and not render_mismatches
        and not empty_artifacts
    )
    return {
        "ok": ok,
        "graph_id": manifest.get("graph_id"),
        "papers": len(set(paper_ids)),
        "stages_completed": stages,
        "complete": sealed,
        "sealed": sealed,
        "checks": checks,
        "out_dir": str(target),
    }


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "BILLABLE_STEPS",
    "BINDING_VERSION",
    "CHECK_PROMPT_VERSION",
    "CHECK_STEP",
    "BIND_STEP",
    "CONTRACT_STEP",
    "CONTRACT_TIMEOUT_SECONDS",
    "DEFAULT_FIGURE_DPI",
    "DEFAULT_GRAPH_ID",
    "DOMAIN_REVIEW_PROFILE",
    "GRAPH_STEP",
    "IMPLEMENTED_STAGES",
    "INDEX_ARTIFACTS",
    "INDEX_STEP",
    "LEDGER_ARTIFACT_FILES",
    "MAX_LONG_SIDE",
    "PAPER_ARTIFACTS",
    "PAPER_ARTIFACTS_V1",
    "PAPER_ARTIFACTS_BY_SCHEMA",
    "REVIEW_ARTIFACT",
    "SCHEMA_VERSION",
    "STAGE_ORDER",
    "VISUAL_STEP",
    "backfill_resumed_specifications",
    "bind_all",
    "bind_paper",
    "check_all",
    "check_coverage",
    "check_paper",
    "binding_coverage",
    "build_layer_bindings",
    "build_pilot_graph_payloads",
    "canonical_bytes",
    "canonical_sha256",
    "contract_all",
    "contract_paper",
    "cost_report",
    "coverage_report",
    "derive_figure_specification",
    "expected_paper_artifacts",
    "graph_all",
    "graph_paper",
    "index_all",
    "index_coverage",
    "figure_render_mismatches",
    "index_report",
    "ledger_artifact_mismatches",
    "load_index_state",
    "load_pilot_ledger",
    "load_resume_state",
    "load_review_resume_state",
    "merge_paper_report",
    "parse_stages",
    "paper_index_report",
    "prepare_all",
    "prepare_paper",
    "review_all",
    "review_paper",
    "review_profile_for",
    "safe_paper_directory",
    "scientific_coverage",
    "validate_pilot_graph_seal",
    "validate_pilot_graph_version",
    "validate_prepare",
    "validate_stages",
    "visual_all",
    "visual_coverage",
    "visual_paper",
    "write_pilot_graph_version",
]
