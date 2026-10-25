"""Adapt one middleware version into the controlled-corruption preference source.


The controlled-corruption family never pays for image editing: a reviewed-good reference
framework figure is taken, exactly one contract violation is injected into it locally, and
the Judge learns the defect that the paper's own contract says must be visible. A
middleware version already holds both halves of that evidence - the paper's method
inventory and the rendered reference figure - in the middleware's own shape. This module
is the adapter that writes them in the shape the corruption chain reads:

``figures.jsonl``
    One row per reviewed-good reference figure: ``figure_id``, ``paper_id``,
    ``figure_path``, ``split`` and ``final_label``; the planner gates on the label.
``contracts/<paper>.json``
    The paper's method inventory as the Judge's D1 contract (``entities``,
    ``relations``, ``training_flow``, ``inference_flow`` and
    ``expected_figure_content``). This is what the corruption planner violates and what
    the region locator matches OCR text against.
``raw/<paper>.json``
    The title and abstract a Judge sample carries beside its images.

Nothing here calls a provider and nothing is invented: a paper that states no inventory, or
whose reference figure was never rendered, is reported as a failure instead of being shipped
with a placeholder contract. The middleware's own selection of what to draw (``must_show``,
``render_priority``, ``visualization_value``) is deliberately not carried over - it is the
figure content plan, and a Judge is never handed that - so the required set is every concept
the paper states rather than a chosen subset.

:func:`run_local_corruption_wave` then runs the CPU-only chain over the source: OCR locates
the modules the inventory names, a review decides which located regions are safe to edit,
the deterministic applier injects one defect per approved region, and the example builder
compiles JudgeExamples. The review is either the VLM-assisted audit (paid and conservative,
the default) or an explicitly labelled zero-cost geometry screen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ..utils import write_json
from .contract_export import snapshot_paper_index, version_reviews
from .corruption_dataset import build_local_corrupted_examples
from .local_corruptions import ELIGIBLE_LABELS, generate_approved_corruptions, generate_region_candidates
from .local_region_reviewer import review_local_regions, screen_regions_geometrically
from .method_inventory import method_inventory_view
from .pilot_graphs import safe_paper_directory
from .source_snapshot_io import sha256_file


CORRUPTION_SOURCE_CONTRACT = "rfs.judge_corruption_source"
CORRUPTION_SOURCE_VERSION = 2
SOURCE_MANIFEST_FILE = "source_bundle.json"
WAVE_REPORT_FILE = "corruption_wave_report.json"
FIGURES_FILE = "figures.jsonl"
CONTRACTS_DIRECTORY = "contracts"
RAW_DIRECTORY = "raw"
GRAPH_DIRECTORY = "graphs"
FIGURE_DIRECTORY = "figures"
REVIEW_FILE = "paper_review.json"
INDEX_FILE = "graph_index.jsonl"
RENDER_SUFFIX = ".render.json"
FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
DEFAULT_LABEL = "\u597d\uff08\u79d1\u7814\uff09"
SPLITS = ("train", "validation", "test")
#: Concept sections that become contract entities, in the order a reader meets them on the
#: figure. The workflow stages stay out of this list: they are the two flows below, so a
#: stage is never asked for as a missing module and a module is never asked for as a stage.
CONCEPT_SECTIONS = ("inputs", "modules", "innovations", "outputs", "optional_details")
#: The planner reads both flows as one ordered list of visible stages. The words that
#: separate a training stage from an inference stage are only used when the paper's own
#: review kept the two workflow halves apart.
FLOW_HINTS = {
    "training": ("train", "offline", "fit", "pretrain", "pre-train"),
    "inference": ("infer", "online", "deploy", "eval", "measure", "prediction", "test"),
}
MAX_EVIDENCE_SPANS = 200
REVIEW_MODES = ("vlm", "geometry")
OCR_ENGINES = {"rapidocr": "run_rapidocr", "easyocr": "run_easyocr", "paddle": "run_paddle_ocr"}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: str | Path, default: Any = None) -> Any:
    source = Path(path)
    if not source.is_file():
        return default
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _statements(value: Any) -> list[str]:
    """Return the non-empty strings of a statement list or a single statement."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    found: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            found.append(item.strip())
        elif isinstance(item, dict):
            text = str(item.get("statement") or item.get("text") or item.get("label") or "").strip()
            if text:
                found.append(text)
    return found


def _step_description(step: Any) -> str:
    """Render one workflow step, keeping its own statement beside its label."""
    if not isinstance(step, dict):
        return str(step or "").strip()
    label = str(step.get("label") or step.get("step_description") or step.get("description") or "").strip()
    seeds = _statements(step.get("seed_statements") or step.get("statements"))
    if label and seeds:
        return f"{label} \u2014 {seeds[0]}"
    return label or (seeds[0] if seeds else "")


def _flow_rows(steps: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for step in steps or []:
        text = _step_description(step)
        if text:
            rows.append({"step_description": text})
    return rows


def _split_flows(stages: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Separate the inventory's stages by the words that name each side of a workflow."""
    training: list[dict[str, str]] = []
    inference: list[dict[str, str]] = []
    for step in stages:
        text = _step_description(step)
        if not text:
            continue
        lowered = text.casefold()
        if any(hint in lowered for hint in FLOW_HINTS["inference"]):
            inference.append({"step_description": text})
        else:
            training.append({"step_description": text})
    return training, inference


def _flows(stages: list[dict[str, Any]], review: dict[str, Any]) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    """Return the two visible flows of a paper and the route that produced them."""
    workflows = review.get("workflows") if isinstance(review.get("workflows"), dict) else {}
    training = _flow_rows(workflows.get("training"))
    inference = _flow_rows(workflows.get("inference"))
    if training or inference:
        return training, inference, "review_workflows"
    training, inference = _split_flows(stages)
    return training, inference, "stages_keyword_split" if stages else "no_stages"


def _evidence_spans(review: dict[str, Any], *, limit: int = MAX_EVIDENCE_SPANS) -> list[dict[str, Any]]:
    """Collect the page quotes a review records, nearest first and without repeats."""
    spans: list[dict[str, Any]] = []
    seen: set[str] = set()

    def walk(value: Any) -> None:
        if len(spans) >= limit:
            return
        if isinstance(value, dict):
            references = value.get("evidence_refs")
            if isinstance(references, list):
                for reference in references:
                    if not isinstance(reference, dict) or len(spans) >= limit:
                        continue
                    key = str(reference.get("evidence_id") or reference.get("quote") or "").strip()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    spans.append({name: reference[name] for name in ("evidence_id", "page", "quote", "section") if name in reference})
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(review)
    return spans


def corruption_contract(inventory: dict[str, Any] | None, review: dict[str, Any] | None = None, *, paper_id: str = "") -> dict[str, Any]:
    """Express one paper's method inventory as the D1 contract the corruption chain reads.

    The inventory states concepts and relations the paper itself describes; the review adds
    the page quotes and, when it kept the two workflow halves apart, the training and
    inference stages. An inventory that states nothing yields a contract with no entities
    rather than a skeleton that pretends the paper stated something.
    """
    view = method_inventory_view(inventory)
    paper_review = review if isinstance(review, dict) else {}
    entities: list[dict[str, Any]] = []
    known: set[str] = set()
    for section in CONCEPT_SECTIONS:
        for row in view.get(section) or []:
            entity_id = str(row.get("id") or "").strip()
            if not entity_id or entity_id in known:
                continue
            known.add(entity_id)
            entities.append({
                "entity_id": entity_id,
                "name": str(row.get("label") or "").strip(),
                "role": str(row.get("role") or section).strip(),
                "section": section,
                "detail_level": str(row.get("detail_level") or "").strip(),
                "evidence_ids": [str(item) for item in (row.get("evidence_ids") or [])],
            })
    relations: list[dict[str, Any]] = []
    for row in view.get("relations") or []:
        source, target = str(row.get("source") or "").strip(), str(row.get("target") or "").strip()
        if not source or not target:
            continue
        relations.append({
            "source": source,
            "target": target,
            "relation_type": str(row.get("type") or row.get("relation_type") or "").strip(),
            "label": str(row.get("label") or "").strip(),
            "evidence_ids": [str(item) for item in (row.get("evidence_ids") or [])],
        })
    resolved = [row for row in relations if row["source"] in known and row["target"] in known]
    training_flow, inference_flow, flows_route = _flows(list(view.get("stages") or []), paper_review)
    expected = {
        "required_entities": [row["entity_id"] for row in entities],
        "required_relations": [f"{row['source']}->{row['target']}" for row in resolved],
        "optional_entities": [],
        "must_not_invent": _statements(view.get("forbidden_inventions")),
    }
    summary = str(view.get("figure_goal") or "").strip() or str(paper_review.get("summary") or "").strip()
    return {
        "paper_id": str(paper_id),
        "method_summary": summary,
        "entities": entities,
        "relations": resolved,
        "training_flow": training_flow,
        "inference_flow": inference_flow,
        "expected_figure_content": expected,
        "evidence_spans": _evidence_spans(paper_review),
        "provenance": {
            "contract": CORRUPTION_SOURCE_CONTRACT,
            "version": CORRUPTION_SOURCE_VERSION,
            "paper_id": str(paper_id),
            "statement_source": "middleware_method_inventory",
            "selection_keys_stripped": True,
            "flows_route": flows_route,
            "relations_dropped_unresolved": len(relations) - len(resolved),
        },
    }


def _figure_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in FIGURE_SUFFIXES and not path.name.endswith(RENDER_SUFFIX)
    )


def _version_index(version: Path) -> dict[str, dict[str, Any]]:
    """Map each sealed paper's directory name to its identity, split and domain."""
    index: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(version / INDEX_FILE):
        paper_id = str(row.get("paper_id") or "").strip()
        if not paper_id:
            continue
        name = safe_paper_directory(paper_id)
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        entry = index.setdefault(name, {"paper_id": paper_id, "split": "", "domain": ""})
        split = str(metadata.get("split") or "").strip()
        domain = str(metadata.get("domain") or "").strip()
        if split and not entry["split"]:
            entry["split"] = split
        if domain and not entry["domain"]:
            entry["domain"] = domain
    return index


def _split_ledger_index(ledger_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Map each frozen ledger row to its paper directory, split, domain and title.

    A sealed version publishes its own index only after the last stage has run, so the frozen
    ledger is what can tell the source which split a paper belongs to before then.
    """
    index: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(ledger_path) if ledger_path not in (None, "") else []:
        paper_id = str(row.get("paper_id") or "").strip()
        if not paper_id:
            continue
        index[safe_paper_directory(paper_id)] = {
            "paper_id": paper_id,
            "split": str(row.get("split") or "").strip(),
            "domain": str(row.get("domain") or "").strip(),
            "title": str(row.get("title") or "").strip(),
        }
    return index


def _group_identity(directory: Path) -> dict[str, Any]:
    """Return the paper identity a middleware directory records about itself.

    A version that has not published its index still names each paper inside its own graph and
    review, which is enough to key the source by the real paper id. No split is stated there, so
    a reader still needs the ledger or the index to place the paper.
    """
    for name in ("scientific_graph.json", "paper_review.json"):
        document = _read_json(directory / name, None)
        if not isinstance(document, dict):
            continue
        provenance = document.get("graph_provenance") if isinstance(document.get("graph_provenance"), dict) else {}
        paper_id = str(document.get("paper_id") or provenance.get("paper_id") or "").strip()
        if paper_id:
            return {"paper_id": paper_id, "split": "", "domain": str(provenance.get("domain") or "").strip()}
    return {}


def _paper_directory_name(version: Path, paper_id: str) -> str:
    """Return the directory a version stored one paper under, as it named it itself."""
    name = safe_paper_directory(paper_id)
    if (version / GRAPH_DIRECTORY / name).is_dir():
        return name
    matches = [path.name for path in sorted((version / GRAPH_DIRECTORY).glob(f"{name}*")) if path.is_dir()]
    return matches[0] if len(matches) == 1 else name


def _review_for(directory: Path, reviews: dict[str, dict[str, Any]], paper_id: str) -> dict[str, Any]:
    stored = _read_json(directory / REVIEW_FILE, None)
    if isinstance(stored, dict) and stored:
        return stored
    return reviews.get(paper_id) or {}


def build_corruption_source(
    version_dir: str | Path,
    out_dir: str | Path,
    *,
    snapshot: str | Path | None = None,
    split_ledger: str | Path | None = None,
    paper_ids: list[str] | None = None,
    limit: int | None = None,
    label: str = DEFAULT_LABEL,
) -> dict[str, Any]:
    """Write the corruption source of one middleware version without calling a provider."""
    version = Path(version_dir)
    if not version.is_dir():
        raise FileNotFoundError(f"middleware version directory does not exist: {version}")
    graph_root = version / GRAPH_DIRECTORY
    if not graph_root.is_dir():
        raise FileNotFoundError(f"middleware version has no {GRAPH_DIRECTORY} directory: {version}")
    if label not in ELIGIBLE_LABELS:
        raise ValueError(f"the corruption chain reads reviewed-good figures; {label!r} is not one of {sorted(ELIGIBLE_LABELS)}")
    available = {path.name: path for path in sorted(graph_root.iterdir()) if path.is_dir()}
    index = _version_index(version)
    splits = _split_ledger_index(split_ledger)
    reviews = version_reviews(version)
    requested = [str(item).strip() for item in (paper_ids or []) if str(item).strip()]
    names = [_paper_directory_name(version, item) for item in requested] if requested else sorted(available)
    if limit is not None:
        names = names[: max(0, int(limit))]
    if not names:
        raise ValueError(f"the version holds no paper directory to adapt: {graph_root}")
    papers = snapshot_paper_index(snapshot) if snapshot not in (None, "") else {}
    root = Path(out_dir).resolve()
    figure_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for name in names:
        directory = available.get(name)
        if directory is None:
            failures.append({"paper": name, "error": "the version has no graph directory for this paper"})
            continue
        identity = splits.get(name) or index.get(name) or _group_identity(directory)
        paper_id = str(identity.get("paper_id") or "").strip()
        split = str(identity.get("split") or "").strip()
        if not paper_id:
            failures.append({"paper": name, "error": "the middleware directory names no paper identity"})
            continue
        if not split:
            failures.append({"paper": name, "error": "no split assignment for this paper; pass --split-ledger or read a version that published its index"})
            continue
        document = _read_json(directory / "paper_method_inventory.json", None) or _read_json(directory / "figure_specification.json", None)
        if not isinstance(document, dict) or not document:
            failures.append({"paper": name, "error": "the middleware directory states no method inventory"})
            continue
        figures = _figure_files(version / FIGURE_DIRECTORY / name)
        if not figures:
            failures.append({"paper": name, "error": "the middleware rendered no reference figure for this paper"})
            continue
        review = _review_for(directory, reviews, paper_id)
        contract = corruption_contract(document, review, paper_id=name)
        ledger_row = papers.get(paper_id) if isinstance(papers.get(paper_id), dict) else {}
        identity_block = review.get("paper_identity") if isinstance(review.get("paper_identity"), dict) else {}
        title = str(identity.get("title") or ledger_row.get("title") or identity_block.get("title") or "").strip()
        abstract = str(ledger_row.get("abstract") or "").strip()
        write_json(root / CONTRACTS_DIRECTORY / f"{name}.json", contract)
        write_json(root / RAW_DIRECTORY / f"{name}.json", {"paper_id": name, "source_paper_id": paper_id, "title": title, "abstract": abstract})
        figure_rows.append({
            "figure_id": f"{name}__reference",
            "paper_id": name,
            "source_paper_id": paper_id,
            "figure_path": str(figures[0].resolve()),
            "image_sha256": sha256_file(figures[0]),
            "split": split,
            "final_label": label,
            "domain": str(identity.get("domain") or ""),
            "candidate_source": "middleware_reference_figure",
            "alternate_figures": [str(path.name) for path in figures[1:]],
        })
        records.append({
            "paper": name,
            "paper_id": paper_id,
            "split": split,
            "entities": len(contract["entities"]),
            "relations": len(contract["relations"]),
            "training_steps": len(contract["training_flow"]),
            "inference_steps": len(contract["inference_flow"]),
            "evidence_spans": len(contract["evidence_spans"]),
            "alternate_figures": max(0, len(figures) - 1),
            "title_source": "ledger" if identity.get("title") else "snapshot" if ledger_row.get("title") else "review" if title else "absent",
        })
    _write_jsonl(root / FIGURES_FILE, figure_rows)
    manifest = {
        "contract": CORRUPTION_SOURCE_CONTRACT,
        "version": CORRUPTION_SOURCE_VERSION,
        "summary": "Reviewed-good reference figures and their D1 contracts, read from one middleware version.",
        "ok": bool(figure_rows) and not failures,
        "source_version": str(version.resolve()),
        "snapshot": str(snapshot) if snapshot not in (None, "") else "",
        "split_ledger": str(split_ledger) if split_ledger not in (None, "") else "",
        "out_dir": str(root),
        "label": label,
        "papers_requested": len(names),
        "papers": len(records),
        "figures": len(figure_rows),
        "failed": len(failures),
        "splits": {split: sum(row["split"] == split for row in figure_rows) for split in SPLITS},
        "domains": _distribution(row["domain"] or "unknown" for row in figure_rows),
        "failures": failures[:40],
        "records": records,
    }
    write_json(root / SOURCE_MANIFEST_FILE, manifest)
    return manifest


def _distribution(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def ocr_function(engine: str) -> Callable[[str | Path, str], list[dict[str, Any]]]:
    """Return one OCR backend by name, imported lazily so an unused engine stays unloaded."""
    name = OCR_ENGINES.get(str(engine or "").strip().lower())
    if name is None:
        raise ValueError(f"unknown OCR engine {engine!r}; expected one of {sorted(OCR_ENGINES)}")
    from ..analysis.paper import ocr as ocr_module

    return getattr(ocr_module, name)


def _preflight_ocr(read: Callable[[str | Path, str], list[dict[str, Any]]], figures_path: Path, engine: str) -> dict[str, Any]:
    """Read one figure before the chain starts, so a missing engine fails loudly and early."""
    eligible = [row for row in _read_jsonl(figures_path) if str(row.get("final_label")) in ELIGIBLE_LABELS]
    if not eligible:
        raise ValueError(f"the source holds no reviewed-good figure to corrupt: {figures_path}")
    probe = Path(str(eligible[0]["figure_path"]))
    try:
        records = read(probe, "en_ch")
    except Exception as error:  # noqa: BLE001 - the engine's own failure is the report
        raise RuntimeError(f"{engine} OCR could not read {probe}: {type(error).__name__}: {error}") from error
    return {"engine": engine, "figure": str(probe), "records": len(records)}


def _mark_unaudited(examples_path: Path, review_method: str) -> int:
    """Record that a wave's examples came from a screen no reviewer stood behind.

    The example builder stamps an approved corruption as human audited, because an approval
    normally means the VLM-assisted audit accepted the region. A geometry screen approves by
    position alone, so every row is rewritten to say so before anything downstream reads it.
    """
    rows = _read_jsonl(examples_path)
    stamped: list[dict[str, Any]] = []
    for row in rows:
        provenance = dict(row.get("provenance") or {})
        provenance["human_audited"] = False
        provenance["review_method"] = review_method
        metadata = dict(row.get("judge_metadata") or {})
        metadata["human_audited"] = False
        stamped.append({**row, "provenance": provenance, "judge_metadata": metadata})
    _write_jsonl(examples_path, stamped)
    return len(stamped)


def run_local_corruption_wave(
    source_dir: str | Path,
    out_dir: str | Path,
    *,
    per_type: int = 100,
    review: str = "vlm",
    ocr_engine: str = "rapidocr",
    workers: int = 3,
    batch_size: int = 5,
    model: str | None = None,
    timeout: int = 180,
    retries: int = 1,
) -> dict[str, Any]:
    """Run the deterministic corruption chain over one prepared source into JudgeExamples.

    Every stage is deterministic for a fixed source, `per_type` and review mode, and each
    stage writes its own artifact, so a repeated run rebuilds the same regions and pairs the
    same regions again. The review stage resumes from its own output file. `workers` only sets
    how many OCR reads run ahead in parallel, so it changes the wall clock and nothing else.
    """
    if review not in REVIEW_MODES:
        raise ValueError(f"unknown review mode {review!r}; expected one of {list(REVIEW_MODES)}")
    source = Path(source_dir)
    figures = source / FIGURES_FILE
    contracts = source / CONTRACTS_DIRECTORY
    if not figures.is_file():
        raise FileNotFoundError(f"the corruption source has no figure manifest: {figures}")
    if not contracts.is_dir():
        raise FileNotFoundError(f"the corruption source has no contract directory: {contracts}")
    root = Path(out_dir).resolve()
    run = ocr_function(ocr_engine)
    stages: dict[str, Any] = {"ocr": _preflight_ocr(run, figures, ocr_engine)}
    candidate_root = root / "candidates"
    stages["candidates"] = generate_region_candidates(figures, contracts, candidate_root / "region_candidates.jsonl", per_type=per_type, ocr_function=run, workers=workers)
    if review == "vlm":
        stages["review"] = review_local_regions(
            candidate_root / "region_candidates.jsonl",
            contracts,
            candidate_root / "region_reviews.jsonl",
            candidate_root / "overlays",
            workers=workers,
            batch_size=batch_size,
            model=model,
            timeout=timeout,
            retries=retries,
        )
    else:
        stages["review"] = screen_regions_geometrically(candidate_root / "region_candidates.jsonl", candidate_root / "region_reviews.jsonl")
    corrupted = root / "corrupted"
    stages["corruptions"] = generate_approved_corruptions(candidate_root / "region_reviews.jsonl", corrupted / "images", corrupted / "corruption_results.jsonl", per_type=per_type)
    stages["examples"] = build_local_corrupted_examples(corrupted / "corruption_results.jsonl", contracts, source / RAW_DIRECTORY, root / "examples")
    examples_path = root / "examples" / "local_corrupted_examples.jsonl"
    stages["examples"]["human_audited"] = review == "vlm"
    if review != "vlm":
        _mark_unaudited(examples_path, review)
    report = {
        "contract": CORRUPTION_SOURCE_CONTRACT,
        "version": CORRUPTION_SOURCE_VERSION,
        "summary": "Deterministic controlled corruptions compiled into JudgeExamples.",
        "source_dir": str(source.resolve()),
        "out_dir": str(root),
        "review": review,
        "ocr_engine": ocr_engine,
        "per_type": int(per_type),
        "examples_path": str((root / "examples" / "local_corrupted_examples.jsonl")),
        "stages": stages,
        "ok": bool(stages["examples"]["examples"]) and not stages["examples"]["errors"],
    }
    write_json(root / WAVE_REPORT_FILE, report)
    return report


__all__ = [
    "CORRUPTION_SOURCE_CONTRACT",
    "CORRUPTION_SOURCE_VERSION",
    "SOURCE_MANIFEST_FILE",
    "WAVE_REPORT_FILE",
    "FIGURES_FILE",
    "CONTRACTS_DIRECTORY",
    "RAW_DIRECTORY",
    "REVIEW_MODES",
    "OCR_ENGINES",
    "build_corruption_source",
    "corruption_contract",
    "ocr_function",
    "run_local_corruption_wave",
]
