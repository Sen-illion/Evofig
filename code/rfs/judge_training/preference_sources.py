"""Read-only adapters for the three Judge preference-pair sources.

Each port wraps artifacts that another line produces. A port validates its inputs
when it opens and raises :class:`SourceUnavailable` when a required path is absent:
nothing here degrades into a substitute source or an empty dataset.

* :class:`MiddlewareSource` - a middleware version directory: ``graphs/<paper>/*.json``
  plus the version level ``graph_index.jsonl`` and ``subgraph_index.jsonl``. This is
  the paper-side ground truth contract and the RAG subgraph material. Its
  ``method_inventory`` hands over the list of concepts the paper itself states, with the
  figure's selection keys stripped, never the middleware's figure content plan.
* :class:`GenerationRunSource` - a ``paper-evolve-editable`` run directory:
  ``scientific_graph.json``, ``round_XX/candidate_XX/``, ``editable_composition.pptx``
  and the ``evolution_*.json(l)`` ledgers. This is the "truth vs generated candidate"
  material, which still needs the generation line to publish run artifacts.
* :class:`CorruptionSource` - reviewed framework figures plus their structured paper
  contracts, which the controlled corruption planner turns into injected defects.

Every port also writes a manifest carrying the source paths, artifact hashes, the
domain distribution and the defect distribution, so a dataset build stays auditable.
Only the corruption source carries defects; the other two report an empty defect
distribution rather than inventing one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .corruption_generator import plan_corruptions
from .judge_schema import DEFECT_LABELS, GATE_DIAGNOSIS_TYPES
from .method_inventory import method_inventory_view, read_method_inventory


SOURCE_CONTRACT = "rfs.judge_preference_source"
SOURCE_VERSION = 1

MIDDLEWARE_REQUIRED = ("graph_index.jsonl", "subgraph_index.jsonl")
GENERATION_REQUIRED = ("scientific_graph.json", "evolution_rounds.json", "evolution_events.jsonl")
LAYER_FILES = {"scientific": "scientific_graph.json", "visual": "visual_graph.json"}
CORRUPTION_REQUIRED = ("figures.jsonl",)
RUN_DELIVERABLES = ("editable_composition.pptx", "figure_program.json")
CANDIDATE_PREVIEWS = ("final_draft_preview.png", "draft_preview.png")

#: Folder names a run's identity search must skip: they hold other folders, not a paper.
IDENTITY_DIRECTORY_STOPWORDS = frozenset({"runs", "graphs", "figures", "output", "outputs", "data", "paperimage", "tmp"})
#: The batch runner nests a run one level down, so an attempt folder names no paper.
ATTEMPT_DIRECTORY_PREFIX = "attempt_"


class SourceUnavailable(RuntimeError):
    """Raised when a preference-pair source is missing artifacts it must have."""


def run_directory_candidates(run_root: str | Path, *, depth: int = 3) -> list[str]:
    """Return the directory names of a run that may carry the paper's real identifier.

    The generation line does not publish the identifier it drew: a run records the kind
    of source (`arxiv`) while the id only survives in the path the batch runner laid
    out, such as `runs/<paper id>/attempt_01`. Names come back nearest first so the
    narrowest match wins, and container or attempt folders are skipped because they name
    no paper.
    """

    candidates: list[str] = []
    path = Path(run_root)
    for _ in range(max(1, int(depth))):
        name = str(path.name or "")
        if name and name not in IDENTITY_DIRECTORY_STOPWORDS and not name.startswith(ATTEMPT_DIRECTORY_PREFIX):
            candidates.append(name)
        parent = path.parent
        if parent == path:
            break
        path = parent
    return candidates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _require(root: Path, names: Iterable[str], source: str) -> None:
    missing = [str(root / name) for name in names if not (root / name).exists()]
    if missing:
        raise SourceUnavailable(f"{source} is missing required artifacts: " + ", ".join(sorted(missing)))


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown").strip().casefold() or "unknown"
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def defect_distribution(kinds: Iterable[str]) -> dict[str, dict[str, int]]:
    """Split injected defect kinds into the gating three and the auxiliary rest."""

    gate: dict[str, int] = {}
    auxiliary: dict[str, int] = {}
    unsupported: dict[str, int] = {}
    for kind in kinds:
        entry = DEFECT_LABELS.get(str(kind))
        if entry is None:
            unsupported[str(kind)] = unsupported.get(str(kind), 0) + 1
            continue
        bucket = gate if entry["diagnosis_type"] in GATE_DIAGNOSIS_TYPES else auxiliary
        key = str(entry["diagnosis_type"])
        bucket[key] = bucket.get(key, 0) + 1
    return {"gate": dict(sorted(gate.items())), "auxiliary": dict(sorted(auxiliary.items())), "unsupported": dict(sorted(unsupported.items()))}


def _manifest(source: str, root: Path, *, artifacts: dict[str, str], counts: dict[str, Any], domains: dict[str, int], defects: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    return {
        "contract": SOURCE_CONTRACT,
        "version": SOURCE_VERSION,
        "source": source,
        "root": str(root),
        "artifacts": dict(sorted(artifacts.items())),
        "counts": dict(sorted(counts.items())),
        "domain_distribution": dict(sorted(domains.items())),
        "defect_distribution": defects,
        "warnings": list(warnings),
        "ok": not warnings,
    }


def _hashed(paths: Iterable[Path]) -> dict[str, str]:
    return {path.name: _sha256(path) for path in paths if path.is_file()}


@dataclass(frozen=True)
class MiddlewareSource:
    """A middleware version directory: paper contracts plus the retrieval index."""

    root: Path
    _index_cache: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        _require(self.root, MIDDLEWARE_REQUIRED, "middleware version")
        if not (self.root / "graphs").is_dir():
            raise SourceUnavailable(f"middleware version has no graphs directory: {self.root / 'graphs'}")

    @property
    def graphs_dir(self) -> Path:
        return self.root / "graphs"

    def index_records(self) -> list[dict[str, Any]]:
        return _jsonl(self.root / "graph_index.jsonl")

    def subgraph_locators(self) -> list[dict[str, Any]]:
        return _jsonl(self.root / "subgraph_index.jsonl")

    def paper_folders(self) -> list[str]:
        return sorted(path.name for path in self.graphs_dir.iterdir() if path.is_dir())

    def indexed_papers(self) -> dict[str, str]:
        """Map every indexed paper id to the folder its layers are stored in."""

        cached = self._index_cache.get("indexed_papers")
        if cached is None:
            cached = {}
            for record in self.index_records():
                source_graph = str((record.get("metadata") or {}).get("source_graph") or "")
                parts = Path(source_graph).parts
                paper_id = str(record.get("paper_id") or "")
                if paper_id and len(parts) >= 3 and parts[0] == "graphs":
                    cached.setdefault(paper_id, parts[1])
            self._index_cache["indexed_papers"] = cached
        return dict(cached)

    def index_maps(self) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
        """Return the index keyed by paper id and by folder, both case insensitive.

        A run may name a paper the way its own path was written rather than the way the
        index spells it, so both keys are folded and the canonical pair comes back with
        the match instead of the spelling that happened to hit.
        """

        cached = self._index_cache.get("index_maps")
        if cached is None:
            by_id: dict[str, tuple[str, str]] = {}
            by_folder: dict[str, tuple[str, str]] = {}
            for paper_id, folder in sorted(self.indexed_papers().items()):
                by_id.setdefault(paper_id.casefold(), (paper_id, folder))
                by_folder.setdefault(folder.casefold(), (paper_id, folder))
            cached = (by_id, by_folder)
            self._index_cache["index_maps"] = cached
        return cached

    def folder_for_run(self, *, paper_id: str = "", run_directories: Iterable[str] = ()) -> tuple[str, str, str]:
        """Resolve which indexed paper a generation run drew, and the route that matched.

        The generation line records `paper_id` as the kind of source it drew - a real
        run holds `arxiv` or `official` - while the identifier itself survives only in
        the run's own directory name, so the recorded id is tried first and the run
        directories after it, against both the indexed ids and the folder names (the
        sanitized ids). Nothing is guessed: no match is a failure that lists every
        candidate that was tried.
        """

        by_id, by_folder = self.index_maps()
        tried: list[str] = []
        routes: list[tuple[str, str]] = [(str(paper_id or "").strip(), "recorded_paper_id")]
        routes.extend((str(name).strip(), "run_directory") for name in run_directories if str(name or "").strip())
        for candidate, route in routes:
            if not candidate:
                continue
            tried.append(f"{route}={candidate}")
            matched = by_id.get(candidate.casefold()) or by_folder.get(candidate.casefold())
            if matched:
                return matched[0], matched[1], route
        raise SourceUnavailable(f"the middleware version has no graph folder for this run; tried {', '.join(tried)}")

    def paper_dir(self, folder: str) -> Path:
        target = self.graphs_dir / str(folder)
        if not target.is_dir():
            raise SourceUnavailable(f"middleware version has no graph folder {folder}: {target}")
        return target

    def layer_graph(self, folder: str, layer: str = "scientific") -> dict[str, Any]:
        name = LAYER_FILES.get(str(layer), f"{layer}_graph.json")
        path = self.paper_dir(folder) / name
        if not path.is_file():
            raise SourceUnavailable(f"middleware folder {folder} has no {layer} layer: {path}")
        return _json(path)

    def method_inventory(self, folder: str) -> dict[str, Any]:
        """Return the paper's own method list, as a Judge may read it.

        Only the read half is handed over: the concepts the paper states, their relations
        and the statements behind them, with the figure's selection keys stripped
        (:mod:`.method_inventory`). A folder that states no method list at all yields an
        empty mapping rather than a failure, because the caller has a scientific graph
        summary to fall back on and an absent list is not an absent paper.
        """

        return method_inventory_view(read_method_inventory(self.paper_dir(folder)))

    def truth_figure(self, folder: str) -> Path:
        """Return the one rendered reference figure of a paper folder.

        The reference figure is what the generation line is compared against, so an absent
        or ambiguous figure is a failure: guessing between two renders would silently
        change the meaning of every pair built from it.
        """

        directory = self.root / "figures" / str(folder)
        if not directory.is_dir():
            raise SourceUnavailable(f"middleware version has no rendered figures for {folder}: {directory}")
        rendered = sorted(path for path in directory.glob("*.png") if path.is_file())
        if len(rendered) != 1:
            raise SourceUnavailable(f"middleware folder {folder} must hold exactly one rendered figure, found {len(rendered)}: {directory}")
        return rendered[0]

    def manifest(self) -> dict[str, Any]:
        records = self.index_records()
        locators = self.subgraph_locators()
        folders = self.paper_folders()
        indexed_folders = set(self.indexed_papers().values())
        return _manifest(
            "middleware_version",
            self.root,
            artifacts=_hashed([self.root / name for name in MIDDLEWARE_REQUIRED] + [self.root / "coverage_report.json", self.root / "graph_manifest.json"]),
            counts={"paper_folders": len(folders), "indexed_papers": len(self.indexed_papers()), "index_records": len(records), "subgraph_locators": len(locators)},
            domains=_distribution(record.get("domain") for record in records),
            defects={},
            warnings=[f"graph folder without an index record: {folder}" for folder in folders if folder not in indexed_folders],
        )


@dataclass(frozen=True)
class GenerationRunSource:
    """A generation run directory: candidates, deliverables and evolution ledgers."""

    root: Path

    def __post_init__(self) -> None:
        _require(self.root, GENERATION_REQUIRED, "generation run")
        if not self.round_dirs():
            raise SourceUnavailable(f"generation run has no round_XX/candidate_XX directories: {self.root}")

    def round_dirs(self) -> list[Path]:
        return sorted(path for path in self.root.glob("round_*") if path.is_dir() and list(path.glob("candidate_*")))

    def candidate_dirs(self, round_index: int) -> list[Path]:
        return sorted(path for path in (self.root / f"round_{int(round_index):02d}").glob("candidate_*") if path.is_dir())

    def candidate_preview(self, round_index: int, variant: int) -> Path:
        folder = self.root / f"round_{int(round_index):02d}" / f"candidate_{int(variant):02d}"
        for name in CANDIDATE_PREVIEWS:
            if (folder / name).is_file():
                return folder / name
        raise SourceUnavailable(f"candidate has no rendered preview: {folder}")

    def deliverables(self) -> dict[str, str]:
        _require(self.root, RUN_DELIVERABLES, "generation run deliverables")
        return {name: str(self.root / name) for name in RUN_DELIVERABLES}

    def evolution_events(self) -> list[dict[str, Any]]:
        return _jsonl(self.root / "evolution_events.jsonl")

    def evolution_rounds(self) -> list[dict[str, Any]]:
        rounds = _json(self.root / "evolution_rounds.json").get("rounds")
        return [row for row in rounds if isinstance(row, dict)] if isinstance(rounds, list) else []

    def manifest(self) -> dict[str, Any]:
        rounds = self.evolution_rounds()
        candidates = [path for round_dir in self.round_dirs() for path in sorted(round_dir.glob("candidate_*")) if path.is_dir()]
        warnings = [f"candidate has no rendered preview: {path}" for path in candidates if not any((path / name).is_file() for name in CANDIDATE_PREVIEWS)]
        return _manifest(
            "generation_run",
            self.root,
            artifacts=_hashed([self.root / name for name in GENERATION_REQUIRED] + [self.root / name for name in RUN_DELIVERABLES]),
            counts={"rounds": len(self.round_dirs()), "evolution_rounds": len(rounds), "candidates": len(candidates), "evolution_events": len(self.evolution_events())},
            domains=_distribution(row.get("domain") for row in rounds),
            defects={},
            warnings=warnings,
        )


@dataclass(frozen=True)
class CorruptionSource:
    """Reviewed framework figures plus the structured contracts they were judged against."""

    figures_path: Path
    structured_dir: Path
    domain_index: Path | None = None

    def __post_init__(self) -> None:
        if not self.figures_path.is_file():
            raise SourceUnavailable(f"framework figure manifest is missing: {self.figures_path}")
        if not self.structured_dir.is_dir():
            raise SourceUnavailable(f"structured contract directory is missing: {self.structured_dir}")
        if self.domain_index is not None and not Path(self.domain_index).is_file():
            raise SourceUnavailable(f"domain index is missing: {self.domain_index}")

    def figures(self) -> list[dict[str, Any]]:
        return _jsonl(self.figures_path)

    def contract(self, paper_id: str) -> dict[str, Any]:
        path = self.structured_dir / f"{paper_id}.json"
        if not path.is_file():
            raise SourceUnavailable(f"structured contract is missing for {paper_id}: {path}")
        return _json(path)

    def domains(self) -> dict[str, str]:
        if self.domain_index is None:
            return {}
        return {str(row.get("paper_id") or ""): str(row.get("domain") or "unknown") for row in _jsonl(Path(self.domain_index))}

    def planned_corruptions(self) -> list[dict[str, Any]]:
        planned: list[dict[str, Any]] = []
        for figure in self.figures():
            paper_id = str(figure.get("paper_id") or figure.get("arxiv_id") or "")
            planned.extend(plan_corruptions(paper_id, str(figure.get("figure_id") or ""), str(figure.get("figure_path") or ""), self.contract(paper_id), str(figure.get("split") or "")))
        return planned

    def manifest(self) -> dict[str, Any]:
        figures = self.figures()
        planned = self.planned_corruptions()
        domains = self.domains()
        return _manifest(
            "controlled_corruption",
            self.figures_path.parent,
            artifacts=_hashed([self.figures_path]),
            counts={"figures": len(figures), "papers": len({str(row.get("paper_id") or "") for row in figures}), "planned_corruptions": len(planned), "contracts": len({str(row.get("paper_id") or "") for row in figures})},
            domains=_distribution(domains.get(str(row.get("paper_id") or ""), "unknown") for row in figures),
            defects=defect_distribution(row.get("corruption_type") for row in planned),
            warnings=[] if domains else ["no domain index supplied; paper domains are reported as unknown"],
        )


def open_middleware_source(version_dir: str | Path) -> MiddlewareSource:
    """Open a middleware version directory, failing when the index is not published."""

    return MiddlewareSource(Path(version_dir))


def open_generation_source(run_dir: str | Path) -> GenerationRunSource:
    """Open a generation run directory, failing when the run is not published."""

    return GenerationRunSource(Path(run_dir))


def open_corruption_source(figures_path: str | Path, structured_dir: str | Path, *, domain_index: str | Path | None = None) -> CorruptionSource:
    """Open the controlled corruption source over reviewed figures and contracts."""

    return CorruptionSource(Path(figures_path), Path(structured_dir), Path(domain_index) if domain_index else None)


def open_source(source: str, *, root: str | Path | None = None, figures: str | Path | None = None, structured: str | Path | None = None, domain_index: str | Path | None = None):
    """Open one of the three ports by name, refusing to guess a missing path."""

    key = str(source or "").strip().casefold()
    if key in {"middleware", "generation"}:
        if not root:
            raise SourceUnavailable(f"the {key} source needs --root")
        return MiddlewareSource(Path(root)) if key == "middleware" else GenerationRunSource(Path(root))
    if key == "corruption":
        if not figures or not structured:
            raise SourceUnavailable("the corruption source needs --figures and --structured")
        return open_corruption_source(figures, structured, domain_index=domain_index)
    raise SourceUnavailable(f"unknown preference source: {source!r}")


def write_source_manifest(source: str, out_path: str | Path, *, root: str | Path | None = None, figures: str | Path | None = None, structured: str | Path | None = None, domain_index: str | Path | None = None) -> dict[str, Any]:
    """Open one port, write its manifest and return it together with its hash.

    The manifest is read from the source artifacts; nothing is invented and a source
    that cannot be opened raises :class:`SourceUnavailable` instead of writing an
    empty manifest.
    """

    manifest = open_source(source, root=root, figures=figures, structured=structured, domain_index=domain_index).manifest()
    body = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = {**manifest, "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
