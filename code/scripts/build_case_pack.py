#!/usr/bin/env python3
"""Collect, for one paper, everything a case study needs to show and to prove.

A case study is only worth reading if a reader can check it, so this script gathers the six
pieces a slot needs -- the source paper, our delivered figure and the process files behind it,
the object audit, the same-paper output of the systems we compare against, and the judge and
repair records -- and writes them next to a ``case.json`` that names every path and its
sha256.

It resolves what exists and records what does not in ``gaps``. It never guesses a substitute:
a baseline whose output cannot be located is a gap, not a blank space in a figure. The author's
original figure is usually not on disk for this corpus; the pack carries the source PDF and
says so, so cropping it stays a deliberate step rather than an assumed one.

Usage (on the host that holds the run trees):

    python scripts/build_case_pack.py \
        --case-list case_studies/case_list.json \
        --run-root <DATA_ROOT>/abl/out/ours_own500 \
        --out-dir <DATA_ROOT>/abl/case_study/packs
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

DOMAINS = ("agent", "biology", "cv", "materials", "rag")

#: Files that carry the claim, the process that produced it, and the record of what was tried.
OURS_FILES = (
    "editable_composition.pptx",
    "preview.png",
    "preview_render_report.json",
    "figure_program.json",
    "editable_graph.json",
    "editable_graph_report.json",
    "graph_bindings.json",
    "visual_graph.json",
    "scientific_graph.json",
    "figure_content_plan.json",
    "paper_method_inventory.json",
    "figure_specification.json",
    "icon_generation_plan.json",
    "global_structure_plan.json",
    "segmentation_plan.json",
    "grounding_gate.json",
    "final_quality_gate.json",
    "judgement.json",
    "evolution_rounds.json",
    "evolution_events.jsonl",
    "evolution_state.json",
    "run_report.json",
    "paper_evolve_editable_report.json",
    "cost_report.json",
)

OURS_NESTED = ("inputs/paper.pdf",)

#: Where the systems in the main table keep their own-set output: one folder per paper under each
#: arm's ``own`` root. A template may list alternatives with ``|``; the first one that exists wins.
#: ``{pid2}`` doubles the first underscore, which one arm's writer did to keep ids unique.
DEFAULT_BASELINES = (
    ("paper2sysarch", "<DATA_ROOT>/efb_main/paper2sysarch/own/{pid}/figure.png"),
    ("livefigure", "<DATA_ROOT>/efb_main/livefigure/own/{pid}/preview.png"),
    (
        "autofigure",
        "<DATA_ROOT>/efb_main/autofigure/own/{pid}/figure.png"
        "|<DATA_ROOT>/efb_main/autofigure/own/{pid}/figure_final.png",
    ),
    ("automatikz", "<DATA_ROOT>/efb_main/automatikz/own/{pid}/figure.png"),
    ("tikzero", "<DATA_ROOT>/efb_main/tikzero/own/{pid}/figure.png"),
    ("image", "<DATA_ROOT>/efb_main/image/own/{pid}/figure.png"),
    ("image_alt", "<DATA_ROOT>/efb_runs/own500_gemini_flash_image/{pid2}/preview.png"),
)


def load_json(path: Path):
    """Read JSON that may sit behind log lines a library wrote to the same file."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            return json.loads(text[index:])
        except json.JSONDecodeError:
            continue
    return None


def dump_json(path: Path, payload) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    try:
        from rfs.utils import write_text

        write_text(path, text)
    except Exception:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_variants(paper_id: str):
    """The id spellings the run trees use for the same paper."""
    parts = paper_id.split("_", 1)
    doubled = parts[0] + "__" + parts[1] if len(parts) == 2 else paper_id
    return [paper_id, doubled]


#: The turbo ruler: the three arms that were re-run with ``z-image-turbo``, so a comparison
#: holds the image model fixed instead of mixing gpt-image-2, a flash image model and turbo.
#: Those runs sit beside the main rows (``<system>_turbo/own``), not under them.
TURBO_BASELINES = (
    ("paper2sysarch", "<DATA_ROOT>/efb_main/paper2sysarch_turbo/own/{pid}/figure.png"),
    (
        "livefigure",
        "<DATA_ROOT>/efb_main/livefigure_turbo/own/{pid}/preview.png"
        "|<DATA_ROOT>/efb_main/livefigure_turbo/own/{pid}/delivery.pptx",
    ),
    ("image", "<DATA_ROOT>/efb_main/image_turbo/own/{pid}/figure.png"),
)

#: Arms the turbo ruler cannot cover, and why. Their output is compared as it stands and the
#: pack says so, rather than dropping them or pretending they share the ruler.
NO_TURBO_VARIANT = {
    "autofigure": "no turbo run of this arm exists; its own image backend is not swapped here",
    "automatikz": "TikZ program: the pipeline has no image model to align",
    "tikzero": "TikZ program: the pipeline has no image model to align",
}


def render_deck(deck: Path, target: Path):
    """Render slide one of a deck to PNG the way our own previews are made: PDF, then PyMuPDF.

    The conversion runs in a private temporary directory: LibreOffice writes a whole user
    profile next to its output, and a pack should carry evidence, not scratch files.
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return "no LibreOffice on this host"
    try:
        import fitz
    except Exception:
        return "PyMuPDF is not importable here"
    workdir = Path(tempfile.mkdtemp(prefix="case-pack-render-"))
    subprocess.run(
        [
            soffice,
            "--headless",
            "--norestore",
            "-env:UserInstallation=file://" + str((workdir / "lo-profile").resolve()),
            "--convert-to",
            "pdf",
            "--outdir",
            str(workdir),
            str(deck),
        ],
        capture_output=True,
        timeout=600,
        check=False,
    )
    pdf = workdir / (deck.stem + ".pdf")
    if not pdf.is_file():
        return "LibreOffice produced no PDF"
    document = fitz.open(str(pdf))
    target.parent.mkdir(parents=True, exist_ok=True)
    document[0].get_pixmap(dpi=150).save(str(target))
    document.close()
    return None


def resolve_baseline(template: str, paper_id: str):
    """First candidate that exists, over both id spellings and every name in the template."""
    for alternative in template.split("|"):
        for variant in id_variants(paper_id):
            candidate = Path(alternative.format(pid=variant, pid2=variant))
            if candidate.is_file():
                return candidate
    return None


FRAMEWORK_HINTS = ("framework", "overview", "architecture", "pipeline", "overall", "workflow")


def render_caption_pages(pdf_path: Path, evidence_path: Path, pack_dir: Path, dpi: int = 200, max_pages: int = 12):
    """Render the pages that carry a figure caption, and say what each caption claims.

    The run never read the authors' figure, so the pack cannot claim one on its own: it hands
    back the pages and the caption text, and flags the captions that read like a framework
    figure. Which figure to crop stays a human decision, recorded in ``source`` when made.
    """
    evidence = load_json(evidence_path) or {}
    entries = evidence.get("evidence") if isinstance(evidence, dict) else evidence
    captions = [
        entry for entry in (entries or []) if isinstance(entry, dict) and entry.get("kind") == "caption"
    ]
    if not captions:
        return [], "source: no figure caption recorded, so no page was rendered for the author's figure"
    pages = []
    for entry in captions:
        page = entry.get("page")
        if isinstance(page, int) and page not in pages:
            pages.append(page)
    pages = pages[:max_pages]
    try:
        import fitz
    except Exception:
        return (
            [{"page": page} for page in pages],
            "source: PyMuPDF is not importable here; render the caption pages by hand",
        )
    document = fitz.open(str(pdf_path))
    rendered = []
    for number in pages:
        if not (1 <= number <= document.page_count):
            continue
        target = pack_dir / "source" / ("page_%04d.png" % number)
        target.parent.mkdir(parents=True, exist_ok=True)
        document[number - 1].get_pixmap(dpi=dpi).save(str(target))
        rendered.append(
            {
                "page": number,
                "image": str(target),
                "sha256": sha256_file(target),
                "captions": [
                    {
                        "bbox": entry.get("bbox"),
                        "text": (entry.get("text") or "")[:400],
                        "looks_like_framework": any(
                            hint in (entry.get("text") or "").lower() for hint in FRAMEWORK_HINTS
                        ),
                    }
                    for entry in captions
                    if entry.get("page") == number
                ],
            }
        )
    document.close()
    return rendered, None


def audit_row(run_dir: Path) -> dict:
    """Count what is actually in the delivered file, from the records written beside it."""
    report = load_json(run_dir / "editable_graph_report.json") or {}
    graph = load_json(run_dir / "editable_graph.json") or {}
    program = load_json(run_dir / "figure_program.json") or {}

    kinds = collections.Counter(
        str(node.get("kind")) for node in (graph.get("nodes") or []) if isinstance(node, dict)
    )
    slots = [slot for slot in (program.get("slots") or []) if isinstance(slot, dict)]
    raster_area = 0.0
    for slot in slots:
        box = slot.get("bbox_percent") or {}
        width, height = box.get("w"), box.get("h")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            raster_area += float(width) * float(height)
    canvas = program.get("canvas") or {}
    return {
        "native_object_count": report.get("object_count"),
        "connector_count": report.get("connector_count"),
        "graph_node_kinds": {key: kinds[key] for key in sorted(kinds)},
        "panel_count": len(program.get("panels") or []),
        "arrow_count": len(program.get("arrows") or []),
        "icon_slot_count": len(slots),
        "image_asset_count": len(program.get("assets") or []),
        "raster_area_ratio": round(min(raster_area, 1.0), 4),
        "canvas_inches": [canvas.get("width_in"), canvas.get("height_in")],
    }


def pack_one(paper_id, domain, run_root, out_dir, baselines, copy: bool, variant: str = "main") -> dict:
    run_dir = Path(run_root) / domain / paper_id
    pack_dir = Path(out_dir) / paper_id
    gaps = []
    if not run_dir.is_dir():
        return {"paper_id": paper_id, "domain": domain, "gaps": [f"run directory not found: {run_dir}"]}

    ours = {}
    for name in OURS_FILES + OURS_NESTED:
        source = run_dir / name
        if not source.is_file():
            continue
        entry = {"source": str(source), "sha256": sha256_file(source), "bytes": source.stat().st_size}
        if copy:
            target = pack_dir / "ours" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            entry["packed"] = str(target)
        ours[name] = entry
    for required in ("editable_composition.pptx", "preview.png", "figure_program.json"):
        if required not in ours:
            gaps.append(f"ours: missing {required}")

    baselines_found = {}
    for system, template, ruler in baselines:
        found = resolve_baseline(template, paper_id)
        if found is None:
            gaps.append(f"baseline {system}: no output found for this paper")
            baselines_found[system] = None
            continue
        entry = {
            "source": str(found),
            "sha256": sha256_file(found),
            "bytes": found.stat().st_size,
            "ruler": ruler,
        }
        if copy:
            target = pack_dir / "baselines" / system / found.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found, target)
            entry["packed"] = str(target)
        if found.suffix.lower() == ".pptx":
            rendered = pack_dir / "baselines" / system / (found.stem + "_slide1.png")
            note = render_deck(found, rendered)
            if note:
                gaps.append(f"baseline {system}: deck found but not rendered ({note})")
            else:
                entry["rendered"] = str(rendered)
                entry["rendered_sha256"] = sha256_file(rendered)
        baselines_found[system] = entry
    source_pdf = ours.get("inputs/paper.pdf", {}).get("source") or str(run_dir / "inputs" / "paper.pdf")
    if not Path(source_pdf).is_file():
        gaps.append("source: no paper PDF beside the run")
    figure_pages, render_note = render_caption_pages(
        Path(source_pdf), run_dir / "key_evidence.json", pack_dir
    )
    if render_note:
        gaps.append(render_note)

    pack = {
        "paper_id": paper_id,
        "domain": domain,
        "variant": variant,
        "run_dir": str(run_dir),
        "audit": audit_row(run_dir),
        "ours": ours,
        "baselines": baselines_found,
        "source": {
            "pdf": source_pdf,
            "figure_pages": figure_pages,
            "author_figure": None,
            "author_figure_note": (
                "not on disk for this corpus; pick the caption that is the framework figure from "
                "figure_pages, crop it from the rendered page, and record the crop here"
            ),
        },
        "gaps": gaps,
    }
    dump_json(pack_dir / "case.json", pack)
    return pack


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--case-list", help="a frozen case list; every pick is packed")
    parser.add_argument("--paper-id", help="pack one paper instead of a whole list")
    parser.add_argument("--domain", help="needed with --paper-id")
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="SYSTEM=PATH_TEMPLATE",
        help="override the default baseline templates (repeatable)",
    )
    parser.add_argument("--no-copy", action="store_true", help="record paths and hashes only")
    parser.add_argument(
        "--variant",
        default="main",
        choices=("main", "turbo"),
        help="main reads each arm's own output; turbo reads the z-image-turbo re-runs",
    )

    args = parser.parse_args()
    if args.baseline:
        baselines = tuple(
            (spec.split("=", 1)[0], spec.split("=", 1)[1], args.variant) for spec in args.baseline
        )
    elif args.variant == "turbo":
        main_rows = dict(DEFAULT_BASELINES)
        baselines = tuple(
            (system, template, "turbo (z-image-turbo)") for system, template in TURBO_BASELINES
        ) + tuple(
            (system, main_rows[system], "main row, no turbo variant: " + reason)
            for system, reason in sorted(NO_TURBO_VARIANT.items())
        )
    else:
        baselines = tuple(
            (system, template, "main row (each arm's own image model)")
            for system, template in DEFAULT_BASELINES
        )

    targets = []
    if args.case_list:
        listing = load_json(Path(args.case_list)) or {}
        for pick in listing.get("picks") or []:
            targets.append((pick["paper_id"], pick["domain"]))
    if args.paper_id:
        domain = args.domain
        if domain is None:
            for candidate in DOMAINS:
                if (Path(args.run_root) / candidate / args.paper_id).is_dir():
                    domain = candidate
                    break
        if domain is None:
            parser.error("--domain is required when the paper is not in a known domain folder")
        targets.append((args.paper_id, domain))
    if not targets:
        parser.error("nothing to pack: pass --case-list or --paper-id")

    summary = []
    for paper_id, domain in targets:
        pack = pack_one(
            paper_id, domain, args.run_root, args.out_dir, baselines, not args.no_copy, args.variant
        )
        summary.append(
            {
                "paper_id": paper_id,
                "domain": domain,
                "files": len(pack.get("ours") or {}),
                "baselines_found": sum(1 for value in (pack.get("baselines") or {}).values() if value),
                "gaps": len(pack.get("gaps") or []),
                "audit": pack.get("audit"),
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
