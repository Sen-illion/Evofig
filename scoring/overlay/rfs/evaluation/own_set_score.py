"""Score one own-set delivery on the four table-1 columns.

This module does not generate a figure and does not read a gold contract.
It scores a preview and a PPTX that are already on disk. An existing
``metrics.json`` from this scorer is left in place unless ``force`` is set.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from ..utils import write_json, write_text
from .livefigure_nine import score_nine
from .native_editability import editability_verdict, measure_native_editability
from .paper_figure_clip import score_paper_figure_clip
from .paper_quiz import score_paper_quiz

OWN_TABLE = "v1"
TORCH_THREADS_ENV_VAR = "RFS_TORCH_THREADS"
CGROUP_CPU_MAX_PATH = "/sys/fs/cgroup/cpu.max"
DEFAULT_MAX_TORCH_THREADS = 4


def _cgroup_cpu_quota(path: str | Path = CGROUP_CPU_MAX_PATH) -> int | None:
    """Whole cores this container's CFS quota allows, or None when it has none.

    ``cpu.max`` reads "<quota> <period>" while a quota is set and "max <period>" while
    the cgroup is unlimited. A fractional quota rounds up, because half a core still
    runs one thread and rounding down would leave the caller with nothing.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8").split()
    except OSError:
        return None
    if len(raw) < 2 or raw[0] == "max":
        return None
    try:
        quota, period = int(raw[0]), int(raw[1])
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return max(1, -(-quota // period))


def torch_thread_budget(
    *,
    override: int | str | None = None,
    cgroup_path: str | Path = CGROUP_CPU_MAX_PATH,
) -> int:
    """Threads this process may hand to torch.

    torch sizes its pool from the host's core count and never from the container's
    quota, so a container capped at half a core still spawns 64 threads that spin-wait
    against that cap. Measured on 2026-10-15 in the no-card container: the same 768^3
    matmul took 0.40 s on one thread and 79 s on 128, and one scored paper went from
    ~20 s to ~12 min. The budget is therefore the smaller of the quota and a small
    ceiling; ``RFS_TORCH_THREADS`` overrides both.
    """
    if override is None:
        override = os.environ.get(TORCH_THREADS_ENV_VAR, "").strip() or None
    if override is not None:
        try:
            return max(1, int(override))
        except (TypeError, ValueError):
            return 1
    budget = min(DEFAULT_MAX_TORCH_THREADS, max(1, os.cpu_count() or 1))
    quota = _cgroup_cpu_quota(cgroup_path)
    if quota is not None:
        budget = min(budget, quota)
    return max(1, budget)


METHOD_HEADINGS = {
    "method",
    "methodology",
    "approach",
    "proposed method",
    "methods",
    "materials and methods",
    "method and materials",
    "methods and materials",
    "materials & methods",
    "materials and methodology",
    "methods and methodology",
    "methods and models",
    "proposed approach",
    "proposed framework",
    "the proposed method",
    "our method",
    "our approach",
    "model",
    "framework",
}
SECTION_STOPS = {"experiment", "experiments", "results", "conclusion", "conclusions", "references"}
ABSTRACT_HEADINGS = {"abstract", "summary", "abstract and introduction"}
#: What ends an abstract: the same stops a method section uses, plus the headings a
#: paper prints between its abstract and its method, which would otherwise fold the
#: introduction's background into the skim.
ABSTRACT_STOPS = SECTION_STOPS | METHOD_HEADINGS | {"introduction", "related work", "background"}

#: A caption begins with the label the paper itself prints: "Figure 3:", "Fig. 2.",
#: "Table 4." The label is what makes a line a caption rather than a sentence that
#: happens to mention a figure.
CAPTION_PREFIX = re.compile(
    r"^\s*(?:fig(?:ure)?s?|tab(?:le)?)\s*\.?\s*[A-Za-z]?\s*\d+\s*[:.\u2014-]",
    re.IGNORECASE,
)
CAPTION_CHAR_LIMIT = 700
ABSTRACT_CHAR_LIMIT = 3000


def shallow_read_from_text(full: str, *, max_captions: int = 12) -> tuple[str, str | None]:
    """Return the abstract and the figure captions -- what a skim of a paper keeps.

    This is the reading a row is handed when the question is what the paper's own
    structure is worth: the abstract names the contribution, the captions name the
    stages the authors drew, and everything between them -- the method, the
    appendix, the experiment tables -- is left behind. It returns an empty string
    and a reason when the text yields neither, so a row that got nothing says so
    instead of being handed an empty read that looks like a thin paper.
    """

    lines = str(full or "").splitlines()
    abstract: list[str] = []
    start = None
    for index, line in enumerate(lines):
        if _heading(line) in ABSTRACT_HEADINGS:
            start = index + 1
            break
    if start is not None:
        for line in lines[start:]:
            heading = _heading(line)
            if abstract and heading in ABSTRACT_STOPS:
                break
            if CAPTION_PREFIX.match(line):
                break
            abstract.append(line)
    abstract_text = "\n".join(abstract).strip()[:ABSTRACT_CHAR_LIMIT]

    captions: list[str] = []
    current: list[str] = []
    for line in lines:
        if CAPTION_PREFIX.match(line):
            if current:
                captions.append(" ".join(current))
            current = [line.strip()]
            if len(captions) >= max_captions:
                break
            continue
        if current:
            if _heading(line) in SECTION_STOPS:
                captions.append(" ".join(current))
                current = []
                continue
            if line.strip():
                current.append(line.strip())
            elif len(" ".join(current)) > CAPTION_CHAR_LIMIT:
                captions.append(" ".join(current))
                current = []
    if current and len(captions) < max_captions:
        captions.append(" ".join(current))
    caption_text = "\n".join(item[:CAPTION_CHAR_LIMIT] for item in captions if item.strip())

    if not abstract_text and not caption_text:
        return "", "no abstract or figure captions in the paper text"
    parts = []
    if abstract_text:
        parts.append("ABSTRACT:\n" + abstract_text)
    if caption_text:
        parts.append("FIGURE CAPTIONS:\n" + caption_text)
    return "\n\n".join(parts), None


def extract_shallow_read(pdf_path: str | Path) -> tuple[str, str | None]:
    """Read a PDF and return its abstract and figure captions."""

    import fitz

    document = fitz.open(Path(pdf_path))
    try:
        full = "\n".join(page.get_text("text") for page in document)
    finally:
        document.close()
    return shallow_read_from_text(full)


def _feature_vector(output: Any) -> list[float]:
    tensor = output
    if getattr(output, "pooler_output", None) is not None:
        tensor = output.pooler_output
    elif getattr(output, "image_embeds", None) is not None:
        tensor = output.image_embeds
    elif getattr(output, "text_embeds", None) is not None:
        tensor = output.text_embeds
    elif isinstance(output, tuple):
        tensor = output[0]
    return [float(value) for value in tensor.detach().float().cpu().reshape(-1).tolist()]


class LocalAltCLIP:
    """CPU AltCLIP from a directory already on disk. This does not download weights."""

    model_name = "BAAI/AltCLIP"

    def __init__(self, model_dir: str | Path, *, threads: int | None = None) -> None:
        import torch
        from transformers import AltCLIPModel, AltCLIPProcessor

        directory = Path(model_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"AltCLIP directory is missing: {directory}")
        self._torch = torch
        self.threads = torch_thread_budget(override=threads)
        try:
            torch.set_num_threads(self.threads)
        except (RuntimeError, ValueError):
            pass
        self.processor = AltCLIPProcessor.from_pretrained(directory, local_files_only=True)
        self.model = AltCLIPModel.from_pretrained(directory, local_files_only=True)
        self.model.to("cpu")
        self.model.eval()

    def embed_image(self, path: Path) -> list[float]:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with self._torch.no_grad():
            output = self.model.get_image_features(**inputs)
        return _feature_vector(output)

    def embed_text(self, text: str) -> list[float]:
        inputs = self.processor(text=[str(text)[:2000]], return_tensors="pt", padding=True, truncation=True)
        with self._torch.no_grad():
            output = self.model.get_text_features(**inputs)
        return _feature_vector(output)


def safe_paper_id(paper_id: str) -> str:
    """Turn a paper id into a single path segment."""
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(paper_id))
    return cleaned[:120] or "paper"


def _heading(line: str) -> str:
    """Normalize a section heading, dropping a numeric or roman-numeral prefix."""
    stripped = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", str(line)).strip().lower()
    stripped = re.sub(r"^(?:[ivxlcdm]{1,6})[.)]\s+", "", stripped)
    return stripped.strip(".:").strip()


def method_section_from_text(full: str) -> tuple[str, str | None]:
    """Return the method section, or an empty string and a reason."""
    lines = str(full or "").splitlines()
    start = None
    for index, line in enumerate(lines):
        heading = _heading(line)
        if heading in METHOD_HEADINGS:
            start = index + 1
            break
    if start is None:
        return "", "missing method section"
    chunk: list[str] = []
    for line in lines[start:]:
        heading = _heading(line)
        if chunk and heading in SECTION_STOPS:
            break
        chunk.append(line)
    text = "\n".join(chunk).strip()
    if not text:
        return "", "missing method section"
    return text[:12000], None


def extract_method_section(pdf_path: str | Path) -> tuple[str, str | None]:
    """Read a PDF and return its method section."""
    import fitz

    document = fitz.open(Path(pdf_path))
    try:
        full = "\n".join(page.get_text("text") for page in document)
    finally:
        document.close()
    return method_section_from_text(full)


#: How much of a paper grounds the ruler when the method section cannot be found.
NINE_GROUNDING_LIMIT = 12000


def nine_grounding_text(text: str, pdf_path: str | Path | None) -> tuple[str, str]:
    """Return what the nine-item ruler grades against, and which read it came from.

    ``nine-v003`` refuses to grade the three fidelity items without the paper's
    own words, and a method section that cannot be located is a property of the
    paper, not of the ruler. The abstract is the next best statement of what the
    paper proposes; a paper whose abstract cannot be found either is handed its
    opening pages. The read comes back with the text, so a row that fell back
    says so instead of looking like a paper scored on its method section.
    """

    if str(text or "").strip():
        return str(text), ""
    if not pdf_path or not Path(pdf_path).is_file():
        return "", ""
    import fitz

    document = fitz.open(Path(pdf_path))
    try:
        full = "\n".join(page.get_text("text") for page in document)
    finally:
        document.close()
    skim, _ = shallow_read_from_text(full)
    if str(skim or "").strip():
        return skim[:NINE_GROUNDING_LIMIT], "abstract"
    body = str(full or "").strip()
    return (body[:NINE_GROUNDING_LIMIT], "body") if body else ("", "")


def _delivery_files(run_dir: Path) -> dict[str, Path]:
    return {
        "preview": run_dir / "preview.png",
        "figure": run_dir / "figure.png",
        "pptx": run_dir / "delivery.pptx",
        "tex": run_dir / "figure.tex",
    }


def _has_delivery(files: dict[str, Path]) -> bool:
    return any(path.is_file() for path in files.values())


def _preview(run_dir: Path, files: dict[str, Path]) -> tuple[Path, str | None]:
    if files["preview"].is_file():
        return files["preview"], None
    if files["figure"].is_file():
        return files["figure"], None
    if not files["pptx"].is_file():
        return files["preview"], "missing preview.png"
    try:
        from ..composition.pptx_render import render_pptx_page

        rendered = render_pptx_page(files["pptx"], files["preview"])
    except Exception as exc:
        return files["preview"], f"{type(exc).__name__}: {exc}"[:200]
    if not files["preview"].is_file():
        reason = str((rendered or {}).get("error") or (rendered or {}).get("reason") or "render produced no preview.png")
        return files["preview"], reason[:200]
    return files["preview"], None


def _already_scored(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and payload.get("own_table") == OWN_TABLE and not payload.get("placeholder"):
        return payload
    return None


def score_own_delivery(
    run_dir: str | Path,
    *,
    paper_id: str = "",
    pdf_path: str | Path | None = None,
    method_text: str | None = None,
    embedder: Any = None,
    paper_quiz_call: Callable[..., dict[str, Any]] | None = None,
    nine_call: Callable[..., dict[str, Any]] | None = None,
    paper_quiz_spec: str | None = None,
    nine_spec: str | None = None,
    force: bool = False,
    allow_question_generation: bool = True,
) -> dict[str, Any]:
    """Score one delivery folder. A folder with no figure file is not written."""
    run = Path(run_dir)
    metrics_path = run / "metrics.json"
    if not force:
        existing = _already_scored(metrics_path)
        if existing is not None:
            return {"ok": True, "skipped": True, "reason": "already_scored", "metrics": existing}
    files = _delivery_files(run)
    if not _has_delivery(files):
        return {"ok": False, "skipped": True, "reason": "delivery_missing", "paper_id": paper_id}
    run.mkdir(parents=True, exist_ok=True)
    if method_text is None and pdf_path and Path(pdf_path).is_file():
        text, method_reason = extract_method_section(pdf_path)
    else:
        text = str(method_text or "")
        method_reason = None if text.strip() else "missing method section"
    write_text(run / "method.txt", text)
    ground_text, nine_fallback = nine_grounding_text(text, pdf_path)
    preview, render_error = _preview(run, files)
    if files["pptx"].is_file():
        editability = editability_verdict(measure_native_editability(files["pptx"]))
    elif files["figure"].is_file() or files["tex"].is_file() or preview.is_file():
        editability = editability_verdict({}, delivery="non_ooxml")
    else:
        editability = editability_verdict({"ok": False, "ooxml_valid": False, "error": "missing delivery.pptx"})
    metrics = {
        "own_table": OWN_TABLE,
        "placeholder": False,
        "paper_id": paper_id,
        "method_reason": method_reason or "",
        "paper_figure_clip": score_paper_figure_clip(preview, text, embedder=embedder, render_error=render_error),
        "vlm_nine": score_nine(
            preview, method_text=ground_text, spec=nine_spec, call=nine_call, render_error=render_error
        ),
        "nine_grounding_fallback": nine_fallback,
        "paper_quiz": score_paper_quiz(
            preview,
            text,
            spec=paper_quiz_spec,
            call=paper_quiz_call,
            render_error=render_error,
            questions_path=run / "questions.json",
            allow_question_generation=allow_question_generation,
        ),
        "editability": editability,
        "preview_render_error": render_error,
    }
    write_json(metrics_path, metrics)
    return {"ok": True, "skipped": False, "reason": "", "metrics_path": str(metrics_path), "metrics": metrics}


def score_own_manifest(
    manifest_path: str | Path,
    runs_root: str | Path,
    *,
    limit: int | None = None,
    force: bool = False,
    embedder: Any = None,
    paper_quiz_call: Callable[..., dict[str, Any]] | None = None,
    nine_call: Callable[..., dict[str, Any]] | None = None,
    paper_quiz_spec: str | None = None,
    nine_spec: str | None = None,
) -> dict[str, Any]:
    """Score deliveries already under ``runs_root``. This does not generate."""
    rows = []
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    root = Path(runs_root)
    counts = {"scored": 0, "already_scored": 0, "delivery_missing": 0, "failed": 0}
    failures: list[dict[str, str]] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        run_dir = root / safe_paper_id(paper_id)
        try:
            result = score_own_delivery(
                run_dir,
                paper_id=paper_id,
                pdf_path=row.get("pdf_path") or None,
                embedder=embedder,
                paper_quiz_call=paper_quiz_call,
                nine_call=nine_call,
                paper_quiz_spec=paper_quiz_spec,
                nine_spec=nine_spec,
                force=force,
            )
        except Exception as exc:
            counts["failed"] += 1
            if len(failures) < 20:
                failures.append({"paper_id": paper_id, "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        if result.get("reason") == "already_scored":
            counts["already_scored"] += 1
        elif result.get("reason") == "delivery_missing":
            counts["delivery_missing"] += 1
        elif result.get("ok"):
            counts["scored"] += 1
        else:
            counts["failed"] += 1
    return {
        "summary": "Own-set table-1 scores for deliveries already on disk. No generation.",
        "ok": counts["failed"] == 0,
        "manifest": str(manifest_path),
        "runs_root": str(root),
        "considered": len(rows),
        "failures": failures,
        **counts,
    }
