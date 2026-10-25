"""Discovery, validation, and source fetching for direct-pipeline benchmarks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import requests

from ..utils import read_json, write_json

DIRECT_SUITE = "paper-evolve-editable"
SUITES = {DIRECT_SUITE}


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def validate_benchmark_case(case_dir: str | Path) -> dict[str, Any]:
    """Validate one direct editable-PPTX benchmark case without running it."""
    root = Path(case_dir).resolve()
    case = _load(root / "case.json")
    errors: list[str] = []
    warnings: list[str] = []
    if str(case.get("suite") or "") != DIRECT_SUITE:
        errors.append(f"case.json suite must be {DIRECT_SUITE}")
    if not str(case.get("case_id") or "").strip():
        errors.append("case.json requires case_id")
    if not isinstance(case.get("thresholds"), dict):
        errors.append("case.json requires thresholds object")
    paper = root / str(case.get("paper") or "inputs/paper.pdf")
    if not paper.is_file():
        if (root / str(case.get("source") or "source.json")).is_file():
            warnings.append(f"paper input not fetched yet: {paper.name}")
        else:
            errors.append(f"missing paper input: {paper.name}")
    expected = root / str(case.get("expected_contract") or "expected_contract.json")
    if not expected.is_file():
        errors.append(f"missing expected contract: {expected.name}")
    else:
        payload = _load(expected)
        if not isinstance(payload.get("required_nodes"), list):
            errors.append("expected contract requires required_nodes list")
        if not isinstance(payload.get("required_edges"), list):
            errors.append("expected contract requires required_edges list")
    return {"summary": "Direct benchmark case validation.", "ok": not errors, "case_dir": str(root), "case_id": case.get("case_id"), "suite": case.get("suite"), "errors": errors, "warnings": warnings}


def fetch_benchmark_case(case_dir: str | Path, force: bool = False) -> dict[str, Any]:
    """Fetch a public paper declared by a direct benchmark case."""
    root = Path(case_dir).resolve()
    case = _load(root / "case.json")
    source_path = root / str(case.get("source") or "source.json")
    source = _load(source_path)
    target = root / str(case.get("paper") or "inputs/paper.pdf")
    if target.is_file() and not force:
        return {"summary": "Benchmark paper already exists.", "ok": True, "status": "reused", "paper": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    urls = source.get("paper_urls") if isinstance(source.get("paper_urls"), list) else []
    if not urls:
        return {"summary": "Benchmark source metadata is missing.", "ok": False, "status": "missing_source", "case_dir": str(root)}
    errors = []
    for url in urls:
        try:
            response = requests.get(str(url), headers={"User-Agent": "EvoFig benchmark fetch/1.0"}, timeout=120)
            response.raise_for_status()
            content = response.content
            if not content.startswith(b"%PDF"):
                raise ValueError("downloaded response is not a PDF")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            report = {"summary": "Benchmark paper fetched.", "ok": True, "status": "downloaded", "paper": str(target), "source_url": str(url), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            write_json(target.parent / "fetch_report.json", report)
            return report
        except Exception as exc:
            errors.append({"url": str(url), "error": str(exc)})
    return {"summary": "Unable to fetch benchmark paper.", "ok": False, "status": "failed", "case_dir": str(root), "errors": errors}


def list_benchmark_cases(benchmarks_root: str | Path, suite: str | None = None) -> dict[str, Any]:
    """List direct-pipeline benchmark cases and their validation status."""
    root = Path(benchmarks_root).resolve()
    suites = [suite] if suite else sorted(SUITES)
    cases = []
    for suite_name in suites:
        case_root = root / suite_name / "cases"
        if not case_root.is_dir():
            continue
        for case_dir in sorted(path for path in case_root.iterdir() if path.is_dir()):
            validation = validate_benchmark_case(case_dir)
            cases.append({"case_id": validation.get("case_id") or case_dir.name, "suite": suite_name, "case_dir": str(case_dir), "valid": validation["ok"], "errors": validation["errors"]})
    return {"summary": "Available direct-pipeline benchmark cases.", "benchmarks_root": str(root), "cases": cases}
