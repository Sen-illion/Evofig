"""Draft gold contracts for framework figures in the public benchmark.

Only papers whose manifest flag is ``yes`` get a draft. An existing structured
draft is cited and not re-extracted. A paper with no draft is left on the
extraction queue unless the caller supplies an extractor. ``unknown`` papers
go to the labeling queue. Nothing here writes ``confirmation.json`` or marks
a draft as accepted. The module/edge denominator stays the count of accepted
contracts, which this step does not increment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..utils import write_json, write_text

PENDING_CONFIRMATION = {"reviewer": "", "decision": "pending", "corrections": [], "updated_at": ""}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _arxiv_key(stem: str) -> str:
    """2301.02083 and 2301.02083.validation both name the paper 2301.02083."""
    parts = stem.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return parts[0] + "." + parts[1]
    return stem


def index_structured_contracts(contracts_dir: str | Path) -> dict[str, Path]:
    """Map an arXiv id to the first structured draft whose name starts with that id."""
    root = Path(contracts_dir)
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for path in sorted(root.glob("*.json")):
        found.setdefault(_arxiv_key(path.name[: -len(path.suffix)]), path)
    return found


def _contract_for(arxiv_ids: list[str], index: dict[str, Path]) -> Path | None:
    for arxiv_id in arxiv_ids:
        path = index.get(arxiv_id)
        if path is not None:
            return path
    return None


def draft_contracts(
    manifest_rows: list[dict[str, Any]],
    *,
    contracts_dir: str | Path,
    out_dir: str | Path,
    extractor: Callable[[dict[str, Any], Path], dict[str, Any]] | None = None,
    exclude_paper_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Write drafts and the labeling queue. Accepted-contract n stays 0."""
    excluded = exclude_paper_ids or set()
    index = index_structured_contracts(contracts_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    drafts: list[dict[str, Any]] = []
    labeling: list[dict[str, Any]] = []
    for row in manifest_rows:
        paper_id = str(row.get("paper_id") or "")
        if paper_id in excluded:
            continue
        flag = str(row.get("framework_figure") or "")
        if flag == "unknown":
            labeling.append(
                {
                    "paper_id": paper_id,
                    "arxiv_id": str(row.get("arxiv_id") or ""),
                    "reason": "framework figure was not decided by an existing human review",
                }
            )
            continue
        if flag != "yes":
            continue
        arxiv_ids = [str(item) for item in (row.get("arxiv_ids") or []) if str(item).strip()]
        if row.get("arxiv_id") and str(row.get("arxiv_id")) not in arxiv_ids:
            arxiv_ids.insert(0, str(row.get("arxiv_id")))
        existing = _contract_for(arxiv_ids, index)
        if existing is not None:
            drafts.append(
                {
                    "paper_id": paper_id,
                    "arxiv_id": arxiv_ids[0] if arxiv_ids else "",
                    "source": "existing_structured_draft",
                    "contract_path": str(existing),
                    "human_confirmation": dict(PENDING_CONFIRMATION),
                    "signed": False,
                }
            )
            continue
        if extractor is None:
            drafts.append(
                {
                    "paper_id": paper_id,
                    "arxiv_id": arxiv_ids[0] if arxiv_ids else "",
                    "source": "needs_extraction",
                    "contract_path": "",
                    "human_confirmation": dict(PENDING_CONFIRMATION),
                    "signed": False,
                    "reason": "no structured draft and no extractor credentials",
                }
            )
            continue
        extracted = extractor(row, out / "extracted" / paper_id)
        drafts.append(
            {
                "paper_id": paper_id,
                "arxiv_id": arxiv_ids[0] if arxiv_ids else "",
                "source": "extracted_draft",
                "contract_path": str(extracted.get("contract_path") or ""),
                "human_confirmation": dict(PENDING_CONFIRMATION),
                "signed": False,
                "ok": bool(extracted.get("ok")),
            }
        )
    write_text(out / "contract_drafts.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in drafts))
    write_text(out / "labeling_queue.jsonl", "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labeling))
    accepted = 0
    report = {
        "summary": "Framework-figure contract drafts. None are human-confirmed.",
        "ok": True,
        "draft_count": len(drafts),
        "cited_existing": sum(1 for row in drafts if row["source"] == "existing_structured_draft"),
        "needs_extraction": sum(1 for row in drafts if row["source"] == "needs_extraction"),
        "extracted": sum(1 for row in drafts if row["source"] == "extracted_draft"),
        "labeling_queue": len(labeling),
        "module_edge_n": accepted,
        "signed_confirmation": False,
    }
    write_json(out / "contract_draft_report.json", report)
    return report
