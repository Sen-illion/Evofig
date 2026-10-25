"""Use a VLM to select one framework overview from per-paper candidates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from rfs.providers import call_vlm_json


PROMPT = """You receive candidate figures from one academic paper, ordered alphabetically from A onward.
Select the single figure that most clearly presents the paper's overall method, system architecture,
or end-to-end framework: modules/components and their relationships, flow, or pipeline. Do not choose
an experimental plot, ablation, qualitative result, table, or isolated submodule. Return JSON only:
{"choice":"A"|"B"|"C"|"none","reason":"brief evidence-based explanation","confidence":0.0}."""


def selected_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return list(payload.get("records", []))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=int(os.getenv("RFS_VLM_TIMEOUT", "90")))
    parser.add_argument("--retries", type=int, default=int(os.getenv("RFS_VLM_RETRIES", "1")))
    args = parser.parse_args()

    summary: dict[str, Any] = {"total": 0, "selected": 0, "none": 0, "no_candidates": 0, "errors": [], "items": {}}
    for row in selected_records(args.selection):
        arxiv_id = str(row["arxiv_id"])
        print(f"[select] {summary['total'] + 1}: {arxiv_id}", flush=True)
        summary["total"] += 1
        candidate_dir = args.candidates_root / arxiv_id
        info_path = candidate_dir / "candidates.json"
        out_dir = args.out_root / arxiv_id
        meta_path = out_dir / "meta.json"
        if meta_path.exists():
            summary["items"][arxiv_id] = "existing"
            continue
        candidates = json.loads(info_path.read_text(encoding="utf-8")).get("candidates", []) if info_path.exists() else []
        image_paths = [candidate_dir / f"cand_{index}.png" for index in range(1, min(len(candidates), args.max_candidates) + 1)]
        image_paths = [path for path in image_paths if path.exists()]
        out_dir.mkdir(parents=True, exist_ok=True)
        if not image_paths:
            meta_path.write_text(json.dumps({"arxiv_id": arxiv_id, "status": "no_candidates", "human_labels": row}, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["no_candidates"] += 1
            summary["items"][arxiv_id] = "no_candidates"
            continue
        try:
            result = call_vlm_json(PROMPT, [str(path) for path in image_paths], model=args.model, timeout=args.timeout, retries=args.retries)
            choice = str(result.get("choice") or "none").upper().strip()
            rank = ord(choice) - ord("A") + 1 if len(choice) == 1 and "A" <= choice <= "Z" else None
            meta = {"arxiv_id": arxiv_id, "status": "selected" if rank and rank <= len(image_paths) else "none", "vlm_choice": choice, "vlm_reason": result.get("reason", ""), "vlm_confidence": result.get("confidence"), "candidates": candidates, "human_labels": row}
            if rank and rank <= len(image_paths):
                shutil.copy2(image_paths[rank - 1], out_dir / "framework.png")
                meta["chosen_rank"] = rank
                summary["selected"] += 1
                summary["items"][arxiv_id] = "selected"
            else:
                summary["none"] += 1
                summary["items"][arxiv_id] = "none"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            summary["errors"].append({"arxiv_id": arxiv_id, "error": str(exc)})
            summary["items"][arxiv_id] = "error"
            print(f"[select] {arxiv_id}: error={str(exc)[:160]}", flush=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
