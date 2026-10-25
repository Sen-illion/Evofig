"""Export frozen Judge examples into multimodal chat-training JSONL."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .prompts import PROMPT_VERSION, build_prompt
from ..contracts.evolution import ACCEPTANCE_ROLE_FAMILY, judge_role_family


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assets(row: dict[str, Any]) -> list[str]:
    assets = [str(row[k]) for k in ("candidate_a", "candidate_b") if row.get(k)]
    if not assets and row.get("preview"):
        assets.append(str(row["preview"]))
    return assets


def _export_process_dataset(source: Path, dest: Path) -> dict[str, Any]:
    records = _read(source)
    dest.mkdir(parents=True, exist_ok=True)
    targets = {
        "process": dest / "process.jsonl",
        "advisory": dest / "advisory.jsonl",
        "acceptance": dest / "acceptance.jsonl",
    }
    lf_root = dest / "llamafactory"
    lf_targets = {name: lf_root / f"{name}.jsonl" for name in targets}
    lf_root.mkdir(parents=True, exist_ok=True)
    exported = {name: 0 for name in targets}
    missing = []
    handles = {name: path.open("w", encoding="utf-8") for name, path in targets.items()}
    lf_handles = {name: path.open("w", encoding="utf-8") for name, path in lf_targets.items()}
    try:
        for row in records:
            assets = _assets(row)
            absent = [item for item in assets if not Path(item).exists()]
            if absent:
                missing.append({"example_id": row.get("example_id"), "images": absent})
                continue
            answer = row.get("labels") if isinstance(row.get("labels"), dict) else {}
            prompt = build_prompt(row)
            metadata = {"example_id": row.get("example_id"), "task_type": row.get("task_type"), "ground_truth_id": row.get("ground_truth_id"), "dataset_version": row.get("dataset_version"), "source": row.get("provenance", {}).get("source"), "prompt_version": PROMPT_VERSION, "judge_role": (row.get("judge_metadata") or {}).get("judge_role")}
            content = [{"type": "text", "text": prompt}, *({"type": "image", "image": item} for item in assets)]
            line = json.dumps({"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))}]}], "metadata": metadata}, ensure_ascii=False, separators=(",", ":")) + "\n"
            lf_line = json.dumps({"messages": [{"role": "user", "content": "<image>" * len(assets) + "\n" + prompt}, {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))}], "images": assets, "metadata": metadata}, ensure_ascii=False, separators=(",", ":")) + "\n"
            handles["process"].write(line)
            lf_handles["process"].write(lf_line)
            exported["process"] += 1
            if str(row.get("task_type") or "") != "repair":
                continue
            family = judge_role_family(str((row.get("judge_metadata") or {}).get("judge_role") or ""))
            name = "acceptance" if family == ACCEPTANCE_ROLE_FAMILY else "advisory"
            handles[name].write(line)
            lf_handles[name].write(lf_line)
            exported[name] += 1
    finally:
        for handle in handles.values():
            handle.close()
        for handle in lf_handles.values():
            handle.close()
    manifest = {
        "schema_version": "judge-process-training-export-v1",
        "source": str(source.resolve()),
        "records": exported["process"],
        "role_counts": {"advisory": exported["advisory"], "acceptance": exported["acceptance"]},
        "forked_trajectories": exported["advisory"] > 0 and exported["acceptance"] > 0,
        "missing_images": missing,
        "sha256": hashlib.sha256(targets["process"].read_bytes()).hexdigest(),
        "llamafactory_sha256": hashlib.sha256(lf_targets["process"].read_bytes()).hexdigest(),
        "formats": ["multimodal_chat", "llamafactory_messages"],
        "role_files": ["advisory.jsonl", "acceptance.jsonl"],
    }
    (dest / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "export_report.json").write_text(json.dumps({"ok": not missing, **manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def export_training_data(dataset: str | Path, out: str | Path, *, format: str = "chat", require_human_audit: bool = False) -> dict[str, Any]:
    """Export a dataset as multimodal chat records or as preference pairs for DPO."""

    if str(format).strip().casefold() == "dpo":
        from .preference_pairs import export_preference_pairs
        return export_preference_pairs(dataset, out, require_human_audit=require_human_audit)
    root = Path(dataset); dest = Path(out); dest.mkdir(parents=True, exist_ok=True)
    process_source = root / "judge_examples.jsonl"
    if process_source.is_file() and not (root / "train.jsonl").is_file():
        return _export_process_dataset(process_source, dest)
    files = {"train": root / "train.jsonl", "validation": root / "validation.jsonl", "hidden_test": root / "hidden_test.jsonl"}
    reports: dict[str, Any] = {}; total = 0; split_ground_truth: dict[str, set[str]] = {}
    for split, source in files.items():
        target = dest / f"{split}.jsonl"; lf_target = dest / "llamafactory" / f"{split}.jsonl"; lf_target.parent.mkdir(parents=True, exist_ok=True)
        records = _read(source); exported = 0; missing: list[dict[str, Any]] = []; truncated: list[dict[str, Any]] = []; split_ground_truth[split] = set()
        with target.open("w", encoding="utf-8") as handle, lf_target.open("w", encoding="utf-8") as lf_handle:
            for row in records:
                split_ground_truth[split].add(str(row.get("ground_truth_id") or ""))
                assets = _assets(row); absent = [p for p in assets if not Path(p).exists()]
                if absent:
                    missing.append({"example_id": row.get("example_id"), "images": absent}); continue
                spans = (row.get("paper") or {}).get("evidence_spans") or []
                if len(spans) > 12: truncated.append({"example_id": row.get("example_id"), "original": len(spans), "used": 12})
                answer = row.get("labels") if isinstance(row.get("labels"), dict) else {}
                content = [{"type": "text", "text": build_prompt(row)}, *({"type": "image", "image": p} for p in assets)]
                item = {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": [{"type": "text", "text": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))}]}], "metadata": {"example_id": row.get("example_id"), "task_type": row.get("task_type"), "ground_truth_id": row.get("ground_truth_id"), "dataset_version": row.get("dataset_version"), "source": row.get("provenance", {}).get("source"), "corruption_type": row.get("provenance", {}).get("corruption_type"), "prompt_version": PROMPT_VERSION}}
                handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"); exported += 1
                lf_item = {"messages": [{"role": "user", "content": "<image>" * len(assets) + "\n" + build_prompt(row)}, {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))}], "images": assets, "metadata": item["metadata"]}
                lf_handle.write(json.dumps(lf_item, ensure_ascii=False, separators=(",", ":")) + "\n")
        digest = hashlib.sha256(target.read_bytes()).hexdigest(); lf_digest = hashlib.sha256(lf_target.read_bytes()).hexdigest()
        reports[split] = {"source_records": len(records), "exported_records": exported, "missing_images": missing, "evidence_truncations": truncated, "sha256": digest, "llamafactory_sha256": lf_digest}; total += exported
    manifest_path = root / "dataset_manifest.json"
    dataset_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None
    overlaps = {"train_validation": sorted(split_ground_truth["train"] & split_ground_truth["validation"]), "train_hidden_test": sorted(split_ground_truth["train"] & split_ground_truth["hidden_test"]), "validation_hidden_test": sorted(split_ground_truth["validation"] & split_ground_truth["hidden_test"])}
    manifest = {"schema_version": "judge-training-export-v1", "dataset": str(root.resolve()), "dataset_manifest_sha256": dataset_digest, "prompt_version": PROMPT_VERSION, "formats": ["multimodal_chat", "llamafactory_messages"], "ground_truth_split_overlaps": overlaps, "splits": reports, "records": total}
    (dest / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = all(not x["missing_images"] for x in reports.values()) and not any(overlaps.values())
    (dest / "export_report.json").write_text(json.dumps({"ok": ok, **manifest}, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
