"""Independently verify whether generated framework corruptions are visible and targeted."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from rfs.providers.client import call_vlm_json, resolve_vlm_model


VerifyFunction = Callable[..., dict[str, Any]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _contract_summary(contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract.get("expected_figure_content") if isinstance(contract.get("expected_figure_content"), dict) else {}
    return {
        "method_summary": contract.get("method_summary"),
        "required_entities": expected.get("required_entities", []),
        "required_relations": expected.get("required_relations", []),
        "must_not_invent": expected.get("must_not_invent", []),
        "training_flow": contract.get("training_flow", []),
        "inference_flow": contract.get("inference_flow", []),
    }


def build_verification_prompt(result: dict[str, Any], contract: dict[str, Any]) -> str:
    """Build a strict source-vs-edit verification prompt."""
    return f"""You are independently auditing a synthetic scientific-framework-diagram edit.
Image 1 is the human-reviewed source. Image 2 is the edited candidate.
Ground-truth paper contract:
{json.dumps(_contract_summary(contract), ensure_ascii=False)}

Requested corruption type: {result.get('corruption_type')}
Requested target: {json.dumps(result.get('target'), ensure_ascii=False)}
Edit instruction: {result.get('edit_instruction')}

Judge only visible changes. Do not infer that the edit succeeded merely from the instruction.
Return one JSON object with exactly these fields:
{{
  "target_error_visible": true or false,
  "detected_error_type": "one corruption type or none",
  "scientific_content_preserved_except_target": true or false,
  "source_identity_preserved": true or false,
  "legible": true or false,
  "confidence": number from 0 to 1,
  "evidence": ["short visible observations"],
  "unintended_changes": ["short visible changes"],
  "reason": "short conclusion"
}}
A sample passes only when the requested error is clearly visible, the detected type matches, source identity is preserved, and no unrelated scientific fact is changed. For correct_but_ugly, scientific content must remain intact and degraded readability must be clearly visible."""


def _normalize_verdict(raw: dict[str, Any], expected_type: str) -> dict[str, Any]:
    detected = str(raw.get("detected_error_type") or "none")
    confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0.0)))
    unintended = raw.get("unintended_changes") if isinstance(raw.get("unintended_changes"), list) else []
    readability_matches = raw.get("legible") is False if expected_type == "correct_but_ugly" else raw.get("legible") is True
    passed = (
        raw.get("target_error_visible") is True
        and detected == expected_type
        and raw.get("scientific_content_preserved_except_target") is True
        and raw.get("source_identity_preserved") is True
        and readability_matches
        and not unintended
        and confidence >= 0.7
    )
    return {
        "passed": passed,
        "target_error_visible": raw.get("target_error_visible") is True,
        "detected_error_type": detected,
        "scientific_content_preserved_except_target": raw.get("scientific_content_preserved_except_target") is True,
        "source_identity_preserved": raw.get("source_identity_preserved") is True,
        "legible": raw.get("legible") is True,
        "confidence": confidence,
        "evidence": raw.get("evidence") if isinstance(raw.get("evidence"), list) else [],
        "unintended_changes": unintended,
        "reason": str(raw.get("reason") or ""),
    }


def verify_corruptions(
    generated_results_path: str | Path,
    structured_dir: str | Path,
    verification_path: str | Path,
    *,
    workers: int = 2,
    retries: int = 1,
    timeout: int = 180,
    model: str | None = None,
    limit: int | None = None,
    verify_function: VerifyFunction = call_vlm_json,
) -> dict[str, Any]:
    """Verify generated items with resume support and persist each verdict atomically."""
    generated = [row for row in _read_jsonl(Path(generated_results_path)) if row.get("status") == "generated_unverified"]
    output = Path(verification_path)
    existing = {str(row["corruption_id"]): row for row in _read_jsonl(output)}
    pending = [row for row in generated if str(row["corruption_id"]) not in existing]
    if limit is not None:
        pending = pending[:max(0, int(limit))]
    contracts = Path(structured_dir)
    lock = threading.Lock()
    resolved_model = resolve_vlm_model("RFS_CRITIC_MODEL", explicit_model=model)

    def persist(row: dict[str, Any]) -> None:
        with lock:
            existing[str(row["corruption_id"])] = row
            _write_jsonl(output, sorted(existing.values(), key=lambda item: str(item["corruption_id"])))

    def verify(row: dict[str, Any]) -> dict[str, Any]:
        contract = json.loads((contracts / f"{row['paper_id']}.json").read_text(encoding="utf-8"))
        metadata: dict[str, Any] = {}
        result = {"corruption_id": row["corruption_id"], "paper_id": row["paper_id"], "figure_id": row["figure_id"], "split": row["split"], "corruption_type": row["corruption_type"], "source_image": row["source_image"], "output_image": row["output_image"], "output_sha256": row["output_sha256"]}
        try:
            raw = verify_function(build_verification_prompt(row, contract), [row["source_image"], row["output_image"]], model=resolved_model, timeout=timeout, retries=retries, call_metadata=metadata)
            verdict = _normalize_verdict(raw, str(row["corruption_type"]))
            result.update({"status": "verified" if verdict["passed"] else "rejected", "verification": verdict, "verifier_model": resolved_model, "provider": metadata, "error": None})
        except Exception as exc:
            result.update({"status": "verification_failed", "verification": None, "verifier_model": resolved_model, "provider": metadata, "error": f"{type(exc).__name__}: {exc}"})
        persist(result)
        return result

    completed: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(8, int(workers)))) as executor:
        futures = [executor.submit(verify, row) for row in pending]
        for future in as_completed(futures):
            completed.append(future.result())
    rows = list(existing.values())
    return {
        "generated": len(generated),
        "verified_this_run": len(completed),
        "passed": sum(row.get("status") == "verified" for row in rows),
        "rejected": sum(row.get("status") == "rejected" for row in rows),
        "verification_failed": sum(row.get("status") == "verification_failed" for row in rows),
        "remaining": len(generated) - len(rows),
        "verification_path": str(output.resolve()),
    }
