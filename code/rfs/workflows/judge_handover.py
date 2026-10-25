"""Drive a trained Judge all the way in: train, predict, gate, register, re-dress Memory.

A Judge used to be handed over by hand. One command trained it, another measured it, a
third gated it against the frozen suite and the bootstrap, a fourth registered the result,
and only then did anyone remember that the Memory an older Judge had written is still
sitting in the library and still being injected. Every step wrote an artifact and every
artifact path had to be typed into the next command, which is how a checkpoint that never
saw the shadow comparison ends up promoted.

This module runs that chain in one call. Each step consumes the artifact the previous step
wrote, each artifact has a fixed location under ``out`` so an interrupted handover resumes
instead of restarting, and the first step that cannot continue stops the chain with a named
reason instead of quietly skipping ahead. Serving the checkpoint -- vLLM or anything else
behind an OpenAI-compatible endpoint -- is deliberately out of scope: prediction is an
injectable seam, so a machine with no endpoint reports ``no_endpoint`` rather than
inventing a Judge of its own.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..contracts.judge_handover import HANDOVER_ROLE_ORDER, normalize_judge_fingerprint, validate_judge_handover_report
from ..judge_training.baseline import run_baseline
from ..judge_training.promotion import build_promotion_report
from ..judge_training.registry import checkpoint_identity, register_checkpoint
from ..judge_training.training import TrainingConfig, train_dpo
from ..providers.client import call_vlm_json, resolve_vlm_model
from ..utils import ensure_dir, read_json, write_json, write_text
from .memory_revalidation import sweep_memory_with_judge


GIT_EXECUTABLE = Path("<PATH>")
TRAINING_DIR = "training"
PREDICTIONS_DIR = "predictions"
PROMOTION_DIR = "promotion"
REGISTRATION_DIR = "registration"
SWEEP_DIR = "sweep"
TRAINING_REPORT_FILE = "training_report.json"
PREDICTIONS_FILE = "predictions.jsonl"
PROMOTION_REPORT_FILE = "promotion_report.json"
REGISTRATION_MANIFEST_FILE = "manifest.json"
SWEEP_REPORT_FILE = "memory_sweep_report.json"
REPORT_JSON = "judge_handover_report.json"
REPORT_MARKDOWN = "judge_handover_report.md"


def handover_project_root() -> Path:
    """Return the checkout a registration is traced back to."""

    return Path(__file__).resolve().parents[2]


def _head_commit(project: Path) -> str:
    """Read HEAD, so a registered Judge names the code that produced it."""

    executable = str(GIT_EXECUTABLE) if GIT_EXECUTABLE.is_file() else "git"
    try:
        completed = subprocess.run([executable, "-C", str(project), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def _endpoint_configured() -> bool:
    """Report whether a Judge endpoint is configured, the way the baseline judges it."""

    return bool(os.getenv("API_BASE")) and bool(os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY"))


def _row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _manifest_field(suite_manifest: str, key: str) -> str:
    """Read a field of a frozen suite, so a manifest alone can supply its dataset."""

    if not suite_manifest:
        return ""
    try:
        raw = read_json(suite_manifest)
    except (OSError, ValueError):
        return ""
    return str((raw if isinstance(raw, dict) else {}).get(key) or "")


def _incoming_model(judge_model: str, model_id: str) -> str:
    """Name the Judge the sweep measures against, resolved the way the loop resolves it.

    The loop stamps the model the Judge was built from, so the sweep has to name the same
    thing or the rows it reads look like another Judge's. Resolution therefore goes through
    the Judge's own environment names first -- but only when one of them is actually set:
    the generic model default names the bootstrap stand-in, not the Judge being handed over,
    so it must not be allowed to stand in for one. The registered model id is the last
    resort, and an unnamed Judge simply records no model.
    """

    explicit = str(judge_model or "").strip()
    if explicit:
        return explicit
    if os.getenv("RFS_ONLINE_JUDGE_MODEL", "").strip() or os.getenv("RFS_CRITIC_MODEL", "").strip():
        return resolve_vlm_model("RFS_ONLINE_JUDGE_MODEL", "RFS_CRITIC_MODEL")
    return str(model_id or "")


def handover_fingerprint(*, roles: Any = (), judge_mode: str = "trained", judge_model: str = "", model_id: str = "", checkpoint: str = "") -> dict[str, Any]:
    """Build the identity the sweep compares Memory against.

    The role is the strongest one the handover earned, so the fingerprint a sweep writes is
    the same shape the evolution loop stamps on each of its own rows. ``checkpoint`` is the
    registered identity of the weights, and it has to be the same string the loop will
    stamp: a sweep that quarantines under one identity and a run that writes rows under
    another would re-quarantine every row it just restored.
    """

    known = {str(role) for role in roles or []}
    strongest = next((role for role in reversed(HANDOVER_ROLE_ORDER) if role in known), "")
    return normalize_judge_fingerprint({"role": strongest, "mode": str(judge_mode or ""), "trained": str(judge_mode or "") != "heuristic", "model": _incoming_model(judge_model, model_id), "checkpoint": str(checkpoint or "")})


@dataclass(frozen=True)
class HandoverOptions:
    """Everything one handover needs, so no step reaches for a global."""

    root: Path
    training_dataset: str
    measured_dataset: str
    suite_manifest: str
    suite_missing: bool
    suite_version: str
    model_id: str
    dataset_version: str
    registry: str
    checkpoint: str
    training_report: str
    predict_from: str
    bootstrap_predictions: str
    no_shadow: bool
    allow_experimental: bool
    judge_mode: str
    judge_model: str
    backend: str
    experience_library: str
    sweep_selection: str
    sweep_limit: int
    git_commit: str
    force: bool
    dry_run: bool
    predict_adapter: Callable[..., dict[str, Any]]
    training_runner: Callable[..., dict[str, Any]]
    sweep_runner: Callable[..., dict[str, Any]] | None
    git_reader: Callable[[Path], str]


def _train_step(options: HandoverOptions, state: dict[str, Any]) -> dict[str, Any]:
    """Reuse or run the training, and hand the checkpoint forward."""

    if options.suite_missing:
        reason = "no_suite"
        detail = f"the frozen suite manifest is not at {options.suite_manifest}, so the checkpoint would have nothing to be gated on"
        return {"status": "pending" if options.dry_run else "blocked", "reason": "" if options.dry_run else reason, "would_block": reason if options.dry_run else "", "detail": detail, "artifacts": {}}
    if not state.get("training_dataset"):
        reason = "no_dataset"
        detail = "neither --dataset nor a suite manifest naming one was given, so there is nothing to train or measure on"
        return {"status": "pending" if options.dry_run else "blocked", "reason": "" if options.dry_run else reason, "would_block": reason if options.dry_run else "", "detail": detail, "artifacts": {}}
    run_report = options.root / TRAINING_DIR / TRAINING_REPORT_FILE
    report_path = Path(options.training_report) if options.training_report else run_report
    checkpoint = str(options.checkpoint or "")
    if checkpoint and not Path(checkpoint).exists():
        return {"status": "pending" if options.dry_run else "blocked", "reason": "" if options.dry_run else "no_checkpoint", "would_block": "no_checkpoint" if options.dry_run else "", "detail": f"the checkpoint is not at {checkpoint}", "artifacts": {}}
    if options.dry_run:
        artifacts = {}
        if checkpoint:
            artifacts["checkpoint"] = checkpoint
        if report_path.is_file():
            artifacts["training_report"] = str(report_path)
        return {"status": "pending", "detail": f"training is planned; its report and checkpoints go under {options.root / TRAINING_DIR}", "artifacts": artifacts}
    if checkpoint and not options.force:
        return {"status": "reused", "detail": "the checkpoint was supplied, so training was not re-run", "artifacts": {"checkpoint": checkpoint}, "checkpoint": checkpoint, "training_report": str(report_path) if report_path.is_file() else ""}
    report = read_json(report_path) if report_path.is_file() else None
    if isinstance(report, dict) and report and not options.force:
        status, detail = "reused", f"the training report at {report_path} was already on disk"
    else:
        report = options.training_runner(TrainingConfig(dataset=state["training_dataset"], out_dir=str(options.root / TRAINING_DIR), backend=options.backend))
        report_path = run_report
        status, detail = "completed", f"training ran on {state['training_dataset']} and wrote its checkpoints under {run_report.parent}"
    if not isinstance(report, dict) or not report:
        return {"status": "blocked", "reason": "no_training_report", "detail": f"training left no report to read at {report_path}", "artifacts": {}}
    saved = report.get("checkpoints") if isinstance(report.get("checkpoints"), list) else []
    if not checkpoint and saved:
        checkpoint = str(saved[-1])
    if not checkpoint:
        return {"status": "blocked", "reason": "no_checkpoint", "detail": f"the training report names no checkpoint, so there is nothing to gate; see {report_path}", "artifacts": {"training_report": str(report_path)}}
    return {"status": status, "detail": detail, "artifacts": {"training_report": str(report_path), "checkpoint": checkpoint}, "checkpoint": checkpoint, "training_report": str(report_path)}


def _predict_step(options: HandoverOptions, state: dict[str, Any]) -> dict[str, Any]:
    """Reuse or run the batch that turns the Judge into predicted rows."""

    if options.predict_from:
        supplied = Path(options.predict_from)
        artifacts = {"predictions": str(supplied)} if supplied.is_file() else {}
        if not supplied.is_file() and not options.dry_run:
            return {"status": "blocked", "reason": "no_predictions", "detail": f"--predict-from names {supplied}, which holds no predicted rows", "artifacts": {}}
        return {"status": "pending" if options.dry_run else "reused", "detail": f"the predicted rows are supplied from {supplied}, so no batch was run", "artifacts": artifacts, "predictions": str(supplied), "predictions_source": "supplied"}
    target = options.root / PREDICTIONS_DIR / PREDICTIONS_FILE
    if options.dry_run:
        would = "" if target.is_file() or _endpoint_configured() else "no_endpoint"
        return {"status": "pending", "detail": f"the Judge would predict {state['measured_dataset']} into {target}", "artifacts": {"predictions": str(target)} if target.is_file() else {}, "would_block": would, "predictions": str(target), "predictions_source": "run"}
    if target.is_file() and not options.force:
        return {"status": "reused", "detail": f"the predicted rows at {target} were already on disk", "artifacts": {"predictions": str(target)}, "predictions": str(target), "predictions_source": "run"}
    if not _endpoint_configured():
        return {"status": "blocked", "reason": "no_endpoint", "detail": "no Judge endpoint is configured: set API_BASE and API_KEY (or GEMINI_API_KEY), or hand the rows over with --predict-from", "artifacts": {}}
    if options.force and target.is_file():
        target.unlink()
    result = run_baseline(state["measured_dataset"], options.root / PREDICTIONS_DIR, model=options.judge_model or None, adapter=options.predict_adapter)
    rows = _row_count(target)
    if not rows:
        return {"status": "blocked", "reason": "no_predictions", "detail": f"the batch wrote no predicted rows; see {options.root / PREDICTIONS_DIR}", "artifacts": {}}
    return {"status": "completed", "detail": f"{rows} rows predicted by {result.get('model')}", "artifacts": {"predictions": str(target)}, "predictions": str(target), "predictions_source": "run"}


def _promote_step(options: HandoverOptions, state: dict[str, Any]) -> dict[str, Any]:
    """Gate the predictions and hand the promotion block forward."""

    report_json = options.root / PROMOTION_DIR / PROMOTION_REPORT_FILE
    shadow_omitted = bool(options.no_shadow) and not options.bootstrap_predictions
    if options.dry_run:
        would = "no_bootstrap_predictions" if not options.bootstrap_predictions and not options.no_shadow else ""
        return {"status": "pending", "detail": f"the promotion gate table is planned at {report_json}", "artifacts": {"promotion_report": str(report_json)} if report_json.is_file() else {}, "would_block": would, "shadow_omitted": shadow_omitted, "promotion": {}}
    if report_json.is_file() and not options.force:
        stored = read_json(report_json)
        promotion = stored.get("promotion") if isinstance(stored, dict) and isinstance(stored.get("promotion"), dict) else {}
        return {"status": "reused", "detail": f"the promotion report at {report_json} was already on disk", "artifacts": {"promotion_report": str(report_json)}, "shadow_omitted": shadow_omitted, "promotion": promotion}
    if not options.bootstrap_predictions and not options.no_shadow:
        return {"status": "blocked", "reason": "no_bootstrap_predictions", "detail": "the shadow comparison needs --bootstrap-predictions; pass --no-shadow to gate without it, which keeps the checkpoint off the frozen role", "artifacts": {}}
    report = build_promotion_report(
        dataset=state["measured_dataset"],
        predictions=state["predictions"],
        suite_manifest=options.suite_manifest or None,
        suite="hidden_test",
        checkpoint=state.get("checkpoint") or None,
        bootstrap_predictions=options.bootstrap_predictions or None,
        training_report=state.get("training_report") or None,
        report_out=options.root / PROMOTION_DIR,
    )
    promotion = report.get("promotion") if isinstance(report.get("promotion"), dict) else {}
    return {"status": "completed", "detail": "the gate table says: " + str(promotion.get("advice") or "no advice"), "artifacts": {"promotion_report": str(report_json)}, "shadow_omitted": shadow_omitted, "promotion": promotion}


def _register_step(options: HandoverOptions, state: dict[str, Any]) -> dict[str, Any]:
    """Register the checkpoint under the roles the gate table earned it."""

    manifest_copy = options.root / REGISTRATION_DIR / REGISTRATION_MANIFEST_FILE
    artifacts = {"manifest": str(manifest_copy), "registry_entry": str(options.registry)} if options.registry else {}
    if options.dry_run:
        would = ""
        if not options.model_id:
            would = "no_model_id"
        elif not options.registry:
            would = "no_registry"
        elif not str(options.git_commit or "").strip() and not options.git_reader(handover_project_root()):
            would = "no_git_commit"
        return {"status": "pending", "detail": f"the checkpoint would be registered in {options.registry or 'no registry'}", "artifacts": artifacts if manifest_copy.is_file() else {}, "would_block": would, "roles": [], "registered_as_experimental": False}
    if manifest_copy.is_file() and not options.force:
        stored = read_json(manifest_copy)
        manifest = stored if isinstance(stored, dict) else {}
        roles = [str(role) for role in manifest.get("roles") or []]
        promotion = manifest.get("promotion") if isinstance(manifest.get("promotion"), dict) else {}
        return {"status": "reused", "detail": f"the registration at {manifest_copy} was already on disk", "artifacts": artifacts, "roles": roles, "checkpoint_identity": checkpoint_identity(manifest), "registered_as_experimental": roles == ["experimental"] and not promotion.get("approved")}
    if not options.model_id:
        return {"status": "blocked", "reason": "no_model_id", "detail": "--model-id is required to register anything", "artifacts": {}}
    if not options.registry:
        return {"status": "blocked", "reason": "no_registry", "detail": "--registry is required to register anything", "artifacts": {}}
    promotion = state.get("promotion") or {}
    as_experimental = False
    if promotion.get("approved"):
        roles = [str(role) for role in promotion.get("roles") or []] or ["experimental"]
    elif options.allow_experimental:
        roles, as_experimental = ["experimental"], True
    else:
        reasons = "; ".join(str(item) for item in promotion.get("reasons") or []) or "the promotion gate did not approve this checkpoint"
        return {"status": "blocked", "reason": "promotion_not_approved", "detail": reasons + "; pass --allow-experimental to register it for experiments only", "artifacts": {}}
    version = str(options.dataset_version or state.get("suite_version") or "")
    if not version:
        return {"status": "blocked", "reason": "no_dataset_version", "detail": "a registration has to name the dataset version the checkpoint was trained on; pass --dataset-version", "artifacts": {}}
    commit = str(options.git_commit or "").strip() or str(options.git_reader(handover_project_root()) or "")
    if not commit:
        return {"status": "blocked", "reason": "no_git_commit", "detail": "no commit could be read from the checkout; pass --git-commit to name the code this Judge was trained from", "artifacts": {}}
    manifest = {
        "model_id": str(options.model_id),
        "dataset_version": version,
        "roles": roles,
        "git_commit": commit,
        "promotion": {
            "approved": bool(promotion.get("approved")),
            "hidden_test_passed": bool(promotion.get("hidden_test_passed")),
            "adversarial_passed": bool(promotion.get("adversarial_passed")),
            "shadow_stable": bool(promotion.get("shadow_stable")),
        },
    }
    register_checkpoint(state["checkpoint"], manifest, options.registry)
    write_json(manifest_copy, manifest)
    return {"status": "completed", "detail": f"registered as {', '.join(roles)} in {options.registry}", "artifacts": artifacts, "roles": roles, "checkpoint_identity": checkpoint_identity(manifest), "registered_as_experimental": as_experimental}


def _sweep_step(options: HandoverOptions, state: dict[str, Any]) -> dict[str, Any]:
    """Re-measure the Memory an older Judge wrote, under the Judge that just arrived."""

    fingerprint = handover_fingerprint(roles=state.get("roles") or [], judge_mode=options.judge_mode, judge_model=options.judge_model, model_id=options.model_id, checkpoint=str(state.get("checkpoint_identity") or ""))
    if not options.experience_library:
        return {"status": "skipped", "reason": "no_experience_library", "detail": "no --experience-library was given, so no Memory was re-measured", "artifacts": {}, "judge_fingerprint": fingerprint, "sweep": {}}
    report_json = options.root / SWEEP_DIR / SWEEP_REPORT_FILE
    if options.dry_run:
        return {"status": "pending", "detail": f"Memory would be swept against {fingerprint['role'] or 'the incoming Judge'}", "artifacts": {"sweep_report": str(report_json)} if report_json.is_file() else {}, "judge_fingerprint": fingerprint, "sweep": {}}
    if report_json.is_file() and not options.force:
        stored = read_json(report_json)
        counts = stored.get("counts") if isinstance(stored, dict) and isinstance(stored.get("counts"), dict) else {}
        return {"status": "reused", "detail": f"the sweep report at {report_json} was already on disk", "artifacts": {"sweep_report": str(report_json)}, "judge_fingerprint": fingerprint, "sweep": counts}
    report = sweep_memory_with_judge(
        experience_library=options.experience_library,
        incoming=fingerprint,
        selection=options.sweep_selection,
        out=options.root / SWEEP_DIR,
        dry_run=False,
        revalidate=True,
        runner=options.sweep_runner,
        limit=options.sweep_limit,
    )
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    restored = sum(1 for item in report.get("revalidated") or [] if isinstance(item, dict) and item.get("outcome") == "restored")
    return {"status": "completed", "detail": f"{counts.get('quarantined', 0)} rows stopped being injected and {restored} came back stamped by the incoming Judge", "artifacts": {"sweep_report": str(report_json)}, "judge_fingerprint": fingerprint, "sweep": counts}


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Judge handover report",
        "",
        f"- Model: `{report['model_id'] or 'unspecified'}`",
        f"- Suite: `{report['suite_id'] or 'unspecified'}`",
        f"- Checkpoint: `{report['checkpoint'] or 'none'}`",
        f"- Registry: `{report['registry'] or 'none'}`",
        f"- Predictions: `{report['predictions_source'] or 'none'}`",
        f"- Roles: {', '.join(report['roles']) or 'none'}",
        f"- Dry run: `{report['dry_run']}`",
        "",
        "| Step | Status | Detail |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {item['step']} | {item['status']} | {item['detail']} |" for item in report["steps"]]
    if report["blockers"]:
        lines += ["", "## Blockers", ""] + [f"- `{item['step']}`: {item['reason']} - {item['detail']}" for item in report["blockers"]]
    if report["sweep"]:
        lines += ["", "## Memory sweep", "", "| Count | Rows |", "| --- | ---: |"] + [f"| {key} | {value} |" for key, value in sorted(report["sweep"].items())]
    return "\n".join(lines) + "\n"


def run_judge_handover(
    *,
    out: str | Path,
    dataset: str | Path = "",
    suite_manifest: str | Path = "",
    model_id: str = "",
    dataset_version: str = "",
    registry: str | Path = "",
    checkpoint: str | Path = "",
    training_report: str | Path = "",
    predict_from: str | Path = "",
    bootstrap_predictions: str | Path = "",
    no_shadow: bool = False,
    allow_experimental: bool = False,
    judge_mode: str = "trained",
    judge_model: str = "",
    backend: str = "synthetic",
    experience_library: str | Path = "",
    sweep_selection: str = "mismatched",
    sweep_limit: int = 0,
    git_commit: str = "",
    force: bool = False,
    dry_run: bool = False,
    predict_adapter: Callable[..., dict[str, Any]] = call_vlm_json,
    training_runner: Callable[..., dict[str, Any]] = train_dpo,
    sweep_runner: Callable[..., dict[str, Any]] | None = None,
    git_reader: Callable[[Path], str] = _head_commit,
) -> dict[str, Any]:
    """Run the whole handover, stopping at the first step that cannot continue.

    Training is reused whenever its report is already on disk, predicting is reused whenever
    the rows are, and every later step reads the artifact the previous one wrote, so a run
    that is interrupted resumes from the last step that finished instead of starting over.
    A dry run plans the same chain and writes nothing.
    """

    root = Path(out)
    if not dry_run:
        root = ensure_dir(root)
    suite_path = str(suite_manifest or "")
    suite_missing = bool(suite_path) and not Path(suite_path).is_file()
    suite_dataset = "" if suite_missing else _manifest_field(suite_path, "dataset")
    options = HandoverOptions(
        root=root,
        training_dataset=str(dataset or suite_dataset or ""),
        measured_dataset=suite_dataset or str(dataset or ""),
        suite_manifest=suite_path,
        suite_missing=suite_missing,
        suite_version="" if suite_missing else _manifest_field(suite_path, "dataset_version"),
        model_id=str(model_id or ""),
        dataset_version=str(dataset_version or ""),
        registry=str(registry or ""),
        checkpoint=str(checkpoint or ""),
        training_report=str(training_report or ""),
        predict_from=str(predict_from or ""),
        bootstrap_predictions=str(bootstrap_predictions or ""),
        no_shadow=bool(no_shadow),
        allow_experimental=bool(allow_experimental),
        judge_mode=str(judge_mode or "trained"),
        judge_model=str(judge_model or ""),
        backend=str(backend or "synthetic"),
        experience_library=str(experience_library or ""),
        sweep_selection=str(sweep_selection or "mismatched"),
        sweep_limit=int(sweep_limit or 0),
        git_commit=str(git_commit or ""),
        force=bool(force),
        dry_run=bool(dry_run),
        predict_adapter=predict_adapter,
        training_runner=training_runner,
        sweep_runner=sweep_runner,
        git_reader=git_reader,
    )
    steps: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    state: dict[str, Any] = {"training_dataset": options.training_dataset, "measured_dataset": options.measured_dataset, "suite_version": options.suite_version}
    stopped = False

    def add(step: str, status: str, *, detail: str = "", reason: str = "", artifacts: dict[str, str] | None = None, would_block: str = "") -> None:
        steps.append({"step": step, "status": status, "detail": detail, "reason": reason, "would_block": would_block, "artifacts": artifacts or {}})
        if status in ("blocked", "failed"):
            blockers.append({"step": step, "reason": reason, "detail": detail})

    def run_step(name: str, action: Callable[[HandoverOptions, dict[str, Any]], dict[str, Any]]) -> None:
        nonlocal stopped
        if stopped:
            add(name, "skipped", reason="earlier_step_blocked", detail=f"the chain stopped before {name} could run")
            return
        try:
            outcome = action(options, state)
        except Exception as exc:  # a step that crashes stops the chain the way a gate failure does
            add(name, "failed", detail=f"{type(exc).__name__}: {exc}")
            stopped = True
            return
        status = str(outcome.get("status") or "pending")
        add(name, status, detail=str(outcome.get("detail") or ""), reason=str(outcome.get("reason") or ""), artifacts=outcome.get("artifacts") or {}, would_block=str(outcome.get("would_block") or ""))
        state.update({key: value for key, value in outcome.items() if key not in ("status", "detail", "reason", "artifacts", "would_block")})
        if status in ("blocked", "failed"):
            stopped = True

    run_step("train", _train_step)
    run_step("predict", _predict_step)
    run_step("promote", _promote_step)
    run_step("register", _register_step)
    run_step("sweep", _sweep_step)

    roles = [str(role) for role in state.get("roles") or []]
    fingerprint = normalize_judge_fingerprint(state.get("judge_fingerprint") or handover_fingerprint(roles=roles, judge_mode=options.judge_mode, judge_model=options.judge_model, model_id=options.model_id, checkpoint=str(state.get("checkpoint_identity") or "")))
    failed = [item["step"] for item in steps if item["status"] in ("blocked", "failed")]
    ok = not failed
    label = options.model_id or "the incoming Judge"
    if options.dry_run:
        summary = f"Judge handover plan for {label}: {len(steps)} steps planned, nothing was run."
    elif ok:
        summary = f"Judge handover for {label} walked every step: " + ", ".join(f"{item['step']} {item['status']}" for item in steps)
    else:
        summary = f"Judge handover for {label} stopped at {blockers[0]['step']}: {blockers[0]['reason']}."
    report = {
        "summary": summary,
        "ok": ok,
        "dry_run": options.dry_run,
        "force": options.force,
        "out_dir": str(root),
        "model_id": options.model_id,
        "dataset_version": str(options.dataset_version or state.get("suite_version") or ""),
        "suite_id": _manifest_field(options.suite_manifest, "suite_id"),
        "suite_manifest": options.suite_manifest,
        "checkpoint": str(state.get("checkpoint") or options.checkpoint or ""),
        "registry": options.registry,
        "judge_fingerprint": fingerprint,
        "predictions_source": str(state.get("predictions_source") or ""),
        "shadow_omitted": bool(state.get("shadow_omitted")),
        "registered_as_experimental": bool(state.get("registered_as_experimental")),
        "roles": roles,
        "promotion": state.get("promotion") or {},
        "sweep": state.get("sweep") or {},
        "steps": steps,
        "blockers": blockers,
    }
    checked = validate_judge_handover_report(report)
    final = {**checked["report"], "validation": {"ok": bool(checked["ok"]), "errors": list(checked["errors"])}, "ok": bool(checked["ok"]) and ok}
    if not options.dry_run:
        write_json(root / REPORT_JSON, final)
        write_text(root / REPORT_MARKDOWN, _markdown(final))
    return final


__all__ = ["HandoverOptions", "handover_fingerprint", "handover_project_root", "run_judge_handover"]
