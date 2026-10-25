"""Command-line interface for the sole direct paper-to-PPTX pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__
from .analysis.paper.inspection import inspect_paper
from .analysis.reference import (
    analyze_reference_figures,
    learn_reference_style,
    select_reference_exemplars,
)
from .cli_commands.categories import (
    EXPERIMENTAL_COMMANDS,
    MAINTENANCE_COMMANDS,
    PRODUCT_COMMANDS,
)
from .cli_commands.decks import register_deck_adoption_command
from .cli_commands.figure_edit import figure_edit_invocation, register_figure_edit
from .cli_commands.experimental import register_judge_command, register_judge_data_commands
from .cli_commands.maintenance import (
    register_quality_commands,
    register_reference_and_inspection_commands,
)
from .cli_commands.product import (
    register_creator_rules,
    register_experience_archive,
    register_paper_evolve_continuous,
    register_paper_evolve_corpus,
    register_paper_evolve_editable,
    register_paper_evolve_patches,
    register_paper_evolve_review,
    register_paper_evolve_session,
    register_paper_evolve_validate,
)
from .evaluation.benchmarking import fetch_benchmark_case, list_benchmark_cases, validate_benchmark_case
from .evaluation.contract_crosscheck import run_contract_crosscheck
from .evaluation.benchmark_contracts import draft_contracts
from .evaluation.benchmark_manifest import export_public_benchmark_manifest
from .evaluation.editable_figure_bench import (
    ALL_SPLITS,
    DEFAULT_LABEL_GROUPS,
    HELD_OUT_SPLITS,
    LAYERS as BENCHMARK_LAYERS,
    prepare_gold_instances,
    run_benchmark_instances,
    score_benchmark,
    verify_gold_instances,
)
from .composition.figure_edit import run_figure_edit
from .composition.pptx_render import RENDER_MODE_ENV, available_backends, render_mode
from .evaluation.outputs import validate_output
from .evaluation.own_set_score import LocalAltCLIP, score_own_manifest
from .evaluation.pdf_extraction_benchmark import run_pdf_extraction_stress_suite
from .judge_training import (
    build_dataset,
    build_suite_manifest,
    compare_shadow_predictions,
    collect_literature_papers,
    evaluate_predictions,
    export_training_data,
    register_checkpoint,
    run_bakeoff,
    run_baseline,
    validate_dataset,
)
from .presentations_qa import run_presentations_qa
from .judge_training.literature_guard import guarded_collection
from .utils import env_present, mask_secret
from .workflows import archive_untransferable_experiences, backfill_prompt_patches, report_creator_rule_ledger, review_answer, review_collect, review_export, review_queue, review_record, review_stats, run_direct_experience_validation, run_paper_evolve_editable


def _json_print(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _probe_executable(name: str, version_arg: str = "-v") -> dict:
    path = shutil.which(name)
    if not path:
        return {"available": False, "path": None, "error": "not found on PATH"}
    try:
        completed = subprocess.run(
            [path, version_arg], capture_output=True, text=True, timeout=5, check=False
        )
    except Exception as exc:
        return {"available": False, "path": path, "error": str(exc)}
    return {
        "available": completed.returncode == 0,
        "path": path,
        "returncode": completed.returncode,
    }


def _doctor() -> dict:
    dependencies = {
        name: importlib.util.find_spec(module) is not None
        for name, module in {
            "Pillow": "PIL",
            "python-pptx": "pptx",
            "PyMuPDF": "fitz",
            "PDFium": "pypdfium2",
            "requests": "requests",
            "OpenCV": "cv2",
        }.items()
    }
    auth = {
        "API_BASE": {"present": env_present("API_BASE")},
        "API_KEY": {"present": env_present("API_KEY"), "masked": mask_secret(os.getenv("API_KEY"))},
        "GEMINI_API_KEY": {"present": env_present("GEMINI_API_KEY"), "masked": mask_secret(os.getenv("GEMINI_API_KEY"))},
        "RFS_ONLINE_JUDGE_MODEL": {"present": env_present("RFS_ONLINE_JUDGE_MODEL")},
        "RFS_FROZEN_JUDGE_MODEL": {"present": env_present("RFS_FROZEN_JUDGE_MODEL")},
        "RFS_IMAGE_MODEL": {"present": env_present("RFS_IMAGE_MODEL")},
    }
    return {
        "summary": "EvoFig direct-pipeline environment check.",
        "ok": all(dependencies.values()),
        "version": __version__,
        "dependencies": dependencies,
        "executables": {
            "git": _probe_executable("git", "--version"),
            "libreoffice": _probe_executable("libreoffice", "--version"),
        },
        # The preview is what the visual reviewer sees; a host with no engine
        # draws it with the program renderer instead, and the run report says so.
        "preview_render": {
            "mode": render_mode(),
            "mode_env": RENDER_MODE_ENV,
            "available_backends": available_backends(),
        },
        "auth": auth,
        "product_commands": list(PRODUCT_COMMANDS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfs",
        description="EvoFig direct paper-to-PPTX evolution CLI",
        epilog=(
            f"Product: {', '.join(PRODUCT_COMMANDS)}. "
            f"Judge: {', '.join(EXPERIMENTAL_COMMANDS)}. "
            f"Maintenance/QA: {', '.join(MAINTENANCE_COMMANDS)}."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    sub = parser.add_subparsers(dest="command", required=True)
    register_reference_and_inspection_commands(sub)
    register_paper_evolve_editable(sub)
    register_paper_evolve_validate(sub)
    register_paper_evolve_corpus(sub)
    register_paper_evolve_session(sub)
    register_paper_evolve_continuous(sub)
    register_paper_evolve_patches(sub)
    register_paper_evolve_review(sub)
    register_experience_archive(sub)
    register_creator_rules(sub)
    register_quality_commands(sub)
    register_judge_command(sub)
    register_judge_data_commands(sub)
    register_deck_adoption_command(sub)
    register_figure_edit(sub)
    return parser


def _run_direct(args: argparse.Namespace) -> dict:
    return run_paper_evolve_editable(
        paper=args.paper,
        paper_id=args.paper_id,
        reference=args.reference,
        out=args.out,
        planner_mode=args.planner_mode,
        read_mode=args.read_mode,
        aspect_ratio=args.aspect_ratio,
        max_rounds=args.max_rounds,
        max_retries_per_experience=args.max_retries_per_experience,
        evolution_scope=args.evolution_scope,
        candidates_per_round=args.candidates_per_round,
        icon_mode=args.icon_mode,
        method_mode=args.method_mode,
        reference_mode=args.reference_mode,
        icon_candidates=args.icon_candidates,
        evolution_enabled=args.evolution == "on",
        judge_mode=args.judge_mode,
        judge_model=args.judge_model,
        online_judge_model=args.online_judge_model,
        frozen_judge_model=args.frozen_judge_model,
        judge_registry=args.registry,
        judge_model_id=args.model_id,
        domain_profile=args.domain_profile,
        experience_enabled=not args.no_experience,
        experience_library=args.experience_library,
        retrieval_statuses=args.retrieval_statuses,
        structure_mode=args.structure_mode,
        text_plan_mode=args.text_plan_mode,
        ppt_plan_mode=args.ppt_plan_mode,
        structure_model=args.structure_model,
        text_plan_model=args.text_plan_model,
        ppt_plan_model=args.ppt_plan_model,
        color_theme_mode=args.color_theme_mode,
        color_theme_model=args.color_theme_model,
        vlm_timeout=args.vlm_timeout,
        vlm_retries=args.vlm_retries,
        reference_library=args.reference_library,
        reference_top_k=args.reference_top_k,
        exemplar_guidance=args.exemplar_guidance == "on",
        graph_context=args.graph_context == "on",
        middle_layer=args.middle_layer,
        figure_target=args.figure_target,
        visual_review_mode=args.visual_review_mode,
        visual_review_model=args.visual_review_model,
        replay_upstream_from=args.replay_upstream_from,
        fork_before=args.fork_before,
        reuse_replayed_icons=args.reuse_replayed_icons,
        upstream_from=args.upstream_from,
        baseline_from=args.baseline_from,
        bulk_figures=args.bulk_figures,
        upstream_retry_enabled=not args.no_upstream_retry,
        acceptance_checkpoint=args.acceptance_checkpoint,
        acceptance_base_model=args.acceptance_base_model,
        online_judge_checkpoint=args.online_judge_checkpoint,
        online_judge_base_model=args.online_judge_base_model,
        judge_holdout_paths=tuple(args.judge_holdout or ()),
    )


def _run_judge(args: argparse.Namespace) -> dict:
    if args.judge_action == "register":
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        result = register_checkpoint(args.checkpoint, manifest, args.registry)
        result["ok"] = True
        return result
    if args.judge_action == "evaluate":
        return evaluate_predictions(args.dataset, args.predictions, checkpoint=args.checkpoint, suite=args.suite, suite_manifest=args.suite_manifest, report_out=args.report_out, strict_gates=args.strict_gates)
    if args.judge_action == "shadow":
        return compare_shadow_predictions(args.dataset, args.predictions, args.bootstrap_predictions, checkpoint=args.checkpoint, suite_manifest=args.suite_manifest, report_out=args.report_out, strict_gates=args.strict_gates)
    if args.judge_action == "train":
        from .judge_training.training import TrainingConfig, train_dpo
        fields = ("backend", "base_model", "beta", "learning_rate", "epochs", "gradient_accumulation", "max_length", "image_max_pixels", "lora_r", "lora_alpha", "lora_dropout", "load_in_4bit", "gradient_checkpointing", "shard_across_gpus", "seed", "max_steps", "save_every", "resume_from", "init_from", "validation_dataset", "eval_limit", "early_stop_margin", "objective", "generate_eval_limit", "selection")
        overrides = {key: getattr(args, key) for key in fields if getattr(args, key) is not None}
        if args.config:
            config = TrainingConfig.from_file(args.config, dataset=args.dataset, out_dir=args.out, **overrides)
        else:
            config = TrainingConfig(dataset=args.dataset, out_dir=args.out, **overrides)
        return train_dpo(config)
    if args.judge_action == "score-choice":
        from .judge_training.choice_items import evaluate_choice_records
        from .judge_training.training import TrainingConfig, build_backend, load_training_records
        from .utils import write_json

        records = load_training_records(args.dataset)
        if args.config:
            config = TrainingConfig.from_file(args.config, dataset=args.dataset, out_dir=str(Path(args.out).parent), base_model=args.base_model)
        else:
            config = TrainingConfig(dataset=args.dataset, out_dir=str(Path(args.out).parent), base_model=args.base_model, objective="sft", selection="choice", load_in_4bit=True)
        policy = build_backend(config)
        if args.checkpoint:
            policy.load(args.checkpoint)
        report = evaluate_choice_records(policy, records, max_new_tokens=int(config.smoke_max_new_tokens), predictions_out=(args.predictions_out or None))
        report["ok"] = not report.get("skipped")
        write_json(args.out, report)
        return report
    if args.judge_action == "gate-choice":
        from .judge_training.choice_gate import build_choice_gate_report
        thresholds: dict[str, float] = {}
        for item in args.threshold or []:
            name, _, value = str(item).partition("=")
            if not name or not _:
                raise SystemExit(f"--threshold expects NAME=VALUE, got {item!r}")
            thresholds[name] = float(value)
        return build_choice_gate_report(
            predictions=args.predictions, dataset=args.dataset, checkpoint=args.checkpoint,
            model_id=args.model_id, dataset_version=args.dataset_version, git_commit=args.git_commit,
            thresholds=thresholds, report_out=args.report_out, strict_gates=args.strict_gates,
        )
    if args.judge_action == "promote":
        from .judge_training.judge_schema import GATE_DIAGNOSIS_TYPES
        from .judge_training.promotion import build_promotion_report
        thresholds = {name: args.class_threshold for name in GATE_DIAGNOSIS_TYPES} if args.class_threshold is not None else None
        return build_promotion_report(dataset=args.dataset, predictions=args.predictions, suite_manifest=args.suite_manifest,
                                      suite=args.suite, checkpoint=args.checkpoint, baseline_predictions=args.baseline_predictions,
                                      bootstrap_predictions=args.bootstrap_predictions, training_report=args.training_report,
                                      class_thresholds=thresholds, adversarial_threshold=args.adversarial_threshold,
                                      report_out=args.report_out, strict_gates=args.strict_gates)
    if args.judge_action == "baseline":
        return run_baseline(args.dataset, args.out, limit=args.limit, model=args.model, timeout=args.timeout, retries=args.retries)
    if args.judge_action == "sweep-memory":
        from .judge_training.registry import checkpoint_identity, resolve_checkpoint
        from .workflows import sweep_memory_with_judge

        model = str(args.judge_model or "")
        checkpoint = ""
        if args.model_id:
            if not args.registry:
                raise ValueError("--model-id needs --registry to resolve the Judge it names")
            resolved = resolve_checkpoint(args.model_id, args.registry)
            model = str(resolved.get("model_id") or args.model_id)
            checkpoint = checkpoint_identity(resolved)
        return sweep_memory_with_judge(
            experience_library=args.experience_library,
            incoming={"role": args.judge_role, "mode": args.judge_mode, "trained": not args.bootstrap_judge, "model": model, "checkpoint": checkpoint},
            selection=args.selection,
            out=args.out,
            dry_run=not args.apply,
            revalidate=args.revalidate,
            limit=args.limit,
        )
    if args.judge_action == "handover":
        from .workflows import run_judge_handover

        return run_judge_handover(
            out=args.out,
            dataset=args.dataset,
            suite_manifest=args.suite_manifest,
            model_id=args.model_id,
            dataset_version=args.dataset_version,
            registry=args.registry,
            checkpoint=args.checkpoint,
            training_report=args.training_report,
            predict_from=args.predict_from,
            bootstrap_predictions=args.bootstrap_predictions,
            no_shadow=args.no_shadow,
            allow_experimental=args.allow_experimental,
            judge_mode=args.judge_mode,
            judge_model=args.judge_model,
            backend=args.backend,
            experience_library=args.experience_library,
            sweep_selection=args.sweep_selection,
            sweep_limit=args.sweep_limit,
            git_commit=args.git_commit,
            force=args.force,
            dry_run=args.dry_run,
        )
    return run_bakeoff(args.dataset, args.out, models=args.models, limit=args.limit, timeout=args.timeout, retries=args.retries)


def _run_judge_data(args: argparse.Namespace) -> dict:
    if args.judge_data_action == "freeze-source":
        from .contracts.source_snapshot import SCOPE_TIERS
        from .judge_training.source_snapshot import freeze_source_snapshot
        return freeze_source_snapshot(args.batch_root, args.out, args.archive_root,
                                      workers=args.workers, preflight=args.preflight,
                                      repair_bindings=args.repair_source_bindings,
                                      tiers=SCOPE_TIERS[args.scope])
    if args.judge_data_action == "validate-source-snapshot":
        from .judge_training.source_snapshot_validation import validate_source_snapshot
        return validate_source_snapshot(args.snapshot, args.archive_root)
    if args.judge_data_action == "render-figure-pages":
        from .judge_training.figure_rendering import render_pending_source_figures
        return render_pending_source_figures(args.batch_root, domains=args.domains, dpi=args.dpi,
                                             limit=args.limit, dry_run=args.dry_run)
    if args.judge_data_action == "recover-source-pdf":
        from .judge_training.source_pdf_recovery import recover_official_source_pdfs
        return recover_official_source_pdfs(args.batch_root, workers=args.workers,
                                            channels=tuple(args.channels), limit=args.limit)
    if args.judge_data_action == "collect-papers":
        return guarded_collection(
            collect_literature_papers,
            args.batch_root,
            year=args.year,
            stage=args.stage,
            domains=args.domains,
            resume=args.resume,
            domain_workers=args.domain_workers,
            paper_workers=args.paper_workers,
            asset_policy=args.asset_policy,
            legacy_c_policy=args.legacy_c_policy,
            canary_per_source=args.canary_per_source,
        )
    if args.judge_data_action == "source-manifest":
        from .judge_training.preference_sources import write_source_manifest
        result = write_source_manifest(
            args.source,
            args.out,
            root=args.root,
            figures=args.figures,
            structured=args.structured,
            domain_index=args.domain_index,
        )
        result["ok"] = bool(result.get("ok", True))
        return result
    if args.judge_data_action == "build-real-pairs":
        from .judge_training.preference_sources import open_generation_source, open_middleware_source
        from .judge_training.real_vs_generated import build_real_vs_generated_examples
        return build_real_vs_generated_examples(
            open_middleware_source(args.middleware),
            open_generation_source(args.run),
            args.out,
            split=args.split,
            include_swapped=args.include_swapped,
            limit=args.limit,
            include_passed=args.include_passed,
        )
    if args.judge_data_action == "build-real-pairs-wave":
        from .judge_training.real_vs_generated_wave import collect_reference_pairs
        return collect_reference_pairs(
            args.subset,
            args.middleware,
            args.wave_root,
            args.out,
            attempt=args.attempt,
            include_swapped=args.include_swapped,
            include_passed=args.include_passed,
            limit=args.limit,
            merge_into=args.merge_into,
        )
    if args.judge_data_action == "map-icon-reuse":
        from .judge_training.icon_reuse import map_corpus
        return map_corpus(
            args.contracts,
            args.legacy_root,
            args.out,
            limit=args.limit,
            adjudicate=args.adjudicate,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            workers=args.workers,
            stage_into=args.stage_into,
            attempt=args.legacy_attempt,
            round_index=args.round_index,
            candidate=args.candidate,
        )
    if args.judge_data_action == "stage-icon-reuse":
        from .judge_training.icon_reuse import stage_from_mapping
        return stage_from_mapping(
            args.mapping,
            args.stage_into,
            routes=[route.strip() for route in str(args.routes).split(",") if route.strip()],
            attempt=args.attempt,
            round_index=args.round_index,
            candidate=args.candidate,
            force=args.force,
            dry_run=args.dry_run,
        )
    if args.judge_data_action == "build-contract-subset":
        from .workflows.corpus_subset import build_contract_subset
        return build_contract_subset(
            args.snapshot,
            contracts=args.contracts,
            legacy_root=args.legacy_root,
            out=args.out,
            subset_id=args.subset_id,
            split=args.split,
            names_out=args.names_out,
            exclude_drawn=args.exclude_drawn,
            mapping=args.mapping,
            legacy_attempt=args.legacy_attempt,
            require_legacy_icons=not getattr(args, "allow_no_reusable_icon", False),
            limit=args.limit,
        )
    if args.judge_data_action == "build":
        result = build_dataset(args.runs, args.out, dataset_version=args.dataset_version)
        result["ok"] = not bool(result.get("errors"))
        return result
    if args.judge_data_action == "controlled-negatives":
        from .judge_training.controlled_negatives import build_controlled_negatives
        result = build_controlled_negatives(args.runs, args.out, round_index=args.round, scripts=args.scripts, split=args.split, limit=args.limit)
        result["ok"] = bool(result.get("rounds"))
        return result
    if args.judge_data_action == "validate":
        return validate_dataset(args.dataset)
    if args.judge_data_action == "audit":
        from .judge_training.dataset_audit import audit_dataset
        result = audit_dataset(args.dataset, out=args.out)
        result["ok"] = result["clean"] if args.strict else True
        return result
    from .judge_training.split_ledger import load_split_ledger
    if args.judge_data_action == "select-pilot":
        from .judge_training.pilot_selection import (
            DEFAULT_PILOT_ID,
            DOMAIN_QUOTAS,
            build_pilot_payloads,
            candidate_pool,
            caption_fallback_labels,
            label_candidates,
            load_figure_index,
            load_label_cache,
            load_snapshot_papers,
            make_structure_gate_runner,
            select_pilot_verified,
            write_pilot_version,
        )
        pilot_id = args.pilot_id or DEFAULT_PILOT_ID
        seed = args.seed or pilot_id
        split_rows = load_split_ledger(args.split)
        papers = load_snapshot_papers(args.snapshot)
        figure_index = load_figure_index(args.snapshot)
        pool = candidate_pool(
            split_rows,
            papers,
            seed=seed,
            quotas=DOMAIN_QUOTAS,
            pool_multiplier=args.pool_multiplier,
        )
        candidates = pool["candidates"]
        if args.no_vlm:
            cache = caption_fallback_labels(candidates, figure_index)
            labeller_name = "caption_fallback"
            label_report = {
                "labeller": labeller_name,
                "candidates": len(candidates),
                "labelled": len(cache),
                "newly_labelled": 0,
                "failures": [],
                "cache": None,
            }
        else:
            labels_path = (
                Path(args.labels)
                if args.labels
                else Path(args.out).parent / "_labels" / f"{pilot_id}.labels.jsonl"
            )
            work_dir = (
                Path(args.work_dir)
                if args.work_dir
                else Path(args.out).parent / "_work" / pilot_id
            )
            label_report = label_candidates(
                candidates,
                args.snapshot,
                figure_index,
                cache_path=labels_path,
                work_dir=work_dir,
                workers=args.workers,
                dpi=args.dpi,
            )
            cache = load_label_cache(labels_path)
            labeller_name = "vlm"
            label_report["candidates"] = len(candidates)
            label_report["labeller"] = labeller_name
        labelled = [
            cache[str(item["paper_id"])]
            for item in candidates
            if str(item["paper_id"]) in cache
        ]
        structure_gate = {"enabled": False}
        if labeller_name == "vlm" and not args.no_structure_gate:
            gate_path = (
                Path(args.structure_gate)
                if args.structure_gate
                else Path(args.out).parent / "_labels" / f"{pilot_id}.structure-gate.jsonl"
            )
            verified = select_pilot_verified(
                labelled,
                seed=seed,
                quotas=DOMAIN_QUOTAS,
                gate=load_label_cache(gate_path),
                gate_runner=make_structure_gate_runner(
                    candidates,
                    args.snapshot,
                    figure_index,
                    cache_path=gate_path,
                    work_dir=work_dir,
                    workers=max(2, args.workers),
                ),
            )
            labelled = verified["labelled"]
            gate_rows = verified["gate"]
            structure_gate = {
                "enabled": True,
                "cache": str(gate_path),
                "asked": len(verified["checked"]),
                "checked": len(gate_rows),
                "rejected": sorted(
                    str(paper_id)
                    for paper_id, row in gate_rows.items()
                    if row.get("is_diagram") is False
                ),
                "rule": "a figure labelled framework, pipeline or technical route is demoted "
                "to other when the second pass says it is not a schematic",
            }
        label_report["structure_gate"] = structure_gate
        payloads = build_pilot_payloads(
            args.split,
            args.snapshot,
            labelled=labelled,
            pool=candidates,
            figure_index=figure_index,
            pilot_id=pilot_id,
            seed=seed,
            quotas=DOMAIN_QUOTAS,
            label_report=label_report,
            labeller_name=labeller_name,
        )
        written = write_pilot_version(args.out, payloads)
        return {
            "ok": bool(payloads["validation"]["ok"]),
            "pilot_id": pilot_id,
            "papers": len(payloads["records"]),
            "candidate_pool": len(candidates),
            "labelled": len(labelled),
            "labeller": labeller_name,
            "label_failures": len(label_report.get("failures") or []),
            "by_split": payloads["manifest"]["by_split"],
            "by_domain": payloads["coverage"]["by_split"]["train"]["by_domain"],
            "shortfalls": payloads["validation"]["shortfalls"],
            "asset_check": payloads["validation"]["asset_check"],
            "sealed": written["sealed"],
            "out_dir": written["out_dir"],
            "files": written["files"],
        }
    if args.judge_data_action == "validate-pilot":
        from .judge_training.pilot_selection import validate_pilot_version
        return validate_pilot_version(args.pilot)
    if args.judge_data_action == "build-pilot-graphs":
        from .judge_training.pilot_graphs import (
            BILLABLE_STEPS,
            CONTRACT_STEP,
            DEFAULT_GRAPH_ID,
            BIND_STEP,
            CHECK_STEP,
            INDEX_STEP,
            REVIEW_STEP,
            VISUAL_STEP,
            build_pilot_graph_payloads,
            write_pilot_graph_version,
        )
        payloads = build_pilot_graph_payloads(
            args.snapshot,
            args.pilot,
            graph_id=args.graph_id or DEFAULT_GRAPH_ID,
            stages=args.stages,
            limit=args.limit,
            model=args.model,
            timeout_seconds=args.timeout,
            retries=args.retries,
            workers=args.workers,
            resume_dir=args.out if args.resume else None,
            figure_dir=Path(args.out) / "figures",
            figure_dpi=args.figure_dpi,
            max_long_side=args.max_long_side,
            probe_top_k=args.probe_top_k,
            probe_limit=args.probe_limit,
            embedding_model=args.embedding_model,
            strict=args.strict_review,
        )
        written = write_pilot_graph_version(args.out, payloads)
        provider_rows = [row for row in payloads["ledger"] if str(row.get("step") or "") in BILLABLE_STEPS]
        provider_tokens = {
            step: sum(int(row.get("total_tokens") or 0) for row in provider_rows if row.get("step") == step)
            for step in BILLABLE_STEPS
            if any(row.get("step") == step for row in payloads["ledger"])
        }
        graphs = payloads["graphs"] or []
        visuals = payloads["visuals"] or []
        return {
            "ok": bool(payloads["validation"]["ok"]),
            "graph_id": payloads["manifest"]["graph_id"],
            "papers": len(payloads["reports"]),
            "stages_requested": payloads["stages_requested"],
            "stages_completed": payloads["stages_completed"],
            "checks": payloads["validation"]["checks"],
            "human_review_required": len(payloads["validation"]["human_review_required"]),
            "estimated_review_tokens": sum(int(row.get("estimated_tokens") or 0) for row in payloads["ledger"]),
            "reviews": len(payloads["reviews"]),
            "review_tokens": provider_tokens.get(REVIEW_STEP, 0),
            "review_token_source": sorted({str(row.get("token_source")) for row in provider_rows if row.get("step") == REVIEW_STEP}),
            "contracts": len(payloads["contracts"]),
            "contract_tokens": provider_tokens.get(CONTRACT_STEP, 0),
            "visuals": len(visuals),
            "visual_tokens": provider_tokens.get(VISUAL_STEP, 0),
            "visual_panels": sum(int((entry.get("visual_graph_report") or {}).get("panel_count") or 0) for entry in visuals),
            "visual_blocks": sum(int((entry.get("visual_graph_report") or {}).get("block_count") or 0) for entry in visuals),
            "visual_edges": sum(int((entry.get("visual_graph_report") or {}).get("edge_count") or 0) for entry in visuals),
            "figure_dpi": args.figure_dpi,
            "bindings": len(payloads["bindings"]),
            "bound_nodes": sum(int((entry.get("binding_report") or {}).get("bound_nodes") or 0) for entry in payloads["bindings"]),
            "bound_edges": sum(int((entry.get("binding_report") or {}).get("bound_edges") or 0) for entry in payloads["bindings"]),
            "claim_coverage": (payloads["coverage"].get("bindings") or {}).get("claim_coverage"),
            "diagnostic_claim_coverage_all_objects": (payloads["coverage"].get("bindings") or {}).get("diagnostic_claim_coverage_all_objects"),
            "diagnostic_required_node_coverage": (payloads["coverage"].get("bindings") or {}).get("diagnostic_required_node_coverage"),
            "reversed_edges": (payloads["coverage"].get("bindings") or {}).get("reversed_edges"),
            "provider_tokens_bind": provider_tokens.get(BIND_STEP, 0),
            "index_records": int((payloads["coverage"].get("index") or {}).get("records") or 0),
            "index_papers": int((payloads["coverage"].get("index") or {}).get("papers") or 0),
            "index_motifs": int((payloads["coverage"].get("index") or {}).get("motifs") or 0),
            "index_subgraph_locations": int((payloads["coverage"].get("index") or {}).get("subgraph_locations") or 0),
            "index_probe_queries": int(((payloads["coverage"].get("index") or {}).get("probe") or {}).get("queries") or 0),
            "index_cross_domain_hit_rate": ((payloads["coverage"].get("index") or {}).get("probe") or {}).get("cross_domain_hit_rate"),
            "index_hidden_test_records": int((payloads["coverage"].get("index") or {}).get("hidden_test_records") or 0),
            "provider_tokens_index": provider_tokens.get(INDEX_STEP, 0),
            "checks": len(payloads["checks"]),
            "check_tokens": provider_tokens.get(CHECK_STEP, 0),
            "check_confirmed": int((payloads["coverage"].get("correspondence") or {}).get("confirmed") or 0),
            "check_missed_by_reader": int((payloads["coverage"].get("correspondence") or {}).get("missed_by_reader") or 0),
            "check_hallucinated_block": int((payloads["coverage"].get("correspondence") or {}).get("hallucinated_block") or 0),
            "check_direction_conflicts": int((payloads["coverage"].get("correspondence") or {}).get("direction_conflicts") or 0),
            "check_mean_required_visible_rate": (payloads["coverage"].get("correspondence") or {}).get("mean_required_visible_rate"),
            "graphs": len(graphs),
            "scientific_nodes": sum(int((entry.get("scientific_graph_report") or {}).get("node_count") or 0) for entry in graphs),
            "scientific_edges": sum(int((entry.get("scientific_graph_report") or {}).get("edge_count") or 0) for entry in graphs),
            "provider_tokens": provider_tokens,
            "provider_token_total": sum(provider_tokens.values()),
            "cost_projection": payloads["costs"]["projection"],
            "failures": payloads["validation"]["failures"],
            "sealed": written["sealed"],
            "out_dir": written["out_dir"],
            "files": written["files"],
        }
    if args.judge_data_action == "validate-pilot-graphs":
        from .judge_training.pilot_graphs import validate_pilot_graph_version
        return validate_pilot_graph_version(args.graphs)
    if args.judge_data_action == "export-contracts":
        from .judge_training.contract_export import export_version_contracts
        manifest = export_version_contracts(
            args.graphs,
            args.out,
            snapshot=args.snapshot,
            paper_ids=args.papers,
            limit=args.limit,
        )
        return {
            "ok": bool(manifest["ok"]),
            "summary": manifest["summary"],
            "source_version": manifest["source_version"],
            "out_root": manifest["out_root"],
            "papers": manifest["papers"],
            "exported": manifest["exported"],
            "failed": manifest["failed"],
            "with_figure_content_plan": manifest["with_figure_content_plan"],
            "inventories_derived": manifest["inventories_derived"],
            "inventories_verified": manifest["inventories_verified"],
            "failures": manifest["failures"],
            "manifest": str(Path(args.out) / "contract_export_manifest.json"),
        }
    if args.judge_data_action == "perceptual-hash":
        from .judge_training.perceptual_hash import (
            build_perceptual_hash_report,
            write_perceptual_hash_report,
        )
        report = build_perceptual_hash_report(
            args.snapshot,
            dpi=args.dpi,
            workers=args.workers,
            near_distance=args.near_distance,
        )
        digest = write_perceptual_hash_report(args.out, report)
        return {
            "ok": bool(report["ok"]),
            "out": args.out,
            "sha256": digest,
            "figures": report["figures_total"],
            "hashed": report["hashed"],
            "failed": report["failed"],
            "distinct_hashes": report["distinct_hashes"],
            "exact_groups": report["exact_groups_total"],
            "near_pairs": report["near_pairs_total"],
        }
    if args.judge_data_action == "split":
        from .judge_training.split_ledger import DEFAULT_SEED, build_split_payloads, write_split_version
        split_id = args.split_id or DEFAULT_SEED
        seed = args.seed or split_id
        report = None
        if args.perceptual_hash_report:
            report = json.loads(Path(args.perceptual_hash_report).read_text(encoding="utf-8"))
        payloads = build_split_payloads(
            args.snapshot,
            seed=seed,
            ratios=args.ratios,
            split_id=split_id,
            perceptual_hash_report=report,
        )
        written = write_split_version(args.out, payloads, perceptual_hash=report)
        return {
            "ok": bool(payloads["validation"]["ok"]),
            "split_id": split_id,
            "seed": seed,
            "papers": payloads["distribution"]["total"],
            "split_counts": payloads["distribution"]["totals"],
            "split_ratios": payloads["distribution"]["ratios"],
            "empty_cells": payloads["distribution"]["empty_cells"],
            "leakage_ok": bool(payloads["leakage"]["ok"]),
            "figure_groups_bound": payloads["manifest"]["figure_groups_bound"],
            "seal_gates": payloads["seal_gates"],
            "sealed": written["sealed"],
            "out_dir": written["out_dir"],
            "files": written["files"],
        }
    if args.judge_data_action == "validate-split":
        from .judge_training.split_ledger import validate_split_version
        return validate_split_version(args.split)
    if args.judge_data_action == "freeze-suite":
        result = build_suite_manifest(args.dataset, args.out, suite=args.suite, dataset_version=args.dataset_version)
        result["ok"] = True
        return result
    if args.judge_data_action == "slice-dpo":
        from .judge_training.dpo_slices import parse_sizes, slice_mixed_dpo

        return slice_mixed_dpo(args.dataset, args.out, exam=args.exam, sizes=parse_sizes(args.sizes), val_size=args.val_size, seed=args.seed)
    if args.judge_data_action == "build-choice":
        from .judge_training.choice_items import build_choice_slices
        from .judge_training.dpo_slices import parse_sizes

        return build_choice_slices(
            {"corruption": args.corruption, "rag": args.rag, "sealed": args.sealed, "v002": args.v002},
            args.out,
            exam=args.exam,
            sizes=parse_sizes(args.sizes),
        )
    if args.judge_data_action == "ground-dpo":
        from .judge_training.grounded_judgements import ground_dpo_export

        return ground_dpo_export(args.dataset, args.out, min_unique_ratio=args.min_unique_ratio)
    if args.judge_data_action == "build-subgraph-pairs":
        from .judge_training.subgraph_pairs import build_subgraph_pairs

        return build_subgraph_pairs(
            args.middleware,
            args.out,
            split=args.split,
            limit=args.limit,
            per_domain=args.per_domain,
            include_swapped=args.include_swapped,
            recall_limit=args.recall_limit,
            same_layer_only=args.same_layer_only,
        )
    if args.judge_data_action == "build-preference-pairs":
        from .judge_training.preference_pairs import build_preference_pairs
        return build_preference_pairs(args.dataset, args.out, families=tuple(args.families), include_swapped=args.include_swapped, limit=args.limit)
    if args.judge_data_action == "review-queue":
        from .judge_training.preference_pairs import build_review_queue
        return build_review_queue(args.pairs, args.out, pairs_per_sheet=args.pairs_per_sheet, limit=args.limit)
    if args.judge_data_action == "mark-reviewed":
        from .judge_training.preference_pairs import apply_human_review
        return apply_human_review(args.pairs, args.decisions, args.out, reviewer=args.reviewer)
    if args.judge_data_action == "build-corruption-source":
        from .judge_training.corruption_wave import build_corruption_source

        result = build_corruption_source(args.middleware, args.out, snapshot=args.snapshot, split_ledger=args.split_ledger, paper_ids=list(args.paper_id or []), limit=args.limit or None, label=args.label)
        result["ok"] = bool(result.get("ok"))
        return result
    if args.judge_data_action == "run-local-corruptions":
        from .judge_training.corruption_wave import run_local_corruption_wave

        return run_local_corruption_wave(args.source, args.out, per_type=args.per_type, review=args.review, ocr_engine=args.ocr_engine, workers=args.workers, batch_size=args.batch_size, model=args.model, timeout=args.timeout, retries=args.retries)
    result = export_training_data(args.dataset, args.out, format=args.format, require_human_audit=args.require_human_audit)
    if args.format == "dpo":
        return result
    result["ok"] = bool(result.get("records")) and all(not item.get("missing_images") for item in result.get("splits", {}).values())
    return result


def _run_session(args: argparse.Namespace) -> dict:
    from .workflows import run_evolution_session

    return run_evolution_session(
        config=args.config,
        out=args.out,
        validation_suite=args.validation_suite or "",
        experience_library=args.experience_library or "",
        hours=args.hours,
        rate_limit_cooldown=args.rate_limit_cooldown,
        rate_limit_retries=args.rate_limit_retries,
        validate=not args.no_validate,
        dry_run=args.dry_run,
        training_export=not args.no_training_export,
        export_split=args.export_split or "",
        acceptance_checkpoint=args.acceptance_checkpoint,
        acceptance_base_model=args.acceptance_base_model,
        judge_holdout_paths=tuple(args.judge_holdout or ()),
    )


def _run_continuous(args: argparse.Namespace) -> dict:
    from .workflows import run_continuous_evolution

    return run_continuous_evolution(
        config=args.config,
        out=args.out,
        experience_library=args.experience_library or "",
        stop_file=args.stop_file or "",
        max_waves=int(args.max_waves or 0),
    )


def _corpus_progress(row: dict) -> None:
    """Print one line per finished attempt so a running batch can be watched."""
    print(
        f"[{row.get('index')}/{row.get('of')}] {row.get('paper_id')} "
        f"attempt {row.get('attempt')} -> {row.get('status')} ({row.get('category')}) "
        f"{row.get('elapsed_seconds')}s"
        + (" - stopping here: the provider says the account is out of quota" if row.get("halted_reason") else ""),
        flush=True,
    )


def _run_paper_evolve_corpus(args: argparse.Namespace) -> dict:
    """Sample a corpus subset, resolve its papers, and run it under supervision."""
    from .workflows.supervised_batch import make_paper_evolve_runner, run_corpus_batch

    shard = _parse_shard(getattr(args, "shard", None))
    if shard is None and int(getattr(args, "workers", 1) or 1) > 1:
        return _run_corpus_shards(args)

    pdf_roots = list(args.pdf_roots or []) or [str(Path.cwd() / "datasets" / "paper_collection")]
    runner = make_paper_evolve_runner(
        reference_library=args.reference_library,
        icon_mode=args.icon_mode,
        vlm_timeout=args.vlm_timeout,
        vlm_retries=args.vlm_retries,
        bulk_figures=bool(getattr(args, "bulk_figures", False)),
        contracts_from=getattr(args, "contracts_from", None),
        frozen_reground_min_ratio=getattr(args, "min_reground_ratio", None),
    )
    return run_corpus_batch(
        snapshot=args.snapshot,
        split_dir=args.split,
        out_root=args.out,
        pdf_roots=pdf_roots,
        archive_root=getattr(args, "archive_root", None),
        subset=args.subset,
        subset_id=args.subset_id,
        per_domain=args.per_domain,
        label_cache=args.label_cache,
        exclude_roots=tuple(getattr(args, "exclude_drawn_root", None) or []),
        sample_only=args.sample_only,
        limit=args.limit,
        shard=shard,
        runner=runner,
        on_attempt=_corpus_progress,
        resume=bool(getattr(args, "resume", False)),
        reuse_upstream=not bool(getattr(args, "fresh_retries", False)),
        keep_first_draft=bool(getattr(args, "keep_first_draft", False)),
        contracts_from=getattr(args, "contracts_from", None),
    )


def _parse_shard(value: str | None) -> tuple[int, int] | None:
    """Read an INDEX/COUNT shard selector."""
    if not value:
        return None
    index, _, total = str(value).partition("/")
    return (int(index), int(total))


def _run_corpus_shards(args: argparse.Namespace) -> dict:
    """Run the subset as independent worker processes and merge what they record.

    Every worker resolves the same paper list and takes every Nth paper, so the
    split needs no shared state.  Each writes its own attempt ledger and the
    parent folds them into one batch summary.
    """
    from .workflows.supervised_batch import merge_shard_ledgers, run_corpus_batch

    pdf_roots = list(args.pdf_roots or []) or [str(Path.cwd() / "datasets" / "paper_collection")]
    prepared = run_corpus_batch(
        snapshot=args.snapshot,
        split_dir=args.split,
        out_root=args.out,
        pdf_roots=pdf_roots,
        archive_root=getattr(args, "archive_root", None),
        subset=args.subset,
        subset_id=args.subset_id,
        per_domain=args.per_domain,
        label_cache=args.label_cache,
        exclude_roots=tuple(getattr(args, "exclude_drawn_root", None) or []),
        sample_only=True,
    )
    if getattr(args, "sample_only", False):
        # A preview that starts paid workers is not a preview: the caller asked
        # for the subset, and the subset is what comes back.
        return prepared
    subset_dir = str(prepared["subset_dir"])
    total = max(1, int(args.workers))
    children = []
    for index in range(total):
        argv = [
            sys.executable, "-m", "rfs.cli", "paper-evolve-corpus",
            "--snapshot", str(args.snapshot),
            "--split", str(args.split),
            "--out", str(args.out),
            "--subset", subset_dir,
            "--shard", f"{index}/{total}",
            "--per-domain", str(args.per_domain),
            "--icon-mode", str(args.icon_mode),
            "--vlm-timeout", str(args.vlm_timeout),
            "--vlm-retries", str(args.vlm_retries),
            "--workers", "1",
        ]
        if args.resume:
            argv.append("--resume")
        if getattr(args, "bulk_figures", False):
            argv.append("--bulk-figures")
        if getattr(args, "fresh_retries", False):
            argv.append("--fresh-retries")
        if getattr(args, "keep_first_draft", False):
            argv.append("--keep-first-draft")
        if args.limit:
            argv += ["--limit", str(args.limit)]
        if args.reference_library:
            argv += ["--reference-library", str(args.reference_library)]
        if getattr(args, "contracts_from", None):
            argv += ["--contracts-from", str(args.contracts_from)]
        if getattr(args, "min_reground_ratio", None) is not None:
            argv += ["--min-reground-ratio", str(args.min_reground_ratio)]
        for root in getattr(args, "exclude_drawn_root", None) or []:
            argv += ["--exclude-drawn-root", str(root)]
        for root in args.pdf_roots or []:
            argv += ["--pdf-root", str(root)]
        if getattr(args, "archive_root", None):
            argv += ["--archive-root", str(args.archive_root)]
        children.append(subprocess.Popen(argv))
    exit_codes = [child.wait() for child in children]
    merged = merge_shard_ledgers(Path(args.out) / "runs", batch_id=str(prepared.get("subset_id") or "batch"))
    merged.update({"workers": total, "worker_exit_codes": exit_codes, "subset_dir": subset_dir})
    return merged


def _run_deck_adoption(args: argparse.Namespace) -> dict:
    """Re-adjudicate an earlier batch's undelivered decks and copy them forward."""
    from .workflows.corpus_subset import load_subset
    from .workflows.deck_adoption import adopt_undelivered_decks

    subset = load_subset(args.subset)
    return adopt_undelivered_decks(sources=args.sources, out_root=args.out, entries=subset.get("papers") or [])


def _benchmark_layers(args: argparse.Namespace) -> tuple[str, ...]:
    """Read the requested metric layers, rejecting unknown ones."""
    requested = [item.strip().upper() for item in str(getattr(args, "layers", "") or "").split(",") if item.strip()]
    unknown = [item for item in requested if item not in BENCHMARK_LAYERS]
    if unknown:
        raise ValueError(f"unknown benchmark layers: {', '.join(unknown)}")
    return tuple(requested) or BENCHMARK_LAYERS


def _run_benchmark_prepare(args: argparse.Namespace) -> dict:
    gold_root = args.out or "datasets/benchmark/gold-v001"
    if getattr(args, "verify", False):
        return verify_gold_instances(gold_root)
    overrides = {
        "papers_path": args.papers,
        "figures_path": args.figures,
        "reviews_path": args.reviews,
        "contracts_dir": args.contracts,
        "pdf_root": args.pdf_root,
        "categories_path": getattr(args, "categories_csv", None),
    }
    label_groups = tuple(part.strip() for part in str(getattr(args, "label_groups", "") or "").split(",") if part.strip())
    if not label_groups:
        label_groups = DEFAULT_LABEL_GROUPS
    unknown = [group for group in label_groups if group not in DEFAULT_LABEL_GROUPS]
    if unknown:
        raise ValueError("unknown label group(s): " + ", ".join(unknown))
    requested_splits = tuple(
        part.strip() for part in str(getattr(args, "splits", "") or "").split(",") if part.strip()
    )
    if requested_splits == ("all",):
        requested_splits = ALL_SPLITS
    include_splits = requested_splits or HELD_OUT_SPLITS
    unknown_splits = [name for name in include_splits if name not in ALL_SPLITS]
    if unknown_splits:
        raise ValueError("unknown split(s): " + ", ".join(unknown_splits))
    return prepare_gold_instances(
        gold_root,
        target=args.target,
        force=args.force,
        label_groups=label_groups,
        include_splits=include_splits,
        **{key: value for key, value in overrides.items() if value},
    )


def _run_benchmark_run(args: argparse.Namespace) -> dict:
    missing = [name for name in ("instances", "systems", "out") if not getattr(args, name)]
    if missing:
        raise ValueError(f"benchmark run requires: {', '.join('--' + name for name in missing)}")
    return run_benchmark_instances(
        instances_dir=args.instances,
        systems_dir=args.systems,
        out_dir=args.out,
        layers=_benchmark_layers(args),
        system_id=args.system_id,
        limit=args.limit,
    )


def _id_set(path: str | None) -> set[str]:
    if not path:
        return set()
    file_path = Path(path)
    if not file_path.is_file():
        return set()
    return {line.strip() for line in file_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}


def _run_benchmark_export_manifest(args: argparse.Namespace) -> dict:
    ledger = args.ledger or "datasets/judge_splits/paper-d0-split-v001/split_ledger.jsonl"
    reviews = [args.reviews or "datasets/judge-framework-review/human_framework_reviews.jsonl"]
    reviews.extend(args.extra_reviews or [])
    return export_public_benchmark_manifest(
        ledger_path=ledger,
        snapshot=args.snapshot or "datasets/judge_source_snapshots/paper-d0-v001",
        reviews_path=reviews,
        framework_papers_path=args.framework_papers or "datasets/judge-framework-v001/papers.jsonl",
        out_dir=args.out or "benchmarks/editable-figure-bench/manifest",
        sciflow_ids=args.sciflow_ids,
        figurebench_test_ids=args.figurebench_test_ids,
        figurebench_dev_ids=args.figurebench_dev_ids,
        access_log=args.access_log if args.access_log is not None else str(Path(ledger).with_name("hidden_test_access.log")),
    )


def _run_benchmark_export_contracts(args: argparse.Namespace) -> dict:
    manifest_path = Path(args.manifest or "benchmarks/editable-figure-bench/manifest/hidden_test_manifest.jsonl")
    if not manifest_path.is_file():
        raise ValueError(f"manifest does not exist: {manifest_path}")
    rows = []
    for line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    extractor = None
    from .providers.client import vlm_credentials_available

    if vlm_credentials_available():
        from .analysis.paper.preparation import prepare_paper_figure_contract

        def extractor(row: dict, out: Path) -> dict:
            pdf = str(row.get("pdf_path") or "")
            if not pdf or not Path(pdf).is_file():
                return {"ok": False, "contract_path": ""}
            result = prepare_paper_figure_contract(pdf, out, planner_mode="vlm")
            contract = out / "figure_content_plan.json"
            return {"ok": bool(result.get("ok")), "contract_path": str(contract if contract.is_file() else "")}

    return draft_contracts(
        rows,
        contracts_dir=args.contracts or "datasets/judge-framework-all/evidence/structured",
        out_dir=args.out or "benchmarks/editable-figure-bench/manifest",
        extractor=extractor,
        exclude_paper_ids=_id_set(args.exclude_papers),
    )


def _run_benchmark_score(args: argparse.Namespace) -> dict:
    runs = args.runs or args.out
    if not runs:
        raise ValueError("benchmark score requires --runs (the directory written by benchmark run)")
    return score_benchmark(runs_dir=runs, out_dir=args.out or runs, layers=_benchmark_layers(args))


def _run_benchmark_score_own(args: argparse.Namespace) -> dict:
    missing = [name for name in ("manifest", "runs") if not getattr(args, name, None)]
    if missing:
        raise ValueError("benchmark score-own requires " + " and ".join("--" + name for name in missing))
    embedder = LocalAltCLIP(args.clip_model) if args.clip_model else None
    return score_own_manifest(
        args.manifest,
        args.runs,
        limit=args.limit,
        force=bool(args.force),
        embedder=embedder,
        paper_quiz_spec=args.paper_quiz,
        nine_spec=args.nine_judge,
    )


def _dispatch(args: argparse.Namespace) -> dict:
    if args.command == "doctor":
        return _doctor()
    if args.command == "analyze-reference-figures":
        return analyze_reference_figures(references=args.references, out=args.out, mode=args.mode, model=args.model, vlm_timeout=args.vlm_timeout, vlm_retries=args.vlm_retries)
    if args.command == "select-reference-exemplars":
        return select_reference_exemplars(paper_review=args.paper_review, library=args.library, out=args.out, selected_template=args.selected_template, top_k=args.top_k)
    if args.command == "learn-reference-style":
        return learn_reference_style(style_reference=args.style_reference, out=args.out, mode=args.mode, model=args.model)
    if args.command == "inspect-pdf":
        return inspect_paper(paper=args.paper, out=args.out, deadline_seconds=args.deadline, ocr_engine=args.ocr_engine, ocr_lang=args.ocr_lang)
    if args.command == "adopt-undelivered-decks":
        return _run_deck_adoption(args)
    if args.command == "paper-evolve-editable":
        return _run_direct(args)
    if args.command == "paper-evolve-validate":
        return run_direct_experience_validation(suite=args.suite, experience_library=args.experience_library, experience_id=args.experience_id, prompt_patch_id=args.prompt_patch_id, out=args.out, reselect_on_failure=not getattr(args, "no_reselect_on_failure", False), validation_profile=getattr(args, "validation_profile", "") or "")
    if args.command == "paper-evolve-corpus":
        return _run_paper_evolve_corpus(args)
    if args.command == "paper-evolve-session":
        return _run_session(args)
    if args.command == "paper-evolve-continuous":
        return _run_continuous(args)
    if args.command == "paper-evolve-review":
        if args.review_action == "collect":
            return review_collect(run_root=args.run_root, out=args.out, limit=args.limit)
        if args.review_action == "record":
            condition = {}
            if str(args.condition or "").strip():
                try:
                    parsed = json.loads(args.condition)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"--condition must be a JSON object: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise SystemExit("--condition must be a JSON object")
                condition = parsed
            return review_record(
                bundle=args.bundle,
                run_root=args.run_root,
                kind=args.kind,
                paper=args.paper,
                item_id=args.item_id,
                choice=args.choice,
                defect=args.defect,
                note=args.note,
                reviewer=args.reviewer,
                stage=args.stage,
                apply_at=args.apply_at,
                directive=args.directive,
                condition=condition,
                dimensions=list(args.dimension or []),
                resident=bool(args.resident),
            )
        if args.review_action == "export":
            return review_export(bundle=args.bundle, out=args.out)
        if args.review_action == "queue":
            return review_queue(run_root=args.run_root)
        if args.review_action == "answer":
            return review_answer(card=args.card, choice=args.choice, note=args.note, reviewer=args.reviewer)
        return review_stats(bundle=args.bundle, out=args.out)
    if args.command == "paper-evolve-patches":
        return backfill_prompt_patches(args.experience_library, out=args.out)
    if args.command == "experience-archive":
        return archive_untransferable_experiences(args.library, out=args.out, dry_run=bool(args.dry_run))
    if args.command == "creator-rules":
        return report_creator_rule_ledger(
            args.library,
            release=str(args.release or ""),
            release_all=bool(args.release_all),
            reason=str(args.reason or ""),
            out=args.out,
        )
    if args.command == "benchmark":
        if args.benchmark_action == "list":
            return list_benchmark_cases(args.root, suite=args.suite)
        if args.benchmark_action == "validate":
            return validate_benchmark_case(args.case)
        if args.benchmark_action == "fetch":
            return fetch_benchmark_case(args.case, force=args.force)
        if args.benchmark_action == "prepare":
            return _run_benchmark_prepare(args)
        if args.benchmark_action == "run":
            return _run_benchmark_run(args)
        if args.benchmark_action == "score":
            return _run_benchmark_score(args)
        if args.benchmark_action == "score-own":
            return _run_benchmark_score_own(args)
        if args.benchmark_action == "export-manifest":
            return _run_benchmark_export_manifest(args)
        if args.benchmark_action == "export-contracts":
            return _run_benchmark_export_contracts(args)
        return run_pdf_extraction_stress_suite(args.out, ocr_engine=args.ocr_engine)
    if args.command == "validate":
        return validate_output(args.out)
    if args.command == "contract-crosscheck":
        return run_contract_crosscheck(args.left, args.right, out=args.out)
    if args.command == "presentations-qa":
        result = run_presentations_qa(out_dir=args.out, pptx=args.pptx, workspace=args.workspace, scale=args.scale, run_inspect=not args.skip_inspect)
        result["ok"] = result.get("presentations_plugin_qa", {}).get("status") != "failed"
        return result
    if args.command == "judge":
        return _run_judge(args)
    if args.command == "judge-data":
        return _run_judge_data(args)
    if args.command == "figure-edit":
        # The parser knows the vocabulary; the composition layer knows the program.
        command, options = figure_edit_invocation(args)
        return run_figure_edit(command, program_path=args.program, out_dir=args.out, **options)
    raise ValueError(f"unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    json_requested = "--json" in argv
    argv = [arg for arg in argv if arg != "--json"]
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = _dispatch(args)
    except Exception as exc:
        result = {"summary": "EvoFig command failed.", "ok": False, "error": str(exc)}
        if json_requested:
            _json_print(result)
        else:
            print(f"error: {exc}")
        return 1
    if json_requested:
        _json_print(result)
    else:
        for key in ("ok", "summary", "out_dir", "pptx", "preview", "status"):
            if key in result:
                print(f"{key}: {result[key]}")
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
