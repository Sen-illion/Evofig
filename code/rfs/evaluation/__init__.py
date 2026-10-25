"""Deterministic and model-assisted quality evaluation APIs."""

from .benchmarking import fetch_benchmark_case, list_benchmark_cases, validate_benchmark_case
from .contract_crosscheck import (
    compare_figure_specifications,
    crosscheck_contract_directories,
    crosscheck_contract_pairs,
    run_contract_crosscheck,
)
from .pdf_extraction_benchmark import run_pdf_extraction_stress_suite
from .outputs import validate_output
from .editable_figure_bench import (
    evaluate_instance,
    prepare_gold_instances,
    resolve_embedder,
    run_benchmark_instances,
    score_benchmark,
    verify_gold_instances,
)
from .native_editability import extract_editable_text, extract_shape_labels, measure_native_editability
from .direct_judges import DirectJudgeInput, build_direct_judges
from .graph_alignment import align_graph_layers
from .grounding_gate import evaluate_grounding_gate
from .decomposability import build_decomposability_report
from .prompt_patch_check import check_prohibited_patches

__all__ = [
    "list_benchmark_cases",
    "fetch_benchmark_case",
    "run_pdf_extraction_stress_suite",
    "validate_benchmark_case",
    "validate_output",
    "prepare_gold_instances",
    "run_benchmark_instances",
    "score_benchmark",
    "verify_gold_instances",
    "evaluate_instance",
    "extract_editable_text",
    "extract_shape_labels",
    "resolve_embedder",
    "measure_native_editability",
    "DirectJudgeInput",
    "build_direct_judges",
    "align_graph_layers",
    "evaluate_grounding_gate",
    "build_decomposability_report",
    "check_prohibited_patches",
    "compare_figure_specifications",
    "crosscheck_contract_directories",
    "crosscheck_contract_pairs",
    "run_contract_crosscheck",
]
