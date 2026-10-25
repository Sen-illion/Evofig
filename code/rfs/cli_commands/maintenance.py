"""Registration helpers for maintenance and reference-library commands."""

from __future__ import annotations


def register_reference_and_inspection_commands(subparsers) -> None:
    doctor = subparsers.add_parser("doctor", help="Check dependencies, PowerPoint, and auth env vars.")
    doctor.add_argument("--json", action="store_true", help="Emit JSON.")

    analyze_references = subparsers.add_parser("analyze-reference-figures", help="Analyze project reference figures into reusable design records.")
    analyze_references.add_argument("--references", default="reference-figures", help="Reference figure root containing excellent/ and negative/. Default: reference-figures.")
    analyze_references.add_argument("--out", default="output/reference_learning_stage1", help="Output directory for reference analysis JSONL, library JSONL, and report.")
    analyze_references.add_argument("--mode", choices=["heuristic", "vlm"], default="heuristic", help="Reference analysis mode. Default: heuristic.")
    analyze_references.add_argument("--model", default="gpt-5.6-sol", help="VLM model for --mode vlm. Default: gpt-5.6-sol.")
    analyze_references.add_argument("--vlm-timeout", type=int, help="Optional VLM timeout in seconds. Defaults to RFS_VLM_TIMEOUT.")
    analyze_references.add_argument("--vlm-retries", type=int, help="Optional VLM retry count. Defaults to RFS_VLM_RETRIES.")
    analyze_references.add_argument("--json", action="store_true", help="Emit JSON.")

    select_exemplars = subparsers.add_parser("select-reference-exemplars", help="Select reference design exemplars for a paper review.")
    select_exemplars.add_argument("--paper-review", required=True, help="paper_review.json for the current paper.")
    select_exemplars.add_argument("--library", required=True, help="Reference library JSONL from analyze-reference-figures.")
    select_exemplars.add_argument("--out", required=True, help="Output directory for exemplar_selection.json and exemplar_style_brief.json.")
    select_exemplars.add_argument("--selected-template", help="Optional selected_template.json for archetype/aspect matching.")
    select_exemplars.add_argument("--top-k", type=int, default=5, help="Positive exemplar count, clamped to 1-20. Default: 5.")
    select_exemplars.add_argument("--json", action="store_true", help="Emit JSON.")

    learn_style = subparsers.add_parser("learn-reference-style", help="Learn reusable visual style preferences from one user style reference image.")
    learn_style.add_argument("--style-reference", required=True, help="User-provided style reference image. Content is not reused.")
    learn_style.add_argument("--out", required=True, help="Output directory for user_style_record.json, user_style_brief.json, and user_style_report.json.")
    learn_style.add_argument("--mode", choices=["heuristic", "vlm"], default="vlm", help="User style learning mode. Default: vlm.")
    learn_style.add_argument("--model", default="gpt-5.6-sol", help="VLM model for --mode vlm. Default: gpt-5.6-sol.")
    learn_style.add_argument("--json", action="store_true", help="Emit JSON.")

    inspect_pdf = subparsers.add_parser("inspect-pdf", help="Extract and diagnose a paper without calling a model.")
    inspect_pdf.add_argument("--paper", required=True, help="Paper PDF or supported document path.")
    inspect_pdf.add_argument("--out", required=True, help="Output directory for the document model and extraction report.")
    inspect_pdf.add_argument("--deadline", type=int, default=180, help="Soft processing deadline in seconds. Default: 180.")
    inspect_pdf.add_argument("--ocr-engine", choices=["auto", "rapidocr", "paddle", "easyocr", "off"], default="auto")
    inspect_pdf.add_argument("--ocr-lang", choices=["en", "ch", "en_ch"], default="en_ch")
    inspect_pdf.add_argument("--json", action="store_true", help="Emit JSON.")


def register_quality_commands(subparsers) -> None:
    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run product acceptance cases and the released EditableFigureBench suite.",
    )
    benchmark.add_argument(
        "benchmark_action",
        choices=[
            "list",
            "validate",
            "fetch",
            "pdf-suite",
            "prepare",
            "run",
            "score",
            "score-own",
            "export-manifest",
            "export-contracts",
        ],
        help="prepare freezes gold instances, run scores systems against them, score aggregates the report, score-own scores deliveries already on disk.",
    )
    benchmark.add_argument("--suite", help="Optional suite filter for list.")
    benchmark.add_argument("--root", default="benchmarks", help="Benchmark root for list. Default: benchmarks.")
    benchmark.add_argument("--case", help="Benchmark case directory for validate, fetch, run, or score.")
    benchmark.add_argument("--case-id", action="append", default=[], help="Optional case id filter for fast-suite. Repeat to select multiple cases.")
    benchmark.add_argument("--out", help="Output directory for reports.")
    benchmark.add_argument("--force", action="store_true", help="Re-fetch an existing local benchmark paper, or rescore an own-set folder that already has metrics.json.")
    benchmark.add_argument("--deadline", type=int, default=180, help="Soft deadline for PDF extraction checks.")
    benchmark.add_argument("--ocr-engine", choices=["auto", "rapidocr", "paddle", "easyocr", "off"], default="off")
    benchmark.add_argument("--rasterize-dpi", type=int, help="Optional scanned-PDF rasterization DPI.")
    benchmark.add_argument("--instances", help="Frozen EditableFigureBench instance directory for run.")
    benchmark.add_argument("--systems", help="Directory holding system outputs for run: <paper_id>/ or <system_id>/<paper_id>/.")
    benchmark.add_argument("--runs", help="Run output directory produced by benchmark run, for score.")
    benchmark.add_argument("--system-id", help="Score a single system under --systems under this name.")
    benchmark.add_argument("--layers", default="L1,L2,L3", help="Comma-separated metric layers for run and score. Default: L1,L2,L3.")
    benchmark.add_argument("--target", type=int, default=150, help="Instance target for prepare. Default: 150.")
    benchmark.add_argument(
        "--label-groups",
        default="good,bad",
        help="Reference-figure quality groups the freeze may use. Pass good to build a release "
        "from human-approved reference figures only. Default: good,bad.",
    )
    benchmark.add_argument(
        "--splits",
        default="",
        help="Comma-separated corpus splits prepare may draw from (train,test,validation) or all. "
        "Default: the held-out splits test,validation.",
    )
    benchmark.add_argument("--papers", help="Override the papers.jsonl used by prepare.")
    benchmark.add_argument("--figures", help="Override the figures.jsonl used by prepare.")
    benchmark.add_argument("--reviews", help="Override the human framework review file used by prepare.")
    benchmark.add_argument(
        "--extra-reviews",
        action="append",
        default=[],
        help="Additional human review ledgers merged by export-manifest. Repeatable; a later ledger wins.",
    )
    benchmark.add_argument("--contracts", help="Override the structured contract directory used by prepare.")
    benchmark.add_argument("--pdf-root", help="Override the corpus PDF directory used by prepare.")
    benchmark.add_argument(
        "--categories-csv",
        help="Override the arXiv category metadata used by prepare for domain strata.",
    )
    benchmark.add_argument("--limit", type=int, help="Optional instance limit for run.")
    benchmark.add_argument("--verify", action="store_true", help="Verify the frozen gold set instead of rebuilding it.")
    benchmark.add_argument("--ledger", help="Split ledger for export-manifest.")
    benchmark.add_argument("--snapshot", help="Source snapshot directory for export-manifest.")
    benchmark.add_argument("--framework-papers", help="Framework-corpus papers.jsonl used only as a join, not as a label.")
    benchmark.add_argument("--sciflow-ids", help="Optional id file for the SciFlow overlap. Missing file is not_downloaded.")
    benchmark.add_argument("--figurebench-test-ids", help="Optional id file for the FigureBench official-test overlap.")
    benchmark.add_argument("--figurebench-dev-ids", help="Optional id file for the FigureBench development overlap.")
    benchmark.add_argument("--access-log", help="Append-only hidden-test access log for export-manifest.")
    benchmark.add_argument("--manifest", help="Manifest jsonl for export-contracts or score-own.")
    benchmark.add_argument("--paper-quiz", choices=["off", "api"], default=None, help="PaperQuiz switch for score-own. Default: RFS_PAPER_QUIZ.")
    benchmark.add_argument("--nine-judge", choices=["off", "api"], default=None, help="Nine-field switch for score-own. Default: RFS_NINE_JUDGE.")
    benchmark.add_argument("--clip-model", help="Local AltCLIP directory for score-own. Omit it and the CLIP column stays null.")
    benchmark.add_argument("--exclude-papers", help="Paper ids that must not enter the public gold, one per line.")
    benchmark.add_argument("--json", action="store_true", help="Emit JSON.")

    validate = subparsers.add_parser("validate", help="Validate an existing EvoFig output directory.")
    validate.add_argument("--out", required=True, help="Output directory to validate.")
    validate.add_argument("--json", action="store_true", help="Emit JSON.")

    crosscheck = subparsers.add_parser(
        "contract-crosscheck",
        help="Compare the figure specifications two directories carry for the same papers.",
    )
    crosscheck.add_argument("--left", action="append", default=[], required=True, help="Left paper directory holding paper_method_inventory.json (or the earlier figure_specification.json). Repeat once per paper.")
    crosscheck.add_argument("--right", action="append", default=[], required=True, help="Right paper directory for the same paper, in the same order. Repeat once per paper.")
    crosscheck.add_argument("--out", help="Optional output directory for contract_crosscheck.json.")
    crosscheck.add_argument("--json", action="store_true", help="Emit JSON.")

    presentations_qa = subparsers.add_parser("presentations-qa", help="Run optional Presentations plugin QA for an existing RFS output directory.")
    presentations_qa.add_argument("--out", required=True, help="RFS output directory containing editable_composition.pptx.")
    presentations_qa.add_argument("--pptx", help="Optional PPTX path. Defaults to <out>/editable_composition.pptx.")
    presentations_qa.add_argument("--workspace", help="Optional Presentations QA workspace. Defaults to <out>/presentations_plugin_qa_workspace.")
    presentations_qa.add_argument("--scale", type=int, default=2, help="Preview render scale. Default: 2.")
    presentations_qa.add_argument("--skip-inspect", action="store_true", help="Write report from PPTX/package metadata without invoking the Presentations plugin.")
    presentations_qa.add_argument("--json", action="store_true", help="Emit JSON.")
