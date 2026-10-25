"""Registration helpers for product workflow commands."""

from __future__ import annotations

from ..workflows.stage_replay import READING_FORK_STAGES, STAGE_ORDER


def register_paper_evolve_editable(subparsers) -> None:
    """Register the sole direct paper-to-PPTX product workflow."""
    parser = subparsers.add_parser("paper-evolve-editable", help="Generate and evolve an editable scientific framework directly from a paper.")
    parser.add_argument("--paper", required=True)
    parser.add_argument(
        "--paper-id",
        default="",
        help="Identifier of the paper this run is about (a DOI or an arXiv id), when the caller knows it. A paper the middle layer indexed under a path keeps only a title, and a run left to read its input path names the paper after that folder -- one name shared by every paper, which detaches the artifacts this run writes from the paper it was started on.",
    )
    parser.add_argument("--reference")
    parser.add_argument("--out", required=True)
    parser.add_argument("--planner-mode", choices=["vlm", "heuristic"], default="heuristic")
    parser.add_argument("--read-mode", choices=["fast", "full"], default="fast", help="How much of the paper is read before anything is drawn. 'fast' is the product's own read: one planning call over a shortened evidence budget, with no reference image handed to the planner. 'full' pays for a review pass of its own first, then hands the planner the reference images and a wider evidence budget, and merges the deterministic review plan into the model's. A different ruler: the review is a call the fast read never makes, so the two are not mixed inside one row.")

    parser.add_argument("--graph-context", choices=["on", "off"], default="on", help="Let the structure planner read the paper's scientific graph and subgraphs. 'off' is the path before the graph was read; 'on' is the new default and the full configuration.")
    parser.add_argument("--middle-layer", choices=["on", "off"], default="on", help="Build the paper's explicit structure (scientific graph, subgraphs, scientific hierarchy plan) and let planning read it. 'off' is the supplementary ablation row: nothing is built while planning, and the graph is rebuilt after the design only as the measurement the Grounding Gate uses. Defaults to 'on'.")
    parser.add_argument("--figure-target", choices=["paper", "overview"], default="overview", help="Which figure the design is aimed at. 'overview' is the product's own reading of the method and is the default; 'paper' follows the panels the paper printed, which is what a public benchmark scores. The target guides the prompts and is measured at the end; it never refuses a delivery.")
    parser.add_argument("--aspect-ratio", default="16:9")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="How many evolution rounds a single-paper run may take (1-12, default 5). The delivered figure is the accepted round with the highest measured score.",
    )
    parser.add_argument("--max-retries-per-experience", type=int, default=3)
    parser.add_argument("--evolution-scope", choices=["global", "block", "joint"], default="joint")
    parser.add_argument("--candidates-per-round", type=int, default=2)
    parser.add_argument("--icon-mode", choices=["placeholder", "image2"], default="image2")
    parser.add_argument("--method-mode", choices=["heuristic", "vlm"], default="heuristic")
    parser.add_argument("--reference-mode", choices=["heuristic", "vlm", "off"], default="heuristic")
    parser.add_argument("--icon-candidates", type=int, default=1)
    parser.add_argument("--structure-mode", choices=["vlm", "heuristic"], default="vlm")
    parser.add_argument("--text-plan-mode", choices=["vlm", "heuristic"], default="vlm")
    parser.add_argument("--ppt-plan-mode", choices=["vlm", "heuristic"], default="vlm")
    parser.add_argument("--structure-model")
    parser.add_argument("--text-plan-model")
    parser.add_argument("--ppt-plan-model")
    parser.add_argument("--color-theme-mode", choices=["vlm", "fallback"], default="vlm")
    parser.add_argument("--color-theme-model")
    parser.add_argument("--vlm-timeout", type=int, default=180)
    parser.add_argument("--vlm-retries", type=int, default=1)
    parser.add_argument("--reference-library")
    parser.add_argument("--reference-top-k", type=int, default=5)
    parser.add_argument("--exemplar-guidance", choices=["on", "off"], default="on")
    parser.add_argument("--evolution", choices=["on", "off"], default="on")
    parser.add_argument("--judge-mode", choices=["heuristic", "vlm", "trained"], default="heuristic")
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--online-judge-model",
        "--advisory-judge-model",
        dest="online_judge_model",
        help="Model for the advisory Judge (建议 Judge; --advisory-judge-model is the same flag).",
    )
    parser.add_argument(
        "--frozen-judge-model",
        "--acceptance-judge-model",
        dest="frozen_judge_model",
        help="Model for the acceptance Judge (验收 Judge; --acceptance-judge-model is the same flag). Frozen describes weights held still in a co-evolution cycle, not the role.",
    )
    parser.add_argument("--registry", help="Judge checkpoint registry; with --model-id it names the trained Judge this run is measured by and stamps its identity on every Memory row.")
    parser.add_argument("--model-id", help="Registered Judge checkpoint to be measured by; needs --registry and --judge-mode trained, and is refused unless the checkpoint is registered for the advisory (online) and acceptance (frozen) roles.")
    parser.add_argument("--acceptance-checkpoint", default="", help="Local choice checkpoint used only to compare the figure before and after a repair.")
    parser.add_argument("--acceptance-base-model", default="", help="Base model the acceptance checkpoint was trained on.")
    parser.add_argument(
        "--advisory-checkpoint",
        "--online-judge-checkpoint",
        dest="online_judge_checkpoint",
        default="",
        help="Local choice checkpoint seated in the advisory Judge instead of the served model (--online-judge-checkpoint is the same flag). The shipped configuration names the same weights in both seats, so the round is advised and scored by one model.",
    )
    parser.add_argument(
        "--advisory-base-model",
        "--online-judge-base-model",
        dest="online_judge_base_model",
        default="",
        help="Base model the advisory checkpoint was trained on.",
    )
    parser.add_argument("--judge-holdout", action="append", dest="judge_holdout", default=[], help="JSONL of papers this run must not draw. Repeat for each exam list. A missing file refuses the run.")
    parser.add_argument("--domain-profile", choices=["auto", "general", "ai-ml-method", "system-platform", "dataset-benchmark", "empirical-science", "survey-review"], default="auto")
    parser.add_argument("--experience-library")
    parser.add_argument("--no-experience", action="store_true")
    parser.add_argument(
        "--retrieval-statuses",
        help="Which Memory rungs this run may draw from. 'strict' (default) injects only rows a paired validation walked to, 'relaxed' also injects a row measured on its own paper, or name the rungs as a comma list.",
    )
    parser.add_argument("--visual-review-mode", choices=["auto", "off", "heuristic", "vlm"], default="auto")
    parser.add_argument("--visual-review-model")
    parser.add_argument("--replay-upstream-from", help="Validated prior run used to freeze upstream contracts and Icon assets for controlled experiments.")
    parser.add_argument(
        "--fork-before",
        # A paired replay must start from artifacts the source can lend, and the reading stages
        # lend nothing -- a fork there is a re-read of the paper rather than a frozen prefix -- so
        # they are not offered here. The continuous loop forks at them by attribution, with no
        # source to inherit from.
        choices=[stage for stage in STAGE_ORDER if stage not in READING_FORK_STAGES],
        default="",
        help="First stage a replayed run plans for itself; everything before it is inherited from --replay-upstream-from, and the named stage and every stage after it are planned again. It is what makes a paired experiment controlled: the pair starts from one frozen prefix and differs only from here on. Empty keeps the old behaviour of freezing the whole figure. Only read when --replay-upstream-from is given.",
    )
    parser.add_argument(
        "--reuse-replayed-icons",
        choices=["on", "off"],
        default="on",
        help="Whether an attempt may keep the Icon assets a replayed source already paid for. On (the default), a slot whose request did not materially move is not drawn twice -- the reading, the contracts and the Icons are what a retry is supposed to inherit. Off draws them all again, which is what a budget experiment needs: Icons are most of what a paper costs, so a rung that inherited them would report the baseline's bill under its own name. Only read when --replay-upstream-from is given.",
    )
    parser.add_argument("--upstream-from", help="Middle layer directory for this paper: when it carries paper_method_inventory.json (or the earlier figure_specification.json) and paper_review.json, the run adopts them as the contract instead of planning its own.")
    parser.add_argument("--baseline-from", help="Finished run directory whose kept figure is the state this run's first round is compared against. Empty means the first round has nothing to compare with, and the record says so instead of scoring it.")
    parser.add_argument("--bulk-figures", action="store_true", help="Bulk run: keep the first design plan instead of paying for a repair call, skip the paid sparse-panel replan, and check the figure once instead of re-checking the repair.")
    parser.add_argument("--no-upstream-retry", action="store_true", help="Do not go back to the method extraction and the structure planning when a measured break escalates; record the refusal instead.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")


def register_paper_evolve_validate(subparsers) -> None:
    """Register cross-paper experience validation."""
    parser = subparsers.add_parser("paper-evolve-validate", help="Run paired experience ON/OFF validation across papers.")
    parser.add_argument("--suite", required=True, help="Validation suite JSON with a graph_index or per-case graph_record so the four-group selector can run.")
    parser.add_argument("--experience-library", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--experience-id", help="Validate one experience through the paired ON/OFF harness.")
    target.add_argument("--prompt-patch-id", help="Validate one prompt patch through the same paired ON/OFF harness.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-reselect-on-failure", action="store_true", help="Do not draw another case set when this one cannot measure the pair (spec 7.7).")
    parser.add_argument("--validation-profile", default="", help="Which bar to measure at: blank keeps the suite's own word and runs the full four-group set; 'quick' asks the single matched paper the row is reproduced on and marks the row quick_validated.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")


def register_paper_evolve_corpus(subparsers) -> None:
    """Register the supervised batch that runs a corpus subset paper by paper."""
    parser = subparsers.add_parser("paper-evolve-corpus", help="Sample a fixed corpus subset and run it through the product workflow with deterministic supervision.")
    parser.add_argument("--snapshot", required=True, help="Frozen source snapshot directory; opened read-only.")
    parser.add_argument("--split", required=True, help="Frozen split version that decides which papers are train or validation.")
    parser.add_argument("--out", required=True, help="Batch root to create: the subset version and every attempt live under it.")
    parser.add_argument("--pdf-root", action="append", dest="pdf_roots", help="Collection directory that holds the paper files; repeatable, defaults to datasets/paper_collection.")
    parser.add_argument("--archive-root", help="The snapshot's own archive root: resolve each sampled paper to the copy the snapshot froze instead of the collection directory it came from.")
    parser.add_argument("--subset", help="Existing subset version to run instead of sampling a new one.")
    parser.add_argument("--exclude-drawn-root", action="append", help="Product root that already holds deliveries: any paper with a run directory in one is dropped before a provider is called. Repeatable.")
    parser.add_argument("--subset-id", default="paper-d0-subset-100", help="Subset version id and sampling seed.")
    parser.add_argument("--per-domain", type=int, default=20, help="Papers to sample per domain.")
    parser.add_argument("--label-cache", help="Pilot label cache JSONL merged in so the structure coverage is reported.")
    parser.add_argument("--sample-only", action="store_true", help="Write the subset and stop before any paid call.")
    parser.add_argument("--limit", type=int, help="Run only the first N sampled papers.")
    parser.add_argument("--resume", action="store_true", help="Skip the papers an earlier run already kept and continue an interrupted one.")
    parser.add_argument("--workers", type=int, default=1, help="Run this many papers at once, one process each; 100 papers serial takes days.")
    parser.add_argument("--shard", help="Run one slice of the subset as INDEX/COUNT; the parent process sets this.")
    parser.add_argument("--reference-library")
    parser.add_argument("--contracts-from", help="Middleware export root holding one contract directory per paper; papers it does not cover are analysed as before.")
    parser.add_argument("--icon-mode", choices=["placeholder", "image2"], default="image2")
    parser.add_argument("--vlm-timeout", type=int, default=240)
    parser.add_argument("--vlm-retries", type=int, default=2)
    parser.add_argument("--bulk-figures", action="store_true", help="Bulk corpus run: keep the first design plan instead of paying for a repair call, skip the paid sparse-panel replan, and check the figure once instead of re-checking the repair.")
    parser.add_argument("--fresh-retries", action="store_true", help="Retry a paper from the start instead of continuing the attempt that already paid for its reading, contracts, and Icons.")
    parser.add_argument("--keep-first-draft", action="store_true", help="Bulk corpus run: keep the figure a paper already delivered instead of paying for another attempt; only an attempt that delivered no figure at all is retried.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")

def register_paper_evolve_session(subparsers) -> None:
    """Register the wall-clock session that runs the whole loop without a human in between."""
    parser = subparsers.add_parser("paper-evolve-session", help="Run the whole evolution loop on one wall-clock budget: evolve papers, validate the experience they earned, promote it, evolve the next paper.")
    parser.add_argument("--config", required=True, help="Session config JSON: the ordered stations (papers to evolve and experiences to validate) and their settings.")
    parser.add_argument("--out", required=True, help="Session root; each station gets its own attempt directory underneath and is never overwritten.")
    parser.add_argument("--validation-suite", help="Validation suite JSON used by every validate station that does not name its own.")
    parser.add_argument("--experience-library", help="Experience library the evolve stations write to and the validate stations judge and promote.")
    parser.add_argument("--hours", type=float, default=5.0, help="Wall-clock budget: the queue stops when it is spent, without cutting a paper in half.")
    parser.add_argument("--rate-limit-cooldown", type=float, default=45.0, help="Seconds to sit out before retrying a station the provider rate limited.")
    parser.add_argument("--rate-limit-retries", type=int, default=3, help="How many times one station may be retried after a 429 before it is recorded as failed.")
    parser.add_argument("--acceptance-checkpoint", default="", help="Local choice checkpoint every evolve station uses to compare the figure before and after a repair.")
    parser.add_argument("--acceptance-base-model", default="", help="Base model that acceptance checkpoint was trained on.")
    parser.add_argument("--judge-holdout", action="append", dest="judge_holdout", default=[], help="JSONL of papers an evolve station must not draw. Repeat for each exam list.")
    parser.add_argument("--no-validate", action="store_true", help="Keep the queue's validate stations out of this session and say so in the report.")
    parser.add_argument("--dry-run", action="store_true", help="Plan the queue, name each station's directory and write nothing.")
    parser.add_argument("--no-training-export", action="store_true", help="Do not export the session's runs into Judge data, preference pairs and a human review queue.")
    parser.add_argument("--export-split", help="Frozen split ledger: the export keeps hidden_test and exploration papers out of the training view.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")


def register_paper_evolve_patches(subparsers) -> None:
    """Register the maintenance command that keeps the Prompt Patch library in step with Memory."""
    parser = subparsers.add_parser("paper-evolve-patches", help="Render the Prompt Patches an existing experience library never derived.")
    actions = parser.add_subparsers(dest="patches_action", required=True)
    backfill = actions.add_parser("backfill", help="Render the missing patch for every measured row and report each row's outcome.")
    backfill.add_argument("--experience-library", required=True, help="Experience library whose rows are rendered into patches.")
    backfill.add_argument("--out", help="Where to write the report; defaults to a file inside the library.")
    backfill.add_argument("--json", action="store_true", help="Emit JSON.")


def register_experience_archive(subparsers) -> None:
    """Register the maintenance command that moves the Memory rows nothing can act on."""

    parser = subparsers.add_parser("experience-archive", help="Move the experience rows that state no lesson another paper can act on into an archive, and leave every row that does.")
    parser.add_argument("--library", required=True, help="Experience library to read as a set of lessons.")
    parser.add_argument("--out", required=True, help="Archive directory: the moved rows are appended to its own library file and the report lands beside them.")
    parser.add_argument("--dry-run", action="store_true", help="Decide every row and move nothing.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")


def register_creator_rules(subparsers) -> None:
    """Register the maintenance command that shows and clears a creator rule's beating streak."""

    parser = subparsers.add_parser("creator-rules", help="Show every creator rule's refutation streak, and clear one or all of them so the next run injects those rules again.")
    parser.add_argument("--library", required=True, help="Experience library whose creator-rule ledger is read.")
    parser.add_argument("--release", default="", help="Clear this rule's refutation streak, so it may be injected again.")
    parser.add_argument("--release-all", action="store_true", help="Clear every rule's refutation streak.")
    parser.add_argument("--reason", default="", help="Why the rules were released; recorded on each ledger event.")
    parser.add_argument("--out", help="Where to write the report; nothing is written when it is empty.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")


def register_paper_evolve_continuous(subparsers) -> None:
    """Register the loop that keeps a fixed set of base papers evolving without stopping."""
    parser = subparsers.add_parser("paper-evolve-continuous", help="Keep the same base papers evolving wave after wave: each iteration inherits the last draft up to the stage an advisory review asked to change, and the loop only stops when the caller says so.")
    parser.add_argument("--config", required=True, help="Continuous config JSON: the fixed base_papers and the evolve options every iteration uses, under a top-level continuous block.")
    parser.add_argument("--out", required=True, help="Continuous root; each paper keeps its iterations under papers/<paper>/iter_NN and is never overwritten.")
    parser.add_argument("--experience-library", help="Experience library every iteration reads from and writes to; a later iteration can pick up what an earlier one earned.")
    parser.add_argument("--stop-file", help="Path whose appearance finishes the wave in flight and stops the loop; defaults to the config's stop_file.")
    parser.add_argument("--max-waves", type=int, default=0, help="Stop after this many waves; 0 keeps going until the stop file or the budget ends it.")
    parser.add_argument("--json", action="store_true", help="Emit JSON.")


def register_paper_evolve_review(subparsers) -> None:
    """Register the human-preference channel: the queue, one verdict, and the export."""

    parser = subparsers.add_parser(
        "paper-evolve-review",
        help="Review the two-figure comparisons a run asked, by hand: build the queue of rounds, record which figure a person thinks is better, and export those verdicts as training rows for the choice seat.",
    )
    actions = parser.add_subparsers(dest="review_action", required=True)

    collect = actions.add_parser("collect", help="Build the queue of comparisons from a run tree, images included.")
    collect.add_argument("--run-root", required=True, help="Self-evolution root to read: every iteration and round under it that asked the two-figure question becomes one queue item.")
    collect.add_argument("--out", required=True, help="Bundle directory to create: the queue file and the two renders per item.")
    collect.add_argument("--limit", type=int, default=0, help="Keep only the newest N comparisons; 0 keeps every one of them.")
    collect.add_argument("--json", action="store_true", help="Emit JSON.")

    record = actions.add_parser("record", help="File one verdict about one comparison, one settled layer, or one preference.")
    record.add_argument("--bundle", default="", help="Bundle directory the queue was collected into; or use --run-root and the verdict is filed in the run's own review folder.")
    record.add_argument("--run-root", default="", help="Self-evolution root this verdict is about; the verdict is filed under its review folder.")
    record.add_argument("--kind", default="figure_choice", choices=["figure_choice", "layer_settled", "preference"], help="What is being filed: which figure is better, that a layer is finished with, or an instruction about how the figure should be drawn.")
    record.add_argument("--item-id", default="", help="The comparison this verdict is about, for a figure choice.")
    record.add_argument("--choice", default="", choices=["A", "B", "tie", "both_bad", ""], help="Which figure is better: A or B, or that they are equally good or equally bad.")
    record.add_argument("--defect", default="", help="What is wrong with the worse figure, one of the menu names.")
    record.add_argument("--note", default="", help="The reviewer's own words; kept as written.")
    record.add_argument("--reviewer", default="", help="Who made this call.")
    record.add_argument("--paper", default="", help="Which paper the verdict is about; required for a settled layer and for a preference.")
    record.add_argument("--stage", default="", help="The pipeline layer this verdict settles, for --kind layer_settled.")
    record.add_argument("--apply-at", default="", help="The pipeline layer a --kind preference is about.")
    record.add_argument("--directive", default="", help="The sentence that layer's prompt should have carried.")
    record.add_argument("--condition", default="", help="JSON object saying when the preference applies.")
    record.add_argument("--dimension", action="append", dest="dimension", default=[], help="One reading answered on its own, as name=choice; repeat for each. Choices are A, B, tie, both_bad.")
    record.add_argument("--resident", action="store_true", help="File the preference as a standing instruction rather than a trial.")
    record.add_argument("--json", action="store_true", help="Emit JSON.")

    queue = actions.add_parser("queue", help="List the judgement points a run is holding for a person, newest first.")
    queue.add_argument("--run-root", required=True, help="Self-evolution root whose human-queue folder is read.")
    queue.add_argument("--json", action="store_true", help="Emit JSON.")

    answer = actions.add_parser("answer", help="Answer one held judgement point by hand: which figure is better, or whether a layer is finished with.")
    answer.add_argument("--card", required=True, help="The card directory the question was filed in, as `paper-evolve-review queue` names it.")
    answer.add_argument("--choice", required=True, help="The answer in the words the card's own kind uses: A or B for a figure choice.")
    answer.add_argument("--note", default="", help="The reviewer's own words; kept as written.")
    answer.add_argument("--reviewer", default="", help="Who made this call.")
    answer.add_argument("--json", action="store_true", help="Emit JSON.")

    export = actions.add_parser("export", help="Write the verdicts that answer the seat's question as training rows.")
    export.add_argument("--bundle", required=True, help="Bundle directory the queue was collected into.")
    export.add_argument("--out", required=True, help="Output JSONL file, or a directory to write one into.")
    export.add_argument("--json", action="store_true", help="Emit JSON.")

    stats = actions.add_parser("stats", help="How much of the queue is answered, and where the person and the seat disagreed.")
    stats.add_argument("--bundle", required=True, help="Bundle directory the queue was collected into.")
    stats.add_argument("--out", help="Optional path to write the report to.")
    stats.add_argument("--json", action="store_true", help="Emit JSON.")
