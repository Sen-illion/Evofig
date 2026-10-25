"""Cross-paper ON/OFF validation workflow for direct evolution experience and patches.

An experience and a prompt patch are validated by the same harness on purpose: both are
Memory picked for a stage, and the only honest evidence that either generalizes is a paired
run that differs in nothing but the record under test. Whichever the caller names lands in
the pair configuration hash, so the two arms of a pair cannot silently differ in more than
the one thing being validated.
"""

from __future__ import annotations

import hashlib
import shutil
import statistics
from pathlib import Path
from typing import Any, Callable

from ..analysis.graph import CASE_GROUPS, COMPILE_ONLY_CASE_QUOTA, CONDITION_KEYS, DEFAULT_CASE_QUOTA, DOMAIN_CASE_GROUPS, FULL_PROFILE, QUICK_PROFILE, GraphIndex, case_set_qualification, outside_condition, outside_graph_scope, quick_case_quota, select_validation_cases
from ..composition import render_rebuild_preview
from ..composition.figure_edit import apply_constraints
from ..contracts.evolution import MECHANISM_ACTIONS, normalize_experience, normalize_pairwise_result
from ..contracts.memory_parameters import layout_rules_from, router_rules_from
from ..contracts.method_figure import method_inventory_path
from ..evolution.direct import ExperienceStore, PromptPatchStore, ValidationCase, measured_score, paired_configuration_hash, resolved_applicability, skeleton_state_hash, validate_experience_cases
from ..evolution.direct.controller import ACTION_BUDGET_KEYS, action_execution_fingerprint
from ..evolution.direct.experience import ACTION_MEMORY_PROFILE
from ..evolution.direct.validation import COMPILE_ONLY_PATH
from ..evaluation.direct_judges import DirectJudgeInput, DirectPairJudgeInput, build_direct_judges, program_readings
from ..utils import ensure_dir, read_json, write_json
from .paper_evolve_editable import run_paper_evolve_editable


#: The ladder outcomes that mean "this set could not answer the question" rather than "the
#: experience failed": a pair nobody measured, a pair whose two arms were not comparable, and a
#: pair whose two arms produced the same products -- the memory changed nothing, so there are no two
#: figures to compare. All of them draw another set instead of spending the row.
RESELECT_REASONS = {"pair_not_measured", "pair_isolation_or_safety_failed", "measurement_noise", "no_contrast_between_on_and_off"}

#: The ladder statuses a set can stop at without having answered its question. ``inconclusive``
#: joins ``overfit_or_failed`` here so a set nobody could tell apart draws another set, exactly as a
#: set nobody measured does: the two are the same kind of answer (none), and only the reason differs.
UNANSWERED_STATUSES = {"overfit_or_failed", "inconclusive"}

#: The same reasons read from the other side. A set that stops for one of them never measured
#: the pair, so it is a reason to draw another set and a reason a run may not spend the row: the
#: absence of a measurement is not a measurement of failure. One name would do, but the two callers
#: read it for different things and the second use is the one that costs a row when it is missed.
UNMEASURED_REASONS = RESELECT_REASONS

#: The stages a pair can fork at, in the order the contract lists them. A pair shares the frozen
#: prefix before the fork and plans from the fork on, so the fork has to be at or after the stage
#: the memory acts at -- otherwise the frozen prefix would freeze away the very move being
#: measured. ``global_planning`` has no prefix worth replaying (the contract itself is re-planned)
#: and a pair forked at ``segmentation`` re-plans the boxes, so neither can share a skeleton; they
#: are reported as unmeasured rather than measured against a figure nobody held equal.
FORK_STAGES = ("segmentation", "theme_planning", "icon_generation", "ppt_compilation", "repair")


#: The refusal a run makes when the paper a case points at carries no figure contract to build.
#: It names the case rather than the row under test: no figure was ever drawn, so the row was never
#: given anything to act on and this set says nothing about it.
UNPREPARED_PAPER_REFUSAL = "Paper preparation failed"

#: The finding written beside a case whose paper cannot be drawn. The control arm writes it and the
#: memory arm reads it, so a paper carrying nothing to draw is read once rather than twice.
UNUSABLE_CASE_FILE = "case_cannot_be_drawn.json"

#: What an arm that never ran reports. The ladder turns it into ``pair_not_measured``, which is
#: already a reason to draw another set, so an undrawable case costs its own reading instead of the
#: whole measurement.
UNUSABLE_CASE_REASON = "paper_cannot_be_prepared"

#: The papers a measurement found nothing to draw in. It is kept beside the attempt rather than
#: inside it because an attempt is finished the moment it writes its report and the next call opens
#: a fresh one; a paper that carries no figure does not become a case by being offered again.
UNDRAWABLE_PAPERS_FILE = "papers_without_a_figure.json"


def _fork_stage_for(apply_at: Any) -> str:
    """Return the stage a pair testing this memory has to fork at."""

    stage = str(apply_at or "").strip()
    return stage if stage in FORK_STAGES else ""


def _unusable_case_arm(case: ValidationCase, arm_dir: Path, *, enabled: bool, target: str, reason: str) -> dict[str, Any]:
    """The arm record for a case whose paper carries no figure contract to build.

    Every field the pair audit and the ladder read is present and empty on purpose. An arm that
    never ran drew no figure, holds no start state and matches no other arm, so reporting a score
    or a hash for it would be reporting a measurement nobody took.
    """

    return {
        "overall_score": None,
        "measured": False,
        "pair_stop_reason": UNUSABLE_CASE_REASON,
        "blocking_issue_count": 0,
        "dimension_scores": {},
        "run_dir": str(arm_dir),
        "ok": False,
        "paired_config_hash": "",
        "initial_state_hash": "",
        "raw_state_hash": "",
        "fork_stage": "",
        "fork_source": "",
        "candidate_policy_hash": "",
        "evaluator_hash": "",
        "memory_enabled": bool(enabled),
        "target_experience_id": str(target) if enabled else None,
        "action_sequence": None,
        "domain": str(case.domain),
        "compiled_program_hash": "",
        "case_unusable": True,
        "case_unusable_reason": str(reason),
        "product_hashes": {},
        "readings": {},
    }


def _note_unusable_case(path: Path, case: ValidationCase, paper_id: str, arm_dir: Path, error: str) -> dict[str, Any]:
    """Write, beside a case, that its paper carries no figure contract to build."""

    finding = {
        "contract": "rfs.unusable_validation_case",
        "version": 1,
        "case_id": str(case.case_id),
        "paper_id": str(paper_id or case.case_id),
        "reason": "the paper carries no figure contract to build",
        "error": str(error)[:1000],
        "arm": str(arm_dir),
        "note": (
            "the target paper could not be turned into a figure contract, so this case never got a figure to "
            "measure; the case is reported as unmeasured and another set is drawn instead of the row being spent "
            "on a paper that was never drawn"
        ),
    }
    write_json(path, finding)
    return finding


def _undrawable_papers_path(root: Path) -> Path:
    """Where the papers this measurement could not draw are remembered.

    One attempt writes its report and the next call opens a new attempt directory, so the list lives
    beside them rather than inside one: the point of it is to survive into the next attempt, which is
    the only place it can do any good.
    """

    return root.parent / UNDRAWABLE_PAPERS_FILE


def _read_undrawable_papers(root: Path) -> list[str]:
    """The papers this measurement already found nothing to draw in."""

    try:
        payload = read_json(_undrawable_papers_path(root))
    except (OSError, ValueError):
        return []
    if isinstance(payload, dict):
        payload = payload.get("paper_ids") or []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if str(item)]


def _note_undrawable_paper(root: Path, paper_id: str, reason: str) -> None:
    """Remember that a paper cannot be drawn, so the selector is not asked to offer it again."""

    name = str(paper_id or "").strip()
    if not name:
        return
    known = _read_undrawable_papers(root)
    if name in known:
        return
    known.append(name)
    write_json(
        _undrawable_papers_path(root),
        {"contract": "rfs.undrawable_validation_papers", "version": 1, "paper_ids": known, "last_reason": str(reason)},
    )


#: The row a measurement is about, pinned into the measurement's own directory the first time it
#: opens. The library it was queued from is not frozen while the measurement runs: rows are
#: archived, retired and rewritten, and a resumed attempt that went back to the library for its
#: row died with "unknown experience" after a wave of generation had already been paid for.
ROW_UNDER_TEST_FILE = "row_under_test.json"
#: That file's contract name. It is named here rather than written out at the one place that pins
#: a live row, because a caller that has to pin a row the library rotated away writes the same
#: shape, and two hand-written copies of a contract name is how a reader ends up skipping a pin.
ROW_UNDER_TEST_CONTRACT = "rfs.validation_row_under_test"
#: The event that spends the mark a row carries when a round introduced a defect. It is written
#: when a validation set showed the defect does not reproduce -- every case stayed safe -- and it
#: is what keeps a later run from capping a row that already cleared itself.
CARRIED_NEW_DEFECT_CLEARED = "carried_new_defect_cleared"


def _arm_resume_source(case_out: Path) -> str:
    """The arm's own directory when it already holds work the per-stage replay can continue from.

    One arm is about forty minutes of generation, and a call that stops inside one used to throw
    all of it away -- the next attempt began from an empty directory and paid for every stage
    again, which is why no validation ever reached a single decided case. The replay refuses a
    source that did not get past paper preparation, so the three artifacts it insists on are the
    test: an arm that wrote them is an arm the replay can keep whatever else it managed to finish,
    and the icons it already drew are the most expensive part of that.

    Only the arm that owns its directory outright is resumed this way. The memory arm replays the
    control arm, and its prefix has to stay byte-identical to that arm's; resuming it from its own
    directory would trade that guarantee for the time it saves, so it keeps replaying the control
    arm and only its post-fork work is paid for again.
    """

    if not case_out.is_dir():
        return ""
    required = ("input_manifest.json", "document_model.json", "method_extraction_report.json")
    if not all((case_out / name).is_file() for name in required):
        return ""
    return str(case_out)


def resolved_target_applicability(experience: dict[str, Any]) -> str:
    """Where a row applies, from the field it stored or from the stage its move acts at.

    One reader, used both to decide whether the row can be measured at all and to draw its case
    set: two readers of the same question is how a row could be selected as one that travels and
    then refused as one that never said where it travels.

    A row that already says where it applies keeps its own answer. The library writes a row's
    applicability once, when the round that earned it could still see the paper's field; requiring
    that field again here asked the row to name a domain it had already recorded under a different
    key, and read a row that named its own domain as one that named none.

    The stage comes from the move's own profile rather than from the row's stored ``apply_at``: a
    row that never wrote one is normalised to the repair stage by contract, which would read every
    unlabeled row as one that acts on the drawing. The profile knows that ``regen_assets`` acts at
    the icons and ``reroute`` at the composition, and that difference is exactly the one that
    decides whether a missing label can be filled in from what the row already says.
    """

    stored = str(experience.get("applicability") or "").strip().casefold()
    if stored in {"common", "domain"}:
        return stored
    action = experience.get("action") if isinstance(experience.get("action"), dict) else {}
    action_type = str(action.get("type") or "")
    profile = ACTION_MEMORY_PROFILE.get(action_type)
    stage = str(profile[1]) if profile else str(experience.get("apply_at") or "")
    return str(
        resolved_applicability(
            {"type": action_type, "applicability": stored},
            apply_at=stage,
            domain_profile={"domain": str((experience.get("trigger") or {}).get("domain") or "")},
        )["applicability"]
    )


def _unmeasurable_reason(experience: dict[str, Any]) -> str:
    """Say why the paired harness cannot examine this row, or ``""`` when it can.

    Two rows cannot be measured at all, and paying for them proves nothing. One never said where
    it applies, so the ladder refuses it before a case is even drawn. The other acts at a stage
    with no prefix a pair can share: freezing that stage would freeze away the very move being
    measured, and letting both arms plan their own figure would compare two different starting
    points. Naming both here refuses the exam before a paper is run rather than after hours of
    generation, and it is what lets the row be retired instead of re-queued for ever.

    "It never said where it applies" is read through the one reader the library writes with, so a
    row whose stage already says where it applies is measurable rather than retired. Reading only
    the stored field made the two sides disagree: the library stores what the proposal wrote, and
    this refused a row that wrote nothing -- which turned "no measurement was possible" into a
    status that read as "this lesson does not work".
    """

    if not resolved_target_applicability(experience):
        return "applicability_unspecified"
    if not _fork_stage_for(experience.get("apply_at")):
        return "stage_not_forkable"
    return ""


#: The program an arm compiled from, written beside the arm. For the control arm this is the
#: program its own run produced; the compiled arm writes the bytes it was handed here, so the
#: two arms' files can be compared rather than the claim that they started alike.
COMPILED_INPUT_NAME = "figure_program_input.json"


def _compiled_program_hash(run_dir: str | Path, *, filename: str = "figure_program.json") -> str:
    """The identity of the program an arm compiled from, so two arms can be shown to share one.

    A compile-only pair is meant to compile the same bytes twice. Reading that off the file each
    arm compiled from is what turns "the two arms started from the same figure" from a claim
    about the protocol into something the case can check: the compiled arm records the hash of
    the copy it was handed, and a runner that quietly drew a different figure would have to have
    written a different program to do it.
    """

    path = Path(str(run_dir or "")) / filename
    if not path.is_file():
        return ""
    try:
        return paired_configuration_hash(read_json(path))
    except (OSError, ValueError):
        return ""


#: The products whose bytes say whether the two arms of a pair produced anything different: the
#: figure program each arm wrote, the editable graph it bound, and the deck it compiled. They are
#: read off the files, not off the names of the moves the arms ran -- an arm can run the same named
#: move and still draw a different figure, and the figure is the thing a reader is asked about.
CONTRAST_PRODUCT_FILES = ("figure_program.json", "editable_composition.pptx")


def _file_digest(path: Path) -> str:
    """The sha256 of a file's bytes, read in chunks so a deck does not have to fit in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _product_hashes(run_dir: str | Path) -> dict[str, str]:
    """The identity of the products one arm wrote, keyed by file name.

    Only files that exist are named: an arm that wrote no deck is an arm that produced nothing to
    compare, and reporting an empty string for a file nobody wrote would read as "identical".
    """

    root = Path(str(run_dir or ""))
    hashes: dict[str, str] = {}
    for name in CONTRAST_PRODUCT_FILES:
        path = root / name
        if path.is_file():
            hashes[name] = _file_digest(path)
    graphs = sorted(root.rglob("editable_graph.json"))
    if graphs:
        hashes["editable_graph.json"] = _file_digest(graphs[-1])
    return hashes


def _arm_readings(run_dir: str | Path) -> dict[str, Any]:
    """The deterministic readings of one arm's own figure, or ``{}`` when it left no program."""

    path = Path(str(run_dir or "")) / "figure_program.json"
    if not path.is_file():
        return {}
    try:
        program = read_json(path)
    except (OSError, ValueError):
        return {}
    return program_readings(program)


def _compile_only_reason(experience: dict[str, Any]) -> str:
    """Say why a row cannot be measured by compiling one figure twice, or ``""`` when it can.

    Only a rule that acts at compile time can be measured this way: it decides how a figure is
    drawn, not what the figure is, so one program can be compiled with it and without it. A row
    acting anywhere else has to plan its own figure, which is a second generation and belongs on
    the paired-run protocol. A rule that states no parameters is refused too: there would be
    nothing for the compiler to honour and the second compilation would be the first one again.
    """

    action = experience.get("action") if isinstance(experience.get("action"), dict) else {}
    if str(action.get("type") or "") not in MECHANISM_ACTIONS:
        return "the row is not a compile-time rule"
    layout = layout_rules_from(action) or layout_rules_from(experience.get("layout_constraints"))
    router = router_rules_from(action) or router_rules_from(experience.get("routing_constraints"))
    if not layout and not router:
        return "the row states no parameters a compiler can honour"
    return ""


def _rows_of_selection(rows: list[dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    """The suite rows a drawn case set names, in the order the set named them."""

    by_paper = {str(item.get("paper_id") or item.get("case_id") or ""): item for item in rows}
    wanted = [str(entry.get("paper_id") or "") for entry in (selection.get("selected") or []) if isinstance(entry, dict)]
    return [by_paper[key] for key in wanted if key in by_paper]


def _unverifiable_report(
    *,
    experience: dict[str, Any],
    experience_id: str,
    reason: str,
    suite: str | Path,
    experience_library: str | Path,
    root: Path,
    spec: dict[str, Any],
    prompt_patch_id: str = "",
    profile: str = FULL_PROFILE,
) -> dict[str, Any]:
    """The report for a row this harness cannot examine, drawn before anything is generated.

    It is written in the same shape as a measured report so a reader that only knows the one
    shape still reads it, and its ``stopped_because`` names the dead end rather than leaving the
    row to look like a plain untried candidate.
    """

    return {
        "experience_id": str(experience.get("experience_id") or experience_id),
        "status": "unverifiable",
        "reason": str(reason),
        "blocked_because": str(reason),
        "validation_profile": str(profile or FULL_PROFILE).strip().casefold() or FULL_PROFILE,
        "target_metric": str(spec.get("target_metric") or "overall_score"),
        "results": [],
        "summary": {"cases": 0, "measured_cases": 0, "passed_cases": 0, "boundary_cases": 0, "mean_target_delta": None},
        "ladder": {"status": "unverifiable", "stopped_because": str(reason), "pairs_isolated": False, "all_cases_safe": False, "group_coverage": {}, "group_uncovered": []},
        "all_pairs_isolated": False,
        "all_cases_measured": False,
        "majority_improved": False,
        "required_shape": {"same_domain": 0, "holdout": 0, "passed": False},
        "suite": str(Path(suite).resolve()),
        "experience_library": str(Path(experience_library).resolve()),
        "out_dir": str(root),
        "subgraph_retrieval": {"used": False, "query_from": "", "selected": []},
        "case_selection": {"used": False, "reason": reason, "selected": [], "uncovered": []},
        "reselection": {"attempted": False, "reason": "the harness cannot examine this row", "first_status": "unverifiable", "status": "unverifiable", "first_case_ids": [], "second_case_ids": []},
        "prompt_patch_id": str(prompt_patch_id or "") or None,
        "pairwise_role": "",
        "cases_without_a_figure": [],
        "papers_without_a_figure": _read_undrawable_papers(root),
    }


def _expected_groups(experience: dict[str, Any], quota: dict[str, Any] | None) -> list[str]:
    """Groups the ladder should require for this experience's applicability."""

    kind = str(experience.get("applicability") or "").strip().casefold()
    names = DOMAIN_CASE_GROUPS if kind == "domain" else CASE_GROUPS
    resolved = dict(quota or DEFAULT_CASE_QUOTA)
    if kind == "domain":
        resolved["cross_domain_same_structure"] = 0
    return [name for name in names if int(resolved.get(name) or 0)]


def _executed_actions(run_dir: Path) -> list[str] | None:
    """Read the repair actions a run actually executed, so ON/OFF contrast is checkable.

    A seeded experience whose action matches what the default policy would have
    chosen produces an identical run, which is no evidence either way. Returns None
    when the run left no round log, which means "not measured" rather than "identical".

    The fingerprint includes the budget the executor used. Two ``relayout_global``
    rounds that differ only in ``dx`` are two different moves; a name-only sequence
    used to report them as identical.
    """
    path = Path(run_dir) / "evolution_rounds.json"
    if not path.exists():
        return None
    actions: list[str] = []
    for item in read_json(path).get("rounds") or []:
        if not isinstance(item, dict):
            continue
        change = item.get("bounded_change") if isinstance(item.get("bounded_change"), dict) else {}
        action = change.get("action") if isinstance(change.get("action"), dict) else item.get("repair_action")
        if not isinstance(action, dict):
            action = {}
        merged = dict(action)
        budget = change.get("applied_budget") if isinstance(change.get("applied_budget"), dict) else {}
        for key in ACTION_BUDGET_KEYS:
            if key in budget:
                merged[key] = budget[key]
        actions.append(action_execution_fingerprint(merged))
    return actions


def _patch_validation_target(patch: dict[str, Any]) -> dict[str, Any]:
    """Shape a prompt patch as the target record the paired harness validates.

    The harness compares paired runs, so it needs the same fields from either kind of
    Memory: an id, the stage, and the scope a run should keep to. Nothing here claims a
    repair action -- the patch travels as ``seed_prompt_patch``, not as an experience. The
    action field therefore carries the patch's own declared action and nothing invented: it
    used to be filled with ``recompile``, which read as a claim the docstring above denies.
    """

    return {
        "experience_id": str(patch.get("patch_id") or ""),
        "scope": "global",
        "memory_type": "repair",
        "apply_at": str(patch.get("apply_at") or "repair"),
        "hypothesis": str(patch.get("directive") or ""),
        "action": {"type": str(patch.get("action_type") or "")},
        # The patch's own scope when it carries one; otherwise the patch id stands in, which is
        # the smallest scope that still lets a counter-example be told apart from a case.
        "graph_scope": dict(patch.get("graph_scope") or {}) or {"object_ids": [str(patch.get("patch_id") or "")]},
        "repair_gain": patch.get("median_repair_gain"),
        "provenance": dict(patch.get("provenance") or {}),
        "excluded_conditions": list(patch.get("excluded_conditions") or []),
        "accepted": True,
        "status": "candidate",
        # A prompt patch is not a domain experience. The common ladder is the one this harness already ran.
        "applicability": "common",
    }


def _source_paper_ids(experience: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    """Name the papers this experience came from, so its case set is not drawn from them.

    A validation case built from the paper that produced the experience reproduces that paper,
    it does not generalize. Only names that identify a paper are collected -- an id, a DOI, an
    arXiv id or, when the suite itself names papers by path, the path -- because a file name is
    not an identity and matching on one would drop an unrelated paper that happens to share it.
    """

    provenance = experience.get("provenance") if isinstance(experience.get("provenance"), dict) else {}
    values = [
        spec.get("source_paper_id"),
        provenance.get("paper_id"),
        provenance.get("doi"),
        provenance.get("arxiv_id"),
        provenance.get("paper"),
        provenance.get("run"),
        experience.get("source_case"),
    ]
    return sorted({str(value).strip() for value in values if str(value or "").strip()})


def _paper_ids_spelled_in_paths(paths: set[str], records: list[dict[str, Any]]) -> set[str]:
    """The indexed papers whose id is spelled out as one whole segment of these paths.

    ``runs/10.1038_s41467-025-66153-5/attempt_01`` is the paper
    ``10.1038/s41467-025-66153-5``: only the separator between the prefix and the suffix is
    respelled. The id has to be a whole segment, so a directory that merely contains the same
    digits is not read as that paper.
    """

    found: set[str] = set()
    for record in records:
        paper_id = str(record.get("paper_id") or "").strip()
        if not paper_id:
            continue
        spellings = {paper_id, paper_id.replace("/", "_"), paper_id.replace("/", "-")}
        for path in paths:
            if spellings & set(Path(path).parts):
                found.add(paper_id)
                break
    return found


def _resolved_path(value: Any) -> str:
    """Resolve one path-shaped identity so two spellings of the same file compare equal."""

    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except (OSError, ValueError):
        return text


def _graph_record_of_source_paper(rows: list[dict[str, Any]], experience: dict[str, Any]) -> dict[str, Any]:
    """The graph record of the paper this experience was measured on, when a case carries it.

    A run records the paper it evolved as ``provenance.run``, which is that paper's resolved
    path, and a suite written with per-case graph records repeats the same path in the case it
    came from. Reading the basis off that case is what lets one suite judge experiences that
    came from different papers instead of selecting every set against one named basis.
    """

    provenance = experience.get("provenance") if isinstance(experience.get("provenance"), dict) else {}
    wanted = {
        _resolved_path(value)
        for value in (provenance.get("run"), provenance.get("paper"), experience.get("source_case"))
        if str(value or "").strip()
    }
    wanted.discard("")
    if not wanted:
        return {}
    for item in rows:
        record = item.get("graph_record")
        if not isinstance(record, dict) or not record.get("record_id"):
            continue
        if _resolved_path(item.get("paper")) in wanted:
            return record
    return {}


def _index_record_of_source_paper(
    rows: list[dict[str, Any]],
    experience: dict[str, Any],
    indexed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """The index record of the paper this experience was measured on.

    A suite can name that paper by path and keep the graph itself in the index.
    The case does not have to repeat the record.
    """

    provenance = experience.get("provenance") if isinstance(experience.get("provenance"), dict) else {}
    wanted = {
        _resolved_path(value)
        for value in (provenance.get("run"), provenance.get("paper"), experience.get("source_case"))
        if str(value or "").strip()
    }
    wanted.discard("")
    if not wanted:
        return {}
    paper_ids = {
        str(item.get("paper_id") or "").strip()
        for item in rows
        if _resolved_path(item.get("paper")) in wanted and str(item.get("paper_id") or "").strip()
    }
    if not paper_ids:
        # The run names the paper by the directory it was handed, and the corpus names that
        # directory after the paper's own id. When the cases name the paper by a path of their
        # own -- an archive path, not the run path -- the id spelled inside the run path is the
        # only thing the two ends still agree on, so the index is looked up by it.
        paper_ids = _paper_ids_spelled_in_paths(wanted, indexed_records)
    if not paper_ids:
        return {}
    for record in indexed_records:
        if str(record.get("paper_id") or "").strip() in paper_ids:
            return record
    return {}


def _shapes_the_row_names(experience: dict[str, Any]) -> list[str]:
    """Name the shapes the row's own figure held, in the row's own words.

    A row carries the shapes it was learned under twice over -- in the trigger it fired on and in
    the condition it holds -- and both are written in the vocabulary the index stores its motif
    histograms in, which is what makes them usable as a query. ``unknown`` is what the extractor
    writes when it could not name a shape, so it is dropped: a basis found by a shape nobody
    could name is not a basis.
    """

    trigger = experience.get("trigger") if isinstance(experience.get("trigger"), dict) else {}
    condition = experience.get("condition") if isinstance(experience.get("condition"), dict) else {}
    named = [str(item) for item in trigger.get("motifs") or []]
    named.extend(str(item) for item in condition.get("defect_patterns") or [])
    return sorted({item for item in named if item and item != "unknown"})


def _basis_record_from_the_row(
    experience: dict[str, Any],
    indexed_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Find the index figure a row that named no paper is measured against, and say which one.

    A row that never said which paper it came from used to be refused here: nothing named the
    paper its case set had to be drawn against, so no exam was ever set and the row sat
    unmeasured for good while its report read as a row that had been tried. The row does say what
    it is about -- the condition it was learned under, the shapes its own figure held, the objects
    it was scoped to -- and the index holds figures those can be answered by.

    What comes back is a real index record, not one composed for the occasion: the case set is
    then drawn against a figure's own structure, domain and embedding, exactly as it is drawn
    against a row's own graph record when the suite carries one. Among the records the row's own
    condition covers, the one taken is the one sharing most of the row's own shapes and then the
    one closest to the typical size of that set -- the basis stands in for the figure the row was
    learned on, and the largest figure a condition happens to admit is not that figure. Its paper
    is kept out of the cases the same way the row's own paper would be.

    A row that names no shape and no condition comes back empty: there is nothing to find it by.
    So does one whose condition no figure in the index answers. The objects the row was scoped to
    are not a basis on their own -- this index is a paper-level one, so it holds no figure that
    shares an object id with the row's own, and a set drawn by scope alone would be drawn by
    nothing.
    """

    condition = experience.get("condition") if isinstance(experience.get("condition"), dict) else {}
    scope = experience.get("graph_scope") if isinstance(experience.get("graph_scope"), dict) else {}
    shapes = _shapes_the_row_names(experience)
    asked = [key for key in CONDITION_KEYS if condition.get(key) not in (None, "", [], {})]
    if not shapes and not asked:
        return {}, {}
    covered = [
        record
        for record in indexed_records
        if not outside_condition(record, condition) and not outside_graph_scope(record, scope)
    ]
    if not covered:
        return {}, {}

    def counts(record: dict[str, Any]) -> tuple[int, int]:
        features = record.get("features") if isinstance(record.get("features"), dict) else {}
        return int(features.get("node_count") or 0), int(features.get("edge_count") or 0)

    def histogram(record: dict[str, Any]) -> dict[str, Any]:
        features = record.get("features") if isinstance(record.get("features"), dict) else {}
        return features.get("motif_histogram") if isinstance(features.get("motif_histogram"), dict) else {}

    def shared_shapes(record: dict[str, Any]) -> int:
        held = histogram(record)
        return sum(1 for name in shapes if name in held)

    sizes = [counts(record) for record in covered]
    middle = (statistics.median([size[0] for size in sizes]), statistics.median([size[1] for size in sizes]))

    def rank(record: dict[str, Any]) -> tuple[float, float, str, str]:
        nodes, edges = counts(record)
        return (
            -float(shared_shapes(record)),
            abs(nodes - middle[0]) + abs(edges - middle[1]),
            str(record.get("record_id") or ""),
            str(record.get("paper_id") or ""),
        )

    basis = min(covered, key=rank)
    note = {
        "why": "the row named no paper, so the set is drawn against the figure in the index its own condition and shapes match",
        "paper_id": str(basis.get("paper_id") or ""),
        "record_id": str(basis.get("record_id") or ""),
        "domain": str(basis.get("domain") or ""),
        "shapes_the_row_names": list(shapes),
        "shapes_shared": int(shared_shapes(basis)),
        "records_the_condition_covers": len(covered),
    }
    return basis, note


def _selection_query(
    experience: dict[str, Any],
    spec: dict[str, Any],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    """Find the graph record a case set is selected against, and say where it came from.

    A suite may name one basis record, which only fits a session that judges one experience. A
    setup that judges several has each row carry its own paper's graph record instead, so the
    set drawn for one experience is the set drawn for that experience's own subgraphs, domain
    and family rather than for whoever the suite happened to name.
    """

    named = spec.get("query_graph_record")
    if isinstance(named, dict) and named:
        return named, "suite"
    context = experience.get("graph_context")
    if isinstance(context, dict) and context:
        return context, "experience"
    source = _graph_record_of_source_paper(rows, experience)
    if source:
        return source, "case_graph_record"
    return {}, "none"


def kept_figure_judge_input(
    run_dir: str | Path,
    *,
    domain: str = "",
    allowed_scope: dict[str, Any] | None = None,
    prior_feedback: list[dict[str, Any]] | None = None,
    progress_sheet: dict[str, Any] | None = None,
) -> DirectJudgeInput | None:
    """Describe a finished run's kept figure the way a Judge question describes one.

    One reader for every question asked about a run that is already on disk, so a paired
    validation, an offline re-score and the continuous runner's boundary review all show the
    model the same figure in the same shape. ``None`` means the run left no kept figure to
    show, which refuses the question instead of guessing what the run produced.
    """

    root = Path(str(run_dir or ""))
    preview = root / "final_draft_preview.png"
    if not preview.is_file():
        preview = root / "draft_preview.png"
    program_path = root / "figure_program.json"
    if not preview.is_file() or not program_path.is_file():
        return None
    program = read_json(program_path)
    method_path = method_inventory_path(root)
    icons_path = root / "icon_generation_plan.json"
    return DirectJudgeInput(
        method=read_json(method_path) if method_path else {},
        layout={"panels": program.get("panels", []), "nodes": program.get("semantic_nodes", []), "connectors": program.get("arrows", [])},
        icons=read_json(icons_path) if icons_path.is_file() else {},
        program=program,
        preview=preview,
        domain_profile={"domain": str(domain or "")},
        prior_feedback=list(prior_feedback or []),
        allowed_scope=dict(allowed_scope or {"scope": "cross_paper_validation"}),
        progress_sheet=dict(progress_sheet) if progress_sheet else None,
    )


def _arm_judge_input(arm: dict[str, Any]) -> DirectJudgeInput | None:
    """Describe one validated arm's kept figure for the paired question."""

    return kept_figure_judge_input(
        arm.get("run_dir"),
        domain=str(arm.get("domain") or ""),
        allowed_scope={"scope": "cross_paper_validation"},
    )


def _pairwise_judge(options: dict[str, Any]) -> Any | None:
    """The acceptance role, when a suite runs one, so two arms can be compared.

    Only a suite that names the VLM judge can answer a two-state question; a suite that runs
    offline has no model to ask and keeps the measured-score comparison, which the report says.
    """

    mode = str(options.get("judge_mode") or "")
    if mode in {"", "heuristic", "off"}:
        return None
    _, frozen = build_direct_judges(
        mode="bootstrap" if mode == "vlm" else mode,
        frozen_model=options.get("frozen_judge_model") or options.get("judge_model"),
        timeout=int(options.get("vlm_timeout") or 180),
        retries=int(options.get("vlm_retries") or 1),
    )
    return frozen


def run_direct_experience_validation(
    suite: str | Path,
    experience_library: str | Path,
    experience_id: str = "",
    out: str | Path = "",
    *,
    prompt_patch_id: str = "",
    runner: Callable[..., dict[str, Any]] = run_paper_evolve_editable,
    reselect_on_failure: bool = True,
    run_options: dict[str, Any] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    validation_profile: str = "",
) -> dict[str, Any]:
    """Validate one experience or one prompt patch on the four-group case set.

    ``reselect_on_failure`` answers §7.7: when the conclusion is ``overfit_or_failed`` because the
    set could not measure the pair rather than because the experience regressed, the selector is
    asked once for another set, and both conclusions stay in the report.

    ``run_options`` overrides the suite's own for this call, which is how a caller that knows
    which Judge seats are in use right now keeps the validation measured by the same seats that
    measured the round the experience came from.
    """

    spec = read_json(suite)
    root = ensure_dir(out).resolve()
    store = ExperienceStore(experience_library)
    patch_store = PromptPatchStore(experience_library)
    patch: dict[str, Any] | None = None
    # What this measurement is about is pinned the first time it opens and read back from its own
    # directory afterwards. The row under test is the row the round earned, not whatever the
    # library happens to hold hours later: a row can be archived, retired or rewritten while the
    # pair runs, and a resumed attempt that asked the library again either measured a different
    # row or found none at all.
    pinned_path = root / ROW_UNDER_TEST_FILE
    pinned = read_json(pinned_path) if pinned_path.is_file() else None
    pinned = pinned if isinstance(pinned, dict) else {}
    # The profile is part of what a measurement is, so it is pinned beside the row: an attempt
    # resumed hours later is finished under the bar it opened with, not under whatever a caller
    # says now. A caller's word decides only while nothing has been pinned.
    pinned_profile = str(pinned.get("validation_profile") or "").strip().casefold()
    profile = str(pinned_profile or validation_profile or spec.get("validation_profile") or FULL_PROFILE).strip().casefold() or FULL_PROFILE
    pinned_experience = pinned.get("experience") if isinstance(pinned.get("experience"), dict) else None
    pinned_patch = pinned.get("prompt_patch") if isinstance(pinned.get("prompt_patch"), dict) else None
    if prompt_patch_id:
        patch = patch_store.get(str(prompt_patch_id)) or pinned_patch
        if not patch:
            raise ValueError(f"Unknown prompt patch: {prompt_patch_id}")
        experience = _patch_validation_target(patch)
    elif pinned_experience:
        experience = pinned_experience
    else:
        experience = store.get(experience_id)
        if not experience:
            raise ValueError(f"Unknown experience: {experience_id}")
    write_json(
        pinned_path,
        {
            "contract": ROW_UNDER_TEST_CONTRACT,
            "version": 1,
            "validation_profile": profile,
            "experience_id": str(experience.get("experience_id") or experience_id),
            "prompt_patch_id": str(prompt_patch_id or ""),
            "experience": experience,
            "prompt_patch": patch,
        },
    )
    target_id = str(prompt_patch_id) if patch is not None else str(experience_id)
    lifecycle = patch_store if patch is not None else store
    # A mark that has already been spent stays spent. The row on disk keeps the field the round
    # wrote, and the event that spent it is what says so; without this the row would be capped
    # again by a run whose own evidence already showed the defect does not reproduce.
    if bool(experience.get("carried_new_defect")) and any(
        str(item.get("event") or "") == CARRIED_NEW_DEFECT_CLEARED for item in lifecycle.events(target_id)
    ):
        experience = {**experience, "carried_new_defect": False}
    # The stage is read off the normalised row, which is the same row ``validate_experience_cases``
    # forks on: a row that names no stage is a repair-stage row by contract, not an unmeasurable
    # one, and reading the raw row here would refuse it for a field it never had to write.
    normalized_target = normalize_experience(experience)
    blocked_because = _unmeasurable_reason(normalized_target)
    if blocked_because:
        # Refused before a case is drawn and before a paper is generated: the row leaves the queue
        # as something the ladder cannot examine, instead of being re-queued on every call.
        refusal = _unverifiable_report(experience=experience, experience_id=experience_id, reason=blocked_because, suite=suite, experience_library=experience_library, root=root, spec=spec, prompt_patch_id=str(prompt_patch_id or ""), profile=profile)
        lifecycle.transition(target_id, "unverifiable", validation_report=str(root / "cross_paper_validation.json"), summary=refusal["summary"], blocked_because=blocked_because)
        lifecycle.append_event(target_id, "row_unverifiable", reason=blocked_because, apply_at=str(normalized_target.get("apply_at") or ""), applicability=str(normalized_target.get("applicability") or ""), applicability_resolved=resolved_target_applicability(normalized_target))
        write_json(root / "cross_paper_validation.json", refusal)
        return refusal
    raw_rows = [item for item in spec.get("cases", []) if isinstance(item, dict)]
    indexed_records: list[dict[str, Any]] = []
    if spec.get("graph_index"):
        indexed_records.extend(GraphIndex(spec["graph_index"]).records())
    indexed_records.extend(item["graph_record"] for item in raw_rows if isinstance(item.get("graph_record"), dict))
    selector = spec.get("selector") if isinstance(spec.get("selector"), dict) else {}
    quota = selector.get("quota") if isinstance(selector.get("quota"), dict) else None
    # The quick profile names one matched paper and nothing else, so it sets the quota itself and
    # overrides a suite's own: the caller lowered the bar on purpose, and a suite written for the
    # full set must not quietly put it back. Every other profile keeps the suite's quota, and the
    # default when the suite names none.
    if profile == QUICK_PROFILE:
        quota = quick_case_quota(normalized_target.get("applicability"))
    # A rule-class row is measured by compiling one figure twice, so it is cheap enough to be
    # asked to travel across more papers than a two-arm row before it is kept. A suite that says
    # how many cases it wants still decides; only a row left to the default is widened, and only
    # when its action is a mechanism action. Every other row keeps the default and goes on being
    # measured by two real runs.
    compile_only_reason = _compile_only_reason(normalized_target)
    if quota is None and not compile_only_reason:
        quota = dict(COMPILE_ONLY_CASE_QUOTA)
    source_papers = _source_paper_ids(experience, spec)
    query_record, query_from = _selection_query(experience, spec, raw_rows)
    if not query_record:
        query_record = _index_record_of_source_paper(raw_rows, experience, indexed_records)
        if query_record:
            query_from = "graph_index"
    basis_note: dict[str, Any] = {}
    if not query_record:
        # A row that names no paper the index holds used to stop here, and a row that stops here is
        # never measured: no case set is drawn, so there is no exam to fail and the row is left
        # sitting in the library for good. The row's own condition and shapes are what it does say
        # about itself, so the index is asked with those instead of with a paper it never named.
        query_record, basis_note = _basis_record_from_the_row(experience, indexed_records)
        if query_record:
            query_from = "row_condition"
    if query_from in {"case_graph_record", "graph_index", "row_condition"}:
        # The paper this experience was measured on is the basis of the selection, so it cannot
        # also be one of the cases: reproducing the paper it came from shows nothing.
        source_papers = sorted({*source_papers, str(query_record.get("paper_id") or "")} - {""})

    def choose(rows: list[dict[str, Any]], *, blocked: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Pick the cases to run and record why each was chosen and why each was left out."""

        if not indexed_records:
            # Nothing to draw a set from. Which of the two halves is missing is what the caller
            # has to act on, so the two are not reported as one sentence.
            return [], {
                "used": False,
                "reason": "the suite named no graph_index and no case carried a graph_record",
                "selected": [],
                "uncovered": [{"group": name, "reason": "the suite named no graph index"} for name in CASE_GROUPS],
            }
        if not query_record:
            return [], {
                "used": False,
                "reason": "nothing named the paper this experience came from and no shape or condition of the row is answered by any figure in the index, so the set had no basis to be drawn against",
                "selected": [],
                "uncovered": [{"group": name, "reason": "the row names no paper, shape or condition this index holds"} for name in CASE_GROUPS],
            }
        selection = select_validation_cases(
            query_record,
            indexed_records,
            quota=quota,
            exclude_paper_ids=[*source_papers, *blocked],
            excluded_conditions=experience.get("excluded_conditions") or [],
            graph_scope=experience.get("graph_scope") or {},
            condition=experience.get("condition") or {},
            top_k=int(selector.get("top_k") or 0),
            applicability=resolved_target_applicability(experience),
        )
        chosen_ids = [str(item.get("paper_id") or "") for item in selection["selected"]]
        by_paper = {str(item.get("paper_id") or item.get("case_id") or ""): item for item in rows}
        picked = [by_paper[paper_id] for paper_id in chosen_ids if paper_id in by_paper]
        if not picked:
            return [], {"used": True, "reason": "no candidate case matched the selection", **selection}
        return picked, {"used": True, **selection}

    def as_cases(rows: list[dict[str, Any]], selection: dict[str, Any]) -> list[ValidationCase]:
        """Build the cases for one set, preferring the group the selector assigned."""

        chosen = {str(item.get("paper_id") or ""): item for item in (selection.get("selected") or [])}
        built: list[ValidationCase] = []
        for index, item in enumerate(rows):
            key = str(item.get("paper_id") or item.get("case_id") or "")
            picked = chosen.get(key) or {}
            group = str(item.get("group") or picked.get("group") or "")
            split = str(item.get("split") or picked.get("split") or "same_domain")
            built.append(ValidationCase(str(item.get("case_id") or f"case_{index:02d}"), str(item.get("domain") or picked.get("domain") or ""), split, group))
        return built

    # A restart must measure the same cases the first call drew, so a drawn set is read back
    # rather than drawn again: re-drawing could pick a different set and throw away the cases the
    # interrupted call had already decided.
    undrawable = _read_undrawable_papers(root)
    selection_path = root / "validation_case_selection.json"
    kept_selection = read_json(selection_path) if selection_path.is_file() else {}
    # Unless a paper the set names has since been found to carry nothing to draw: reading that set
    # back would only refuse it the same way a second time, so it is drawn again with that paper
    # left out. A paper with no figure does not become a case by being offered again.
    kept_names_an_undrawable = isinstance(kept_selection, dict) and bool(
        {str(item.get("paper_id") or item.get("case_id") or "") for item in (kept_selection.get("selected") or [])} & set(undrawable)
    )
    if isinstance(kept_selection, dict) and kept_selection.get("used") and not kept_names_an_undrawable:
        case_selection = {**kept_selection, "resumed": True}
        raw_cases = _rows_of_selection(raw_rows, case_selection)
        if not raw_cases:
            case_selection = {**kept_selection, "resumed": True, "reason": "the drawn set names no case this suite still holds"}
    else:
        raw_cases, case_selection = choose(raw_rows, blocked=[*undrawable])
        write_json(selection_path, {key: value for key, value in case_selection.items() if key != "resumed"})
    retrieval_report: dict[str, Any] = {
        "used": bool(case_selection.get("used")),
        "query_from": query_from,
        "selected": list(case_selection.get("selected") or []),
    }
    if basis_note:
        retrieval_report["basis"] = dict(basis_note)
    cases = as_cases(raw_cases, case_selection)
    qualification = case_set_qualification(
        resolved_target_applicability(normalized_target),
        quota=quota,
        pool_counts=case_selection.get("candidate_counts"),
        profile=profile,
    )
    write_json(root / "case_set_qualification.json", qualification)
    if case_selection.get("used") and not qualification["ok"]:
        # An exam that cannot reach the ladder is refused before a paper is run. A suite that
        # asked for none of a required group, or a pool that cannot supply it, would otherwise
        # spend the whole run to arrive at a conclusion the case set had already fixed.
        refused_target = str(prompt_patch_id or "") if patch is not None else str(experience_id)
        refused_lifecycle = patch_store if patch is not None else store
        refusal = {
            "experience_id": str(experience.get("experience_id") or experience_id),
            "status": "unverifiable",
            "reason": str(qualification["reason"]),
            "blocked_because": str(qualification["reason"]),
            "validation_profile": profile,
            "case_set_qualification": qualification,
            "target_metric": str(spec.get("target_metric") or "overall_score"),
            "results": [],
            "summary": {"cases": 0, "measured_cases": 0, "passed_cases": 0, "boundary_cases": 0, "mean_target_delta": None},
            "ladder": {
                "status": "unverifiable",
                "stopped_because": str(qualification["reason"]),
                "pairs_isolated": False,
                "all_cases_safe": False,
                "group_coverage": dict(qualification["candidate_counts"]),
                "group_uncovered": [dict(item) for item in qualification["missing"]],
            },
            "all_pairs_isolated": False,
            "all_cases_measured": False,
            "majority_improved": False,
            "required_shape": {"same_domain": 0, "holdout": 0, "passed": False},
            "suite": str(Path(suite).resolve()),
            "experience_library": str(Path(experience_library).resolve()),
            "out_dir": str(root),
            "subgraph_retrieval": retrieval_report,
            "case_selection": case_selection,
            "reselection": {"attempted": False, "reason": "the case set was refused before any run", "first_status": "unverifiable", "status": "unverifiable", "first_case_ids": [], "second_case_ids": []},
            "prompt_patch_id": str(prompt_patch_id or "") or None,
            "pairwise_role": "",
            "cases_without_a_figure": [],
            "papers_without_a_figure": _read_undrawable_papers(root),
        }
        # The exam cannot be set for this row: no group the ladder needs can be drawn from this
        # suite's pool. It is named, retired, and not paid for again on the next call.
        refused_lifecycle.transition(refused_target, "unverifiable", validation_report=str(root / "cross_paper_validation.json"), summary=refusal["summary"], blocked_because=str(qualification["reason"]))
        refused_lifecycle.append_event(refused_target, "case_set_refused", reason=str(qualification["reason"]), missing=[dict(item) for item in qualification["missing"]])
        write_json(root / "cross_paper_validation.json", refusal)
        return refusal
    suite_options = spec.get("run_options") if isinstance(spec.get("run_options"), dict) else {}
    # A caller that owns the seats names them here: the suite was written with whatever checkpoint
    # was seated when it was built, and a validation decided by a different model than the round
    # that earned the experience would be measuring something else.
    common = {**suite_options, **(run_options or {})}
    # The pair is decided by showing both arms' kept figures together, the same way a round is
    # decided: one arm is the control (A) and the other carries the memory row (B). A suite with
    # no VLM judge behind it keeps the measured-score comparison and the report names which ran.
    first_case_options = raw_cases[0].get("run_options") if raw_cases and isinstance(raw_cases[0].get("run_options"), dict) else {}
    comparison_judge = _pairwise_judge({**common, **first_case_options})

    def make_runner(rows: list[dict[str, Any]], row_cases: list[ValidationCase], subdir: str = "") -> Callable[[ValidationCase, bool, dict[str, Any]], dict[str, Any]]:
        """Bind one set's rows to the case ids its own cases use.

        A reselected set is a different set of cases, so the lookup that turns a case back into
        the row it came from has to be built per set rather than shared.
        """

        by_id = {case.case_id: raw for case, raw in zip(row_cases, rows)}

        return lambda case, enabled, record: paired_runner(case, enabled, record, by_id, subdir)

    def paired_runner(case: ValidationCase, enabled: bool, record: dict[str, Any], by_id: dict[str, dict[str, Any]], subdir: str = "") -> dict[str, Any]:
        raw = by_id[case.case_id]
        case_root = (root / subdir) if subdir else root
        case_out = case_root / case.case_id / ("on" if enabled else "off")
        unusable = case_root / case.case_id / UNUSABLE_CASE_FILE
        if enabled and unusable.is_file():
            # The control arm could not draw this paper, so there is nothing here for the row to act
            # on. Reading the finding is the whole of this arm's work: a second reading would pay for
            # the same refusal and could not produce a figure either way.
            try:
                finding = read_json(unusable)
            except (OSError, ValueError):
                finding = {}
            arm = _unusable_case_arm(
                case,
                case_out,
                enabled=True,
                target=str(record.get("experience_id") or ""),
                reason=str((finding or {}).get("reason") or "the paper carries no figure contract to build"),
            )
            write_json(case_out / "paired_validation_result.json", arm)
            return arm
        # One arm is about forty minutes of generation. An arm that already ran and left its
        # measurement is reused instead of being paid for twice, which is what lets a set that
        # was interrupted mid-arm still make progress.
        try:
            kept_arm = read_json(case_out / "paired_validation_result.json")
        except (OSError, ValueError):
            kept_arm = None
        if isinstance(kept_arm, dict) and kept_arm.get("initial_state_hash"):
            return kept_arm
        options = {**common, **(raw.get("run_options") if isinstance(raw.get("run_options"), dict) else {})}
        options.update({
            "paper": raw["paper"],
            "reference": raw.get("reference"),
            "out": case_out,
            "domain_profile": raw.get("domain_profile") or case.domain,
            "evolution_enabled": True,
            "experience_enabled": False,
            "seed_experience": record if enabled and patch is None else None,
            "seed_prompt_patch": patch if enabled and patch is not None else None,
            "evolution_scope": str(record.get("scope") or "block"),
            "max_rounds": max(2, int(options.get("max_rounds", 3))),
            "fixed_rounds": True,
            "run_seed": int(raw.get("seed") or spec.get("seed") or 1729),
        })
        # The memory arm replays the control arm and forks where the memory acts, so the only
        # difference between the two arms is the record under test. Both of these describe the
        # protocol rather than the configuration, so neither belongs in the pair hash: the
        # control arm has no replay source by construction.
        fork_stage = _fork_stage_for(record.get("apply_at"))
        fork_source = case_out.parent / "off"
        if enabled and fork_stage:
            options.update({"replay_upstream_from": str(fork_source), "fork_before": fork_stage})
        else:
            # An interrupted control arm continues its own unfinished run: the per-stage replay
            # keeps every stage that attempt finished and re-plans only the rest, so the icons it
            # already drew are not drawn again.
            resume_source = _arm_resume_source(case_out)
            if resume_source:
                options["reuse_upstream_from"] = resume_source
                write_json(
                    case_out / "arm_resume.json",
                    {
                        "contract": "rfs.arm_resume",
                        "version": 1,
                        "source": resume_source,
                        "note": "this attempt continues the control arm's own unfinished run instead of starting it again",
                    },
                )
        # ``reuse_upstream_from`` says how this attempt was resumed, not what is being measured,
        # so it stays out of the pair hash for the same reason the replay source does: a resumed
        # attempt and a fresh one have to describe the same pair.
        pair_configuration = {key: value for key, value in options.items() if key not in {"out", "seed_experience", "seed_prompt_patch", "replay_upstream_from", "reuse_upstream_from", "fork_before"}}
        # Whichever record is under test belongs in the hash: a pair that toggles a patch
        # must not share a configuration hash with a pair that toggles an experience.
        pair_configuration["prompt_patch_id"] = str(prompt_patch_id or "")
        config_hash = paired_configuration_hash(pair_configuration)
        candidate_policy_hash = paired_configuration_hash({key: pair_configuration.get(key) for key in ("paper", "reference", "max_rounds", "candidates_per_round", "icon_candidates", "icon_mode", "run_seed", "evolution_scope", "fixed_rounds")})
        # The registry binding belongs with the other evaluator settings: two arms measured
        # by two different checkpoints served under the same model name would otherwise
        # share an evaluator hash and read as one comparison.
        evaluator_hash = paired_configuration_hash({key: pair_configuration.get(key) for key in ("judge_mode", "judge_model", "online_judge_model", "frozen_judge_model", "judge_registry", "judge_model_id", "visual_review_mode", "visual_review_model")})
        try:
            result = runner(**options)
        except ValueError as error:
            if UNPREPARED_PAPER_REFUSAL not in str(error):
                raise
            # This case's own paper has nothing to draw, which is a fact about the case rather than
            # about the row under test. It is recorded beside the case and reported as an unmeasured
            # arm: the ladder then refuses the set on ``pair_not_measured``, which is already a
            # reason to draw another set, instead of the whole measurement ending on its first case.
            finding = _note_unusable_case(unusable, case, str(raw.get("paper_id") or case.case_id), case_out, str(error))
            arm = _unusable_case_arm(
                case,
                case_out,
                enabled=bool(enabled),
                target=str(record.get("experience_id") or ""),
                reason=str(finding["reason"]),
            )
            write_json(case_out / "paired_validation_result.json", arm)
            return arm
        # Both scores are still read off the accepted state -- never off the advisory answer, which
        # carries none -- and they stay in the record as the trend. The case itself is decided by
        # the comparison below when the suite runs a judge that can answer one.
        judgement = next(
            (
                item
                for item in (result.get("acceptance_judgement"), result.get("judgement"))
                if isinstance(item, dict) and item
            ),
            {},
        )
        snapshot_path = case_out / "round_00" / "state_before.json"
        raw_state_hash = ""
        initial_state_hash = ""
        if snapshot_path.exists():
            snapshot = read_json(snapshot_path)
            raw_state_hash = str(snapshot.get("state_hash") or "")
            # The pair is compared on the skeleton, not on the bytes: every stage plans its own
            # colours and writes its own names, so a hash that counts those differences refuses
            # every honest pair. The raw hash stays in the record for diagnosis only.
            initial_state_hash = skeleton_state_hash(snapshot.get("state") or {})
        # A missing start state is not a licence to compare the two configurations instead: two
        # arms that were never shown to start from the same figure have not been measured against
        # each other, and the pair audit has to record that rather than pass on a technicality.
        # Falling back to the configuration hash made an unmeasured pair indistinguishable from a
        # clean one, which is the more dangerous of the two possible mistakes.
        pair_stop_reason = "" if initial_state_hash else "initial_state_not_measured"
        paired = {
            # An arm that could not be scored is recorded as unmeasured, never as zero: the pair
            # difference of two unscored runs is not a measurement of anything.
            "overall_score": measured_score(judgement),
            "measured": measured_score(judgement) is not None and bool(initial_state_hash),
            "pair_stop_reason": pair_stop_reason,
            "blocking_issue_count": int(judgement.get("blocking_issue_count") or 0),
            "dimension_scores": judgement.get("dimension_scores") or {},
            "run_dir": str(case_out),
            "ok": bool(result.get("ok")),
            "paired_config_hash": config_hash,
            "initial_state_hash": initial_state_hash,
            "raw_state_hash": raw_state_hash,
            "fork_stage": fork_stage,
            "fork_source": str(fork_source) if enabled and fork_stage else "",
            "candidate_policy_hash": candidate_policy_hash,
            "evaluator_hash": evaluator_hash,
            "memory_enabled": bool(enabled),
            "target_experience_id": record.get("experience_id") if enabled else None,
            "action_sequence": _executed_actions(case_out),
            "domain": case.domain,
            "compiled_program_hash": _compiled_program_hash(case_out),
            # What the arm produced and what its own figure measures. The pair's contrast is read
            # off these rather than off the names of the moves the two arms ran, so a memory that
            # changes the figure without changing the move's name is not read as "no contrast".
            "product_hashes": _product_hashes(case_out),
            "readings": _arm_readings(case_out),
        }
        write_json(case_out / "paired_validation_result.json", paired)
        return paired

    def compile_only_runner(case: ValidationCase, control: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
        """Compile the control arm's own figure a second time, with the rule applied.

        A rule-class row does not need a second generation: it changes how the figure is drawn,
        and the figure is the control arm's, already on disk. So the compiled arm is written
        beside the control arm from the same program, rendered by the same deterministic
        renderer, and shown to the comparison Judge with the same method and icon inputs the
        control arm showed -- so the only thing between the two arms is the rule.
        """

        control_dir = Path(str(control.get("run_dir") or ""))
        arm_dir = control_dir.parent / "on"
        layout = layout_rules_from(record.get("action")) or layout_rules_from(record.get("layout_constraints"))
        router = router_rules_from(record.get("action")) or router_rules_from(record.get("routing_constraints"))
        program_path = control_dir / "figure_program.json"
        if not program_path.is_file() or (not layout and not router):
            # Nothing to compile. The case is reported as unmeasured rather than as a pair that
            # agreed: an arm with no figure is not an arm that drew the same figure.
            return {**control, "run_dir": str(control_dir), "memory_enabled": True, "path": COMPILE_ONLY_PATH, "initial_state_hash": "", "measured": False, "compiled_program_hash": "", "pair_stop_reason": "compile_only_arm_drew_nothing"}
        ensure_dir(arm_dir)
        source_program = read_json(program_path)
        # Keep the bytes this arm compiled from, so the identity check compares two files that
        # both exist rather than two claims about one protocol.
        write_json(arm_dir / COMPILED_INPUT_NAME, source_program)
        updated, compile_report = apply_constraints(source_program, layout=layout or None, routing=router or None, out_dir=arm_dir)
        write_json(arm_dir / "figure_program.json", updated)
        render_rebuild_preview(updated, arm_dir, arm_dir / "final_draft_preview.png")
        # The Judge is shown the figure, and it reads the method and icon inputs from the arm's
        # own directory. Copying them keeps the two arms differing in the figure alone.
        inventory = method_inventory_path(control_dir)
        for source in (inventory, control_dir / "icon_generation_plan.json"):
            if source and Path(source).is_file():
                shutil.copyfile(source, arm_dir / Path(source).name)
        return {
            **control,
            "run_dir": str(arm_dir),
            "memory_enabled": True,
            "path": COMPILE_ONLY_PATH,
            # No model scored this arm on its own, and inventing the control arm's number would
            # read as a measurement that never happened. The pair is decided by the comparison.
            "overall_score": None,
            "dimension_scores": {},
            "measured": False,
            "target_experience_id": str(record.get("experience_id") or ""),
            "action_sequence": [str((record.get("action") or {}).get("type") or "")],
            "compiled_program_hash": _compiled_program_hash(arm_dir, filename=COMPILED_INPUT_NAME),
            "compile_report": compile_report,
            # The compiled arm's products and readings, so the escalation check can tell a second
            # compilation the rule changed from one that came out identical to its control.
            "product_hashes": _product_hashes(arm_dir),
            "readings": _arm_readings(arm_dir),
        }

    def pairwise_between_arms(off: dict[str, Any], on: dict[str, Any]) -> dict[str, Any]:
        """Ask which arm's kept figure a reader is better served by; B is the arm with the row."""

        before, after = _arm_judge_input(off), _arm_judge_input(on)
        names = {"candidate_a": "the run without the memory row", "candidate_b": "the run with the memory row"}
        if before is None or after is None:
            return {**normalize_pairwise_result({}, role=getattr(comparison_judge, "role", "unknown"), **names), "pairwise_unavailable": "one of the two arms left no kept figure to compare"}
        return comparison_judge.evaluate_pairwise(DirectPairJudgeInput(before=before, after=after, **names))

    # A rule-class row is measured by compiling one figure twice, which needs no second
    # generation; every other row keeps the paired-run protocol. Which one ran is written into
    # the report, together with the reason when the cheap path was not taken.
    compile_only_callable: Callable[[ValidationCase, dict[str, Any], dict[str, Any]], dict[str, Any]] | None = None
    compile_only_note: dict[str, Any] = {"used": False, "reason": compile_only_reason}
    if not compile_only_reason:
        if comparison_judge is None:
            compile_only_note["reason"] = "the suite runs no comparison Judge, so a compiled pair could not be decided"
        else:
            compile_only_callable = compile_only_runner
            compile_only_note = {
                "used": True,
                "reason": "",
                "layout_rules": layout_rules_from(normalized_target.get("action")) or layout_rules_from(normalized_target.get("layout_constraints")),
                "router_rules": router_rules_from(normalized_target.get("action")) or router_rules_from(normalized_target.get("routing_constraints")),
            }

    def case_progress(entry: dict[str, Any]) -> None:
        """Say, while the set is still running, that another case is decided.

        A validation is hours long and single-worker, so without this the loop's log goes silent
        for the whole set and a stalled measurement looks exactly like a running one.
        """

        if on_progress is not None:
            on_progress({"case_id": str(entry.get("case_id") or ""), "measured": bool(entry.get("measured")), "noise": bool(entry.get("noise")), "passed": bool(entry.get("passed")), "split": str(entry.get("split") or ""), "target_delta": entry.get("target_delta")})

    def run_set(rows: list[dict[str, Any]], selection: dict[str, Any], *, subdir: str = "") -> tuple[dict[str, Any], list[ValidationCase]]:
        """Run one case set and return its report with the cases it actually ran."""

        row_cases = as_cases(rows, selection)
        cases_root = (root / subdir / "cases") if subdir else (root / "cases")
        outcome = validate_experience_cases(
            experience,
            row_cases,
            make_runner(rows, row_cases, subdir),
            target_metric=str(spec.get("target_metric") or "overall_score"),
            tolerance=float(spec.get("tolerance") or 0.01),
            pairwise=pairwise_between_arms if comparison_judge is not None else None,
            expected_groups=_expected_groups(experience, quota),
            judge_samples=int(spec.get("judge_samples") or 3),
            case_dir=cases_root,
            compile_only=compile_only_callable,
            on_case_decided=case_progress,
            profile=profile,
        )
        return outcome, row_cases

    report, cases = run_set(raw_cases, case_selection)
    reselection: dict[str, Any] = {
        "attempted": False,
        "reason": "the first set was not refused for an unmeasured or unsafe pair",
        "first_status": report["status"],
        "status": report["status"],
        "first_case_ids": [case.case_id for case in cases],
        "second_case_ids": [],
    }
    stopped = str((report.get("ladder") or {}).get("stopped_because") or "")
    if reselect_on_failure and report["status"] in UNANSWERED_STATUSES and stopped in RESELECT_REASONS:
        # §7.7: a set that could not measure the pair is not a verdict on the experience, so the
        # selector is asked once more for a set drawn from papers this one did not use.
        used = [str(item.get("paper_id") or item.get("case_id") or "") for item in raw_cases]
        # The papers this call already found nothing to draw in are left out of the second draw as
        # well: a fresh set that names one of them would refuse itself the same way.
        alternate_rows, alternate_selection = choose(raw_rows, blocked=[*used, *_read_undrawable_papers(root)])
        alternate_cases = as_cases(alternate_rows, alternate_selection)
        if not alternate_rows or {case.case_id for case in alternate_cases} == {case.case_id for case in cases}:
            reselection["reason"] = f"the first set could not measure the pair ({stopped}) and the pool holds no other case set"
        else:
            report, cases = run_set(alternate_rows, alternate_selection, subdir="reselected")
            case_selection = alternate_selection
            reselection.update({
                "attempted": True,
                "reason": f"the first set could not measure the pair ({stopped}); another set was drawn",
                "status": report["status"],
                "second_case_ids": [case.case_id for case in cases],
            })
    # Every case whose paper carried nothing to draw is remembered, so the next attempt does not
    # draw it again and pay for the same refusal a second time.
    undrawable_findings = []
    for finding_path in sorted(root.rglob(UNUSABLE_CASE_FILE)):
        try:
            finding = read_json(finding_path)
        except (OSError, ValueError):
            continue
        if not isinstance(finding, dict):
            continue
        name = str(finding.get("paper_id") or finding.get("case_id") or "")
        _note_undrawable_paper(root, name, str(finding.get("reason") or ""))
        undrawable_findings.append({"case_id": str(finding.get("case_id") or ""), "paper_id": name, "reason": str(finding.get("reason") or ""), "finding": str(finding_path)})
    report.update({"suite": str(Path(suite).resolve()), "experience_library": str(Path(experience_library).resolve()), "out_dir": str(root), "validation_profile": profile, "subgraph_retrieval": retrieval_report, "case_selection": case_selection, "reselection": reselection, "compile_only": compile_only_note, "prompt_patch_id": str(prompt_patch_id or "") or None, "pairwise_role": getattr(comparison_judge, "role", "") if comparison_judge is not None else "", "cases_without_a_figure": undrawable_findings, "papers_without_a_figure": _read_undrawable_papers(root)})
    # A set that ended without measuring the pair says nothing about the row. Writing that as the
    # row's own status would take it out of the library's usable set for a reason it had no part in
    # -- and the row is still the only thing that ever measured itself on the paper it came from --
    # so the attempt is recorded in full and the rung the row holds stands.
    if str((report.get("ladder") or {}).get("stopped_because") or "") in UNMEASURED_REASONS:
        lifecycle.append_event(target_id, "validation_unmeasured", reason=str((report.get("ladder") or {}).get("stopped_because") or ""), validation_report=str(root / "cross_paper_validation.json"), summary=dict(report.get("summary") or {}), cases_without_a_figure=[item["case_id"] for item in undrawable_findings])
    else:
        lifecycle.transition(target_id, report["status"], validation_report=str(root / "cross_paper_validation.json"), summary=report["summary"], blocked_because=str((report.get("ladder") or {}).get("stopped_because") or ""))
    if bool((report.get("ladder") or {}).get("carried_new_defect_cleared")):
        # The row introduced a defect on the round that earned it and carried the mark into this
        # run. Every case here stayed safe, which is the run saying the defect does not
        # reproduce, so the mark is spent and the rung this run reached stands.
        lifecycle.append_event(target_id, CARRIED_NEW_DEFECT_CLEARED, status=str(report["status"]), reason="every case in this set stayed safe, so the defect the row introduced did not reproduce")
    if reselection["attempted"]:
        lifecycle.append_event(target_id, "suite_reselected", reason=str(reselection["reason"]), first_status=str(reselection["first_status"]), status=str(reselection["status"]), first_case_ids=list(reselection["first_case_ids"]), second_case_ids=list(reselection["second_case_ids"]))
    write_json(root / "cross_paper_validation.json", report)
    return report
