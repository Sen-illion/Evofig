"""Persist versioned prompt patches and assemble the ones a stage may inject.

A patch is rendered from an experience that already measured its own repair gain. The sentence
it carries is the seat's -- the reflection writes which requirement a stage's prompt failed to
state and how it should read instead -- and the standing is the measurement's: a patch with no
measured round behind it stays a candidate and is injected nowhere. So the library is still a
second view of the same measured Memory rather than a second place where instructions could be
invented, and every patch can still be traced to the case that justified it. Patches share the
experience lifecycle: they are candidates until a validation promotes them, a rollback disables
a version while the history stays readable, and the rung a patch earns from the paper that
produced it is injected back into that paper alone.

Assembly answers which patch a stage actually receives. Patches that would contradict each
other are never injected together; the winner is decided by the same fixed precedence the
experience Router uses, and every loser is reported with the reason it lost.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

from ...contracts.prompt_patch import (
    PROMPT_ASSEMBLY_ORDER,
    PROMPT_PATCH_CONTRACT,
    PROMPT_PATCH_LIFECYCLE,
    PROMPT_PATCH_REFUTATION_LIMIT,
    PROMPT_PATCH_TERMINAL,
    PROMPT_PATCH_VERSION,
    is_enforceable_prohibition,
    normalize_prompt_patch,
    prompt_patch_id,
    readable_condition,
    render_patch_directive,
    validate_prompt_patch,
)
from ...contracts.evolution import APPLY_AT_STAGES, normalize_judge_fingerprint
from .file_lock import append_jsonl, exclusive_lock, write_json_atomic
from .store import ALLOWED_APPEND_STATUSES, DIRECT_DOMAIN

SCHEMA_VERSION = "1.0"
PATCH_LIBRARY_FILE = "prompt_patches.jsonl"
PATCH_EVENTS_FILE = "prompt_patch_events.jsonl"
PATCH_ASSEMBLY_CONTRACT = "rfs.prompt_patch_assembly"
PATCH_ASSEMBLY_VERSION = 1
# The same rungs the experience store retrieves at: a patch carries no more evidence than the
# row it was rendered from.
DEFAULT_ASSEMBLY_STATUSES = frozenset({"same_domain_validated", "cross_domain_validated", "promoted"})

#: The rung a patch earns from the paper that produced it. It is deliberately not in the set
#: above: the measurement behind it is one run's own round, so it is worth enough to be injected
#: back into that run and not enough to reach a paper that measured nothing.
CURRENT_CASE_STATUS = "current_case_validated"
OWN_CASE_ASSEMBLY_STATUSES = DEFAULT_ASSEMBLY_STATUSES | {CURRENT_CASE_STATUS}

#: The revision of a stage's prompt that carries no patch at all: the wording the pipeline ships.
BASE_PROMPT_REVISION = "base"

# Which kind of patch an action's Memory can become, when the experience measured a gain.
GEOMETRY_ACTIONS = frozenset({"relayout_global", "relayout_slots", "resize_nodes", "reroute", "repair_interfaces"})
# The measured metrics whose limit is a floor rather than a ceiling, so a prohibition reads
# "the metric has to stay at or above the limit" instead of "at or below it".
FLOOR_METRICS = frozenset({"min_panel_occupancy"})


def patch_set_revision(stage: str, patches: Any) -> str:
    """Name the version of one stage's prompt, so a change of version is readable.

    ``base`` is the shipped wording with nothing riding on it. Any other revision names the exact
    set of patch versions the stage ran under, so a record that says a stage changed its prompt can
    say which one it changed to, and two rounds that name the same revision ran the same prompt.
    """

    items = sorted(
        f"{str(item.get('patch_id') or '')}@{int(item.get('version') or 0)}"
        for item in (patches or [])
        if isinstance(item, dict) and str(item.get("patch_id") or "")
    )
    if not items:
        return BASE_PROMPT_REVISION
    return "rev-" + hashlib.sha1(f"{stage}|{'|'.join(items)}".encode("utf-8")).hexdigest()[:12]


def prompt_revision_record(stage: str, patches: Any) -> dict[str, Any]:
    """The version of one stage's prompt, and whether anything unpromoted rides on it.

    A patch that has only been measured on the paper that wrote it makes the prompt a trial
    version: it is injected for that paper alone, and the record says so rather than leaving the
    reader to work the rung out from the patch ids.
    """

    chosen = [item for item in (patches or []) if isinstance(item, dict)]
    statuses = sorted({str(item.get("status") or "") for item in chosen if str(item.get("status") or "")})
    return {
        "base_revision": BASE_PROMPT_REVISION,
        "revision": patch_set_revision(stage, chosen),
        "trial": any(status not in DEFAULT_ASSEMBLY_STATUSES for status in statuses),
        "patch_ids": [str(item.get("patch_id") or "") for item in chosen if item.get("patch_id")],
        "patch_statuses": statuses,
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def patch_class_for(experience: dict[str, Any]) -> str:
    """Choose the class from what the experience measured, never from its prose.

    A repair the round refused is evidence for a prohibition; a repair the round
    accepted that moved bounded geometry becomes an executable patch; everything else
    stays advisory. Absolute ``repair_gain`` is trend evidence only: pairwise can keep
    a figure whose scores did not rise, and that row is still not a prohibition.
    """

    action = experience.get("action") if isinstance(experience.get("action"), dict) else {}
    action_type = str(action.get("type") or action.get("action") or "")
    if not experience.get("accepted"):
        return "prohibited"
    if action_type in GEOMETRY_ACTIONS or any(_number(action.get(key)) is not None for key in ("factor", "dx", "dy")):
        return "executable_geometry"
    return "advisory"


def build_prompt_patch(
    *,
    experience: dict[str, Any] | None,
    apply_at: str = "",
    patch_class: str = "",
    status: str = "candidate",
    version: int = PROMPT_PATCH_VERSION,
    condition: dict[str, Any] | None = None,
    authored: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Render one patch from a measured experience, or report that it cannot be one.

    ``condition`` overrides the condition the directive quotes, which is how a prohibition
    is given the measurable bound the deterministic check enforces; without it a refused
    repair keeps the human-readable condition it measured worse under.

    ``authored`` is the patch the reflection wrote -- where it applies, when, and the sentence
    the prompt should carry. Its wording is preferred to the rendered template because a
    sentence is the thing that has to change and a template cannot know which one; its
    ``apply_at`` is honoured only when it names a stage the pipeline runs, and its condition is
    read through the same filter as every other, so a bucket that reads as nothing cannot ride
    in on it. A model answer that names no stage, or names a requirement in no words at all, is
    discarded and the row's own measurement renders the directive instead.
    """

    source = experience if isinstance(experience, dict) else {}
    experience_id = str(source.get("experience_id") or "")
    if not experience_id:
        return None
    action = source.get("action") if isinstance(source.get("action"), dict) else {}
    resolved_class = str(patch_class or patch_class_for(source))
    written = authored if isinstance(authored, dict) else {}
    written_apply_at = str(written.get("apply_at") or "").strip()
    written_directive = str(written.get("directive") or "").strip()
    written_condition = readable_condition(written.get("condition"))
    resolved_condition = readable_condition(condition)
    if not resolved_condition:
        resolved_condition = written_condition
    if not resolved_condition:
        resolved_condition = readable_condition(source.get("condition"))
    if not resolved_condition:
        resolved_condition = readable_condition(source.get("context"))
    directive = written_directive or render_patch_directive(patch_class=resolved_class, condition=resolved_condition, action=action)
    resolved_apply_at = str(apply_at or written_apply_at or source.get("apply_at") or "")
    provenance = source.get("provenance") if isinstance(source.get("provenance"), dict) else {}
    cases = source.get("validated_cases") if isinstance(source.get("validated_cases"), list) else []
    gain = _number(source.get("repair_gain"))
    identifier = prompt_patch_id(
        experience_ids=[experience_id],
        apply_at=resolved_apply_at,
        patch_class=resolved_class,
        directive=directive,
    )
    return normalize_prompt_patch(
        {
            "patch_id": identifier,
            "version": int(version),
            "apply_at": resolved_apply_at,
            "class": resolved_class,
            "action_type": str(action.get("type") or action.get("action") or ""),
            "written_by": "model" if written_directive else "template",
            "condition": resolved_condition,
            "excluded_conditions": [dict(item) for item in (source.get("excluded_conditions") or []) if isinstance(item, dict)],
            "graph_scope": dict(source.get("graph_scope") or {}) if isinstance(source.get("graph_scope"), dict) else {},
            "directive": directive,
            "parameters": {key: action.get(key) for key in ("factor", "dx", "dy")},
            "source_experience_ids": [experience_id],
            "support": max(1, len(cases)),
            "median_repair_gain": gain,
            "status": status,
            "provenance": {
                "source": str(provenance.get("source") or ""),
                "run": str(provenance.get("run") or ""),
                "measured": bool(provenance.get("measured")),
                "round": provenance.get("round"),
            },
        }
    )


def prohibition_condition_for_experience(experience: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the measurable bound a refusal has to be enforced under, or nothing.

    A prohibition is enforcement only when a machine can check it, and the only metrics the
    check reads are the ones the decomposability report measured. The bound is the limit that
    was in force in the round that refused the repair, not the single value that round
    happened to measure: the limit is what the pipeline already treats as broken.
    """

    source = experience if isinstance(experience, dict) else {}
    observed = source.get("observed_effect") if isinstance(source.get("observed_effect"), dict) else {}
    for item in observed.get("decomposability_violations") or []:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric") or "").strip()
        limit = item.get("threshold")
        if not metric or isinstance(limit, bool) or not isinstance(limit, (int, float)):
            continue
        return {"metric": metric, "op": ">=" if metric in FLOOR_METRICS else "<=", "value": limit}
    return None


def build_patches_for_experience(experience: dict[str, Any] | None, *, authored: dict[str, Any] | None = None) -> dict[str, Any]:
    """Render the one patch a measured row supports, or say why it supports none.

    The row decides the class from what it measured, and a refusal needs the measurable bound
    its round recorded. A refusal without one becomes no patch at all rather than a directive
    the checker would report as unchecked, because a prohibition nobody can verify is not a
    prohibition.
    """

    source = experience if isinstance(experience, dict) else {}
    if not str(source.get("experience_id") or ""):
        return {"patch": None, "reason": "the row carries no experience_id"}
    condition = None
    if patch_class_for(source) == "prohibited":
        condition = prohibition_condition_for_experience(source)
        if condition is None:
            return {
                "patch": None,
                "reason": "prohibition_condition_unavailable",
                "note": "A refusal is enforced against the decomposability metrics, so it needs the measured break its round recorded.",
            }
    derived_from = ""
    written = authored if isinstance(authored, dict) else {}
    if written and not str(written.get("directive") or "").strip():
        # The seat answered the field and left it blank, which is allowed and is not a failure:
        # the row's own measurement is what renders the sentence in that case.
        written = {}
    if not written and str(source.get("guidance_source") or "") == "model":
        # The seat wrote a rule and no patch. A rule states how a figure should be drawn, which is
        # the same thing a patch states about a stage's prompt, so the sentence it already wrote is
        # rendered into the patch instead of a template standing in for it.
        sentence = str(source.get("guidance") or "").strip()
        if sentence:
            written = {"apply_at": str(source.get("apply_at") or ""), "condition": {}, "directive": sentence}
            derived_from = "rule"
    discarded = ""
    if written and str(written.get("apply_at") or "").strip() not in APPLY_AT_STAGES:
        discarded = "the patch names a stage the pipeline does not run"
        written = {}
        derived_from = ""
    rendered = build_prompt_patch(experience=source, condition=condition, authored=written)
    if rendered is None:
        return {"patch": None, "reason": "the row cannot render a patch"}
    validation = validate_prompt_patch(rendered)
    if not validation["ok"]:
        return {"patch": None, "reason": "contract_validation_failed", "errors": list(validation["errors"])}
    return {
        "patch": validation["patch"],
        "reason": "rendered",
        "authored_by": str(validation["patch"].get("written_by") or "template"),
        "authored_discarded": discarded,
        # Empty when the patch carries the seat's own sentence or a template's; "rule" when the
        # sentence is the rule the seat stated, which is the one derivation this step performs.
        "derived_from": derived_from,
        "permit": patch_work_permit(source, validation["patch"]),
    }


def patch_work_permit(experience: dict[str, Any] | None, patch: dict[str, Any] | None) -> dict[str, Any]:
    """Decide whether a patch has earned the rung that lets its own run inject it.

    The prose is the seat's; the standing is not. A patch may be injected back into the run that
    produced it only on a measurement that stands behind it, and the measurement is the same one
    the round itself was decided on: the comparison of the two figures preferred the one this
    patch's own repair drew. Asking for a positive absolute gain *as well* was asking for a second,
    different proof -- and since the loop now accepts a round the comparison preferred even when
    the score did not rise, a patch written by such a round could never be injected at all. The
    absolute reading is written on the permit as the accompanying number it is, and decides
    nothing. The one other licence stays as it was: a round that was refused grants only the
    prohibition the round's own break measured, and only when a machine can check its bound.
    """

    source = experience if isinstance(experience, dict) else {}
    if not isinstance(patch, dict) or not str(patch.get("patch_id") or ""):
        return {"granted": False, "reason": "there is no patch to permit"}
    if str(patch.get("class") or "") == "prohibited":
        if is_enforceable_prohibition(patch.get("condition")):
            return {
                "granted": True,
                "status": CURRENT_CASE_STATUS,
                "reason": "the round measured this repair worsening the figure under a bound the checker enforces",
            }
        return {"granted": False, "reason": "a prohibition with no measurable bound is not enforceable"}
    if not source.get("accepted"):
        return {"granted": False, "reason": "the row was not an accepted measurement"}
    gain = _number(source.get("repair_gain"))
    return {
        "granted": True,
        "status": CURRENT_CASE_STATUS,
        "reason": "the comparison this round was decided on preferred the figure this repair drew",
        "repair_gain": gain,
        "score_rose": bool(gain is not None and gain > 0),
        "pairwise_verdict": str(source.get("pairwise_verdict") or ""),
    }


def select_own_case_candidates(
    records: Any,
    *,
    paper_key: str = "",
    permitted_ids: Any = (),
) -> list[dict[str, Any]]:
    """Keep every row a run may draw on, scoping the own-case rung to its own paper.

    A patch that stands on one paper's own measurement is offered back to that paper and to
    no other. Two things can say a row belongs to this run: the run permitted it a moment ago,
    or the row's provenance names the paper this run is reading. The first is what lets a round
    this run just permitted ride in the very next round that plans that stage, and the second
    is what lets a later iteration of the same paper pick the patch up again. Every other rung
    passes through untouched and is judged by its own status at assembly time.
    """

    permitted = {str(item) for item in permitted_ids} if permitted_ids else set()
    key = str(paper_key or "")
    kept: list[dict[str, Any]] = []
    for record in records if isinstance(records, (list, tuple)) else []:
        if not isinstance(record, dict):
            continue
        if str(record.get("status") or "") == CURRENT_CASE_STATUS:
            provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
            if str(record.get("patch_id") or "") not in permitted and str(provenance.get("run") or "") != key:
                continue
        kept.append(record)
    return kept


def patch_sentence_key(patch: dict[str, Any]) -> tuple:
    """What makes two patches the same sentence: the stage it applies at and the words themselves.

    The class is left out on purpose. The same sentence filed once as an executable repair and once
    as advice is one sentence, and counting them as two is how eighteen patches came to hold five
    different things. The wording is compared exactly -- normalised for case and spacing only --
    because two directives that differ in a word are two sentences and neither can be discarded.
    """

    directive = " ".join(str(patch.get("directive") or "").split()).casefold()
    return (str(patch.get("apply_at") or ""), directive)


def merge_patch_sentences(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """One more round that asked for the same sentence: one row, one more support behind it.

    A duplicate is not a finding, so it is not filed as a second patch: the version that already
    stands is rewritten with the extra measurement folded in, and the experience that said it is
    added to the row's own list of sources. Whoever reads the library afterwards sees how many
    rounds asked for the sentence rather than how many times it was written down.
    """

    sources = {str(item) for item in (existing.get("source_experience_ids") or []) if str(item)}
    sources.update(str(item) for item in (incoming.get("source_experience_ids") or []) if str(item))
    gains = [value for value in (_number(existing.get("median_repair_gain")), _number(incoming.get("median_repair_gain"))) if value is not None]
    return {
        **existing,
        "version": int(existing.get("version") or 0) + 1,
        "support": max(1, int(existing.get("support") or 0)) + max(1, int(incoming.get("support") or 0)),
        "source_experience_ids": sorted(sources),
        "median_repair_gain": round(sum(gains) / len(gains), 4) if gains else None,
        # The newer wording of a condition that has moved on is kept: the sentence is the same,
        # and the condition under which it applies is what the later round measured it under.
        "condition": dict(incoming.get("condition") or existing.get("condition") or {}),
    }


def patch_safety_tier(patch: dict[str, Any]) -> int:
    """Rank how safe a patch is to inject: a prohibition outranks a preference."""

    return 2 if str(patch.get("class") or "") == "prohibited" else 1


def patch_gain(patch: dict[str, Any]) -> float:
    value = _number(patch.get("median_repair_gain"))
    return round(float(value), 4) if value is not None else 0.0


def patch_precedence_key(patch: dict[str, Any]) -> tuple:
    """Order two patches competing for one stage, deterministically and audibly."""

    return (
        -patch_safety_tier(patch),
        -patch_gain(patch),
        -int(patch.get("support") or 0),
        str(patch.get("patch_id") or ""),
        -int(patch.get("version") or 0),
    )


def condition_signature(patch: dict[str, Any]) -> str:
    condition = patch.get("condition") if isinstance(patch.get("condition"), dict) else {}
    return json.dumps(condition, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _assembly_report(*, apply_at: str, selected: list[dict[str, Any]], rejected: list[dict[str, Any]], considered: int, statuses: list[str]) -> dict[str, Any]:
    return {
        "contract": PATCH_ASSEMBLY_CONTRACT,
        "version": PATCH_ASSEMBLY_VERSION,
        "apply_at": apply_at,
        "order": list(PROMPT_ASSEMBLY_ORDER),
        "statuses": list(statuses),
        "considered": int(considered),
        "selected": [str(item.get("patch_id") or "") for item in selected],
        "rejected": rejected,
        "reason": "no patch applies here" if not selected else f"{len(selected)} patch(es) assembled for {apply_at}",
    }


def assemble_prompt_patches(
    patches: Any,
    *,
    apply_at: str,
    statuses: Any = None,
    required_forbidden: Any = (),
) -> dict[str, Any]:
    """Choose the patches one stage receives, and report why the others were refused.

    Only the newest non-terminal version of a patch is a candidate. Patches that contradict
    each other -- same stage, same action, same condition, opposite direction -- are resolved
    by precedence so the prompt never carries both an instruction and its prohibition.
    """

    allowed = {str(item) for item in statuses} if statuses else set(DEFAULT_ASSEMBLY_STATUSES)
    stage = str(apply_at or "")
    considered = 0
    latest: dict[str, dict[str, Any]] = {}
    for item in patches if isinstance(patches, (list, tuple)) else []:
        patch = normalize_prompt_patch(item if isinstance(item, dict) else {})
        if not patch["patch_id"] or str(patch["apply_at"]) != stage:
            continue
        if str(patch["status"]) in PROMPT_PATCH_TERMINAL or str(patch["status"]) not in allowed:
            continue
        considered += 1
        existing = latest.get(patch["patch_id"])
        if existing is None or int(patch["version"]) > int(existing["version"]):
            latest[patch["patch_id"]] = patch
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for patch in latest.values():
        groups.setdefault((str(patch["action_type"]), condition_signature(patch)), []).append(patch)
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: (item[0], item[1])):
        ranked = sorted(groups[key], key=patch_precedence_key)
        winner = ranked[0]
        selected.append(winner)
        for loser in ranked[1:]:
            rejected.append(
                {
                    "patch_id": str(loser.get("patch_id") or ""),
                    "version": int(loser.get("version") or 0),
                    "class": str(loser.get("class") or ""),
                    "reason": f"conflicts with {winner.get('patch_id')} for {winner.get('action_type') or 'the stage'}, which declares the safer boundary",
                }
            )
    selected.sort(key=patch_precedence_key)
    return {"patches": selected, "report": _assembly_report(apply_at=stage, selected=selected, rejected=rejected, considered=considered, statuses=sorted(allowed))}


class PromptPatchStore:
    """Persist prompt patches and their lifecycle events beside the experience library."""

    def __init__(self, path: str | Path):
        self.root = Path(path).resolve()
        self.domain = DIRECT_DOMAIN
        self.path = self.root / PATCH_LIBRARY_FILE
        self.events_path = self.root / PATCH_EVENTS_FILE
        self.manifest_path = self.root / "prompt_patch_manifest.json"

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def append_result(self, record: dict[str, Any]) -> dict[str, Any]:
        patch_id = str((record or {}).get("patch_id") or "")
        validation = validate_prompt_patch(record if isinstance(record, dict) else {})
        if not validation["ok"]:
            return {"ok": False, "reason": "contract_validation_failed", "patch_id": patch_id, "errors": list(validation["errors"])}
        normalized = validation["patch"]
        if normalized["status"] not in ALLOWED_APPEND_STATUSES:
            return {"ok": False, "reason": "status_not_allowed", "patch_id": patch_id, "status": normalized["status"], "allowed": sorted(ALLOWED_APPEND_STATUSES)}
        # One lock around the duplicate check, the append and the manifest: a wave writes this
        # library from more than one paper at a time.
        with exclusive_lock(self.path):
            rows = self._read()
            if any(str(item.get("patch_id")) == patch_id and int(item.get("version") or 0) == int(normalized["version"]) for item in rows):
                return {"ok": False, "reason": "duplicate_patch_version", "patch_id": patch_id, "version": normalized["version"]}
            self.root.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.path, normalized)
            write_json_atomic(self.manifest_path, {"schema_version": SCHEMA_VERSION, "records": len(rows) + 1, "patches": sorted({str(item.get("patch_id")) for item in rows} | {patch_id})})
        return {"ok": True, "reason": "appended", "patch_id": patch_id, "version": normalized["version"], "records": len(rows) + 1}

    def merge_sentence(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Fold a patch that says the same thing as one already standing into that one.

        The library is what decides; the caller does not have to know what is in it. A row already
        retired is not a home for a new measurement, so only the live versions are considered, and
        an incoming patch that *is* the row already filed is left to the ordinary append, which
        refuses the duplicate by version.
        """

        if not isinstance(patch, dict) or not str(patch.get("patch_id") or ""):
            return {"ok": True, "merged": False}
        key = patch_sentence_key(patch)
        existing: dict[str, Any] | None = None
        for record in self.records_with_status():
            if str(record.get("status") or "") in PROMPT_PATCH_TERMINAL:
                continue
            if patch_sentence_key(record) != key:
                continue
            if existing is None or int(record.get("version") or 0) > int(existing.get("version") or 0):
                existing = record
        if existing is None or str(existing.get("patch_id") or "") == str(patch.get("patch_id") or ""):
            return {"ok": True, "merged": False}
        merged = merge_patch_sentences(existing, patch)
        stored = self.append_result(merged)
        patch_id = str(existing.get("patch_id") or "")
        if not stored.get("ok"):
            return {"ok": False, "merged": False, "reason": str(stored.get("reason") or ""), "patch_id": patch_id}
        return {
            "ok": True,
            "merged": True,
            "patch_id": patch_id,
            "version": int(merged.get("version") or 0),
            "support": int(merged.get("support") or 0),
            "patch": merged,
        }

    def append(self, record: dict[str, Any]) -> bool:
        return bool(self.append_result(record)["ok"])

    def records(self) -> list[dict[str, Any]]:
        """Return the readable patch rows, for callers that only need to read them."""

        return self._read()

    def append_event(self, patch_id: str, event: str, **payload: Any) -> dict[str, Any]:
        record = {"patch_id": str(patch_id), "event": str(event), **payload}
        self.root.mkdir(parents=True, exist_ok=True)
        return append_jsonl(self.events_path, record)

    def events(self, patch_id: str = "") -> list[dict[str, Any]]:
        """Return the lifecycle events, optionally only those of one patch."""

        if not self.events_path.exists():
            return []
        want = str(patch_id or "")
        rows: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            if want and str(value.get("patch_id") or "") != want:
                continue
            rows.append(value)
        return rows

    def lifecycle_status(self, patch_id: str, default: str | None = None) -> str | None:
        status = default
        for row in self._read():
            if str(row.get("patch_id") or "") == str(patch_id):
                status = str(row.get("status") or status)
        for event in self.events(patch_id):
            if event.get("status"):
                status = str(event["status"])
        return status

    def records_with_status(self) -> list[dict[str, Any]]:
        """Return every version with its effective status, so a lifecycle event is not ignored."""

        return [{**record, "status": self.lifecycle_status(str(record.get("patch_id") or ""), str(record.get("status") or ""))} for record in self._read()]

    def revalidation_fingerprint(self, patch_id: str) -> dict[str, Any] | None:
        """Return the Judge a re-validation last re-measured a patch under, or None if never."""

        found: dict[str, Any] | None = None
        for event in self.events(patch_id):
            if isinstance(event.get("judge_fingerprint"), dict):
                found = normalize_judge_fingerprint(event["judge_fingerprint"])
        return found

    def last_validation_report(self, patch_id: str) -> str:
        """Return the newest validation report an event points at, so a sweep can re-run it."""

        found = ""
        for event in self.events(patch_id):
            if str(event.get("validation_report") or ""):
                found = str(event["validation_report"])
        return found

    def get(self, patch_id: str, version: int | None = None) -> dict[str, Any] | None:
        found: dict[str, Any] | None = None
        for record in self._read():
            if str(record.get("patch_id") or "") != str(patch_id):
                continue
            if version is not None and int(record.get("version") or 0) != int(version):
                continue
            found = record
        if found is None:
            return None
        return {**found, "status": self.lifecycle_status(patch_id, str(found.get("status") or "candidate"))}

    def versions(self, patch_id: str) -> list[dict[str, Any]]:
        """Return every recorded version of one patch, so a rollback has a history."""

        rows = [record for record in self._read() if str(record.get("patch_id") or "") == str(patch_id)]
        return sorted(rows, key=lambda item: int(item.get("version") or 0))

    def transition(self, patch_id: str, status: str, **payload: Any) -> dict[str, Any]:
        return self.append_event(patch_id, "status_changed", status=status, **payload)

    def disable(self, patch_id: str, *, reason: str = "") -> dict[str, Any]:
        """Roll a patch back: its versions stay readable but none of them is injected."""

        return self.append_event(patch_id, "disabled", status="disabled", reason=str(reason))

    def refutation_streak(self, patch_id: str) -> int:
        """How many times in a row this patch has been measured and lost.

        The count is read from the patch's own lifecycle events rather than kept in one run's
        memory, so two iterations of the same paper cannot each see a fresh streak. A win -- an
        ``upheld`` event -- resets it, because a patch that helps two rounds out of three is
        still a patch that helps.
        """

        streak = 0
        for event in self.events(str(patch_id)):
            name = str(event.get("event") or "")
            if name == "upheld":
                streak = 0
            elif name == "refuted":
                streak += 1
        return streak

    def select(self, *, apply_at: str, statuses: Any = None, required_forbidden: Any = ()) -> dict[str, Any]:
        """Assemble the patches one stage receives from the whole library."""

        candidates = []
        for record in self._read():
            patch = normalize_prompt_patch(record)
            effective = self.lifecycle_status(str(patch["patch_id"]), str(patch["status"]))
            candidates.append({**patch, "status": effective or patch["status"]})
        return assemble_prompt_patches(candidates, apply_at=apply_at, statuses=statuses, required_forbidden=required_forbidden)


__all__ = [
    "BASE_PROMPT_REVISION",
    "CURRENT_CASE_STATUS",
    "DEFAULT_ASSEMBLY_STATUSES",
    "OWN_CASE_ASSEMBLY_STATUSES",
    "FLOOR_METRICS",
    "GEOMETRY_ACTIONS",
    "PATCH_ASSEMBLY_CONTRACT",
    "PATCH_ASSEMBLY_VERSION",
    "PATCH_EVENTS_FILE",
    "PATCH_LIBRARY_FILE",
    "PROMPT_PATCH_CONTRACT",
    "PROMPT_PATCH_LIFECYCLE",
    "PROMPT_PATCH_REFUTATION_LIMIT",
    "PROMPT_PATCH_TERMINAL",
    "PROMPT_PATCH_VERSION",
    "PromptPatchStore",
    "assemble_prompt_patches",
    "patch_set_revision",
    "prompt_revision_record",
    "build_patches_for_experience",
    "build_prompt_patch",
    "condition_signature",
    "patch_class_for",
    "patch_gain",
    "patch_precedence_key",
    "patch_safety_tier",
    "patch_sentence_key",
    "patch_work_permit",
    "merge_patch_sentences",
    "prohibition_condition_for_experience",
    "select_own_case_candidates",
]
