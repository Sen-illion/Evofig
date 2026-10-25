"""Append-only Memory store for the direct editable evolution pipeline.

Retrieval ranks records by the stage they apply at and by how similar their motifs and
structure are to the query, using the similarity measures the graph index already
defines, and every row keeps the components behind its score. When no embedding can be
produced the retrieval reports the mode and the embedder it tried, so a structure-only
order is never passed off as a vector order.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ...contracts.evolution import normalize_judge_fingerprint, validate_experience
from ...providers.embeddings import EmbeddingUnavailable, resolve_embedder
from .actions import normalize_mechanism_rule
from .experience import count_bucket
from .file_lock import append_jsonl, exclusive_lock, write_json_atomic
from .rule_binding import rule_binding_gate, split_rule_binding
from .retrieval_query import (
    EXCLUDED_CONDITION_EVENT,
    condition_similarity,
    embedding_similarity,
    exclusion_reason,
    exclusion_view,
    motif_overlap,
    query_features,
    record_text,
    structure_similarity,
)

SCHEMA_VERSION = "2.0"
DIRECT_DOMAIN = "direct_editable"
ALLOWED_APPEND_STATUSES = {"candidate", "current_case_validated", "quick_validated", "same_domain_validated", "cross_domain_validated", "domain_validated", "promoted", "retrying", "overfit_or_failed", "disabled", "refuted", "inconclusive", "common_knowledge", "measurement_noise", "unverifiable"}

#: The event a restated lesson writes instead of a second row. The row that already carries the
#: move grows a condition and an evidence entry there, so the library keeps one line per lesson
#: while the conditions it was measured under stay readable.
MERGED_WITNESS_EVENT = "merged_into"


#: The rung classes a restatement may fold within. Folding is only allowed between two rows the
#: ladder stands in the same place as: a validated lesson restating a validated one is one lesson,
#: and two candidates saying the same thing are one candidate, but folding a candidate into a
#: validated row would hand the newer measurement to a rung it never earned, and folding anything
#: into a row that was refuted, retired or never measured would bury the record of that measurement
#: -- which is the one thing the ladder has to be able to see about it.
MERGE_RUNG_CLASSES = {
    "candidate": "candidate",
    "current_case_validated": "own_case",
    "quick_validated": "validated",
    "same_domain_validated": "validated",
    "cross_domain_validated": "validated",
    "domain_validated": "validated",
    "promoted": "validated",
    "common_knowledge": "validated",
}


def merge_rung_class(status: Any) -> str:
    """Which group of rungs a row stands on, or ``""`` for a row the ladder has no position for."""

    return MERGE_RUNG_CLASSES.get(str(status or ""), "")


def _rule_sentence(record: dict[str, Any]) -> str:
    """The words a row says its lesson in, whichever of the row's shapes carried them."""

    rule = record.get("rule") if isinstance(record.get("rule"), dict) else {}
    strategy = record.get("strategy") if isinstance(record.get("strategy"), dict) else {}
    for value in (record.get("guidance"), rule.get("text"), rule.get("guidance"), strategy.get("instruction"), record.get("hypothesis")):
        text = " ".join(str(value or "").split())
        if text:
            return text.casefold()
    return ""


def _rule_stance(record: dict[str, Any]) -> str:
    """Whether the row says the move works or says not to make it.

    The same words and the same move can be either, and the two are opposite lessons: a row that
    measured a gain and a row that was refused carry the same sentence about the same repair. They
    must not fold into one another, or the refusal would be filed as a success.
    """

    if record.get("accepted") is False:
        return "refused"
    if str(record.get("status") or "") in {"refuted", "overfit_or_failed"}:
        return "refused"
    return "works"


def merge_rule_key(record: dict[str, Any]) -> str:
    """What makes two rows the same lesson: the move, the words it is said in, and its scope.

    The condition is deliberately not part of the key. One lesson answers many conditions -- that
    is what makes it a lesson -- and the conditions it was measured under are what the merged row's
    own tables carry. The words are part of it, and that half matters more than it looks: the same
    action type is not the same lesson, and keying on the action alone folded every ``reroute`` row
    in the library into the first one, which would have hidden the conditions the later ones were
    measured under behind a row that never saw them. An empty key means the row names no move, or
    says nothing in words, which is not a lesson to fold.
    """

    action = record.get("action") if isinstance(record.get("action"), dict) else {}
    if not action:
        rule = record.get("rule") if isinstance(record.get("rule"), dict) else {}
        action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
    if not str(action.get("type") or ""):
        return ""
    sentence = _rule_sentence(record)
    if not sentence:
        return ""
    parameters: dict[str, Any] = {}
    for holder in ("constraints", "params"):
        value = action.get(holder)
        if isinstance(value, dict):
            parameters.update(value)
    forbidden = sorted({str(item) for item in (record.get("forbidden_changes") or []) if str(item)})
    return json.dumps(
        {
            "type": str(action.get("type")),
            "words": sentence,
            "stance": _rule_stance(record),
            "parameters": parameters,
            "forbidden": forbidden,
            "scope": str(record.get("scope") or ""),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
RETRIEVAL_REPORT_CONTRACT = "rfs.experience_retrieval_report"
# The one place that says which rungs a run may inject. A rung belongs here once a paired
# validation walked far enough for the row to stand behind a gain elsewhere; the rung an evolve
# station writes for itself before any pair ran (``current_case_validated``) deliberately does
# not.
DEFAULT_RETRIEVAL_STATUSES = frozenset({"same_domain_validated", "cross_domain_validated", "domain_validated", "promoted"})

# The bar a run may lower to on purpose: the same rungs plus the two names a row carries before any
# pair has been run elsewhere -- ``candidate``, which is the status a station writes and the one
# the library actually stores, and ``current_case_validated``, which is what the ladder leaves on a
# row whose reproduction on a second paper was not shown. Injecting either is a bet: the row was
# measured on the paper it came from and nowhere else. The point of the relaxed bar is that the bet
# is taken knowingly, by a caller that named it, and never by default. Nothing here changes what
# the validation ladder writes; it only decides what a run is willing to draw from.
RELAXED_RETRIEVAL_STATUSES = DEFAULT_RETRIEVAL_STATUSES | {"candidate", "current_case_validated", "quick_validated"}

# The names a caller may use for the bar instead of spelling the rungs out.
RETRIEVAL_STATUS_PRESETS: dict[str, frozenset[str]] = {
    "strict": DEFAULT_RETRIEVAL_STATUSES,
    "relaxed": RELAXED_RETRIEVAL_STATUSES,
}


def resolve_retrieval_statuses(spec: Any) -> set[str] | None:
    """Read the retrieval bar a caller named, or None when it named none.

    A run that names no bar keeps the default. A name outside the presets is refused rather than
    read as "no bar": a typo that silently widened the library to every status would let a run
    inject rows its own record claims it did not read, which is worse than failing to start.
    """

    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None
    if isinstance(spec, str):
        named = spec.strip().casefold()
        if named in RETRIEVAL_STATUS_PRESETS:
            return set(RETRIEVAL_STATUS_PRESETS[named])
        wanted = {item.strip() for item in named.split(",") if item.strip()}
    elif isinstance(spec, (list, tuple, set, frozenset)):
        wanted = {str(item).strip() for item in spec if str(item).strip()}
    else:
        raise ValueError(f"a retrieval bar must be a preset name or a list of rungs, not {type(spec).__name__}")
    unknown = sorted(wanted - ALLOWED_APPEND_STATUSES)
    if unknown:
        raise ValueError(f"unknown retrieval rung(s): {', '.join(unknown)}")
    return wanted

RETRIEVAL_WEIGHTS = {"motif": 0.42, "structure": 0.22, "condition": 0.26, "embedding": 0.10}
BUCKET_DISTANCE = {0: 1.0, 1: 0.5}
ARCHIVE_REPORT_CONTRACT = "rfs.experience_archive_report"
ARCHIVE_REPORT_FILE = "archive_report.json"


def untransferable_reason(record: dict[str, Any] | None, *, figure: dict[str, Any] | None = None) -> str:
    """Why no later paper could be drawn by this row, or ``""`` when one could.

    The bar is the gate a new row is admitted by, applied to a row that is already stored: a row
    that names the figure it came from, or that states no rule a later paper could act on, is not
    a lesson. Guidance is the model's own prose and the one part that always travels, so a row
    carrying any is kept whatever the gate says about the ids in its action -- and whatever its
    status is, including the rows that lost their paired validation. A row is judged on its own
    text, so the same row is judged the same way by whoever reads the report later.
    """

    source = record if isinstance(record, dict) else {}
    if str(split_rule_binding(source)[0].get("guidance") or "").strip():
        return ""
    gate = rule_binding_gate(
        source,
        default_action_fingerprint=str(source.get("default_action_fingerprint") or ""),
        figure=figure,
    )
    if not gate["ok"]:
        return str(gate["reason"] or "no_transferable_rule")
    # The gate admits a mechanism-level row on its action name alone, and the name is not the
    # rule: the parameters are. A mechanism row without them can never be injected, so it is a
    # shell like any other row that says nothing another paper can act on.
    action = source.get("action")
    if normalize_mechanism_rule(action if isinstance(action, dict) else {}):
        return ""
    return "no_transferable_rule"


def _domain_experience_matches(record: dict[str, Any], context: dict[str, Any] | None) -> bool:
    """A domain-validated row is injectable only in the domain it names."""

    query = context if isinstance(context, dict) else {}
    paper_domain = str(query.get("domain") or "").strip()
    if bool(query.get("domain_missing")) or paper_domain.casefold() in {"", "unknown", "missing"}:
        return False
    named = str(record.get("applicability_domain") or "").strip()
    return bool(named) and named == paper_domain


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _retrieval_embedder(spec: Any) -> tuple[Any, dict[str, Any]]:
    """Resolve the embedder a retrieval asked for, or report why it could not be used."""

    requested = str(spec or "hashing-ngram")
    try:
        encoder = resolve_embedder(spec)
        fingerprint = encoder.fingerprint()
        if int(fingerprint.get("dimension") or 0) <= 0:
            raise EmbeddingUnavailable("the embedder reports no usable dimension")
    except (EmbeddingUnavailable, OSError, ValueError) as exc:
        return None, {"spec": requested, "kind": "unavailable", "fingerprint": {}, "reason": f"the requested embedder is unavailable: {exc}"}
    return encoder, {"spec": requested, "kind": str(fingerprint.get("kind") or ""), "fingerprint": fingerprint, "reason": ""}


def _rank_score(components: dict[str, Any]) -> float:
    """Average the available similarity components with their fixed weights."""

    available = {name: float(value) for name, value in components.items() if value is not None and name in RETRIEVAL_WEIGHTS}
    denominator = sum(RETRIEVAL_WEIGHTS[name] for name in available)
    return sum(RETRIEVAL_WEIGHTS[name] * value for name, value in available.items()) / denominator if denominator else 0.0


def _hit_reason(components: dict[str, Any]) -> str:
    """Name the components that placed a record, so a hit can be audited."""

    parts = [f"{name}={float(value):.2f}" for name, value in components.items() if value is not None]
    return "ranked by " + ", ".join(parts) if parts else "no similarity component was available for this query"


def _query_vector(query: dict[str, Any], encoder: Any) -> list[float]:
    text = str(query.get("text") or "").strip()
    if encoder is None or not text:
        return []
    return [float(item) for item in encoder.encode([text])[0]]


def _record_vector(record: dict[str, Any], encoder: Any) -> list[float]:
    stored = record.get("embedding")
    if isinstance(stored, list) and stored:
        return [float(item) for item in stored]
    if encoder is None:
        return []
    text = record_text(record)
    return [float(item) for item in encoder.encode([text])[0]] if text else []


def _retrieval_report(*, mode: str, degraded: bool, reason: str, candidates: int, apply_at: str = "", statuses: list[str] | None = None, embedder: dict[str, Any] | None = None, embedded_rows: int = 0, excluded_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "contract": RETRIEVAL_REPORT_CONTRACT,
        "version": 1,
        "mode": mode,
        "degraded": bool(degraded),
        "reason": str(reason),
        "apply_at": str(apply_at or ""),
        "statuses": list(statuses or []),
        "candidates": int(candidates),
        "embedded_rows": int(embedded_rows),
        # A row kept out by an excluded condition is named here rather than dropped, so a
        # thin result set is always explainable from the report alone.
        "excluded_rows": list(excluded_rows or []),
        "weights": dict(RETRIEVAL_WEIGHTS),
        "embedder": dict(embedder or {"spec": "", "kind": "", "fingerprint": {}, "reason": ""}),
    }


class ExperienceStore:
    """Persist direct node/edge/slot/object Memory and lifecycle events."""

    def __init__(self, path: str | Path):
        self.root = Path(path).resolve()
        self.path = self.root / "judge_experiences.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.events_path = self.root / "experience_events.jsonl"

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and str(value.get("domain") or DIRECT_DOMAIN) == DIRECT_DOMAIN:
                rows.append(value)
        return rows

    def append_result(self, record: dict[str, Any], *, figure: dict[str, Any] | None = None) -> dict[str, Any]:
        """Append one Memory record and report why it was rejected when it is.

        ``figure`` is the layout the row was measured on. The gate needs it to check the one
        kind of case binding a pattern cannot see: a rule that names a box by the name the
        figure gave it.
        """
        experience_id = str(record.get("experience_id") or "")
        status = record.get("status")
        if status not in ALLOWED_APPEND_STATUSES:
            return {"ok": False, "reason": "status_not_allowed", "experience_id": experience_id, "status": status, "allowed": sorted(ALLOWED_APPEND_STATUSES)}
        # The row is read as a rule before it is read as a contract: a rule that names the figure
        # it came from cannot be retrieved for another paper, and a rule that restates the move
        # the run would have made anyway is not a finding. Either way the library stays out of it.
        # The round's own policy move is read off the row and handed to the gate, because the
        # gate cannot compute it: only the caller that ran the round knows what the mechanism
        # would have done without Memory. Passing it is what makes the ``common_knowledge``
        # branch reachable -- for a long time nothing ever did, so the gate was dead code.
        gate = rule_binding_gate(record, default_action_fingerprint=str(record.get("default_action_fingerprint") or ""), figure=figure)
        if not gate["ok"]:
            return {"ok": False, "reason": gate["reason"], "experience_id": experience_id, "status": gate["reason"], "evidence": list(gate["evidence"]), "rule": gate["rule"]}
        validation = validate_experience(record)
        if not validation["ok"]:
            return {"ok": False, "reason": "contract_validation_failed", "experience_id": experience_id, "errors": list(validation["errors"])}
        normalized = {**record, "domain": DIRECT_DOMAIN}
        # The duplicate check and the append have to be one step: two runs of a wave read the
        # same rows, both find no duplicate, and both write the row. The lock makes the check
        # and the write one decision, and the manifest lands by rename so a reader never sees a
        # count that belongs to a different set of rows.
        with exclusive_lock(self.path):
            rows = self._read()
            if any(item.get("experience_id") == normalized.get("experience_id") for item in rows):
                return {"ok": False, "reason": "duplicate_experience_id", "experience_id": experience_id}
            merged_into = self._merge_target(rows, normalized)
            if merged_into:
                self.append_event(
                    merged_into,
                    MERGED_WITNESS_EVENT,
                    merged_into=merged_into,
                    merged_from=experience_id,
                    condition=normalized.get("condition") or {},
                    trigger=normalized.get("trigger") or {},
                    source_case=str((normalized.get("provenance") or {}).get("run") or ""),
                    round=(normalized.get("provenance") or {}).get("round"),
                    repair_gain=normalized.get("repair_gain"),
                    outcome=normalized.get("outcome") or {},
                )
                return {"ok": True, "reason": "merged_into_existing", "experience_id": merged_into, "merged_from": experience_id}
            self.root.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.path, normalized)
            write_json_atomic(self.manifest_path, {"schema_version": SCHEMA_VERSION, "records": len(rows) + 1})
        return {"ok": True, "reason": "appended", "experience_id": experience_id, "records": len(rows) + 1}

    def _merge_target(self, rows: list[dict[str, Any]], normalized: dict[str, Any]) -> str:
        """The id of the row this one restates, or ``""`` when it is a lesson of its own.

        A row that passed while carrying a new defect is never folded, in either direction: that
        record is the one thing the ladder has to be able to see about it, and folding it into an
        older row would hide it behind a row whose own ladder position says nothing about it.
        Neither is a row whose rung class differs from the row it would fold into, nor one the
        ladder has no position for at all: see ``MERGE_RUNG_CLASSES`` for why both halves matter.
        """

        if bool(normalized.get("carried_new_defect")):
            return ""
        rung = merge_rung_class(normalized.get("status"))
        if not rung:
            return ""
        key = merge_rule_key(normalized)
        if not key:
            return ""
        for row in rows:
            if bool(row.get("carried_new_defect")):
                continue
            if merge_rung_class(row.get("status")) != rung:
                continue
            if str(row.get("experience_id") or "") == str(normalized.get("experience_id") or ""):
                continue
            if merge_rule_key(row) == key:
                return str(row.get("experience_id") or "")
        return ""

    def merged_witnesses(self) -> dict[str, list[dict[str, Any]]]:
        """The conditions and evidence folded into each row by a restated lesson, by row id."""

        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in self.events():
            if str(event.get("event") or "") != MERGED_WITNESS_EVENT:
                continue
            grouped.setdefault(str(event.get("experience_id") or ""), []).append(dict(event))
        return grouped

    def records(self) -> list[dict[str, Any]]:
        """Return the readable Memory rows, for callers that only need to read them."""

        return self._read()

    def _archive_decisions(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        for row in rows:
            reason = untransferable_reason(row)
            decisions.append(
                {
                    "experience_id": str(row.get("experience_id") or ""),
                    "status": str(row.get("status") or ""),
                    "decision": "archived" if reason else "kept",
                    "reason": reason or "carries a lesson another paper can act on",
                }
            )
        return decisions

    def _archive_report(self, decisions: list[dict[str, Any]], *, destination: Path, dry_run: bool, records_after: int) -> dict[str, Any]:
        counts: dict[str, int] = {}
        archived = [decision for decision in decisions if decision["decision"] == "archived"]
        for decision in archived:
            counts[decision["reason"]] = counts.get(decision["reason"], 0) + 1
        return {
            "contract": ARCHIVE_REPORT_CONTRACT,
            "version": 1,
            "library": str(self.path),
            "archive_rows": str(destination / self.path.name),
            "dry_run": bool(dry_run),
            "declared_records": len(decisions),
            "archived": len(archived),
            "kept": len(decisions) - len(archived),
            # What the manifest says once the move is done, so a count that disagrees with the
            # report is visible rather than something a reader has to add up.
            "records_after": int(records_after),
            "counts_by_reason": counts,
            "rows": decisions,
        }

    def archive_untransferable(self, out_dir: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
        """Move the rows no later paper could act on into an archive, and keep the rest.

        The library is read as a set of lessons rather than as a log, so what decides is not how
        old a row is but whether another paper can be drawn by it. Everything is judged from the
        row itself, and the report names every row and why it went where it did. Nothing is
        deleted: the moved rows are appended to the archive's own library file, which may already
        hold an earlier archive, and the report lands beside them.
        """

        destination = Path(out_dir).resolve()
        rows = self.records()
        decisions = self._archive_decisions(rows)
        if dry_run:
            return self._archive_report(decisions, destination=destination, dry_run=True, records_after=len(rows))
        destination.mkdir(parents=True, exist_ok=True)
        archive_path = destination / self.path.name
        with exclusive_lock(self.path):
            # Re-read under the lock and decide again: a wave that appended while the rows were
            # being judged is judged here too, so a row that arrived meanwhile is not dropped by a
            # partition taken before it existed.
            rows = self.records()
            live = [row for row in rows if not untransferable_reason(row)]
            archived = [row for row in rows if untransferable_reason(row)]
            if archived:
                with exclusive_lock(archive_path):
                    for row in archived:
                        append_jsonl(archive_path, row)
            temporary = self.path.with_name(self.path.name + ".archive.tmp")
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in live), encoding="utf-8")
            os.replace(temporary, self.path)
            write_json_atomic(self.manifest_path, {"schema_version": SCHEMA_VERSION, "records": len(live)})
            report = self._archive_report(self._archive_decisions(rows), destination=destination, dry_run=False, records_after=len(live))
            write_json_atomic(destination / ARCHIVE_REPORT_FILE, report)
        return report

    def append(self, record: dict[str, Any]) -> bool:
        return bool(self.append_result(record)["ok"])

    def integrity_report(self) -> dict[str, Any]:
        """Report rows this store silently skips, so a thin library is explainable."""
        malformed: list[int] = []
        foreign_domain: list[dict[str, Any]] = []
        kept = 0
        if self.path.exists():
            for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    malformed.append(number)
                    continue
                if not isinstance(value, dict):
                    malformed.append(number)
                elif str(value.get("domain") or DIRECT_DOMAIN) != DIRECT_DOMAIN:
                    foreign_domain.append({"line": number, "domain": value.get("domain")})
                else:
                    kept += 1
        return {"path": str(self.path), "exists": self.path.exists(), "readable_records": kept, "malformed_lines": malformed, "foreign_domain_rows": foreign_domain}

    def append_event(self, experience_id: str, event: str, **payload: Any) -> dict[str, Any]:
        record = {"experience_id": str(experience_id), "event": str(event), **payload}
        self.root.mkdir(parents=True, exist_ok=True)
        return append_jsonl(self.events_path, record)

    def events(self, experience_id: str = "") -> list[dict[str, Any]]:
        """Return the lifecycle events, optionally only those of one row."""

        if not self.events_path.exists():
            return []
        want = str(experience_id or "")
        rows: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            if want and str(value.get("experience_id") or "") != want:
                continue
            rows.append(value)
        return rows

    def lifecycle_status(self, experience_id: str, default: str | None = None) -> str | None:
        status = default
        for row in self._read():
            if row.get("experience_id") == experience_id:
                status = str(row.get("status") or status)
        for event in self.events(experience_id):
            if event.get("status"):
                status = str(event["status"])
        return status

    def records_with_status(self) -> list[dict[str, Any]]:
        """Return every row with its effective status, so a lifecycle event is not ignored."""

        return [{**record, "status": self.lifecycle_status(str(record.get("experience_id") or ""), str(record.get("status") or ""))} for record in self._read()]

    def appended_exclusions(self) -> dict[str, list[dict[str, Any]]]:
        """Return the exclusion conditions events added, grouped by the row they belong to.

        A round that ran under an experience and did not survive the Gate appends a
        condition instead of rewriting the row: the append-only log stays append-only, and
        retrieval folds the conditions back in when it decides what is eligible here.
        """

        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in self.events():
            if str(event.get("event") or "") != EXCLUDED_CONDITION_EVENT:
                continue
            condition = event.get("condition")
            if isinstance(condition, dict):
                grouped.setdefault(str(event.get("experience_id") or ""), []).append(dict(condition))
        return grouped

    def effective_excluded_conditions(self, record: dict[str, Any], *, appended: dict[str, list[dict[str, Any]]] | None = None) -> list[dict[str, Any]]:
        """Return the conditions a row may not be reused under: its own and the appended ones."""

        grouped = appended if isinstance(appended, dict) else self.appended_exclusions()
        conditions = [dict(item) for item in (record.get("excluded_conditions") or []) if isinstance(item, dict)]
        conditions += grouped.get(str(record.get("experience_id") or ""), [])
        return conditions

    def revalidation_fingerprint(self, experience_id: str) -> dict[str, Any] | None:
        """Return the Judge a re-validation last re-measured a row under, or None if never."""

        found: dict[str, Any] | None = None
        for event in self.events(experience_id):
            if isinstance(event.get("judge_fingerprint"), dict):
                found = normalize_judge_fingerprint(event["judge_fingerprint"])
        return found

    def last_validation_report(self, experience_id: str) -> str:
        """Return the newest validation report an event points at, so a sweep can re-run it."""

        found = ""
        for event in self.events(experience_id):
            if str(event.get("validation_report") or ""):
                found = str(event["validation_report"])
        return found

    def get(self, experience_id: str) -> dict[str, Any] | None:
        for record in self._read():
            if str(record.get("experience_id") or "") == str(experience_id):
                return {**record, "status": self.lifecycle_status(experience_id, str(record.get("status") or "candidate"))}
        return None

    def transition(self, experience_id: str, status: str, **payload: Any) -> dict[str, Any]:
        return self.append_event(experience_id, "status_changed", status=status, **payload)

    def retrieve(self, limit: int = 5, *, domain: str = DIRECT_DOMAIN, context: dict[str, Any] | None = None, statuses: set[str] | None = None, apply_at: str = "", embedder: Any = None) -> list[dict[str, Any]]:
        """Return the Memory records worth injecting for this state."""

        return self.retrieve_with_report(limit=limit, domain=domain, context=context, statuses=statuses, apply_at=apply_at, embedder=embedder)["rows"]

    def retrieve_with_report(self, limit: int = 5, *, domain: str = DIRECT_DOMAIN, context: dict[str, Any] | None = None, statuses: set[str] | None = None, apply_at: str = "", embedder: Any = None) -> dict[str, Any]:
        """Rank records by stage, motif and structure, and report how they were ranked.

        A record is only a candidate when its lifecycle status is eligible and, when a
        stage is named, when it applies at that stage. The order then comes from motif
        overlap, size similarity, condition match and, when it can be produced, embedding
        similarity -- each row keeping the components behind its score.
        """

        if domain != DIRECT_DOMAIN:
            return {"rows": [], "report": _retrieval_report(mode="unsupported_domain", degraded=True, reason=f"the store only serves {DIRECT_DOMAIN}", candidates=0, apply_at=apply_at)}
        query = query_features(context)
        view = exclusion_view(query)
        appended = self.appended_exclusions()
        allowed = {str(item) for item in statuses} if statuses else set(DEFAULT_RETRIEVAL_STATUSES)
        stage = str(apply_at or "").strip()
        candidates = []
        excluded_rows: list[dict[str, Any]] = []
        for record in self._read():
            experience_id = str(record.get("experience_id") or "")
            effective_status = self.lifecycle_status(experience_id, str(record.get("status") or ""))
            if effective_status not in allowed:
                continue
            if stage and str(record.get("apply_at") or "") != stage:
                continue
            conditions = [dict(item) for item in (record.get("excluded_conditions") or []) if isinstance(item, dict)]
            conditions += appended.get(experience_id, [])
            reasons = [reason for reason in (exclusion_reason(view, condition) for condition in conditions) if reason]
            if reasons:
                excluded_rows.append({"experience_id": experience_id, "reasons": list(dict.fromkeys(reasons)), "excluded_conditions": conditions})
                continue
            if effective_status == "domain_validated" and not _domain_experience_matches(record, context):
                excluded_rows.append({"experience_id": experience_id, "reasons": ["domain applicability does not match this paper"], "excluded_conditions": conditions})
                continue
            candidates.append((record, effective_status))
        encoder, embedder_report = _retrieval_embedder(embedder)
        query_vector = _query_vector(query, encoder)
        scored = []
        for record, effective_status in candidates:
            vector = _record_vector(record, encoder)
            components = {
                "motif": motif_overlap(query, record),
                "structure": structure_similarity(query, record.get("condition") or record.get("context") or {}),
                "condition": condition_similarity(query, record.get("condition") or record.get("context") or {}),
                "embedding": embedding_similarity(query_vector, vector),
            }
            scored.append((_rank_score(components), components, effective_status, record))
        scored.sort(key=lambda item: (-item[0], -_number((item[3].get("outcome") or {}).get("after_score")), str(item[3].get("experience_id") or "")))
        rows = [
            {
                "strategy_id": record.get("strategy_id"),
                "domain": DIRECT_DOMAIN,
                "condition": record.get("condition", {}),
                "context": record.get("context", record.get("condition", {})),
                "strategy": record.get("strategy", {}),
                "action": record.get("action"),
                "apply_at": str(record.get("apply_at") or ""),
                "status": effective_status,
                "forbidden_changes": [str(item) for item in (record.get("forbidden_changes") or []) if str(item)],
                "evidence": {
                    "experience_id": record.get("experience_id"),
                    "match_score": round(total, 6),
                    "after_score": _number((record.get("outcome") or {}).get("after_score")),
                    "components": components,
                    "hit_reason": _hit_reason(components),
                    "support": len(record.get("validated_cases") or []),
                    "repair_gain": record.get("repair_gain"),
                    "provenance_source": str((record.get("provenance") or {}).get("source") or ""),
                },
            }
            for total, components, effective_status, record in scored[: max(0, int(limit))]
        ]
        mode = "structure_and_embedding" if query_vector and embedder_report["kind"] not in {"", "unavailable"} else "structure_only"
        reason = "" if mode == "structure_and_embedding" else embedder_report["reason"] or "no query or record text could be embedded"
        return {
            "rows": rows,
            "report": _retrieval_report(
                mode=mode,
                degraded=mode != "structure_and_embedding",
                reason=reason,
                candidates=len(candidates),
                apply_at=stage,
                statuses=sorted(allowed),
                embedder=embedder_report,
                embedded_rows=sum(1 for _, components, _, _ in scored if components["embedding"] is not None),
                excluded_rows=excluded_rows,
            ),
        }
