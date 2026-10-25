"""Split a Memory row into the rule that travels and the binding that stays behind.

An experience earns its name by travelling: it is a claim about how figures are drawn, and it is
worth keeping only when it can change a figure it has never seen. A row whose rule says
``rel_12`` has not made a claim -- it has described one drawing -- and the next paper has no
``rel_12``, so nothing can ever be retrieved for it.

A rule is read in three parts, and only one of them is written by a model. ``condition`` says
when the rule holds, and the pipeline computes it from the figure: a condition written by the
same model that proposed the repair would be free to describe whatever made that repair look
right. ``action`` is the move, taken from the closed vocabulary the executor can actually run.
``guidance`` is the part a model does write -- how to apply that move, and when not to -- and it
is the only prose the library keeps.

Everything that names the figure it was measured on goes to the ``binding``: the object ids, the
boxes, the sentence about this figure's own defect, and this figure's own labels. The library
keeps the rule; the binding stays in the run directory as the evidence of what the measurement
was about.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from ...analysis.graph.retrieval import SCOPE_ID_KEYS


#: The parts of a row a rule may carry: what it claims, when it holds, and the move it makes.
#: ``guidance`` is the prose half and the only one a model writes. ``hypothesis`` is rendered
#: from all three by :func:`render_rule_statement`, so the row always carries one sentence that
#: says what to do instead of a sentence about how this round scored.
RULE_KEYS = (
    "memory_type",
    "apply_at",
    "scope",
    "applicability",
    "applicability_domain",
    "hypothesis",
    "guidance",
    "condition",
    "excluded_conditions",
    "required_motifs",
    "expected_effect",
)

#: Prose that describes one drawing rather than a way of drawing. It stays in the binding: a
#: sentence about "the New Query box" is evidence about a measurement, not a claim another paper
#: can act on, and no other paper has a box called that.
BINDING_PROSE_KEYS = ("problem_type",)

#: Action parameters that name the figure the action was measured on rather than the move. The
#: defect sentence and the effect the repair was asked for are two more of them: both are about
#: this figure's own boxes, and both used to travel with the rule and be handed to the next
#: paper's prompt as if they described that paper's figure.
BINDING_ACTION_KEYS = (*SCOPE_ID_KEYS, "graph_ids", "subgraph_ids", "problem", "expected_effect", "evidence")

#: The moves that change how figures are drawn rather than this figure's geometry.
MECHANISM_ACTIONS = ("router_constraint", "layout_constraint")

#: An object id the pipeline mints per figure: ``rel_12``, ``node_alpha``, ``slot_3`` and kin.
#: The lookahead keeps the coarse statistics -- ``node_count_bucket`` and friends -- out of it,
#: because a bucket is a shape a rule is allowed to talk about and an id is not.
CASE_OBJECT_ID = re.compile(r"\b(?:rel|conn|edge|node|slot|panel|icon|asset|subgraph|sg)_(?!count\b)[A-Za-z0-9]+\b", re.IGNORECASE)

#: A coordinate, which is the same binding written as a number instead of a name.
COORDINATE = re.compile(r"\b[xywh]\s*[:=]\s*-?\d|\b\d?\.\d{1,3}\s*[,/]\s*\d?\.\d{1,3}\b")

#: Words a figure label may share with ordinary prose. A label that is one of these cannot be
#: read as "the rule named this figure's box", because the same word is how a rule would talk
#: about the thing in general. A label of two or more words is never on this list: no rule has a
#: reason to write out a panel's full title unless it is talking about that panel.
GENERIC_LABEL_WORDS = frozenset({
    "agent", "answer", "arrow", "axis", "batch", "block", "box", "cache", "caption", "chart",
    "context", "data", "database", "decoder", "document", "edge", "embedding", "encoder",
    "feature", "figure", "filter", "frame", "gate", "graph", "group", "head", "history", "icon",
    "image", "index", "information", "input", "label", "layer", "legend", "line", "loss",
    "matrix", "memory", "method", "metric", "model", "module", "network", "node", "output",
    "panel", "paper", "plot", "policy", "pool", "process", "prompt", "query", "question",
    "reference", "response", "result", "results", "retrieval", "reward", "sample", "score",
    "search", "section", "source", "stage", "state", "step", "system", "table", "target",
    "task", "text", "title", "token", "training", "tree", "user", "value", "vector", "weights",
})

#: Action parameters that are part of a rule's identity or its binding, never a magnitude the
#: rendered sentence should spell out.
_ACTION_PARAMETER_SKIP = frozenset({
    "action", "type", "scope_type", "applicability", "problem", "expected_effect",
    "forbidden_changes", "experience_id", "source", "transferable_advice", "reason", "note",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def figure_labels(figure: dict[str, Any] | None) -> list[str]:
    """Every name the figure gives to one of its own parts.

    This is what makes "the rule names this figure" checkable instead of a guess. The labels are
    read out of the plan the pipeline already wrote, so the check never has to recognise a noun:
    it compares the rule's words against the only words that are certainly this figure's.
    """

    source = figure if isinstance(figure, dict) else {}
    names: list[str] = []
    for key in ("nodes", "semantic_nodes", "cards", "slots", "panels", "groups", "labels", "arrows", "connectors"):
        items = source.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str):
                names.append(item)
                continue
            if not isinstance(item, dict):
                continue
            for field in ("label", "title", "name", "text"):
                value = str(item.get(field) or "").strip()
                if value:
                    names.append(value)
                    break
    return sorted(dict.fromkeys(name for name in names if name))


def _label_hits(text: str, labels: Iterable[str]) -> list[str]:
    """Which of the figure's own labels the text names."""

    haystack = " ".join(str(text or "").casefold().split())
    if not haystack:
        return []
    hits: list[str] = []
    for label in labels:
        needle = " ".join(str(label or "").casefold().split())
        if len(needle) < 3:
            continue
        if " " in needle:
            if needle in haystack:
                hits.append(str(label))
            continue
        if needle in GENERIC_LABEL_WORDS:
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", haystack):
            hits.append(str(label))
    return hits


def _rule_prose(rule: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Every free-text field the rule carries, with the name of the field it came from."""

    source = rule if isinstance(rule, dict) else {}
    fields: list[tuple[str, str]] = []
    for key in ("hypothesis", "guidance"):
        text = str(source.get(key) or "").strip()
        if text:
            fields.append((key, text))
    action = source.get("action")
    if isinstance(action, dict):
        for key in ("problem", "expected_effect", "transferable_advice", "reason", "note"):
            text = str(action.get(key) or "").strip()
            if text:
                fields.append((f"action.{key}", text))
    return fields


def split_rule_binding(record: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one row as a rule that travels and a binding that does not.

    The action is split as well: ``slot_ids`` names the figure the move was measured on and
    belongs to the binding, while a mechanism parameter such as a router constraint belongs to
    the rule, because that is the part the next paper can act on. So does the sentence about the
    defect, which is why the split carries prose keys on both sides rather than text only in the
    rule.
    """

    source = record if isinstance(record, dict) else {}
    rule = {key: source[key] for key in RULE_KEYS if key in source}
    binding: dict[str, Any] = {key: source[key] for key in SCOPE_ID_KEYS if key in source}
    for key in ("graph_scope", "binding", "source_case", "experience_id", *BINDING_PROSE_KEYS):
        if key in source:
            binding[key] = source[key]
    action = source.get("action")
    if isinstance(action, dict):
        rule["action"] = {key: value for key, value in action.items() if key not in BINDING_ACTION_KEYS}
        named = {key: action[key] for key in BINDING_ACTION_KEYS if key in action}
        if named:
            binding["action"] = named
    return rule, binding


def _condition_phrase(condition: Any) -> str:
    """When the rule holds, in the words the structure itself uses."""

    source = condition if isinstance(condition, dict) else {}
    parts: list[str] = []
    patterns = [str(item) for item in (source.get("defect_patterns") or []) if str(item)]
    if patterns:
        parts.append("the figure shows " + ", ".join(patterns))
    for key, phrase in (("node_count_bucket", "node count"), ("connector_count_bucket", "connector count")):
        value = str(source.get(key) or "").strip()
        if value and value != "unknown":
            parts.append("%s %s" % (phrase, value))
    return " and ".join(parts) if parts else "this condition holds"


def _action_phrase(action: Any) -> str:
    """What the rule does, with the magnitudes the executor needs and nothing else."""

    source = action if isinstance(action, dict) else {}
    name = str(source.get("type") or source.get("action") or "").strip()
    if not name:
        return "no bounded move"
    parameters: list[str] = []
    for key in sorted(source):
        if key in _ACTION_PARAMETER_SKIP or key in BINDING_ACTION_KEYS:
            continue
        value = source[key]
        if isinstance(value, bool):
            parameters.append("%s=%s" % (key, str(value).lower()))
        elif isinstance(value, (int, float)):
            parameters.append("%s=%s" % (key, value))
        elif isinstance(value, str) and value.strip():
            parameters.append("%s=%s" % (key, value.strip()))
        elif isinstance(value, dict) and value:
            parameters.append("%s=%s" % (key, _canonical(value)))
    return name + (" (" + ", ".join(parameters) + ")" if parameters else "")


def render_rule_statement(*, condition: Any, action: Any, guidance: str = "") -> str:
    """One sentence a later paper can act on: when it holds, what it does, and how to do it.

    The frame is rendered from the condition and the move, so the sentence always names a stage
    the executor can run and a shape it can recognise. The guidance is the model's own prose and
    is appended rather than trusted with the frame: the model decides how to say it, the
    pipeline decides what a rule is allowed to claim.
    """

    sentence = "When %s: %s." % (_condition_phrase(condition), _action_phrase(action))
    extra = " ".join(str(guidance or "").split())
    return (sentence + " " + extra).strip()


def rule_statement(rule: dict[str, Any] | None) -> str:
    """The rule as one piece of text, which is what the gate reads."""

    source = rule if isinstance(rule, dict) else {}
    parts = [str(source.get("hypothesis") or ""), str(source.get("guidance") or "")]
    for key in ("condition", "action"):
        if isinstance(source.get(key), dict):
            parts.append(_canonical(source[key]))
    return " ".join(part for part in parts if part)


def case_object_evidence(rule: dict[str, Any] | None, *, figure: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Name every way the rule has bound itself to the figure it came from.

    Three ways, and the third is the one that used to slip through: an object id, a coordinate,
    and a name the figure itself gave to one of its parts. The old check could only see the first
    two, so a rule that said "the New Query box is too small" passed -- the box is a real node of
    a real figure, and no pattern for ``node_`` was ever going to match a human name.
    """

    source = rule if isinstance(rule, dict) else {}
    evidence: list[dict[str, Any]] = []
    labels = figure_labels(figure)
    for field, text in _rule_prose(source):
        match = CASE_OBJECT_ID.search(text)
        if match:
            evidence.append({"kind": "case_object_id", "field": field, "value": match.group(0)})
        match = COORDINATE.search(text)
        if match:
            evidence.append({"kind": "coordinate", "field": field, "value": match.group(0)})
        hits = _label_hits(text, labels)
        if hits:
            evidence.append({"kind": "case_figure_label", "field": field, "value": ", ".join(hits)})
    condition = source.get("condition")
    if isinstance(condition, dict):
        named = sorted(key for key in condition if key in SCOPE_ID_KEYS)
        if named:
            evidence.append({"kind": "case_object_id", "field": "condition", "value": ", ".join(named)})
    return evidence


def rule_binding_gate(
    record: dict[str, Any] | None, *, default_action_fingerprint: str = "", figure: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Decide whether a row may enter the library, and which part of it would.

    Three refusals come out of here. A rule that names the figure it came from -- by object id,
    by coordinate, or by one of that figure's own labels -- is refused as
    ``rule_binds_to_case_objects``: it cannot be retrieved for another paper, so keeping it would
    only make the library look larger than what it can act on. A rule that restates the move the
    run would have made anyway is refused as ``common_knowledge``: it is not a finding, and a
    library that files its own defaults as discoveries is why a pair whose two arms did the same
    thing read as a failed experience instead of as a wasted measurement. And a round that wrote
    no guidance and named no mechanism-level move has nothing in it that travels, so it is
    refused as ``no_transferable_rule`` rather than filed as one.
    """

    rule, binding = split_rule_binding(record)
    evidence = case_object_evidence(rule, figure=figure)
    if evidence:
        return {"ok": False, "reason": "rule_binds_to_case_objects", "rule": rule, "binding": binding, "evidence": evidence}
    fingerprint = str(default_action_fingerprint or (record or {}).get("default_action_fingerprint") or "")
    if fingerprint and _same_move(rule.get("action"), fingerprint):
        return {"ok": False, "reason": "common_knowledge", "rule": rule, "binding": binding, "evidence": [{"kind": "action_fingerprint", "field": "action", "value": fingerprint}]}
    action_type = str((rule.get("action") or {}).get("type") or "")
    if not str(rule.get("guidance") or "").strip() and action_type not in MECHANISM_ACTIONS:
        return {
            "ok": False,
            "reason": "no_transferable_rule",
            "rule": rule,
            "binding": binding,
            "evidence": [{"kind": "no_transferable_rule", "field": "guidance", "value": "the round wrote no guidance and named no mechanism-level move"}],
        }
    return {"ok": True, "reason": "", "rule": rule, "binding": binding, "evidence": []}


def _same_move(action: Any, fingerprint: str) -> bool:
    """Whether the rule's move is the one the fingerprint names."""

    from .controller import action_execution_fingerprint

    return action_execution_fingerprint(action if isinstance(action, dict) else {}) == fingerprint


__all__ = [
    "BINDING_ACTION_KEYS",
    "BINDING_PROSE_KEYS",
    "CASE_OBJECT_ID",
    "COORDINATE",
    "GENERIC_LABEL_WORDS",
    "MECHANISM_ACTIONS",
    "RULE_KEYS",
    "case_object_evidence",
    "figure_labels",
    "render_rule_statement",
    "rule_binding_gate",
    "rule_statement",
    "split_rule_binding",
]
