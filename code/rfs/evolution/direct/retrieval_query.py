"""Turn a pipeline state into the query the Memory store ranks against.

Kept apart from the store so the same normalisation can be reused by the Router and by
the tests, and so the store stays about persistence and ordering.
"""

from __future__ import annotations

from typing import Any, Sequence

from ...analysis.graph import cosine_similarity, histogram_similarity
from .experience import count_bucket


BUCKET_ORDER = ("low", "medium", "high")
CONDITION_KEYS = ("domain", "topology")
# The condition keys a query can compare, so an exclusion can be honoured or refused.
# Anything else an entry names (a motif, a metric) cannot be confirmed against a query
# and therefore never excludes a row on its own.
#: The condition keys an appended refusal may name. ``defect_patterns`` is included so a
#: refusal can say *which shape* of figure the move failed on, not only how big it was.
EXCLUSION_KEYS = ("domain", "topology", "node_count_bucket", "connector_count_bucket", "defect_patterns")
EXCLUDED_CONDITION_EVENT = "experience_condition_excluded"


def _number_of(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _count_of(context: dict[str, Any], count_key: str, collection_keys: Sequence[str]) -> int | None:
    direct = _number_of(context.get(count_key))
    if direct is not None:
        return max(0, direct)
    for key in collection_keys:
        value = context.get(key)
        if isinstance(value, (list, tuple)):
            return len(value)
    return None


def query_features(context: dict[str, Any] | None) -> dict[str, Any]:
    """Normalise one pipeline context into motifs, counts, conditions and query text."""

    source = context if isinstance(context, dict) else {}
    motifs = [str(item).strip() for item in (source.get("motifs") or source.get("required_motifs") or []) if str(item).strip()]
    action = source.get("action") if isinstance(source.get("action"), dict) else {}
    text = " ".join(
        part
        for part in (
            str(source.get("problem") or ""),
            str(source.get("topology") or ""),
            " ".join(motifs),
            str(action.get("type") or ""),
        )
        if part.strip()
    )
    return {
        "motifs": motifs,
        "node_count": _count_of(source, "node_count", ("nodes", "semantic_nodes")),
        "connector_count": _count_of(source, "connector_count", ("connectors", "edges", "arrows")),
        "condition": {key: source[key] for key in CONDITION_KEYS if str(source.get(key) or "").strip()},
        "text": text,
    }


def exclusion_view(query: dict[str, Any]) -> dict[str, str]:
    """Return the comparable condition a query offers an excluded row.

    The query already normalises sizes into buckets for structure similarity, so the same
    buckets are reused here: a row excluded for "medium nodes, low connectors" is refused
    for exactly the states that match, and for no others.
    """

    view: dict[str, str] = {}
    for key, value in (query.get("condition") or {}).items():
        if str(value or "").strip():
            view[str(key)] = str(value)
    for count_key, bucket_key in (("node_count", "node_count_bucket"), ("connector_count", "connector_count_bucket")):
        count = query.get(count_key)
        if count is not None and not view.get(bucket_key):
            view[bucket_key] = count_bucket(int(count))
    return {key: value for key, value in view.items() if key in EXCLUSION_KEYS}


def _exclusion_pairs(condition: dict[str, Any]) -> dict[str, str]:
    """Read one exclusion entry, whichever of its two shapes it uses.

    A flat entry names its conditions directly; a ``{dimension, value}`` entry names one of
    them in its own field, so both are folded into the same comparable pairs.
    """

    pairs: dict[str, str] = {}
    dimension, value = str(condition.get("dimension") or ""), str(condition.get("value") or "")
    if dimension and value:
        pairs[dimension] = value
    for key in EXCLUSION_KEYS:
        text = str(condition.get(key) or "")
        if text:
            pairs[key] = text
    return pairs


def exclusion_reason(view: dict[str, str], condition: dict[str, Any]) -> str:
    """Return why a row is excluded from this query, or "" when it is not excluded.

    A row is excluded only when every key the entry names and the query can compare is
    equal, and at least one such key exists. An entry that names nothing this query knows,
    or names a condition the query cannot confirm, does not exclude anything.
    """

    if not isinstance(condition, dict):
        return ""
    comparable = {key: value for key, value in _exclusion_pairs(condition).items() if key in view and view[key]}
    if not comparable:
        return ""
    if all(view[key] == value for key, value in comparable.items()):
        return str(condition.get("reason") or "excluded_condition")
    return ""


def _bucket_distance(left: str, right: str) -> float:
    if left not in BUCKET_ORDER or right not in BUCKET_ORDER:
        return 1.0 if left == right else 0.0
    distance = abs(BUCKET_ORDER.index(left) - BUCKET_ORDER.index(right))
    return 1.0 if distance == 0 else 0.5 if distance == 1 else 0.0


def structure_similarity(query: dict[str, Any], condition: dict[str, Any]) -> float | None:
    """Compare the query's object counts with the buckets a record was measured under."""

    pairs: list[float] = []
    for count_key, bucket_key in (("node_count", "node_count_bucket"), ("connector_count", "connector_count_bucket")):
        count, bucket = query.get(count_key), str((condition or {}).get(bucket_key) or "")
        if count is None or not bucket:
            continue
        pairs.append(_bucket_distance(count_bucket(int(count)), bucket))
    return sum(pairs) / len(pairs) if pairs else None


def condition_similarity(query: dict[str, Any], condition: dict[str, Any]) -> float | None:
    """Return how much of the query's condition a record was measured under matches."""

    wanted = query.get("condition") or {}
    if not wanted:
        return None
    matches = sum(1 for key, value in wanted.items() if str((condition or {}).get(key) or "") == str(value))
    return matches / len(wanted)


def motif_overlap(query: dict[str, Any], record: dict[str, Any]) -> float | None:
    """Return the motif overlap of a query and a record, or None when the query is silent."""

    motifs = list(query.get("motifs") or [])
    if not motifs:
        return None
    return histogram_similarity(motifs, [str(item) for item in record.get("required_motifs") or []])


def record_text(record: dict[str, Any]) -> str:
    """Return the deterministic text a Memory record is embedded from."""

    action = record.get("action") if isinstance(record.get("action"), dict) else {}
    return " ".join(
        part
        for part in (
            str(record.get("hypothesis") or ""),
            str(record.get("problem_type") or ""),
            str(action.get("type") or action.get("action") or ""),
        )
        if part.strip()
    )


def embedding_similarity(query_vector: list[float], record_vector: list[float]) -> float | None:
    """Return a non-negative cosine, or None when either side has no vector."""

    if not query_vector or not record_vector:
        return None
    return max(0.0, cosine_similarity(query_vector, record_vector))
