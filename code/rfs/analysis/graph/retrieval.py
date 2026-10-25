"""Select structurally relevant but content-diverse graph validation cases.

Two layers are kept apart. :func:`vector_recall` is the coarse, cheap pass: it keeps
the ``limit`` most similar records by stored embedding. :func:`rank_validation_cases`
is the structural pass: motif histograms, size, content and domain diversity decide the
final order, with the embedding similarity as one small term. The weighted score itself
is unchanged, so a recalled subset is ranked exactly like the full candidate pool.

Recall fails loudly when an embedding is missing, because a silent fall back to
structure-only ranking would hide an index that was never embedded.

A subgraph is recalled and reranked by its own pair of functions,
:func:`vector_recall_subgraphs` and :func:`rank_subgraph_candidates`, which score the
locator rows of ``subgraph_index.jsonl`` with the same arithmetic and the same loud
failure when a vector is missing.

The similarity measures themselves -- :func:`cosine_similarity`,
:func:`histogram_similarity`, :func:`jaccard_similarity` and :func:`motif_similarity` --
are public, because the Memory store ranks its own records with the same arithmetic
instead of growing a second, subtly different definition of "similar".
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

from .defect_patterns import defect_patterns, missing_patterns


def _cosine(left: list[float], right: list[float]) -> float:
    return cosine_similarity(left, right)


def _jaccard(left: set[str], right: set[str]) -> float:
    return jaccard_similarity(left, right)


def _motif_similarity(query: dict[str, Any], candidate: dict[str, Any]) -> float:
    return motif_similarity(query, candidate)


def cosine_similarity(left: Iterable[float] | None, right: Iterable[float] | None) -> float:
    """Return the cosine of two dense vectors; unequal or empty vectors score 0."""

    first = [float(item) for item in left] if isinstance(left, (list, tuple)) else []
    second = [float(item) for item in right] if isinstance(right, (list, tuple)) else []
    if not first or len(first) != len(second):
        return 0.0
    denominator = math.sqrt(sum(item * item for item in first)) * math.sqrt(sum(item * item for item in second))
    return sum(a * b for a, b in zip(first, second)) / denominator if denominator else 0.0


def _counts(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(key): float(item) for key, item in value.items() if _is_number(item)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item): 1.0 for item in value if str(item).strip()}
    return {}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def histogram_similarity(left: Any, right: Any) -> float:
    """Return the overlap of two count maps or name collections, in ``[0, 1]``.

    ``sum(min) / sum(max)`` reduces to the Jaccard index for two sets, so a motif
    histogram and a list of motif names are compared by the same rule.
    """

    first, second = _counts(left), _counts(right)
    keys = set(first) | set(second)
    denominator = sum(max(first.get(key, 0.0), second.get(key, 0.0)) for key in keys)
    overlap = sum(min(first.get(key, 0.0), second.get(key, 0.0)) for key in keys)
    return overlap / denominator if denominator else 1.0


def jaccard_similarity(left: Any, right: Any) -> float:
    """Return the Jaccard index of two name collections, treating them as counts."""

    return histogram_similarity(left, right)


def motif_similarity(query: dict[str, Any], candidate: dict[str, Any]) -> float:
    """Return the motif-histogram overlap of two graph index records."""

    q = (query.get("features") or {}).get("motif_histogram") or {}
    c = (candidate.get("features") or {}).get("motif_histogram") or {}
    return histogram_similarity(q, c)


RECALL_LIMIT = 32


def _embedding_of(record: dict[str, Any]) -> list[float]:
    value = record.get("embedding") if isinstance(record, dict) else None
    return [float(item) for item in value] if isinstance(value, list) else []


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("paper_id") or ""), str(record.get("record_id") or "")


def vector_recall(
    query: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    *,
    limit: int,
    exclude_paper_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Return the ``limit`` most similar records, ready for the structural rerank."""

    excluded = {str(item) for item in exclude_paper_ids} | {str(query.get("paper_id") or "")}
    pool = [candidate for candidate in candidates if str(candidate.get("paper_id") or "") not in excluded]
    embedded = [_embedding_of(candidate) for candidate in pool]
    missing = [str(key[0] or key[1] or "?") for key, vector in zip((_record_key(item) for item in pool), embedded) if not vector]
    if not _embedding_of(query):
        missing.append(str(query.get("paper_id") or query.get("record_id") or "query"))
    if missing:
        raise ValueError("vector recall needs a stored embedding for every record; missing for: " + ", ".join(sorted(missing)[:10]))
    query_vector = _embedding_of(query)
    ranked = sorted(pool, key=lambda item: (-_cosine(query_vector, _embedding_of(item)), _record_key(item)))
    return ranked[:max(0, int(limit))]


def rank_validation_cases_with_recall(
    query: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    *,
    top_k: int = 5,
    recall_limit: int = RECALL_LIMIT,
    exclude_paper_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Recall by embedding, then rerank with the unchanged structural score."""

    recalled = vector_recall(query, candidates, limit=max(int(recall_limit), int(top_k)), exclude_paper_ids=exclude_paper_ids)
    return rank_validation_cases(query, recalled, top_k=top_k, exclude_paper_ids=exclude_paper_ids)


def _locator_key(locator: dict[str, Any]) -> tuple[str, str]:
    """Identify one subgraph locator; a subgraph is unique inside its paper."""

    return str(locator.get("paper_id") or ""), str(locator.get("subgraph_id") or "")


def _structure_of(locator: dict[str, Any]) -> tuple[str, int, int]:
    """Return the motif, node count and edge count of one locator.

    The counts live on the locator row itself, so a signature block is read only when a
    caller supplied one; nothing here depends on subgraph payloads being materialized.
    """

    signature = locator.get("signature")
    source = signature if isinstance(signature, dict) and signature else locator
    return (
        str(source.get("motif") or "unknown"),
        int(source.get("node_count") or 0),
        int(source.get("edge_count") or 0),
    )


def _same_layer(query: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return str(query.get("graph_kind") or "") == str(candidate.get("graph_kind") or "")


def _locator_pool(
    query: dict[str, Any],
    locators: Iterable[dict[str, Any]],
    *,
    exclude_paper_ids: Iterable[str],
    same_layer_only: bool,
) -> list[dict[str, Any]]:
    """Return the candidate locators one subgraph query may be matched against."""

    excluded = {str(item) for item in exclude_paper_ids} | {str(query.get("paper_id") or "")}
    query_key = _locator_key(query)
    pool = [
        item for item in locators
        if str(item.get("paper_id") or "") not in excluded and _locator_key(item) != query_key
    ]
    if same_layer_only:
        pool = [item for item in pool if _same_layer(query, item)]
    return pool


def _require_embeddings(query: dict[str, Any], pool: Iterable[dict[str, Any]]) -> None:
    """Fail loudly when a locator has no stored vector."""

    missing = [str(key[1] or key[0] or "?") for key, vector in zip((_locator_key(item) for item in pool), (_embedding_of(item) for item in pool)) if not vector]
    if not _embedding_of(query):
        missing.append(str(query.get("subgraph_id") or query.get("paper_id") or "query"))
    if missing:
        raise ValueError("subgraph recall needs a stored embedding for every locator; missing for: " + ", ".join(sorted(missing)[:10]))


def vector_recall_subgraphs(
    query: dict[str, Any],
    locators: Iterable[dict[str, Any]],
    *,
    limit: int,
    exclude_paper_ids: Iterable[str] = (),
    same_layer_only: bool = True,
) -> list[dict[str, Any]]:
    """Return the ``limit`` most similar subgraph locators, ready for the rerank.

    The candidate pool may be narrowed to the query's own layer, because a scientific
    stage and a visual block of the same shape are not interchangeable evidence.
    """

    pool = _locator_pool(query, locators, exclude_paper_ids=exclude_paper_ids, same_layer_only=same_layer_only)
    _require_embeddings(query, pool)
    query_vector = _embedding_of(query)
    ranked = sorted(pool, key=lambda item: (-_cosine(query_vector, _embedding_of(item)), _locator_key(item)))
    return ranked[:max(0, int(limit))]


def rank_subgraph_candidates(
    query: dict[str, Any],
    locators: Iterable[dict[str, Any]],
    *,
    top_k: int = 5,
    recall_limit: int = RECALL_LIMIT,
    exclude_paper_ids: Iterable[str] = (),
    same_layer_only: bool = True,
) -> list[dict[str, Any]]:
    """Recall subgraphs by vector, then rerank them by structure and layer.

    The weights mirror the record level ranking: structure dominates, the layer is a
    tie breaker and the embedding similarity stays a small term, because "the same
    structure" is a property of the graph and not of the wording.
    """

    recalled = vector_recall_subgraphs(
        query,
        locators,
        limit=max(int(recall_limit), int(top_k)),
        exclude_paper_ids=exclude_paper_ids,
        same_layer_only=same_layer_only,
    )
    query_motif, query_nodes, query_edges = _structure_of(query)
    ranked = []
    for candidate in recalled:
        other_motif, other_nodes, other_edges = _structure_of(candidate)
        motif_match = 1.0 if query_motif == other_motif and query_motif != "unknown" else 0.0
        size_similarity = 1.0 - min(1.0, (abs(query_nodes - other_nodes) + abs(query_edges - other_edges)) / max(1, query_nodes + query_edges, other_nodes + other_edges))
        layer_match = 1.0 if _same_layer(query, candidate) else 0.0
        embedding_similarity = _cosine(_embedding_of(query), _embedding_of(candidate))
        structure_score = 0.75 * motif_match + 0.25 * size_similarity
        score = 0.68 * structure_score + 0.22 * layer_match + 0.10 * max(0.0, embedding_similarity)
        ranked.append({
            "locator": candidate,
            "score": round(score, 6),
            "structure_similarity": round(structure_score, 6),
            "motif_match": round(motif_match, 6),
            "size_similarity": round(size_similarity, 6),
            "layer_match": round(layer_match, 6),
            "embedding_similarity": round(embedding_similarity, 6),
            "same_motif": bool(motif_match),
            "same_layer": bool(layer_match),
            "selection_reason": "similar_subgraph_structure" if motif_match else "vector_neighbour_with_another_structure",
        })
    ranked.sort(key=lambda item: (-item["score"], str(item["locator"].get("subgraph_id") or "")))
    return ranked[:max(0, int(top_k))]


def rank_validation_cases(
    query: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    *,
    top_k: int = 5,
    exclude_paper_ids: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Rank same-logic cases while rewarding cross-domain/content diversity."""

    excluded = {str(item) for item in exclude_paper_ids} | {str(query.get("paper_id") or "")}
    query_tags = set(query.get("content_tags") or [])
    query_domain = str(query.get("domain") or "unknown")
    ranked = []
    for candidate in candidates:
        if str(candidate.get("paper_id") or "") in excluded:
            continue
        motif = _motif_similarity(query, candidate)
        q_nodes = int((query.get("features") or {}).get("node_count") or 0)
        c_nodes = int((candidate.get("features") or {}).get("node_count") or 0)
        q_edges = int((query.get("features") or {}).get("edge_count") or 0)
        c_edges = int((candidate.get("features") or {}).get("edge_count") or 0)
        size_similarity = 1.0 - min(1.0, (abs(q_nodes - c_nodes) + abs(q_edges - c_edges)) / max(1, q_nodes + q_edges, c_nodes + c_edges))
        tag_similarity = _jaccard(query_tags, set(candidate.get("content_tags") or []))
        content_diversity = 1.0 - tag_similarity
        domain_diversity = 1.0 if str(candidate.get("domain") or "unknown") != query_domain else 0.0
        embedding_similarity = _cosine(query.get("embedding") or [], candidate.get("embedding") or [])
        structure_score = 0.75 * motif + 0.25 * size_similarity
        score = 0.68 * structure_score + 0.14 * content_diversity + 0.10 * domain_diversity + 0.08 * max(0.0, embedding_similarity)
        ranked.append({
            "record": candidate,
            "score": round(score, 6),
            "structure_similarity": round(structure_score, 6),
            "motif_similarity": round(motif, 6),
            "content_diversity": round(content_diversity, 6),
            "cross_domain": bool(domain_diversity),
            "embedding_similarity": round(embedding_similarity, 6),
            "selection_reason": "similar_subgraph_logic_with_domain_or_content_variation",
        })
    ranked.sort(key=lambda item: (-item["score"], str(item["record"].get("paper_id") or "")))
    return ranked[:max(0, int(top_k))]


#: The four groups a case set has to cover (docx §7.5): the same domain with the same structure,
#: another domain with the same structure, the same domain with another structure, and a
#: boundary counter-example the experience is not supposed to apply to.
CASE_GROUPS = (
    "same_domain_same_structure",
    "cross_domain_same_structure",
    "same_domain_cross_structure",
    "boundary",
)

#: A domain experience is not asked to travel. The cross-domain isomorphic group stays out.
DOMAIN_CASE_GROUPS = (
    "same_domain_same_structure",
    "same_domain_cross_structure",
    "boundary",
)

#: How many cases each group is asked for when the suite does not say.
DEFAULT_CASE_QUOTA = {
    "same_domain_same_structure": 2,
    "cross_domain_same_structure": 1,
    "same_domain_cross_structure": 1,
    "boundary": 1,
}

#: How many cases a rule-class row is asked for. Such a row is measured by compiling the same
#: figure twice instead of regenerating it in two arms, which costs minutes rather than hours, so
#: the same rule can be asked to travel across more papers before it is kept. Only rows whose
#: action is a mechanism action take this quota; every other row keeps the suite's own.
COMPILE_ONLY_CASE_QUOTA = {
    "same_domain_same_structure": 3,
    "cross_domain_same_structure": 2,
    "same_domain_cross_structure": 2,
    "boundary": 2,
}

#: The name of the profile that runs the whole four-group set. It is the default everywhere, so a
#: caller that names no profile keeps the ladder exactly as it was.
FULL_PROFILE = "full"

#: The name of the profile that runs one matched paper instead of the set. A caller names it when
#: it has decided that a row which already won on the paper it came from only needs to be
#: reproduced once on a paper whose shape matches before the run may draw from it. What it gives
#: up is named here rather than hidden: with no boundary counter-example nothing measures a figure
#: the row was never meant to touch, and with no second same-domain case reproduction on the
#: source figure is not re-shown. A row kept this way is written to a rung of its own, so the bet
#: stays readable off the row instead of living only in whoever configured the run.
QUICK_PROFILE = "quick"
QUICK_CASE_QUOTA = {
    "same_domain_same_structure": 0,
    "cross_domain_same_structure": 1,
    "same_domain_cross_structure": 0,
    "boundary": 0,
}
#: The same single case for a row that claims no travel: it cannot be asked across domains, so the
#: one matched paper it is asked about is another shape inside its own domain.
DOMAIN_QUICK_CASE_QUOTA = {
    "same_domain_same_structure": 0,
    "cross_domain_same_structure": 0,
    "same_domain_cross_structure": 1,
    "boundary": 0,
}


def quick_case_quota(applicability: Any = "common") -> dict[str, int]:
    """The one matched case the ``quick`` profile asks for, chosen for the row's applicability."""

    kind = str(applicability or "").strip().casefold()
    return dict(DOMAIN_QUICK_CASE_QUOTA if kind == "domain" else QUICK_CASE_QUOTA)


def required_case_groups(applicability: str, *, profile: str = FULL_PROFILE) -> tuple[str, ...]:
    """The groups an exam has to be able to fill before it is worth running.

    The ``quick`` profile requires none, because one matched paper is the whole of the evidence
    the caller asked for. Every other profile keeps the contract: a row that claims to travel has
    to face another domain, and every exam needs a case the row was never meant to apply to --
    without a boundary, "this repairs the figure" and "this is applied wherever it is seen" cannot
    be told apart, and the ladder can never be climbed.
    """

    if str(profile or FULL_PROFILE).strip().casefold() == QUICK_PROFILE:
        return ()
    return tuple(REQUIRED_CASE_GROUPS.get(str(applicability or "").strip().casefold(), ()))


#: The ladder rung a group feeds. A boundary case is its own rung: it only has to stay safe.
SPLIT_FOR_GROUP = {
    "same_domain_same_structure": "same_domain",
    "same_domain_cross_structure": "same_domain",
    "cross_domain_same_structure": "holdout",
    "boundary": "boundary",
}

#: The record keys a Memory row may name when it says which contexts it does not apply to.
CONDITION_KEYS = ("domain", "topology", "node_count_bucket", "connector_count_bucket", "defect_patterns")

#: The keys an experience uses to say which objects it is scoped to.
SCOPE_ID_KEYS = ("object_ids", "node_ids", "semantic_node_ids", "slot_ids", "connector_ids", "edge_ids")

#: A candidate counts as the same structure when its motif histogram overlaps the query's by at
#: least this much. The value is reported with every selection, so the choice is auditable.
SAME_STRUCTURE_THRESHOLD = 0.5


def domain_is_missing(value: Any) -> bool:
    """Whether a paper has no domain label that a same-domain group can use.

    ``general`` is a label a caller wrote on purpose. An empty value, ``unknown``
    and ``missing`` are not the same domain as each other.
    """

    return str(value or "").strip().casefold() in {"", "unknown", "missing"}


def split_for_group(group: str) -> str:
    """Return the ladder rung a group belongs to, or nothing when the group is unknown."""

    return SPLIT_FOR_GROUP.get(str(group or ""), "")


def count_bucket(count: int) -> str:
    """Bucket an object count the way a Memory condition states it.

    The names and thresholds are the ones the evolve station writes into a condition: ``low``
    under five, ``medium`` under nine, ``high`` from nine. They used to differ -- ``small`` and
    ``large`` here, ``low`` and ``high`` there -- and a condition that names a count could
    therefore never match the record it was written about, which silently made every such
    condition unenforceable.
    """

    return "high" if count >= 9 else "medium" if count >= 5 else "low"


def _bucket(count: int) -> str:
    return count_bucket(count)


def _identity_of(record: dict[str, Any]) -> dict[str, str]:
    identity = record.get("paper_identity") if isinstance(record.get("paper_identity"), dict) else {}
    return {
        "paper_id": str(record.get("paper_id") or identity.get("paper_id") or ""),
        "doi": str(record.get("doi") or identity.get("doi") or ""),
        "arxiv_id": str(record.get("arxiv_id") or identity.get("arxiv_id") or ""),
    }


def paper_family(record: dict[str, Any]) -> str:
    """Name the family one paper belongs to, so its own versions are not read as other papers.

    A preprint and the version of record, or two articles from the same proceedings series with
    different article numbers, are the same family: a case drawn from one of them cannot test an
    experience its sibling produced. The family is deliberately coarse -- the arXiv base id, or
    the DOI's registrant plus series -- because a de-leak rule that lets a sibling through is
    worse than one that drops a genuine neighbour.
    """

    identity = _identity_of(record)
    arxiv = identity["arxiv_id"].strip().casefold()
    if arxiv:
        return "arxiv:" + re.sub(r"v\d+$", "", arxiv)
    doi = identity["doi"].strip().casefold()
    if doi:
        registrant, _, suffix = doi.partition("/")
        if suffix:
            segments = suffix.split(".")
            series = ".".join(segments[:-1]) if len(segments) > 1 else suffix
            return f"doi:{registrant}/{series}"
        return "doi:" + registrant
    return ""


def _record_defects(record: dict[str, Any]) -> list[str]:
    """The defect patterns a record can answer for, from whatever shape it carries them in."""

    features = record.get("features") if isinstance(record.get("features"), dict) else {}
    for named in (record.get("defect_patterns"), features.get("defect_patterns")):
        if isinstance(named, list):
            return sorted({str(item) for item in named if str(item)})
    figure = record.get("figure") if isinstance(record.get("figure"), dict) else record.get("layout") if isinstance(record.get("layout"), dict) else None
    return defect_patterns(figure) if figure else []


def _record_buckets(record: dict[str, Any]) -> dict[str, Any]:
    """Read the comparable properties a Memory condition may name.

    ``defect_patterns`` answers with the whole list rather than one value, because a condition
    names a shape ("a one-to-many fan-out that crosses panels") and the figure either has that
    shape or does not.
    """

    features = record.get("features") if isinstance(record.get("features"), dict) else {}
    return {
        "domain": str(record.get("domain") or ""),
        "topology": str(record.get("topology") or features.get("topology") or ""),
        "node_count_bucket": str(record.get("node_count_bucket") or _bucket(int(features.get("node_count") or 0))),
        "connector_count_bucket": str(record.get("connector_count_bucket") or _bucket(int(features.get("edge_count") or 0))),
        "defect_patterns": _record_defects(record),
    }


def _answerable_buckets(record: dict[str, Any]) -> dict[str, Any]:
    """The condition dimensions this record positively answers about itself.

    A key counts only when the record actually carries the property. An index record that says
    nothing about its defects cannot be shown to fall outside a condition that names one, and a
    record with no counts at all cannot be shown to fall outside a condition about sizes. Reading
    an absent key as a mismatch would turn every candidate into a boundary case, which is the
    opposite of the negative control the ladder needs.
    """

    values = _record_buckets(record)
    features = record.get("features") if isinstance(record.get("features"), dict) else {}
    answerable: dict[str, Any] = {}
    if not domain_is_missing(record.get("domain")):
        answerable["domain"] = values["domain"]
    if str(record.get("topology") or "").strip():
        answerable["topology"] = values["topology"]
    if int(features.get("node_count") or record.get("node_count") or 0) > 0:
        answerable["node_count_bucket"] = values["node_count_bucket"]
    if int(features.get("edge_count") or record.get("connector_count") or 0) > 0:
        answerable["connector_count_bucket"] = values["connector_count_bucket"]
    named = record.get("defect_patterns") if isinstance(record.get("defect_patterns"), list) else features.get("defect_patterns")
    if isinstance(named, list) or isinstance(record.get("figure"), dict) or isinstance(record.get("layout"), dict):
        answerable["defect_patterns"] = values["defect_patterns"]
    return answerable


def outside_condition(record: dict[str, Any], condition: dict[str, Any] | None) -> str:
    """Return why the experience's own condition does not cover this record, or nothing.

    A condition is the sentence an experience was learned under, so it also says where the
    experience makes no claim. A candidate that answers one of those dimensions and answers it
    differently is a figure the experience was never meant to apply to: the boundary counter
    example the ladder asks for, and the only kind it can actually get from a paper-level index.
    """

    if not isinstance(condition, dict) or not condition:
        return ""
    asked = {key: condition[key] for key in CONDITION_KEYS if key in condition and condition[key] not in (None, "", [], {})}
    if not asked:
        return ""
    answerable = _answerable_buckets(record)
    checked = {key: wanted for key, wanted in asked.items() if key in answerable}
    if not checked:
        return ""
    misses: list[str] = []
    for key, wanted in checked.items():
        if key == "defect_patterns":
            missing = missing_patterns({key: wanted}, answerable[key])
            if missing:
                misses.append(f"{key} lacks {'/'.join(missing)}")
            continue
        if str(answerable[key]) != str(wanted):
            misses.append(f"{key}={answerable[key]} not {wanted}")
    if not misses:
        return ""
    return "the experience's condition does not cover this figure: " + "; ".join(misses)


def matched_condition(record: dict[str, Any], conditions: Iterable[dict[str, Any]] | None) -> str:
    """Return the excluded condition this record hits, or nothing when it hits none.

    A condition counts when every key it names is one this record can answer and all of them
    match, and at least one of them is comparable: a condition nobody can check would exclude
    every candidate, which is not what "the experience does not apply here" means.
    """

    values = _record_buckets(record)
    for condition in conditions or []:
        if not isinstance(condition, dict):
            continue
        dimension = str(condition.get("dimension") or "")
        if dimension and "value" in condition:
            if values.get(dimension) and values[dimension] == str(condition.get("value")):
                return f"{dimension}={condition['value']}"
            continue
        compared = {key: condition[key] for key in CONDITION_KEYS if key in condition and condition[key] not in (None, "", [], {})}
        matched = bool(compared)
        for key, wanted in compared.items():
            if key == "defect_patterns":
                # A shape is asked for as a whole: every pattern named has to be present, and a
                # figure with extra defects still answers the question.
                missing = missing_patterns(condition, values.get("defect_patterns") or [])
                if missing:
                    matched = False
                    break
                continue
            if values.get(key) != str(wanted):
                matched = False
                break
        if matched:
            return ", ".join(f"{key}={wanted if key == 'defect_patterns' else str(wanted)}" for key, wanted in compared.items())
    return ""


def outside_graph_scope(record: dict[str, Any], graph_scope: dict[str, Any] | None) -> str:
    """Return why this record falls outside the experience's own object scope, or nothing.

    Only decided when both sides name objects: a paper-level record that names none cannot be
    shown to be outside a scope it never speaks about, and guessing there would drop a case for
    no reason the report could defend.
    """

    scope = graph_scope if isinstance(graph_scope, dict) else {}
    wanted = {str(item) for key in SCOPE_ID_KEYS for item in (scope.get(key) or []) if str(item)}
    if not wanted:
        return ""
    held = {str(item) for key in SCOPE_ID_KEYS for item in (record.get(key) or []) if str(item)}
    if not held or wanted & held:
        return ""
    return "holds none of the objects this experience is scoped to"


def boundary_reason(
    record: dict[str, Any],
    *,
    excluded_conditions: Iterable[dict[str, Any]] | None = None,
    graph_scope: dict[str, Any] | None = None,
    condition: dict[str, Any] | None = None,
) -> str:
    """Return why this candidate is a boundary counter-example, or nothing.

    Three ways a candidate can be the negative control: it matches a condition the experience
    explicitly excluded, it holds none of the objects the experience is scoped to, or it is a
    figure the experience's own condition does not cover. The third is what makes the group
    reachable at all when a suite draws from a paper-level index, where no candidate shares an
    object id with the experience and no row writes an excluded condition.
    """

    excluded = matched_condition(record, excluded_conditions)
    if excluded:
        return f"matches an excluded condition of this experience ({excluded})"
    scope = outside_graph_scope(record, graph_scope)
    if scope:
        return f"outside the experience's graph scope: {scope}"
    return outside_condition(record, condition)


#: The groups a case set has to be able to fill before the exam is worth running at all.
#: An experience that claims to travel has to face another domain, and every exam needs a case
#: the experience was never meant to apply to: without a boundary, "this repairs the figure" and
#: "this is applied wherever it is seen" cannot be told apart, so the ladder can never be climbed.
REQUIRED_CASE_GROUPS = {
    "common": ("cross_domain_same_structure", "boundary"),
    "domain": ("boundary",),
}


def case_set_qualification(
    applicability: str,
    *,
    quota: dict[str, Any] | None = None,
    pool_counts: dict[str, Any] | None = None,
    profile: str = FULL_PROFILE,
) -> dict[str, Any]:
    """Decide whether an exam can reach the ladder, before a single paper is run.

    A suite that asked for none of a required group and a pool that holds none of it are both
    unwinnable. They are refused here, for different reasons, instead of being played out to
    arrive at a conclusion the case set had already fixed. This is checked before any run so a
    bad suite costs nothing.
    """

    kind = str(applicability or "").strip().casefold()
    resolved = {name: int((quota or DEFAULT_CASE_QUOTA).get(name) or 0) for name in CASE_GROUPS}
    if kind == "domain":
        resolved["cross_domain_same_structure"] = 0
    counts = {name: int((pool_counts or {}).get(name) or 0) for name in CASE_GROUPS}
    required = list(required_case_groups(kind, profile=profile))
    missing: list[dict[str, Any]] = []
    for name in required:
        if resolved[name] <= 0:
            missing.append({"group": name, "reason": "configured_zero"})
        elif counts[name] <= 0:
            missing.append({"group": name, "reason": "no_candidate_in_pool"})
    if kind not in REQUIRED_CASE_GROUPS:
        reason = "applicability_unspecified"
    else:
        reason = "case_set_quota_incomplete" if missing else ""
    return {
        "contract": "rfs.case_set_qualification",
        "version": 1,
        "ok": not missing and kind in REQUIRED_CASE_GROUPS,
        "reason": reason,
        "applicability": kind if kind in REQUIRED_CASE_GROUPS else "",
        "required_groups": required,
        "quota": resolved,
        "candidate_counts": counts,
        "missing": missing,
        "profile": str(profile or FULL_PROFILE).strip().casefold() or FULL_PROFILE,
    }


def select_validation_cases(
    query: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    *,
    quota: dict[str, Any] | None = None,
    exclude_paper_ids: Iterable[str] = (),
    excluded_conditions: Iterable[dict[str, Any]] | None = None,
    graph_scope: dict[str, Any] | None = None,
    same_structure_threshold: float = SAME_STRUCTURE_THRESHOLD,
    top_k: int = 0,
    applicability: str = "common",
    condition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pick a case set that covers the groups this experience's applicability asks for.

    Recall and rerank come first, exactly as they do for a single ranked list; this function
    only decides which of the ranked candidates to keep. A candidate is dropped before it can be
    chosen when it is the query paper itself, one of the papers the experience came from, or a
    sibling of either (the same arXiv base id, or the same DOI registrant and series), because a
    case drawn from the experience's own family tests nothing about generalization.
    """

    kind = str(applicability or "").strip().casefold()
    groups = DOMAIN_CASE_GROUPS if kind == "domain" else CASE_GROUPS
    resolved_quota = {name: int((quota or DEFAULT_CASE_QUOTA).get(name) or 0) for name in CASE_GROUPS}
    if kind == "domain":
        resolved_quota["cross_domain_same_structure"] = 0
    if kind not in {"common", "domain"}:
        return {
            "selected": [],
            "contract": "rfs.validation_case_selection",
            "version": 1,
            "recall": "not_run",
            "same_structure_threshold": float(same_structure_threshold),
            "applicability": "",
            "quota": resolved_quota,
            "groups": {name: [] for name in CASE_GROUPS},
            "uncovered": [{"group": name, "reason": "applicability was not written"} for name in CASE_GROUPS],
            "duplicate_paper_candidates": 0,
            "candidate_counts": {name: 0 for name in CASE_GROUPS},
            "excluded": [],
        }
    pool = [item for item in candidates if isinstance(item, dict)]
    blocked = {str(item) for item in exclude_paper_ids if str(item)}
    query_identity = _identity_of(query)
    query_families = {family for family in (paper_family(query),) if family}
    excluded: list[dict[str, Any]] = []
    keep: list[dict[str, Any]] = []
    for candidate in pool:
        identity = _identity_of(candidate)
        paper_id = identity["paper_id"]
        if paper_id and (paper_id in blocked or paper_id == query_identity["paper_id"]):
            excluded.append({"paper_id": paper_id, "reason": "the query paper itself or a source paper of this experience"})
            continue
        family = paper_family(candidate)
        if family and family in query_families:
            excluded.append({"paper_id": paper_id, "reason": f"the same paper family as the query ({family})"})
            continue
        keep.append(candidate)
    recall = "structural_only"
    if _embedding_of(query) and keep and all(_embedding_of(item) for item in keep):
        ranked = rank_validation_cases_with_recall(query, keep, top_k=max(len(keep), 1))
        recall = "vector_then_structure"
    else:
        ranked = rank_validation_cases(query, keep, top_k=max(len(keep), 1))
    threshold = float(same_structure_threshold)
    chosen: dict[str, list[dict[str, Any]]] = {name: [] for name in CASE_GROUPS}
    # One paper can hold several graph records, so a quota filled from two records of the same
    # paper is one case counted twice. A paper enters the set once, in whichever group it is
    # ranked into first; each later record of it is reported instead of silently admitted, and
    # the count is kept so a "quota of two" that is really one paper can be seen.
    used_papers: set[str] = set()
    duplicate_paper_candidates = 0
    # How many ranked candidates fell into each group, before the quota and the paper dedup
    # dropped any of them. This is the pool side of the qualification check: a group asked for
    # but absent here cannot be supplied by this suite at all.
    candidate_counts: dict[str, int] = {name: 0 for name in CASE_GROUPS}
    for entry in ranked:
        record = entry["record"]
        identity = _identity_of(record)
        query_domain = str(query.get("domain") or "")
        record_domain = str(record.get("domain") or "")
        query_missing = domain_is_missing(query_domain)
        record_missing = domain_is_missing(record_domain)
        same_domain = not query_missing and not record_missing and query_domain == record_domain
        cross_domain = not query_missing and not record_missing and query_domain != record_domain
        same_structure = float(entry["motif_similarity"]) >= threshold
        reason = boundary_reason(record, excluded_conditions=excluded_conditions, graph_scope=graph_scope, condition=condition)
        if reason:
            group = "boundary"
        elif kind == "domain" and cross_domain:
            excluded.append({"paper_id": identity["paper_id"], "reason": "a domain experience does not draw another domain"})
            continue
        elif not same_domain and not cross_domain:
            excluded.append({"paper_id": identity["paper_id"], "reason": "a paper without a domain cannot enter a same-domain group"})
            continue
        elif cross_domain and same_structure:
            group, reason = "cross_domain_same_structure", "another domain with the same structure"
        elif cross_domain:
            # The ladder asks for one cross-domain group and it is the same-structure one;
            # another domain with another structure is not a group this set has to cover.
            excluded.append({"paper_id": identity["paper_id"], "reason": "another domain with another structure, which is no group this ladder asks for"})
            continue
        elif same_structure:
            group, reason = "same_domain_same_structure", "the same domain with the same structure"
        else:
            group, reason = "same_domain_cross_structure", "the same domain with another structure"
        candidate_counts[group] = candidate_counts.get(group, 0) + 1
        if identity["paper_id"] and identity["paper_id"] in used_papers:
            duplicate_paper_candidates += 1
            excluded.append({"paper_id": identity["paper_id"], "reason": "the same paper is already a case"})
            continue
        if len(chosen[group]) >= resolved_quota[group]:
            excluded.append({"paper_id": identity["paper_id"], "reason": f"the {group} quota is already filled"})
            continue
        if identity["paper_id"]:
            used_papers.add(identity["paper_id"])
        chosen[group].append({
            "paper_id": identity["paper_id"],
            "record_id": str(record.get("record_id") or ""),
            "domain": str(record.get("domain") or ""),
            "group": group,
            "split": split_for_group(group),
            "reason": reason,
            "score": entry["score"],
            "motif_similarity": entry["motif_similarity"],
            "content_diversity": entry.get("content_diversity"),
        })
    selected: list[dict[str, Any]] = []
    for name in CASE_GROUPS:
        selected.extend(chosen[name])
    if int(top_k or 0) > 0:
        selected = selected[: int(top_k)]
        kept = {item["paper_id"] for item in selected}
        chosen = {name: [item for item in rows if item["paper_id"] in kept] for name, rows in chosen.items()}
    uncovered = [
        {"group": name, "reason": "no candidate in the pool belonged to this group" if resolved_quota[name] else "the suite asked for none of this group"}
        for name in CASE_GROUPS
        if resolved_quota[name] and not chosen[name]
    ]
    report = {
        "selected": [
            {
                "paper_id": item["paper_id"],
                "score": item["score"],
                "structure_similarity": item["motif_similarity"],
                "cross_domain": item["group"].startswith("cross_domain"),
                "group": item["group"],
                "split": item["split"],
                "reason": item["reason"],
            }
            for item in selected
        ],
        "contract": "rfs.validation_case_selection",
        "version": 1,
        "recall": recall,
        "same_structure_threshold": threshold,
        "applicability": kind,
        "quota": resolved_quota,
        "groups": {name: [item["paper_id"] for item in chosen[name]] for name in CASE_GROUPS},
        "uncovered": uncovered,
        "duplicate_paper_candidates": duplicate_paper_candidates,
        "candidate_counts": candidate_counts,
        "excluded": excluded,
    }
    return report
