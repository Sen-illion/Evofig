"""Deterministic entity and relation matching for the EditableFigureBench L1 layer.

The gold contracts name things in `snake_case` identifiers (`exploration_probability_alpha`)
while a generated figure names them in human labels ("Exploration probability
alpha"). This module normalises both sides and reports precision, recall, and F1
for nodes and edges, at two published strictness levels:

- `strict` compares normalised identifiers, so only spelling variants collapse.
- `soft` also accepts token-set containment and a small, published alias table.
- `embedding` is optional and off by default: it scores labels with a captured text
  embedder (see `EMBEDDING_ENV_VAR`) for paraphrases the token sets miss.

The strict and soft numbers are always reported and are fully offline, so nothing
here needs a credential and the same inputs always produce the same scores. The
embedding level is opt-in and its absence is reported, never silently substituted.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterable, Sequence

#: Camel-case boundaries become token boundaries before comparison.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")

#: Greek letters appear in both contract identifiers and rendered labels.
_GREEK = {
    "\u03b1": "alpha",
    "\u03b2": "beta",
    "\u03b3": "gamma",
    "\u03b4": "delta",
    "\u03b5": "epsilon",
    "\u03b7": "eta",
    "\u03b8": "theta",
    "\u03bb": "lambda",
    "\u03bc": "mu",
    "\u03c0": "pi",
    "\u03c1": "rho",
    "\u03c3": "sigma",
    "\u03c4": "tau",
    "\u03c6": "phi",
    "\u03c9": "omega",
}

#: Words that carry no discriminating meaning in a scientific figure label.
STOPWORDS = frozenset(
    {"a", "an", "the", "of", "and", "for", "to", "in", "on", "with", "by", "via", "using", "based"}
)

#: Published, deliberately small alias table for relation types whose wording
#: differs but whose meaning and direction do not.
RELATION_TYPE_ALIASES = {
    "provideinput": "feed",
    "inputinto": "feed",
    "outputto": "output",
    "produceoutput": "output",
    "flowto": "flow",
    "dataflow": "flow",
}

#: Token-overlap threshold a soft node match must reach.
SOFT_MATCH_THRESHOLD = 0.6

#: Cosine threshold an embedding-backed node match must reach. Set above the
#: token threshold because embeddings score unrelated scientific labels high.
EMBEDDING_MATCH_THRESHOLD = 0.72

#: Environment variable that opts the L1 layer into embedding matching.
#: Empty or unset means the level is reported as disabled.
EMBEDDING_ENV_VAR = "RFS_BINDING_EMBEDDER"

#: Unmatched entries kept in the report for failure analysis.
MAX_REPORTED_UNMATCHED = 40


def normalize_tokens(value: Any) -> tuple[str, ...]:
    """Split a label or identifier into comparable lowercase tokens."""
    text = str(value if value is not None else "")
    for letter, name in _GREEK.items():
        text = text.replace(letter, f" {name} ")
    text = _CAMEL_BOUNDARY.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text.lower())
    tokens: list[str] = []
    for token in text.split():
        if not token or token in STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tuple(tokens)


def normalize_key(value: Any) -> str:
    """Return the canonical identifier form of a label."""
    return "_".join(normalize_tokens(value))


def normalize_relation_type(value: Any) -> str:
    """Normalise a relation type, collapsing wording variants that keep direction.

    Keys in RELATION_TYPE_ALIASES are written in already-normalised form, so
    "provides input to" reaches the table as "provideinput".
    """
    joined = "".join(token for token in normalize_tokens(value) if token != "into")
    return RELATION_TYPE_ALIASES.get(joined, joined)


def _token_set(value: Any) -> frozenset[str]:
    return frozenset(normalize_tokens(value))


def _similarity(left: Any, right: Any) -> float:
    """Token overlap for the soft level.

    Single-character tokens are dropped before comparing: in a figure label they
    are indices or subscripts ("agent i", "alpha_i"), not part of the name, and
    keeping them stops "alpha_i" from binding a module named
    exploration_probability_alpha. A label that carries no longer token keeps its
    original tokens, so a bare symbol never matches by becoming empty.
    """
    left_raw = _token_set(left)
    right_raw = _token_set(right)
    if not left_raw or not right_raw:
        return 1.0 if left_raw == right_raw else 0.0
    left_tokens = frozenset(token for token in left_raw if len(token) > 1)
    right_tokens = frozenset(token for token in right_raw if len(token) > 1)
    if not left_tokens or not right_tokens:
        left_tokens, right_tokens = left_raw, right_raw
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return 1.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, or 0.0 if either is degenerate."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_square = 0.0
    right_square = 0.0
    for first, second in zip(left, right):
        dot += first * second
        left_square += first * first
        right_square += second * second
    if left_square <= 0.0 or right_square <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_square) * math.sqrt(right_square))


def embedding_scorer(embedder: Any, *, threshold: float = EMBEDDING_MATCH_THRESHOLD) -> Callable[[Any, Any], float]:
    """Score two labels by the cosine similarity of their captured embeddings.

    The embedder is expected to be the process-wide cached provider, so every
    distinct label is embedded at most once even though the greedy matcher asks
    about every gold/predicted pair.
    """

    def score(left: Any, right: Any) -> float:
        vectors = embedder.encode([str(left), str(right)])
        return cosine_similarity(vectors[0], vectors[1])

    return score


def _entity_scorer(mode: str, embedder: Any = None) -> tuple[Callable[[Any, Any], float], float]:
    if mode == "strict":
        return (lambda left, right: 1.0 if normalize_key(left) == normalize_key(right) else 0.0), 1.0
    if mode == "embedding":
        if embedder is None:
            raise ValueError("embedding matching needs an embedder")
        return embedding_scorer(embedder), EMBEDDING_MATCH_THRESHOLD
    return _similarity, SOFT_MATCH_THRESHOLD


def _greedy_match(
    gold: Sequence[Any],
    predicted: Sequence[Any],
    score: Callable[[Any, Any], float],
    threshold: float,
) -> list[tuple[int, int, float]]:
    """Pair gold and predicted entries one-to-one, best score first."""
    candidates: list[tuple[float, int, int]] = []
    for gold_index, gold_item in enumerate(gold):
        for predicted_index, predicted_item in enumerate(predicted):
            value = score(gold_item, predicted_item)
            if value >= threshold:
                candidates.append((value, gold_index, predicted_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_gold: set[int] = set()
    used_predicted: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for value, gold_index, predicted_index in candidates:
        if gold_index in used_gold or predicted_index in used_predicted:
            continue
        used_gold.add(gold_index)
        used_predicted.add(predicted_index)
        pairs.append((gold_index, predicted_index, value))
    return pairs


def _prf(matched: int, predicted: int, gold: int) -> dict[str, Any]:
    if predicted:
        precision = matched / predicted
    else:
        precision = 1.0 if gold == 0 else 0.0
    if gold:
        recall = matched / gold
    else:
        recall = 1.0 if predicted == 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "matched": matched,
        "gold_count": gold,
        "predicted_count": predicted,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _unmatched(values: Sequence[Any], used: set[int]) -> list[str]:
    return [str(values[index]) for index in range(len(values)) if index not in used][:MAX_REPORTED_UNMATCHED]


def match_entities(
    gold: Sequence[Any],
    predicted: Sequence[Any],
    *,
    mode: str = "strict",
    embedder: Any = None,
) -> dict[str, Any]:
    """Match generated node labels against the gold required entities.

    `embedder` is only read in `embedding` mode; the offline levels ignore it.
    """
    score, threshold = _entity_scorer(mode, embedder)
    pairs = _greedy_match(gold, predicted, score, threshold)
    report = _prf(len(pairs), len(predicted), len(gold))
    report["mode"] = mode
    report["unmatched_gold"] = _unmatched(gold, {item[0] for item in pairs})
    report["unmatched_predicted"] = _unmatched(predicted, {item[1] for item in pairs})
    return report


def relation_triple(value: Any) -> tuple[str, str, str]:
    """Read a relation from either a mapping or a three-item sequence."""
    if isinstance(value, dict):
        return (
            str(value.get("source") or ""),
            str(value.get("target") or ""),
            str(value.get("relation_type") or value.get("type") or ""),
        )
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (str(value[0]), str(value[1]), str(value[2]))
    raise ValueError(f"unsupported relation entry: {value!r}")


def match_relations(gold: Sequence[Any], predicted: Sequence[Any], *, mode: str = "strict") -> dict[str, Any]:
    """Match generated edges against the gold required relations.

    Direction is never relaxed: a reversed edge never matches. Only the wording
    of the relation type is normalised, and only in soft mode. When the gold
    contract records a relation without a type, its endpoints alone decide.
    """
    gold_triples = [relation_triple(item) for item in gold]
    predicted_triples = [relation_triple(item) for item in predicted]
    endpoint_score, endpoint_threshold = _entity_scorer(mode)
    type_normalizer = normalize_relation_type if mode != "strict" else normalize_key

    def score(left: tuple[str, str, str], right: tuple[str, str, str]) -> float:
        # A gold relation with no recorded type is scored on its endpoints alone,
        # because the frozen contract never stated which relation it was.
        if normalize_key(left[2]) and type_normalizer(left[2]) != type_normalizer(right[2]):
            return 0.0
        if endpoint_score(left[0], right[0]) < endpoint_threshold:
            return 0.0
        if endpoint_score(left[1], right[1]) < endpoint_threshold:
            return 0.0
        return 1.0

    pairs = _greedy_match(gold_triples, predicted_triples, score, 1.0)
    report = _prf(len(pairs), len(predicted_triples), len(gold_triples))
    report["mode"] = mode
    report["unmatched_gold"] = [str(gold_triples[index]) for index in range(len(gold_triples)) if index not in {item[0] for item in pairs}][:MAX_REPORTED_UNMATCHED]
    report["unmatched_predicted"] = [str(predicted_triples[index]) for index in range(len(predicted_triples)) if index not in {item[1] for item in pairs}][:MAX_REPORTED_UNMATCHED]
    return report


def rendered_entity_recall(gold: Sequence[Any], rendered_text: str) -> dict[str, Any]:
    """Estimate how many gold entities appear in the deck's editable text.

    This is a heuristic proxy: it answers "can a reader see this entity in the
    delivered slide" without an OCR pass, so it is reported alongside, never
    instead of, the program-level node scores.
    """
    haystack = _token_set(rendered_text)
    hits = 0
    missing: list[str] = []
    for item in gold:
        tokens = _token_set(item)
        if tokens and tokens <= haystack:
            hits += 1
        elif tokens and len(tokens & haystack) / len(tokens) >= SOFT_MATCH_THRESHOLD:
            hits += 1
        else:
            missing.append(str(item))
    total = len(gold)
    return {
        "matched": hits,
        "gold_count": total,
        "recall": round((hits / total) if total else 1.0, 4),
        "unmatched_gold": missing[:MAX_REPORTED_UNMATCHED],
        "method": "normalised token containment against editable PPTX text",
    }



#: Characters that only appear in mathematical notation, never in a module label.
_NOTATION_MARKERS = re.compile(r"[\\^={}\[\]]|[∈→←∑√∫∏≤≥≠⋅×±]|\$")

#: A function-call label such as "Mask(X)" or "g(o_t)".
_SYMBOL_CALL = re.compile(r"^([A-Za-z][A-Za-z0-9]{0,15})\s*[\(\[](.*)[\)\]]$")


def is_notation(value: Any) -> bool:
    """Whether a label is mathematical notation rather than a named module.

    The frozen contracts name modules, inputs and outputs (agent_policy), but a
    planner can also turn an equation into a node (o_{-i}, Mask(X), g_{att}(o_t)).
    Those labels cannot bind to a module term and would otherwise count as extra,
    unfaithful nodes, so the L1 layer separates them and reports the split
    instead of scoring it.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return False
    if _NOTATION_MARKERS.search(text):
        return True
    call = _SYMBOL_CALL.match(text)
    if call:
        arguments = [part for part in re.split(r"[,\s]+", call.group(2)) if part]
        if arguments and all(len(part) <= 3 for part in arguments):
            return True
    tokens = normalize_tokens(text)
    return bool(tokens) and all(len(token) <= 2 for token in tokens)


def split_notation(values: Sequence[Any]) -> tuple[list[str], list[str]]:
    """Split labels into named modules and mathematical notation, order preserved."""
    modules: list[str] = []
    notation: list[str] = []
    for value in values:
        text = str(value if value is not None else "")
        (notation if is_notation(text) else modules).append(text)
    return modules, notation


def module_coverage(
    gold: Sequence[Any],
    predicted: Sequence[Any],
    *,
    mode: str = "strict",
    embedder: Any = None,
) -> dict[str, Any]:
    """How many frozen gold modules the figure represents at all.

    This is the recall side read on its own. Precision over drawn nodes is not a
    module-faithfulness signal, because a figure may legitimately split one gold
    module into several boxes; the extra boxes are reported by unbound_nodes.

    Every predicted label is a candidate binding, notation included: a contract
    can name a module after a symbol (exploration_probability_alpha) that the
    figure legitimately writes as a formula (alpha_i). How often that happened is
    reported as notation_modules_matched rather than hidden.
    """
    predicted = list(predicted)
    gold_modules, gold_notation = split_notation(gold)
    _, predicted_notation = split_notation(predicted)
    score, threshold = _entity_scorer(mode, embedder)
    pairs = _greedy_match(gold_modules, predicted, score, threshold)
    matched = len(pairs)
    return {
        "mode": mode,
        "matched": matched,
        "gold_module_count": len(gold_modules),
        "coverage": round(matched / len(gold_modules), 4) if gold_modules else 1.0,
        "missing": _unmatched(gold_modules, {item[0] for item in pairs}),
        "gold_notation_excluded": len(gold_notation),
        "predicted_notation_count": len(predicted_notation),
        "notation_modules_matched": sum(1 for pair in pairs if is_notation(predicted[pair[1]])),
    }


def unbound_nodes(
    gold: Sequence[Any],
    predicted: Sequence[Any],
    *,
    mode: str = "strict",
    embedder: Any = None,
) -> dict[str, Any]:
    """Drawn module labels that bind to no frozen gold module.

    This is an upper bound on fabricated content, not a verdict: a box that only
    subdivides a gold module also lands here, so the labels are kept for the
    failure taxonomy and the must_not_invent hard gate stays the fabrication gate.

    A predicted label that is pure notation and bound to nothing is an equation
    the planner drew, not a module box, so it is excluded from the rate and
    counted separately. Notation that does bind a gold module is excluded here
    too, because module_coverage already credits it.
    """
    predicted = list(predicted)
    gold_modules, _ = split_notation(gold)
    score, threshold = _entity_scorer(mode, embedder)
    pairs = _greedy_match(gold_modules, predicted, score, threshold)
    bound = {item[1] for item in pairs}
    unbound_indexes = [index for index in range(len(predicted)) if index not in bound and not is_notation(predicted[index])]
    notation_unbound = sum(1 for index in range(len(predicted)) if index not in bound and is_notation(predicted[index]))
    total = len(predicted) - notation_unbound
    unbound = len(unbound_indexes)
    return {
        "mode": mode,
        "unbound": unbound,
        "predicted_module_count": total,
        "rate": round(unbound / total, 4) if total else 0.0,
        "labels": [predicted[index] for index in unbound_indexes][:MAX_REPORTED_UNMATCHED],
        "notation_unbound_excluded": notation_unbound,
    }

__all__ = [
    "MAX_REPORTED_UNMATCHED",
    "RELATION_TYPE_ALIASES",
    "SOFT_MATCH_THRESHOLD",
    "STOPWORDS",
    "is_notation",
    "match_entities",
    "match_relations",
    "normalize_key",
    "normalize_relation_type",
    "module_coverage",
    "normalize_tokens",
    "split_notation",
    "unbound_nodes",
    "relation_triple",
    "rendered_entity_recall",
]
