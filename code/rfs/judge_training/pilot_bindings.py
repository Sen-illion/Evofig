"""Bind the scientific and visual graph layers of one Pilot paper.

Phase 3 builds two independent layers per paper: the scientific layer read from
the paper text and the visual layer read from the paper's own framework figure.
This module decides which scientific node each drawn block realizes, which
scientific relation each drawn arrow carries, and it records everything it could
not bind instead of inventing a link.

Matching runs in four layers, in this order of authority:

1. What the reader actually saw. The check stage asks the provider, concept by
   concept, which text the paper's own figure carries for it. That answer is the
   strongest available evidence that a concept is on the figure, so it decides the
   mapping first, and it may point several concepts at one drawn box.
2. Reading the labels directly. The scientific layer writes sentences and the
   figure carries short phrases, so this pass only reaches what the wording already
   shares. It stays the fully independent reading, and its own count is reported
   separately, so the improvement stays attributable to one layer.
3. Structure. A drawn block has to sit where the concept sits: a pure input must
   not be bound to a block that something flows into, a merge must not be bound to
   a block with one arrow in, a bound block should belong to a panel. A violation
   never moves a binding; it lowers confidence and raises needs_review.
4. Semantics. When the strings cannot reach a pair, an embedder may rank it, but a
   vector never makes a binding: a semantic-only pair is kept as weak evidence and
   stays out of the coverage. Nothing frozen depends on this layer, and when no
   embedder is configured the deterministic layers run exactly as before.

Beyond the concept-first question -- where was this concept drawn -- the stage
answers the figure-first one. ``claim_figure_objects`` gives every object the
figure draws, blocks, row labels and arrows alike, exactly one conclusion about
what it is, because a figure also draws things the paper's concept list never
mentions and a count that starts from concepts can only call those nothing.

The stage calls no VLM. The only optional provider call is the embedder, which the
caller opts into by name; the run records which provider it used, how many texts it
embedded and, when a call failed, why it fell back to the deterministic layers.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Iterable, Sequence

from rfs.contracts import normalize_graph_bindings, stable_graph_id, validate_graph_bindings
from rfs.providers.api_embeddings import api_embedder, api_model_name, is_api_spec
from rfs.providers.embeddings import EmbeddingUnavailable, TextEmbedder, resolve_embedder

FIGURE_CLAIMS_CONTRACT = "rfs.figure_claims"
FIGURE_CLAIMS_VERSION = 4
BINDING_VERSION = "pilot-layer-bindings-v7"
LABEL_MATCH_METHOD = "label_match"
CHECK_MATCH_METHOD = "image_label_match"
SEMANTIC_MATCH_METHOD = "semantic_match"
ENDPOINT_MATCH_METHOD = "endpoint_match"
DECLARED_OUTSIDE_METHOD = "reader_declared_outside"
ENDPOINT_OUTSIDE_METHOD = "endpoint_outside_paper_method"
PROVENANCE_METHOD = "paper_text_match"
LOCAL_ROLE_METHOD = "local_rule"
READER_ROLE_METHOD = "reader_role"
UNREADABLE_LABEL_METHOD = "unreadable_label"
ENDPOINT_ANNOTATION_METHOD = "endpoint_figure_annotation"
DECLARED_ANNOTATION_METHOD = "reader_declared_annotation"
NOTE_REGION_METHOD = "figure_notation_region"
CLAIM_BOUND = "bound"
CLAIM_OUTSIDE_PAPER_METHOD = "outside_paper_method"
CLAIM_IN_PAPER_NOT_IN_METHOD = "in_paper_not_in_method"
CLAIM_FIGURE_ANNOTATION = "figure_annotation"
CLAIM_UNRESOLVED = "unresolved"
CLAIM_BLOCK = "block"
CLAIM_ROW_LABEL = "row_label"
CLAIM_ARROW = "arrow"
CLAIM_OTHER = "other"
CLAIM_NOTE_REGION = "notation_region"
CLAIM_CONCLUSIONS = (
    CLAIM_BOUND,
    CLAIM_OUTSIDE_PAPER_METHOD,
    CLAIM_IN_PAPER_NOT_IN_METHOD,
    CLAIM_FIGURE_ANNOTATION,
    CLAIM_UNRESOLVED,
)
ROLE_METHOD_OBJECT = "method_object"
ROLE_EXTERNAL_METHOD = "external_method"
ROLE_STAGE_ANNOTATION = "stage_annotation"
ROLE_SYMBOL = "symbol"
ROLE_DECORATIVE = "decorative"
ROLES = (ROLE_METHOD_OBJECT, ROLE_EXTERNAL_METHOD, ROLE_STAGE_ANNOTATION, ROLE_SYMBOL, ROLE_DECORATIVE)
ROLE_ALIASES = {
    "method": ROLE_METHOD_OBJECT,
    "module": ROLE_METHOD_OBJECT,
    "method_object": ROLE_METHOD_OBJECT,
    "external": ROLE_EXTERNAL_METHOD,
    "outside": ROLE_EXTERNAL_METHOD,
    "external_method": ROLE_EXTERNAL_METHOD,
    "annotation": ROLE_STAGE_ANNOTATION,
    "stage": ROLE_STAGE_ANNOTATION,
    "stage_annotation": ROLE_STAGE_ANNOTATION,
    "panel_title": ROLE_STAGE_ANNOTATION,
    "symbol": ROLE_SYMBOL,
    "formula": ROLE_SYMBOL,
    "decorative": ROLE_DECORATIVE,
    "decoration": ROLE_DECORATIVE,
}
ROLE_EXCLUDED_FROM_MEANING = frozenset({ROLE_DECORATIVE, ROLE_SYMBOL})
# The roles that name the figure explaining itself rather than a step of the method: a
# panel title, a formula drawn as its own object, a scale tick. Only these three may end
# an unclaimed object as a figure annotation. ``method_object`` is the opposite claim --
# the reading says this object *is* a step of the method -- so an unclaimed one stays
# unresolved and is looked up in the paper instead of being filed away as notation, and
# ``external_method`` speaks for another paper's method, not for this figure's own.
ROLE_FIGURE_NOTATION = frozenset({ROLE_STAGE_ANNOTATION, ROLE_SYMBOL, ROLE_DECORATIVE})
# The three answers the correspondence review may give about how a figure carries a
# concept. They live here because the review is read through the Bindings layer: a
# review that disagrees with what the claims pass sees is recorded, never obeyed.
HOW_AS_BLOCK = "as_block"
HOW_INSIDE_OTHER_GRAPHICS = "inside_other_graphics"
HOW_NOT_DRAWN = "not_drawn"
HOWS = (HOW_AS_BLOCK, HOW_INSIDE_OTHER_GRAPHICS, HOW_NOT_DRAWN)
CLAIM_TIER_OWN_LABEL = 1
CLAIM_TIER_READER_TEXT = 2
CLAIM_TIER_SEMANTIC = 3
CLAIM_TIER_DECLARED = 4
EXACT_MATCH = 1.0
CONTAINMENT_MATCH = 0.85
STRONG_MATCH = 0.6
WEAK_MATCH = 0.35
SEMANTIC_STRONG = 0.6
SEMANTIC_NEAR = 0.5
MULTI_MATCH_MARGIN = 0.05
STRUCTURE_PENALTY = 0.6
CONFIDENCE_FLOOR = 0.1
MIN_CONTAINMENT_CHARS = 4
MIN_TOKEN_CHARS = 2
BLOCK_KIND = "module"
PANEL_KIND = "panel"
NOTE_REGION_KIND = "notation_region"
BINDABLE_KINDS = (BLOCK_KIND, PANEL_KIND)
MAX_CANDIDATES = 3
MAX_CONFLICT_EXAMPLES = 20
MAX_UNACCOUNTED_BLOCKS = 20
MAX_SELECTION_RECORDS = 60
MAX_ANNOTATION_RECORDS = 40
MIN_REPEAT_LABEL_CHARS = 4
BOX_TOLERANCE = 0.02
FEEDBACK_RELATIONS = frozenset({"feedback", "feedback_loop", "iteration", "iterate", "return_flow", "loop"})
EMBEDDING_OFF_SPECS = frozenset({"", "off", "none", "disabled"})
_NON_WORD = re.compile(r"[^\w]+")
_BARE_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
_SINGLE_LETTER = re.compile(r"^[a-z]$")
_SIZE_TOKEN = re.compile(r"^\d+x(?:\d+x)*\d+$")
_FORMULA_MARK = re.compile(r"[\\^_{}=<>←→⟨⟩|$]|\d[x×]\d")
_STAGE_WORD = re.compile(r"^(?:panel|main|sub|fig|figure|step|stage|diagram|workflow|section|part|legend|key|colorbar|colourbar|scale)\b")
_WORD_TOKEN = re.compile(r"[a-z]{3,}")
# The words a label may carry and still be notation only: the Greek letter names and the
# LaTeX fragments a reading leaves behind when it copies a formula out of a figure. Any
# other word is content, and content is never taken out of the denominator by shape.
_MATH_WORDS = frozenset({
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota",
    "kappa", "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "upsilon", "phi",
    "chi", "psi", "omega", "pi", "hat", "bar", "tilde", "dot", "ddot", "vec",
    "frac", "dfrac", "sqrt", "sum", "prod", "int", "lim", "min", "max", "arg",
    "exp", "log", "ln", "grad", "div", "det", "tr", "diag", "cdot", "times",
    "left", "right", "begin", "end", "text", "mathcal", "mathbb", "mathrm",
    "mathbf", "mathsf", "mathtt", "bmatrix", "matrix", "cases", "operatorname",
    "quad", "qquad", "partial", "nabla", "overline", "underline",
})


def graph_sha256(graph: dict[str, Any]) -> str:
    """Hash one layer the way the artifact ledger hashes it.

    A graph's id is stable for its paper, so it says nothing about the graph's
    content: only a hash of the graph itself can tell a binding that was built
    against the current layer from one built against a superseded revision.
    """
    canonical = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_label(value: Any) -> str:
    """Fold one visible label into a comparable form."""
    text = str(value or "").casefold()
    return " ".join(_NON_WORD.sub(" ", text).split())


def label_tokens(label: str) -> set[str]:
    """Tokens of a normalized label, dropping one-character noise."""
    return {token for token in label.split(" ") if len(token) >= MIN_TOKEN_CHARS}


def label_score(scientific_label: Any, visual_label: Any) -> float:
    """Return how strongly one scientific label matches one visible label."""
    first = normalize_label(scientific_label)
    second = normalize_label(visual_label)
    if not first or not second:
        return 0.0
    if first == second:
        return EXACT_MATCH
    if len(first) >= MIN_CONTAINMENT_CHARS and len(second) >= MIN_CONTAINMENT_CHARS and (first in second or second in first):
        return CONTAINMENT_MATCH
    left = label_tokens(first)
    right = label_tokens(second)
    if not left or not right:
        return 0.0
    return round(len(left & right) / len(left | right), 6)


def _node_map(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(node.get("id") or ""): node
        for node in ((graph or {}).get("nodes") or [])
        if isinstance(node, dict) and str(node.get("id") or "")
    }


def _edge_map(graph: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    return {
        str(edge.get("id") or ""): edge
        for edge in ((graph or {}).get("edges") or [])
        if isinstance(edge, dict) and str(edge.get("id") or "")
    }


def _degree_map(graph: dict[str, Any] | None) -> dict[str, tuple[int, int]]:
    """Return the in- and out-degree of every node of one layer."""
    degrees: dict[str, tuple[int, int]] = {}
    for node in _node_map(graph):
        degrees[node] = (0, 0)
    for edge in _edge_map(graph).values():
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in degrees:
            inbound, outbound = degrees[source]
            degrees[source] = (inbound, outbound + 1)
        if target in degrees:
            inbound, outbound = degrees[target]
            degrees[target] = (inbound + 1, outbound)
    return degrees


def _panel_present(visual_graph: dict[str, Any] | None) -> bool:
    return any(str(node.get("kind") or "") == PANEL_KIND for node in _node_map(visual_graph).values())


def _bindable_targets(visual_nodes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the visual nodes a concept may bind to: drawn blocks and panel labels.

    A panel is the row or group name of the figure, and on a comparison figure that name
    is where the author states which row is theirs. The pool used to hold drawn blocks
    only, so a concept the figure carries as a row label could never bind, even when the
    reading quoted that label word for word. Panel labels are first-class targets here;
    they are offered, never forced, because the score still has to clear the same
    threshold a drawn block does.
    """
    return {
        visual_id: node
        for visual_id, node in visual_nodes.items()
        if str(node.get("kind") or "") in BINDABLE_KINDS
    }


def reader_object_role(visual_node: dict[str, Any]) -> str | None:
    """Return the role the reading gave one figure object, folded onto the five known roles.

    Only a role the reading chose for the object itself counts. The provider's
    older ``visual_role`` tag answers a different question -- what kind of graphic
    the box is -- so it is never read as one of the five.
    """
    properties = visual_node.get("properties") if isinstance(visual_node.get("properties"), dict) else {}
    raw = str(properties.get("role") or "").strip().casefold()
    return ROLE_ALIASES.get(raw)


def content_words(label: Any) -> list[str]:
    """Return the ordinary words of one label, formula marks and math names taken out.

    A drawn object may carry a formula and still be content: "Classification Loss
    $\\mathcal{L}_{cls}$" names a loss of the method, while "U <- U - eta grad_U" is
    notation. The Greek letter names and the LaTeX fragments go, and what the wording
    still says decides the two.
    """
    return [token for token in _WORD_TOKEN.findall(str(label or "").casefold()) if token not in _MATH_WORDS]


def object_is_unlabeled(visual_node: dict[str, Any]) -> bool:
    """Say whether an object's own text is missing rather than merely short.

    The v1 reading fell back to the provider id when it could not read a block's
    text and wrote ``text_confidence: 0`` beside it, so an id-shaped label is a gap
    in our own reading, not a label. The v2 reading leaves the label empty instead,
    and the shared graph contract then fills it with the node's own id, which is the
    same gap written differently. Neither shape says anything about the figure.
    """
    label = str(visual_node.get("label") or "").strip()
    if not label:
        return True
    properties = visual_node.get("properties") if isinstance(visual_node.get("properties"), dict) else {}
    source_id = str(properties.get("source_id") or "").strip()
    if source_id and label == source_id:
        return True
    node_id = str(visual_node.get("id") or "").strip()
    if node_id and label == node_id:
        return True
    confidence = properties.get("text_confidence")
    return confidence == 0 and len(label) <= 3 and not _FORMULA_MARK.search(label.casefold())


def local_object_role(visual_node: dict[str, Any]) -> str | None:
    """Classify an object nothing claimed by the shape of its own wording.

    The rules only run on objects no concept and no reading declaration answered
    for, so they cannot move a binding, and they only name wording that is notation
    rather than content: a bare number or scale tick is decoration, a formula or a
    sub-three-character token is a symbol, a panel or legend title is a stage
    annotation. A formula is notation only while no ordinary word survives in it.

    Two shapes are deliberately left unclassified. An object whose own text our
    reading could not produce, and a label that carries an ordinary word beside a
    formula, are both gaps or content rather than notation; calling either one
    decoration would take it out of the denominator on the strength of our own
    reading, so both stay counted as content, and stay unresolved until something
    else explains them.
    """
    if object_is_unlabeled(visual_node):
        return None
    label = str(visual_node.get("label") or "").strip()
    normalized = normalize_label(label)
    if not normalized:
        # A label that is nothing but glyphs -- "+", "-", "x", "..." -- is the
        # figure's own arithmetic rather than wording we failed to read: it is
        # notation, and notation leaves the denominator. An absent label never
        # reaches this line, because the unread test above has already taken it.
        return ROLE_SYMBOL
    tokens = normalized.split(" ")
    if all(_BARE_NUMBER.match(token) or _SIZE_TOKEN.match(token) or _SINGLE_LETTER.match(token) for token in tokens):
        return ROLE_DECORATIVE
    if _FORMULA_MARK.search(label.casefold()) and not content_words(label):
        return ROLE_SYMBOL
    if len(normalized) <= 3:
        return ROLE_SYMBOL
    if _STAGE_WORD.match(normalized):
        return ROLE_STAGE_ANNOTATION
    return None


def repeat_label_group(visual_node: dict[str, Any]) -> str:
    """Return the wording key under which identical drawings of one object group."""
    if str(visual_node.get("kind") or "") != BLOCK_KIND or object_is_unlabeled(visual_node):
        return ""
    normalized = normalize_label(visual_node.get("label"))
    if len(normalized) < MIN_REPEAT_LABEL_CHARS or not any(character.isalpha() for character in normalized):
        return ""
    return normalized


def local_repeat_groups(visual_nodes: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Group the blocks one figure draws with the same wording.

    The v2 reading marks a repeated drawing itself; until then the same wording
    drawn twice in one figure is the only local evidence of repetition, and it is
    enough to offer the free copies to the concept that already owns one.
    """
    groups: dict[str, list[str]] = {}
    for visual_id in sorted(visual_nodes):
        key = repeat_label_group(visual_nodes[visual_id])
        if key:
            groups.setdefault(key, []).append(visual_id)
    return {key: ids for key, ids in groups.items() if len(ids) > 1}


def reader_repeat_of(visual_node: dict[str, Any]) -> dict[str, Any] | None:
    """Return the repetition the v2 reading recorded for one object, when it did."""
    properties = visual_node.get("properties") if isinstance(visual_node.get("properties"), dict) else {}
    key = str(properties.get("repeat_group") or "").strip()
    if not key:
        return None
    index = properties.get("repeat_index")
    count = properties.get("repeat_count")
    return {
        "key": key,
        "index": int(index) if isinstance(index, (int, float)) and not isinstance(index, bool) else None,
        "count": int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) else None,
    }


def _box_of(visual_node: dict[str, Any]) -> dict[str, float] | None:
    """Return one object's normalized box, when the reading recorded a usable one."""
    properties = visual_node.get("properties") if isinstance(visual_node.get("properties"), dict) else {}
    box = properties.get("box")
    if not isinstance(box, dict):
        return None
    numbers: dict[str, float] = {}
    for key in ("x", "y", "w", "h"):
        value = box.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numbers[key] = float(value)
    return numbers if numbers["w"] > 0 and numbers["h"] > 0 else None


def _box_contains(outer: dict[str, float], inner: dict[str, float]) -> bool:
    return (
        inner["x"] >= outer["x"] - BOX_TOLERANCE
        and inner["y"] >= outer["y"] - BOX_TOLERANCE
        and inner["x"] + inner["w"] <= outer["x"] + outer["w"] + BOX_TOLERANCE
        and inner["y"] + inner["h"] <= outer["y"] + outer["h"] + BOX_TOLERANCE
    )


def _box_holds_point(box: dict[str, float], x: float, y: float) -> bool:
    return (
        box["x"] - BOX_TOLERANCE <= x <= box["x"] + box["w"] + BOX_TOLERANCE
        and box["y"] - BOX_TOLERANCE <= y <= box["y"] + box["h"] + BOX_TOLERANCE
    )


def owning_block(visual_id: str, visual_nodes: dict[str, dict[str, Any]]) -> str:
    """Return the smallest drawn block that contains this object, or an empty string.

    A formula drawn inside a box explains that box: the reading draws it as its own
    object, so geometry is the only local evidence of which object it belongs to.
    When nothing contains it, the annotation is recorded without an owner instead
    of being attached to the nearest block by guesswork.
    """
    box = _box_of(visual_nodes.get(visual_id) or {})
    if box is None:
        return ""
    center_x = box["x"] + box["w"] / 2
    center_y = box["y"] + box["h"] / 2
    best = ""
    best_area: float | None = None
    for other_id in sorted(visual_nodes):
        if other_id == visual_id:
            continue
        other = visual_nodes[other_id]
        if str(other.get("kind") or "") != BLOCK_KIND:
            continue
        other_box = _box_of(other)
        if other_box is None:
            continue
        if not (_box_contains(other_box, box) or _box_holds_point(other_box, center_x, center_y)):
            continue
        area = other_box["w"] * other_box["h"]
        if best_area is None or area < best_area:
            best, best_area = other_id, area
    return best


def _bound_visual_ids(match: dict[str, Any] | None) -> list[str]:
    """Return every drawn block one scientific node is bound to."""
    if not match:
        return []
    listed = [str(value) for value in (match.get("visual_node_ids") or []) if str(value)]
    if listed:
        return listed
    primary = str(match.get("visual_node_id") or "")
    return [primary] if primary else []


def _concept_visual_ids(node_bindings: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Return the drawn objects each concept is bound to, in binding order."""
    bound: dict[str, list[str]] = {}
    for item in node_bindings:
        if not isinstance(item, dict):
            continue
        scientific_id = str(item.get("scientific_node_id") or "")
        if scientific_id:
            bound[scientific_id] = [str(value) for value in item.get("visual_node_ids") or [] if str(value)]
    return bound


class _SemanticScorer:
    """Cosine similarity over one embedder, which degrades instead of guessing.

    The scorer embeds every distinct label of a paper once and answers from that cache,
    so the cost of the layer is one request per paper rather than one per comparison. A
    provider that cannot be loaded, or a call that fails, turns the layer off and keeps
    the reason: the deterministic layers then decide exactly the bindings they always did.
    """

    def __init__(self, embedder: TextEmbedder | None = None, *, status: str = "off", error: str = "") -> None:
        self._embedder = embedder
        self._vectors: dict[str, list[float]] = {}
        self.status = status
        self.error = error
        self.fingerprint: dict[str, Any] | None = None
        self.texts = 0
        self.calls = 0
        self.total_tokens = 0

    @property
    def available(self) -> bool:
        return self._embedder is not None

    def prepare(self, texts: Iterable[str]) -> None:
        """Embed every distinct label once; turn the layer off when that fails.

        One embedder serves the whole run, so its counters are cumulative: this layer
        reports the delta its own paper caused, not the total the process has reached.
        Without that, every paper would carry its predecessors' usage into the ledger.
        """
        if self._embedder is None:
            return
        wanted = sorted({str(text) for text in texts if str(text).strip()})
        before = self._usage()
        try:
            self.fingerprint = dict(self._embedder.fingerprint())
            self.status = str(self.fingerprint.get("kind") or "unknown")
            vectors = self._embedder.encode(wanted) if wanted else []
        except EmbeddingUnavailable as exc:
            self._disable(str(exc))
            return
        except (OSError, ValueError) as exc:
            self._disable("embedder could not be loaded: " + str(exc))
            return
        self._vectors = {text: list(vector) for text, vector in zip(wanted, vectors)}
        usage = self._usage()
        self.texts = len(wanted)
        self.calls = max(0, int(usage.get("calls") or 0) - int(before.get("calls") or 0))
        self.total_tokens = max(0, int(usage.get("total_tokens") or 0) - int(before.get("total_tokens") or 0))

    def _disable(self, reason: str) -> None:
        self._embedder = None
        self._vectors = {}
        self.status = "unavailable"
        self.error = reason

    def _usage(self) -> dict[str, Any]:
        usage = getattr(self._embedder, "usage", None)
        return dict(usage()) if callable(usage) else {}

    def similarity(self, left: Any, right: Any) -> float | None:
        """Return the cosine of two prepared labels, or None without a semantic layer."""
        if self._embedder is None:
            return None
        first = self._vectors.get(str(left or ""))
        second = self._vectors.get(str(right or ""))
        if not first or not second or len(first) != len(second):
            return None
        return round(sum(one * two for one, two in zip(first, second)), 6)

    def usage_report(self) -> dict[str, Any]:
        """Return what this layer used and whether it was available at all."""
        return {
            "status": self.status,
            "error": self.error,
            "fingerprint": dict(self.fingerprint) if isinstance(self.fingerprint, dict) else None,
            "texts": int(self.texts),
            "calls": int(self.calls),
            "total_tokens": int(self.total_tokens),
        }


def _pair_scores(scientific_label: Any, visual_label: Any, semantic: "_SemanticScorer | None") -> tuple[float, float | None]:
    """Return the string score of a pair and its semantic score when one exists."""
    value = semantic.similarity(scientific_label, visual_label) if semantic is not None else None
    return label_score(scientific_label, visual_label), value


def _binding_texts(
    scientific_nodes: dict[str, dict[str, Any]],
    visual_nodes: dict[str, dict[str, Any]],
    figure_text_review: dict[str, Any] | None,
) -> list[str]:
    """Return every label the semantic layer may be asked about, once each."""
    texts: list[str] = []
    for node in scientific_nodes.values():
        if str(node.get("kind") or "") != PANEL_KIND:
            texts.append(str(node.get("label") or ""))
    for node in visual_nodes.values():
        if str(node.get("kind") or "") == BLOCK_KIND:
            texts.append(str(node.get("label") or ""))
    for concept in (figure_text_review or {}).get("concepts") or []:
        if isinstance(concept, dict):
            texts.append(str(concept.get("image_label") or ""))
            texts.append(str(concept.get("label") or ""))
    return texts


def resolve_binding_embedder(spec: str | None) -> tuple[TextEmbedder | None, dict[str, Any]]:
    """Resolve the bind stage's embedder spec and say what it resolved to.

    "off" and "none" ask for the deterministic layers only, "api" or "api:<model>" for
    the paid endpoint, and anything else is handed to the local resolver so "hashing"
    and a pinned "st:" directory keep working. A spec that cannot be loaded returns no
    embedder plus the reason, instead of quietly binding with a different provider.
    """
    text = "" if spec is None else str(spec).strip()
    folded = text.casefold()
    if folded in EMBEDDING_OFF_SPECS:
        return None, {"status": "off", "spec": text, "error": ""}
    if is_api_spec(text):
        model = api_model_name(text)
        return api_embedder(model), {"status": "api_embeddings", "spec": "api:" + model, "error": ""}
    try:
        embedder = resolve_embedder(text)
    except EmbeddingUnavailable as exc:
        return None, {"status": "unavailable", "spec": text, "error": str(exc)}
    return embedder, {"status": "local", "spec": text, "error": ""}


def match_layer_nodes(
    scientific_graph: dict[str, Any],
    visual_graph: dict[str, Any],
    semantic: "_SemanticScorer | None" = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Match scientific nodes to the figure's own labels, one target per scientific node.

    Greedy assignment over the highest scores, with a deterministic tie-break on ids, so
    a rerun always reproduces the same pairs. Every target is used at most once; a label
    that repeats stays with its best match and the others are reported. A target is a
    drawn block or a panel label, so a concept the figure states as a row name binds to
    that row instead of falling through.

    A pair the strings cannot reach may still be offered by the semantic layer, but the
    vector never decides by itself: it neither ranks this pass nor makes a semantic-only
    pair strong. The order stays the strings' own, so the count this pass produces is the
    same string baseline whether or not an embedding is switched on; the vector can only
    offer a residual pair as weak evidence, and an embedding can then never inflate
    coverage.
    """
    scientific_nodes = _node_map(scientific_graph)
    visual_nodes = _node_map(visual_graph)
    targets = _bindable_targets(visual_nodes)
    candidates: list[tuple[int, float, float, str, str]] = []
    near_misses = 0
    for scientific_id, scientific in scientific_nodes.items():
        if str(scientific.get("kind") or "") == PANEL_KIND:
            continue
        label = scientific.get("label")
        if not normalize_label(label):
            continue
        for visual_id, visual in targets.items():
            string, semantic_score = _pair_scores(label, visual.get("label"), semantic)
            value = 0.0 if semantic_score is None else float(semantic_score)
            if string < WEAK_MATCH:
                if value < SEMANTIC_STRONG:
                    if value >= SEMANTIC_NEAR:
                        near_misses += 1
                    continue
                candidates.append((0, string, value, scientific_id, visual_id))
                continue
            candidates.append((1 if string >= STRONG_MATCH else 0, string, value, scientific_id, visual_id))
    # The rank is the strings' own: a vector tie-break here would move pairs this pass
    # has always decided, and the point of this pass is to stay the string baseline.
    candidates.sort(key=lambda item: (-item[0], -item[1], item[3], item[4]))
    matches: dict[str, dict[str, Any]] = {}
    taken_visual: dict[str, str] = {}
    skipped: list[dict[str, Any]] = []
    semantic_only = 0
    for strong, string, semantic_score, scientific_id, visual_id in candidates:
        if scientific_id in matches:
            continue
        if visual_id in taken_visual:
            skipped.append({"scientific_node_id": scientific_id, "visual_node_id": visual_id, "score": round(max(string, semantic_score), 6), "reason": "block_already_bound"})
            continue
        matches[scientific_id] = {
            "visual_node_id": visual_id,
            "score": round(max(string, semantic_score), 6),
            "string": round(string, 6),
            "semantic": round(semantic_score, 6),
            "strong": bool(strong),
        }
        taken_visual[visual_id] = scientific_id
        if not strong:
            semantic_only += 1
    audit = {
        "scored_pairs": len(candidates),
        "skipped_duplicate_blocks": skipped[:20],
        "unlabeled_visual_blocks": sorted(
            visual_id
            for visual_id, visual in visual_nodes.items()
            if str(visual.get("kind") or "") == BLOCK_KIND and not normalize_label(visual.get("label"))
        ),
        "matched_visual_blocks": sum(
            1 for visual_id in taken_visual
            if str((targets.get(visual_id) or {}).get("kind") or "") == BLOCK_KIND
        ),
        "matched_panel_targets": sum(
            1 for visual_id in taken_visual
            if str((targets.get(visual_id) or {}).get("kind") or "") == PANEL_KIND
        ),
        "visual_blocks": sum(1 for visual in visual_nodes.values() if str(visual.get("kind") or "") == BLOCK_KIND),
        "panel_targets": sum(1 for visual in targets.values() if str(visual.get("kind") or "") == PANEL_KIND),
        "semantic_weak_nodes": semantic_only,
        "semantic_near_misses": near_misses,
    }
    return matches, audit


def match_layer_nodes_via_check(
    visual_graph: dict[str, Any],
    figure_text_review: dict[str, Any] | None,
    semantic: "_SemanticScorer | None" = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Bind the concepts the check stage already read off the same figure.

    The check stage asks the provider, concept by concept, which text the paper's figure
    carries for it. That answer is the bridge the label match cannot cross, and it is the
    primary criterion here: this pass runs on its own, independently of the label match,
    and the caller decides which side wins when the two disagree. Only concepts the
    provider marked visible take part, so a concept the figure never drew stays unbound
    instead of being given a block it does not own.

    A block may carry more than one concept: the scientific layer often splits one drawn
    box into several concepts, and refusing the second concept would drop a link the
    provider confirmed. Sharing is counted, not hidden. A panel label is a target as
    well, so a concept the figure states as a row name binds to that row.
    """
    concepts = [item for item in ((figure_text_review or {}).get("concepts") or []) if isinstance(item, dict)]
    visual_nodes = _node_map(visual_graph)
    targets = _bindable_targets(visual_nodes)
    candidates: list[tuple[float, float, float, str, str, float]] = []
    skipped_not_visible = 0
    skipped_no_image_label = 0
    skipped_no_match = 0
    near_misses = 0
    for concept in concepts:
        scientific_id = str(concept.get("concept_id") or "")
        if not scientific_id:
            continue
        if not concept.get("visible"):
            skipped_not_visible += 1
            continue
        image_label = str(concept.get("image_label") or "").strip()
        if not image_label:
            skipped_no_image_label += 1
            continue
        confidence = concept.get("confidence")
        confidence_value = float(confidence) if isinstance(confidence, (int, float)) and not isinstance(confidence, bool) else 0.0
        best_id = ""
        best_key = (0.0, 0.0, 0.0)
        for visual_id in sorted(targets):
            visual = targets[visual_id]
            string, semantic_score = _pair_scores(image_label, visual.get("label"), semantic)
            value = 0.0 if semantic_score is None else float(semantic_score)
            if string < STRONG_MATCH and value < SEMANTIC_STRONG:
                if value >= SEMANTIC_NEAR or string >= WEAK_MATCH:
                    near_misses += 1
                continue
            key = (max(string, value), string, value)
            if key > best_key:
                best_id, best_key = visual_id, key
        if not best_id:
            skipped_no_match += 1
            continue
        candidates.append((best_key[0], best_key[1], best_key[2], scientific_id, best_id, confidence_value))
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3], item[4]))
    matches: dict[str, dict[str, Any]] = {}
    claimed: set[str] = set()
    shared = 0
    semantic_nodes = 0
    for score, string, semantic_score, scientific_id, visual_id, confidence_value in candidates:
        if visual_id in claimed:
            shared += 1
        claimed.add(visual_id)
        matches[scientific_id] = {
            "visual_node_id": visual_id,
            "score": round(score, 6),
            "string": round(string, 6),
            "semantic": round(semantic_score, 6),
            "strong": bool(string >= STRONG_MATCH or semantic_score >= SEMANTIC_STRONG),
            "check_confidence": confidence_value,
        }
        if string < STRONG_MATCH and semantic_score >= SEMANTIC_STRONG:
            semantic_nodes += 1
    audit = {
        "image_label_bound": len(matches),
        "image_label_shared_blocks": shared,
        "image_label_bound_panels": sum(
            1 for item in matches.values()
            if str((targets.get(str(item.get("visual_node_id") or "")) or {}).get("kind") or "") == PANEL_KIND
        ),
        "image_label_skipped_not_visible": skipped_not_visible,
        "image_label_skipped_no_image_label": skipped_no_image_label,
        "image_label_skipped_no_match": skipped_no_match,
        "check_concepts": len(concepts),
        "semantic_fact_nodes": semantic_nodes,
        "semantic_fact_near_misses": near_misses,
    }
    return matches, audit


def match_layer_edges(
    scientific_graph: dict[str, Any],
    visual_graph: dict[str, Any],
    node_matches: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Match scientific relations to drawn arrows through the bound endpoints.

    Every drawn block a concept is bound to is offered as an endpoint, so a module the
    figure draws in two panels still carries the arrows that touch either copy.
    """
    scientific_edges = _edge_map(scientific_graph)
    visual_edges = _edge_map(visual_graph)
    by_endpoints: dict[tuple[str, str], list[str]] = {}
    for visual_id, edge in visual_edges.items():
        key = (str(edge.get("source") or ""), str(edge.get("target") or ""))
        by_endpoints.setdefault(key, []).append(visual_id)
    matches: dict[str, dict[str, Any]] = {}
    reversed_ids: list[str] = []
    type_mismatches: list[dict[str, str]] = []
    for scientific_id, edge in scientific_edges.items():
        sources = _bound_visual_ids(node_matches.get(str(edge.get("source") or "")))
        targets = _bound_visual_ids(node_matches.get(str(edge.get("target") or "")))
        if not sources or not targets:
            continue
        forward: list[str] = []
        backward: list[str] = []
        for source_visual in sources:
            for target_visual in targets:
                forward.extend(by_endpoints.get((source_visual, target_visual), []))
                backward.extend(by_endpoints.get((target_visual, source_visual), []))
        forward = sorted(set(forward))
        backward = sorted(set(backward))
        if forward:
            matches[scientific_id] = {"visual_edge_ids": forward, "direction": "forward"}
        elif backward:
            matches[scientific_id] = {"visual_edge_ids": backward, "direction": "reversed"}
            reversed_ids.append(scientific_id)
        else:
            continue
        scientific_relation = str(edge.get("relation_type") or "").casefold()
        for visual_id in matches[scientific_id]["visual_edge_ids"]:
            visual_relation = str((visual_edges.get(visual_id) or {}).get("relation_type") or "").casefold()
            if (scientific_relation in FEEDBACK_RELATIONS) != (visual_relation in FEEDBACK_RELATIONS):
                type_mismatches.append({"scientific_edge_id": scientific_id, "visual_edge_id": visual_id, "scientific_relation": scientific_relation, "visual_relation": visual_relation})
    audit = {
        "reversed_edges": sorted(reversed_ids),
        "relation_type_mismatches": type_mismatches[:20],
        "matched_visual_edges": sum(len(item["visual_edge_ids"]) for item in matches.values()),
        "visual_edges": len(visual_edges),
    }
    return matches, audit


def structure_violations(
    scientific: dict[str, Any],
    scientific_degrees: dict[str, tuple[int, int]],
    visual_ids: Sequence[str],
    visual_nodes: dict[str, dict[str, Any]],
    visual_degrees: dict[str, tuple[int, int]],
    *,
    panels_present: bool,
) -> list[str]:
    """Return the structural ways a bound block contradicts its scientific node.

    Every rule is a statement about degree and panel membership rather than about role
    wording, because the two layers name their roles in their own vocabularies. A
    violation lowers confidence and asks for review; it never removes the binding.

    The degree rules speak about drawn blocks, so they are read over the block targets
    only: a panel label carries no arrows of its own, and being bound to one is not an
    isolated block. Panel membership is asked of the blocks, and a panel target answers
    it by being the panel itself.
    """
    if not visual_ids:
        return []
    blocks = [visual_id for visual_id in visual_ids if str((visual_nodes.get(visual_id) or {}).get("kind") or "") == BLOCK_KIND]
    panel_targets = [visual_id for visual_id in visual_ids if str((visual_nodes.get(visual_id) or {}).get("kind") or "") == PANEL_KIND]
    violations: list[str] = []
    if blocks:
        inbound, outbound = scientific_degrees.get(str(scientific.get("id") or ""), (0, 0))
        block_in = sum(visual_degrees.get(visual_id, (0, 0))[0] for visual_id in blocks)
        block_out = sum(visual_degrees.get(visual_id, (0, 0))[1] for visual_id in blocks)
        if (inbound or outbound) and not (block_in or block_out):
            violations.append("isolated_block")
        if inbound == 0 and outbound > 0 and block_in > 0:
            violations.append("direction_conflict")
        if outbound == 0 and inbound > 0 and block_out > 0:
            violations.append("direction_conflict")
        if inbound >= 2 and block_in <= 1:
            violations.append("topology_conflict")
        if outbound >= 2 and block_out <= 1:
            violations.append("topology_conflict")
    if panels_present and not panel_targets and all(not ((visual_nodes.get(visual_id) or {}).get("panel_ids") or []) for visual_id in blocks):
        violations.append("panel_missing")
    return violations


def _candidate_rows(
    scientific_label: Any,
    visual_nodes: dict[str, dict[str, Any]],
    semantic: "_SemanticScorer | None",
    *,
    accepted: str,
) -> list[dict[str, Any]]:
    """Return the best few targets of one concept, with what each of them scored."""
    rows: list[dict[str, Any]] = []
    targets = _bindable_targets(visual_nodes)
    for visual_id in sorted(targets):
        visual = targets[visual_id]
        string, semantic_score = _pair_scores(scientific_label, visual.get("label"), semantic)
        value = 0.0 if semantic_score is None else float(semantic_score)
        score = max(string, value)
        if score <= 0.0 and visual_id != accepted:
            continue
        rows.append({
            "visual_node_id": visual_id,
            "score": round(score, 6),
            "string": round(string, 6),
            "semantic": None if semantic_score is None else round(float(semantic_score), 6),
            "accepted": visual_id == accepted,
        })
    rows.sort(key=lambda item: (-item["score"], -item["string"], item["visual_node_id"]))
    return rows[:MAX_CANDIDATES]


def _multi_bound_blocks(
    node_matches: dict[str, dict[str, Any]],
    scientific_nodes: dict[str, dict[str, Any]],
    visual_nodes: dict[str, dict[str, Any]],
    semantic: "_SemanticScorer | None",
) -> dict[str, list[str]]:
    """Give a concept the free copies of its block that the figure draws again.

    A module the figure draws in two panels is one concept with two drawn blocks. Sharing
    one block between concepts is already allowed; this adds the opposite direction, and
    it only ever adds a block that scores as strongly as the accepted one, is within the
    margin, and that no other concept claims as its own.
    A panel is never added here: a row label names a group, it is not a second drawing of
    the same module.
    """
    claimed = {str(match.get("visual_node_id") or "") for match in node_matches.values()}
    extras: dict[str, list[str]] = {}
    used: set[str] = set()
    for scientific_id in sorted(node_matches):
        match = node_matches[scientific_id]
        best = str(match.get("visual_node_id") or "")
        if not best:
            continue
        extra: list[str] = []
        for visual_id in sorted(visual_nodes):
            visual = visual_nodes[visual_id]
            if str(visual.get("kind") or "") != BLOCK_KIND:
                continue
            if visual_id == best or visual_id in claimed or visual_id in used:
                continue
            string, semantic_score = _pair_scores(scientific_nodes[scientific_id].get("label"), visual.get("label"), semantic)
            value = 0.0 if semantic_score is None else float(semantic_score)
            score = max(string, value)
            if score >= STRONG_MATCH and float(match.get("score") or 0.0) - score <= MULTI_MATCH_MARGIN:
                extra.append(visual_id)
        if extra:
            extras[scientific_id] = extra
            used.update(extra)
    return extras


def _second_instance_blocks(
    node_matches: dict[str, dict[str, Any]],
    scientific_nodes: dict[str, dict[str, Any]],
    visual_nodes: dict[str, dict[str, Any]],
    semantic: "_SemanticScorer | None",
) -> dict[str, list[str]]:
    """Give a concept every other copy of its drawing, and near-identical copies.

    Two rules, in this order. First the figure's own repetition: when two blocks
    carry the same wording, the free copy is the same drawing again, and it joins
    the concept that owns its group even when the concept's own label never reaches
    the copy's wording -- the reading read that wording for the concept, so a
    repeated drawing must keep all of its copies. Second, the older margin rule for
    copies whose wording is merely close, which stays exactly as it was. A panel is
    never a copy: a row label names a group, not a second drawing of the module.
    """
    groups = local_repeat_groups(visual_nodes)
    group_of = {visual_id: key for key, ids in groups.items() for visual_id in ids}
    claimed = {str(match.get("visual_node_id") or "") for match in node_matches.values()}
    taken = set(claimed)
    extras: dict[str, list[str]] = {}
    for scientific_id in sorted(node_matches):
        best = str(node_matches[scientific_id].get("visual_node_id") or "")
        key = group_of.get(best)
        if not key:
            continue
        extra = [visual_id for visual_id in groups[key] if visual_id != best and visual_id not in taken]
        if extra:
            extras[scientific_id] = extra
            taken.update(extra)
    margin = _multi_bound_blocks(node_matches, scientific_nodes, visual_nodes, semantic)
    for scientific_id, extra_ids in margin.items():
        fresh = [visual_id for visual_id in extra_ids if visual_id not in taken]
        if not fresh:
            continue
        extras[scientific_id] = [*extras.get(scientific_id, []), *fresh]
        taken.update(fresh)
    return extras


def _figure_first_report(
    visual_graph: dict[str, Any],
    figure_text_review: dict[str, Any] | None,
    node_bindings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Account for the figure the way a reader does: block by block.

    The author's figure is the standard, so the question here is "what did the author
    draw, and does the paper explain it?", never "what did the author leave out". A
    drawn block the paper never names is either a gap in our own reading or a block
    that belongs outside the contract; it is listed for review, never called an error.
    """
    visual_nodes = _node_map(visual_graph)
    drawn = {
        visual_id
        for visual_id, node in visual_nodes.items()
        if str(node.get("kind") or "") == BLOCK_KIND
    }
    claimed: set[str] = set()
    for item in node_bindings:
        claimed.update(str(value) for value in item.get("visual_node_ids") or [])
    declared = {
        str(block.get("label") or "").strip().casefold()
        for block in _declared_outside_blocks(figure_text_review)
    }
    unaccounted = sorted(
        visual_id
        for visual_id in drawn - claimed
        if str((visual_nodes.get(visual_id) or {}).get("label") or "").strip().casefold() not in declared
    )
    return {
        "drawn_blocks": len(drawn),
        "claimed_blocks": len(claimed & drawn),
        "declared_outside_contract": len(declared),
        "unaccounted_blocks": len(unaccounted),
        "unaccounted_block_ids": unaccounted[:MAX_UNACCOUNTED_BLOCKS],
    }


def _selection_samples(
    scientific_graph: dict[str, Any],
    figure_text_review: dict[str, Any] | None,
    *,
    node_bindings: Sequence[dict[str, Any]] | None = None,
    block_ids: Iterable[str] = (),
    roles: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Record the author's trade-off as data: what the figure kept, and how it kept it.

    The content layer answers whether a concept reaches the figure at all. A concept the
    author wrote inside another drawing -- in a formula, under an arrow, in a row name --
    is content the author kept, so it counts as drawn, and only a concept the figure
    never carries counts as an omission. The expression layer then says how it got there:
    as a block of its own, or inside something else the figure draws. The two are kept
    apart because they teach different things: what to draw, and how to draw it.

    This is the negative half of what a figure teaches, and the reason the author's
    omissions are kept at all instead of being counted as misses.

    The judged set is every concept of the scientific layer the review answered about, so
    the sample is the whole inventory minus what the figure drew. The required subset is
    reported beside it, because the earlier runs measured that subset only and those
    numbers have to stay comparable.

    ``as_block`` counts a drawing of the concept's own, so an object the claims pass
    classified as notation -- a formula or a scale tick -- never counts as the block a
    concept was drawn as, even when the review read the concept's wording on it.

    The form the review chose for a concept is kept beside the local one as
    ``reader_how``. It is evidence rather than a decision: the sample is filed by what
    the claims pass can see on the figure, and every disagreement is counted so the
    difference stays visible instead of being averaged into the sample.
    """
    required = [str(value) for value in scientific_graph.get("required_node_ids") or []]
    if figure_text_review is None:
        return {
            "required_concepts": len(required),
            "review_available": False,
            "judged": 0,
            "drawn": None,
            "not_drawn": None,
            "as_block": None,
            "inside_other_graphics": None,
            "required_judged": 0,
            "required_drawn": None,
            "required_not_drawn": None,
            "reader_how_conflicts": None,
            "records": [],
            "records_truncated": False,
        }
    answers = {
        str(concept.get("concept_id") or ""): concept
        for concept in figure_text_review.get("concepts") or []
        if isinstance(concept, dict) and str(concept.get("concept_id") or "")
    }
    scientific_nodes = _node_map(scientific_graph)
    blocks = {str(value) for value in block_ids}
    drawn_objects = _concept_visual_ids(node_bindings or [])
    required_set = set(required)
    judged = sorted(node_id for node_id in scientific_nodes if node_id in answers)
    drawn = as_block = required_judged = required_drawn = how_conflicts = 0
    records: list[dict[str, Any]] = []
    for node_id in judged:
        answer = answers[node_id]
        visible = bool(answer.get("visible"))
        own_block = visible and any(
            visual_id in blocks and (roles or {}).get(visual_id) not in ROLE_EXCLUDED_FROM_MEANING
            for visual_id in drawn_objects.get(node_id, [])
        )
        if visible:
            drawn += 1
        if own_block:
            as_block += 1
        if node_id in required_set:
            required_judged += 1
            if visible:
                required_drawn += 1
        local_how = HOW_AS_BLOCK if own_block else (HOW_INSIDE_OTHER_GRAPHICS if visible else HOW_NOT_DRAWN)
        reader_how = str(answer.get("reader_how") or "")
        if reader_how and reader_how != local_how:
            how_conflicts += 1
        if len(records) < MAX_SELECTION_RECORDS:
            node = scientific_nodes[node_id]
            records.append({
                "concept_id": node_id,
                "label": str(node.get("label") or answer.get("label") or ""),
                "kind": str(node.get("kind") or ""),
                "required": node_id in required_set,
                "drawn": visible,
                "as_block": bool(own_block),
                "inside_other_graphics": bool(visible and not own_block),
                "reader_how": reader_how,
                "image_label": str(answer.get("image_label") or ""),
                "where": str(answer.get("where") or ""),
            })
    return {
        "required_concepts": len(required),
        "review_available": True,
        "judged": len(judged),
        "drawn": drawn,
        "not_drawn": len(judged) - drawn,
        "as_block": as_block,
        "inside_other_graphics": drawn - as_block,
        "required_judged": required_judged,
        "required_drawn": required_drawn,
        "required_not_drawn": required_judged - required_drawn,
        "reader_how_conflicts": how_conflicts,
        "records": records,
        "records_truncated": len(judged) > len(records),
    }


def _declared_outside_blocks(figure_text_review: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the labels the reading itself declared as outside this paper's method."""
    declared: list[dict[str, Any]] = []
    source = figure_text_review or {}
    rows = source.get("unlisted_content")
    if rows is None:
        # A review written before the split only carried extra_blocks, and those rows were
        # the content the figure draws outside the concept list.
        rows = source.get("extra_blocks")
    for block in rows or []:
        if not isinstance(block, dict):
            continue
        label = str(block.get("label") or "").strip()
        if label:
            declared.append({"label": label, "where": str(block.get("where") or "")})
    return declared


def _declared_region_annotations(figure_text_review: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the figure's own annotations as the review declared them.

    A legend, a panel title, a scale or a colour bar is the figure explaining itself:
    it is notation rather than a step of the method, and only the review that read the
    figure may say so. Rows keep the wording as printed and what the annotation explains.
    """
    declared: list[dict[str, Any]] = []
    for item in (figure_text_review or {}).get("region_annotations") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        explains = str(item.get("explains") or "").strip()
        if label or explains:
            declared.append({
                "label": label,
                "kind": str(item.get("kind") or "").strip(),
                "explains": explains,
                "where": str(item.get("where") or ""),
            })
    return declared


def _declared_outside_match(label: Any, declared: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the declaration one drawn label answers to, when the reading made one."""
    for item in declared:
        if label_score(item.get("label"), label) >= CONTAINMENT_MATCH:
            return item
    return None


def _claim_tier(status: str, method: str) -> int:
    """Rank one piece of evidence the way a reader would rank it.

    The object's own label comes first: the label pass compares the figure's label
    with the concept's label and nothing else, so a pair it decided was decided by
    what the figure itself says. The text the correspondence review read off the
    figure for a concept comes second, because it reaches the object through a
    concept rather than through the object's own words. A vector's opinion comes
    last, and only ever as weak evidence.
    """
    if status == "weak":
        return CLAIM_TIER_SEMANTIC
    if method == CHECK_MATCH_METHOD:
        return CLAIM_TIER_READER_TEXT
    return CLAIM_TIER_OWN_LABEL


def _candidate_reason(score: float, bound_elsewhere: bool) -> str:
    """Say why one candidate did not win, in the terms its own score is read in."""
    if bound_elsewhere:
        return "the concept is bound to another object of the figure"
    if score < STRONG_MATCH:
        return "the agreement is below the match threshold"
    return "the object was claimed by a stronger piece of evidence"


def _endpoint_report(
    end: str,
    visual_id: str,
    conclusions: dict[str, str],
    claims_by_visual: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Describe one end of a drawn arrow: what it is, and which concepts it carries."""
    accepted = sorted({
        str(claim.get("scientific_node_id") or "")
        for claim in claims_by_visual.get(visual_id, [])
        if claim.get("accepted") and claim.get("scientific_node_id")
    })
    return {
        "end": end,
        "visual_node_id": visual_id,
        "conclusion": conclusions.get(visual_id, "missing"),
        "scientific_node_ids": accepted,
    }


def _arrow_candidates(
    ends: Sequence[dict[str, Any]],
    candidates_by_visual: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Return the best few concepts the unnamed end of an arrow could be.

    An end that already answers to a concept is not a candidate for anything: what an
    unexplained arrow needs is a name for the end nobody could name.
    """
    rows: list[dict[str, Any]] = []
    for end in ends:
        if str(end.get("conclusion") or "") == CLAIM_BOUND:
            continue
        visual_id = str(end.get("visual_node_id") or "")
        for item in candidates_by_visual.get(visual_id, []):
            rows.append({**item, "end": str(end.get("end") or ""), "visual_node_id": visual_id})
    rows.sort(key=lambda row: (-float(row.get("score") or 0.0), row["end"], str(row.get("scientific_node_id") or "")))
    return rows[:MAX_CANDIDATES]


def claim_figure_objects(
    scientific_graph: dict[str, Any],
    visual_graph: dict[str, Any],
    figure_text_review: dict[str, Any] | None = None,
    *,
    bindings: dict[str, Any] | None = None,
    provenance: Callable[[Any], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Give every object the figure draws exactly one conclusion about what it is.

    The bind layer asks where each concept of the paper was drawn. This layer asks
    the opposite question, because a figure draws things the paper's concept list
    does not explain -- a second method in a comparison row, a legend, the equations
    inside a box -- and a count that only ever starts from concepts can say nothing
    about them. Every drawn block, every row label and every arrow therefore ends
    with exactly one conclusion:

    ``bound``
        the object answers to a scientific node (a block or a row label) or to a
        scientific relation between two bound objects (an arrow).
    ``outside_paper_method``
        the object is content this paper's own method does not contain. Only the
        reading may say this, by naming the label in its extra blocks: a concept the
        paper declares and the figure leaves out is the author's trade-off, never an
        outside method, so absence from the concept list never decides this.
    ``in_paper_not_in_method``
        the paper carries the object's wording as contiguous text on a known page,
        but the method list does not contain it: the figure printed something the
        article says elsewhere, not a concept of the method this paper proposes.
        Only a strict phrase match decides this, never a bag of words.
    ``figure_annotation``
        the object is the figure's own notation rather than one of its content
        objects: a legend or panel title, a formula or symbol, a scale tick. The
        reading's role decides this when it names one, and three local rules decide
        it for the v1 readings that never carried roles. An arrow with such an object
        at either end belongs here too, because it explains the notation the figure
        prints about itself. An object our own reading could not put text on is never
        filed here: that is a gap in our reading, so it stays counted as content.
    ``unresolved``
        nothing explains the object yet. It carries the best three candidates and
        why each of them was refused, so the gap is reviewable instead of silent.

    Evidence is ranked the way a reader ranks it -- the object's own label, then the
    text the correspondence review read off the figure, then a vector's opinion --
    and every piece of evidence is kept on the object, so a disagreement between the
    label pass and the review stays visible instead of being averaged away.

    ``needs_review`` marks a claim that contradicts itself rather than an object that
    is merely unresolved: evidence that disagrees about what the object is, a
    declaration that the object is outside the method while a concept claims it, or
    an arrow whose relation is missing or runs backwards.

    ``role`` records what kind of thing the object is, so an object excluded from the
    meaningful denominator is excluded by name and counted, never silently dropped.
    ``repeat_group`` records the drawing the object repeats, ``belongs_to`` and
    ``annotations`` attach notation to the object it explains, and ``provenance``
    carries the page a phrase match came from.
    """
    paper_id = str(scientific_graph.get("paper_id") or visual_graph.get("paper_id") or "")
    scientific_nodes = _node_map(scientific_graph)
    visual_nodes = _node_map(visual_graph)
    visual_edges = _edge_map(visual_graph)
    node_bindings = [item for item in ((bindings or {}).get("node_bindings") or []) if isinstance(item, dict)]
    edge_bindings = [item for item in ((bindings or {}).get("edge_bindings") or []) if isinstance(item, dict)]
    declared = _declared_outside_blocks(figure_text_review)
    declared_annotations = _declared_region_annotations(figure_text_review)
    blocks = {
        visual_id
        for visual_id, node in visual_nodes.items()
        if str(node.get("kind") or "") == BLOCK_KIND
    }
    claims_by_visual: dict[str, list[dict[str, Any]]] = {}
    candidates_by_visual: dict[str, list[dict[str, Any]]] = {}
    for item in node_bindings:
        scientific_id = str(item.get("scientific_node_id") or "")
        scientific_label = str((scientific_nodes.get(scientific_id) or {}).get("label") or item.get("scientific_label") or "")
        status = str(item.get("status") or "")
        method = str((item.get("provenance") or {}).get("method") or "")
        signals = item.get("signals") or {}
        visual_ids = [str(value) for value in item.get("visual_node_ids") or [] if str(value)]
        for visual_id in visual_ids:
            claims_by_visual.setdefault(visual_id, []).append({
                "method": method,
                "scientific_node_id": scientific_id,
                "scientific_label": scientific_label,
                "score": round(float(item.get("score") or 0.0), 6),
                "string": round(float(signals.get("string") or 0.0), 6),
                "semantic": signals.get("semantic"),
                "priority": _claim_tier(status, method),
                "accepted": True,
            })
        for row in item.get("candidates") or []:
            visual_id = str(row.get("visual_node_id") or "")
            if not visual_id:
                continue
            score = round(float(row.get("score") or 0.0), 6)
            candidates_by_visual.setdefault(visual_id, []).append({
                "scientific_node_id": scientific_id,
                "scientific_label": scientific_label,
                "score": score,
                "string": round(float(row.get("string") or 0.0), 6),
                "semantic": row.get("semantic"),
                "reason": _candidate_reason(score, bool(visual_ids)),
            })
    conclusions: dict[str, str] = {}
    objects: list[dict[str, Any]] = []
    groups = local_repeat_groups(visual_nodes)
    group_of = {visual_id: key for key, ids in groups.items() for visual_id in ids}
    roles: dict[str, str | None] = {}
    owners: dict[str, str] = {}
    annotations_by_owner: dict[str, list[dict[str, Any]]] = {}
    for visual_id in sorted(visual_nodes):
        node = visual_nodes[visual_id]
        if str(node.get("kind") or "") not in BINDABLE_KINDS:
            continue
        role = reader_object_role(node)
        if role is None and not claims_by_visual.get(visual_id):
            own_label = str(node.get("label") or "")
            own_declaration = _declared_outside_match(own_label, declared) if normalize_label(own_label) else None
            if own_declaration is None:
                role = local_object_role(node)
        roles[visual_id] = role
        if role != ROLE_SYMBOL:
            continue
        glyph = str(node.get("label") or "").strip()
        if not glyph:
            continue
        owner = owning_block(visual_id, visual_nodes)
        owners[visual_id] = owner
        if owner:
            annotations_by_owner.setdefault(owner, []).append({
                "glyph": glyph,
                "meaning": "",
                "object_id": visual_id,
            })
    provenance_searched = 0
    provenance_found = 0
    for visual_id in sorted(visual_nodes):
        node = visual_nodes[visual_id]
        kind = str(node.get("kind") or "")
        object_kind = {BLOCK_KIND: CLAIM_BLOCK, PANEL_KIND: CLAIM_ROW_LABEL, NOTE_REGION_KIND: CLAIM_NOTE_REGION}.get(kind, CLAIM_OTHER)
        label = str(node.get("label") or "")
        role = roles.get(visual_id)
        claims = sorted(
            claims_by_visual.get(visual_id, []),
            key=lambda item: (item["priority"], -item["score"], item["scientific_node_id"]),
        )
        declaration = _declared_outside_match(label, declared) if normalize_label(label) else None
        annotation_declaration = (
            _declared_outside_match(label, declared_annotations) if normalize_label(label) else None
        )
        candidates: list[dict[str, Any]] = []
        needs_review = False
        provenance_hit: dict[str, Any] | None = None
        if kind == NOTE_REGION_KIND:
            # The reading put a legend, key or scale into its own region: whatever it
            # names is the figure explaining itself, never a step of the paper's method.
            conclusion = CLAIM_FIGURE_ANNOTATION
            role = ROLE_STAGE_ANNOTATION
            claims = [{
                "method": NOTE_REGION_METHOD,
                "accepted": True,
                "priority": CLAIM_TIER_DECLARED,
                "note": "the reading filed this object as a region of the figure's own notation",
            }]
        elif claims:
            conclusion = CLAIM_BOUND
            best = claims[0]["priority"]
            for item in claims:
                item["accepted"] = item["priority"] == best
            if len({item["priority"] for item in claims}) > 1:
                needs_review = True
            if declaration is not None:
                needs_review = True
                claims.append({
                    "method": DECLARED_OUTSIDE_METHOD,
                    "accepted": False,
                    "priority": CLAIM_TIER_DECLARED,
                    "label": declaration["label"],
                    "where": declaration["where"],
                })
            if role in ROLE_EXCLUDED_FROM_MEANING:
                needs_review = True
        elif annotation_declaration is not None:
            # The review read this object as one of the figure's own annotations. A concept
            # that claims the object outranks this, so it only decides an unclaimed object.
            conclusion = CLAIM_FIGURE_ANNOTATION
            claims = [{
                "method": DECLARED_ANNOTATION_METHOD,
                "accepted": True,
                "priority": CLAIM_TIER_DECLARED,
                "label": annotation_declaration["label"],
                "explains": annotation_declaration["explains"],
                "where": annotation_declaration["where"],
            }]
            needs_review = declaration is not None
        elif declaration is not None:
            conclusion = CLAIM_OUTSIDE_PAPER_METHOD
            claims = [{
                "method": DECLARED_OUTSIDE_METHOD,
                "accepted": True,
                "priority": CLAIM_TIER_DECLARED,
                "label": declaration["label"],
                "where": declaration["where"],
            }]
        elif role in ROLE_FIGURE_NOTATION:
            conclusion = CLAIM_FIGURE_ANNOTATION
            claims = [{
                "method": READER_ROLE_METHOD if reader_object_role(node) else LOCAL_ROLE_METHOD,
                "accepted": True,
                "priority": CLAIM_TIER_DECLARED,
                "role": role,
                "note": "the object carries the figure's own notation, not a method concept",
            }]
        elif role == ROLE_EXTERNAL_METHOD:
            conclusion = CLAIM_OUTSIDE_PAPER_METHOD
            claims = [{
                "method": READER_ROLE_METHOD if reader_object_role(node) else LOCAL_ROLE_METHOD,
                "accepted": True,
                "priority": CLAIM_TIER_DECLARED,
                "role": role,
                "note": "the reading filed this object under another paper's method",
            }]
        else:
            if provenance is not None and normalize_label(label) and not object_is_unlabeled(node):
                provenance_searched += 1
                provenance_hit = provenance(label)
            if provenance_hit is not None:
                conclusion = CLAIM_IN_PAPER_NOT_IN_METHOD
                provenance_found += 1
                claims = [{
                    "method": PROVENANCE_METHOD,
                    "accepted": True,
                    "priority": CLAIM_TIER_OWN_LABEL,
                    "page": provenance_hit.get("page"),
                    "matches": provenance_hit.get("matches"),
                }]
            else:
                conclusion = CLAIM_UNRESOLVED
                candidates = sorted(
                    candidates_by_visual.get(visual_id, []),
                    key=lambda item: (-item["score"], item["scientific_node_id"]),
                )[:MAX_CANDIDATES]
                if object_is_unlabeled(node):
                    # Our own reading produced no text for this object, which says nothing
                    # about what the figure drew. It stays counted as content, and the gap
                    # is named instead of being quietly filed away as decoration.
                    needs_review = True
                    claims = [{
                        "method": UNREADABLE_LABEL_METHOD,
                        "accepted": True,
                        "priority": CLAIM_TIER_DECLARED,
                        "note": "the reading produced no text for this object, so nothing here says what the figure drew",
                    }]
        repeat = reader_repeat_of(node)
        key = group_of.get(visual_id)
        if repeat is not None:
            local_size = len(groups.get(str(repeat.get("key") or ""), [])) or 1
            repeat = {**repeat, "source": "reading", "local_size": local_size}
            if isinstance(repeat.get("count"), int) and repeat["count"] != local_size:
                needs_review = True
        elif key:
            repeat = {
                "key": key,
                "index": groups[key].index(visual_id) + 1,
                "count": len(groups[key]),
                "source": "local_labels",
                "local_size": len(groups[key]),
            }
        else:
            repeat = None
        conclusions[visual_id] = conclusion
        objects.append({
            "object_id": visual_id,
            "kind": object_kind,
            "visual_kind": kind,
            "label": label,
            "conclusion": conclusion,
            "role": role,
            "repeat_group": repeat,
            "belongs_to": owners.get(visual_id, ""),
            "annotations": annotations_by_owner.get(visual_id, [])[:MAX_ANNOTATION_RECORDS],
            "provenance": provenance_hit,
            "claims": claims,
            "candidates": candidates,
            "needs_review": needs_review,
        })
    relations_of_edge: dict[str, dict[str, Any]] = {}
    for item in edge_bindings:
        for visual_edge_id in item.get("visual_edge_ids") or []:
            relations_of_edge[str(visual_edge_id)] = {
                "scientific_edge_id": str(item.get("scientific_edge_id") or ""),
                "direction": str(item.get("direction") or ""),
            }
    for edge_id in sorted(visual_edges):
        edge = visual_edges[edge_id]
        source_id = str(edge.get("source") or "")
        target_id = str(edge.get("target") or "")
        ends = [
            _endpoint_report("source", source_id, conclusions, claims_by_visual),
            _endpoint_report("target", target_id, conclusions, claims_by_visual),
        ]
        end_conclusions = {end["conclusion"] for end in ends}
        claims = []
        candidates = []
        if end_conclusions == {CLAIM_BOUND}:
            conclusion = CLAIM_BOUND
            relation = relations_of_edge.get(edge_id)
            if relation is not None and relation["scientific_edge_id"]:
                direction = relation["direction"] or "forward"
                claims = [{
                    "method": ENDPOINT_MATCH_METHOD,
                    "accepted": direction == "forward",
                    "priority": CLAIM_TIER_OWN_LABEL,
                    "scientific_edge_id": relation["scientific_edge_id"],
                    "direction": direction,
                }]
                needs_review = direction != "forward"
            else:
                claims = [{
                    "method": ENDPOINT_MATCH_METHOD,
                    "accepted": False,
                    "priority": CLAIM_TIER_OWN_LABEL,
                    "scientific_edge_id": "",
                    "direction": "",
                    "note": "both ends answer to concepts but no scientific relation connects them",
                }]
                needs_review = True
        elif CLAIM_UNRESOLVED in end_conclusions or "missing" in end_conclusions:
            conclusion = CLAIM_UNRESOLVED
            candidates = _arrow_candidates(ends, candidates_by_visual)
            needs_review = False
        elif CLAIM_FIGURE_ANNOTATION in end_conclusions:
            # An arrow that touches a legend, a formula or a scale explains the notation
            # the figure prints about itself; it is the figure's own annotation layer, not
            # a step of the paper's method.
            conclusion = CLAIM_FIGURE_ANNOTATION
            claims = [{
                "method": ENDPOINT_ANNOTATION_METHOD,
                "accepted": True,
                "priority": CLAIM_TIER_DECLARED,
                "visual_node_ids": [source_id, target_id],
            }]
            needs_review = False
        else:
            conclusion = CLAIM_OUTSIDE_PAPER_METHOD
            claims = [{
                "method": ENDPOINT_OUTSIDE_METHOD,
                "accepted": True,
                "priority": CLAIM_TIER_DECLARED,
                "visual_node_ids": [source_id, target_id],
            }]
            needs_review = False
        conclusions[edge_id] = conclusion
        objects.append({
            "object_id": edge_id,
            "kind": CLAIM_ARROW,
            "visual_kind": "edge",
            "label": str((edge.get("properties") or {}).get("label") or ""),
            "conclusion": conclusion,
            "role": None,
            "repeat_group": None,
            "belongs_to": "",
            "annotations": [],
            "provenance": None,
            "claims": claims,
            "candidates": candidates,
            "endpoints": ends,
            "needs_review": needs_review,
        })

    def counted(kind_value: str, conclusion_value: str | None = None) -> int:
        return sum(
            1
            for item in objects
            if item["kind"] == kind_value and (conclusion_value is None or item["conclusion"] == conclusion_value)
        )

    total = len(objects)
    bound = counted(CLAIM_BLOCK, CLAIM_BOUND) + counted(CLAIM_ROW_LABEL, CLAIM_BOUND) + counted(CLAIM_ARROW, CLAIM_BOUND) + counted(CLAIM_OTHER, CLAIM_BOUND)
    outside = (
        counted(CLAIM_BLOCK, CLAIM_OUTSIDE_PAPER_METHOD)
        + counted(CLAIM_ROW_LABEL, CLAIM_OUTSIDE_PAPER_METHOD)
        + counted(CLAIM_ARROW, CLAIM_OUTSIDE_PAPER_METHOD)
        + counted(CLAIM_OTHER, CLAIM_OUTSIDE_PAPER_METHOD)
    )
    provenance_rows = (
        counted(CLAIM_BLOCK, CLAIM_IN_PAPER_NOT_IN_METHOD)
        + counted(CLAIM_ROW_LABEL, CLAIM_IN_PAPER_NOT_IN_METHOD)
        + counted(CLAIM_ARROW, CLAIM_IN_PAPER_NOT_IN_METHOD)
        + counted(CLAIM_OTHER, CLAIM_IN_PAPER_NOT_IN_METHOD)
    )
    annotation = (
        counted(CLAIM_BLOCK, CLAIM_FIGURE_ANNOTATION)
        + counted(CLAIM_ROW_LABEL, CLAIM_FIGURE_ANNOTATION)
        + counted(CLAIM_ARROW, CLAIM_FIGURE_ANNOTATION)
        + counted(CLAIM_OTHER, CLAIM_FIGURE_ANNOTATION)
    )
    # A region of the figure's own notation is excluded by name: it is neither a concept
    # nor a drawing of one, so counting it as an object would inflate the denominator.
    meaningful = [
        item
        for item in objects
        if item.get("role") not in ROLE_EXCLUDED_FROM_MEANING and item.get("kind") != CLAIM_NOTE_REGION
    ]
    resolved_meaningful = sum(1 for item in meaningful if item["conclusion"] != CLAIM_UNRESOLVED)
    annotatable = [item for item in objects if item.get("role") == ROLE_SYMBOL and str(item.get("label") or "").strip()]
    attached = sum(1 for item in annotatable if str(item.get("belongs_to") or ""))
    summary = {
        "objects": total,
        "blocks": counted(CLAIM_BLOCK),
        "row_labels": counted(CLAIM_ROW_LABEL),
        "arrows": counted(CLAIM_ARROW),
        "other": counted(CLAIM_OTHER),
        "notation_regions": counted(CLAIM_NOTE_REGION),
        "bound": bound,
        "outside_paper_method": outside,
        "in_paper_not_in_method": provenance_rows,
        "figure_annotation": annotation,
        "unresolved": total - bound - outside - annotation - provenance_rows,
        "claim_coverage": round(resolved_meaningful / len(meaningful), 4) if meaningful else 1.0,
        "diagnostic_claim_coverage_all_objects": round((bound + outside) / total, 4) if total else 1.0,
        "meaningful_objects": len(meaningful),
        "resolved_meaningful": resolved_meaningful,
        "excluded_from_meaning": total - len(meaningful),
        "roles": {role: sum(1 for item in objects if item.get("role") == role) for role in ROLES},
        "repeat_groups": len(groups),
        "repeat_instances": sum(len(ids) for ids in groups.values()),
        "repeat_groups_from_reading": sum(
            1 for item in objects if str((item.get("repeat_group") or {}).get("source") or "") == "reading"
        ),
        "annotation_coverage": round(attached / len(annotatable), 4) if annotatable else 1.0,
        "annotations_attached": attached,
        "annotations_unanchored": len(annotatable) - attached,
        "provenance_searched": provenance_searched,
        "provenance_found": provenance_found,
        "provenance_coverage": round(provenance_found / provenance_searched, 4) if provenance_searched else 0.0,
        "unlabeled_objects": sum(
            1
            for item in objects
            if item.get("kind") in (CLAIM_BLOCK, CLAIM_ROW_LABEL)
            and object_is_unlabeled(visual_nodes.get(str(item.get("object_id")) or "") or {})
        ),
        "bound_blocks": counted(CLAIM_BLOCK, CLAIM_BOUND),
        "outside_blocks": counted(CLAIM_BLOCK, CLAIM_OUTSIDE_PAPER_METHOD),
        "unresolved_blocks": counted(CLAIM_BLOCK, CLAIM_UNRESOLVED),
        "bound_row_labels": counted(CLAIM_ROW_LABEL, CLAIM_BOUND),
        "outside_row_labels": counted(CLAIM_ROW_LABEL, CLAIM_OUTSIDE_PAPER_METHOD),
        "unresolved_row_labels": counted(CLAIM_ROW_LABEL, CLAIM_UNRESOLVED),
        "bound_arrows": counted(CLAIM_ARROW, CLAIM_BOUND),
        "outside_arrows": counted(CLAIM_ARROW, CLAIM_OUTSIDE_PAPER_METHOD),
        "annotation_arrows": counted(CLAIM_ARROW, CLAIM_FIGURE_ANNOTATION),
        "unresolved_arrows": counted(CLAIM_ARROW, CLAIM_UNRESOLVED),
        "needs_review": sum(1 for item in objects if item["needs_review"]),
        "declared_outside_labels": len(declared),
        "declared_region_labels": len(declared_annotations),
    }
    return {
        "contract": FIGURE_CLAIMS_CONTRACT,
        "version": FIGURE_CLAIMS_VERSION,
        "paper_id": paper_id,
        "binding_version": BINDING_VERSION,
        "objects": objects,
        "summary": summary,
        "selection_samples": _selection_samples(
            scientific_graph,
            figure_text_review,
            node_bindings=node_bindings,
            block_ids=blocks,
            roles=roles,
        ),
    }


def _selection_summary(samples: dict[str, Any]) -> dict[str, Any]:
    """Return the counts of one selection sample set, without its per concept records."""
    return {key: value for key, value in samples.items() if key not in ("records", "records_truncated")}


def build_layer_bindings(
    scientific_graph: dict[str, Any],
    visual_graph: dict[str, Any],
    figure_text_review: dict[str, Any] | None = None,
    *,
    embedder: TextEmbedder | None = None,
    embedder_status: str = "off",
    embedder_error: str = "",
    provenance: Callable[[Any], dict[str, Any] | None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the cross-layer bindings of one paper plus their audit.

    Without a stored figure_text_review the label pass and the structural layer run on
    their own, so a bind that precedes its check behaves exactly as it always did.
    """
    paper_id = str(scientific_graph.get("paper_id") or visual_graph.get("paper_id") or "")
    scientific_graph_sha256 = graph_sha256(scientific_graph)
    visual_graph_sha256 = graph_sha256(visual_graph)
    scientific_nodes = _node_map(scientific_graph)
    visual_nodes = _node_map(visual_graph)
    semantic = _SemanticScorer(embedder, status=embedder_status, error=embedder_error)
    semantic.prepare(_binding_texts(scientific_nodes, visual_nodes, figure_text_review))
    label_matches, node_audit = match_layer_nodes(scientific_graph, visual_graph, semantic)
    check_matches, check_audit = match_layer_nodes_via_check(visual_graph, figure_text_review, semantic)
    # The stored review answers about the concept list its own check stage was asked
    # about. A concept the current layer no longer declares cannot be bound to
    # anything, so its answer is dropped here and counted instead of raising: the
    # check is a paid stage, and answers about a superseded list are the one input a
    # rebind cannot recompute for free.
    stale_fact_ids = sorted(set(check_matches) - set(scientific_nodes))
    for scientific_id in stale_fact_ids:
        check_matches.pop(scientific_id)
    if stale_fact_ids:
        check_audit["stale_fact_concepts"] = len(stale_fact_ids)
    label_matched_ids = set(label_matches) - set(check_matches)
    overrides: list[str] = []
    conflicts: list[dict[str, Any]] = []
    for scientific_id in sorted(set(label_matches) & set(check_matches)):
        label_visual = str(label_matches[scientific_id].get("visual_node_id") or "")
        fact_visual = str(check_matches[scientific_id].get("visual_node_id") or "")
        if label_visual == fact_visual:
            continue
        conflicts.append({
            "scientific_node_id": scientific_id,
            "label_visual_node_id": label_visual,
            "fact_visual_node_id": fact_visual,
        })
        if label_matches[scientific_id].get("strong"):
            overrides.append(scientific_id)
    node_matches = {**label_matches, **check_matches}
    extras = _second_instance_blocks(node_matches, scientific_nodes, visual_nodes, semantic)
    for scientific_id, extra_ids in extras.items():
        primary = str((node_matches.get(scientific_id) or {}).get("visual_node_id") or "")
        node_matches[scientific_id] = {**node_matches[scientific_id], "visual_node_ids": [primary, *extra_ids]}
    edge_matches, edge_audit = match_layer_edges(scientific_graph, visual_graph, node_matches)
    scientific_degrees = _degree_map(scientific_graph)
    visual_degrees = _degree_map(visual_graph)
    panels_present = _panel_present(visual_graph)
    node_bindings: list[dict[str, Any]] = []
    for scientific_id in sorted(scientific_nodes):
        match = node_matches.get(scientific_id)
        visual_id = str((match or {}).get("visual_node_id") or "")
        visual_ids = [visual_id, *extras.get(scientific_id, [])] if visual_id else []
        score = float(match["score"]) if match else 0.0
        if not match:
            status = "unbound"
        elif match.get("strong"):
            status = "bound"
        else:
            status = "weak"
        panel_ids: list[str] = []
        for bound_id in visual_ids:
            bound = visual_nodes.get(bound_id) or {}
            if str(bound.get("kind") or "") == PANEL_KIND:
                panel_ids.append(bound_id)
            panel_ids.extend(sorted(str(value) for value in bound.get("panel_ids") or []))
        violations = structure_violations(
            scientific_nodes[scientific_id],
            scientific_degrees,
            visual_ids,
            visual_nodes,
            visual_degrees,
            panels_present=panels_present,
        ) if visual_ids else []
        confidence = round(max(CONFIDENCE_FLOOR, score * (STRUCTURE_PENALTY ** len(violations))), 6) if visual_ids else 0.0
        if not match:
            method = LABEL_MATCH_METHOD
        elif scientific_id in check_matches:
            method = CHECK_MATCH_METHOD
        elif not match.get("strong") and float(match.get("semantic") or 0.0) > float(match.get("string") or 0.0):
            method = SEMANTIC_MATCH_METHOD
        else:
            method = LABEL_MATCH_METHOD
        method_provenance: dict[str, Any] = {
            "stage": "bind",
            "binding_version": BINDING_VERSION,
            "method": method,
        }
        if scientific_id in check_matches:
            method_provenance["check_confidence"] = check_matches[scientific_id].get("check_confidence")
        if scientific_id in overrides:
            method_provenance["reader_overrode_label_match"] = True
        node_bindings.append({
            "binding_id": stable_graph_id("node_binding", paper_id, scientific_id),
            "scientific_node_id": scientific_id,
            "visual_node_ids": list(visual_ids),
            "panel_ids": sorted(set(panel_ids)),
            "slot_ids": [],
            "port_ids": [],
            "ppt_object_ids": [],
            "status": status,
            "score": round(score, 6),
            "confidence": confidence,
            "needs_review": bool(violations) or (bool(visual_ids) and confidence < STRONG_MATCH),
            "signals": {
                "fact": scientific_id in check_matches,
                "string": round(float(match.get("string") or 0.0), 6) if match else 0.0,
                "semantic": (round(float(match.get("semantic") or 0.0), 6) if match and match.get("semantic") is not None else None),
                "structure": list(violations),
            },
            "candidates": _candidate_rows(scientific_nodes[scientific_id].get("label"), visual_nodes, semantic, accepted=visual_id),
            "scientific_label": str(scientific_nodes[scientific_id].get("label") or ""),
            "visual_label": str((visual_nodes.get(visual_id) or {}).get("label") or "") if visual_id else "",
            "provenance": method_provenance,
        })
    edge_bindings: list[dict[str, Any]] = []
    mismatch_ids = {str(item.get("scientific_edge_id") or "") for item in edge_audit["relation_type_mismatches"]}
    for scientific_id in sorted(_edge_map(scientific_graph)):
        match = edge_matches.get(scientific_id)
        reversed_here = bool(match) and match["direction"] == "reversed"
        mismatched = scientific_id in mismatch_ids
        confidence = 1.0
        if reversed_here:
            confidence *= 0.5
        if mismatched:
            confidence *= 0.7
        edge_bindings.append({
            "binding_id": stable_graph_id("edge_binding", paper_id, scientific_id),
            "scientific_edge_id": scientific_id,
            "visual_edge_ids": list(match["visual_edge_ids"]) if match else [],
            "connector_ids": [],
            "status": "bound" if match and match["direction"] == "forward" else ("reversed" if match else "unbound"),
            "direction": match["direction"] if match else None,
            "confidence": round(confidence, 6) if match else 0.0,
            "needs_review": bool(match) and (reversed_here or mismatched),
            "provenance": {"stage": "bind", "binding_version": BINDING_VERSION, "method": ENDPOINT_MATCH_METHOD},
        })
    usage = semantic.usage_report()
    bindings = normalize_graph_bindings({
        "binding_set_id": stable_graph_id("graph_bindings", paper_id),
        "paper_id": paper_id,
        "node_bindings": node_bindings,
        "edge_bindings": edge_bindings,
        "provenance": {
            "stage": "bind",
            "binding_version": BINDING_VERSION,
            "scientific_graph_id": scientific_graph.get("graph_id"),
            "visual_graph_id": visual_graph.get("graph_id"),
            "scientific_graph_sha256": scientific_graph_sha256,
            "visual_graph_sha256": visual_graph_sha256,
            "provider_calls": int(usage["calls"]),
            "check_supplied": figure_text_review is not None,
            "embedding": usage,
        },
    })
    validation = validate_graph_bindings(bindings, scientific_graph=scientific_graph, visual_graph=visual_graph)
    required_nodes = [str(value) for value in scientific_graph.get("required_node_ids") or []]
    required_edges = [str(value) for value in scientific_graph.get("required_edge_ids") or []]
    bound_nodes = {item["scientific_node_id"] for item in node_bindings if item["status"] == "bound"}
    bound_edges = {item["scientific_edge_id"] for item in edge_bindings if item["status"] == "bound"}
    statuses: dict[str, int] = {}
    for item in node_bindings:
        statuses[item["status"]] = statuses.get(item["status"], 0) + 1
    scores = [item["score"] for item in node_bindings if item["visual_node_ids"]]
    label_pass_bound = {scientific_id for scientific_id, match in label_matches.items() if match.get("strong")}
    semantic_weak = {
        scientific_id
        for scientific_id, match in label_matches.items()
        if not match.get("strong") and match.get("semantic") is not None and float(match.get("semantic") or 0.0) > float(match.get("string") or 0.0)
    }
    final_status = {item["scientific_node_id"]: item["status"] for item in node_bindings}
    figure_first = _figure_first_report(visual_graph, figure_text_review, node_bindings)
    # The claims payload travels back in the report and is stored as its own artifact
    # by the stage. The report copy carries the summary, so a stored report never
    # holds the whole object list of every paper.
    claims = claim_figure_objects(
        scientific_graph,
        visual_graph,
        figure_text_review,
        bindings=bindings,
        provenance=provenance,
    )
    claim_summary = claims["summary"]
    selection = _selection_summary(claims["selection_samples"])
    report = {
        "paper_id": paper_id,
        "ok": bool(validation["ok"]),
        "errors": list(validation["errors"]),
        "warnings": (
            [str(len(stale_fact_ids)) + " stored check answers name concepts the current scientific layer no longer declares"]
            if stale_fact_ids
            else []
        ),
        "binding_version": BINDING_VERSION,
        "stale_fact_nodes": len(stale_fact_ids),
        "figure_first": figure_first,
        "figure_claims": claims,
        "claim_bound_objects": int(claim_summary.get("bound") or 0),
        "claim_outside_objects": int(claim_summary.get("outside_paper_method") or 0),
        "claim_in_paper_not_in_method_objects": int(claim_summary.get("in_paper_not_in_method") or 0),
        "claim_figure_annotation_objects": int(claim_summary.get("figure_annotation") or 0),
        "claim_unresolved_objects": int(claim_summary.get("unresolved") or 0),
        "claim_meaningful_objects": int(claim_summary.get("meaningful_objects") or 0),
        "claim_coverage": claim_summary.get("claim_coverage"),
        "diagnostic_claim_coverage_all_objects": claim_summary.get("diagnostic_claim_coverage_all_objects"),
        "claim_roles": dict(claim_summary.get("roles") or {}),
        "repeat_groups": int(claim_summary.get("repeat_groups") or 0),
        "repeat_instances": int(claim_summary.get("repeat_instances") or 0),
        "annotation_coverage": claim_summary.get("annotation_coverage"),
        "annotations_attached": int(claim_summary.get("annotations_attached") or 0),
        "annotations_unanchored": int(claim_summary.get("annotations_unanchored") or 0),
        "provenance_searched": int(claim_summary.get("provenance_searched") or 0),
        "provenance_found": int(claim_summary.get("provenance_found") or 0),
        "provenance_coverage": claim_summary.get("provenance_coverage"),
        "selection_samples": selection,
        "node_count": len(node_bindings),
        "edge_count": len(edge_bindings),
        "node_statuses": statuses,
        "bound_nodes": len(bound_nodes),
        "bound_edges": len(bound_edges),
        "required_nodes": len(required_nodes),
        "required_edges": len(required_edges),
        "required_nodes_unbound": [value for value in required_nodes if value not in bound_nodes],
        "required_edges_unbound": [value for value in required_edges if value not in bound_edges],
        "diagnostic_required_node_coverage": 1.0 if not required_nodes else round(len(bound_nodes & set(required_nodes)) / len(required_nodes), 4),
        "diagnostic_required_edge_coverage": 1.0 if not required_edges else round(len(bound_edges & set(required_edges)) / len(required_edges), 4),
        "label_match_bound_nodes": len(bound_nodes & label_matched_ids),
        "image_label_bound_nodes": len(bound_nodes & set(check_matches)),
        "fact_bound_nodes": len(bound_nodes & set(check_matches)),
        "label_pass_bound_nodes": len(bound_nodes & label_pass_bound),
        "fact_override_nodes": len(bound_nodes & set(overrides)),
        "fact_vs_label_conflicts": len(conflicts),
        "fact_vs_label_conflict_examples": conflicts[:MAX_CONFLICT_EXAMPLES],
        "semantic_weak_nodes": sum(1 for scientific_id in semantic_weak if final_status.get(scientific_id) == "weak"),
        "semantic_ranked_nodes": int(check_audit.get("semantic_fact_nodes") or 0),
        "semantic_near_misses": int(node_audit.get("semantic_near_misses") or 0) + int(check_audit.get("semantic_fact_near_misses") or 0),
        "structure_conflict_nodes": sum(1 for item in node_bindings if item["signals"]["structure"]),
        "needs_review_nodes": sum(1 for item in node_bindings if item["needs_review"] and item["visual_node_ids"]),
        "multi_bound_nodes": sum(1 for item in node_bindings if len(item["visual_node_ids"]) > 1),
        "panel_bound_nodes": sum(
            1 for item in node_bindings
            if any(str((visual_nodes.get(visual_id) or {}).get("kind") or "") == PANEL_KIND for visual_id in item["visual_node_ids"])
        ),
        "embedding_status": usage["status"],
        "embedding_error": usage["error"],
        "embedding_fingerprint": usage["fingerprint"],
        "embedding_texts": int(usage["texts"]),
        "embedding_calls": int(usage["calls"]),
        "embedding_tokens": int(usage["total_tokens"]),
        "diagnostic_label_match_required_coverage": 1.0 if not required_nodes else round(len(bound_nodes & label_matched_ids & set(required_nodes)) / len(required_nodes), 4),
        "check_audit": dict(check_audit),
        "reversed_edges": list(edge_audit["reversed_edges"]),
        "relation_type_mismatches": list(edge_audit["relation_type_mismatches"]),
        "unlabeled_visual_blocks": list(node_audit["unlabeled_visual_blocks"]),
        "skipped_duplicate_blocks": list(node_audit["skipped_duplicate_blocks"]),
        "matched_visual_blocks": node_audit["matched_visual_blocks"],
        "visual_blocks": node_audit["visual_blocks"],
        "matched_visual_edges": edge_audit["matched_visual_edges"],
        "visual_edges": edge_audit["visual_edges"],
        "mean_binding_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "provider_calls": int(usage["calls"]),
    }
    if report["errors"]:
        return bindings, report
    return bindings, report


__all__ = [
    "BINDING_VERSION",
    "BLOCK_KIND",
    "CHECK_MATCH_METHOD",
    "CLAIM_ARROW",
    "CLAIM_BLOCK",
    "CLAIM_BOUND",
    "CLAIM_CONCLUSIONS",
    "CLAIM_FIGURE_ANNOTATION",
    "CLAIM_IN_PAPER_NOT_IN_METHOD",
    "CLAIM_NOTE_REGION",
    "CLAIM_OUTSIDE_PAPER_METHOD",
    "CLAIM_ROW_LABEL",
    "CLAIM_UNRESOLVED",
    "DECLARED_ANNOTATION_METHOD",
    "ENDPOINT_ANNOTATION_METHOD",
    "FIGURE_CLAIMS_CONTRACT",
    "FIGURE_CLAIMS_VERSION",
    "HOWS",
    "HOW_AS_BLOCK",
    "HOW_INSIDE_OTHER_GRAPHICS",
    "HOW_NOT_DRAWN",
    "LABEL_MATCH_METHOD",
    "LOCAL_ROLE_METHOD",
    "NOTE_REGION_KIND",
    "NOTE_REGION_METHOD",
    "PROVENANCE_METHOD",
    "READER_ROLE_METHOD",
    "ROLES",
    "ROLE_DECORATIVE",
    "ROLE_EXTERNAL_METHOD",
    "ROLE_METHOD_OBJECT",
    "ROLE_STAGE_ANNOTATION",
    "ROLE_SYMBOL",
    "SEMANTIC_MATCH_METHOD",
    "SEMANTIC_STRONG",
    "STRONG_MATCH",
    "UNREADABLE_LABEL_METHOD",
    "WEAK_MATCH",
    "build_layer_bindings",
    "claim_figure_objects",
    "graph_sha256",
    "label_score",
    "label_tokens",
    "local_object_role",
    "local_repeat_groups",
    "match_layer_edges",
    "match_layer_nodes",
    "match_layer_nodes_via_check",
    "normalize_label",
    "object_is_unlabeled",
    "owning_block",
    "reader_object_role",
    "reader_repeat_of",
    "resolve_binding_embedder",
    "structure_violations",
]
