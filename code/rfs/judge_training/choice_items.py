"""Turn certified figure pairs into two-field multiple-choice Judge items.

The completion is only which figure is wrong and which defect it has. Scores stay in
``DEFECT_LABELS`` for the pipeline; they are not part of the text the model learns.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..utils import write_json
from .dpo_slices import exam_paper_ids
from .judge_schema import DEFECT_LABELS, GATE_DIAGNOSIS_TYPES

CHOICE_CONTRACT = "rfs.judge_choice_item"
FAMILIES = ("corruption", "rag", "sealed", "v002")
HARD_DEFECTS = GATE_DIAGNOSIS_TYPES
#: Diagnosis names the model may pick. ``assumed_worse`` is asked as ``no_defect``.
CHOICE_DEFECTS = tuple(sorted({
    *(str(item["diagnosis_type"]) for item in DEFECT_LABELS.values() if str(item["diagnosis_type"]) != "assumed_worse"),
    "no_defect",
}))
FAMILY_QUOTAS: dict[int, dict[str, int]] = {
    20: {"corruption": 7, "rag": 5, "sealed": 4, "v002": 4},
    200: {"corruption": 70, "rag": 50, "sealed": 40, "v002": 40},
    1000: {"corruption": 350, "rag": 250, "sealed": 200, "v002": 200},
    5000: {"corruption": 1750, "rag": 1250, "sealed": 1000, "v002": 1000},
}
VAL_QUOTAS = {"corruption": 12, "rag": 8, "sealed": 6, "v002": 6}
TYPE_CAP = 0.4
LEGAL_RATE_FLOOR = 0.5
#: Choice rows use these domain ids. Anything else, including ``unknown``, is omitted.
KNOWN_CHOICE_DOMAINS = frozenset({"agent", "biology", "cv", "materials", "rag"})


def choice_json(figure: str, defect: str) -> str:
    """Return the only completion this stage accepts."""

    return json.dumps({"wrong_figure": figure, "defect": defect}, ensure_ascii=False, separators=(",", ":"))


def menu_defect(raw: str) -> str:
    """Map an injected or reported type onto a menu id.

    ``assumed_worse`` has no named defect, so the correct option is ``no_defect``.
    Arrow injections that share one diagnosis collapse to ``wrong_arrow_target``.
    """

    key = str(raw or "").strip()
    if key in {"", "assumed_worse"}:
        return "no_defect" if key == "assumed_worse" else ""
    if key in DEFECT_LABELS:
        diagnosis = str(DEFECT_LABELS[key]["diagnosis_type"])
        return "no_defect" if diagnosis == "assumed_worse" else diagnosis
    if key in CHOICE_DEFECTS:
        return key
    return ""


def defect_menu(gold: str, pair_id: str) -> list[str]:
    """One correct type plus three stable distractors, in a stable order."""

    if gold not in CHOICE_DEFECTS:
        raise ValueError(f"gold defect is not on the menu: {gold}")
    pool = [item for item in CHOICE_DEFECTS if item != gold]
    rng = random.Random(int(hashlib.sha256(pair_id.encode("utf-8")).hexdigest()[:8], 16))
    picked = rng.sample(pool, 3)
    menu = [gold, *picked]
    rng.shuffle(menu)
    return menu


def parse_choice(text: str, menu: Sequence[str]) -> tuple[str, str] | None:
    """Accept one JSON object and nothing else. Extra prose is not an answer.

    The two fields the question requires have to be exactly right; anything else the same answer
    carries is passed over instead of making it illegal. The acceptance seat is asked a second
    question in the same answer, and an answer that takes the second one must not cost the
    comparison it was also asked for.
    """

    payload = choice_object(text)
    if payload is None:
        return None
    figure = payload.get("wrong_figure")
    defect = payload.get("defect")
    if figure not in {"A", "B"} or defect not in set(menu):
        return None
    return str(figure), str(defect)


def choice_object(text: str) -> dict[str, Any] | None:
    """The one JSON object an answer carries, or None when it carries none.

    Only an object is an answer: a code fence, prose around it or a bare list is not one. This is
    the reader every choice question shares, so a second question asked in the same answer cannot
    change how the first one is read.

    One thing is not prose: an answer that was cut off before it closed. The seat used to be asked
    for two fields and now may be asked for several, and a generation that runs out of tokens
    mid-string carries the two fields the comparison is scored on and then stops. Reading that as
    no answer at all would cost a round its comparison because it also took the second question, so
    an unterminated object is read for the two required fields and nothing else. A closed object
    with anything after it is still prose, and still not an answer.
    """

    raw = str(text or "").strip()
    if not raw or raw.startswith("```"):
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _cut_off_fields(raw)
    return payload if isinstance(payload, dict) else None


#: The two fields, read straight off an answer that never closed.
_FIELD_OF_ANSWER = {key: re.compile(r'"%s"\s*:\s*"([^"]*)"' % key) for key in ("wrong_figure", "defect")}


def _cut_off_fields(raw: str) -> dict[str, Any] | None:
    """The required fields of an object that was cut off before it closed, or None.

    A ``}`` anywhere in the text means the object did close, so whatever failed to parse is
    something else and is left alone.
    """

    if not raw.startswith("{") or "}" in raw:
        return None
    found: dict[str, Any] = {}
    for key, pattern in _FIELD_OF_ANSWER.items():
        match = pattern.search(raw)
        if match:
            found[key] = match.group(1)
    return found if len(found) == len(_FIELD_OF_ANSWER) else None


#: The two fields the pair question keeps requiring, and the ones the direction question adds.
CHOICE_FIELDS = ("wrong_figure", "defect")
DIRECTION_FIELDS = ("layer", "settled", "still_wrong", "next_layer", "because")

#: The completion the seat is told to write. The two required fields come first and are the two
#: the comparison is scored on, so a seat that stops after them is still read as an answer to the
#: comparison it was asked, and a seat that also answers the layer question is read for both.
CHOICE_REPLY = 'Reply with only {"wrong_figure":"A or B","defect":"one listed option"}. '
CHOICE_DIRECTION_REPLY = (
    'Reply with only {"wrong_figure":"A or B","defect":"one listed option","layer":"<the layer '
    'this round is on>","settled":true or false,"still_wrong":"<what is still wrong with that '
    'layer, empty once it is finished>","next_layer":"<the layer below it the defect belongs to, '
    'empty while it is not finished>","because":"<the reading or the thing you saw that says '
    'so>"}. '
)


def _truth(value: Any) -> bool | None:
    """Read a yes/no answer written as a boolean or as a word."""

    if isinstance(value, bool):
        return value
    word = str(value or "").strip().casefold()
    if word in {"true", "yes", "y", "1"}:
        return True
    if word in {"false", "no", "n", "0"}:
        return False
    return None


def parse_direction(payload: Mapping[str, Any], stages: Sequence[str] = ()) -> tuple[dict[str, Any], str]:
    """Read the direction block out of one answer, field by field.

    The order is the one the question asks in: whether the layer this round is on is finished
    with, and only then which layer below it the defect the figure still has belongs to. A layer
    is never read as finished because the answer is silent about it, and no field may throw the
    answer away: each is taken on its own and the ones that could not be read are named in the
    reason, so a round that got a partial answer keeps the part it got.
    """

    nested = payload.get("direction") if isinstance(payload.get("direction"), dict) else None
    source: Mapping[str, Any] = nested if nested else payload
    known = {str(item) for item in stages}
    layer = str(source.get("layer") or source.get("stage") or source.get("working_on") or "").strip()
    if known and layer not in known:
        layer = ""
    next_layer = str(source.get("next_layer") or source.get("next_working_on") or "").strip()
    if known and next_layer not in known:
        next_layer = ""
    still_wrong = str(source.get("still_wrong") or source.get("defect_here") or "").strip()
    because = str(source.get("because") or source.get("reading") or "").strip()
    settled = _truth(source.get("settled"))
    if settled is None and (still_wrong or next_layer):
        # The branch is written in the fields themselves; a missing flag is read from them.
        settled = bool(next_layer) and not still_wrong
    note = ""
    if settled is False and not still_wrong:
        note = "the answer did not say what is still wrong with the layer it is on"
    elif settled is True and not next_layer:
        note = "the answer did not say which layer below it the defect belongs to"
    return {
        "layer": layer,
        "settled": settled,
        "still_wrong": still_wrong,
        "next_layer": next_layer,
        "because": because,
    }, note


def parse_choice_answer(text: str, menu: Sequence[str], stages: Sequence[str] = ()) -> dict[str, Any]:
    """The pair answer plus the direction block, or an empty record when the pair is illegal.

    The pair is the whole answer when nothing asks about the layer, so an answer read here and one
    read by :func:`parse_choice` are the same answer whenever the second question was not asked.
    ``direction_problem`` says why the direction half is missing, and it is ``read`` when that
    half was answered in full.
    """

    payload = choice_object(text)
    if payload is None:
        return {}
    figure = payload.get("wrong_figure")
    defect = payload.get("defect")
    if figure not in {"A", "B"} or defect not in set(menu):
        return {}
    direction, note = parse_direction(payload, stages)
    answered = any(direction.get(key) not in (None, "") for key in DIRECTION_FIELDS)
    return {
        "wrong_figure": str(figure),
        "defect": str(defect),
        "direction": direction if answered else {},
        "direction_problem": (note or "read") if answered else "the answer did not take the direction question",
    }


def direction_question(layer: str, stages: Sequence[str]) -> str:
    """The second question, in the order a person asked it in: this layer first, then below it."""

    order = ", ".join(str(item) for item in stages)
    where = f"It is working on the layer {layer}"
    where += f", and the layers in order are: {order}. " if order else ". "
    return (
        f"Then say where this round stands. {where}"
        "Judge that layer before anything below it: if planning it again would not make the figure "
        "better it is finished with, so answer settled true and name in next_layer the layer below "
        "it that the defect the figure still has belongs to. If it is not good yet, answer settled "
        "false, leave next_layer empty and name in still_wrong what is still wrong with it, in the "
        "same words the defect menu uses. Working on a layer below an open one never needs this "
        "answer; going back over a finished one does."
    )


def choice_prompt(
    title: str,
    evidence: str,
    menu: Sequence[str],
    domain: str = "",
    *,
    rule: str = "",
    readings: str = "",
    layer: str = "",
    stages: Sequence[str] = (),
    direction: bool = False,
) -> str:
    """Ask for the two fields, and for the direction when there is a layer to ask about.

    A known domain is named and an unknown one is left out. ``rule`` is handed in rather than
    written here: it is the one trade a person has already said out loud, and the seat that
    compares two figures and the seat that measures them are told the same thing in the same
    words. ``readings`` is the loop's own measurement of both figures, labelled as what it is, so
    the seat may weigh it and is never asked to obey it. A caller that passes neither adds
    nothing, and the question renders exactly as it did before.
    """

    options = ", ".join(menu)
    name = str(domain or "").strip()
    domain_sentence = f"Domain: {name}. " if name in KNOWN_CHOICE_DOMAINS else ""
    parts = [
        "Look at figure A and figure B. Choose which figure is wrong and which defect it has. ",
        "wrong_figure must be A or B. ",
        f"defect must be one of: {options}. ",
        CHOICE_DIRECTION_REPLY if direction else CHOICE_REPLY,
        domain_sentence,
    ]
    if rule:
        parts.append(f"{rule} ")
    if readings:
        parts.append(f"Readings of the two figures, measured by the loop and evidence only, not a verdict: {readings} ")
    if direction:
        parts.append(direction_question(layer, stages) + " ")
    parts.append(f"Paper title: {title}\nPaper evidence: {evidence}")
    return "".join(parts)


def prompt_leaks_answer(prompt: str, gold: str, menu: Sequence[str]) -> bool:
    """Whether the gold defect is stated anywhere outside the option list."""

    options = ", ".join(menu)
    marker = f"defect must be one of: {options}."
    if marker not in prompt:
        return True
    body = prompt.replace(marker, "", 1)
    return gold in body or "注入" in prompt


def _paper_id(row: Mapping[str, Any]) -> str:
    paper = row.get("paper") if isinstance(row.get("paper"), dict) else {}
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return str(row.get("paper_id") or paper.get("paper_id") or provenance.get("source_arxiv_id") or "").strip()


def _diagnosis(row: Mapping[str, Any]) -> tuple[str, str]:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    diagnosis = labels.get("diagnosis") if isinstance(labels.get("diagnosis"), list) else []
    first = diagnosis[0] if diagnosis and isinstance(diagnosis[0], dict) else {}
    figure = str(first.get("figure") or "")
    raw = str(first.get("type") or "")
    if not raw:
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        raw = str(provenance.get("corruption_type") or "")
    if figure not in {"A", "B"}:
        preferred = str(labels.get("preferred_candidate") or "")
        figure = "B" if preferred == "A" else "A" if preferred == "B" else ""
    return figure, menu_defect(raw)


def _text_bits(row: Mapping[str, Any]) -> tuple[str, str]:
    paper = row.get("paper") if isinstance(row.get("paper"), dict) else {}
    title = str(paper.get("title") or _paper_id(row) or "untitled")
    abstract = str(paper.get("abstract") or "")
    return title[:240], abstract[:400]


def pair_from_example(row: Mapping[str, Any], family: str) -> dict[str, Any] | None:
    """Read one unswapped pairwise example. Swapped copies are rebuilt later."""

    if family not in (*FAMILIES, "exam"):
        raise ValueError(f"unknown choice family: {family}")
    example_id = str(row.get("example_id") or "")
    if example_id.endswith("__swap") or str(row.get("task_type") or "pairwise_preference") != "pairwise_preference":
        return None
    figure, defect = _diagnosis(row)
    image_a = str(row.get("candidate_a") or "")
    image_b = str(row.get("candidate_b") or "")
    if figure not in {"A", "B"} or not defect or not image_a or not image_b or image_a == image_b:
        return None
    if family == "sealed" and defect == "missing_core_entity":
        return None
    good, bad = (image_a, image_b) if figure == "B" else (image_b, image_a)
    title, evidence = _text_bits(row)
    pair_id = str(row.get("pair_group") or example_id)
    return {
        "pair_id": pair_id,
        "paper_id": _paper_id(row),
        "domain": str(row.get("domain") or "unknown"),
        "family": family,
        "defect": defect,
        "good": good,
        "bad": bad,
        "title": title,
        "evidence": evidence,
    }


def render_pair(pair: Mapping[str, Any], *, swapped: bool) -> dict[str, Any]:
    """Render one orientation. The swapped copy flips the figure and keeps the menu."""

    menu = defect_menu(str(pair["defect"]), str(pair["pair_id"]))
    figure = "A" if swapped else "B"
    images = [str(pair["bad"]), str(pair["good"])] if swapped else [str(pair["good"]), str(pair["bad"])]
    prompt = choice_prompt(str(pair["title"]), str(pair["evidence"]), menu, domain=str(pair.get("domain") or ""))
    distractor = next(item for item in menu if item != pair["defect"])
    other = "B" if figure == "A" else "A"
    suffix = "__swap" if swapped else ""
    content = [{"type": "text", "text": prompt}, {"type": "image", "image": images[0]}, {"type": "image", "image": images[1]}]
    return {
        "schema_version": CHOICE_CONTRACT,
        "example_id": f"{pair['pair_id']}{suffix}",
        "pair_id": str(pair["pair_id"]),
        "paper_id": str(pair["paper_id"]),
        "domain": str(pair["domain"]),
        "family": str(pair["family"]),
        # ``gold_in_A`` means the certified figure is the first image the model sees; the swapped
        # copy of the same pair must answer the other way, which is what the pair-order gate reads.
        "image_kind": "gold_in_A" if not swapped else "gold_in_B",
        "defect": str(pair["defect"]),
        "defect_menu": menu,
        "gold_figure": figure,
        "gold_defect": str(pair["defect"]),
        "images": images,
        "chosen": choice_json(figure, str(pair["defect"])),
        "rejected": choice_json(other, distractor),
        "messages": [{"role": "user", "content": content}],
    }


def preference_records(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split one rendered side into DPO rows that each change a single field.

    ``no_defect`` has no named scientific error, so it does not become a
    figure preference. A row that flips both the figure and the defect is not emitted.
    """

    chosen = json.loads(str(row["chosen"]))
    rejected = json.loads(str(row["rejected"]))
    figure = str(chosen["wrong_figure"])
    defect = str(chosen["defect"])
    other = str(rejected["wrong_figure"])
    distractor = str(rejected["defect"])
    records: list[dict[str, Any]] = []
    if defect != "no_defect":
        records.append(_preference_row(row, "figure", choice_json(other, defect)))
    records.append(_preference_row(row, "defect", choice_json(figure, distractor)))
    return records


def export_figure_preferences(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Write only the figure-swap DPO rows of a supervised choice JSONL.

    Each supervised side flips both the figure and the defect. This keeps the
    row that changes the figure and leaves the defect word alone, and drops
    ``no_defect`` sides, which have no figure preference.
    """

    written = 0
    skipped_no_figure = 0
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open(encoding="utf-8") as incoming, target.open("w", encoding="utf-8") as outgoing:
        for line in incoming:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("preference") or ""):
                raise ValueError("export_figure_preferences expects supervised rows, not rows that are already split")
            figure_rows = [item for item in preference_records(row) if item.get("preference") == "figure"]
            if not figure_rows:
                skipped_no_figure += 1
                continue
            for item in figure_rows:
                chosen = json.loads(str(item["chosen"]))
                rejected = json.loads(str(item["rejected"]))
                if chosen.get("defect") != rejected.get("defect") or chosen.get("wrong_figure") == rejected.get("wrong_figure"):
                    raise ValueError(f"figure preference must change only the figure: {item.get('example_id')}")
                outgoing.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
    return {"rows": written, "skipped_no_figure": skipped_no_figure, "path": str(target)}


def _preference_row(row: Mapping[str, Any], kind: str, rejected: str) -> dict[str, Any]:
    written = dict(row)
    written["example_id"] = f"{row['example_id']}__{kind}"
    written["preference"] = kind
    written["rejected"] = rejected
    return written


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.name.endswith(".partial.jsonl"):
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file or every finished JSONL under a directory."""

    root = Path(path)
    files = [root] if root.is_file() else sorted(item for item in root.rglob("*.jsonl") if item.is_file())
    rows: list[dict[str, Any]] = []
    for item in files:
        rows.extend(_read_jsonl(item))
    return rows


def load_family(path: str | Path, family: str) -> list[dict[str, Any]]:
    """Load one source and keep a single orientation of each pair.

    A corruption selection file already names the injected image, the train rank, and
    the holdout papers. Those ranks are the 20 / 200 / 1000 prefixes, so they are kept.
    """

    rows = load_examples(path)
    if family == "corruption" and rows and "injected_image" in rows[0]:
        return _load_corruption_selection(rows)
    seen: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for row in rows:
        pair = pair_from_example(row, family)
        if pair is None or not pair["paper_id"] or pair["pair_id"] in seen:
            continue
        seen.add(str(pair["pair_id"]))
        pairs.append(pair)
    return pairs


def _load_corruption_selection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        defect = menu_defect(str(row.get("defect") or row.get("corruption_type") or ""))
        paper_id = str(row.get("paper_id") or "").strip()
        good = str(row.get("source_image") or "")
        bad = str(row.get("injected_image") or "")
        pair_id = str(row.get("candidate_id") or f"{paper_id}-{defect}")
        if not defect or not paper_id or not good or not bad or pair_id in seen:
            continue
        seen.add(pair_id)
        pairs.append({
            "pair_id": pair_id,
            "paper_id": paper_id,
            "domain": str(row.get("domain") or "unknown"),
            "family": "corruption",
            "defect": defect,
            "good": good,
            "bad": bad,
            "title": str(row.get("title") or paper_id),
            "evidence": str(row.get("evidence") or ""),
            "role": str(row.get("role") or ""),
            "rank": int(row.get("rank") or 0),
        })
    return pairs


def _cap(count: int, fraction: float = TYPE_CAP) -> int:
    return int(count * fraction)


def _exceeds_cap(selected: Sequence[Mapping[str, Any]], key_fn, targets: Sequence[int]) -> bool:
    if not selected:
        return False
    last = selected[-1]
    for target in targets:
        if len(selected) > target:
            continue
        allowed = _cap(target)
        count = sum(1 for item in selected if key_fn(item) == key_fn(last))
        if count > allowed:
            return True
    return False


def stratified_order(items: Sequence[Mapping[str, Any]], targets: Sequence[int], key_fn) -> list[dict[str, Any]]:
    """Round-robin so every nested prefix stays within the per-type cap."""

    if not targets:
        return []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(key_fn(item))].append(dict(item))
    keys = sorted(groups)
    cursors = {key: 0 for key in keys}
    selected: list[dict[str, Any]] = []
    goal = targets[-1]
    while len(selected) < goal:
        progressed = False
        for key in keys:
            if len(selected) >= goal:
                break
            bucket = groups[key]
            cursor = cursors[key]
            if cursor >= len(bucket):
                continue
            trial = [*selected, bucket[cursor]]
            if _exceeds_cap(trial, key_fn, targets):
                continue
            selected.append(bucket[cursor])
            cursors[key] = cursor + 1
            progressed = True
        if not progressed:
            break
    return selected


def _ensure_no_defect(ordered: list[dict[str, Any]], pool: Sequence[Mapping[str, Any]], targets: Sequence[int]) -> list[dict[str, Any]]:
    """Keep at least one ``no_defect`` pair inside each v002 prefix that has room."""

    result = list(ordered)
    for target in targets:
        if target > len(result) or any(item.get("defect") == "no_defect" for item in result[:target]):
            continue
        later = next((index for index in range(target, len(result)) if result[index].get("defect") == "no_defect"), None)
        if later is not None:
            result[target - 1], result[later] = result[later], result[target - 1]
            continue
        spare = next((dict(item) for item in pool if item.get("defect") == "no_defect" and item["pair_id"] not in {row["pair_id"] for row in result}), None)
        if spare is not None:
            result[target - 1] = spare
    return result


def _block_papers(pairs: Sequence[Mapping[str, Any]], blocked: set[str]) -> list[dict[str, Any]]:
    return [dict(item) for item in pairs if str(item["paper_id"]) not in blocked]


def _order_family(items: Sequence[Mapping[str, Any]], targets: Sequence[int], *, family: str) -> list[dict[str, Any]]:
    if family == "corruption" and items and items[0].get("rank"):
        return [dict(item) for item in items]
    key_fn = (lambda item: item["domain"]) if family == "rag" else (lambda item: item["defect"])
    ordered = stratified_order(items, targets, key_fn)
    if family == "v002":
        ordered = _ensure_no_defect(ordered, items, targets)
    return ordered


def cut_choice_slices(
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    exam_papers: set[str] | None = None,
    sizes: Sequence[int] = (20, 200, 1000, 5000),
) -> dict[str, Any]:
    """Cut nested train slices and a 32-pair held-out set. Papers stay in one split."""

    wanted = tuple(sorted(int(size) for size in sizes))
    unknown = [size for size in wanted if size not in FAMILY_QUOTAS]
    if unknown:
        raise ValueError(f"no choice quota for sizes: {unknown}")
    exam = set(exam_papers or ())
    held: dict[str, list[dict[str, Any]]] = {}
    used: set[str] = set(exam)
    corruption = [dict(item) for item in (pools.get("corruption") or [])]
    assigned = [item for item in corruption if item.get("role")]
    families_for_val = FAMILIES
    if assigned:
        holdout = [item for item in assigned if item.get("role") == "holdout"]
        taken = stratified_order(holdout, [VAL_QUOTAS["corruption"]], lambda item: item["defect"])
        if len(taken) < VAL_QUOTAS["corruption"]:
            return {"ok": False, "reason": "held-out short for corruption", "have": len(taken), "need": VAL_QUOTAS["corruption"]}
        held["corruption"] = taken[: VAL_QUOTAS["corruption"]]
        holdout_papers = {str(item["paper_id"]) for item in holdout}
        train_rows = sorted((item for item in assigned if item.get("role") == "train"), key=lambda item: (int(item.get("rank") or 0), str(item["defect"]), str(item["paper_id"])))
        used.update(holdout_papers)
        reserved = set(used) | {str(item["paper_id"]) for item in train_rows}
        families_for_val = tuple(family for family in FAMILIES if family != "corruption")
        pools = {**pools, "corruption": train_rows}
    else:
        reserved = used
    for family in families_for_val:
        available = _block_papers(pools.get(family) or [], reserved)
        taken = stratified_order(available, [VAL_QUOTAS[family]], (lambda item: item["domain"]) if family == "rag" else (lambda item: item["defect"]))
        if len(taken) < VAL_QUOTAS[family]:
            return {"ok": False, "reason": f"held-out short for {family}", "have": len(taken), "need": VAL_QUOTAS[family]}
        held[family] = taken[: VAL_QUOTAS[family]]
        reserved.update(str(item["paper_id"]) for item in held[family])
    ordered: dict[str, list[dict[str, Any]]] = {}
    for family in FAMILIES:
        targets = [FAMILY_QUOTAS[size][family] for size in wanted]
        block = used if assigned and family == "corruption" else reserved
        available = _block_papers(pools.get(family) or [], block)
        ordered[family] = _order_family(available, targets, family=family)
    fitting = [size for size in wanted if all(len(ordered[family]) >= FAMILY_QUOTAS[size][family] for family in FAMILIES)]
    if not fitting:
        short = {family: len(ordered[family]) for family in FAMILIES}
        return {"ok": False, "reason": "no train size can be filled", "have": short}
    trains = {size: {family: ordered[family][: FAMILY_QUOTAS[size][family]] for family in FAMILIES} for size in fitting}
    val_pairs = [item for family in FAMILIES for item in held[family]]
    skipped = [size for size in wanted if size not in fitting]
    return {"ok": True, "sizes": fitting, "skipped_sizes": skipped, "train": trains, "val": val_pairs, "blocked_papers": sorted(used)}


def _write_rows(path: Path, pairs: Sequence[Mapping[str, Any]], *, swap: bool, preferences: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            for swapped in (False, True) if swap else (False,):
                rendered = render_pair(pair, swapped=swapped)
                rows = preference_records(rendered) if preferences else [rendered]
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_choice_slices(
    sources: Mapping[str, str],
    out_dir: str | Path,
    *,
    exam: str = "",
    sizes: Sequence[int] = (20, 200, 1000, 5000),
    seed: int = 0,
) -> dict[str, Any]:
    """Load the four sources, cut the nested slices, and write choice JSONL."""

    del seed  # order is hash-stable per pair; sampling does not use a global shuffle
    pools = {family: load_family(path, family) for family, path in sources.items() if path}
    if "sealed" in pools and "v002" in pools:
        sealed_papers = {str(item["paper_id"]) for item in pools["sealed"]}
        pools["v002"] = [item for item in pools["v002"] if str(item["paper_id"]) not in sealed_papers]
    exam_ids = exam_paper_ids(exam) if exam else set()
    cut = cut_choice_slices(pools, exam_papers=exam_ids, sizes=sizes)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    if not cut.get("ok"):
        write_json(dest / "choice_slice_report.json", cut)
        return cut
    for size, families in cut["train"].items():
        rows = [item for family in FAMILIES for item in families[family]]
        _write_rows(dest / f"train-{size}.jsonl", rows, swap=True, preferences=True)
    _write_rows(dest / "val-32.jsonl", cut["val"], swap=True)
    if exam:
        exam_pairs = []
        for row in load_examples(exam):
            pair = pair_from_example(row, "exam")
            if pair is not None:
                exam_pairs.append(pair)
        _write_rows(dest / "exam-choice.jsonl", exam_pairs, swap=False)
    report = {
        "ok": True,
        "contract": CHOICE_CONTRACT,
        "sizes": list(cut["sizes"]),
        "skipped_sizes": list(cut.get("skipped_sizes") or []),
        "val_pairs": len(cut["val"]),
        "train_pairs": {str(size): sum(len(items) for items in families.values()) for size, families in cut["train"].items()},
        "exam_papers": len(exam_ids),
    }
    write_json(dest / "choice_slice_report.json", report)
    return report


def choice_is_better(row: Mapping[str, Any], best: Mapping[str, Any] | None) -> bool:
    """No pick may fall, and at least one must rise. A collapsed legal rate cannot win.

    The held-out validator is small, so both picks frequently tie at the same
    coarse value. Requiring a strict rise in *both* then rejects a checkpoint
    that matches the incumbent and beats it on one pick, which is the ordinary
    way a later step wins; on a 32-pair set that degenerates into keeping the
    first checkpoint that lands on the high corner. Ties therefore fall through
    to the other pick instead of blocking, and an exact match still keeps the
    earlier step.
    """

    if float(row.get("legal_rate") or 0.0) < LEGAL_RATE_FLOOR:
        return False
    if best is None:
        return True
    figure = float(row.get("figure_accuracy") or 0.0)
    defect = float(row.get("defect_accuracy") or 0.0)
    best_figure = float(best.get("figure_accuracy") or 0.0)
    best_defect = float(best.get("defect_accuracy") or 0.0)
    return figure >= best_figure and defect >= best_defect and (figure > best_figure or defect > best_defect)


def _side_orientation(record: Mapping[str, Any]) -> str:
    """Which slot the certified figure sits in: ``gold_in_A`` or ``gold_in_B``.

    Assets rendered before the field existed carry an empty ``image_kind``, so the swapped
    copy is recognised by its ``__swap`` example id instead of by the missing field.
    """

    kind = str(record.get("image_kind") or "")
    if kind in {"gold_in_A", "gold_in_B"}:
        return kind
    example_id = str(record.get("example_id") or "")
    swapped = "__" in example_id and example_id.rpartition("__")[2].endswith("swap")
    return "gold_in_B" if swapped else "gold_in_A"


def evaluate_choice_records(
    policy: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int = 0,
    max_new_tokens: int = 64,
    predictions_out: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a short answer per row and score pairs only when every side is right.

    When ``predictions_out`` is given, the per-side parse is written as JSONL first, so a
    promotion gate can read the same run the numbers came from instead of re-scoring it.
    """

    if not callable(getattr(policy, "generate", None)):
        return {"skipped": True, "reason": "backend has no generate"}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("pair_id") or record.get("example_id"))].append(record)
    pair_ids = list(grouped)
    if limit > 0:
        pair_ids = pair_ids[:limit]
    legal_rows = 0
    rows = 0
    figure_hits = 0
    defect_hits = 0
    hard = {name: [0, 0] for name in HARD_DEFECTS}
    predictions: list[dict[str, Any]] = []
    for pair_id in pair_ids:
        sides = grouped[pair_id]
        figure_ok = True
        defect_ok = True
        gold_defect = str(sides[0].get("gold_defect") or "")
        for record in sides:
            rows += 1
            menu = list(record.get("defect_menu") or [])
            parsed = parse_choice(policy.generate(record, max_new_tokens=max_new_tokens), menu)
            predictions.append({
                "example_id": str(record.get("example_id") or ""),
                "pair_id": pair_id,
                "paper_id": str(record.get("paper_id") or ""),
                "domain": str(record.get("domain") or ""),
                "image_kind": _side_orientation(record),
                "gold_figure": str(record.get("gold_figure") or ""),
                "gold_defect": gold_defect,
                "legal": parsed is not None,
                "pred_figure": parsed[0] if parsed else "",
                "pred_defect": parsed[1] if parsed else "",
            })
            if parsed is None:
                figure_ok = False
                defect_ok = False
                continue
            legal_rows += 1
            figure, defect = parsed
            if figure != record.get("gold_figure"):
                figure_ok = False
            if defect != record.get("gold_defect"):
                defect_ok = False
        if gold_defect in hard:
            hard[gold_defect][1] += 1
            if defect_ok:
                hard[gold_defect][0] += 1
        figure_hits += 1 if figure_ok else 0
        defect_hits += 1 if defect_ok else 0
    total = len(pair_ids)
    if predictions_out is not None:
        target = Path(predictions_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "skipped": False,
        "pairs": total,
        "rows": rows,
        "legal_rate": round(legal_rows / rows, 6) if rows else 0.0,
        "figure_accuracy": round(figure_hits / total, 6) if total else 0.0,
        "defect_accuracy": round(defect_hits / total, 6) if total else 0.0,
        "hard_recall": {name: round(hits / seen, 6) if seen else None for name, (hits, seen) in hard.items()},
    }
