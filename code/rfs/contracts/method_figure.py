"""Stable contract for paper-grounded method content shown in a figure."""

from __future__ import annotations

import re
from pathlib import Path

from typing import Any

from .priority import priority_number

METHOD_FIGURE_CONTRACT_VERSION = 2

#: The two files a middleware paper directory states its method contract in.
#:
#: They were renamed once: the figure specification became the paper method
#: inventory (what the paper states) and the method figure content became the
#: figure content plan (which of it the figure draws). A reader accepts both
#: names, so a directory written before the rename stays readable while a new
#: run writes the current one.
METHOD_INVENTORY_FILE = "paper_method_inventory.json"
LEGACY_INVENTORY_FILE = "figure_specification.json"
FIGURE_CONTENT_PLAN_FILE = "figure_content_plan.json"
LEGACY_PLAN_FILE = "method_figure_content.json"
INVENTORY_FILES = (METHOD_INVENTORY_FILE, LEGACY_INVENTORY_FILE)
PLAN_FILES = (FIGURE_CONTENT_PLAN_FILE, LEGACY_PLAN_FILE)


def method_inventory_path(directory: str | Path, *, plan_fallback: bool = True) -> Path | None:
    """Return the file a directory states the paper's method in, or ``None``.

    A directory holding only the plan is answered with the plan when ``plan_fallback``
    is set: the plan carries the same concepts plus the figure's own selection, so a
    reader that drops the selection reads the same thing either way.
    """
    root = Path(directory)
    for name in INVENTORY_FILES + (PLAN_FILES if plan_fallback else ()):
        path = root / name
        if path.is_file():
            return path
    return None


def figure_content_plan_path(directory: str | Path) -> Path | None:
    """Return the file a directory stores the figure content plan in, or ``None``."""
    root = Path(directory)
    for name in PLAN_FILES:
        path = root / name
        if path.is_file():
            return path
    return None


_DETAIL_LEVEL_WORDS = (
    ("stage", ("stage", "phase", "step", "stage_", "macro", "overview", "block")),
    ("mechanism", ("mechanism", "process", "operation", "algorithm", "procedure", "method", "technique", "policy", "loss", "objective", "training")),
    ("artifact", ("artifact", "data", "dataset", "result", "output", "metric", "state", "embedding", "table", "figure", "parameter", "hyperparameter", "prompt", "score")),
    ("module", ("module", "component", "unit", "sub", "detail", "head", "layer", "encoder", "decoder", "network", "model", "agent", "tool")),
)


def _detail_level_alias(value: str) -> str:
    """Read an off-contract ``detail_level`` as the closest legal level.

    The readers write words the contract never listed -- "submodule", "step",
    "head" -- and rejecting the whole method contract over the spelling threw
    away a figure the paper had already been understood for.
    """
    lowered = str(value or "").strip().casefold()
    for level, words in _DETAIL_LEVEL_WORDS:
        if any(word in lowered for word in words):
            return level
    return "module"


def _normalize_item(item: dict[str, Any], *, borrow_id_as_label: bool = True) -> dict[str, Any]:
    """Normalize one evidence-aware method object without inventing fields."""
    normalized = dict(item)
    normalized["id"] = str(normalized.get("id") or "").strip()
    label = normalized.get("label") or normalized.get("name")
    if not label and borrow_id_as_label:
        label = normalized["id"]
    normalized["label"] = str(label or "").strip()
    normalized["parent_id"] = str(normalized.get("parent_id") or normalized.get("parent_stage_id") or "").strip() or None
    detail_level = str(normalized.get("detail_level") or "module").strip().lower()
    detail_aliases = {
        "system": "stage",
        "pipeline": "stage",
        "framework": "stage",
        "component": "module",
        "operation": "mechanism",
        "state": "artifact",
        "data": "artifact",
        "result": "artifact",
    }
    resolved_detail_level = detail_aliases.get(detail_level) or _detail_level_alias(detail_level)
    if resolved_detail_level != detail_level:
        # The reading is kept next to the repair so a report can say which words
        # were read as what instead of silently flattening the hierarchy.
        normalized["detail_level_repaired_from"] = detail_level
    normalized["detail_level"] = resolved_detail_level
    normalized["scientific_importance"] = priority_number(normalized.get("scientific_importance") or normalized.get("importance"))
    normalized["visualization_value"] = priority_number(normalized.get("visualization_value") or normalized.get("visual_priority"))
    normalized["must_show"] = bool(normalized.get("must_show", False))
    normalized["should_render"] = bool(normalized.get("should_render", normalized.get("must_show", False)))
    normalized["render_priority"] = int(normalized.get("render_priority") or normalized.get("priority") or 2)
    normalized["evidence_ids"] = [str(value) for value in normalized.get("evidence_ids", []) if str(value).strip()] if isinstance(normalized.get("evidence_ids"), list) else []
    normalized["input_ids"] = [str(value) for value in normalized.get("input_ids", []) if str(value).strip()] if isinstance(normalized.get("input_ids"), list) else []
    normalized["output_ids"] = [str(value) for value in normalized.get("output_ids", []) if str(value).strip()] if isinstance(normalized.get("output_ids"), list) else []
    return normalized


def _relation_endpoints(relation: dict[str, Any]) -> tuple[str, str]:
    """Read one normalized relation's endpoints under both accepted key spellings."""
    source = str(relation.get("source") or relation.get("source_id") or "").strip()
    target = str(relation.get("target") or relation.get("target_id") or "").strip()
    return source, target


def _resolve_relation_references(items: list[dict[str, Any]], endpoints: dict[str, tuple[str, str]], field: str, index: int) -> None:
    """Resolve relation ids used as object ports in place.

    A provider sometimes names the relation instead of the object it connects.
    Because the relation is declared in the same contract, the reference still
    resolves deterministically: an input port becomes the relation's source and
    an output port its target. Unknown ids stay untouched so the validator can
    still report them.
    """
    for item in items:
        own_id = str(item.get("id") or "").strip()
        resolved: list[str] = []
        for value in item.get(field, []):
            declared = endpoints.get(value)
            candidate = declared[index] if declared is not None else value
            if candidate and candidate != own_id and candidate not in resolved:
                resolved.append(candidate)
        item[field] = resolved


def _name_key(item: dict[str, Any]) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(item.get("name") or "").casefold())


def _label_key(item: dict[str, Any]) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(item.get("label") or "").casefold())


def fold_alias_nodes(content: dict[str, Any]) -> dict[str, Any]:
    """Merge records that only restate an already-declared node.

    A method planner can name one mechanism twice: once as a module and again
    as the paper innovation. Keeping both drew two identical cards on the
    slide. The contract keeps the first declaration, unions the evidence and
    remaps every reference, so no scientific content is dropped.
    """

    fields = ("stages", "modules", "inputs", "outputs", "optional_details")
    by_name_parent: dict[tuple[str, str], dict[str, Any]] = {}
    by_label_parent: dict[tuple[str, str], dict[str, Any]] = {}
    for field in fields:
        for item in content.get(field) or []:
            if not isinstance(item, dict):
                continue
            name_key = _name_key(item)
            parent_key = str(item.get("parent_id") or "")
            if name_key and (name_key, parent_key) not in by_name_parent:
                by_name_parent[(name_key, parent_key)] = item
            label_key = _label_key(item)
            if label_key and (label_key, parent_key) not in by_label_parent:
                by_label_parent[(label_key, parent_key)] = item
    kept: list[dict[str, Any]] = []
    merges: list[dict[str, Any]] = []
    for item in content.get("innovations") or []:
        if not isinstance(item, dict):
            continue
        parent_key = str(item.get("parent_id") or "")
        target = by_name_parent.get((_name_key(item), parent_key)) or by_label_parent.get((_label_key(item), parent_key))
        alias_id = str(item.get("id") or "")
        target_id = str(target.get("id") or "") if isinstance(target, dict) else ""
        if not target_id or target_id == alias_id:
            kept.append(item)
            continue
        target["evidence_ids"] = list(dict.fromkeys([*(target.get("evidence_ids") or []), *(item.get("evidence_ids") or [])]))
        if _name_key(target) == _label_key(target) and len(str(item.get("name") or "")) > len(str(target.get("name") or "")):
            target["name"] = str(item.get("name"))
        if item.get("should_render"):
            target["should_render"] = True
        merges.append({"alias_id": alias_id, "merged_into": target_id, "reason": "restates_declared_node", "evidence_ids": list(item.get("evidence_ids") or [])})
    if not merges:
        return content
    content["innovations"] = kept
    remap = {item["alias_id"]: item["merged_into"] for item in merges if item["alias_id"]}
    for relation in content.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        for field in ("source", "target", "source_id", "target_id"):
            value = str(relation.get(field) or "")
            if value in remap:
                relation[field] = remap[value]
    for field in (*fields, "innovations"):
        for item in content.get(field) or []:
            if not isinstance(item, dict):
                continue
            for key in ("input_ids", "output_ids"):
                if isinstance(item.get(key), list):
                    item[key] = [remap.get(str(value), value) for value in item[key]]
    provenance = content.get("provenance") if isinstance(content.get("provenance"), dict) else {}
    content["provenance"] = {**provenance, "alias_merges": [*(provenance.get("alias_merges") or []), *merges]}
    return content


OBJECT_FIELDS = ("inputs", "outputs", "modules", "stages", "innovations", "optional_details")
OBJECT_DETAIL_LEVEL_RANK = {"stage": 0, "module": 1, "mechanism": 2, "artifact": 3}


def _object_id(item: dict[str, Any]) -> str:
    return str(item.get("id") or "").strip()


def _name_or_label_key(item: dict[str, Any]) -> str:
    return _name_key(item) or _label_key(item)


def _declares_same_node(kept: dict[str, Any], duplicate: dict[str, Any]) -> bool:
    """True when two declarations that share one id read as one node.

    A reader routinely states one node twice -- under ``modules`` and again
    under ``stages`` -- and the shared id is what says they are the same node.
    Measured over the delivered corpus every duplicate id was exactly that,
    with labels that are equal or one contained in the other.
    """
    kept_key = _name_or_label_key(kept)
    duplicate_key = _name_or_label_key(duplicate)
    if not kept_key or not duplicate_key:
        return True
    return kept_key == duplicate_key or kept_key in duplicate_key or duplicate_key in kept_key


def _merge_restated_node(kept: dict[str, Any], duplicate: dict[str, Any]) -> None:
    """Fold a second declaration of one node into the first without losing content."""
    kept["evidence_ids"] = list(dict.fromkeys([*(kept.get("evidence_ids") or []), *(duplicate.get("evidence_ids") or [])]))
    own_id = _object_id(kept)
    for key in ("input_ids", "output_ids"):
        merged = [value for value in dict.fromkeys([*(kept.get(key) or []), *(duplicate.get(key) or [])]) if value and value != own_id]
        kept[key] = merged
    for key in ("must_show", "should_render"):
        kept[key] = bool(kept.get(key) or duplicate.get(key))
    # The longer wording is the more explicit one; nothing is invented by it.
    for key in ("label", "name"):
        if len(str(duplicate.get(key) or "")) > len(str(kept.get(key) or "")):
            kept[key] = duplicate[key]
    kept_level = str(kept.get("detail_level") or "")
    duplicate_level = str(duplicate.get("detail_level") or "")
    if OBJECT_DETAIL_LEVEL_RANK.get(duplicate_level, 9) < OBJECT_DETAIL_LEVEL_RANK.get(kept_level, 9):
        kept["detail_level"] = duplicate_level


def _unique_object_id(base: str, taken: set[str]) -> str:
    index = 2
    while f"{base}_{index}" in taken:
        index += 1
    return f"{base}_{index}"


def _product_alias_key(value: Any) -> str:
    """Read a handoff name the way node names are read: punctuation and case blind."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _relation_carrying_product(relations: list[dict[str, Any]], value: str, own_id: str, field: str, resolved: dict[str, dict[str, Any]] | None = None) -> tuple[dict[str, Any], str] | None:
    """Find the arrow a handoff named in a port travels on, so its name is kept.

    A reader writes the name of a handoff -- ``solver_assignment``,
    ``reward_signal``, ``advantages`` -- into ``input_ids``/``output_ids``
    without declaring it as a node, because on the figure that handoff *is* the
    arrow. Removing the name threw away a reading the reader did state.

    The arrow is found in two strengths, and the caller records which one it
    was so nothing reads as verified that is not:

    * ``arrow_label`` / ``arrow_label_at_this_node`` -- the relation's own label
      says the same thing, which is the reader naming its own arrow.
    * ``sole_arrow_at_this_node`` -- no label matches, so the name goes on the
      only relation leaving the naming node (an output port) or the only one
      entering it (an input port). This is an inference from the reader's
      sentence, not a statement it made.
    * ``same_name_as_another_port`` -- the reader wrote this same name on a port
      that has already been settled -- one node's output and another's input is
      how a handoff is normally described -- so the second mention goes on the
      same relation. Ports are read in field order, so which mention settles
      first is fixed for a given contract.

    A self loop carries nothing, and a port with several candidate arrows keeps
    the old behaviour and stays reported as dropped.
    """
    key = _product_alias_key(value)
    if not key:
        return None
    settled = (resolved or {}).get(key)
    if settled is not None:
        return settled["relation"], "same_name_as_another_port"
    carriers = [item for item in relations if _relation_endpoints(item)[0] and _relation_endpoints(item)[0] != _relation_endpoints(item)[1]]
    labelled = [item for item in carriers if _product_alias_key(item.get("label")) == key]
    if len(labelled) == 1:
        return labelled[0], "arrow_label"
    attached = [item for item in carriers if _relation_endpoints(item)[0 if field == "output_ids" else 1] == own_id]
    if labelled:
        on_node = [item for item in labelled if item in attached]
        return (on_node[0], "arrow_label_at_this_node") if len(on_node) == 1 else None
    return (attached[0], "sole_arrow_at_this_node") if len(attached) == 1 else None


def repair_object_references(content: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Repair reference errors that are deterministic to repair, and record each one.

    Three families are repaired here because each of them was, by itself, the
    reason a finished deck was discarded:

    * one node declared under two fields, which the validator read as a
      duplicate id,
    * a ``parent_id`` no declared node owns,
    * an ``input_ids``/``output_ids`` entry no declared node owns,
    * an ``input_ids``/``output_ids`` entry that names a handoff the figure
      draws as an arrow, which is kept on that relation instead of thrown away.

    Repairs only remove a reference that cannot resolve or fold a restatement
    into the node it restates; no node, port or relation is invented. Anything
    else -- a missing id, an illegal ``detail_level``, an undeclared relation
    endpoint -- stays an error for the validator to report.
    """
    relations = [item for item in content.get("relations") or [] if isinstance(item, dict)]
    settled_handoffs: dict[str, dict[str, Any]] = {}
    stated: dict[str, tuple[str, dict[str, Any]]] = {}
    repairs: dict[str, list[dict[str, Any]]] = {"restatements": [], "renamed_ids": [], "dropped_parent_ids": [], "dropped_ports": [], "bound_product_aliases": []}
    for field in OBJECT_FIELDS:
        kept_items: list[dict[str, Any]] = []
        for item in content.get(field) or []:
            if not isinstance(item, dict):
                continue
            object_id = _object_id(item)
            previous = stated.get(object_id) if object_id else None
            if previous is None:
                if object_id:
                    stated[object_id] = (field, item)
                kept_items.append(item)
                continue
            kept_field, kept = previous
            if kept_field == field and not _declares_same_node(kept, item):
                renamed = _unique_object_id(object_id, set(stated))
                item["id"] = renamed
                stated[renamed] = (field, item)
                repairs["renamed_ids"].append({"id": object_id, "renamed_to": renamed, "field": field})
                kept_items.append(item)
                continue
            _merge_restated_node(kept, item)
            repairs["restatements"].append({"id": object_id, "dropped_from": field, "kept_in": kept_field, "label": str(kept.get("label") or "")})
        content[field] = kept_items
    known = set(stated)
    for field in OBJECT_FIELDS:
        for item in content.get(field) or []:
            if not isinstance(item, dict):
                continue
            own_id = _object_id(item)
            parent_id = str(item.get("parent_id") or "")
            if parent_id and parent_id not in known:
                item["parent_id"] = None
                repairs["dropped_parent_ids"].append({"id": own_id, "parent_id": parent_id})
            for key in ("input_ids", "output_ids"):
                values = [str(value) for value in item.get(key) or [] if str(value) and str(value) != own_id]
                kept_values = [value for value in values if value in known]
                still_dropped = []
                for value in values:
                    if value in known:
                        continue
                    found = _relation_carrying_product(relations, value, own_id, key, settled_handoffs)
                    if found is None:
                        still_dropped.append(value)
                        continue
                    relation, match = found
                    settled_handoffs.setdefault(_product_alias_key(value), {"relation": relation, "match": match})
                    carried = relation.setdefault("product_aliases", [])
                    if not any(isinstance(entry, dict) and str(entry.get("name")) == value for entry in carried):
                        carried.append({"name": value, "declared_by": own_id, "field": key, "match": match})
                    repairs["bound_product_aliases"].append({"id": own_id, "field": key, "product": value, "relation_id": str(relation.get("id") or ""), "match": match})
                if still_dropped:
                    repairs["dropped_ports"].append({"id": own_id, "field": key, "dropped": still_dropped})
                item[key] = kept_values
    return repairs


def normalize_method_figure_content(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}

    def records(key: str, *, borrow_id_as_label: bool = True) -> list[dict[str, Any]]:
        value = source.get(key)
        return [_normalize_item(item, borrow_id_as_label=borrow_id_as_label) for item in value if isinstance(item, dict)] if isinstance(value, list) else []
    # A relation id is a coordinate, not a word the paper printed. Letting it stand in
    # for a missing label is how "panelrel_i_04" reached the figure.
    relations = records("relations", borrow_id_as_label=False)
    content = {
        "contract": "rfs.method_figure_content",
        "version": int(source.get("version") or METHOD_FIGURE_CONTRACT_VERSION),
        "figure_goal": str(source.get("figure_goal") or "Explain the paper method."),
        "inputs": records("inputs"),
        "outputs": records("outputs"),
        "modules": records("modules"),
        "relations": relations,
        "stages": records("stages"),
        "innovations": records("innovations"),
        "optional_details": records("optional_details"),
        "forbidden_inventions": [str(item) for item in source.get("forbidden_inventions", []) if str(item).strip()] if isinstance(source.get("forbidden_inventions"), list) else [],
        "uncertainties": [str(item) for item in source.get("uncertainties", []) if str(item).strip()] if isinstance(source.get("uncertainties"), list) else [],
        "provenance": dict(source.get("provenance")) if isinstance(source.get("provenance"), dict) else {},
    }
    endpoints = {relation["id"]: _relation_endpoints(relation) for relation in relations if relation["id"]}
    endpoints = {key: value for key, value in endpoints.items() if value[0] and value[1]}
    if endpoints:
        for field, index in (("input_ids", 0), ("output_ids", 1)):
            for name in OBJECT_FIELDS:
                _resolve_relation_references(content[name], endpoints, field, index)
    # Alias folding runs first so the repairs below also see the references it
    # remaps, and the repairs stay recorded on the contract's provenance.
    folded = fold_alias_nodes(content)
    repairs = repair_object_references(folded)
    if any(repairs.values()):
        provenance = folded.get("provenance") if isinstance(folded.get("provenance"), dict) else {}
        folded["provenance"] = {**provenance, "normalization_repairs": repairs}
    return folded


def validate_method_figure_content(content: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_method_figure_content(content)
    errors: list[str] = []
    repairs: list[str] = []
    object_fields = ("modules", "inputs", "outputs", "stages", "innovations", "optional_details")
    all_objects = [item for field in object_fields for item in normalized[field]]
    ids = [str(item.get("id") or "").strip() for item in all_objects]
    if len([item for item in ids if item]) != len(set(item for item in ids if item)):
        errors.append("method figure node IDs must be unique")
    known = {item for item in ids if item}
    for index, item in enumerate(all_objects):
        if not item.get("id"):
            errors.append(f"method object[{index}] requires id")
        if item.get("detail_level") not in {"stage", "module", "mechanism", "artifact"}:
            errors.append(f"method object[{index}] has invalid detail_level")
        elif item.get("detail_level_repaired_from"):
            repairs.append(f"method object[{index}] ({item.get('id')}) reads detail_level {item['detail_level_repaired_from']!r} as {item['detail_level']!r}")
        parent_id = str(item.get("parent_id") or "")
        if parent_id and parent_id not in known:
            errors.append(f"method object[{index}] references unknown parent_id: {parent_id}")
        for field in ("input_ids", "output_ids"):
            unknown = [value for value in item.get(field, []) if value not in known]
            if unknown:
                errors.append(f"method object[{index}] references unknown {field}: {unknown[0]}")
    # What normalization repaired is reported next to what it could not, so a
    # salvaged contract says which words were read as what.
    stored = ((normalized.get("provenance") or {}).get("normalization_repairs") or {}) if isinstance(normalized.get("provenance"), dict) else {}
    for record in stored.get("restatements") or []:
        repairs.append(f"node {record.get('id')} was stated again under {record.get('dropped_from')}; kept the {record.get('kept_in')} declaration and merged its evidence")
    for record in stored.get("renamed_ids") or []:
        repairs.append(f"node {record.get('id')} was stated twice with different labels under {record.get('field')}; the second is drawn as {record.get('renamed_to')}")
    for record in stored.get("dropped_parent_ids") or []:
        repairs.append(f"node {record.get('id')} named an undeclared parent {record.get('parent_id')}; it is drawn at the top level")
    for record in stored.get("dropped_ports") or []:
        repairs.append(f"node {record.get('id')} named undeclared {record.get('field')} {record.get('dropped')}; the reference is dropped")
    for record in stored.get("bound_product_aliases") or []:
        repairs.append(f"node {record.get('id')} named the handoff {record.get('product')} in {record.get('field')}; it is kept as the product carried by relation {record.get('relation_id')} ({record.get('match')})")
    for index, relation in enumerate(normalized["relations"]):
        source = str(relation.get("source") or relation.get("source_id") or "").strip()
        target = str(relation.get("target") or relation.get("target_id") or "").strip()
        if not source or not target:
            errors.append(f"relation[{index}] requires source and target")
        elif known and (source not in known or target not in known):
            errors.append(f"relation[{index}] endpoint is not declared: {source}->{target}")
    if not normalized["modules"] and not normalized["inputs"] and not normalized["outputs"]:
        errors.append("method figure content requires at least one node")
    # An object the reader wants to draw without citing anything is still
    # drawable: it is listed as unsourced and the figure is delivered, rather
    # than voiding the run over a missing citation.
    evidence_pending = sorted(str(item.get("id")) for item in all_objects if item.get("should_render") and not item.get("evidence_ids"))
    if evidence_pending:
        repairs.append(f"{len(evidence_pending)} objects are drawn without an evidence id: {', '.join(evidence_pending[:6])}")
    return {
        "ok": not errors,
        "errors": errors,
        "repairs": repairs,
        "evidence_pending_ids": evidence_pending,
        "node_count": len(known),
        "relation_count": len(normalized["relations"]),
    }
