from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from ...utils import ensure_dir, write_json
from ..paper import normalize_quality_label
from .figure_kind import classify_figure_kind, is_framework_figure


def _reference_library_status(configured: Path | None, readable: str | Path | None) -> str:
    """Name which of four states a run is in.

    "Nothing was configured" and "a library was configured and turned out to hold no positive
    exemplar" are different defects -- the first is configuration, the second is data -- and a
    record that only says ``not_found`` cannot tell them apart.
    """

    if configured is None:
        return "not_configured"
    if not configured.exists():
        return "configured_missing"
    if not readable:
        return "configured_but_unread"
    return "selected"


class NoFrameworkExemplarError(ValueError):
    """A library holds positive exemplars, but none of them is a framework figure."""

    def __init__(self, message: str, *, excluded: dict[str, list[str]] | None = None) -> None:
        super().__init__(message)
        self.excluded = excluded or {}


def _exclusion_ledger(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Name every record that is not a framework figure, and the evidence that decided it."""

    ledger: list[dict[str, Any]] = []
    for record in records:
        verdict = classify_figure_kind(record)
        if verdict["kind"] == "framework":
            continue
        ledger.append(
            {
                "id": record.get("id"),
                "kind": verdict["kind"],
                "evidence": (verdict["data_evidence"] or verdict["framework_context"])[:4],
            }
        )
    return ledger


def _excluded_ids(ledger: list[dict[str, Any]]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {"data_panel": [], "unknown": []}
    for entry in ledger:
        grouped.setdefault(str(entry.get("kind")), []).append(entry.get("id"))
    return grouped


def resolve_project_exemplars(*, root: str | Path, paper_review: dict[str, Any], selected_template: dict[str, Any], reference_library: str | Path | None, top_k: int, configured_library: str | Path | None = None) -> dict[str, Any] | None:
    """Select project-level curated exemplars as visual-only guidance."""

    root = Path(root)
    configured = Path(configured_library) if configured_library else (Path(reference_library) if reference_library else None)
    resolution: dict[str, Any] = {
        "status": _reference_library_status(configured, reference_library),
        "source": str(configured) if configured else None,
        "selection": "explicit_or_project_default",
    }
    if not reference_library or configured is None or not configured.exists():
        write_json(root / "reference_library_resolution.json", resolution)
        return None
    try:
        selected = select_reference_exemplars(
            paper_review=paper_review,
            library=reference_library,
            out=root,
            selected_template=selected_template,
            top_k=top_k,
        )
    except NoFrameworkExemplarError as exc:
        resolution["status"] = "configured_no_framework_positive"
        resolution["detail"] = "the library holds positive exemplars, but none of them is a framework figure"
        resolution["excluded_by_kind"] = exc.excluded
        write_json(root / "reference_library_resolution.json", resolution)
        return None
    except ValueError as exc:
        if "positive exemplars" not in str(exc):
            raise
        resolution["status"] = "configured_empty"
        resolution["detail"] = "the library exists but holds no positive exemplar"
        write_json(root / "reference_library_resolution.json", resolution)
        return None
    resolution.update(
        {
            "status": "selected",
            "record_count": selected.get("record_count"),
            "positive_count": selected.get("positive_count"),
            "framework_positive_count": selected.get("framework_positive_count"),
            "excluded_by_kind": selected.get("excluded_by_kind"),
            "negative_count": selected.get("negative_count"),
        }
    )
    write_json(root / "reference_library_resolution.json", resolution)
    return selected


def _load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {source}")
    return data


def load_reference_library(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Reference library does not exist: {source}")
    records = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Reference library line {line_number} is not a JSON object")
        label = str(record.get("quality_label") or "").strip().casefold()
        label_map = {"优": "positive", "优秀": "positive", "good": "positive", "excellent": "positive", "正向": "positive", "差": "negative", "劣": "negative", "bad": "negative", "negative": "negative", "负向": "negative"}
        explicit_label_map = {"\u4f18": "positive", "\u4f18\u79c0": "positive", "\u6b63\u5411": "positive", "\u5dee": "negative", "\u574f": "negative", "\u8d1f\u5411": "negative"}
        record["quality_label"] = normalize_quality_label(explicit_label_map.get(label, label_map.get(label, label)))
        records.append(record)
    if not records:
        raise ValueError(f"Reference library is empty: {source}")
    return records


def _statements(review: dict[str, Any]) -> str:
    fields = ["research_questions", "central_claims", "modules", "relations", "innovations", "inputs", "outputs"]
    parts: list[str] = []
    for field in fields:
        for item in review.get(field, []) if isinstance(review.get(field), list) else []:
            if isinstance(item, dict):
                parts.append(str(item.get("statement") or item.get("visible_label") or item.get("name") or ""))
    workflows = review.get("workflows") if isinstance(review.get("workflows"), dict) else {}
    for items in workflows.values():
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                parts.append(str(item.get("statement") or ""))
    return " ".join(parts).lower()


def paper_features_from_review(review: dict[str, Any], selected_template: dict[str, Any] | None = None) -> dict[str, Any]:
    text = _statements(review)
    relation_types = {str(item.get("relation_type") or item.get("type") or "").lower() for item in review.get("relations", []) if isinstance(item, dict)}
    workflows = review.get("workflows") if isinstance(review.get("workflows"), dict) else {}
    module_count = len(review.get("modules", []) if isinstance(review.get("modules"), list) else [])
    desired_density = "very_high" if module_count >= 12 else "high" if module_count >= 6 else "medium"
    actual_roles = {
        str(item.get("visual_role") or item.get("role") or "").strip()
        for field in ("inputs", "modules", "outputs", "innovations")
        for item in (review.get(field, []) if isinstance(review.get(field), list) else [])
        if isinstance(item, dict) and str(item.get("visual_role") or item.get("role") or "").strip()
    }
    return {
        "domain_profile": review.get("domain_profile"),
        "module_count": module_count,
        "relation_count": len(review.get("relations", []) if isinstance(review.get("relations"), list) else []),
        "has_loop": bool(workflows.get("feedback")) or "feedback" in relation_types or any(token in text for token in ["iterative", "loop", "feedback"]),
        "has_tree": any(token in text for token in ["tree", "branch", "prune", "search"]),
        "has_multimodal": any(token in text for token in ["multimodal", "multi-modal", "image", "video", "audio", "document"]),
        "has_retrieval": any(token in text for token in ["retrieval", "retrieve", "index", "rag", "knowledge graph"]),
        "desired_density": desired_density,
        "color_role_count": max(1, len(actual_roles)),
        "selected_template_id": (selected_template or {}).get("template_id"),
        "selected_aspect_ratio": (selected_template or {}).get("source_aspect_ratio") or (selected_template or {}).get("aspect_ratio"),
    }


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if result > 0 else None
    except Exception:
        return None


def _record_text(record: dict[str, Any]) -> str:
    values = [
        record.get("archetype"),
        record.get("layout", {}).get("macro_pattern") if isinstance(record.get("layout"), dict) else "",
        record.get("density", {}).get("level") if isinstance(record.get("density"), dict) else "",
        " ".join(record.get("structure", {}).get("section_roles", []) if isinstance(record.get("structure"), dict) else []),
        " ".join(record.get("connectors", {}).get("grammar", []) if isinstance(record.get("connectors"), dict) else []),
    ]
    return " ".join(str(value or "") for value in values).lower()


def _score_record(record: dict[str, Any], features: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    archetype = str(record.get("archetype") or "")
    if archetype and archetype == features.get("selected_template_id"):
        score += 2.0
        reasons.append("selected_template_match")
    desired_ratio = _safe_float(features.get("selected_aspect_ratio"))
    record_ratio = _safe_float((record.get("image") or {}).get("aspect_ratio") if isinstance(record.get("image"), dict) else None)
    if desired_ratio and record_ratio:
        ratio_score = max(0.0, 1.0 - abs(record_ratio - desired_ratio) / max(desired_ratio, 0.1))
        score += ratio_score
        reasons.append(f"aspect_ratio_fit={ratio_score:.2f}")
    density = record.get("density") if isinstance(record.get("density"), dict) else {}
    if density.get("level") == features.get("desired_density"):
        score += 0.8
        reasons.append("density_match")
    text = _record_text(record)
    for feature, tokens in [
        ("has_loop", ["loop", "feedback", "curved", "iteration"]),
        ("has_tree", ["tree", "branch"]),
        ("has_multimodal", ["multimodal", "parallel", "multi", "modal"]),
        ("has_retrieval", ["retrieval", "retrieve", "index", "merge"]),
    ]:
        if features.get(feature) and any(token in text for token in tokens):
            score += 0.6
            reasons.append(feature)
    module_estimate = density.get("module_count_estimate")
    try:
        if module_estimate is not None and features["module_count"]:
            distance = abs(float(module_estimate) - float(features["module_count"]))
            module_score = max(0.0, 1.0 - distance / max(float(features["module_count"]), 1.0))
            score += 0.6 * module_score
            reasons.append(f"module_count_fit={module_score:.2f}")
    except Exception:
        pass
    recipe = record.get("color_recipe") if isinstance(record.get("color_recipe"), dict) else {}
    semantic_roles = {
        str(role)
        for color in recipe.get("colors", [])
        if isinstance(color, dict)
        for role in color.get("semantic_roles", [])
        if str(role).strip()
    }
    desired_role_count = int(features.get("color_role_count") or 1)
    if semantic_roles:
        role_fit = max(0.0, 1.0 - abs(len(semantic_roles) - desired_role_count) / max(desired_role_count, len(semantic_roles), 1))
        score += 0.25 * role_fit
        reasons.append(f"color_role_fit={role_fit:.2f}")
    if recipe.get("background_color_id"):
        score += 0.05
        reasons.append("explicit_background_recipe")
    return round(score, 4), reasons or ["fallback_order"]


def _unique(values: list[Any], limit: int = 12) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        if value in (None, "", [], {}) or key in seen:
            continue
        seen.add(key)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def _most_common(values: list[str], fallback: str) -> str:
    filtered = [value for value in values if value and value != "unknown"]
    if not filtered:
        return fallback
    return Counter(filtered).most_common(1)[0][0]


def _compact_color_recipe(record: dict[str, Any]) -> dict[str, Any] | None:
    """Keep transferable color evidence without paper-specific semantics."""
    recipe = record.get("color_recipe") if isinstance(record.get("color_recipe"), dict) else {}
    if str((recipe.get("analysis") or {}).get("semantic_mode") or "") != "vlm":
        return None
    colors = []
    for item in recipe.get("colors", []):
        if not isinstance(item, dict) or not item.get("color_id") or not item.get("hex"):
            continue
        colors.append({
            "color_id": str(item["color_id"]),
            "hex": str(item["hex"]),
            "area_ratio": round(float(item.get("area_ratio") or 0.0), 4),
            "spatial_zones": list(item.get("spatial_zones") or [])[:4],
            "used_for": list(item.get("used_for") or [])[:5],
            "semantic_roles": list(item.get("semantic_roles") or [])[:5],
        })
    if not colors:
        return None
    return {
        "recipe_id": str(recipe.get("recipe_id") or record.get("id") or ""),
        "background_color_id": str(recipe.get("background_color_id") or ""),
        "colors": colors,
        "relationships": recipe.get("relationships") if isinstance(recipe.get("relationships"), dict) else {},
        "transfer_rules": list(recipe.get("transfer_rules") or [])[:8],
        "avoid_copying": list(recipe.get("avoid_copying") or [])[:5],
    }


def build_exemplar_style_brief(selected: list[dict[str, Any]], negative: list[dict[str, Any]], library_path: str | Path) -> dict[str, Any]:
    positives = [item["record"] for item in selected]
    negatives = [item["record"] for item in negative]
    structure_experience = [record.get("structure_experience") for record in positives if isinstance(record.get("structure_experience"), dict)]
    negative_experience = [record.get("structure_experience") for record in negatives if isinstance(record.get("structure_experience"), dict)]
    library_file = Path(library_path)
    manifest_candidates = [
        library_file.with_name("reference_structure_experience_manifest.json"),
        library_file.with_name("reference_experience_manifest.json"),
    ]
    manifest = {}
    for candidate in manifest_candidates:
        if candidate.exists():
            try:
                manifest = _load_json(candidate)
            except Exception:
                manifest = {}
            break
    global_rules = {
        "composition_patterns": manifest.get("composition_patterns", []),
        "global_geometry_rules": manifest.get("global_geometry_rules", {}),
        "panel_sizing_rules": manifest.get("panel_sizing_rules", {}),
        "anti_patterns": manifest.get("anti_patterns", []),
        "evidence_references": manifest.get("evidence_references", []),
    }
    color_recipes = [recipe for record in positives if (recipe := _compact_color_recipe(record)) is not None][:3]
    return {
        "summary": "Reference exemplar style brief for direct structure and Icon planning.",
        "source_library": str(Path(library_path).resolve()),
        "selected_positive_ids": [record.get("id") for record in positives],
        "selected_negative_ids": [record.get("id") for record in negatives],
        "layout_rules": _unique([record.get("layout", {}).get("macro_pattern") for record in positives if isinstance(record.get("layout"), dict)] + [item for record in positives if isinstance(record.get("layout"), dict) for item in record.get("layout", {}).get("reading_order", [])]),
        "structure_rules": _unique([record.get("structure", {}).get("grouping_style") for record in positives if isinstance(record.get("structure"), dict)] + [item for record in positives if isinstance(record.get("structure"), dict) for item in record.get("structure", {}).get("section_roles", [])]),
        "density_target": {
            "level": _most_common([str(record.get("density", {}).get("level") or "") for record in positives if isinstance(record.get("density"), dict)], "unknown"),
            "whitespace_policy": _most_common([str(record.get("density", {}).get("whitespace_policy") or "") for record in positives if isinstance(record.get("density"), dict)], "unknown"),
        },
        "hierarchy_rules": _unique([record.get("hierarchy", {}).get("primary_focus") for record in positives if isinstance(record.get("hierarchy"), dict)] + [item for record in positives if isinstance(record.get("hierarchy"), dict) for item in record.get("hierarchy", {}).get("emphasis_methods", [])]),
        "color_rules": _unique([record.get("color", {}).get("contrast_policy") for record in positives if isinstance(record.get("color"), dict)]),
        "color_experience": {
            "selected_recipes": color_recipes,
            "shared_patterns": _unique([
                value
                for recipe in color_recipes
                for value in recipe.get("transfer_rules", [])
            ], limit=8),
            "candidate_harmonies": _unique([
                recipe.get("relationships", {}).get("harmony")
                for recipe in color_recipes
                if isinstance(recipe.get("relationships"), dict)
            ], limit=3),
        },
        "text_rules": _unique([record.get("text", {}).get("label_style") for record in positives if isinstance(record.get("text"), dict)] + [record.get("text", {}).get("title_style") for record in positives if isinstance(record.get("text"), dict)]),
        "connector_rules": _unique([item for record in positives if isinstance(record.get("connectors"), dict) for item in record.get("connectors", {}).get("grammar", [])]),
        "avoid_rules": _unique([record.get("layout", {}).get("macro_pattern") for record in negatives if isinstance(record.get("layout"), dict)] + [item for record in negatives if isinstance(record.get("text"), dict) for item in record.get("text", {}).get("forbidden", [])]),
        "anti_copy_rules": _unique([item for record in positives + negatives if isinstance(record.get("safety"), dict) for item in record.get("safety", {}).get("forbidden_copy_terms", [])] + [item for record in positives + negatives if isinstance(record.get("safety"), dict) for item in record.get("safety", {}).get("forbidden_copy_objects", [])]),
        "structure_experience": {
            "composition_patterns": structure_experience[:8],
            "anti_patterns": negative_experience[:8],
            "evidence_policy": "Use only as paper-agnostic organization guidance; every instantiated section and submodule requires current-paper evidence.",
        },
        "stage4_manifest": global_rules,
        "stage4_execution_policy": {
            "enabled": True,
            "mode": "explicit_experience_constraints",
            "rule_classes": ["advisory", "executable_geometry", "prohibited"],
            "source_manifest": str(next((p.resolve() for p in manifest_candidates if p.exists()), "")),
        },
    }


def select_reference_exemplars(
    paper_review: dict[str, Any] | str | Path,
    library: str | Path,
    out: str | Path,
    selected_template: dict[str, Any] | str | Path | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    review = _load_json(paper_review) if isinstance(paper_review, (str, Path)) else paper_review
    template = _load_json(selected_template) if isinstance(selected_template, (str, Path)) else (selected_template or {})
    records = load_reference_library(library)
    positives = [record for record in records if record.get("quality_label") == "positive"]
    negatives = [record for record in records if record.get("quality_label") == "negative"]
    if not positives:
        raise ValueError("Reference library has no positive exemplars")
    ledger = _exclusion_ledger(positives)
    framework_positives = [record for record in positives if is_framework_figure(record)]
    if not framework_positives:
        raise NoFrameworkExemplarError(
            "Reference library has no positive exemplars that are framework figures",
            excluded=_excluded_ids(ledger),
        )
    positives = framework_positives
    features = paper_features_from_review(review, template)
    scored_positive = []
    for record in positives:
        score, reasons = _score_record(record, features)
        scored_positive.append({"id": record.get("id"), "score": score, "reasons": reasons, "record": record})
    scored_positive.sort(key=lambda item: (-item["score"], str(item["id"])))
    selected_positive = scored_positive[: max(1, min(20, int(top_k)))]
    scored_negative = []
    for record in negatives:
        score, reasons = _score_record(record, features)
        scored_negative.append({"id": record.get("id"), "score": score, "reasons": reasons, "record": record})
    scored_negative.sort(key=lambda item: (-item["score"], str(item["id"])))
    selected_negative = scored_negative[:3]
    brief = build_exemplar_style_brief(selected_positive, selected_negative, library)
    output = ensure_dir(out)
    selection = {
        "summary": "Deterministic reference exemplar selection.",
        "ok": True,
        "source_library": str(Path(library).resolve()),
        "out_dir": str(output.resolve()),
        "record_count": len(records),
        "positive_count": len(positives) + len(ledger),
        "framework_positive_count": len(positives),
        "excluded_by_kind": _excluded_ids(ledger),
        "excluded_detail": ledger,
        "negative_count": len(negatives),
        "top_k": max(1, min(20, int(top_k))),
        "paper_features": features,
        "selected_positive": [{"id": item["id"], "score": item["score"], "reasons": item["reasons"]} for item in selected_positive],
        "selected_negative": [{"id": item["id"], "score": item["score"], "reasons": item["reasons"]} for item in selected_negative],
        "selection_path": str(output / "exemplar_selection.json"),
        "brief_path": str(output / "exemplar_style_brief.json"),
    }
    write_json(output / "exemplar_selection.json", selection)
    write_json(output / "exemplar_style_brief.json", brief)
    return {"summary": "Reference exemplars selected.", "ok": True, **selection, "brief": brief}


def _under_any(path: Path, roots: list[Path]) -> bool:
    """Whether a figure sits inside a directory this selection must not read from."""

    for root in roots:
        try:
            if path == root or path.is_relative_to(root):
                return True
        except (OSError, ValueError):
            continue
    return False


def _fold_key(text: str) -> str:
    """One key written the way a path part writes it (``10.1145/x`` and ``10.1145_x`` agree)."""

    return str(text or "").strip().replace("\\", "_").replace("/", "_").lower()


def _path_name_forms(path_text: str) -> set[str]:
    """The names a figure's path is built from, folded for comparison.

    Whole path parts are returned, never the path as one string. A key that merely appears
    inside a longer name is a different name -- ``paper`` is not ``paperimage`` -- and matching
    it loosely is what empties the pool.
    """

    forms: set[str] = set()
    for part in str(path_text or "").replace("\\", "/").split("/"):
        folded = _fold_key(part)
        if folded:
            forms.add(folded)
    return forms


def _identifier_keys(tokens: Iterable[str]) -> tuple[set[str], set[str]]:
    """Split caller keys into the ones that can name a paper and the ones that cannot.

    A caller that failed to learn the paper's identifier falls back to naming the paper after
    its input file, and that fallback can be as generic as ``paper``: a key that names no paper
    and, matched loosely, would empty the pool. A key counts as naming a paper when it reads as
    an identifier -- long enough to be one, and carrying a digit. Both halves come back so the
    dropped keys are reported rather than silently discarded.
    """

    kept: set[str] = set()
    dropped: set[str] = set()
    for token in tokens:
        text = str(token or "").strip()
        if not text:
            continue
        if len(text) >= 6 and any(character.isdigit() for character in text):
            kept.add(_fold_key(text))
        else:
            dropped.add(text)
    return kept, dropped


def _is_named_for(path_forms: set[str], keys: set[str]) -> bool:
    """Whether one of the caller's keys is a name this path carries.

    A whole path part matches outright. A part that continues a key with a separator
    (``<key>_fig1.png``) is the same figure named with a suffix, which is how a figure exported
    from the paper's own run is written.
    """

    for key in keys:
        for form in path_forms:
            if form == key:
                return True
            if form.startswith(key) and form[len(key):][:1] in {"_", "-", "."}:
                return True
    return False


def select_gap_reference(
    paper_review: dict[str, Any] | str | Path,
    library: str | Path,
    *,
    selected_template: dict[str, Any] | str | Path | None = None,
    exclude_ids: Iterable[str] = (),
    exclude_paths: Iterable[str | Path] = (),
    exclude_tokens: Iterable[str] = (),
) -> dict[str, Any]:
    """Pick the one good figure a paper's own output should be measured against.

    Two things make a figure usable for this question. It has to be the kind of figure this
    project draws -- a framework of labelled modules and connectors -- because a gap named
    against a data panel is a gap about a different object, and a seat shown one answers
    "theirs plots a physical field and ours does not", which nothing downstream can act on.
    And it must not be one of the paper's own figures: a paper compared with itself has no gap
    to find, and the seat would be asked to prefer our drawing to our drawing. A record is kept
    out when one of the paper's keys is one of the names in the path it came from -- whole path
    parts are compared, so the folder a paper's own run wrote is recognised while a folder that
    merely shares some of its letters is not. Callers pass whatever identifier they managed to
    learn, and a run that learned none names its paper after its input file; a key too generic to
    name a paper is dropped and reported rather than allowed to empty the pool.

    The library is where the kind is decided, not the caller: the classifier that already
    separates framework from data figures is the same one the planner selects exemplars with,
    so the two cannot disagree about what a good framework figure is.
    """

    review = _load_json(paper_review) if isinstance(paper_review, (str, Path)) else paper_review
    template = _load_json(selected_template) if isinstance(selected_template, (str, Path)) else (selected_template or {})
    records = load_reference_library(library)
    positives = [record for record in records if record.get("quality_label") == "positive"]
    framework = [record for record in positives if is_framework_figure(record)]
    blocked_ids = {str(item) for item in exclude_ids if str(item or "").strip()}
    blocked_roots = [Path(item).resolve() for item in exclude_paths if str(item or "").strip()]
    blocked_keys, ignored_keys = _identifier_keys(exclude_tokens)
    excluded: list[dict[str, Any]] = []
    usable: list[dict[str, Any]] = []
    for record in framework:
        path = Path(str(record.get("source_path") or ""))
        reason = ""
        if str(record.get("id") or "") in blocked_ids:
            reason = "the figure belongs to the paper being evolved"
        elif not path.is_file():
            reason = "the image is not readable"
        elif blocked_roots and _under_any(path.resolve(), blocked_roots):
            reason = "the figure sits under a path this selection must not read"
        elif blocked_keys and _is_named_for(_path_name_forms(str(record.get("source_path") or "")), blocked_keys):
            reason = "the figure is one of the paper being evolved"
        if reason:
            excluded.append({"id": record.get("id"), "reason": reason})
        else:
            usable.append(record)
    features = paper_features_from_review(review, template)
    scored = []
    for record in usable:
        score, reasons = _score_record(record, features)
        scored.append({"id": record.get("id"), "score": score, "reasons": reasons, "record": record})
    scored.sort(key=lambda item: (-item["score"], str(item["id"])))
    selected = scored[0] if scored else {}
    record = selected.get("record") if isinstance(selected.get("record"), dict) else {}
    return {
        "summary": "One framework exemplar selected to read this paper's output against it.",
        "ok": bool(selected),
        "reason": "" if selected else (
            "the reference library holds no positive framework figure this paper may be measured against"
            if framework else "the reference library holds no positive framework figure at all"
        ),
        "source_library": str(Path(library).resolve()),
        "record_count": len(records),
        "positive_count": len(positives),
        "framework_positive_count": len(framework),
        "usable_count": len(usable),
        "excluded": excluded,
        "exclude_keys_applied": sorted(blocked_keys),
        "exclude_keys_ignored": sorted(ignored_keys),
        "paper_features": features,
        "selected": (
            {
                "id": record.get("id"),
                "source_path": str(record.get("source_path") or ""),
                "archetype": str(record.get("archetype") or ""),
                "why_good": [str(item) for item in (record.get("why_good") or [])][:6],
                "score": selected.get("score"),
                "reasons": selected.get("reasons"),
            }
            if selected
            else {}
        ),
    }
