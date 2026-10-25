"""Domain profiles for direct paper-to-PPTX evolution."""

from __future__ import annotations

from typing import Any


DOMAIN_ALIASES = {
    "ai-ml-method": "ai_method",
    "system-platform": "computer_system",
    "dataset-benchmark": "dataset_or_benchmark",
    "empirical-science": "empirical_science",
    "survey-review": "survey_or_taxonomy",
    "general": "general",
}

DOMAIN_PROFILES: dict[str, dict[str, Any]] = {
    "general": {"icon_style": "clean scientific illustration", "rubric": ["clear causal flow", "faithful terminology"]},
    "ai_method": {"icon_style": "abstract computational scientific icon with tensors, modules, or data flow", "rubric": ["training and inference are unambiguous", "tensor and module flow is directional"]},
    "computer_system": {"icon_style": "technical systems architecture icon with services, devices, storage, or network components", "rubric": ["component ownership is clear", "request and data paths are distinguishable"]},
    "biomedical": {"icon_style": "restrained biomedical scientific illustration with anatomically or experimentally meaningful shapes", "rubric": ["biological scale and experimental order are not conflated", "visual metaphor does not invent biological entities"]},
    "biology": {"icon_style": "restrained biomedical scientific illustration with anatomically or experimentally meaningful shapes", "rubric": ["biological scale and experimental order are not conflated", "visual metaphor does not invent biological entities"]},
    "empirical_science": {"icon_style": "scientific experimental apparatus or measurement icon", "rubric": ["experimental sequence is explicit", "observations are separated from interpretations"]},
    "dataset_or_benchmark": {"icon_style": "dataset, annotation, evaluation, and metric oriented scientific icon", "rubric": ["dataset construction and evaluation are separated", "splits and metrics are not confused"]},
    "survey_or_taxonomy": {"icon_style": "taxonomy and category oriented scientific icon", "rubric": ["hierarchy and membership are clear", "categories do not imply unsupported causality"]},
}


def _missing_domain() -> dict[str, Any]:
    return {"domain": "", "domain_missing": True, "icon_style": "", "rubric": []}


def resolve_domain_profile(requested: str | None, paper_review: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a stable evolution domain from CLI input or paper review metadata.

    ``auto`` with no domain on the paper stays missing. It is not rewritten as
    ``general``, and a missing domain cannot be placed in a same-domain group.
    The general icon style is used only when a caller asks for ``general``.
    """

    value = str(requested or "auto")
    if value == "auto":
        review = paper_review if isinstance(paper_review, dict) else {}
        value = str(review.get("domain_profile") or review.get("selected_domain_profile") or "").strip()
        if not value:
            return _missing_domain()
    key = DOMAIN_ALIASES.get(value, value.replace("-", "_"))
    if key not in DOMAIN_PROFILES:
        return _missing_domain()
    return {"domain": key, "domain_missing": False, **DOMAIN_PROFILES[key]}
