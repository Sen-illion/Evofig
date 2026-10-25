"""Resolve concise academic icon metaphors and reference-derived style rules."""

from __future__ import annotations

from typing import Any


ACADEMIC_ICON_RULES = [
    "one dominant semantic symbol with at most one small supporting mark",
    "flat two-dimensional vector-like rendering with restrained pastel fills",
    "clean uniform medium outline and simple geometry",
    "immediately recognizable when reduced inside a paper framework node",
    "transparent background with no scene, card, shadow, or decorative environment",
]

_METAPHORS = (
    (("tool", "api", "function"), "one compact crossed wrench-and-gear symbol"),
    (("search", "retriev", "lookup"), "one magnifying glass over two small document tiles"),
    (("revis", "refin", "edit", "correct"), "one document sheet with a small pencil and check mark"),
    (("plan", "schedule", "strategy"), "one simple three-step flow glyph with connected dots"),
    (("input", "instruction", "prompt", "query"), "one document entering a small input tray"),
    (("output", "result", "response", "answer"), "one document leaving an output tray with a check mark"),
    (("agent", "actor", "model"), "one simple assistant head silhouette with a small gear"),
    (("memory", "database", "store"), "one compact stacked-database cylinder"),
    (("score", "judge", "critic", "evaluat"), "one clipboard with a check mark and a small rating bar"),
    (("tree", "mcts", "branch"), "one compact branching tree with three circular terminal nodes"),
)


def resolve_primary_object(node: dict[str, Any], subject: str) -> str:
    """Prefer a planned metaphor, then use a stable paper-icon fallback."""
    planned = str(node.get("primary_object") or node.get("visual_metaphor") or "").strip()
    if planned and planned.casefold() not in {subject.casefold(), f"scientific visual for {subject}".casefold()}:
        return planned
    low = f"{subject} {node.get('role') or ''}".casefold()
    for keywords, metaphor in _METAPHORS:
        if any(keyword in low for keyword in keywords):
            return metaphor
    return f"one simple flat symbolic object representing {subject}"


def reference_icon_rules(exemplar_brief: dict[str, Any] | None) -> list[str]:
    """Combine the canonical user-requested grammar with concise exemplar rules."""
    rules = list(ACADEMIC_ICON_RULES)
    brief = exemplar_brief if isinstance(exemplar_brief, dict) else {}
    candidates = []
    for key in ("icon_rules", "style_rules", "hierarchy_rules"):
        value = brief.get(key)
        if isinstance(value, list):
            candidates.extend(str(item).strip() for item in value if str(item).strip())
    for item in candidates:
        low = item.casefold()
        if any(word in low for word in ("icon", "symbol", "outline", "flat", "stroke", "illustration")) and item not in rules:
            rules.append(item)
        if len(rules) >= 8:
            break
    return rules

