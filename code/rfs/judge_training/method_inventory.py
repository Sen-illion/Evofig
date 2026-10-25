"""Hand a Judge the paper's own method list, never the figure's own selection.

One middleware contract call answers two questions at once: which concepts a paper states
(the read), and which of them are worth drawing (the selection, carried by ``must_show``,
``should_render``, ``render_priority`` and ``visualization_value``). A Judge sample must
carry only the first answer. Handing over the second would make "agrees with the taste of a
model that was never trained for this" the reference the Judge is scored against, which is
exactly the signal this training line exists to avoid.

:func:`method_inventory_view` keeps what the paper states - every concept's ``id``,
``label``, ``role``, ``detail_level``, ``evidence_ids`` and the statements it was read
from, the relations between concepts, what the paper forbids inventing, and the figure's
goal - and drops everything that only says how the figure was planned: the four selection
keys above, the extractor's own importance weight, the layout grouping ids
(``parent_id``/``input_ids``/``output_ids``) and any page-layout plan. The view is
idempotent, so a stored inventory and a re-derived one read the same way.

:func:`read_method_inventory` finds one paper's inventory in a directory that may store it
under either its current or its earlier name, and answers a directory holding only the
figure content plan with that plan: the plan states the same concepts plus the selection,
so its read half is the same view. A directory holding neither yields an empty mapping,
so a caller can tell "the paper states nothing here" from "the paper states this" instead
of receiving a look alike.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts.method_figure import (
    FIGURE_CONTENT_PLAN_FILE,
    INVENTORY_FILES,
    LEGACY_INVENTORY_FILE,
    LEGACY_PLAN_FILE,
    METHOD_INVENTORY_FILE,
    PLAN_FILES,
    figure_content_plan_path,
    method_inventory_path,
)

#: What the middleware's own call decided about the figure, which a Judge must not see.
SELECTION_KEYS = ("must_show", "should_render", "render_priority", "visualization_value")

CONCEPT_SECTIONS = ("inputs", "outputs", "modules", "innovations", "stages", "optional_details")
STATEMENT_SECTIONS = ("forbidden_inventions", "uncertainties")
CONCEPT_KEYS = ("id", "label", "role", "detail_level", "evidence_ids", "seed_statements")
RELATION_KEYS = ("id", "source", "target", "type", "label", "evidence_ids")


def _picked(row: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: row[key] for key in keys if key in row}


def _rows(value: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    return [_picked(row, keys) for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _statements(value: Any) -> list[str]:
    return [str(item) for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []


def _read(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def method_inventory_view(document: dict[str, Any] | None) -> dict[str, Any]:
    """Return the read half of one paper's method inventory.

    An empty document yields an empty view rather than a skeleton, so an absent inventory
    is never mistaken for a paper that states nothing.
    """
    if not isinstance(document, dict) or not document:
        return {}
    view: dict[str, Any] = {name: _rows(document.get(name), CONCEPT_KEYS) for name in CONCEPT_SECTIONS}
    view["relations"] = _rows(document.get("relations"), RELATION_KEYS)
    view["figure_goal"] = str(document.get("figure_goal") or "")
    for name in STATEMENT_SECTIONS:
        view[name] = _statements(document.get(name))
    return view


def inventory_path(directory: str | Path, *, plan_fallback: bool = True) -> Path | None:
    """Return the file one directory stores the inventory in, or ``None``.

    The stored name changed once, so a reader accepts both. A directory that only holds
    the figure content plan is answered with the plan: it carries the same concepts, and
    the view taken of it drops the selection the same way.
    """
    return method_inventory_path(directory, plan_fallback=plan_fallback)


def read_method_inventory(directory: str | Path, *, plan_fallback: bool = True) -> dict[str, Any]:
    """Return the stored method inventory of one directory; absent or unreadable gives ``{}``."""
    return _read(inventory_path(directory, plan_fallback=plan_fallback))


def plan_path(directory: str | Path) -> Path | None:
    """Return the file one directory stores the figure content plan in, or ``None``."""
    return figure_content_plan_path(directory)


def read_figure_content_plan(directory: str | Path) -> dict[str, Any]:
    """Return the stored figure content plan of one directory; absent or unreadable gives ``{}``."""
    return _read(plan_path(directory))


__all__ = [
    "CONCEPT_KEYS",
    "CONCEPT_SECTIONS",
    "FIGURE_CONTENT_PLAN_FILE",
    "INVENTORY_FILES",
    "LEGACY_INVENTORY_FILE",
    "LEGACY_PLAN_FILE",
    "METHOD_INVENTORY_FILE",
    "PLAN_FILES",
    "RELATION_KEYS",
    "SELECTION_KEYS",
    "STATEMENT_SECTIONS",
    "inventory_path",
    "method_inventory_view",
    "plan_path",
    "read_figure_content_plan",
    "read_method_inventory",
]
