"""Settle a refused round object by object, and keep the parts worth keeping.

A round used to be all or nothing: the figure rolled back to the snapshot it started from, so
every part of a change that was worth keeping went with the part that was not. On a figure whose
geometry is a hundred small decisions that is how a loop spends forty rounds rebuilding the same
figure -- one object at a time, from a clean base, is where the "always the same failed
experiment" came from.

This module settles a refused round object by object instead. The reading of an object is taken
from the same deterministic ruler the round's own geometry report is built from, so what
"worse" means here and what the gate calls a defect cannot drift apart: the breaches, the
overlaps and the crossings an object takes part in, plus how full its panel is.

The settlement is tried widest first and stops at the first step that leaves the figure no worse
than the snapshot, so a settlement can never hand back a figure with more faults than the one it
was protecting:

*Tier 1.* An object that came out better is kept, and so is one that moved without coming out
worse -- unless the comparison named it. Only the objects that came out worse, and the objects
the comparison blamed, go back to the snapshot.

*Tier 2.* Only the objects that came out better are kept; everything else goes back.

*Tier 3.* The whole round goes back. This is the floor: whatever the steps above did, the figure
ends up where it started.

Which objects the comparison blamed is read off the defect it named rather than guessed at: a
defect about overlap points at the objects that overlap, a defect about arrows or module order
points at the connectors that cross. A defect that is prose about the science -- a missing stage,
a corrupted term -- points at no object, so it forces nothing back and the settlement falls to
what the readings themselves say.

Two cases have no object to settle and are refused rather than guessed at.

*A re-plan.* A move that rewrote the plan -- a re-segmentation, a structure re-plan -- carries
object ids that did not exist before, so "this object came out worse" has no meaning: there is
nothing to compare. Those rounds already roll back whole, and it is said out loud here rather
than merged into something nobody planned.

*An unfaithful merge.* When nothing was kept and nothing was added, the merged figure has to be
the snapshot exactly; a merge that differs dragged candidate content in on its own -- icons and
programs that follow no object were the way that happened -- and the round rolls back whole with
``merge_unfaithful`` recorded rather than shipping a figure nobody settled.

A panel whose objects the move deleted is settled at the panel: its objects are restored unless
the panel itself came out no worse, and either way the deletion is written down as
``object_orphaned`` rather than dropped in silence. An object that exists only in the candidate
is settled on its own: it is added only while the figure is no worse with it in it, so nothing
is swept in with a reading nobody took.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ...evaluation.geometry_quality import evaluate_density_quality, evaluate_geometry_quality


PARTIAL_ROLLBACK_CONTRACT = "rfs.partial_rollback"
PARTIAL_ROLLBACK_VERSION = 1
OBJECT_READINGS_CONTRACT = "rfs.object_readings"
#: Below this share of the objects surviving, the move re-planned the figure and there is nothing
#: to settle object by object. Above it, the change is an edit of the figure rather than a new one.
OBJECT_OVERLAP_FLOOR = 0.6
#: How much fuller a panel has to be before a lower occupancy counts as a move in the other
#: direction rather than as the same reading.
OCCUPANCY_EPSILON = 0.005
SETTLED_RESTORED = "restored"
SETTLED_DELETION_KEPT = "deletion_kept"

#: The steps a settlement is tried in, widest first. The first step whose merge leaves the figure
#: no worse than the snapshot is the one the round settles at, and the report says which.
TIER_WIDEST = 1
TIER_IMPROVED_ONLY = 2
TIER_WHOLE = 3
SETTLEMENT_TIERS = (TIER_WIDEST, TIER_IMPROVED_ONLY, TIER_WHOLE)
#: The reading a named defect points at. A defect whose name is not here is prose about the
#: science rather than about the drawing, so it points at no object and forces nothing back.
BLAME_READING_FIELDS: dict[str, str] = {
    "overlap": "overlaps",
    "module_order_error": "crossings",
    "wrong_arrow_target": "crossings",
}
#: A defect about arrows or module order is about the connectors that cross, so only a connector
#: is forced back for it; a defect about overlap is about whatever overlaps.
BLAME_KINDS: dict[str, str] = {
    "module_order_error": "connector",
    "wrong_arrow_target": "connector",
}
#: The fields of one object's reading. Two merges that agree on all of them are the same figure
#: for every purpose this module settles.
READING_FIELDS = ("kind", "panel_id", "geometry", "boundary_breaches", "overlaps", "crossings", "panel_occupancy", "faults")


def _layout_of(state: dict[str, Any] | None) -> dict[str, Any]:
    source = state if isinstance(state, dict) else {}
    layout = source.get("layout") if isinstance(source.get("layout"), dict) else {}
    return layout


def figure_program_from_state(state: dict[str, Any] | None) -> dict[str, Any]:
    """The figure program the geometry ruler reads, out of a round's own state.

    A candidate carries the program it compiled; a snapshot taken before the first move carries
    only the layout. Both are handed to the ruler the same way, so an object is read with one
    definition whichever side of the round it sits on.
    """

    source = state if isinstance(state, dict) else {}
    program = source.get("program") if isinstance(source.get("program"), dict) else {}
    layout = _layout_of(source)
    figure = {
        "panels": layout.get("panels") or program.get("panels") or [],
        "semantic_nodes": layout.get("nodes") or program.get("semantic_nodes") or [],
        "arrows": layout.get("connectors") or program.get("arrows") or [],
    }
    for key in ("slots", "text_program", "title_block", "density_targets", "canvas", "topology"):
        if isinstance(program.get(key), (dict, list)) and program.get(key):
            figure[key] = program[key]
        elif isinstance(source.get(key), (dict, list)) and source.get(key):
            figure[key] = source[key]
    return figure


def _panel_occupancy(density: dict[str, Any]) -> dict[str, float]:
    readings: dict[str, float] = {}
    for item in density.get("panels") or []:
        if not isinstance(item, dict):
            continue
        try:
            readings[str(item.get("panel_id") or "")] = float(item.get("occupancy") or 0.0)
        except (TypeError, ValueError):
            continue
    return readings


def _geometry(value: Any) -> Any:
    """An object's own shape, rounded so float dust is not read as a move.

    A node or a panel is read from its box; a connector from the path it was drawn along. Two
    readings that differ in this value are not the same object in the same place, whatever their
    fault counts say.
    """

    if isinstance(value, dict):
        return [round(float(value.get(key) or 0.0), 6) for key in ("x", "y", "w", "h")]
    if isinstance(value, list):
        return [
            [round(float(point[0]), 6), round(float(point[1]), 6)]
            for point in value
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
    return None


def _blank(kind: str, panel_id: str = "") -> dict[str, Any]:
    return {
        "kind": str(kind),
        "panel_id": str(panel_id),
        "geometry": None,
        "boundary_breaches": 0,
        "overlaps": 0,
        "crossings": 0,
        "panel_occupancy": 0.0,
    }


def object_readings(state: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """One reading per object: the breaches, overlaps and crossings it takes part in, and how
    full its panel is.

    The ruler is the figure-quality module's own, so an object the gate calls broken is broken
    here too. Every fault record names the objects it is about, and every object named anywhere
    gets a reading even when the reading is all zeroes -- an object that is clean has to be
    distinguishable from an object that is missing.
    """

    figure = figure_program_from_state(state)
    layout = _layout_of(state)
    geometry = evaluate_geometry_quality(figure)
    density = evaluate_density_quality(figure)
    occupancy = _panel_occupancy(density)
    panel_of = {str(item.get("id") or ""): str(item.get("panel_id") or "") for item in layout.get("nodes") or [] if isinstance(item, dict)}
    readings: dict[str, dict[str, Any]] = {}
    for item in layout.get("panels") or []:
        if isinstance(item, dict) and str(item.get("id") or ""):
            key = str(item["id"])
            readings[key] = _blank("panel", key)
            readings[key]["geometry"] = _geometry(item.get("bbox_percent"))
    for item in layout.get("nodes") or []:
        if isinstance(item, dict) and str(item.get("id") or ""):
            key = str(item["id"])
            readings[key] = _blank("node", panel_of.get(key, ""))
            readings[key]["geometry"] = _geometry(item.get("bbox_percent"))
    for item in layout.get("connectors") or []:
        if isinstance(item, dict) and str(item.get("id") or ""):
            key = str(item["id"])
            readings[key] = _blank("connector", panel_of.get(str(item.get("source") or ""), "") )
            readings[key]["geometry"] = _geometry(item.get("path_percent"))
    for key, reading in readings.items():
        reading["panel_occupancy"] = round(float(occupancy.get(str(reading.get("panel_id") or key) or "", 0.0)), 6)

    def _touch(object_id: str, field: str) -> None:
        reading = readings.get(str(object_id))
        if reading is not None:
            reading[field] = int(reading[field]) + 1

    for item in geometry.get("boundary_violations") or []:
        if not isinstance(item, dict):
            continue
        for key in ("node_id", "slot_id", "object_id"):
            if str(item.get(key) or ""):
                _touch(str(item[key]), "boundary_breaches")
    for item in geometry.get("node_overlaps") or []:
        if isinstance(item, dict):
            _touch(str(item.get("node_a") or ""), "overlaps")
            _touch(str(item.get("node_b") or ""), "overlaps")
    for item in geometry.get("connector_label_overlaps") or []:
        if isinstance(item, dict):
            _touch(str(item.get("node_id") or ""), "overlaps")
            _touch(str(item.get("label_id") or ""), "overlaps")
    for item in geometry.get("box_passes") or []:
        if isinstance(item, dict):
            _touch(str(item.get("connector_id") or ""), "crossings")
    for item in geometry.get("text_passes") or []:
        if isinstance(item, dict):
            _touch(str(item.get("connector_id") or ""), "crossings")
    for item in geometry.get("icon_text_overlaps") or []:
        if isinstance(item, dict):
            _touch(str(item.get("node_id") or ""), "overlaps")
    for key, reading in readings.items():
        reading["faults"] = faults_of(reading)
    return readings


def faults_of(reading: dict[str, Any] | None) -> int:
    """How many visible faults one object takes part in."""

    source = reading if isinstance(reading, dict) else {}
    return int(source.get("boundary_breaches") or 0) + int(source.get("overlaps") or 0) + int(source.get("crossings") or 0)


def object_came_out_worse(before: dict[str, Any] | None, after: dict[str, Any] | None) -> bool:
    """Whether an object is worse than it was, on its own reading.

    More faults is worse. With the same faults, an object in a panel that lost more than a
    tolerance of its fill is worse too: a node that kept its own box while its panel emptied out
    is a node that no longer sits in a figure a reader can read.
    """

    was, now = faults_of(before), faults_of(after)
    if now != was:
        return now > was
    try:
        was_fill = float((before or {}).get("panel_occupancy") or 0.0)
        now_fill = float((after or {}).get("panel_occupancy") or 0.0)
    except (TypeError, ValueError):
        return False
    return now_fill + OCCUPANCY_EPSILON < was_fill


def object_moved(before: dict[str, Any] | None, after: dict[str, Any] | None) -> bool:
    """Whether an object's own shape changed, so its new reading is of a different object.

    A box that was redrawn somewhere else, or a connector whose path was rerouted, is not the
    same object in the same place; keeping its new place in a figure whose neighbours are being
    put back is what turned a settlement into a figure with more faults than the snapshot.
    """

    was = (before or {}).get("geometry")
    now = (after or {}).get("geometry")
    if was is None or now is None:
        return False
    return was != now


def _by_id(items: Any) -> dict[str, dict[str, Any]]:
    return {str(item.get("id") or ""): item for item in (items or []) if isinstance(item, dict) and str(item.get("id") or "")}


def _merge_objects(
    *, before_items: Any, after_items: Any, keep: set[str], additions: set[str] | None = None
) -> list[dict[str, Any]]:
    """Take a preserved object from the candidate and every other one from the snapshot.

    An object that exists only in the candidate is not swept in: it is added only when the
    settlement gave it a verdict, and the verdict is that the figure is no worse with it.
    """

    allowed = additions if additions is not None else set()
    by_before, by_after = _by_id(before_items), _by_id(after_items)
    merged: list[dict[str, Any]] = []
    for object_id, item in by_before.items():
        if object_id in by_after and object_id in keep:
            merged.append(deepcopy(by_after[object_id]))
        else:
            merged.append(deepcopy(item))
    for object_id, item in by_after.items():
        if object_id not in by_before and object_id in allowed:
            merged.append(deepcopy(item))
    return merged


def blamed_objects(readings: dict[str, Any], categories: Any = None) -> set[str]:
    """The objects a comparison's named defect points at.

    The comparison does not hand back object ids, so a named defect is read against the readings:
    a defect about overlap points at every object that overlaps, a defect about arrows or module
    order points at the connectors that cross. A name that points at nothing -- a missing stage is
    prose about the paper, not about an object -- blames no object, and the settlement then falls
    back to what the readings themselves say.
    """

    names = {str(item).strip() for item in (categories or ()) if str(item).strip()}
    blamed: set[str] = set()
    for object_id, reading in (readings or {}).items():
        source = reading if isinstance(reading, dict) else {}
        for name in names:
            field = BLAME_READING_FIELDS.get(name)
            if not field:
                continue
            kind = BLAME_KINDS.get(name)
            if kind and str(source.get("kind") or "") != kind:
                continue
            if int(source.get(field) or 0) > 0:
                blamed.add(str(object_id))
    return blamed


def _readings_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Whether two sets of object readings describe the same figure, object for object."""

    if set(left) != set(right):
        return False
    for object_id, reading in left.items():
        other = right.get(object_id) or {}
        for field in READING_FIELDS:
            if reading.get(field) != other.get(field):
                return False
    return True


def _settles_to_the_snapshot(
    readings_before: dict[str, Any], merged_state: dict[str, Any], dropped: set[str]
) -> bool:
    """Whether a merge is exactly the snapshot, object for object, minus the deletions it kept.

    Read when the merge was meant to keep nothing: a merge that still differs has dragged
    candidate content in on its own, and that is what ``merge_unfaithful`` is for.
    """

    expected = {object_id: reading for object_id, reading in readings_before.items() if object_id not in dropped}
    return _readings_equal(expected, object_readings(merged_state))


def _kept_content(
    keep: Any, readings_before: dict[str, Any], readings_after: dict[str, Any]
) -> bool:
    """Whether a kept object is content the settlement actually took from the candidate.

    An object the merge kept whose reading is the same on both sides is not a change: a panel
    that came through untouched is written down as kept and is the snapshot's panel either way.
    The merge then has to be the snapshot exactly, and this is what tells the two apart.
    """

    for object_id in keep:
        was = readings_before.get(object_id)
        now = readings_after.get(object_id)
        if was is None or now is None:
            return True
        if any(was.get(field) != now.get(field) for field in READING_FIELDS):
            return True
    return False


def _reverted_rows(
    ids: Any,
    readings_before: dict[str, Any],
    readings_after: dict[str, Any],
    worse: set[str],
    blamed: set[str],
) -> list[dict[str, Any]]:
    """Why each object went back to the snapshot."""

    rows: list[dict[str, Any]] = []
    for object_id in sorted(ids):
        was = readings_before.get(object_id) or {}
        now = readings_after.get(object_id) or {}
        if object_id in blamed:
            basis = "blamed"
            why = "the comparison named this object in the defect it refused the round for, so it goes back to the snapshot"
        elif object_id in worse:
            basis = "worse"
            why = (
                "this object came out with more faults than it went in with"
                if faults_of(now) != faults_of(was)
                else "this object kept its own faults while its panel emptied out"
            )
        elif object_moved(was, now):
            basis = "unimproved"
            why = "this object moved without coming out better, so it goes back to the snapshot"
        else:
            basis = "unimproved"
            why = "this object came out no better than it went in, so it goes back to the snapshot"
        rows.append(
            {
                "object_id": object_id,
                "kind": str(now.get("kind") or was.get("kind") or ""),
                "panel_id": str(now.get("panel_id") or was.get("panel_id") or ""),
                "before_faults": faults_of(was),
                "after_faults": faults_of(now),
                "basis": basis,
                "why": why,
            }
        )
    return rows


def _kept_rows(
    ids: Any,
    readings_before: dict[str, Any],
    readings_after: dict[str, Any],
    improved: set[str],
) -> list[dict[str, Any]]:
    """Why each object was taken from the refused candidate."""

    rows: list[dict[str, Any]] = []
    for object_id in sorted(ids):
        was = readings_before.get(object_id) or {}
        now = readings_after.get(object_id) or {}
        if object_id in improved:
            basis = "improved"
            why = "this object came out with fewer faults than it went in with"
        else:
            basis = "unchanged"
            why = (
                "this object moved without coming out worse, so its new place is kept"
                if object_moved(was, now)
                else "this object stayed where it was and did not come out worse, so it is kept"
            )
        rows.append(
            {
                "object_id": object_id,
                "kind": str(now.get("kind") or was.get("kind") or ""),
                "panel_id": str(now.get("panel_id") or was.get("panel_id") or ""),
                "before_faults": faults_of(was),
                "after_faults": faults_of(now),
                "basis": basis,
                "why": why,
            }
        )
    return rows


def _merge_icons(before_state: dict[str, Any], after_state: dict[str, Any], kept: set[str]) -> dict[str, Any]:
    """Icons follow the objects: an icon comes from whichever side its object comes from.

    An asset is named by the object it belongs to when it says so (``object_id`` / ``node_id`` /
    ``parent_id``): an object the settlement kept brings the candidate's icon, an object it gave
    back brings the snapshot's, and an icon the candidate dropped is gone with the object it
    belonged to. An asset that names no object has no verdict to follow, so it comes from the
    snapshot -- letting one in from the candidate is how a merge that was meant to keep nothing
    dragged candidate content into the figure anyway.
    """

    def _rows(state: dict[str, Any]) -> list[dict[str, Any]]:
        icons = state.get("icons") if isinstance(state.get("icons"), dict) else {}
        return [item for item in (icons.get("icons") or []) if isinstance(item, dict)]

    def _owner(item: dict[str, Any]) -> str:
        return str(item.get("object_id") or item.get("node_id") or item.get("parent_id") or "")

    rows: list[dict[str, Any]] = []
    taken: set[str] = set()
    for item in _rows(after_state):
        if _owner(item) and _owner(item) in kept:
            key = str(item.get("asset_id") or item.get("id") or "")
            rows.append(deepcopy(item))
            if key:
                taken.add(key)
    for item in _rows(before_state):
        owner = _owner(item)
        if owner and owner in kept:
            continue
        key = str(item.get("asset_id") or item.get("id") or "")
        if key and key in taken:
            continue
        rows.append(deepcopy(item))
    merged = deepcopy(before_state.get("icons")) if isinstance(before_state.get("icons"), dict) else {}
    merged["icons"] = rows
    return merged


def _slot_keep(program: dict[str, Any], keep: set[str]) -> set[str]:
    """Which slots follow an object that was kept.

    A slot is named by its node (``parent_id``, or its own id without the ``slot_`` prefix), so a
    slot is kept exactly when the object it is drawn into is kept, and rolls back with it.
    """

    keep_ids: set[str] = set()
    for item in program.get("slots") or []:
        if not isinstance(item, dict):
            continue
        slot_id = str(item.get("id") or "")
        if not slot_id:
            continue
        parent = str(item.get("parent_id") or "") or slot_id.removeprefix("slot_")
        if parent in keep:
            keep_ids.add(slot_id)
    return keep_ids


def _merge_program(
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    keep: set[str],
    dropped: set[str],
    additions: set[str] | None = None,
) -> dict[str, Any]:
    """The figure program with the kept objects taken from the candidate and the rest from the
    snapshot.

    The layout and the program are the same figure seen twice, so they have to be settled with one
    verdict: an object kept in the layout is kept in the program too, and an object given back
    comes back in both. An object the move deleted leaves the program the same way it leaves the
    layout. A slot or an icon has no reading of its own, so it follows the object it belongs to and
    that object's verdict is its verdict.
    """

    allowed = additions if additions is not None else set()
    before_program = before_state.get("program") if isinstance(before_state.get("program"), dict) else {}
    after_program = after_state.get("program") if isinstance(after_state.get("program"), dict) else {}
    if not after_program:
        return deepcopy(before_program)
    merged = deepcopy(before_program)
    merged["semantic_nodes"] = _merge_objects(
        before_items=before_program.get("semantic_nodes"),
        after_items=after_program.get("semantic_nodes"),
        keep=keep,
        additions=allowed,
    )
    merged["arrows"] = _merge_objects(
        before_items=before_program.get("arrows"),
        after_items=after_program.get("arrows"),
        keep=keep,
        additions=allowed,
    )
    merged["slots"] = _merge_objects(
        before_items=before_program.get("slots"),
        after_items=after_program.get("slots"),
        keep=_slot_keep(after_program, keep),
        additions=_slot_keep(after_program, keep | allowed),
    )
    if dropped:
        for section in ("semantic_nodes", "arrows", "slots"):
            merged[section] = [
                item for item in (merged.get(section) or []) if str(item.get("id") or "") not in dropped
            ]
    return merged


def _panel_verdicts(readings_before: dict[str, Any], readings_after: dict[str, Any]) -> dict[str, bool]:
    """For each panel, whether it came out worse than it went in."""

    verdicts: dict[str, bool] = {}
    for object_id, reading in readings_before.items():
        if str(reading.get("kind") or "") != "panel":
            continue
        verdicts[object_id] = object_came_out_worse(reading, readings_after.get(object_id))
    return verdicts


def plan_partial_rollback(
    before_state: dict[str, Any] | None,
    after_state: dict[str, Any] | None,
    *,
    round_index: int = 0,
    action: dict[str, Any] | None = None,
    blame_categories: Any = None,
) -> dict[str, Any]:
    """Settle a refused round object by object, and hand back the state that keeps the good parts.

    The settlement is tried widest first and stops at the first step whose merge leaves the
    figure no worse than the snapshot. Only the objects a step keeps are taken from the refused
    candidate; everything else comes from the snapshot, including the icons of the objects that
    were given back. When a step keeps nothing at all, the merge is checked against the snapshot
    object by object, and a merge that quietly differs is refused as ``merge_unfaithful`` rather
    than shipped.

    ``blame_categories`` is the defect the comparison named when it refused this round. It is
    read as the objects to give back -- the objects that overlap, or the connectors that cross --
    so the model's own answer about where the figure is worse is what steers the settlement.
    """

    before = before_state if isinstance(before_state, dict) else {}
    after = after_state if isinstance(after_state, dict) else {}
    report: dict[str, Any] = {
        "contract": PARTIAL_ROLLBACK_CONTRACT,
        "version": PARTIAL_ROLLBACK_VERSION,
        "round": int(round_index),
        "action": dict(action) if isinstance(action, dict) else {},
        "applied": False,
        "tier": 0,
        "tiers": [],
        "blame": [],
        "merge_unfaithful": False,
        "reason": "",
        "overlap_ratio": 0.0,
        "kept": [],
        "reverted": [],
        "orphaned": [],
        "additions": [],
        "before_faults": 0,
        "after_faults": 0,
        "merged_faults": 0,
        "readings_improved": False,
        "merged_state": {},
        "summary": "",
    }
    if not before or not after:
        report["reason"] = "a partial rollback needs both the state before the round and the one it produced"
        return report
    readings_before, readings_after = object_readings(before), object_readings(after)
    shared = set(readings_before) & set(readings_after)
    denominator = max(1, len(readings_before))
    overlap = round(len(shared) / denominator, 4)
    report["overlap_ratio"] = overlap
    report["before_faults"] = sum(faults_of(item) for item in readings_before.values())
    report["after_faults"] = sum(faults_of(item) for item in readings_after.values())
    if overlap < OBJECT_OVERLAP_FLOOR:
        report["reason"] = "the move re-planned the figure, so there are no objects to settle one by one"
        report["summary"] = "rolled back whole: the move rewrote the plan rather than editing the figure"
        return report

    blamed = blamed_objects(readings_after, blame_categories)
    report["blame"] = sorted(blamed & shared)
    worse = {
        object_id
        for object_id in shared
        if object_came_out_worse(readings_before[object_id], readings_after[object_id])
    }
    improved = {
        object_id
        for object_id in shared
        if faults_of(readings_after[object_id]) < faults_of(readings_before[object_id])
    }
    # A round that moved no object, added none and deleted none has nothing to settle: there is no
    # object whose verdict could differ from the snapshot's, so the whole round goes back before
    # any reading is weighed. Everything else is settled, even when every object it moved came
    # out worse -- the report then says so object by object instead of silently doing nothing.
    reading_changed = {
        object_id
        for object_id in shared
        if any(
            readings_before[object_id].get(field) != readings_after[object_id].get(field)
            for field in ("geometry", "boundary_breaches", "overlaps", "crossings", "panel_occupancy")
        )
    }
    additions_present = set(readings_after) - set(readings_before)

    panel_verdicts = _panel_verdicts(readings_before, readings_after)
    dropped: set[str] = set()
    for object_id in sorted(set(readings_before) - set(readings_after)):
        reading = readings_before[object_id]
        kind = str(reading.get("kind") or "")
        panel_id = object_id if kind == "panel" else str(reading.get("panel_id") or "")
        panel_worse = bool(panel_verdicts.get(panel_id)) if panel_id else False
        settled = SETTLED_RESTORED if (panel_worse or not panel_id) else SETTLED_DELETION_KEPT
        if settled == SETTLED_DELETION_KEPT:
            dropped.add(object_id)
        report["orphaned"].append(
            {
                "object_id": object_id,
                "kind": kind,
                "panel_id": panel_id,
                "settled": settled,
                "why": (
                    "the move deleted this object and its panel came out worse, so it is restored"
                    if settled == SETTLED_RESTORED and panel_worse
                    else "the move deleted this object and no panel it belongs to can be named, so it is restored"
                    if settled == SETTLED_RESTORED
                    else "the move deleted this object and its panel came out no worse, so the deletion is kept"
                ),
            }
        )
    if not reading_changed and not additions_present and not dropped:
        report["reason"] = "the round moved no object, added none and deleted none, so there is nothing to give back"
        report["summary"] = "rolled back whole: the round moved nothing to settle"
        return report
    layout_before, layout_after = _layout_of(before), _layout_of(after)

    def build(keep_ids: set[str], addition_ids: set[str]) -> dict[str, Any]:
        merged_state = deepcopy(before)
        merged_layout = merged_state.setdefault("layout", {})
        if not isinstance(merged_layout, dict):
            merged_layout = {}
            merged_state["layout"] = merged_layout
        for key, section in (("panels", "panels"), ("nodes", "nodes"), ("connectors", "connectors")):
            merged_layout[section] = _merge_objects(
                before_items=layout_before.get(key),
                after_items=layout_after.get(key),
                keep=keep_ids,
                additions=addition_ids,
            )
        if dropped:
            for section in ("panels", "nodes", "connectors"):
                merged_layout[section] = [
                    item for item in merged_layout[section] if str(item.get("id") or "") not in dropped
                ]
        merged_state["program"] = _merge_program(before, after, keep_ids, dropped, additions=addition_ids)
        merged_state["icons"] = _merge_icons(before, after, keep_ids | addition_ids)
        return merged_state

    def faults_in(state: dict[str, Any]) -> int:
        return sum(faults_of(item) for item in object_readings(state).values())

    def settle_additions(keep_ids: set[str]) -> tuple[set[str], list[dict[str, Any]]]:
        # An object that exists only in the candidate is settled one at a time: it is added only
        # while the figure with it stays no worse than the snapshot, so every one of them carries
        # a verdict instead of being swept in with the objects whose readings were never taken.
        addition_ids: set[str] = set()
        rows: list[dict[str, Any]] = []
        for object_id in sorted(additions_present):
            reading = readings_after[object_id]
            trial = addition_ids | {object_id}
            kept_it = faults_in(build(keep_ids, trial)) <= int(report["before_faults"])
            if kept_it:
                addition_ids = trial
            rows.append(
                {
                    "object_id": object_id,
                    "kind": str(reading.get("kind") or ""),
                    "panel_id": str(reading.get("panel_id") or ""),
                    "faults": faults_of(reading),
                    "kept": bool(kept_it),
                    "why": (
                        "the figure is no worse with this object in it, so the addition is kept"
                        if kept_it
                        else "the figure came out worse with this object in it, so the addition is turned down"
                    ),
                }
            )
        return addition_ids, rows

    for tier in SETTLEMENT_TIERS:
        if tier == TIER_WIDEST:
            keep = {object_id for object_id in shared if object_id not in worse and object_id not in blamed}
        elif tier == TIER_IMPROVED_ONLY:
            keep = {object_id for object_id in improved if object_id not in blamed}
        else:
            keep = set()
        addition_ids, addition_rows = settle_additions(keep)
        merged = build(keep, addition_ids)
        merged_faults = faults_in(merged)
        if merged_faults > int(report["before_faults"]):
            report["tiers"].append(
                {
                    "tier": tier,
                    "merged_faults": merged_faults,
                    "before_faults": int(report["before_faults"]),
                    "accepted": False,
                    "why": "the merge left the figure with more faults than the snapshot",
                }
            )
            continue
        # Nothing kept that changed and nothing added means the merge has to be the snapshot,
        # object for object. A merge that differs dragged candidate content in on its own, so the
        # round rolls back whole rather than shipping a figure nobody settled.
        settles_to_snapshot = _settles_to_the_snapshot(readings_before, merged, dropped)
        if not _kept_content(keep, readings_before, readings_after) and not addition_ids and not settles_to_snapshot:
            report["merge_unfaithful"] = True
            report["tiers"].append(
                {
                    "tier": tier,
                    "merged_faults": merged_faults,
                    "before_faults": int(report["before_faults"]),
                    "accepted": False,
                    "why": "the merge kept nothing but did not come out exactly like the snapshot",
                }
            )
            continue
        report["tiers"].append(
            {
                "tier": tier,
                "merged_faults": merged_faults,
                "before_faults": int(report["before_faults"]),
                "accepted": True,
                "why": "the merge left the figure no worse than the snapshot",
            }
        )
        report["tier"] = tier
        report["merged_faults"] = merged_faults
        report["readings_improved"] = merged_faults <= int(report["before_faults"])
        report["additions"] = addition_rows
        report["reverted"] = _reverted_rows(shared - keep, readings_before, readings_after, worse, blamed)
        report["kept"] = _kept_rows(keep, readings_before, readings_after, improved)
        report["applied"] = True
        kept_additions = sum(1 for item in addition_rows if item.get("kept"))
        restored = len([item for item in report["orphaned"] if item["settled"] == SETTLED_RESTORED])
        report["reason"] = (
            f"settled at tier {tier}: kept {len(keep)} object(s) from the refused round and gave back "
            f"{len(shared - keep)}"
        )
        report["summary"] = (
            f"settled at tier {tier}: kept {len(keep)} object(s) that were no worse than they went in, "
            f"gave back {len(shared - keep)}, kept {kept_additions} object(s) that exist only in the "
            f"candidate, restored {restored} deleted object(s) and settled {len(dropped)} deletion(s): "
            f"{report['before_faults']} fault(s) before, {merged_faults} after the settlement"
        )
        report["merged_state"] = merged
        return report

    if report["merge_unfaithful"]:
        report["reason"] = "the merge kept nothing but did not come out exactly like the snapshot, so no step could be taken"
        report["summary"] = "rolled back whole: the merge was not faithful to the snapshot"
    else:
        report["reason"] = "keeping the good parts left the figure with more faults than it started with"
        report["summary"] = "rolled back whole: the settlement measured worse than the snapshot"
    return report


def validate_partial_rollback(report: dict[str, Any] | None) -> list[str]:
    """Check a settlement report is readable and does not claim a merge it did not make."""

    source = report if isinstance(report, dict) else {}
    errors: list[str] = []
    if str(source.get("contract") or "") != PARTIAL_ROLLBACK_CONTRACT:
        errors.append("a partial rollback report must carry its own contract name")
    applied = bool(source.get("applied"))
    merged = source.get("merged_state") if isinstance(source.get("merged_state"), dict) else {}
    if applied and not merged:
        errors.append("an applied settlement must carry the merged state it settled into")
    if not applied and merged:
        errors.append("a settlement that was not applied must carry no merged state")
    if applied and not (
        source.get("reverted")
        or source.get("kept")
        or source.get("orphaned")
        or [item for item in source.get("additions") or [] if isinstance(item, dict) and item.get("kept")]
    ):
        errors.append("an applied settlement must have settled at least one object")
    if source.get("tier") not in (0, TIER_WIDEST, TIER_IMPROVED_ONLY, TIER_WHOLE):
        errors.append("a settlement must say which tier it settled at")
    for item in source.get("orphaned") or []:
        if not isinstance(item, dict):
            errors.append("every orphaned object is written as an object")
            continue
        if str(item.get("settled") or "") not in {SETTLED_RESTORED, SETTLED_DELETION_KEPT}:
            errors.append(f"an orphaned object must say how it was settled: {item.get('object_id')}")
    for item in source.get("additions") or []:
        if not isinstance(item, dict):
            errors.append("every object that exists only in the candidate is written as an object")
            continue
        if not isinstance(item.get("kept"), bool):
            errors.append(f"an object that exists only in the candidate must carry a verdict: {item.get('object_id')}")
        elif not str(item.get("why") or "").strip():
            errors.append(f"an object that exists only in the candidate must say why: {item.get('object_id')}")
    return errors


__all__ = [
    "BLAME_KINDS",
    "BLAME_READING_FIELDS",
    "OBJECT_OVERLAP_FLOOR",
    "OBJECT_READINGS_CONTRACT",
    "OCCUPANCY_EPSILON",
    "PARTIAL_ROLLBACK_CONTRACT",
    "PARTIAL_ROLLBACK_VERSION",
    "READING_FIELDS",
    "SETTLEMENT_TIERS",
    "SETTLED_DELETION_KEPT",
    "SETTLED_RESTORED",
    "TIER_IMPROVED_ONLY",
    "TIER_WHOLE",
    "TIER_WIDEST",
    "blamed_objects",
    "faults_of",
    "figure_program_from_state",
    "object_came_out_worse",
    "object_moved",
    "object_readings",
    "plan_partial_rollback",
    "validate_partial_rollback",
]
