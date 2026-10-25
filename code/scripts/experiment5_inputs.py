"""Freeze the inputs Experiment 5 reads off a row, before its scores exist.

Experiment 5 (``docs/supplementary-experiments.md``) is not a row of its own: it re-reads the
complete row -- fifty papers per wave, promotion at every wave boundary -- and groups the
deliveries by the generation that produced them. Two of its three inputs therefore cost
nothing at run time and only have to be copied somewhere durable: which paper landed in which
generation, and how many experiences the library held when that generation started. The third
-- the per-paper nine-item, editability and official scores -- is filled in later by the
scoring pass the row needs anyway.

The generation each paper belongs to is read from the row's own record rather than re-derived
from the paper list, because that record is also where the boundary state was written; a group
recomputed from the list would look identical and silently drop the state if the driver's cut
ever changes. Two rows write generations two different ways and both are read here:

* a wave row (``ours_own500``, ``no_middle_layer``) writes ``waves/wave_NN.json`` per generation,
  each carrying the library's snapshot at that boundary;
* a continuous row (``paper-evolve-continuous``, the iteration rounds added on the finished
  decks) writes one state file, ``continuous_state.json``, whose per-paper history holds one
  entry per iteration. There a generation is an iteration number, and what the round carried in
  is read off the entries themselves, because a continuous round has no wave boundary to
  snapshot.

The second shape exists because the two-stage split spends its rounds after the first half has
already been drawn: those rounds are iterations of a paper's own ladder, not waves of a batch,
and Experiment 5 needs both halves in one table.

Usage::

    python scripts/experiment5_inputs.py --arm ours_own500
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_WAVE_SIZE = 50
DELIVERY_NAME = "editable_composition.pptx"
CONTINUOUS_STATE_FILE = "continuous_state.json"
PAPERS_DIRNAME = "papers"

HOW_TO_FILL_THE_SCORES = (
    "python -m rfs.cli benchmark score-own --manifest <the row's paper list> "
    "--runs <this arm dir> --nine-judge api --json"
)


def _read_json(path: Path, default):
    """Read a JSON file, falling back to ``default`` when it is absent or unreadable."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a missing report is a gap, not a crash
        return default


def safe(name: str) -> str:
    """The directory a paper id is filed under, spelled the way the driver spells it.

    A wave file names a paper by its id (``iclr:2026:10010667``) while the run files it under a
    sanitised name (``iclr_2026_10010667``), so the two ends are joined with the driver's own
    rule. The unit test pins this copy against ``scripts/run_ablation_arm.safe`` so the two
    cannot drift into a skeleton that reports every paper as undelivered.
    """

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def delivery_index(out_root: Path) -> dict:
    """Map paper id to the directory that paper's delivery lives in.

    A wave file names its papers by id alone, and the run files them under their domain, so the
    two ends are joined here instead of by guessing a domain from the id.
    """

    index: dict = {}
    if not out_root.is_dir():
        return index
    for domain in sorted(out_root.iterdir()):
        if not domain.is_dir() or domain.name == "waves":
            continue
        for paper in sorted(domain.iterdir()):
            if paper.is_dir():
                index.setdefault(paper.name, paper)
    return index


def _delivered(paper_dir: Path | None) -> bool:
    if paper_dir is None:
        return False
    if (paper_dir / DELIVERY_NAME).is_file():
        return True
    return any(paper_dir.glob("round_*/candidate_*/" + DELIVERY_NAME))


def _paper_row(paper_id: str, paper_dir: Path | None) -> dict:
    """One paper's free fields: cost, rounds, and the empty score slots to fill later."""

    cost = _read_json(paper_dir / "cost_report.json", {}) if paper_dir else {}
    state = _read_json(paper_dir / "evolution_state.json", {}) if paper_dir else {}
    return {
        "paper_id": paper_id,
        "delivery": str(paper_dir) if paper_dir else "",
        "delivered": _delivered(paper_dir),
        "generation_seconds": cost.get("elapsed_seconds"),
        "generation_usd": cost.get("generation_usd"),
        "rounds_completed": state.get("rounds_completed"),
        # Filled by the scoring pass; null means "not scored yet", never "scored zero".
        "nine": None,
        "editable": None,
        "node_f1": None,
        "edge_f1": None,
    }


def _injectable(snapshot: dict) -> int:
    """How many experiences the library offered a generation when it started."""

    by_status = snapshot.get("by_status") if isinstance(snapshot, dict) else None
    if isinstance(by_status, dict):
        try:
            return int(by_status.get("promoted") or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def build_skeleton(out_root: Path, *, wave_size: int = DEFAULT_WAVE_SIZE, papers: list | None = None) -> dict:
    """Group one row's deliveries by generation, with its library snapshot at each boundary."""

    arm_dir = Path(out_root)
    if (arm_dir / CONTINUOUS_STATE_FILE).is_file():
        return build_continuous_skeleton(arm_dir)
    index = delivery_index(arm_dir)
    wave_files = sorted((arm_dir / "waves").glob("wave_*.json")) if arm_dir.is_dir() else []
    source = "wave_files"
    waves: list = []
    if wave_files:
        for path in wave_files:
            record = _read_json(path, {})
            waves.append(
                {
                    "wave": int(record.get("wave") or len(waves) + 1),
                    "papers": [str(item) for item in (record.get("papers") or [])],
                    "library_before": record.get("library_before") or {},
                    "library_after": record.get("library_after") or {},
                    "candidates": len(record.get("candidates") or []),
                    "examined": len(record.get("examined") or []),
                    "validations": record.get("validations") or [],
                    "promoted_between_waves": bool(record.get("promoted_between_waves")),
                    "wave_file": str(path),
                }
            )
    elif papers:
        source = "recomputed_from_list"
        step = int(wave_size or DEFAULT_WAVE_SIZE)
        ids = [str(row.get("paper_id") or "") for row in papers if isinstance(row, dict)]
        ids = [item for item in ids if item]
        for start in range(0, len(ids), step):
            waves.append(
                {
                    "wave": len(waves) + 1,
                    "papers": ids[start:start + step],
                    "library_before": {},
                    "library_after": {},
                    "candidates": 0,
                    "examined": 0,
                    "validations": [],
                    "promoted_between_waves": None,
                    "wave_file": "",
                }
            )
    for wave in waves:
        rows = [_paper_row(paper_id, index.get(safe(paper_id))) for paper_id in wave["papers"]]
        wave["rows"] = rows
        wave["delivered"] = sum(1 for row in rows if row["delivered"])
        wave["injectable_at_start"] = _injectable(wave["library_before"])
    return {
        "experiment": 5,
        "arm": arm_dir.name,
        "arm_dir": str(arm_dir),
        "source": source,
        "wave_size": int(wave_size or DEFAULT_WAVE_SIZE),
        "generations": len(waves),
        "papers": sum(len(wave["papers"]) for wave in waves),
        "delivered": sum(wave["delivered"] for wave in waves),
        "generation_usd": round(
            sum(row["generation_usd"] or 0.0 for wave in waves for row in wave["rows"]), 4
        ),
        "unit": "wave",
        "how_to_fill_the_scores": HOW_TO_FILL_THE_SCORES,
        "waves": waves,
    }


def _iteration_delivery(arm_dir: Path, name: str, record: dict) -> Path:
    """The directory one iteration published, preferring the path the loop itself recorded."""

    for key in ("best", "out"):
        value = str(record.get(key) or "")
        if value:
            return Path(value)
    return arm_dir / PAPERS_DIRNAME / name / ("iter_%02d" % int(record.get("iteration") or 0))


def _iteration_row(arm_dir: Path, name: str, entry: dict, record: dict) -> dict:
    """One iteration as a row of the same shape a wave row's papers have.

    A round that inherited a finished deck and a round that was drawn from scratch both land
    here, and ``inherited_from`` is what tells them apart: a reader that mixed them would be
    comparing a figure an earlier round drew with one this round drew.
    """

    delivery = _iteration_delivery(arm_dir, name, record)
    cost = _read_json(delivery / "cost_report.json", {})
    if not cost:
        cost = record.get("round_cost") if isinstance(record.get("round_cost"), dict) else {}
    state = _read_json(delivery / "evolution_state.json", {})
    reuse = record.get("reuse") if isinstance(record.get("reuse"), dict) else {}
    rounds = state.get("rounds_completed")
    if rounds is None:
        rounds = record.get("rounds")
    return {
        "paper_id": str(entry.get("paper_id") or name),
        "paper": name,
        "iteration": int(record.get("iteration") or 0),
        "delivery": str(delivery),
        "delivered": _delivered(delivery),
        "generation_seconds": cost.get("elapsed_seconds"),
        "generation_usd": cost.get("generation_usd"),
        "rounds_completed": rounds,
        "inherited_from": str(record.get("reuse_from") or ""),
        "reuse_accepted": bool(reuse.get("accepted")),
        "reuse_declined_reason": str(reuse.get("declined_reason") or ""),
        "accepted_rounds": int(record.get("accepted_rounds") or 0),
        "experiences_injected": list(record.get("experiences_injected") or []),
        "candidate_experience_ids": list(record.get("candidate_experience_ids") or []),
        "archive_size": int(record.get("archive_size") or 0),
        # Filled by the scoring pass; null means "not scored yet", never "scored zero".
        "nine": None,
        "editable": None,
        "node_f1": None,
        "edge_f1": None,
    }


def build_continuous_skeleton(arm_dir: Path) -> dict:
    """Group a continuous row's iterations by generation, one generation per iteration number.

    The grouping comes from the loop's own state file rather than from the directories, for the
    same reason the wave path reads the wave files: the state is the loop's account of what it
    did, including the rounds that failed and the ones that inherited nothing and redrew.
    """

    state = _read_json(arm_dir / CONTINUOUS_STATE_FILE, {})
    papers = state.get("papers") if isinstance(state.get("papers"), dict) else {}
    validations = state.get("validations") if isinstance(state.get("validations"), dict) else {}
    grouped: dict = {}
    for name in sorted(papers):
        entry = papers[name] if isinstance(papers[name], dict) else {}
        for record in entry.get("history") or []:
            if not isinstance(record, dict):
                continue
            row = _iteration_row(arm_dir, name, entry, record)
            grouped.setdefault(row["iteration"], []).append(row)
    measured: dict = {}
    for experience_id, record in validations.items():
        if isinstance(record, dict):
            measured.setdefault(int(record.get("iteration") or 0), []).append(
                {"experience_id": experience_id, **record}
            )
    waves: list = []
    for iteration in sorted(grouped):
        rows = sorted(grouped[iteration], key=lambda item: item["paper"])
        waves.append(
            {
                "wave": iteration,
                "papers": [row["paper"] for row in rows],
                "library_before": {},
                "library_after": {},
                "candidates": sum(len(row["candidate_experience_ids"]) for row in rows),
                "examined": len(measured.get(iteration) or []),
                "validations": measured.get(iteration) or [],
                "promoted_between_waves": None,
                "wave_file": str(arm_dir / CONTINUOUS_STATE_FILE),
                "inherited": sum(1 for row in rows if row["reuse_accepted"]),
                # A continuous round has no boundary snapshot, so what it read is counted off
                # the entries themselves: the experiences each paper actually carried in.
                "injected_in_generation": sum(len(row["experiences_injected"]) for row in rows),
                "generation_usd": round(sum(row["generation_usd"] or 0.0 for row in rows), 4),
                "rows": rows,
                "delivered": sum(1 for row in rows if row["delivered"]),
            }
        )
    return {
        "experiment": 5,
        "arm": arm_dir.name,
        "arm_dir": str(arm_dir),
        "source": "continuous_state",
        "unit": "iteration",
        "wave_size": None,
        "generations": len(waves),
        "papers": sum(len(wave["papers"]) for wave in waves),
        "delivered": sum(wave["delivered"] for wave in waves),
        "generation_usd": round(
            sum(row["generation_usd"] or 0.0 for wave in waves for row in wave["rows"]), 4
        ),
        "how_to_fill_the_scores": HOW_TO_FILL_THE_SCORES,
        "waves": waves,
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the Experiment 5 input skeleton for one row.")
    parser.add_argument("--run-root", default="<DATA_ROOT>/abl", help="the run root the row was launched with")
    parser.add_argument("--arm", required=True, help="the row whose wave files are read")
    parser.add_argument("--wave-size", type=int, default=DEFAULT_WAVE_SIZE)
    parser.add_argument("--papers", default="", help="paper list to fall back on before the first wave file exists")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    papers = None
    if args.papers and Path(args.papers).is_file():
        papers = [json.loads(line) for line in Path(args.papers).read_text(encoding="utf-8").splitlines() if line.strip()]
    arm_dir = Path(args.run_root) / "out" / args.arm
    skeleton = build_skeleton(arm_dir, wave_size=args.wave_size, papers=papers)

    out_json = arm_dir / "experiment5_inputs.json"
    out_rows = arm_dir / "experiment5_papers.jsonl"
    if arm_dir.is_dir():
        out_json.write_text(json.dumps(skeleton, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with out_rows.open("w", encoding="utf-8") as fh:
            for wave in skeleton["waves"]:
                for row in wave["rows"]:
                    fh.write(json.dumps({"wave": wave["wave"], **row}, ensure_ascii=False) + "\n")

    if args.json:
        print(json.dumps(skeleton, ensure_ascii=False))
        return 0

    print("arm=%s unit=%s source=%s generations=%d papers=%d delivered=%d cost=$%.2f"
          % (skeleton["arm"], skeleton.get("unit") or "wave", skeleton["source"], skeleton["generations"],
             skeleton["papers"], skeleton["delivered"], skeleton["generation_usd"]))
    for wave in skeleton["waves"]:
        read = wave.get("injectable_at_start")
        carried = ("injectable_at_start=%d" % read) if read is not None else (
            "injected_in_generation=%d" % wave.get("injected_in_generation", 0)
        )
        inherited = "" if "inherited" not in wave else " inherited=%d" % wave["inherited"]
        print("  generation %02d: papers=%d delivered=%d %s examined=%d validations=%d%s"
              % (wave["wave"], len(wave["papers"]), wave["delivered"], carried,
                 wave["examined"], len(wave["validations"]), inherited))
    if skeleton["source"] == "wave_files":
        print("wrote %s and %s" % (out_json, out_rows))
    else:
        print("no wave file written yet; the grouping above is recomputed from the paper list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

