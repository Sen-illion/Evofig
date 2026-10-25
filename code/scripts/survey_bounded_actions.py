"""Measure whether each bounded repair action actually improves a figure.

The self-evolution loop assumes its ten bounded actions are repairs. Cross-paper
validation showed `relayout_global` rolling back on every case and actively
regressing two of them, so that assumption needs evidence before more machinery
is built on top of it.

For each (paper, action) pair this forces the action as round 1 via the same
seed-experience path the ON/OFF harness uses, then records what the frozen judge
decided. Results stream to JSONL so a partial run is still usable, and existing
rows are skipped so the survey can be resumed.

Usage:
    python scripts/survey_bounded_actions.py --out out/action_survey
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from rfs.evolution.direct.actions import SUPPORTED_ACTIONS
from rfs.utils import read_json
from rfs.workflows.paper_evolve_editable import run_paper_evolve_editable

# Papers that already have a completed baseline run to replay upstream stages from.
DEFAULT_PAPERS = {
    "nlp_acl64": "out/stage2_validation/run/nlp_acl64/off",
    "nlp_emnlp25": "out/stage2_validation/run/nlp_emnlp25/off",
    "materials_2d": "out/stage2_validation/run/materials_2d/off",
    "nlp_emnlp25_smoke": "out/stage1_memory_typed",
}

# Which object ids each action needs to have anything to act on.
ACTION_SCOPE = {
    "reanalyze_semantics": ("semantic", "semantic_node_ids"),
    "replan_structure": ("global", None),
    "resegment": ("global", None),
    "relayout_global": ("global", None),
    "relayout_slots": ("slot", "slot_ids"),
    "resize_nodes": ("global", "object_ids"),
    "reroute": ("connector", "edge_ids"),
    "repair_interfaces": ("interface", "edge_ids"),
    "regen_assets": ("slot", "slot_ids"),
    "recompile": ("global", None),
}

RUN_OPTIONS = {
    "planner_mode": "heuristic", "method_mode": "heuristic", "structure_mode": "heuristic",
    "text_plan_mode": "heuristic", "ppt_plan_mode": "heuristic", "color_theme_mode": "fallback",
    "reference_mode": "off", "icon_mode": "placeholder", "judge_mode": "heuristic",
    "candidates_per_round": 1, "icon_candidates": 1, "max_rounds": 2,
    "fixed_rounds": True, "run_seed": 1729, "experience_enabled": False,
}


def _target_ids(baseline: Path) -> dict[str, list[str]]:
    """Collect real object ids from a baseline run so scoped actions have targets."""
    program = read_json(baseline / "figure_program.json") if (baseline / "figure_program.json").exists() else {}
    return {
        "slot_ids": [str(item.get("id")) for item in program.get("slots", []) if isinstance(item, dict) and item.get("id")][:1],
        "object_ids": [str(item.get("id")) for item in program.get("semantic_nodes", []) if isinstance(item, dict) and item.get("id")][:2],
        "edge_ids": [str(item.get("id")) for item in program.get("arrows", []) if isinstance(item, dict) and item.get("id")][:1],
        "semantic_node_ids": [str(item.get("id")) for item in program.get("semantic_nodes", []) if isinstance(item, dict) and item.get("id")][:1],
    }


def _build_action(action_type: str, ids: dict[str, list[str]]) -> dict:
    scope_type, id_key = ACTION_SCOPE[action_type]
    action = {"type": action_type, "scope_type": scope_type, "problem": f"survey: force {action_type}"}
    if id_key:
        action[id_key] = ids.get(id_key, [])
    return action


def _round_one_outcome(run_dir: Path) -> dict:
    """Read what the frozen judge decided about the forced round-1 repair."""
    rounds = (read_json(run_dir / "evolution_rounds.json") or {}).get("rounds") or []
    by_index = {int(item.get("round", -1)): item for item in rounds if isinstance(item, dict)}
    first, second = by_index.get(0) or {}, by_index.get(1) or {}
    decision = second.get("repair_decision") or {}
    executed = (second.get("bounded_change") or {}).get("action") or {}
    before = (first.get("frozen_judgement") or {}).get("dimension_scores") or {}
    after = (second.get("frozen_judgement") or {}).get("dimension_scores") or {}
    return {
        "reached_round_1": bool(second),
        "executed_action": str(executed.get("type") or ""),
        "action_applied": bool((second.get("bounded_change") or {}).get("applied")),
        "accepted": bool(second.get("accepted")),
        "reason": str(decision.get("reason") or ""),
        "before_score": decision.get("before_score"),
        "after_score": decision.get("after_score"),
        "score_delta": decision.get("score_delta"),
        "new_blockers": decision.get("new_blockers"),
        "dimension_deltas": {k: round(float(after[k]) - float(before[k]), 4) for k in sorted(set(before) & set(after)) if isinstance(before[k], (int, float)) and isinstance(after[k], (int, float))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/action_survey")
    parser.add_argument("--papers", nargs="*", default=sorted(DEFAULT_PAPERS))
    parser.add_argument("--actions", nargs="*", default=sorted(SUPPORTED_ACTIONS))
    parser.add_argument("--replay-upstream", action="store_true", help="Reuse the baseline's upstream stages; requires a replay-compatible baseline.")
    args = parser.parse_args()

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results_path = root / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((row.get("paper"), row.get("action")))

    for paper_key in args.papers:
        baseline = Path(DEFAULT_PAPERS[paper_key]).resolve()
        pdf = baseline / "inputs" / "paper.pdf"
        if not pdf.exists():
            print(f"[skip] {paper_key}: no baseline paper at {pdf}", flush=True)
            continue
        ids = _target_ids(baseline)
        for action_type in args.actions:
            if (paper_key, action_type) in done:
                print(f"[done] {paper_key} / {action_type}", flush=True)
                continue
            run_dir = root / paper_key / action_type
            row = {"paper": paper_key, "action": action_type, "run_dir": str(run_dir), "target_ids": ids}
            try:
                run_paper_evolve_editable(
                    paper=str(pdf), out=str(run_dir),
                    replay_upstream_from=str(baseline) if args.replay_upstream else None,
                    seed_experience={"experience_id": f"survey-{action_type}", "action": _build_action(action_type, ids)},
                    **RUN_OPTIONS,
                )
                row.update({"ok": True, **_round_one_outcome(run_dir)})
            except Exception as exc:  # a failing action is itself a survey result
                row.update({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-800:]})
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{'ok ' if row.get('ok') else 'ERR'}] {paper_key} / {action_type}: accepted={row.get('accepted')} delta={row.get('score_delta')} reason={row.get('reason') or row.get('error','')[:60]}", flush=True)


if __name__ == "__main__":
    main()
