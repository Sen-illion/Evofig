"""The two tiers one round's advice lands in, and the rung a measurement earns.

A round answers one question in two halves: what to change about this figure, and what a later
figure should be drawn by instead. These tests pin the two halves apart, pin which rung may reach
another paper, and pin the paper a same-domain row still has to face.
"""

import json
import time
import unittest
from pathlib import Path
import tempfile

from rfs.contracts.evolution import normalize_experience
from rfs.contracts.prompt_patch import PROMPT_PATCH_VERSION, prompt_patch_id
from rfs.contracts.suggestion_tiers import (
    LOCAL_FIX_SCOPE,
    PORTABLE_RULE_SCOPE,
    suggestion_tiers_record,
    validate_suggestion_tiers,
)
from rfs.evolution.direct.prompt_patch import (
    CURRENT_CASE_STATUS,
    DEFAULT_ASSEMBLY_STATUSES,
    OWN_CASE_ASSEMBLY_STATUSES,
    assemble_prompt_patches,
    patch_work_permit,
    select_own_case_candidates,
)
from rfs.evolution.direct import ExperienceStore
from rfs.utils import write_json, write_text
from rfs.workflows.continuous_evolution import CONTINUOUS_EVENTS, CONTINUOUS_STATE, run_continuous_evolution


def _patch(*, patch_id: str = "patch-1", status: str = "candidate", apply_at: str = "ppt_compilation") -> dict:
    return {
        "patch_id": patch_id,
        "version": PROMPT_PATCH_VERSION,
        "apply_at": apply_at,
        "class": "advisory",
        "directive": "when the connectors cross panels, run each one down a shared channel",
        "source_experience_ids": ["direct-exp-1"],
        "median_repair_gain": 0.2,
        "status": status,
        "provenance": {"source": "round", "run": "/run", "measured": True, "round": 1},
    }


class TwoTierRecordTests(unittest.TestCase):
    def test_a_round_lands_the_fix_it_ran_and_the_rule_it_may_not_run(self):
        record = suggestion_tiers_record(
            3,
            local_fix={
                "apply_at": "ppt_compilation",
                "requirement": "a connector leaves its node on the side that faces its target",
                "directive": "route every connector down a channel shared with its neighbours",
                "action": {"type": "reroute", "scope_type": "connector"},
                "evidence": [{"stage": "ppt_compilation", "field": "stage_trace.metrics.crossing_count", "value": 9}],
                "executed": True,
            },
            portable_rule={"apply_at": "ppt_compilation", "directive": "route shared channels", "class": "advisory"},
            authored_by="model",
            transferable_half=True,
        )

        self.assertEqual("this_paper", record["local_fix"]["scope"])
        self.assertTrue(record["local_fix"]["runs_here"])
        self.assertEqual(PORTABLE_RULE_SCOPE, record["portable_rule"]["scope"])
        self.assertFalse(record["portable_rule"]["runs_here"])
        self.assertFalse(record["portable_rule"]["travels"])
        self.assertTrue(record["transferable_half"])
        self.assertEqual([], validate_suggestion_tiers(record))

    def test_a_rule_that_rendered_a_patch_travels(self):
        record = suggestion_tiers_record(
            4,
            local_fix={"apply_at": "segmentation", "requirement": "the main flow is drawn large"},
            portable_rule={"patch_id": "patch-1", "apply_at": "segmentation", "class": "advisory"},
        )
        self.assertTrue(record["portable_rule"]["travels"])
        self.assertEqual([], validate_suggestion_tiers(record))

    def test_a_stage_the_pipeline_does_not_run_is_a_contract_error(self):
        record = suggestion_tiers_record(1, local_fix={"apply_at": "nowhere"}, portable_rule={})
        errors = validate_suggestion_tiers(record)
        self.assertTrue(any("outside the pipeline order" in item for item in errors), errors)

    def test_the_portable_tier_may_not_claim_it_ran_here(self):
        record = suggestion_tiers_record(1, local_fix={"apply_at": "segmentation"}, portable_rule={"apply_at": "segmentation"})
        # The builder always writes the tier's own flags; a record that contradicts them is what
        # the checker is for, so the contradiction is written by hand here.
        record["portable_rule"]["runs_here"] = True
        self.assertTrue(any("may not run in the round" in item for item in validate_suggestion_tiers(record)))


class InjectionRungTests(unittest.TestCase):
    def test_a_candidate_is_injected_nowhere_even_on_its_own_paper(self):
        # The rungs a run may draw from are the contract, not the status a caller happens to pass:
        # a candidate is not among them, so it is assembled nowhere.
        selection = assemble_prompt_patches([_patch(status="candidate")], apply_at="ppt_compilation", statuses=OWN_CASE_ASSEMBLY_STATUSES)
        self.assertEqual([], selection["patches"])
        own_case = assemble_prompt_patches([_patch(status=CURRENT_CASE_STATUS)], apply_at="ppt_compilation", statuses=OWN_CASE_ASSEMBLY_STATUSES)
        self.assertEqual(1, len(own_case["patches"]), "the own-case rung is drawable at all; which paper may draw it is the scoping's job")

    def test_the_own_case_rung_reaches_its_own_paper_and_no_other(self):
        rows = [normalize_experience({"experience_id": "direct-exp-1", "applicability": "common"})]
        patch = _patch(status=CURRENT_CASE_STATUS)
        mine = select_own_case_candidates([patch], paper_key="/papers/mine", permitted_ids={"patch-1"})
        theirs = select_own_case_candidates([patch], paper_key="/papers/theirs", permitted_ids=set())
        self.assertEqual(["patch-1"], [item["patch_id"] for item in mine])
        self.assertEqual([], theirs)

    def test_only_a_validated_rung_reaches_another_paper(self):
        rows = [
            {"patch_id": "p-candidate", "version": 1, "apply_at": "ppt_compilation", "class": "advisory", "directive": "x", "source_experience_ids": ["e"], "median_repair_gain": 0.1, "status": "candidate", "provenance": {"source": "r", "run": "/run", "measured": True}},
            {"patch_id": "p-promoted", "version": 1, "apply_at": "ppt_compilation", "class": "advisory", "directive": "y", "source_experience_ids": ["e"], "median_repair_gain": 0.1, "status": "promoted", "provenance": {"source": "r", "run": "/run", "measured": True}},
        ]
        selection = assemble_prompt_patches(rows, apply_at="ppt_compilation", statuses=DEFAULT_ASSEMBLY_STATUSES)
        self.assertEqual(["p-promoted"], [item["patch_id"] for item in selection["patches"]])


class WorkPermitTests(unittest.TestCase):
    """A patch earns its rung from the measurement behind it, never from the words."""

    def test_a_row_with_no_measurement_is_granted_nothing(self):
        permit = patch_work_permit({"accepted": False, "repair_gain": None}, _patch())
        self.assertFalse(permit["granted"])

    def test_a_measured_gain_is_granted_the_own_case_rung(self):
        permit = patch_work_permit({"accepted": True, "repair_gain": 0.3}, _patch())
        self.assertTrue(permit["granted"])
        self.assertEqual(CURRENT_CASE_STATUS, permit["status"])

    def test_a_prohibition_needs_a_bound_a_machine_can_check(self):
        unenforceable = _patch(patch_id="p-bad")
        unenforceable["class"] = "prohibited"
        unenforceable["condition"] = {"defect": "interface:box_pass"}
        self.assertFalse(patch_work_permit({"accepted": False}, unenforceable)["granted"])
        enforceable = dict(unenforceable)
        enforceable["condition"] = {"metric": "max_cross_panel_connectors", "op": "<=", "value": 2.0}
        self.assertTrue(patch_work_permit({"accepted": False}, enforceable)["granted"])


def _config(**overrides) -> dict:
    block = {
        "base_papers": [{"paper": "/papers/one.pdf", "paper_id": "one", "domain": "biology"}],
        "budget_hours": 24.0,
        "iteration_rounds": 2,
        "evolve": {"judge_mode": "vlm"},
    }
    block.update(overrides)
    return {"continuous": block}


def _file_candidate(out_dir: Path, experience_id: str, options: dict, *, status: str) -> None:
    write_json(out_dir / "round_00" / "experience_candidate.json", {"experience_id": experience_id, "applicability": "common"})
    library = Path(str(options.get("experience_library") or ""))
    library.mkdir(parents=True, exist_ok=True)
    row = {"experience_id": experience_id, "applicability": "common", "status": status}
    with (library / "judge_experiences.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _evolve_runner(experience_id: str, status: str):
    def runner(**options):
        out = Path(options["out"])
        out.mkdir(parents=True, exist_ok=True)
        best = out / "round_00" / "candidate_00"
        best.mkdir(parents=True, exist_ok=True)
        write_json(out / "evolution_state.json", {"best_candidate_dir": str(best), "best_score": 1.0})
        write_text(out / "final_draft_preview.png", "png")
        _file_candidate(out, experience_id, options, status=status)
        return {"ok": True, "out_dir": str(out), "rounds_completed": 2}

    return runner


def _events(out: Path) -> list[dict]:
    path = out / CONTINUOUS_EVENTS
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _wait_for(predicate, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class CrossDomainRungTests(unittest.TestCase):
    """The one other-domain paper, and the rung a pass on it earns."""

    def test_a_second_paper_preferring_the_row_is_the_cross_domain_rung(self):
        experience = "direct-exp-cross"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)

            def validate(**options):
                return {
                    "status": "quick_validated",
                    "validation_profile": "quick",
                    "ladder": {},
                    "summary": {"measured_cases": 1},
                    "all_pairs_isolated": True,
                    "all_cases_measured": True,
                }

            run_continuous_evolution(
                _config(validation={"suite": "/suites/quick.json", "validation_profile": "quick"}),
                out,
                experience_library=str(out / "library"),
                evolve_runner=_evolve_runner(experience, CURRENT_CASE_STATUS),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=validate,
                clock=lambda: 0.0,
                max_waves=1,
            )

            self.assertTrue(_wait_for(lambda: any(item["event"] == "cross_domain_validated" for item in _events(out))))
            statuses = [str(row.get("status") or "") for row in ExperienceStore(str(out / "library")).records_with_status()]
            self.assertIn("cross_domain_validated", statuses)

    def test_a_row_that_stopped_at_the_same_domain_rung_is_owed_one_more_paper(self):
        experience = "direct-exp-same"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            seen: list[str] = []

            def validate(**options):
                seen.append(str(options.get("validation_profile") or ""))
                return {
                    "status": "same_domain_validated",
                    "ladder": {"stopped_because": "cross_domain_generalization_not_shown"},
                    "summary": {"measured_cases": 3},
                    "all_pairs_isolated": True,
                    "all_cases_measured": True,
                }

            run_continuous_evolution(
                _config(validation={"suite": "/suites/full.json"}),
                out,
                experience_library=str(out / "library"),
                evolve_runner=_evolve_runner(experience, "candidate"),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=validate,
                clock=lambda: 0.0,
                max_waves=1,
            )

            # The verdict is settled by the validation worker, which is a thread: the call can
            # return before the worker has read the result, so the queue entry is waited for
            # rather than assumed. The wait is on the event because the worker writes the state
            # entry first and the event second, so an event that exists means the state does too.
            self.assertTrue(
                _wait_for(lambda: any(item["event"] == "cross_domain_follow_up_queued" for item in _events(out)), timeout=20),
                _events(out),
            )
            queued = [item for item in _events(out) if item["event"] == "cross_domain_follow_up_queued"]
            self.assertEqual(experience, queued[0]["experience_id"])
            # The follow-up runs behind the same single worker, so no second heavy job is added
            # beside a running one. It is marked in the state, so a call that ends before the
            # worker reaches it leaves the work owed rather than lost.
            state = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            self.assertEqual("queued", state["cross_domain_follow_ups"][experience])

            # The next call queues it before anything else and measures it at the cheap profile.
            run_continuous_evolution(
                _config(validation={"suite": "/suites/full.json"}),
                out,
                experience_library=str(out / "library"),
                evolve_runner=_evolve_runner(experience, "same_domain_validated"),
                review_runner=lambda *_a: {"ok": False, "reason": "", "stage_feedback": []},
                validate_runner=validate,
                clock=lambda: 0.0,
                max_waves=1,
            )
            self.assertTrue(_wait_for(lambda: "quick" in seen, timeout=20), seen)
            decided = json.loads((out / CONTINUOUS_STATE).read_text(encoding="utf-8"))
            self.assertEqual("refused", decided["cross_domain_follow_ups"][experience])


if __name__ == "__main__":
    unittest.main()