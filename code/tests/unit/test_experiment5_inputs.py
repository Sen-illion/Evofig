"""Unit tests for scripts/experiment5_inputs.py.

Experiment 5 reads the complete row's deliveries grouped by generation, so the grouping and
what the row read at each boundary are the two things that must not be invented. These tests
pin both for each shape a row writes: a wave row groups by its own wave files and takes the
"injectable" count from the snapshot those files carry, a continuous row groups by the
iteration numbers in its state file and counts what each round carried in, and a row with no
generation record yet is recomputed from the paper list and says so.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "experiment5_inputs", ROOT / "scripts" / "experiment5_inputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inputs = load_module()


class Experiment5InputsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name) / "out" / "ours_own500"

    def tearDown(self):
        self._tmp.cleanup()

    def _delivery(self, domain, paper, *, delivered=True, cost=1.25, rounds=1):
        paper_dir = self.root / domain / paper
        paper_dir.mkdir(parents=True, exist_ok=True)
        if delivered:
            (paper_dir / "editable_composition.pptx").write_bytes(b"pptx")
        (paper_dir / "cost_report.json").write_text(
            json.dumps({"elapsed_seconds": 120.0, "generation_usd": cost}), encoding="utf-8"
        )
        (paper_dir / "evolution_state.json").write_text(
            json.dumps({"rounds_completed": rounds}), encoding="utf-8"
        )
        return paper_dir

    def _wave(self, index, papers, *, promoted=0, examined=0):
        waves = self.root / "waves"
        waves.mkdir(parents=True, exist_ok=True)
        (waves / ("wave_%02d.json" % index)).write_text(
            json.dumps(
                {
                    "wave": index,
                    "papers": papers,
                    "library_before": {"by_status": {"promoted": promoted}} if promoted else {"rows": 0},
                    "library_after": {"by_status": {"promoted": promoted}},
                    "candidates": ["a", "b"],
                    "examined": ["a"] * examined,
                    "validations": [{"experience_id": "a", "rc": 0}] if examined else [],
                    "promoted_between_waves": bool(examined),
                }
            ),
            encoding="utf-8",
        )

    def test_wave_files_group_papers_and_carry_the_library_snapshot(self):
        self._delivery("agent", "p1")
        self._delivery("biology", "p2")
        self._delivery("agent", "p3", delivered=False, cost=None)
        self._wave(1, ["p1", "p2"], promoted=0)
        self._wave(2, ["p3"], promoted=7, examined=2)

        skeleton = inputs.build_skeleton(self.root)

        self.assertEqual("wave_files", skeleton["source"])
        self.assertEqual(2, skeleton["generations"])
        self.assertEqual(3, skeleton["papers"])
        self.assertEqual(2, skeleton["delivered"])
        self.assertEqual(0, skeleton["waves"][0]["injectable_at_start"])
        self.assertEqual(7, skeleton["waves"][1]["injectable_at_start"])
        self.assertEqual(2, skeleton["waves"][1]["examined"])
        self.assertTrue(skeleton["waves"][1]["promoted_between_waves"])
        self.assertEqual("p3", skeleton["waves"][1]["rows"][0]["paper_id"])
        self.assertEqual(1, skeleton["waves"][1]["rows"][0]["rounds_completed"])
        self.assertFalse(skeleton["waves"][1]["rows"][0]["delivered"])
        self.assertIsNone(skeleton["waves"][0]["rows"][0]["nine"])

    def test_a_paper_delivered_under_the_round_directory_counts(self):
        paper_dir = self.root / "agent" / "p4"
        (paper_dir / "round_00" / "candidate_00").mkdir(parents=True)
        (paper_dir / "round_00" / "candidate_00" / "editable_composition.pptx").write_bytes(b"pptx")
        self._wave(1, ["p4"])

        skeleton = inputs.build_skeleton(self.root)

        self.assertTrue(skeleton["waves"][0]["rows"][0]["delivered"])

    def test_no_wave_file_yet_is_recomputed_from_the_list_and_says_so(self):
        self._delivery("agent", "p1")
        papers = [{"paper_id": "p1"}, {"paper_id": "p2"}, {"paper_id": "p3"}]

        skeleton = inputs.build_skeleton(self.root, wave_size=2, papers=papers)

        self.assertEqual("recomputed_from_list", skeleton["source"])
        self.assertEqual([2, 1], [len(wave["papers"]) for wave in skeleton["waves"]])
        self.assertEqual(0, skeleton["waves"][0]["injectable_at_start"])
        self.assertEqual([], skeleton["waves"][0]["validations"])
        self.assertEqual(1, skeleton["delivered"])

    def test_missing_library_snapshot_reads_as_zero_injectable(self):
        self.assertEqual(0, inputs._injectable({}))
        self.assertEqual(0, inputs._injectable({"by_status": {"rejected": 4}}))
        self.assertEqual(0, inputs._injectable({"by_status": {"promoted": "none"}}))
        self.assertEqual(4, inputs._injectable({"by_status": {"promoted": 4}}))

    def test_a_paper_named_by_id_is_found_under_its_sanitised_directory(self):
        self._delivery("agent", "iclr_2026_10010667")
        self._wave(1, ["iclr:2026:10010667"])

        skeleton = inputs.build_skeleton(self.root)

        row = skeleton["waves"][0]["rows"][0]
        self.assertEqual("iclr:2026:10010667", row["paper_id"])
        self.assertTrue(row["delivered"])
        self.assertIn("iclr_2026_10010667", row["delivery"])

    def test_the_sanitiser_is_the_drivers_own(self):
        spec = importlib.util.spec_from_file_location(
            "run_ablation_arm_safe_check", ROOT / "scripts" / "run_ablation_arm.py"
        )
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)

        for name in ("iclr:2026:10010667", "10.1145/3770855.3818074", "plain", "", "a b", "x.y_z-1"):
            self.assertEqual(driver.safe(name), inputs.safe(name), name)

    def _iteration(self, name, iteration, *, inherited_from="", reuse_accepted=None, usd=0.5,
                   delivered=True, injected=(), candidates=(), rounds=1, declined_reason=""):
        """One iteration's directory and the state entry the loop would have written for it.

        A round with nothing to inherit is not a round whose inheritance was declined, so the
        verdict defaults to whatever ``inherited_from`` says rather than to "accepted".
        """

        if reuse_accepted is None:
            reuse_accepted = bool(inherited_from)

        iter_dir = self.root / "papers" / name / ("iter_%02d" % iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)
        if delivered:
            (iter_dir / "editable_composition.pptx").write_bytes(b"pptx")
        (iter_dir / "cost_report.json").write_text(
            json.dumps({"elapsed_seconds": 200.0, "generation_usd": usd}), encoding="utf-8"
        )
        (iter_dir / "evolution_state.json").write_text(
            json.dumps({"rounds_completed": rounds}), encoding="utf-8"
        )
        return {
            "paper": name,
            "iteration": iteration,
            "out": str(iter_dir),
            "best": str(iter_dir),
            "reuse_from": inherited_from,
            "reuse": {"accepted": reuse_accepted, "declined_reason": declined_reason},
            "accepted_rounds": 1 if reuse_accepted else 0,
            "rounds": rounds,
            "experiences_injected": list(injected),
            "candidate_experience_ids": list(candidates),
            "archive_size": 2,
        }

    def _state(self, histories, *, validations=None, paper_ids=None):
        papers = {}
        for name, records in histories.items():
            papers[name] = {
                "paper_id": (paper_ids or {}).get(name, name),
                "history": records,
            }
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "continuous_state.json").write_text(
            json.dumps({"papers": papers, "validations": validations or {}}), encoding="utf-8"
        )

    def test_a_continuous_row_groups_its_iterations_by_round(self):
        first_a = self._iteration("a", 1, usd=0.4)
        second_a = self._iteration("a", 2, inherited_from=str(self.root / "papers" / "a" / "iter_01"), usd=0.6)
        first_b = self._iteration("b", 1, usd=0.2)
        self._state({"a": [first_a, second_a], "b": [first_b]})

        skeleton = inputs.build_skeleton(self.root)

        self.assertEqual("continuous_state", skeleton["source"])
        self.assertEqual("iteration", skeleton["unit"])
        self.assertEqual(2, skeleton["generations"])
        self.assertEqual(3, skeleton["papers"])
        self.assertEqual(3, skeleton["delivered"])
        self.assertAlmostEqual(1.2, skeleton["generation_usd"], places=6)
        self.assertEqual([2, 1], [len(wave["papers"]) for wave in skeleton["waves"]])
        self.assertEqual(0, skeleton["waves"][0]["inherited"])
        self.assertEqual(1, skeleton["waves"][1]["inherited"])

        continued = skeleton["waves"][1]["rows"][0]
        self.assertEqual(2, continued["iteration"])
        self.assertTrue(continued["reuse_accepted"])
        self.assertIn("iter_01", continued["inherited_from"])
        self.assertEqual(1, continued["rounds_completed"])

    def test_a_round_that_was_not_scored_is_null_never_zero(self):
        self._state({"a": [self._iteration("a", 1)]})

        row = inputs.build_skeleton(self.root)["waves"][0]["rows"][0]

        self.assertIsNone(row["nine"])
        self.assertIsNone(row["editable"])
        self.assertIsNone(row["node_f1"])
        self.assertIsNone(row["edge_f1"])

    def test_a_round_that_inherited_nothing_says_so(self):
        self._state({"a": [self._iteration("a", 1, reuse_accepted=False, declined_reason="stage_missing")]})

        row = inputs.build_skeleton(self.root)["waves"][0]["rows"][0]

        self.assertFalse(row["reuse_accepted"])
        self.assertEqual("stage_missing", row["reuse_declined_reason"])
        self.assertEqual(0, inputs.build_skeleton(self.root)["waves"][0]["inherited"])

    def test_what_a_continuous_round_carried_in_is_counted_off_its_entries(self):
        self._state({"a": [self._iteration("a", 1, injected=["exp-1", "exp-2"])]})

        wave = inputs.build_skeleton(self.root)["waves"][0]

        self.assertEqual(2, wave["injected_in_generation"])
        self.assertNotIn("injectable_at_start", wave)

    def test_a_continuous_validation_is_filed_under_its_own_iteration(self):
        self._state(
            {"a": [self._iteration("a", 1), self._iteration("a", 2, inherited_from="x")]},
            validations={"exp-1": {"stub": True, "paper": "a", "iteration": 2, "status": "promoted"}},
        )

        skeleton = inputs.build_skeleton(self.root)

        self.assertEqual([], skeleton["waves"][0]["validations"])
        self.assertEqual(1, len(skeleton["waves"][1]["validations"]))
        self.assertEqual("exp-1", skeleton["waves"][1]["validations"][0]["experience_id"])
        self.assertEqual(1, skeleton["waves"][1]["examined"])

    def test_main_writes_the_continuous_table_too(self):
        self._state({"a": [self._iteration("a", 1)]})

        code = inputs.main(["--run-root", str(self.root.parents[1]), "--arm", "ours_own500", "--json"])

        self.assertEqual(0, code)
        written = json.loads((self.root / "experiment5_inputs.json").read_text(encoding="utf-8"))
        self.assertEqual("iteration", written["unit"])
        rows = (self.root / "experiment5_papers.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(1, json.loads(rows[0])["wave"])

    def test_main_writes_both_files(self):
        self._delivery("agent", "p1")
        self._wave(1, ["p1"], promoted=0)

        code = inputs.main(
            ["--run-root", str(self.root.parents[1]), "--arm", "ours_own500", "--json"]
        )

        self.assertEqual(0, code)
        written = json.loads((self.root / "experiment5_inputs.json").read_text(encoding="utf-8"))
        self.assertEqual(1, written["generations"])
        rows = (self.root / "experiment5_papers.jsonl").read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(1, len(rows))
        self.assertEqual(1, json.loads(rows[0])["wave"])


if __name__ == "__main__":
    unittest.main()

