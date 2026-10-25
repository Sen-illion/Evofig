"""Unit tests for scripts/run_ablation_arm.py.

The driver shells out to the real product chain, so these tests cover the parts that decide
*what* gets run and *what* gets recorded: which switches an arm passes, which the protocol
refuses to let it pass, the session config it builds, and the refusals that keep a row from
starting before a human has approved it. The chain itself is never executed.
"""

import contextlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "run_ablation_arm", ROOT / "scripts" / "run_ablation_arm.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = load_driver()

PROTOCOL = {
    "schema": "rfs.experiment-protocol",
    "version": 1,
    "generation": {
        "planner_mode": "vlm",
        "read_mode": "fast",
        "method_mode": "vlm",
        "structure_mode": "vlm",
        "text_plan_mode": "vlm",
        "ppt_plan_mode": "vlm",
        "color_theme_mode": "vlm",
        "visual_review_mode": "vlm",
        "judge_mode": "vlm",
        "reference_mode": "heuristic",
        "icon_mode": "image2",
        "graph_context": "on",
        "middle_layer": "on",
        "figure_target": "overview",
        "domain_profile": "auto",
        "aspect_ratio": "16:9",
        "max_rounds": 3,
        "candidates_per_round": 1,
        "icon_candidates": 1,
        "vlm_timeout": 300,
        "vlm_retries": 2,
        "evolution_enabled": True,
        "evolution_scope": "joint",
        "experience_enabled": True,
        "reference_top_k": 5,
        "reuse_replayed_icons": "on",
    },
    "judge": {
        "advisory_checkpoint": "/ckpt/step-003174",
        "advisory_base_model": "/models/Qwen3.5-9B",
        "acceptance_checkpoint": "/ckpt/step-003174",
        "acceptance_base_model": "/models/Qwen3.5-9B",
    },
    "data": {"judge_holdout": {"lists_by_host": {"experiment": ["/nowhere/val-32.jsonl"]}}},
    "arms": [
        {"id": "full", "driver": "session", "status": "awaiting_go", "require_holdout": True,
         "experience_library": "host", "validation_suite": "host"},
        {"id": "no_memory", "driver": "session", "status": "awaiting_go", "experience_library": "",
         "overrides": {"experience_enabled": False}},
        {
            "id": "no_middle_layer", "driver": "editable", "status": "blocked",
            "status_note": "the planning-side bypass does not exist", "experience_library": "host",
        },
        {"id": "no_repair", "driver": "none", "status": "free"},
    ],
    "hosts": {
        "experiment": {
            "hostname": "box-experiment",
            "code": "/code/EvoFig-se",
            "python": "/envs/paperimage/bin/python",
            "env_file": "/code/.env",
            "run_root": "/run",
            "paper_list": "/run/own_500.jsonl",
            "experience_library": "/run/library",
            "validation_suite": "/run/validation_suite.json",
        }
    },
}


def write_protocol(body, directory):
    path = pathlib.Path(directory) / "protocol.json"
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return path


def argv_for(*extra):
    return ["run_ablation_arm.py", *extra]


class LoadProtocolTests(unittest.TestCase):
    def test_digest_names_the_bytes_that_were_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_protocol(PROTOCOL, tmp)
            body, digest = driver.load_protocol(path)
            self.assertEqual(body["schema"], "rfs.experiment-protocol")
            self.assertEqual(len(digest), 64)
            path.write_text(json.dumps(PROTOCOL, ensure_ascii=False) + " ", encoding="utf-8")
            _, again = driver.load_protocol(path)
            self.assertNotEqual(digest, again)

    def test_shipped_protocol_parses_and_names_the_arms(self):
        body, digest = driver.load_protocol(ROOT / "configs" / "experiment_protocol.json")
        self.assertEqual(body["schema"], "rfs.experiment-protocol")
        ids = [arm["id"] for arm in body["arms"]]
        self.assertEqual(
            ids,
            [
                "full",
                "no_evolve",
                "no_memory",
                "no_memory_product",
                "no_middle_layer",
                "no_repair",
                "unit_cost",
                "ours_own500",
                "ours_sciflow",
                "ours_figurebench",
                "bestofn_1",
                "bestofn_2",
                "bestofn_3",
                "bestofn_4",
                "topk_5",
                "topk_3",
                "topk_8",
            ],
        )
        self.assertEqual(len(digest), 64)
        for arm in body["arms"]:
            self.assertIn(arm["status"], ("runnable", "awaiting_go", "blocked", "free"))

    def test_the_no_memory_row_reads_the_legacy_batch_instead_of_running(self):
        """The user's call: that row is the 2026-10-04 batch, with the product twin kept aside."""

        body, _ = driver.load_protocol(ROOT / "configs" / "experiment_protocol.json")
        arm = driver.arm_by_id(body, "no_memory")
        self.assertEqual(arm["driver"], "none")
        self.assertEqual(arm["status"], "free")
        self.assertIn("2026-10-04", arm["status_note"])
        self.assertIn("10-04", arm["reads"])
        twin = driver.arm_by_id(body, "no_memory_product")
        self.assertEqual(twin["driver"], "none")
        self.assertEqual(twin["status"], "free")
        self.assertIn("ours_own500", twin["reads"])

    def test_the_no_middle_layer_row_differs_from_the_main_row_once(self):
        """The draft rows are a controlled pair: one switch apart, so a gap is that switch."""

        body, _ = driver.load_protocol(ROOT / "configs" / "experiment_protocol.json")
        host = body["hosts"]["main"]

        def flags(arm_id):
            arm = driver.arm_by_id(body, arm_id)
            cmd = driver.editable_command(body, arm, host, "/p.pdf", "/out", [])
            pairs = {}
            index = 0
            while index < len(cmd):
                if cmd[index].startswith("--"):
                    if index + 1 < len(cmd) and not cmd[index + 1].startswith("--"):
                        pairs[cmd[index]] = cmd[index + 1]
                        index += 2
                    else:
                        pairs[cmd[index]] = ""
                        index += 1
                else:
                    index += 1
            return pairs

        main = flags("ours_own500")
        ablation = flags("no_middle_layer")
        differing = sorted(
            key for key in set(main) | set(ablation) if main.get(key) != ablation.get(key)
        )
        self.assertEqual(differing, ["--middle-layer"])
        self.assertEqual(main["--middle-layer"], "on")
        self.assertEqual(ablation["--middle-layer"], "off")
        # Neither draft row seats a trained Judge or reads the experience library, so neither pays
        # for a seat and the row is the figure it drew, which is what makes the one switch readable.
        for row in (main, ablation):
            self.assertEqual(row["--acceptance-checkpoint"], "")
            self.assertEqual(row["--advisory-checkpoint"], "")
            self.assertIn("--no-experience", row)

    def test_the_self_evolution_rows_run_in_waves_of_one_size(self):
        """Waves are what make a row self-evolving, so all three rows have to interleave alike."""

        body, _ = driver.load_protocol(ROOT / "configs" / "experiment_protocol.json")
        rows = ("full", "no_evolve", "no_middle_layer")
        self.assertEqual({driver.arm_by_id(body, key)["driver"] for key in rows}, {"waves"})
        self.assertEqual({driver.arm_by_id(body, key)["wave_size"] for key in rows}, {driver.DEFAULT_WAVE_SIZE})
        self.assertEqual((driver.arm_by_id(body, "no_evolve").get("overrides") or {}).get("validate"), False)

    def test_the_no_middle_layer_row_turns_the_explicit_structure_off(self):
        """Its one switch is the paper's explicit structure, so the driver has to pass it."""

        body, _ = driver.load_protocol(ROOT / "configs" / "experiment_protocol.json")
        arm = driver.arm_by_id(body, "no_middle_layer")
        host = body["hosts"]["main"]
        self.assertEqual(arm["status"], "awaiting_go")
        self.assertIn("--middle-layer off", arm["status_note"])
        self.assertEqual((arm.get("overrides") or {}).get("middle_layer"), "off")
        cmd = driver.editable_command(body, arm, host, "/p.pdf", "/out", [])
        self.assertEqual(cmd[cmd.index("--middle-layer") + 1], "off")
        # The row leaves the figure target alone: if a benchmark run needs the
        # paper's own panels it is set in the arm, not smuggled in here.
        self.assertEqual(cmd[cmd.index("--figure-target") + 1], "overview")
        full = driver.arm_by_id(body, "full")
        full_cmd = driver.editable_command(body, full, host, "/p.pdf", "/out", [])
        self.assertEqual(full_cmd[full_cmd.index("--middle-layer") + 1], "on")


class HostTests(unittest.TestCase):
    def test_host_block_is_chosen_by_hostname_or_key(self):
        key, block = driver.host_section(PROTOCOL, hostname="box-experiment")
        self.assertEqual((key, block["code"]), ("experiment", "/code/EvoFig-se"))
        self.assertEqual(driver.host_section(PROTOCOL, key="experiment")[1]["run_root"], "/run")

    def test_an_unknown_machine_is_refused_instead_of_guessed(self):
        with self.assertRaises(SystemExit):
            driver.host_section(PROTOCOL, hostname="somewhere-else")

    def test_holdout_lists_are_filed_per_host_and_never_borrowed(self):
        self.assertEqual(
            driver.holdout_lists(PROTOCOL, "experiment"), ([], ["/nowhere/val-32.jsonl"])
        )
        with self.assertRaises(SystemExit):
            driver.holdout_lists(PROTOCOL, "main")

    def test_a_paper_list_is_chosen_by_the_arm_and_the_host(self):
        host = dict(PROTOCOL["hosts"]["experiment"])
        host["paper_lists"] = {"sciflow": "/run/sciflow_500.jsonl"}
        self.assertEqual(driver.paper_list_for(host, "sciflow"), "/run/sciflow_500.jsonl")
        self.assertEqual(driver.paper_list_for(host, ""), "/run/own_500.jsonl")

    def test_a_list_the_host_does_not_file_is_refused_not_substituted(self):
        host = dict(PROTOCOL["hosts"]["experiment"])
        host["paper_lists"] = {"own_500": "/run/own_500.jsonl"}
        with self.assertRaises(SystemExit):
            driver.paper_list_for(host, "sciflow")

    def test_the_shipped_protocol_files_the_same_five_exams_on_every_host(self):
        body, _ = driver.load_protocol(ROOT / "configs" / "experiment_protocol.json")
        by_host = body["data"]["judge_holdout"]["lists_by_host"]
        # `main-de` is the same machine as `main` under a checkout of its own, and `experiment4`
        # is the four-card box, so every key files the same exams; the key exists because the
        # guard is chosen by host key, not by hostname.
        self.assertEqual(sorted(by_host), ["experiment", "experiment4", "main", "main-de"])
        names = {key: [path.rsplit("/", 1)[1] for path in paths] for key, paths in by_host.items()}
        for key in sorted(by_host):
            self.assertEqual(len(names[key]), 5)
            self.assertEqual(names[key], names["main"])


class ForkAndLibraryTests(unittest.TestCase):
    """The two switches the inference-budget arms add, and the refusals around them."""

    def _shipped(self):
        body, _ = driver.load_protocol(ROOT / "configs" / "experiment_protocol.json")
        return body

    def test_a_forked_arm_inherits_one_papers_directory_from_its_source_arm(self):
        body = self._shipped()
        arm = driver.arm_by_id(body, "bestofn_4")
        row = {"domain": "cv", "paper_id": "2401.0001v1"}
        source, stage = driver.replay_prefix(arm, "/run", row)
        self.assertEqual(stage, "segmentation")
        # Compared piecewise: os.path.join spells the same tree with the local separator, and the
        # arm reads it on Linux where the two spellings are the same string.
        self.assertEqual(
            [piece for piece in source.replace("\\", "/").split("/") if piece],
            ["run", "out", "bestofn_1", "cv", "2401.0001v1"],
        )

    def test_a_baseline_arm_inherits_nothing(self):
        body = self._shipped()
        self.assertEqual(driver.replay_prefix(driver.arm_by_id(body, "bestofn_1"), "/run", {}), ("", ""))

    def test_a_fork_without_a_stage_is_refused_instead_of_run_uncontrolled(self):
        with self.assertRaises(SystemExit):
            driver.replay_prefix({"id": "x", "replay_from_arm": "bestofn_1"}, "/run", {})

    def test_the_library_comes_from_the_run_host(self):
        body = self._shipped()
        arm = driver.arm_by_id(body, "topk_8")
        host = body["hosts"]["main-de"]
        library = driver.library_for(arm, host)
        self.assertEqual(library, host["reference_library"])
        self.assertIn("reference_library.jsonl", library)

    def test_an_arm_reading_a_library_refuses_a_host_that_files_none(self):
        body = self._shipped()
        arm = driver.arm_by_id(body, "topk_8")
        with self.assertRaises(SystemExit):
            driver.library_for(arm, dict(body["hosts"]["experiment"]))

    def test_the_fork_switches_are_only_written_when_there_is_a_prefix(self):
        body = self._shipped()
        arm = driver.arm_by_id(body, "bestofn_2")
        host = body["hosts"]["experiment"]
        cmd = driver.editable_command(body, arm, host, "/p.pdf", "/out", [])
        self.assertNotIn("--replay-upstream-from", cmd)
        self.assertEqual(cmd[cmd.index("--candidates-per-round") + 1], "2")
        cmd = driver.editable_command(
            body, arm, host, "/p.pdf", "/out", [], replay_from="/src", fork_before="segmentation"
        )
        self.assertEqual(cmd[cmd.index("--replay-upstream-from") + 1], "/src")
        self.assertEqual(cmd[cmd.index("--fork-before") + 1], "segmentation")

    def test_a_topk_arm_hands_the_library_before_the_suffixes(self):
        body = self._shipped()
        arm = driver.arm_by_id(body, "topk_3")
        host = body["hosts"]["main-de"]
        cmd = driver.editable_command(body, arm, host, "/p.pdf", "/out", [])
        self.assertEqual(cmd[cmd.index("--reference-library") + 1], host["reference_library"])
        self.assertEqual(cmd[cmd.index("--reference-top-k") + 1], "3")

    def test_the_two_draft_rows_read_the_gallery_the_user_picked(self):
        body = self._shipped()
        host = body["hosts"]["main"]
        self.assertTrue(host["reference_library"].endswith("reference_library.jsonl"))
        for arm_id in ("ours_own500", "no_middle_layer"):
            arm = driver.arm_by_id(body, arm_id)
            self.assertEqual(arm.get("reference_library"), "host", arm_id)
            self.assertEqual(driver.library_for(arm, host), host["reference_library"])

    def test_a_draft_row_reading_the_gallery_refuses_a_host_that_files_none(self):
        body = self._shipped()
        arm = driver.arm_by_id(body, "ours_own500")
        with self.assertRaises(SystemExit):
            driver.library_for(arm, dict(body["hosts"]["experiment"]))

    def test_the_draft_rows_hand_the_gallery_to_the_command(self):
        body = self._shipped()
        host = body["hosts"]["main"]
        arm = driver.arm_by_id(body, "ours_own500")
        cmd = driver.editable_command(body, arm, host, "/p.pdf", "/out", [])
        self.assertEqual(cmd[cmd.index("--reference-library") + 1], host["reference_library"])


class UpstreamContractTests(unittest.TestCase):
    """The middle layer's tree is read per paper, and a paper it misses runs on its own."""

    def _shipped(self):
        return json.loads(
            (ROOT / "configs" / "experiment_protocol.json").read_text(encoding="utf-8")
        )

    def test_the_tree_is_resolved_per_paper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "graphs"
            (root / "paper-a").mkdir(parents=True)
            host = {"upstream_graphs": str(root)}

            directory = driver.upstream_contract_for(
                host, {"paper_id": "paper-a", "pdf_path": "/papers/paper-a.pdf"}
            )

            self.assertEqual(directory, str(root / "paper-a"))

    def test_a_paper_the_tree_does_not_cover_runs_on_its_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "graphs"
            root.mkdir()
            host = {"upstream_graphs": str(root)}

            self.assertEqual(driver.upstream_contract_for(host, {"paper_id": "absent"}), "")

    def test_a_host_that_files_no_tree_passes_nothing(self):
        self.assertEqual(driver.upstream_contract_for({}, {"paper_id": "paper-a"}), "")
        self.assertEqual(
            driver.upstream_contract_for({"upstream_graphs": "/nope"}, {"paper_id": "paper-a"}), ""
        )

    def test_the_shipped_main_host_names_the_middle_layers_tree(self):
        body = self._shipped()
        root = body["hosts"]["main"]["upstream_graphs"]

        self.assertIn("graph_middleware", root)
        self.assertTrue(root.endswith("/graphs"), root)
        # The retrieval-depth arms are filed on `main-de` and must stay on their own
        # ruler: the middleware's inventory would be a second difference in their pair.
        self.assertNotIn("upstream_graphs", body["hosts"]["main-de"])

    def test_the_command_carries_the_directory_only_when_there_is_one(self):
        body = self._shipped()
        arm = driver.arm_by_id(body, "ours_own500")
        host = body["hosts"]["main"]

        cmd = driver.editable_command(body, arm, host, "/p.pdf", "/out", [])
        self.assertNotIn("--upstream-from", cmd)

        cmd = driver.editable_command(
            body, arm, host, "/p.pdf", "/out", [], upstream_contract="/graphs/paper-a"
        )
        self.assertEqual(cmd[cmd.index("--upstream-from") + 1], "/graphs/paper-a")


class SettingsTests(unittest.TestCase):
    def test_an_arm_delta_sits_on_top_of_the_generation_block(self):
        arm = driver.arm_by_id(PROTOCOL, "no_memory")
        values = driver.settings_for(PROTOCOL, arm)
        self.assertFalse(values["experience_enabled"])
        self.assertEqual(values["planner_mode"], "vlm")

    def test_unknown_arm_is_refused(self):
        with self.assertRaises(SystemExit):
            driver.arm_by_id(PROTOCOL, "no_such_arm")

    def test_the_host_sentinel_reads_the_run_host(self):
        host = PROTOCOL["hosts"]["experiment"]
        self.assertEqual(driver.from_host("host", host, "experience_library"), "/run/library")
        self.assertEqual(driver.from_host("", host, "experience_library"), "")


class ReadListTests(unittest.TestCase):
    def write_list(self, directory, count):
        path = pathlib.Path(directory) / "papers.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for index in range(count):
                fh.write(json.dumps({"paper_id": "p%02d" % index, "pdf_path": "/pdfs/%d.pdf" % index}) + "\n")
        return path

    def test_a_stride_spans_the_whole_list_instead_of_the_first_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_list(tmp, 20)
            rows = driver.read_list(str(path), 4, 5)
            self.assertEqual([row["paper_id"] for row in rows], ["p00", "p05", "p10", "p15"])

    def test_without_a_stride_the_first_papers_are_taken(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_list(tmp, 20)
            rows = driver.read_list(str(path), 2)
            self.assertEqual([row["paper_id"] for row in rows], ["p00", "p01"])

    def test_the_cost_sample_asks_for_a_stride(self):
        body, _ = driver.load_protocol(ROOT / "configs" / "experiment_protocol.json")
        arm = driver.arm_by_id(body, "unit_cost")
        self.assertEqual(arm["limit"], 20)
        self.assertGreater(arm["stride"], 1)


class RuntimeEnvTests(unittest.TestCase):
    def test_prose_keys_are_not_exported_as_variables(self):
        body = json.loads(json.dumps(PROTOCOL))
        body["runtime_env"] = {"RFS_PREVIEW_RENDER": "auto", "_note": "prose, not a variable"}
        self.assertEqual(driver.runtime_env(body), {"RFS_PREVIEW_RENDER": "auto"})

    def test_the_shipped_protocol_pins_the_rate_limit_sleep(self):
        body, _ = driver.load_protocol(ROOT / "configs" / "experiment_protocol.json")
        self.assertEqual(driver.runtime_env(body).get("RFS_VLM_RATE_LIMIT_SLEEP_SECONDS"), "30")


class EditableCommandTests(unittest.TestCase):
    def test_every_switch_comes_from_the_protocol(self):
        arm = driver.arm_by_id(PROTOCOL, "no_memory")
        host = PROTOCOL["hosts"]["experiment"]
        cmd = driver.editable_command(PROTOCOL, arm, host, "/p.pdf", "/out", [])
        self.assertIn("--planner-mode", cmd)
        self.assertEqual(cmd[cmd.index("--planner-mode") + 1], "vlm")
        self.assertEqual(cmd[cmd.index("--candidates-per-round") + 1], "1")
        self.assertEqual(cmd[cmd.index("--max-rounds") + 1], "3")
        self.assertIn("--no-experience", cmd)
        self.assertNotIn("--experience-library", cmd)
        self.assertEqual(cmd[cmd.index("--advisory-checkpoint") + 1], "/ckpt/step-003174")
        self.assertEqual(cmd[cmd.index("--acceptance-base-model") + 1], "/models/Qwen3.5-9B")
        self.assertEqual(cmd[-1], "--json")

    def test_changing_the_protocol_changes_the_command(self):
        body = json.loads(json.dumps(PROTOCOL))
        body["generation"]["candidates_per_round"] = 2
        arm = driver.arm_by_id(body, "no_memory")
        host = body["hosts"]["experiment"]
        cmd = driver.editable_command(body, arm, host, "/p.pdf", "/out", [])
        self.assertEqual(cmd[cmd.index("--candidates-per-round") + 1], "2")

    def test_an_arm_that_keeps_the_library_refuses_without_one(self):
        arm = {"id": "custom", "experience_library": ""}
        host = {"experience_library": ""}
        with self.assertRaises(SystemExit):
            driver.editable_command(PROTOCOL, arm, host, "/p.pdf", "/out", [])

    def test_evolution_off_is_written_as_a_value_not_a_missing_flag(self):
        body = json.loads(json.dumps(PROTOCOL))
        arm = {"id": "custom", "experience_library": "host", "overrides": {"evolution_enabled": False}}
        host = body["hosts"]["experiment"]
        cmd = driver.editable_command(body, arm, host, "/p.pdf", "/out", [])
        self.assertEqual(cmd[cmd.index("--evolution") + 1], "off")


class SessionTests(unittest.TestCase):
    PAPERS = [
        {"paper_id": "10.1038/x", "domain": "biology", "pdf_path": "/pdfs/x.pdf"},
        {"paper_id": "10.1038/y", "domain": "biology", "pdf_path": "/pdfs/y.pdf"},
    ]

    def test_one_evolve_station_per_paper_carries_the_arm_delta(self):
        arm = driver.arm_by_id(PROTOCOL, "no_memory")
        body = driver.session_config(PROTOCOL, arm, self.PAPERS, [], "s1")
        self.assertEqual(body["session_id"], "s1")
        self.assertEqual(len(body["stations"]), 2)
        for station in body["stations"]:
            self.assertEqual(station["kind"], "evolve")
            self.assertFalse(station["experience_enabled"])
            self.assertEqual(station["planner_mode"], "vlm")
            self.assertEqual(station["online_judge_checkpoint"], "/ckpt/step-003174")

    def test_holdout_lists_that_exist_are_handed_to_the_stations(self):
        arm = driver.arm_by_id(PROTOCOL, "no_memory")
        body = driver.session_config(PROTOCOL, arm, self.PAPERS, ["/run/val-32.jsonl"], "s1")
        self.assertEqual(body["stations"][0]["judge_holdout_paths"], ["/run/val-32.jsonl"])

    def test_no_validate_is_added_only_for_the_arm_that_drops_the_ladder(self):
        host = PROTOCOL["hosts"]["experiment"]
        flat = driver.arm_by_id(PROTOCOL, "full")
        self.assertNotIn("--no-validate", driver.session_command(PROTOCOL, flat, host, "/c.json", "/o", []))
        evolved = {"id": "no_evolve", "driver": "session", "overrides": {"validate": False}, "experience_library": "host", "validation_suite": "host"}
        self.assertIn("--no-validate", driver.session_command(PROTOCOL, evolved, host, "/c.json", "/o", []))


class WaveTests(unittest.TestCase):
    """The driver that lets a paper read what earlier papers earned without going serial."""

    def run_main(self, protocol_path, *extra):
        with mock.patch.object(driver.sys, "argv", argv_for("--protocol", str(protocol_path), *extra)):
            return driver.main()

    def test_a_list_is_cut_into_waves_that_keep_its_order(self):
        rows = [{"paper_id": "p%02d" % index} for index in range(1, 8)]
        waves = driver.wave_slices(rows, 3)
        self.assertEqual([len(wave) for wave in waves], [3, 3, 1])
        self.assertEqual([row["paper_id"] for row in waves[0]], ["p01", "p02", "p03"])
        self.assertEqual([row["paper_id"] for row in waves[2]], ["p07"])

    def test_an_arm_that_names_no_wave_size_gets_the_default_one(self):
        rows = [{"paper_id": "p%03d" % index} for index in range(1, driver.DEFAULT_WAVE_SIZE + 2)]
        self.assertEqual([len(wave) for wave in driver.wave_slices(rows, 0)], [driver.DEFAULT_WAVE_SIZE, 1])

    def test_a_wave_reads_the_candidates_its_papers_wrote(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = {"paper_id": "10.1/x", "domain": "cv"}
            out = pathlib.Path(driver.paper_out(tmp, row))
            (out / "round_00").mkdir(parents=True)
            (out / "round_00" / "experience_candidate.json").write_text(
                json.dumps({"experience_id": "direct-exp-1", "applicability": " Domain "}), encoding="utf-8"
            )
            (out / "round_01").mkdir()
            (out / "round_01" / "experience_candidate.json").write_text("{not json", encoding="utf-8")
            unrun = {"paper_id": "10.1/y", "domain": "cv"}
            self.assertEqual(driver.wave_candidates([row, unrun], tmp), {"direct-exp-1": "domain"})

    def test_the_validate_command_takes_its_seats_from_the_suite(self):
        arm = driver.arm_by_id(PROTOCOL, "full")
        host = PROTOCOL["hosts"]["experiment"]
        cmd = driver.validate_command(PROTOCOL, arm, host, "direct-exp-1", "/waves/validate/1")
        self.assertEqual(cmd[cmd.index("--suite") + 1], host["validation_suite"])
        self.assertEqual(cmd[cmd.index("--experience-library") + 1], host["experience_library"])
        self.assertEqual(cmd[cmd.index("--experience-id") + 1], "direct-exp-1")
        self.assertEqual(cmd[cmd.index("--out") + 1], "/waves/validate/1")
        self.assertEqual(cmd[-1], "--json")

    def test_a_library_snapshot_pins_the_bytes_a_wave_read(self):
        self.assertEqual(driver.library_snapshot("")["rows"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            library = pathlib.Path(tmp) / "library"
            library.mkdir()
            before = driver.library_snapshot(str(library))
            self.assertEqual(len(before["digest"]), 64)
            (library / "judge_experiences.jsonl").write_text(
                '{"experience_id": "direct-exp-1", "status": "candidate"}\n', encoding="utf-8"
            )
            after = driver.library_snapshot(str(library))
            self.assertEqual(len(after["digest"]), 64)
            self.assertNotEqual(before["digest"], after["digest"])

    def test_a_wave_row_without_a_validation_suite_refuses_to_start(self):
        """Without a suite every wave reads the same library, which is the row it is not."""

        body = json.loads(json.dumps(PROTOCOL))
        body["arms"][0]["driver"] = "waves"
        body["arms"][0]["require_holdout"] = False
        body["hosts"]["experiment"]["validation_suite"] = ""
        with tempfile.TemporaryDirectory() as tmp:
            path = write_protocol(body, tmp)
            with mock.patch.dict(driver.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
                with self.assertRaises(SystemExit) as caught:
                    self.run_main(path, "--arm", "full", "--host", "experiment", "--confirm-go", "--dry-run", "--out", tmp)
            self.assertIn("validation suite", str(caught.exception))

    def test_a_wave_plan_names_its_waves_and_pays_for_nothing(self):
        body = json.loads(json.dumps(PROTOCOL))
        body["arms"][0]["driver"] = "waves"
        body["arms"][0]["wave_size"] = 2
        body["arms"][0]["require_holdout"] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = write_protocol(body, tmp)
            listing = pathlib.Path(tmp) / "papers.jsonl"
            listing.write_text(
                "\n".join(json.dumps({"paper_id": "p%d" % index, "domain": "cv"}) for index in (1, 2, 3)),
                encoding="utf-8",
            )
            buffer = io.StringIO()
            with mock.patch.dict(driver.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
                with contextlib.redirect_stdout(buffer):
                    code = self.run_main(
                        path, "--arm", "full", "--host", "experiment", "--confirm-go", "--dry-run",
                        "--list", str(listing), "--out", tmp,
                    )
            self.assertEqual(code, 0)
            printed = buffer.getvalue()
            self.assertIn("waves=2 wave_size=2", printed)
            self.assertIn("wave 01: 2 papers", printed)
            self.assertIn("wave 02: 1 papers", printed)
            self.assertIn("paper-evolve-validate", printed)


class RefusalTests(unittest.TestCase):
    """A row must not start before a human has approved its plan, hours and cost."""

    def run_main(self, protocol_path, *extra):
        with mock.patch.object(driver.sys, "argv", argv_for("--protocol", str(protocol_path), *extra)):
            return driver.main()

    def test_a_blocked_arm_says_why_instead_of_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_protocol(PROTOCOL, tmp)
            with self.assertRaises(SystemExit) as caught:
                self.run_main(path, "--arm", "no_middle_layer", "--host", "experiment", "--dry-run")
            self.assertIn("bypass", str(caught.exception))

    def test_an_arm_waiting_for_a_go_needs_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_protocol(PROTOCOL, tmp)
            with self.assertRaises(SystemExit) as caught:
                self.run_main(path, "--arm", "no_memory", "--host", "experiment", "--dry-run")
            self.assertIn("--confirm-go", str(caught.exception))

    def test_an_arm_that_only_reads_another_arm_does_not_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_protocol(PROTOCOL, tmp)
            with self.assertRaises(SystemExit):
                self.run_main(path, "--arm", "no_repair", "--host", "experiment", "--dry-run")

    def test_a_required_guard_that_is_absent_stops_the_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_protocol(PROTOCOL, tmp)
            with self.assertRaises(SystemExit) as caught:
                self.run_main(path, "--arm", "full", "--host", "experiment", "--confirm-go", "--dry-run")
            self.assertIn("guarded", str(caught.exception))


class CardRotationTests(unittest.TestCase):
    """One paper loads both judge seats, so a row's workers have to be spread over the cards.

    Aiming a whole arm at whichever device looked emptiest when it started is what put three
    papers on one 80 GB card: two of them already sit at about 73 GB, so the third paper's
    advisory seat ran out of memory and the row went on without a judge instead of refusing.
    """

    def run_main(self, protocol_path, *extra):
        with mock.patch.object(driver.sys, "argv", argv_for("--protocol", str(protocol_path), *extra)):
            return driver.main()

    def test_a_card_carries_two_papers_and_no_more(self):
        self.assertEqual(driver.MAX_PROCESSES_PER_GPU, 2)

    def test_the_cards_a_machine_reports_are_used_in_turn(self):
        self.assertEqual(driver.gpu_rotation({"CUDA_VISIBLE_DEVICES": "0,1"}, 4), ["0", "1"])
        self.assertEqual(driver.gpu_rotation({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, 8), ["0", "1", "2", "3"])

    def test_a_pinned_card_stays_the_only_card_used(self):
        self.assertEqual(driver.gpu_rotation({"CUDA_VISIBLE_DEVICES": "3"}, 2), ["3"])

    def test_a_fan_out_no_card_can_carry_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            driver.gpu_rotation({"CUDA_VISIBLE_DEVICES": "0,1"}, 5)
        self.assertIn("37-43 GB", str(caught.exception))

    def test_a_worker_over_the_ceiling_is_refused_before_a_dry_run_prints_a_plan(self):
        body = json.loads(json.dumps(PROTOCOL))
        body["arms"][0].update({"driver": "editable", "status": "runnable", "require_holdout": False})
        with tempfile.TemporaryDirectory() as tmp:
            path = write_protocol(body, tmp)
            with mock.patch.dict(driver.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                with self.assertRaises(SystemExit) as caught:
                    self.run_main(
                        path, "--arm", "full", "--host", "experiment", "--concurrency", "3",
                        "--dry-run", "--out", tmp,
                    )
            self.assertIn("card carries at most 2", str(caught.exception))

    def test_a_row_without_seats_may_fill_a_card(self):
        self.assertEqual(driver.MAX_PROCESSES_PER_GPU_WITHOUT_SEATS, 4)
        self.assertEqual(
            driver.gpu_rotation({"CUDA_VISIBLE_DEVICES": "0,1"}, 8, with_seats=False), ["0", "1"]
        )
        with self.assertRaises(SystemExit) as caught:
            driver.gpu_rotation({"CUDA_VISIBLE_DEVICES": "0,1"}, 9, with_seats=False)
        self.assertIn("seats no trained Judge", str(caught.exception))

    def test_each_paper_of_a_row_is_launched_on_its_own_card(self):
        body = json.loads(json.dumps(PROTOCOL))
        body["arms"][0].update(
            {"driver": "editable", "status": "runnable", "require_holdout": False,
             "concurrency": 4, "experience_library": "host"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_protocol(body, tmp)
            listing = pathlib.Path(tmp) / "papers.jsonl"
            listing.write_text(
                "\n".join(
                    json.dumps({"paper_id": "p%d" % index, "domain": "cv", "pdf_path": "/pdfs/%d.pdf" % index})
                    for index in range(1, 5)
                ),
                encoding="utf-8",
            )
            handed = []

            class Done:
                returncode = 0
                stdout = ""
                stderr = ""

            def fake_run(cmd, cwd=None, env=None, **kwargs):
                handed.append(env.get("CUDA_VISIBLE_DEVICES"))
                return Done()

            with mock.patch.dict(driver.os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
                with mock.patch.object(driver, "require_judge_assets", lambda *a, **k: None):
                    with mock.patch.object(driver, "load_env", lambda *a, **k: {"CUDA_VISIBLE_DEVICES": "0,1"}):
                        with mock.patch.object(driver.subprocess, "run", fake_run):
                            self.run_main(
                                path, "--arm", "full", "--host", "experiment", "--out", tmp, "--list", str(listing)
                            )
            self.assertEqual(sorted(handed), ["0", "0", "1", "1"])


class ReadModeTests(unittest.TestCase):
    """The driver passes the reading mode, and the command it builds has to parse."""

    def test_the_command_the_driver_builds_is_one_the_product_cli_parses(self):
        from rfs.cli import build_parser

        arm = {"id": "row", "experience_library": "host", "overrides": {"read_mode": "full"}}
        cmd = driver.editable_command(
            PROTOCOL, arm, PROTOCOL["hosts"]["experiment"], "/pdfs/a.pdf", "/out/a", []
        )

        parsed = build_parser().parse_args(cmd[3:])

        self.assertEqual(parsed.read_mode, "full")
        self.assertEqual(parsed.paper, "/pdfs/a.pdf")
        self.assertEqual(parsed.out, "/out/a")

    def test_a_row_that_names_no_mode_reads_the_paper_the_shared_way(self):
        arm = {"id": "row", "experience_library": "host"}
        cmd = driver.editable_command(
            PROTOCOL, arm, PROTOCOL["hosts"]["experiment"], "/pdfs/a.pdf", "/out/a", []
        )
        self.assertEqual(cmd[cmd.index("--read-mode") + 1], "fast")


class JudgeSeatTests(unittest.TestCase):
    """The two seats are the only local model a round loads, so an arm has to be able to drop them.

    A row whose delivery is the figure the round drew and nothing the seats decided pays for two
    9B seats per paper and never reads their answer. This is the switch that takes them off, and
    the two tests that matter are that the switch reaches the command line and that a row which
    names a Judge key the driver does not know is refused instead of quietly measured by the
    seats it thought it had turned off.
    """

    def seats_off(self):
        return {"id": "row", "judge": {"advisory_checkpoint": "", "acceptance_checkpoint": ""}}

    def test_an_arm_without_a_judge_block_inherits_the_shared_seats(self):
        arm = {"id": "row"}
        self.assertTrue(driver.seats_a_trained_judge(PROTOCOL, arm))
        self.assertEqual(
            driver.judge_settings(PROTOCOL, arm)["acceptance_checkpoint"], "/ckpt/step-003174"
        )

    def test_an_arm_that_blanks_the_checkpoints_seats_no_trained_judge(self):
        self.assertFalse(driver.seats_a_trained_judge(PROTOCOL, self.seats_off()))
        judge = driver.judge_settings(PROTOCOL, self.seats_off())
        self.assertEqual(judge["advisory_checkpoint"], "")
        self.assertEqual(judge["acceptance_checkpoint"], "")
        # A key the arm did not name is left as the shared block has it.
        self.assertEqual(judge["acceptance_base_model"], "/models/Qwen3.5-9B")

    def test_a_judge_key_the_driver_does_not_know_is_refused(self):
        arm = {"id": "row", "judge": {"acceptance_chekpoint": ""}}
        with self.assertRaises(SystemExit) as caught:
            driver.judge_settings(PROTOCOL, arm)
        self.assertIn("does not know", str(caught.exception))

    def test_the_command_carries_the_seats_the_arm_asked_for(self):
        arm = dict(self.seats_off(), experience_library="host")
        cmd = driver.editable_command(
            PROTOCOL, arm, PROTOCOL["hosts"]["experiment"], "/pdfs/a.pdf", "/out/a", []
        )

        def value_of(flag):
            return cmd[cmd.index(flag) + 1]

        self.assertEqual(value_of("--acceptance-checkpoint"), "")
        self.assertEqual(value_of("--advisory-checkpoint"), "")
        self.assertEqual(value_of("--judge-mode"), "vlm")

    def test_a_row_without_seats_is_not_asked_for_the_judge_assets(self):
        body = json.loads(json.dumps(PROTOCOL))
        body["arms"][0].update(
            {"driver": "editable", "status": "runnable", "require_holdout": False,
             "experience_library": "", "concurrency": 1,
             "overrides": {"experience_enabled": False},
             "judge": {"advisory_checkpoint": "", "acceptance_checkpoint": ""}}
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_protocol(body, tmp)
            listing = pathlib.Path(tmp) / "papers.jsonl"
            listing.write_text(
                json.dumps({"paper_id": "p1", "domain": "cv", "pdf_path": "/pdfs/1.pdf"}),
                encoding="utf-8",
            )
            asked = []

            class Done:
                returncode = 0
                stdout = ""
                stderr = ""

            with mock.patch.dict(driver.os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                with mock.patch.object(driver, "require_judge_assets", lambda *a, **k: asked.append(a)):
                    with mock.patch.object(driver, "load_env", lambda *a, **k: {"CUDA_VISIBLE_DEVICES": "0"}):
                        with mock.patch.object(driver.subprocess, "run", lambda *a, **k: Done()):
                            with mock.patch.object(
                                driver.sys, "argv",
                                argv_for("--protocol", str(path), "--arm", "full", "--host", "experiment",
                                         "--out", tmp, "--list", str(listing)),
                            ):
                                self.assertEqual(driver.main(), 0)
            self.assertEqual(asked, [])
            stamp = json.loads(
                (pathlib.Path(tmp) / "out" / "full" / "protocol.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stamp["judge_seats"], "heuristic")
            self.assertEqual(stamp["judge_overrides"]["acceptance_checkpoint"], "")


class MainRowProtocolTests(unittest.TestCase):
    """The main row and its no-memory twin are one run, which is a property of the protocol."""

    def setUp(self):
        self.shipped = json.loads(
            (ROOT / "configs" / "experiment_protocol.json").read_text(encoding="utf-8")
        )

    def test_the_main_own_set_row_fills_all_four_cards(self):
        main = driver.arm_by_id(self.shipped, "ours_own500")
        self.assertEqual(main["driver"], "waves")
        self.assertEqual(main["wave_size"], 50)
        # Four papers on each of the four cards is the ceiling a row without seats carries.
        self.assertEqual(main["concurrency"], 16)
        self.assertEqual(main["papers"], "own_500")
        self.assertFalse(main["overrides"].get("validate", True))
        # Two papers a card was the rule while every paper loaded two 9B seats. This row seats
        # none, so the ceiling that applies to it is the one the chain itself sets.
        self.assertFalse(driver.seats_a_trained_judge(self.shipped, main))

    def test_the_no_memory_row_reads_the_main_row_instead_of_running(self):
        twin = driver.arm_by_id(self.shipped, "no_memory_product")
        self.assertEqual(twin["status"], "free")
        self.assertEqual(twin["driver"], "none")
        self.assertIn("ours_own500", twin["reads"])

    def test_the_four_comparable_rows_share_a_wave_size_and_a_library(self):
        wanted = ("ours_own500", "no_middle_layer")
        for arm_id in wanted:
            arm = driver.arm_by_id(self.shipped, arm_id)
            self.assertEqual(arm["wave_size"], 50, arm_id)
            self.assertEqual(arm["experience_library"], "host", arm_id)
            self.assertEqual(arm["concurrency"], 16, arm_id)

    def test_the_draft_rows_take_the_trained_seats_off(self):
        for arm_id in ("ours_own500", "no_middle_layer"):
            arm = driver.arm_by_id(self.shipped, arm_id)
            self.assertEqual(
                arm["judge"], {"advisory_checkpoint": "", "acceptance_checkpoint": ""}, arm_id
            )
            self.assertFalse(driver.seats_a_trained_judge(self.shipped, arm), arm_id)
            self.assertEqual(arm["overrides"]["judge_mode"], "heuristic", arm_id)
            self.assertFalse(arm["overrides"]["evolution_enabled"], arm_id)
            self.assertFalse(arm["overrides"]["experience_enabled"], arm_id)
            self.assertIn("paper-evolve-continuous", arm["status_note"], arm_id)

    def test_every_other_row_still_runs_the_trained_judge(self):
        # The two draft rows are the only rows that changed: the rows already measured, and the
        # rows still to come, keep the seats they were designed around.
        for arm in self.shipped["arms"]:
            if arm["id"] in ("ours_own500", "no_middle_layer"):
                continue
            self.assertTrue(driver.seats_a_trained_judge(self.shipped, arm), arm["id"])

    def test_the_icon_endpoint_is_pinned_with_the_memory_hint(self):
        env = self.shipped["runtime_env"]
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertIn("<API_HOST>", env["GEMINI_GEN_IMG_URL"])



class ArmRulerTests(unittest.TestCase):
    """A row may name an environment of its own; today that is the icon model it draws under.

    Two families of rows are being drawn by different image models on purpose, so the protocol's
    shared block has to keep stating one default and every row that departs from it has to say so
    where the row is filed. Otherwise a delivery drawn by the cheap model is read as a delivery of
    the family it is being compared against.
    """

    def setUp(self):
        self.shipped = json.loads(
            (ROOT / "configs" / "experiment_protocol.json").read_text(encoding="utf-8")
        )

    def test_an_arms_env_is_layered_over_the_shared_block(self):
        protocol = {"runtime_env": {"A": "1", "B": "2", "_prose": "not a variable"}}
        env = driver.runtime_env(protocol, {"env": {"B": "3", "_note": "prose", "C": "4"}})
        self.assertEqual(env, {"A": "1", "B": "3", "C": "4"})

    def test_an_arm_without_an_env_gets_the_shared_block(self):
        protocol = {"runtime_env": {"A": "1", "_prose": "x"}}
        self.assertEqual(driver.runtime_env(protocol), {"A": "1"})
        self.assertEqual(driver.runtime_env(protocol, {"id": "plain"}), {"A": "1"})

    def test_the_owning_rows_draw_under_the_cheap_icon_model(self):
        for arm_id in (
            "full", "no_evolve", "no_middle_layer", "unit_cost",
            "ours_own500", "ours_sciflow", "ours_figurebench",
        ):
            arm = driver.arm_by_id(self.shipped, arm_id)
            self.assertEqual((arm.get("env") or {}).get("RFS_IMAGE_MODEL"), "z-image-turbo", arm_id)

    def test_the_axis_rows_keep_the_ruler_they_were_measured_with(self):
        for arm_id in ("bestofn_1", "bestofn_2", "bestofn_3", "bestofn_4", "topk_3", "topk_5", "topk_8"):
            arm = driver.arm_by_id(self.shipped, arm_id)
            self.assertFalse((arm.get("env") or {}).get("RFS_IMAGE_MODEL"), arm_id)
        self.assertNotEqual(self.shipped["runtime_env"]["RFS_IMAGE_MODEL"], "z-image-turbo")

if __name__ == "__main__":
    unittest.main()
