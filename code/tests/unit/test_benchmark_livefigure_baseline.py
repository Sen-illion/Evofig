"""Unit tests for scripts/benchmark_livefigure_baseline.py.

The harness talks to a model and to the authors' checkout, so these tests cover
the parts that decide what the model sees and what comes back into the benchmark:
prompt loading, the paper-only conversion contract, JSONL shape, deck selection and
the collect step. No model call and no LiveFigure code is executed.
"""

import importlib.util
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location("benchmark_livefigure_baseline", ROOT / "scripts" / "benchmark_livefigure_baseline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = load_module()


class PromptTests(unittest.TestCase):
    def test_the_shipped_prompt_has_the_paper_slot(self):
        template = harness.load_prompt(ROOT / "benchmarks/editable-figure-bench/baselines/livefigure/prompts/describe-v001.md")
        self.assertIn(harness.PAPER_MARKER, template)
        self.assertGreater(len(template), 200)

    def test_a_template_without_the_slot_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "prompt.md"
            path.write_text("no slot here", encoding="utf-8")
            with self.assertRaises(ValueError):
                harness.load_prompt(path)

    def test_build_prompt_only_fills_the_paper_slot(self):
        built = harness.build_prompt("before {paper_text} after", "PAPER BODY")
        self.assertEqual("before PAPER BODY after", built)

    def test_fingerprint_is_stable_and_content_bound(self):
        first = harness.prompt_fingerprint("template")
        self.assertEqual(first, harness.prompt_fingerprint("template"))
        self.assertNotEqual(first, harness.prompt_fingerprint("template "))

    def test_the_baseline_config_pins_the_upstream_commit(self):
        config = json.loads((ROOT / "benchmarks/editable-figure-bench/baselines/livefigure/baseline.json").read_text(encoding="utf-8"))
        self.assertEqual("livefigure", config["baseline"])
        self.assertEqual(40, len(config["pinned_commit"]))
        self.assertIsNone(config["license"])
        self.assertIn("V2_CaptionAndContext", config["conditions"])


class DescriptionTests(unittest.TestCase):
    def test_a_complete_answer_is_accepted_and_trimmed(self):
        caption, description = harness.validate_description({"caption": " System  overview ", "description": " A -> B "})
        self.assertEqual("System  overview", caption)
        self.assertEqual("A -> B", description)

    def test_a_missing_key_is_refused(self):
        with self.assertRaises(ValueError):
            harness.validate_description({"caption": "only this"})
        with self.assertRaises(ValueError):
            harness.validate_description({"description": "   "})
        with self.assertRaises(ValueError):
            harness.validate_description("not an object")

    def test_the_dataset_line_maps_a_case_directory_back_to_the_instance(self):
        line = harness.dataset_line("2401.00001", "cap", "desc")
        self.assertEqual("2401.00001", line["paper_name"])
        self.assertEqual("Fig1", line["figure_label"])
        self.assertEqual("cap", line["caption"])
        self.assertEqual("desc", line["description"])
        self.assertEqual(harness.case_directory("/lf", "2401.00001").name, "2401.00001_Fig1")

    def test_the_conversion_step_never_reads_the_gold_contract(self):
        """Only the paper reaches the model: the gold is not part of the baseline input."""
        source = (ROOT / "scripts/benchmark_livefigure_baseline.py").read_text(encoding="utf-8")
        describe_one = source.split("def describe_one")[1].split("def run_describe")[0]
        self.assertNotIn("contract", describe_one)
        self.assertIn("paper.pdf", describe_one)

    def test_the_prompt_asks_for_a_paper_only_conversion(self):
        template = harness.load_prompt(ROOT / "benchmarks/editable-figure-bench/baselines/livefigure/prompts/describe-v001.md")
        self.assertIn("Never invent", template)
        self.assertIn("Use only what the paper states", template)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _case(self, paper_id="2401.00001"):
        case = self.root / "livefigure" / (paper_id + "_Fig1")
        (case / "V2_CaptionAndContext").mkdir(parents=True)
        return case

    def _instances(self, ids):
        for paper_id in ids:
            child = self.root / "instances" / paper_id
            child.mkdir(parents=True)
            (child / "contract.json").write_text("{}", encoding="utf-8")

    def test_select_deck_prefers_the_newest_generated_deck(self):
        case = self._case()
        older = case / "V2_CaptionAndContext/01_code_iter_0_try_0.pptx"
        newer = case / "V2_CaptionAndContext/03_code_iter_2_try_0.pptx"
        older.write_bytes(b"old")
        newer.write_bytes(b"new")
        (case / "V2_CaptionAndContext/notes.pptx").write_bytes(b"ignored")
        self.assertEqual(newer, harness.select_deck(case))

    def test_select_deck_reports_nothing_when_the_case_has_not_run(self):
        self.assertIsNone(harness.select_deck(self.root / "absent"))

    def test_select_preview_prefers_the_deck_directory(self):
        case = self._case()
        deck = case / "V2_CaptionAndContext/01_code_iter_0_try_0.pptx"
        deck.write_bytes(b"deck")
        far = case / "other/preview.png"
        far.parent.mkdir(parents=True)
        far.write_bytes(b"far")
        near = case / "V2_CaptionAndContext/render.png"
        near.write_bytes(b"near")
        self.assertEqual(near, harness.select_preview(case, deck))

    def test_collect_writes_the_benchmark_system_layout(self):
        self._instances(["2401.00001", "2401.00002"])
        case = self._case("2401.00001")
        deck = case / "V2_CaptionAndContext/01_code_iter_0_try_0.pptx"
        deck.write_bytes(b"deck")
        (case / "V2_CaptionAndContext/preview.png").write_bytes(b"png")
        args = SimpleNamespace(
            livefigure_root=str(self.root / "livefigure"),
            instances=str(self.root / "instances"),
            systems=str(self.root / "systems"),
            system_id="livefigure",
            json=False,
        )
        self.assertEqual(0, harness.run_collect(args))
        target = self.root / "systems/livefigure/2401.00001"
        self.assertEqual(b"deck", (target / "editable_composition.pptx").read_bytes())
        self.assertEqual(b"png", (target / "preview.png").read_bytes())
        self.assertFalse((self.root / "systems/livefigure/2401.00002").exists())
        report = json.loads((self.root / "systems/livefigure_collect_report.json").read_text(encoding="utf-8"))
        self.assertEqual(1, report["collected"])
        self.assertEqual(["2401.00002"], report["missing"])

    def test_instances_are_only_directories_with_a_contract(self):
        self._instances(["b", "a"])
        (self.root / "instances/no-contract").mkdir()
        self.assertEqual(["a", "b"], harness.discover_instances(self.root / "instances"))
        self.assertEqual(["a"], harness.discover_instances(self.root / "instances", limit=1))

    def test_a_missing_instance_root_is_an_error(self):
        with self.assertRaises(ValueError):
            harness.discover_instances(self.root / "nope")


if __name__ == "__main__":
    unittest.main()
