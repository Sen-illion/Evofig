"""Choice-item quotas, menus, and short-answer checkpoint selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.judge_training.choice_items import (
    CHOICE_DEFECTS,
    FAMILY_QUOTAS,
    build_choice_slices,
    choice_is_better,
    choice_json,
    menu_defect,
    parse_choice,
    export_figure_preferences,
    preference_records,
    prompt_leaks_answer,
    render_pair,
)
from rfs.judge_training.training import StepOutcome, TrainingConfig, train_dpo


def _example(family_paper: str, defect: str, *, domain: str = "cv") -> dict:
    kind = "assumed_worse" if defect == "no_defect" else "missing_relation" if defect == "wrong_arrow_target" else defect
    return {
        "example_id": f"pair-{family_paper}",
        "paper_id": family_paper,
        "domain": domain,
        "task_type": "pairwise_preference",
        "candidate_a": f"/tmp/{family_paper}-good.png",
        "candidate_b": f"/tmp/{family_paper}-bad.png",
        "pair_group": f"group-{family_paper}",
        "paper": {"title": f"Title {family_paper}", "abstract": "A method overview."},
        "labels": {
            "preferred_candidate": "A",
            "diagnosis": [{"figure": "B", "type": kind, "target": "node"}],
        },
        "provenance": {"corruption_type": kind},
    }


def _pool(prefix: str, defect: str, count: int, *, domain: str = "cv") -> list[dict]:
    return [_example(f"{prefix}-{defect}-{index}", defect, domain=domain) for index in range(count)]


def _sources(root: Path) -> dict[str, str]:
    corruption = []
    for defect in ("module_order_error", "overlap", "wrong_arrow_target", "terminology_corruption", "missing_core_entity", "missing_core_stage", "correct_but_ugly"):
        corruption.extend(_pool("c", defect, 8))
    rag = []
    for domain, count in (("agent", 6), ("biology", 6), ("cv", 6), ("materials", 6), ("rag", 6)):
        rag.extend(_pool("r", f"{domain}", count, domain=domain))
    # rag pool above uses defect name equal to domain, which is not a menu defect.
    # Rebuild rag with a real defect and varying domains.
    rag = []
    domains = ("agent", "biology", "cv", "materials", "rag")
    for index in range(30):
        rag.append(_example(f"r-{index}", "module_order_error", domain=domains[index % 5]))
    sealed = []
    for defect in ("module_order_error", "overlap", "wrong_arrow_target", "terminology_corruption", "missing_core_stage", "invented_module"):
        sealed.extend(_pool("s", defect, 4))
    sealed.append(_example("s-entity", "missing_core_entity"))
    v002 = _pool("v", "correct_but_ugly", 8) + _pool("v", "no_defect", 4) + _pool("v", "missing_core_entity", 8) + _pool("v", "terminology_corruption", 4)
    v002.append(_example("s-module_order_error-0", "missing_core_entity"))
    paths = {}
    for name, rows in (("corruption", corruption), ("rag", rag), ("sealed", sealed), ("v002", v002)):
        path = root / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        paths[name] = str(path)
    exam = root / "exam.jsonl"
    exam.write_text(json.dumps(_example("exam-paper", "overlap")) + "\n", encoding="utf-8")
    paths["exam"] = str(exam)
    return paths


class ChoiceItemTests(unittest.TestCase):
    def test_assumed_worse_is_no_defect_and_the_arrow_name_collapses(self):
        self.assertEqual(menu_defect("assumed_worse"), "no_defect")
        self.assertEqual(menu_defect("missing_relation"), "wrong_arrow_target")

    def test_menu_hides_the_answer_outside_the_option_list_and_swap_flips_the_figure(self):
        pair = {
            "pair_id": "group-1",
            "paper_id": "paper-1",
            "domain": "cv",
            "family": "corruption",
            "defect": "overlap",
            "good": "/tmp/good.png",
            "bad": "/tmp/bad.png",
            "title": "Title",
            "evidence": "A method overview.",
        }
        forward = render_pair(pair, swapped=False)
        swapped = render_pair(pair, swapped=True)
        self.assertEqual(forward["gold_figure"], "B")
        self.assertEqual(swapped["gold_figure"], "A")
        self.assertEqual(forward["defect_menu"], swapped["defect_menu"])
        self.assertEqual(len(forward["defect_menu"]), 4)
        self.assertIn("overlap", forward["defect_menu"])
        prompt = forward["messages"][0]["content"][0]["text"]
        self.assertIn("Domain: cv. ", prompt)
        self.assertFalse(prompt_leaks_answer(prompt, "overlap", forward["defect_menu"]))
        self.assertIsNone(parse_choice(prompt, forward["defect_menu"]))
        unnamed = render_pair({**pair, "domain": "unknown", "pair_id": "group-unknown"}, swapped=False)
        unnamed_prompt = unnamed["messages"][0]["content"][0]["text"]
        self.assertNotIn("Domain:", unnamed_prompt)
        self.assertFalse(prompt_leaks_answer(unnamed_prompt, "overlap", unnamed["defect_menu"]))
        self.assertEqual(swapped["messages"][0]["content"][0]["text"], prompt)
        self.assertEqual(parse_choice(forward["chosen"], forward["defect_menu"]), ("B", "overlap"))
        self.assertIsNone(parse_choice(forward["chosen"] + " because B overlaps", forward["defect_menu"]))

    def test_nested_slices_exclude_the_exam_and_drop_sealed_missing_entities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _sources(root)
            report = build_choice_slices(
                {key: paths[key] for key in ("corruption", "rag", "sealed", "v002")},
                root / "out",
                exam=paths["exam"],
                sizes=(20,),
            )
            self.assertTrue(report["ok"], report)
            train = [json.loads(line) for line in (root / "out" / "train-20.jsonl").read_text(encoding="utf-8").splitlines()]
            val = [json.loads(line) for line in (root / "out" / "val-32.jsonl").read_text(encoding="utf-8").splitlines()]
            papers = {row["paper_id"] for row in train + val}
            self.assertNotIn("exam-paper", papers)
            self.assertTrue(papers.isdisjoint({row["paper_id"] for row in train} & {row["paper_id"] for row in val}))
            self.assertEqual(report["train_pairs"]["20"], 20)
            self.assertEqual(report["val_pairs"], 32)
            self.assertTrue(all(row["defect"] in CHOICE_DEFECTS for row in train))
            sealed = [row for row in train + val if row["family"] == "sealed"]
            self.assertTrue(all(row["defect"] != "missing_core_entity" for row in sealed))
            self.assertTrue(all(row["preference"] in {"figure", "defect"} for row in train))
            self.assertTrue(all("preference" not in row for row in val))
            self.assertGreater(len(train), 40)
            self.assertTrue(all(row["preference"] == "defect" for row in train if row["defect"] == "no_defect"))

    def test_preassigned_corruption_ranks_stay_nested(self):
        from rfs.judge_training.choice_items import cut_choice_slices

        types = ("module_order_error", "overlap", "wrong_arrow_target", "terminology_corruption", "missing_core_entity", "missing_core_stage", "correct_but_ugly")
        corruption = []
        for defect in types:
            for rank in range(1, 51):
                corruption.append({"pair_id": f"c-{defect}-{rank}", "paper_id": f"c-{defect}-{rank}", "domain": "cv", "family": "corruption", "defect": defect, "good": "g", "bad": "b", "title": "t", "evidence": "e", "role": "train", "rank": rank})
            for hold in (1, 2):
                corruption.append({"pair_id": f"c-{defect}-h{hold}", "paper_id": f"c-{defect}-h{hold}", "domain": "cv", "family": "corruption", "defect": defect, "good": "g", "bad": "b", "title": "t", "evidence": "e", "role": "holdout", "rank": 0})
        rag = [{"pair_id": f"r-{i}", "paper_id": f"r-{i}", "domain": ("agent", "biology", "cv", "materials", "rag")[i % 5], "family": "rag", "defect": "module_order_error", "good": "g", "bad": "b", "title": "t", "evidence": "e"} for i in range(300)]
        sealed = [{"pair_id": f"s-{i}", "paper_id": f"s-{i}", "domain": "cv", "family": "sealed", "defect": ("module_order_error", "overlap", "wrong_arrow_target", "terminology_corruption", "missing_core_stage", "invented_module")[i % 6], "good": "g", "bad": "b", "title": "t", "evidence": "e"} for i in range(250)]
        v002_defects = ("correct_but_ugly", "no_defect", "missing_core_entity", "terminology_corruption")
        v002 = [{"pair_id": f"v-{i}", "paper_id": f"v-{i}", "domain": "cv", "family": "v002", "defect": v002_defects[i % 4], "good": "g", "bad": "b", "title": "t", "evidence": "e"} for i in range(250)]
        cut = cut_choice_slices({"corruption": corruption, "rag": rag, "sealed": sealed, "v002": v002}, sizes=(20, 200, 1000))
        self.assertTrue(cut["ok"], cut)
        self.assertEqual([item["rank"] for item in cut["train"][20]["corruption"]], [1] * 7)
        self.assertEqual(len(cut["train"][200]["corruption"]), 70)
        self.assertEqual(len(cut["train"][1000]["corruption"]), 350)
        small = {item["pair_id"] for item in cut["train"][20]["corruption"]}
        middle = {item["pair_id"] for item in cut["train"][200]["corruption"]}
        large = {item["pair_id"] for item in cut["train"][1000]["corruption"]}
        self.assertTrue(small < middle < large)
        self.assertTrue(all(item.get("role") != "holdout" for family in cut["train"][1000].values() for item in family))


class ChoiceSelectionTests(unittest.TestCase):
    def test_a_checkpoint_must_never_lower_a_pick_and_keep_a_legal_rate(self):
        first = {"legal_rate": 1.0, "figure_accuracy": 0.4, "defect_accuracy": 0.4, "step": 10}
        self.assertTrue(choice_is_better(first, None))
        self.assertFalse(choice_is_better({"legal_rate": 0.2, "figure_accuracy": 0.9, "defect_accuracy": 0.9}, first))
        self.assertTrue(choice_is_better({"legal_rate": 1.0, "figure_accuracy": 0.5, "defect_accuracy": 0.4}, first))
        self.assertTrue(choice_is_better({"legal_rate": 1.0, "figure_accuracy": 0.4, "defect_accuracy": 0.5}, first))
        self.assertTrue(choice_is_better({"legal_rate": 1.0, "figure_accuracy": 0.5, "defect_accuracy": 0.5}, first))
        self.assertFalse(choice_is_better({"legal_rate": 1.0, "figure_accuracy": 0.5, "defect_accuracy": 0.3}, first))
        self.assertFalse(choice_is_better({"legal_rate": 1.0, "figure_accuracy": 0.3, "defect_accuracy": 0.5}, first))
        self.assertFalse(choice_is_better({"legal_rate": 1.0, "figure_accuracy": 0.4, "defect_accuracy": 0.4}, first))

    def test_a_pick_that_merely_ties_does_not_block_the_other_pick(self):
        # The held-out set is small, so a later step that matches the incumbent on one
        # pick and beats it on the other is the ordinary winner. It must be adopted.
        incumbent = {"legal_rate": 1.0, "figure_accuracy": 0.90625, "defect_accuracy": 0.71875, "step": 100}
        later = {"legal_rate": 1.0, "figure_accuracy": 0.90625, "defect_accuracy": 0.75, "step": 400}
        self.assertTrue(choice_is_better(later, incumbent))
        higher_figure = {"legal_rate": 1.0, "figure_accuracy": 0.9375, "defect_accuracy": 0.71875, "step": 1900}
        self.assertTrue(choice_is_better(higher_figure, incumbent))

    def test_choice_selection_does_not_rank_long_json(self):
        pair = {
            "pair_id": "group-1",
            "paper_id": "paper-1",
            "domain": "cv",
            "family": "corruption",
            "defect": "overlap",
            "good": "/tmp/good.png",
            "bad": "/tmp/bad.png",
            "title": "Title",
            "evidence": "A method overview.",
        }
        row = render_pair(pair, swapped=False)

        class Fake:
            name = "fake"

            def sft_step(self, record, chosen):
                return StepOutcome(loss=0.1, margin=1.0)

            def generate(self, record, max_new_tokens=64):
                return record["chosen"]

            def flush(self):
                return None

            def save(self, directory):
                Path(directory).mkdir(parents=True, exist_ok=True)

            def state(self):
                return {"name": "fake"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            train.write_text(json.dumps(row) + "\n", encoding="utf-8")
            val.write_text(json.dumps(row) + "\n", encoding="utf-8")
            config = TrainingConfig(
                dataset=str(train),
                out_dir=str(root / "run"),
                backend="synthetic",
                objective="sft",
                selection="choice",
                epochs=1,
                save_every=5,
                smoke_limit=0,
                pairwise_patience=2,
                collapse_margin=5,
                validation_dataset=str(val),
                smoke_max_new_tokens=32,
            )
            report = train_dpo(config, backend=Fake())
        self.assertEqual(report["validation"]["best"]["selection"], "choice")
        self.assertEqual(report["validation"]["best"]["figure_accuracy"], 1.0)
        self.assertEqual(report["stopped_early_reason"], "")
        self.assertEqual(choice_json("B", "overlap"), row["chosen"])


class ChoicePreferenceTests(unittest.TestCase):
    def test_each_dpo_row_changes_one_field_and_ugly_pairs_keep_a_figure_preference(self):
        pair = {
            "pair_id": "group-ugly",
            "paper_id": "paper-ugly",
            "domain": "cv",
            "family": "v002",
            "defect": "correct_but_ugly",
            "good": "/tmp/good.png",
            "bad": "/tmp/bad.png",
            "title": "Title",
            "evidence": "A method overview.",
        }
        rows = preference_records(render_pair(pair, swapped=False))
        self.assertEqual([row["preference"] for row in rows], ["figure", "defect"])
        figure = json.loads(rows[0]["rejected"])
        defect = json.loads(rows[1]["rejected"])
        self.assertEqual(figure, {"wrong_figure": "A", "defect": "correct_but_ugly"})
        self.assertEqual(defect["wrong_figure"], "B")
        self.assertNotEqual(defect["defect"], "correct_but_ugly")
        plain = render_pair({**pair, "defect": "no_defect", "pair_id": "group-plain"}, swapped=False)
        kinds = [row["preference"] for row in preference_records(plain)]
        self.assertEqual(kinds, ["defect"])

    def test_figure_export_drops_defect_rows_and_no_defect(self):
        ugly = render_pair({
            "pair_id": "group-ugly",
            "paper_id": "paper-ugly",
            "domain": "cv",
            "family": "v002",
            "defect": "correct_but_ugly",
            "good": "/tmp/good.png",
            "bad": "/tmp/bad.png",
            "title": "Title",
            "evidence": "A method overview.",
        }, swapped=False)
        plain = render_pair({
            "pair_id": "group-plain",
            "paper_id": "paper-plain",
            "domain": "cv",
            "family": "v002",
            "defect": "no_defect",
            "good": "/tmp/good.png",
            "bad": "/tmp/bad.png",
            "title": "Title",
            "evidence": "A method overview.",
        }, swapped=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "train.jsonl"
            source.write_text(json.dumps(ugly) + "\n" + json.dumps(plain) + "\n", encoding="utf-8")
            report = export_figure_preferences(source, root / "figure.jsonl")
            rows = [json.loads(line) for line in (root / "figure.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(report["rows"], 1)
        self.assertEqual(report["skipped_no_figure"], 1)
        self.assertEqual(rows[0]["preference"], "figure")
        chosen = json.loads(rows[0]["chosen"])
        rejected = json.loads(rows[0]["rejected"])
        self.assertEqual(chosen["defect"], rejected["defect"])
        self.assertNotEqual(chosen["wrong_figure"], rejected["wrong_figure"])


class ChoiceQuotaTests(unittest.TestCase):
    def test_5000_keeps_the_same_mix_and_contains_1000(self):
        self.assertEqual(sum(FAMILY_QUOTAS[5000].values()), 5000)
        self.assertEqual(FAMILY_QUOTAS[5000]["corruption"], 1750)
        for family, count in FAMILY_QUOTAS[1000].items():
            self.assertGreaterEqual(FAMILY_QUOTAS[5000][family], count)


class ChoiceConfigTests(unittest.TestCase):
    def test_the_choice_template_turns_ranking_stops_off(self):
        path = Path(__file__).resolve().parents[2] / "configs" / "judge" / "train-9b-choice.json"
        config = TrainingConfig.from_file(path, dataset="train.jsonl", out_dir="run")
        self.assertEqual(config.selection, "choice")
        self.assertEqual(config.objective, "sft")
        self.assertEqual(config.pairwise_patience, 0)
        self.assertEqual(config.collapse_margin, 0)
        self.assertEqual(config.generate_eval_limit, 0)
        self.assertEqual(config.smoke_max_new_tokens, 64)


if __name__ == "__main__":
    unittest.main()
