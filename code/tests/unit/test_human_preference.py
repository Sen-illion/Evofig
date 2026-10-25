"""The human-preference channel: the queue, a verdict, and what a verdict becomes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.evolution.direct.human_preference import (
    HUMAN_CHOICES,
    HUMAN_PREFERENCE_ENTRY_CONTRACT,
    HumanReviewStore,
    collect_review_items,
    export_choice_items,
    human_review_stats,
    pairwise_choice,
    preference_entry,
    review_kind,
    unconsumed_verdicts,
    validate_human_review,
    write_review_bundle,
)

MENU = ["correct_but_ugly", "missing_core_stage", "no_defect"]


def _round(root: Path, paper: str, iteration: str, round_name: str, *, accepted: bool, choice: str = "B", before=0.5, after=0.7) -> Path:
    round_dir = root / "papers" / paper / iteration / round_name
    (round_dir / "candidate_00").mkdir(parents=True, exist_ok=True)
    (round_dir / "candidate_00" / "final_draft_preview.png").write_bytes(f"{paper}-{iteration}-{round_name}".encode("utf-8"))
    (round_dir / "repair_decision.json").write_text(
        json.dumps(
            {
                "accepted": accepted,
                "reason": "pairwise_preferred" if accepted else "new_defect",
                "comparison": "pairwise",
                "before_score": before,
                "after_score": after,
                "pairwise": {"preferred_candidate": choice, "margin": "clear", "defect": "correct_but_ugly", "wrong_figure": "A"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return round_dir


class CollectReviewItemsTests(unittest.TestCase):
    def test_the_accepted_round_becomes_the_next_rounds_left_hand_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _round(root, "P1", "iter_01", "round_00", accepted=True)
            _round(root, "P1", "iter_01", "round_01", accepted=True)
            items = collect_review_items(root)
            # The first round of a tree has nothing to compare against, so it is not a question
            # anybody can answer; the second one is, and its left-hand figure is the first.
            self.assertEqual([item["item_id"] for item in items], ["P1__iter_01__round_01"])
            self.assertTrue(items[0]["images"]["A"].endswith("round_00/candidate_00/final_draft_preview.png".replace("/", str(Path("/")))))
            self.assertEqual(items[0]["model_choice"], "B")

    def test_a_round_that_could_not_be_reviewed_still_moves_the_baseline(self) -> None:
        """The first round of a run is not reviewable; losing it emptied every queue behind it."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _round(root, "P1", "iter_01", "round_00", accepted=True)
            _round(root, "P1", "iter_02", "round_00", accepted=True)
            items = collect_review_items(root)
            self.assertEqual([item["item_id"] for item in items], ["P1__iter_02__round_00"])
            self.assertIn("previous iteration", items[0]["before_source"])

    def test_the_two_images_of_one_item_are_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _round(root, "P1", "iter_01", "round_00", accepted=True)
            _round(root, "P1", "iter_01", "round_01", accepted=False)
            item = collect_review_items(root)[0]
            self.assertEqual(sorted(item["image_hashes"]), ["A", "B"])
            self.assertTrue(all(len(value) == 64 for value in item["image_hashes"].values()))
            # A rolled-back round still asked the question, so it is still reviewable.
            self.assertFalse(item["accepted"])

    def test_a_round_that_asked_no_pair_question_is_not_a_queue_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dir = _round(root, "P1", "iter_01", "round_00", accepted=True)
            round_dir2 = _round(root, "P1", "iter_01", "round_01", accepted=True)
            (round_dir2 / "repair_decision.json").write_text(json.dumps({"accepted": True, "comparison": "score"}), encoding="utf-8")
            write_review_bundle(Path(tmp) / "bundle", root)
            payload = json.loads((Path(tmp) / "bundle" / "items.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["items"], [])
            self.assertTrue((Path(tmp) / "bundle" / "images").exists())


class PairwiseChoiceTests(unittest.TestCase):
    def test_the_choice_is_read_from_whichever_home_the_record_keeps_it_in(self) -> None:
        self.assertEqual(pairwise_choice({"preferred_candidate": "B"}), "B")
        self.assertEqual(pairwise_choice({"pairwise": {"choice": "A"}}), "A")
        self.assertEqual(pairwise_choice({"choice": "A"}), "A")
        self.assertEqual(pairwise_choice({"preferred_candidate": "neither"}), "")
        self.assertEqual(pairwise_choice({}), "")


class ReviewContractTests(unittest.TestCase):
    def test_a_verdict_needs_a_choice_and_a_reviewer(self) -> None:
        result = validate_human_review({"item_id": "x", "human_choice": "maybe", "reviewer": ""})
        self.assertFalse(result["ok"])
        self.assertEqual(len(result["errors"]), 2)

    def test_a_preference_says_what_is_wrong_or_says_it_in_a_note(self) -> None:
        silent = validate_human_review({"item_id": "x", "human_choice": "A", "reviewer": "me"})
        self.assertFalse(silent["ok"])
        noted = validate_human_review({"item_id": "x", "human_choice": "A", "reviewer": "me", "human_note": "标题被切了"})
        self.assertTrue(noted["ok"])

    def test_a_defect_outside_the_menu_is_refused(self) -> None:
        good = validate_human_review({"item_id": "x", "human_choice": "A", "reviewer": "me", "human_defect": "correct_but_ugly"}, defect_menu=MENU)
        self.assertTrue(good["ok"])
        bad = validate_human_review({"item_id": "x", "human_choice": "A", "reviewer": "me", "human_defect": "made_up"}, defect_menu=MENU)
        self.assertFalse(bad["ok"])

    def test_a_changed_mind_is_appended_rather_than_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HumanReviewStore(tmp)
            store.record({"item_id": "x", "human_choice": "A", "reviewer": "me", "human_note": "first"}, defect_menu=MENU)
            store.record({"item_id": "x", "human_choice": "B", "reviewer": "me", "human_note": "second"}, defect_menu=MENU)
            self.assertEqual(len(store.rows()), 2)
            self.assertEqual(store.state()["items"]["x"]["human_choice"], "B")


class ExportTests(unittest.TestCase):
    def _item(self) -> dict:
        return {
            "item_id": "P1__iter_01__round_01",
            "paper": "P1",
            "iteration": 1,
            "round": 1,
            "model_choice": "A",
            "model_defect": "correct_but_ugly",
            "images": {"A": "images/x/A.png", "B": "images/x/B.png"},
            "image_hashes": {"A": "a" * 64, "B": "b" * 64},
        }

    def test_a_preference_becomes_the_choice_the_seat_answers(self) -> None:
        item = self._item()
        row = export_choice_items([item], [{"item_id": item["item_id"], "human_choice": "B", "human_defect": "missing_core_stage", "reviewer": "me"}], defect_menu=MENU)["items"][0]
        self.assertEqual(row["gold_figure"], "A")  # the reviewer said B is better, so A is wrong
        self.assertEqual(row["gold_defect"], "missing_core_stage")
        self.assertEqual(row["family"], "human_review")
        self.assertEqual(json.loads(row["chosen"])["wrong_figure"], "A")
        self.assertEqual(row["provenance"]["model_choice"], "A")
        self.assertEqual(len(row["messages"][0]["content"]), 3)

    def test_the_two_answers_the_question_cannot_carry_stay_out_of_training(self) -> None:
        item = self._item()
        rendered = export_choice_items(
            [item],
            [
                {"item_id": item["item_id"], "human_choice": "tie"},
                {"item_id": item["item_id"], "human_choice": "both_bad"},
            ],
            defect_menu=MENU,
        )
        self.assertEqual(rendered["items"], [])
        # Both answers are about the same comparison and neither one is a figure/defect choice,
        # so each row is set aside with the reason it could not become a training item.
        self.assertEqual(len(rendered["skipped"]), 2)
        self.assertEqual({row["reason"] for row in rendered["skipped"]}, {"this answer is not a figure/defect choice"})

    def test_where_the_person_and_the_seat_disagree(self) -> None:
        item = self._item()
        stats = human_review_stats([item], [{"item_id": item["item_id"], "human_choice": "B", "human_defect": "missing_core_stage"}])
        self.assertEqual(stats["decided"], 1)
        self.assertEqual(stats["agreed"], 0)
        self.assertEqual(stats["disagreed"], 1)
        self.assertEqual(stats["agreement_rate"], 0.0)
        self.assertEqual(stats["model_wrong"][0]["model_choice"], "A")

    def test_agreement_and_an_unanswered_queue(self) -> None:
        item = self._item()
        stats = human_review_stats([item, {"item_id": "other"}], [{"item_id": item["item_id"], "human_choice": "A"}])
        self.assertEqual(stats["queue"], 2)
        self.assertEqual(stats["reviewed"], 1)
        self.assertEqual(stats["unreviewed"], 1)
        self.assertEqual(stats["agreement_rate"], 1.0)
        self.assertEqual(stats["model_wrong"], [])


class InstructionTests(unittest.TestCase):
    """A verdict that is about the work rather than about one comparison, and where it lands."""

    def test_a_settled_layer_names_its_paper_its_stage_and_why(self) -> None:
        base = {"kind": "layer_settled", "reviewer": "me", "human_note": "every module is on the figure"}
        missing_paper = validate_human_review({**base, "stage": "paper_reading"})
        self.assertFalse(missing_paper["ok"])
        missing_stage = validate_human_review({**base, "paper": "paper_one"})
        self.assertFalse(missing_stage["ok"])
        made_up_stage = validate_human_review({**base, "paper": "paper_one", "stage": "the_vibes"})
        self.assertFalse(made_up_stage["ok"])
        ok = validate_human_review({**base, "paper": "paper_one", "stage": "paper_reading"})
        self.assertTrue(ok["ok"], ok["errors"])
        self.assertFalse(validate_human_review({**base, "paper": "paper_one", "stage": "paper_reading", "human_note": ""})["ok"])

    def test_a_preference_carries_the_sentence_the_stage_should_have_written(self) -> None:
        self.assertFalse(validate_human_review({"kind": "preference", "paper": "paper_one", "apply_at": "theme_planning", "directive": "", "reviewer": "me"})["ok"])
        self.assertFalse(validate_human_review({"kind": "preference", "paper": "paper_one", "apply_at": "nowhere", "directive": "keep it calm", "reviewer": "me"})["ok"])
        ok = validate_human_review(
            {"kind": "preference", "paper": "paper_one", "apply_at": "theme_planning", "directive": "keep it calm", "condition": {"defect_patterns": ["clashing_palette"]}, "reviewer": "me"}
        )
        self.assertTrue(ok["ok"], ok["errors"])

    def test_a_preference_entry_has_the_shape_a_rule_and_a_patch_share(self) -> None:
        entry = preference_entry(
            {
                "apply_at": "ppt_compilation",
                "condition": {"defect_patterns": ["cross_panel"]},
                "directive": "Route connectors on shared vertical channels.",
                "reviewer": "me",
                "review_id": "human-review-x",
            }
        )
        self.assertEqual(HUMAN_PREFERENCE_ENTRY_CONTRACT, entry["contract"])
        self.assertEqual("human", entry["source"])
        self.assertEqual("candidate", entry["status"])
        self.assertEqual(["apply_at", "condition", "directive"], [key for key in ("apply_at", "condition", "directive") if key in entry])
        self.assertTrue(entry["entry_id"].startswith("human-pref-"))
        # The same sentence filed twice is one entry, not two.
        self.assertEqual(entry["entry_id"], preference_entry({"apply_at": "ppt_compilation", "condition": {"defect_patterns": ["cross_panel"]}, "directive": "Route connectors on shared vertical channels.", "reviewer": "me", "review_id": "human-review-x"})["entry_id"])

    def test_a_trial_stands_only_when_it_is_promoted_and_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HumanReviewStore(tmp)
            recorded = store.record(
                {"kind": "preference", "paper": "paper_one", "apply_at": "theme_planning", "directive": "keep it calm", "reviewer": "me", "human_note": "the palette is too loud"}
            )
            self.assertTrue(recorded["ok"], recorded)
            self.assertEqual("candidate", recorded["entry"]["status"])
            self.assertEqual(1, store.state()["entries"])
            self.assertEqual(0, store.state()["resident"])
            first = store.promote(recorded["entry"]["entry_id"], by="loop", because="the round kept it")
            self.assertTrue(first["changed"])
            again = store.promote(recorded["entry"]["entry_id"], by="loop", because="the round kept it")
            self.assertFalse(again["changed"])
            self.assertEqual("resident", again["entry"]["status"])
            self.assertEqual(1, store.state()["resident"])
            self.assertEqual(1, len(store.rows()))  # promoting says nothing new, so the log keeps one row

    def test_a_resident_preference_is_filed_standing_from_the_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = HumanReviewStore(tmp)
            recorded = store.record(
                {"kind": "preference", "paper": "paper_one", "apply_at": "theme_planning", "directive": "keep it calm", "reviewer": "me", "resident": True}
            )
            self.assertEqual("resident", recorded["entry"]["status"])
            self.assertEqual(1, store.state()["resident"])

    def test_a_verdict_is_read_once_and_only_by_the_paper_it_names(self) -> None:
        rows = [
            {"kind": "layer_settled", "paper": "paper_one", "review_id": "a"},
            {"kind": "layer_settled", "paper": "paper_two", "review_id": "b"},
            {"kind": "preference", "paper": "paper_one", "review_id": "c"},
            {"item_id": "P1__iter_01__round_01", "human_choice": "A", "review_id": "d"},
        ]
        mine = unconsumed_verdicts(rows, [], paper="paper_one")
        self.assertEqual(["a", "c"], [row["review_id"] for row in mine])
        self.assertEqual([], unconsumed_verdicts(rows, ["a", "c"], paper="paper_one"))
        self.assertEqual(["b"], [row["review_id"] for row in unconsumed_verdicts(rows, [], paper="paper_two")])
        self.assertEqual("figure_choice", review_kind(rows[3]))
        self.assertEqual("figure_choice", review_kind({}))

    def test_an_instruction_is_not_an_answer_to_the_comparison_queue(self) -> None:
        item = {"item_id": "x", "model_choice": "A"}
        stats = human_review_stats(
            [item],
            [
                {"item_id": "x", "human_choice": "A", "review_id": "a"},
                {"kind": "preference", "paper": "paper_one", "review_id": "b"},
                {"kind": "layer_settled", "paper": "paper_one", "review_id": "c"},
            ],
        )
        self.assertEqual(1, stats["reviewed"])
        self.assertEqual(2, stats["instructions"])
        self.assertEqual(1, stats["agreed"])


class ReviewCommandTests(unittest.TestCase):
    """The command a person actually runs: a run root, and the three kinds of verdict."""

    def test_a_verdict_can_be_filed_straight_into_a_run_root(self) -> None:
        from rfs.workflows.human_review import review_record

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorded = review_record(
                run_root=root,
                kind="preference",
                paper="paper_one",
                apply_at="theme_planning",
                directive="Keep the palette calm.",
                condition={"defect_patterns": ["clashing_palette"]},
                dimensions=["layout=A", "color=B"],
                note="81 lays out best, 95 colours best",
                reviewer="<AUTHOR>",
            )
            self.assertTrue(recorded["ok"], recorded)
            self.assertEqual("preference", recorded["kind"])
            self.assertEqual(1, recorded["entries"])
            self.assertEqual(0, recorded["resident"])
            self.assertEqual(str(root / "human_review"), recorded["bundle"])
            entry = json.loads((root / "human_review" / "human_preference_entries.json").read_text(encoding="utf-8"))["entries"][0]
            self.assertEqual({"layout": "A", "color": "B"}, entry["dimensions"])
            self.assertEqual("candidate", entry["status"])

    def test_a_person_can_settle_a_layer_through_the_same_command(self) -> None:
        from rfs.workflows.human_review import review_record

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorded = review_record(
                run_root=root,
                kind="layer_settled",
                paper="paper_one",
                stage="segmentation",
                note="the panels are where they should be",
                reviewer="<AUTHOR>",
            )
            self.assertTrue(recorded["ok"], recorded)
            self.assertEqual("layer_settled", recorded["kind"])

    def test_a_verdict_with_neither_a_bundle_nor_a_run_root_is_refused(self) -> None:
        from rfs.workflows.human_review import review_record

        with self.assertRaises(ValueError):
            review_record(item_id="x", choice="A", reviewer="me")


class ChoiceVocabularyTests(unittest.TestCase):
    def test_the_vocabulary_is_the_four_answers_the_page_offers(self) -> None:
        self.assertEqual(HUMAN_CHOICES, ("A", "B", "tie", "both_bad"))


if __name__ == "__main__":
    unittest.main()
