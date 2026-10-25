"""The reference-gap question: what it accepts, and the figure it is asked against.

The question only earns its place if three things hold: an answer that names no class counts as
the question going unanswered rather than as a gap found, the figure handed over is always one of
this project's own kind and never the paper's own drawing, and no measurement seat is ever shown
the reference. Each of those is a way the channel could look alive while measuring nothing, so
each has a test.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rfs.analysis.reference import select_gap_reference
from rfs.evaluation.direct_judges import (
    BootstrapVLMDirectJudge,
    CheckpointAdvisoryJudge,
    HeuristicDirectJudge,
    ReferenceGapInput,
)
from rfs.evaluation.reference_gap import (
    GAP_CLASSES,
    MACRO_ACTIONS,
    build_reference_gap_prompt,
    normalize_reference_gap,
    reference_gap_conformance,
    reference_gap_summary,
)
from rfs.workflows.paper_evolve_editable import _ask_reference_gap


def _good_answer(**overrides):
    answer = {
        "gap_class": "layout_structure",
        "direction": "give the middle pipeline the widest column and push the call-outs to the margin",
        "evidence": ["ours reads as one undifferentiated row of boxes"],
        "confidence": 0.7,
        "requested_actions": [{"action": "resegment", "expected_effect": "one dominant lane with smaller flanks"}],
    }
    answer.update(overrides)
    return answer


def _framework_record(record_id: str, path: Path, **overrides):
    record = {
        "id": record_id,
        "quality_label": "positive",
        "source_path": str(path),
        "archetype": "linear_workflow",
        "why_good": ["the reading order is unmistakable"],
        "structure": {"section_roles": ["input", "module", "output"]},
        "layout": {"macro_pattern": "left to right"},
        "connectors": {"grammar": ["arrow"]},
        "density": {"level": "medium"},
    }
    record.update(overrides)
    return record


def _data_record(record_id: str, path: Path, **overrides):
    record = {
        "id": record_id,
        "quality_label": "positive",
        "source_path": str(path),
        "archetype": "quantitative panel",
        "layout": {"macro_pattern": "small-multiple plot grid"},
        "structure": {"section_roles": ["axis", "legend"]},
    }
    record.update(overrides)
    return record


class ReferenceGapConformanceTests(unittest.TestCase):
    def test_an_answer_that_names_no_class_is_the_question_unanswered(self):
        problems = reference_gap_conformance(_good_answer(gap_class="looks_nicer"))
        self.assertTrue(any("gap_class" in problem for problem in problems))

    def test_an_answer_that_asks_for_no_action_is_the_question_unanswered(self):
        problems = reference_gap_conformance(_good_answer(requested_actions=[]))
        self.assertIn("no requested_actions at all", problems)

    def test_a_local_action_is_not_an_answer_to_this_question(self):
        problems = reference_gap_conformance(
            _good_answer(requested_actions=[{"action": "resize_nodes"}])
        )
        self.assertTrue(any("not a macro action" in problem for problem in problems))

    def test_a_macro_answer_is_accepted(self):
        self.assertEqual(reference_gap_conformance(_good_answer()), [])
        for action in MACRO_ACTIONS:
            self.assertEqual(
                reference_gap_conformance(_good_answer(requested_actions=[{"action": action}])), []
            )

    def test_a_missing_direction_is_refused(self):
        self.assertIn("no direction was named", reference_gap_conformance(_good_answer(direction=" ")))


class ReferenceGapRecordTests(unittest.TestCase):
    def test_silence_and_an_unusable_class_are_told_apart(self):
        silent = normalize_reference_gap({}, problem="the seat could not be read")
        unusable = normalize_reference_gap(_good_answer(gap_class="looks_nicer"))
        self.assertFalse(silent["readable"])
        self.assertFalse(silent["conforms"])
        self.assertTrue(unusable["readable"])
        self.assertFalse(unusable["conforms"])

    def test_a_readable_answer_keeps_the_actions_it_named(self):
        record = normalize_reference_gap(_good_answer())
        self.assertTrue(record["conforms"])
        self.assertEqual(record["gap_class"], "layout_structure")
        self.assertEqual([item["action"] for item in record["requested_actions"]], ["resegment"])

    def test_the_summary_counts_what_the_question_got(self):
        summary = reference_gap_summary(
            [
                normalize_reference_gap(_good_answer()),
                normalize_reference_gap({}, problem="unreadable"),
            ]
        )
        self.assertEqual(summary["asked"], 2)
        self.assertEqual(summary["readable"], 1)
        self.assertEqual(summary["conforming"], 1)
        self.assertEqual(summary["gap_classes"], {"layout_structure": 1, "(none)": 1})

    def test_the_prompt_carries_the_catalog_wording_and_forbids_no_change(self):
        prompt = build_reference_gap_prompt("good-001", "reads well", "our output")
        self.assertIn("No change is needed", prompt)
        for action in MACRO_ACTIONS:
            self.assertIn(action, prompt)
        for gap_class in GAP_CLASSES:
            self.assertIn(gap_class, prompt)


class GapReferenceSelectionTests(unittest.TestCase):
    def _library(self, root: Path, records: list[dict]) -> Path:
        path = root / "reference_library.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
        return path

    def test_a_better_scoring_data_panel_is_still_not_the_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            framework = root / "framework.png"
            framework.write_bytes(b"png")
            data = root / "data.png"
            data.write_bytes(b"png")
            library = self._library(
                root,
                [
                    _data_record("panel_0001", data, archetype="wanted_template"),
                    _framework_record("good_0009", framework),
                ],
            )
            selection = select_gap_reference({}, library, selected_template={"template_id": "wanted_template"})
            self.assertTrue(selection["ok"])
            self.assertEqual(selection["selected"]["id"], "good_0009")
            self.assertEqual(selection["framework_positive_count"], 1)

    def test_the_papers_own_figure_is_kept_out_of_the_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mine = root / "10.1145_3770854.3780184" / "iter_34" / "final_draft_preview.png"
            mine.parent.mkdir(parents=True)
            mine.write_bytes(b"png")
            other = root / "good-002.png"
            other.write_bytes(b"png")
            library = self._library(
                root,
                [
                    _framework_record("mine_0001", mine),
                    _framework_record("good_0002", other),
                ],
            )
            selection = select_gap_reference({}, library, exclude_tokens=["10.1145_3770854.3780184"])
            self.assertEqual(selection["selected"]["id"], "good_0002")
            self.assertEqual(
                [item["reason"] for item in selection["excluded"]],
                ["the figure is one of the paper being evolved"],
            )

    def test_a_key_that_merely_appears_inside_a_folder_name_keeps_nothing_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "data" / "paperimage" / "library" / "curated_0001.png"
            good.parent.mkdir(parents=True)
            good.write_bytes(b"png")
            library = self._library(root, [_framework_record("curated_0001", good)])
            # The pool lives under a project folder whose name contains the key a run that never
            # learned its paper's identifier falls back to. Reading that key as a substring
            # empties the pool over a folder no figure came from, so it is dropped instead.
            selection = select_gap_reference({}, library, exclude_tokens=["paper", ""])
            self.assertTrue(selection["ok"])
            self.assertEqual(selection["selected"]["id"], "curated_0001")
            self.assertEqual(selection["excluded"], [])
            self.assertEqual(selection["exclude_keys_ignored"], ["paper"])

    def test_a_slice_of_the_identifier_is_not_the_papers_own_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mine = root / "10.1145_3770854.3780184" / "iter_34" / "final_draft_preview.png"
            mine.parent.mkdir(parents=True)
            mine.write_bytes(b"png")
            library = self._library(root, [_framework_record("mine_0001", mine)])
            # A folder is claimed only when a key is one of its whole names: the tail of an
            # identifier appearing inside a longer folder name is a different name.
            selection = select_gap_reference({}, library, exclude_tokens=["3770854.3780184"])
            self.assertTrue(selection["ok"])
            self.assertEqual(selection["selected"]["id"], "mine_0001")
            self.assertEqual(selection["excluded"], [])

    def test_the_slash_form_of_the_identifier_matches_the_folder_folded_from_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mine = root / "10.1145_3770854.3780184" / "iter_34" / "final_draft_preview.png"
            mine.parent.mkdir(parents=True)
            mine.write_bytes(b"png")
            other = root / "good-0002.png"
            other.write_bytes(b"png")
            library = self._library(
                root,
                [_framework_record("mine_0001", mine), _framework_record("good_0002", other)],
            )
            selection = select_gap_reference({}, library, exclude_tokens=["10.1145/3770854.3780184"])
            self.assertEqual(selection["selected"]["id"], "good_0002")
            self.assertEqual(
                [item["reason"] for item in selection["excluded"]],
                ["the figure is one of the paper being evolved"],
            )
            self.assertEqual(selection["exclude_keys_applied"], ["10.1145_3770854.3780184"])

    def test_a_figure_named_after_the_paper_with_a_suffix_is_still_its_own(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mine = root / "exports" / "10.1145_3770854.3780184_fig1.png"
            mine.parent.mkdir(parents=True)
            mine.write_bytes(b"png")
            other = root / "good-0002.png"
            other.write_bytes(b"png")
            library = self._library(
                root,
                [_framework_record("mine_0001", mine), _framework_record("good_0002", other)],
            )
            selection = select_gap_reference({}, library, exclude_tokens=["10.1145/3770854.3780184"])
            self.assertEqual(selection["selected"]["id"], "good_0002")

    def test_a_library_of_data_panels_answers_with_a_reason_instead_of_a_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.png"
            data.write_bytes(b"png")
            library = self._library(root, [_data_record("panel_0001", data)])
            selection = select_gap_reference({}, library)
            self.assertFalse(selection["ok"])
            self.assertEqual(selection["selected"], {})
            self.assertIn("no positive framework figure", selection["reason"])

    def test_an_unreadable_image_is_reported_rather_than_handed_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = self._library(root, [_framework_record("good_0001", root / "missing.png")])
            selection = select_gap_reference({}, library)
            self.assertFalse(selection["ok"])
            self.assertEqual([item["reason"] for item in selection["excluded"]], ["the image is not readable"])


class SeatTests(unittest.TestCase):
    def test_a_seat_with_no_model_has_nothing_to_say(self):
        record = HeuristicDirectJudge().reference_gap(ReferenceGapInput(reference_path="a.png", ours_path="b.png"))
        self.assertFalse(record["readable"])
        self.assertIn("no model to ask", record["problems"][0])

    def test_the_acceptance_seat_is_never_shown_a_reference(self):
        called: list[Any] = []

        def ask(prompt, images):
            called.append((prompt, images))
            return _good_answer()

        seat = BootstrapVLMDirectJudge(frozen=True, ask_json=ask)
        record = seat.reference_gap(ReferenceGapInput(reference_path="a.png", ours_path="b.png"))
        self.assertFalse(record["readable"])
        self.assertEqual(called, [])
        self.assertIn("not shown a reference figure", record["problems"][0])

    def test_the_advisory_seat_asks_the_question_with_both_figures(self):
        seen: dict[str, Any] = {}

        def ask(prompt, images):
            seen["prompt"] = prompt
            seen["images"] = list(images)
            return _good_answer()

        seat = BootstrapVLMDirectJudge(ask_json=ask)
        record = seat.reference_gap(
            ReferenceGapInput(
                reference_path="reference.png",
                ours_path="ours.png",
                reference_id="good-001",
                reference_note="reads well",
                ours_note="our output",
            )
        )
        self.assertTrue(record["conforms"])
        self.assertEqual(seen["images"], ["reference.png", "ours.png"])
        self.assertIn("reference_gap", seen["prompt"])
        self.assertEqual(record["reference_id"], "good-001")

    def test_a_checkpoint_that_cannot_be_read_is_recorded_as_silence(self):
        def ask(prompt, images):
            raise ValueError("cut off inside the object")

        seat = CheckpointAdvisoryJudge(checkpoint="", base_model="", ask_json=ask)
        record = seat.reference_gap(ReferenceGapInput(reference_path="a.png", ours_path="b.png"))
        self.assertFalse(record["readable"])
        self.assertTrue(record["retried"])
        self.assertTrue(record["problems"])


class RoundWiringTests(unittest.TestCase):
    class _Judge:
        def __init__(self):
            self.calls: list[Any] = []

        def reference_gap(self, gap_input):
            self.calls.append(gap_input)
            return normalize_reference_gap(_good_answer())

    def test_the_question_is_skipped_when_no_reference_was_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            events = Path(tmp) / "events.jsonl"
            judge = self._Judge()
            record = _ask_reference_gap(
                judge,
                gap_reference={"ok": False, "selected": {}},
                selected={"dir": str(Path(tmp) / "candidate_00")},
                round_index=1,
                event_path=events,
            )
            self.assertEqual(record, {})
            self.assertEqual(judge.calls, [])
            self.assertFalse(events.exists())

    def test_the_question_is_asked_once_and_leaves_an_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate_00"
            candidate.mkdir()
            (candidate / "final_draft_preview.png").write_bytes(b"png")
            reference = root / "good-001.png"
            reference.write_bytes(b"png")
            events = root / "events.jsonl"
            judge = self._Judge()
            record = _ask_reference_gap(
                judge,
                gap_reference={"ok": True, "selected": {"id": "good-001", "source_path": str(reference), "why_good": ["reads well"]}},
                selected={"dir": str(candidate)},
                round_index=2,
                event_path=events,
            )
            self.assertTrue(record["conforms"])
            self.assertEqual(len(judge.calls), 1)
            self.assertEqual(judge.calls[0].reference_id, "good-001")
            self.assertTrue(judge.calls[0].ours_path.name, "final_draft_preview.png")
            written = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([item["event"] for item in written], ["reference_gap_recorded"])
            self.assertEqual(written[0]["gap_class"], "layout_structure")

    def test_a_reference_whose_image_is_gone_is_not_asked_about(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate_00"
            candidate.mkdir()
            (candidate / "final_draft_preview.png").write_bytes(b"png")
            judge = self._Judge()
            record = _ask_reference_gap(
                judge,
                gap_reference={"ok": True, "selected": {"id": "gone", "source_path": str(root / "gone.png")}},
                selected={"dir": str(candidate)},
                round_index=0,
                event_path=root / "events.jsonl",
            )
            self.assertEqual(record, {})
            self.assertEqual(judge.calls, [])


if __name__ == "__main__":
    unittest.main()