"""The advisory role's comparison question: two figures, one verdict, one defect class.

The Judge was trained on two figures at a time. Handed one figure and asked what is wrong with it,
it answers "no defect" whatever the figure looks like -- not because the figure is good, but because
that is not the question it learned. These tests pin the wiring that lets the version the loop has
already accepted travel beside this round's candidate, and pin that neither measurement role is ever
handed that second image.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rfs.evaluation.direct_judges import (
    BootstrapVLMDirectJudge,
    DirectJudgeInput,
    DirectPairJudgeInput,
    build_direct_judges,
)
from rfs.evaluation.judge_prompt import canonical_prompt, online_judge_row
from rfs.judge_training.judge_schema import DIAGNOSIS_TYPES
from rfs.judge_training.prompts import output_schema
from rfs.workflows.paper_evolve_editable import _accepted_preview, _preview_path


def _png(root, name):
    path = Path(root) / name
    path.write_bytes(b"png")
    return path


def _candidate(preview=None, accepted=None):
    return DirectJudgeInput(method={}, layout={}, icons={}, program={}, preview=preview, comparison_preview=accepted)


class _Capture:
    """A stand-in for the model that records what it was asked."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt, images):
        self.calls.append((prompt, [str(item) for item in images]))
        return {"stage_feedback": []}


class AdvisoryComparisonQuestionTests(unittest.TestCase):
    def test_the_accepted_render_travels_beside_the_candidate_as_a_then_b(self):
        seen = _Capture()
        advisory, _ = build_direct_judges(mode="bootstrap", online_model="m", frozen_model="m", advisory_checkpoint="/ckpt", advisory_ask=seen)
        with TemporaryDirectory() as root:
            current, accepted = _png(root, "current.png"), _png(root, "accepted.png")
            advisory.evaluate(_candidate(preview=current, accepted=accepted))
        self.assertEqual(1, len(seen.calls))
        self.assertEqual([str(accepted), str(current)], seen.calls[0][1])

    def test_the_question_asks_for_a_verdict_and_a_defect_class(self):
        seen = _Capture()
        advisory, _ = build_direct_judges(mode="bootstrap", online_model="m", frozen_model="m", advisory_checkpoint="/ckpt", advisory_ask=seen)
        with TemporaryDirectory() as root:
            current, accepted = _png(root, "current.png"), _png(root, "accepted.png")
            advisory.evaluate(_candidate(preview=current, accepted=accepted))
        prompt = seen.calls[0][0]
        self.assertIn("the first is A, the version the loop has already accepted", prompt)
        self.assertIn("the second is B, this round's candidate", prompt)
        self.assertIn('"preferred_candidate": "A|B|tie"', prompt)
        self.assertIn('"diagnosis"', prompt)
        for defect in DIAGNOSIS_TYPES:
            self.assertIn(defect, prompt, defect)
        # The defect menu is the training contract's own, not a second vocabulary invented here.
        self.assertEqual(DIAGNOSIS_TYPES, tuple(sorted(set(DIAGNOSIS_TYPES))))
        # Both questions now show two images, so the sentences that name them must stay apart:
        # a reader -- or a filter -- that finds "the first is A" must not think it found the pair.
        self.assertNotIn("the state it has just produced", prompt)

    def test_the_figure_is_still_asked_alone_when_nothing_accepted_exists_yet(self):
        seen = _Capture()
        advisory, _ = build_direct_judges(mode="bootstrap", online_model="m", frozen_model="m", advisory_checkpoint="/ckpt", advisory_ask=seen)
        with TemporaryDirectory() as root:
            current = _png(root, "current.png")
            advisory.evaluate(_candidate(preview=current))
        prompt, images = seen.calls[0]
        self.assertEqual([str(current)], images)
        self.assertNotIn("Two images are supplied", prompt)
        self.assertNotIn("preferred_candidate", prompt)

    def test_an_accepted_render_that_was_never_written_falls_back_to_one_image(self):
        seen = _Capture()
        advisory, _ = build_direct_judges(mode="bootstrap", online_model="m", frozen_model="m", advisory_checkpoint="/ckpt", advisory_ask=seen)
        with TemporaryDirectory() as root:
            current = _png(root, "current.png")
            advisory.evaluate(_candidate(preview=current, accepted=Path(root) / "missing.png"))
        prompt, images = seen.calls[0]
        self.assertEqual([str(current)], images)
        self.assertNotIn("preferred_candidate", prompt)

    def test_the_inherited_figure_is_the_accepted_state_until_a_round_here_is_accepted(self):
        # A round that followed a rollback has accepted nothing yet, so the state the loop
        # has already accepted is the kept figure the iteration was handed. Reading the
        # accepted directory alone left that empty, so the seat was asked about one figure.
        with TemporaryDirectory() as root:
            inherited = _png(root, "inherited.png")
            current = _png(root, "current.png")
            baseline = _candidate(preview=inherited)
            accepted = _accepted_preview(None, baseline)
            self.assertEqual(inherited, accepted)

            seen = _Capture()
            advisory, _ = build_direct_judges(mode="bootstrap", online_model="m", frozen_model="m", advisory_checkpoint="/ckpt", advisory_ask=seen)
            advisory.evaluate(_candidate(preview=current, accepted=accepted))
            prompt, images = seen.calls[0]
            self.assertEqual([str(inherited), str(current)], images)
            self.assertIn("the first is A, the version the loop has already accepted", prompt)
            self.assertIn('"preferred_candidate"', prompt)

    def test_a_round_accepted_here_replaces_the_inherited_figure(self):
        with TemporaryDirectory() as root:
            inherited = _png(root, "inherited.png")
            accepted_dir = Path(root) / "round_00" / "candidate_00"
            accepted_dir.mkdir(parents=True)
            render = accepted_dir / "final_draft_preview.png"
            render.write_bytes(b"png")
            self.assertEqual(render, _accepted_preview(accepted_dir, _candidate(preview=inherited)))

    def test_an_accepted_directory_without_a_render_does_not_fall_back(self):
        # A round here was accepted, so the inherited figure is no longer the state the loop
        # has already accepted: showing it as A would ask the seat about the wrong pair.
        with TemporaryDirectory() as root:
            inherited = _png(root, "inherited.png")
            empty_dir = Path(root) / "round_00" / "candidate_00"
            empty_dir.mkdir(parents=True)
            self.assertIsNone(_accepted_preview(empty_dir, _candidate(preview=inherited)))

    def test_the_acceptance_seat_is_never_handed_the_second_image(self):
        seen = _Capture()
        seat = BootstrapVLMDirectJudge(frozen=True, ask_json=seen)
        with TemporaryDirectory() as root:
            current, accepted = _png(root, "current.png"), _png(root, "accepted.png")
            seat.evaluate(_candidate(preview=current, accepted=accepted))
        prompt, images = seen.calls[0]
        self.assertEqual([str(current)], images)
        self.assertNotIn("preferred_candidate", prompt)
        self.assertNotIn("the version the loop has already accepted", prompt)

    def test_the_pairwise_question_keeps_its_own_two_images_in_its_own_order(self):
        seen = _Capture()
        seat = BootstrapVLMDirectJudge(frozen=True, ask_json=seen)
        with TemporaryDirectory() as root:
            before, after, spare = _png(root, "before.png"), _png(root, "after.png"), _png(root, "spare.png")
            seat.evaluate_pairwise(DirectPairJudgeInput(before=_candidate(preview=before, accepted=spare), after=_candidate(preview=after)))
        prompt, images = seen.calls[0]
        self.assertEqual([str(before), str(after)], images)
        self.assertIn("the first is A, the state the loop already accepted", prompt)


class TheComparisonIsTheOnlyDifferenceTests(unittest.TestCase):
    def test_the_advisory_schema_grows_exactly_two_fields(self):
        plain = output_schema("repair", role="online_trained")
        compared = output_schema("repair", role="online_trained", comparison=True)
        self.assertNotIn("preferred_candidate", plain)
        self.assertNotIn("diagnosis", plain)
        self.assertEqual(["preferred_candidate"], [key for key in compared if key not in plain][:1])
        self.assertIn("diagnosis", compared)
        self.assertEqual(set(plain) - {"preferred_candidate"}, set(compared) - {"preferred_candidate", "diagnosis"})

    def test_a_comparison_never_reaches_a_measurement_schema(self):
        self.assertNotIn("preferred_candidate", output_schema("repair", role="frozen_choice", comparison=True))
        self.assertNotIn("diagnosis", output_schema("repair", role="acceptance", comparison=True))

    def test_a_row_without_a_comparison_carries_no_comparison_key(self):
        row = online_judge_row(_candidate(preview="x.png"), role="online_trained")
        self.assertNotIn("comparison", row["process_context"])
        self.assertNotIn("comparison", online_judge_row(_candidate(preview="x.png"), role="online_trained", comparison=False)["process_context"])

    def test_a_comparison_key_reaches_the_advisory_row_and_no_other(self):
        advisory = online_judge_row(_candidate(preview="x.png"), role="online_trained", comparison=True)
        self.assertIn("comparison", advisory["process_context"])
        self.assertEqual("B: this round's candidate", advisory["process_context"]["comparison"]["image_2"])
        none_for_acceptance = online_judge_row(_candidate(preview="x.png"), role="frozen_choice", comparison=True)
        self.assertNotIn("comparison", none_for_acceptance["process_context"])
        none_for_pairwise = online_judge_row(_candidate(preview="x.png"), role="online_trained", pairwise=True, comparison=True)
        self.assertNotIn("comparison", none_for_pairwise["process_context"])

    def test_a_prompt_without_a_comparison_never_mentions_a_pair(self):
        prompt = canonical_prompt(online_judge_row(_candidate(preview="x.png"), role="online_trained"))
        self.assertNotIn("preferred_candidate", prompt)
        self.assertNotIn("A: the version the loop has already accepted", prompt)


class TheAcceptedRenderIsResolvedTests(unittest.TestCase):
    def test_the_loop_hands_over_the_render_a_candidate_directory_holds(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            self.assertIsNone(_preview_path(None))
            self.assertIsNone(_preview_path(root))
            (root / "draft_preview.png").write_bytes(b"png")
            self.assertEqual(root / "draft_preview.png", _preview_path(root))
            (root / "final_draft_preview.png").write_bytes(b"png")
            self.assertEqual(root / "final_draft_preview.png", _preview_path(root))


if __name__ == "__main__":
    unittest.main()