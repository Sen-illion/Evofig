"""A Judge call with no figure must not hand the processor an empty image list.

The advisory seat reflects on a round with the outcome as text and no image at all. The
transformers image processor indexes the list it is given, so ``images=[]`` raised
``IndexError: list index out of range`` inside the processor -- not inside the model -- and
every reflection the live run ever made came back empty for that reason. The same shape is
reachable from the answer path whenever a preview is missing, so the guard is on the call
rather than on the one caller that happened to hit it first.

A reflection that still cannot be read is a named outcome: the reason travels on it, and an
answer that was cut off is asked for once more, briefer -- the same second chance the answer
path already had. Silence and "the seat had nothing to say" must not look alike.
"""

import contextlib
import unittest

from rfs.evaluation.direct_judges import REFLECTION_BREVITY_RETRY, CheckpointAdvisoryJudge
from rfs.judge_training.training import TransformersVlmBackend, _processor_call


class _Ids:
    shape = (1, 5)

    def clone(self):
        return self


class _Generated:
    def __getitem__(self, item):
        return self


class _RecordingProcessor:
    """A processor that keeps how it was called, so the call shape can be asserted."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        return "PROMPT"

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"input_ids": _Ids()}

    def batch_decode(self, ids, **kwargs):
        return ["{\"outcome\": \"improved\"}"]


class _Model:
    device = "cpu"

    def eval(self):
        return None

    def generate(self, **kwargs):
        return _Generated()


class _Stub:
    """Everything ``generate`` touches, without a checkpoint behind it."""

    def __init__(self, processor):
        self.processor = processor
        self.model = _Model()
        self.torch = type("_Torch", (), {"no_grad": staticmethod(contextlib.nullcontext)})()


class ProcessorCallTests(unittest.TestCase):
    def test_no_images_leaves_the_argument_out(self):
        processor = _RecordingProcessor()
        _processor_call(processor, text="PROMPT", images=[])
        self.assertEqual(len(processor.calls), 1)
        self.assertNotIn("images", processor.calls[0])
        self.assertEqual(processor.calls[0]["text"], ["PROMPT"])

    def test_images_are_passed_when_there_are_any(self):
        processor = _RecordingProcessor()
        _processor_call(processor, text="PROMPT", images=["a.png"])
        self.assertEqual(processor.calls[0]["images"], ["a.png"])

    def test_generate_without_a_figure_never_hands_over_an_empty_list(self):
        processor = _RecordingProcessor()
        record = {"messages": [{"role": "user", "content": [{"type": "text", "text": "reflect"}]}], "images": []}
        answer = TransformersVlmBackend.generate(_Stub(processor), record, max_new_tokens=8)
        self.assertIn("improved", answer)
        self.assertEqual(len(processor.calls), 1)
        self.assertNotIn("images", processor.calls[0])

    def test_generate_with_a_figure_still_passes_it(self):
        processor = _RecordingProcessor()
        record = {"messages": [{"role": "user", "content": [{"type": "text", "text": "judge"}]}], "images": ["figure.png"]}
        TransformersVlmBackend.generate(_Stub(processor), record, max_new_tokens=8)
        self.assertEqual(processor.calls[0]["images"], ["figure.png"])


class ReflectionFailureTests(unittest.TestCase):
    """A reflection that could not be read is a named outcome, never a silent empty one."""

    def test_an_unreadable_reflection_carries_its_reason(self):
        calls = []

        def explode(prompt, images):
            calls.append(prompt)
            raise RuntimeError("list index out of range")

        judge = CheckpointAdvisoryJudge(ask_json=explode)
        reflection = judge.reflect({"decision": {"accepted": True}, "round": 2})
        self.assertIn("list index out of range", reflection["provider_error"])
        self.assertIn("retry", reflection["provider_error"])
        self.assertTrue(reflection["reflection_parse_retry"])
        self.assertTrue(str(reflection.get("next_policy") or ""))
        # A seat that had something to say and said too much is asked again, briefer, not dropped.
        self.assertEqual(len(calls), 2)
        self.assertIn(REFLECTION_BREVITY_RETRY, calls[1])

    def test_a_reflection_that_only_lands_on_the_second_ask_is_kept(self):
        calls = []

        def cut_off_then_complete(prompt, images):
            calls.append(prompt)
            if len(calls) == 1:
                raise ValueError("Unterminated string starting at line 1 column 4510")
            return {
                "outcome": "improved",
                "next_policy": "revise_rule",
                "rule": {"guidance": "space a crowded panel before adding anything to it"},
            }

        judge = CheckpointAdvisoryJudge(ask_json=cut_off_then_complete)
        reflection = judge.reflect({"decision": {"accepted": False, "reason": "new_blocker"}})
        self.assertEqual(reflection["next_policy"], "revise_rule")
        self.assertEqual(reflection["rule"]["guidance"], "space a crowded panel before adding anything to it")
        self.assertTrue(reflection["reflection_parse_retry"])
        self.assertFalse(reflection.get("provider_error"))

    def test_a_reflection_that_lands_first_time_is_not_asked_twice(self):
        calls = []

        def complete(prompt, images):
            calls.append(prompt)
            return {
                "outcome": "improved",
                "next_policy": "keep",
                "rule": {"type": "layout_constraint", "params": {"align_grid": 0.05}},
                "prompt_patch": {
                    "apply_at": "ppt_compilation",
                    "condition": {"defect_patterns": ["crowded_panel"]},
                    "directive": "Keep the panels on a shared alignment grid.",
                },
            }

        judge = CheckpointAdvisoryJudge(ask_json=complete)
        reflection = judge.reflect({"decision": {"accepted": True}})
        self.assertEqual(len(calls), 1)
        self.assertFalse(reflection["reflection_parse_retry"])
        self.assertFalse(reflection.get("provider_error"))

    def test_a_readable_reflection_keeps_what_the_model_said(self):
        judge = CheckpointAdvisoryJudge(
            ask_json=lambda prompt, images: {
                "outcome": "improved",
                "next_policy": "revise_rule",
                "rule": {"guidance": "when the same panel is crowded, space it before adding anything"},
            }
        )
        reflection = judge.reflect({"decision": {"accepted": True}, "round": 2})
        self.assertEqual(reflection["next_policy"], "revise_rule")
        self.assertEqual(reflection["rule"]["guidance"], "when the same panel is crowded, space it before adding anything")
        self.assertFalse(reflection.get("provider_error"))


if __name__ == "__main__":
    unittest.main()
