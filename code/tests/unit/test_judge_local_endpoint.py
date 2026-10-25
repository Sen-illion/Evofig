"""Unit tests for the local Judge endpoint's request parsing.

The server itself needs a GPU stack, so the parts tested here are the pure helpers that
turn one chat request into prompt text and inline images, plus the two response wrappers
the client reads back. Those are the parts that decide whether a trained checkpoint is
reached at all, and they run without torch installed.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_endpoint():
    spec = importlib.util.spec_from_file_location("judge_serve_local_under_test", ROOT / "scripts" / "judge_serve_local.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


endpoint = load_endpoint()
PNG_BYTES = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


class RequestParsingTests(unittest.TestCase):
    def test_a_plain_string_content_is_one_text_block(self):
        self.assertEqual(endpoint.content_blocks("hello"), [{"type": "text", "text": "hello"}])

    def test_a_list_content_keeps_only_the_object_blocks(self):
        blocks = endpoint.content_blocks([{"type": "text", "text": "a"}, "junk", 7, {"type": "image_url", "image_url": {"url": "data:x;base64,AA=="}}])
        self.assertEqual(len(blocks), 2)
        self.assertEqual(endpoint.text_from_blocks(blocks), "a")

    def test_any_other_content_becomes_no_blocks(self):
        self.assertEqual(endpoint.content_blocks(None), [])
        self.assertEqual(endpoint.content_blocks(42), [])

    def test_text_blocks_are_joined_in_order(self):
        blocks = [{"type": "text", "text": "first"}, {"type": "image_url", "image_url": {"url": "data:x;base64,AA=="}}, {"type": "text", "text": "second"}]
        self.assertEqual(endpoint.text_from_blocks(blocks), "first\nsecond")

    def test_an_image_url_is_read_from_both_shapes(self):
        blocks = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
            {"type": "image", "image_url": "data:image/png;base64,BB=="},
            {"type": "image_url", "image_url": {"url": ""}},
        ]
        self.assertEqual(endpoint.image_urls_from_blocks(blocks), ["data:image/png;base64,AA==", "data:image/png;base64,BB=="])

    def test_a_data_url_is_decoded_to_its_bytes(self):
        url = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")
        self.assertEqual(endpoint.decode_data_url(url), PNG_BYTES)

    def test_a_remote_url_is_refused(self):
        with self.assertRaises(ValueError):
            endpoint.decode_data_url("https://example.invalid/figure.png")

    def test_a_non_base64_payload_is_refused(self):
        with self.assertRaises(ValueError):
            endpoint.decode_data_url("data:image/png;base64,not base64!!")

    def test_one_request_yields_the_prompt_and_the_images(self):
        payload = {
            "model": "judge-pilot-v001",
            "messages": [
                {"role": "system", "content": "answer as JSON"},
                {"role": "user", "content": [{"type": "text", "text": "which is better?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode("ascii")}}]},
            ],
        }
        text, images = endpoint.read_messages(payload)
        self.assertEqual(text, "answer as JSON\nwhich is better?")
        self.assertEqual(images, [PNG_BYTES])

    def test_a_request_without_messages_is_refused(self):
        with self.assertRaises(ValueError):
            endpoint.read_messages({})

    def test_a_request_without_text_is_refused(self):
        payload = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]}
        with self.assertRaises(ValueError):
            endpoint.read_messages(payload)


class PromptModeTests(unittest.TestCase):
    def test_the_serving_prompt_closes_the_thinking_block_by_default(self):
        self.assertEqual(endpoint.chat_template_kwargs(False), {"enable_thinking": False})

    def test_thinking_is_opt_in(self):
        self.assertEqual(endpoint.chat_template_kwargs(True), {"enable_thinking": True})

    def test_the_flag_defaults_to_no_thinking(self):
        self.assertFalse(endpoint.build_parser().parse_args(["--base-model", "m"]).thinking)
        self.assertTrue(endpoint.build_parser().parse_args(["--base-model", "m", "--thinking"]).thinking)

    def test_the_decode_budget_defaults_to_1024(self):
        self.assertEqual(endpoint.DEFAULT_MAX_NEW_TOKENS, 1024)
        self.assertEqual(endpoint.build_parser().parse_args(["--base-model", "m"]).max_new_tokens, 1024)
        self.assertEqual(endpoint.build_parser().parse_args(["--base-model", "m", "--max-new-tokens", "1536"]).max_new_tokens, 1536)

    def test_the_vision_budget_is_opt_in_and_matches_the_training_value(self):
        # Serving and scoring must see the same picture: the flag exists so the server can
        # apply the same cap `judge train` and `judge score-choice` use. Absent means the
        # processor default, which is what older endpoints did.
        self.assertEqual(endpoint.build_parser().parse_args(["--base-model", "m"]).image_max_pixels, 0)
        self.assertEqual(endpoint.build_parser().parse_args(["--base-model", "m", "--image-max-pixels", "250000"]).image_max_pixels, 250000)


class ResponseShapeTests(unittest.TestCase):
    def test_a_completion_is_wrapped_in_the_shape_the_client_reads(self):
        body = endpoint.completion_body(model="judge-pilot-v001", answer='{"status": "pass"}', prompt_tokens=11, completion_tokens=3)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "judge-pilot-v001")
        self.assertEqual(body["choices"][0]["message"], {"role": "assistant", "content": '{"status": "pass"}'})
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(body["usage"], {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14})

    def test_an_empty_decode_is_labelled_empty_not_stop(self):
        self.assertEqual(endpoint.classify_completion("", 0, 1024), "empty")
        self.assertEqual(endpoint.classify_completion("   ", 12, 1024), "empty")
        body = endpoint.completion_body(model="j", answer="", prompt_tokens=11, completion_tokens=0, finish_reason="empty")
        self.assertEqual(body["choices"][0]["finish_reason"], "empty")
        self.assertEqual(body["choices"][0]["message"]["content"], "")

    def test_a_capped_unparseable_answer_is_labelled_length(self):
        self.assertEqual(endpoint.classify_completion('{"diagnosis": [', 1024, 1024), "length")
        self.assertEqual(endpoint.classify_completion('{"ok": true}', 1024, 1024), "stop")
        self.assertEqual(endpoint.classify_completion('{"ok": true}', 40, 1024), "stop")

    def test_an_error_is_wrapped_where_the_client_can_read_it(self):
        body = endpoint.error_body("boom")
        self.assertEqual(body["error"]["message"], "boom")


class LaunchGuardTests(unittest.TestCase):
    def test_a_missing_adapter_directory_stops_before_loading_anything(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = endpoint.main(["--base-model", "some/model", "--adapter", str(ROOT / "no-such-adapter"), "--port", "0"])
        self.assertEqual(status, 2)
        self.assertIn("adapter_config.json", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
