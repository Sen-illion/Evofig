import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from rfs.providers.client import call_vlm_json, extract_json


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {"choices": [{"message": {"content": '{"summary":"ok"}'}}]}

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self
            raise error

    def json(self):
        return self._payload


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}


class VlmClientTests(unittest.TestCase):
    def test_extract_json_accepts_a_raw_newline_inside_a_string(self):
        raw = '{\n  "summary": "line one\nline two",\n  "ok": true\n}'
        parsed = extract_json(raw)
        self.assertEqual(parsed["summary"], "line one\nline two")
        self.assertTrue(parsed["ok"])

    def test_extract_json_replaces_a_lone_surrogate(self):
        # Half of an astral character survives json.loads but raises on every
        # utf-8 encode; the parsed answer must come back encodable instead of
        # aborting a paid stage when the artifact is written.
        parsed = extract_json('{"label": "\\ud835 tail"}')
        self.assertEqual(parsed["label"], "\ufffd tail")

    def test_extract_json_still_rejects_a_truncated_answer(self):
        with self.assertRaises(Exception):
            extract_json('{"summary": "no closing brace"')

    def test_extract_json_repairs_a_missing_comma_between_entries(self):
        # "Expecting ',' delimiter" is what nearly every unparseable plan says;
        # the content is intact, only the punctuation is missing.
        parsed = extract_json('{\n  "summary": "first"\n  "ok": true\n}')
        self.assertEqual(parsed, {"summary": "first", "ok": True})

    def test_extract_json_repairs_a_trailing_comma(self):
        parsed = extract_json('{"nodes": ["a", "b",], "ok": true,}')
        self.assertEqual(parsed["nodes"], ["a", "b"])
        self.assertTrue(parsed["ok"])

    def test_extract_json_unwraps_an_answer_a_gateway_wrapped(self):
        # The provider handed the plan back wrapped at a fixed column, so line
        # breaks landed inside an identifier and inside a number. Re-sending
        # that request reproduces the same wrapping, so the answer has to be
        # read instead of bought twice.
        wrapped = (
            '{\n  "nodes": [\n    {\n      "id": "node_pretrained_\npolicy",\n'
            '      "bbox_percent": {\n        "x": 0.\n05,\n        "y": 0.15\n      }\n    }\n  ],\n'
            '  "canvas": {"aspect_ratio": "16:9"\n\n  }\n}'
        )
        parsed = extract_json(wrapped)
        self.assertEqual(parsed["nodes"][0]["id"], "node_pretrained_policy")
        self.assertEqual(parsed["nodes"][0]["bbox_percent"]["x"], 0.05)
        self.assertEqual(parsed["canvas"]["aspect_ratio"], "16:9")

    def test_extract_json_unwraps_a_wrapped_answer_that_lost_a_comma(self):
        wrapped = '{\n  "first": 1\n  "second": "two_\nparts"\n}'
        parsed = extract_json(wrapped)
        self.assertEqual(parsed, {"first": 1, "second": "two_parts"})

    def test_extract_json_reads_a_backslash_the_model_meant_literally(self):
        # Two plans in the 2026-09-21 corpus run died on "Invalid \escape" and
        # fell back to a placeholder stage. The backslash was meant literally
        # (a path, a LaTeX macro), so making it literal is the whole repair.
        windows_path = '{"pdf": "<PATH>", "note": "see \\alpha"}'
        parsed = extract_json(windows_path)
        self.assertEqual(parsed["pdf"], "<PATH>")
        self.assertEqual(parsed["note"], "see \\alpha")

    def test_extract_json_keeps_the_escapes_json_does_allow(self):
        answer = '{"text": "line\\nbreak \\"quoted\\" \\u00e9 \\\\ done"}'
        parsed = extract_json(answer)
        self.assertEqual(parsed["text"], 'line\nbreak "quoted" \u00e9 \\ done')

    def test_unwrapping_never_finishes_a_cut_off_answer(self):
        truncated = '{\n  "summary": "the model stopped mid_\nsentence'
        with self.assertRaises(Exception):
            extract_json(truncated)

    def test_a_cut_off_answer_is_never_completed_by_guessing(self):
        # Closing the brackets here would deliver half a plan as if it were the
        # whole one, which is worse than asking for it again.
        with self.assertRaises(Exception):
            extract_json('{"a": [1, 2, "b"}')

    def test_a_malformed_answer_is_retried_with_the_parser_complaint(self):
        cut_off = '{"summary": "half", "nodes": ['
        complete = '{"summary": "half", "nodes": ["a"]}'
        metadata = {}
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "reference.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
                with patch("rfs.providers.client.time.sleep"), patch("rfs.providers.client.requests.post") as post:
                    post.side_effect = [
                        FakeResponse(payload={"choices": [{"message": {"content": cut_off}}]}),
                        FakeResponse(payload={"choices": [{"message": {"content": complete}}]}),
                    ]
                    result = call_vlm_json("Return the plan as JSON", [image], timeout=7, retries=1, call_metadata=metadata)
        self.assertEqual(result["nodes"], ["a"])
        self.assertEqual(post.call_count, 2)
        first = json.loads(post.call_args_list[0].kwargs["data"])
        second = json.loads(post.call_args_list[1].kwargs["data"])
        self.assertEqual(len(first["messages"][0]["content"]), 2)
        # The repair carries the text only: the images are the expensive part
        # and the defect is syntax, not evidence.
        self.assertEqual(len(second["messages"][0]["content"]), 1)
        complaint = second["messages"][0]["content"][0]["text"]
        self.assertIn("Return the plan as JSON", complaint)
        self.assertIn("PARSER ERROR", complaint)
        self.assertIn('"nodes": [', complaint)
        self.assertEqual(metadata["failure_categories"], ["invalid_json"])
        self.assertEqual(metadata["attempts"], 2)
        self.assertTrue(metadata["success"])

    def test_call_vlm_json_retries_timeout_then_succeeds(self):
        with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.providers.client.time.sleep") as sleep, patch("rfs.providers.client.requests.post") as post:
                post.side_effect = [requests.Timeout("slow"), FakeResponse()]
                result = call_vlm_json("prompt", [], timeout=7, retries=1)
        self.assertEqual(result["summary"], "ok")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs["timeout"], 7)
        sleep.assert_called_once()

    def test_call_vlm_json_retries_429_and_5xx(self):
        with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.providers.client.time.sleep"), patch("rfs.providers.client.requests.post") as post:
                post.side_effect = [FakeResponse(429), FakeResponse(502), FakeResponse()]
                result = call_vlm_json("prompt", [], timeout=7, retries=2)
        self.assertEqual(result["summary"], "ok")
        self.assertEqual(post.call_count, 3)

    def test_a_rejected_request_keeps_the_provider_message_and_is_not_repeated(self):
        # A 400 is the request's own fault: repeating it spends budget and still
        # fails. The body has to survive, otherwise "400 Bad Request" says nothing.
        rejected = FakeResponse(400)
        rejected.text = '{"error":{"message":"prompt is too long"}}'
        metadata = {}
        with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.providers.client.time.sleep") as sleep, patch("rfs.providers.client.requests.post") as post:
                post.return_value = rejected
                with self.assertRaises(RuntimeError) as caught:
                    call_vlm_json("prompt", [], timeout=7, retries=2, call_metadata=metadata)
        self.assertEqual(post.call_count, 1)
        self.assertIn("400", str(caught.exception))
        self.assertIn("prompt is too long", str(caught.exception))
        sleep.assert_not_called()
        self.assertEqual(metadata["failure_categories"], ["http"])

    @patch("rfs.providers.client.requests.post")
    def test_records_provider_reported_token_usage(self, post):
        post.return_value = FakeResponse(payload={"choices": [{"message": {"content": '{"summary":"ok"}'}}], "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500}})
        metadata = {}
        with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "secret-token"}, clear=False):
            call_vlm_json("Return JSON", [], model="test-model", timeout=10, retries=0, call_metadata=metadata)

        self.assertEqual(metadata["token_source"], "provider")
        self.assertEqual(metadata["prompt_tokens"], 1200)
        self.assertEqual(metadata["completion_tokens"], 300)
        self.assertEqual(metadata["total_tokens"], 1500)
        self.assertEqual(metadata["usage"]["total_tokens"], 1500)
        self.assertNotIn("secret-token", str(metadata))

    @patch("rfs.providers.client.requests.post")
    def test_falls_back_to_estimated_tokens_without_provider_usage(self, post):
        post.return_value = FakeResponse()
        metadata = {}
        with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "secret-token"}, clear=False):
            call_vlm_json("Return JSON", [], model="test-model", timeout=10, retries=0, call_metadata=metadata)

        self.assertEqual(metadata["token_source"], "estimated")
        self.assertEqual(metadata["prompt_tokens"], metadata["estimated_tokens"])
        self.assertIsNone(metadata["usage"])
        self.assertIsNone(metadata["completion_tokens"])
        self.assertIsNone(metadata["total_tokens"])

    @patch("rfs.providers.client.time.sleep", return_value=None)
    @patch("rfs.providers.client.requests.post", side_effect=requests.Timeout("slow"))
    def test_failed_call_still_accounts_prompt_tokens(self, _post, _sleep):
        metadata = {}
        with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "secret-token"}, clear=False):
            with self.assertRaises(requests.Timeout):
                call_vlm_json("Return JSON", [], model="test-model", timeout=10, retries=1, call_metadata=metadata)

        self.assertEqual(metadata["token_source"], "estimated")
        self.assertEqual(metadata["prompt_tokens"], metadata["estimated_tokens"])
        self.assertEqual(metadata["attempts"], 2)

    @patch("rfs.providers.client.requests.post")
    def test_usage_record_keeps_partial_provider_payload(self, post):
        post.return_value = FakeResponse(payload={"choices": [{"message": {"content": '{"summary":"ok"}'}}], "usage": {"completion_tokens": "42"}})
        metadata = {}
        with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "secret-token"}, clear=False):
            call_vlm_json("Return JSON", [], model="test-model", timeout=10, retries=0, call_metadata=metadata)

        self.assertEqual(metadata["token_source"], "provider")
        self.assertEqual(metadata["completion_tokens"], 42)
        self.assertIsNone(metadata["prompt_tokens"])

    @patch("rfs.providers.client.time.sleep", return_value=None)
    @patch("rfs.providers.client.requests.post", side_effect=[requests.exceptions.SSLError("EOF during TLS"), _Response()])
    def test_retry_metadata_records_recovery_without_secrets(self, _post, _sleep):
        metadata = {}
        with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "secret-token"}, clear=False):
            result = call_vlm_json("Return JSON", [], model="test-model", timeout=10, retries=1, call_metadata=metadata)

        self.assertTrue(result["ok"])
        self.assertEqual(metadata["attempts"], 2)
        self.assertEqual(metadata["retries_used"], 1)
        self.assertTrue(metadata["success"])
        self.assertEqual(metadata["failure_categories"], ["tls"])
        self.assertNotIn("secret-token", str(metadata))

    @patch("rfs.providers.client.requests.post")
    def test_expired_deadline_skips_provider_call(self, post):
        metadata = {}
        with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "secret-token"}, clear=False):
            with self.assertRaisesRegex(TimeoutError, "deadline budget exhausted"):
                call_vlm_json("Return JSON", [], model="test-model", timeout=30, retries=2, call_metadata=metadata, deadline_at=0.0)

        post.assert_not_called()
        self.assertEqual(metadata["attempts"], 0)
        self.assertEqual(metadata["retries_used"], 0)
        self.assertEqual(metadata["failure_categories"], ["timeout"])
        self.assertTrue(metadata["deadline_reached"])

    def test_an_empty_completion_is_not_retried_as_invalid_json(self):
        metadata = {}
        with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.providers.client.time.sleep") as sleep, patch("rfs.providers.client.requests.post") as post:
                post.return_value = FakeResponse(payload={"choices": [{"message": {"content": ""}, "finish_reason": "empty"}]})
                with self.assertRaisesRegex(RuntimeError, "empty completion"):
                    call_vlm_json("judge this pair", [], timeout=180, retries=2, call_metadata=metadata)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(metadata["failure_categories"], ["empty_completion"])
        self.assertEqual(metadata["attempts"], 1)
        self.assertFalse(metadata["success"])

    def test_a_truncated_completion_is_not_retried_as_invalid_json(self):
        metadata = {}
        with patch.dict("os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "key"}, clear=False):
            with patch("rfs.providers.client.time.sleep") as sleep, patch("rfs.providers.client.requests.post") as post:
                post.return_value = FakeResponse(payload={"choices": [{"message": {"content": '{"diagnosis": ['}, "finish_reason": "length"}]})
                with self.assertRaisesRegex(RuntimeError, "token cap"):
                    call_vlm_json("audit the figure", [], timeout=180, retries=2, call_metadata=metadata)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(metadata["failure_categories"], ["truncated_completion"])


if __name__ == "__main__":
    unittest.main()
