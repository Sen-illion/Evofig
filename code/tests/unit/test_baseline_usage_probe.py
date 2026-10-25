"""The baseline usage probe classifies responses, and never hides a call it could not measure."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "baseline_usage_probe", ROOT / "scripts" / "baseline_usage_probe" / "usage_probe.py"
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class RecordTests(unittest.TestCase):
    def test_chat_response_carries_its_token_block(self) -> None:
        line = PROBE.record(
            "https://api.example/v1/chat/completions",
            {"model": "gemini-3.1-flash-lite"},
            {"usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}},
        )
        self.assertEqual(line["kind"], "text")
        self.assertEqual(line["model"], "gemini-3.1-flash-lite")
        self.assertEqual(line["prompt_tokens"], 120)
        self.assertEqual(line["completion_tokens"], 30)

    def test_gemini_native_usage_block_is_read_too(self) -> None:
        line = PROBE.record(
            "https://api.example/v1beta/models:generateContent",
            None,
            {"usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3}},
        )
        self.assertEqual(line["kind"], "text")
        self.assertEqual(line["total_tokens"], 10)

    def test_image_endpoint_is_billed_per_image_not_per_token(self) -> None:
        line = PROBE.record(
            "https://api.example/v1/images/generations",
            {"model": "z-image-turbo"},
            {"data": [{"b64_json": "..."}]},
        )
        self.assertEqual(line["kind"], "image")
        self.assertEqual(line["model"], "z-image-turbo")
        self.assertNotIn("prompt_tokens", line)

    def test_fetching_the_generated_asset_is_not_a_billable_call(self) -> None:
        line = PROBE.record(
            "https://oss.example.com/20261009/abc.png?Expires=1&Signature=x",
            None,
            b"\x89PNG\r\n\x1a\n",
        )
        self.assertEqual(line["kind"], "other")

    def test_a_response_without_a_usage_block_is_counted_not_dropped(self) -> None:
        line = PROBE.record("https://api.example/v1/chat/completions", None, {"choices": []})
        self.assertEqual(line["kind"], "unmeasured")
        self.assertIn("usage", line["reason"])

    def test_streamed_response_is_counted_as_unmeasured(self) -> None:
        line = PROBE.record(
            "https://api.example/v1/chat/completions", None, {"choices": []}, streamed=True
        )
        self.assertEqual(line["kind"], "unmeasured")
        self.assertEqual(line["reason"], "streamed response")


class SummariseTests(unittest.TestCase):
    def _write(self, rows: list[dict]) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.close()
        self.addCleanup(lambda: os.unlink(handle.name))
        return Path(handle.name)

    def test_complete_log_reports_probe_basis(self) -> None:
        path = self._write([
            {"kind": "text", "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            {"kind": "text", "prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            {"kind": "image"},
            {"kind": "image"},
            {"kind": "image"},
        ])
        summary = PROBE.summarise(path)
        self.assertEqual(summary["usage_source"], "probe")
        self.assertEqual(summary["text_prompt_tokens"], 105)
        self.assertEqual(summary["text_completion_tokens"], 21)
        self.assertEqual(summary["image_calls"], 3)
        self.assertEqual(summary["unmeasured_calls"], 0)

    def test_one_unmeasured_call_downgrades_the_paper(self) -> None:
        path = self._write([{"kind": "text", "prompt_tokens": 1}, {"kind": "unmeasured"}])
        summary = PROBE.summarise(path)
        self.assertEqual(summary["usage_source"], "probe_partial")
        self.assertEqual(summary["unmeasured_calls"], 1)

    def test_absent_log_is_missing_not_zero(self) -> None:
        summary = PROBE.summarise("/nonexistent/usage.jsonl")
        self.assertEqual(summary["usage_source"], "missing")
        self.assertEqual(summary["text_calls"], 0)

    def test_a_download_an_older_probe_called_unmeasured_is_not_a_gap(self) -> None:
        """A row already on disk is not bought again just to be relabelled."""

        path = self._write([
            {"kind": "text", "prompt_tokens": 100, "completion_tokens": 20},
            {
                "kind": "unmeasured",
                "reason": "no usage block in response",
                "url": "https://oss.example/7d/36/a/b.png?Expires=1&Signature=x",
            },
        ])
        summary = PROBE.summarise(path)
        self.assertEqual(summary["usage_source"], "probe")
        self.assertEqual(summary["unmeasured_calls"], 0)
        self.assertEqual(summary["text_prompt_tokens"], 100)

    def test_torn_line_is_skipped(self) -> None:
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
        handle.write('{"kind": "text", "prompt_tokens": 9}\n')
        handle.write('{"kind": "text", "prompt_tok\n')
        handle.close()
        self.addCleanup(lambda: os.unlink(handle.name))
        summary = PROBE.summarise(Path(handle.name))
        self.assertEqual(summary["text_prompt_tokens"], 9)
        self.assertEqual(summary["usage_source"], "probe")


class GateTests(unittest.TestCase):
    def test_probe_writes_nothing_until_a_driver_names_an_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "usage.jsonl"
            os.environ.pop("RFS_USAGE_PROBE_OUT", None)
            PROBE.note("https://api.example/v1/chat/completions", None, {"usage": {"prompt_tokens": 1}})
            self.assertFalse(target.exists())
            self.assertFalse(PROBE.enabled())


class InstallTests(unittest.TestCase):
    """The transports the two baseline stacks actually use must all be wrapped."""

    def test_requests_and_both_httpx_clients_are_wrapped(self) -> None:
        import requests

        original = requests.adapters.HTTPAdapter.send
        self.addCleanup(lambda: setattr(requests.adapters.HTTPAdapter, "send", original))
        wrapped = PROBE.install()
        self.assertIn("requests", wrapped)
        self.assertTrue(getattr(requests.adapters.HTTPAdapter.send, "_rfs_usage_probe", False))

        if importlib.util.find_spec("httpx") is None:
            return  # httpx only ships on the baseline machines, not in every dev checkout
        import httpx

        sync_original = httpx.Client.send
        async_original = httpx.AsyncClient.send
        self.addCleanup(lambda: setattr(httpx.Client, "send", sync_original))
        self.addCleanup(lambda: setattr(httpx.AsyncClient, "send", async_original))
        wrapped = PROBE.install()
        self.assertIn("httpx", wrapped)
        self.assertIn("httpx-async", wrapped)
        self.assertTrue(getattr(httpx.Client.send, "_rfs_usage_probe", False))
        self.assertTrue(getattr(httpx.AsyncClient.send, "_rfs_usage_probe", False))


if __name__ == "__main__":
    unittest.main()
