import os
import unittest
from unittest import mock

from rfs.providers.client import resolve_vlm_retries, resolve_vlm_timeout


class VlmTimeoutResolutionTests(unittest.TestCase):
    def test_explicit_values_pass_through(self):
        self.assertEqual(resolve_vlm_timeout(42), 42)
        self.assertEqual(resolve_vlm_retries(3), 3)

    def test_none_falls_back_to_the_documented_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RFS_VLM_TIMEOUT", None)
            os.environ.pop("RFS_VLM_RETRIES", None)
            self.assertEqual(resolve_vlm_timeout(None), 180)
            self.assertEqual(resolve_vlm_retries(None), 1)

    def test_environment_overrides_the_default(self):
        with mock.patch.dict(os.environ, {"RFS_VLM_TIMEOUT": "90", "RFS_VLM_RETRIES": "2"}):
            self.assertEqual(resolve_vlm_timeout(None), 90)
            self.assertEqual(resolve_vlm_retries(None), 2)

    def test_a_junk_environment_value_is_ignored(self):
        with mock.patch.dict(os.environ, {"RFS_VLM_TIMEOUT": "not-a-number"}):
            self.assertEqual(resolve_vlm_timeout(None), 180)


class CallVlmJsonAcceptsNoneTests(unittest.TestCase):
    def test_none_timeout_and_retries_do_not_crash_before_the_request(self):
        from rfs.providers import client

        with mock.patch.dict(os.environ, {"API_BASE": "https://example.invalid", "API_KEY": "x"}):
            with mock.patch.object(client.requests, "post", side_effect=RuntimeError("no network")):
                with self.assertRaises(Exception) as ctx:
                    client.call_vlm_json("prompt", [], timeout=None, retries=None)
        self.assertNotIn("NoneType", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
