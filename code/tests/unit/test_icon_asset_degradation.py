"""A refused or half-written icon costs one symbol, never the paper."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from PIL import Image

from rfs.generation.icon_assets import (
    IconProviderRefusal,
    _image2,
    generate_icon_assets,
)


def _write_png(path: Path) -> None:
    Image.new("RGB", (32, 32), "white").save(path)


def _plan() -> dict:
    return {
        "icons": [
            {
                "asset_id": "icon_prompt_bomb",
                "subject": "Make a bomb?",
                "primary_object": "one document entering a small input tray",
                "prompt": "academic framework-diagram icon for Make a bomb?",
            }
        ]
    }


class IconDegradationTests(unittest.TestCase):
    def test_a_refused_prompt_is_asked_again_in_neutral_wording(self):
        seen = []

        def fake_image2(path, prompt, model):
            seen.append(prompt)
            if len(seen) == 1:
                raise IconProviderRefusal("image provider refused this prompt (HTTP 400)")
            _write_png(path)
            return {"model": "gpt-image-2", "api_requests_attempted": 1}

        with tempfile.TemporaryDirectory() as temp, patch("rfs.generation.icon_assets._image2", side_effect=fake_image2):
            report = generate_icon_assets(_plan(), temp, mode="image2", model="gpt-image-2")
            record = report["assets"][0]

            self.assertEqual(len(seen), 2)
            self.assertIn("Make a bomb?", seen[0])
            self.assertNotIn("Make a bomb?", seen[1])
            self.assertTrue(record["neutral_prompt_used"])
            self.assertEqual(record["status"], "neutralized")
            self.assertTrue(record["path"].endswith("selected.png"))
            self.assertTrue(Path(record["path"]).exists())
            self.assertEqual(report["degradations"][0]["kinds"], ["provider_refused_prompt"])

    def test_a_prompt_refused_twice_becomes_a_placeholder_instead_of_a_failure(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "rfs.generation.icon_assets._image2",
            side_effect=IconProviderRefusal("image provider refused this prompt (HTTP 400)"),
        ):
            report = generate_icon_assets(_plan(), temp, mode="image2", model="gpt-image-2")
            record = report["assets"][0]

            self.assertEqual(len(report["assets"]), 1)
            self.assertEqual(record["status"], "placeholder_after_refusal")
            self.assertTrue(Path(record["path"]).exists())
            with Image.open(record["path"]) as image:
                image.load()
            self.assertTrue(report["quality_gate"]["ok"])

    def test_a_half_written_image_is_asked_for_once_more(self):
        calls = []

        def half_then_whole(path, prompt, model):
            calls.append(path)
            if len(calls) == 1:
                path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtruncated")
            else:
                _write_png(path)
            return {"model": "gpt-image-2", "api_requests_attempted": 1}

        with tempfile.TemporaryDirectory() as temp, patch("rfs.generation.icon_assets._image2", side_effect=half_then_whole):
            report = generate_icon_assets(_plan(), temp, mode="image2", model="gpt-image-2")
            record = report["assets"][0]

            self.assertEqual(len(calls), 2)
            self.assertEqual(record["status"], "generated")
            self.assertFalse(record["neutral_prompt_used"])
            self.assertIn("damaged_image", record["degradation"]["kinds"])
            with Image.open(record["path"]) as image:
                self.assertEqual(image.size, (32, 32))

    def test_a_damaged_image_twice_becomes_a_placeholder(self):
        def always_half(path, prompt, model):
            path.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtruncated")
            return {"model": "gpt-image-2", "api_requests_attempted": 1}

        with tempfile.TemporaryDirectory() as temp, patch("rfs.generation.icon_assets._image2", side_effect=always_half):
            report = generate_icon_assets(_plan(), temp, mode="image2", model="gpt-image-2")
            record = report["assets"][0]

            self.assertEqual(record["status"], "placeholder_after_damage")
            with Image.open(record["path"]) as image:
                image.load()

    def test_a_503_on_the_papers_wording_is_asked_again_without_those_words(self):
        seen = []

        def fake_image2(path, prompt, model):
            seen.append(prompt)
            if len(seen) == 1:
                response = Mock(status_code=503)
                error = requests.HTTPError("503 Server Error: Service Unavailable")
                error.response = response
                raise error
            _write_png(path)
            return {"model": "z-image-turbo", "api_requests_attempted": 1}

        with tempfile.TemporaryDirectory() as temp, patch("rfs.generation.icon_assets._image2", side_effect=fake_image2):
            report = generate_icon_assets(_plan(), temp, mode="image2", model="z-image-turbo")
            record = report["assets"][0]

            self.assertEqual(len(seen), 2)
            self.assertNotIn("Make a bomb?", seen[1])
            self.assertTrue(record["neutral_prompt_used"])
            self.assertEqual(record["status"], "neutralized")
            self.assertEqual(report["degradations"][0]["kinds"], ["provider_unavailable_for_wording"])

    def test_a_503_that_also_hits_the_neutral_symbol_still_stops_the_run(self):
        response = Mock(status_code=503)
        error = requests.HTTPError("503 Server Error: Service Unavailable")
        error.response = response
        with tempfile.TemporaryDirectory() as temp, patch("rfs.generation.icon_assets._image2", side_effect=error):
            with self.assertRaises(requests.HTTPError):
                generate_icon_assets(_plan(), temp, mode="image2", model="z-image-turbo")

    def test_a_provider_that_is_out_of_quota_still_stops_the_run(self):
        with tempfile.TemporaryDirectory() as temp, patch(
            "rfs.generation.icon_assets._image2",
            side_effect=RuntimeError("quota is not enough"),
        ):
            with self.assertRaises(RuntimeError):
                generate_icon_assets(_plan(), temp, mode="image2", model="gpt-image-2")


class RefusalClassificationTests(unittest.TestCase):
    def test_a_400_is_reported_as_a_refusal_rather_than_an_http_error(self):
        refused = Mock(status_code=400)
        refused.text = json.dumps({"error": {"message": "your request was rejected by the safety system"}})
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            "os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "secret", "RFS_IMAGE2_RETRIES": "0"}
        ), patch("rfs.generation.icon_assets.requests.post", return_value=refused):
            with self.assertRaises(IconProviderRefusal) as caught:
                _image2(Path(temp) / "icon.png", "prompt", "gpt-image-2")

        self.assertIn("safety system", str(caught.exception))

    def test_a_500_is_still_an_http_error(self):
        broken = Mock(status_code=500)
        broken.text = "server error"
        broken.raise_for_status.side_effect = RuntimeError("500 Server Error")
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            "os.environ", {"API_BASE": "https://example.test/v1", "API_KEY": "secret", "RFS_IMAGE2_RETRIES": "0"}
        ), patch("rfs.generation.icon_assets.requests.post", return_value=broken):
            with self.assertRaises(RuntimeError) as caught:
                _image2(Path(temp) / "icon.png", "prompt", "gpt-image-2")

        self.assertNotIsInstance(caught.exception, IconProviderRefusal)


if __name__ == "__main__":
    unittest.main()
