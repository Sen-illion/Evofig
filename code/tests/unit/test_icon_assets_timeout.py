import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rfs.generation.icon_assets import _image2


class IconAssetsTimeoutTests(unittest.TestCase):
    def test_image2_timeout_uses_environment_override(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"b64_json": "aGVsbG8="}]}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "secret", "RFS_IMAGE2_TIMEOUT": "300"}), patch("rfs.generation.icon_assets.requests.post", return_value=response) as post:
            _image2(Path(temp) / "icon.png", "prompt", "gpt-image-2")
        self.assertEqual(post.call_args.kwargs["timeout"], 300)

    def test_image2_retries_rate_limit(self):
        limited = Mock(status_code=429, headers={"Retry-After": "0"})
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {"data": [{"b64_json": "aGVsbG8="}]}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"API_BASE": "https://example.test/v1", "API_KEY": "secret", "RFS_IMAGE2_RATE_LIMIT_RETRIES": "1"}), patch("rfs.generation.icon_assets.requests.post", side_effect=[limited, response]) as post:
            result = _image2(Path(temp) / "icon.png", "prompt", "gpt-image-2")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result["api_requests_attempted"], 2)
