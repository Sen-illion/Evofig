import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rfs.providers.gemini_image import generate_gemini_image, is_gemini_image_model


class GeminiImageTests(unittest.TestCase):
    def test_gemini_image_model_names(self):
        self.assertTrue(is_gemini_image_model("gemini-2.5-flash-image"))
        self.assertFalse(is_gemini_image_model("gpt-image-2"))
        self.assertFalse(is_gemini_image_model("gemini-3.1-flash-lite"))

    def test_generate_writes_inline_image_and_usage(self):
        encoded = base64.b64encode(b"png-bytes").decode("ascii")
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": encoded}}]}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 1290, "totalTokenCount": 1302},
        }
        env = {"GEMINI_GEN_IMG_URL": "https://images.example/v1beta/models/gemini-2.5-flash-image:generateContent", "API_KEY": "secret"}
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, env), patch("rfs.providers.gemini_image.requests.post", return_value=response) as post:
            icon = Path(temp) / "icon.png"
            result = generate_gemini_image(icon, "a neural encoder", "gemini-2.5-flash-image")
            self.assertEqual(icon.read_bytes(), b"png-bytes")
        self.assertEqual(post.call_args.args[0], "https://images.example/v1beta/models/gemini-2.5-flash-image:generateContent")
        self.assertEqual(post.call_args.kwargs["json"]["generationConfig"]["responseModalities"], ["IMAGE"])
        self.assertEqual(result["usage"]["image_count"], 1)
        self.assertEqual(result["usage"]["completion_tokens"], 1290)
