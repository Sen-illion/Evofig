"""Serve one locally trained Judge checkpoint over the Judge's own JSON API contract.

Every caller in this pipeline -- the evolution loop, the baseline batch, the shadow
comparison and the handover -- reaches a Judge through ``rfs.providers.client.call_vlm_json``,
which posts an OpenAI-shaped chat request to ``{API_BASE}/chat/completions``. That is the
only shape a Judge has ever been called in, so a checkpoint we trained replaces a vendor
endpoint by answering the same request on our own machine: no caller changes, and no second
in-process calling path to drift from the first.

The server loads the base model once, optionally stacks a LoRA adapter saved by
``rfs judge train``, and answers with the model's own text. It never post-processes the
answer into JSON: the client already parses the answer and re-asks with the parser's
complaint when the model emits something else, and rewriting that here would hide the
model's mistakes from the very pipeline that is supposed to learn from them.

Only the standard library is imported at module scope. ``torch``, ``transformers`` and
``peft`` are imported when the model is loaded, so the request-parsing helpers below stay
importable and testable on a machine that has no GPU stack installed.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_NEW_TOKENS = 1024
IMAGE_BLOCK_TYPES = ("image_url", "image")


def content_blocks(content: Any) -> list[dict[str, Any]]:
    """Return the content of one chat message as a list of blocks.

    A plain string is the ordinary single-text-block case; a list is the multimodal one.
    """

    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]
    return []


def text_from_blocks(blocks: Iterable[dict[str, Any]]) -> str:
    """Join every text block of one message."""

    parts = [str(block.get("text") or "") for block in blocks if str(block.get("type") or "") in ("text", "input_text")]
    return "\n".join(part for part in parts if part)


def image_urls_from_blocks(blocks: Iterable[dict[str, Any]]) -> list[str]:
    """Collect the image URLs of one message, in the order they were sent."""

    urls: list[str] = []
    for block in blocks:
        if str(block.get("type") or "") not in IMAGE_BLOCK_TYPES:
            continue
        source = block.get("image_url")
        url = str((source or {}).get("url") or "") if isinstance(source, dict) else str(source or "")
        if url:
            urls.append(url)
    return urls


def decode_data_url(url: str) -> bytes:
    """Decode one ``data:`` URL into raw bytes.

    Only inline data URLs are accepted: the pipeline sends previews that way, and fetching a
    remote URL would let a request make this server reach the network on its behalf.
    """

    head, _, payload = str(url).partition(",")
    if not head.startswith("data:") or not head.endswith(";base64"):
        raise ValueError("only base64 data URLs are accepted")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"the image payload is not valid base64: {exc}") from exc


def chat_template_kwargs(thinking: bool) -> dict[str, Any]:
    """Keyword arguments that keep the prompt inside the distribution the Judge trained on.

    Training renders the assistant turn with ``add_generation_prompt=False`` and the JSON completion,
    so the model never sees a `` thinking`` prefix. Serving with the template default
    (``add_generation_prompt=True``) appends one, which drops the checkpoint into a reasoning mode it
    was never trained for: the base model answers with pages of analysis and never reaches the JSON, and
    at any token cap the answer is the analysis rather than the judgement. ``enable_thinking`` closes
    that block; ``--thinking`` puts the template default back for a caller that wants it.
    """

    return {"enable_thinking": bool(thinking)}


def classify_completion(answer: str, completion_tokens: int, max_new_tokens: int) -> str:
    """Return the OpenAI-shaped ``finish_reason`` the client uses to decide whether to retry.

    An empty decode used to go out as ``stop`` with ``content=""``. The client then treated it as
    a JSON syntax error and re-asked until the timeout budget was gone. ``empty`` and ``length``
    are the two cases where asking again cannot help: the model wrote nothing, or it filled the
    token cap without ever closing the JSON.
    """

    body = str(answer or "").strip()
    if not body:
        return "empty"
    if int(completion_tokens) < int(max_new_tokens):
        return "stop"
    try:
        from rfs.providers.client import extract_json

        extract_json(body)
    except Exception:
        return "length"
    return "stop"


def read_messages(payload: dict[str, Any]) -> tuple[str, list[bytes]]:
    """Turn one request into the prompt text and the inline images that go with it."""

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("the request carries no messages")
    texts: list[str] = []
    images: list[bytes] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        blocks = content_blocks(message.get("content"))
        text = text_from_blocks(blocks)
        if text:
            texts.append(text)
        images.extend(decode_data_url(url) for url in image_urls_from_blocks(blocks))
    if not texts:
        raise ValueError("the request carries no text content")
    return "\n".join(texts), images


class LocalJudge:
    """One checkpoint answering the Judge question on this machine."""

    def __init__(self, *, base_model: str, adapter: str = "", load_in_4bit: bool = True, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, thinking: bool = False, image_max_pixels: int = 0) -> None:
        self.base_model = base_model
        self.adapter = adapter
        self.load_in_4bit = bool(load_in_4bit)
        self.max_new_tokens = int(max_new_tokens)
        self.thinking = bool(thinking)
        self.image_max_pixels = int(image_max_pixels or 0)
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {"loaded": False, "served_model_name": "", "device": "", "adapter": str(adapter or "")}
        self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor

        from rfs.judge_training.training import resolve_vision_auto_class

        import transformers

        kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "device_map": {"": 0}}
        if self.load_in_4bit:
            import bitsandbytes  # noqa: F401 - imported for its side effect of registering the quantiser

            kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.processor = AutoProcessor.from_pretrained(self.base_model, trust_remote_code=True)
        # Serving and scoring must see the same picture. ``rfs judge train`` and
        # ``judge score-choice`` cap the vision budget (``image_max_pixels``), so the server
        # applies the same cap; leaving it off would serve the Judge a sharper figure than it
        # was measured on. ``0`` keeps the processor default for callers who want that.
        if self.image_max_pixels:
            from rfs.judge_training.training import apply_vision_budget, TrainingUnavailable

            try:
                applied = apply_vision_budget(self.processor, self.image_max_pixels)
            except TrainingUnavailable as exc:
                raise SystemExit(f"--image-max-pixels cannot be applied: {exc}") from exc
            self.state["vision_budget"] = applied
        self.model = resolve_vision_auto_class(transformers).from_pretrained(self.base_model, **kwargs)
        if self.adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter)
        self.model.eval()
        self.state.update({"loaded": True, "device": str(getattr(self.model, "device", "")), "served_model_name": self.base_model})

    def _conversation(self, text: str, images: list[bytes]) -> list[dict[str, Any]]:
        from PIL import Image
        import io

        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for raw in images:
            content.append({"type": "image", "image": Image.open(io.BytesIO(raw)).convert("RGB")})
        return [{"role": "user", "content": content}]

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Answer one chat request and return the model's own text."""

        text, images = read_messages(payload)
        conversation = self._conversation(text, images)
        requested = payload.get("max_tokens") if payload.get("max_tokens") is not None else payload.get("max_new_tokens")
        cap = int(requested) if requested not in (None, "") else self.max_new_tokens
        with self.lock:
            inputs = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **chat_template_kwargs(self.thinking))
            inputs = {key: (value.to(self.model.device) if hasattr(value, "to") else value) for key, value in inputs.items()}
            prompt_tokens = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
            generated = self.model.generate(**inputs, max_new_tokens=cap, do_sample=False)
            answer_ids = generated[:, prompt_tokens:] if prompt_tokens else generated
            answer = self.processor.batch_decode(answer_ids, skip_special_tokens=True)[0]
        answer = str(answer).strip()
        completion_tokens = int(answer_ids.shape[1])
        return {
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "image_count": len(images),
            "finish_reason": classify_completion(answer, completion_tokens, cap),
        }


def completion_body(*, model: str, answer: str, prompt_tokens: int, completion_tokens: int, finish_reason: str = "stop") -> dict[str, Any]:
    """Wrap one answer in the response shape ``call_vlm_json`` reads back."""

    return {
        "id": f"chatcmpl-local-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": str(finish_reason or "stop")}],
        "usage": {"prompt_tokens": int(prompt_tokens), "completion_tokens": int(completion_tokens), "total_tokens": int(prompt_tokens) + int(completion_tokens)},
    }


def error_body(message: str) -> dict[str, Any]:
    return {"error": {"message": str(message), "type": "invalid_request_error"}}


class _Handler(BaseHTTPRequestHandler):
    server_version = "EvoFigJudge/1.0"
    judge: LocalJudge
    served_model_name: str

    def log_message(self, message: str, *args: Any) -> None:  # noqa: A003 - the base class names it
        sys.stderr.write(f"{self.address_string()} {message % args}\n")

    def _send(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - the base class names it
        if self.path.rstrip("/") not in ("/health", "/v1/health"):
            self._send(404, error_body(f"unknown path: {self.path}"))
            return
        self._send(200, {"status": "ok", "model": self.served_model_name, **self.judge.state})

    def do_POST(self) -> None:  # noqa: N802 - the base class names it
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, error_body(f"unknown path: {self.path}"))
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("the request body is not a JSON object")
            if payload.get("stream"):
                raise ValueError("this server answers whole completions only; send stream=false")
            result = self.judge.complete(payload)
        except Exception as exc:  # noqa: BLE001 - a bad request must be answered, not crash the server
            self._send(400, error_body(f"{type(exc).__name__}: {exc}"))
            return
        self._send(200, completion_body(model=self.served_model_name, answer=result["answer"], prompt_tokens=result["prompt_tokens"], completion_tokens=result["completion_tokens"], finish_reason=str(result.get("finish_reason") or "stop")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve one trained Judge checkpoint over the Judge JSON API contract.")
    parser.add_argument("--base-model", required=True, help="Base checkpoint directory or Hub id the adapter was trained on.")
    parser.add_argument("--adapter", default="", help="LoRA adapter directory written by `rfs judge train`; omit to serve the base model.")
    parser.add_argument("--served-model-name", default="", help="Name reported to callers; the base model by default.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="Decode budget. 1024 finishes an error_audit; 640 truncates it. Raise to 1536 for a retry of a truncated id.")
    parser.add_argument("--thinking", action="store_true", help="Render the assistant turn with the template's thinking block; off by default because the Judge is trained to answer without one.")
    parser.add_argument("--image-max-pixels", type=int, default=0, help="Cap the vision budget the way `judge train`/`judge score-choice` do; 0 keeps the processor default. Set it to the training value so serving matches scoring.")
    parser.add_argument("--no-4bit", action="store_false", dest="load_in_4bit", default=True, help="Load the base model without 4-bit quantisation.")
    return parser


def serve(*, base_model: str, adapter: str = "", served_model_name: str = "", host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, load_in_4bit: bool = True, thinking: bool = False, image_max_pixels: int = 0) -> None:
    """Load the checkpoint and answer requests until interrupted."""

    judge = LocalJudge(base_model=base_model, adapter=adapter, load_in_4bit=load_in_4bit, max_new_tokens=max_new_tokens, thinking=thinking, image_max_pixels=image_max_pixels)
    name = served_model_name or os.path.basename(str(base_model).rstrip("/")) or str(base_model)
    handler = type("_BoundHandler", (_Handler,), {"judge": judge, "served_model_name": name})
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    sys.stderr.write(f"serving {name} on http://{host}:{port}/v1  (health: /health)\n")
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adapter = str(args.adapter or "")
    if adapter and not (Path(adapter) / "adapter_config.json").is_file():
        sys.stderr.write(f"no LoRA adapter at {adapter}: expected adapter_config.json\n")
        return 2
    serve(
        base_model=str(args.base_model),
        adapter=adapter,
        served_model_name=str(args.served_model_name or ""),
        host=str(args.host),
        port=int(args.port),
        max_new_tokens=int(args.max_new_tokens),
        load_in_4bit=bool(args.load_in_4bit),
        thinking=bool(args.thinking),
        image_max_pixels=int(args.image_max_pixels),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
