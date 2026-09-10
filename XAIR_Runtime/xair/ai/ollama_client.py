"""Ollama HTTP client for structured AIS generation."""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class OllamaResponse:
    content: str
    model: str
    latency_ms: float
    raw: dict


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        num_ctx: int | None = None,
    ):
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
        # Keep timeouts short so a hung runner cannot stall the campaign for minutes.
        self.timeout_s = float(
            timeout_s if timeout_s is not None else os.environ.get("OLLAMA_TIMEOUT_S", "60")
        )
        self.max_retries = int(
            max_retries if max_retries is not None else os.environ.get("OLLAMA_MAX_RETRIES", "3")
        )
        # Default Ollama vision contexts of 32k balloon KV/compute graph and cause hangs
        # under long campaigns; 8k is ample for AIS JSON + one/two images.
        self.num_ctx = int(
            num_ctx if num_ctx is not None else os.environ.get("OLLAMA_NUM_CTX", "8192")
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        format_json: bool = True,
        model: str | None = None,
        images: list[str] | None = None,
    ) -> OllamaResponse:
        """Call POST /api/chat. images = list of base64-encoded PNG/JPG."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": 0.0,
            },
        }
        if format_json:
            payload["format"] = "json"
        if images and messages:
            # Ollama vision: images on last user message
            msgs = [dict(m) for m in messages]
            msgs[-1]["images"] = images
            payload["messages"] = msgs

        body = json.dumps(payload).encode()
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    f"{self.host}/api/chat",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = json.loads(resp.read().decode())
                latency_ms = (time.perf_counter() - t0) * 1000.0
                content = raw.get("message", {}).get("content", "")
                return OllamaResponse(
                    content=content,
                    model=payload["model"],
                    latency_ms=latency_ms,
                    raw=raw,
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
                last_err = e
                if attempt < self.max_retries:
                    # Runner EOF/500 often needs several seconds while Ollama respawns CUDA.
                    delay = 2.0 * (attempt + 1)
                    if isinstance(e, urllib.error.HTTPError) and e.code >= 500:
                        delay = 5.0 * (attempt + 1)
                    time.sleep(delay)
        raise RuntimeError(f"Ollama chat failed after {self.max_retries + 1} attempts: {last_err}")

    @staticmethod
    def encode_image(path: str, max_side: int | None = None) -> str:
        """
        Base64-encode an image for Ollama vision.

        Large VisA frames (~1300-1500px) make the CUDA runner abort with
        std::runtime_error during sampling; downscaling to ~1024px keeps
        grounding intact and avoids those hard crashes under LXC passthrough.
        """
        side = int(
            max_side
            if max_side is not None
            else os.environ.get("OLLAMA_IMAGE_MAX_SIDE", "1024")
        )
        try:
            from PIL import Image
        except ImportError:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("ascii")

        import io

        with Image.open(path) as im:
            rgb = im.convert("RGB")
            if max(rgb.size) > side > 0:
                rgb = rgb.copy()
                rgb.thumbnail((side, side))
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("ascii")

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def unload(self, model: str | None = None) -> None:
        """Drop a model from VRAM so the next load is not squeezed by a leftover resident."""
        name = model or self.model
        payload = json.dumps({"model": name, "keep_alive": 0}).encode()
        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
        except Exception:
            # Best-effort: campaign can continue even if unload races with a busy runner.
            pass
