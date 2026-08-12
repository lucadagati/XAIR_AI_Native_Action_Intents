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
        timeout_s: float = 120.0,
        max_retries: int = 2,
    ):
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.timeout_s = timeout_s
        self.max_retries = max_retries

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
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Ollama chat failed after {self.max_retries + 1} attempts: {last_err}")

    @staticmethod
    def encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False
