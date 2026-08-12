"""Mock Ollama client tests (no GPU)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from xair.ai.ollama_client import OllamaClient


def test_chat_parses_response():
    client = OllamaClient(host="http://127.0.0.1:9", model="test")
    payload = json.dumps({
        "message": {"content": '{"action":"RESUME"}'},
    }).encode()

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return payload

    with patch("urllib.request.urlopen", return_value=FakeResp()):
        resp = client.chat([{"role": "user", "content": "hi"}], format_json=True)
    assert "RESUME" in resp.content
    assert resp.latency_ms >= 0


def test_health_false_on_error():
    client = OllamaClient(host="http://127.0.0.1:59999")
    assert client.health() is False
