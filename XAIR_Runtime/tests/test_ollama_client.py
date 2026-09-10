"""Mock Ollama client tests (no GPU)."""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import patch

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


def test_unload_is_best_effort():
    client = OllamaClient(host="http://127.0.0.1:9", model="test")

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return b"{}"

    with patch("urllib.request.urlopen", return_value=FakeResp()) as mocked:
        client.unload("qwen2.5vl:3b")
    assert mocked.called


def test_encode_image_downscales(tmp_path):
    from PIL import Image

    img = tmp_path / "big.jpg"
    Image.new("RGB", (1600, 1200), color=(10, 20, 30)).save(img, quality=95)
    b64 = OllamaClient.encode_image(str(img), max_side=1024)
    raw = base64.b64decode(b64)
    out = Image.open(io.BytesIO(raw))
    assert max(out.size) <= 1024
    assert out.size[0] == 1024
