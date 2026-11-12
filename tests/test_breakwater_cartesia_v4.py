"""Runs CartesiaTTS.synthesize against a local fake Cartesia HTTP server using the real SDK."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

PCM_CHUNKS = [b"\x01\x00\x02\x00\x03\x00\x04\x00", b"\x05\x00\x06\x00"]


class _Handler(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *args):  # silence
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        type(self).requests.append((self.path, body))
        if self.path.split("?")[0].rstrip("/").endswith("/tts/sse"):
            lines = []
            for chunk in PCM_CHUNKS:
                event = {
                    "type": "chunk",
                    "data": base64.b64encode(chunk).decode(),
                    "done": False,
                    "status_code": 206,
                    "step_time": 12.5,
                    "context_id": "ctx-1",
                }
                lines.append("data: " + json.dumps(event) + "\n\n")
            done = {
                "type": "done",
                "data": "",
                "done": True,
                "status_code": 200,
                "context_id": "ctx-1",
            }
            lines.append("data: " + json.dumps(done) + "\n\n")
            payload = "".join(lines).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            self.close_connection = True
        else:
            payload = json.dumps({"error": "not found"}).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


@pytest.fixture
def fake_cartesia(monkeypatch):
    _Handler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("CARTESIA_BASE_URL", base_url)

    import cartesia

    # Older SDKs have no CARTESIA_BASE_URL variable: default them to the server.
    for name in ("Cartesia", "AsyncCartesia"):
        cls = getattr(cartesia, name)
        original = cls.__init__
        if "base_url" in inspect.signature(original).parameters:

            def make(orig):
                def __init__(self, *args, **kwargs):
                    if kwargs.get("base_url") is None:
                        kwargs["base_url"] = base_url
                    orig(self, *args, **kwargs)

                return __init__

            monkeypatch.setattr(cls, "__init__", make(original))
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()


def test_cartesia_tts_synthesize_streams_pcm_frames(fake_cartesia):
    from voicetest.providers.tts import CartesiaTTS
    from voicetest.providers.targets import SAMPLE_RATE

    async def run():
        tts = CartesiaTTS(api_key="test-key")
        return [f async for f in tts.synthesize("hello there")]

    frames = asyncio.run(run())

    assert any(path.split("?")[0].endswith("/tts/sse") for path, _ in _Handler.requests)
    assert [f.samples for f in frames] == PCM_CHUNKS
    assert all(f.sample_rate == SAMPLE_RATE for f in frames)
