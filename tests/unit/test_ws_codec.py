"""
Unit tests for the Twilio Media Streams codec bridge (μ-law 8 kHz ↔ PCM16).
No network — pure transcode + framing round-trips.
"""

from __future__ import annotations

import base64
import struct

from voicetest.engine.ws_server import (
    StreamRegistry,
    pcm16_to_ulaw,
    ulaw_to_pcm16,
)


def test_ulaw_pcm_roundtrip_preserves_length():
    # 160 samples = 20 ms @ 8 kHz, Twilio's typical frame size.
    pcm = struct.pack("<160h", *([1000] * 160))
    ulaw = pcm16_to_ulaw(pcm)
    assert len(ulaw) == 160  # μ-law is 1 byte/sample
    back = ulaw_to_pcm16(ulaw)
    assert len(back) == len(pcm)  # PCM16 is 2 bytes/sample


def test_ulaw_pcm_roundtrip_approximates_signal():
    # μ-law is lossy but should stay close for mid-range amplitudes.
    pcm = struct.pack("<4h", 0, 4000, -4000, 8000)
    back = struct.unpack("<4h", ulaw_to_pcm16(pcm16_to_ulaw(pcm)))
    originals = (0, 4000, -4000, 8000)
    for o, b in zip(originals, back):
        assert abs(o - b) < 600  # within μ-law quantization tolerance


def test_base64_payload_decodes_to_pcm():
    # Simulate a Twilio media payload (base64 μ-law) → engine PCM.
    ulaw = pcm16_to_ulaw(struct.pack("<80h", *([2000] * 80)))
    payload = base64.b64encode(ulaw).decode("ascii")
    pcm = ulaw_to_pcm16(base64.b64decode(payload))
    assert len(pcm) == 80 * 2


# ---------------------------------------------------------------------------
# StreamRegistry rendezvous
# ---------------------------------------------------------------------------


def test_registry_register_get_unregister():
    from voicetest.providers.targets import AudioStream

    reg = StreamRegistry()
    stream = AudioStream()
    ev = reg.register("abc", stream)
    assert reg.get("abc") is stream
    assert not ev.is_set()

    reg.mark_connected("abc")
    assert ev.is_set()

    reg.unregister("abc")
    assert reg.get("abc") is None


def test_registry_get_unknown_returns_none():
    assert StreamRegistry().get("nope") is None
