"""
ws_server.py — Twilio Media Streams WebSocket bridge.

This is the ONLY place phone audio codecs live (ADR: transcode at the transport
edge only). Twilio Media Streams sends/receives **8 kHz μ-law (G.711)** as
base64 inside JSON frames; the rest of the engine speaks PCM linear16 @ 8 kHz
(``AudioFrame``). We transcode μ-law↔PCM16 here and nowhere else.

Rendezvous model
----------------
``TwilioOutbound.connect()`` builds an :class:`AudioStream`, registers it under a
``stream_id`` in the module-level :data:`registry`, then dials Twilio with TwiML
pointing at ``{PUBLIC_WS_URL}/twilio-stream/{stream_id}``. When Twilio answers and
connects back to that WebSocket, :func:`twilio_stream` looks the stream up by id
and bridges audio. The WS route and the target run in the **same process**, so the
registry is a plain in-memory dict.

Frame protocol (Twilio → us):  {"event": "start"|"media"|"stop"|"mark", ...}
Frame protocol (us → Twilio):  {"event": "media", "streamSid": ..., "media": {"payload": b64}}
"""

from __future__ import annotations

import asyncio
import audioop  # deprecated in 3.13; use audioop-lts there. μ-law↔PCM16 transcode.
import base64

from fastapi import WebSocket, WebSocketDisconnect

from voicetest.log import vlog
from voicetest.providers.targets import SAMPLE_RATE, AudioFrame, AudioStream

# ---------------------------------------------------------------------------
# Stream registry — the rendezvous between TwilioOutbound.connect() and the WS
# ---------------------------------------------------------------------------


class StreamRegistry:
    """Maps a stream_id → the AudioStream that TwilioOutbound is waiting to bridge."""

    def __init__(self) -> None:
        self._streams: dict[str, AudioStream] = {}
        # Set when the WS for a given id actually connects (so connect() can
        # distinguish "answered" from "no-answer / never connected").
        self._connected: dict[str, asyncio.Event] = {}

    def register(self, stream_id: str, stream: AudioStream) -> asyncio.Event:
        self._streams[stream_id] = stream
        ev = asyncio.Event()
        self._connected[stream_id] = ev
        vlog("ws", f"registered stream_id={stream_id} (now waiting for Twilio to connect)")
        return ev

    def get(self, stream_id: str) -> AudioStream | None:
        return self._streams.get(stream_id)

    def mark_connected(self, stream_id: str) -> None:
        ev = self._connected.get(stream_id)
        if ev is not None:
            ev.set()

    def unregister(self, stream_id: str) -> None:
        existed = self._streams.pop(stream_id, None) is not None
        self._connected.pop(stream_id, None)
        vlog("ws", f"unregistered stream_id={stream_id} (existed={existed})")


# One shared registry for the process.
registry = StreamRegistry()


# ---------------------------------------------------------------------------
# Codec: μ-law (Twilio) ↔ PCM16 (engine)
# ---------------------------------------------------------------------------


def ulaw_to_pcm16(ulaw_bytes: bytes) -> bytes:
    """Twilio μ-law 8 kHz → PCM signed 16-bit (what Deepgram/the engine expect)."""
    return audioop.ulaw2lin(ulaw_bytes, 2)  # width=2 bytes → 16-bit


def pcm16_to_ulaw(pcm_bytes: bytes) -> bytes:
    """Engine PCM16 → Twilio μ-law 8 kHz."""
    return audioop.lin2ulaw(pcm_bytes, 2)


# ---------------------------------------------------------------------------
# The WebSocket endpoint Twilio connects back to
# ---------------------------------------------------------------------------


async def twilio_stream(ws: WebSocket, stream_id: str) -> None:
    """
    Bridge a Twilio Media Streams WebSocket to the engine's AudioStream.

    Mounted by the app as ``@app.websocket("/twilio-stream/{stream_id}")``.
    Runs two pumps concurrently until Twilio sends ``stop`` or the socket drops.
    """
    await ws.accept()
    vlog("ws", f">>> WebSocket OPENED for stream_id={stream_id} (Twilio connected back!)")
    stream = registry.get(stream_id)
    if stream is None:
        # This is a common failure mode: Twilio connected but the stream_id is
        # unknown/expired (e.g. the run already gave up, or a stale ngrok URL is
        # pointing at a server that was restarted). The call will drop here.
        vlog("ws", f"!!! NO stream registered for {stream_id} — closing WS 1008. "
                   "Twilio reached us but the call leg was already gone.")
        await ws.close(code=1008)  # policy violation: unknown/expired stream id
        return

    registry.mark_connected(stream_id)
    twilio_stream_sid: str | None = None
    media_in = 0
    media_out = 0

    async def pump_in() -> None:
        """Twilio media frames → engine inbound (the agent's voice, to transcribe)."""
        nonlocal twilio_stream_sid, media_in
        try:
            while True:
                msg = await ws.receive_json()
                event = msg.get("event")
                if event == "start":
                    twilio_stream_sid = msg.get("start", {}).get("streamSid") or msg.get(
                        "streamSid"
                    )
                    vlog("ws", f"event=start streamSid={twilio_stream_sid}")
                elif event == "media":
                    media_in += 1
                    if media_in == 1 or media_in % 100 == 0:
                        vlog("ws", f"inbound media frame #{media_in} (Twilio→engine)")
                    payload = msg["media"]["payload"]
                    pcm = ulaw_to_pcm16(base64.b64decode(payload))
                    await stream.put_inbound(AudioFrame(samples=pcm, sample_rate=SAMPLE_RATE))
                elif event == "mark":
                    vlog("ws", f"event=mark {msg.get('mark', {})}")
                elif event == "stop":
                    vlog("ws", f"event=stop — Twilio closed the stream (total inbound={media_in})")
                    break
        except WebSocketDisconnect:
            vlog("ws", f"pump_in: WebSocketDisconnect (inbound frames received={media_in})")
        except Exception as exc:  # noqa: BLE001
            vlog("ws", f"pump_in ERROR: {type(exc).__name__}: {exc}")
        finally:
            await stream.close_inbound()

    async def pump_out() -> None:
        """Engine outbound (caller's synthesized voice) → Twilio media frames."""
        nonlocal media_out
        try:
            async for frame in stream.outbound_frames():
                media_out += 1
                if media_out == 1 or media_out % 100 == 0:
                    vlog("ws", f"outbound media frame #{media_out} (engine→Twilio)")
                payload = base64.b64encode(pcm16_to_ulaw(frame.samples)).decode("ascii")
                out = {"event": "media", "media": {"payload": payload}}
                if twilio_stream_sid:
                    out["streamSid"] = twilio_stream_sid
                else:
                    # Twilio silently drops media frames sent before the `start`
                    # event arrives (no streamSid yet) — worth flagging.
                    vlog("ws", "WARNING: sending outbound media before streamSid known "
                               "(Twilio may drop it)")
                try:
                    await ws.send_json(out)
                except (WebSocketDisconnect, RuntimeError) as exc:
                    vlog("ws", f"pump_out: send failed ({type(exc).__name__}) — stopping")
                    return
            vlog("ws", f"pump_out: outbound stream ended (total outbound={media_out})")
        except Exception as exc:  # noqa: BLE001
            vlog("ws", f"pump_out ERROR: {type(exc).__name__}: {exc}")

    try:
        await asyncio.gather(pump_in(), pump_out())
    finally:
        vlog("ws", f"<<< bridge closed stream_id={stream_id} "
                   f"(inbound={media_in}, outbound={media_out})")
        registry.unregister(stream_id)
