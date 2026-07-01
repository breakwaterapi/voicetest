"""CallTarget interface and in-memory implementations: LocalLoopback and TranscriptReplay.

The real telephony target (TwilioOutbound) lives in ``providers/twilio_target.py``
so the optional Twilio SDK is only imported when a real call is placed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from voicetest.models import Turn

# ---------------------------------------------------------------------------
# Audio primitives
# ---------------------------------------------------------------------------

SAMPLE_RATE = 8000  # 8 kHz — G.711 phone channel rate (see tech-stack.md)


@dataclass
class AudioFrame:
    """A chunk of raw PCM audio. 16-bit signed, mono, at SAMPLE_RATE."""

    samples: bytes  # raw PCM bytes
    sample_rate: int = SAMPLE_RATE


# ---------------------------------------------------------------------------
# AudioStream — bidirectional audio channel between engine and target
# ---------------------------------------------------------------------------


class AudioStream:
    """
    Bidirectional in-memory audio pipe.

    The engine writes outbound (caller) frames via put_outbound().
    The engine reads inbound (agent) frames via inbound_frames().
    LocalLoopback / TwilioOutbound put frames on the inbound side.
    """

    def __init__(self) -> None:
        self._inbound: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._outbound: asyncio.Queue[AudioFrame | None] = asyncio.Queue()

    # engine → target
    async def put_outbound(self, frame: AudioFrame) -> None:
        await self._outbound.put(frame)

    async def close_outbound(self) -> None:
        await self._outbound.put(None)

    # target → engine
    async def put_inbound(self, frame: AudioFrame) -> None:
        await self._inbound.put(frame)

    async def close_inbound(self) -> None:
        await self._inbound.put(None)

    async def inbound_frames(self) -> AsyncIterator[AudioFrame]:
        while True:
            frame = await self._inbound.get()
            if frame is None:
                return
            yield frame

    async def outbound_frames(self) -> AsyncIterator[AudioFrame]:
        while True:
            frame = await self._outbound.get()
            if frame is None:
                return
            yield frame


# ---------------------------------------------------------------------------
# CallTarget protocol
# ---------------------------------------------------------------------------

AgentFn = Callable[[list[Turn]], str]


@runtime_checkable
class CallTarget(Protocol):
    """
    The thing the synthetic caller engine talks to.
    connect() returns a bidirectional AudioStream; the engine drives it from there.
    """

    async def connect(self) -> AudioStream: ...

    async def disconnect(self) -> None: ...


# ---------------------------------------------------------------------------
# Stub STT / TTS used by LocalLoopback (no network, no real audio)
# ---------------------------------------------------------------------------


def _text_to_frame(text: str, sample_rate: int = SAMPLE_RATE) -> AudioFrame:
    """
    Encode text as a length-prefixed UTF-8 payload inside an AudioFrame.
    The loopback "STT" reverses this to recover the transcript without any
    real speech synthesis or recognition.

    Payload is padded to an even byte count so NoiseMixer's 16-bit PCM
    struct.unpack never fails on odd-length text.
    """
    payload = text.encode()
    length = len(payload).to_bytes(4, "big")
    raw = length + payload
    if len(raw) % 2:
        raw += b"\x00"  # pad to even — _frame_to_text uses the length prefix, not total size
    return AudioFrame(samples=raw, sample_rate=sample_rate)


def _frame_to_text(frame: AudioFrame) -> str | None:
    """Decode a frame produced by _text_to_frame; returns None on malformed input."""
    data = frame.samples
    if len(data) < 4:
        return None
    length = int.from_bytes(data[:4], "big")
    if len(data) < 4 + length:
        return None
    return data[4 : 4 + length].decode()


# ---------------------------------------------------------------------------
# Default cooperative booking agent
# ---------------------------------------------------------------------------


def _default_agent_fn(turns: list[Turn]) -> str:
    """A minimal cooperative restaurant-booking agent used as the default stub."""
    if not turns:
        return "Hello, thank you for calling. How can I help you today?"

    last_caller = next(
        (t.text.lower() for t in reversed(turns) if t.speaker == "caller"), ""
    )

    if any(w in last_caller for w in ("book", "reserv", "table")):
        return "Of course! For how many people, and what date and time were you thinking?"
    if any(w in last_caller for w in ("two", "2", "four", "4", "six", "6")):
        return "Great. And what date and time would you like?"
    if any(w in last_caller for w in ("tomorrow", "friday", "saturday", "sunday", "monday")):
        return "Perfect. And what time works best for you?"
    if any(c.isdigit() for c in last_caller) and any(
        w in last_caller for w in ("pm", "am", "o'clock", ":")
    ):
        return "Wonderful, I have that booked for you. Is there anything else I can help you with?"
    if any(w in last_caller for w in ("no", "that's all", "thank", "bye", "goodbye")):
        return "Great, thank you for calling. Goodbye!"
    return "I'm sorry, could you repeat that please?"


# ---------------------------------------------------------------------------
# LocalLoopback
# ---------------------------------------------------------------------------


class LocalLoopback:
    """
    In-process call target. Drives a pluggable agent_fn so tests can inject
    a broken/slow agent without any code changes to the engine.

    Audio is faked: outbound frames are decoded back to text (stub STT),
    agent_fn is called, and the reply is encoded back to frames (stub TTS).
    No real audio hardware, codec, or network is involved.
    """

    def __init__(self, agent_fn: AgentFn | None = None) -> None:
        self._agent_fn: AgentFn = agent_fn or _default_agent_fn
        self._stream: AudioStream | None = None
        self._task: asyncio.Task[None] | None = None
        self._turns: list[Turn] = []
        self._clock_ms: int = 0

    async def connect(self) -> AudioStream:
        self._stream = AudioStream()
        self._task = asyncio.create_task(self._run())
        return self._stream

    async def disconnect(self) -> None:
        if self._stream:
            # Signal _run() to finish iterating outbound frames, then wait for it.
            await self._stream.close_outbound()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.CancelledError:
                pass
        self._stream = None
        self._task = None

    async def _run(self) -> None:
        """Read outbound (caller) frames, decode, call agent_fn, emit inbound frames."""
        assert self._stream is not None
        buffer: list[str] = []

        async for frame in self._stream.outbound_frames():
            text = _frame_to_text(frame)
            if text is None:
                continue

            # Record the caller turn
            start_ms = self._clock_ms
            self._clock_ms += len(text) * 10  # rough duration heuristic
            self._turns.append(
                Turn(speaker="caller", text=text, start_ms=start_ms, end_ms=self._clock_ms)
            )
            buffer.append(text)

            # Generate agent reply
            reply = self._agent_fn(list(self._turns))

            reply_start = self._clock_ms
            self._clock_ms += len(reply) * 10
            self._turns.append(
                Turn(
                    speaker="agent",
                    text=reply,
                    start_ms=reply_start,
                    end_ms=self._clock_ms,
                )
            )

            reply_frame = _text_to_frame(reply)
            await self._stream.put_inbound(reply_frame)

        await self._stream.close_inbound()

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)


# ---------------------------------------------------------------------------
# TranscriptReplay
# ---------------------------------------------------------------------------


@dataclass
class TranscriptReplay:
    """
    Replays a canned list of Turns as inbound audio frames.
    Produces no live audio — the scorer uses it to evaluate pre-recorded
    conversations without running the engine at all.
    """

    _turns: list[Turn] = field(default_factory=list)
    _stream: AudioStream | None = field(default=None, repr=False)

    def __init__(self, turns: list[Turn]) -> None:
        self._turns = list(turns)
        self._stream = None

    async def connect(self) -> AudioStream:
        self._stream = AudioStream()
        asyncio.create_task(self._feed())
        return self._stream

    async def disconnect(self) -> None:
        self._stream = None

    async def _feed(self) -> None:
        """Emit agent turns as inbound frames; caller turns are already in the transcript."""
        assert self._stream is not None
        for turn in self._turns:
            if turn.speaker == "agent":
                frame = _text_to_frame(turn.text)
                await self._stream.put_inbound(frame)
        await self._stream.close_inbound()

    @property
    def turns(self) -> list[Turn]:
        """Expose the canned turns unchanged for the scorer."""
        return list(self._turns)
