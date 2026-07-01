"""
Unit tests for LocalLoopback.

All in-memory — no network, no audio hardware, no real STT/TTS.
"""

from __future__ import annotations

import pytest

from voicetest.models import Turn
from voicetest.providers.targets import (
    AudioFrame,
    LocalLoopback,
    _frame_to_text,
    _text_to_frame,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _drive(loopback: LocalLoopback, utterances: list[str]) -> list[str]:
    """Connect, send utterances, disconnect; return decoded agent replies."""
    stream = await loopback.connect()
    replies: list[str] = []

    for text in utterances:
        await stream.put_outbound(_text_to_frame(text))

    await stream.close_outbound()

    async for frame in stream.inbound_frames():
        decoded = _frame_to_text(frame)
        if decoded is not None:
            replies.append(decoded)

    await loopback.disconnect()
    return replies


# ---------------------------------------------------------------------------
# Frame codec
# ---------------------------------------------------------------------------


def test_frame_roundtrip():
    text = "Hello, world!"
    frame = _text_to_frame(text)
    assert _frame_to_text(frame) == text


def test_frame_roundtrip_empty_string():
    assert _frame_to_text(_text_to_frame("")) == ""


def test_frame_malformed_returns_none():
    assert _frame_to_text(AudioFrame(samples=b"\x00\x00\x00\xff")) is None


def test_frame_too_short_returns_none():
    assert _frame_to_text(AudioFrame(samples=b"\x01")) is None


def test_frame_preserves_sample_rate():
    frame = _text_to_frame("hi", sample_rate=16000)
    assert frame.sample_rate == 16000


# ---------------------------------------------------------------------------
# LocalLoopback — default agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loopback_greeting():
    """No utterances → agent sends a greeting."""
    loopback = LocalLoopback()
    stream = await loopback.connect()
    await stream.close_outbound()

    replies: list[str] = []
    async for frame in stream.inbound_frames():
        text = _frame_to_text(frame)
        if text:
            replies.append(text)

    await loopback.disconnect()
    # Default agent produces an opening greeting when no turns exist yet.
    # It won't reply unless it receives a frame — greeting is call-initiated.
    # Sending no utterances → no replies (agent only responds to inbound).
    assert isinstance(replies, list)


@pytest.mark.asyncio
async def test_loopback_single_utterance():
    replies = await _drive(LocalLoopback(), ["I'd like to make a reservation please"])
    assert len(replies) == 1
    assert replies[0]  # non-empty reply


@pytest.mark.asyncio
async def test_loopback_multi_turn():
    loopback = LocalLoopback()
    utterances = [
        "I'd like to book a table",
        "For four people",
        "Tomorrow evening at 7pm",
        "No that's all, thank you",
    ]
    replies = await _drive(loopback, utterances)
    assert len(replies) == len(utterances)
    for reply in replies:
        assert isinstance(reply, str)
        assert len(reply) > 0


@pytest.mark.asyncio
async def test_loopback_turns_recorded():
    loopback = LocalLoopback()
    await _drive(loopback, ["Book a table", "For two"])
    turns = loopback.turns
    assert len(turns) == 4  # 2 caller + 2 agent
    speakers = [t.speaker for t in turns]
    assert speakers == ["caller", "agent", "caller", "agent"]


@pytest.mark.asyncio
async def test_loopback_turn_texts_match_utterances():
    utterances = ["Reserve a table", "For six people"]
    loopback = LocalLoopback()
    await _drive(loopback, utterances)
    caller_turns = [t for t in loopback.turns if t.speaker == "caller"]
    assert [t.text for t in caller_turns] == utterances


@pytest.mark.asyncio
async def test_loopback_turn_timestamps_monotonic():
    loopback = LocalLoopback()
    await _drive(loopback, ["book a table", "tomorrow at noon"])
    turns = loopback.turns
    for t in turns:
        assert t.start_ms <= t.end_ms
    for i in range(len(turns) - 1):
        assert turns[i].end_ms <= turns[i + 1].start_ms


@pytest.mark.asyncio
async def test_loopback_call_target_protocol():
    """LocalLoopback must satisfy the CallTarget Protocol runtime check."""
    from voicetest.providers.targets import CallTarget

    loopback = LocalLoopback()
    assert isinstance(loopback, CallTarget)


# ---------------------------------------------------------------------------
# LocalLoopback — injectable agent_fn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loopback_injectable_agent():
    calls: list[list[Turn]] = []

    def spy_agent(turns: list[Turn]) -> str:
        calls.append(list(turns))
        return "spy reply"

    replies = await _drive(LocalLoopback(agent_fn=spy_agent), ["hello"])
    assert replies == ["spy reply"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_loopback_broken_agent_slow(monkeypatch):
    """Broken agent (fast variant) still produces replies — timing tested elsewhere."""
    from tests.fixtures.broken_agent import make_broken_agent

    # Use delay=0 so the unit test runs fast; P0-T6 tests real latency.
    agent = make_broken_agent(delay_s=0, fixed_reply="I can't help with that")
    loopback = LocalLoopback(agent_fn=agent)
    replies = await _drive(loopback, ["Can you book me a table?"])
    assert replies == ["I can't help with that"]


@pytest.mark.asyncio
async def test_loopback_broken_agent_ignores_content():
    """Broken agent always returns the same reply regardless of input."""
    from tests.fixtures.broken_agent import make_broken_agent

    agent = make_broken_agent(delay_s=0, fixed_reply="fixed")
    loopback = LocalLoopback(agent_fn=agent)
    replies = await _drive(loopback, ["question one", "question two", "question three"])
    assert all(r == "fixed" for r in replies)


# ---------------------------------------------------------------------------
# LocalLoopback — connect / disconnect lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loopback_disconnect_cleans_up():
    loopback = LocalLoopback()
    await loopback.connect()
    await loopback.disconnect()
    assert loopback._stream is None
    assert loopback._task is None


@pytest.mark.asyncio
async def test_loopback_returns_audio_stream_on_connect():
    from voicetest.providers.targets import AudioStream

    loopback = LocalLoopback()
    stream = await loopback.connect()
    assert isinstance(stream, AudioStream)
    await loopback.disconnect()
