"""
Unit tests for TranscriptReplay.

Verifies that canned turns are exposed unchanged for the scorer,
and that inbound frames match agent turns in the transcript.
"""

from __future__ import annotations

import pytest

from voicetest.models import Turn
from voicetest.providers.targets import (
    CallTarget,
    TranscriptReplay,
    _frame_to_text,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CANNED_TURNS = [
    Turn(speaker="agent", text="Hello, how can I help?", start_ms=0, end_ms=500),
    Turn(speaker="caller", text="I'd like a table for two", start_ms=600, end_ms=1200),
    Turn(speaker="agent", text="Certainly! What date?", start_ms=1300, end_ms=1800),
    Turn(speaker="caller", text="Tomorrow evening", start_ms=1900, end_ms=2400),
    Turn(speaker="agent", text="Booked. Anything else?", start_ms=2500, end_ms=3000),
    Turn(speaker="caller", text="No, thank you", start_ms=3100, end_ms=3500),
]


async def _collect_inbound(replay: TranscriptReplay) -> list[str]:
    stream = await replay.connect()
    texts: list[str] = []
    async for frame in stream.inbound_frames():
        text = _frame_to_text(frame)
        if text is not None:
            texts.append(text)
    await replay.disconnect()
    return texts


# ---------------------------------------------------------------------------
# turns property — scorer interface
# ---------------------------------------------------------------------------


def test_turns_returns_all_canned_turns():
    replay = TranscriptReplay(CANNED_TURNS)
    assert replay.turns == CANNED_TURNS


def test_turns_returns_copy():
    replay = TranscriptReplay(CANNED_TURNS)
    got = replay.turns
    got.append(Turn(speaker="caller", text="extra", start_ms=0, end_ms=100))
    assert len(replay.turns) == len(CANNED_TURNS)


def test_turns_preserves_order():
    replay = TranscriptReplay(CANNED_TURNS)
    for expected, actual in zip(CANNED_TURNS, replay.turns):
        assert actual.speaker == expected.speaker
        assert actual.text == expected.text
        assert actual.start_ms == expected.start_ms
        assert actual.end_ms == expected.end_ms


def test_turns_empty_transcript():
    replay = TranscriptReplay([])
    assert replay.turns == []


def test_turns_caller_only():
    turns = [Turn(speaker="caller", text="hello", start_ms=0, end_ms=300)]
    replay = TranscriptReplay(turns)
    assert replay.turns == turns


# ---------------------------------------------------------------------------
# connect / inbound frames — only agent turns emitted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inbound_frames_match_agent_turns():
    replay = TranscriptReplay(CANNED_TURNS)
    texts = await _collect_inbound(replay)
    expected = [t.text for t in CANNED_TURNS if t.speaker == "agent"]
    assert texts == expected


@pytest.mark.asyncio
async def test_inbound_frames_empty_when_no_agent_turns():
    turns = [Turn(speaker="caller", text="hello", start_ms=0, end_ms=300)]
    replay = TranscriptReplay(turns)
    texts = await _collect_inbound(replay)
    assert texts == []


@pytest.mark.asyncio
async def test_inbound_frames_empty_transcript():
    replay = TranscriptReplay([])
    texts = await _collect_inbound(replay)
    assert texts == []


@pytest.mark.asyncio
async def test_inbound_frames_agent_only():
    turns = [
        Turn(speaker="agent", text="Hi there", start_ms=0, end_ms=300),
        Turn(speaker="agent", text="How can I help?", start_ms=400, end_ms=800),
    ]
    replay = TranscriptReplay(turns)
    texts = await _collect_inbound(replay)
    assert texts == ["Hi there", "How can I help?"]


# ---------------------------------------------------------------------------
# CallTarget protocol
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_transcript_replay_satisfies_call_target_protocol():
    replay = TranscriptReplay(CANNED_TURNS)
    assert isinstance(replay, CallTarget)


@pytest.mark.asyncio
async def test_connect_returns_audio_stream():
    from voicetest.providers.targets import AudioStream

    replay = TranscriptReplay(CANNED_TURNS)
    stream = await replay.connect()
    assert isinstance(stream, AudioStream)
    # drain so the background task finishes cleanly
    async for _ in stream.inbound_frames():
        pass
    await replay.disconnect()


@pytest.mark.asyncio
async def test_disconnect_clears_stream():
    replay = TranscriptReplay(CANNED_TURNS)
    await replay.connect()
    await _collect_inbound(replay)
    assert replay._stream is None


# ---------------------------------------------------------------------------
# Scorer use-case: turns unchanged after connect/disconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_turns_unchanged_after_connect_disconnect():
    replay = TranscriptReplay(CANNED_TURNS)
    await _collect_inbound(replay)
    assert replay.turns == CANNED_TURNS
