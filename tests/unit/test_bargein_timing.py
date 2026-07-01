"""
P1-T3 — BargeInInjector timing and event tests.

Uses LocalLoopback so no real audio hardware or network is needed.
Tests verify:
- An `interruption` event is logged on the configured agent turn
- The event carries the correct agent_turn and caller_says fields
- The injection fires only once even across multiple turns
- The barge-in utterance appears in outbound traffic (injected as a frame)
- Unit-level burst counting is correct
"""

from __future__ import annotations

import asyncio

import pytest

from voicetest.engine.processors.bargein import BargeInInjector
from voicetest.models import Event
from voicetest.providers.targets import AudioFrame, AudioStream, _text_to_frame
from voicetest.providers.tts import StubTTS
from voicetest.scenarios.models import BargeInConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_injector(
    *,
    on_agent_turn: int = 2,
    trigger_after_ms: int = 0,
    caller_says: str = "Wait, hold on.",
    enabled: bool = True,
    clock_val: int = 0,
) -> tuple[BargeInInjector, AudioStream, list[Event]]:
    stream = AudioStream()
    events: list[Event] = []
    tick = [clock_val]
    config = BargeInConfig(
        enabled=enabled,
        on_agent_turn=on_agent_turn,
        trigger_after_agent_speaks_ms=trigger_after_ms,
        caller_says=caller_says,
    )
    injector = BargeInInjector.from_config(config, stream, StubTTS(), events, lambda: tick[0])
    return injector, stream, events


def _fake_frame() -> AudioFrame:
    return _text_to_frame("agent speaking")


# ---------------------------------------------------------------------------
# Burst counting unit tests (synchronous)
# ---------------------------------------------------------------------------


def test_first_frame_increments_turn_count() -> None:
    injector, _, _ = _make_injector(enabled=True)
    injector._track_burst(0)
    assert injector._agent_turn_count == 1


def test_consecutive_frames_same_burst() -> None:
    injector, _, _ = _make_injector(enabled=True)
    injector._track_burst(0)
    injector._last_frame_at_ms = 0
    injector._track_burst(50)   # < _INTER_TURN_GAP_MS → same burst
    assert injector._agent_turn_count == 1


def test_gap_creates_new_burst() -> None:
    from voicetest.engine.processors.bargein import _INTER_TURN_GAP_MS
    injector, _, _ = _make_injector(on_agent_turn=99, enabled=True)  # no trigger
    injector._track_burst(0)
    injector._last_frame_at_ms = 0
    injector._in_burst = False  # simulate end-of-burst
    injector._track_burst(_INTER_TURN_GAP_MS + 1)
    assert injector._agent_turn_count == 2


def test_trigger_not_scheduled_on_wrong_turn() -> None:
    injector, _, _ = _make_injector(on_agent_turn=3, enabled=True)
    injector._track_burst(0)
    assert injector._trigger_task is None


@pytest.mark.asyncio
async def test_trigger_scheduled_on_correct_turn() -> None:
    injector, _, _ = _make_injector(on_agent_turn=1, trigger_after_ms=50, enabled=True)
    await injector.put_inbound(_fake_frame())  # triggers scheduling via put_inbound (async)
    assert injector._trigger_task is not None
    injector._trigger_task.cancel()  # clean up
    try:
        await injector._trigger_task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Async: interruption event logged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interruption_event_logged_on_correct_turn() -> None:
    injector, stream, events = _make_injector(
        on_agent_turn=1, trigger_after_ms=0, caller_says="Hold on!"
    )
    # Simulate one inbound frame (agent turn 1)
    await injector.put_inbound(_fake_frame())
    await injector.wait_for_trigger()

    kinds = [e.kind for e in events]
    assert "interruption" in kinds


@pytest.mark.asyncio
async def test_interruption_event_has_correct_agent_turn() -> None:
    injector, stream, events = _make_injector(
        on_agent_turn=1, trigger_after_ms=0, caller_says="Excuse me."
    )
    await injector.put_inbound(_fake_frame())
    await injector.wait_for_trigger()

    ev = next(e for e in events if e.kind == "interruption")
    assert ev.data["agent_turn"] == 1


@pytest.mark.asyncio
async def test_interruption_event_has_correct_caller_says() -> None:
    injector, stream, events = _make_injector(
        on_agent_turn=1, trigger_after_ms=0, caller_says="Actually wait."
    )
    await injector.put_inbound(_fake_frame())
    await injector.wait_for_trigger()

    ev = next(e for e in events if e.kind == "interruption")
    assert ev.data["caller_says"] == "Actually wait."


@pytest.mark.asyncio
async def test_interruption_event_agent_was_speaking_true() -> None:
    injector, stream, events = _make_injector(
        on_agent_turn=1, trigger_after_ms=0, caller_says="Wait."
    )
    await injector.put_inbound(_fake_frame())
    await injector.wait_for_trigger()

    ev = next(e for e in events if e.kind == "interruption")
    assert ev.data["agent_was_speaking"] is True


@pytest.mark.asyncio
async def test_interruption_fires_only_once() -> None:
    """Even if many frames arrive on the trigger turn, event fires once."""
    injector, stream, events = _make_injector(
        on_agent_turn=1, trigger_after_ms=0, caller_says="One sec."
    )
    for _ in range(5):
        await injector.put_inbound(_fake_frame())
    await injector.wait_for_trigger()

    interruptions = [e for e in events if e.kind == "interruption"]
    assert len(interruptions) == 1


@pytest.mark.asyncio
async def test_interruption_only_on_configured_turn() -> None:
    """Turn 1 passes without trigger; turn 2 fires."""
    injector, stream, events = _make_injector(
        on_agent_turn=2, trigger_after_ms=0, caller_says="Wait."
    )
    # Turn 1: mark end-of-burst manually to simulate silence gap
    await injector.put_inbound(_fake_frame())
    injector._in_burst = False

    assert not any(e.kind == "interruption" for e in events)

    # Turn 2: new burst
    await injector.put_inbound(_fake_frame())
    await injector.wait_for_trigger()
    assert any(e.kind == "interruption" for e in events)


@pytest.mark.asyncio
async def test_barge_in_utterance_emitted_outbound() -> None:
    """The injected caller_says text appears as an outbound frame."""
    injector, stream, events = _make_injector(
        on_agent_turn=1, trigger_after_ms=0, caller_says="Hold on."
    )
    await injector.put_inbound(_fake_frame())
    await injector.wait_for_trigger()

    # StubTTS produces a _text_to_frame frame; drain the outbound queue
    frames = []
    while not stream._outbound.empty():
        frames.append(stream._outbound.get_nowait())
    assert len(frames) >= 1


@pytest.mark.asyncio
async def test_inbound_frame_forwarded_to_stream() -> None:
    """put_inbound must always forward to the underlying stream."""
    injector, stream, events = _make_injector(
        on_agent_turn=2, trigger_after_ms=0, enabled=True
    )
    frame = _fake_frame()
    await injector.put_inbound(frame)

    received = stream._inbound.get_nowait()
    assert received.samples == frame.samples


# ---------------------------------------------------------------------------
# Pipeline-level integration via run_call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_barge_in_logs_interruption_event() -> None:
    """run_call with barge_in.enabled produces an interruption event."""
    from voicetest.engine.run_call import run_call
    from voicetest.providers.targets import LocalLoopback
    from voicetest.scenarios import BUILTIN_DIR, load_scenario
    from voicetest.scenarios.models import BargeInConfig, Limits

    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-double-books.yaml")
    assert scenario.barge_in.enabled is True

    short = scenario.model_copy(
        update={
            "limits": Limits(max_turns=3, max_duration_s=30),
            "barge_in": BargeInConfig(
                enabled=True,
                on_agent_turn=1,
                trigger_after_agent_speaks_ms=0,
                caller_says="Wait — hold on.",
            ),
        }
    )
    result = await run_call(short, LocalLoopback())
    event_kinds = {e.kind for e in result.events}
    assert "interruption" in event_kinds
