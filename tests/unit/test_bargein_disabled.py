"""
P1-T3 — BargeInInjector disabled-passthrough tests.

When BargeInConfig.enabled is False:
- No interruption events are logged
- Inbound frames are forwarded bitwise-identical
- No outbound injection frames appear
- Pipeline runs to completion without error
"""

from __future__ import annotations

import pytest

from voicetest.engine.processors.bargein import BargeInInjector
from voicetest.models import Event
from voicetest.providers.targets import AudioFrame, AudioStream, _text_to_frame
from voicetest.providers.tts import StubTTS
from voicetest.scenarios.models import BargeInConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _disabled_injector() -> tuple[BargeInInjector, AudioStream, list[Event]]:
    stream = AudioStream()
    events: list[Event] = []
    config = BargeInConfig(enabled=False, on_agent_turn=1, trigger_after_agent_speaks_ms=0)
    injector = BargeInInjector.from_config(config, stream, StubTTS(), events, lambda: 0)
    return injector, stream, events


def _fake_frame() -> AudioFrame:
    return _text_to_frame("agent reply here")


# ---------------------------------------------------------------------------
# No events when disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_interruption_event_when_disabled() -> None:
    injector, _, events = _disabled_injector()
    await injector.put_inbound(_fake_frame())
    await injector.wait_for_trigger()
    assert not any(e.kind == "interruption" for e in events)


@pytest.mark.asyncio
async def test_no_events_at_all_when_disabled() -> None:
    injector, _, events = _disabled_injector()
    for _ in range(5):
        await injector.put_inbound(_fake_frame())
    await injector.wait_for_trigger()
    assert events == []


@pytest.mark.asyncio
async def test_trigger_task_never_created_when_disabled() -> None:
    injector, _, _ = _disabled_injector()
    await injector.put_inbound(_fake_frame())
    assert injector._trigger_task is None


# ---------------------------------------------------------------------------
# Passthrough: inbound frames forwarded unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_forwards_inbound_frame() -> None:
    injector, stream, _ = _disabled_injector()
    frame = _fake_frame()
    await injector.put_inbound(frame)
    received = stream._inbound.get_nowait()
    assert received.samples == frame.samples


@pytest.mark.asyncio
async def test_disabled_forwards_multiple_frames_in_order() -> None:
    injector, stream, _ = _disabled_injector()
    texts = ["first", "second", "third"]
    frames = [_text_to_frame(t) for t in texts]
    for f in frames:
        await injector.put_inbound(f)
    for expected in frames:
        received = stream._inbound.get_nowait()
        assert received.samples == expected.samples


# ---------------------------------------------------------------------------
# No outbound injection when disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_no_outbound_injection() -> None:
    injector, stream, _ = _disabled_injector()
    for _ in range(3):
        await injector.put_inbound(_fake_frame())
    await injector.wait_for_trigger()
    assert stream._outbound.empty()


# ---------------------------------------------------------------------------
# Burst tracking skipped when disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_task_created_when_disabled() -> None:
    """Even after multiple frames, no trigger task is created when disabled."""
    injector, _, _ = _disabled_injector()
    for _ in range(5):
        await injector.put_inbound(_fake_frame())
    assert injector._trigger_task is None


# ---------------------------------------------------------------------------
# Pipeline-level: disabled barge_in scenario produces no interruption events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_no_interruption_when_disabled() -> None:
    from voicetest.engine.run_call import run_call
    from voicetest.providers.targets import LocalLoopback
    from voicetest.scenarios import BUILTIN_DIR, load_scenario
    from voicetest.scenarios.models import Limits

    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert scenario.barge_in.enabled is False
    short = scenario.model_copy(update={"limits": Limits(max_turns=2, max_duration_s=30)})

    result = await run_call(short, LocalLoopback())
    assert not any(e.kind == "interruption" for e in result.events)


@pytest.mark.asyncio
async def test_pipeline_completes_without_error_when_disabled() -> None:
    from voicetest.engine.run_call import run_call
    from voicetest.models import CallOutcome
    from voicetest.providers.targets import LocalLoopback
    from voicetest.scenarios import BUILTIN_DIR, load_scenario
    from voicetest.scenarios.models import Limits

    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert scenario.barge_in.enabled is False
    short = scenario.model_copy(update={"limits": Limits(max_turns=2, max_duration_s=30)})

    result = await run_call(short, LocalLoopback())
    assert result.outcome == CallOutcome.COMPLETED
    assert result.error is None
