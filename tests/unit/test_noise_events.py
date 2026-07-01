"""
P1-T2 — NoiseMixer event logging tests.

Verifies that:
- noise_on is logged when mixer.start() is called (enabled=True)
- noise_off is logged when mixer.stop() is called (enabled=True)
- events carry timestamps from the clock function
- noise_on appears before noise_off in the event list
- no events emitted when enabled=False
"""

from __future__ import annotations

import pytest

from voicetest.engine.processors.noise import NoiseMixer
from voicetest.models import Event
from voicetest.providers.targets import AudioStream
from voicetest.scenarios.models import NoiseConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixer(
    enabled: bool,
    clock_start: int = 0,
) -> tuple[NoiseMixer, list[Event], list[int]]:
    """
    Return (mixer, events, tick_list).

    The clock function returns tick_list[0] each call, so tests can control
    timestamps by mutating tick_list.
    """
    stream = AudioStream()
    events: list[Event] = []
    tick = [clock_start]

    def clock() -> int:
        return tick[0]

    config = NoiseConfig(enabled=enabled, profile="cafe", level_db=-12)
    mixer = NoiseMixer.from_config(config, stream, events, clock)
    return mixer, events, tick


# ---------------------------------------------------------------------------
# Enabled: noise_on / noise_off emitted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noise_on_event_emitted_on_start() -> None:
    mixer, events, _ = _make_mixer(enabled=True)
    await mixer.start()
    kinds = [e.kind for e in events]
    assert "noise_on" in kinds


@pytest.mark.asyncio
async def test_noise_off_event_emitted_on_stop() -> None:
    mixer, events, _ = _make_mixer(enabled=True)
    await mixer.start()
    await mixer.stop()
    kinds = [e.kind for e in events]
    assert "noise_off" in kinds


@pytest.mark.asyncio
async def test_noise_on_before_noise_off() -> None:
    mixer, events, _ = _make_mixer(enabled=True)
    await mixer.start()
    await mixer.stop()
    kinds = [e.kind for e in events]
    assert kinds.index("noise_on") < kinds.index("noise_off")


@pytest.mark.asyncio
async def test_noise_on_timestamp() -> None:
    mixer, events, tick = _make_mixer(enabled=True, clock_start=42)
    await mixer.start()
    on_event = next(e for e in events if e.kind == "noise_on")
    assert on_event.at_ms == 42


@pytest.mark.asyncio
async def test_noise_off_timestamp() -> None:
    mixer, events, tick = _make_mixer(enabled=True, clock_start=0)
    await mixer.start()
    tick[0] = 5000  # advance clock before stop
    await mixer.stop()
    off_event = next(e for e in events if e.kind == "noise_off")
    assert off_event.at_ms == 5000


@pytest.mark.asyncio
async def test_noise_off_timestamp_after_noise_on_timestamp() -> None:
    mixer, events, tick = _make_mixer(enabled=True, clock_start=100)
    await mixer.start()
    tick[0] = 200
    await mixer.stop()
    on_ts = next(e for e in events if e.kind == "noise_on").at_ms
    off_ts = next(e for e in events if e.kind == "noise_off").at_ms
    assert off_ts >= on_ts


@pytest.mark.asyncio
async def test_exactly_one_noise_on_per_start() -> None:
    mixer, events, _ = _make_mixer(enabled=True)
    await mixer.start()
    on_events = [e for e in events if e.kind == "noise_on"]
    assert len(on_events) == 1


@pytest.mark.asyncio
async def test_exactly_one_noise_off_per_stop() -> None:
    mixer, events, _ = _make_mixer(enabled=True)
    await mixer.start()
    await mixer.stop()
    off_events = [e for e in events if e.kind == "noise_off"]
    assert len(off_events) == 1


@pytest.mark.asyncio
async def test_events_are_Event_instances() -> None:
    mixer, events, _ = _make_mixer(enabled=True)
    await mixer.start()
    await mixer.stop()
    for e in events:
        assert isinstance(e, Event)


# ---------------------------------------------------------------------------
# Disabled: no events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_noise_on_when_disabled() -> None:
    mixer, events, _ = _make_mixer(enabled=False)
    await mixer.start()
    assert not any(e.kind == "noise_on" for e in events)


@pytest.mark.asyncio
async def test_no_noise_off_when_disabled() -> None:
    mixer, events, _ = _make_mixer(enabled=False)
    await mixer.start()
    await mixer.stop()
    assert not any(e.kind == "noise_off" for e in events)


@pytest.mark.asyncio
async def test_no_events_at_all_when_disabled() -> None:
    mixer, events, _ = _make_mixer(enabled=False)
    await mixer.start()
    await mixer.stop()
    assert events == []


# ---------------------------------------------------------------------------
# Events appended to caller-supplied list (shared reference)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_appended_to_shared_list() -> None:
    """Mixer must append to the same list object passed at construction."""
    stream = AudioStream()
    shared_events: list[Event] = []
    tick = [0]
    config = NoiseConfig(enabled=True, profile="cafe", level_db=-12)
    mixer = NoiseMixer.from_config(config, stream, shared_events, lambda: tick[0])
    await mixer.start()
    await mixer.stop()
    assert len(shared_events) == 2  # noise_on + noise_off
