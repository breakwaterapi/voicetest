"""
P1-T2 — NoiseMixer disabled-passthrough tests.

When NoiseConfig.enabled is False:
- put_outbound forwards frames bitwise-identical (PCM bytes unchanged)
- sample_rate is preserved
- no noise events are logged
- pipeline still runs end-to-end without error (integration smoke)
"""

from __future__ import annotations

import math
import struct

import pytest

from voicetest.engine.processors.noise import NoiseMixer
from voicetest.models import Event
from voicetest.providers.targets import SAMPLE_RATE, AudioFrame, AudioStream
from voicetest.scenarios.models import NoiseConfig

# ---------------------------------------------------------------------------
# Helper: we can't import _make_pcm from noise.py (it's test-local in mix tests)
# so define it here too.
# ---------------------------------------------------------------------------


def _make_pcm(n_samples: int, amplitude: int = 10000, freq: int = 1000) -> bytes:
    vals = [
        int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        for i in range(n_samples)
    ]
    return struct.pack(f"<{n_samples}h", *vals)


def _disabled_mixer() -> tuple[NoiseMixer, AudioStream, list[Event]]:
    stream = AudioStream()
    events: list[Event] = []
    config = NoiseConfig(enabled=False, profile="cafe", level_db=-12)
    mixer = NoiseMixer.from_config(config, stream, events, lambda: 0)
    return mixer, stream, events


# ---------------------------------------------------------------------------
# Passthrough: PCM bytes unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disabled_passthrough_bytes_identical() -> None:
    mixer, stream, _ = _disabled_mixer()
    pcm = _make_pcm(160)
    await mixer.put_outbound(AudioFrame(samples=pcm, sample_rate=SAMPLE_RATE))
    out = stream._outbound.get_nowait()
    assert out.samples == pcm


@pytest.mark.asyncio
async def test_disabled_passthrough_sample_rate_preserved() -> None:
    mixer, stream, _ = _disabled_mixer()
    frame = AudioFrame(samples=_make_pcm(80), sample_rate=SAMPLE_RATE)
    await mixer.put_outbound(frame)
    out = stream._outbound.get_nowait()
    assert out.sample_rate == SAMPLE_RATE


@pytest.mark.asyncio
async def test_disabled_passthrough_multiple_frames() -> None:
    mixer, stream, _ = _disabled_mixer()
    frames = [_make_pcm(160, freq=f) for f in (500, 700, 900)]
    for pcm in frames:
        await mixer.put_outbound(AudioFrame(samples=pcm, sample_rate=SAMPLE_RATE))

    for expected_pcm in frames:
        out = stream._outbound.get_nowait()
        assert out.samples == expected_pcm


@pytest.mark.asyncio
async def test_disabled_passthrough_empty_frame() -> None:
    mixer, stream, _ = _disabled_mixer()
    empty = AudioFrame(samples=b"", sample_rate=SAMPLE_RATE)
    await mixer.put_outbound(empty)
    out = stream._outbound.get_nowait()
    assert out.samples == b""


@pytest.mark.asyncio
async def test_disabled_passthrough_large_frame() -> None:
    mixer, stream, _ = _disabled_mixer()
    pcm = _make_pcm(SAMPLE_RATE)  # 1 second worth
    await mixer.put_outbound(AudioFrame(samples=pcm, sample_rate=SAMPLE_RATE))
    out = stream._outbound.get_nowait()
    assert out.samples == pcm


# ---------------------------------------------------------------------------
# No events when disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_events_emitted_on_start_when_disabled() -> None:
    mixer, _, events = _disabled_mixer()
    await mixer.start()
    assert events == []


@pytest.mark.asyncio
async def test_no_events_emitted_on_stop_when_disabled() -> None:
    mixer, _, events = _disabled_mixer()
    await mixer.start()
    await mixer.stop()
    assert events == []


@pytest.mark.asyncio
async def test_no_events_emitted_on_put_outbound_when_disabled() -> None:
    mixer, _, events = _disabled_mixer()
    pcm = _make_pcm(160)
    await mixer.put_outbound(AudioFrame(samples=pcm, sample_rate=SAMPLE_RATE))
    assert events == []


# ---------------------------------------------------------------------------
# Pipeline smoke: disabled noise doesn't break run_call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_with_noise_disabled_completes() -> None:
    """run_call with a noise-disabled scenario must complete without error."""
    from voicetest.engine.run_call import run_call
    from voicetest.models import CallOutcome
    from voicetest.providers.targets import LocalLoopback
    from voicetest.scenarios import BUILTIN_DIR, load_scenario
    from voicetest.scenarios.models import Limits

    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert scenario.noise.enabled is False
    short = scenario.model_copy(update={"limits": Limits(max_turns=2, max_duration_s=30)})

    result = await run_call(short, LocalLoopback())
    assert result.outcome == CallOutcome.COMPLETED
    assert result.error is None


@pytest.mark.asyncio
async def test_pipeline_with_noise_enabled_scenario_completes() -> None:
    """run_call with a noise-enabled scenario must complete without error."""
    from voicetest.engine.run_call import run_call
    from voicetest.models import CallOutcome
    from voicetest.providers.targets import LocalLoopback
    from voicetest.scenarios import BUILTIN_DIR, load_scenario
    from voicetest.scenarios.models import Limits

    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-mumbled-phone.yaml")
    assert scenario.noise.enabled is True
    short = scenario.model_copy(update={"limits": Limits(max_turns=2, max_duration_s=30)})

    result = await run_call(short, LocalLoopback())
    assert result.outcome == CallOutcome.COMPLETED
    assert result.error is None


@pytest.mark.asyncio
async def test_noise_on_off_events_present_in_call_result() -> None:
    """When noise is enabled, call result events include noise_on and noise_off."""
    from voicetest.engine.run_call import run_call
    from voicetest.providers.targets import LocalLoopback
    from voicetest.scenarios import BUILTIN_DIR, load_scenario
    from voicetest.scenarios.models import Limits

    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-mumbled-phone.yaml")
    assert scenario.noise.enabled is True

    # Use a very short limit so the test completes quickly without hitting max_duration_s.
    short_scenario = scenario.model_copy(update={"limits": Limits(max_turns=2, max_duration_s=30)})
    result = await run_call(short_scenario, LocalLoopback())
    event_kinds = {e.kind for e in result.events}
    assert "noise_on" in event_kinds
    assert "noise_off" in event_kinds


@pytest.mark.asyncio
async def test_noise_events_absent_when_disabled_in_call_result() -> None:
    """When noise is disabled, call result events must not contain noise_on/off."""
    from voicetest.engine.run_call import run_call
    from voicetest.providers.targets import LocalLoopback
    from voicetest.scenarios import BUILTIN_DIR, load_scenario
    from voicetest.scenarios.models import Limits

    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert scenario.noise.enabled is False
    short = scenario.model_copy(update={"limits": Limits(max_turns=2, max_duration_s=30)})

    result = await run_call(short, LocalLoopback())
    event_kinds = {e.kind for e in result.events}
    assert "noise_on" not in event_kinds
    assert "noise_off" not in event_kinds
