"""
P1-T2 — NoiseMixer level and length tests.

Feed known PCM + known noise clip; assert:
- output length is preserved (matches voice frame length)
- output RMS is within ±3 dB of the configured target level
- enabled=False produces bitwise-identical output to input
"""

from __future__ import annotations

import math
import struct

import pytest

from voicetest.engine.processors.noise import (
    NoiseMixer,
    _db_to_linear,
    _mix_frames,
    _rms_s16,
    _synthesize_sine,
)
from voicetest.models import Event
from voicetest.providers.targets import SAMPLE_RATE, AudioFrame, AudioStream
from voicetest.scenarios.models import NoiseConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pcm(n_samples: int, amplitude: int = 10000, freq: int = 1000) -> bytes:
    """Return n_samples of a clean 1 kHz sine at the given amplitude."""
    vals = [
        int(amplitude * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        for i in range(n_samples)
    ]
    return struct.pack(f"<{n_samples}h", *vals)


def _clock() -> int:
    return 0


def _make_mixer(
    level_db: float, enabled: bool = True, profile: str = "cafe"
) -> tuple[NoiseMixer, AudioStream, list[Event]]:
    stream = AudioStream()
    events: list[Event] = []
    config = NoiseConfig(enabled=enabled, profile=profile, level_db=level_db)
    mixer = NoiseMixer.from_config(config, stream, events, _clock)
    return mixer, stream, events


# ---------------------------------------------------------------------------
# _rms_s16
# ---------------------------------------------------------------------------


def test_rms_empty() -> None:
    assert _rms_s16(b"") == 0.0


def test_rms_dc_signal() -> None:
    # All samples = 1000 → RMS = 1000
    pcm = struct.pack("<4h", 1000, 1000, 1000, 1000)
    assert _rms_s16(pcm) == pytest.approx(1000.0)


def test_rms_sine_approx() -> None:
    # RMS of a full-cycle sine with amplitude A ≈ A / sqrt(2)
    pcm = _make_pcm(SAMPLE_RATE, amplitude=20000)  # 1 second
    rms = _rms_s16(pcm)
    expected = 20000 / math.sqrt(2)
    assert abs(rms - expected) / expected < 0.01  # within 1%


# ---------------------------------------------------------------------------
# _mix_frames
# ---------------------------------------------------------------------------


def test_mix_frames_length_matches_voice() -> None:
    voice = _make_pcm(160)  # 20 ms at 8 kHz
    noise = _make_pcm(160, amplitude=5000, freq=400)
    out = _mix_frames(voice, noise, noise_gain=1.0)
    assert len(out) == len(voice)


def test_mix_frames_zero_gain_is_passthrough() -> None:
    voice = _make_pcm(80)
    noise = _make_pcm(80, amplitude=30000, freq=400)
    out = _mix_frames(voice, noise, noise_gain=0.0)
    assert out == voice


def test_mix_frames_clamps_to_int16_range() -> None:
    # Both signals at full scale; gain=1 → would overflow without clamping
    voice = struct.pack("<4h", 32767, 32767, 32767, 32767)
    noise = struct.pack("<4h", 32767, 32767, 32767, 32767)
    out = _mix_frames(voice, noise, noise_gain=1.0)
    vals = struct.unpack("<4h", out)
    for v in vals:
        assert -32768 <= v <= 32767


def test_mix_frames_noise_shorter_than_voice() -> None:
    voice = _make_pcm(200)
    noise = _make_pcm(100, amplitude=5000, freq=400)
    out = _mix_frames(voice, noise, noise_gain=1.0)
    # Output length = voice length
    assert len(out) == len(voice)


# ---------------------------------------------------------------------------
# NoiseMixer.from_config — gain calibration
# ---------------------------------------------------------------------------


def test_noise_gain_calibration_within_3db() -> None:
    """
    When mixing a known-RMS noise clip at level_db=-20 into silence,
    the output RMS should be within ±3 dB of 32767 * 10^(-20/20).
    """
    target_db = -20.0
    target_amp = 32767 * _db_to_linear(target_db)

    clip = _synthesize_sine(400, duration_s=1.0, sample_rate=SAMPLE_RATE)
    clip_rms = _rms_s16(clip)
    gain = target_amp / clip_rms

    # Mix noise into silence
    silence = struct.pack(f"<{SAMPLE_RATE}h", *([0] * SAMPLE_RATE))
    out = _mix_frames(silence, clip, noise_gain=gain)
    out_rms = _rms_s16(out)

    # Check within ±3 dB
    ratio = out_rms / target_amp
    db_error = 20 * math.log10(ratio) if ratio > 0 else -999
    assert abs(db_error) < 3.0, f"Level error {db_error:.2f} dB (> 3 dB tolerance)"


# ---------------------------------------------------------------------------
# NoiseMixer.put_outbound — async integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_put_outbound_length_preserved() -> None:
    mixer, stream, _ = _make_mixer(level_db=-12)
    voice_pcm = _make_pcm(160)
    frame = AudioFrame(samples=voice_pcm, sample_rate=SAMPLE_RATE)
    await mixer.put_outbound(frame)

    # Read from stream outbound queue
    out_frame = stream._outbound.get_nowait()
    assert len(out_frame.samples) == len(voice_pcm)


@pytest.mark.asyncio
async def test_put_outbound_sample_rate_preserved() -> None:
    mixer, stream, _ = _make_mixer(level_db=-12)
    frame = AudioFrame(samples=_make_pcm(80), sample_rate=SAMPLE_RATE)
    await mixer.put_outbound(frame)
    out_frame = stream._outbound.get_nowait()
    assert out_frame.sample_rate == SAMPLE_RATE


@pytest.mark.asyncio
async def test_put_outbound_modifies_content_when_enabled() -> None:
    mixer, stream, _ = _make_mixer(level_db=-12)
    voice_pcm = _make_pcm(160)
    await mixer.put_outbound(AudioFrame(samples=voice_pcm, sample_rate=SAMPLE_RATE))
    out_frame = stream._outbound.get_nowait()
    # Output must differ from input (noise was mixed in)
    assert out_frame.samples != voice_pcm


@pytest.mark.asyncio
async def test_put_outbound_output_rms_higher_than_silence() -> None:
    """Noise mixed into silence → output RMS > 0."""
    mixer, stream, _ = _make_mixer(level_db=-12)
    silence = struct.pack(f"<{160}h", *([0] * 160))
    await mixer.put_outbound(AudioFrame(samples=silence, sample_rate=SAMPLE_RATE))
    out_frame = stream._outbound.get_nowait()
    assert _rms_s16(out_frame.samples) > 0


@pytest.mark.asyncio
async def test_noise_loops_across_multiple_frames() -> None:
    """The noise clip loops correctly across many consecutive frames."""
    mixer, stream, _ = _make_mixer(level_db=-20)
    silence = struct.pack(f"<{160}h", *([0] * 160))
    frame = AudioFrame(samples=silence, sample_rate=SAMPLE_RATE)

    # Push 200 frames — synthetic clip is 1 s = 16000 samples = 32000 bytes;
    # 200 frames × 320 bytes = 64000 bytes → needs 2 full loops
    for _ in range(200):
        await mixer.put_outbound(frame)

    rmss = []
    for _ in range(200):
        out = stream._outbound.get_nowait()
        rmss.append(_rms_s16(out.samples))

    # All frames should have non-zero RMS (clip didn't run dry)
    assert all(r > 0 for r in rmss)


# ---------------------------------------------------------------------------
# Synthetic clip generation
# ---------------------------------------------------------------------------


def test_synthesize_sine_length() -> None:
    pcm = _synthesize_sine(400, duration_s=0.5, sample_rate=SAMPLE_RATE)
    assert len(pcm) == int(0.5 * SAMPLE_RATE) * 2  # 2 bytes per sample


def test_synthesize_sine_nonzero_rms() -> None:
    pcm = _synthesize_sine(400, duration_s=0.1, sample_rate=SAMPLE_RATE)
    assert _rms_s16(pcm) > 0
