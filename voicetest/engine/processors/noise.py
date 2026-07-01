"""
Background-noise injection processor.

Architecture position (architecture.md §4.2):
    TTS → [NoiseMixer] → Transport (outbound)

NoiseMixer wraps an AudioStream and mixes a looping noise bed into every
outbound PCM frame at `level_db` relative to the raw noise clip amplitude.
It logs `noise_on` / `noise_off` events to a shared event list and is a
passthrough when `NoiseConfig.enabled` is False.

Audio facts (tech-stack.md):
- Phone channel is 8 kHz, 16-bit signed PCM (G.711 μ-law at the edge).
- All mixing happens at SAMPLE_RATE = 8000 Hz.
- The noise clip is loaded once, resampled to SAMPLE_RATE if needed, and
  looped in-memory for the duration of the call.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from voicetest.models import Event
from voicetest.providers.targets import SAMPLE_RATE, AudioFrame, AudioStream

if TYPE_CHECKING:
    from voicetest.scenarios.models import NoiseConfig


# ---------------------------------------------------------------------------
# Clip loading
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets" / "noise"


def _load_clip_pcm(profile: str) -> bytes:
    """
    Load a noise clip for `profile` (cafe | street | car) and return raw
    16-bit signed PCM bytes at SAMPLE_RATE.

    Search order:
    1. assets/noise/<profile>.wav  — real royalty-free clip (production)
    2. assets/noise/<profile>.pcm  — raw PCM blob (alternative)
    3. Synthetic fallback — a quiet sine tone, so tests run without files.
    """
    for suffix, loader in ((".wav", _load_wav), (".pcm", _load_raw_pcm)):
        path = ASSETS_DIR / f"{profile}{suffix}"
        if path.exists():
            return loader(path)

    # Synthetic fallback: 1-second sine at a profile-specific frequency.
    freq = {"cafe": 400, "street": 300, "car": 200}.get(profile, 350)
    return _synthesize_sine(freq, duration_s=1.0, sample_rate=SAMPLE_RATE)


def _load_wav(path: Path) -> bytes:
    """Read a WAV file and return 16-bit signed PCM at SAMPLE_RATE."""
    import wave

    with wave.open(str(path), "rb") as wf:
        src_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    # Convert to mono 16-bit if necessary
    samples = _to_mono_s16(raw, n_channels, sampwidth)

    if src_rate != SAMPLE_RATE:
        samples = _resample_linear(samples, src_rate, SAMPLE_RATE)

    return samples


def _load_raw_pcm(path: Path) -> bytes:
    """Load a raw 16-bit signed PCM file assumed to already be at SAMPLE_RATE."""
    return path.read_bytes()


def _to_mono_s16(raw: bytes, n_channels: int, sampwidth: int) -> bytes:
    """Convert multi-channel / non-16-bit PCM to mono 16-bit."""
    if sampwidth == 1:
        # 8-bit unsigned → 16-bit signed
        vals = [((b - 128) << 8) for b in raw]
    elif sampwidth == 2:
        n = len(raw) // 2
        vals = list(struct.unpack(f"<{n}h", raw))
    elif sampwidth == 3:
        # 24-bit → 16-bit (drop low byte)
        vals = []
        for i in range(0, len(raw) - 2, 3):
            v = struct.unpack("<i", raw[i : i + 3] + b"\x00")[0] >> 8
            vals.append(v)
    elif sampwidth == 4:
        n = len(raw) // 4
        raw_vals = struct.unpack(f"<{n}i", raw)
        vals = [v >> 16 for v in raw_vals]
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        # Average channels
        mono = []
        for i in range(0, len(vals), n_channels):
            chunk = vals[i : i + n_channels]
            mono.append(int(sum(chunk) / len(chunk)))
        vals = mono

    return struct.pack(f"<{len(vals)}h", *vals)


def _resample_linear(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample from src_rate to dst_rate."""
    n = len(pcm) // 2
    src = struct.unpack(f"<{n}h", pcm)

    ratio = src_rate / dst_rate
    out_len = int(n / ratio)
    out: list[int] = []

    for i in range(out_len):
        pos = i * ratio
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        val = int(src[lo] * (1 - frac) + src[hi] * frac)
        out.append(max(-32768, min(32767, val)))

    return struct.pack(f"<{len(out)}h", *out)


def _synthesize_sine(freq: int, duration_s: float, sample_rate: int) -> bytes:
    """Generate a mono 16-bit sine tone (amplitude 0.15 full-scale)."""
    n = int(duration_s * sample_rate)
    amplitude = int(32767 * 0.15)
    vals = [
        int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(n)
    ]
    return struct.pack(f"<{n}h", *vals)


# ---------------------------------------------------------------------------
# dB helpers
# ---------------------------------------------------------------------------


def _db_to_linear(db: float) -> float:
    """Convert a dB level (relative to full-scale) to a linear multiplier."""
    return 10 ** (db / 20.0)


def _rms_s16(pcm: bytes) -> float:
    """Root-mean-square of a 16-bit signed PCM buffer. Returns 0.0 for empty."""
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    vals = struct.unpack(f"<{n}h", pcm)
    mean_sq = sum(v * v for v in vals) / n
    return math.sqrt(mean_sq)


def _mix_frames(voice: bytes, noise: bytes, noise_gain: float) -> bytes:
    """
    Mix voice PCM with noise PCM at noise_gain (linear).

    Both buffers must be 16-bit signed. If lengths differ, noise is truncated
    or the shorter voice frame drives the output length.
    """
    n_voice = len(voice) // 2
    n_noise = len(noise) // 2
    n = min(n_voice, n_noise)

    voice_vals = struct.unpack(f"<{n_voice}h", voice)
    noise_vals = struct.unpack(f"<{n_noise}h", noise)

    out: list[int] = []
    for i in range(n):
        mixed = int(voice_vals[i] + noise_vals[i] * noise_gain)
        out.append(max(-32768, min(32767, mixed)))
    # Append remaining voice samples unmodified if voice > noise
    for i in range(n, n_voice):
        out.append(voice_vals[i])

    return struct.pack(f"<{len(out)}h", *out)


# ---------------------------------------------------------------------------
# NoiseMixer
# ---------------------------------------------------------------------------


class NoiseMixer:
    """
    Wraps an AudioStream's put_outbound path to mix a noise bed into outgoing
    PCM frames.

    Usage:
        mixer = NoiseMixer.from_config(config, stream, events, clock_fn)
        await mixer.start()
        # replace stream.put_outbound calls with mixer.put_outbound(frame)
        await mixer.stop()

    When config.enabled is False this is a transparent passthrough and no
    noise_on / noise_off events are emitted.
    """

    def __init__(
        self,
        stream: AudioStream,
        events: list[Event],
        clock_fn: Callable[[], int],
        clip_pcm: bytes,
        noise_gain: float,
        enabled: bool,
    ) -> None:
        self._stream = stream
        self._events = events
        self._clock_fn = clock_fn
        self._clip = clip_pcm  # pre-resampled, looped on demand
        self._gain = noise_gain
        self._enabled = enabled
        self._clip_pos = 0  # byte offset into _clip (always even)

    @classmethod
    def from_config(
        cls,
        config: NoiseConfig,
        stream: AudioStream,
        events: list[Event],
        clock_fn: Callable[[], int],
    ) -> NoiseMixer:
        if config.enabled:
            clip = _load_clip_pcm(config.profile)
            rms = _rms_s16(clip)
            # Compute gain so that: 20*log10(rms * gain / 32767) ≈ level_db
            # level_db is relative to full-scale (e.g. -12 dB FS)
            target_amp = 32767 * _db_to_linear(config.level_db)
            gain = (target_amp / rms) if rms > 0 else 0.0
        else:
            clip = b""
            gain = 0.0
        return cls(stream, events, clock_fn, clip, gain, config.enabled)

    def _next_noise_chunk(self, n_bytes: int) -> bytes:
        """Return n_bytes of looped noise PCM."""
        result = bytearray()
        remaining = n_bytes
        while remaining > 0:
            end = self._clip_pos + remaining
            chunk = self._clip[self._clip_pos : end]
            result.extend(chunk)
            remaining -= len(chunk)
            self._clip_pos = end
            if self._clip_pos >= len(self._clip):
                self._clip_pos = 0  # loop
        return bytes(result)

    async def start(self) -> None:
        if self._enabled:
            self._events.append(Event(kind="noise_on", at_ms=self._clock_fn()))

    async def stop(self) -> None:
        if self._enabled:
            self._events.append(Event(kind="noise_off", at_ms=self._clock_fn()))

    async def put_outbound(self, frame: AudioFrame) -> None:
        """Mix noise into frame then forward to the underlying stream."""
        if not self._enabled or not self._clip:
            await self._stream.put_outbound(frame)
            return

        noise_chunk = self._next_noise_chunk(len(frame.samples))
        mixed = _mix_frames(frame.samples, noise_chunk, self._gain)
        await self._stream.put_outbound(AudioFrame(samples=mixed, sample_rate=frame.sample_rate))
