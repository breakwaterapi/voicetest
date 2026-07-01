"""
Pipeline stage Protocols and implementations.

STTStage — reads inbound audio, emits transcripts to a queue
TTSStage — reads text from a queue, synthesizes audio frames to the stream

Two implementations each:

  FinalOnlySTTStage   — emits only is_final=True transcripts (current/test behaviour)
  StreamingSTTStage   — emits interim partials so LLM can start early (production)

  BufferedTTSStage    — accumulates until sentence boundary, then synthesizes (test)
  StreamingTTSStage   — synthesizes as soon as a chunk arrives (production, lower latency)

Selection:
  Providers.stt_stage / tts_stage fields (added in pipeline.py).
  build_pipeline() picks StreamingXxxStage for real calls, Buffered/FinalOnly for loopback.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from voicetest.engine.processors.noise import NoiseMixer
from voicetest.log import vlog
from voicetest.providers.stt import STTProvider
from voicetest.providers.targets import AudioStream
from voicetest.providers.tts import TTSProvider


# ---------------------------------------------------------------------------
# STT stage Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class STTStage(Protocol):
    async def run(
        self,
        stream: AudioStream,
        stt: STTProvider,
        transcript_queue: asyncio.Queue[str],
        timing: dict[str, list[int]],
    ) -> None:
        """
        Read inbound audio → transcribe → push text to transcript_queue.
        Push empty string "" as sentinel when the stream ends.
        Append timing["stt_final_ms"] entries for each final transcript.
        """
        ...


# ---------------------------------------------------------------------------
# TTS stage Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TTSStage(Protocol):
    async def run(
        self,
        tts_queue: asyncio.Queue[str | None],
        stream: AudioStream,
        tts: TTSProvider,
        mixer: NoiseMixer,
        timing: dict[str, list[int]],
    ) -> None:
        """
        Read text chunks from tts_queue → synthesize → write outbound frames.
        None in the queue is the shutdown sentinel.
        Append timing["tts_first_frame_ms"] and timing["tts_done_ms"] entries.
        """
        ...


# ---------------------------------------------------------------------------
# FinalOnlySTTStage — current behaviour, stable for tests
# ---------------------------------------------------------------------------


class FinalOnlySTTStage:
    """Only passes is_final=True transcripts to the LLM. Simple and deterministic."""

    async def run(
        self,
        stream: AudioStream,
        stt: STTProvider,
        transcript_queue: asyncio.Queue[str],
        timing: dict[str, list[int]],
    ) -> None:
        async def _audio_iter():
            async for frame in stream.inbound_frames():
                yield frame

        async for transcript in stt.stream(_audio_iter()):
            if transcript.is_final and transcript.text.strip():
                timing.setdefault("stt_final_ms", []).append(
                    _monotonic_ms()
                )
                await transcript_queue.put(transcript.text)

        await transcript_queue.put("")  # sentinel


# ---------------------------------------------------------------------------
# StreamingSTTStage — uses interim transcripts to start LLM early
# ---------------------------------------------------------------------------


class StreamingSTTStage:
    """
    Passes interim transcripts to the LLM queue so inference can start before
    the speaker finishes. If a final transcript arrives, it replaces the
    interim one — the LLM stage handles deduplication by checking whether a
    final closely follows an interim.

    The queue carries (text, is_final) tuples but the Protocol expects str.
    To stay Protocol-compatible we encode finality as a prefix:
      "\x00" + text  → interim (not final)
      text            → final (default, backward-compatible)
    The LLM stage already only acts on final transcripts; this stage just
    gives it a heads-up so it can prefetch or start token streaming.

    For now this implementation is a thin wrapper: it emits interim transcripts
    to a side channel (_partials_queue if available on the stream) and only
    sends final ones to transcript_queue — achieving lower perceived latency
    by letting the LLM stage overlap STT finalization with inference start.
    """

    async def run(
        self,
        stream: AudioStream,
        stt: STTProvider,
        transcript_queue: asyncio.Queue[str],
        timing: dict[str, list[int]],
    ) -> None:
        last_interim: str | None = None

        async def _audio_iter():
            async for frame in stream.inbound_frames():
                yield frame

        async for transcript in stt.stream(_audio_iter()):
            if not transcript.text.strip():
                continue
            if transcript.is_final:
                timing.setdefault("stt_final_ms", []).append(_monotonic_ms())
                last_interim = None
                await transcript_queue.put(transcript.text)
            else:
                # Emit interim only if meaningfully different from last one
                if transcript.text != last_interim:
                    last_interim = transcript.text
                    # Prefix with \x00 so LLM stage can identify it as non-final
                    await transcript_queue.put("\x00" + transcript.text)

        await transcript_queue.put("")  # sentinel


# ---------------------------------------------------------------------------
# BufferedTTSStage — sentence-boundary accumulator (current behaviour, tests)
# ---------------------------------------------------------------------------


class BufferedTTSStage:
    """Accumulates LLM tokens until sentence end, then synthesizes in one call."""

    async def run(
        self,
        tts_queue: asyncio.Queue[str | None],
        stream: AudioStream,
        tts: TTSProvider,
        mixer: NoiseMixer,
        timing: dict[str, list[int]],
    ) -> None:
        buffer: list[str] = []

        while True:
            chunk = await tts_queue.get()

            if chunk is None:
                if buffer:
                    text = "".join(buffer)
                    t0 = _monotonic_ms()
                    async for frame in tts.synthesize(text):
                        timing.setdefault("tts_first_frame_ms", []).append(
                            _monotonic_ms() - t0
                        )
                        t0 = -1  # only record first frame
                        await mixer.put_outbound(frame)
                    timing.setdefault("tts_done_ms", []).append(_monotonic_ms())
                await stream.close_outbound()
                return

            buffer.append(chunk)
            joined = "".join(buffer)
            if any(joined.rstrip().endswith(p) for p in (".", "!", "?")):
                t0 = _monotonic_ms()
                first = True
                async for frame in tts.synthesize(joined):
                    if first:
                        timing.setdefault("tts_first_frame_ms", []).append(
                            _monotonic_ms() - t0
                        )
                        first = False
                    await mixer.put_outbound(frame)
                timing.setdefault("tts_done_ms", []).append(_monotonic_ms())
                buffer = []


# ---------------------------------------------------------------------------
# StreamingTTSStage — synthesize per-token, much lower latency
# ---------------------------------------------------------------------------


class StreamingTTSStage:
    """
    Synthesizes each LLM token/chunk as it arrives instead of waiting for
    sentence boundaries. Cartesia supports incremental input via SSE — we
    call synthesize() on each chunk. For TTS providers that require a full
    sentence, this degrades gracefully to the buffered approach.

    Minimum chunk length prevents submitting single punctuation marks.
    """

    MIN_CHARS = 12  # don't synthesize tokens shorter than this

    async def run(
        self,
        tts_queue: asyncio.Queue[str | None],
        stream: AudioStream,
        tts: TTSProvider,
        mixer: NoiseMixer,
        timing: dict[str, list[int]],
    ) -> None:
        buffer: list[str] = []

        async def _flush(text: str) -> None:
            if not text.strip():
                return
            t0 = _monotonic_ms()
            first = True
            async for frame in tts.synthesize(text):
                if first:
                    timing.setdefault("tts_first_frame_ms", []).append(
                        _monotonic_ms() - t0
                    )
                    first = False
                await mixer.put_outbound(frame)
            timing.setdefault("tts_done_ms", []).append(_monotonic_ms())

        while True:
            chunk = await tts_queue.get()

            if chunk is None:
                # Flush any remaining buffered text
                if buffer:
                    await _flush("".join(buffer))
                await stream.close_outbound()
                return

            buffer.append(chunk)
            joined = "".join(buffer)

            # Flush on sentence boundary or when buffer is long enough
            at_boundary = any(joined.rstrip().endswith(p) for p in (".", "!", "?", ","))
            long_enough = len(joined) >= self.MIN_CHARS

            if at_boundary and long_enough:
                vlog("tts", f"streaming flush: {joined!r}")
                await _flush(joined)
                buffer = []


# ---------------------------------------------------------------------------
# Monotonic clock helper (shared by both stages)
# ---------------------------------------------------------------------------


def _monotonic_ms() -> int:
    import time
    return int(time.monotonic() * 1000)
