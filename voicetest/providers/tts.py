"""TTS provider interface and implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from voicetest.providers.targets import AudioFrame, SAMPLE_RATE, _text_to_frame


@runtime_checkable
class TTSProvider(Protocol):
    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]: ...


# ---------------------------------------------------------------------------
# Cartesia (real provider — requires CARTESIA_API_KEY)
# ---------------------------------------------------------------------------


class CartesiaTTS:
    """
    Cartesia Sonic-3 / Turbo streaming TTS.
    Import deferred so tests using StubTTS never need the SDK.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "sonic-2",
        voice_id: str = "a0e99841-438c-4a64-b679-ae501e7d6091",
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._voice_id = voice_id

    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        try:
            from cartesia import AsyncCartesia
        except ImportError as e:
            raise RuntimeError("cartesia not installed — pip install cartesia") from e

        # cartesia v1.x: tts.sse(...) is a coroutine returning an async generator
        # of dict chunks ({"audio": <pcm bytes>, ...}) — NOT an async context
        # manager yielding objects with `.data` (that was a pre-1.0 shape).
        client = AsyncCartesia(api_key=self._api_key)
        try:
            generator = await client.tts.sse(
                transcript=text,
                voice_id=self._voice_id,
                model_id=self._model_id,
                output_format={
                    "container": "raw",
                    "encoding": "pcm_s16le",
                    "sample_rate": SAMPLE_RATE,
                },
                stream=True,
            )
            async for chunk in generator:
                audio = chunk["audio"] if isinstance(chunk, dict) else getattr(chunk, "audio", None)
                if audio:
                    yield AudioFrame(samples=audio, sample_rate=SAMPLE_RATE)
        finally:
            # Avoid the "coroutine AsyncCartesia.close was never awaited" leak.
            await client.close()


# ---------------------------------------------------------------------------
# Stub (no network — used by tests)
# ---------------------------------------------------------------------------


class StubTTS:
    """
    Encodes text as a length-prefixed frame (mirrors _text_to_frame / LocalLoopback).
    LocalLoopback's stub STT can decode these back to text, completing the loop.
    """

    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]:
        yield _text_to_frame(text)
