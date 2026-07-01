"""STT provider interface and implementations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from voicetest.providers.targets import AudioFrame


@dataclass
class Transcript:
    text: str
    is_final: bool


@runtime_checkable
class STTProvider(Protocol):
    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]: ...


# ---------------------------------------------------------------------------
# Deepgram Flux (real provider — requires DEEPGRAM_API_KEY)
# ---------------------------------------------------------------------------


class DeepgramSTT:
    """
    Deepgram Flux streaming STT.  Uses built-in end-of-turn detection (ADR-5).
    Import is deferred so tests that use StubSTT never require the SDK.
    """

    def __init__(self, api_key: str, model: str = "nova-3") -> None:
        self._api_key = api_key
        self._model = model

    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        try:
            from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
        except ImportError as e:
            raise RuntimeError("deepgram-sdk not installed — pip install deepgram-sdk") from e

        client = DeepgramClient(self._api_key)
        dg_connection = client.listen.asynclive.v("1")

        queue: asyncio.Queue[Transcript | None] = asyncio.Queue()

        # Deepgram SDK v3 invokes handlers as handler(connection, result=..., **kw),
        # so the first positional arg is the connection — absorb it as `_conn`.
        async def on_message(_conn: object, result: object, **_: object) -> None:
            alt = result.channel.alternatives[0]
            if alt.transcript:
                await queue.put(Transcript(text=alt.transcript, is_final=result.is_final))

        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        opts = LiveOptions(
            model=self._model,
            language="en-US",
            smart_format=True,
            encoding="linear16",
            sample_rate=8000,
            channels=1,
        )
        await dg_connection.start(opts)

        async def feed() -> None:
            async for frame in audio:
                await dg_connection.send(frame.samples)
            await dg_connection.finish()
            await queue.put(None)

        asyncio.create_task(feed())

        while True:
            item = await queue.get()
            if item is None:
                return
            yield item


# ---------------------------------------------------------------------------
# Stub (no network — used by tests)
# ---------------------------------------------------------------------------


class StubSTT:
    """
    Decodes length-prefixed text frames (from LocalLoopback / _text_to_frame)
    and emits them as final transcripts.  No network, no audio codec.
    """

    async def stream(self, audio: AsyncIterator[AudioFrame]) -> AsyncIterator[Transcript]:
        from voicetest.providers.targets import _frame_to_text

        async for frame in audio:
            text = _frame_to_text(frame)
            if text is not None:
                yield Transcript(text=text, is_final=True)
