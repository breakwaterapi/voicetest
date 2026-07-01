"""
Barge-in / interruption injection processor.

Architecture position (architecture.md §4.3):
    Transport(inbound) → [BargeInInjector] → STT

BargeInInjector wraps the inbound side of an AudioStream.  It counts distinct
agent-speaking bursts (sequences of consecutive inbound frames separated by
silence gaps).  When the Nth burst starts (on_agent_turn) and
trigger_after_agent_speaks_ms have elapsed, it races a scripted caller
utterance outbound — simulating the caller speaking over the agent.

The injector logs an `interruption` event with:
  - at_ms: when the injection fired
  - agent_turn: which agent turn triggered it
  - caller_says: the injected text
  - agent_was_speaking: True if the inbound stream was still active at fire time

When barge_in.enabled is False this is a transparent passthrough — inbound
frames are forwarded unchanged and no interruption events are logged.

Interaction with the pipeline:
  - The injector wraps `put_inbound` so STT still receives all agent frames.
  - On trigger it writes the barge-in utterance to the outbound queue directly
    via a TTS provider, bypassing the normal LLM→TTS path for that one turn.
  - It does NOT cancel/interrupt the inbound stream; the transcript for the
    overlapping agent speech is still captured so scoring can detect
    failure-to-stop (ADR-7: computed, not judged).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from voicetest.models import Event
from voicetest.providers.targets import AudioFrame, AudioStream
from voicetest.providers.tts import TTSProvider

if TYPE_CHECKING:
    from voicetest.scenarios.models import BargeInConfig


# ---------------------------------------------------------------------------
# Silence detection heuristic
# ---------------------------------------------------------------------------

# Two consecutive inbound frames with the same (zero-length or stub-sentinel)
# content are treated as a gap between agent turns.  For real audio this would
# be VAD; in the stub environment (LocalLoopback) each agent reply is one frame
# burst.  We use the simpler rule: a new agent turn starts whenever we see a
# fresh inbound frame after a gap of at least _INTER_TURN_GAP_MS.
_INTER_TURN_GAP_MS = 200  # ms of silence = new agent turn


# ---------------------------------------------------------------------------
# BargeInInjector
# ---------------------------------------------------------------------------


class BargeInInjector:
    """
    Wraps an AudioStream's inbound side to watch for agent-speaking events,
    then injects a scripted caller utterance after the configured delay.

    Usage:
        injector = BargeInInjector.from_config(config, stream, tts, events, clock_fn)
        # Use injector.put_inbound() instead of stream.put_inbound() in the target.
        # Or wrap the stream via injector.wrap_stream() which patches inbound routing.
        await injector.close()  # in finally block — cancels any pending trigger task
    """

    def __init__(
        self,
        stream: AudioStream,
        tts: TTSProvider,
        events: list[Event],
        clock_fn: Callable[[], int],
        on_agent_turn: int,
        trigger_after_ms: int,
        caller_says: str,
        enabled: bool,
    ) -> None:
        self._stream = stream
        self._tts = tts
        self._events = events
        self._clock_fn = clock_fn
        self._on_agent_turn = on_agent_turn
        self._trigger_after_ms = trigger_after_ms
        self._caller_says = caller_says
        self._enabled = enabled

        self._agent_turn_count = 0          # bursts seen so far
        self._in_burst = False              # currently receiving a burst?
        self._last_frame_at_ms: int = 0     # monotonic ms of last inbound frame
        self._trigger_task: asyncio.Task[None] | None = None
        self._fired = False                 # trigger fires at most once

    @classmethod
    def from_config(
        cls,
        config: BargeInConfig,
        stream: AudioStream,
        tts: TTSProvider,
        events: list[Event],
        clock_fn: Callable[[], int],
    ) -> BargeInInjector:
        return cls(
            stream=stream,
            tts=tts,
            events=events,
            clock_fn=clock_fn,
            on_agent_turn=config.on_agent_turn,
            trigger_after_ms=config.trigger_after_agent_speaks_ms,
            caller_says=config.caller_says,
            enabled=config.enabled,
        )

    # ------------------------------------------------------------------
    # Public: frame routing (testable directly)
    # ------------------------------------------------------------------

    async def put_inbound(self, frame: AudioFrame) -> None:
        """
        Track agent-speaking bursts and forward frame to the underlying stream.
        Call this instead of stream.put_inbound when routing through the injector.
        After install(), stream.put_inbound is redirected here automatically.
        """
        now = self._clock_fn()

        if self._enabled and not self._fired:
            should_trigger = self._track_burst(now)
            if should_trigger:
                self._schedule_trigger(now)

        await self._stream._inbound.put(frame)  # write directly to the queue
        self._last_frame_at_ms = now

    async def close_inbound(self) -> None:
        """Forward the inbound-close sentinel and cancel any pending trigger."""
        await self._stream.close_inbound()
        if self._trigger_task and not self._trigger_task.done():
            self._trigger_task.cancel()

    # ------------------------------------------------------------------
    # Stream interception (used by the pipeline)
    # ------------------------------------------------------------------

    def install(self) -> None:
        """
        Redirect stream.put_inbound through this injector so any caller of
        stream.put_inbound (e.g. LocalLoopback) is transparently intercepted.
        Called once after connect() returns the stream.
        """
        injector = self

        async def _patched_put_inbound(frame: AudioFrame) -> None:
            await injector.put_inbound(frame)

        self._stream.put_inbound = _patched_put_inbound  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # Burst tracking
    # ------------------------------------------------------------------

    def _track_burst(self, now: int) -> bool:
        """
        Detect when a new agent-speaking burst begins.  Returns True when
        this call starts the trigger turn (so the caller can schedule the task
        from an async context).

        A burst begins when we were NOT in a burst, or enough ms have elapsed
        since the last frame to constitute a gap.
        """
        gap = now - self._last_frame_at_ms
        is_new_burst = not self._in_burst or (gap > _INTER_TURN_GAP_MS)

        if is_new_burst:
            self._in_burst = True
            self._agent_turn_count += 1
            if self._agent_turn_count == self._on_agent_turn:
                return True
        return False

    def _schedule_trigger(self, burst_start_ms: int) -> None:
        """Schedule the barge-in injection. Must be called from async context."""
        delay_s = self._trigger_after_ms / 1000.0
        self._trigger_task = asyncio.create_task(
            self._fire(burst_start_ms, delay_s),
            name="bargein-trigger",
        )

    async def _fire(self, burst_start_ms: int, delay_s: float) -> None:
        """Wait for delay then inject the caller utterance outbound."""
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            return

        fire_ms = self._clock_fn()
        self._fired = True

        # Log the interruption event
        self._events.append(
            Event(
                kind="interruption",
                at_ms=fire_ms,
                data={
                    "agent_turn": self._on_agent_turn,
                    "caller_says": self._caller_says,
                    "trigger_after_ms": self._trigger_after_ms,
                    "burst_start_ms": burst_start_ms,
                    "agent_was_speaking": True,
                },
            )
        )

        # Inject the caller utterance as outbound audio (TTS → transport)
        # This races with whatever the normal TTS stage is doing, which is the
        # point — the caller is interrupting.
        async for frame in self._tts.synthesize(self._caller_says):
            await self._stream.put_outbound(frame)

    async def wait_for_trigger(self) -> None:
        """Await the trigger task if it was scheduled.  No-op if not enabled."""
        if self._trigger_task:
            try:
                await self._trigger_task
            except asyncio.CancelledError:
                pass
