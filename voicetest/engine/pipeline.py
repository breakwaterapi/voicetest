"""
Pipecat-based caller pipeline.

Architecture (from architecture.md):
    Transport → STTStage → Persona LLM → TTSStage → Transport

Stages are now proper Protocols (voicetest.engine.stages) so they're
independently swappable:
  - FinalOnlySTTStage / StreamingSTTStage
  - BufferedTTSStage  / StreamingTTSStage

The Providers dataclass carries stt_stage and tts_stage alongside the
existing provider interfaces so build_pipeline can choose the right pair
per environment (streaming for real calls, buffered/final-only for tests).

Per-stage timing is collected in a shared `timing` dict and emitted as a
"stage_timing" event at call end so the score layer can pinpoint which
stage causes latency spikes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from voicetest.engine.processors.bargein import BargeInInjector
from voicetest.engine.processors.noise import NoiseMixer
from voicetest.engine.stages import (
    BufferedTTSStage,
    FinalOnlySTTStage,
    STTStage,
    StreamingSTTStage,
    StreamingTTSStage,
    TTSStage,
)
from voicetest.log import vlog
from voicetest.models import Event, Turn
from voicetest.providers.llm import LLMProvider, StubLLM
from voicetest.providers.stt import STTProvider, StubSTT
from voicetest.providers.targets import AudioStream
from voicetest.providers.tts import StubTTS, TTSProvider
from voicetest.scenarios import Persona, Scenario

# ---------------------------------------------------------------------------
# Providers bundle
# ---------------------------------------------------------------------------


@dataclass
class Providers:
    stt: STTProvider = field(default_factory=StubSTT)
    tts: TTSProvider = field(default_factory=StubTTS)
    llm: LLMProvider = field(default_factory=StubLLM)
    # Stage implementations — None means "pick the default for this provider type"
    stt_stage: STTStage | None = None
    tts_stage: TTSStage | None = None


def default_providers() -> Providers:
    return Providers()


def real_providers(
    stt: STTProvider,
    tts: TTSProvider,
    llm: LLMProvider,
) -> Providers:
    """Providers bundle for a real call — uses streaming stages."""
    return Providers(
        stt=stt,
        tts=tts,
        llm=llm,
        stt_stage=StreamingSTTStage(),
        tts_stage=StreamingTTSStage(),
    )


# ---------------------------------------------------------------------------
# Persona prompt assembly
# ---------------------------------------------------------------------------


def assemble_system_prompt(persona: Persona) -> str:
    """
    Combine persona.system_prompt + goal + difficult_behaviors into the
    caller LLM system prompt. Kept tight for latency (ADR-5).
    """
    behaviors = "\n".join(f"- {b}" for b in persona.difficult_behaviors)
    return (
        f"{persona.system_prompt.strip()}\n\n"
        f"YOUR GOAL: {persona.goal}\n\n"
        "IMPORTANT — you MUST exhibit ALL of these behaviors during the call:\n"
        f"{behaviors}\n\n"
        "Rules:\n"
        "- You are the CALLER, not an assistant. Never be helpful or cooperative.\n"
        "- Keep replies short (1-3 sentences max). Real phone calls are terse.\n"
        "- Do NOT break character under any circumstances.\n"
        "- When your goal is achieved or the call naturally ends, say goodbye."
    )


def assemble_opening_line(persona: Persona) -> str:
    """
    Generate a natural opening line from the persona goal.

    Rather than prepending "Hello, <goal literal>" (which no human says),
    we ask the LLM to generate a natural first sentence. For stub/loopback
    mode we use a sensible fallback so no LLM call is needed at startup.

    The pipeline calls this synchronously at construction time to avoid
    blocking the asyncio loop during WS handshake.
    """
    goal = persona.goal.strip().rstrip(".")
    # Extract the core action from the goal for a natural opening
    # e.g. "Book a haircut for next Saturday afternoon" → "I'd like to book a haircut"
    goal_lower = goal.lower()
    if goal_lower.startswith("book") or goal_lower.startswith("schedule"):
        verb = goal_lower.split()[0]
        rest = goal[len(verb):].strip()
        return f"Hi, I'd like to {verb}{' ' + rest if rest else ''}."
    if goal_lower.startswith("cancel"):
        return f"Hi, I need to {goal.lower()}."
    if goal_lower.startswith("check") or goal_lower.startswith("find out"):
        return f"Hi, I was hoping to {goal.lower()}."
    # Generic fallback — still better than "Hello, <goal>"
    return f"Hi, {goal}."


# ---------------------------------------------------------------------------
# Lightweight monotonic clock
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        import time
        self._start = time.monotonic()

    def now(self) -> int:
        import time
        return int((time.monotonic() - self._start) * 1000)


# ---------------------------------------------------------------------------
# LLM stage coroutine
# ---------------------------------------------------------------------------


async def _llm_stage(
    transcript_queue: asyncio.Queue[str],
    tts_queue: asyncio.Queue[str | None],
    stream: AudioStream,
    system_prompt: str,
    llm: LLMProvider,
    turns: list[Turn],
    events: list[Event],
    clock: _Clock,
    max_turns: int,
    timing: dict[str, Any],
) -> None:
    """Read agent transcripts → respond via LLM → push TTS text chunks."""
    messages: list[dict] = []

    async def _end_call() -> None:
        await tts_queue.put(None)
        await stream.close_inbound()

    while True:
        raw = await transcript_queue.get()

        # Interim transcript (prefixed with \x00 by StreamingSTTStage) — skip for now.
        # Future: use to pre-warm KV cache or start token streaming.
        if raw.startswith("\x00"):
            vlog("pipe", f"interim transcript (skipped): {raw[1:]!r}")
            continue

        if raw == "":
            vlog("pipe", "inbound (agent) stream ended — closing conversation")
            await tts_queue.put(None)
            return

        vlog("pipe", f"agent said: {raw!r}")
        agent_rx_ms = clock.now()
        timing.setdefault("agent_rx_ms", []).append(agent_rx_ms)

        turns.append(Turn(speaker="agent", text=raw, start_ms=agent_rx_ms, end_ms=agent_rx_ms))
        messages.append({"role": "user", "content": raw})

        if len([t for t in turns if t.speaker == "caller"]) >= max_turns:
            await _end_call()
            return

        caller_start = clock.now()
        timing.setdefault("llm_start_ms", []).append(caller_start)

        chunks: list[str] = []
        first_token_recorded = False
        async for chunk in llm.respond(messages, system_prompt):
            if not first_token_recorded:
                timing.setdefault("llm_first_token_ms", []).append(clock.now() - caller_start)
                first_token_recorded = True
            chunks.append(chunk)
            await tts_queue.put(chunk)

        caller_text = "".join(chunks)
        caller_end = clock.now()
        timing.setdefault("llm_done_ms", []).append(caller_end - caller_start)

        turns.append(
            Turn(speaker="caller", text=caller_text, start_ms=caller_start, end_ms=caller_end)
        )
        messages.append({"role": "assistant", "content": caller_text})

        vlog("pipe", f"caller replied: {caller_text!r}")
        events.append(Event(kind="caller_turn", at_ms=caller_start, data={"text": caller_text}))

        lower = caller_text.lower()
        if any(w in lower for w in ("goodbye", "bye", "thanks, goodbye", "thank you, goodbye")):
            await _end_call()
            return


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class CallerPipeline:
    """Wired pipeline ready to run. Call .run(stream) to execute one conversation."""

    scenario: Scenario
    providers: Providers
    turns: list[Turn] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    _clock: _Clock = field(default_factory=_Clock)

    async def run(self, stream: AudioStream) -> None:
        """
        Drive the full STTStage→LLM→TTSStage loop until:
        - the inbound (agent) stream closes, OR
        - max_turns is hit, OR
        - the caller says goodbye.
        """
        system_prompt = assemble_system_prompt(self.scenario.persona)
        max_turns = self.scenario.limits.max_turns

        transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Shared timing dict — each stage appends entries keyed by stage name.
        # Emitted as a "stage_timing" event at call end for the score layer.
        timing: dict[str, Any] = {}

        self.events.append(Event(kind="call_start", at_ms=self._clock.now()))

        mixer = NoiseMixer.from_config(
            self.scenario.noise, stream, self.events, self._clock.now
        )
        await mixer.start()

        bargein = BargeInInjector.from_config(
            self.scenario.barge_in,
            stream,
            self.providers.tts,
            self.events,
            self._clock.now,
        )
        bargein.install()

        # Pick stage implementations — fall back to safe defaults
        stt_stage: STTStage = self.providers.stt_stage or FinalOnlySTTStage()
        tts_stage: TTSStage = self.providers.tts_stage or BufferedTTSStage()

        # Synthesize the caller's opening line BEFORE starting the gather loop
        # so the WS connection is already live and the pipeline is running before
        # we block on TTS. Opening is derived from the persona goal naturally.
        opening = assemble_opening_line(self.scenario.persona)
        self.turns.append(
            Turn(speaker="caller", text=opening, start_ms=0, end_ms=self._clock.now())
        )
        vlog("pipe", f"synthesizing caller opening line: {opening!r}")
        n_frames = 0
        try:
            async for frame in self.providers.tts.synthesize(opening):
                n_frames += 1
                await mixer.put_outbound(frame)
        except Exception as exc:  # noqa: BLE001
            vlog("pipe", f"!!! TTS synth of opening line FAILED: {type(exc).__name__}: {exc}")
            raise
        vlog("pipe", f"opening line synthesized into {n_frames} audio frame(s)")

        try:
            await asyncio.gather(
                stt_stage.run(stream, self.providers.stt, transcript_queue, timing),
                _llm_stage(
                    transcript_queue,
                    tts_queue,
                    stream,
                    system_prompt,
                    self.providers.llm,
                    self.turns,
                    self.events,
                    self._clock,
                    max_turns,
                    timing,
                ),
                tts_stage.run(tts_queue, stream, self.providers.tts, mixer, timing),
            )
        finally:
            await mixer.stop()
            await bargein.wait_for_trigger()

        self.events.append(Event(kind="call_end", at_ms=self._clock.now()))
        # Emit all collected timing data for the score layer to pick up
        if timing:
            self.events.append(Event(kind="stage_timing", at_ms=self._clock.now(), data=timing))
        vlog("pipe", f"conversation ended ({len(self.turns)} turns total)")


def build_pipeline(
    scenario: Scenario,
    providers: Providers | None = None,
) -> CallerPipeline:
    """Factory used by run_call. Returns a fresh CallerPipeline each call."""
    return CallerPipeline(
        scenario=scenario,
        providers=providers or default_providers(),
    )
