"""
run_call — stateless entry point for a single synthetic call.

Contract (from P0-T3 spec):
- Never raises on call failure — maps errors to CallOutcome.
- Enforces scenario.limits (max_duration_s, max_turns via pipeline).
- Returns a partial CallResult; recorder fills recording_path in P0-T5.
"""

from __future__ import annotations

import asyncio

from voicetest.engine.pipeline import CallerPipeline, Providers, build_pipeline
from voicetest.log import vlog
from voicetest.models import CallOutcome, CallResult, Event
from voicetest.providers.targets import CallTarget
from voicetest.scenarios import Scenario


async def run_call(
    scenario: Scenario,
    target: CallTarget,
    providers: Providers | None = None,
) -> CallResult:
    """
    Run one complete synthetic call.

    Parameters
    ----------
    scenario : Scenario
        Fully validated scenario (persona + limits + scoring rubric).
    target : CallTarget
        Where to send the call — LocalLoopback, TranscriptReplay, or TwilioOutbound.
    providers : Providers | None
        STT / TTS / LLM bundle.  Defaults to stub providers (suitable for tests).

    Returns
    -------
    CallResult
        Partial result: turns + events + outcome + duration.
        recording_path is None (filled by recorder in P0-T5).
    """
    pipeline = build_pipeline(scenario, providers)

    try:
        stream = await target.connect()
    except Exception as exc:  # noqa: BLE001
        # A target may raise OutcomeError to declare a specific first-class
        # outcome (no-answer, voicemail) that is NOT a test failure. Anything
        # else at connect time is treated as a failure to reach the agent.
        from voicetest.providers.twilio_target import OutcomeError

        if isinstance(exc, OutcomeError):
            outcome = exc.outcome
        else:
            outcome = CallOutcome.NO_ANSWER
        vlog("call", f"connect() failed → outcome={outcome}: {type(exc).__name__}: {exc}")
        return CallResult(
            scenario_id=scenario.id,
            outcome=outcome,
            turns=[],
            events=[Event(kind="connect_error", at_ms=0, data={"error": str(exc)})],
            recording_path=None,
            duration_ms=0,
            error=str(exc),
        )

    vlog("call", f"connected — running pipeline (max_duration={scenario.limits.max_duration_s}s)")
    try:
        await asyncio.wait_for(
            pipeline.run(stream),
            timeout=scenario.limits.max_duration_s,
        )
        outcome = CallOutcome.COMPLETED
        error = None
        vlog("call", "pipeline finished normally")
    except asyncio.TimeoutError:
        outcome = CallOutcome.COMPLETED  # Timed out = call ran to limit, counts as completed
        error = None
        vlog("call", "pipeline hit max_duration limit (counts as completed)")
        pipeline.events.append(
            Event(kind="max_duration_hit", at_ms=int(scenario.limits.max_duration_s * 1000))
        )
    except Exception as exc:  # noqa: BLE001
        outcome = CallOutcome.ERROR
        error = str(exc)
        vlog("call", f"!!! pipeline raised → ERROR: {type(exc).__name__}: {exc}")
        pipeline.events.append(Event(kind="error", at_ms=0, data={"error": str(exc)}))
    finally:
        await target.disconnect()

    # Derive duration from last event timestamp
    duration_ms = pipeline.events[-1].at_ms if pipeline.events else 0

    return CallResult(
        scenario_id=scenario.id,
        outcome=outcome,
        turns=list(pipeline.turns),
        events=list(pipeline.events),
        recording_path=None,
        duration_ms=duration_ms,
        error=error,
    )
