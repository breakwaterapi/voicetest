"""
Unit / integration tests: scenario limits enforcement.

- max_turns: 2 → pipeline stops after 2 caller turns.
- max_duration_s: very short → TimeoutError is caught; outcome is COMPLETED.
"""

from __future__ import annotations

import pytest

from tests.fixtures import mini_scenario, stub_providers
from voicetest.engine.run_call import run_call
from voicetest.models import CallOutcome
from voicetest.providers.targets import LocalLoopback
from voicetest.scenarios import Limits, Persona, Scenario, ScoringConfig


def scenario_with_limits(max_turns: int, max_duration_s: int = 60) -> Scenario:
    return Scenario(
        id="test-limits",
        category="appointment_booking",
        title="Limits test scenario",
        version=1,
        persona=Persona(
            system_prompt="You are a caller.",
            goal="Book a table.",
            difficult_behaviors=["Be difficult about party size."],
        ),
        scoring=ScoringConfig(objective="Agent books a table."),
        limits=Limits(max_duration_s=max_duration_s, max_turns=max_turns),
    )


@pytest.mark.asyncio
async def test_max_turns_2_terminates_after_2_caller_turns() -> None:
    """A scenario with max_turns=2 stops after the caller has spoken twice."""
    scenario = scenario_with_limits(max_turns=2)
    providers = stub_providers(
        responses=[
            "Hi, I'd like a table please.",
            "For two people at 7pm.",
            "Next Saturday.",  # Should never be reached
            "Yes thank you.",
        ]
    )
    target = LocalLoopback()
    result = await run_call(scenario, target, providers)

    caller_turns = [t for t in result.turns if t.speaker == "caller"]
    # Caller made at most max_turns turns (opening + 1 LLM turn = 2)
    assert len(caller_turns) <= scenario.limits.max_turns
    assert result.outcome == CallOutcome.COMPLETED


@pytest.mark.asyncio
async def test_max_turns_1_terminates_immediately() -> None:
    """max_turns=1 means the caller says the opening line and that's it."""
    scenario = scenario_with_limits(max_turns=1)
    providers = stub_providers(
        responses=["Hi, I'd like to book. Thank you, goodbye!"]
    )
    target = LocalLoopback()
    result = await run_call(scenario, target, providers)

    caller_turns = [t for t in result.turns if t.speaker == "caller"]
    assert len(caller_turns) <= 1
    assert result.outcome == CallOutcome.COMPLETED


@pytest.mark.asyncio
async def test_max_duration_fires_completed_not_error() -> None:
    """A very short max_duration_s doesn't cause an ERROR outcome."""
    import asyncio

    scenario = scenario_with_limits(max_turns=50, max_duration_s=1)

    # Slow agent that will cause the timeout to fire
    async def slow_agent_fn(turns):
        await asyncio.sleep(2)
        return "Hello, how can I help?"

    class SlowLoopback:
        async def connect(self):
            from voicetest.providers.targets import AudioStream

            self._stream = AudioStream()
            import asyncio

            self._task = asyncio.create_task(self._run())
            return self._stream

        async def disconnect(self):
            if hasattr(self, "_task"):
                self._task.cancel()

        async def _run(self):
            from voicetest.providers.targets import _frame_to_text, _text_to_frame

            async for frame in self._stream.outbound_frames():
                await asyncio.sleep(5)  # deliberate delay to trigger timeout
                reply = "Hello how can I help?"
                await self._stream.put_inbound(_text_to_frame(reply))
            await self._stream.close_inbound()

    providers = stub_providers(responses=["Hi there. Goodbye!"])
    result = await run_call(scenario, SlowLoopback(), providers)

    # Timeout should be caught; outcome must not be ERROR
    assert result.outcome in (CallOutcome.COMPLETED, CallOutcome.NO_ANSWER)


@pytest.mark.asyncio
async def test_run_call_never_raises() -> None:
    """run_call must never propagate exceptions — all errors become outcomes."""

    class ExplodingTarget:
        async def connect(self):
            raise RuntimeError("catastrophic failure")

        async def disconnect(self):
            pass

    scenario = mini_scenario()
    result = await run_call(scenario, ExplodingTarget(), stub_providers())

    # Should return a result, not raise
    assert isinstance(result.outcome, CallOutcome)
    assert result.outcome in (CallOutcome.NO_ANSWER, CallOutcome.ERROR)
