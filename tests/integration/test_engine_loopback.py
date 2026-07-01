"""
Integration test: run_call against LocalLoopback with stub providers.

No network.  No real audio.  Asserts a multi-turn CallResult with alternating
caller/agent turns and outcome=COMPLETED.
"""

from __future__ import annotations

import pytest

from tests.fixtures import mini_scenario, stub_providers
from voicetest.engine.run_call import run_call
from voicetest.models import CallOutcome
from voicetest.providers.targets import LocalLoopback


@pytest.mark.asyncio
async def test_loopback_multi_turn_call() -> None:
    """Full STT→LLM→TTS loop through LocalLoopback; result has alternating turns."""
    scenario = mini_scenario(max_turns=6)
    providers = stub_providers(
        responses=[
            "Hi, I'd like to book a table please.",
            "For two people, this Saturday evening.",
            "Actually next Saturday. 7pm works.",
            "Yes, that's great. Thank you, goodbye!",
        ]
    )
    target = LocalLoopback()

    result = await run_call(scenario, target, providers)

    assert result.outcome == CallOutcome.COMPLETED
    assert result.scenario_id == scenario.id
    assert result.recording_path is None  # P0-T5 fills this

    # Must have at least one caller and one agent turn
    speakers = [t.speaker for t in result.turns]
    assert "caller" in speakers
    assert "agent" in speakers

    # Turns must alternate (caller starts, agent responds, …)
    assert result.turns[0].speaker == "caller"
    for i in range(1, len(result.turns)):
        assert result.turns[i].speaker != result.turns[i - 1].speaker, (
            f"turn {i} has same speaker as turn {i-1}: {result.turns[i].speaker}"
        )

    # Events must include call_start and call_end
    event_kinds = {e.kind for e in result.events}
    assert "call_start" in event_kinds
    assert "call_end" in event_kinds

    # Duration is non-negative
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_loopback_no_error_on_connect_failure() -> None:
    """run_call never raises — a connect failure becomes CallOutcome.NO_ANSWER."""

    class FailingTarget:
        async def connect(self):
            raise ConnectionError("simulated connect failure")

        async def disconnect(self):
            pass

    scenario = mini_scenario()
    result = await run_call(scenario, FailingTarget(), stub_providers())

    assert result.outcome == CallOutcome.NO_ANSWER
    assert result.error is not None
    assert result.turns == []


@pytest.mark.asyncio
async def test_loopback_with_injected_agent_fn() -> None:
    """LocalLoopback accepts an injected agent_fn; pipeline handles it gracefully."""
    from voicetest.providers.targets import LocalLoopback

    call_count = 0

    def counting_agent(turns):
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            return "Great, all booked. Is there anything else? Goodbye!"
        return "Of course, for how many people and what time?"

    scenario = mini_scenario(max_turns=4)
    providers = stub_providers(
        responses=[
            "Hi, I'd like to book a table for two please.",
            "7pm next Saturday.",
            "No, thank you. Goodbye!",
        ]
    )
    target = LocalLoopback(agent_fn=counting_agent)

    result = await run_call(scenario, target, providers)

    assert result.outcome == CallOutcome.COMPLETED
    assert len(result.turns) >= 2
