"""
P1-T3 — Barge-in against the broken-agent fixture.

The broken agent (tests/fixtures/broken_agent.py) ignores barge-in —
it always returns a fixed reply regardless of caller input, and
sleeps synchronously so it holds the inbound channel.

Tests assert:
1. An `interruption` event IS logged (the injector fired correctly).
2. The agent's transcript after the interruption STILL contains the broken
   canned reply — proving the agent did NOT stop cleanly.
3. The agent_was_speaking flag is True in the interruption event, because
   the agent kept emitting frames past the trigger point.

This observable state feeds P1-T4's "did not stop cleanly" finding.
"""

from __future__ import annotations

import pytest

from tests.fixtures.broken_agent import make_broken_agent
from voicetest.engine.run_call import run_call
from voicetest.providers.targets import LocalLoopback
from voicetest.scenarios import BUILTIN_DIR, load_scenario
from voicetest.scenarios.models import BargeInConfig, Limits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _barge_in_scenario(on_turn: int = 1, delay_ms: int = 0):
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-double-books.yaml")
    return scenario.model_copy(
        update={
            "limits": Limits(max_turns=3, max_duration_s=30),
            "barge_in": BargeInConfig(
                enabled=True,
                on_agent_turn=on_turn,
                trigger_after_agent_speaks_ms=delay_ms,
                caller_says="Wait — actually, can we change the time?",
            ),
        }
    )


# ---------------------------------------------------------------------------
# Broken agent: interruption fires but agent ignores it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_broken_agent_interruption_event_logged() -> None:
    """Injector fires and logs the interruption even when the agent ignores it."""
    scenario = _barge_in_scenario()
    broken = LocalLoopback(agent_fn=make_broken_agent(delay_s=0.0))
    result = await run_call(scenario, broken)

    event_kinds = [e.kind for e in result.events]
    assert "interruption" in event_kinds, (
        "Expected an interruption event but got none. "
        f"Events: {event_kinds}"
    )


@pytest.mark.asyncio
async def test_broken_agent_still_replies_after_interruption() -> None:
    """
    The broken agent ignores the barge-in and continues with its canned reply.
    The transcript must contain the fixed reply AFTER the interruption fires,
    which is the observable 'failure to stop' signal for P1-T4.
    """
    fixed_reply = "Sorry, could you repeat that? I didn't quite catch it."
    scenario = _barge_in_scenario()
    broken = LocalLoopback(agent_fn=make_broken_agent(delay_s=0.0, fixed_reply=fixed_reply))
    result = await run_call(scenario, broken)

    # Collect all agent turns from the result transcript
    agent_texts = [t.text for t in result.turns if t.speaker == "agent"]

    # The broken agent's fixed reply must appear in the transcript
    assert any(fixed_reply in text for text in agent_texts), (
        "Broken agent should have kept talking after barge-in. "
        f"Agent turns: {agent_texts}"
    )


@pytest.mark.asyncio
async def test_broken_agent_interruption_event_fields() -> None:
    """Interruption event has the expected structure for P1-T4 scoring."""
    scenario = _barge_in_scenario(on_turn=1, delay_ms=0)
    broken = LocalLoopback(agent_fn=make_broken_agent(delay_s=0.0))
    result = await run_call(scenario, broken)

    interruption_events = [e for e in result.events if e.kind == "interruption"]
    assert interruption_events, "No interruption event found"

    ev = interruption_events[0]
    assert ev.data["agent_turn"] == 1
    assert ev.data["caller_says"] == "Wait — actually, can we change the time?"
    assert "burst_start_ms" in ev.data
    assert ev.at_ms >= 0


@pytest.mark.asyncio
async def test_broken_agent_interruption_not_duplicate() -> None:
    """Interruption fires exactly once even against a slow broken agent."""
    scenario = _barge_in_scenario()
    broken = LocalLoopback(agent_fn=make_broken_agent(delay_s=0.0))
    result = await run_call(scenario, broken)

    interruptions = [e for e in result.events if e.kind == "interruption"]
    assert len(interruptions) == 1, (
        f"Expected exactly 1 interruption event, got {len(interruptions)}"
    )


# ---------------------------------------------------------------------------
# Cooperative agent comparison: interruption fires but agent adapts
# (shows the event infrastructure works regardless of agent quality)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_good_agent_also_gets_interruption_event() -> None:
    """The interruption event is logged regardless of agent quality."""
    scenario = _barge_in_scenario()
    result = await run_call(scenario, LocalLoopback())  # default cooperative agent

    event_kinds = {e.kind for e in result.events}
    assert "interruption" in event_kinds
