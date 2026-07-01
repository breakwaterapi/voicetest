"""
P1-T4 — Barge-in handling scoring tests.

Uses bargein_handling() from compute.py against synthetic CallResult fixtures.
No LLM calls.

Scenarios:
- agent kept talking past the interruption → barge_in_handling fails
- agent stopped cleanly + context recovered → passes
- agent stopped but restarted with a "could you repeat" → context not recovered
- no interruption event → dimension is present=False, passed=True
"""

from __future__ import annotations

from voicetest.models import CallOutcome, CallResult, Event, Turn
from voicetest.scoring.compute import BargeInResult, bargein_handling

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SCENARIO_ID = "test-scenario"


def _make_call(turns: list[Turn], events: list[Event]) -> CallResult:
    return CallResult(
        scenario_id=_SCENARIO_ID,
        outcome=CallOutcome.COMPLETED,
        turns=turns,
        events=events,
        recording_path=None,
        duration_ms=5000,
    )


def _interruption_event(at_ms: int = 1000) -> Event:
    return Event(
        kind="interruption",
        at_ms=at_ms,
        data={
            "agent_turn": 1,
            "caller_says": "Wait, hold on.",
            "agent_was_speaking": True,
            "burst_start_ms": at_ms - 200,
            "trigger_after_ms": 200,
        },
    )


# ---------------------------------------------------------------------------
# No barge-in: no interruption events
# ---------------------------------------------------------------------------


def test_no_interruption_events_returns_not_present() -> None:
    call = _make_call(
        turns=[
            Turn(speaker="agent", text="Hello, how can I help you?", start_ms=0, end_ms=500),
            Turn(speaker="caller", text="I need to book an appointment.",
                 start_ms=600, end_ms=1000),
        ],
        events=[],
    )
    result = bargein_handling(call)
    assert result.present is False


def test_no_interruption_passes() -> None:
    call = _make_call(turns=[], events=[])
    result = bargein_handling(call)
    assert result.passed is True


def test_no_interruption_reason_mentions_no_event() -> None:
    call = _make_call(turns=[], events=[])
    result = bargein_handling(call)
    assert "no barge-in" in result.reason.lower() or "event" in result.reason.lower()


# ---------------------------------------------------------------------------
# Agent kept talking (fail case)
# ---------------------------------------------------------------------------


def test_agent_kept_talking_fails() -> None:
    fire_ms = 1000
    call = _make_call(
        turns=[
            # Agent turn that started AFTER fire_ms but within overlap window
            Turn(
                speaker="agent",
                text="As I was mentioning, the appointment is on Tuesday.",
                start_ms=fire_ms + 50,
                end_ms=fire_ms + 800,
            ),
            Turn(speaker="caller", text="Wait, hold on.",
                 start_ms=fire_ms + 100, end_ms=fire_ms + 400),
        ],
        events=[_interruption_event(at_ms=fire_ms)],
    )
    result = bargein_handling(call)
    assert result.agent_stopped is False
    assert result.passed is False


def test_agent_kept_talking_overlap_turns_populated() -> None:
    fire_ms = 1000
    overlap_text = "As I was saying, Tuesday works for us."
    call = _make_call(
        turns=[
            Turn(
                speaker="agent",
                text=overlap_text,
                start_ms=fire_ms + 100,
                end_ms=fire_ms + 600,
            ),
        ],
        events=[_interruption_event(at_ms=fire_ms)],
    )
    result = bargein_handling(call)
    assert overlap_text in result.overlap_turns


def test_agent_kept_talking_reason_mentions_overlap() -> None:
    fire_ms = 1000
    call = _make_call(
        turns=[
            Turn(
                speaker="agent",
                text="Continuing my explanation despite the interruption.",
                start_ms=fire_ms + 200,
                end_ms=fire_ms + 700,
            ),
        ],
        events=[_interruption_event(at_ms=fire_ms)],
    )
    result = bargein_handling(call)
    assert "kept talking" in result.reason.lower() or "overlap" in result.reason.lower()


# ---------------------------------------------------------------------------
# Agent stopped cleanly (pass case)
# ---------------------------------------------------------------------------


def test_agent_stopped_cleanly_passes() -> None:
    fire_ms = 1000
    call = _make_call(
        turns=[
            # Agent turn starts well AFTER the overlap window
            Turn(
                speaker="agent",
                text="Of course! How can I help you with that?",
                start_ms=fire_ms + 600,
                end_ms=fire_ms + 1200,
            ),
        ],
        events=[_interruption_event(at_ms=fire_ms)],
    )
    result = bargein_handling(call)
    assert result.agent_stopped is True
    assert result.passed is True


def test_agent_stopped_cleanly_no_overlap_turns() -> None:
    fire_ms = 1000
    call = _make_call(
        turns=[
            Turn(
                speaker="agent",
                text="Sure, what would you like to change?",
                start_ms=fire_ms + 700,
                end_ms=fire_ms + 1200,
            ),
        ],
        events=[_interruption_event(at_ms=fire_ms)],
    )
    result = bargein_handling(call)
    assert result.overlap_turns == []


def test_post_interruption_latency_computed() -> None:
    fire_ms = 1000
    agent_start = fire_ms + 800
    call = _make_call(
        turns=[
            Turn(
                speaker="agent",
                text="Absolutely, let me update that for you.",
                start_ms=agent_start,
                end_ms=agent_start + 500,
            ),
        ],
        events=[_interruption_event(at_ms=fire_ms)],
    )
    result = bargein_handling(call)
    assert result.post_interruption_latency_ms == 800


# ---------------------------------------------------------------------------
# Context recovery heuristic
# ---------------------------------------------------------------------------


def test_restart_phrase_context_not_recovered() -> None:
    fire_ms = 1000
    call = _make_call(
        turns=[
            Turn(
                speaker="agent",
                text="Sorry, could you repeat that? I didn't catch what you said.",
                start_ms=fire_ms + 700,
                end_ms=fire_ms + 1200,
            ),
        ],
        events=[_interruption_event(at_ms=fire_ms)],
    )
    result = bargein_handling(call)
    assert result.context_recovered is False
    assert result.passed is False


def test_topical_response_context_recovered() -> None:
    fire_ms = 1000
    call = _make_call(
        turns=[
            Turn(
                speaker="agent",
                text="No problem! Thursday at 2pm is available. Shall I book that?",
                start_ms=fire_ms + 700,
                end_ms=fire_ms + 1300,
            ),
        ],
        events=[_interruption_event(at_ms=fire_ms)],
    )
    result = bargein_handling(call)
    assert result.context_recovered is True


def test_context_not_recovered_fails() -> None:
    fire_ms = 1000
    call = _make_call(
        turns=[
            Turn(
                speaker="agent",
                text="Pardon? Let me start over. How can I help you?",
                start_ms=fire_ms + 700,
                end_ms=fire_ms + 1200,
            ),
        ],
        events=[_interruption_event(at_ms=fire_ms)],
    )
    result = bargein_handling(call)
    assert result.passed is False


# ---------------------------------------------------------------------------
# BargeInResult shape
# ---------------------------------------------------------------------------


def test_result_is_bargeinresult_instance() -> None:
    result = bargein_handling(_make_call(turns=[], events=[]))
    assert isinstance(result, BargeInResult)


def test_result_has_all_fields() -> None:
    result = bargein_handling(_make_call(turns=[], events=[]))
    assert hasattr(result, "present")
    assert hasattr(result, "agent_stopped")
    assert hasattr(result, "context_recovered")
    assert hasattr(result, "post_interruption_latency_ms")
    assert hasattr(result, "overlap_turns")
    assert hasattr(result, "passed")
    assert hasattr(result, "reason")
