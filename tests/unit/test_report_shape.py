"""
P1-T4 — Report shape tests.

Verifies that score() returns a Report with all 5 dimensions, that the
tool_call_accuracy dimension is marked deferred (not a fail), and that
the failure breakdown links recording path and turn index.

Uses CannedJudge — no LLM calls.
"""

from __future__ import annotations

import pytest

from voicetest.models import CallOutcome, CallResult, Event, Turn
from voicetest.scenarios import BUILTIN_DIR, load_scenario
from voicetest.scenarios.models import BargeInConfig, Limits, NoiseConfig
from voicetest.scoring.judge import CannedJudge, CriterionResult, JudgeResult, NoiseResilienceResult
from voicetest.scoring.score import _failure_breakdown, score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RECORDING = "/tmp/fake-recording.wav"
_FIVE_DIMS = {
    "goal_completion", "latency", "barge_in_handling", "noise_resilience", "tool_call_accuracy"
}


def _base_scenario():
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    return scenario.model_copy(update={"limits": Limits(max_turns=2, max_duration_s=30)})


def _completed_call(events: list[Event] | None = None) -> CallResult:
    return CallResult(
        scenario_id="test",
        outcome=CallOutcome.COMPLETED,
        turns=[
            Turn(speaker="agent", text="Hello, how can I help?", start_ms=0, end_ms=400),
            Turn(speaker="caller", text="I'd like to book an appointment.",
                 start_ms=500, end_ms=900),
            Turn(speaker="agent", text="Sure, what date works for you?",
                 start_ms=1000, end_ms=1400),
        ],
        events=events or [],
        recording_path=_RECORDING,
        duration_ms=2000,
    )


def _all_pass_judge() -> CannedJudge:
    return CannedJudge(
        JudgeResult(
            goal_completed=True,
            reason="All criteria met.",
            criteria=[CriterionResult(text="Booked appointment", met=True, evidence="confirmed")],
            noise_resilience=None,
        )
    )


def _all_fail_judge() -> CannedJudge:
    return CannedJudge(
        JudgeResult(
            goal_completed=False,
            reason="Goal not met.",
            criteria=[CriterionResult(text="Booked appointment", met=False, evidence="not found")],
            noise_resilience=None,
        )
    )


# ---------------------------------------------------------------------------
# All 5 dimensions present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_has_all_five_dimensions() -> None:
    scenario = _base_scenario()
    report = await score(_completed_call(), scenario, _all_pass_judge())
    assert set(report.dimensions.keys()) == _FIVE_DIMS


@pytest.mark.asyncio
async def test_all_five_dims_present_on_fail() -> None:
    scenario = _base_scenario()
    report = await score(_completed_call(), scenario, _all_fail_judge())
    assert set(report.dimensions.keys()) == _FIVE_DIMS


@pytest.mark.asyncio
async def test_non_completed_still_has_computed_dims() -> None:
    """Non-completed calls skip the judge but still include computed dimensions."""
    scenario = _base_scenario()
    call = CallResult(
        scenario_id="test",
        outcome=CallOutcome.ERROR,
        turns=[],
        events=[],
        recording_path=None,
        duration_ms=0,
        error="Connection reset",
    )
    report = await score(call, scenario, _all_pass_judge())
    # goal_completion is omitted (no judge call for non-completed)
    assert "latency" in report.dimensions
    assert "barge_in_handling" in report.dimensions
    assert "noise_resilience" in report.dimensions
    assert "tool_call_accuracy" in report.dimensions
    assert report.passed is False


# ---------------------------------------------------------------------------
# tool_call_accuracy is always deferred (not a fail)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_dim_is_deferred() -> None:
    scenario = _base_scenario()
    report = await score(_completed_call(), scenario, _all_pass_judge())
    dim = report.dimensions["tool_call_accuracy"]
    assert dim["deferred"] is True


@pytest.mark.asyncio
async def test_tool_call_dim_score_is_none() -> None:
    scenario = _base_scenario()
    report = await score(_completed_call(), scenario, _all_pass_judge())
    assert report.dimensions["tool_call_accuracy"]["score"] is None


@pytest.mark.asyncio
async def test_tool_call_deferred_does_not_fail_overall() -> None:
    """A deferred tool_call dimension must never cause overall failure."""
    scenario = _base_scenario()
    report = await score(_completed_call(), scenario, _all_pass_judge())
    # Passed because goal met + no barge-in + noise disabled + tool_call deferred
    assert report.passed is True


@pytest.mark.asyncio
async def test_tool_call_dim_reason_mentions_phase_2() -> None:
    scenario = _base_scenario()
    report = await score(_completed_call(), scenario, _all_pass_judge())
    reason = report.dimensions["tool_call_accuracy"]["reason"].lower()
    assert "phase 2" in reason or "deferred" in reason


# ---------------------------------------------------------------------------
# Failure breakdown links recording + turn index
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_breakdown_links_recording_path() -> None:
    from voicetest.scoring.compute import bargein_handling

    bi = bargein_handling(_completed_call())
    judge_result = JudgeResult(
        goal_completed=False,
        reason="Not met",
        criteria=[CriterionResult(text="Book appt", met=False, evidence="missing")],
    )
    failures = _failure_breakdown(_completed_call(), judge_result, bi)
    assert len(failures) > 0
    assert failures[0]["recording_path"] == _RECORDING


def test_failure_breakdown_has_dimension_field() -> None:
    from voicetest.scoring.compute import bargein_handling

    bi = bargein_handling(_completed_call())
    judge_result = JudgeResult(
        goal_completed=False,
        reason="Not met",
        criteria=[CriterionResult(text="Book appt", met=False, evidence="missing")],
    )
    failures = _failure_breakdown(_completed_call(), judge_result, bi)
    for f in failures:
        assert "dimension" in f
        assert "criterion" in f
        assert "evidence" in f
        assert "recording_path" in f
        assert "turn_index" in f


def test_barge_in_failure_links_turn_index() -> None:
    """When agent kept talking, failure links to the overlapping turn index."""
    from voicetest.scoring.compute import BargeInResult

    overlap_text = "Continuing despite interruption."
    call = _completed_call(
        events=[
            Event(
                kind="interruption",
                at_ms=500,
                data={"agent_turn": 1, "caller_says": "Wait", "agent_was_speaking": True,
                      "burst_start_ms": 300, "trigger_after_ms": 200},
            )
        ]
    )
    # Override turns so one overlaps
    call = CallResult(
        scenario_id=call.scenario_id,
        outcome=call.outcome,
        turns=[
            Turn(speaker="agent", text=overlap_text, start_ms=550, end_ms=900),
        ],
        events=call.events,
        recording_path=_RECORDING,
        duration_ms=call.duration_ms,
    )
    bi = BargeInResult(
        present=True,
        agent_stopped=False,
        context_recovered=True,
        post_interruption_latency_ms=0,
        overlap_turns=[overlap_text],
        passed=False,
        reason="Agent kept talking",
    )
    failures = _failure_breakdown(call, None, bi)
    barge_failures = [f for f in failures if f["dimension"] == "barge_in_handling"]
    assert barge_failures
    assert barge_failures[0]["turn_index"] == 0


# ---------------------------------------------------------------------------
# Report fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_has_call_field() -> None:
    scenario = _base_scenario()
    call = _completed_call()
    report = await score(call, scenario, _all_pass_judge())
    assert report.call is call


@pytest.mark.asyncio
async def test_report_passed_true_when_all_ok() -> None:
    scenario = _base_scenario()
    report = await score(_completed_call(), scenario, _all_pass_judge())
    assert report.passed is True


@pytest.mark.asyncio
async def test_report_passed_false_when_goal_failed() -> None:
    scenario = _base_scenario()
    report = await score(_completed_call(), scenario, _all_fail_judge())
    assert report.passed is False


@pytest.mark.asyncio
async def test_report_summary_is_nonempty_string() -> None:
    scenario = _base_scenario()
    report = await score(_completed_call(), scenario, _all_pass_judge())
    assert isinstance(report.summary, str)
    assert len(report.summary) > 0


@pytest.mark.asyncio
async def test_report_summary_contains_verdict() -> None:
    scenario = _base_scenario()
    report = await score(_completed_call(), scenario, _all_pass_judge())
    assert "PASSED" in report.summary or "FAILED" in report.summary


# ---------------------------------------------------------------------------
# Barge-in + noise active combination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_with_noise_and_bargein_has_all_dims() -> None:
    """Scenario with both noise and barge_in active still produces all 5 dims."""
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-double-books.yaml")
    scenario = scenario.model_copy(
        update={
            "limits": Limits(max_turns=2, max_duration_s=30),
            "noise": NoiseConfig(enabled=True, profile="cafe", level_db=-12),
            "barge_in": BargeInConfig(
                enabled=True,
                on_agent_turn=1,
                trigger_after_agent_speaks_ms=0,
                caller_says="Wait.",
            ),
        }
    )
    judge = CannedJudge(
        JudgeResult(
            goal_completed=True,
            reason="All ok",
            criteria=[CriterionResult(text="Crit", met=True)],
            noise_resilience=NoiseResilienceResult(passed=True, missed_values=[], reason="ok"),
        )
    )
    call = _completed_call()
    report = await score(call, scenario, judge)
    assert set(report.dimensions.keys()) == _FIVE_DIMS
