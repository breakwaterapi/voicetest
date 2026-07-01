"""
Tests for score() on non-COMPLETED outcomes (NO_ANSWER, VOICEMAIL, ERROR).

Verifies:
- passed=False for all non-COMPLETED outcomes
- Judge is NEVER called
- Report is returned (not raised)
- Summary mentions the outcome
"""

from __future__ import annotations

import pytest

from voicetest.models import CallOutcome, CallResult, Event
from voicetest.scenarios.models import Limits, Persona, Scenario, ScoringConfig
from voicetest.scoring.judge import CannedJudge, CriterionResult, JudgeResult
from voicetest.scoring.score import score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _scenario() -> Scenario:
    return Scenario(
        id="test-no-answer",
        category="appointment_booking",
        title="No-answer test",
        version=1,
        persona=Persona(
            system_prompt="You are a caller.",
            goal="Book a table.",
            difficult_behaviors=["Be unhelpful."],
        ),
        scoring=ScoringConfig(
            objective="Agent books a table.",
            pass_criteria=["Agent confirmed the booking."],
        ),
        limits=Limits(max_duration_s=60, max_turns=10),
    )


def _spy_judge() -> CannedJudge:
    """A CannedJudge that should never be called — call_count stays 0."""
    return CannedJudge(
        JudgeResult(
            goal_completed=True,
            reason="Should not be called.",
            criteria=[CriterionResult(text="anything", met=True)],
        )
    )


def _call(outcome: CallOutcome, error: str | None = None) -> CallResult:
    return CallResult(
        scenario_id="test-no-answer",
        outcome=outcome,
        turns=[],
        events=[Event(kind="connect_error", at_ms=0, data={"error": error or ""})],
        recording_path=None,
        duration_ms=0,
        error=error,
    )


# ---------------------------------------------------------------------------
# NO_ANSWER
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_answer_passed_is_false() -> None:
    scenario = _scenario()
    report = await score(_call(CallOutcome.NO_ANSWER), scenario, judge=_spy_judge())
    assert report.passed is False


@pytest.mark.asyncio
async def test_no_answer_judge_not_called() -> None:
    scenario = _scenario()
    judge = _spy_judge()
    await score(_call(CallOutcome.NO_ANSWER), scenario, judge=judge)
    assert judge.call_count == 0


@pytest.mark.asyncio
async def test_no_answer_returns_report() -> None:
    scenario = _scenario()
    report = await score(_call(CallOutcome.NO_ANSWER), scenario, judge=_spy_judge())
    assert hasattr(report, "call")
    assert hasattr(report, "passed")
    assert hasattr(report, "dimensions")
    assert hasattr(report, "summary")


@pytest.mark.asyncio
async def test_no_answer_summary_mentions_outcome() -> None:
    scenario = _scenario()
    report = await score(_call(CallOutcome.NO_ANSWER), scenario, judge=_spy_judge())
    assert "no_answer" in report.summary.lower() or "no answer" in report.summary.lower() or "not complete" in report.summary.lower()


# ---------------------------------------------------------------------------
# VOICEMAIL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voicemail_passed_is_false() -> None:
    scenario = _scenario()
    report = await score(_call(CallOutcome.VOICEMAIL), scenario, judge=_spy_judge())
    assert report.passed is False


@pytest.mark.asyncio
async def test_voicemail_judge_not_called() -> None:
    scenario = _scenario()
    judge = _spy_judge()
    await score(_call(CallOutcome.VOICEMAIL), scenario, judge=judge)
    assert judge.call_count == 0


# ---------------------------------------------------------------------------
# ERROR
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_passed_is_false() -> None:
    scenario = _scenario()
    report = await score(_call(CallOutcome.ERROR, error="network failure"), scenario, judge=_spy_judge())
    assert report.passed is False


@pytest.mark.asyncio
async def test_error_judge_not_called() -> None:
    scenario = _scenario()
    judge = _spy_judge()
    await score(_call(CallOutcome.ERROR, error="boom"), scenario, judge=judge)
    assert judge.call_count == 0


@pytest.mark.asyncio
async def test_error_does_not_raise() -> None:
    """score() never raises, even for error outcomes."""
    scenario = _scenario()
    report = await score(_call(CallOutcome.ERROR, error="catastrophic"), scenario, judge=_spy_judge())
    assert report is not None


# ---------------------------------------------------------------------------
# Non-completed outcome still has latency dimension
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_answer_has_latency_dimension() -> None:
    scenario = _scenario()
    report = await score(_call(CallOutcome.NO_ANSWER), scenario, judge=_spy_judge())
    assert "latency" in report.dimensions


@pytest.mark.asyncio
async def test_no_answer_no_goal_completion_dimension() -> None:
    """Goal-completion dimension only exists for COMPLETED calls."""
    scenario = _scenario()
    report = await score(_call(CallOutcome.NO_ANSWER), scenario, judge=_spy_judge())
    assert "goal_completion" not in report.dimensions
