"""
Tests for score() using CannedJudge — no network.

Verifies:
- Report shape (all fields present)
- passed=True when goal_completed and all criteria met
- passed=False when goal_completed but some criteria not met
- passed=False when goal not completed
- Exactly one judge call per completed call
- Latency dimension is always present
- Judge is NOT called for non-COMPLETED outcomes
"""

from __future__ import annotations

import pytest

from voicetest.models import CallOutcome, CallResult, Event, Turn
from voicetest.scenarios.models import Limits, Persona, Scenario, ScoringConfig
from voicetest.scoring.judge import CannedJudge, CriterionResult, JudgeResult
from voicetest.scoring.score import score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_scenario(pass_criteria: list[str] | None = None) -> Scenario:
    return Scenario(
        id="test-scenario",
        category="appointment_booking",
        title="Test",
        version=1,
        persona=Persona(
            system_prompt="You are a caller.",
            goal="Book a table.",
            difficult_behaviors=["Be difficult."],
        ),
        scoring=ScoringConfig(
            objective="Agent books a table.",
            pass_criteria=pass_criteria or [
                "Agent confirmed the date.",
                "Agent confirmed the party size.",
            ],
        ),
        limits=Limits(max_duration_s=60, max_turns=10),
    )


def make_completed_call(turns: list[Turn] | None = None) -> CallResult:
    turns = turns or [
        Turn(speaker="caller", text="Hi I'd like to book", start_ms=0, end_ms=400),
        Turn(speaker="agent", text="Of course, when?", start_ms=600, end_ms=900),
        Turn(speaker="caller", text="Next Saturday at 7pm for two", start_ms=1000, end_ms=1500),
        Turn(speaker="agent", text="Booked! Is there anything else?", start_ms=1700, end_ms=2100),
    ]
    return CallResult(
        scenario_id="test-scenario",
        outcome=CallOutcome.COMPLETED,
        turns=turns,
        events=[Event(kind="call_end", at_ms=2100)],
        recording_path=None,
        duration_ms=2100,
    )


def all_met_judge(scenario: Scenario) -> CannedJudge:
    return CannedJudge(JudgeResult(
        goal_completed=True,
        reason="The agent booked the table correctly.",
        criteria=[CriterionResult(text=c, met=True, evidence="confirmed") for c in scenario.scoring.pass_criteria],
    ))


def partial_judge(scenario: Scenario) -> CannedJudge:
    """First criterion met, rest not."""
    criteria = [
        CriterionResult(text=c, met=(i == 0), evidence="found" if i == 0 else "not found")
        for i, c in enumerate(scenario.scoring.pass_criteria)
    ]
    return CannedJudge(JudgeResult(goal_completed=True, reason="Partial.", criteria=criteria))


def goal_not_met_judge(scenario: Scenario) -> CannedJudge:
    return CannedJudge(JudgeResult(
        goal_completed=False,
        reason="The agent failed to complete the booking.",
        criteria=[CriterionResult(text=c, met=False, evidence="not found") for c in scenario.scoring.pass_criteria],
    ))


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_has_required_fields() -> None:
    scenario = make_scenario()
    call = make_completed_call()
    report = await score(call, scenario, judge=all_met_judge(scenario))

    assert hasattr(report, "call")
    assert hasattr(report, "passed")
    assert hasattr(report, "dimensions")
    assert hasattr(report, "summary")


@pytest.mark.asyncio
async def test_report_call_is_original() -> None:
    scenario = make_scenario()
    call = make_completed_call()
    report = await score(call, scenario, judge=all_met_judge(scenario))
    assert report.call is call


@pytest.mark.asyncio
async def test_goal_completion_dimension_present() -> None:
    scenario = make_scenario()
    report = await score(make_completed_call(), scenario, judge=all_met_judge(scenario))
    assert "goal_completion" in report.dimensions


@pytest.mark.asyncio
async def test_latency_dimension_present() -> None:
    scenario = make_scenario()
    report = await score(make_completed_call(), scenario, judge=all_met_judge(scenario))
    assert "latency" in report.dimensions


@pytest.mark.asyncio
async def test_goal_dimension_has_expected_keys() -> None:
    scenario = make_scenario()
    report = await score(make_completed_call(), scenario, judge=all_met_judge(scenario))
    dim = report.dimensions["goal_completion"]
    assert "goal_completed" in dim
    assert "reason" in dim
    assert "criteria" in dim


@pytest.mark.asyncio
async def test_latency_dimension_has_expected_keys() -> None:
    scenario = make_scenario()
    report = await score(make_completed_call(), scenario, judge=all_met_judge(scenario))
    dim = report.dimensions["latency"]
    assert "p50_ms" in dim
    assert "p95_ms" in dim
    assert "warn" in dim


# ---------------------------------------------------------------------------
# passed logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_passed_when_goal_and_all_criteria_met() -> None:
    scenario = make_scenario()
    report = await score(make_completed_call(), scenario, judge=all_met_judge(scenario))
    assert report.passed is True


@pytest.mark.asyncio
async def test_failed_when_some_criteria_not_met() -> None:
    scenario = make_scenario()
    report = await score(make_completed_call(), scenario, judge=partial_judge(scenario))
    assert report.passed is False


@pytest.mark.asyncio
async def test_failed_when_goal_not_completed() -> None:
    scenario = make_scenario()
    report = await score(make_completed_call(), scenario, judge=goal_not_met_judge(scenario))
    assert report.passed is False


@pytest.mark.asyncio
async def test_failed_when_goal_met_but_no_criteria() -> None:
    # Edge case: goal_completed=True but criteria list is empty → passes
    # (no criteria to fail)
    scenario = make_scenario(pass_criteria=[])
    judge = CannedJudge(JudgeResult(goal_completed=True, reason="Done.", criteria=[]))
    report = await score(make_completed_call(), scenario, judge=judge)
    assert report.passed is True


# ---------------------------------------------------------------------------
# Exactly one judge call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exactly_one_judge_call_for_completed() -> None:
    scenario = make_scenario()
    judge = all_met_judge(scenario)
    await score(make_completed_call(), scenario, judge=judge)
    assert judge.call_count == 1


@pytest.mark.asyncio
async def test_judge_not_called_for_no_answer() -> None:
    scenario = make_scenario()
    judge = all_met_judge(scenario)
    call = CallResult(
        scenario_id="test-scenario",
        outcome=CallOutcome.NO_ANSWER,
        turns=[],
        events=[],
        recording_path=None,
        duration_ms=0,
    )
    await score(call, scenario, judge=judge)
    assert judge.call_count == 0


@pytest.mark.asyncio
async def test_judge_not_called_for_error() -> None:
    scenario = make_scenario()
    judge = all_met_judge(scenario)
    call = CallResult(
        scenario_id="test-scenario",
        outcome=CallOutcome.ERROR,
        turns=[],
        events=[],
        recording_path=None,
        duration_ms=0,
        error="boom",
    )
    await score(call, scenario, judge=judge)
    assert judge.call_count == 0


# ---------------------------------------------------------------------------
# Summary field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_is_nonempty_string() -> None:
    scenario = make_scenario()
    report = await score(make_completed_call(), scenario, judge=all_met_judge(scenario))
    assert isinstance(report.summary, str)
    assert len(report.summary) > 0


@pytest.mark.asyncio
async def test_summary_contains_pass_or_fail() -> None:
    scenario = make_scenario()
    passed_report = await score(make_completed_call(), scenario, judge=all_met_judge(scenario))
    failed_report = await score(make_completed_call(), scenario, judge=goal_not_met_judge(scenario))

    assert "PASSED" in passed_report.summary
    assert "FAILED" in failed_report.summary


# ---------------------------------------------------------------------------
# Criteria forwarded correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_criteria_count_matches_scenario() -> None:
    criteria_texts = ["Criterion A.", "Criterion B.", "Criterion C."]
    scenario = make_scenario(pass_criteria=criteria_texts)
    judge = all_met_judge(scenario)
    report = await score(make_completed_call(), scenario, judge=judge)
    goal_dim = report.dimensions["goal_completion"]
    assert len(goal_dim["criteria"]) == len(criteria_texts)
