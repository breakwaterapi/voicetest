"""
P1-T4 — Noise resilience scoring tests (canned judge).

Tests that the noise_resilience dimension is populated correctly based on
the judge result.  Uses CannedJudge — no LLM calls.

Scenarios:
- noise active + judge says missed key number → noise dim fails, overall fails
- noise active + judge says all values captured → noise dim passes
- noise disabled → noise dim is inactive (score=None, no pass/fail)
- noise active but no ground_truth → noise dim not evaluated (score=None)
"""

from __future__ import annotations

import pytest

from voicetest.models import CallOutcome, CallResult, Turn
from voicetest.scenarios import BUILTIN_DIR, load_scenario
from voicetest.scenarios.models import Limits, NoiseConfig
from voicetest.scoring.judge import CannedJudge, CriterionResult, JudgeResult, NoiseResilienceResult
from voicetest.scoring.score import score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCENARIO_ID = "test-scenario"


def _completed_call(turns: list[Turn] | None = None) -> CallResult:
    return CallResult(
        scenario_id=_SCENARIO_ID,
        outcome=CallOutcome.COMPLETED,
        turns=turns or [
            Turn(speaker="agent", text="Hello, what is your phone number?", start_ms=0, end_ms=500),
            Turn(speaker="caller", text="It's 617-555-0182.", start_ms=600, end_ms=1000),
            Turn(speaker="agent", text="Got it, 617-555-0182. Confirmed.",
                 start_ms=1100, end_ms=1500),
        ],
        events=[],
        recording_path="/fake/recording.wav",
        duration_ms=2000,
    )


def _noise_scenario(enabled: bool = True):
    """Load a noise scenario and force max_turns=2 for speed."""
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-mumbled-phone.yaml")
    return scenario.model_copy(
        update={
            "limits": Limits(max_turns=2, max_duration_s=30),
            "noise": NoiseConfig(enabled=enabled, profile="cafe", level_db=-12),
        }
    )


def _canned_judge(goal_completed: bool, noise_passed: bool, missed: list[str]) -> CannedJudge:
    return CannedJudge(
        JudgeResult(
            goal_completed=goal_completed,
            reason="Canned result",
            criteria=[CriterionResult(text="Criterion 1", met=goal_completed)],
            noise_resilience=NoiseResilienceResult(
                passed=noise_passed,
                missed_values=missed,
                reason="Noise evaluation reason",
            ),
        )
    )


# ---------------------------------------------------------------------------
# Noise active + missed values → fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noise_dim_fails_when_judge_says_missed() -> None:
    scenario = _noise_scenario(enabled=True)
    judge = _canned_judge(goal_completed=True, noise_passed=False, missed=["617-555-0182"])
    call = _completed_call()

    report = await score(call, scenario, judge)
    dim = report.dimensions["noise_resilience"]
    assert dim["score"] == 0
    assert dim["passed"] is False


@pytest.mark.asyncio
async def test_noise_dim_lists_missed_values() -> None:
    scenario = _noise_scenario(enabled=True)
    judge = _canned_judge(goal_completed=True, noise_passed=False, missed=["617-555-0182"])
    call = _completed_call()

    report = await score(call, scenario, judge)
    dim = report.dimensions["noise_resilience"]
    assert "617-555-0182" in dim["missed_values"]


@pytest.mark.asyncio
async def test_overall_fails_when_noise_dim_fails() -> None:
    scenario = _noise_scenario(enabled=True)
    judge = _canned_judge(goal_completed=True, noise_passed=False, missed=["key-number"])
    call = _completed_call()

    report = await score(call, scenario, judge)
    assert report.passed is False


# ---------------------------------------------------------------------------
# Noise active + all values captured → passes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noise_dim_passes_when_all_captured() -> None:
    scenario = _noise_scenario(enabled=True)
    judge = _canned_judge(goal_completed=True, noise_passed=True, missed=[])
    call = _completed_call()

    report = await score(call, scenario, judge)
    dim = report.dimensions["noise_resilience"]
    assert dim["score"] == 1
    assert dim["passed"] is True


@pytest.mark.asyncio
async def test_overall_passes_when_noise_captured_and_goal_met() -> None:
    scenario = _noise_scenario(enabled=True)
    judge = _canned_judge(goal_completed=True, noise_passed=True, missed=[])
    call = _completed_call()

    report = await score(call, scenario, judge)
    assert report.passed is True


@pytest.mark.asyncio
async def test_noise_active_dim_has_active_true() -> None:
    scenario = _noise_scenario(enabled=True)
    judge = _canned_judge(goal_completed=True, noise_passed=True, missed=[])
    call = _completed_call()

    report = await score(call, scenario, judge)
    assert report.dimensions["noise_resilience"]["active"] is True


# ---------------------------------------------------------------------------
# Noise disabled → dimension inactive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noise_dim_inactive_when_noise_disabled() -> None:
    scenario = _noise_scenario(enabled=False)
    # Judge returns no noise_resilience (noise not active, not passed in)
    judge = CannedJudge(
        JudgeResult(
            goal_completed=True,
            reason="Canned",
            criteria=[CriterionResult(text="Criterion 1", met=True)],
            noise_resilience=None,
        )
    )
    call = _completed_call()

    report = await score(call, scenario, judge)
    dim = report.dimensions["noise_resilience"]
    assert dim["active"] is False
    assert dim["score"] is None


@pytest.mark.asyncio
async def test_noise_disabled_dim_does_not_fail_overall() -> None:
    """Inactive noise dim must not penalise overall pass/fail."""
    scenario = _noise_scenario(enabled=False)
    judge = CannedJudge(
        JudgeResult(
            goal_completed=True,
            reason="Canned",
            criteria=[CriterionResult(text="Criterion 1", met=True)],
            noise_resilience=None,
        )
    )
    call = _completed_call()

    report = await score(call, scenario, judge)
    assert report.passed is True


# ---------------------------------------------------------------------------
# Single judge call (ADR-7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exactly_one_judge_call_when_noise_active() -> None:
    scenario = _noise_scenario(enabled=True)
    judge = _canned_judge(goal_completed=True, noise_passed=True, missed=[])
    call = _completed_call()

    await score(call, scenario, judge)
    assert judge.call_count == 1


@pytest.mark.asyncio
async def test_exactly_one_judge_call_when_noise_disabled() -> None:
    scenario = _noise_scenario(enabled=False)
    judge = CannedJudge(
        JudgeResult(
            goal_completed=True,
            reason="Canned",
            criteria=[],
            noise_resilience=None,
        )
    )
    call = _completed_call()

    await score(call, scenario, judge)
    assert judge.call_count == 1
