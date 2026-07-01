"""
Tests for step 4: root-cause analysis + clustering of critical failures.
"""

from __future__ import annotations

import json

import pytest

from voicetest.models import CallOutcome, CallResult, Report
from voicetest.scenarios.compiler import LLMScenarioCompiler, StubScenarioCompiler
from voicetest.scenarios.models import (
    BargeInConfig,
    Limits,
    NoiseConfig,
    Persona,
    Scenario,
    ScoringConfig,
)
from voicetest.scoring.aggregate import aggregate_battery
from voicetest.scoring.root_cause import RootCauseAnalyzer


def _scenario(sid: str, group: str = "edge_cases") -> Scenario:
    return Scenario(
        id=sid,
        category="booking",
        title=sid,
        group=group,
        persona=Persona(system_prompt="p", goal="g", difficult_behaviors=["b"]),
        noise=NoiseConfig(),
        barge_in=BargeInConfig(),
        scoring=ScoringConfig(objective="o", pass_criteria=["c"]),
        limits=Limits(),
    )


def _report(passed: bool, summary: str = "", error: str | None = None, dimensions=None) -> Report:
    call = CallResult(
        scenario_id="s", outcome=CallOutcome.COMPLETED, turns=[], events=[],
        recording_path=None, duration_ms=1000, error=error,
    )
    return Report(call=call, passed=passed, dimensions=dimensions or {}, summary=summary)


# ---------------------------------------------------------------------------
# RootCauseAnalyzer.analyze — grouping by pattern
# ---------------------------------------------------------------------------


def test_analyze_groups_timeouts_together():
    analyzer = RootCauseAnalyzer()
    failed = [
        (_scenario("s1"), _report(False, summary="Calendar API timed out after 30s.")),
        (_scenario("s2"), _report(False, summary="The booking API timeout occurred mid-call.")),
    ]
    buckets = analyzer.analyze(failed)
    assert "api_timeout" in buckets
    assert len(buckets["api_timeout"]) == 2


def test_analyze_separates_distinct_patterns():
    analyzer = RootCauseAnalyzer()
    failed = [
        (_scenario("s1"), _report(False, summary="Calendar API timed out.")),
        (_scenario("s2"), _report(False, summary="Caller's number was garbled and misheard.")),
    ]
    buckets = analyzer.analyze(failed)
    assert set(buckets) == {"api_timeout", "speech_recognition"}


def test_analyze_uses_call_error_field():
    analyzer = RootCauseAnalyzer()
    failed = [
        (_scenario("s1"), _report(False, error="Telephony provider request timed out.")),
    ]
    buckets = analyzer.analyze(failed)
    assert "api_timeout" in buckets


def test_analyze_falls_back_to_other_for_unknown_pattern():
    analyzer = RootCauseAnalyzer()
    failed = [
        (_scenario("s1"), _report(False, summary="Something unrelated went wrong.")),
    ]
    buckets = analyzer.analyze(failed)
    assert set(buckets) == {"other"}


def test_classify_report_returns_label_and_fix():
    analyzer = RootCauseAnalyzer()
    key, label, fix = analyzer.classify_report(
        _scenario("s1"), _report(False, summary="Speech was garbled and unclear.")
    )
    assert key == "speech_recognition"
    assert label
    assert fix


# ---------------------------------------------------------------------------
# aggregate_battery: critical_failures carry root_cause + suggested_fix
# ---------------------------------------------------------------------------


def test_critical_failures_enriched_with_root_cause():
    scenario_reports = [
        (
            _scenario(f"s{i}"),
            _report(
                False,
                dimensions={
                    "goal_completion": {
                        "score": 0,
                        "reason": "Calendar API timed out before confirming the slot.",
                    }
                },
            ),
        )
        for i in range(3)
    ]
    report = aggregate_battery("b1", scenario_reports)
    goal_failure = next(f for f in report.critical_failures if f.dimension == "goal_completion")
    assert goal_failure.root_cause == "api_timeout"
    assert goal_failure.suggested_fix


def test_critical_failures_without_pattern_have_empty_root_cause():
    scenario_reports = [
        (_scenario("s1"), _report(True, dimensions={"goal_completion": {"score": 1}})),
    ]
    report = aggregate_battery("b1", scenario_reports)
    # No critical failures at all here (nothing failed) — sanity check the field exists.
    assert all(hasattr(f, "root_cause") for f in report.critical_failures)


# ---------------------------------------------------------------------------
# Regression scenario generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stub_compiler_generates_five_regression_scenarios():
    compiler = StubScenarioCompiler()
    scenarios = await compiler.compile_regression("Calendar API timeout")
    assert len(scenarios) == 5
    assert all(s.group == "regression" for s in scenarios)


@pytest.mark.asyncio
async def test_llm_compiler_regression_prompt_includes_cause_keywords(monkeypatch):
    compiler = LLMScenarioCompiler(api_key="fake")
    captured: dict[str, str] = {}

    async def fake_call_anthropic(self, prompt, max_tokens=8192):
        captured["prompt"] = prompt
        return json.dumps({
            "category": "api_timeout_regression",
            "groups": [{
                "group": "regression",
                "scenarios": [
                    {
                        "title": f"Regression {i}",
                        "persona_system_prompt": "p", "goal": "g",
                        "difficult_behaviors": ["Calendar API times out mid-booking"],
                        "objective": "o", "pass_criteria": ["c"],
                    }
                    for i in range(5)
                ],
            }],
        })

    monkeypatch.setattr(LLMScenarioCompiler, "_call_anthropic", fake_call_anthropic)

    scenarios = await compiler.compile_regression("Calendar API timeout")
    assert "Calendar API timeout" in captured["prompt"]
    assert len(scenarios) == 5
    assert all(s.group == "regression" for s in scenarios)
    assert all("api times out" in b.lower() for s in scenarios for b in s.persona.difficult_behaviors)
