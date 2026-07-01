"""
score() — combine judge + computed metrics into a 5-dimension Report.

Dimensions
----------
1. goal_completion  — judge (one LLM call, ADR-7)
2. latency          — computed; warns but does NOT fail
3. barge_in_handling — computed from events; fails if agent kept talking
4. noise_resilience — judge (folded into the single LLM call when noise active)
5. tool_call_accuracy — deferred (Phase 2); always marked "needs API access"

Rules
-----
- Non-COMPLETED outcomes → Report.passed=False, no judge call.
- passed = goal_completed AND every pass_criterion.met AND barge_in passed
           AND noise_resilience passed (when active).
- Latency warns but does NOT fail.
- Exactly one judge call per COMPLETED call.
"""

from __future__ import annotations

from voicetest.models import CallOutcome, CallResult, Report
from voicetest.scenarios.models import Scenario
from voicetest.scoring.compute import BargeInResult, LatencyResult, bargein_handling, latency
from voicetest.scoring.judge import (
    JudgeProvider,
    JudgeResult,
    NoiseResilienceResult,
    _format_transcript,
    make_judge,
)

# ---------------------------------------------------------------------------
# Dimension builders
# ---------------------------------------------------------------------------


def _latency_dimension(lat: LatencyResult) -> dict:
    return {
        "score": None,
        "p50_ms": lat.p50_ms,
        "p95_ms": lat.p95_ms,
        "warn": lat.warn,
        "per_turn": [
            {"agent_turn_index": t.agent_turn_index, "gap_ms": t.gap_ms}
            for t in lat.per_turn
        ],
        "reason": (
            f"p50={lat.p50_ms}ms  p95={lat.p95_ms}ms"
            + (" ⚠ above threshold" if lat.warn else "")
        ),
    }


def _goal_dimension(judge_result: JudgeResult) -> dict:
    return {
        "score": 1 if judge_result.goal_completed else 0,
        "goal_completed": judge_result.goal_completed,
        "reason": judge_result.reason,
        "criteria": [
            {"text": c.text, "met": c.met, "evidence": c.evidence}
            for c in judge_result.criteria
        ],
    }


def _bargein_dimension(bi: BargeInResult) -> dict:
    return {
        "score": 1 if bi.passed else 0,
        "present": bi.present,
        "agent_stopped": bi.agent_stopped,
        "context_recovered": bi.context_recovered,
        "post_interruption_latency_ms": bi.post_interruption_latency_ms,
        "overlap_turns": bi.overlap_turns,
        "passed": bi.passed,
        "reason": bi.reason,
    }


def _noise_dimension(nr: NoiseResilienceResult | None, noise_active: bool) -> dict:
    if not noise_active:
        return {
            "score": None,
            "active": False,
            "reason": "Noise not active for this scenario",
        }
    if nr is None:
        return {
            "score": None,
            "active": True,
            "reason": "No ground truth configured — noise resilience not evaluated",
        }
    return {
        "score": 1 if nr.passed else 0,
        "active": True,
        "passed": nr.passed,
        "missed_values": nr.missed_values,
        "reason": nr.reason,
    }


def _tool_call_dimension() -> dict:
    return {
        "score": None,
        "deferred": True,
        "reason": "Tool-call accuracy requires API access — deferred to Phase 2",
    }


# ---------------------------------------------------------------------------
# Failure breakdown helpers
# ---------------------------------------------------------------------------


def _failure_breakdown(
    call: CallResult,
    judge_result: JudgeResult | None,
    bi: BargeInResult,
) -> list[dict]:
    """
    Return one entry per failure, each linking to recording path + transcript turn index.
    """
    failures: list[dict] = []

    if judge_result is not None:
        for i, c in enumerate(judge_result.criteria):
            if not c.met:
                failures.append(
                    {
                        "dimension": "goal_completion",
                        "criterion": c.text,
                        "evidence": c.evidence,
                        "recording_path": call.recording_path,
                        "turn_index": None,
                    }
                )

    if bi.present and not bi.agent_stopped:
        # Find the first overlapping agent turn index
        for i, turn in enumerate(call.turns):
            if turn.speaker == "agent" and turn.text in bi.overlap_turns:
                failures.append(
                    {
                        "dimension": "barge_in_handling",
                        "criterion": "Agent should stop when caller interrupts",
                        "evidence": turn.text[:120],
                        "recording_path": call.recording_path,
                        "turn_index": i,
                    }
                )
                break

    if bi.present and not bi.context_recovered:
        failures.append(
            {
                "dimension": "barge_in_handling",
                "criterion": "Agent should recover context after interruption",
                "evidence": bi.reason,
                "recording_path": call.recording_path,
                "turn_index": None,
            }
        )

    return failures


def _most_common_failure(failures: list[dict]) -> str:
    if not failures:
        return "No failures."
    dims = [f["dimension"] for f in failures]
    most_common = max(set(dims), key=dims.count)
    count = dims.count(most_common)
    return f"Most common failure: {most_common} ({count} failure(s))."


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def _build_summary(
    call: CallResult,
    judge_result: JudgeResult | None,
    lat: LatencyResult,
    bi: BargeInResult,
    passed: bool,
    failures: list[dict],
) -> str:
    if call.outcome != CallOutcome.COMPLETED:
        return f"Call did not complete (outcome={call.outcome}). Not scored."

    verdict = "PASSED" if passed else "FAILED"
    latency_note = f"p50={lat.p50_ms}ms p95={lat.p95_ms}ms" + (
        " ⚠ latency warn" if lat.warn else ""
    )

    lines = [f"{verdict} — {judge_result.reason if judge_result else ''}"]

    if judge_result:
        for c in judge_result.criteria:
            lines.append(f"  {'✓' if c.met else '✗'} {c.text}")

    if bi.present:
        lines.append(f"  {'✓' if bi.passed else '✗'} Barge-in handling: {bi.reason}")

    lines.append(f"  Latency: {latency_note}")
    lines.append(_most_common_failure(failures))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def score(
    call: CallResult,
    scenario: Scenario,
    judge: JudgeProvider | None = None,
) -> Report:
    """
    Score a CallResult against its scenario.

    Parameters
    ----------
    call     : CallResult from run_call.
    scenario : The Scenario the call was run against.
    judge    : JudgeProvider to use.  Defaults to make_judge() (real LLM).
               Pass a CannedJudge in tests.

    Returns
    -------
    Report with 5 dimensions: goal_completion, latency, barge_in_handling,
    noise_resilience, tool_call_accuracy.
    """
    lat = latency(call, scenario.scoring.latency)
    bi = bargein_handling(call)

    # Non-completed calls: no judge, auto-fail
    if call.outcome != CallOutcome.COMPLETED:
        return Report(
            call=call,
            passed=False,
            dimensions={
                "latency": _latency_dimension(lat),
                "barge_in_handling": _bargein_dimension(bi),
                "noise_resilience": _noise_dimension(None, scenario.noise.enabled),
                "tool_call_accuracy": _tool_call_dimension(),
            },
            summary=f"Call did not complete (outcome={call.outcome}). Not scored.",
        )

    # Resolve noise ground truth for the single judge call
    noise_active = scenario.noise.enabled
    noise_ground_truth = scenario.persona.ground_truth if noise_active else None
    # Only pass ground truth if it is non-empty
    if noise_ground_truth is not None and not noise_ground_truth:
        noise_ground_truth = None

    if judge is None:
        judge = make_judge()

    transcript = _format_transcript(call.turns)
    # One judge call — folds in noise resilience when active (ADR-7)
    judge_result = await judge.score(transcript, scenario.scoring, noise_ground_truth)

    all_criteria_met = all(c.met for c in judge_result.criteria)
    noise_ok = (
        judge_result.noise_resilience is None
        or judge_result.noise_resilience.passed
    )
    passed = judge_result.goal_completed and all_criteria_met and bi.passed and noise_ok

    failures = _failure_breakdown(call, judge_result, bi)

    dimensions = {
        "goal_completion": _goal_dimension(judge_result),
        "latency": _latency_dimension(lat),
        "barge_in_handling": _bargein_dimension(bi),
        "noise_resilience": _noise_dimension(judge_result.noise_resilience, noise_active),
        "tool_call_accuracy": _tool_call_dimension(),
    }

    summary = _build_summary(call, judge_result, lat, bi, passed, failures)

    return Report(
        call=call,
        passed=passed,
        dimensions=dimensions,
        summary=summary,
    )
