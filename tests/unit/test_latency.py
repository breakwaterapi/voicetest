"""
Unit tests for voicetest.scoring.compute.latency().

Uses hand-built CallResult objects with known turn timing so that
p50/p95 values are deterministic and easy to verify.
"""

from __future__ import annotations

from voicetest.models import CallOutcome, CallResult, Event, Turn
from voicetest.scenarios.models import LatencyThresholds
from voicetest.scoring.compute import LatencyResult, latency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_call(turns: list[Turn], outcome: CallOutcome = CallOutcome.COMPLETED) -> CallResult:
    return CallResult(
        scenario_id="test",
        outcome=outcome,
        turns=turns,
        events=[],
        recording_path=None,
        duration_ms=turns[-1].end_ms if turns else 0,
    )


def caller(text: str, start: int, end: int) -> Turn:
    return Turn(speaker="caller", text=text, start_ms=start, end_ms=end)


def agent(text: str, start: int, end: int) -> Turn:
    return Turn(speaker="agent", text=text, start_ms=start, end_ms=end)


# ---------------------------------------------------------------------------
# Basic gap calculation
# ---------------------------------------------------------------------------


def test_single_turn_gap() -> None:
    call = make_call([
        caller("Hi I'd like to book", 0, 500),
        agent("Of course, for how many?", 700, 1200),
    ])
    result = latency(call)
    assert len(result.per_turn) == 1
    assert result.per_turn[0].gap_ms == 200   # 700 - 500


def test_gap_is_zero_when_agent_starts_before_caller_ends() -> None:
    # Overlapping turns (shouldn't happen in practice, but must not go negative)
    call = make_call([
        caller("Hello", 0, 800),
        agent("Hi there", 600, 1000),
    ])
    result = latency(call)
    assert result.per_turn[0].gap_ms == 0


def test_multi_turn_gaps() -> None:
    call = make_call([
        caller("Hello", 0, 400),
        agent("Hi", 600, 900),          # gap = 200
        caller("Book a table", 1000, 1500),
        agent("For how many?", 1700, 2000),  # gap = 200
        caller("Two people", 2100, 2400),
        agent("What time?", 2500, 2800),     # gap = 100
    ])
    result = latency(call)
    assert len(result.per_turn) == 3
    assert result.per_turn[0].gap_ms == 200
    assert result.per_turn[1].gap_ms == 200
    assert result.per_turn[2].gap_ms == 100


# ---------------------------------------------------------------------------
# p50 / p95
# ---------------------------------------------------------------------------


def test_p50_p95_single_sample() -> None:
    call = make_call([
        caller("Hello", 0, 100),
        agent("Hi", 400, 600),  # gap = 300
    ])
    result = latency(call)
    assert result.p50_ms == 300
    assert result.p95_ms == 300


def test_p50_p95_two_samples() -> None:
    call = make_call([
        caller("A", 0, 100),
        agent("B", 300, 400),   # gap = 200
        caller("C", 500, 600),
        agent("D", 1100, 1300), # gap = 500
    ])
    result = latency(call)
    # sorted: [200, 500]
    # p50 nearest-rank @ 50%: rank=ceil(2*0.5)=1 → 200
    # p95 nearest-rank @ 95%: rank=ceil(2*0.95)=2 → 500
    assert result.p50_ms == 200
    assert result.p95_ms == 500


def test_p50_of_three_samples() -> None:
    # gaps: 100, 200, 300 → p50 = 200
    turns = []
    pairs = [(0, 50, 150, 200), (300, 350, 550, 600), (700, 750, 1050, 1100)]
    for cs, ce, as_, ae in pairs:
        turns.append(caller("x", cs, ce))
        turns.append(agent("y", as_, ae))
    call = make_call(turns)
    result = latency(call)
    gaps = [t.gap_ms for t in result.per_turn]
    assert sorted(gaps) == [100, 200, 300]
    assert result.p50_ms == 200


def test_empty_turns_returns_zeros() -> None:
    call = make_call([])
    result = latency(call)
    assert result.p50_ms == 0
    assert result.p95_ms == 0
    assert result.per_turn == []


def test_caller_only_turns_no_samples() -> None:
    call = make_call([caller("Hello", 0, 500)])
    result = latency(call)
    assert result.per_turn == []
    assert result.p50_ms == 0


def test_agent_only_turns_no_samples() -> None:
    # Agent turn with no preceding caller turn is skipped
    call = make_call([agent("Welcome", 0, 400)])
    result = latency(call)
    assert result.per_turn == []


def test_agent_greeting_skipped() -> None:
    # First agent turn has no preceding caller → skipped; second has one
    call = make_call([
        agent("Welcome to the restaurant", 0, 500),
        caller("Hi I'd like to book", 600, 900),
        agent("Of course!", 1000, 1200),  # gap = 100
    ])
    result = latency(call)
    assert len(result.per_turn) == 1
    assert result.per_turn[0].gap_ms == 100


# ---------------------------------------------------------------------------
# Threshold warnings
# ---------------------------------------------------------------------------


def test_no_warn_below_threshold() -> None:
    call = make_call([
        caller("Hi", 0, 100),
        agent("Hello", 200, 400),  # gap = 100
    ])
    thresholds = LatencyThresholds(p50_warn_ms=800, p95_warn_ms=1500)
    result = latency(call, thresholds)
    assert result.warn is False


def test_warn_when_p50_exceeds_threshold() -> None:
    call = make_call([
        caller("Hi", 0, 100),
        agent("Hello", 1100, 1300),  # gap = 1000
    ])
    thresholds = LatencyThresholds(p50_warn_ms=800, p95_warn_ms=1500)
    result = latency(call, thresholds)
    assert result.warn is True


def test_warn_when_p95_exceeds_threshold() -> None:
    # 9 fast turns + 1 very slow: p50 ok, p95 bad
    turns: list[Turn] = []
    t = 0
    for _ in range(9):
        turns.append(caller("x", t, t + 100))
        turns.append(agent("y", t + 200, t + 400))   # gap = 100
        t += 500
    # one slow turn
    turns.append(caller("x", t, t + 100))
    turns.append(agent("y", t + 2000, t + 2200))     # gap = 1900
    t += 2500

    call = make_call(turns)
    thresholds = LatencyThresholds(p50_warn_ms=800, p95_warn_ms=1500)
    result = latency(call, thresholds)
    assert result.p95_ms > 1500
    assert result.warn is True


def test_no_warn_when_no_thresholds() -> None:
    call = make_call([
        caller("Hi", 0, 100),
        agent("Hello", 5000, 6000),  # huge gap
    ])
    result = latency(call, thresholds=None)
    assert result.warn is False


# ---------------------------------------------------------------------------
# LatencyResult type
# ---------------------------------------------------------------------------


def test_returns_latency_result_type() -> None:
    call = make_call([])
    result = latency(call)
    assert isinstance(result, LatencyResult)
