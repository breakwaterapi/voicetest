"""
Computed metrics — no LLM required.

Covers:
  - latency(): per-turn response latency (p50/p95)
  - bargein_handling(): analyse the interruption window from events+transcript

All times are in milliseconds, sourced from Turn.start_ms / Turn.end_ms
set by the pipeline clock (no wall-clock drift between turns).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from voicetest.models import CallResult, Event, Turn
from voicetest.scenarios.models import LatencyThresholds

# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


@dataclass
class TurnLatency:
    agent_turn_index: int    # index into call.turns
    agent_text: str
    gap_ms: int              # caller_end_ms → agent_start_ms


@dataclass
class LatencyResult:
    per_turn: list[TurnLatency] = field(default_factory=list)
    p50_ms: int = 0
    p95_ms: int = 0
    warn: bool = False       # True if p50 or p95 exceeds thresholds


def _percentile(values: list[int], pct: float) -> int:
    """Nearest-rank percentile, returns 0 for empty input."""
    if not values:
        return 0
    sorted_vals = sorted(values)
    # nearest-rank formula (1-based)
    rank = max(1, int(len(sorted_vals) * pct / 100 + 0.5))
    return sorted_vals[min(rank, len(sorted_vals)) - 1]


def latency(call: CallResult, thresholds: LatencyThresholds | None = None) -> LatencyResult:
    """
    Compute per-turn response latency from a CallResult.

    A latency sample is produced for each (caller, agent) adjacent pair
    in call.turns.  If no caller turn precedes an agent turn the sample
    is skipped (e.g. an unsolicited agent greeting).
    """
    samples: list[TurnLatency] = []
    turns = call.turns

    for i, turn in enumerate(turns):
        if turn.speaker != "agent":
            continue
        # Find the most-recent preceding caller turn
        prev_caller: Turn | None = None
        for j in range(i - 1, -1, -1):
            if turns[j].speaker == "caller":
                prev_caller = turns[j]
                break

        if prev_caller is None:
            continue

        gap = max(0, turn.start_ms - prev_caller.end_ms)
        samples.append(TurnLatency(agent_turn_index=i, agent_text=turn.text, gap_ms=gap))

    gaps = [s.gap_ms for s in samples]
    p50 = _percentile(gaps, 50)
    p95 = _percentile(gaps, 95)

    warn = False
    if thresholds is not None:
        warn = p50 > thresholds.p50_warn_ms or p95 > thresholds.p95_warn_ms

    return LatencyResult(per_turn=samples, p50_ms=p50, p95_ms=p95, warn=warn)


# ---------------------------------------------------------------------------
# Barge-in handling analysis
# ---------------------------------------------------------------------------

# Agent turns that overlap with the interruption window are counted as
# "kept talking" — a fail signal.  We use a generous window: any agent turn
# that STARTS within this many ms after the interruption event fires counts as
# overlapping (the agent text was already being synthesised at trigger time).
_BARGEIN_OVERLAP_WINDOW_MS = 500


@dataclass
class BargeInResult:
    """Computed analysis of how the agent handled a barge-in."""

    present: bool = False          # barge_in.enabled AND an interruption event fired
    agent_stopped: bool = True     # False = agent kept talking past the interruption
    context_recovered: bool = True # True = first post-interruption agent turn is topical
    post_interruption_latency_ms: int = 0  # gap between interruption fire and next agent turn
    overlap_turns: list[str] = field(default_factory=list)  # agent texts that overlapped
    passed: bool = True
    reason: str = "No barge-in configured"


def bargein_handling(call: CallResult) -> BargeInResult:
    """
    Analyse the interruption window in a completed call.

    Heuristics (ADR-7 § barge-in scoring):
    1. agent_stopped=False when any agent turn starts within
       _BARGEIN_OVERLAP_WINDOW_MS of the interruption event (the agent was
       synthesising speech it could not retract).
    2. context_recovered=True when the first agent turn AFTER the interruption
       window does NOT begin with "I'm sorry, could you repeat" or similar
       restart phrases — a proxy for context retention.
    3. post_interruption_latency_ms = start_ms of first post-interruption agent
       turn minus interruption event at_ms.
    """
    interruption_events = [e for e in call.events if e.kind == "interruption"]

    if not interruption_events:
        return BargeInResult(present=False, passed=True, reason="No barge-in event fired")

    # Use the first interruption event (fires at most once per call)
    ev: Event = interruption_events[0]
    fire_ms: int = ev.at_ms

    overlap_turns: list[str] = []
    first_post: Turn | None = None

    for turn in call.turns:
        if turn.speaker != "agent":
            continue
        if turn.start_ms <= fire_ms + _BARGEIN_OVERLAP_WINDOW_MS:
            # Agent turn started during or just before the interruption window
            if turn.start_ms >= fire_ms:
                overlap_turns.append(turn.text)
        else:
            # First agent turn clearly after the interruption
            if first_post is None:
                first_post = turn

    agent_stopped = len(overlap_turns) == 0

    # Context recovery heuristic: did the agent restart vs pick up where it was?
    _RESTART_PHRASES = (
        "sorry, could you repeat",
        "could you repeat that",
        "i didn't catch",
        "i'm sorry, i didn't",
        "pardon",
        "let me start over",
        "as i was saying",
    )
    context_recovered = True
    if first_post is not None:
        lower = first_post.text.lower()
        context_recovered = not any(p in lower for p in _RESTART_PHRASES)

    post_latency = 0
    if first_post is not None:
        post_latency = max(0, first_post.start_ms - fire_ms)

    passed = agent_stopped and context_recovered

    if not agent_stopped:
        reason = (
            f"Agent kept talking after interruption ({len(overlap_turns)} overlap turn(s)). "
            "Expected agent to stop and yield the floor."
        )
    elif not context_recovered:
        reason = (
            "Agent stopped but restarted from scratch instead of recovering context. "
            f"First post-interruption turn: {first_post.text[:80]!r}"
        )
    else:
        reason = (
            "Agent stopped cleanly and recovered context. "
            f"Post-interruption latency: {post_latency}ms"
        )

    return BargeInResult(
        present=True,
        agent_stopped=agent_stopped,
        context_recovered=context_recovered,
        post_interruption_latency_ms=post_latency,
        overlap_turns=overlap_turns,
        passed=passed,
        reason=reason,
    )
