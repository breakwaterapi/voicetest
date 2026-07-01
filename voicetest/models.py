from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Literal


@dataclass
class Turn:
    speaker: Literal["caller", "agent"]
    text: str
    start_ms: int
    end_ms: int


@dataclass
class Event:
    kind: str
    at_ms: int
    data: dict = field(default_factory=dict)


class CallOutcome(StrEnum):
    COMPLETED = "completed"
    NO_ANSWER = "no_answer"
    VOICEMAIL = "voicemail"
    ERROR = "error"


@dataclass
class CallResult:
    scenario_id: str
    outcome: CallOutcome
    turns: list[Turn]
    events: list[Event]
    recording_path: str | None
    duration_ms: int
    error: str | None = None


@dataclass
class Report:
    call: CallResult
    passed: bool
    dimensions: dict  # name -> {score, reason}
    summary: str


@dataclass
class ProductionCall:
    """A single call ingested from production logs (not re-run — result only)."""

    id: str
    agent_id: str
    transcript: str
    outcome: str
    error: str | None
    detected_at: datetime


# ---------------------------------------------------------------------------
# Battery — a full auto-generated stress-test suite over one agent
# ---------------------------------------------------------------------------


@dataclass
class DimensionSummary:
    """Roll-up of one scoring dimension across every scenario in a battery."""

    passed: int
    failed: int
    not_applicable: int  # scenarios where the dimension wasn't tested (score None)
    critical: bool       # failed in >50% of the scenarios where it applied


@dataclass
class CriticalFailure:
    """A failure pattern that recurred across the battery — the headline finding."""

    dimension: str
    description: str
    affected_count: int
    total_applicable: int
    evidence: list[str] = field(default_factory=list)  # up to 3 transcript/reason snippets
    root_cause: str = ""  # categorized cause key, e.g. "api_timeout", "speech_recognition"
    suggested_fix: str = ""  # concrete action the developer can take


@dataclass
class NodeCoverage:
    """Roll-up of every scenario targeting one AgentModel graph node."""

    node_id: str
    label: str
    total: int
    passed: int
    failed: int


@dataclass
class BatteryReport:
    """
    The aggregate of an entire stress-test battery.

    `verdict` is one of: "production_ready" | "needs_work" | "not_ready".
    `coverage_pct` is the fraction of (group × dimension) cells that actually ran
    (i.e. were applicable), expressed 0–100.
    `graph_coverage` is populated only when the battery was generated from a
    confirmed AgentModel — one entry per graph node, with pass/fail counts from
    the scenarios tagged with that node_id. Empty for ungrounded batteries.
    """

    battery_id: str
    total: int
    passed: int
    coverage_pct: float
    verdict: str
    dimensions: dict[str, DimensionSummary]      # dimension name -> summary
    critical_failures: list[CriticalFailure]
    scenario_results: list[dict]  # [{scenario_id, group, title, passed, summary}]
    graph_coverage: list[NodeCoverage] = field(default_factory=list)
