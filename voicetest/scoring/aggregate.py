"""
aggregate_battery() — roll up many scored Reports into one BatteryReport.

A *battery* is a full auto-generated stress-test suite over a single agent:
~20 scenarios spread across behavioural groups (baseline, confused_caller,
impatient_caller, edge_cases, audio_stress, adversarial, regression).

This module is pure — no I/O, no LLM, no DB. It takes (Scenario, Report) pairs
and produces the headline view: a per-dimension pass/fail roll-up, the recurring
critical failures, a coverage score, and an overall verdict.

Dimension applicability
-----------------------
A dimension "applies" to a scenario when its report dimension has a non-None
score. Latency never has a 0/1 score (it only warns), so it is treated
specially: it applies whenever the call completed, and counts as failed when
`warn` is true.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from voicetest.models import (
    BatteryReport,
    CriticalFailure,
    DimensionSummary,
    NodeCoverage,
    Report,
)
from voicetest.scenarios.models import Scenario
from voicetest.scoring.root_cause import RootCauseAnalyzer

if TYPE_CHECKING:
    from voicetest.agent_model.models import AgentModel

# Dimensions we roll up, in display order.
_SCORED_DIMENSIONS = ["goal_completion", "barge_in_handling", "noise_resilience"]

_DIMENSION_LABELS = {
    "goal_completion": "Goal completion",
    "barge_in_handling": "Barge-in handling",
    "noise_resilience": "Noise resilience",
    "latency": "Latency",
}


def _dimension_applies(dim: dict | None) -> bool:
    """A scored dimension applies when it ran and produced a 0/1 score."""
    return bool(dim) and dim.get("score") is not None


def _summarize_dimension(name: str, reports: list[Report]) -> DimensionSummary:
    passed = failed = na = 0
    for r in reports:
        dim = r.dimensions.get(name)

        if name == "latency":
            # Latency applies whenever the dimension exists with measured percentiles.
            if not dim or dim.get("p50_ms") is None:
                na += 1
            elif dim.get("warn"):
                failed += 1
            else:
                passed += 1
            continue

        if not _dimension_applies(dim):
            na += 1
        elif dim.get("score") == 1:
            passed += 1
        else:
            failed += 1

    applicable = passed + failed
    critical = applicable > 0 and (failed / applicable) > 0.5
    return DimensionSummary(
        passed=passed, failed=failed, not_applicable=na, critical=critical
    )


def _dimension_failure_reason(name: str, report: Report) -> str | None:
    """Pull a one-line reason snippet from a failing dimension, if any."""
    dim = report.dimensions.get(name)
    if not dim:
        return None
    if name == "latency":
        if dim.get("warn"):
            return dim.get("reason") or "Latency above threshold"
        return None
    if dim.get("score") == 0:
        return dim.get("reason") or f"{name} failed"
    return None


def _build_critical_failures(
    summaries: dict[str, DimensionSummary],
    scenario_reports: list[tuple[Scenario, Report]],
) -> list[CriticalFailure]:
    analyzer = RootCauseAnalyzer()
    failures: list[CriticalFailure] = []
    for name, summ in summaries.items():
        if not summ.critical:
            continue
        evidence: list[str] = []
        root_cause_votes: dict[str, int] = {}
        for scenario, r in scenario_reports:
            reason = _dimension_failure_reason(name, r)
            if not reason:
                continue
            if len(evidence) < 3:
                evidence.append(reason)
            key, _, _ = analyzer.classify_report(scenario, r)
            root_cause_votes[key] = root_cause_votes.get(key, 0) + 1

        root_cause = max(root_cause_votes, key=root_cause_votes.get) if root_cause_votes else ""
        _, suggested_fix = analyzer.label_and_fix(root_cause) if root_cause else ("", "")

        failures.append(
            CriticalFailure(
                dimension=name,
                description=_DIMENSION_LABELS.get(name, name),
                affected_count=summ.failed,
                total_applicable=summ.passed + summ.failed,
                evidence=evidence,
                root_cause=root_cause,
                suggested_fix=suggested_fix,
            )
        )
    # Headline the most-affected failure first.
    failures.sort(key=lambda f: f.affected_count, reverse=True)
    return failures


def _verdict(passed: int, total: int) -> str:
    if total == 0:
        return "not_ready"
    rate = passed / total
    if rate >= 0.85:
        return "production_ready"
    if rate >= 0.50:
        return "needs_work"
    return "not_ready"


def _build_graph_coverage(
    scenario_reports: list[tuple[Scenario, Report]],
    agent_model: "AgentModel | None",
) -> list[NodeCoverage]:
    """
    Roll up (Scenario, Report) pairs by the AgentModel node they target.

    Returns one entry per node in the model — including nodes with ZERO
    scenarios pointed at them, which is the whole point: "Booking -> Timeout:
    0%" is only visible if untested nodes still show up in the list.
    Returns [] when there's no agent_model (ungrounded battery) — the caller
    falls back to the plain dimension-coverage view.
    """
    if agent_model is None or not agent_model.nodes:
        return []

    by_node: dict[str, list[Report]] = {n.id: [] for n in agent_model.nodes}
    for scenario, report in scenario_reports:
        if scenario.node_id in by_node:
            by_node[scenario.node_id].append(report)

    coverage: list[NodeCoverage] = []
    for node in agent_model.nodes:
        reports = by_node[node.id]
        passed = sum(1 for r in reports if r.passed)
        coverage.append(
            NodeCoverage(
                node_id=node.id,
                label=node.label,
                total=len(reports),
                passed=passed,
                failed=len(reports) - passed,
            )
        )
    return coverage


def aggregate_battery(
    battery_id: str,
    scenario_reports: list[tuple[Scenario, Report]],
    agent_model: "AgentModel | None" = None,
) -> BatteryReport:
    """
    Roll up (Scenario, Report) pairs into one BatteryReport.

    Pass the confirmed `agent_model` used to generate this battery (if any) to
    also compute per-node graph coverage — untested nodes show up at 0% rather
    than being silently absent.
    """
    reports = [r for _, r in scenario_reports]
    total = len(reports)
    passed = sum(1 for r in reports if r.passed)

    dimensions: dict[str, DimensionSummary] = {}
    for name in (*_SCORED_DIMENSIONS, "latency"):
        dimensions[name] = _summarize_dimension(name, reports)

    # Coverage = applicable cells / total possible cells across all dimensions.
    total_cells = total * len(dimensions)
    applicable_cells = sum(
        d.passed + d.failed for d in dimensions.values()
    )
    coverage_pct = (applicable_cells / total_cells * 100.0) if total_cells else 0.0

    critical_failures = _build_critical_failures(dimensions, scenario_reports)

    scenario_results = [
        {
            "scenario_id": s.id,
            "group": s.group,
            "title": s.title,
            "passed": r.passed,
            "summary": r.summary,
        }
        for s, r in scenario_reports
    ]

    return BatteryReport(
        battery_id=battery_id,
        total=total,
        passed=passed,
        coverage_pct=round(coverage_pct, 1),
        verdict=_verdict(passed, total),
        dimensions=dimensions,
        critical_failures=critical_failures,
        scenario_results=scenario_results,
        graph_coverage=_build_graph_coverage(scenario_reports, agent_model),
    )
