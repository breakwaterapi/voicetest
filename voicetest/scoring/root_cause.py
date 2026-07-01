"""
RootCauseAnalyzer — group failed (Scenario, Report) pairs by *why* they failed.

Pure heuristic, no ML: grep the report summary, dimension failure reasons, and
call error for known failure-mode keywords. Each root cause maps to a brief
categorized key (e.g. "api_timeout") plus a suggested fix a developer can act
on immediately.
"""

from __future__ import annotations

from voicetest.models import Report
from voicetest.scenarios.models import Scenario

# Ordered so more specific patterns are checked before generic ones.
_RULES: list[tuple[str, tuple[str, ...], str, str]] = [
    (
        "api_timeout",
        ("timeout", "timed out", "time out"),
        "API/tool call timeout",
        "Add retry with backoff or raise the timeout on the slow tool call.",
    ),
    (
        "api_error",
        ("api error", "api failed", "500", "503", "internal server error"),
        "Upstream API error",
        "Check the failing API's error logs and add graceful fallback handling.",
    ),
    (
        "not_found",
        ("not found", "no results", "no availability", "no match"),
        "Lookup returned no results",
        "Verify the query params sent to the lookup and handle empty results gracefully.",
    ),
    (
        "malformed_data",
        ("malformed", "invalid format", "parse error", "could not parse"),
        "Malformed data from a tool/API response",
        "Add response validation and a clear fallback message when parsing fails.",
    ),
    (
        "speech_recognition",
        ("garbled", "misheard", "mumbled", "unclear audio", "could not understand", "did not understand"),
        "Speech recognition failure",
        "Improve ASR prompt/grammar biasing for this field or add a confirm-back step.",
    ),
    (
        "prompt_ambiguity",
        ("ambiguous", "unclear intent", "misinterpreted", "conflicting"),
        "Agent misinterpreted an ambiguous caller request",
        "Add clarifying-question logic before committing to an interpretation.",
    ),
    (
        "barge_in_handling",
        ("barge-in", "barge in", "interruption", "did not stop", "kept talking"),
        "Agent mishandled a caller interruption",
        "Ensure the agent stops speaking immediately on detected barge-in and recovers context.",
    ),
    (
        "goal_incomplete",
        ("goal not completed", "did not complete", "failed to book", "failed to schedule"),
        "Agent failed to complete the caller's goal",
        "Review the conversation flow for the missing step and add explicit confirmation.",
    ),
]

_DEFAULT_KEY = "other"
_DEFAULT_LABEL = "Uncategorized failure"
_DEFAULT_FIX = "Review the transcript manually — no known pattern matched."


def _failure_text(scenario: Scenario, report: Report) -> str:  # noqa: ARG001
    """Concatenate everything that might mention why a call failed."""
    parts = [report.summary or ""]
    if report.call.error:
        parts.append(report.call.error)
    for dim in report.dimensions.values():
        if isinstance(dim, dict):
            reason = dim.get("reason")
            if reason:
                parts.append(reason)
    return " ".join(parts).lower()


def _classify(text: str) -> tuple[str, str, str]:
    for key, keywords, label, fix in _RULES:
        if any(kw in text for kw in keywords):
            return key, label, fix
    return _DEFAULT_KEY, _DEFAULT_LABEL, _DEFAULT_FIX


class RootCauseAnalyzer:
    """Groups failed (Scenario, Report) pairs into root-cause buckets."""

    def analyze(
        self, failed: list[tuple[Scenario, Report]]
    ) -> dict[str, list[str]]:
        """
        Returns {root_cause_key: [evidence snippet, ...]} for every failed pair.

        Evidence is the raw failure text (summary/error/reason) truncated to a
        readable snippet — used both for grouping and for display in the UI.
        """
        buckets: dict[str, list[str]] = {}
        for scenario, report in failed:
            text = _failure_text(scenario, report)
            key, _, _ = _classify(text)
            snippet = (report.summary or report.call.error or "").strip()
            if snippet:
                buckets.setdefault(key, []).append(snippet)
        return buckets

    def label_and_fix(self, root_cause_key: str) -> tuple[str, str]:
        """Look up the display label + suggested fix for a root cause key."""
        for key, _, label, fix in _RULES:
            if key == root_cause_key:
                return label, fix
        return _DEFAULT_LABEL, _DEFAULT_FIX

    def classify_report(self, scenario: Scenario, report: Report) -> tuple[str, str, str]:
        """Classify a single failed (Scenario, Report) pair -> (key, label, fix)."""
        return _classify(_failure_text(scenario, report))

    def classify_text(self, text: str) -> tuple[str, str, str]:
        """Classify raw failure text (e.g. a production call transcript/error) -> (key, label, fix)."""
        return _classify(text.lower())
