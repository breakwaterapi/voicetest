"""
Integration test: `run --broken` — scorer trust gate.

From testing-strategy.md:
  "running a scenario against the broken agent must produce a **failing** report
   with the correct failure reasons. If the scorer passes the broken agent,
   that's a test failure."

Verifies:
- CLI exits with code 1 (passed=False)
- report.json has passed=False
- goal_completion dimension reflects failure
- All pass_criteria are marked not met
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicetest.cli import main


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def test_broken_exits_nonzero(output_dir: Path) -> None:
    """--broken must exit non-zero (failed report)."""
    exit_code = main(["run", "--broken", "--output", str(output_dir)])
    assert exit_code != 0


def test_broken_exits_1(output_dir: Path) -> None:
    exit_code = main(["run", "--broken", "--output", str(output_dir)])
    assert exit_code == 1


def test_broken_writes_report_json(output_dir: Path) -> None:
    main(["run", "--broken", "--output", str(output_dir)])
    report_files = list(output_dir.rglob("report.json"))
    assert len(report_files) >= 1


def test_broken_report_passed_is_false(output_dir: Path) -> None:
    """Core scorer-trust assertion: broken agent must produce passed=False."""
    main(["run", "--broken", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    assert data["passed"] is False, (
        "Scorer PASSED the broken agent — scorer trust gate FAILED. "
        "The broken-agent fixture must produce a failing report."
    )


def test_broken_report_has_goal_completion(output_dir: Path) -> None:
    main(["run", "--broken", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    assert "goal_completion" in data["dimensions"]


def test_broken_goal_completed_false(output_dir: Path) -> None:
    main(["run", "--broken", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    goal = data["dimensions"]["goal_completion"]
    assert goal["goal_completed"] is False, "Broken agent goal should not be completed"


def test_broken_all_criteria_not_met(output_dir: Path) -> None:
    """Every pass_criterion must be marked not met for the broken agent."""
    main(["run", "--broken", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    criteria = data["dimensions"]["goal_completion"]["criteria"]
    assert len(criteria) > 0, "No criteria in report"
    for c in criteria:
        assert c["met"] is False, (
            f"Criterion '{c['text']}' was marked met for a broken agent — "
            "scorer trust gate FAILED"
        )


def test_broken_report_has_reason(output_dir: Path) -> None:
    main(["run", "--broken", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    reason = data["dimensions"]["goal_completion"]["reason"]
    assert isinstance(reason, str) and len(reason) > 0


def test_broken_report_has_latency_dimension(output_dir: Path) -> None:
    main(["run", "--broken", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    assert "latency" in data["dimensions"]


def test_broken_call_outcome_completed(output_dir: Path) -> None:
    """The broken agent still answers — call outcome is COMPLETED."""
    main(["run", "--broken", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    assert data["call"]["outcome"] == "completed"
