"""
Integration test: `run --loopback` end-to-end.

Verifies:
- CLI exits with code 0 (passed=True with CannedJudge that says pass)
- report.json is written and contains a `passed` field
- report.json contains turns, outcome, and scenario_id
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicetest.cli import main


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


def test_loopback_exits_0(output_dir: Path) -> None:
    """run --loopback with the default scenario should exit 0 (passed)."""
    exit_code = main(["run", "--loopback", "--output", str(output_dir)])
    assert exit_code == 0


def test_loopback_writes_report_json(output_dir: Path) -> None:
    """A report.json file must be created under the output directory."""
    main(["run", "--loopback", "--output", str(output_dir)])
    report_files = list(output_dir.rglob("report.json"))
    assert len(report_files) >= 1, "No report.json found under output dir"


def test_loopback_report_has_passed_field(output_dir: Path) -> None:
    main(["run", "--loopback", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    assert "passed" in data, "report.json missing 'passed' field"


def test_loopback_report_passed_is_true(output_dir: Path) -> None:
    main(["run", "--loopback", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    assert data["passed"] is True


def test_loopback_report_has_call(output_dir: Path) -> None:
    main(["run", "--loopback", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    assert "call" in data
    call = data["call"]
    assert "scenario_id" in call
    assert "outcome" in call
    assert "turns" in call


def test_loopback_report_outcome_completed(output_dir: Path) -> None:
    main(["run", "--loopback", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    assert data["call"]["outcome"] == "completed"


def test_loopback_report_has_dimensions(output_dir: Path) -> None:
    main(["run", "--loopback", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    assert "dimensions" in data
    assert "latency" in data["dimensions"]
    assert "goal_completion" in data["dimensions"]


def test_loopback_with_explicit_scenario(output_dir: Path) -> None:
    """--scenario can be given as a builtin id."""
    exit_code = main([
        "run", "--loopback",
        "--scenario", "appointment-booking-confused-date",
        "--output", str(output_dir),
    ])
    assert exit_code == 0


def test_loopback_report_has_summary(output_dir: Path) -> None:
    main(["run", "--loopback", "--output", str(output_dir)])
    report_file = next(output_dir.rglob("report.json"))
    data = json.loads(report_file.read_text())
    assert "summary" in data
    assert isinstance(data["summary"], str)
    assert len(data["summary"]) > 0


def test_list_subcommand_exits_0() -> None:
    exit_code = main(["list"])
    assert exit_code == 0
