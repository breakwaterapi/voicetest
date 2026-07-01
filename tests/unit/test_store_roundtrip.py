"""
test_store_roundtrip.py — runs/calls/reports persist and reload correctly.

Uses an in-memory SQLite database (":memory:" is not supported by sqlite3.connect
across connections, so we use a tmp_path file instead).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from voicetest.models import CallOutcome, CallResult, Event, Report, Turn
from voicetest.store import RunStatus, Store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Store:
    db_path = tmp_path / "test.db"
    return Store(db_url=f"sqlite:///{db_path}")


def _make_call_result(outcome: CallOutcome = CallOutcome.COMPLETED) -> CallResult:
    return CallResult(
        scenario_id="test-scenario",
        outcome=outcome,
        turns=[
            Turn(speaker="caller", text="Hello", start_ms=0, end_ms=500),
            Turn(speaker="agent", text="Hi there", start_ms=600, end_ms=1200),
        ],
        events=[Event(kind="start", at_ms=0)],
        recording_path=None,
        duration_ms=1200,
        error=None,
    )


def _make_report(call_result: CallResult, passed: bool = True) -> Report:
    return Report(
        call=call_result,
        passed=passed,
        dimensions={
            "goal_completion": {"score": 1 if passed else 0, "reason": "done"},
            "latency": {"score": None, "p50_ms": 300, "p95_ms": 600, "warn": False},
        },
        summary="PASSED — test" if passed else "FAILED — test",
    )


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------


def test_create_and_get_run(store: Store) -> None:
    run_id = str(uuid.uuid4())
    store.create_run(run_id, "scenario-abc", "+15550001234", 3)

    row = store.get_run(run_id)
    assert row is not None
    assert row["id"] == run_id
    assert row["scenario_id"] == "scenario-abc"
    assert row["number"] == "+15550001234"
    assert row["n"] == 3
    assert row["status"] == RunStatus.PENDING
    assert row["completed"] == 0


def test_update_run_status(store: Store) -> None:
    run_id = str(uuid.uuid4())
    store.create_run(run_id, "s", "+10000000001", 2)

    store.update_run_status(run_id, RunStatus.RUNNING, 1)
    row = store.get_run(run_id)
    assert row["status"] == RunStatus.RUNNING
    assert row["completed"] == 1

    store.update_run_status(run_id, RunStatus.COMPLETED, 2)
    row = store.get_run(run_id)
    assert row["status"] == RunStatus.COMPLETED
    assert row["completed"] == 2


def test_get_run_missing_returns_none(store: Store) -> None:
    assert store.get_run("nonexistent-id") is None


# ---------------------------------------------------------------------------
# Call CRUD
# ---------------------------------------------------------------------------


def test_create_and_update_call(store: Store) -> None:
    run_id = str(uuid.uuid4())
    call_id = str(uuid.uuid4())
    store.create_run(run_id, "s", "+10000000001", 1)
    store.create_call(call_id, run_id, 0)

    row = store.get_call(call_id)
    assert row is not None
    assert row["id"] == call_id
    assert row["run_id"] == run_id
    assert row["call_index"] == 0
    assert row["outcome"] is None  # not yet filled

    result = _make_call_result()
    store.update_call(call_id, result)

    row = store.get_call(call_id)
    assert row["outcome"] == CallOutcome.COMPLETED
    assert row["duration_ms"] == 1200
    assert row["error"] is None


def test_call_result_json_roundtrip(store: Store) -> None:
    run_id = str(uuid.uuid4())
    call_id = str(uuid.uuid4())
    store.create_run(run_id, "s", "+10000000001", 1)
    store.create_call(call_id, run_id, 0)

    result = _make_call_result()
    store.update_call(call_id, result)

    row = store.get_call(call_id)
    data = json.loads(row["result_json"])
    assert data["scenario_id"] == "test-scenario"
    assert data["outcome"] == "completed"
    assert len(data["turns"]) == 2


def test_get_calls_for_run_ordering(store: Store) -> None:
    run_id = str(uuid.uuid4())
    store.create_run(run_id, "s", "+10000000001", 3)

    ids = [str(uuid.uuid4()) for _ in range(3)]
    for i, cid in enumerate(ids):
        store.create_call(cid, run_id, i)

    rows = store.get_calls_for_run(run_id)
    assert [r["call_index"] for r in rows] == [0, 1, 2]
    assert [r["id"] for r in rows] == ids


def test_get_call_missing_returns_none(store: Store) -> None:
    assert store.get_call("nope") is None


# ---------------------------------------------------------------------------
# Report CRUD
# ---------------------------------------------------------------------------


def test_save_and_get_report(store: Store) -> None:
    run_id = str(uuid.uuid4())
    call_id = str(uuid.uuid4())
    report_id = str(uuid.uuid4())
    store.create_run(run_id, "s", "+10000000001", 1)
    store.create_call(call_id, run_id, 0)

    result = _make_call_result()
    store.update_call(call_id, result)

    report = _make_report(result, passed=True)
    store.save_report(report_id, call_id, run_id, report)

    row = store.get_report_for_call(call_id)
    assert row is not None
    assert row["id"] == report_id
    assert row["passed"] == 1
    assert "goal_completion" in json.loads(row["dimensions"])
    assert "PASSED" in row["summary"]


def test_get_reports_for_run(store: Store) -> None:
    run_id = str(uuid.uuid4())
    store.create_run(run_id, "s", "+10000000001", 2)

    for i in range(2):
        cid = str(uuid.uuid4())
        rid = str(uuid.uuid4())
        store.create_call(cid, run_id, i)
        result = _make_call_result()
        store.update_call(cid, result)
        store.save_report(rid, cid, run_id, _make_report(result, passed=(i == 0)))

    rows = store.get_reports_for_run(run_id)
    assert len(rows) == 2
    assert rows[0]["call_index"] == 0
    assert rows[1]["call_index"] == 1
    assert rows[0]["passed"] == 1
    assert rows[1]["passed"] == 0


def test_report_missing_returns_none(store: Store) -> None:
    run_id = str(uuid.uuid4())
    call_id = str(uuid.uuid4())
    store.create_run(run_id, "s", "+10000000001", 1)
    store.create_call(call_id, run_id, 0)
    assert store.get_report_for_call(call_id) is None


# ---------------------------------------------------------------------------
# Non-COMPLETED call outcome stored correctly
# ---------------------------------------------------------------------------


def test_non_completed_outcome_stored(store: Store) -> None:
    run_id = str(uuid.uuid4())
    call_id = str(uuid.uuid4())
    store.create_run(run_id, "s", "+10000000001", 1)
    store.create_call(call_id, run_id, 0)

    result = _make_call_result(outcome=CallOutcome.NO_ANSWER)
    result = CallResult(
        scenario_id=result.scenario_id,
        outcome=CallOutcome.NO_ANSWER,
        turns=[],
        events=[],
        recording_path=None,
        duration_ms=0,
        error="connect timed out",
    )
    store.update_call(call_id, result)

    row = store.get_call(call_id)
    assert row["outcome"] == "no_answer"
    assert row["error"] == "connect timed out"
