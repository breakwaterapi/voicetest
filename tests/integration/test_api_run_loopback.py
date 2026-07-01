"""
test_api_run_loopback.py — end-to-end API test using loopback + stub judge.

POST /runs with a loopback number kicks off N calls, GET /runs/{id} shows
progress → done, GET /runs/{id}/report has a pass rate.

No telephony. No real LLM calls. Uses a CannedJudge and LocalLoopback.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from voicetest.providers.targets import LocalLoopback
from voicetest.scenarios.models import ScoringConfig
from voicetest.scoring.judge import CriterionResult, JudgeResult
from voicetest.store import Store

# ---------------------------------------------------------------------------
# CannedJudge — always returns a passing result, no LLM call
# ---------------------------------------------------------------------------


class CannedJudge:
    def __init__(self, goal_completed: bool = True) -> None:
        self._goal_completed = goal_completed

    async def score(
        self,
        transcript: str,
        rubric: ScoringConfig,
        noise_ground_truth: dict[str, Any] | None = None,
    ) -> JudgeResult:
        criteria = [
            CriterionResult(text=c, met=self._goal_completed, evidence="stub")
            for c in rubric.pass_criteria
        ]
        return JudgeResult(
            goal_completed=self._goal_completed,
            reason=(
                "stub judge — always passes"
                if self._goal_completed
                else "stub judge — always fails"
            ),
            criteria=criteria,
        )


# ---------------------------------------------------------------------------
# App fixture — overrides store, judge, and target factory
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client(tmp_path: Path):
    """
    Return a TestClient wired to an isolated Store and a CannedJudge.
    The target factory always returns a LocalLoopback so no real calls are made.
    """
    import voicetest.api as api_module
    import voicetest.jobs as jobs_module

    # Isolated store
    store = Store(db_url=f"sqlite:///{tmp_path / 'test.db'}")
    api_module._store = store

    # Patch start_run in api_module (the imported name) so the route handler
    # uses our stub judge + loopback target instead of real providers.
    _original_jobs_start = jobs_module.start_run
    _original_api_start = api_module.start_run

    def _patched_start_run(run_id, scenario, number, n, store, judge=None, target_factory=None):
        return _original_jobs_start(
            run_id,
            scenario,
            number,
            n,
            store,
            judge=CannedJudge(goal_completed=True),
            target_factory=lambda _num: LocalLoopback(),
        )

    api_module.start_run = _patched_start_run

    from voicetest.api import app

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client

    # Restore
    api_module.start_run = _original_api_start
    api_module._store = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOOPBACK_NUMBER = "loopback"
SCENARIO_ID = "appointment-booking-confused-date"


def _wait_for_completion(client: TestClient, run_id: str, timeout: float = 30.0) -> dict:
    """Poll GET /runs/{run_id} until status is COMPLETED (or FAILED), with timeout."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.05)
    raise TimeoutError(f"Run {run_id} did not complete within {timeout}s")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_post_runs_returns_run_id(api_client: TestClient) -> None:
    resp = api_client.post(
        "/runs",
        json={"scenario_id": SCENARIO_ID, "number": LOOPBACK_NUMBER, "n": 1},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "run_id" in body
    assert isinstance(body["run_id"], str)


def test_get_run_initially_shows_pending_or_running(api_client: TestClient) -> None:
    resp = api_client.post(
        "/runs",
        json={"scenario_id": SCENARIO_ID, "number": LOOPBACK_NUMBER, "n": 1},
    )
    run_id = resp.json()["run_id"]
    resp2 = api_client.get(f"/runs/{run_id}")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["status"] in ("pending", "running", "completed")
    assert data["n"] == 1


def test_run_completes_with_n_calls(api_client: TestClient) -> None:
    n = 2
    resp = api_client.post(
        "/runs",
        json={"scenario_id": SCENARIO_ID, "number": LOOPBACK_NUMBER, "n": n},
    )
    run_id = resp.json()["run_id"]
    data = _wait_for_completion(api_client, run_id)
    assert data["status"] == "completed"
    assert data["completed"] == n
    assert len(data["calls"]) == n


def test_get_run_calls_have_outcomes(api_client: TestClient) -> None:
    resp = api_client.post(
        "/runs",
        json={"scenario_id": SCENARIO_ID, "number": LOOPBACK_NUMBER, "n": 1},
    )
    run_id = resp.json()["run_id"]
    data = _wait_for_completion(api_client, run_id)
    for call in data["calls"]:
        assert call["outcome"] is not None
        assert call["call_id"] is not None


def test_report_has_pass_rate(api_client: TestClient) -> None:
    n = 3
    resp = api_client.post(
        "/runs",
        json={"scenario_id": SCENARIO_ID, "number": LOOPBACK_NUMBER, "n": n},
    )
    run_id = resp.json()["run_id"]
    _wait_for_completion(api_client, run_id)

    resp2 = api_client.get(f"/runs/{run_id}/report")
    assert resp2.status_code == 200
    report = resp2.json()
    assert report["total"] == n
    assert report["completed"] == n
    assert "pass_rate" in report
    assert isinstance(report["pass_rate"], float)
    assert 0.0 <= report["pass_rate"] <= 1.0


def test_report_call_breakdown(api_client: TestClient) -> None:
    resp = api_client.post(
        "/runs",
        json={"scenario_id": SCENARIO_ID, "number": LOOPBACK_NUMBER, "n": 2},
    )
    run_id = resp.json()["run_id"]
    _wait_for_completion(api_client, run_id)

    report = api_client.get(f"/runs/{run_id}/report").json()
    assert len(report["calls"]) == 2
    for call_rpt in report["calls"]:
        assert "passed" in call_rpt
        assert "summary" in call_rpt
        assert "dimensions" in call_rpt


def test_get_run_missing_returns_404(api_client: TestClient) -> None:
    resp = api_client.get(f"/runs/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_report_missing_returns_404(api_client: TestClient) -> None:
    resp = api_client.get(f"/runs/{uuid.uuid4()}/report")
    assert resp.status_code == 404


def test_invalid_scenario_returns_404(api_client: TestClient) -> None:
    resp = api_client.post(
        "/runs",
        json={"scenario_id": "nonexistent-scenario", "number": LOOPBACK_NUMBER, "n": 1},
    )
    assert resp.status_code == 404


def test_blocked_number_returns_422(api_client: TestClient) -> None:
    """A number not in the allowlist and not a loopback sentinel is rejected."""
    import voicetest.api as api_module
    from voicetest.config import Settings

    # Temporarily set a non-empty allowlist that doesn't include the target number
    original = api_module._settings

    class _FakeSettings(Settings):
        verified_dest_numbers: str | None = "+19990000001"  # type: ignore[assignment]

    api_module._settings = _FakeSettings()
    try:
        resp = api_client.post(
            "/runs",
            json={"scenario_id": SCENARIO_ID, "number": "+15559999999", "n": 1},
        )
        assert resp.status_code == 422
    finally:
        api_module._settings = original


def test_scenarios_endpoint_returns_list(api_client: TestClient) -> None:
    resp = api_client.get("/scenarios")
    assert resp.status_code == 200
    scenarios = resp.json()
    assert isinstance(scenarios, list)
    assert len(scenarios) >= 1
    assert all("id" in s and "title" in s for s in scenarios)


def test_n_zero_returns_422(api_client: TestClient) -> None:
    resp = api_client.post(
        "/runs",
        json={"scenario_id": SCENARIO_ID, "number": LOOPBACK_NUMBER, "n": 0},
    )
    assert resp.status_code == 422
