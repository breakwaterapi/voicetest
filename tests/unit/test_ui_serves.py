"""
test_ui_serves.py — verify the UI page renders and scenario dropdown data is available.

Checks:
- GET / returns 200 with HTML content
- The page contains form elements and the scenario script hook
- GET /scenarios returns a non-empty list with id + title fields
- The transcript endpoint exists and returns the right shape

No telephony, no real LLM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from voicetest.store import Store


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    import voicetest.api as api_module

    api_module._store = Store(db_url=f"sqlite:///{tmp_path / 'test.db'}")
    api_module._settings = None  # reset so default (no allowlist) is used

    from voicetest.api import app

    with TestClient(app) as c:
        yield c

    api_module._store = None


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_ui_returns_200(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200


def test_ui_content_type_is_html(client: TestClient) -> None:
    resp = client.get("/")
    assert "text/html" in resp.headers["content-type"]


def test_ui_has_number_input(client: TestClient) -> None:
    html = client.get("/").text
    assert 'id="battery-number"' in html


def test_ui_has_description_input(client: TestClient) -> None:
    html = client.get("/").text
    assert 'id="battery-desc"' in html


def test_ui_has_known_issues_input(client: TestClient) -> None:
    html = client.get("/").text
    assert 'id="battery-issues"' in html


def test_ui_has_run_button(client: TestClient) -> None:
    html = client.get("/").text
    assert "startBattery" in html


def test_ui_has_progress_section(client: TestClient) -> None:
    html = client.get("/").text
    assert "progress" in html.lower()


def test_ui_has_report_section(client: TestClient) -> None:
    html = client.get("/").text
    assert "report" in html.lower()


def test_ui_polls_battery_status(client: TestClient) -> None:
    html = client.get("/").text
    assert "pollBattery" in html
    assert "/battery/" in html


def test_ui_shows_critical_failures(client: TestClient) -> None:
    html = client.get("/").text
    assert "critical" in html.lower()


def test_ui_shows_verdict(client: TestClient) -> None:
    html = client.get("/").text
    assert "verdict" in html.lower()


def test_ui_shows_group_matrix(client: TestClient) -> None:
    html = client.get("/").text
    assert "group-matrix" in html or "renderGroupMatrix" in html


# ---------------------------------------------------------------------------
# GET /scenarios (the data source for the dropdown)
# ---------------------------------------------------------------------------


def test_scenarios_returns_200(client: TestClient) -> None:
    resp = client.get("/scenarios")
    assert resp.status_code == 200


def test_scenarios_returns_list(client: TestClient) -> None:
    data = client.get("/scenarios").json()
    assert isinstance(data, list)


def test_scenarios_non_empty(client: TestClient) -> None:
    data = client.get("/scenarios").json()
    assert len(data) >= 1


def test_scenarios_have_id_and_title(client: TestClient) -> None:
    data = client.get("/scenarios").json()
    for s in data:
        assert "id" in s
        assert "title" in s
        assert isinstance(s["id"], str) and s["id"]
        assert isinstance(s["title"], str) and s["title"]


def test_scenarios_have_category(client: TestClient) -> None:
    data = client.get("/scenarios").json()
    for s in data:
        assert s["category"] in ("appointment_booking", "lead_qualification", "faq_support")


# ---------------------------------------------------------------------------
# GET /runs/{id}/transcript shape (uses an already-created call)
# ---------------------------------------------------------------------------


def test_transcript_endpoint_missing_run_returns_404(client: TestClient) -> None:
    resp = client.get("/runs/nonexistent-run/transcript")
    assert resp.status_code == 404


def test_transcript_endpoint_returns_list_for_empty_run(
    client: TestClient, tmp_path: Path
) -> None:
    import uuid

    import voicetest.api as api_module

    store = api_module._store
    assert store is not None
    run_id = str(uuid.uuid4())
    store.create_run(run_id, "s", "loopback", 1)

    resp = client.get(f"/runs/{run_id}/transcript")
    assert resp.status_code == 200
    assert resp.json() == []
