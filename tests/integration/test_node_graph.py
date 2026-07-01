"""
Step 3: node graph UI — verify the graph panel renders on the page and that
clicking a node / generating more scenarios works end-to-end against the
/battery/generate-for-node endpoint.

No telephony, no real LLM: the compiler is a StubScenarioCompiler.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from voicetest.agent_model.models import AgentModel, GraphNode
from voicetest.scenarios.compiler import StubScenarioCompiler
from voicetest.store import Store


def _confirmed_model_with_nodes(*node_ids: str) -> AgentModel:
    model = AgentModel(
        goal="Book appointments.",
        confirmed=True,
        nodes=[GraphNode(id=nid, label=nid.title()) for nid in node_ids],
    )
    model.ensure_node_ids()
    return model


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    import voicetest.api as api_module

    api_module._store = Store(db_url=f"sqlite:///{tmp_path / 'test.db'}")
    api_module._settings = None
    api_module._compiler = StubScenarioCompiler()
    added_ids = set(api_module._compiled_scenarios.keys())

    from voicetest.api import app

    with TestClient(app) as c:
        yield c

    api_module._store = None
    api_module._compiler = None
    for sid in set(api_module._compiled_scenarios.keys()) - added_ids:
        del api_module._compiled_scenarios[sid]


# ---------------------------------------------------------------------------
# Graph panel renders in the page
# ---------------------------------------------------------------------------


def test_ui_has_graph_panel_markup(client: TestClient) -> None:
    html = client.get("/").text
    assert 'id="node-graph-section"' in html
    assert 'id="graph-panel-body"' in html
    assert "renderGraphPanel" in html
    assert "showNodePanel" in html


def test_ui_has_generate_for_node_hook(client: TestClient) -> None:
    html = client.get("/").text
    assert "generateForNode" in html
    assert "/battery/generate-for-node" in html


# ---------------------------------------------------------------------------
# POST /battery/generate-for-node
# ---------------------------------------------------------------------------


def test_generate_for_node_requires_confirmed_model(client: TestClient) -> None:
    resp = client.post(
        "/battery/generate-for-node",
        json={"agent_model_id": "nonexistent", "node_id": "greeting", "count": 10},
    )
    assert resp.status_code == 404


def test_generate_for_node_rejects_unknown_node(client: TestClient) -> None:
    import voicetest.api as api_module

    model = _confirmed_model_with_nodes("greeting", "confirm")
    api_module._store.save_agent_model(model)

    resp = client.post(
        "/battery/generate-for-node",
        json={"agent_model_id": model.id, "node_id": "made-up", "count": 10},
    )
    assert resp.status_code == 404


def test_generate_for_node_returns_scenarios_targeting_that_node(
    client: TestClient,
) -> None:
    import voicetest.api as api_module

    model = _confirmed_model_with_nodes("greeting", "confirm")
    api_module._store.save_agent_model(model)

    resp = client.post(
        "/battery/generate-for-node",
        json={"agent_model_id": model.id, "node_id": "confirm", "count": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    scenarios = body["scenarios"]
    assert len(scenarios) == 5
    assert all(s["node_id"] == "confirm" for s in scenarios)


def test_generate_for_node_caches_scenarios_for_later_lookup(
    client: TestClient,
) -> None:
    import voicetest.api as api_module

    model = _confirmed_model_with_nodes("greeting", "confirm")
    api_module._store.save_agent_model(model)

    resp = client.post(
        "/battery/generate-for-node",
        json={"agent_model_id": model.id, "node_id": "confirm", "count": 3},
    )
    scenario_ids = [s["id"] for s in resp.json()["scenarios"]]

    all_scenarios = client.get("/scenarios").json()
    all_ids = {s["id"] for s in all_scenarios}
    assert all(sid in all_ids for sid in scenario_ids)
