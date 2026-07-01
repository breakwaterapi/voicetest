"""Unit tests for the Agent Model (extraction, coercion, persistence)."""

from __future__ import annotations

import tempfile

import pytest

from voicetest.agent_model.extractor import StubAgentModelExtractor
from voicetest.agent_model.models import AgentModel, coerce_agent_model
from voicetest.store import Store


@pytest.mark.asyncio
async def test_stub_extract_produces_renderable_model():
    model = await StubAgentModelExtractor().extract("books haircuts")
    assert model.goal
    assert model.nodes and model.edges
    node_ids = {n.id for n in model.nodes}
    # Every edge must reference nodes that actually exist.
    assert all(e.source in node_ids and e.target in node_ids for e in model.edges)


def test_coerce_accepts_loose_string_lists_and_relinks_edges():
    loose = {
        "goal": "g",
        "required_info": ["Name", "Phone"],
        "tools": ["Calendar API"],
        "business_rules": ["never double-book"],
        "risks": ["calendar timeout"],
        "nodes": ["Greeting", "Confirm"],
        # edge references nodes by LABEL, not id — must be relinked to ids
        "edges": [{"source": "Greeting", "target": "Confirm", "condition": "ok"}],
    }
    m = coerce_agent_model(loose, source_description="src")
    assert m.required_info[0].key == "name"
    assert m.tools[0].name == "Calendar API"
    assert m.edges[0].source == "greeting" and m.edges[0].target == "confirm"


def test_ensure_node_ids_drops_dangling_edges():
    m = AgentModel(
        nodes=[{"id": "a", "label": "A"}],  # type: ignore[list-item]
        edges=[{"source": "a", "target": "ghost"}],  # type: ignore[list-item]
    )
    m.ensure_node_ids()
    assert m.edges == []  # edge to a non-existent node is dropped


@pytest.mark.asyncio
async def test_store_round_trip_and_upsert():
    db = tempfile.mktemp(suffix=".db")
    store = Store(db_url="sqlite:///" + db)
    model = await StubAgentModelExtractor().extract("x")

    store.save_agent_model(model)
    got = store.get_agent_model(model.id)
    assert got is not None and got.id == model.id and got.confirmed is False

    model.confirmed = True
    store.save_agent_model(model)  # upsert on same id
    again = store.get_agent_model(model.id)
    assert again is not None and again.confirmed is True

    assert store.get_agent_model("missing") is None
