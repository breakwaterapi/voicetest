"""
Tests for step 2: grounding scenario generation + coverage in a confirmed
AgentModel (description -> AgentModel -> scenarios, with per-node coverage).
"""

from __future__ import annotations

import json

import pytest

from voicetest.agent_model.models import AgentModel, BusinessRule, GraphNode
from voicetest.models import CallOutcome, CallResult, Report
from voicetest.scenarios.compiler import LLMScenarioCompiler, _build_battery
from voicetest.scoring.aggregate import aggregate_battery


def _model_with_nodes(*node_ids: str) -> AgentModel:
    return AgentModel(
        goal="Book appointments.",
        nodes=[GraphNode(id=nid, label=nid.title()) for nid in node_ids],
    )


def _fake_report(passed: bool) -> Report:
    call = CallResult(
        scenario_id="s", outcome=CallOutcome.COMPLETED, turns=[], events=[],
        recording_path=None, duration_ms=1000,
    )
    return Report(call=call, passed=passed, dimensions={}, summary="")


# ---------------------------------------------------------------------------
# Compiler: node_id validation
# ---------------------------------------------------------------------------


def test_build_battery_keeps_valid_node_ids():
    raw = json.dumps({
        "category": "booking",
        "groups": [{
            "group": "baseline",
            "scenarios": [{
                "title": "Happy path",
                "node_id": "confirm",
                "persona_system_prompt": "p", "goal": "g",
                "difficult_behaviors": ["b"], "objective": "o",
                "pass_criteria": ["c"],
            }],
        }],
    })
    scenarios = _build_battery(raw, valid_node_ids={"greeting", "confirm"})
    assert scenarios[0].node_id == "confirm"


def test_build_battery_drops_hallucinated_node_id():
    """A node_id that doesn't exist in the confirmed model must not silently
    corrupt graph coverage — it should be cleared rather than trusted."""
    raw = json.dumps({
        "category": "booking",
        "groups": [{
            "group": "baseline",
            "scenarios": [{
                "title": "Happy path",
                "node_id": "made_up_node",
                "persona_system_prompt": "p", "goal": "g",
                "difficult_behaviors": ["b"], "objective": "o",
                "pass_criteria": ["c"],
            }],
        }],
    })
    scenarios = _build_battery(raw, valid_node_ids={"greeting", "confirm"})
    assert scenarios[0].node_id == ""


def test_build_battery_without_valid_node_ids_trusts_whatever_is_given():
    """Ungrounded (no AgentModel) path — node_id passes through untouched."""
    raw = json.dumps({
        "category": "booking",
        "groups": [{"group": "baseline", "scenarios": [{
            "title": "Happy path", "node_id": "anything",
            "persona_system_prompt": "p", "goal": "g",
            "difficult_behaviors": ["b"], "objective": "o", "pass_criteria": ["c"],
        }]}],
    })
    scenarios = _build_battery(raw, valid_node_ids=None)
    assert scenarios[0].node_id == "anything"


@pytest.mark.asyncio
async def test_compile_battery_grounded_prompt_references_model(monkeypatch):
    """When an AgentModel is passed, the LLM prompt must reference its nodes/
    risks/rules by name (the whole point of grounding) — not just the raw text."""
    model = _model_with_nodes("greeting", "confirm")
    model.business_rules = [BusinessRule(rule="never double-book")]

    compiler = LLMScenarioCompiler(api_key="fake")
    captured: dict[str, str] = {}

    async def fake_call_anthropic(self, prompt, max_tokens=8192):
        captured["prompt"] = prompt
        return json.dumps({"category": "booking", "groups": []})

    monkeypatch.setattr(LLMScenarioCompiler, "_call_anthropic", fake_call_anthropic)

    await compiler.compile_battery("a booking bot", agent_model=model)
    assert "greeting" in captured["prompt"]
    assert "confirm" in captured["prompt"]
    assert "never double-book" in captured["prompt"]


@pytest.mark.asyncio
async def test_compile_battery_without_model_uses_ungrounded_prompt(monkeypatch):
    compiler = LLMScenarioCompiler(api_key="fake")
    captured: dict[str, str] = {}

    async def fake_call_anthropic(self, prompt, max_tokens=8192):
        captured["prompt"] = prompt
        return json.dumps({"category": "booking", "groups": []})

    monkeypatch.setattr(LLMScenarioCompiler, "_call_anthropic", fake_call_anthropic)

    await compiler.compile_battery("a booking bot")
    assert "CONFIRMED AGENT MODEL" not in captured["prompt"]


# ---------------------------------------------------------------------------
# aggregate_battery: graph coverage
# ---------------------------------------------------------------------------


def test_graph_coverage_includes_untested_nodes_at_zero():
    """The whole point: a node with ZERO scenarios must still appear, at 0%,
    not be silently absent from the report."""
    from voicetest.scenarios.compiler import _build_variant

    model = _model_with_nodes("greeting", "confirm", "escalate")
    scenario = _build_variant(
        {"title": "t", "node_id": "confirm", "persona_system_prompt": "p",
         "goal": "g", "difficult_behaviors": ["b"], "objective": "o",
         "pass_criteria": ["c"]},
        category="booking", group="baseline", valid_node_ids={"greeting", "confirm", "escalate"},
    )
    report = aggregate_battery("b1", [(scenario, _fake_report(True))], agent_model=model)

    by_id = {n.node_id: n for n in report.graph_coverage}
    assert set(by_id) == {"greeting", "confirm", "escalate"}
    assert by_id["confirm"].total == 1 and by_id["confirm"].passed == 1
    assert by_id["greeting"].total == 0  # untested — still present
    assert by_id["escalate"].total == 0


def test_graph_coverage_empty_when_no_agent_model():
    from voicetest.scenarios.compiler import _build_variant

    scenario = _build_variant(
        {"title": "t", "persona_system_prompt": "p", "goal": "g",
         "difficult_behaviors": ["b"], "objective": "o", "pass_criteria": ["c"]},
        category="booking", group="baseline",
    )
    report = aggregate_battery("b1", [(scenario, _fake_report(True))])
    assert report.graph_coverage == []
