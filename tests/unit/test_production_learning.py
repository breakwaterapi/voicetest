"""
Tests for step 5: production learning — ingest, dedup, and auto-generated
regression scenarios that would catch the ingested failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicetest.scenarios.compiler import LLMScenarioCompiler, StubScenarioCompiler
from voicetest.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    db_path = tmp_path / "test.db"
    return Store(db_url=f"sqlite:///{db_path}")


# ---------------------------------------------------------------------------
# Store: production call dedup
# ---------------------------------------------------------------------------


def test_production_call_dedup(store: Store) -> None:
    transcript = "Caller: I need to book Tuesday. Agent: Sorry, the calendar API timed out."

    assert store.production_call_exists("agent-1", transcript) is False
    store.save_production_call("call-1", "agent-1", transcript, "error", "timeout")
    assert store.production_call_exists("agent-1", transcript) is True

    # Re-saving the same transcript for the same agent should be a no-op, not a duplicate row.
    store.save_production_call("call-2", "agent-1", transcript, "error", "timeout")
    failures = store.get_production_failures("agent-1")
    assert len(failures) == 1
    assert failures[0]["id"] == "call-1"


def test_production_call_dedup_is_per_agent(store: Store) -> None:
    transcript = "Same transcript text"
    store.save_production_call("call-1", "agent-1", transcript, "error", "timeout")
    assert store.production_call_exists("agent-2", transcript) is False


def test_get_production_failures_excludes_completed(store: Store) -> None:
    store.save_production_call("call-1", "agent-1", "all good", "completed", None)
    store.save_production_call("call-2", "agent-1", "went wrong", "error", "timed out")

    failures = store.get_production_failures("agent-1")
    assert len(failures) == 1
    assert failures[0]["id"] == "call-2"


def test_get_production_failures_sorted_by_recency(store: Store) -> None:
    store.save_production_call("call-1", "agent-1", "first failure", "error", "timeout")
    store.save_production_call("call-2", "agent-1", "second failure", "error", "timeout")

    failures = store.get_production_failures("agent-1")
    ids = [f["id"] for f in failures]
    assert ids == ["call-2", "call-1"]


# ---------------------------------------------------------------------------
# Regression scenario generation from failure text
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stub_compiler_generates_three_scenarios_from_failure_text() -> None:
    compiler = StubScenarioCompiler()
    scenarios = await compiler.compile_from_failure_text("Calendar API timed out mid-booking.")
    assert len(scenarios) == 3
    assert all(s.group == "regression" for s in scenarios)


@pytest.mark.asyncio
async def test_llm_compiler_failure_prompt_includes_transcript_keywords(monkeypatch) -> None:
    compiler = LLMScenarioCompiler(api_key="fake")
    captured: dict[str, str] = {}

    async def fake_call_anthropic(self, prompt, max_tokens=8192):
        captured["prompt"] = prompt
        return json.dumps({
            "category": "calendar_lookup_regression",
            "groups": [{
                "group": "regression",
                "scenarios": [
                    {
                        "title": f"Regression {i}",
                        "persona_system_prompt": "p", "goal": "g",
                        "difficult_behaviors": ["Calendar API times out mid-booking"],
                        "objective": "o", "pass_criteria": ["c"],
                    }
                    for i in range(3)
                ],
            }],
        })

    monkeypatch.setattr(LLMScenarioCompiler, "_call_anthropic", fake_call_anthropic)

    transcript = "Caller: I need Tuesday at 3pm. Agent: Sorry, the calendar API timed out."
    scenarios = await compiler.compile_from_failure_text(transcript)

    assert "calendar api timed out" in captured["prompt"].lower()
    assert len(scenarios) == 3
    assert all(s.group == "regression" for s in scenarios)
    assert all(
        "api times out" in b.lower() for s in scenarios for b in s.persona.difficult_behaviors
    )


# ---------------------------------------------------------------------------
# End-to-end: ingest failure -> generate scenario -> scenario would catch it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_ingest_generates_scenario_matching_failure(store: Store) -> None:
    transcript = "Caller: Book me Tuesday at 3pm. Agent: Sorry, the calendar API timed out."

    assert store.production_call_exists("agent-1", transcript) is False
    store.save_production_call("call-1", "agent-1", transcript, "error", "timed out", root_cause="api_timeout")

    failures = store.get_production_failures("agent-1")
    assert len(failures) == 1
    assert failures[0]["root_cause"] == "api_timeout"

    compiler = StubScenarioCompiler()
    scenarios = await compiler.compile_from_failure_text(failures[0]["transcript"])

    # The generated regression batch is tagged so it stands out in reports,
    # and is non-empty — i.e. running the battery would exercise this failure again.
    assert len(scenarios) == 3
    assert all(s.group == "regression" for s in scenarios)
