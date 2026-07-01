"""
Test fixtures for the voicetest suite.

Exports:
- make_broken_agent / make_mishearing_agent  (pre-existing, in broken_agent.py)
- stub_providers()    — Providers bundle with StubSTT/StubTTS/StubLLM
- mini_scenario()     — minimal 2-turn Scenario for fast tests
"""

from __future__ import annotations

from voicetest.engine.pipeline import Providers
from voicetest.providers.llm import StubLLM
from voicetest.providers.stt import StubSTT
from voicetest.providers.tts import StubTTS
from voicetest.scenarios import Limits, Persona, Scenario, ScoringConfig


def stub_providers(responses: list[str] | None = None) -> Providers:
    """Return a Providers bundle backed entirely by stubs (no network)."""
    return Providers(
        stt=StubSTT(),
        tts=StubTTS(),
        llm=StubLLM(responses=responses),
    )


def mini_scenario(max_turns: int = 10) -> Scenario:
    """Minimal hardcoded scenario for integration tests."""
    return Scenario(
        id="test-mini-booking",
        category="appointment_booking",
        title="Mini booking test",
        version=1,
        persona=Persona(
            system_prompt="You are calling a restaurant to book a table.",
            goal="Book a table for two people.",
            difficult_behaviors=[
                "Give the wrong party size first, then correct yourself.",
                "Be slightly impatient.",
            ],
        ),
        scoring=ScoringConfig(objective="Agent books a table for two people."),
        limits=Limits(max_duration_s=30, max_turns=max_turns),
    )
