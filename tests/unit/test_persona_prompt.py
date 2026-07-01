"""
Unit tests: persona system prompt assembly.

Asserts that every difficult_behavior entry and the goal appear verbatim
in the assembled prompt.
"""

from __future__ import annotations

import pytest

from voicetest.engine.pipeline import assemble_system_prompt
from voicetest.scenarios import Persona


def make_persona(behaviors: list[str], goal: str = "Book a table.") -> Persona:
    return Persona(
        system_prompt="You are calling a restaurant.",
        goal=goal,
        difficult_behaviors=behaviors,
    )


def test_all_behaviors_present() -> None:
    behaviors = [
        "Give the WRONG date first, then correct yourself.",
        "Mumble the phone number the first time.",
        "Get impatient if the agent takes too long.",
        "Ask one out-of-scope question.",
    ]
    persona = make_persona(behaviors)
    prompt = assemble_system_prompt(persona)

    for behavior in behaviors:
        assert behavior in prompt, f"Behavior not found in prompt: {behavior!r}"


def test_goal_present() -> None:
    goal = "Book a table for four people on Friday at 8pm."
    persona = make_persona(["Be rude if asked twice."], goal=goal)
    prompt = assemble_system_prompt(persona)
    assert goal in prompt


def test_system_prompt_included() -> None:
    persona = Persona(
        system_prompt="You are a grumpy caller who hates waiting.",
        goal="Cancel an appointment.",
        difficult_behaviors=["Sigh loudly every other sentence."],
    )
    prompt = assemble_system_prompt(persona)
    assert "grumpy caller" in prompt


def test_single_behavior() -> None:
    behavior = "Only speak in questions."
    persona = make_persona([behavior])
    prompt = assemble_system_prompt(persona)
    assert behavior in prompt


def test_multiple_behaviors_all_present() -> None:
    behaviors = [f"Behavior number {i}." for i in range(10)]
    persona = make_persona(behaviors)
    prompt = assemble_system_prompt(persona)
    for b in behaviors:
        assert b in prompt


def test_prompt_is_nonempty_string() -> None:
    persona = make_persona(["Do something difficult."])
    prompt = assemble_system_prompt(persona)
    assert isinstance(prompt, str)
    assert len(prompt) > 50


def test_persona_rejects_empty_behaviors() -> None:
    with pytest.raises(Exception):
        Persona(
            system_prompt="You are a caller.",
            goal="Book something.",
            difficult_behaviors=[],
        )
