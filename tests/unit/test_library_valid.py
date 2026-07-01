"""
P1-T1 — Scenario library validation tests.

Verifies that every builtin scenario in voicetest/scenarios/builtin/:
  - Parses without error (load_all succeeds)
  - Has >= 2 difficult_behaviors
  - Has >= 1 pass_criteria
  - Has a unique id
  - Belongs to a valid category
  - At least one scenario enables noise
  - At least one scenario enables barge_in
"""

from __future__ import annotations

import pytest

from voicetest.scenarios import BUILTIN_DIR, Scenario, load_all

VALID_CATEGORIES = {"appointment_booking", "lead_qualification", "faq_support"}


@pytest.fixture(scope="module")
def all_scenarios() -> list[Scenario]:
    return load_all(BUILTIN_DIR)


# ---------------------------------------------------------------------------
# Collection-level invariants
# ---------------------------------------------------------------------------


def test_at_least_six_personas(all_scenarios: list[Scenario]) -> None:
    assert len(all_scenarios) >= 6, (
        f"Expected >= 6 builtin scenarios, found {len(all_scenarios)}"
    )


def test_all_three_categories_present(all_scenarios: list[Scenario]) -> None:
    categories = {s.category for s in all_scenarios}
    assert categories == VALID_CATEGORIES, (
        f"Missing categories: {VALID_CATEGORIES - categories}"
    )


def test_ids_are_unique(all_scenarios: list[Scenario]) -> None:
    ids = [s.id for s in all_scenarios]
    assert len(ids) == len(set(ids)), f"Duplicate scenario ids: {ids}"


def test_at_least_one_noise_enabled(all_scenarios: list[Scenario]) -> None:
    noise_scenarios = [s for s in all_scenarios if s.noise.enabled]
    assert noise_scenarios, "No builtin scenario has noise.enabled=true"


def test_at_least_one_barge_in_enabled(all_scenarios: list[Scenario]) -> None:
    barge_in_scenarios = [s for s in all_scenarios if s.barge_in.enabled]
    assert barge_in_scenarios, "No builtin scenario has barge_in.enabled=true"


# ---------------------------------------------------------------------------
# Per-scenario invariants
# ---------------------------------------------------------------------------


def test_each_scenario_valid_category(all_scenarios: list[Scenario]) -> None:
    for s in all_scenarios:
        assert s.category in VALID_CATEGORIES, (
            f"{s.id}: invalid category '{s.category}'"
        )


def test_each_scenario_at_least_two_difficult_behaviors(
    all_scenarios: list[Scenario],
) -> None:
    for s in all_scenarios:
        count = len(s.persona.difficult_behaviors)
        assert count >= 2, (
            f"{s.id}: only {count} difficult_behavior(s), need >= 2"
        )


def test_each_scenario_difficult_behaviors_are_nonempty_strings(
    all_scenarios: list[Scenario],
) -> None:
    for s in all_scenarios:
        for i, b in enumerate(s.persona.difficult_behaviors):
            assert isinstance(b, str) and b.strip(), (
                f"{s.id}: difficult_behaviors[{i}] is blank or not a string"
            )


def test_each_scenario_at_least_one_pass_criterion(
    all_scenarios: list[Scenario],
) -> None:
    for s in all_scenarios:
        count = len(s.scoring.pass_criteria)
        assert count >= 1, (
            f"{s.id}: has {count} pass_criteria, need >= 1"
        )


def test_each_scenario_has_nonempty_goal(all_scenarios: list[Scenario]) -> None:
    for s in all_scenarios:
        assert s.persona.goal.strip(), f"{s.id}: persona.goal is blank"


def test_each_scenario_has_nonempty_system_prompt(
    all_scenarios: list[Scenario],
) -> None:
    for s in all_scenarios:
        assert len(s.persona.system_prompt.strip()) > 20, (
            f"{s.id}: persona.system_prompt is too short or blank"
        )


def test_each_scenario_has_nonempty_scoring_objective(
    all_scenarios: list[Scenario],
) -> None:
    for s in all_scenarios:
        assert s.scoring.objective.strip(), f"{s.id}: scoring.objective is blank"


def test_each_scenario_valid_limits(all_scenarios: list[Scenario]) -> None:
    for s in all_scenarios:
        assert s.limits.max_duration_s > 0, (
            f"{s.id}: max_duration_s must be positive"
        )
        assert s.limits.max_turns > 0, f"{s.id}: max_turns must be positive"


# ---------------------------------------------------------------------------
# Category coverage — at least 2 personas per category
# ---------------------------------------------------------------------------


def test_appointment_booking_has_at_least_two_personas(
    all_scenarios: list[Scenario],
) -> None:
    count = sum(1 for s in all_scenarios if s.category == "appointment_booking")
    assert count >= 2, f"Only {count} appointment_booking scenario(s)"


def test_lead_qualification_has_at_least_two_personas(
    all_scenarios: list[Scenario],
) -> None:
    count = sum(1 for s in all_scenarios if s.category == "lead_qualification")
    assert count >= 2, f"Only {count} lead_qualification scenario(s)"


def test_faq_support_has_at_least_two_personas(all_scenarios: list[Scenario]) -> None:
    count = sum(1 for s in all_scenarios if s.category == "faq_support")
    assert count >= 2, f"Only {count} faq_support scenario(s)"


# ---------------------------------------------------------------------------
# Noise / barge_in coverage per category
# ---------------------------------------------------------------------------


def test_noise_or_barge_in_in_each_category(all_scenarios: list[Scenario]) -> None:
    """Each category must have at least one scenario exercising noise or barge_in."""
    for cat in VALID_CATEGORIES:
        cat_scenarios = [s for s in all_scenarios if s.category == cat]
        has_noise_or_barge = any(
            s.noise.enabled or s.barge_in.enabled for s in cat_scenarios
        )
        assert has_noise_or_barge, (
            f"Category '{cat}' has no scenario with noise or barge_in enabled"
        )
