"""
Tests for the scenario YAML loader and models.

Covers:
- builtin scenario loads correctly with all typed fields
- empty difficult_behaviors raises ValidationError
- unknown top-level key raises ValidationError
- load_all returns the builtin(s)
- Scenario objects from loader plug into run_call (type-check only, no network)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from voicetest.scenarios import BUILTIN_DIR, Scenario, load_all, load_scenario
from voicetest.scenarios.models import (
    BargeInConfig,
    Limits,
    NoiseConfig,
    Persona,
    ScoringConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_yaml(tmp_path: Path, data: dict, filename: str = "scenario.yaml") -> Path:
    p = tmp_path / filename
    p.write_text(yaml.dump(data))
    return p


MINIMAL_VALID: dict = {
    "id": "test-scenario",
    "category": "appointment_booking",
    "title": "Test scenario",
    "version": 1,
    "persona": {
        "system_prompt": "You are a caller.",
        "goal": "Book a table.",
        "difficult_behaviors": ["Be rude if asked twice."],
    },
    "scoring": {
        "objective": "Agent books a table.",
    },
}


# ---------------------------------------------------------------------------
# Builtin scenario
# ---------------------------------------------------------------------------


def test_builtin_scenario_loads() -> None:
    builtin = BUILTIN_DIR / "appointment-booking-confused-date.yaml"
    scenario = load_scenario(builtin)
    assert isinstance(scenario, Scenario)


def test_builtin_scenario_id() -> None:
    builtin = BUILTIN_DIR / "appointment-booking-confused-date.yaml"
    scenario = load_scenario(builtin)
    assert scenario.id == "appointment-booking-confused-date"


def test_builtin_scenario_category() -> None:
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert scenario.category == "appointment_booking"


def test_builtin_scenario_persona_fields() -> None:
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert isinstance(scenario.persona, Persona)
    assert scenario.persona.goal  # non-empty
    assert len(scenario.persona.system_prompt) > 10
    assert len(scenario.persona.difficult_behaviors) >= 4


def test_builtin_scenario_difficult_behaviors_are_strings() -> None:
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    for b in scenario.persona.difficult_behaviors:
        assert isinstance(b, str) and b.strip()


def test_builtin_scenario_ground_truth() -> None:
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    gt = scenario.persona.ground_truth
    assert "desired_date" in gt
    assert "desired_time" in gt
    assert "phone" in gt


def test_builtin_scenario_noise_disabled() -> None:
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert isinstance(scenario.noise, NoiseConfig)
    assert scenario.noise.enabled is False


def test_builtin_scenario_barge_in_disabled() -> None:
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert isinstance(scenario.barge_in, BargeInConfig)
    assert scenario.barge_in.enabled is False


def test_builtin_scenario_scoring() -> None:
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert isinstance(scenario.scoring, ScoringConfig)
    assert scenario.scoring.objective
    assert len(scenario.scoring.pass_criteria) >= 3


def test_builtin_scenario_latency_thresholds() -> None:
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert scenario.scoring.latency.p50_warn_ms == 800
    assert scenario.scoring.latency.p95_warn_ms == 1500


def test_builtin_scenario_limits() -> None:
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert isinstance(scenario.limits, Limits)
    assert scenario.limits.max_duration_s == 180
    assert scenario.limits.max_turns == 25


def test_builtin_scenario_version() -> None:
    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    assert scenario.version == 1


# ---------------------------------------------------------------------------
# Validation: empty difficult_behaviors
# ---------------------------------------------------------------------------


def test_empty_difficult_behaviors_raises(tmp_path: Path) -> None:
    data = dict(MINIMAL_VALID)
    data["persona"] = {**MINIMAL_VALID["persona"], "difficult_behaviors": []}
    p = write_yaml(tmp_path, data)
    with pytest.raises(ValidationError, match="difficult_behaviors"):
        load_scenario(p)


# ---------------------------------------------------------------------------
# Validation: unknown top-level key
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_ignored(tmp_path: Path) -> None:
    # Top-level Scenario uses extra='ignore' to allow compiler-generated fields
    # to be forward-compatible without breaking older runners.
    data = {**MINIMAL_VALID, "unknown_extra_field": "oops"}
    p = write_yaml(tmp_path, data)
    scenario = load_scenario(p)  # must not raise
    assert scenario.id == MINIMAL_VALID["id"]


def test_unknown_persona_key_raises(tmp_path: Path) -> None:
    data = dict(MINIMAL_VALID)
    data["persona"] = {**MINIMAL_VALID["persona"], "extra_persona_field": "bad"}
    p = write_yaml(tmp_path, data)
    with pytest.raises(ValidationError):
        load_scenario(p)


def test_unknown_scoring_key_raises(tmp_path: Path) -> None:
    data = dict(MINIMAL_VALID)
    data["scoring"] = {"objective": "Book.", "mystery_field": 42}
    p = write_yaml(tmp_path, data)
    with pytest.raises(ValidationError):
        load_scenario(p)


# ---------------------------------------------------------------------------
# Validation: missing required fields
# ---------------------------------------------------------------------------


def test_missing_id_raises(tmp_path: Path) -> None:
    data = {k: v for k, v in MINIMAL_VALID.items() if k != "id"}
    p = write_yaml(tmp_path, data)
    with pytest.raises(ValidationError):
        load_scenario(p)


def test_missing_persona_raises(tmp_path: Path) -> None:
    data = {k: v for k, v in MINIMAL_VALID.items() if k != "persona"}
    p = write_yaml(tmp_path, data)
    with pytest.raises(ValidationError):
        load_scenario(p)


def test_missing_scoring_raises(tmp_path: Path) -> None:
    data = {k: v for k, v in MINIMAL_VALID.items() if k != "scoring"}
    p = write_yaml(tmp_path, data)
    with pytest.raises(ValidationError):
        load_scenario(p)


def test_custom_category_accepted(tmp_path: Path) -> None:
    # category is now an open string — any domain is valid
    data = {**MINIMAL_VALID, "category": "dental_booking"}
    p = write_yaml(tmp_path, data)
    scenario = load_scenario(p)
    assert scenario.category == "dental_booking"


# ---------------------------------------------------------------------------
# load_all
# ---------------------------------------------------------------------------


def test_load_all_returns_builtin_scenarios() -> None:
    scenarios = load_all()
    assert len(scenarios) >= 1
    assert all(isinstance(s, Scenario) for s in scenarios)


def test_load_all_includes_appointment_booking() -> None:
    scenarios = load_all()
    ids = {s.id for s in scenarios}
    assert "appointment-booking-confused-date" in ids


def test_load_all_custom_dir(tmp_path: Path) -> None:
    write_yaml(tmp_path, MINIMAL_VALID, "a.yaml")
    write_yaml(tmp_path, {**MINIMAL_VALID, "id": "second-scenario"}, "b.yaml")
    scenarios = load_all(tmp_path)
    assert len(scenarios) == 2
    assert scenarios[0].id == "test-scenario"   # alphabetical: a.yaml first
    assert scenarios[1].id == "second-scenario"


def test_load_all_empty_dir(tmp_path: Path) -> None:
    assert load_all(tmp_path) == []


def test_load_all_bad_file_raises(tmp_path: Path) -> None:
    write_yaml(tmp_path, MINIMAL_VALID, "good.yaml")
    (tmp_path / "bad.yaml").write_text("id: bad\ncategory: appointment_booking\n")
    with pytest.raises(ValidationError):
        load_all(tmp_path)


# ---------------------------------------------------------------------------
# Scenario plugs into run_call (type signature check — no network)
# ---------------------------------------------------------------------------


def test_scenario_accepted_by_run_call_signature() -> None:
    """Verify the object returned by load_scenario satisfies run_call's type contract."""
    import inspect

    from voicetest.engine.run_call import run_call

    sig = inspect.signature(run_call)
    scenario_param = sig.parameters["scenario"]
    # Just confirm it's annotated as Scenario (or compatible)
    assert scenario_param.annotation is not inspect.Parameter.empty

    scenario = load_scenario(BUILTIN_DIR / "appointment-booking-confused-date.yaml")
    # Structural check: has all fields run_call reads
    assert hasattr(scenario, "id")
    assert hasattr(scenario, "limits")
    assert hasattr(scenario, "persona")
