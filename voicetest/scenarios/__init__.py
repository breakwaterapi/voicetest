"""
voicetest.scenarios — public API.

Models live in .models; the loader in .loader.
Everything is re-exported here so existing imports from `voicetest.scenarios`
continue to work unchanged.
"""

from voicetest.scenarios.loader import BUILTIN_DIR, load_all, load_scenario
from voicetest.scenarios.models import (
    BargeInConfig,
    LatencyThresholds,
    Limits,
    NoiseConfig,
    Persona,
    Scenario,
    ScoringConfig,
)

__all__ = [
    # models
    "BargeInConfig",
    "LatencyThresholds",
    "Limits",
    "NoiseConfig",
    "Persona",
    "Scenario",
    "ScoringConfig",
    # loader
    "BUILTIN_DIR",
    "load_scenario",
    "load_all",
]

# ---------------------------------------------------------------------------
# Built-in Phase-0 scenario (programmatic, no file I/O)
# ---------------------------------------------------------------------------

BOOKING_SCENARIO = Scenario(
    id="appointment-booking-confused-date",
    category="appointment_booking",
    title="Books a haircut but gives the wrong date first",
    version=1,
    persona=Persona(
        system_prompt=(
            "You are calling a hair salon to book a haircut. You are a real, slightly "
            "distracted human, NOT a helpful assistant. Speak in short, natural phone "
            "sentences. Do not be articulate or organized."
        ),
        goal="Book a haircut for next Saturday afternoon.",
        difficult_behaviors=[
            "Give the WRONG date first ('this Saturday'), then correct yourself once.",
            "Mumble the phone number the first time you say it.",
            "Get impatient and sigh if the agent takes more than a few seconds.",
            "Ask one out-of-scope question (do they sell shampoo?).",
        ],
        ground_truth={
            "desired_date": "next Saturday",
            "desired_time": "afternoon",
            "phone": "617-555-0142",
        },
    ),
    scoring=ScoringConfig(
        objective="The agent correctly books: next Saturday, afternoon, phone 617-555-0142.",
        pass_criteria=[
            "Agent captured the corrected date (next Saturday, not this Saturday).",
            "Agent confirmed the time as afternoon.",
            "Agent read back or confirmed the phone number correctly.",
        ],
    ),
    limits=Limits(max_duration_s=180, max_turns=25),
)
