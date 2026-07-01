# Scenario / Persona Spec (declarative config)

A scenario is one YAML file. It fully describes a synthetic caller + how to score the
call. **No code changes to add a scenario.** The loader validates against this schema
(pydantic) and fails loudly on unknown/missing fields.

The product's core quality lever (doc §4.1): personas must be **uncooperative**.
LLMs default to people-pleasing — clear, helpful, on-script. That tests an agent
against a caller who doesn't exist. Every persona MUST specify explicit difficult
behaviors.

## Schema

```yaml
id: appointment-booking-confused-date      # unique, kebab-case
category: appointment_booking              # appointment_booking | lead_qualification | faq_support
title: "Books a haircut but gives the wrong date first"
version: 1

persona:
  # The system prompt for the caller-brain LLM. Be specific and difficult.
  system_prompt: |
    You are calling a hair salon to book a haircut. You are a real, slightly
    distracted human, NOT a helpful assistant. Speak in short, natural phone
    sentences. Do not be articulate or organized.
  goal: "Book a haircut for next Saturday afternoon."
  # Explicit difficulty behaviors — the heart of the test. Engine/persona prompt
  # weaves these in. Keep them concrete and observable.
  difficult_behaviors:
    - "Give the WRONG date first ('this Saturday'), then correct yourself once."
    - "Mumble the phone number the first time you say it."
    - "Get impatient and sigh if the agent takes more than a few seconds."
    - "Ask one out-of-scope question (do they sell shampoo?)."

  ground_truth:        # what you KNOW the caller actually means (for noise scoring)
    desired_date: "next Saturday"
    desired_time: "afternoon"
    phone: "617-555-0142"

# Optional — Phase 1 features. Omit for Phase 0 scenarios.
noise:
  enabled: false
  profile: cafe        # cafe | street | car  (clips in tests/fixtures or assets)
  level_db: -12        # mix level relative to caller voice
barge_in:
  enabled: false
  # interrupt the agent N ms after it STARTS speaking, on the Kth agent turn
  trigger_after_agent_speaks_ms: 800
  on_agent_turn: 2
  caller_says: "Wait, actually — can you make it Sunday instead?"

# Scoring rubric the judge reads verbatim. Keep objectives binary where possible.
scoring:
  objective: "The agent correctly books: next Saturday, afternoon, phone 617-555-0142."
  pass_criteria:
    - "Agent captured the corrected date (next Saturday, not this Saturday)."
    - "Agent confirmed the time as afternoon."
    - "Agent read back or confirmed the phone number correctly."
  # latency thresholds for the COMPUTED latency dimension (ms)
  latency:
    p50_warn_ms: 800
    p95_warn_ms: 1500

# Call control
limits:
  max_duration_s: 180
  max_turns: 25
```

## Field rules
- `difficult_behaviors` is **required** and must be non-empty. The loader rejects a
  persona with zero difficult behaviors (guards against people-pleaser scenarios).
- `ground_truth` is required for any scenario where noise scoring is meaningful; it's
  the source of truth the judge compares the agent's understanding against.
- `noise` / `barge_in` blocks are optional and default to `enabled: false`.
- `scoring.pass_criteria` are the exact bullets the judge evaluates; phrase each as a
  checkable statement.

## Built-in scenarios (Phase 1 target: 3 categories)
- `appointment_booking` — booking with corrections, mumbled details, impatience.
- `lead_qualification` — evasive/vague answers, changes mind, off-topic tangents.
- `faq_support` — asks out-of-scope, interrupts, repeats a misunderstood question.

Phase 0 ships **one** hardcoded scenario (an appointment-booking persona) to prove
the loop end-to-end.
