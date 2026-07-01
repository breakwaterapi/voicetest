# P1-T1 — Scenario library (3 categories × difficult personas)

**Context pack:** `scenario-spec.md` + `conventions.md` + this file.
**Depends on:** Phase 0 (loader exists)

## Goal
Author the launch scenario library: 3 categories, several **uncooperative** personas
each. This is the product's top quality lever (doc §4.1) — personas must induce failure,
not cooperate.

## Deliverables
- `voicetest/scenarios/builtin/*.yaml` — for each category below, 2–3 personas:
  - **appointment_booking**: wrong-date-then-correct, mumbled phone number, impatient
    customer, double-books / changes time mid-call.
  - **lead_qualification**: evasive/vague budget, changes mind, off-topic tangents,
    reluctant to give contact info.
  - **faq_support**: out-of-scope question, repeats a misunderstood question, talks over
    the agent (barge_in enabled), angry/curt tone.
- Each YAML: full `persona` with non-empty `difficult_behaviors`, `ground_truth`, and
  concrete `scoring.pass_criteria`. Enable `noise`/`barge_in` on a subset.

## Tests to write
- `test_library_valid.py`: `load_all` parses every builtin; each has ≥2
  difficult_behaviors and ≥1 pass_criteria; ids unique; categories valid.

## Definition of done
- [ ] ≥3 categories, ≥6 personas total, all loading and validating
- [ ] Every persona is demonstrably difficult (reviewed against §4.1, not people-pleasing)
- [ ] At least one persona each exercises noise and barge-in
