# P0-T4 — Scenario YAML loader + one hardcoded scenario

**Context pack:** `tech-stack.md` + `conventions.md` + `scenario-spec.md` + this file.
**Depends on:** P0-T1

## Goal
Load + validate scenario YAML files into typed `Scenario` objects, and ship one
complete appointment-booking scenario to drive Phase 0 end-to-end.

## Deliverables
- `voicetest/scenarios/models.py`: pydantic `Scenario`, `Persona`, `NoiseConfig`,
  `BargeInConfig`, `Scoring`, `Limits` matching `scenario-spec.md` exactly.
- `voicetest/scenarios/loader.py`: `load_scenario(path) -> Scenario` and
  `load_all(dir) -> list[Scenario]`. Validates and **fails loudly** on unknown/missing
  fields. Enforces the rule: `difficult_behaviors` must be non-empty (guards against
  people-pleaser personas).
- `voicetest/scenarios/builtin/appointment-booking-confused-date.yaml`: a full,
  difficult appointment-booking scenario (use the example in `scenario-spec.md` as the
  starting point; `noise`/`barge_in` disabled for Phase 0).

## Tests to write
- `test_scenario_loader.py`:
  - loads the builtin scenario; asserts fields parse and types are correct.
  - a YAML with empty `difficult_behaviors` raises a validation error.
  - a YAML with an unknown top-level key raises.
  - `load_all` returns all builtins.

## Definition of done
- [ ] Builtin scenario loads into a typed `Scenario`
- [ ] Validation rejects people-pleaser (empty difficult_behaviors) and unknown fields
- [ ] `Scenario` plugs directly into `run_call` (P0-T3)
