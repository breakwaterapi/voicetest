# P0-T3 — Caller engine (Pipecat pipeline) against LocalLoopback

**Context pack:** `tech-stack.md` + `conventions.md` + `architecture.md` + this file.
**Depends on:** P0-T1, P0-T2

## Goal
Build the synthetic-caller pipeline — STT → persona LLM → TTS — as a Pipecat pipeline,
and run a full conversation against `LocalLoopback` with **no telephony**. This is the
heart of the engine.

## Deliverables
- `voicetest/providers/stt.py`, `tts.py`, `llm.py`: thin wrappers behind the
  interfaces in `tech-stack.md`, each with:
  - a real provider impl (Deepgram Flux STT, Cartesia TTS, configurable LLM), AND
  - a **stub impl** (`StubSTT/StubTTS/StubLLM` in `tests/fixtures`) for tests.
  - Provider chosen via `config.py` (model from `LLM_CALLER_MODEL`).
- `voicetest/engine/pipeline.py`: `build_pipeline(scenario, target, providers)` wiring
  Transport(from target) → STT → LLM(persona system prompt) → TTS → Transport.
- `voicetest/engine/run_call.py`: `async run_call(scenario, target, providers=None)
  -> CallResult`. Stateless. Enforces `scenario.limits` (max_duration, max_turns).
  Returns a partial CallResult (recorder fills the rest in P0-T5).
- Persona prompt assembly: combine `persona.system_prompt` + `difficult_behaviors`
  into the caller LLM system prompt (the difficult behaviors must be present).

## Implementation notes
- Use **Deepgram Flux** end-of-turn for caller-side turn detection (ADR-5); don't add
  a separate caller VAD.
- The persona LLM drives toward `persona.goal` while exhibiting difficult behaviors —
  resist the people-pleaser default (testing-strategy + scenario-spec).
- Keep the system prompt tight (latency). Caller latency should not be the bottleneck.
- `run_call` must map connect failures / no-answer to `CallOutcome`, never raise.

## Tests to write (integration, no network)
- `test_engine_loopback.py`: run `run_call` with stub providers against `LocalLoopback`
  + a hardcoded mini-scenario; assert a multi-turn `CallResult` with alternating
  caller/agent turns and `outcome=COMPLETED`.
- `test_persona_prompt.py`: assert assembled system prompt contains every
  `difficult_behaviors` entry and the goal.
- `test_limits.py`: a scenario with `max_turns: 2` terminates after 2 turns.

## Definition of done
- [ ] Full STT→LLM→TTS loop runs against LocalLoopback with stubs, no network
- [ ] Swapping providers is config-only
- [ ] `run_call` is stateless and never raises on call failure (returns outcome)
