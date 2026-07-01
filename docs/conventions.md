# Conventions — repo layout, code style, testing rules

**Attach this to every build task.** These rules are binding for all generated code.

## Repo layout (target)

```
voicetest/
├── voicetest/                  # the package
│   ├── __init__.py
│   ├── config.py               # pydantic-settings, loads env
│   ├── providers/              # thin provider interfaces + impls
│   │   ├── targets.py          # CallTarget: LocalLoopback, TranscriptReplay, TwilioOutbound
│   │   ├── stt.py  tts.py  llm.py  judge.py
│   ├── engine/
│   │   ├── pipeline.py         # builds the Pipecat pipeline
│   │   ├── processors/         # custom processors
│   │   │   ├── recorder.py     # PCM + per-turn timestamps + event log
│   │   │   ├── noise.py        # noise-bed mixer (Phase 1)
│   │   │   └── bargein.py      # interruption scripter (Phase 1)
│   │   └── run_call.py         # run_call(scenario, target) -> CallResult
│   ├── scenarios/              # *.yaml scenario configs + loader
│   ├── scoring/
│   │   ├── compute.py          # latency, barge-in, noise-diff (no LLM)
│   │   └── judge.py            # single structured judge call
│   ├── models.py               # dataclasses/pydantic: Scenario, CallResult, Report...
│   ├── store.py                # Store + BlobStore (SQLite + disk)
│   ├── api.py                  # FastAPI app (Phase 1)
│   └── cli.py                  # `python -m voicetest.cli ...`
├── tests/
│   ├── unit/                   # fast, no network, no audio HW
│   ├── integration/            # local loopback end-to-end, no telephony
│   └── fixtures/               # canned transcripts, broken-agent stub, noise clips
├── scenarios/                  # user-facing scenario YAMLs (also voicetest/scenarios for builtins)
├── recordings/                 # gitignored
├── .env.example  .env(gitignored)
├── pyproject.toml / requirements.txt
└── README.md
```

## Code style
- Type hints everywhere. `ruff` for lint+format, `mypy` for types (advisory).
- Dataclasses or pydantic for all data shapes (`models.py`). No loose dicts crossing
  module boundaries.
- Pure functions where possible. The engine's `run_call` must be **stateless** and
  return a `CallResult` — no globals, so N calls parallelize trivially.
- Errors over silent failures: no-answer / voicemail / connect-error are *first-class
  CallResult outcomes*, never exceptions that look like a failed test.
- Provider impls behind the interfaces in `tech-stack.md`. Engine code imports the
  interface, never a concrete provider.

## Testing rules (NON-NEGOTIABLE — every task ships with tests)
1. **No real phone calls in tests.** Tests run against `LocalLoopback` or
   `TranscriptReplay` only. CI never dials Twilio.
2. Each task's "Tests to write" section is part of its definition of done.
3. **Unit tests** mock providers; **integration tests** run the real pipeline against
   the local loopback with deterministic stub STT/LLM/TTS.
4. The scorer is validated against the **broken-agent fixture** (a deliberately bad
   loopback agent) — a green score on a known-bad agent is a test failure.
5. Determinism: judge/LLM calls in tests use canned responses or temp 0 + fixed seed
   where the provider supports it; assert on structure, not exact prose.
6. `pytest` + `pytest-asyncio`. Fast suite (`tests/unit`) must run < 5s with no network.

## Git / commits
- Conventional-commit style messages. One task ≈ one focused commit/PR.
- Never commit `.env`, recordings, or API keys.

## Definition of done (applies to every task)
- [ ] Deliverable files created per the task
- [ ] Tests written and passing (`pytest`)
- [ ] `ruff` clean
- [ ] No real external calls in the test path
- [ ] Task's explicit acceptance criteria met
