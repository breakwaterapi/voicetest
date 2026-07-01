# Roadmap & Task Index

Build in order. Each task file is self-contained; attach `tech-stack.md` +
`conventions.md` + the task file (+ any listed dependency) to your LLM. Don't skip
the tests — they're part of each task's definition of done.

## Phase 0 — Prove the engine (the MVP you want now)
Goal: place ONE real call, behave like a difficult caller, record it, score it.
Magic moment: *"it caught a real bug and I have the recording."*
Telephony is the LAST step (ADR-2) — everything before it runs locally for free.

| Task | File | Depends on | Outcome |
|------|------|-----------|---------|
| P0-T1 | `tasks/phase-0/P0-T1-project-setup.md` | — | repo, config, deps, models, test harness |
| P0-T2 | `tasks/phase-0/P0-T2-call-targets.md` | T1 | CallTarget interface + LocalLoopback + TranscriptReplay |
| P0-T3 | `tasks/phase-0/P0-T3-caller-engine.md` | T1,T2 | Pipecat pipeline STT→LLM→TTS vs LocalLoopback |
| P0-T4 | `tasks/phase-0/P0-T4-scenario-loader.md` | T1 | YAML scenario loader + 1 hardcoded scenario |
| P0-T5 | `tasks/phase-0/P0-T5-recorder.md` | T3 | recorder processor + CallResult artifact |
| P0-T6 | `tasks/phase-0/P0-T6-scoring-mvp.md` | T5 | goal-completion (judge) + computed latency |
| P0-T7 | `tasks/phase-0/P0-T7-twilio-outbound.md` | T3,T5 | TwilioOutbound target — real outbound call |
| P0-T8 | `tasks/phase-0/P0-T8-cli.md` | all | `python -m voicetest.cli run` end-to-end + report.json |

**Phase 0 done when:** CLI runs the hardcoded scenario against a real phone number,
produces a recording + transcript + scored report; and the broken-agent fixture
(LocalLoopback) yields a correct *fail*.

## Phase 1 — The MVP product
Goal: 3 scenario categories with uncooperative personas, noise + barge-in, full
scoring (minus tool-call), a no-auth web page, shareable report URL.

| Task | File | Depends on | Outcome |
|------|------|-----------|---------|
| P1-T1 | `tasks/phase-1/P1-T1-scenario-library.md` | P0 | 3 categories × several difficult personas |
| P1-T2 | `tasks/phase-1/P1-T2-noise-injection.md` | P0-T3 | noise mixer processor + tests |
| P1-T3 | `tasks/phase-1/P1-T3-bargein-injection.md` | P0-T3 | barge-in scripter processor + tests |
| P1-T4 | `tasks/phase-1/P1-T4-full-scoring.md` | P0-T6 | 5 dims (minus tool-call) + structured report |
| P1-T5 | `tasks/phase-1/P1-T5-api-jobrunner.md` | P0-T8 | FastAPI + asyncio job runner + SQLite store |
| P1-T6 | `tasks/phase-1/P1-T6-web-ui.md` | P1-T5 | one page: paste #, pick scenario, run N, report |
| P1-T7 | `tasks/phase-1/P1-T7-shareable-report.md` | P1-T6 | unguessable report URL |

**Phase 1 done when:** you'd post the demo in the Vapi Discord without embarrassment.

## Phase 2 — Vapi API path (data unlock) — outline only, detail later
- Add Vapi assistant-ID + API-key path alongside phone path.
- Event stream → **tool-call verification** (closes the state-blind gap).
- Auto-fill scenario suggestions from the agent config.
- Instrument which entry point converts better.

## Phase 3 — Growth feature — outline only
- **GitHub Action** with `fail-threshold` on pass rate (the word-of-mouth feature).
- Saved/rerunnable suites; concurrent calls (`asyncio.gather` + semaphore); alerts.

## Phase 4 — Business — outline only
- Signup/auth, free tier + paid plans, team seats.
- Expand targets: Retell, Bland, ElevenLabs Conversational.

## Risks to keep designing around (from doc §7)
- People-pleaser personas → enforced by scenario-spec (difficult_behaviors required).
- Outbound consent → verified-number allowlist in TwilioOutbound (ADR-9).
- No-answer/voicemail → first-class CallResult outcomes, not false failures.
- Latency precision → measure on captured inbound timeline + VAD events only.
- Cost when parallelizing → watch per-call spend; stub everything in tests.
