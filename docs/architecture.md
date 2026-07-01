# Architecture & Design Decisions

Read once for orientation. Per-task LLM context comes from `tech-stack.md` +
`conventions.md` + the task file — not this document.

## Core idea

The simulated caller **is** a voice agent: it listens (STT), thinks (persona LLM),
and speaks (TTS) in real time over a phone line — exactly like the agents under test.
The differences: (1) its prompt makes it a *difficult human caller*, (2) it injects
noise and interruptions on purpose, (3) everything is timestamped and recorded for
offline scoring.

## System diagram (MVP)

```
Web page / CLI ──► FastAPI + asyncio job runner
                         │ per call: run_call(scenario, target) -> CallResult
                         ▼
┌──────────────────── CALLER ENGINE (Pipecat) ───────────────────────┐
│  CallTarget (interface)                                             │
│   ├─ LocalLoopback     (dev: stub agent, no telephony, no cost)     │
│   ├─ TranscriptReplay  (test: canned transcript, scoring only)      │
│   └─ TwilioOutbound    (prod: real phone via Media Streams WS)      │
│                                                                     │
│  Transport ─► STT ─► Persona LLM ─► TTS ─► [Barge-in] ─► [Noise] ─► Transport
│                                                                     │
│  Recorder processor: PCM frames + per-turn timestamps + event log   │
└─────────────────────────────────────────────────────────────────────┘
                         │ on call end -> CallResult artifact
                         ▼
        Scorer (offline): 1 structured judge call + computed metrics
                         │
                         ▼
                      Report
```

A **test run = N independent calls**. Each call is one pure coroutine
`run_call(scenario, target) -> CallResult` with no shared state. Sequential in
Phase 0/1; parallelize later with `asyncio.gather` + semaphore.

## Key decisions (ADR-style)

### ADR-1: Pipecat as orchestration framework
Pipeline-first, lets us insert custom processors (noise, barge-in, recorder)
anywhere. Zero local infra. 60+ swappable providers. Chosen over LiveKit
(infra-first; overhead for single-caller→single-agent).

### ADR-2: Build engine without telephony first
The full caller loop + noise + barge-in + recorder + scoring is developed and
tested against `LocalLoopback` before Twilio is wired. Real calls cost money and
iterate slowly. **Twilio is the last integration of Phase 0.** This is the biggest
cost/speed lever in the whole build.

### ADR-3: CallTarget interface (3 implementations)
`LocalLoopback` (dev stub agent), `TranscriptReplay` (scoring tests), `TwilioOutbound`
(real call). The engine never knows which it's talking to. This is the core
testability decision — it makes a voice product unit-testable.

### ADR-4: Scenarios are declarative config, not code
One YAML per scenario: persona prompt, difficulty behaviors, noise profile, barge-in
script, success objective, scoring rubric. Add scenarios with zero code. Judge reads
the exact rubric. See `scenario-spec.md`.

### ADR-5: Deepgram Flux for caller-side turn-taking
Built-in end-of-turn detection replaces a separate VAD on the caller side, removing
a class of turn-taking bugs. (VAD on the *inbound* track is still needed to detect
when the agent starts speaking, for barge-in + latency.)

### ADR-6: Recording via Twilio dual-channel; timestamps via Pipecat
Dual-channel recording gives cleanly separated agent vs caller tracks server-side —
reliable attribution for scoring. Pipecat's recorder processor owns per-turn
timestamps + the event log. Local dev records PCM frames directly.

### ADR-7: One judge call per call
Latency, barge-in timing, and noise ground-truth-diff are **computed**, not judged.
Only goal-completion + noise-interpretation use the LLM, folded into a single
structured-JSON judge call. ~3–5× cheaper than per-dimension calls.

### ADR-8: Provider-agnostic LLM/STT/TTS via interfaces
Pinned defaults in `tech-stack.md`, but every provider sits behind a thin interface
so swapping is a config change. Models/pricing in this space reset ~quarterly.

### ADR-9: Compliance allowlist in code
`TwilioOutbound` refuses to dial numbers not on a verified allowlist. Survives the
upgrade off the Twilio trial (which enforces this for free).

## Known gaps (intentional for MVP)

- **Tool-call accuracy is deferred** — phone-only testing is "state-blind": we see
  what was *said*, not whether the backend action fired. Closing this is the headline
  reason for the Phase 2 Vapi API path.
- **No auth / no billing** — deferred to Phase 4. Reports are shared via unguessable URL.
