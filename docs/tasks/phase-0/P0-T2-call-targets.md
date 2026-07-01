# P0-T2 — CallTarget interface + LocalLoopback + TranscriptReplay

**Context pack:** `tech-stack.md` + `conventions.md` + `architecture.md` (ADR-3) + this file.
**Depends on:** P0-T1

## Goal
Define the `CallTarget` interface and the two non-telephony implementations the rest
of Phase 0 develops and tests against. (TwilioOutbound comes in P0-T7.)

## Deliverables
- `voicetest/providers/targets.py`:
  - `CallTarget` Protocol: `async connect() -> AudioStream`, `async disconnect()`,
    plus a way for the engine to read inbound audio and write outbound audio. Model it
    on whatever transport abstraction Pipecat expects so it drops into the pipeline in
    P0-T3 — i.e. the target yields a Pipecat-compatible transport/frame source.
  - `LocalLoopback`: an in-process "agent under test". Takes a pluggable
    `agent_fn(transcript) -> reply_text` (default: a simple cooperative booking agent).
    Wraps stub STT/TTS so the engine talks to it as if over a wire, fully in-memory.
  - `TranscriptReplay`: constructed from a list of `Turn`s; produces no live audio —
    it exists so the scorer (P0-T6) can run on canned conversations with no engine.

## Implementation notes
- `LocalLoopback` must support an injectable agent so the **broken-agent fixture**
  (slow / ignores barge-in / mis-hears) can be supplied in tests — see
  `testing-strategy.md`.
- Keep audio representation consistent (PCM frame dataclass + sample rate) so noise
  (P1) can be mixed without conversions.
- No network, no real STT/TTS in either implementation.

## Tests to write
- `test_local_loopback.py`: drive a few caller utterances, assert the default agent
  produces coherent replies and the loop terminates.
- `test_transcript_replay.py`: build from canned `Turn`s, assert it exposes them
  unchanged for the scorer.
- A fixture factory `make_broken_agent()` in `tests/fixtures` returning an `agent_fn`
  that is slow + ignores interruptions (used later, define it now).

## Definition of done
- [ ] Engine in P0-T3 can accept any `CallTarget` without code changes
- [ ] Both impls fully in-memory; tests have no network
- [ ] broken-agent fixture factory exists
