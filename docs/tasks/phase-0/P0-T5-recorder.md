# P0-T5 — Recorder processor + CallResult artifact

**Context pack:** `tech-stack.md` + `conventions.md` + `architecture.md` (ADR-6) + this file.
**Depends on:** P0-T3

## Goal
Capture everything needed for scoring: per-turn transcript with timestamps, an event
log, and the call audio. Produce a complete `CallResult` + a persisted artifact.

## Deliverables
- `voicetest/engine/processors/recorder.py`: a Pipecat processor inserted in the
  pipeline that, **on a single captured timeline**:
  - records each `Turn` (speaker, text, start_ms, end_ms),
  - logs `Event`s: vad_start/vad_stop on the inbound (agent) track, turn boundaries
    (and later noise_on/off, interruption — hooks for P1),
  - captures outbound+inbound PCM frames to a WAV (local dev). Prefer **dual-channel**
    (caller=left, agent=right) so attribution is trivial.
- `voicetest/store.py` (minimal): `BlobStore` writing recordings to `VOICETEST_BLOB_DIR`,
  and a function to serialize a `CallResult` to JSON next to its recording.
- Wire the recorder into `run_call` so it returns a fully-populated `CallResult`
  (turns, events, recording_path, duration_ms).

## Implementation notes
- **Timeline discipline (latency precision):** all `*_ms` are relative to call start on
  *your* captured audio timeline. Never use wall-clock across a phone bridge.
- VAD on the **inbound** track gives the ground-truth boundary for "agent started
  speaking" — log it as an event; it's what P0-T6 latency and P1 barge-in consume.
- For TwilioOutbound later, audio also comes from Twilio dual-channel recording; the
  recorder still owns timestamps/events. Keep the audio source pluggable.

## Tests to write (integration, no network)
- `test_recorder_loopback.py`: run `run_call` against LocalLoopback with stub TTS that
  emits known-length PCM; assert turns have monotonic non-overlapping timestamps, a WAV
  is written, and `CallResult.recording_path` exists.
- `test_callresult_json.py`: serialize/reload a CallResult round-trips.

## Definition of done
- [ ] `run_call` returns a complete `CallResult` (turns + events + recording + duration)
- [ ] All timestamps on one captured timeline; inbound VAD events logged
- [ ] WAV + CallResult JSON persisted via BlobStore
