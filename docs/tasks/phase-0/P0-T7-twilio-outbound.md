# P0-T7 — TwilioOutbound target (the real call)

**Context pack:** `tech-stack.md` + `conventions.md` + `architecture.md` (ADR-2, ADR-6,
ADR-9) + `P0-T2` + this file.
**Depends on:** P0-T3, P0-T5

## Goal
Implement the `TwilioOutbound` `CallTarget`: originate a real outbound call and bridge
its audio to the Pipecat pipeline over a Twilio Media Streams WebSocket. This is the
**last** Phase 0 integration — everything else already works against LocalLoopback.

## Deliverables
- `voicetest/providers/targets.py` → add `TwilioOutbound`:
  - Originate via Twilio REST API to the destination number, returning TwiML with
    `<Connect><Stream url="{PUBLIC_WS_URL}"/>`.
  - A WebSocket server (FastAPI/uvicorn route or standalone) that Twilio connects to;
    bridge μ-law 8 kHz frames ↔ Pipecat transport (Pipecat ships a Twilio serializer —
    use it).
  - **ADR-9 guard:** refuse to dial any number not in `VERIFIED_DEST_NUMBERS`; raise a
    clear error before placing the call.
  - **No-answer / voicemail / error handling:** use Twilio Answering Machine Detection;
    map to `CallOutcome.NO_ANSWER / VOICEMAIL / ERROR` so they're not false test fails.
  - Enable Twilio **dual-channel recording**; pass the recording to the recorder/BlobStore.
- `voicetest/engine/ws_server.py`: the Media Streams WS endpoint that builds the
  pipeline (P0-T3) with this target and runs `run_call`.
- Docs note in README/setup: run `ngrok http <port>`, set `PUBLIC_WS_URL` to the wss URL.

## Implementation notes
- Reference `pipecat-ai/pipecat-quickstart-phone-bot` for the Twilio↔Pipecat WS bridge;
  invert it from answering to **originating** an outbound call.
- 8 kHz μ-law is expected (narrowband) — transcode only at the transport edge.
- Keep secrets in env (`config.py`). This target needs Twilio + Deepgram + Cartesia +
  LLM keys set.

## Tests to write
- `test_twilio_allowlist.py` (unit, mocked Twilio client): dialing a non-allowlisted
  number raises before any REST call; an allowlisted number proceeds.
- `test_outcome_mapping.py`: simulated AMD results map to the right `CallOutcome`.
- **Manual smoke (documented, not in CI):** dial your own verified number, confirm a
  recording + transcript + report come back.

## Definition of done
- [ ] A real outbound call connects, converses, and yields a `CallResult` + recording
- [ ] Un-verified numbers are refused in code
- [ ] No-answer/voicemail/error are first-class outcomes, not exceptions
- [ ] CI never places a real call (allowlist + outcome tests use mocks)
