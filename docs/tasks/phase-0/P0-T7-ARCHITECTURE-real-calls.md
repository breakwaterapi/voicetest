# Architecture: Real Phone Calls + Live Conversing LLM

This is the design for the **one missing integration** that lets you dial a real
number (e.g. `+18573132352`) and have the synthetic caller LLM actually converse
over the phone. Read this before building `TwilioOutbound`.

---

## TL;DR — what's already done vs. what's missing

| Layer | Status | Where |
|-------|--------|-------|
| Persona LLM (the "brain") | ✅ **Done, real** | `providers/llm.py` → `OpenAILLM`, `AnthropicLLM` |
| STT (hear the agent) | ✅ **Done, real** | `providers/stt.py` → `DeepgramSTT` |
| TTS (caller's voice) | ✅ **Done, real** | `providers/tts.py` → `CartesiaTTS` |
| Conversation loop (STT→LLM→TTS) | ✅ **Done** | `engine/pipeline.py` |
| Noise / barge-in / recorder | ✅ **Done** | `engine/processors/` |
| Scoring + report | ✅ **Done** | `scoring/` |
| `CallTarget` interface | ✅ **Done** | `providers/targets.py` |
| `LocalLoopback` (fake call) | ✅ **Done** | `providers/targets.py` |
| **`TwilioOutbound` (real call)** | ❌ **MISSING** | this doc |
| **Media Streams WS server** | ❌ **MISSING** | this doc |
| **`_make_target` wiring for real #s** | ❌ **MISSING** | `jobs.py:37` raises today |

**The brain already converses.** What's missing is the *pipe* that carries phone
audio in and out of the existing pipeline. That's it. ~1 file + 1 WS endpoint +
3 lines of wiring.

---

## Why "the dropdown isn't working"

The dropdown is fine — `/scenarios` returns all 9. The real failure: you pick a
scenario, type `+18573132352`, hit Run. The run is accepted (HTTP 202) but the
background job dies immediately because `jobs.py:_make_target()` only handles
loopback numbers and raises `ValueError` for any real number. There is no code
path that places a real call. This doc fills that gap.

---

## The core problem: audio transport mismatch

Everything downstream of the transport speaks the project's in-memory
`AudioStream` (see `providers/targets.py`):

```
AudioStream:
  put_outbound(frame)   # engine → target  (the caller's synthesized voice)
  inbound_frames()      # target → engine  (the agent's voice, to transcribe)
```

The engine just reads/writes `AudioFrame`s on this queue. `LocalLoopback` fakes
it in-process. **`TwilioOutbound` must implement the same interface but back it
with a real phone call** — frames come from / go to Twilio over a WebSocket.

Two impedance mismatches to bridge:

1. **Who initiates.** Twilio Media Streams is *inbound*: Twilio dials the number,
   then **connects back to your WebSocket server**. So a real call needs a
   running WS server that Twilio can reach (via ngrok in dev). `connect()` can't
   just return a stream synchronously — it must (a) start/lookup the WS server,
   (b) place the REST call telling Twilio to stream to that WS, then (c) hand back
   an `AudioStream` that the WS handler feeds once Twilio connects.

2. **Audio codec.** The phone channel is **8 kHz μ-law (G.711)**, base64-encoded
   in Twilio's JSON WS frames. Deepgram is configured for `linear16` @ 8 kHz
   (`stt.py:61`), and Cartesia must emit the same. So at the transport edge we
   transcode **μ-law ↔ PCM16** in both directions. Nowhere else.

---

## Target architecture (real call)

```
                    ┌──────── your machine ────────┐
  POST /runs        │                               │
  (number)          │   jobs.execute_run            │
      │             │      └─ run_call(scn, target) │
      ▼             │            target = TwilioOutbound(number)
 FastAPI api.py ────┤                 │             │
                    │       1. connect():           │
                    │          - allowlist guard (ADR-9)
                    │          - REST: create call, TwiML <Connect><Stream
                    │            url=PUBLIC_WS_URL/twilio-stream/{call_sid}>
                    │          - register a pending AudioStream for call_sid
                    │          - return that AudioStream
                    │                 │             │
   Twilio Cloud ◄───┼── REST dial ────┘             │
        │           │                               │
        │ rings +18573132352, person answers        │
        │           │                               │
        └── WS ────►│  /twilio-stream/{sid}  (ws_server.py, FastAPI WS route)
   (μ-law 8k base64)│      ├─ recv: Twilio "media" → μ-law→PCM16 → stream.put_inbound
                    │      └─ send: stream.outbound_frames → PCM16→μ-law → Twilio
                    │                 │             │
                    │     (the SAME engine pipeline runs unchanged:)
                    │     inbound PCM → DeepgramSTT → PersonaLLM → CartesiaTTS → outbound PCM
                    └───────────────────────────────┘
                                      │ call ends
                                      ▼
                            CallResult → score → report
```

Key property (ADR-3): **the engine never knows it's a real call.** Same
`run_call`, same pipeline, same scoring. Only the `CallTarget` implementation
changes.

---

## Components to build

### 1. `TwilioOutbound` (in `providers/targets.py`)

Implements `CallTarget` (`connect`/`disconnect`).

```
class TwilioOutbound:
    def __init__(self, number, settings, registry): ...

    async def connect(self) -> AudioStream:
        # ADR-9: refuse non-allowlisted numbers BEFORE any REST call
        guard_allowlist(self.number, self.settings)
        stream = AudioStream()
        call_sid_placeholder = uuid4()           # correlate REST call ↔ WS
        registry.register(call_sid_placeholder, stream)
        twiml = f'<Response><Connect><Stream url="{PUBLIC_WS_URL}/twilio-stream/{id}"/></Connect></Response>'
        # Twilio REST: client.calls.create(to=number, from_=FROM, twiml=twiml,
        #              record=True, recording_channels="dual",
        #              machine_detection="Enable")   # AMD → outcome mapping
        # Store the real call_sid; await first WS frame OR timeout(no-answer).
        return stream

    async def disconnect(self):
        # close stream, hang up via REST if still live, fetch recording URL
```

Responsibilities:
- **Allowlist guard (ADR-9):** raise before dialing if number ∉ `VERIFIED_DEST_NUMBERS`.
  (Move the existing check out of `api.py:_allowlist_check` into a shared helper so
  both the API and the target enforce it.)
- **Originate** via Twilio REST with `<Connect><Stream>` TwiML.
- **Outcome mapping (not exceptions):** use Twilio Answering Machine Detection →
  map to `CallOutcome.NO_ANSWER / VOICEMAIL / ERROR`. A voicemail pickup must NOT
  count as a failed *test* — it's a NO_ANSWER-class outcome.
- **Recording:** enable `record=True, recording_channels="dual"`; on disconnect,
  fetch the recording URL and hand it to the recorder/BlobStore so it shows up in
  the report's `<audio>` player.

### 2. `ws_server.py` — the Media Streams WebSocket endpoint

A FastAPI WebSocket route (mount on the **same** uvicorn app as `api.py`, or a
sibling app — same process so it shares the in-memory `registry`).

```
@app.websocket("/twilio-stream/{stream_id}")
async def twilio_stream(ws, stream_id):
    await ws.accept()
    stream = registry.get(stream_id)          # the AudioStream connect() made
    # Twilio sends JSON: {"event":"start"/"media"/"stop", "media":{"payload": b64 μ-law}}
    async def pump_in():   # Twilio → engine
        async for msg in ws.iter_json():
            if msg["event"] == "media":
                pcm = ulaw_to_pcm16(b64decode(msg["media"]["payload"]))
                await stream.put_inbound(AudioFrame(pcm))
            elif msg["event"] == "stop":
                await stream.close_inbound(); break
    async def pump_out():  # engine → Twilio
        async for frame in stream.outbound_frames():
            payload = b64encode(pcm16_to_ulaw(frame.samples))
            await ws.send_json({"event":"media","streamSid":sid,"media":{"payload":payload}})
    await asyncio.gather(pump_in, pump_out)
```

This is the *only* new place audio codecs live. `audioop.lin2ulaw` /
`audioop.ulaw2lin` (stdlib, or `audioop-lts` on 3.13+) handle the transcode.

> **Pipecat alternative:** Pipecat ships a `FastAPIWebsocketTransport` +
> `TwilioFrameSerializer` that does the μ-law bridge for you. The repo's pipeline
> is currently hand-rolled (`engine/pipeline.py`), *not* real Pipecat, despite
> ADR-1. Two honest options:
> - **(A) Minimal:** keep the hand-rolled pipeline, write the ~40-line WS bridge
>   above. Least new surface, you already understand the pipeline. **Recommended
>   to ship first.**
> - **(B) Pipecat-native:** port the pipeline to a real Pipecat `Pipeline` and use
>   its Twilio serializer. More idiomatic, more code churn. Do this later if/when
>   you want Pipecat's VAD, interruption handling, and metrics for free.

### 3. Wire `_make_target` (in `jobs.py`)

Replace the `raise ValueError` at `jobs.py:48` with:

```
return TwilioOutbound(number, settings, registry)
```

Pass `settings` + the shared `registry` into `execute_run` (or construct the
target via a factory the API injects, mirroring the existing `target_factory`
param — that hook already exists at `jobs.py:60`).

### 4. UI / API honesty (small, do regardless)

Today a real-number run silently fails in the background. At minimum:
- Surface `run.status == "failed"` + the error in the poll response (it's already
  stored — the UI just shows "failed" with no reason). Add an `error` field to
  `RunStatusResponse` so the page can display *why*.
- Until `PUBLIC_WS_URL` is set, `TwilioOutbound.connect()` should raise a clear
  "telephony not configured — set PUBLIC_WS_URL / run ngrok" message, mapped to a
  422 the UI shows.

---

## Audio format cheat-sheet (don't fight these)

| Edge | Format |
|------|--------|
| Twilio Media Streams WS | μ-law 8 kHz, base64 in JSON `media.payload` |
| Internal `AudioFrame` | PCM linear16, 8 kHz, mono (`SAMPLE_RATE=8000`) |
| Deepgram input | `encoding="linear16", sample_rate=8000` (already set, `stt.py:61`) |
| Cartesia output | must emit PCM16 @ 8 kHz to match |

Transcode **only** at the WS edge: `ulaw→pcm16` inbound, `pcm16→ulaw` outbound.

---

## Setup / runbook (dev)

1. **Env** (`.env.local`): `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`,
   `TWILIO_FROM_NUMBER` (a Twilio number you own), `VERIFIED_DEST_NUMBERS`
   (include `+18573132352`), `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, an LLM key,
   `LLM_CALLER_MODEL`.
2. **Tunnel:** `ngrok http 8000` → copy the `wss://…ngrok…` URL into
   `PUBLIC_WS_URL`.
3. **Run:** `uvicorn voicetest.api:app --port 8000` (the WS route lives on the
   same app).
4. **Smoke test:** open the UI, pick a scenario, enter your verified number, Run.
   Your phone rings; the persona LLM converses; on hangup you get transcript +
   recording + score.

> On a Twilio **trial** account you can only dial numbers you've verified in the
> console — which conveniently matches ADR-9's allowlist. `+18573132352` must be
> verified there too.

---

## Failure modes to handle (so they're outcomes, not crashes)

| Situation | Detect | Map to |
|-----------|--------|--------|
| Nobody answers | Twilio call status `no-answer`/`busy` | `CallOutcome.NO_ANSWER` |
| Voicemail/machine | Twilio AMD `machine_*` | `CallOutcome.VOICEMAIL` |
| WS never connects | timeout in `connect()` | `CallOutcome.ERROR` + clear msg |
| `PUBLIC_WS_URL` unset | check in `connect()` | 422 to UI, never dial |
| Non-allowlisted number | allowlist guard | raise before REST (ADR-9) |

`run_call.py` already wraps the call and maps exceptions to `CallOutcome` — feed
these in via the target rather than letting them bubble as raw errors.

---

## Tests (CI must never place a real call)

- `test_twilio_allowlist.py` — mock Twilio client; non-allowlisted number raises
  before any REST call; allowlisted proceeds.
- `test_outcome_mapping.py` — simulated AMD/status results → correct `CallOutcome`.
- `test_ws_codec.py` — round-trip `pcm16→ulaw→pcm16` and Twilio JSON framing,
  no network.
- **Manual smoke (documented, not CI):** dial your own verified number.

---

## Build order (smallest shippable first)

1. Extract allowlist guard into a shared helper; add `error` to `RunStatusResponse`
   so failures are visible. *(unblocks debugging immediately)*
2. `ws_server.py` WS route + μ-law codec + `registry`. Test with `test_ws_codec`.
3. `TwilioOutbound.connect/disconnect` with allowlist guard + REST dial + AMD.
   Test with mocked Twilio.
4. Wire `_make_target` → `TwilioOutbound`. 
5. Recording fetch → BlobStore so the report audio player works.
6. Manual smoke call to `+18573132352`.

Steps 1–4 get you a real, conversing call. 5–6 complete the artifact loop.
