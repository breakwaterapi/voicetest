# Tech Stack — pinned components, interfaces, env vars

**Attach this to every build task.** Models/pricing are *defaults, not commitments*
(this space resets ~quarterly). All providers sit behind interfaces so swapping is a
config change.

## Language & runtime
- **Python 3.11+**, single language for the whole stack (Pipecat is Python).
- Package/deps: `uv` (preferred) or `pip` + `requirements.txt`.
- Async-first: the engine is asyncio; FastAPI for the API.

## Components

| Layer | Default | Interface | Notes |
|-------|---------|-----------|-------|
| Orchestration | **Pipecat v1.x** | — | pipeline of processors |
| Telephony | **Twilio Programmable Voice + Media Streams** | `CallTarget` | outbound via REST + `<Stream>` TwiML over WS |
| STT (caller hears agent) | **Deepgram Flux** | `STTProvider` | built-in end-of-turn; replaces caller-side VAD |
| TTS (caller voice) | **Cartesia Sonic-3 / Turbo** | `TTSProvider` | low latency so caller isn't the bottleneck |
| Caller brain (real-time) | **swappable** (e.g. GPT-4o-mini / Claude Haiku 4.5) | `LLMProvider` | tight system prompt; latency-sensitive |
| Judge (offline) | **swappable** (e.g. GPT-4o / Claude Sonnet) @ **temp 0** | `JudgeProvider` | structured JSON output |
| API | **FastAPI** | — | |
| DB | **SQLite** (MVP) → Postgres later | `Store` | runs / scenarios / results |
| Object storage | **local disk** (MVP) → S3 later | `BlobStore` | call recordings |
| Job execution | **asyncio tasks** (MVP) | — | no Redis/Celery until proven necessary |
| Local dev tunnel | **ngrok** | — | exposes WS server to Twilio |

## Interfaces (define these in `voicetest/providers/` — keep them thin)

```python
class CallTarget(Protocol):
    """The thing the synthetic caller talks to."""
    async def connect(self) -> AudioStream: ...      # opens the call
    async def disconnect(self) -> None: ...
    # impls: LocalLoopback, TranscriptReplay, TwilioOutbound

class STTProvider(Protocol):
    async def stream(self, audio) -> AsyncIterator[Transcript]: ...

class TTSProvider(Protocol):
    async def synthesize(self, text: str) -> AsyncIterator[AudioFrame]: ...

class LLMProvider(Protocol):
    async def respond(self, messages, system: str) -> AsyncIterator[str]: ...

class JudgeProvider(Protocol):
    async def score(self, transcript, rubric) -> JudgeResult: ...  # temp 0, JSON
```

## Audio facts (don't fight these)
- Phone channel is **8 kHz μ-law (G.711)**, narrowband. This is a *feature*: noise
  testing is realistic. All noise mixing happens at the correct sample rate.
- Internal processing sample rate per provider docs; transcode at transport edges only.

## Config & secrets
- All secrets via env vars, loaded through one `config.py` (pydantic-settings). Never
  hardcode keys. `.env.example` lists every var; `.env` is gitignored.

Required env vars (only the ones a given target/provider needs must be set):
```
# Telephony
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_FROM_NUMBER=
VERIFIED_DEST_NUMBERS=    # comma-separated allowlist (ADR-9)
PUBLIC_WS_URL=            # ngrok wss URL for Media Streams
# STT / TTS
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
# LLMs (set whichever provider you choose)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
LLM_CALLER_MODEL=         # e.g. gpt-4o-mini | claude-haiku-4-5
LLM_JUDGE_MODEL=          # e.g. gpt-4o | claude-sonnet-4-6
# App
VOICETEST_DB_URL=sqlite:///voicetest.db
VOICETEST_BLOB_DIR=./recordings
```

## Cost model (rough, per ~2-min call — sanity check, not a quote)
Twilio voice (per-min) + Deepgram STT + Cartesia TTS + caller-LLM tokens +
1 judge call. Lands in **low single-digit cents/call**. Watch it when parallelizing.
Telnyx/Plivo (~$0.005/min outbound) are cheaper telephony swaps post-MVP.
