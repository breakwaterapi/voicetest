# Testing Strategy — how to test a voice product without burning call $

The hard part of testing VoiceTest is that the thing under construction is itself a
real-time audio system. The strategy: **make audio optional at three layers** so most
logic is testable with no telephony, no audio hardware, and no LLM spend.

## The three test targets (via the `CallTarget` interface — ADR-3)

1. **`TranscriptReplay`** — feeds a canned conversation transcript. Used to test the
   **scorer** (compute metrics + judge) in isolation. Zero audio. Deterministic.
2. **`LocalLoopback`** — a real Pipecat pipeline where the "agent under test" is a
   local stub. Tests the **engine** (turn-taking, recorder, noise, barge-in) with
   stub STT/LLM/TTS. No telephony, no network.
3. **`TwilioOutbound`** — real phone. **Never used in automated tests.** Exercised
   only by manual/smoke runs you trigger by hand.

## Test layers

| Layer | Location | Targets | Speed | Spend |
|-------|----------|---------|-------|-------|
| Unit | `tests/unit` | mocks | <5s, no net | none |
| Integration | `tests/integration` | LocalLoopback + stub providers | seconds | none |
| Scorer validation | `tests/integration` | TranscriptReplay | seconds | none/canned |
| Manual smoke | run by hand | TwilioOutbound | ~minutes | real cents |

## Stub providers (in `tests/fixtures`)
- **StubSTT** — turns scripted text into "transcripts" on cue (or echoes injected text).
- **StubLLM** — returns scripted lines for the persona; deterministic.
- **StubTTS** — emits silent/constant PCM frames of correct length (so timing/recorder
  logic is exercised without real synthesis).
- **CannedJudge** — returns a fixed `JudgeResult` so scoring-pipeline tests don't call
  a real LLM. A separate, opt-in test (marked `@pytest.mark.llm`, skipped in CI) hits
  the real judge against a golden transcript to catch prompt regressions.

## The broken-agent fixture (the trust anchor)
`tests/fixtures/broken_agent.py` — a `LocalLoopback` agent that is deliberately bad:
ignores barge-in (keeps talking), responds slowly (injects delay), and mis-hears the
key number under noise. **Test assertion:** running a scenario against the broken
agent must produce a **failing** report with the *correct* failure reasons. If the
scorer passes the broken agent, that's a test failure. This is how you trust a green
report on a real agent.

## What each phase must prove (test gates)
- **P0**: `run_call` against LocalLoopback produces a complete `CallResult` (transcript
  + timestamps + recording). Scorer produces goal-completion + latency. Broken-agent
  smoke shows a fail.
- **P1**: noise & barge-in processors have unit tests (correct mix level / correct
  interrupt timing); full 5-dim scorer validated against TranscriptReplay goldens;
  broken-agent fixture fails on the *right* dimensions.

## CI
GitHub Actions runs `tests/unit` + `tests/integration` only. No secrets needed for the
green path (stubs/canned). The real-LLM and real-telephony tests are manual/local.
