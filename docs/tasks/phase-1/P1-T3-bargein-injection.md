# P1-T3 — Barge-in / interruption injection processor

**Context pack:** `tech-stack.md` + `conventions.md` + `architecture.md` (§4.3) + this file.
**Depends on:** P0-T3 (pipeline), P0-T5 (recorder/VAD events)

## Goal
On a scripted moment, make the caller speak **over** the agent to test interruption
handling. Then capture the signals scoring needs: did the agent stop, recover context,
or restart; did latency spike.

## Deliverables
- `voicetest/engine/processors/bargein.py`: watches inbound-track VAD (agent started
  speaking). On `scenario.barge_in.on_agent_turn`, after
  `trigger_after_agent_speaks_ms`, forces the caller TTS to emit `caller_says`,
  cutting the agent off. Logs an `interruption` event (with the agent's position).
- Wiring: active only when `scenario.barge_in.enabled`.

## Implementation notes
- Use Pipecat's frame/VAD events for "agent started speaking" — that's the trigger
  anchor. All timing on the captured timeline.
- The processor only *causes* the interruption + logs the event; judging recovery
  happens in P1-T4 from the transcript + events.

## Tests to write
- `test_bargein_timing.py` (integration vs LocalLoopback): with a stub agent that
  speaks a long line, assert the interruption fires on the right turn ~the configured
  ms after the agent starts, and an `interruption` event is logged.
- `test_bargein_disabled.py`: disabled = no interruption events.
- Against the **broken agent** (ignores barge-in): assert the event shows the agent
  kept talking (feeds the P1-T4 "did not stop cleanly" finding).

## Definition of done
- [ ] Interruption fires at the scripted moment, logged as an event
- [ ] Disabled = no effect
- [ ] Broken-agent case observably shows failure-to-stop
