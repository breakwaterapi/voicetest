# P1-T2 — Background-noise injection processor

**Context pack:** `tech-stack.md` + `conventions.md` + `architecture.md` (§4.2) + this file.
**Depends on:** P0-T3 (pipeline)

## Goal
Mix a background-noise bed into the caller's **outgoing** audio at a configurable level,
so the agent-under-test's STT hits realistic degradation. Then scoring checks whether
the agent still captured the key info.

## Deliverables
- `voicetest/engine/processors/noise.py`: a processor placed **between TTS and
  transport** that mixes a noise clip into outgoing PCM at `noise.level_db`, at the
  correct sample rate (8 kHz μ-law at the phone edge). Logs `noise_on`/`noise_off`
  events to the recorder.
- A small library of royalty-free clips (cafe/street/car) in `assets/noise/` (or
  fixtures for tests). Loaded per `NoiseConfig.profile`.
- Wiring: processor only active when `scenario.noise.enabled`.

## Implementation notes
- Mix at the right sample rate to avoid artifacts; resample the clip once at load.
- The caller's spoken content is **ground truth** (you generated it) — store it so the
  noise-resilience judge (P1-T4) can compare against what the agent understood.

## Tests to write
- `test_noise_mix.py`: feed known PCM + known noise clip; assert output level matches
  target dB within tolerance and length is preserved.
- `test_noise_events.py`: noise_on/off events logged with correct timeline.
- `test_noise_disabled.py`: with `enabled: false`, output PCM is unchanged.

## Definition of done
- [ ] Noise mixes at the configured dB without breaking timing
- [ ] Events logged; disabled = passthrough
- [ ] Caller ground-truth available to the scorer
