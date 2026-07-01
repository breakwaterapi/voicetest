# P1-T4 — Full scoring (5 dims minus tool-call) + structured report

**Context pack:** `tech-stack.md` + `conventions.md` + `architecture.md` (ADR-7) + this file.
**Depends on:** P0-T6 (scoring base), P1-T2, P1-T3

## Goal
Extend scoring from 2 dimensions to the full launch set (minus tool-call accuracy,
which is Phase 2), and produce the launch report.

## Dimensions
1. **Goal completion** — judge (already in P0-T6).
2. **Latency** — computed (already in P0-T6); report p50/p95 + per-turn.
3. **Barge-in handling** — analysis of the interruption window from events+transcript:
   did agent audio stop, context recovered vs restarted, latency spike after.
4. **Noise resilience** — judge compares caller **ground truth** vs what the agent
   understood (its transcript/responses) under noise.
5. **Tool-call accuracy** — **deferred**; report as "needs API access" (Phase 2), not a fail.

## Deliverables
- Extend `voicetest/scoring/compute.py`: barge-in window analysis (stop detection,
  restart-vs-recover heuristic, post-interruption latency).
- Extend `voicetest/scoring/judge.py`: single judge call now also returns
  noise-resilience evaluation when noise was active (still one call/ ADR-7).
- `voicetest/scoring/report.py`: build the launch `Report`:
  overall pass/fail, per-dimension {score, reason}, per-call breakdown linking the
  recording + the highlighted transcript turn for each failure, a plain-English
  "most common failure" summary, and an LLM-generated fix suggestion.

## Tests to write
- `test_bargein_scoring.py`: events showing agent-kept-talking → barge-in dim fails;
  clean stop + context recovery → passes.
- `test_noise_scoring.py` (canned judge): agent missed the key number under noise →
  noise dim flags it; correct capture → passes.
- `test_report_shape.py`: report includes all 5 dims (tool-call = "needs API access"),
  failure breakdown links recording + turn.

## Definition of done
- [ ] All 5 dimensions present; tool-call marked deferred (not a fail)
- [ ] Still one judge call per call
- [ ] Report links each failure to its recording + transcript turn + fix suggestion
