# P0-T6 — Scoring MVP: goal-completion (judge) + computed latency

**Context pack:** `tech-stack.md` + `conventions.md` + `architecture.md` (ADR-7) + this file.
**Depends on:** P0-T5

## Goal
Turn a `CallResult` into a scored `Report` with two dimensions: **goal completion**
(LLM judge) and **latency** (computed). One judge call per call (ADR-7).

## Deliverables
- `voicetest/scoring/compute.py`:
  - `latency(call) -> {p50_ms, p95_ms, per_turn, warn}`: for each agent turn, gap
    between caller's last audio end and agent's first audio start (from events/turns on
    the captured timeline). Compare to `scenario.scoring.latency` thresholds.
- `voicetest/scoring/judge.py`:
  - `JudgeProvider` impl (configurable model, **temperature 0**) that takes the
    transcript + `scenario.scoring` rubric and returns **structured JSON**:
    `{goal_completed: bool, reason: str, criteria: [{text, met: bool}]}`.
  - `CannedJudge` stub for tests.
- `voicetest/scoring/score.py`: `score(call, scenario) -> Report` combining judge +
  latency into `Report` (`passed` = goal_completed AND all pass_criteria met; latency
  is reported, warns but doesn't fail in P0).

## Implementation notes
- Single judge call returns all judged fields at once (cheap). Prompt the judge to
  evaluate each `pass_criteria` bullet verbatim and quote evidence from the transcript.
- Non-COMPLETED outcomes (no-answer/voicemail/error) → `Report.passed=False` with a
  clear summary, **no judge call** (don't pay to judge a call that didn't happen).

## Tests to write
- `test_latency.py`: hand-built CallResult with known turn timing → assert p50/p95.
- `test_score_canned.py`: TranscriptReplay + CannedJudge → assert Report shape and
  `passed` logic (pass when criteria met, fail when not).
- `test_score_no_answer.py`: NO_ANSWER outcome → passed=False, no judge invoked.
- (opt-in, `@pytest.mark.llm`, skipped in CI) real judge on a golden transcript.

## Definition of done
- [ ] `score()` returns a `Report` with goal-completion + latency dimensions
- [ ] Exactly one judge call per completed call; none for failed-connect calls
- [ ] Judge runs at temp 0 and returns validated structured JSON
