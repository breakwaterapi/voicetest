# P1-T6 — Web page: paste #, pick scenario, run N, view report

**Context pack:** `tech-stack.md` + `conventions.md` + this file.
**Depends on:** P1-T5 (API)

## Goal
The thinnest possible UI: one page to start a run and watch the report. No auth, no
routing complexity.

## Deliverables
- A single page (server-rendered HTML + a little JS, or one small React page — pick the
  lighter option) served by FastAPI:
  - inputs: phone number, scenario dropdown (from `GET /scenarios`), number of calls N.
  - "Run" → `POST /runs`, then poll `GET /runs/{id}` and show live progress
    (`completed/total`, per-call pass/fail as they land).
  - report view: overall pass rate, per-call breakdown, each failure links to its
    recording (audio player) + highlighted transcript turn + fix suggestion.
- Keep styling minimal; correctness and clarity over polish.

## Tests to write
- `test_ui_serves.py`: page renders, scenario dropdown is populated from the API.
- (Manual) run a loopback run from the page and confirm progress + report render.

## Definition of done
- [ ] From one page: pick scenario, enter number, run N, watch progress, read report
- [ ] Failures show audio + transcript + fix suggestion
- [ ] Works against loopback for demoing without spending call $
