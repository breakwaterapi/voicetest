# P1-T5 — FastAPI + asyncio job runner + SQLite store

**Context pack:** `tech-stack.md` + `conventions.md` + this file.
**Depends on:** P0-T8 (engine+scorer usable as functions)

## Goal
Expose "run N calls for a scenario against a number" as an API with a background job
runner and persistent storage. No auth.

## Deliverables
- `voicetest/store.py` (extend): SQLite via the `Store` interface — tables for
  `runs`, `calls`, `reports`. `BlobStore` already handles recordings.
- `voicetest/jobs.py`: an asyncio-based runner. A **run** = N calls. Phase 1 executes
  calls **sequentially** (design for later `asyncio.gather` + semaphore, but don't
  parallelize yet). Each call: `run_call` → `score` → persist. Track progress.
- `voicetest/api.py` (FastAPI):
  - `POST /runs` `{scenario_id, number, n}` → creates a run, kicks off the job, returns
    `run_id`.
  - `GET /runs/{id}` → status + progress (`completed/total`) + per-call summaries.
  - `GET /runs/{id}/report` → aggregate report (pass rate + per-call breakdown).
  - `GET /scenarios` → builtin list.
  - `GET /recordings/{call_id}` → audio file.

## Implementation notes
- Reuse the verified-number allowlist (ADR-9) at the API boundary too.
- Non-COMPLETED call outcomes surface as themselves in progress, not as crashes.
- Keep the job runner in-process (asyncio task) — no Redis/Celery (per ADR/roadmap).

## Tests to write
- `test_api_run_loopback.py`: `POST /runs` with a loopback flag runs N calls, `GET`
  shows progress→done, report has a pass rate. (Uses stubs; no telephony.)
- `test_store_roundtrip.py`: runs/calls/reports persist and reload.

## Definition of done
- [ ] Create a run, poll status, fetch aggregate report — all over HTTP
- [ ] N calls run sequentially and persist to SQLite
- [ ] Allowlist enforced at the API; failed-connect calls reported, not crashed
