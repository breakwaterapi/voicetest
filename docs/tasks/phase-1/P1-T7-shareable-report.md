# P1-T7 — Shareable report URL

**Context pack:** `conventions.md` + this file.
**Depends on:** P1-T6

## Goal
Make any run's report shareable via an unguessable URL (MVP-grade sharing — no auth).

## Deliverables
- Give each run a high-entropy `share_token` (e.g. `secrets.token_urlsafe(16)`),
  stored with the run.
- `GET /r/{share_token}` → a **read-only** report view (same render as P1-T6's report,
  no controls to start runs, recordings served via tokenized paths).
- A "Copy share link" button on the report page.

## Implementation notes
- Read-only endpoint must not expose the run-creation API or other runs.
- Recording URLs under a shared report must also be guarded by the token (don't leak
  raw `call_id` blob paths).

## Tests to write
- `test_share_readonly.py`: a valid token renders the report; a wrong token 404s;
  the shared view exposes no run-creation/control endpoints.

## Definition of done — **and Phase 1 / launch-readiness gate**
- [ ] Any run produces a copyable, unguessable, read-only report link
- [ ] You'd post the demo link in the Vapi Discord without embarrassment
