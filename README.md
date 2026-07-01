# VoiceTest

Automated testing for voice AI agents. VoiceTest places a real outbound phone call
to a voice agent, **acts like a difficult human caller** following a scenario,
records everything, and produces a scored pass/fail report.

The synthetic caller is itself a voice agent (STT → persona LLM → TTS) running over
a phone line. The product value is in three things competitors skip: **uncooperative
personas**, **background-noise injection**, and **barge-in/interruption injection**.

---

## How to use these docs with an LLM (token-efficient workflow)

These docs are designed so you build the app one task at a time, attaching only the
context a task needs. **Do not paste the whole `docs/` folder into your LLM.**

For any build task, attach exactly:

1. `docs/tech-stack.md`  — pinned components, swappable interfaces, env vars
2. `docs/conventions.md`  — repo layout, code style, testing rules
3. The **one** task file you're building, e.g. `docs/tasks/phase-0/P0-T3-caller-engine.md`
4. Only the prior-task files that task lists under **Depends on** (usually 0–1)

That's the "context pack." Everything else (`architecture.md`, `roadmap.md`) is
read-once orientation for *you*, the human — not per-task LLM context.

Each task file is self-contained: goal, deliverables, implementation notes, the
tests to write, and a definition of done. Build them in order; don't skip the tests.

---

## Document map

| File | Purpose | Who reads it |
|------|---------|--------------|
| `docs/architecture.md` | Final architecture + design decisions (ADRs) | You, once |
| `docs/tech-stack.md` | Pinned stack, provider interfaces, env vars, cost table | Every task |
| `docs/conventions.md` | Repo layout, style, testing philosophy | Every task |
| `docs/scenario-spec.md` | Declarative scenario/persona YAML schema | Scenario tasks |
| `docs/testing-strategy.md` | How to test voice without burning call $ | You + test tasks |
| `docs/roadmap.md` | Phase overview + task index | You, to plan |
| `docs/tasks/phase-0/*` | The MVP you want now (place 1 real call, score it) | One at a time |
| `docs/tasks/phase-1/*` | The launchable product | One at a time |

---

## The magic moment (definition of Phase 0 success)

> "I gave it my agent's phone number and it caught a real bug — and I have the
> recording to prove it."

Start at `docs/roadmap.md`, then `docs/tasks/phase-0/P0-T1-project-setup.md`.
