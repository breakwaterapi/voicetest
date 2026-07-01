# Design: Agentic + Streamlined Flow (v2)

Status: proposal · Author: design session 2026-07-01 · Owner: @khalid.d

Goal: take the current **Describe → Confirm → Pick → Run** wizard to a
production-grade, Hamming-class experience on two axes at once —
**a genuinely agentic synthetic caller** and **a streamlined, modern UX** —
without losing the existing moat (uncooperative personas, noise, barge-in).

---

## 1. Where we are (grounded in the code)

| Concern | Today | File |
|---|---|---|
| Understanding | One-shot LLM extraction, temp 0, prose-only | [agent_model/extractor.py](../voicetest/agent_model/extractor.py) |
| Caller | LLM per-turn but driven by a **static** `system_prompt` + `difficult_behaviors` it "MUST exhibit ALL of" | [engine/pipeline.py](../voicetest/engine/pipeline.py) `assemble_system_prompt` |
| Scenarios | Pre-compiled static YAML; `Persona` = fixed prompt | [scenarios/models.py](../voicetest/scenarios/models.py) |
| Scoring | LLM judge over `pass_criteria: list[str]` | [scoring/judge.py](../voicetest/scoring/judge.py) |
| Coverage | Graph-node based; battery can target a node | [scenarios/compiler.py](../voicetest/scenarios/compiler.py) |
| Prod loop | ingest → failures → regenerate scenarios | [api.py](../voicetest/api.py) |

**Honest read:** the caller is *scripted-adaptive* — it reacts to the transcript
but is told to perform a fixed checklist of behaviors rather than pursue a goal.
Scoring is *vibes-based* — an LLM mood over a prose list, non-reproducible.
The UX has a *hard confirm gate* that dead-ends when extraction is slightly off.

---

## 2. Target architecture

Five changes. #1 and #2 create the agentic core; #3–#5 are the streamlined UX
and the robustness that lets you trust the verdicts.

### 2.1 Agentic caller (the keystone)

Replace the static-behavior persona with a **goal-directed policy** the caller
reasons over every turn.

New `Persona` shape (additive — keep `system_prompt` for back-compat):

```python
class Persona(_StrictModel):
    system_prompt: str                 # identity/voice only, not the script
    goal: str                          # what this caller is trying to achieve
    must_verify: list[str] = []        # facts the caller must extract/confirm
                                       #   e.g. "the confirmed appointment time"
    difficulty: Literal["easy","medium","hard"] = "medium"
    tactics: list[str] = []            # OPTIONAL nudges, not a mandatory checklist
    give_up_after_turns: int | None = None
    ground_truth: dict[str, Any] = {}
```

Turn loop gains an explicit **caller state machine** (in `engine/pipeline.py`),
not just a chat history:

```
state = { goal, satisfied: bool, verified: {fact -> value|None}, frustration: 0..1 }
each turn:
  1. observe: last agent utterance
  2. update: did it satisfy the goal? did it reveal a must_verify fact?
  3. decide: continue / probe harder / escalate difficulty / give up / say goodbye
  4. act: emit next utterance consistent with difficulty + frustration
```

Key shift: difficulty becomes a **dial the caller applies adaptively**
(a "hard" caller withholds info, interrupts, changes its mind) instead of a
literal list it recites. This is what makes it feel like a real adversary and
catches bugs scripts can't. The state (`verified`, `satisfied`) is also what
feeds assertion scoring below — no separate judge guesswork about whether the
goal was met.

Latency guardrail (ADR-5 still holds): the decide step is a *small* structured
call (or folded into the same generation with a JSON preamble) so we don't add a
round-trip per turn. Prototype both; measure p95.

### 2.2 Assertion-based scoring (trust the verdict)

Replace `pass_criteria: list[str]` → LLM mood with **typed, independently-checked
assertions**. Each yields pass/fail + evidence; the run verdict is a deterministic
aggregate, not a single judge call.

```python
class Assertion(_StrictModel):
    id: str
    kind: Literal["llm_judge","state","latency","regex","tool_call"]
    description: str                   # human-readable
    check: dict[str, Any]              # kind-specific params
    severity: Literal["blocker","major","minor"] = "major"
```

- `state` assertions read the caller state machine (§2.1) — zero LLM flake
  (e.g. "caller's callback number was echoed back before booking").
- `llm_judge` assertions are scoped to **one** claim each → far more reproducible
  than one judge over a paragraph. Run each at temp 0; optionally 3-vote for
  blockers.
- `latency` reads the existing `stage_timing` events already emitted by the pipeline.

Verdict = `fail if any blocker fails; else score = weighted pass rate`. The
report shows a **per-assertion checklist**, which also becomes your coverage
surface (which rules/risks are actually exercised).

Generation change: the compiler emits assertions alongside each scenario,
derived from the `AgentModel`'s `business_rules` and `risks` (already structured).

### 2.3 Discovery call before generation (ground the model)

Between extract and generate, place **one exploratory call** that lets the
platform correct its own understanding before spending 20 calls on a wrong model.

```
describe → extract draft → [discovery call] → reconciled AgentModel → generate
```

The discovery caller is a cooperative "happy path" persona whose only job is to
walk the main flow once. We diff observed behavior against the draft model
(tools actually invoked? info actually requested? terminal reached?) and surface
corrections inline. Cheap (1 call), high trust payoff, and it makes the whole
thing feel like it *knows* the agent. Optional/skippable for loopback.

### 2.4 Conversational canvas (kill the confirm gate)

The wizard becomes a **single living canvas** instead of a 4-step gate:

- Auto-advance: on extract, immediately show the model **and** a first pass of
  scenarios (optimistic) — no "confirm" button required to proceed.
- The `AgentModel` is edited **conversationally**: "it also takes payments" →
  re-extract deltas, don't re-run the whole thing. (New endpoint:
  `POST /agent-model/{id}/refine` taking a natural-language instruction.)
- The conversation graph stays as the spatial anchor (it's good), but clicking a
  node streams newly generated scenarios in place.
- Steps collapse from **4 gates → 1 canvas with 2 optional confirmations**
  (discovery + run cost).

Modern feel = streaming, inline edit, no dead-end forms.

### 2.5 Run robustness (production-ready)

- **Flake retry:** re-run a scenario once on `CallOutcome` transport errors or
  a judge-disagreement above threshold; mark `flaky` rather than fail.
- **Determinism:** seed personas; pin judge model + temp 0; record model IDs in
  the report so runs are comparable across time.
- **Trajectory output:** persist the caller state timeline per turn (already
  half-there via events) so a failure is *explainable*, not just red.
- **Dedup + budgets:** cap concurrent real calls; dedup near-identical generated
  scenarios before charging the user for calls.

---

## 3. Migration plan (additive, non-breaking)

| Phase | Ships | Risk |
|---|---|---|
| P1 | Agentic caller behind a `Persona.difficulty` flag; old path stays default | Low — additive fields, old YAML still valid |
| P2 | Assertion model + `state`/`latency` kinds; keep `pass_criteria` as an auto-migrated `llm_judge` assertion | Low |
| P3 | Discovery call endpoint + reconcile UI | Med — new call cost |
| P4 | Canvas UX (`/refine`, streaming, auto-advance) | Med — frontend rewrite of the served HTML |
| P5 | Robustness (retry/seed/trajectory/dedup) | Low |

Each phase is independently shippable and testable with loopback (no telephony $).

## 4. Open decisions

1. **Caller decide-step latency:** separate structured call vs. JSON-preamble in
   the same generation. → prototype both, pick on measured p95.
2. **Judge voting:** single-shot vs. 3-vote for blocker assertions (cost vs.
   reproducibility). → 3-vote blockers only.
3. **Discovery call:** on by default or opt-in? → default on for real telephony,
   off for loopback.
4. **Canvas:** extend the server-rendered HTML in `api.py`, or split a real
   frontend? → recommend keeping single-file HTML for now, add SSE streaming.

## 5. What we explicitly keep

The moat is untouched and reinforced: uncooperative personas (now *more* real via
adaptive difficulty), noise injection, barge-in, and the production-learning loop.
This design makes them *sharper*, not replaced.
