# P0-T1 — Project setup

**Context pack:** `tech-stack.md` + `conventions.md` + this file.
**Depends on:** —

## Goal
Stand up the repo skeleton, dependency management, config loader, core data models,
and the test harness — so every later task slots in without restructuring.

## Deliverables
- `pyproject.toml` (or `requirements.txt`) with: pipecat-ai, fastapi, uvicorn,
  pydantic, pydantic-settings, pyyaml, pytest, pytest-asyncio, ruff. (Provider SDKs
  added in their tasks.)
- Package tree from `conventions.md` (empty `__init__.py` modules where listed).
- `voicetest/config.py` — pydantic-settings `Settings` loading every env var in
  `tech-stack.md`. All optional except where a feature needs them; never crash on
  import when a key is missing.
- `voicetest/models.py` — dataclasses/pydantic for the data shapes below.
- `.env.example` listing all env vars; `.gitignore` (`.env`, `recordings/`, `*.db`).
- `tests/` tree with `tests/unit`, `tests/integration`, `tests/fixtures`, and a
  trivial passing `tests/unit/test_smoke.py`.
- `pytest.ini` / `[tool.pytest]` config with `asyncio_mode = auto`.

## Core models (`models.py`) — minimum fields
```python
@dataclass
class Turn:            # one utterance
    speaker: Literal["caller", "agent"]
    text: str
    start_ms: int      # on captured timeline
    end_ms: int

@dataclass
class Event:           # interruption, vad_start, vad_stop, noise_on, etc.
    kind: str
    at_ms: int
    data: dict

class CallOutcome(str, Enum):
    COMPLETED = "completed"
    NO_ANSWER = "no_answer"
    VOICEMAIL = "voicemail"
    ERROR = "error"

@dataclass
class CallResult:
    scenario_id: str
    outcome: CallOutcome
    turns: list[Turn]
    events: list[Event]
    recording_path: str | None
    duration_ms: int
    error: str | None = None

@dataclass
class Report:          # filled by scorer in P0-T6 / P1-T4
    call: CallResult
    passed: bool
    dimensions: dict   # name -> {score, reason}
    summary: str
```

## Tests to write
- `test_smoke.py`: imports the package, asserts `Settings()` constructs with no env set.
- `test_models.py`: round-trip construct each model; `CallOutcome` enum values.

## Definition of done
- [ ] `pytest tests/unit` green in <5s, no network
- [ ] `ruff check` clean
- [ ] `Settings()` works with an empty environment
- [ ] Package importable as `import voicetest`
