"""
Judge provider interface and implementations.

The judge evaluates one completed call transcript against the scenario's
scoring rubric and returns structured JSON (ADR-7: one call per scored call,
temperature 0, no streaming needed).

When noise was active the single judge call also evaluates noise resilience
— whether the agent correctly captured ground_truth values despite noise.
This keeps to exactly one LLM call per completed call.

JudgeResult schema:
  {
    "goal_completed": bool,
    "reason": str,
    "criteria": [{"text": str, "met": bool, "evidence": str}],
    "noise_resilience": {               # present only when noise_ground_truth provided
      "passed": bool,
      "missed_values": ["..."],         # ground-truth values the agent got wrong
      "reason": str
    } | null
  }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from voicetest.models import Turn
from voicetest.scenarios.models import ScoringConfig

# ---------------------------------------------------------------------------
# JudgeResult
# ---------------------------------------------------------------------------


@dataclass
class CriterionResult:
    text: str
    met: bool
    evidence: str = ""


@dataclass
class NoiseResilienceResult:
    passed: bool
    missed_values: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class JudgeResult:
    goal_completed: bool
    reason: str
    criteria: list[CriterionResult] = field(default_factory=list)
    noise_resilience: NoiseResilienceResult | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class JudgeProvider(Protocol):
    async def score(
        self,
        transcript: str,
        rubric: ScoringConfig,
        noise_ground_truth: dict[str, Any] | None = None,
    ) -> JudgeResult: ...


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a strict call-quality evaluator. "
    "You will be given a phone call transcript and a scoring rubric. "
    "Evaluate the rubric honestly based solely on what the transcript contains. "
    "Return ONLY valid JSON — no markdown, no prose outside the JSON object."
)

_USER_TEMPLATE = """
TRANSCRIPT:
{transcript}

OBJECTIVE: {objective}

PASS CRITERIA (evaluate each one):
{criteria_list}

Return a JSON object with EXACTLY this shape:
{{
  "goal_completed": <true|false>,
  "reason": "<1-2 sentences quoting evidence from the transcript>",
  "criteria": [
    {{"text": "<criterion text>", "met": <true|false>, "evidence": "<short quote or 'not found'>"}}
  ]
}}
The "criteria" array must have exactly {n_criteria} items, in the same order as above.
""".strip()

_NOISE_SECTION = """

NOISE RESILIENCE EVALUATION:
The call was run under background noise. The ground truth values the agent should have captured are:
{ground_truth_json}

In addition to the fields above, add a "noise_resilience" field:
{{
  "noise_resilience": {{
    "passed": <true if agent correctly captured all ground truth values, false otherwise>,
    "missed_values": ["<value the agent got wrong or missed>", ...],
    "reason": "<1-2 sentences on noise impact>"
  }}
}}
"""


def _format_transcript(turns: list[Turn]) -> str:
    lines = [f"{t.speaker.upper()}: {t.text}" for t in turns]
    return "\n".join(lines)


def _build_prompt(
    transcript: str,
    rubric: ScoringConfig,
    noise_ground_truth: dict[str, Any] | None = None,
) -> str:
    criteria_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(rubric.pass_criteria))
    base = _USER_TEMPLATE.format(
        transcript=transcript,
        objective=rubric.objective,
        criteria_list=criteria_list,
        n_criteria=len(rubric.pass_criteria),
    )
    if noise_ground_truth:
        base = base + _NOISE_SECTION.format(
            ground_truth_json=json.dumps(noise_ground_truth, indent=2)
        )
    return base


def _strip_json_fences(raw: str) -> str:
    """
    Some chat models wrap JSON in ```json … ``` fences despite instructions.
    Strip a single leading/trailing markdown code fence so json.loads succeeds.
    """
    s = raw.strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _parse_response(
    raw: str,
    rubric: ScoringConfig,
    noise_ground_truth: dict[str, Any] | None = None,
) -> JudgeResult:
    """Parse and validate the raw JSON string returned by the LLM."""
    try:
        data = json.loads(_strip_json_fences(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {exc}\nRaw: {raw!r}") from exc

    criteria = [
        CriterionResult(
            text=c.get("text", rubric.pass_criteria[i] if i < len(rubric.pass_criteria) else ""),
            met=bool(c.get("met", False)),
            evidence=c.get("evidence", ""),
        )
        for i, c in enumerate(data.get("criteria", []))
    ]

    noise_resilience: NoiseResilienceResult | None = None
    if noise_ground_truth is not None:
        nr = data.get("noise_resilience") or {}
        noise_resilience = NoiseResilienceResult(
            passed=bool(nr.get("passed", False)),
            missed_values=list(nr.get("missed_values", [])),
            reason=str(nr.get("reason", "")),
        )

    return JudgeResult(
        goal_completed=bool(data.get("goal_completed", False)),
        reason=str(data.get("reason", "")),
        criteria=criteria,
        noise_resilience=noise_resilience,
    )


# ---------------------------------------------------------------------------
# OpenAI judge (real, temp=0)
# ---------------------------------------------------------------------------


class OpenAIJudge:
    """GPT-4o (or configurable model) judge at temperature 0."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self._api_key = api_key
        self._model = model

    async def score(
        self,
        transcript: str,
        rubric: ScoringConfig,
        noise_ground_truth: dict[str, Any] | None = None,
    ) -> JudgeResult:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError("openai not installed — pip install openai") from e

        client = AsyncOpenAI(api_key=self._api_key)
        response = await client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _build_prompt(transcript, rubric, noise_ground_truth)},
            ],
        )
        raw = response.choices[0].message.content or ""
        return _parse_response(raw, rubric, noise_ground_truth)


# ---------------------------------------------------------------------------
# Anthropic judge (real, temp=0)
# ---------------------------------------------------------------------------


class AnthropicJudge:
    """Claude Sonnet (or configurable model) judge at temperature 0."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._api_key = api_key
        self._model = model

    async def score(
        self,
        transcript: str,
        rubric: ScoringConfig,
        noise_ground_truth: dict[str, Any] | None = None,
    ) -> JudgeResult:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic not installed — pip install anthropic") from e

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        message = await client.messages.create(
            model=self._model,
            system=_SYSTEM,
            messages=[
                {"role": "user", "content": _build_prompt(transcript, rubric, noise_ground_truth)}
            ],
            max_tokens=1024,
            temperature=0,
        )
        raw = message.content[0].text if message.content else ""
        return _parse_response(raw, rubric, noise_ground_truth)


# ---------------------------------------------------------------------------
# CannedJudge — deterministic stub for tests
# ---------------------------------------------------------------------------


class CannedJudge:
    """
    Returns a pre-configured JudgeResult without any network call.
    Suitable for testing `score()` logic independent of LLM behaviour.
    """

    def __init__(self, result: JudgeResult) -> None:
        self._result = result
        self.call_count = 0

    async def score(
        self,
        transcript: str,
        rubric: ScoringConfig,
        noise_ground_truth: dict[str, Any] | None = None,
    ) -> JudgeResult:
        self.call_count += 1
        return self._result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_judge(model: str | None = None) -> OpenAIJudge | AnthropicJudge:
    """Instantiate the real judge from config (LLM_JUDGE_MODEL)."""
    from voicetest.config import Settings

    settings = Settings()
    chosen = model or settings.llm_judge_model or "gpt-4o"

    if chosen.startswith("claude"):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for Claude judge models")
        return AnthropicJudge(api_key=settings.anthropic_api_key, model=chosen)

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY required for OpenAI judge models")
    return OpenAIJudge(api_key=settings.openai_api_key, model=chosen)
