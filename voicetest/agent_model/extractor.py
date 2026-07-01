"""
AgentModelExtractor — turn a plain-English agent description into an AgentModel.

This is the keystone step: `description -> AgentModel`. The extracted model is a
DRAFT the developer confirms/edits; downstream scenario generation and coverage
reference the confirmed model rather than the raw text.

Mirrors ScenarioCompiler's structure (Protocol + LLM impl + Stub) so it reuses the
same LLM provider/config and is unit-testable without network access.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from voicetest.agent_model.models import AgentModel, coerce_agent_model

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentModelExtractor(Protocol):
    async def extract(self, description: str) -> AgentModel:
        """Turn a plain-English agent description into a draft AgentModel."""
        ...


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert voice-AI analyst. Given a plain-English description of what a
voice agent does, you build a precise structured understanding of that agent:
its goal, the information it must collect, the tools it calls, the business rules
it must obey, the risks it faces, and a coarse conversation graph.

You infer a reasonable structure even when the description is terse — this is a
DRAFT the developer will confirm. Prefer a small, clean graph (4-8 nodes) that
captures the real flow over an exhaustive one.

Return ONLY a raw JSON object — no markdown fences, no prose outside the JSON.
"""

_USER_TEMPLATE = """\
AGENT DESCRIPTION:
{description}

Return a JSON object with exactly this shape:
{{
  "goal": "<one sentence: the agent's primary objective>",
  "required_info": [
    {{"key": "<snake_case>", "label": "<human label>", "required": true}}
  ],
  "tools": [
    {{"name": "<tool/API name>", "purpose": "<what the agent uses it for>"}}
  ],
  "business_rules": [
    {{"rule": "<a policy the agent must obey, one sentence>"}}
  ],
  "risks": [
    {{"description": "<a realistic failure mode worth testing>", "node_id": "<id of the related node, or empty>"}}
  ],
  "nodes": [
    {{"id": "<snake_case id>", "label": "<short label>", "objective": "<what happens at this step>"}}
  ],
  "edges": [
    {{"source": "<node id>", "target": "<node id>", "condition": "<branch label, e.g. success/failure/none>"}}
  ]
}}

Rules:
- The graph must start with a greeting/entry node and end at a terminal node
  (confirm, escalate, hang up). Include failure/branch edges where the flow can
  diverge (e.g. tool success vs. timeout).
- Every edge's source and target MUST be an id that appears in "nodes".
- required_info keys are snake_case and reflect what the agent actually collects.
- tools are only external systems the agent calls (calendar, CRM, payments, DB).
  If none are implied, return an empty list.
- risks should be concrete and testable (e.g. "calendar API times out mid-booking",
  not "the agent might fail").
- Return ONLY the raw JSON object — no markdown fences, no prose.
"""


def _strip_fences(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        end = s.rfind("```")
        if end != -1:
            s = s[:end]
    return s.strip()


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Parse the model's response, tolerating stray prose around the JSON."""
    s = _strip_fences(raw)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Fall back to the first {...} block if the model wrapped it in prose.
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


# ---------------------------------------------------------------------------
# LLM implementation
# ---------------------------------------------------------------------------


class LLMAgentModelExtractor:
    """Extract an AgentModel using Claude or OpenAI (reuses the judge model)."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        self._api_key = api_key
        self._model = model

    async def extract(self, description: str) -> AgentModel:
        prompt = _USER_TEMPLATE.format(description=description.strip())

        if self._model.startswith("claude"):
            raw = await self._call_anthropic(prompt)
        else:
            raw = await self._call_openai(prompt)

        data = _extract_json_object(raw)
        return coerce_agent_model(data, source_description=description.strip())

    async def _call_anthropic(self, prompt: str, max_tokens: int = 4096) -> str:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic not installed — pip install anthropic") from e

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        message = await client.messages.create(
            model=self._model,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )
        return message.content[0].text if message.content else ""

    async def _call_openai(self, prompt: str, max_tokens: int = 4096) -> str:
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
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Stub (tests, no network)
# ---------------------------------------------------------------------------


class StubAgentModelExtractor:
    """Returns a fixed AgentModel regardless of description (tests/loopback)."""

    def __init__(self, model: AgentModel | None = None) -> None:
        self._model = model

    async def extract(self, description: str) -> AgentModel:
        if self._model is not None:
            m = self._model.model_copy(deep=True)
            m.source_description = description.strip()
            return m
        return coerce_agent_model(_default_stub_extraction(), description.strip())


def _default_stub_extraction() -> dict[str, Any]:
    return {
        "goal": "Book appointments for callers.",
        "required_info": [
            {"key": "name", "label": "Name", "required": True},
            {"key": "date", "label": "Date", "required": True},
            {"key": "time", "label": "Time", "required": True},
            {"key": "phone", "label": "Callback number", "required": True},
        ],
        "tools": [{"name": "Calendar API", "purpose": "Check and book slots"}],
        "business_rules": [
            {"rule": "Verify the callback phone number before booking."},
            {"rule": "Never double-book a time slot."},
        ],
        "risks": [
            {"description": "Calendar API times out mid-booking", "node_id": "check_calendar"},
            {"description": "Caller never provides a phone number", "node_id": "collect_info"},
        ],
        "nodes": [
            {"id": "greeting", "label": "Greeting", "objective": "Greet the caller."},
            {"id": "collect_info", "label": "Collect Info", "objective": "Gather name, date, time, phone."},
            {"id": "check_calendar", "label": "Check Calendar", "objective": "Look up availability."},
            {"id": "confirm", "label": "Confirm", "objective": "Confirm the booking."},
            {"id": "escalate", "label": "Escalate", "objective": "Hand off to a human."},
        ],
        "edges": [
            {"source": "greeting", "target": "collect_info", "condition": "none"},
            {"source": "collect_info", "target": "check_calendar", "condition": "info complete"},
            {"source": "check_calendar", "target": "confirm", "condition": "available"},
            {"source": "check_calendar", "target": "escalate", "condition": "unavailable/timeout"},
        ],
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_extractor() -> LLMAgentModelExtractor:
    """Instantiate the real extractor from config (reuses LLM_JUDGE_MODEL)."""
    from voicetest.config import Settings

    settings = Settings()
    chosen = settings.llm_judge_model or "claude-sonnet-4-6"

    if chosen.startswith("claude"):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for agent model extractor")
        return LLMAgentModelExtractor(api_key=settings.anthropic_api_key, model=chosen)

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY required for agent model extractor")
    return LLMAgentModelExtractor(api_key=settings.openai_api_key, model=chosen)
