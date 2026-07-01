"""LLM provider interface and implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    async def respond(
        self, messages: list[dict], system: str
    ) -> AsyncIterator[str]: ...


# ---------------------------------------------------------------------------
# OpenAI (real provider — requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------


class OpenAILLM:
    """GPT-4o-mini (or any openai-compatible model) streaming LLM."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._model = model

    async def respond(self, messages: list[dict], system: str) -> AsyncIterator[str]:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError("openai not installed — pip install openai") from e

        client = AsyncOpenAI(api_key=self._api_key)
        full_messages = [{"role": "system", "content": system}, *messages]
        stream = await client.chat.completions.create(
            model=self._model,
            messages=full_messages,  # type: ignore[arg-type]
            stream=True,
            temperature=1.0,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ---------------------------------------------------------------------------
# Anthropic (real provider — requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------


class AnthropicLLM:
    """Claude Haiku 4.5 (or any Anthropic model) streaming LLM."""

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5") -> None:
        self._api_key = api_key
        self._model = model

    async def respond(self, messages: list[dict], system: str) -> AsyncIterator[str]:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("anthropic not installed — pip install anthropic") from e

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        async with client.messages.stream(
            model=self._model,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=512,
            temperature=1.0,
        ) as stream:
            async for text in stream.text_stream:
                yield text


# ---------------------------------------------------------------------------
# Stub (no network — used by tests)
# ---------------------------------------------------------------------------


class StubLLM:
    """
    Returns canned responses from a pre-configured sequence.
    When the sequence is exhausted it repeats the last entry.
    Suitable for deterministic multi-turn test scripts.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or [
            "Hi, I'd like to book a table please.",
            "For two people, tomorrow evening around 7pm.",
            "Yes, that's correct. Thank you, goodbye!",
        ]
        self._index = 0

    async def respond(self, messages: list[dict], system: str) -> AsyncIterator[str]:
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        yield response


# ---------------------------------------------------------------------------
# Factory — picks provider from config
# ---------------------------------------------------------------------------


def make_llm_provider(model: str | None = None) -> OpenAILLM | AnthropicLLM:
    """Instantiate the real LLM provider based on LLM_CALLER_MODEL config."""
    from voicetest.config import Settings

    settings = Settings()
    chosen = model or settings.llm_caller_model or "gpt-4o-mini"

    if chosen.startswith("claude"):
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY required for Claude models")
        return AnthropicLLM(api_key=settings.anthropic_api_key, model=chosen)

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY required for OpenAI models")
    return OpenAILLM(api_key=settings.openai_api_key, model=chosen)
