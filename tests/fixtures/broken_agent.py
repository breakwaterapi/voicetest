"""
Broken-agent fixture factory.

Returns an agent_fn that is deliberately bad in ways the scorer must detect:
  - Slow: injects an asyncio sleep so response latency is always high.
  - Ignores barge-in: always responds with a fixed canned reply regardless
    of what the caller said.
  - Mis-hears numbers: replaces digit words / digits in the last caller turn
    with "umm" to simulate mis-recognition under noise.

Usage in tests:
    from tests.fixtures.broken_agent import make_broken_agent
    loopback = LocalLoopback(agent_fn=make_broken_agent())
"""

from __future__ import annotations

import re
from collections.abc import Callable

from voicetest.models import Turn

AgentFn = Callable[[list[Turn]], str]

_DIGIT_PATTERN = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
    re.IGNORECASE,
)


def make_broken_agent(
    *,
    delay_s: float = 2.0,
    fixed_reply: str = "Sorry, could you repeat that? I didn't quite catch it.",
) -> AgentFn:
    """
    Return an agent_fn that:
    - Sleeps for `delay_s` seconds (simulating a slow/unresponsive agent).
    - Always returns `fixed_reply` (ignores barge-in / caller content).

    The sleep is synchronous (time.sleep) inside the fn so it blocks the
    loopback loop — this is intentional: it models an agent that holds the
    line without yielding, which is exactly what barge-in tests need to detect.
    """
    import time

    def _broken(turns: list[Turn]) -> str:  # noqa: ARG001
        time.sleep(delay_s)
        return fixed_reply

    return _broken


def make_mishearing_agent(
    *,
    delay_s: float = 0.5,
) -> AgentFn:
    """
    Variant: responds quickly but replaces any digit / number word with "umm",
    simulating mis-recognition under noise. Useful for noise-scoring tests.
    """

    def _mishearing(turns: list[Turn]) -> str:
        if not turns:
            return "Hello, how can I help?"
        last_caller = next(
            (t.text for t in reversed(turns) if t.speaker == "caller"), ""
        )
        mangled = _DIGIT_PATTERN.sub("umm", last_caller)
        return f"I heard: {mangled}. Is that correct?"

    return _mishearing
