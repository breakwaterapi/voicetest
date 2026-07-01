"""
log.py — dead-simple, unmissable terminal logging for debugging the call flow.

Every high-signal step in the telephony/engine path calls ``vlog(tag, msg)``.
Output goes to stderr with a millisecond timestamp and a colored ``[tag]`` prefix
so you can watch a live call unfold in the terminal running uvicorn, e.g.::

    19:42:01.123 [job ] run started run=ab12 scenario=… number=+1857… n=1
    19:42:01.456 [twilio] dialing +1857… from +1833… sid=CA…
    19:42:09.789 [ws  ] >>> WebSocket OPENED for stream_id=… (Twilio connected!)
    19:42:09.790 [ws  ] event=start streamSid=MZ…
    19:42:09.812 [pipe] caller opening line synthesized (… frames)

This is intentionally print-based (not the logging module) so it shows up no
matter how uvicorn/pytest configure logging. Set VOICETEST_DEBUG=0 to silence.
"""

from __future__ import annotations

import os
import sys
import time

_ENABLED = os.environ.get("VOICETEST_DEBUG", "1") != "0"
_START = time.monotonic()

# ANSI colors per tag so the eye can follow one subsystem at a time.
_COLORS = {
    "job": "\033[35m",      # magenta
    "twilio": "\033[36m",   # cyan
    "ws": "\033[32m",       # green
    "pipe": "\033[33m",     # yellow
    "call": "\033[34m",     # blue
}
_RESET = "\033[0m"


def vlog(tag: str, msg: str) -> None:
    """Print a timestamped, tagged debug line to stderr (flushed immediately)."""
    if not _ENABLED:
        return
    elapsed = time.monotonic() - _START
    color = _COLORS.get(tag, "")
    line = f"{elapsed:8.3f}s {color}[{tag:^6}]{_RESET} {msg}"
    print(line, file=sys.stderr, flush=True)
