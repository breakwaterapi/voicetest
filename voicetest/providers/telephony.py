"""
telephony.py — shared helpers for real outbound calls (ADR-9 compliance).

The allowlist guard lives here so BOTH the API layer (fast 422 before a run is
created) and the TwilioOutbound target (defence-in-depth, raises before any REST
call) enforce the same rule — there is exactly one definition of "verified".
"""

from __future__ import annotations

from voicetest.config import Settings


class AllowlistError(ValueError):
    """Raised when a destination number is not in the verified allowlist."""


def normalize_number(number: str) -> str:
    """Strip spaces/dashes so '+1 857-313-2352' compares equal to '+18573132352'."""
    return number.replace("-", "").replace(" ", "")


def is_loopback(number: str) -> bool:
    """Loopback sentinel and +0… test numbers never touch real telephony."""
    return number == "loopback" or number.startswith("+0")


def check_allowlist(number: str, settings: Settings) -> None:
    """
    Raise AllowlistError if *number* is a real number not in VERIFIED_DEST_NUMBERS.

    Loopback / +0… numbers are always allowed (they never dial out). This is the
    single source of truth for ADR-9; the API maps the error to HTTP 422 and the
    target raises it before any Twilio REST call.
    """
    if is_loopback(number):
        return
    allowed = settings.verified_dest_list
    normalized = normalize_number(number)
    if not allowed or normalized not in allowed:
        raise AllowlistError(
            f"Number {number!r} is not in the verified allowlist. "
            "Set VERIFIED_DEST_NUMBERS or use a loopback number for testing."
        )
