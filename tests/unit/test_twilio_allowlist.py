"""
Unit tests for the ADR-9 allowlist guard and TwilioOutbound.connect() — all with a
mocked Twilio client. CI must NEVER place a real call.
"""

from __future__ import annotations

import pytest

from voicetest.config import Settings
from voicetest.providers.telephony import AllowlistError, check_allowlist, is_loopback
from voicetest.providers.twilio_target import (
    OutcomeError,
    TelephonyNotConfiguredError,
    TwilioOutbound,
)


def _settings(**over) -> Settings:
    base = dict(
        verified_dest_numbers="+18573132352, +15551230000",
        twilio_account_sid="AC_test",
        twilio_auth_token="tok_test",
        twilio_from_number="+15550009999",
        public_ws_url="wss://example.ngrok.io",
    )
    base.update(over)
    return Settings(**base)


# ---------------------------------------------------------------------------
# Pure allowlist guard
# ---------------------------------------------------------------------------


def test_loopback_sentinels_always_allowed():
    s = _settings(verified_dest_numbers=None)
    check_allowlist("loopback", s)  # no raise
    check_allowlist("+00000000000", s)  # no raise
    assert is_loopback("loopback") and is_loopback("+0123")


def test_allowlisted_number_passes():
    check_allowlist("+18573132352", _settings())  # no raise


def test_number_normalized_before_compare():
    # Spaces/dashes must not matter.
    check_allowlist("+1 857-313-2352", _settings())  # no raise


def test_non_allowlisted_number_raises():
    with pytest.raises(AllowlistError):
        check_allowlist("+19998887777", _settings())


def test_empty_allowlist_rejects_real_numbers():
    with pytest.raises(AllowlistError):
        check_allowlist("+18573132352", _settings(verified_dest_numbers=None))


# ---------------------------------------------------------------------------
# TwilioOutbound.connect — dialing guarded, mocked client
# ---------------------------------------------------------------------------


class _FakeCall:
    sid = "CA_fake_sid"


class _FakeCalls:
    def __init__(self) -> None:
        self.create_kwargs = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return _FakeCall()


class _FakeClient:
    def __init__(self) -> None:
        self.calls = _FakeCalls()


class _Registry:
    """Minimal StreamRegistry double whose connect-event we control."""

    def __init__(self, autoconnect: bool) -> None:
        self.autoconnect = autoconnect
        self.registered: dict = {}

    def register(self, stream_id, stream):
        import asyncio

        ev = asyncio.Event()
        if self.autoconnect:
            ev.set()  # simulate Twilio connecting back instantly (answered)
        self.registered[stream_id] = (stream, ev)
        return ev

    def unregister(self, stream_id):
        self.registered.pop(stream_id, None)


@pytest.mark.asyncio
async def test_connect_refuses_non_allowlisted_before_any_rest_call():
    client = _FakeClient()
    target = TwilioOutbound(
        "+19998887777", settings=_settings(), registry=_Registry(True), client=client
    )
    with pytest.raises(AllowlistError):
        await target.connect()
    # The crucial assertion: no REST call was attempted.
    assert client.calls.create_kwargs is None


@pytest.mark.asyncio
async def test_connect_requires_public_ws_url():
    target = TwilioOutbound(
        "+18573132352",
        settings=_settings(public_ws_url=None),
        registry=_Registry(True),
        client=_FakeClient(),
    )
    with pytest.raises(TelephonyNotConfiguredError):
        await target.connect()


@pytest.mark.asyncio
async def test_connect_dials_allowlisted_number_with_stream_twiml():
    client = _FakeClient()
    target = TwilioOutbound(
        "+18573132352", settings=_settings(), registry=_Registry(True), client=client
    )
    stream = await target.connect()
    kw = client.calls.create_kwargs
    assert kw is not None
    assert kw["to"] == "+18573132352"
    assert kw["from_"] == "+15550009999"
    assert "<Connect><Stream" in kw["twiml"]
    assert "/twilio-stream/" in kw["twiml"]
    assert kw["record"] is True
    assert kw["recording_channels"] == "dual"
    assert kw["machine_detection"] == "Enable"
    assert target.call_sid == "CA_fake_sid"
    assert stream is not None


@pytest.mark.asyncio
async def test_connect_times_out_to_no_answer(monkeypatch):
    # Registry never sets the connected event → connect() should declare NO_ANSWER.
    monkeypatch.setattr(
        "voicetest.providers.twilio_target._CONNECT_TIMEOUT_S", 0.05, raising=True
    )
    target = TwilioOutbound(
        "+18573132352", settings=_settings(), registry=_Registry(False), client=_FakeClient()
    )
    with pytest.raises(OutcomeError) as ei:
        await target.connect()
    from voicetest.models import CallOutcome

    assert ei.value.outcome == CallOutcome.NO_ANSWER
