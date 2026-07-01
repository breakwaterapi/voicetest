"""
twilio_target.py — TwilioOutbound: the real outbound-call CallTarget.

Implements the CallTarget protocol (connect/disconnect) backed by a real phone
call. The engine never knows it's talking to a real phone (ADR-3): connect()
returns the same AudioStream the loopback target returns; the ws_server bridges
Twilio's μ-law audio into it.

Flow
----
1. connect():
   - ADR-9 allowlist guard (raises BEFORE any REST call).
   - require PUBLIC_WS_URL (else clear "telephony not configured" error).
   - build an AudioStream, register it under a stream_id, dial Twilio with TwiML
     <Connect><Stream url=PUBLIC_WS_URL/twilio-stream/{stream_id}>.
   - wait for the WS to connect (Twilio answered) OR time out → NO_ANSWER.
   - Twilio AMD result later maps to VOICEMAIL/NO_ANSWER.
2. disconnect(): close the stream, hang up if live, fetch the dual-channel
   recording URL for the report.

Outcomes are surfaced via the OutcomeError exception which run_call maps to a
CallOutcome — a voicemail/no-answer is NOT a test failure.
"""

from __future__ import annotations

import asyncio
import uuid

from voicetest.config import Settings
from voicetest.log import vlog
from voicetest.models import CallOutcome
from voicetest.providers.targets import AudioStream
from voicetest.providers.telephony import check_allowlist, normalize_number


class TelephonyNotConfiguredError(RuntimeError):
    """Raised when required Twilio/tunnel config is missing."""


class OutcomeError(Exception):
    """
    Raised to signal a non-COMPLETED call outcome (no-answer, voicemail).

    Carries the CallOutcome so run_call can record it as a first-class outcome
    rather than a generic ERROR. These are NOT test failures.
    """

    def __init__(self, outcome: CallOutcome, message: str) -> None:
        super().__init__(message)
        self.outcome = outcome


# How long to wait for Twilio to connect the Media Stream back to us before
# declaring no-answer. Real outbound calls ring for ~20-30s.
_CONNECT_TIMEOUT_S = 45.0


class TwilioOutbound:
    """Real outbound call target via Twilio Programmable Voice + Media Streams."""

    def __init__(
        self,
        number: str,
        settings: Settings | None = None,
        registry=None,  # StreamRegistry; defaults to the shared ws_server.registry
        client=None,  # twilio.rest.Client; injected in tests
    ) -> None:
        self.number = number
        self._settings = settings or Settings()
        if registry is None:
            from voicetest.engine.ws_server import registry as default_registry

            registry = default_registry
        self._registry = registry
        self._client = client

        self._stream_id: str | None = None
        self._stream: AudioStream | None = None
        self._call_sid: str | None = None

    # ------------------------------------------------------------------ client

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from twilio.rest import Client
        except ImportError as e:  # pragma: no cover - exercised only without SDK
            raise TelephonyNotConfiguredError(
                "twilio not installed — pip install twilio"
            ) from e
        s = self._settings
        if not (s.twilio_account_sid and s.twilio_auth_token):
            raise TelephonyNotConfiguredError(
                "Twilio not configured — set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."
            )
        self._client = Client(s.twilio_account_sid, s.twilio_auth_token)
        return self._client

    # ----------------------------------------------------------------- connect

    async def connect(self) -> AudioStream:
        s = self._settings

        # ADR-9: defence-in-depth — refuse non-allowlisted numbers before dialing.
        check_allowlist(self.number, s)

        if not s.public_ws_url:
            raise TelephonyNotConfiguredError(
                "PUBLIC_WS_URL is not set — start `ngrok http <port>` and set "
                "PUBLIC_WS_URL to the wss:// URL so Twilio can stream audio back."
            )
        if not s.twilio_from_number:
            raise TelephonyNotConfiguredError(
                "TWILIO_FROM_NUMBER is not set — required to place an outbound call."
            )

        self._stream_id = uuid.uuid4().hex
        self._stream = AudioStream()
        connected = self._registry.register(self._stream_id, self._stream)

        ws_url = s.public_ws_url.rstrip("/") + f"/twilio-stream/{self._stream_id}"
        twiml = f'<Response><Connect><Stream url="{ws_url}"/></Connect></Response>'

        to = normalize_number(self.number)
        from_ = normalize_number(s.twilio_from_number)
        vlog("twilio", f"dialing to={to} from={from_}")
        vlog("twilio", f"stream WS url Twilio will connect back to: {ws_url}")
        vlog("twilio", f"TwiML: {twiml}")

        client = self._get_client()
        # client.calls.create is blocking (requests-based) → run off the loop.
        try:
            call = await asyncio.to_thread(
                client.calls.create,
                to=to,
                from_=from_,
                twiml=twiml,
                record=True,
                recording_channels="dual"
            )
        except Exception as exc:  # noqa: BLE001
            # Surface the exact Twilio REST error (auth, unverified number,
            # geo-permission, trial restriction) instead of a silent failure.
            vlog("twilio", f"!!! calls.create FAILED: {type(exc).__name__}: {exc}")
            self._registry.unregister(self._stream_id)
            raise
        self._call_sid = getattr(call, "sid", None)
        vlog("twilio", f"call created sid={self._call_sid} status={getattr(call, 'status', '?')}")
        vlog("twilio", f"waiting up to {_CONNECT_TIMEOUT_S:.0f}s for Twilio to connect the "
                       "media stream (= callee answered AND pressed a key on trial)…")

        # Wait for Twilio to actually connect the Media Stream (= answered).
        try:
            await asyncio.wait_for(connected.wait(), timeout=_CONNECT_TIMEOUT_S)
        except asyncio.TimeoutError as e:
            vlog("twilio", f"!!! TIMEOUT — Twilio never connected the stream within "
                           f"{_CONNECT_TIMEOUT_S:.0f}s. Causes: no answer; callee didn't press "
                           "a key on the trial gate; PUBLIC_WS_URL/ngrok unreachable; or "
                           f"<Connect><Stream> failed. sid={self._call_sid}")
            self._registry.unregister(self._stream_id)
            raise OutcomeError(
                CallOutcome.NO_ANSWER,
                f"No answer from {self.number} within {_CONNECT_TIMEOUT_S:.0f}s.",
            ) from e

        vlog("twilio", f"media stream CONNECTED for sid={self._call_sid} — bridging audio")
        return self._stream

    # -------------------------------------------------------------- disconnect

    async def disconnect(self) -> None:
        vlog("twilio", f"disconnect() called for sid={self._call_sid}")
        if self._stream is not None:
            await self._stream.close_outbound()
        if self._stream_id is not None:
            self._registry.unregister(self._stream_id)
        # Best-effort hang up if the call is still live.
        if self._call_sid and self._client is not None:
            try:
                await asyncio.to_thread(
                    self._client.calls(self._call_sid).update, status="completed"
                )
                vlog("twilio", f"hung up call sid={self._call_sid}")
            except Exception as exc:  # noqa: BLE001 - hangup is best-effort
                vlog("twilio", f"hangup best-effort failed (ignored): {exc}")
        self._stream = None
        self._stream_id = None

    @property
    def call_sid(self) -> str | None:
        return self._call_sid
