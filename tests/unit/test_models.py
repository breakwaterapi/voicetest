from voicetest.models import CallOutcome, CallResult, Event, Report, Turn


def test_turn():
    t = Turn(speaker="caller", text="hello", start_ms=0, end_ms=500)
    assert t.speaker == "caller"
    assert t.text == "hello"
    assert t.start_ms == 0
    assert t.end_ms == 500


def test_event():
    e = Event(kind="vad_start", at_ms=123, data={"confidence": 0.9})
    assert e.kind == "vad_start"
    assert e.at_ms == 123
    assert e.data["confidence"] == 0.9


def test_event_default_data():
    e = Event(kind="noise_on", at_ms=0)
    assert e.data == {}


def test_call_outcome_values():
    assert CallOutcome.COMPLETED == "completed"
    assert CallOutcome.NO_ANSWER == "no_answer"
    assert CallOutcome.VOICEMAIL == "voicemail"
    assert CallOutcome.ERROR == "error"


def test_call_outcome_is_str():
    assert isinstance(CallOutcome.COMPLETED, str)


def test_call_result():
    turns = [Turn(speaker="agent", text="Hi", start_ms=0, end_ms=300)]
    events = [Event(kind="vad_stop", at_ms=300)]
    result = CallResult(
        scenario_id="s1",
        outcome=CallOutcome.COMPLETED,
        turns=turns,
        events=events,
        recording_path="/tmp/rec.wav",
        duration_ms=5000,
    )
    assert result.scenario_id == "s1"
    assert result.outcome == CallOutcome.COMPLETED
    assert len(result.turns) == 1
    assert result.error is None


def test_call_result_with_error():
    result = CallResult(
        scenario_id="s2",
        outcome=CallOutcome.ERROR,
        turns=[],
        events=[],
        recording_path=None,
        duration_ms=0,
        error="connection refused",
    )
    assert result.error == "connection refused"
    assert result.recording_path is None


def test_report():
    result = CallResult(
        scenario_id="s1",
        outcome=CallOutcome.COMPLETED,
        turns=[],
        events=[],
        recording_path=None,
        duration_ms=1000,
    )
    report = Report(
        call=result,
        passed=True,
        dimensions={"latency": {"score": 0.9, "reason": "fast"}},
        summary="All good",
    )
    assert report.passed is True
    assert report.dimensions["latency"]["score"] == 0.9
    assert report.summary == "All good"
