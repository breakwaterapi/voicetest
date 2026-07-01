"""
The real judge models sometimes wrap their JSON in ```json … ``` markdown fences.
_parse_response must tolerate that (regression test for a run-blocking bug).
"""

from __future__ import annotations

from voicetest.scenarios.models import ScoringConfig
from voicetest.scoring.judge import _parse_response, _strip_json_fences

_RUBRIC = ScoringConfig(objective="resolve the issue", pass_criteria=["did the thing"])

_BODY = (
    '{"goal_completed": true, "reason": "ok", '
    '"criteria": [{"text": "did the thing", "met": true, "evidence": "yes"}]}'
)


def test_strip_fences_json_label():
    assert _strip_json_fences("```json\n" + _BODY + "\n```") == _BODY


def test_strip_fences_plain():
    assert _strip_json_fences("```\n" + _BODY + "\n```") == _BODY


def test_strip_fences_noop_on_bare_json():
    assert _strip_json_fences(_BODY) == _BODY


def test_parse_response_accepts_fenced_json():
    result = _parse_response("```json\n" + _BODY + "\n```", _RUBRIC)
    assert result.goal_completed is True
    assert result.criteria[0].met is True


def test_parse_response_accepts_bare_json():
    result = _parse_response(_BODY, _RUBRIC)
    assert result.goal_completed is True
