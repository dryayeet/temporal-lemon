import json

import pytest
import requests

import emotion
from emotion import (
    DEFAULT_EMOTION,
    EMOTION_LABELS,
    classify_emotion,
    format_emotion_block,
)


# ---------- _parse ----------

def test_parse_plain_json():
    raw = json.dumps({
        "primary": "sadness", "intensity": 0.6,
        "underlying_need": "feel heard", "undertones": ["loneliness"],
    })
    out = emotion._parse(raw)
    assert out["primary"] == "sadness"
    assert out["intensity"] == 0.6
    assert out["underlying_need"] == "feel heard"
    assert out["undertones"] == ["loneliness"]


def test_parse_strips_json_fence():
    raw = "```json\n" + json.dumps({"primary": "joy", "intensity": 0.4}) + "\n```"
    out = emotion._parse(raw)
    assert out["primary"] == "joy"
    assert out["intensity"] == 0.4


def test_parse_clamps_intensity_to_unit_interval():
    out = emotion._parse(json.dumps({"primary": "anger", "intensity": 4.2}))
    assert out["intensity"] == 1.0
    out = emotion._parse(json.dumps({"primary": "anger", "intensity": -1}))
    assert out["intensity"] == 0.0


def test_parse_drops_unknown_primary_label():
    out = emotion._parse(json.dumps({"primary": "wibble", "intensity": 0.5}))
    assert out["primary"] == "neutral"


def test_parse_filters_unknown_undertones():
    raw = json.dumps({
        "primary": "sadness", "intensity": 0.5,
        "undertones": ["loneliness", "wibble", "anger"],
    })
    assert emotion._parse(raw)["undertones"] == ["loneliness", "anger"]


def test_parse_caps_undertones_at_three():
    raw = json.dumps({
        "primary": "sadness", "intensity": 0.5,
        "undertones": EMOTION_LABELS[:6],  # six valid labels
    })
    assert len(emotion._parse(raw)["undertones"]) == 3


def test_parse_handles_null_underlying_need():
    out = emotion._parse(json.dumps({
        "primary": "neutral", "intensity": 0.2, "underlying_need": None,
    }))
    assert out["underlying_need"] is None


def test_parse_rejects_non_object():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        emotion._parse("[1, 2, 3]")


# ---------- format_emotion_block ----------

def test_format_includes_primary_and_intensity():
    out = format_emotion_block({
        "primary": "sadness", "intensity": 0.7,
        "underlying_need": "be heard", "undertones": ["loneliness"],
    })
    assert "<user_emotion>" in out
    assert "sadness" in out
    assert "be heard" in out
    assert "loneliness" in out


def test_format_handles_missing_fields():
    out = format_emotion_block({"primary": "neutral", "intensity": 0.1})
    assert "neutral" in out
    assert "unclear" in out  # null underlying_need
    assert "none" in out     # empty undertones


def test_format_intensity_word_buckets():
    assert "mild" in format_emotion_block({"primary": "joy", "intensity": 0.1})
    assert "moderate" in format_emotion_block({"primary": "joy", "intensity": 0.4})
    assert "strong" in format_emotion_block({"primary": "joy", "intensity": 0.7})
    assert "very strong" in format_emotion_block({"primary": "joy", "intensity": 0.95})


# ---------- classify_emotion (HTTP mocked) ----------

class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload) if isinstance(payload, dict) else str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f"{self.status_code}")
            err.response = self
            raise err

    def json(self):
        return self._payload


def test_classify_happy_path(monkeypatch):
    fake_emotion = {"primary": "anger", "intensity": 0.8, "undertones": []}
    fake_payload = {"choices": [{"message": {"content": json.dumps(fake_emotion)}}]}
    monkeypatch.setattr(emotion.requests, "post", lambda *a, **kw: _FakeResponse(fake_payload))

    out = classify_emotion("I'm furious about this", recent_msgs=[])
    assert out["primary"] == "anger"
    assert out["intensity"] == 0.8


def test_classify_returns_default_on_http_error(monkeypatch):
    monkeypatch.setattr(
        emotion.requests, "post",
        lambda *a, **kw: _FakeResponse({"error": "boom"}, status=500),
    )
    assert classify_emotion("hi") == DEFAULT_EMOTION


def test_classify_returns_default_on_bad_json(monkeypatch):
    fake = {"choices": [{"message": {"content": "not json"}}]}
    monkeypatch.setattr(emotion.requests, "post", lambda *a, **kw: _FakeResponse(fake))
    assert classify_emotion("hi") == DEFAULT_EMOTION


def test_classify_returns_default_on_request_exception(monkeypatch):
    def raising_post(*a, **kw):
        raise requests.ConnectionError("network down")
    monkeypatch.setattr(emotion.requests, "post", raising_post)
    assert classify_emotion("hi") == DEFAULT_EMOTION


def test_classify_passes_recent_msgs_into_prompt(monkeypatch):
    captured = {}
    def fake_post(*a, **kw):
        captured["body"] = kw.get("json", {})
        return _FakeResponse({"choices": [{"message": {"content": json.dumps({
            "primary": "neutral", "intensity": 0.1
        })}}]})
    monkeypatch.setattr(emotion.requests, "post", fake_post)

    classify_emotion("now what?", recent_msgs=[
        {"role": "user", "content": "earlier message"},
        {"role": "assistant", "content": "earlier reply"},
    ])
    prompt_text = captured["body"]["messages"][0]["content"]
    assert "earlier message" in prompt_text
    assert "earlier reply" in prompt_text
