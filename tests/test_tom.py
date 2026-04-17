import json

import pytest
import requests

import tom
from tom import DEFAULT_TOM, format_tom_block, theory_of_mind


# ---------- _parse ----------

def test_parse_extracts_three_keys():
    raw = json.dumps({
        "feeling": "tired and overwhelmed",
        "avoid": "don't pivot to advice",
        "what_helps": "just listen",
    })
    out = tom._parse(raw)
    assert out["feeling"] == "tired and overwhelmed"
    assert out["avoid"] == "don't pivot to advice"
    assert out["what_helps"] == "just listen"


def test_parse_strips_json_fence():
    raw = "```json\n" + json.dumps({"feeling": "x", "avoid": "y", "what_helps": "z"}) + "\n```"
    out = tom._parse(raw)
    assert out == {"feeling": "x", "avoid": "y", "what_helps": "z"}


def test_parse_handles_missing_keys():
    out = tom._parse(json.dumps({"feeling": "ok"}))
    assert out["feeling"] == "ok"
    assert out["avoid"] is None
    assert out["what_helps"] is None


def test_parse_treats_empty_strings_as_none():
    out = tom._parse(json.dumps({"feeling": "", "avoid": "  ", "what_helps": "x"}))
    assert out["feeling"] is None
    assert out["avoid"] is None
    assert out["what_helps"] == "x"


def test_parse_rejects_non_object():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        tom._parse('"a string"')


# ---------- format_tom_block ----------

def test_format_includes_all_three():
    out = format_tom_block({"feeling": "sad", "avoid": "advice", "what_helps": "listen"})
    assert "<theory_of_mind>" in out
    assert "sad" in out
    assert "advice" in out
    assert "listen" in out


def test_format_handles_nones():
    out = format_tom_block({"feeling": None, "avoid": None, "what_helps": None})
    assert "unclear" in out
    assert out.count("(no specific guidance)") == 2


# ---------- theory_of_mind (HTTP mocked) ----------

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
    def json(self): return self._payload


def test_tom_happy_path(monkeypatch):
    fake_tom = {"feeling": "anxious", "avoid": "rushing", "what_helps": "slow down"}
    fake_payload = {"choices": [{"message": {"content": json.dumps(fake_tom)}}]}
    monkeypatch.setattr(tom.requests, "post", lambda *a, **kw: _FakeResponse(fake_payload))

    out = theory_of_mind("the deadline is tomorrow", emotion={"primary": "anxiety", "intensity": 0.7})
    assert out["feeling"] == "anxious"
    assert out["what_helps"] == "slow down"


def test_tom_returns_default_on_error(monkeypatch):
    monkeypatch.setattr(tom.requests, "post",
                        lambda *a, **kw: _FakeResponse({"err": "boom"}, status=500))
    assert theory_of_mind("hi") == DEFAULT_TOM


def test_tom_returns_default_on_bad_json(monkeypatch):
    fake = {"choices": [{"message": {"content": "garbage"}}]}
    monkeypatch.setattr(tom.requests, "post", lambda *a, **kw: _FakeResponse(fake))
    assert theory_of_mind("hi") == DEFAULT_TOM


def test_tom_includes_emotion_in_prompt(monkeypatch):
    captured = {}
    def fake_post(*a, **kw):
        captured["body"] = kw.get("json", {})
        return _FakeResponse({"choices": [{"message": {"content": json.dumps(DEFAULT_TOM)}}]})
    monkeypatch.setattr(tom.requests, "post", fake_post)

    theory_of_mind("test", emotion={"primary": "fear", "intensity": 0.9, "undertones": ["anxiety"]})
    prompt = captured["body"]["messages"][0]["content"]
    assert "fear" in prompt
    assert "0.90" in prompt
