import json

import pytest
import requests

import state
from state import (
    DEFAULT_STATE,
    format_internal_state,
    load_state,
    parse_state_response,
    save_state,
    update_internal_state,
)


# ---------- parse_state_response ----------

def test_parse_plain_json():
    raw = json.dumps({
        "mood": "good", "energy": "low", "engagement": "deep",
        "emotional_thread": None, "recent_activity": None, "disposition": "warm",
    })
    out = parse_state_response(raw, fallback=DEFAULT_STATE)
    assert out["mood"] == "good"
    assert out["energy"] == "low"


def test_parse_strips_json_fence():
    raw = "```json\n" + json.dumps({"mood": "low"}) + "\n```"
    out = parse_state_response(raw, fallback=DEFAULT_STATE)
    assert out["mood"] == "low"
    assert out["energy"] == DEFAULT_STATE["energy"]


def test_parse_strips_bare_fence():
    raw = "```\n" + json.dumps({"mood": "content"}) + "\n```"
    out = parse_state_response(raw, fallback=DEFAULT_STATE)
    assert out["mood"] == "content"


def test_parse_drops_unknown_keys():
    raw = json.dumps({"mood": "good", "bogus_key": "x"})
    out = parse_state_response(raw, fallback=DEFAULT_STATE)
    assert set(out) == set(DEFAULT_STATE)
    assert "bogus_key" not in out


def test_parse_rejects_non_object():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_state_response("[1,2,3]", fallback=DEFAULT_STATE)


def test_parse_raises_on_malformed_json():
    with pytest.raises(json.JSONDecodeError):
        parse_state_response("not json at all", fallback=DEFAULT_STATE)


# ---------- format_internal_state ----------

def test_format_includes_all_fields():
    out = format_internal_state(DEFAULT_STATE)
    assert "<internal_state>" in out
    assert "Mood: neutral" in out
    assert "Energy: medium" in out
    assert "Engagement level: normal" in out
    assert "Disposition toward this person right now: warm" in out


def test_format_handles_null_fields():
    out = format_internal_state(DEFAULT_STATE)
    assert "nothing specific" in out
    assert "nothing worth mentioning" in out


# ---------- load_state / save_state (via db) ----------

def test_load_returns_default_when_db_empty():
    out = load_state()
    assert out == DEFAULT_STATE
    assert out is not DEFAULT_STATE   # fresh copy


def test_save_load_roundtrip():
    nudged = dict(DEFAULT_STATE, mood="tired", energy="low", emotional_thread="long day")
    save_state(nudged)
    loaded = load_state()
    assert loaded["mood"] == "tired"
    assert loaded["energy"] == "low"
    assert loaded["emotional_thread"] == "long day"


def test_load_returns_latest_snapshot():
    save_state(dict(DEFAULT_STATE, mood="good"))
    save_state(dict(DEFAULT_STATE, mood="anxious"))
    save_state(dict(DEFAULT_STATE, mood="content"))
    assert load_state()["mood"] == "content"


def test_load_fills_missing_keys_from_defaults():
    # Save a partial dict directly via db to simulate a schema mismatch
    import db as db_mod
    db_mod.save_state_snapshot({"mood": "good"})
    loaded = load_state()
    assert loaded["mood"] == "good"
    assert loaded["disposition"] == DEFAULT_STATE["disposition"]


# ---------- update_internal_state (HTTP mocked) ----------

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


def test_update_happy_path(monkeypatch):
    nudged = dict(DEFAULT_STATE, mood="good", energy="high")
    fake_payload = {"choices": [{"message": {"content": json.dumps(nudged)}}]}
    monkeypatch.setattr(state.requests, "post", lambda *a, **kw: _FakeResponse(fake_payload))

    out = update_internal_state(DEFAULT_STATE, "user said something", "bot replied")
    assert out["mood"] == "good"
    assert out["energy"] == "high"


def test_update_returns_previous_on_http_error(monkeypatch):
    monkeypatch.setattr(
        state.requests, "post",
        lambda *a, **kw: _FakeResponse({"error": "boom"}, status=500),
    )
    assert update_internal_state(DEFAULT_STATE, "u", "b") == DEFAULT_STATE


def test_update_returns_previous_on_bad_json(monkeypatch):
    fake = {"choices": [{"message": {"content": "not json"}}]}
    monkeypatch.setattr(state.requests, "post", lambda *a, **kw: _FakeResponse(fake))
    assert update_internal_state(DEFAULT_STATE, "u", "b") == DEFAULT_STATE


def test_update_returns_previous_on_request_exception(monkeypatch):
    def raising_post(*a, **kw):
        raise requests.ConnectionError("network down")
    monkeypatch.setattr(state.requests, "post", raising_post)
    assert update_internal_state(DEFAULT_STATE, "u", "b") == DEFAULT_STATE
