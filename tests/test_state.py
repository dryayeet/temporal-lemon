"""Tests for the legacy 6-field state module (`storage/state.py`).

This module is DEPRECATED — superseded by `storage/lemon_state.py` in stage 3
of the dyadic-state architecture. These tests cover only the surface that
remains: parsing helpers, load/save round-trips, and session-start overrides.
The new lemon-side schema is tested in `tests/test_lemon_state.py`.

`update_internal_state` and `format_internal_state` were removed in earlier
work — those tests are gone with them.
"""
import json

import pytest

from storage.state import (
    DEFAULT_STATE,
    SESSION_START_OVERRIDES,
    fresh_session_state,
    load_state,
    parse_state_response,
    save_state,
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
    # Save a partial dict via db to simulate a schema mismatch from older snapshots
    from storage import db as db_mod
    db_mod.save_state_snapshot({"mood": "good"})
    loaded = load_state()
    assert loaded["mood"] == "good"
    assert loaded["disposition"] == DEFAULT_STATE["disposition"]


# ---------- fresh_session_state ----------

def test_fresh_session_on_empty_db_uses_upbeat_baseline():
    out = fresh_session_state()
    assert out["energy"] == SESSION_START_OVERRIDES["energy"]
    assert out["engagement"] == SESSION_START_OVERRIDES["engagement"]
    assert out["mood"] == SESSION_START_OVERRIDES["mood"]
    assert out["disposition"] == SESSION_START_OVERRIDES["disposition"]


def test_fresh_session_overrides_drained_previous_state():
    drained = dict(DEFAULT_STATE,
                   mood="tired", energy="low", engagement="low",
                   disposition="slightly reserved",
                   emotional_thread="long day",
                   recent_activity="was talking with the user earlier")
    save_state(drained)

    out = fresh_session_state()
    # upbeat overrides applied
    assert out["energy"] == SESSION_START_OVERRIDES["energy"]
    assert out["engagement"] == SESSION_START_OVERRIDES["engagement"]
    assert out["mood"] == SESSION_START_OVERRIDES["mood"]
    assert out["disposition"] == SESSION_START_OVERRIDES["disposition"]
    # cross-session continuity preserved
    assert out["emotional_thread"] == "long day"
    assert out["recent_activity"] == "was talking with the user earlier"


def test_session_start_overrides_cover_every_energy_field():
    # sanity: the override set must touch each field that represents "am I into this"
    assert set(SESSION_START_OVERRIDES) == {"mood", "energy", "engagement", "disposition"}
