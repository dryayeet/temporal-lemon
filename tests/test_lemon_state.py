"""Tests for the lemon-side three-layer state (dyadic-state stage 3)."""
import copy

from prompts import format_lemon_state, LEMON_STATE_TAG
from storage import db
from storage.lemon_state import (
    DEFAULT_LEMON_STATE,
    LEMON_SESSION_START_STATE,
    apply_delta,
    fresh_lemon_session_state,
    load_lemon_state,
    migrate_legacy_state,
    save_lemon_state,
    validate_delta,
    validate_lemon_state,
)


# ---------- DEFAULT_LEMON_STATE comes from persona ----------

def test_default_pulls_traits_from_persona():
    from persona import LEMON_TRAITS
    assert DEFAULT_LEMON_STATE["traits"] == LEMON_TRAITS


def test_default_pulls_adaptations_from_persona():
    from persona import LEMON_ADAPTATIONS
    assert DEFAULT_LEMON_STATE["adaptations"]["values"] == LEMON_ADAPTATIONS["values"]
    assert DEFAULT_LEMON_STATE["adaptations"]["relational_stance"] == LEMON_ADAPTATIONS["relational_stance"]


def test_default_values_are_schwartz_tagged():
    """Persona values should ship tagged with Schwartz categories."""
    vals = DEFAULT_LEMON_STATE["adaptations"]["values"]
    assert all(isinstance(v, dict) for v in vals)
    assert all("label" in v and "schwartz" in v for v in vals)
    # Specific tags expected by the persona constants
    by_label = {v["label"]: v["schwartz"] for v in vals}
    assert by_label["honesty"] == "universalism"
    assert by_label["warmth without performance"] == "benevolence"
    assert by_label["calm"] == "security"


def test_fresh_session_repulls_values_from_persona():
    """If a previous snapshot has stale untagged values, fresh_lemon_session_state
    should re-pull tagged values from persona."""
    from persona import LEMON_ADAPTATIONS
    stale = copy.deepcopy(DEFAULT_LEMON_STATE)
    stale["adaptations"]["values"] = ["honesty", "calm"]   # legacy untagged
    save_lemon_state(stale)
    fresh = fresh_lemon_session_state()
    assert fresh["adaptations"]["values"] == LEMON_ADAPTATIONS["values"]


def test_load_normalizes_legacy_untagged_lemon_values():
    """Old snapshots with raw string values should normalize on load
    (independent of fresh_lemon_session_state's persona re-pull)."""
    from storage import db
    legacy = copy.deepcopy(DEFAULT_LEMON_STATE)
    legacy["adaptations"]["values"] = ["honesty", "calm"]
    db.save_lemon_state_snapshot(legacy)
    loaded = load_lemon_state()
    assert all(isinstance(v, dict) for v in loaded["adaptations"]["values"])
    labels = [v["label"] for v in loaded["adaptations"]["values"]]
    assert "honesty" in labels
    assert "calm" in labels


def test_default_state_is_neutral_pad():
    s = DEFAULT_LEMON_STATE["state"]
    assert s["pleasure"] == 0.0
    assert s["arousal"] == 0.0
    assert s["dominance"] == 0.0
    assert s["mood_label"] == "neutral"


# ---------- session-start re-pegging ----------

def test_fresh_session_repegs_pad_to_baseline():
    fresh = fresh_lemon_session_state()
    assert fresh["state"]["pleasure"] == LEMON_SESSION_START_STATE["pleasure"]
    assert fresh["state"]["mood_label"] == LEMON_SESSION_START_STATE["mood_label"]


def test_fresh_session_resets_relational_stance():
    """Even if a previous session ended with a shifted stance, fresh sessions
    re-peg to the persona baseline."""
    drained = copy.deepcopy(DEFAULT_LEMON_STATE)
    drained["adaptations"]["relational_stance"] = "polite but a little reserved"
    drained["state"]["pleasure"] = -0.4
    save_lemon_state(drained)

    fresh = fresh_lemon_session_state()
    assert fresh["adaptations"]["relational_stance"] == DEFAULT_LEMON_STATE["adaptations"]["relational_stance"]


def test_fresh_session_preserves_concerns():
    """Concerns and goals carry over so cross-session continuity works."""
    carried = copy.deepcopy(DEFAULT_LEMON_STATE)
    carried["adaptations"]["concerns"] = ["something happened with the user yesterday"]
    save_lemon_state(carried)

    fresh = fresh_lemon_session_state()
    assert "something happened with the user yesterday" in fresh["adaptations"]["concerns"]


# ---------- legacy migration ----------

def test_migrate_legacy_tired_low_state():
    legacy = {
        "mood": "tired", "energy": "low", "engagement": "low",
        "emotional_thread": "long day", "recent_activity": None,
        "disposition": "slightly reserved",
    }
    out = migrate_legacy_state(legacy)
    assert out["state"]["mood_label"] == "tired"
    assert out["state"]["arousal"] < 0   # tired + low energy = low arousal
    assert out["adaptations"]["relational_stance"] == "polite but a little reserved"
    assert out["adaptations"]["concerns"] == ["long day"]


def test_migrate_legacy_anxious_high_state():
    legacy = {"mood": "anxious", "energy": "high", "disposition": "warm"}
    out = migrate_legacy_state(legacy)
    assert out["state"]["mood_label"] == "anxious"
    assert out["state"]["arousal"] > 0.3   # anxious + high energy = high arousal
    assert out["adaptations"]["relational_stance"] == "warm and present"


def test_migrate_legacy_unknown_mood_falls_back_to_neutral():
    legacy = {"mood": "mysterious", "energy": "medium"}
    out = migrate_legacy_state(legacy)
    assert out["state"]["mood_label"] == "neutral"


def test_migrate_drops_recent_activity():
    legacy = {"mood": "good", "recent_activity": "watching TV"}
    out = migrate_legacy_state(legacy)
    # recent_activity is intentionally not preserved in the new schema
    assert "watching TV" not in str(out)


# ---------- load_lemon_state with legacy fallback ----------

def test_load_uses_legacy_state_when_no_lemon_state():
    """Cold-start migration: if there's an old state_snapshot but no
    lemon_state_snapshot, load_lemon_state should migrate the legacy row."""
    db.save_state_snapshot({
        "mood": "low", "energy": "low", "engagement": "low",
        "emotional_thread": "tough call earlier",
        "recent_activity": None, "disposition": "warm",
    })
    out = load_lemon_state()
    assert out["state"]["mood_label"] == "low"
    assert "tough call earlier" in out["adaptations"]["concerns"]


def test_load_returns_default_when_db_empty():
    out = load_lemon_state()
    assert out == DEFAULT_LEMON_STATE


def test_load_prefers_lemon_state_over_legacy():
    """If both tables have rows, lemon_state_snapshots wins."""
    db.save_state_snapshot({"mood": "anxious", "energy": "high"})
    new_shape = copy.deepcopy(DEFAULT_LEMON_STATE)
    new_shape["state"]["mood_label"] = "happy"
    save_lemon_state(new_shape)

    out = load_lemon_state()
    assert out["state"]["mood_label"] == "happy"


# ---------- validators reuse user_state pattern ----------

def test_validate_falls_back_to_lemon_default():
    out = validate_lemon_state({}, fallback=None)
    assert out == DEFAULT_LEMON_STATE


def test_validate_clamps_pad():
    out = validate_lemon_state({"state": {"pleasure": 5.0}}, fallback=DEFAULT_LEMON_STATE)
    assert out["state"]["pleasure"] == 1.0


# ---------- delta application via shared helpers ----------

def test_apply_delta_nudges_pad():
    delta = validate_delta({"pad": {"pleasure": 0.10, "arousal": 0.0, "dominance": 0.0}})
    out = apply_delta(DEFAULT_LEMON_STATE, delta)
    assert abs(out["state"]["pleasure"] - 0.10) < 1e-9


# ---------- format block ----------

def test_format_lemon_state_block_includes_tag():
    out = format_lemon_state(DEFAULT_LEMON_STATE)
    assert LEMON_STATE_TAG in out
    assert "</lemon_state>" in out


def test_format_lemon_state_includes_mood_and_pad():
    s = copy.deepcopy(DEFAULT_LEMON_STATE)
    s["state"] = {"pleasure": 0.30, "arousal": 0.10, "dominance": 0.0, "mood_label": "content"}
    out = format_lemon_state(s)
    assert "content" in out
    assert "pleasure +0.30" in out


def test_format_lemon_state_renders_concerns():
    s = copy.deepcopy(DEFAULT_LEMON_STATE)
    s["adaptations"]["concerns"] = ["user seemed off earlier"]
    out = format_lemon_state(s)
    assert "user seemed off earlier" in out


def test_format_lemon_state_collapses_empty_concerns():
    out = format_lemon_state(DEFAULT_LEMON_STATE)
    # When concerns is empty, the formatter emits the "nothing in particular" line
    assert "Quietly on your mind: nothing in particular" in out


# ---------- save / load round-trip ----------

def test_save_load_roundtrip():
    s = copy.deepcopy(DEFAULT_LEMON_STATE)
    s["state"]["mood_label"] = "tired"
    s["state"]["pleasure"] = -0.2
    s["adaptations"]["concerns"] = ["something specific"]
    save_lemon_state(s)
    loaded = load_lemon_state()
    assert loaded["state"]["mood_label"] == "tired"
    assert abs(loaded["state"]["pleasure"] - (-0.2)) < 1e-9
    assert loaded["adaptations"]["concerns"] == ["something specific"]
