"""Tests for the user-side three-layer state (dyadic-state stage 1)."""
import copy

from prompts import format_user_state_block
from storage.user_state import (
    DEFAULT_USER_STATE,
    MOOD_LABELS,
    apply_delta,
    fresh_user_session_state,
    is_cold_start,
    load_user_state,
    save_user_state,
    validate_delta,
    validate_user_state,
)


# ---------- validate_user_state ----------

def test_validate_clamps_pad_to_unit_interval():
    parsed = {"state": {"pleasure": 5.0, "arousal": -3.0, "dominance": 0.2, "mood_label": "calm"}}
    out = validate_user_state(parsed, fallback=DEFAULT_USER_STATE)
    assert out["state"]["pleasure"] == 1.0
    assert out["state"]["arousal"] == -1.0
    assert out["state"]["dominance"] == 0.2


def test_validate_drops_unknown_mood_label():
    parsed = {"state": {"mood_label": "delighted"}}  # not in MOOD_LABELS
    out = validate_user_state(parsed, fallback=DEFAULT_USER_STATE)
    assert out["state"]["mood_label"] == "neutral"


def test_validate_caps_adaptation_list_length():
    parsed = {"adaptations": {"current_goals": [f"goal {i}" for i in range(20)]}}
    out = validate_user_state(parsed, fallback=DEFAULT_USER_STATE)
    assert len(out["adaptations"]["current_goals"]) == 5


def test_validate_clamps_traits():
    parsed = {"traits": {"openness": 2.5, "neuroticism": -4.0}}
    out = validate_user_state(parsed, fallback=DEFAULT_USER_STATE)
    assert out["traits"]["openness"] == 1.0
    assert out["traits"]["neuroticism"] == -1.0


def test_validate_drops_unknown_trait_keys():
    parsed = {"traits": {"openness": 0.4, "rizz": 0.9}}
    out = validate_user_state(parsed, fallback=DEFAULT_USER_STATE)
    assert "rizz" not in out["traits"]
    assert set(out["traits"].keys()) == set(DEFAULT_USER_STATE["traits"].keys())


def test_validate_dedupes_string_lists_case_insensitively():
    parsed = {"adaptations": {"values": ["family", "Family", "career"]}}
    out = validate_user_state(parsed, fallback=DEFAULT_USER_STATE)
    assert len(out["adaptations"]["values"]) == 2
    # Legacy strings normalize to {label, schwartz=None}
    assert all(isinstance(v, dict) for v in out["adaptations"]["values"])
    assert all(v["schwartz"] is None for v in out["adaptations"]["values"])


def test_validate_preserves_tagged_value_dicts():
    parsed = {"adaptations": {"values": [
        {"label": "family", "schwartz": "benevolence"},
        {"label": "career", "schwartz": "achievement"},
    ]}}
    out = validate_user_state(parsed, fallback=DEFAULT_USER_STATE)
    labels = {v["label"]: v["schwartz"] for v in out["adaptations"]["values"]}
    assert labels["family"] == "benevolence"
    assert labels["career"] == "achievement"


def test_validate_drops_unknown_schwartz_tag_to_null():
    parsed = {"adaptations": {"values": [{"label": "x", "schwartz": "not_a_tag"}]}}
    out = validate_user_state(parsed, fallback=DEFAULT_USER_STATE)
    assert out["adaptations"]["values"] == [{"label": "x", "schwartz": None}]


def test_validate_canonicalizes_alias_schwartz_tags():
    parsed = {"adaptations": {"values": [{"label": "creativity", "schwartz": "self-direction"}]}}
    out = validate_user_state(parsed, fallback=DEFAULT_USER_STATE)
    assert out["adaptations"]["values"][0]["schwartz"] == "self_direction"


# ---------- validate_delta ----------

def test_validate_delta_clamps_pad_magnitude():
    parsed = {"pad": {"pleasure": 0.9, "arousal": -0.7, "dominance": 0.05}}
    out = validate_delta(parsed)
    # _PAD_NUDGE_CAP = 0.15
    assert out["pad"]["pleasure"] == 0.15
    assert out["pad"]["arousal"] == -0.15
    assert out["pad"]["dominance"] == 0.05


def test_validate_delta_freezes_traits():
    parsed = {"trait_nudges": {"openness": 0.5, "neuroticism": -0.3}}
    out = validate_delta(parsed)
    # _TRAIT_NUDGE_CAP = 0.02
    assert out["trait_nudges"]["openness"] == 0.02
    assert out["trait_nudges"]["neuroticism"] == -0.02


def test_validate_delta_zero_on_garbage():
    out = validate_delta("not a dict")
    assert out["pad"] == {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0}
    assert out["mood_label"] is None
    assert out["trait_nudges"] == {}
    assert out["goal_add"] == []


def test_validate_delta_caps_list_lengths():
    parsed = {"goal_add": ["a", "b", "c", "d"]}
    out = validate_delta(parsed)
    assert len(out["goal_add"]) == 2


# ---------- apply_delta ----------

def test_apply_delta_pad_nudge_clamps_to_unit_interval():
    prev = copy.deepcopy(DEFAULT_USER_STATE)
    prev["state"]["pleasure"] = 0.95
    out = apply_delta(prev, {"pad": {"pleasure": 0.15, "arousal": 0.0, "dominance": 0.0}})
    assert out["state"]["pleasure"] == 1.0   # clamped after nudge


def test_apply_delta_freezes_traits_via_validation_first():
    # Even if a delta sneaks past with a 0.5 trait nudge, validate_delta caps it at 0.02
    prev = copy.deepcopy(DEFAULT_USER_STATE)
    baseline = prev["traits"]["openness"]
    raw_delta = {"trait_nudges": {"openness": 0.5}}
    sane_delta = validate_delta(raw_delta)
    out = apply_delta(prev, sane_delta)
    # nudge applied on top of whatever baseline DEFAULT_USER_STATE seeds
    assert abs(out["traits"]["openness"] - (baseline + 0.02)) < 1e-9


def test_apply_delta_adds_goals_dedup():
    prev = copy.deepcopy(DEFAULT_USER_STATE)
    prev["adaptations"]["current_goals"] = ["prep for exam"]
    delta = validate_delta({"goal_add": ["prep for exam", "thesis defense"]})
    out = apply_delta(prev, delta)
    assert out["adaptations"]["current_goals"] == ["prep for exam", "thesis defense"]


def test_apply_delta_removes_goals_then_adds():
    prev = copy.deepcopy(DEFAULT_USER_STATE)
    prev["adaptations"]["current_goals"] = ["prep for exam", "fix bug"]
    delta = validate_delta({"goal_remove": ["prep for exam"], "goal_add": ["start thesis"]})
    out = apply_delta(prev, delta)
    assert "prep for exam" not in out["adaptations"]["current_goals"]
    assert "fix bug" in out["adaptations"]["current_goals"]
    assert "start thesis" in out["adaptations"]["current_goals"]


def test_apply_delta_mood_label_only_when_set():
    prev = copy.deepcopy(DEFAULT_USER_STATE)
    prev["state"]["mood_label"] = "calm"
    out = apply_delta(prev, validate_delta({}))   # no mood label in delta
    assert out["state"]["mood_label"] == "calm"
    out2 = apply_delta(prev, validate_delta({"mood_label": "tired"}))
    assert out2["state"]["mood_label"] == "tired"


def test_apply_delta_does_not_mutate_prev():
    prev = copy.deepcopy(DEFAULT_USER_STATE)
    snapshot = copy.deepcopy(prev)
    apply_delta(prev, validate_delta({"pad": {"pleasure": 0.1, "arousal": 0.0, "dominance": 0.0}}))
    assert prev == snapshot


def test_apply_delta_adds_tagged_value():
    prev = copy.deepcopy(DEFAULT_USER_STATE)
    delta = validate_delta({"value_add": [{"label": "family", "schwartz": "benevolence"}]})
    out = apply_delta(prev, delta)
    assert {"label": "family", "schwartz": "benevolence"} in out["adaptations"]["values"]


def test_apply_delta_value_add_promotes_null_tag_to_real_tag():
    """If the existing value has schwartz=None and the same label arrives with
    a real tag, the tag fills in."""
    prev = copy.deepcopy(DEFAULT_USER_STATE)
    prev["adaptations"]["values"] = [{"label": "honesty", "schwartz": None}]
    delta = validate_delta({"value_add": [{"label": "honesty", "schwartz": "universalism"}]})
    out = apply_delta(prev, delta)
    assert out["adaptations"]["values"] == [{"label": "honesty", "schwartz": "universalism"}]


def test_apply_delta_value_add_dedupes_by_label_case_insensitive():
    prev = copy.deepcopy(DEFAULT_USER_STATE)
    prev["adaptations"]["values"] = [{"label": "Family", "schwartz": "benevolence"}]
    delta = validate_delta({"value_add": [{"label": "family", "schwartz": "tradition"}]})
    out = apply_delta(prev, delta)
    # Existing tag is real; doesn't get overwritten
    assert len(out["adaptations"]["values"]) == 1
    assert out["adaptations"]["values"][0]["schwartz"] == "benevolence"


# ---------- is_cold_start ----------

def test_default_is_cold_start():
    assert is_cold_start(DEFAULT_USER_STATE)


def test_populated_is_not_cold_start():
    s = copy.deepcopy(DEFAULT_USER_STATE)
    s["state"]["pleasure"] = 0.1
    assert not is_cold_start(s)


def test_concerns_alone_break_cold_start():
    s = copy.deepcopy(DEFAULT_USER_STATE)
    s["adaptations"]["concerns"] = ["something"]
    assert not is_cold_start(s)


# ---------- format_user_state_block ----------

def test_format_cold_start_emits_first_read_message():
    out = format_user_state_block(None)
    assert "First read of this person" in out
    assert "<user_state>" in out
    assert "</user_state>" in out


def test_format_cold_start_default_state_collapses():
    out = format_user_state_block(DEFAULT_USER_STATE)
    assert "First read of this person" in out


def test_format_cold_start_still_surfaces_trait_baseline():
    """Even when PAD/adaptations are all-default, the configured Big 5
    baseline should be rendered so lemon reads it from turn one."""
    out = format_user_state_block(DEFAULT_USER_STATE)
    assert "Roughly:" in out
    assert "high openness" in out


def test_default_user_state_has_calibrated_trait_baseline():
    """DEFAULT_USER_STATE seeds calibrated Big 5 values for this user, not
    population zero. Traits still drift via trait_nudges (capped tiny)."""
    traits = DEFAULT_USER_STATE["traits"]
    assert traits["openness"] == 0.75
    assert traits["conscientiousness"] == 0.70
    assert traits["extraversion"] == 0.55
    assert traits["agreeableness"] == 0.62
    assert traits["neuroticism"] == -0.20


def test_format_includes_pad_numbers():
    s = copy.deepcopy(DEFAULT_USER_STATE)
    s["state"] = {"pleasure": -0.2, "arousal": 0.1, "dominance": -0.05, "mood_label": "tired"}
    out = format_user_state_block(s)
    assert "tired" in out
    assert "pleasure -0.20" in out
    assert "arousal +0.10" in out


def test_format_renders_adaptations():
    s = copy.deepcopy(DEFAULT_USER_STATE)
    s["state"]["mood_label"] = "calm"
    s["state"]["pleasure"] = 0.1
    s["adaptations"]["current_goals"] = ["prep tuesday exam"]
    s["adaptations"]["values"] = ["family"]
    s["adaptations"]["concerns"] = ["unprepared"]
    s["adaptations"]["relational_stance"] = "open, slightly tired"
    out = format_user_state_block(s)
    assert "On their mind: prep tuesday exam" in out
    assert "Cares about: family" in out
    assert "Worries: unprepared" in out
    assert "How they're showing up: open, slightly tired" in out


# ---------- load_user_state / save_user_state ----------

def test_load_returns_default_when_db_empty():
    out = load_user_state()
    assert out == DEFAULT_USER_STATE
    assert out is not DEFAULT_USER_STATE   # fresh copy


def test_save_load_roundtrip():
    s = copy.deepcopy(DEFAULT_USER_STATE)
    s["state"]["mood_label"] = "tired"
    s["state"]["pleasure"] = -0.2
    s["traits"]["agreeableness"] = 0.6
    s["adaptations"]["current_goals"] = ["prep exam"]
    s["adaptations"]["values"] = [{"label": "family", "schwartz": "benevolence"}]
    save_user_state(s)
    loaded = load_user_state()
    assert loaded["state"]["mood_label"] == "tired"
    assert loaded["state"]["pleasure"] == -0.2
    assert loaded["traits"]["agreeableness"] == 0.6
    assert loaded["adaptations"]["current_goals"] == ["prep exam"]
    assert loaded["adaptations"]["values"] == [{"label": "family", "schwartz": "benevolence"}]


def test_save_load_legacy_string_values_normalize_on_read():
    """A snapshot saved before Schwartz tagging shipped (raw string values)
    should normalize to the tagged shape when read back."""
    from storage import db
    legacy = copy.deepcopy(DEFAULT_USER_STATE)
    legacy["adaptations"]["values"] = ["honesty", "family"]   # legacy shape
    db.save_user_state_snapshot(legacy)
    loaded = load_user_state()
    assert loaded["adaptations"]["values"] == [
        {"label": "honesty", "schwartz": None},
        {"label": "family",  "schwartz": None},
    ]


def test_load_returns_latest_snapshot():
    save_user_state(copy.deepcopy(DEFAULT_USER_STATE))
    s = copy.deepcopy(DEFAULT_USER_STATE)
    s["state"]["mood_label"] = "happy"
    save_user_state(s)
    s2 = copy.deepcopy(DEFAULT_USER_STATE)
    s2["state"]["mood_label"] = "tired"
    save_user_state(s2)
    assert load_user_state()["state"]["mood_label"] == "tired"


def test_fresh_user_session_has_no_overrides():
    """The user side intentionally has no session-start reset — they bring
    whatever they were carrying. Asymmetric dynamics with the lemon side."""
    s = copy.deepcopy(DEFAULT_USER_STATE)
    s["state"]["mood_label"] = "low"
    s["adaptations"]["concerns"] = ["heavy week"]
    save_user_state(s)
    fresh = fresh_user_session_state()
    assert fresh["state"]["mood_label"] == "low"
    assert "heavy week" in fresh["adaptations"]["concerns"]


# ---------- MOOD_LABELS sanity ----------

def test_mood_labels_includes_neutral():
    assert "neutral" in MOOD_LABELS


def test_mood_labels_no_duplicates():
    assert len(MOOD_LABELS) == len(set(MOOD_LABELS))
