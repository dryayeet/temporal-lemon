"""Tests for `empathy/emotion.py` + the related reading-block formatter.

`classify_emotion` was retired when emotion + ToM merged into the single
`empathy/user_read.read_user` round-trip; that flow is covered by mocks in
`test_pipeline.py`. `format_emotion_block` was folded into
`format_reading_block` (the unified phasic block) in `prompts/__init__.py`.

What's still in `empathy/emotion.py`: the schema validator (`_validate`),
the JSON parser (`_parse`), and the family map for mood-congruent retrieval
(`family_of`, `emotion_relatedness`). Those are tested here.
"""
import json

import pytest

from empathy import emotion
from empathy.emotion import EMOTION_FAMILIES, emotion_relatedness, family_of
from prompts import EMOTION_LABELS, format_reading_block


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


# ---------- family map ----------

def test_every_label_has_a_family():
    assert set(EMOTION_LABELS) == set(EMOTION_FAMILIES)


def test_pride_in_self_conscious_cluster():
    """Tracy & Robins: pride shares the self-representation substrate
    with shame/guilt/embarrassment despite the opposite valence."""
    assert family_of("pride") == "self_conscious"
    assert family_of("shame") == "self_conscious"


def test_relief_in_positive_cluster():
    assert family_of("relief") == "positive"


def test_relatedness_same_label_is_one_except_neutral():
    assert emotion_relatedness("sadness", "sadness") == 1.0
    # Neutral exact-match doesn't count — every turn is mostly neutral.
    assert emotion_relatedness("neutral", "neutral") == 0.0


def test_relatedness_same_family_is_half():
    assert emotion_relatedness("sadness", "loneliness") == 0.5
    assert emotion_relatedness("anger", "frustration") == 0.5
    assert emotion_relatedness("pride", "shame") == 0.5    # self-conscious cluster


def test_relatedness_different_family_is_zero():
    assert emotion_relatedness("joy", "anger") == 0.0
    assert emotion_relatedness("tired", "sadness") == 0.0  # low_arousal vs sad


def test_relatedness_neutral_never_scores():
    assert emotion_relatedness("neutral", "joy") == 0.0
    assert emotion_relatedness("sadness", "neutral") == 0.0


# ---------- format_reading_block (unified phasic block) ----------

def test_reading_block_includes_emotion_and_tom():
    out = format_reading_block(
        emotion={"primary": "sadness", "intensity": 0.7,
                 "underlying_need": "be heard",
                 "undertones": ["loneliness"]},
        tom={"feeling": "tired and let down",
             "avoid": "don't jump to advice",
             "what_helps": "stay with it"},
    )
    assert "<reading>" in out
    assert "sadness" in out
    assert "loneliness" in out
    assert "be heard" in out
    assert "tired and let down" in out
    assert "advice" in out
    assert "stay with it" in out


def test_reading_block_intensity_word_buckets():
    def block_for(intensity):
        return format_reading_block(
            {"primary": "joy", "intensity": intensity, "undertones": []},
            {"feeling": None, "avoid": None, "what_helps": None},
        )
    assert "mild" in block_for(0.1)
    assert "moderate" in block_for(0.4)
    assert "strong" in block_for(0.7)
    assert "very strong" in block_for(0.95)


def test_reading_block_handles_null_tom_fields():
    out = format_reading_block(
        {"primary": "neutral", "intensity": 0.1, "undertones": []},
        {"feeling": None, "avoid": None, "what_helps": None},
    )
    # Doesn't crash; renders 'unclear' / '(no specific guidance)' fallbacks
    assert "neutral" in out
    assert "unclear" in out
