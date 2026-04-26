"""Tests for the Schwartz universal-values vocabulary + helpers."""
from prompts.schwartz import (
    SCHWARTZ_DESCRIPTIONS,
    SCHWARTZ_VALUES,
    coerce_schwartz,
    is_schwartz_value,
    normalize_value_entry,
)


# ---------- vocabulary ----------

def test_ten_values_present():
    assert len(SCHWARTZ_VALUES) == 10


def test_no_duplicates():
    assert len(SCHWARTZ_VALUES) == len(set(SCHWARTZ_VALUES))


def test_every_value_has_description():
    assert set(SCHWARTZ_VALUES) == set(SCHWARTZ_DESCRIPTIONS.keys())


# ---------- is_schwartz_value / coerce_schwartz ----------

def test_canonical_values_recognized():
    for v in SCHWARTZ_VALUES:
        assert is_schwartz_value(v)
        assert coerce_schwartz(v) == v


def test_aliases_coerced():
    assert coerce_schwartz("self-direction") == "self_direction"
    assert coerce_schwartz("Self Direction") == "self_direction"
    assert coerce_schwartz("selfdirection") == "self_direction"
    assert coerce_schwartz("BENEVOLENCE") == "benevolence"


def test_higher_order_axes_coerced_to_nearest():
    # Self-Transcendence and Self-Enhancement aren't in the 10 — coerce to nearest
    assert coerce_schwartz("self_transcendence") == "universalism"
    assert coerce_schwartz("self_enhancement") == "achievement"
    assert coerce_schwartz("openness") == "self_direction"
    assert coerce_schwartz("conservation") == "security"


def test_garbage_returns_none():
    assert coerce_schwartz("not a real category") is None
    assert coerce_schwartz(None) is None
    assert coerce_schwartz(42) is None
    assert coerce_schwartz("") is None
    assert not is_schwartz_value("not a real category")


# ---------- normalize_value_entry ----------

def test_string_input_becomes_untagged_dict():
    out = normalize_value_entry("honesty")
    assert out == {"label": "honesty", "schwartz": None}


def test_dict_input_passes_through_with_canonical_tag():
    out = normalize_value_entry({"label": "family", "schwartz": "benevolence"})
    assert out == {"label": "family", "schwartz": "benevolence"}


def test_dict_with_alias_tag_canonicalizes():
    out = normalize_value_entry({"label": "creativity", "schwartz": "Self-Direction"})
    assert out == {"label": "creativity", "schwartz": "self_direction"}


def test_dict_with_unknown_tag_drops_to_null():
    out = normalize_value_entry({"label": "fairness", "schwartz": "garbage"})
    assert out == {"label": "fairness", "schwartz": None}


def test_empty_string_returns_none():
    assert normalize_value_entry("") is None
    assert normalize_value_entry("   ") is None


def test_non_string_label_returns_none():
    assert normalize_value_entry({"label": 42, "schwartz": "achievement"}) is None
    assert normalize_value_entry({"schwartz": "achievement"}) is None


def test_label_truncated_to_max_len():
    long = "x" * 200
    out = normalize_value_entry(long, max_len=50)
    assert out is not None
    assert len(out["label"]) == 50


def test_other_types_return_none():
    assert normalize_value_entry(None) is None
    assert normalize_value_entry(42) is None
    assert normalize_value_entry(["honesty"]) is None
