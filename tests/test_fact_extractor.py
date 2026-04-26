"""Tests for `empathy/fact_extractor.py`.

The standalone `extract_facts()` LLM call was retired when fact extraction
merged into `empathy/post_exchange.bookkeep` (which now runs in a daemon
thread post-reply). The bookkeep flow is exercised end-to-end by
`test_pipeline.py`. The prompt builder also moved out — it lives in
`prompts.build_bookkeep_prompt` now.

What's still in `empathy/fact_extractor.py`: the `_parse` / `_validate`
key-and-value hygiene + the dedup gate (`_reconcile_key`, `_strip_noise`)
that prevents the LLM from inventing key mutations like
`prajwal_sleep_status_v2`. Those are tested here.
"""
import json

import pytest

from empathy.fact_extractor import _parse, _reconcile_key, _strip_noise, _validate


# ---------- _parse ----------

def test_parse_plain_json():
    raw = json.dumps({"exam_date": "tuesday", "sister_name": "riya"})
    out = _parse(raw, max_new=3)
    assert out == {"exam_date": "tuesday", "sister_name": "riya"}


def test_parse_strips_json_fence():
    raw = "```json\n" + json.dumps({"city": "Bangalore"}) + "\n```"
    assert _parse(raw, max_new=3) == {"city": "Bangalore"}


def test_parse_empty_object_returns_empty_dict():
    assert _parse("{}", max_new=3) == {}


def test_parse_rejects_non_object():
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _parse("[1, 2, 3]", max_new=3)


def test_parse_drops_invalid_key_chars():
    raw = json.dumps({"Exam Date": "tuesday", "1st_exam": "wed", "ok_key": "yes"})
    # "Exam Date" has a space; "1st_exam" starts with a digit.
    assert _parse(raw, max_new=3) == {"ok_key": "yes"}


def test_parse_lowercases_uppercase_key():
    raw = json.dumps({"CITY": "Bangalore"})
    # Validator lowercases keys before applying the regex check.
    assert _parse(raw, max_new=3) == {"city": "Bangalore"}


def test_parse_drops_empty_value():
    raw = json.dumps({"name": "", "city": "  ", "ok": "yes"})
    assert _parse(raw, max_new=3) == {"ok": "yes"}


def test_parse_drops_none_value():
    raw = json.dumps({"name": None, "ok": "yes"})
    assert _parse(raw, max_new=3) == {"ok": "yes"}


def test_parse_truncates_long_value():
    long_value = "x" * 400
    out = _parse(json.dumps({"thing": long_value}), max_new=3)
    assert len(out["thing"]) == 200


def test_parse_caps_at_max_new():
    raw = json.dumps({"a": "1", "b": "2", "c": "3", "d": "4", "e": "5"})
    assert len(_parse(raw, max_new=3)) == 3


def test_parse_coerces_numeric_value_to_string():
    raw = '{"age": 25}'
    assert _parse(raw, max_new=3) == {"age": "25"}


# ---------- _validate (when existing_keys is provided, dedup kicks in) ----------

def test_validate_with_no_existing_keys_passes_through():
    out = _validate({"city": "Bangalore"}, max_new=3, existing_keys=())
    assert out == {"city": "Bangalore"}


def test_validate_dedups_against_existing_keys():
    """If the proposed key is a noisy variant of an existing one, reconcile
    it back to the canonical key so the upsert lands on the same row."""
    existing = {"sleep_status"}
    out = _validate({"current_sleep_status": "exhausted"}, max_new=3, existing_keys=existing)
    # The "current_" filler should be stripped, mapping back to "sleep_status".
    assert out == {"sleep_status": "exhausted"}


# ---------- _strip_noise ----------

def test_strip_noise_removes_modifier_suffixes():
    assert _strip_noise("sleep_status_final") == "sleep_status"
    assert _strip_noise("sleep_status_v2") == "sleep_status"
    assert _strip_noise("sleep_status_updated") == "sleep_status"
    assert _strip_noise("sleep_status_clarified") == "sleep_status"


def test_strip_noise_removes_filler_prefixes():
    assert _strip_noise("current_sleep_status") == "sleep_status"
    assert _strip_noise("latest_mood") == "mood"
    assert _strip_noise("recent_activity_status") == "activity_status"


def test_strip_noise_removes_filler_infix():
    # 'current' between two semantic tokens should also vanish
    assert _strip_noise("prajwal_current_sleep_status") == "prajwal_sleep_status"


def test_strip_noise_leaves_clean_keys_alone():
    assert _strip_noise("exam_date") == "exam_date"
    assert _strip_noise("sister_name") == "sister_name"
    assert _strip_noise("city") == "city"


# ---------- _reconcile_key ----------

def test_reconcile_returns_existing_key_when_noisy_variant():
    existing = ["sleep_status", "exam_date"]
    assert _reconcile_key("current_sleep_status", existing) == "sleep_status"
    assert _reconcile_key("sleep_status_final", existing) == "sleep_status"


def test_reconcile_returns_none_when_no_match():
    existing = ["city", "exam_date"]
    assert _reconcile_key("favorite_color", existing) is None


def test_reconcile_returns_canonical_key_for_exact_match():
    """When the new key already matches an existing one exactly, the
    reconciler returns that same key — meaning the caller upserts in place
    rather than creating a near-duplicate."""
    existing = ["sleep_status"]
    assert _reconcile_key("sleep_status", existing) == "sleep_status"
