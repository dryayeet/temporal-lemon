"""Tests for `empathy/tom.py` + the legacy `format_tom_block` formatter.

The standalone `theory_of_mind()` LLM call was retired when emotion + ToM
merged into the single `empathy/user_read.read_user` round-trip; that flow
is covered by mocks in `test_pipeline.py`. The unified pre-reply prompt
block is now `<reading>` (combined emotion + ToM); see `test_emotion.py`
for that. `format_tom_block` is kept as a legacy formatter in `prompts`
so existing imports / older tooling don't break.

What's still in `empathy/tom.py`: `DEFAULT_TOM`, `_validate`, `_parse`.
"""
import json

import pytest

from empathy import tom
from empathy.tom import DEFAULT_TOM
from prompts import format_tom_block


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


def test_default_has_three_nullable_fields():
    assert set(DEFAULT_TOM.keys()) == {"feeling", "avoid", "what_helps"}
    assert all(v is None for v in DEFAULT_TOM.values())


# ---------- format_tom_block (legacy formatter, kept in prompts/) ----------

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
