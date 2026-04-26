"""Tests for `llm/chat.py`.

`humanize_delay` and `HUMANIZE_STREAM` were removed when streaming pacing was
dropped from the chat call (the web UI now buffers a full reply for the
empathy post-check before delivering it as a single SSE token event). The
prepare_messages tests still apply — Anthropic-style cache_control wrapping
on the persona block is unchanged.
"""
from core import config
from llm.chat import prepare_messages


# ---------- prepare_messages ----------

def test_prepare_wraps_persona_block_when_caching_enabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_PROMPT_CACHE", True)
    history = [
        {"role": "system", "content": "<Who you are>\nfull persona prompt..."},
        {"role": "system", "content": "<time_context>10:15</time_context>"},
        {"role": "user", "content": "hi"},
    ]
    out = prepare_messages(history)

    persona = out[0]
    assert isinstance(persona["content"], list)
    assert persona["content"][0]["type"] == "text"
    assert persona["content"][0]["cache_control"] == {"type": "ephemeral"}

    # other messages stay plain strings
    assert isinstance(out[1]["content"], str)
    assert isinstance(out[2]["content"], str)


def test_prepare_passthrough_when_caching_disabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_PROMPT_CACHE", False)
    history = [
        {"role": "system", "content": "<Who you are>\nx"},
        {"role": "user", "content": "hi"},
    ]
    out = prepare_messages(history)
    assert out == history


def test_prepare_only_wraps_persona_not_other_system_blocks(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_PROMPT_CACHE", True)
    history = [
        {"role": "system", "content": "<time_context>x</time_context>"},
        {"role": "system", "content": "<lemon_state>x</lemon_state>"},
    ]
    out = prepare_messages(history)
    # neither has the persona tag → both stay as plain strings
    for msg in out:
        assert isinstance(msg["content"], str)
