import config
from chat import humanize_delay, prepare_messages


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
        {"role": "system", "content": "<internal_state>x</internal_state>"},
    ]
    out = prepare_messages(history)
    # neither has the persona tag → both stay as plain strings
    for msg in out:
        assert isinstance(msg["content"], str)


# ---------- humanize_delay ----------

def test_humanize_returns_zero_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "HUMANIZE_STREAM", False)
    assert humanize_delay("hi", "medium") == 0.0


def test_humanize_low_energy_is_slower_than_high(monkeypatch):
    monkeypatch.setattr(config, "HUMANIZE_STREAM", True)
    # average over many samples to wash out jitter
    n = 200
    low = sum(humanize_delay("a", "low") for _ in range(n)) / n
    high = sum(humanize_delay("a", "high") for _ in range(n)) / n
    assert low > high


def test_humanize_punctuation_adds_extra_pause(monkeypatch):
    monkeypatch.setattr(config, "HUMANIZE_STREAM", True)
    n = 200
    plain = sum(humanize_delay("a", "medium") for _ in range(n)) / n
    after_period = sum(humanize_delay("a.", "medium") for _ in range(n)) / n
    assert after_period > plain
