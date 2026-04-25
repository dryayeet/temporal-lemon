import json

import pytest
import requests

import config
import pipeline
from empathy import fact_extractor
from empathy.fact_extractor import _build_prompt, _parse, extract_facts
from storage import db


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
    # "Exam Date" has a space (invalid), "1st_exam" starts with a digit (invalid)
    assert _parse(raw, max_new=3) == {"ok_key": "yes"}


def test_parse_drops_uppercase_key():
    raw = json.dumps({"CITY": "Bangalore"})
    # uppercase not allowed; we lowercase before validating, so this actually passes
    # but we also want to confirm the normalized key form ends up lowercase
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


# ---------- _build_prompt ----------

def test_prompt_includes_all_inputs():
    prompt = _build_prompt(
        user_msg="my exam is on tuesday",
        bot_reply="oh that's soon, you prepping?",
        existing_facts={"city": "Bangalore"},
        recent_msgs=[
            {"role": "user", "content": "earlier stuff"},
            {"role": "assistant", "content": "earlier reply"},
        ],
        max_new=3,
    )
    assert "my exam is on tuesday" in prompt
    assert "oh that's soon" in prompt
    assert "city: Bangalore" in prompt
    assert "earlier stuff" in prompt
    assert "earlier reply" in prompt
    assert "at most 3" in prompt


def test_prompt_handles_empty_existing_facts():
    prompt = _build_prompt("hi", "hi back", {}, None, max_new=3)
    assert "(none yet)" in prompt
    assert "(no prior turns)" in prompt


# ---------- extract_facts (HTTP mocked) ----------

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


def test_extract_facts_happy_path(monkeypatch):
    fake_content = json.dumps({"exam_date": "tuesday"})
    fake_payload = {"choices": [{"message": {"content": fake_content}}]}
    monkeypatch.setattr(fact_extractor.requests, "post",
                        lambda *a, **kw: _FakeResponse(fake_payload))

    out = extract_facts("my exam is on tuesday", "got it, good luck")
    assert out == {"exam_date": "tuesday"}


def test_extract_facts_empty_on_http_error(monkeypatch):
    monkeypatch.setattr(fact_extractor.requests, "post",
                        lambda *a, **kw: _FakeResponse({"error": "boom"}, status=500))
    assert extract_facts("x", "y") == {}


def test_extract_facts_empty_on_bad_json(monkeypatch):
    fake = {"choices": [{"message": {"content": "not json"}}]}
    monkeypatch.setattr(fact_extractor.requests, "post",
                        lambda *a, **kw: _FakeResponse(fake))
    assert extract_facts("x", "y") == {}


def test_extract_facts_empty_on_connection_error(monkeypatch):
    def raising(*a, **kw):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(fact_extractor.requests, "post", raising)
    assert extract_facts("x", "y") == {}


# ---------- pipeline integration ----------

def _base_history():
    return [
        {"role": "system", "content": "<Who you are>\npersona"},
        {"role": "system", "content": "<time_context>10:00</time_context>"},
        {"role": "system", "content": "<internal_state>x</internal_state>"},
    ]


def _install_pipeline_mocks(monkeypatch, draft="hey, that sounds fun"):
    monkeypatch.setattr(pipeline, "classify_emotion",
                        lambda *a, **kw: {"primary": "joy", "intensity": 0.3,
                                          "underlying_need": None, "undertones": []})
    monkeypatch.setattr(pipeline, "theory_of_mind",
                        lambda *a, **kw: {"feeling": "up", "avoid": None, "what_helps": None})
    monkeypatch.setattr(pipeline, "generate_reply", lambda *a, **kw: draft)


def test_pipeline_persists_extracted_facts(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    monkeypatch.setattr(config, "ENABLE_AUTO_FACTS", True)
    _install_pipeline_mocks(monkeypatch)
    monkeypatch.setattr(fact_extractor, "extract_facts",
                        lambda *a, **kw: {"exam_date": "tuesday"})

    sid = db.start_session()
    reply, trace = pipeline.run_empathy_turn(
        "my exam is on tuesday", _base_history(), session_id=sid
    )

    assert trace.facts_extracted == {"exam_date": "tuesday"}
    assert db.get_facts() == {"exam_date": "tuesday"}


def test_pipeline_survives_extractor_exception(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    monkeypatch.setattr(config, "ENABLE_AUTO_FACTS", True)
    _install_pipeline_mocks(monkeypatch, draft="sure thing")

    def boom(*a, **kw):
        raise RuntimeError("extractor crashed")
    monkeypatch.setattr(fact_extractor, "extract_facts", boom)

    sid = db.start_session()
    reply, trace = pipeline.run_empathy_turn("x", _base_history(), session_id=sid)

    assert reply == "sure thing"
    assert trace.facts_extracted == {}
    assert db.get_facts() == {}


def test_pipeline_skips_extractor_when_auto_facts_disabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    monkeypatch.setattr(config, "ENABLE_AUTO_FACTS", False)
    _install_pipeline_mocks(monkeypatch)

    called = {"n": 0}
    def tracking(*a, **kw):
        called["n"] += 1
        return {"should_not": "appear"}
    monkeypatch.setattr(fact_extractor, "extract_facts", tracking)

    sid = db.start_session()
    reply, trace = pipeline.run_empathy_turn("hi", _base_history(), session_id=sid)

    assert called["n"] == 0
    assert trace.facts_extracted == {}
    assert db.get_facts() == {}


def test_pipeline_emits_noting_phase(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    monkeypatch.setattr(config, "ENABLE_AUTO_FACTS", True)
    _install_pipeline_mocks(monkeypatch)
    monkeypatch.setattr(fact_extractor, "extract_facts",
                        lambda *a, **kw: {"city": "Bangalore"})

    phases = []
    sid = db.start_session()
    pipeline.run_empathy_turn("I'm in Bangalore", _base_history(),
                              session_id=sid, on_phase=phases.append)

    assert "making a note" in phases


def test_pipeline_caps_facts_at_max_per_turn(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    monkeypatch.setattr(config, "ENABLE_AUTO_FACTS", True)
    monkeypatch.setattr(config, "AUTO_FACTS_MAX_PER_TURN", 2)
    _install_pipeline_mocks(monkeypatch)
    # extractor returns 4 facts; pipeline should only upsert the first 2
    monkeypatch.setattr(fact_extractor, "extract_facts",
                        lambda *a, **kw: {"a": "1", "b": "2", "c": "3", "d": "4"})

    sid = db.start_session()
    pipeline.run_empathy_turn("info dump", _base_history(), session_id=sid)

    saved = db.get_facts()
    assert len(saved) == 2
