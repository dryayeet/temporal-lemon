import json

import pytest

import config
import db
import pipeline
from pipeline import run_empathy_turn


# ---------- helpers ----------

def base_history():
    return [
        {"role": "system", "content": "<Who you are>\npersona..."},
        {"role": "system", "content": "<time_context>10:00</time_context>"},
        {"role": "system", "content": "<internal_state>x</internal_state>"},
    ]


class _FakeReply:
    """Mock for both classify_emotion and theory_of_mind so the pipeline runs without HTTP."""
    def __init__(self, emotion=None, tom=None, draft="hey, that sounds rough"):
        self.emotion = emotion or {"primary": "sadness", "intensity": 0.6,
                                    "underlying_need": "feel heard", "undertones": []}
        self.tom = tom or {"feeling": "sad", "avoid": "advice", "what_helps": "listen"}
        self.draft = draft
        self.draft_calls = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(pipeline, "classify_emotion",
                            lambda *a, **kw: self.emotion)
        monkeypatch.setattr(pipeline, "theory_of_mind",
                            lambda *a, **kw: self.tom)
        def gen(*a, **kw):
            self.draft_calls += 1
            return self.draft
        monkeypatch.setattr(pipeline, "generate_reply", gen)


# ---------- short-circuit when disabled ----------

def test_pipeline_off_skips_aux_calls(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", False)
    fake = _FakeReply(draft="straight reply")
    fake.install(monkeypatch)

    sid = db.start_session()
    reply, trace = run_empathy_turn("hi", base_history(), session_id=sid)
    assert reply == "straight reply"
    assert trace.pipeline_used is False
    assert trace.emotion is None
    assert fake.draft_calls == 1


# ---------- full pipeline runs all steps ----------

def test_pipeline_on_runs_all_steps(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    monkeypatch.setattr(config, "EMPATHY_RETRY_ON_FAIL", True)
    fake = _FakeReply(draft="hey, that sounds rough. want to talk about it?")
    fake.install(monkeypatch)

    sid = db.start_session()
    reply, trace = run_empathy_turn("I had a rough day", base_history(), session_id=sid)

    assert trace.pipeline_used is True
    assert trace.emotion["primary"] == "sadness"
    assert trace.tom["feeling"] == "sad"
    assert trace.draft == reply
    assert trace.check is not None
    assert trace.check.passed is True
    assert trace.regenerated is False


# ---------- regenerates on failed check ----------

def test_pipeline_regenerates_on_check_failure(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    monkeypatch.setattr(config, "EMPATHY_RETRY_ON_FAIL", True)

    drafts = iter([
        "at least you'll be fine",          # bad — minimizing
        "yeah, that's a lot. with you.",    # good
    ])
    fake = _FakeReply()
    fake.install(monkeypatch)
    monkeypatch.setattr(pipeline, "generate_reply", lambda *a, **kw: next(drafts))

    sid = db.start_session()
    reply, trace = run_empathy_turn("my dog died", base_history(), session_id=sid)

    assert trace.regenerated is True
    assert "at least" not in reply
    assert reply == "yeah, that's a lot. with you."


def test_pipeline_does_not_regenerate_when_retry_disabled(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    monkeypatch.setattr(config, "EMPATHY_RETRY_ON_FAIL", False)
    fake = _FakeReply(draft="at least you can move on")
    fake.install(monkeypatch)

    sid = db.start_session()
    reply, trace = run_empathy_turn("rough", base_history(), session_id=sid)

    assert trace.check is not None and not trace.check.passed
    assert trace.regenerated is False
    assert reply == "at least you can move on"


# ---------- logging side effects ----------

def test_pipeline_logs_user_msg_with_emotion(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    fake = _FakeReply(emotion={"primary": "anger", "intensity": 0.7,
                                "underlying_need": None, "undertones": []})
    fake.install(monkeypatch)

    sid = db.start_session()
    run_empathy_turn("furious", base_history(), session_id=sid)

    rows = db.session_messages(sid)
    user_rows = [r for r in rows if r["role"] == "user"]
    assert len(user_rows) == 1
    assert user_rows[0]["emotion"] == "anger"
    assert user_rows[0]["intensity"] == 0.7


def test_pipeline_logs_assistant_reply(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    fake = _FakeReply(draft="hey there")
    fake.install(monkeypatch)

    sid = db.start_session()
    run_empathy_turn("hi", base_history(), session_id=sid)

    rows = db.session_messages(sid)
    assistant_rows = [r for r in rows if r["role"] == "assistant"]
    assert len(assistant_rows) == 1
    assert assistant_rows[0]["content"] == "hey there"


# ---------- on_phase callback ----------

def test_pipeline_emits_phase_events(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    fake = _FakeReply()
    fake.install(monkeypatch)

    phases = []
    sid = db.start_session()
    run_empathy_turn("hi", base_history(), session_id=sid, on_phase=phases.append)

    # at least the four core phases should fire
    assert "reading you" in phases
    assert "remembering" in phases
    assert "thinking" in phases
    assert "replying" in phases


# ---------- doesn't mutate base_history ----------

def test_pipeline_does_not_mutate_base_history(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    fake = _FakeReply()
    fake.install(monkeypatch)

    sid = db.start_session()
    base = base_history()
    before = json.dumps(base)
    run_empathy_turn("hi", base, session_id=sid)
    assert json.dumps(base) == before
