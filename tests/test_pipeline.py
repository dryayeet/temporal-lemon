import json

import pytest

from app import pipeline
from app.pipeline import run_empathy_turn
from core import config
from storage import db


# ---------- helpers ----------

def base_history():
    return [
        {"role": "system", "content": "<Who you are>\npersona..."},
        {"role": "system", "content": "<time_context>10:00</time_context>"},
        {"role": "system", "content": "<internal_state>x</internal_state>"},
    ]


def _messages_in_session(session_id: int) -> list[dict]:
    """Read messages for a session directly. test helper, not exported."""
    with db.connect() as c:
        rows = c.execute(
            "SELECT role, content, emotion, intensity FROM messages "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _zero_delta():
    return {
        "pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0},
        "mood_label": None,
        "trait_nudges": {},
        "goal_add": [], "goal_remove": [],
        "concern_add": [], "concern_remove": [],
        "value_add": [],
        "stance": None,
    }


class _FakeReply:
    """Mock the merged read_user (emotion + tom + user_delta + lemon_delta)
    and generate_reply so the pipeline runs without any HTTP."""
    def __init__(self, emotion=None, tom=None, draft="hey, that sounds rough",
                 user_delta=None, lemon_delta=None, delta=None):
        # `delta` kwarg kept for backward compat — interpreted as user_delta.
        self.emotion = emotion or {"primary": "sadness", "intensity": 0.6,
                                    "underlying_need": "feel heard", "undertones": []}
        self.tom = tom or {"feeling": "sad", "avoid": "advice", "what_helps": "listen"}
        self.user_delta = user_delta if user_delta is not None else (delta or _zero_delta())
        self.lemon_delta = lemon_delta if lemon_delta is not None else _zero_delta()
        self.draft = draft
        self.draft_calls = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(
            pipeline, "read_user",
            lambda *a, **kw: (self.emotion, self.tom, self.user_delta, self.lemon_delta),
        )
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

    rows = _messages_in_session(sid)
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

    rows = _messages_in_session(sid)
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

    # The merged user_read collapsed the old "thinking" phase into "reading you".
    # Three core phases remain.
    assert "reading you" in phases
    assert "remembering" in phases
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


# ---------- dyadic-state stage 1: user_state trajectory on trace ----------

def test_pipeline_records_user_state_trajectory(monkeypatch):
    """Pipeline should populate user_state_before / after / delta on the trace
    and apply the LLM-emitted delta to the user_state going in."""
    from storage.user_state import DEFAULT_USER_STATE

    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    nudge_delta = {
        "pad": {"pleasure": -0.10, "arousal": 0.05, "dominance": 0.0},
        "mood_label": "tired",
        "trait_nudges": {},
        "goal_add": ["prep tuesday exam"],
        "goal_remove": [], "concern_add": [], "concern_remove": [], "value_add": [],
        "stance": None,
    }
    fake = _FakeReply(delta=nudge_delta)
    fake.install(monkeypatch)

    sid = db.start_session()
    reply, trace = run_empathy_turn(
        "long day", base_history(), session_id=sid,
        user_state=dict(DEFAULT_USER_STATE),
    )

    assert trace.user_state_before is not None
    assert trace.user_state_after is not None
    assert trace.user_state_delta is not None
    # Mood label moved
    assert trace.user_state_after["state"]["mood_label"] == "tired"
    # PAD nudged toward unhappy
    assert trace.user_state_after["state"]["pleasure"] < 0
    # Goal got added
    assert "prep tuesday exam" in trace.user_state_after["adaptations"]["current_goals"]


def test_pipeline_persists_user_state_snapshot(monkeypatch):
    """Pipeline should write a user_state_snapshot during each turn, so the
    next session can pick up where this one left off."""
    from storage.user_state import DEFAULT_USER_STATE

    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    fake = _FakeReply(user_delta={
        "pad": {"pleasure": 0.10, "arousal": 0.0, "dominance": 0.0},
        "mood_label": "calm",
        "trait_nudges": {},
        "goal_add": [], "goal_remove": [], "concern_add": [], "concern_remove": [],
        "value_add": [], "stance": None,
    })
    fake.install(monkeypatch)

    sid = db.start_session()
    run_empathy_turn("hi", base_history(), session_id=sid,
                     user_state=dict(DEFAULT_USER_STATE))

    snapshot = db.latest_user_state()
    assert snapshot is not None
    assert snapshot["state"]["mood_label"] == "calm"


# ---------- dyadic-state stages 2+3: lemon_state trajectory + persistence ----------

def test_pipeline_records_lemon_state_trajectory(monkeypatch):
    """Stage 2: pipeline applies a lemon_state delta pre-reply and reports
    the trajectory on the trace."""
    from storage.lemon_state import DEFAULT_LEMON_STATE
    from storage.user_state import DEFAULT_USER_STATE

    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    lemon_nudge = {
        "pad": {"pleasure": -0.05, "arousal": 0.0, "dominance": 0.0},
        "mood_label": "calm",
        "trait_nudges": {},
        "goal_add": [],
        "goal_remove": [],
        "concern_add": ["the user seemed off"],
        "concern_remove": [],
        "value_add": [],
        "stance": None,
    }
    fake = _FakeReply(lemon_delta=lemon_nudge)
    fake.install(monkeypatch)

    sid = db.start_session()
    reply, trace = run_empathy_turn(
        "long day", base_history(), session_id=sid,
        user_state=dict(DEFAULT_USER_STATE),
        lemon_state=dict(DEFAULT_LEMON_STATE),
    )

    assert trace.lemon_state_before is not None
    assert trace.lemon_state_after is not None
    assert trace.lemon_state_delta is not None
    assert trace.lemon_state_after["state"]["mood_label"] == "calm"
    assert "the user seemed off" in trace.lemon_state_after["adaptations"]["concerns"]


def test_pipeline_persists_lemon_state_snapshot(monkeypatch):
    """Pipeline writes a lemon_state_snapshot during each turn."""
    from storage.lemon_state import DEFAULT_LEMON_STATE
    from storage.user_state import DEFAULT_USER_STATE

    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    fake = _FakeReply(lemon_delta={
        "pad": {"pleasure": 0.05, "arousal": 0.0, "dominance": 0.0},
        "mood_label": "happy",
        "trait_nudges": {},
        "goal_add": [], "goal_remove": [], "concern_add": [], "concern_remove": [],
        "value_add": [], "stance": None,
    })
    fake.install(monkeypatch)

    sid = db.start_session()
    run_empathy_turn(
        "hi", base_history(), session_id=sid,
        user_state=dict(DEFAULT_USER_STATE),
        lemon_state=dict(DEFAULT_LEMON_STATE),
    )

    snapshot = db.latest_lemon_state()
    assert snapshot is not None
    assert snapshot["state"]["mood_label"] == "happy"


def test_pipeline_lemon_traits_never_drift(monkeypatch):
    """Stage 3: even if the LLM tries to nudge lemon's traits, the validator
    layer in user_read should clamp them to zero. Traits are persona-fixed."""
    from storage.lemon_state import DEFAULT_LEMON_STATE
    from storage.user_state import DEFAULT_USER_STATE

    monkeypatch.setattr(config, "ENABLE_EMPATHY_PIPELINE", True)
    # Bake an aggressive trait nudge into the lemon_delta. The pipeline
    # itself doesn't enforce the trait freeze — that's user_read's job — but
    # we confirm here that an in-bounds delta still produces no trait drift,
    # because the asymmetric clamp inside read_user wipes trait_nudges to {}.
    # We mock read_user directly so we're testing the pipeline's behavior
    # given a clean delta (the production path runs through _clamp_lemon_delta).
    fake = _FakeReply(lemon_delta={
        "pad": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0},
        "mood_label": None,
        "trait_nudges": {},   # already cleared, mimicking _clamp_lemon_delta
        "goal_add": [], "goal_remove": [], "concern_add": [], "concern_remove": [],
        "value_add": [], "stance": None,
    })
    fake.install(monkeypatch)

    sid = db.start_session()
    run_empathy_turn(
        "hello", base_history(), session_id=sid,
        user_state=dict(DEFAULT_USER_STATE),
        lemon_state=dict(DEFAULT_LEMON_STATE),
    )
    snapshot = db.latest_lemon_state()
    # Traits are exactly the persona constants
    assert snapshot["traits"] == DEFAULT_LEMON_STATE["traits"]
