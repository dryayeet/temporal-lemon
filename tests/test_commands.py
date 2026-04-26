import copy

from commands import ChatContext, dispatch, is_command
from storage import db
from storage.lemon_state import DEFAULT_LEMON_STATE


def make_ctx(**overrides):
    # use a real session so foreign-key constraints in db are satisfied
    sid = db.start_session()
    lemon_state = copy.deepcopy(DEFAULT_LEMON_STATE)
    lemon_state["state"]["mood_label"] = "happy"
    base = ChatContext(
        history=[
            {"role": "system", "content": "<persona>"},
            {"role": "assistant", "content": "hey"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "what's up"},
        ],
        lemon_state=lemon_state,
        chat_model="test/model",
        session_id=sid,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


# ---------- is_command ----------

def test_is_command_recognizes_slash():
    assert is_command("/help")
    assert is_command("/state foo")


def test_is_command_rejects_plain_text():
    assert not is_command("help")
    assert not is_command("/")
    assert not is_command("")


# ---------- /help ----------

def test_help_lists_known_commands():
    out = dispatch("/help", make_ctx()).output
    assert "/help" in out
    assert "/state" in out
    assert "/facts" in out
    assert "/reset" in out


def test_unknown_command_hints_help():
    out = dispatch("/nonsense", make_ctx()).output
    assert "unknown" in out.lower()
    assert "/help" in out


# ---------- /state ----------

def test_state_renders_lemon_state_block():
    ctx = make_ctx()
    out = dispatch("/state", ctx).output
    assert "lemon_state" in out
    assert "mood: happy" in out  # mood_label was set in make_ctx
    assert "traits" in out
    assert "PAD" in out


# ---------- /reset ----------

def test_reset_returns_state_to_defaults():
    ctx = make_ctx()
    res = dispatch("/reset", ctx)
    assert ctx.lemon_state == DEFAULT_LEMON_STATE
    assert res.reload_state is True


# ---------- /facts /remember /forget ----------

def test_facts_reports_empty_when_no_facts():
    out = dispatch("/facts", make_ctx()).output
    assert "no facts" in out


def test_remember_then_facts_then_forget():
    ctx = make_ctx()
    out = dispatch("/remember city=Bangalore", ctx).output
    assert "remembered" in out

    listing = dispatch("/facts", ctx).output
    assert "city" in listing and "Bangalore" in listing

    dispatch("/forget city", ctx)
    assert db.get_facts() == {}


def test_remember_rejects_bad_format():
    out = dispatch("/remember just-a-key", make_ctx()).output
    assert "usage" in out.lower()


def test_forget_reports_missing_key():
    out = dispatch("/forget never-stored", make_ctx()).output
    assert "no fact" in out


# ---------- /history /rewind ----------

def test_history_prints_recent_exchanges():
    out = dispatch("/history", make_ctx()).output
    assert "what's up" in out
    assert "<persona>" not in out  # system msgs hidden


def test_rewind_drops_two_non_system_messages():
    ctx = make_ctx()
    before = len(ctx.history)
    dispatch("/rewind", ctx)
    assert len(ctx.history) == before - 2
    # the persona system msg must remain
    assert any(m["role"] == "system" for m in ctx.history)


# ---------- /model ----------

def test_model_with_no_arg_reports_current():
    out = dispatch("/model", make_ctx()).output
    assert "test/model" in out


def test_model_switches_for_session():
    ctx = make_ctx()
    dispatch("/model anthropic/claude-haiku-4.5", ctx)
    assert ctx.chat_model == "anthropic/claude-haiku-4.5"


# ---------- /sessions ----------

def test_sessions_lists_db_sessions():
    sid1 = db.start_session()
    sid2 = db.start_session()
    out = dispatch("/sessions", make_ctx()).output
    assert f"#{sid1}" in out
    assert f"#{sid2}" in out


# ---------- /quit /exit ----------

def test_quit_sets_exit_flag():
    ctx = make_ctx()
    dispatch("/quit", ctx)
    assert ctx.exit_requested is True


def test_exit_sets_exit_flag():
    ctx = make_ctx()
    dispatch("/exit", ctx)
    assert ctx.exit_requested is True
