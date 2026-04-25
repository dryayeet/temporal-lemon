from prompt_stack import compress_history, replace_system_block


def sys_msg(content):
    return {"role": "system", "content": content}


def user_msg(content):
    return {"role": "user", "content": content}


def asst_msg(content):
    return {"role": "assistant", "content": content}


# ---------- replace_system_block ----------

def test_replace_inserts_when_absent():
    history = [sys_msg("<persona>..."), user_msg("hi")]
    out = replace_system_block(history, "<time_context>", "<time_context>10:00</time_context>", position=1)
    assert out[0]["content"].startswith("<persona>")
    assert out[1] == {"role": "system", "content": "<time_context>10:00</time_context>"}
    assert out[2] == user_msg("hi")


def test_replace_removes_existing_and_reinserts():
    history = [
        sys_msg("<persona>..."),
        sys_msg("<time_context>old</time_context>"),
        user_msg("hi"),
    ]
    out = replace_system_block(history, "<time_context>", "<time_context>new</time_context>", position=1)
    time_blocks = [m for m in out if "<time_context>" in m["content"]]
    assert len(time_blocks) == 1
    assert "new" in time_blocks[0]["content"]
    assert out[1] == {"role": "system", "content": "<time_context>new</time_context>"}


def test_replace_only_touches_matching_tag():
    history = [
        sys_msg("<persona>..."),
        sys_msg("<internal_state>old</internal_state>"),
    ]
    out = replace_system_block(history, "<time_context>", "<time_context>x</time_context>", position=1)
    # original internal_state block must survive
    assert any("<internal_state>" in m["content"] for m in out)


# ---------- compress_history ----------

def test_compress_noop_when_under_threshold():
    history = [sys_msg("<persona>"), user_msg("a"), asst_msg("b")]
    assert compress_history(history, keep_recent=8) == history


def test_compress_folds_old_turns_into_summary():
    history = [sys_msg("<persona>")]
    # 10 conversation turns, keep_recent=4 — 6 should be folded
    for i in range(5):
        history.append(user_msg(f"u{i}"))
        history.append(asst_msg(f"a{i}"))

    out = compress_history(history, keep_recent=4)

    # structure: [persona, summary_block, last 4 convo turns]
    assert out[0] == sys_msg("<persona>")
    assert out[1]["role"] == "system"
    assert "<earlier_conversation>" in out[1]["content"]
    assert "u0" in out[1]["content"] and "a2" in out[1]["content"]
    # last 4 verbatim, in order
    assert [m["content"] for m in out[-4:]] == ["u3", "a3", "u4", "a4"]


def test_compress_preserves_all_system_messages():
    history = [
        sys_msg("<persona>"),
        sys_msg("<time_context>"),
        sys_msg("<internal_state>"),
    ]
    for i in range(12):
        history.append(user_msg(f"u{i}"))

    out = compress_history(history, keep_recent=4)
    system_contents = [m["content"] for m in out if m["role"] == "system"]
    # three original system blocks + one summary block
    assert "<persona>" in system_contents
    assert "<time_context>" in system_contents
    assert "<internal_state>" in system_contents
    assert any("<earlier_conversation>" in c for c in system_contents)
