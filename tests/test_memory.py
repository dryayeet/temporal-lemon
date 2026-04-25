from storage import db
from storage.memory import format_memory_block, relevant_memories


def _seed_user_msg(session_id, content, emotion, intensity=0.5):
    return db.log_message(
        session_id, "user", content,
        emotion=emotion, intensity=intensity, salience=intensity,
    )


def test_relevant_memories_excludes_current_session():
    s1 = db.start_session()
    _seed_user_msg(s1, "old sad msg", "sadness")
    s2 = db.start_session()  # current session
    _seed_user_msg(s2, "today's sad msg", "sadness")

    rows = relevant_memories(emotion="sadness", current_session_id=s2, limit=5)
    contents = [r["content"] for r in rows]
    assert "old sad msg" in contents
    assert "today's sad msg" not in contents


def test_relevant_memories_filters_by_emotion():
    s1 = db.start_session()
    _seed_user_msg(s1, "sad one", "sadness")
    _seed_user_msg(s1, "happy one", "joy")
    s2 = db.start_session()

    sad_rows = relevant_memories("sadness", current_session_id=s2)
    assert [r["content"] for r in sad_rows] == ["sad one"]

    joy_rows = relevant_memories("joy", current_session_id=s2)
    assert [r["content"] for r in joy_rows] == ["happy one"]


def test_relevant_memories_neutral_returns_empty():
    s1 = db.start_session()
    _seed_user_msg(s1, "meh", "neutral")
    assert relevant_memories("neutral") == []


def test_relevant_memories_empty_emotion_returns_empty():
    assert relevant_memories("") == []


def test_relevant_memories_orders_newest_first():
    s1 = db.start_session()
    _seed_user_msg(s1, "first", "anger")
    _seed_user_msg(s1, "second", "anger")
    _seed_user_msg(s1, "third", "anger")
    s2 = db.start_session()

    rows = relevant_memories("anger", current_session_id=s2, limit=5)
    assert [r["content"] for r in rows] == ["third", "second", "first"]


def test_relevant_memories_respects_limit():
    s1 = db.start_session()
    for i in range(10):
        _seed_user_msg(s1, f"msg {i}", "fear")
    s2 = db.start_session()

    rows = relevant_memories("fear", current_session_id=s2, limit=3)
    assert len(rows) == 3


# ---------- format_memory_block ----------

def test_format_empty_returns_empty_string():
    assert format_memory_block([]) == ""


def test_format_wraps_in_tag():
    rows = [{"content": "hi", "emotion": "joy", "created_at": "2026-04-17T10:00:00"}]
    out = format_memory_block(rows)
    assert out.startswith("<emotional_memory>")
    assert out.endswith("</emotional_memory>")
    assert "joy" in out
    assert '"hi"' in out


def test_format_truncates_long_content():
    long = "x" * 500
    rows = [{"content": long, "emotion": "sadness", "created_at": "2026-04-17T10:00:00"}]
    out = format_memory_block(rows)
    assert "..." in out
    assert long not in out
