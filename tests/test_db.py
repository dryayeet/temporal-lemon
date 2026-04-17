import db


def test_session_lifecycle():
    sid = db.start_session()
    assert isinstance(sid, int) and sid > 0

    db.log_message(sid, "user", "hi")
    db.log_message(sid, "assistant", "hey")
    db.end_session(sid)

    sessions = db.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == sid
    assert sessions[0]["msg_count"] == 2
    assert sessions[0]["ended_at"] is not None


def test_session_messages_in_order():
    sid = db.start_session()
    db.log_message(sid, "user", "first")
    db.log_message(sid, "assistant", "second")
    db.log_message(sid, "user", "third")

    msgs = db.session_messages(sid)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [m["content"] for m in msgs] == ["first", "second", "third"]


def test_state_snapshot_returns_latest():
    assert db.latest_state() is None
    sid = db.start_session()
    db.save_state_snapshot({"mood": "good"}, session_id=sid)
    db.save_state_snapshot({"mood": "tired"}, session_id=sid)
    assert db.latest_state() == {"mood": "tired"}


def test_facts_upsert_and_delete():
    db.upsert_fact("city", "Bangalore")
    db.upsert_fact("pet", "cat named pickle")
    assert db.get_facts() == {"city": "Bangalore", "pet": "cat named pickle"}

    # upsert overwrites value, not inserts a duplicate
    db.upsert_fact("city", "Mumbai")
    assert db.get_facts()["city"] == "Mumbai"

    assert db.delete_fact("pet") is True
    assert "pet" not in db.get_facts()
    assert db.delete_fact("nonexistent") is False


def test_facts_clear():
    db.upsert_fact("a", "1")
    db.upsert_fact("b", "2")
    db.clear_facts()
    assert db.get_facts() == {}


def test_list_sessions_orders_newest_first():
    s1 = db.start_session()
    s2 = db.start_session()
    s3 = db.start_session()
    rows = db.list_sessions()
    assert [r["id"] for r in rows] == [s3, s2, s1]


def test_list_sessions_respects_limit():
    for _ in range(5):
        db.start_session()
    rows = db.list_sessions(limit=2)
    assert len(rows) == 2


# ---------- emotion fields ----------

def test_log_message_stores_emotion_fields():
    sid = db.start_session()
    db.log_message(sid, "user", "I'm sad", emotion="sadness", intensity=0.7, salience=0.7)

    msgs = db.session_messages(sid)
    assert msgs[0]["emotion"] == "sadness"
    assert msgs[0]["intensity"] == 0.7
    assert msgs[0]["salience"] == 0.7


def test_log_message_emotion_fields_default_to_none():
    sid = db.start_session()
    db.log_message(sid, "assistant", "hey")
    msgs = db.session_messages(sid)
    assert msgs[0]["emotion"] is None
    assert msgs[0]["intensity"] is None


def test_find_messages_by_emotion_filters_by_role_and_label():
    sid = db.start_session()
    db.log_message(sid, "user", "u-sad", emotion="sadness", intensity=0.5)
    db.log_message(sid, "assistant", "a-sad", emotion="sadness", intensity=0.5)
    db.log_message(sid, "user", "u-joy", emotion="joy", intensity=0.5)

    rows = db.find_messages_by_emotion("sadness", limit=10)
    assert [r["content"] for r in rows] == ["u-sad"]


def test_find_messages_by_emotion_excludes_session():
    s1 = db.start_session()
    db.log_message(s1, "user", "old", emotion="anger", intensity=0.5)
    s2 = db.start_session()
    db.log_message(s2, "user", "new", emotion="anger", intensity=0.5)

    rows = db.find_messages_by_emotion("anger", exclude_session_id=s2)
    assert [r["content"] for r in rows] == ["old"]


def test_find_recent_messages_orders_newest_first():
    s1 = db.start_session()
    db.log_message(s1, "user", "first")
    db.log_message(s1, "assistant", "second")
    db.log_message(s1, "user", "third")

    rows = db.find_recent_messages(limit=2)
    assert [r["content"] for r in rows] == ["third", "second"]


def test_schema_version_recorded():
    db.start_session()  # triggers schema setup
    with db.connect() as c:
        rows = c.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
        versions = [r["version"] for r in rows]
        assert db.LATEST_VERSION in versions
