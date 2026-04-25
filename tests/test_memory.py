"""Unit tests for the composite-scored memory retriever.

Accuracy benchmarking lives separately in `eval/test_retrieval.py` —
this file only covers the mechanical contracts (signature, exclusion,
limit, score-shape, fallback path).
"""
from prompts import format_memory_block
from storage import db
from storage.memory import (
    _build_fts_query,
    _normalize_bm25,
    _tokenize_for_fts,
    relevant_memories,
)


# ---------- tokenization helpers ----------

def test_tokenizer_drops_stopwords_and_short_tokens():
    tokens = _tokenize_for_fts("the exam is on Tuesday and i am scared")
    assert "exam" in tokens
    assert "tuesday" in tokens
    assert "scared" in tokens
    assert "the" not in tokens
    assert "is" not in tokens  # short
    assert "and" not in tokens
    assert "i" not in tokens


def test_tokenizer_handles_hinglish_stopwords():
    tokens = _tokenize_for_fts("yaar mera exam hai kal subah")
    assert "exam" in tokens
    assert "subah" in tokens
    assert "yaar" not in tokens  # in stopword set
    assert "hai" not in tokens   # in stopword set


def test_build_fts_query_or_combines():
    q = _build_fts_query("exam tuesday prep")
    assert q is not None
    assert " OR " in q
    assert "exam" in q


def test_build_fts_query_returns_none_for_empty_or_all_stopwords():
    assert _build_fts_query("") is None
    assert _build_fts_query("the and you i") is None


def test_build_fts_query_dedupes():
    q = _build_fts_query("exam exam exam tuesday")
    assert q == "exam OR tuesday"


def test_normalize_bm25_uniform_scores_get_uniform_output():
    out = _normalize_bm25([-2.0, -2.0, -2.0])
    assert out[0] == out[1] == out[2]
    assert 0.5 < out[0] < 1.0  # negative bm25 = relevant, so > 0.5


def test_normalize_bm25_more_relevant_gets_higher_score():
    out = _normalize_bm25([-3.0, -1.0, -2.0])
    # smaller raw bm25 = higher relevance → larger normalized
    assert out[0] > out[2] > out[1]
    assert all(0.0 < x < 1.0 for x in out)


def test_normalize_bm25_zero_maps_to_half():
    # Zero relevance (BM25 = 0) sits at the sigmoid midpoint.
    assert _normalize_bm25([0.0]) == [0.5]


def test_normalize_bm25_close_scores_stay_close():
    """Sigmoid means small BM25 deltas don't blow up to full-range deltas
    (the property that makes the intensity/emotion terms meaningful)."""
    out = _normalize_bm25([-2.0, -2.1])
    assert abs(out[0] - out[1]) < 0.05


# ---------- retrieval ----------

def _seed_user_msg(session_id, content, emotion=None, intensity=None):
    return db.log_message(
        session_id, "user", content,
        emotion=emotion, intensity=intensity, salience=intensity,
    )


def test_retrieval_excludes_current_session():
    s1 = db.start_session()
    _seed_user_msg(s1, "exam tomorrow stressing me out", emotion="anxiety", intensity=0.6)
    s2 = db.start_session()
    _seed_user_msg(s2, "the exam went badly today", emotion="disappointment", intensity=0.5)

    out = relevant_memories(
        user_msg="how did the exam go",
        emotion="disappointment",
        intensity=0.5,
        current_session_id=s2,
    )
    contents = [r["content"] for r in out]
    assert "exam tomorrow stressing me out" in contents
    assert "the exam went badly today" not in contents


def test_retrieval_respects_limit():
    s1 = db.start_session()
    for i in range(8):
        _seed_user_msg(s1, f"talked about topic {i}", emotion="joy", intensity=0.4)
    s2 = db.start_session()

    out = relevant_memories(
        user_msg="topic", emotion="joy", intensity=0.4,
        current_session_id=s2, limit=3,
    )
    assert len(out) == 3


def test_retrieval_returns_results_on_neutral_emotion():
    """The old code early-returned on neutral; new behavior must NOT."""
    s1 = db.start_session()
    _seed_user_msg(s1, "ate biryani for lunch", emotion="neutral", intensity=0.2)
    s2 = db.start_session()

    out = relevant_memories(
        user_msg="lunch was biryani again",
        emotion="neutral",
        intensity=0.2,
        current_session_id=s2,
    )
    assert len(out) >= 1
    assert "biryani" in out[0]["content"]


def test_retrieval_returns_empty_when_no_messages_at_all():
    s = db.start_session()
    out = relevant_memories(
        user_msg="anything", emotion="joy", intensity=0.5,
        current_session_id=s,
    )
    assert out == []


def test_retrieval_attaches_score_and_breakdown():
    s1 = db.start_session()
    _seed_user_msg(s1, "I love eating biryani on weekends",
                   emotion="joy", intensity=0.7)
    s2 = db.start_session()

    out = relevant_memories(
        user_msg="want to eat biryani tonight",
        emotion="joy", intensity=0.5,
        current_session_id=s2,
    )
    assert len(out) >= 1
    r = out[0]
    assert "score" in r and 0.0 <= r["score"] <= 1.0
    assert "breakdown" in r
    for key in ("lex", "rec", "int", "emo", "total"):
        assert key in r["breakdown"]


def test_emotion_match_boosts_over_unrelated():
    """Same lexical match, different emotion — same-family wins."""
    s1 = db.start_session()
    sad_id = _seed_user_msg(s1, "rough day at work today",
                            emotion="sadness", intensity=0.6)
    happy_id = _seed_user_msg(s1, "rough day at work was funny",
                              emotion="joy", intensity=0.6)
    s2 = db.start_session()

    out = relevant_memories(
        user_msg="work was rough",
        emotion="loneliness",  # same family as sadness, different family from joy
        intensity=0.6,
        current_session_id=s2,
        limit=5,
    )
    ids = [r["id"] for r in out]
    # Both should appear; sadness one should rank higher than joy one.
    assert sad_id in ids
    assert happy_id in ids
    sad_rank = ids.index(sad_id)
    happy_rank = ids.index(happy_id)
    assert sad_rank < happy_rank


def test_fallback_when_no_lexical_overlap():
    """When the FTS query has no hits, recency/emotion-only retrieval
    still returns something usable."""
    s1 = db.start_session()
    mid = _seed_user_msg(s1, "totally unrelated content here",
                         emotion="anxiety", intensity=0.7)
    s2 = db.start_session()

    out = relevant_memories(
        user_msg="zzz qqq xxx",  # nothing matches the corpus
        emotion="anxiety",
        intensity=0.7,
        current_session_id=s2,
    )
    assert len(out) >= 1
    # Lexical signal should be zero in the breakdown (fallback path).
    assert out[0]["breakdown"]["lex"] == 0.0


# ---------- format_memory_block (unchanged) ----------

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
