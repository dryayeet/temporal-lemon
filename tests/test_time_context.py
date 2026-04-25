from datetime import datetime, timedelta

from prompts import get_time_context
from temporal.clock import session_duration_note, time_of_day_label


def test_time_of_day_boundaries():
    assert time_of_day_label(4) == "very late night / early hours"
    assert time_of_day_label(5) == "morning"
    assert time_of_day_label(9) == "morning"
    assert time_of_day_label(10) == "afternoon"
    assert time_of_day_label(16) == "afternoon"
    assert time_of_day_label(17) == "evening"
    assert time_of_day_label(20) == "evening"
    assert time_of_day_label(21) == "late night"
    assert time_of_day_label(23) == "late night"
    assert time_of_day_label(0) == "very late night / early hours"
    assert time_of_day_label(2) == "very late night / early hours"


def test_session_duration_buckets():
    assert "just started" in session_duration_note(0)
    assert "just started" in session_duration_note(1)
    assert "5 minutes" in session_duration_note(5)
    assert "20 minutes" in session_duration_note(20)
    assert "long conversation" in session_duration_note(45)


def test_get_time_context_contains_expected_fields():
    start = datetime(2026, 4, 17, 10, 0)
    now = start + timedelta(minutes=15)
    ctx = get_time_context(start, now=now)

    assert "<time_context>" in ctx
    assert "</time_context>" in ctx
    assert "2026-04-17" in ctx
    assert "10:15" in ctx
    assert "Friday" in ctx       # April 17, 2026 is a Friday
    assert "afternoon" in ctx    # 10:15 falls in afternoon bucket
    assert "15 minutes" in ctx


def test_get_time_context_fresh_session():
    start = datetime(2026, 4, 17, 23, 30)
    ctx = get_time_context(start, now=start)
    assert "late night" in ctx
    assert "just started" in ctx
