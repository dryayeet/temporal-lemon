"""Shared session-context helpers used by both the CLI (`lem.py`) and the
web server (`web.py`).

Builds the initial system-prompt stack, refreshes the time/lemon_state/facts
blocks between turns, and runs the post-exchange bookkeeping thread (now
facts-only — state updates moved pre-reply with stage 2 of the dyadic-state
architecture).
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

import config
from commands import ChatContext
from empathy.post_exchange import bookkeep
from logging_setup import get_logger
from prompt_stack import replace_system_block
from prompts import (
    FACTS_TAG,
    LEMON_PROMPT,
    LEMON_STATE_TAG,
    TIME_TAG,
    format_lemon_state,
    format_user_facts,
    get_time_context,
)
from storage import db

log = get_logger("session_context")


def initial_history(lemon_state: dict, session_start: datetime) -> list[dict]:
    """Build the seed system-prompt stack for a fresh session."""
    history: list[dict] = [
        {"role": "system", "content": LEMON_PROMPT},
        {"role": "system", "content": get_time_context(session_start)},
        {"role": "system", "content": format_lemon_state(lemon_state)},
    ]
    facts = db.get_facts()
    facts_block = format_user_facts(facts)
    if facts_block:
        history.append({"role": "system", "content": facts_block})
    log.info("initial_history facts=%d blocks=%d", len(facts), len(history))
    return history


def refresh_base_blocks(
    history: list[dict],
    lemon_state: dict,
    session_start: datetime,
) -> list[dict]:
    """Return a copy of `history` with the time/lemon_state/facts system blocks
    replaced to reflect the current snapshot. The persona prompt at position 0
    is untouched so prompt caching keeps hitting."""
    h = list(history)
    h = replace_system_block(h, TIME_TAG, get_time_context(session_start), position=1)
    h = replace_system_block(h, LEMON_STATE_TAG, format_lemon_state(lemon_state), position=2)
    facts_block = format_user_facts(db.get_facts())
    if facts_block:
        h = replace_system_block(h, FACTS_TAG, facts_block, position=3)
    return h


def run_bookkeeping(
    ctx: ChatContext,
    session_id: int,
    user_msg: str,
    reply: str,
    trace,
    recent_snapshot: list[dict],
    model: str,
    lock: threading.Lock,
) -> None:
    """Post-exchange fact extraction. Runs in a daemon thread so the caller
    (REPL or SSE handler) never waits on it. Failures are logged and swallowed.

    With stages 2+3, lemon-state updates moved pre-reply (via the merged
    user_read pass), so this thread no longer touches state — only facts.
    """
    started = time.time()
    try:
        existing = db.get_facts()
        if config.ENABLE_AUTO_FACTS:
            new_facts = bookkeep(
                user_msg=user_msg,
                bot_reply=reply,
                existing_facts=existing,
                recent_msgs=recent_snapshot,
                model=model,
                max_new=config.AUTO_FACTS_MAX_PER_TURN,
            )
        else:
            new_facts = {}

        upserted = 0
        with lock:
            for k, v in list(new_facts.items())[:config.AUTO_FACTS_MAX_PER_TURN]:
                db.upsert_fact(k, v, source_session_id=session_id)
                upserted += 1
            trace.facts_extracted = new_facts
        log.info(
            "bookkeep_done upserted=%d elapsed_ms=%d",
            upserted, int((time.time() - started) * 1000),
        )
    except Exception as e:
        log.error("bookkeep_failed error=%r", e)
