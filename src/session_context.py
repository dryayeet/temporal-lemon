"""Shared session-context helpers used by both the CLI (`lem.py`) and the
web server (`web.py`).

These used to live duplicated in both entry points: building the initial
system-prompt history, refreshing time/state/facts blocks between turns,
and running the post-exchange bookkeeping thread. They're byte-identical
work so they belong in one place.
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Optional

import config
from commands import ChatContext
from empathy.post_exchange import bookkeep
from prompt_stack import replace_system_block
from prompts import (
    FACTS_TAG,
    LEMON_PROMPT,
    format_internal_state,
    format_user_facts,
    get_time_context,
)
from storage import db
from storage.state import save_state


def initial_history(internal_state: dict, session_start: datetime) -> list[dict]:
    """Build the seed system-prompt stack for a fresh session."""
    history: list[dict] = [
        {"role": "system", "content": LEMON_PROMPT},
        {"role": "system", "content": get_time_context(session_start)},
        {"role": "system", "content": format_internal_state(internal_state)},
    ]
    facts_block = format_user_facts(db.get_facts())
    if facts_block:
        history.append({"role": "system", "content": facts_block})
    return history


def refresh_base_blocks(
    history: list[dict],
    internal_state: dict,
    session_start: datetime,
) -> list[dict]:
    """Return a copy of `history` with the time/state/facts system blocks
    replaced to reflect the current snapshot. The persona prompt at position
    0 is untouched so prompt caching keeps hitting."""
    h = list(history)
    h = replace_system_block(h, "<time_context>", get_time_context(session_start), position=1)
    h = replace_system_block(h, "<internal_state>", format_internal_state(internal_state), position=2)
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
    state_snapshot: dict,
    model: str,
    lock: threading.Lock,
) -> None:
    """Merged fact + state bookkeeping. Runs in a daemon thread so the caller
    (REPL or SSE handler) never waits on it. Failures are logged and swallowed
    so a bookkeep hiccup never affects the reply that already shipped.
    """
    try:
        existing = db.get_facts()
        if config.ENABLE_AUTO_FACTS:
            new_facts, new_state = bookkeep(
                user_msg=user_msg,
                bot_reply=reply,
                existing_facts=existing,
                current_state=state_snapshot,
                recent_msgs=recent_snapshot,
                model=model,
                max_new=config.AUTO_FACTS_MAX_PER_TURN,
            )
        else:
            new_facts, new_state = {}, state_snapshot

        with lock:
            for k, v in list(new_facts.items())[:config.AUTO_FACTS_MAX_PER_TURN]:
                db.upsert_fact(k, v, source_session_id=session_id)
            ctx.internal_state = new_state
            save_state(new_state, session_id=session_id)
            # surface extracted facts on the trace so /trace and /why report
            # them (after a short delay — bookkeeping runs post-reply).
            trace.facts_extracted = new_facts
    except Exception as e:
        print(f"  [bookkeeping thread failed: {e}]")
