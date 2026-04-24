"""Lemon CLI entry point: REPL with empathy pipeline, slash commands, SQLite memory."""
import random
import threading
from datetime import datetime
from typing import Optional

import requests

import config
import db
from commands import ChatContext, dispatch, is_command
from config import CHAT_MODEL, KEEP_RECENT_TURNS
from facts import FACTS_TAG, format_user_facts
from history import replace_system_block
from pipeline import recent_messages_for_context, run_empathy_turn
from post_exchange import bookkeep
from prompt import LEMON_OPENERS, LEMON_PROMPT
from state import format_internal_state, fresh_session_state, save_state
from time_context import get_time_context


def build_initial_history(internal_state: dict, session_start: datetime) -> list[dict]:
    history: list[dict] = [
        {"role": "system", "content": LEMON_PROMPT},
        {"role": "system", "content": get_time_context(session_start)},
        {"role": "system", "content": format_internal_state(internal_state)},
    ]
    facts_block = format_user_facts(db.get_facts())
    if facts_block:
        history.append({"role": "system", "content": facts_block})
    return history


def refresh_base_blocks(ctx: ChatContext, session_start: datetime) -> list[dict]:
    """Return a copy of ctx.history with time/state/facts system blocks refreshed."""
    h = list(ctx.history)
    h = replace_system_block(h, "<time_context>", get_time_context(session_start), position=1)
    h = replace_system_block(h, "<internal_state>", format_internal_state(ctx.internal_state), position=2)
    facts_block = format_user_facts(db.get_facts())
    if facts_block:
        h = replace_system_block(h, FACTS_TAG, facts_block, position=3)
    return h


def _run_bookkeeping(
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
    """Merged fact + state bookkeeping. Runs in a daemon thread so the REPL
    never waits for it. Failures are swallowed."""
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
            trace.facts_extracted = new_facts
    except Exception as e:
        print(f"  [bookkeeping thread failed: {e}]")


def main() -> None:
    session_start = datetime.now()
    session_id = db.start_session()
    internal_state = fresh_session_state()
    save_state(internal_state, session_id=session_id)

    ctx = ChatContext(
        history=build_initial_history(internal_state, session_start),
        internal_state=internal_state,
        chat_model=CHAT_MODEL,
        session_id=session_id,
    )

    # Guards ctx.internal_state / ctx.history while a bookkeeping thread is
    # mutating them in parallel with the main REPL loop.
    ctx_lock = threading.Lock()
    last_bg: Optional[threading.Thread] = None

    print("lemon — type /help for commands, /quit to leave\n")

    first_message = random.choice(LEMON_OPENERS)
    print(f"lemon: {first_message}\n")
    ctx.history.append({"role": "assistant", "content": first_message})
    db.log_message(session_id, "assistant", first_message)

    try:
        while not ctx.exit_requested:
            try:
                user_input = input("you: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            if is_command(user_input):
                result = dispatch(user_input, ctx)
                print(f"\n{result.output}\n")
                continue

            with ctx_lock:
                base_history = refresh_base_blocks(ctx, session_start)

            try:
                reply, trace = run_empathy_turn(
                    user_msg=user_input,
                    base_history=base_history,
                    model=ctx.chat_model,
                    session_id=session_id,
                    keep_recent_turns=KEEP_RECENT_TURNS,
                    on_phase=lambda phase: print(f"  · {phase}...", flush=True),
                )
            except (requests.RequestException, RuntimeError) as e:
                print(f"\n[chat error: {e}]\n")
                continue

            with ctx_lock:
                ctx.last_trace = trace
                ctx.history.append({"role": "user", "content": user_input})
                ctx.history.append({"role": "assistant", "content": reply})
                recent_snapshot = recent_messages_for_context(ctx.history)
                state_snapshot = dict(ctx.internal_state)
                model_snapshot = ctx.chat_model

            print(f"lemon: {reply}\n")

            last_bg = threading.Thread(
                target=_run_bookkeeping,
                args=(
                    ctx, session_id, user_input, reply, trace,
                    recent_snapshot, state_snapshot, model_snapshot, ctx_lock,
                ),
                daemon=True,
            )
            last_bg.start()
    finally:
        # Wait briefly for the final bookkeeping thread so its state write
        # doesn't race the cleanup save_state below.
        if last_bg is not None and last_bg.is_alive():
            last_bg.join(timeout=10)
        save_state(ctx.internal_state, session_id=session_id)
        db.end_session(session_id)


if __name__ == "__main__":
    main()
