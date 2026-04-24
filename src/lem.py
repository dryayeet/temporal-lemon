"""Lemon CLI entry point: REPL with empathy pipeline, slash commands, SQLite memory."""
import random
from datetime import datetime

import requests

import db
from commands import ChatContext, dispatch, is_command
from config import CHAT_MODEL, KEEP_RECENT_TURNS, STATE_UPDATE_EVERY
from facts import FACTS_TAG, format_user_facts
from history import replace_system_block
from pipeline import run_empathy_turn
from prompt import LEMON_OPENERS, LEMON_PROMPT
from state import (
    format_internal_state,
    fresh_session_state,
    save_state,
    update_internal_state,
)
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

    print("lemon — type /help for commands, /quit to leave\n")

    first_message = random.choice(LEMON_OPENERS)
    print(f"lemon: {first_message}\n")
    ctx.history.append({"role": "assistant", "content": first_message})
    db.log_message(session_id, "assistant", first_message)

    exchange_count = 0

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

            ctx.last_trace = trace
            ctx.history.append({"role": "user", "content": user_input})
            ctx.history.append({"role": "assistant", "content": reply})

            print(f"lemon: {reply}\n")

            exchange_count += 1
            if exchange_count % STATE_UPDATE_EVERY == 0:
                ctx.internal_state = update_internal_state(
                    ctx.internal_state, user_input, reply
                )
                save_state(ctx.internal_state, session_id=session_id)
    finally:
        save_state(ctx.internal_state, session_id=session_id)
        db.end_session(session_id)


if __name__ == "__main__":
    main()
