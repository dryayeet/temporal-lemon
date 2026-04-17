"""Lemon CLI entry point: REPL with streaming chat, slash commands, SQLite memory."""
import random
import sys
from datetime import datetime

import requests

import db
from chat import stream_chat
from commands import ChatContext, dispatch, is_command
from config import CHAT_MODEL, KEEP_RECENT_TURNS, STATE_UPDATE_EVERY
from facts import FACTS_TAG, format_user_facts
from history import compress_history, replace_system_block
from prompt import LEMON_OPENERS, LEMON_PROMPT
from state import (
    format_internal_state,
    load_state,
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


def main() -> None:
    session_start = datetime.now()
    session_id = db.start_session()
    internal_state = load_state()

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

            ctx.history.append({"role": "user", "content": user_input})
            db.log_message(session_id, "user", user_input)

            # refresh ephemeral system blocks
            ctx.history = replace_system_block(
                ctx.history, "<time_context>", get_time_context(session_start), position=1
            )
            ctx.history = replace_system_block(
                ctx.history, "<internal_state>", format_internal_state(ctx.internal_state), position=2
            )
            facts_block = format_user_facts(db.get_facts())
            if facts_block:
                ctx.history = replace_system_block(ctx.history, FACTS_TAG, facts_block, position=3)
            ctx.history = compress_history(ctx.history, keep_recent=KEEP_RECENT_TURNS)

            try:
                reply = stream_chat(
                    ctx.history,
                    energy=ctx.internal_state.get("energy", "medium"),
                    model=ctx.chat_model,
                )
            except (requests.RequestException, RuntimeError) as e:
                print(f"\n[chat error: {e}]\n")
                ctx.history.pop()
                continue

            ctx.history.append({"role": "assistant", "content": reply})
            db.log_message(session_id, "assistant", reply)
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
