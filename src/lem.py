"""Lemon CLI entry point: REPL with empathy pipeline, slash commands, SQLite memory."""
import random
import threading
from datetime import datetime
from typing import Optional

import requests

from commands import ChatContext, dispatch, is_command
from config import CHAT_MODEL, KEEP_RECENT_TURNS
from pipeline import recent_messages_for_context, run_empathy_turn
from prompt.persona import LEMON_OPENERS
from session_context import initial_history, refresh_base_blocks, run_bookkeeping
from storage import db
from storage.state import fresh_session_state, save_state


def main() -> None:
    session_start = datetime.now()
    session_id = db.start_session()
    internal_state = fresh_session_state()
    save_state(internal_state, session_id=session_id)

    ctx = ChatContext(
        history=initial_history(internal_state, session_start),
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
                base_history = refresh_base_blocks(ctx.history, ctx.internal_state, session_start)

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
                target=run_bookkeeping,
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
