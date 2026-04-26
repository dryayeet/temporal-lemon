"""Lemon CLI entry point: REPL with empathy pipeline, slash commands, SQLite memory."""
import random
import threading
from datetime import datetime
from typing import Optional

import requests

import config
from commands import ChatContext, dispatch, is_command
from config import CHAT_MODEL, KEEP_RECENT_TURNS
from logging_setup import get_logger, setup_logging
from pipeline import recent_messages_for_context, run_empathy_turn
from prompts import LEMON_OPENERS
from session_context import initial_history, refresh_base_blocks, run_bookkeeping
from storage import db
from storage.lemon_state import fresh_lemon_session_state, save_lemon_state
from storage.user_state import fresh_user_session_state


def main() -> None:
    setup_logging()
    log = get_logger("lem")
    log.info(
        "cli_startup chat_model=%s state_model=%s",
        config.CHAT_MODEL, config.STATE_MODEL,
    )

    session_start = datetime.now()
    session_id = db.start_session()
    lemon_state = fresh_lemon_session_state()
    save_lemon_state(lemon_state, session_id=session_id)
    user_state = fresh_user_session_state()

    ctx = ChatContext(
        history=initial_history(lemon_state, session_start),
        lemon_state=lemon_state,
        chat_model=CHAT_MODEL,
        session_id=session_id,
        user_state=user_state,
    )

    # Guards ctx.lemon_state / ctx.user_state / ctx.history while the bookkeeping
    # thread runs in parallel.
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
                base_history = refresh_base_blocks(ctx.history, ctx.lemon_state, session_start)

            try:
                reply, trace = run_empathy_turn(
                    user_msg=user_input,
                    base_history=base_history,
                    model=ctx.chat_model,
                    session_id=session_id,
                    keep_recent_turns=KEEP_RECENT_TURNS,
                    on_phase=lambda phase: print(f"  · {phase}...", flush=True),
                    user_state=ctx.user_state,
                    lemon_state=ctx.lemon_state,
                )
            except (requests.RequestException, RuntimeError) as e:
                print(f"\n[chat error: {e}]\n")
                continue

            with ctx_lock:
                ctx.last_trace = trace
                ctx.history.append({"role": "user", "content": user_input})
                ctx.history.append({"role": "assistant", "content": reply})
                # Pull both freshly-updated states off the trace; pipeline
                # already persisted them. Next turn reads from these values.
                if getattr(trace, "user_state_after", None) is not None:
                    ctx.user_state = trace.user_state_after
                if getattr(trace, "lemon_state_after", None) is not None:
                    ctx.lemon_state = trace.lemon_state_after
                recent_snapshot = recent_messages_for_context(ctx.history)
                model_snapshot = ctx.chat_model

            print(f"lemon: {reply}\n")

            last_bg = threading.Thread(
                target=run_bookkeeping,
                args=(
                    ctx, session_id, user_input, reply, trace,
                    recent_snapshot, model_snapshot, ctx_lock,
                ),
                daemon=True,
            )
            last_bg.start()
    finally:
        # Wait briefly for the final bookkeeping thread (facts only now;
        # state writes happen in the pipeline already).
        if last_bg is not None and last_bg.is_alive():
            last_bg.join(timeout=10)
        save_lemon_state(ctx.lemon_state, session_id=session_id)
        db.end_session(session_id)


if __name__ == "__main__":
    main()
