"""Lemon web UI: FastAPI + single-page HTML, SSE streaming for chat.

Run with:
    uvicorn web:app --reload --app-dir src
or:
    python src/web.py
"""
from __future__ import annotations

import json
import random
import threading
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterator

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

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

app = FastAPI(title="lemon")

# ---------- single-process session state ----------
# This server is meant for the user themselves — one chat at a time.
# A lock guards the shared ChatContext from interleaving across requests.
_lock = Lock()
_session_start = datetime.now()
_session_id = db.start_session()
_internal_state = fresh_session_state()
save_state(_internal_state, session_id=_session_id)


def _initial_history() -> list[dict]:
    h = [
        {"role": "system", "content": LEMON_PROMPT},
        {"role": "system", "content": get_time_context(_session_start)},
        {"role": "system", "content": format_internal_state(_internal_state)},
    ]
    facts_block = format_user_facts(db.get_facts())
    if facts_block:
        h.append({"role": "system", "content": facts_block})
    return h


_ctx = ChatContext(
    history=_initial_history(),
    internal_state=_internal_state,
    chat_model=CHAT_MODEL,
    session_id=_session_id,
)

# greet on startup so the UI has something to render on first paint
_first = random.choice(LEMON_OPENERS)
_ctx.history.append({"role": "assistant", "content": _first})
db.log_message(_session_id, "assistant", _first)


def _refresh_base_blocks() -> list[dict]:
    h = list(_ctx.history)
    h = replace_system_block(h, "<time_context>", get_time_context(_session_start), position=1)
    h = replace_system_block(h, "<internal_state>", format_internal_state(_ctx.internal_state), position=2)
    facts_block = format_user_facts(db.get_facts())
    if facts_block:
        h = replace_system_block(h, FACTS_TAG, facts_block, position=3)
    return h


# ---------- request models ----------

class ChatRequest(BaseModel):
    message: str


class CommandRequest(BaseModel):
    text: str


# ---------- index ----------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ---------- chat (SSE) ----------

@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    msg = req.message.strip()
    if not msg:
        raise HTTPException(400, "empty message")

    if is_command(msg):
        raise HTTPException(400, "use POST /command for slash commands")

    return StreamingResponse(_stream_reply(msg), media_type="text/event-stream")


def _stream_reply(user_msg: str) -> Iterator[bytes]:
    """SSE generator: emit phase events, then run the pipeline, then deliver
    the reply. Fact + state bookkeeping fires AFTER `done` in a daemon thread
    so the user never waits on it."""
    phase_queue: list[str] = []

    def push_phase(phase: str) -> None:
        phase_queue.append(phase)

    with _lock:
        base_history = _refresh_base_blocks()
        model = _ctx.chat_model

    try:
        # we yield the first phase immediately so the UI sees "lemon is reading you..."
        yield _sse("phase", "reading you")

        reply, trace = run_empathy_turn(
            user_msg=user_msg,
            base_history=base_history,
            model=model,
            session_id=_session_id,
            keep_recent_turns=KEEP_RECENT_TURNS,
            on_phase=push_phase,
        )

        # flush queued phase events (skipping the first which we already sent)
        for phase in phase_queue[1:]:
            yield _sse("phase", phase)

    except (requests.RequestException, RuntimeError) as e:
        yield _sse("error", str(e))
        return

    with _lock:
        _ctx.last_trace = trace
        _ctx.history.append({"role": "user", "content": user_msg})
        _ctx.history.append({"role": "assistant", "content": reply})
        # snapshot inputs the bookkeeping thread will need; take them now while
        # we hold the lock so they stay consistent with the history we just appended.
        recent_snapshot = recent_messages_for_context(_ctx.history)
        state_snapshot = dict(_ctx.internal_state)
        model_snapshot = _ctx.chat_model

    yield _sse("token", reply)
    yield _sse("done", reply)

    threading.Thread(
        target=_run_bookkeeping,
        args=(user_msg, reply, trace, recent_snapshot, state_snapshot, model_snapshot),
        daemon=True,
    ).start()


def _run_bookkeeping(
    user_msg: str,
    reply: str,
    trace,
    recent_snapshot: list[dict],
    state_snapshot: dict,
    model: str,
) -> None:
    """Run the merged fact+state bookkeeping call, then apply its results.

    Failures are swallowed so a bad bookkeep never affects the user-visible
    reply (which already shipped) or subsequent turns.
    """
    try:
        existing = db.get_facts()
        new_facts, new_state = bookkeep(
            user_msg=user_msg,
            bot_reply=reply,
            existing_facts=existing,
            current_state=state_snapshot,
            recent_msgs=recent_snapshot,
            model=model,
            max_new=config.AUTO_FACTS_MAX_PER_TURN,
        ) if config.ENABLE_AUTO_FACTS else ({}, state_snapshot)

        with _lock:
            for k, v in list(new_facts.items())[:config.AUTO_FACTS_MAX_PER_TURN]:
                db.upsert_fact(k, v, source_session_id=_session_id)
            _ctx.internal_state = new_state
            save_state(new_state, session_id=_session_id)
            # surface the extracted facts on the trace so /trace and /why
            # still report them (albeit after a short delay).
            trace.facts_extracted = new_facts
    except Exception as e:
        print(f"  [bookkeeping thread failed: {e}]")


def _sse(event: str, data: str) -> bytes:
    """Encode an SSE event. Data is JSON-wrapped so newlines/quotes are safe."""
    payload = json.dumps({"event": event, "data": data})
    return f"data: {payload}\n\n".encode("utf-8")


# ---------- slash commands ----------

@app.post("/command")
def command(req: CommandRequest) -> JSONResponse:
    text = req.text.strip()
    if not is_command(text):
        raise HTTPException(400, "not a slash command")
    with _lock:
        result = dispatch(text, _ctx)
    return JSONResponse({"output": result.output, "exit": _ctx.exit_requested})


# ---------- introspection ----------

@app.get("/state")
def get_state() -> JSONResponse:
    return JSONResponse(_ctx.internal_state)


@app.get("/facts")
def get_facts_endpoint() -> JSONResponse:
    return JSONResponse(db.get_facts())


@app.get("/sessions")
def get_sessions() -> JSONResponse:
    return JSONResponse(db.list_sessions(limit=20))


@app.get("/history")
def get_history() -> JSONResponse:
    convo = [
        {"role": m["role"], "content": m["content"]}
        for m in _ctx.history if m["role"] != "system"
    ]
    return JSONResponse(convo)


@app.get("/trace")
def get_trace() -> JSONResponse:
    """Return the most recent pipeline trace as JSON, for debugging from the UI."""
    trace = _ctx.last_trace
    if trace is None:
        return JSONResponse({"available": False})
    return JSONResponse({
        "available": True,
        "pipeline_used": getattr(trace, "pipeline_used", False),
        "emotion": getattr(trace, "emotion", None),
        "tom": getattr(trace, "tom", None),
        "memories": [
            {"when": m.get("created_at"), "emotion": m.get("emotion"),
             "content": m.get("content")}
            for m in (getattr(trace, "memories", None) or [])
        ],
        "regenerated": getattr(trace, "regenerated", False),
        "check": (
            None if getattr(trace, "check", None) is None
            else {
                "passed": trace.check.passed,
                "failures": trace.check.failures,
            }
        ),
        "facts_extracted": getattr(trace, "facts_extracted", {}) or {},
    })


# ---------- entry point ----------

def main() -> None:
    import uvicorn
    uvicorn.run("web:app", host="127.0.0.1", port=8000, reload=False, app_dir=str(Path(__file__).parent))


if __name__ == "__main__":
    main()
