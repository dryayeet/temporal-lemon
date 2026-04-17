"""Lemon web UI: FastAPI + single-page HTML, SSE streaming for chat.

Run with:
    uvicorn web:app --reload --app-dir src
or:
    python src/web.py
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterator

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import db
from chat import humanize_delay, iter_chat
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

app = FastAPI(title="lemon")

# ---------- single-process session state ----------
# This server is meant for the user themselves — one chat at a time.
# A lock guards the shared ChatContext from interleaving across requests.
_lock = Lock()
_session_start = datetime.now()
_session_id = db.start_session()
_internal_state = load_state()


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
_exchange_count = 0

# greet on startup so the UI has something to render on first paint
_first = random.choice(LEMON_OPENERS)
_ctx.history.append({"role": "assistant", "content": _first})
db.log_message(_session_id, "assistant", _first)


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
        # commands aren't streamed — easier as a separate endpoint
        raise HTTPException(400, "use POST /command for slash commands")

    return StreamingResponse(_stream_reply(msg), media_type="text/event-stream")


def _stream_reply(user_msg: str) -> Iterator[bytes]:
    """SSE generator: yield each token, then a DONE event with the full reply."""
    global _exchange_count

    with _lock:
        _ctx.history.append({"role": "user", "content": user_msg})
        db.log_message(_session_id, "user", user_msg)

        _ctx.history = replace_system_block(
            _ctx.history, "<time_context>", get_time_context(_session_start), position=1
        )
        _ctx.history = replace_system_block(
            _ctx.history, "<internal_state>", format_internal_state(_ctx.internal_state), position=2
        )
        facts_block = format_user_facts(db.get_facts())
        if facts_block:
            _ctx.history = replace_system_block(_ctx.history, FACTS_TAG, facts_block, position=3)
        _ctx.history = compress_history(_ctx.history, keep_recent=KEEP_RECENT_TURNS)

        history_snapshot = list(_ctx.history)
        energy = _ctx.internal_state.get("energy", "medium")
        model = _ctx.chat_model

    chunks: list[str] = []
    try:
        for delta in iter_chat(history_snapshot, model=model):
            chunks.append(delta)
            yield _sse("token", delta)
            time.sleep(humanize_delay(delta, energy))
    except (requests.RequestException, RuntimeError) as e:
        with _lock:
            _ctx.history.pop()  # roll back the user message
        yield _sse("error", str(e))
        return

    full = "".join(chunks)

    with _lock:
        _ctx.history.append({"role": "assistant", "content": full})
        db.log_message(_session_id, "assistant", full)
        _exchange_count += 1
        if _exchange_count % STATE_UPDATE_EVERY == 0:
            _ctx.internal_state = update_internal_state(_ctx.internal_state, user_msg, full)
            save_state(_ctx.internal_state, session_id=_session_id)

    yield _sse("done", full)


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


# ---------- entry point ----------

def main() -> None:
    import uvicorn
    uvicorn.run("web:app", host="127.0.0.1", port=8000, reload=False, app_dir=str(Path(__file__).parent))


if __name__ == "__main__":
    main()
