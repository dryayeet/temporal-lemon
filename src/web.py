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

from commands import ChatContext, dispatch, is_command
from config import CHAT_MODEL, KEEP_RECENT_TURNS
from pipeline import recent_messages_for_context, run_empathy_turn
from prompts import LEMON_OPENERS
from session_context import initial_history, refresh_base_blocks, run_bookkeeping
from storage import db
from storage.state import fresh_session_state, save_state

app = FastAPI(title="lemon")

# ---------- single-process session state ----------
# This server is meant for the user themselves — one chat at a time.
# A lock guards the shared ChatContext from interleaving across requests.
_lock = Lock()
_session_start = datetime.now()
_session_id = db.start_session()
_internal_state = fresh_session_state()
save_state(_internal_state, session_id=_session_id)

_ctx = ChatContext(
    history=initial_history(_internal_state, _session_start),
    internal_state=_internal_state,
    chat_model=CHAT_MODEL,
    session_id=_session_id,
)

# greet on startup so the UI has something to render on first paint
_first = random.choice(LEMON_OPENERS)
_ctx.history.append({"role": "assistant", "content": _first})
db.log_message(_session_id, "assistant", _first)

# The HTML template is static; read it once at import instead of on every GET.
_INDEX_HTML = (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")


# ---------- request models ----------

class ChatRequest(BaseModel):
    message: str


class CommandRequest(BaseModel):
    text: str


# ---------- index ----------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML)


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
        base_history = refresh_base_blocks(_ctx.history, _ctx.internal_state, _session_start)
        model = _ctx.chat_model

    try:
        # yield the first phase immediately so the UI has something to render
        # while run_empathy_turn blocks. Pipeline phases are buffered in
        # phase_queue and flushed below, deduping consecutive repeats so we
        # don't re-emit "reading you" when the pipeline happens to start there.
        yield _sse("phase", "reading you")
        last_phase = "reading you"

        reply, trace = run_empathy_turn(
            user_msg=user_msg,
            base_history=base_history,
            model=model,
            session_id=_session_id,
            keep_recent_turns=KEEP_RECENT_TURNS,
            on_phase=push_phase,
        )

        for phase in phase_queue:
            if phase == last_phase:
                continue
            yield _sse("phase", phase)
            last_phase = phase

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
        target=run_bookkeeping,
        args=(
            _ctx, _session_id, user_msg, reply, trace,
            recent_snapshot, state_snapshot, model_snapshot, _lock,
        ),
        daemon=True,
    ).start()


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
