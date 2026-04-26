"""Lemon web UI: FastAPI + single-page HTML, SSE streaming for chat.

Run with:
    uvicorn app.web:app --reload --app-dir src
or:
    python -m app.web   (from src/, with src on PYTHONPATH)

Interactive API docs available at:
    http://127.0.0.1:8000/docs    (Swagger UI)
    http://127.0.0.1:8000/redoc   (ReDoc)
    http://127.0.0.1:8000/openapi.json
"""
from __future__ import annotations

import json
import random
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, Optional

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from core import config
from core.config import CHAT_MODEL, KEEP_RECENT_TURNS
from core.logging_setup import get_logger, preview, setup_logging
from prompts import LEMON_OPENERS
from storage import db
from storage.lemon_state import fresh_lemon_session_state, save_lemon_state
from storage.user_state import fresh_user_session_state

from .commands import ChatContext, dispatch, is_command
from .pipeline import recent_messages_for_context, run_empathy_turn
from .session_context import initial_history, refresh_base_blocks, run_bookkeeping

# Logging must be configured before any module-level code that may emit logs.
setup_logging()
log = get_logger("web")


# ---------- FastAPI metadata (drives /docs and /openapi.json) ----------

API_DESCRIPTION = """
**lemon** is a single-user empathetic chat companion built on a per-turn
empathy pipeline (emotion + theory-of-mind read → memory retrieval →
draft → empathy check → optional regeneration). Facts and internal
state are persisted to SQLite and updated in a background thread after
each reply.

This API powers the bundled web UI. Endpoints are grouped as:

* **chat** — the streaming chat endpoint and slash-command dispatcher
* **introspection** — read-only views into state, facts, sessions, history, last pipeline trace
* **health** — liveness and readiness probes
"""

API_TAGS = [
    {"name": "chat", "description": "Send messages and run slash commands."},
    {"name": "introspection", "description": "Read-only views into lemon's state and history."},
    {"name": "health", "description": "Liveness and readiness probes."},
]

app = FastAPI(
    title="lemon",
    description=API_DESCRIPTION,
    version="0.2.0",
    openapi_tags=API_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------- request-id + timing middleware ----------

@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    rid = uuid.uuid4().hex[:8]
    started = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        elapsed_ms = int((time.time() - started) * 1000)
        log.error(
            "http_unhandled %s %s error=%r elapsed_ms=%d",
            request.method, request.url.path, e, elapsed_ms,
        )
        raise
    elapsed_ms = int((time.time() - started) * 1000)
    log.info(
        "http %s %s -> %d %dms",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    response.headers["X-Request-ID"] = rid
    return response


# ---------- single-process session state ----------
# This server is meant for the user themselves — one chat at a time.
# A lock guards the shared ChatContext from interleaving across requests.
_lock = Lock()
_session_start = datetime.now()
_session_id = db.start_session()
_lemon_state = fresh_lemon_session_state()
save_lemon_state(_lemon_state, session_id=_session_id)
_user_state = fresh_user_session_state()

_ctx = ChatContext(
    history=initial_history(_lemon_state, _session_start),
    lemon_state=_lemon_state,
    chat_model=CHAT_MODEL,
    session_id=_session_id,
    user_state=_user_state,
)

# greet on startup so the UI has something to render on first paint
_first = random.choice(LEMON_OPENERS)
_ctx.history.append({"role": "assistant", "content": _first})
db.log_message(_session_id, "assistant", _first)
log.info(
    "server_startup session=%s chat_model=%s state_model=%s",
    _session_id, config.CHAT_MODEL, config.STATE_MODEL,
)

# Templates and static assets sit at src/ alongside the package folders, so
# they're one level UP from app/. The HTML template is read once at import.
_SRC_ROOT = Path(__file__).resolve().parent.parent
_INDEX_HTML = (_SRC_ROOT / "templates" / "index.html").read_text(encoding="utf-8")

# Brand asset (favicon + on-page logo). Loaded once into memory so we don't
# need aiofiles/StaticFiles. ~940KB; trivial for a single-user local app.
_LEMON_PNG_BYTES = (_SRC_ROOT / "static" / "lemon.png").read_bytes()


# ---------- request / response models ----------

class ChatRequest(BaseModel):
    """Body for POST /chat. `message` must be non-empty and not a slash command."""
    message: str = Field(..., description="User's message text", min_length=1, examples=["hey, how was your day?"])


class CommandRequest(BaseModel):
    """Body for POST /command. `text` must start with `/`."""
    text: str = Field(..., description="Slash command, e.g. `/help` or `/facts`", examples=["/help"])


class CommandResponse(BaseModel):
    output: str = Field(..., description="Human-readable result of the command")
    exit: bool = Field(..., description="True when the command requested session shutdown")


class PingResponse(BaseModel):
    pong: bool = Field(True, description="Always true on a live server")


class HealthDB(BaseModel):
    ok: bool = Field(..., description="True if SELECT 1 succeeded")
    error: Optional[str] = Field(None, description="Stringified error if `ok` is False")


class HealthConfig(BaseModel):
    chat_model: str
    state_model: str
    empathy_pipeline: bool
    empathy_retry_on_fail: bool
    auto_facts: bool
    auto_facts_max_per_turn: int
    prompt_cache: bool
    memory_retrieval_limit: int
    keep_recent_turns: int


class HealthResponse(BaseModel):
    status: str = Field(..., description="`ok` if all checks passed, `degraded` otherwise")
    uptime_seconds: int = Field(..., description="Seconds since the server process started")
    session_id: int = Field(..., description="The single chat session this server holds open")
    db: HealthDB
    config: HealthConfig


# ---------- index ----------

@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["chat"],
    summary="Bundled chat UI",
    description="Serves the single-page HTML chat client.",
)
def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML)


# ---------- brand asset (favicon + on-page logo) ----------

def _png_response() -> Response:
    return Response(
        content=_LEMON_PNG_BYTES,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return _png_response()


@app.get("/static/lemon.png", include_in_schema=False)
def lemon_png() -> Response:
    return _png_response()


# ---------- chat (SSE) ----------

@app.post(
    "/chat",
    tags=["chat"],
    summary="Stream a chat reply (SSE)",
    description=(
        "Runs the user message through the empathy pipeline and streams the "
        "result as Server-Sent Events. Event types: `phase` (pipeline phase "
        "name), `token` (the full reply, sent in one event), `done` (final "
        "marker), `error` (on transport/HTTP failure). Each event payload is "
        "a JSON object `{event, data}`."
    ),
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "SSE stream"},
        400: {"description": "Empty message or slash-command sent to /chat"},
    },
)
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
        base_history = refresh_base_blocks(_ctx.history, _ctx.lemon_state, _session_start)
        model = _ctx.chat_model
        user_state_in = dict(_ctx.user_state) if _ctx.user_state else None
        lemon_state_in = dict(_ctx.lemon_state) if _ctx.lemon_state else None

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
            user_state=user_state_in,
            lemon_state=lemon_state_in,
        )

        for phase in phase_queue:
            if phase == last_phase:
                continue
            yield _sse("phase", phase)
            last_phase = phase

    except (requests.RequestException, RuntimeError) as e:
        log.error("chat_stream_error error=%r", e)
        yield _sse("error", str(e))
        return

    with _lock:
        _ctx.last_trace = trace
        _ctx.history.append({"role": "user", "content": user_msg})
        _ctx.history.append({"role": "assistant", "content": reply})
        # Pull both freshly-updated tonic states off the trace; pipeline
        # already persisted them. Next turn reads from these values.
        if getattr(trace, "user_state_after", None) is not None:
            _ctx.user_state = trace.user_state_after
        if getattr(trace, "lemon_state_after", None) is not None:
            _ctx.lemon_state = trace.lemon_state_after
        # Snapshot inputs the bookkeeping thread will need (facts-only now,
        # so just recent_msgs and the model name).
        recent_snapshot = recent_messages_for_context(_ctx.history)
        model_snapshot = _ctx.chat_model

    yield _sse("token", reply)
    yield _sse("done", reply)

    threading.Thread(
        target=run_bookkeeping,
        args=(
            _ctx, _session_id, user_msg, reply, trace,
            recent_snapshot, model_snapshot, _lock,
        ),
        daemon=True,
    ).start()


def _sse(event: str, data: str) -> bytes:
    """Encode an SSE event. Data is JSON-wrapped so newlines/quotes are safe."""
    payload = json.dumps({"event": event, "data": data})
    return f"data: {payload}\n\n".encode("utf-8")


# ---------- slash commands ----------

@app.post(
    "/command",
    response_model=CommandResponse,
    tags=["chat"],
    summary="Run a slash command",
    description="Dispatches a slash command (e.g. `/help`, `/facts`, `/reset`). Does not call the LLM.",
)
def command(req: CommandRequest) -> JSONResponse:
    text = req.text.strip()
    if not is_command(text):
        raise HTTPException(400, "not a slash command")
    with _lock:
        result = dispatch(text, _ctx)
    log.info("command cmd=%r exit=%s", preview(text, 40), _ctx.exit_requested)
    return JSONResponse({"output": result.output, "exit": _ctx.exit_requested})


# ---------- introspection ----------

@app.get(
    "/state",
    tags=["introspection"],
    summary="Lemon's current internal state (three-layer schema)",
    description=(
        "Lemon's three-layer tonic state: traits (Big 5), characteristic "
        "adaptations (goals/values/concerns/stance), and PAD core affect "
        "(pleasure / arousal / dominance plus a derived mood label). Updated "
        "each turn from the empathy pipeline's lemon_state delta."
    ),
)
def get_state() -> JSONResponse:
    return JSONResponse(_ctx.lemon_state or {})


@app.get(
    "/user_state",
    tags=["introspection"],
    summary="The user's inferred persistent state (dyadic-state stage 1)",
    description=(
        "The three-layer user state: traits (Big 5), characteristic adaptations "
        "(goals/values/concerns/stance), and PAD core affect (pleasure / arousal "
        "/ dominance plus a derived mood label). Updated each turn from the "
        "empathy pipeline's user_state delta."
    ),
)
def get_user_state() -> JSONResponse:
    return JSONResponse(_ctx.user_state or {})


@app.get(
    "/facts",
    tags=["introspection"],
    summary="All stored user facts",
    description="Key/value map of everything lemon has remembered about the user. Read-only.",
)
def get_facts_endpoint() -> JSONResponse:
    return JSONResponse(db.get_facts())


@app.get(
    "/sessions",
    tags=["introspection"],
    summary="Recent sessions (latest 20)",
    description="One row per chat session, with start/end timestamps and message count.",
)
def get_sessions() -> JSONResponse:
    return JSONResponse(db.list_sessions(limit=20))


@app.get(
    "/history",
    tags=["introspection"],
    summary="Conversation history (this session)",
    description="The current session's user/assistant turns. System blocks are excluded.",
)
def get_history() -> JSONResponse:
    convo = [
        {"role": m["role"], "content": m["content"]}
        for m in _ctx.history if m["role"] != "system"
    ]
    return JSONResponse(convo)


@app.get(
    "/trace",
    tags=["introspection"],
    summary="Most recent pipeline trace",
    description=(
        "Returns the intermediate outputs from the last empathy-pipeline run: "
        "emotion read, theory-of-mind read, retrieved memories, empathy-check "
        "result, regeneration flag, and any auto-extracted facts."
    ),
)
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


# ---------- health / liveness ----------

_PROCESS_START = time.time()


@app.get(
    "/ping",
    response_model=PingResponse,
    tags=["health"],
    summary="Liveness probe",
    description="Trivial liveness check — returns immediately with `{pong: true}`. Use for load-balancer / k8s liveness.",
)
def ping() -> PingResponse:
    return PingResponse(pong=True)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Readiness probe",
    description=(
        "Verifies SQLite connectivity and reports the active config. "
        "Returns `status=ok` when the DB responds to `SELECT 1`, "
        "`status=degraded` otherwise (still HTTP 200 — inspect the body)."
    ),
)
def health() -> HealthResponse:
    db_ok = True
    db_error: Optional[str] = None
    try:
        with db.connect() as c:
            c.execute("SELECT 1").fetchone()
    except Exception as e:
        db_ok = False
        db_error = repr(e)
        log.warning("health_db_failed error=%s", db_error)

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        uptime_seconds=int(time.time() - _PROCESS_START),
        session_id=_session_id,
        db=HealthDB(ok=db_ok, error=db_error),
        config=HealthConfig(
            chat_model=config.CHAT_MODEL,
            state_model=config.STATE_MODEL,
            empathy_pipeline=config.ENABLE_EMPATHY_PIPELINE,
            empathy_retry_on_fail=config.EMPATHY_RETRY_ON_FAIL,
            auto_facts=config.ENABLE_AUTO_FACTS,
            auto_facts_max_per_turn=config.AUTO_FACTS_MAX_PER_TURN,
            prompt_cache=config.ENABLE_PROMPT_CACHE,
            memory_retrieval_limit=config.MEMORY_RETRIEVAL_LIMIT,
            keep_recent_turns=config.KEEP_RECENT_TURNS,
        ),
    )


# ---------- entry point ----------

def main() -> None:
    import uvicorn
    uvicorn.run(
        "app.web:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        app_dir=str(_SRC_ROOT),
    )


if __name__ == "__main__":
    main()
