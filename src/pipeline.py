"""Empathy pipeline orchestrator.

Per turn:
    merged read (emotion + theory-of-mind, one LLM call) → retrieve memories
    → inject system blocks → generate draft → sentiment-mirror check
    → regenerate-once on failure → return final reply + trace.

Post-exchange bookkeeping (fact extraction + state nudge) no longer runs
inside the pipeline — callers run it in a background thread after the reply
has been delivered, so the user never waits on it. See `post_exchange.py`.

The CLI and web entry points both call `run_empathy_turn`. When the empathy
pipeline is disabled (`config.ENABLE_EMPATHY_PIPELINE = False`), this becomes
a thin wrapper around `chat.generate_reply`.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

import config
from empathy.empathy_check import CheckResult, check_response
from empathy.user_read import read_user
from llm.chat import generate_reply
from logging_setup import get_logger, preview
from prompt_stack import compress_history
from prompts import (
    CRITIQUE_TAG,
    LEMON_STATE_TAG,
    MEMORY_TAG,
    READING_TAG,
    USER_STATE_TAG,
    format_critique_block,
    format_lemon_state,
    format_memory_block,
    format_reading_block,
    format_user_state_block,
)
from storage import db
from storage.lemon_state import (
    DEFAULT_LEMON_STATE,
    apply_delta as apply_lemon_state_delta,
    save_lemon_state,
)
from storage.memory import relevant_memories
from storage.user_state import (
    DEFAULT_USER_STATE,
    apply_delta as apply_user_state_delta,
    save_user_state,
)

log = get_logger("pipeline")

# Phase labels used by the SSE relay so the web UI can show "lemon is reading
# you...", etc. CLI ignores them by default.
#
# THINKING was merged into READING when emotion+ToM collapsed into one call.
# NOTING was dropped when fact/state bookkeeping moved to a background thread
# that fires AFTER the reply is delivered.
PHASE_READING = "reading you"
PHASE_REMEMBERING = "remembering"
PHASE_REPLYING = "replying"
PHASE_REVISING = "rephrasing"


@dataclass
class PipelineTrace:
    """Snapshot of every intermediate output from one pipeline run."""
    emotion: Optional[dict] = None
    memories: list[dict] = field(default_factory=list)
    tom: Optional[dict] = None
    draft: Optional[str] = None
    check: Optional[CheckResult] = None
    regenerated: bool = False
    final: Optional[str] = None
    pipeline_used: bool = False
    facts_extracted: dict = field(default_factory=dict)
    # Dyadic-state stage 1: user_state trajectory across this turn.
    user_state_before: Optional[dict] = None
    user_state_after: Optional[dict] = None
    user_state_delta: Optional[dict] = None
    # Dyadic-state stage 2 + 3: lemon_state trajectory across this turn.
    lemon_state_before: Optional[dict] = None
    lemon_state_after: Optional[dict] = None
    lemon_state_delta: Optional[dict] = None


def _last_leading_system_index(history: list[dict]) -> int:
    """Index just after the contiguous block of system messages at the front of `history`."""
    i = 0
    while i < len(history) and history[i]["role"] == "system":
        i += 1
    return i


def _inject_block(history: list[dict], tag: str, content: str) -> list[dict]:
    """Drop any existing block with `tag`; insert `content` after the leading system block."""
    filtered = [
        m for m in history
        if not (m["role"] == "system" and tag in m["content"])
    ]
    pos = _last_leading_system_index(filtered)
    filtered.insert(pos, {"role": "system", "content": content})
    return filtered


def recent_messages_for_context(history: list[dict], n: int = 6) -> list[dict]:
    """Pull the last n non-system messages out of `history` so classifiers
    and the bookkeeping thread get the same conversational view."""
    convo = [m for m in history if m["role"] != "system"]
    return convo[-n:]


def run_empathy_turn(
    user_msg: str,
    base_history: list[dict],
    model: Optional[str] = None,
    session_id: Optional[int] = None,
    keep_recent_turns: Optional[int] = None,
    on_phase: Optional[Callable[[str], None]] = None,
    user_state: Optional[dict] = None,
    lemon_state: Optional[dict] = None,
) -> tuple[str, PipelineTrace]:
    """Run the empathy pipeline for one user message. Returns (final_reply, trace).

    `base_history` should contain the refreshed time/lemon_state/facts system
    blocks AND the prior conversation. It must NOT yet include `user_msg` —
    the pipeline appends it after running the pre-generation passes.

    The pipeline does not mutate `base_history`. Caller appends user_msg and
    the final reply to its own history after this returns.

    `on_phase` is an optional callback for surfacing progress (used by web SSE).

    `user_state` and `lemon_state` are the persistent tonic states going INTO
    this turn. The pipeline reads them, has the LLM emit per-turn deltas
    against them (in the merged user_read pass), applies and persists both,
    then conditions reply generation on the freshly-updated states. The
    trajectory is reported on the trace.
    """
    trace = PipelineTrace()
    keep_recent_turns = keep_recent_turns or config.KEEP_RECENT_TURNS
    turn_id = uuid.uuid4().hex[:8]
    turn_started = time.time()

    log.info(
        "turn_start turn=%s session=%s msg=%r pipeline=%s",
        turn_id, session_id, preview(user_msg, 60), config.ENABLE_EMPATHY_PIPELINE,
    )

    # ---------- short-circuit when pipeline disabled ----------
    if not config.ENABLE_EMPATHY_PIPELINE:
        if on_phase:
            on_phase(PHASE_REPLYING)
        history = list(base_history)
        history.append({"role": "user", "content": user_msg})
        history = compress_history(history, keep_recent=keep_recent_turns)
        if session_id is not None:
            db.log_message(session_id, "user", user_msg)
        reply = generate_reply(history, model=model)
        if session_id is not None:
            db.log_message(session_id, "assistant", reply)
        trace.final = reply
        log.info(
            "turn_done turn=%s reply_chars=%d elapsed_ms=%d pipeline=off",
            turn_id, len(reply), int((time.time() - turn_started) * 1000),
        )
        return reply, trace

    trace.pipeline_used = True
    recent = recent_messages_for_context(base_history)

    # Dyadic-state: tonic states for both agents going INTO this turn.
    user_state_before = user_state if user_state is not None else dict(DEFAULT_USER_STATE)
    lemon_state_before = lemon_state if lemon_state is not None else dict(DEFAULT_LEMON_STATE)
    trace.user_state_before = user_state_before
    trace.lemon_state_before = lemon_state_before

    # ---------- 1. merged read: emotion + ToM + user-delta + lemon-delta ----------
    if on_phase:
        on_phase(PHASE_READING)
    phase_started = time.time()
    emotion, tom, user_state_delta, lemon_state_delta = read_user(
        user_msg,
        recent_msgs=recent,
        current_user_state=user_state_before,
        current_lemon_state=lemon_state_before,
        model=model,
    )
    user_state_after = apply_user_state_delta(user_state_before, user_state_delta)
    lemon_state_after = apply_lemon_state_delta(lemon_state_before, lemon_state_delta)
    trace.emotion = emotion
    trace.tom = tom
    trace.user_state_delta = user_state_delta
    trace.user_state_after = user_state_after
    trace.lemon_state_delta = lemon_state_delta
    trace.lemon_state_after = lemon_state_after
    log.info(
        "phase=reading elapsed_ms=%d emotion=%s user_mood=%s lemon_mood=%s",
        int((time.time() - phase_started) * 1000),
        emotion.get("primary"),
        user_state_after.get("state", {}).get("mood_label"),
        lemon_state_after.get("state", {}).get("mood_label"),
    )

    # Persist both tonic states immediately. A crash during retrieval/generation
    # still preserves the read. Fire-and-forget; failures don't break the turn.
    try:
        save_user_state(user_state_after, session_id=session_id)
    except Exception as e:
        log.warning("user_state_save_failed error=%r", e)
    try:
        save_lemon_state(lemon_state_after, session_id=session_id)
    except Exception as e:
        log.warning("lemon_state_save_failed error=%r", e)

    # log the user message NOW with its emotion fields
    if session_id is not None:
        db.log_message(
            session_id, "user", user_msg,
            emotion=emotion.get("primary"),
            intensity=emotion.get("intensity"),
            salience=emotion.get("intensity"),
        )

    # ---------- 2. retrieve memories (DB only, no LLM) ----------
    if on_phase:
        on_phase(PHASE_REMEMBERING)
    phase_started = time.time()
    memories = relevant_memories(
        user_msg=user_msg,
        emotion=emotion.get("primary", "neutral"),
        intensity=emotion.get("intensity", 0.0),
        current_session_id=session_id,
        limit=config.MEMORY_RETRIEVAL_LIMIT,
    )
    log.info(
        "phase=remembering elapsed_ms=%d retrieved=%d",
        int((time.time() - phase_started) * 1000), len(memories),
    )
    trace.memories = memories

    # ---------- 3. inject blocks + draft ----------
    # Stage 3 prompt-block layout:
    #   persona (cache anchor), time_context, lemon_state (refreshed pre-reply),
    #   user_facts, [emotional_memory], <user_state>, <reading>.
    # The <lemon_state> block was refreshed in `refresh_base_blocks` from
    # lemon_state_before; we now overwrite it with the JUST-updated
    # lemon_state_after so reply generation reads the freshly-nudged state.
    history = list(base_history)
    blocks_injected = []
    history = _inject_block(history, LEMON_STATE_TAG, format_lemon_state(lemon_state_after))
    blocks_injected.append("lemon_state")
    if memories:
        history = _inject_block(history, MEMORY_TAG, format_memory_block(memories))
        blocks_injected.append("memory")
    # <user_state> (tonic) sits before <reading> (phasic) so the model reads
    # tonic-then-phasic for the user, paralleling the lemon ordering.
    history = _inject_block(history, USER_STATE_TAG, format_user_state_block(user_state_after))
    blocks_injected.append("user_state")
    history = _inject_block(history, READING_TAG, format_reading_block(emotion, tom))
    blocks_injected.append("reading")

    history.append({"role": "user", "content": user_msg})
    pre_compress_len = len(history)
    history = compress_history(history, keep_recent=keep_recent_turns)
    if len(history) != pre_compress_len:
        log.info(
            "history_compressed pre=%d post=%d", pre_compress_len, len(history),
        )

    if on_phase:
        on_phase(PHASE_REPLYING)
    phase_started = time.time()
    draft = generate_reply(history, model=model)
    log.info(
        "phase=replying elapsed_ms=%d chars=%d",
        int((time.time() - phase_started) * 1000), len(draft),
    )
    trace.draft = draft

    # ---------- 4. post-check ----------
    check = check_response(user_msg, draft, emotion)
    trace.check = check
    log.info(
        "empathy_check passed=%s failures=%d",
        check.passed, len(getattr(check, "failures", []) or []),
    )

    final = draft
    if not check.passed and config.EMPATHY_RETRY_ON_FAIL:
        if on_phase:
            on_phase(PHASE_REVISING)
        phase_started = time.time()
        retry_history = _inject_block(history, CRITIQUE_TAG, format_critique_block(draft, check.critique))
        try:
            second = generate_reply(retry_history, model=model)
            if second.strip():
                final = second
                trace.regenerated = True
                log.info(
                    "regenerated elapsed_ms=%d chars=%d",
                    int((time.time() - phase_started) * 1000), len(final),
                )
            else:
                log.warning("regenerate_empty turn=%s — kept draft", turn_id)
        except Exception as e:
            log.error("regenerate_failed turn=%s error=%r", turn_id, e)

    if session_id is not None:
        db.log_message(session_id, "assistant", final)
    trace.final = final

    log.info(
        "turn_done turn=%s chars=%d regenerated=%s elapsed_ms=%d",
        turn_id, len(final), trace.regenerated,
        int((time.time() - turn_started) * 1000),
    )

    # Post-exchange bookkeeping (facts + state nudge) runs OUTSIDE the pipeline
    # in a background thread — see web.py / lem.py. Pipeline returns now so
    # the caller can deliver the reply to the user immediately.
    return final, trace
