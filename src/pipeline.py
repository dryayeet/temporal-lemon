"""Empathy pipeline orchestrator.

Per turn:
    classify emotion → retrieve relevant memories → theory-of-mind side pass
    → inject system blocks → generate draft → sentiment-mirror check
    → regenerate-once on failure → return final reply + trace.

The CLI and web entry points both call `run_empathy_turn`. When the empathy
pipeline is disabled (`config.ENABLE_EMPATHY_PIPELINE = False`), this becomes
a thin wrapper around `chat.generate_reply`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import config
import db
from chat import generate_reply
from emotion import EMOTION_TAG, classify_emotion, format_emotion_block
from empathy_check import CheckResult, check_response
from history import compress_history
from memory import MEMORY_TAG, format_memory_block, relevant_memories
from tom import TOM_TAG, format_tom_block, theory_of_mind

CRITIQUE_TAG = "<empathy_retry>"

# Phase labels used by the SSE relay so the web UI can show "lemon is reading
# you...", "...thinking...", etc. CLI ignores them by default.
PHASE_READING = "reading you"
PHASE_REMEMBERING = "remembering"
PHASE_THINKING = "thinking"
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


def _recent_messages_for_context(history: list[dict], n: int = 6) -> list[dict]:
    """Pull the last n non-system messages out of `history` to give classifiers context."""
    convo = [m for m in history if m["role"] != "system"]
    return convo[-n:]


def run_empathy_turn(
    user_msg: str,
    base_history: list[dict],
    energy: str = "medium",
    model: Optional[str] = None,
    session_id: Optional[int] = None,
    keep_recent_turns: Optional[int] = None,
    on_phase: Optional[Callable[[str], None]] = None,
) -> tuple[str, PipelineTrace]:
    """Run the empathy pipeline for one user message. Returns (final_reply, trace).

    `base_history` should contain the refreshed time/state/facts system blocks
    AND the prior conversation. It must NOT yet include `user_msg` — the pipeline
    appends it after running the pre-generation passes.

    The pipeline does not mutate `base_history`. Caller appends user_msg and the
    final reply to its own history after this returns.

    `on_phase` is an optional callback for surfacing progress (used by web SSE).
    """
    trace = PipelineTrace()
    keep_recent_turns = keep_recent_turns or config.KEEP_RECENT_TURNS

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
        return reply, trace

    trace.pipeline_used = True
    recent = _recent_messages_for_context(base_history)

    # ---------- 1. classify emotion ----------
    if on_phase:
        on_phase(PHASE_READING)
    emotion = classify_emotion(user_msg, recent_msgs=recent, model=model)
    trace.emotion = emotion

    # log the user message NOW with its emotion fields
    if session_id is not None:
        db.log_message(
            session_id, "user", user_msg,
            emotion=emotion.get("primary"),
            intensity=emotion.get("intensity"),
            salience=emotion.get("intensity"),
        )

    # ---------- 2. retrieve memories ----------
    if on_phase:
        on_phase(PHASE_REMEMBERING)
    memories = relevant_memories(
        emotion=emotion.get("primary", "neutral"),
        current_session_id=session_id,
        limit=config.MEMORY_RETRIEVAL_LIMIT,
    )
    trace.memories = memories

    # ---------- 3. ToM pass ----------
    if on_phase:
        on_phase(PHASE_THINKING)
    tom = theory_of_mind(user_msg, emotion=emotion, recent_msgs=recent, model=model)
    trace.tom = tom

    # ---------- 4. inject blocks + draft ----------
    history = list(base_history)
    if memories:
        history = _inject_block(history, MEMORY_TAG, format_memory_block(memories))
    history = _inject_block(history, EMOTION_TAG, format_emotion_block(emotion))
    history = _inject_block(history, TOM_TAG, format_tom_block(tom))

    history.append({"role": "user", "content": user_msg})
    history = compress_history(history, keep_recent=keep_recent_turns)

    if on_phase:
        on_phase(PHASE_REPLYING)
    draft = generate_reply(history, model=model)
    trace.draft = draft

    # ---------- 5. post-check ----------
    check = check_response(user_msg, draft, emotion)
    trace.check = check

    final = draft
    if not check.passed and config.EMPATHY_RETRY_ON_FAIL:
        if on_phase:
            on_phase(PHASE_REVISING)
        retry_history = _inject_block(history, CRITIQUE_TAG, _critique_block(check, draft))
        try:
            second = generate_reply(retry_history, model=model)
            if second.strip():
                final = second
                trace.regenerated = True
        except Exception as e:
            print(f"  [empathy retry failed: {e}]")

    if session_id is not None:
        db.log_message(session_id, "assistant", final)
    trace.final = final
    return final, trace


def _critique_block(check: CheckResult, draft: str) -> str:
    """Wrap a CheckResult critique as a system block for the regenerate call."""
    snippet = draft[:200] + ("..." if len(draft) > 200 else "")
    return f"""
<empathy_retry>
You just produced this draft: "{snippet}"

{check.critique}
</empathy_retry>
""".strip()
