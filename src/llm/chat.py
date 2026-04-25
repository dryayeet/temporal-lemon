"""Chat-completion call: prompt caching and streaming."""
import json
import time
from typing import Iterable, Iterator, Optional

import requests

import config
from config import OPENROUTER_HEADERS, OPENROUTER_URL
from logging_setup import get_logger, preview
from prompts import PERSONA_TAG  # marks the persona system block as cacheable

log = get_logger("llm.chat")


def _wrap_for_cache(content: str) -> list[dict]:
    """Anthropic-style structured content with an ephemeral cache breakpoint."""
    return [{
        "type": "text",
        "text": content,
        "cache_control": {"type": "ephemeral"},
    }]


def prepare_messages(history: list[dict]) -> list[dict]:
    """Convert the working history into the wire format.

    If prompt caching is enabled, mark the persona system block as cacheable.
    Other messages stay as plain strings.
    """
    if not config.ENABLE_PROMPT_CACHE:
        return history

    out = []
    for msg in history:
        if msg["role"] == "system" and PERSONA_TAG in msg["content"]:
            out.append({"role": "system", "content": _wrap_for_cache(msg["content"])})
        else:
            out.append(msg)
    return out


def _summarize_history(messages: list[dict]) -> str:
    """Return a compact 'sys=N user=N asst=N total_chars=N' summary string."""
    by_role = {"system": 0, "user": 0, "assistant": 0}
    total_chars = 0
    for m in messages:
        by_role[m["role"]] = by_role.get(m["role"], 0) + 1
        c = m["content"]
        if isinstance(c, list):
            total_chars += sum(len(b.get("text", "")) for b in c if isinstance(b, dict))
        else:
            total_chars += len(c)
    return (
        f"sys={by_role.get('system', 0)} "
        f"user={by_role.get('user', 0)} "
        f"asst={by_role.get('assistant', 0)} "
        f"total_chars={total_chars}"
    )


def iter_chat(history: list[dict], model: Optional[str] = None) -> Iterator[str]:
    """Yield content deltas from the chat model. No pacing, no printing.

    Both the CLI and the web UI consume this. Errors propagate as RuntimeError
    so callers can roll back the user message.
    """
    wire_messages = prepare_messages(history)
    payload = {
        "model": model or config.CHAT_MODEL,
        "temperature": 0.75,
        "top_p": 0.95,
        "frequency_penalty": 0.2,
        "max_tokens": 400,
        "messages": wire_messages,
        "stream": True,
    }

    log.info(
        "event=chat_request model=%s %s temp=%.2f top_p=%.2f freq_pen=%.2f "
        "max_tokens=%d cache=%s stream=True",
        payload["model"], _summarize_history(wire_messages),
        payload["temperature"], payload["top_p"], payload["frequency_penalty"],
        payload["max_tokens"], config.ENABLE_PROMPT_CACHE,
    )
    log.debug("event=chat_request_body payload=%s", json.dumps(payload)[:4000])

    started = time.time()
    with requests.post(
        OPENROUTER_URL,
        headers=OPENROUTER_HEADERS,
        json=payload,
        stream=True,
        timeout=60,
    ) as response:
        if response.status_code >= 400:
            body = response.text[:500]
            log.error(
                "event=chat_http_error status=%d elapsed_ms=%d body=%s",
                response.status_code, int((time.time() - started) * 1000), body,
            )
            raise RuntimeError(f"HTTP {response.status_code}: {body}")

        log.info("event=chat_response_open status=%d", response.status_code)

        chunks: list[str] = []
        try:
            for delta in _iter_sse_deltas(response.iter_lines(decode_unicode=True)):
                chunks.append(delta)
                yield delta
        finally:
            elapsed_ms = int((time.time() - started) * 1000)
            full = "".join(chunks)
            log.info(
                "event=chat_response_done chars=%d chunks=%d elapsed_ms=%d preview=%r",
                len(full), len(chunks), elapsed_ms, preview(full, 80),
            )
            log.debug("event=chat_response_full content=%s", full)


def generate_reply(history: list[dict], model: Optional[str] = None) -> str:
    """Buffered (non-streamed) generation. Used by the empathy pipeline's draft phase."""
    return "".join(iter_chat(history, model=model))


def _iter_sse_deltas(lines: Iterable[str]) -> Iterator[str]:
    """Parse OpenRouter SSE lines and yield content deltas as strings."""
    for raw_line in lines:
        if not raw_line or not raw_line.startswith("data:"):
            continue
        data = raw_line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        delta = event.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            yield delta
