"""Chat-completion call: prompt caching, streaming, humanized output pacing."""
import json
import random
import time
from typing import Iterable, Iterator, Optional

import requests

import config
from config import OPENROUTER_HEADERS, OPENROUTER_URL

# tag of the system block we want to mark cacheable. The persona prompt is
# long (~5KB) and stable across turns — perfect cache hit. Time/state blocks
# change every turn and stay uncached.
PERSONA_TAG = "<Who you are>"

ENERGY_SPEED_MULT = {
    "low": 1.45,
    "medium": 1.0,
    "high": 0.7,
}


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


def humanize_delay(token: str, energy: str) -> float:
    """Return seconds to sleep after `token`. Zero if humanizing is disabled."""
    if not config.HUMANIZE_STREAM:
        return 0.0
    mult = ENERGY_SPEED_MULT.get(energy, 1.0)
    base = config.HUMANIZE_BASE_SECONDS * mult
    delay = base * random.uniform(0.5, 1.5)
    if token and token[-1] in ".!?,":
        delay += config.HUMANIZE_PUNCT_PAUSE * mult * random.uniform(0.7, 1.3)
    return delay


def iter_chat(history: list[dict], model: Optional[str] = None) -> Iterator[str]:
    """Yield content deltas from the chat model. No pacing, no printing.

    Both the CLI and the web UI consume this. Errors propagate as RuntimeError
    so callers can roll back the user message.
    """
    payload = {
        "model": model or config.CHAT_MODEL,
        "temperature": 0.75,
        "top_p": 0.95,
        "frequency_penalty": 0.2,
        "max_tokens": 400,
        "messages": prepare_messages(history),
        "stream": True,
    }

    with requests.post(
        OPENROUTER_URL,
        headers=OPENROUTER_HEADERS,
        json=payload,
        stream=True,
        timeout=60,
    ) as response:
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:400]}")

        yield from _iter_sse_deltas(response.iter_lines(decode_unicode=True))


def stream_chat(
    history: list[dict],
    energy: str = "medium",
    model: Optional[str] = None,
) -> str:
    """CLI helper: stream `history` to stdout with humanized pacing, return the full reply."""
    print("lemon: ", end="", flush=True)
    chunks: list[str] = []
    for delta in iter_chat(history, model=model):
        print(delta, end="", flush=True)
        chunks.append(delta)
        time.sleep(humanize_delay(delta, energy))
    print("\n")
    return "".join(chunks)


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
