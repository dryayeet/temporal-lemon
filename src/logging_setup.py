"""Process-wide logging configuration.

Call `setup_logging()` once at process startup (CLI: `lem.main`; web:
import-time of `web.py`). Subsequent `get_logger(name)` calls return
standard `logging.Logger` instances under the `lemon.*` namespace.

Levels (env LEMON_LOG_LEVEL, default INFO)
------------------------------------------
INFO  — one line per pipeline phase (with timing), per HTTP request,
        per DB write, per LLM round-trip outcome.
DEBUG — additionally dumps full request/response bodies and prompts.
WARNING — fallbacks (parse failure → defaults, HTTP non-2xx, etc.).
ERROR — pipeline-breaking errors.

Optional file sink via env LEMON_LOG_FILE (path written alongside stderr).

Helpers
-------
`shape_of(obj)`   describes a structure (keys + value types) without
                  dumping the content. Use for logging "what came back"
                  without leaking 5KB of prompt text into INFO.
`preview(text)`   single-line, length-capped snippet for log lines.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-26s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """Configure the `lemon.*` logger tree. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.getenv("LEMON_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)
    handlers: list[logging.Handler] = []

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(formatter)
    handlers.append(stderr)

    log_file = os.getenv("LEMON_LOG_FILE")
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        handlers.append(fh)

    root = logging.getLogger("lemon")
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
    root.setLevel(level)
    root.propagate = False  # keep our records out of the root logger

    # Tame third-party chatter so DEBUG mode is still readable.
    for noisy in ("urllib3", "httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    root.debug("event=logging_ready level=%s file=%s", level_name, log_file or "<stderr only>")


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the `lemon.` root namespace."""
    if not name.startswith("lemon."):
        name = f"lemon.{name}"
    return logging.getLogger(name)


# ---------- payload-safe formatters ----------

def shape_of(obj: Any, max_depth: int = 2, max_items: int = 6) -> str:
    """Describe an object's structure without dumping content.

    Examples
    --------
    >>> shape_of({"a": 1, "b": [1, 2, 3]})
    '{a: int, b: list[3]}'
    >>> shape_of({"primary": "joy", "intensity": 0.4, "undertones": []})
    '{primary: str[3], intensity: float, undertones: list[0]}'
    """
    return _describe(obj, max_depth, max_items)


def preview(text: Any, max_chars: int = 100) -> str:
    """Single-line, length-capped snippet for log lines."""
    if text is None:
        return "<None>"
    s = str(text).replace("\n", "\\n").replace("\r", "")
    if len(s) > max_chars:
        return s[: max_chars - 3] + "..."
    return s


def _describe(obj: Any, depth: int, max_items: int) -> str:
    if obj is None:
        return "None"
    if depth <= 0:
        return type(obj).__name__
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = list(obj.items())[:max_items]
        body = ", ".join(f"{k}: {_describe(v, depth - 1, max_items)}" for k, v in items)
        more = f", ...+{len(obj) - max_items}" if len(obj) > max_items else ""
        return "{" + body + more + "}"
    if isinstance(obj, list):
        return f"list[{len(obj)}]"
    if isinstance(obj, tuple):
        return f"tuple[{len(obj)}]"
    if isinstance(obj, str):
        return f"str[{len(obj)}]"
    return type(obj).__name__
