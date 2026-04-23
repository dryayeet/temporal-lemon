"""Slash-command dispatcher used by both the CLI and web frontends.

A command takes a `ChatContext` and returns a `CommandResult`. Commands can
mutate context (`reset`, `rewind`, `model`) or just read it (`state`, `facts`).
The CLI prints `result.output`; the web UI sends it as a system bubble.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import config
import db
import state as state_mod


@dataclass
class ChatContext:
    """Mutable state shared with the slash-command handlers."""
    history: list[dict] = field(default_factory=list)
    internal_state: dict = field(default_factory=dict)
    chat_model: str = ""
    session_id: Optional[int] = None
    exit_requested: bool = False
    last_trace: Optional[object] = None   # most recent PipelineTrace, for /why


@dataclass
class CommandResult:
    output: str
    # the dispatcher may signal that the host loop should fully reset its
    # working memory, e.g. after `/reset`
    reload_state: bool = False


CommandFn = Callable[[ChatContext, str], CommandResult]
_REGISTRY: dict[str, tuple[CommandFn, str]] = {}


def command(name: str, help_text: str) -> Callable[[CommandFn], CommandFn]:
    def decorator(fn: CommandFn) -> CommandFn:
        _REGISTRY[name] = (fn, help_text)
        return fn
    return decorator


def is_command(text: str) -> bool:
    return text.startswith("/") and len(text) > 1


def dispatch(text: str, ctx: ChatContext) -> CommandResult:
    """Resolve and run a slash command. Unknown commands return a help nudge."""
    parts = text[1:].split(maxsplit=1)
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    entry = _REGISTRY.get(name)
    if entry is None:
        return CommandResult(f"unknown command: /{name}. try /help.")
    fn, _ = entry
    return fn(ctx, args)


# ---------- handlers ----------

@command("help", "show this list of commands")
def _help(ctx: ChatContext, args: str) -> CommandResult:
    lines = ["available commands:"]
    for name, (_, htext) in sorted(_REGISTRY.items()):
        lines.append(f"  /{name:<12} {htext}")
    return CommandResult("\n".join(lines))


@command("state", "show lemon's current internal state")
def _state(ctx: ChatContext, args: str) -> CommandResult:
    return CommandResult(json.dumps(ctx.internal_state, indent=2))


@command("reset", "reset internal state to defaults (does not erase facts or history)")
def _reset(ctx: ChatContext, args: str) -> CommandResult:
    ctx.internal_state = dict(state_mod.DEFAULT_STATE)
    state_mod.save_state(ctx.internal_state, session_id=ctx.session_id)
    return CommandResult("internal state reset to defaults.", reload_state=True)


@command("facts", "list everything lemon remembers about you")
def _facts(ctx: ChatContext, args: str) -> CommandResult:
    facts = db.get_facts()
    if not facts:
        return CommandResult("(no facts stored yet)")
    return CommandResult("\n".join(f"  {k} = {v}" for k, v in facts.items()))


@command("remember", "store a fact: /remember key=value")
def _remember(ctx: ChatContext, args: str) -> CommandResult:
    if "=" not in args:
        return CommandResult("usage: /remember key=value")
    key, _, value = args.partition("=")
    key = key.strip()
    value = value.strip()
    if not key or not value:
        return CommandResult("both key and value are required")
    db.upsert_fact(key, value, source_session_id=ctx.session_id)
    return CommandResult(f"remembered: {key} = {value}")


@command("forget", "remove a fact: /forget key")
def _forget(ctx: ChatContext, args: str) -> CommandResult:
    key = args.strip()
    if not key:
        return CommandResult("usage: /forget key")
    deleted = db.delete_fact(key)
    return CommandResult(f"forgot {key}." if deleted else f"no fact named {key}.")


@command("history", "show the last N exchanges in this session (default 5)")
def _history(ctx: ChatContext, args: str) -> CommandResult:
    n = int(args) if args.strip().isdigit() else 5
    convo = [m for m in ctx.history if m["role"] != "system"]
    recent = convo[-(n * 2):] if convo else []
    if not recent:
        return CommandResult("(no exchanges yet in this session)")
    return CommandResult("\n".join(f"{m['role']}: {m['content']}" for m in recent))


@command("rewind", "drop the last exchange (your last message + lemon's reply)")
def _rewind(ctx: ChatContext, args: str) -> CommandResult:
    removed = 0
    while ctx.history and ctx.history[-1]["role"] != "system" and removed < 2:
        ctx.history.pop()
        removed += 1
    return CommandResult(f"rewound {removed} message(s).")


@command("model", "switch the chat model for this session: /model anthropic/claude-sonnet-4.6")
def _model(ctx: ChatContext, args: str) -> CommandResult:
    name = args.strip()
    if not name:
        return CommandResult(f"current model: {ctx.chat_model}")
    ctx.chat_model = name
    return CommandResult(f"chat model set to {name} for this session.")


@command("sessions", "list recent sessions stored in the db")
def _sessions(ctx: ChatContext, args: str) -> CommandResult:
    rows = db.list_sessions(limit=10)
    if not rows:
        return CommandResult("(no past sessions)")
    lines = ["recent sessions:"]
    for r in rows:
        end = r["ended_at"] or "(ongoing)"
        lines.append(f"  #{r['id']:<4} started {r['started_at']}  ended {end}  msgs={r['msg_count']}")
    return CommandResult("\n".join(lines))


@command("empathy", "toggle the empathy pipeline: /empathy on|off (no arg shows status)")
def _empathy(ctx: ChatContext, args: str) -> CommandResult:
    arg = args.strip().lower()
    if arg in ("on", "enable", "true", "1"):
        config.ENABLE_EMPATHY_PIPELINE = True
        return CommandResult("empathy pipeline: ON")
    if arg in ("off", "disable", "false", "0"):
        config.ENABLE_EMPATHY_PIPELINE = False
        return CommandResult("empathy pipeline: OFF")
    if not arg:
        status = "ON" if config.ENABLE_EMPATHY_PIPELINE else "OFF"
        return CommandResult(f"empathy pipeline: {status}")
    return CommandResult("usage: /empathy on|off")


@command("why", "show the empathy pipeline trace for the last reply")
def _why(ctx: ChatContext, args: str) -> CommandResult:
    trace = ctx.last_trace
    if trace is None:
        return CommandResult("no trace yet — send a message first.")
    if not getattr(trace, "pipeline_used", False):
        return CommandResult("empathy pipeline was off for the last reply.")

    lines = ["last reply's pipeline trace:"]
    emo = getattr(trace, "emotion", None)
    if emo:
        lines.append(
            f"  emotion: {emo.get('primary')} (intensity {emo.get('intensity', 0):.2f})"
        )
        if emo.get("undertones"):
            lines.append(f"           undertones: {', '.join(emo['undertones'])}")
        if emo.get("underlying_need"):
            lines.append(f"           underlying need: {emo['underlying_need']}")

    tom = getattr(trace, "tom", None)
    if tom:
        lines.append(f"  feeling:   {tom.get('feeling') or '(none)'}")
        lines.append(f"  avoid:     {tom.get('avoid') or '(none)'}")
        lines.append(f"  do:        {tom.get('what_helps') or '(none)'}")

    memories = getattr(trace, "memories", []) or []
    lines.append(f"  memories used: {len(memories)}")

    check = getattr(trace, "check", None)
    if check is not None:
        if check.passed:
            lines.append("  post-check: passed")
        else:
            lines.append(f"  post-check: failed ({', '.join(check.failures)})")
            if getattr(trace, "regenerated", False):
                lines.append("  -> regenerated once with critique")

    facts = getattr(trace, "facts_extracted", {}) or {}
    if facts:
        lines.append(f"  facts saved: {', '.join(f'{k}={v}' for k, v in facts.items())}")
    return CommandResult("\n".join(lines))


@command("quit", "exit the chat (alias: /exit)")
def _quit(ctx: ChatContext, args: str) -> CommandResult:
    ctx.exit_requested = True
    return CommandResult("bye.")


@command("exit", "exit the chat")
def _exit(ctx: ChatContext, args: str) -> CommandResult:
    ctx.exit_requested = True
    return CommandResult("bye.")
