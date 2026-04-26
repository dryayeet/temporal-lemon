"""Slash-command dispatcher used by both the CLI and web frontends.

A command takes a `ChatContext` and returns a `CommandResult`. Commands can
mutate context (`reset`, `rewind`, `model`) or just read it (`state`, `facts`).
The CLI prints `result.output`; the web UI sends it as a system bubble.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from core import config
from storage import db
from storage import lemon_state as lemon_state_mod


@dataclass
class ChatContext:
    """Mutable state shared with the slash-command handlers."""
    history: list[dict] = field(default_factory=list)
    chat_model: str = ""
    session_id: Optional[int] = None
    exit_requested: bool = False
    last_trace: Optional[object] = None   # most recent PipelineTrace, for /why
    # Dyadic-state: both agents have a three-layer tonic state object. Updated
    # each turn from trace.{user,lemon}_state_after by the pipeline.
    user_state: dict = field(default_factory=dict)
    lemon_state: dict = field(default_factory=dict)


@dataclass
class CommandResult:
    output: str
    # the dispatcher may signal that the host loop should fully reset its
    # working memory, e.g. after `/reset`
    reload_state: bool = False


CommandFn = Callable[[ChatContext, str], CommandResult]
_REGISTRY: dict[str, tuple[CommandFn, str]] = {}


def _render_values(values) -> str:
    """Render a list of value entries (tagged dicts or plain strings) as
    a comma-separated string. Tagged: 'label (tag)'; untagged: 'label'."""
    bits = []
    for v in values or []:
        if isinstance(v, dict):
            label = v.get("label") or ""
            tag = v.get("schwartz")
            bits.append(f"{label} ({tag})" if tag else label)
        elif isinstance(v, str):
            bits.append(v)
    return ", ".join(b for b in bits if b)


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


@command("state", "show lemon's current internal state (traits / adaptations / PAD)")
def _state(ctx: ChatContext, args: str) -> CommandResult:
    if not ctx.lemon_state:
        return CommandResult("(no lemon_state yet)")
    s = ctx.lemon_state.get("state", {})
    traits = ctx.lemon_state.get("traits", {})
    adapt = ctx.lemon_state.get("adaptations", {})
    lines = [
        "lemon_state:",
        f"  mood: {s.get('mood_label', 'neutral')}",
        f"  PAD: pleasure {float(s.get('pleasure', 0.0)):+.2f}, "
        f"arousal {float(s.get('arousal', 0.0)):+.2f}, "
        f"dominance {float(s.get('dominance', 0.0)):+.2f}",
        "  traits:",
    ]
    for k in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        lines.append(f"    {k:<18} {float(traits.get(k, 0.0)):+.2f}")
    lines.append("  adaptations:")
    lines.append(f"    goals:    {', '.join(adapt.get('current_goals') or []) or '(none)'}")
    lines.append(f"    values:   {_render_values(adapt.get('values') or []) or '(none)'}")
    lines.append(f"    concerns: {', '.join(adapt.get('concerns') or []) or '(none)'}")
    lines.append(f"    stance:   {adapt.get('relational_stance') or '(none)'}")
    return CommandResult("\n".join(lines))


@command("user_state", "show the user's inferred persistent state (traits / adaptations / PAD)")
def _user_state(ctx: ChatContext, args: str) -> CommandResult:
    if not ctx.user_state:
        return CommandResult("(no user_state yet — first read of this person)")
    s = ctx.user_state.get("state", {})
    traits = ctx.user_state.get("traits", {})
    adapt = ctx.user_state.get("adaptations", {})
    lines = [
        "user_state:",
        f"  mood: {s.get('mood_label', 'neutral')}",
        f"  PAD: pleasure {float(s.get('pleasure', 0.0)):+.2f}, "
        f"arousal {float(s.get('arousal', 0.0)):+.2f}, "
        f"dominance {float(s.get('dominance', 0.0)):+.2f}",
        "  traits:",
    ]
    for k in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
        lines.append(f"    {k:<18} {float(traits.get(k, 0.0)):+.2f}")
    lines.append("  adaptations:")
    lines.append(f"    goals:    {', '.join(adapt.get('current_goals') or []) or '(none)'}")
    lines.append(f"    values:   {_render_values(adapt.get('values') or []) or '(none)'}")
    lines.append(f"    concerns: {', '.join(adapt.get('concerns') or []) or '(none)'}")
    lines.append(f"    stance:   {adapt.get('relational_stance') or '(none)'}")
    return CommandResult("\n".join(lines))


@command("reset", "reset lemon's state to persona defaults (does not erase facts or history)")
def _reset(ctx: ChatContext, args: str) -> CommandResult:
    import copy
    ctx.lemon_state = copy.deepcopy(lemon_state_mod.DEFAULT_LEMON_STATE)
    lemon_state_mod.save_lemon_state(ctx.lemon_state, session_id=ctx.session_id)
    return CommandResult("lemon's state reset to persona defaults.", reload_state=True)


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

    # Dyadic-state: surface trajectory for both agents.
    def _summarize_trajectory(label: str, before, after, delta):
        if before is None or after is None:
            return
        before_mood = before.get("state", {}).get("mood_label", "?")
        after_mood = after.get("state", {}).get("mood_label", "?")
        if before_mood != after_mood:
            lines.append(f"  {label} mood: {before_mood} -> {after_mood}")
        else:
            lines.append(f"  {label} mood: {after_mood}")
        if not delta:
            return
        pad = delta.get("pad") or {}
        nudges = []
        for k in ("pleasure", "arousal", "dominance"):
            v = float(pad.get(k, 0.0))
            if abs(v) > 0.005:
                nudges.append(f"{k[0]}{v:+.2f}")
        if nudges:
            lines.append(f"    PAD nudge: {' '.join(nudges)}")
        adds = (delta.get("goal_add") or []) + (delta.get("concern_add") or [])
        if adds:
            lines.append(f"    added: {', '.join(adds)}")
        removes = (delta.get("goal_remove") or []) + (delta.get("concern_remove") or [])
        if removes:
            lines.append(f"    resolved: {', '.join(removes)}")

    _summarize_trajectory(
        "user",
        getattr(trace, "user_state_before", None),
        getattr(trace, "user_state_after", None),
        getattr(trace, "user_state_delta", None),
    )
    _summarize_trajectory(
        "lemon",
        getattr(trace, "lemon_state_before", None),
        getattr(trace, "lemon_state_after", None),
        getattr(trace, "lemon_state_delta", None),
    )

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


@command("search", "search past messages: /search <query>")
def _search(ctx: ChatContext, args: str) -> CommandResult:
    q = args.strip()
    if not q:
        return CommandResult("usage: /search <query>")
    rows = db.find_messages_by_fts(fts_query=q, candidate_pool=10)
    if not rows:
        return CommandResult(f"(no matches for '{q}')")
    lines = [f"matches for '{q}':"]
    for r in rows[:8]:
        when = (r.get("created_at") or "")[:16]
        snippet = (r.get("content") or "").replace("\n", " ")
        if len(snippet) > 100:
            snippet = snippet[:97] + "..."
        lines.append(f"  #{r['session_id']:<3} {when}  {snippet}")
    return CommandResult("\n".join(lines))


@command("recall", "show past messages tagged with an emotion: /recall <emotion>")
def _recall(ctx: ChatContext, args: str) -> CommandResult:
    emotion = args.strip().lower()
    if not emotion:
        return CommandResult("usage: /recall <emotion> (e.g. joy, sadness, anxiety)")
    rows = db.find_messages_by_emotion(emotion, exclude_session_id=ctx.session_id, limit=8)
    if not rows:
        return CommandResult(f"(no past messages tagged '{emotion}')")
    lines = [f"past messages tagged '{emotion}':"]
    for r in rows:
        when = (r.get("created_at") or "")[:16]
        intensity = r.get("intensity")
        intensity_str = f" ({intensity:.2f})" if intensity is not None else ""
        snippet = (r.get("content") or "").replace("\n", " ")
        if len(snippet) > 100:
            snippet = snippet[:97] + "..."
        lines.append(f"  {when}{intensity_str}  {snippet}")
    return CommandResult("\n".join(lines))


@command("stats", "show counts: messages, facts, sessions")
def _stats(ctx: ChatContext, args: str) -> CommandResult:
    with db.connect() as c:
        total_msgs = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        total_facts = c.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        total_sessions = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        this_session_msgs = c.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (ctx.session_id,)
        ).fetchone()[0]
    return CommandResult(
        "stats:\n"
        f"  this session: {this_session_msgs} messages\n"
        f"  all sessions: {total_sessions}\n"
        f"  all messages: {total_msgs}\n"
        f"  facts stored: {total_facts}"
    )


@command("config", "show the current behaviour flags")
def _config(ctx: ChatContext, args: str) -> CommandResult:
    return CommandResult(
        "config:\n"
        f"  chat model:        {ctx.chat_model}\n"
        f"  state model:       {config.STATE_MODEL}\n"
        f"  empathy pipeline:  {'on' if config.ENABLE_EMPATHY_PIPELINE else 'off'}\n"
        f"  empathy retry:     {'on' if config.EMPATHY_RETRY_ON_FAIL else 'off'}\n"
        f"  auto facts:        {'on' if config.ENABLE_AUTO_FACTS else 'off'} (max {config.AUTO_FACTS_MAX_PER_TURN}/turn)\n"
        f"  prompt cache:      {'on' if config.ENABLE_PROMPT_CACHE else 'off'}\n"
        f"  memory retrieval:  top {config.MEMORY_RETRIEVAL_LIMIT}\n"
        f"  keep recent turns: {config.KEEP_RECENT_TURNS}"
    )


@command("clear", "drop visible chat history this session (db is untouched)")
def _clear(ctx: ChatContext, args: str) -> CommandResult:
    before = len(ctx.history)
    ctx.history = [m for m in ctx.history if m["role"] == "system"]
    dropped = before - len(ctx.history)
    return CommandResult(f"cleared {dropped} message(s). past sessions in db are untouched.")


@command("export", "export this session's chat as plain text")
def _export(ctx: ChatContext, args: str) -> CommandResult:
    convo = [m for m in ctx.history if m["role"] != "system"]
    if not convo:
        return CommandResult("(nothing to export yet)")
    lines = [f"# session #{ctx.session_id}", ""]
    for m in convo:
        speaker = "you" if m["role"] == "user" else "lemon"
        lines.append(f"{speaker}: {m['content']}")
        lines.append("")
    return CommandResult("\n".join(lines).rstrip())


@command("autofacts", "toggle automatic fact extraction: /autofacts on|off")
def _autofacts(ctx: ChatContext, args: str) -> CommandResult:
    arg = args.strip().lower()
    if arg in ("on", "enable", "true", "1"):
        config.ENABLE_AUTO_FACTS = True
        return CommandResult("auto facts: ON")
    if arg in ("off", "disable", "false", "0"):
        config.ENABLE_AUTO_FACTS = False
        return CommandResult("auto facts: OFF")
    if not arg:
        status = "ON" if config.ENABLE_AUTO_FACTS else "OFF"
        return CommandResult(f"auto facts: {status}")
    return CommandResult("usage: /autofacts on|off")


@command("cache", "toggle prompt caching: /cache on|off")
def _cache(ctx: ChatContext, args: str) -> CommandResult:
    arg = args.strip().lower()
    if arg in ("on", "enable", "true", "1"):
        config.ENABLE_PROMPT_CACHE = True
        return CommandResult("prompt cache: ON")
    if arg in ("off", "disable", "false", "0"):
        config.ENABLE_PROMPT_CACHE = False
        return CommandResult("prompt cache: OFF")
    if not arg:
        status = "ON" if config.ENABLE_PROMPT_CACHE else "OFF"
        return CommandResult(f"prompt cache: {status}")
    return CommandResult("usage: /cache on|off")


@command("quit", "exit the chat (alias: /exit)")
@command("exit", "exit the chat (alias: /quit)")
def _quit(ctx: ChatContext, args: str) -> CommandResult:
    ctx.exit_requested = True
    return CommandResult("bye.")
