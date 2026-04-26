# Slash commands

Type any of these in the CLI or web input. Anything starting with `/` is dispatched through `commands.py`; everything else goes to the chat model.

| command                       | what it does                                                          |
| ----------------------------- | --------------------------------------------------------------------- |
| `/help`                       | list every command and its summary                                    |
| `/state`                      | print lemon's current internal state as JSON                          |
| `/reset`                      | reset internal state to defaults (does not erase facts or history)    |
| `/facts`                      | list everything stored in the `facts` table                           |
| `/remember key=value`         | upsert a fact lemon will see in the `<user_facts>` block next turn    |
| `/forget key`                 | delete a fact                                                         |
| `/history [n]`                | show the last N exchanges in this session (default 5)                 |
| `/rewind`                     | drop the last exchange (your message + lemon's reply)                 |
| `/clear`                      | drop the in-memory chat history this session (database is untouched)  |
| `/export`                     | dump this session's chat as plain text                                |
| `/model name`                 | switch chat model for this session (e.g. `anthropic/claude-haiku-4.5`)|
| `/sessions`                   | list recent sessions stored in the database                           |
| `/search query`               | full-text search past messages (FTS5 over `messages.content`)         |
| `/recall emotion`             | past user messages tagged with the given emotion                      |
| `/stats`                      | counts: messages this session, sessions, total messages, facts        |
| `/config`                     | show current behaviour flags (model, empathy, auto-facts, cache, …)   |
| `/empathy on\|off`            | toggle the empathy pipeline (no arg shows current status)             |
| `/autofacts on\|off`          | toggle automatic fact extraction (no arg shows current status)        |
| `/cache on\|off`              | toggle Anthropic-style prompt caching (no arg shows current status)   |
| `/why`                        | show the empathy-pipeline trace for the last reply                    |
| `/quit` or `/exit`            | end the chat                                                          |

## Examples

**Tell lemon something to remember:**
```
you: /remember college_year=second
remembered: college_year = second
```
Next turn, lemon's prompt includes a `<user_facts>` block that lists `college_year: second`.

**See the current mood / energy:**
```
you: /state
{
  "mood": "good",
  "energy": "medium",
  "engagement": "deep",
  "emotional_thread": "curious about exam result",
  "recent_activity": null,
  "disposition": "warm"
}
```

**Undo a misfire:**
```
you: I'm so embarrassed
lemon: ah no, what happened?
you: /rewind
rewound 2 message(s).
```
Now the previous exchange is gone from the history that gets sent to the model.

**Try a different model for a single conversation:**
```
you: /model anthropic/claude-haiku-4.5
chat model set to anthropic/claude-haiku-4.5 for this session.
```
The override lasts until the process exits — `config.CHAT_MODEL` is unchanged.

**See why lemon answered the way it did:**
```
you: I had a rough day
lemon: yeah, that sounds heavy. what happened?
you: /why
last reply's pipeline trace:
  emotion: sadness (intensity 0.65)
           underlying need: feel heard, not solved
  feeling:   tired and let down
  avoid:     don't jump to advice
  do:        stay with it, ask one open question
  memories used: 2
  post-check: passed
```

**Turn the pipeline off (e.g. for cost or latency):**
```
you: /empathy off
empathy pipeline: OFF
```
Drops to one user-facing chat call per turn (skips `user_read`, memory retrieval, and the post-check). The backgrounded bookkeeping call still runs after each reply so facts/state keep accumulating.

`/autofacts off` and `/cache off` are the matching toggles for fact extraction and prompt caching, with the same semantics — runtime-only, reverts on next process start.

**Search past messages by content:**
```
you: /search exam
matches for 'exam':
  #61  2026-04-26T19:33  worried about my exam tuesday
  #58  2026-04-23T14:02  exam went better than i thought
  …
```
Hits go through the FTS5 index with the porter stemmer, so `exam` matches `exams` / `examined`.

**Pull up past messages by emotion:**
```
you: /recall sadness
past messages tagged 'sadness':
  2026-04-21T22:14 (0.78)  i didn't think it would hit me this hard
  …
```
Reads from the `emotion` column the empathy pipeline writes per user turn. `intensity` shown in parentheses.

**Quick situational view:**
```
you: /stats
stats:
  this session: 12 messages
  all sessions: 64
  all messages: 478
  facts stored: 175

you: /config
config:
  chat model:        anthropic/claude-haiku-4.5
  state model:       anthropic/claude-haiku-4.5
  empathy pipeline:  on
  empathy retry:     on
  auto facts:        on (max 3/turn)
  prompt cache:      on
  memory retrieval:  top 3
  keep recent turns: 8
```

**Wipe the visible thread without touching memory:**
```
you: /clear
cleared 14 message(s). past sessions in db are untouched.
```
Useful when the in-context history is getting long and you want a fresh visual conversation, but you still want lemon's facts and past sessions intact for retrieval.

## Adding your own command

Every command is just a function decorated with `@command(name, help_text)`. The decorator registers it with the dispatcher.

```python
@command("mood", "force a mood: /mood happy")
def _mood(ctx: ChatContext, args: str) -> CommandResult:
    new = args.strip()
    if not new:
        return CommandResult(f"current mood: {ctx.internal_state['mood']}")
    ctx.internal_state["mood"] = new
    state_mod.save_state(ctx.internal_state, session_id=ctx.session_id)
    return CommandResult(f"mood forced to {new}.")
```

Drop that into `src/commands.py` and it's available in both CLI and web.
