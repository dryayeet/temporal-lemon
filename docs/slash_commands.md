# Slash commands

Type any of these in the CLI or web input. Anything starting with `/` is dispatched through `commands.py`; everything else goes to the chat model.

| command                       | what it does                                                          |
| ----------------------------- | --------------------------------------------------------------------- |
| `/help`                       | list every command and its summary                                    |
| `/state`                      | render lemon's current three-layer state (mood, PAD, traits, adaptations) |
| `/user_state`                 | render the user's inferred three-layer state                          |
| `/reset`                      | reset lemon's state to persona defaults (does not erase facts or history) |
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
| `/why`                        | show the empathy-pipeline trace for the last reply (both agents' state trajectories) |
| `/quit` or `/exit`            | end the chat                                                          |

## Examples

**Tell lemon something to remember:**
```
you: /remember college_year=second
remembered: college_year = second
```
Next turn, lemon's prompt includes a `<user_facts>` block that lists `college_year: second`.

**See lemon's current three-layer state:**
```
you: /state
lemon_state:
  mood: content
  PAD: pleasure +0.30, arousal +0.00, dominance +0.10
  traits:
    openness           +0.50
    conscientiousness  -0.20
    extraversion       +0.30
    agreeableness      +0.80
    neuroticism        -0.60
  adaptations:
    goals:    be present for the user, match their energy without forcing it
    values:   honesty, warmth without performance, calm
    concerns: (none)
    stance:   close friend, not assistant
```

**Or pull up what lemon thinks about the user right now:**
```
you: /user_state
user_state:
  mood: tired
  PAD: pleasure -0.20, arousal +0.10, dominance -0.05
  traits:
    openness           +0.00
    conscientiousness  +0.00
    extraversion       +0.00
    agreeableness      +0.60
    neuroticism        -0.30
  adaptations:
    goals:    prep tuesday exam
    values:   family
    concerns: feeling unprepared
    stance:   open, slightly tired
```
Both blocks render the same three-layer schema (Big 5 traits, characteristic adaptations, PAD core affect plus a derived mood label). See `docs/dyadic_state.md` for the design.

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
  user mood: neutral -> low
    PAD nudge: p-0.15 a+0.05
    added: rough day
  lemon mood: content
    PAD nudge: p-0.05
    added: user had a rough day today
  post-check: passed
```
The two trajectory blocks (`user mood` / `lemon mood`) are the dyadic-state delta for each agent across the turn. Lemon stays close to her baseline because her dynamics are damped harder than the user's; the schema is the same.

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
@command("mood", "force lemon's mood label: /mood happy")
def _mood(ctx: ChatContext, args: str) -> CommandResult:
    from storage.user_state import MOOD_LABELS
    from storage.lemon_state import save_lemon_state
    new = args.strip()
    if not new:
        return CommandResult(f"current mood: {ctx.lemon_state['state']['mood_label']}")
    if new not in MOOD_LABELS:
        return CommandResult(f"unknown mood {new!r}. valid: {', '.join(MOOD_LABELS)}")
    ctx.lemon_state["state"]["mood_label"] = new
    save_lemon_state(ctx.lemon_state, session_id=ctx.session_id)
    return CommandResult(f"mood forced to {new}.")
```

Drop that into `src/commands.py` and it's available in both CLI and web.

`ctx.lemon_state` and `ctx.user_state` are the two three-layer state objects (see `docs/dyadic_state.md` §6). Each has the same shape: `traits` (Big 5), `adaptations` (current_goals / values / concerns / relational_stance), `state` (PAD coordinates + mood_label).
