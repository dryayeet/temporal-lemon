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
| `/model name`                 | switch chat model for this session (e.g. `anthropic/claude-haiku-4.5`)|
| `/sessions`                   | list recent sessions stored in the database                           |
| `/empathy on\|off`            | toggle the empathy pipeline (no arg shows current status)             |
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
Drops to one chat call per turn (no emotion classifier, no ToM, no post-check).

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
