# How the LLM interprets lemon's state (numbers vs prose)

> The whole dyadic-state schema is mostly floats: PAD coordinates in
> [-1, +1], Big 5 traits in [-1, +1], emotion intensity in [0, 1]. But the
> LLM is not a regression model. It does not "feel" 0.62 vs 0.55 the way
> the schema does. This doc spells out how each numeric field is translated
> into something the model can actually act on, and where the translation
> can fool you.

Read alongside `docs/dyadic_state.md` (the schema) and
`src/prompts/__init__.py` (the actual rendering code). Section numbers in
this doc don't map onto the dyadic_state numbering; this is a cross-cutting
view focused on representation.

---

## 1. The core trick: render to prose, keep the number as an anchor

The model never sees a bare float. Every numeric field is rendered into a
**descriptor word + the number in parentheses**, e.g. `pleasure +0.35` or
`high openness`. Two reasons:

- **Words give the model a category** (`high`, `low`, `mild`, `tired`).
  The model's pretraining ties those words to behaviour patterns. "low
  arousal" maps to a real cluster of texts the model has seen; `0.18` does
  not.
- **The raw number gives the model magnitude.** Two adjacent renderings
  with the same descriptor (`high openness 0.55` vs `high openness 0.85`)
  will read slightly differently because the number anchors which end of
  "high" you mean.

Wherever we strip the number, the model loses the magnitude and behaves
flatter. Wherever we strip the descriptor, the model is left with a scalar
it cannot reliably ground. We always include both.

---

## 2. PAD core affect (pleasure / arousal / dominance)

### 2.1 What it is

Three floats in [-1, +1] modelling the user's (and separately, lemon's)
*tonic core affect* per Mehrabian's PAD model.

- **pleasure**: positive vs negative valence
- **arousal**: activated vs sleepy
- **dominance**: in-control vs overwhelmed

### 2.2 How it reaches the LLM

Inside `<user_state>` and `<lemon_state>`, both blocks render the live PAD
as a single line, prefixed with the categorical mood label:

    Mood: tired (pleasure -0.20, arousal -0.40, dominance -0.10)

The categorical label is the descriptor (`tired`); the three numbers are
the magnitude anchors. The model reads "this person is tired in *that*
specific way" rather than just "this person is tired."

### 2.3 What the LLM does with it

The model uses the *sign + magnitude* combination to modulate its reply
register. In practice:

- Negative pleasure with low arousal: shorter sentences, slower rhythm,
  fewer questions.
- Negative pleasure with high arousal: still short, but more grounding
  language ("yeah, that sounds rough"), no upbeat openers.
- Positive pleasure with high arousal: looser, can match an exclamation
  or two.
- Dominance < 0 with negative pleasure: the user feels overwhelmed; the
  model softens directives, never stacks questions.

The *exact* numeric thresholds the model uses are not deterministic;
they're emergent from training. But the rendering is designed so that
crossing zero on any axis triggers a noticeable register change because
the *mood label* will usually flip with it (see §3).

### 2.4 What it cannot do

PAD is a coarse affect coordinate. It does not encode topic, intent, or
specific worry. The model cannot infer "they're tired *because of work*"
from PAD alone; that comes from `<user_facts>`, `<reading>`, and the
conversation itself.

---

## 3. Mood label (the categorical view of PAD)

### 3.1 What it is

A whitelist of 10 folksy strings (`src/storage/user_state.py:67-78`):

    neutral, calm, content, happy, excited,
    anxious, low, tense, tired, frustrated

It is currently picked by the LLM during the user-read delta and clamped
to the whitelist by `validate_user_state`. Per the algorithmic-state
roadmap (`docs/algorithmic_state_options.md` §1.1) it can be derived
deterministically from PAD via nearest-centroid lookup; that hasn't
shipped yet.

### 3.2 How it reaches the LLM

Already covered in §2.2: it is the descriptor word that prefixes the PAD
line.

### 3.3 What the LLM does with it

This is the most load-bearing label in the prompt for the model's reply
register. PAD numbers nudge tone; the mood label sets the *frame*.
"`Mood: tired`" reliably collapses reply length; "`Mood: excited`" reliably
expands it. The model reads the label as a category it has thousands of
training examples of, then uses the PAD numbers to dial within that
category.

---

## 4. Big 5 traits (OCEAN)

### 4.1 What it is

Five floats in [-1, +1], one per axis (openness, conscientiousness,
extraversion, agreeableness, neuroticism). Slow-drift / essentially frozen.

### 4.2 How it reaches the LLM

Through the **`_trait_descriptor` function**
(`src/prompts/__init__.py:417-428`). Each axis is bucketed by magnitude
and rendered as a short phrase, or dropped entirely if it's near the
population mean:

| Magnitude band       | Sign | Descriptor          |
|----------------------|------|---------------------|
| abs(v) < 0.15        | any  | (dropped — silent)  |
| v >= 0.5             | +    | `high <label>`      |
| 0 < v < 0.5          | +    | `somewhat <label>`  |
| -0.5 < v < 0         | -    | `slightly low <label>` |
| v <= -0.5            | -    | `low <label>`       |

The five descriptors are joined into one prose line:

    Roughly: high openness, high conscientiousness, high extraversion,
    high agreeableness, slightly low neuroticism.

In the model-facing user_read prompt the labels are softened to lay
adjectives (`open`, `structured`, `outgoing`, `agreeable`, `reactive`)
so the model doesn't read them as clinical jargon
(`src/prompts/__init__.py:_format_user_state_for_read`).

### 4.3 What the LLM does with it

Trait descriptors do not modulate the *current turn's* tone the way PAD
does. They shape lemon's *priors* about who they are talking to:

- `high openness` → assume the user is fine with abstract or speculative
  riffs; don't anchor every reply to concrete facts.
- `high conscientiousness` → don't undersell their commitments; they
  remember what they said they'd do.
- `low neuroticism` → don't read every short reply as distress.

Because the bands are coarse (only four buckets per axis after the
silence cutoff), the model treats traits as *typed* rather than
*continuous*. A change from 0.55 to 0.65 on extraversion will not visibly
change behaviour. A change from 0.45 to 0.55 (crossing the `somewhat` →
`high` boundary) will.

### 4.4 Why we surface them at cold start

DEFAULT_USER_STATE seeds calibrated traits even when PAD/adaptations are
empty (see `docs/dyadic_state.md` and the prompt-stack implementation in
`format_user_state_block`). At cold start the prompt looks like:

    First read of this person — let your reply do the inferring.
    Roughly: high openness, high conscientiousness, ...

The "first read" notice tells the model that *mood and goals* are fresh,
not that the *person* is fresh. Without the trait line the model would
flatten the user to a population mean for the first several turns.

---

## 5. Adaptations (goals / values / concerns / stance)

### 5.1 What it is

Already prose in storage:

- `current_goals: list[str]` (≤ 5)
- `values: list[{label, schwartz}]`
- `concerns: list[str]` (≤ 5)
- `relational_stance: str | None`

No numeric translation step needed. The LLM reads them verbatim through
prose wrappers like `Default mode: ...`, `What you value: ...`,
`Quietly on your mind: ...`.

### 5.2 What the LLM does with it

This is the **most behaviour-shaping data the LLM sees**, by a wide
margin. A single string in `current_goals` ("be present for the user")
has more pull on the next reply than every number in PAD combined,
because it reads as a literal directive. The earlier therapist-leak bug
(see git log around the `<state_instructions>` rewrite) was almost
entirely caused by an over-earnest adaptation phrase, not by the numeric
state.

Phrasing matters more than content here. "Be present for the user" reads
as a mission statement. "Just chat back like a person" reads as a default
mode. Same conceptual content; very different effect on the reply.

---

## 6. Phasic emotion + theory-of-mind (the per-turn read)

### 6.1 What it is

Output of the merged `read_user()` STATE_MODEL call
(`src/empathy/user_read.py`). Two sub-objects:

- **emotion**: `{primary, intensity, undertones, underlying_need}`. The
  primary is one of 23 categorical labels (`EMOTION_LABELS`,
  `src/prompts/__init__.py:338`), intensity in [0, 1].
- **tom**: three free-text fields (`feeling`, `avoid`, `what_helps`).

### 6.2 How it reaches the LLM

Combined into a single `<reading>` block by `format_reading_block`
(`src/prompts/__init__.py:348-385`). The intensity float is bucketed
into one of four words before it's printed:

| intensity range | word          |
|-----------------|---------------|
| < 0.30          | mild          |
| 0.30–0.59       | moderate      |
| 0.60–0.84       | strong        |
| ≥ 0.85          | very strong   |

Rendered:

    Primary feeling: sadness (moderate, intensity 0.45)
    Undertones: tired, lonely
    What they probably want: feel heard, not solved
    What they're actually feeling: ...
    What helps: ...
    What to avoid: ...

### 6.3 What the LLM does with it

This block is the most *acutely* load-bearing per turn. PAD describes how
the person *carries themselves into* the turn; `<reading>` describes
*what landed* in this specific message. The model uses `<reading>` to
choose the immediate move (validate, acknowledge, change subject, sit
quietly) and uses `<user_state>` to color how it executes that move.

The `intensity_word` bucketing is deliberately coarse (four words, not
ten) so the model reads it as a typed escalation ladder rather than a
continuous knob. Crossing from `moderate` to `strong` triggers a real
behaviour change; nudging intensity from 0.42 to 0.48 does not.

---

## 7. The empathy pipeline (post-check + regenerate)

### 7.1 What it is

A pure-Python, regex-based check that runs *after* the main reply is
generated (`src/empathy/empathy_check.py`). No LLM call. Twelve
detectors, each returning either `None` (passed) or a short critique
string. If any fail, the pipeline calls the chat model **once more** with
the critique appended as a system block.

The detectors are:

    minimizing             "at least...", "could be worse"
    toxic_positivity       silver-lining clichés
    advice_pivot           unsolicited advice while user is upset
    polarity_mismatch      cheery opener under negative emotion
    validation_cascade     stacked validation phrases
    therapy_speak          clinical labelling ("sounds like anxiety")
    self_centering         opener that recenters on the responder
    sycophancy             "great question" agreement-inflation
    false_equivalence      hijacks the user's moment with own story
    lecturing              "what you need to realize"
    performative_empathy   "as someone who cares..."
    question_stacking      3+ questions when user is overwhelmed

### 7.2 How "the LLM interprets" it

The regenerate critique is itself a small system block. The model reads:

    Your previous draft had problems. Rewrite it without these patterns:
    - <critique 1>
    - <critique 2>
    Keep the same intent but address the issues. Don't apologize for the
    previous draft.

That single block is enough on the second pass because the model already
saw its first draft and the failure labels are concrete enough that it
can edit-rather-than-restart. The pipeline does not recompute PAD or
emotion or ToM between draft and regenerate; the registers stay fixed
and only the critique changes.

### 7.3 What it cannot do

- It cannot detect *content* failures (factual mistakes, hallucinated
  facts about the user). Those need the bookkeep / facts pipeline.
- It cannot detect register failures the regex does not encode. The
  earlier `<internal_state>` leak slipped past every detector because
  none of them check for "model wrote a fake reasoning block."
- The retry only fires once. If the regenerated draft also fails, the
  failed-but-regenerated draft ships anyway.

The detectors are rigid by design (see `docs/empathy_research.md`): the
goal is to *prevent specific failure modes*, not to maximize empathy.
Pushing the model away from advice-pivot is the kind of intervention
where false negatives are cheaper than false positives.

---

## 8. Where each block lands in the prompt stack

The order matters because the model reads top-down and weights
later-positioned context more strongly. The pipeline assembles them in
roughly this order (`src/app/pipeline.py`):

1. **Persona** (`<Who you are>`, `<Voice and tone rules>`,
   `<state_instructions>`, …) — cached; identity-shaping.
2. **Time context** (`<time_context>`) — coarse session/time-of-day
   colouring.
3. **`<lemon_state>`** — lemon's tonic state going into this turn.
4. **`<user_state>`** — the user's tonic state (background read).
5. **`<user_facts>`** — durable facts the bookkeep pipeline has stored.
6. **`<emotional_memory>`** — relevant moments from past sessions.
7. **`<earlier_conversation>`** — compressed older turns of this chat.
8. **`<reading>`** — the per-turn phasic read (emotion + ToM).
9. **The conversation history itself** — the actual messages, last.

`<reading>` sits last among the system blocks deliberately: it is the
most "right now" piece of context, and the model reads later-positioned
content as more salient.

---

## 9. Caveats and where this representation can fool you

- **Numeric precision is theatre below the descriptor band width.**
  Moving openness from 0.55 to 0.65 has no behavioural effect because
  both render `high openness`. If you want a behaviour change, move
  across a band boundary (the bands are at ±0.15 / ±0.50).
- **The model treats some labels as instructions even when framed as
  description.** A `current_goals` entry of "be present for the user"
  reads as a directive. Phrasing decides whether a piece of state shapes
  the reply or directs it.
- **Stale adaptations contaminate fresh reads.** A `concerns` entry from
  a prior session that says "user is running on empty" will continue to
  colour replies long after the user has bounced back. Concerns are
  durable by design (see `fresh_lemon_session_state`); cleaning them is
  a separate decision.
- **Cross-session continuity is asymmetric.** Lemon re-pegs PAD,
  values, stance, and goals on every session start; concerns persist.
  The user side has no session-start reset. So whatever the user was
  feeling at the end of the last conversation is what they "walk into"
  the next one with — by design, but worth knowing when debugging "why
  is lemon still being soft."
- **The empathy regenerate is invisible to logs.** When a critique
  fires, the log just shows the second draft. Diagnosing tone problems
  requires inspecting the trace (e.g. via `/empathy` slash command or
  the structured pipeline trace).

---

## 10. Quick reference: where each thing lives

| Field                | Storage type        | Where rendered to prose                                  |
|----------------------|---------------------|----------------------------------------------------------|
| pleasure / arousal / dominance | `float [-1, +1]` | `format_user_state_block`, `format_lemon_state` (PAD line) |
| mood_label           | `str` (10-vocab)    | Same line, as the descriptor prefix                       |
| traits (Big 5)       | `dict[str, float]`  | `_trait_descriptor` (4 magnitude bands)                   |
| current_goals        | `list[str]` (≤5)    | `Default mode: ...` line                                  |
| values               | `list[{label, schwartz}]` | `What you value: <label> (<schwartz>)`              |
| concerns             | `list[str]` (≤5)    | `Quietly on your mind: ...`                               |
| relational_stance    | `str \| None`        | `Stance with this person: ...`                            |
| emotion.primary      | `str` (23-vocab)    | `Primary feeling: <label> (<word>, intensity X.XX)`      |
| emotion.intensity    | `float [0, 1]`      | Bucketed to 4 words: mild / moderate / strong / very strong |
| emotion.undertones   | `list[str]` (≤3)    | `Undertones: ...`                                         |
| emotion.underlying_need | `str \| None`     | `What they probably want: ...`                            |
| tom.feeling / avoid / what_helps | `str`     | Three lines in the `<reading>` block                      |

If you ever want to see the live prompt blocks for the current DB state,
this one-liner does it (run from project root):

```python
import sys; sys.path.insert(0, 'src')
from storage.user_state import load_user_state
from storage.lemon_state import fresh_lemon_session_state
from prompts import format_user_state_block, format_lemon_state
print(format_lemon_state(fresh_lemon_session_state()))
print(format_user_state_block(load_user_state()))
```
