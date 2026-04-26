# Dyadic state architecture

> **Status: stages 1, 2, and 3 SHIPPED.**
> Both lemon and the user are now modelled with the same three-layer schema
> (Big 5 traits + characteristic adaptations + PAD core affect). Per-turn
> state updates happen pre-reply for both agents in a single merged LLM
> call; response generation reads from freshly-updated states. The legacy
> 6-field internal_state shape is deprecated in favour of the unified
> three-layer schema.
>
> Author: lemon, in discussion with the project assistant. Initial draft
> 2026-04-26; stages 2+3 landed 2026-04-27.

---

## 1. Context: why we are even considering this

### 1.1 The product framing

Lemon is built around a goal that is qualitatively different from most
"helpful assistant" chatbots:

> Simulate how two humans interact. That includes a lot of steps, but more
> importantly it includes the *underlying state of both parties* and the way
> they interact *based on that state*.

Concretely: a real friend you talk to does not simply parse your message and
emit an empathetic response. They show up to the conversation with a mood,
some energy level, things on their mind from earlier today, and a stance
they bring to you. Your message lands on top of all of that. Their reply is
the natural output of *who they are right now* meeting *what you just said*.
Then *both* people's states update.

That's the simulation target. Anything less is a smaller game.

### 1.2 The originating prompt

The architectural concern was raised by the user, paraphrased:

> "I want to simulate how humans interact. ... I told lemon to track my mood
> (the `/why` pipeline) and have an internal state within it too — to have
> semblance of its own state or mood. But right now what that internal
> state *is* is a guide on what's happening and how lemon should respond to
> the user, which is fine, but that's the next step, right? First it should
> have an internal state and *then* think how it should reply. So lemon and
> user both should have a unified internal state representation, and then
> the 'how lemon should reply' part should come into play."

The point cuts in two directions, both worth keeping separate:

1. **Symmetry**: the user and lemon should both be modelled as agents with
   the same *kind* of internal state, not asymmetrically.
2. **Ordering**: response generation should be *downstream* of the state
   update, not the other way around. State first, response second.

This document treats those two as one architectural shift because they are
mutually reinforcing.

### 1.3 What this document is not

- Not an implementation plan. No code is being changed yet.
- Not a description of current behaviour — that's `architecture.md` and
  `memory_architecture.md`. This is a *proposal*.
- Not a settled design. Several open questions are flagged in §6 and need
  answering before any code is written.

---

## 2. Where we are today

### 2.1 Lemon's state (tonic, persisted)

`storage/state.py::DEFAULT_STATE`:

```python
{
    "mood":             str,   # neutral | good | low | happy | anxious | restless | tired | content
    "energy":           str,   # low | medium | high
    "engagement":       str,   # low | normal | deep
    "emotional_thread": str | None,  # what's on lemon's mind across turns
    "recent_activity":  str | None,  # only when grounded by chat
    "disposition":      str,   # warm | normal | slightly reserved
}
```

Six fields. Tonic — persists across turns inside a session, persisted to
SQLite at `state_snapshots`, with `SESSION_START_OVERRIDES` re-pegging
mood/energy/engagement/disposition at session boundaries (so lemon does not
inherit last session's drained energy when "picking up the phone fresh").
`emotional_thread` and `recent_activity` survive across sessions.

State is updated **after** the reply, by `empathy/post_exchange.py`, in a
background daemon thread so the user does not wait on it. The prompt that
drives the update enforces "subtle nudges only" — large categorical leaps
(e.g. `neutral → grief`) are explicitly disallowed.

### 2.2 User's state (phasic only, ephemeral)

The user has *no persistent state object*. What exists per turn:

- `<user_emotion>` block: `{primary, intensity, undertones, underlying_need}`
  emitted by `empathy/user_read.py` before reply.
- `<theory_of_mind>` block: `{feeling, avoid, what_helps}` emitted by the
  same merged call.
- The user's `emotion`, `intensity`, `salience` are written to the
  `messages` row in SQLite, so historical phasic data exists at the
  per-message level — but there is no aggregation into a tonic user-mood
  object. Past emotion is reachable via memory retrieval composite scoring,
  not as "the user is currently sad-leaning."

So the user is modelled as a stateless source of messages with per-message
emotional signal. No persistent representation of "where the user is right
now in their life" or "where they were when they walked into this session."

### 2.3 The asymmetry, named

| dimension | lemon | user |
| --- | --- | --- |
| tonic state | yes (6 fields, persisted) | no |
| phasic state | no | yes (per-message emotion + ToM) |
| update cadence | post-reply, damped | per-read, fresh each turn |
| schema vocabulary | small mood vocabulary (8 values) | rich emotion vocabulary (23 labels) |
| persistence | SQLite snapshots | only as per-message columns |
| cross-session continuity | partial (emotional_thread, recent_activity) | none — only via memory retrieval |

This asymmetry is **structural**, not just a vocabulary difference. The
earlier discussion in the conversation defended the *vocabulary* asymmetry
on persona-stability grounds. That defence is correct as far as it goes,
but it is answering the wrong question. The right question is whether the
*structures themselves* should be symmetric. They probably should.

### 2.4 The double-duty problem

The current `internal_state` is doing two jobs at once:

1. **State** — what lemon is right now (mood, energy, etc.).
2. **Response guide** — implicit instructions to the response generator
   about how to behave (e.g. `disposition: warm` is read by the generator
   as "be warm").

In a clean dyadic model these are different objects. The state describes
who the agent is. A *separate* layer translates state into response
posture. Today the two are conflated because there is only one place to
put state-shaped information.

---

## 3. The proposal, stated precisely

### 3.1 Architectural shift

**From:** asymmetric structure where lemon has tonic + the user is read
phasically; response generation does double duty as state-encoder.

**To:** symmetric structure where both agents have the same kind of state
object; response generation is downstream and reads from both states.

### 3.2 Both agents get two layers

Per agent (lemon **and** user):

- **Tonic state** — slow-moving background. Mood, energy, what's on their
  mind, current focus, stance toward the other agent.
- **Phasic event** — acute reaction to the latest exchange. Per-turn,
  ephemeral.

The *experienced affect* at any moment is the combination of tonic state
and the most recent phasic event — this matches Russell's core-affect-
plus-appraisal framing (cited in §4.1).

### 3.3 Pipeline reordering: state first, response second

The new turn flow:

1. **Read** the user's message → produce user phasic event (current
   `user_read` already does this, with appraisal).
2. **Update user's tonic state** — nudge based on the phasic event. NEW.
3. **Compute lemon's phasic reaction** — what does lemon "feel" in
   response to what was just said. (Small, optional — see persona-drift
   risk in §7.1.)
4. **Update lemon's tonic state** — nudge based on the phasic reaction
   and ToM read. Currently happens in `post_exchange` AFTER reply;
   MOVED to before.
5. **Generate response** conditioned on `(lemon_state, user_state,
   lemon_phasic, user_phasic, tom)`. Reply is what an agent in that
   state would naturally say to an agent in that state.
6. **Post-exchange**: facts extraction, optional secondary confirmation
   pass on the state nudge based on what lemon actually said.

The fundamental change is step 4 (lemon state update) moves to **before**
the reply, so the reply genuinely reflects an updated internal state
rather than describing an after-the-fact narrative. Steps 2 and 3 are
new additions that make the user side and lemon's reactive layer first-
class.

### 3.4 Theory of mind: separate from user_state

ToM is **lemon's model of user_state**. The actual user_state is a
separate, ground-truthier object inferred from observed user behaviour.
ToM and user_state diverge when lemon misreads — and that divergence is
itself a useful signal (§6.4). They should be different objects in the
architecture, not merged.

---

## 4. Psychological grounding

The proposal is not novel; it has substantial backing in the affective-
science literature and matches how the dialogue-systems research
community has been moving for several years.

### 4.1 Tonic vs phasic affect (Russell, Sels et al.)

Russell's *core affect* theory treats every person's affective state as
having a tonic background (valence × arousal coordinates that drift
slowly) plus phasic appraisals of events that move the coordinates. This
is the canonical psychology version of "have a state, then react." Sels
et al. (PLOS One 2025, *From dyadic coping to emotional sharing and
multimodal interpersonal synchrony*) explicitly separate the two
timescales in experimental design: tonic processes are mood/appraisal
changes measured before/after a sharing task; phasic processes are
moment-to-moment dynamics during the task. Different timescales, both
real.

### 4.2 Dyadic systems modelling (Bodie et al., Butler)

Bodie & colleagues, *A Dynamic Dyadic Systems Approach to Interpersonal
Communication* (Oxford JoC, 2021), argue that conversations are
**interdependent dual-state systems**. Each interlocutor has internal
state; each state evolves under two forces — **self-dependence** (the
tendency of someone's own state to persist) and **interpersonal
dependence** (the influence of the other's state). State-space-grid
methods empirically model the joint trajectory.

Butler, *Emotions are temporal interpersonal systems* (Current Opinion in
Psychology), makes the case for emotion specifically: emotional states
are not properties of individuals but of *dyads over time*. Modelling
either party in isolation loses information. This is exactly the gap
lemon's current architecture has.

### 4.3 Three-layer affect (Egges et al.)

Egges, Kshirsagar & Magnenat-Thalmann, *Generic personality and emotion
simulation for conversational agents* (CAVW, 2004), give the canonical
computational framing for an affective agent: stable **personality** →
medium-term **mood** → acute **emotion**. Each layer has different
update dynamics. Personality drifts on a scale of months; mood on the
order of hours; emotion on seconds. This three-layer structure applies
*to whichever agent is being simulated*, symmetrically.

In lemon's terms: lemon has a hardcoded personality (the system prompt),
a tracked mood (`internal_state`), and currently no explicit emotion
layer. The user has nothing on the personality layer (could be
`facts`-derived), nothing on the mood layer (the gap), and a phasic
emotion read per turn.

### 4.4 Hierarchical trait/state decoding

Tay et al. (Scientific Reports, 2025), *A hierarchical trait and state
model for decoding dyadic social interactions*, formalises the same
idea computationally: traits are stable patterns that differ across
individuals; states are phasic patterns that vary over time within
individuals, oscillating around their traits. Their model decodes BOTH
participants' state from multimodal dyadic interaction data. The
explicit framing — trait + state, dyadic, both interlocutors — is
exactly the right scaffold.

### 4.5 Empathetic dialogue systems: known gap

Ma et al., *A survey on empathetic dialogue systems*, surveys the field
and confirms that most systems track **only the user's emotion**;
the agent's affective state is handled implicitly through persona
prompts. The asymmetry lemon currently has is the *default* in the
field. Closing it is recognised as a frontier direction but not
yet productionised in most companion systems.

### 4.6 Bot's own affect, when it shows up

Spitale et al., *Socio-conversational systems: Three challenges at the
crossroads of fields*, calls out that "socio-conversational systems
must employ neural architectures that mirror dyadic processes by
analysing the user's states and generating the agent's utterances
within the dyadic process of two interlocutors working together."
Translation: agent state and user state are co-equal inputs to
generation, not "user state in, agent reply out."

---

## 5. Architectural precedent

Closest existing implementations of the symmetric pattern:

### 5.1 Park et al. 2023 — Generative Agents (Stanford)

**Reference:** Park, O'Brien, Cai, Morris, Liang, Bernstein,
*Generative Agents: Interactive Simulacra of Human Behavior*, UIST 2023.
arXiv 2304.03442.

Architecture: every agent has the *same* components — memory stream,
reflection, planning. Agents interact by reading each other's behaviour
and updating their own state. There is no privileged "user" agent —
all participants are agents of the same kind. When a human is in the
loop, the human is just another agent from the system's perspective.

This is the cleanest reference for the symmetric pattern. The key
takeaway for lemon: the agent and the user can share architecture
*even if* the human isn't actually being simulated by an LLM — what
matters is that the *system's representation* of both is symmetric, so
that the response-generation step is uniform.

### 5.2 MemoryBank / SiliconFriend (Zhong et al. 2023)

**Reference:** Zhong, Guo, Gao, Ye, Wang, *MemoryBank: Enhancing Large
Language Models with Long-Term Memory*, AAAI 2024. arXiv 2305.10250.

Tracks long-term **user** properties — personality profile, dialogue
summaries, with Ebbinghaus-curve forgetting. Solves the user-state
persistence problem, but the bot's affective state is still implicit
(LoRA-tuned persona, no explicit running state). So MemoryBank is
half of what's needed: persistent user representation without the
symmetry.

### 5.3 The empathic-CA platforms (mental health context)

The systematic review by JMIR Mental Health (*Empathic Conversational
Agent Platform Designs and Their Evaluation*) shows that mental-health
focused empathic CAs almost universally track the user's emotional
state (PHQ-9, GAD-7, derived signals) but rarely model the bot's own
state — the bot is treated as a stable, neutral interviewer. This is a
*deliberate* asymmetry for clinical reasons (you don't want a therapist
agent to "have a bad day"), but it's the wrong default for a
*companion* simulation, which is what lemon is trying to be.

### 5.4 Affective embodied conversational agents

Older line of work — Egges (2004), Becker-Asano's WASABI architecture,
Marsella & Gratch's EMA — all treat the agent as a full affective
system with personality + mood + emotion layers. Symmetric framing is
natural in those systems because they were originally built for
multi-agent simulation (NPCs in environments). The LLM-companion
literature has somehow regressed from this; lemon's proposal would
return to the older, more principled framing with modern LLM tooling.

---

## 6. Unified internal-state representation (LOCKED)

The representation question — *what does an internal-state object actually
look like, psychologically* — has been settled. Per agent, a **three-layer
representation** following McAdams' integrative framework, with each layer
operationalised by the most empirically-defensible model in its category.

### 6.1 The three layers

| layer | timescale | what it represents | model used | dynamics |
| --- | --- | --- | --- | --- |
| **Traits** | months / years (essentially static) | dispositional tendencies | **Big 5 / OCEAN** | hardcoded for lemon; slowly inferred for the user |
| **Characteristic adaptations** | weeks / months (evolves) | goals, values, concerns, relational stance | structured strings (free-form, optionally Schwartz-tagged) | accumulates and prunes across sessions |
| **State (mood)** | hours / days (drifts) | current background affect | **PAD (Pleasure-Arousal-Dominance)** + derived categorical mood label | nudged each turn by phasic events, reverts toward trait-defined home range |

Plus, **per turn (ephemeral)**:

| object | timescale | what it represents | model used |
| --- | --- | --- | --- |
| **Phasic emotion event** | seconds | acute reaction to the latest exchange | the existing 23-label categorical (already implemented) + intensity |

This is McAdams' three-level model (Levels 1, 2, 3) with the affect layer
implemented per Russell core affect / Mehrabian-Russell PAD, and phasic
events as the *input* that nudges the state layer (per Whole Trait Theory:
states are samples from trait-shaped distributions).

### 6.2 Concrete schema (applies to both lemon and user)

```python
INTERNAL_STATE = {
    # Layer 1: Big 5 traits (each in [-1, +1])
    "traits": {
        "openness":          float,
        "conscientiousness": float,
        "extraversion":      float,
        "agreeableness":     float,
        "neuroticism":       float,
    },

    # Layer 2: characteristic adaptations
    "adaptations": {
        "current_goals":     list[str],   # what they're trying to do
        "values":            list[str],   # what matters to them
        "concerns":          list[str],   # what's on their mind
        "relational_stance": str | None,  # how they show up to the other agent
    },

    # Layer 3: PAD core affect
    "state": {
        "pleasure":   float,   # [-1, +1]
        "arousal":    float,   # [-1, +1]
        "dominance":  float,   # [-1, +1]
        "mood_label": str,     # derived; whitelisted from a small folksy set
    },
}
```

Phasic event (per turn, ephemeral, **separate** from this object):

```python
PHASIC = {
    "event_emotion": str,    # one of EMOTION_LABELS (the 23-label vocab)
    "intensity":     float,  # [0, 1]
    "appraisal":     str | None,  # what about the message triggered this
}
```

### 6.3 Why these specific picks

**Big 5 over MBTI / cognitive functions / Enneagram.** The empirical case
is one-sided. MBTI test-retest reliability is so weak that 39–76% of
people get a different type after five weeks; the cognitive functions
layer has even less empirical support; Enneagram has mixed reliability
and validity. Big 5 has decades of cross-cultural validation, predicts
real-life outcomes ~2× better than MBTI, and is continuous (not
categorical). It's also already the de facto standard for LLM persona
research (Serapio-García et al. 2023; PersonaLLM; *LLMs Simulate Big5*
2024). Using anything weaker as the trait layer would be a structural
mistake. **HEXACO** (Big 5 + Honesty-Humility) is a reasonable upgrade
path if more moral-dimension coverage is wanted later; not needed for
stage 1.

**PAD over PANAS / categorical-only.** PAD is continuous (smooth
nudging, matches the "subtle nudges only" prompt discipline), three-
dimensional (the dominance axis captures "in control vs at the mercy
of things" — directly relevant for empathy), and is the dominant
choice in computational affective agents (used in ALMA, in animated
character systems, in PAD-based emotional contagion models). PANAS
misses dominance entirely and its two axes (PA, NA) only cover one of
four variants of positive/negative affect each. Pure categorical mood
labels can't blend continuously and don't compose under nudging.

**Categorical mood label derived from PAD.** Keep a small whitelisted
label set (e.g. `neutral, calm, content, happy, excited, anxious, low,
tense, tired, frustrated`) for prompt readability — but compute it as
a function of PAD coordinates, not as a separate primary
representation. The LLM reads the label like a friend reads a vibe;
the PAD coordinates underneath give the system smooth dynamics.

**Free-form adaptations (with structured slots).** `current_goals`,
`values`, `concerns`, `relational_stance` are short strings. Could be
Schwartz-tagged later (achievement, benevolence, etc., 10 universal
values). For stage 1, free-form is more flexible and matches lemon's
existing fact-storage style.

**Phasic emotion stays separate.** The 23-label categorical event
(already implemented in `empathy/emotion.py`) is the *input* that
nudges layer 3. Don't merge them — phasic and tonic operate on
different timescales, and conflating them was the original design
muddle this whole proposal exists to fix.

### 6.4 The trait–state bridge: Whole Trait Theory

Fleeson's Whole Trait Theory (2001, refined 2015 and 2025) gives the
clean theoretical link between traits and states: **traits are density
distributions of states**. A trait of "extraversion = 0.6" means the
agent's typical extraversion across daily life is centred around 0.6,
with a spread that's also a stable individual difference. State
fluctuates day-to-day and turn-to-turn within that trait-shaped
distribution.

For lemon's dynamics:

- Phasic event → state nudge (clamped magnitude per turn).
- State drifts back toward a trait-defined home range over time
  (mean-reversion).
- Trait nudges per turn are tiny (≤0.02) — essentially frozen.
- Adaptations evolve at a third rate: more responsive than traits, less
  responsive than state.

This is what makes the state layer *honest* rather than reactive: state
moves, but it's anchored to the trait distribution underneath.

### 6.5 Theory of mind: still separate, still distinct from user_state

ToM is **lemon's *model* of user_state**, not user_state itself. Same
shape (or a subset), different epistemic status. When ToM and the
inferred user_state diverge, that's signal: lemon misread the user. In
later stages this divergence can drive self-correction. For stage 1,
ToM stays in its existing form (`feeling`, `avoid`, `what_helps`); the
new user_state object lives alongside it and is fed into the response
generation as a separate block.

### 6.6 Storage layout

| object | scope | persistence |
| --- | --- | --- |
| `lemon_state` (today's `internal_state`) | single-user | `state_snapshots` table — UNCHANGED in stage 1 |
| `user_state` (NEW, three layers) | single-user | NEW `user_state_snapshots` table |
| `lemon_traits` & `lemon_adaptations` | static persona | constants in NEW `src/persona.py` |
| `phasic emotion` per user turn | per-message | already in `messages.emotion`/`intensity`/`salience` columns |
| `phasic emotion` per lemon turn | per-message | NOT IMPLEMENTED in stage 1 (lemon-side phasic is stage 2+) |
| `tom` per turn | per-turn ephemeral | UNCHANGED — flows through pipeline trace, not persisted |

A new `user_state_snapshots` table parallel to `state_snapshots` (rather
than extending it with a discriminator column) keeps trajectory history
for `/why` debugging, doesn't require backfill of existing rows, and is
purely additive to the schema.

---

## 7. Design decisions (resolved for stage 1)

The decisions below were open in earlier drafts. They are now resolved
for stage 1 implementation. Stage 2 / 3 may revisit some of them.

### 7.1 How damped is lemon's phasic update?

The risk: in a symmetric architecture, lemon's tonic state can be
swayed turn-to-turn by the user's emotional weather, producing a moody-
mirror persona. This is bad — friends are stabler than the people they
support.

**Mitigation:** lemon's phasic-to-tonic update is **heavily damped**.
The current `post_exchange` prompt already enforces "subtle nudges
only." That same constraint applies in the new architecture, just at a
different point in the pipeline. The user's update can be more
responsive because users genuinely swing more turn-to-turn than long-
term-stable companions do. **The schema is symmetric; the dynamics are
asymmetric, and that asymmetry is psychologically honest.**

### 7.2 Reset behaviour — does the user get session-start overrides too?

Lemon currently re-pegs `mood`, `energy`, `engagement`, `disposition` at
the start of every session. Realistically a *user* doesn't reset like
that — they bring whatever they carried in.

**Lean: no session-start override on the user side.** The user's tonic
state persists across sessions as-is. The previous session ended at
some state; that's where the next session starts (with an optional
"some time has passed" decay applied). This is asymmetric reset
behaviour with the symmetric schema, and it's the right call.

### 7.3 LLM round trips — RESOLVED

**Decision: extend `user_read`. No new round trip.** The single
existing pre-reply call now emits emotion + ToM + a user-state delta
in one merged JSON response. `post_exchange` is unchanged in stage 1.
Output JSON gets longer (modest cost) but the call count stays the
same.

Stages 2 and 3 may add lemon-side state updates pre-reply (currently
post-reply via `post_exchange`); whether that costs an extra call is a
stage-2 decision.

### 7.4 Cold-start for user_state — RESOLVED

**Decision:** initialise user_state from `DEFAULT_USER_STATE` (zeroed
PAD, zeroed traits, empty adaptations, neutral mood label) on first
contact. The block formatter recognises the all-zero/empty case and
emits a degraded version ("First read of this person — let your reply
do the inferring") to signal low confidence to the LLM. By turn 3-5,
small per-turn nudges have populated enough of the state for the
normal block format to take over. No special "calibration mode" flag.

Cross-session: previous session's end-state is the new session's
start-state. **No session-start overrides on the user side** (this is
the §7.2 decision: a user genuinely brings their carried-in mood; not
resetting them is the right call).

### 7.5 Prompt block layout — SHIPPED stage 3

Final layout in the system block stack:

```
position 0: <Who you are>          (persona — cache anchor, UNTOUCHED)
position 1: <time_context>
position 2: <lemon_state>          (renamed from <internal_state>, three-layer schema)
position 3: <user_facts>           (if facts exist)

per-turn injections (in pipeline order):
  <lemon_state>                    (re-injected with the freshly-nudged
                                    state, AFTER read_user pass)
  <emotional_memory>               (if memories)
  <user_state>                     (tonic, three-layer schema)
  <reading>                        (NEW unified block: phasic emotion + ToM)
```

Tonic-then-phasic for both agents. `<lemon_state>` is refreshed
twice — once via `refresh_base_blocks` from carried-in state, then
overwritten by the pipeline with the post-nudge state — so the reply
generator reads the updated lemon_state (this is the stage 2 ordering
guarantee: state-first, response-second).

### 7.6 `<user_state>` block format — RESOLVED

**Decision:** compact prose with PAD numbers in parentheses. Not JSON.

Example shape:

```
<user_state>
Background read of the person you're talking to. Lets your responses
match where they are right now, not just the latest message. Do not
narrate this.

Mood: calm (pleasure +0.15, arousal -0.10, dominance 0.00)
Roughly: somewhat agreeable, average extraversion, low neuroticism.
On their mind: prepping for tuesday's exam.
Cares about: family, doing well academically.
Worries: feeling unprepared.
How they're showing up: open, slightly tired.
</user_state>
```

Empty / cold-start case collapses to: "First read of this person —
let your reply do the inferring."

### 7.7 Schema field names — RESOLVED

**Decision: literally identical names** between `lemon_state` and
`user_state`. `traits.openness`, `state.pleasure`, etc. — same shape,
same field names. Asymmetry shows up in *dynamics* (lemon's traits
hardcoded; lemon's state damped harder) and in *update site* (lemon
post-reply via `post_exchange`; user pre-reply via `user_read`), not
in structure.

For stage 1 the lemon-side `internal_state` keeps its current 6-field
shape — moving lemon to the new three-layer schema is stage 3. Stage 1
ships with the user side on the new schema and lemon on the old one;
they're not yet symmetric in field names but they ARE symmetric in
existence (both have a persisted state object).

### 7.8 `/why` enrichment — RESOLVED for stage 1

**Decision:** stage 1 adds three new fields to the `/why` output:
`user_state_before`, `user_state_after`, `user_state_delta`. The full
trait/adaptation/PAD trajectory becomes visible. A new `/user_state`
slash command shows the current user_state in compact form, paralleling
the existing `/state` (lemon).

---

## 8. Costs and risks

### 8.1 Persona drift / moody-mirror

Discussed in §7.1. Mitigation: aggressive damping on lemon's phasic-
to-tonic update. Test by running multi-turn synthetic conversations
where the user oscillates emotionally; lemon's tonic state should NOT
oscillate at the same frequency.

### 8.2 Cold-start noise on user_state

User tonic state is unreliable for the first few turns of a new user.
Mitigation: low-confidence defaults, slow update gain at the start,
explicit confidence field that downweights its influence on response
generation when the state is fresh.

### 8.3 LLM cost / latency

Discussed in §7.3. Mitigation: fold updates into existing round trips;
do not add new ones. Worst case adds ~10% to per-turn token cost; best
case is zero added round trips.

### 8.4 More state to test / debug / observe

The complexity surface grows. Mitigation: the `/why` enrichment in
§7.8 makes the new state observable. Trace logging needs to capture
both states' before/after on each turn.

### 8.5 Prompt size growth

More state injected = bigger system block. Mitigation: prompt-cache
the stable parts (already done for the persona). The dynamic state
blocks are small (low hundreds of tokens combined).

### 8.6 Persona drift from frequent state changes (separate from §8.1)

Even with damping, repeatedly seeing varied state values in the prompt
could drift the persona's voice over very long conversations.
Mitigation: persona block is the cache anchor and stays untouched;
state blocks are smaller and clearly framed as "current snapshot, not
identity."

---

## 9. Migration order

Stage gates so the work is interruptible at each step.

### Stage 1 (SHIPPED 2026-04-26): persistent user_state with three-layer schema

- Add `src/storage/user_state.py` with `DEFAULT_USER_STATE`, validators,
  `apply_delta`, load/save wrappers. Mirrors the lemon-side `state.py`.
- Add `user_state_snapshots` table via additive migration; expose
  `latest_user_state` and `save_user_state_snapshot` from `storage/db.py`.
- Add `src/persona.py` with `LEMON_TRAITS` (Big 5) and `LEMON_ADAPTATIONS`
  (goals/values/concerns/stance) constants.
- Extend `prompts.build_user_read_prompt` to accept the current
  user_state and emit a `user_state_delta` sub-object alongside emotion
  + ToM. Add `format_user_state_block` and `USER_STATE_TAG`.
- Update `empathy/user_read.read_user` to return a 3-tuple `(emotion,
  tom, user_state_delta)` with safe fallback to a zero-delta.
- Update `pipeline.run_empathy_turn` to load/persist user_state and
  inject the new `<user_state>` block. Extend `PipelineTrace`.
- Wire `fresh_user_session_state` through `session_context`, `lem.py`,
  `web.py`. Extend `ChatContext` with a `user_state` field.
- Add `/user_state` slash command; extend `/why` with the new fields.
- New `tests/test_user_state.py`; updates to `test_pipeline.py`,
  `test_db.py`, `test_state.py` (the last is already broken — fix as
  part of this stage).

**No pipeline reordering** in stage 1. Lemon's existing `internal_state`
remains as-is (6-field shape), updated post-reply. Stage 1 is purely
*additive*.

Most of the simulation value comes from this stage alone, because it
finally gives lemon a persistent representation of "where the user
is."

### Stage 2 (SHIPPED 2026-04-27): lemon's state update moved pre-reply

Lemon's tonic state nudge moved out of `post_exchange.bookkeep` (which
ran post-reply in a background thread) and into the merged user_read
pass that now runs pre-reply. The response generator reads from a
*freshly-updated* lemon state on every turn, so reply tone genuinely
reflects what lemon "feels" in response to the just-arrived message.
`post_exchange` shrunk to facts-only.

Round-trip count is unchanged: the new lemon-side delta rides along on
the existing user_read call, which now emits four sub-objects
(emotion, tom, user_state_delta, lemon_state_delta) in one LLM
response. The `_clamp_lemon_delta` helper in `empathy/user_read.py`
applies asymmetric damping on top of the standard validator (lemon's
PAD ±0.10 vs the user's ±0.15; lemon's traits and values are frozen).

This is the "state first, response second" half of the proposal.

### Stage 3 (SHIPPED 2026-04-27): schema unification + prompt-block restructure

Lemon's runtime state migrated to the same three-layer schema as the
user side. Concretely:

- **New module `src/storage/lemon_state.py`** mirrors `user_state.py`,
  built on top of `persona.LEMON_TRAITS` and `persona.LEMON_ADAPTATIONS`
  for the static layers. PAD core affect drifts via the pre-reply nudge.
  `fresh_lemon_session_state` re-pegs PAD to a session-start baseline
  (`LEMON_SESSION_START_STATE`) and resets relational_stance to the
  persona baseline; concerns and goals carry over for cross-session
  continuity.
- **New `lemon_state_snapshots` table** parallel to
  `user_state_snapshots`. Migration entry (3, ...). The legacy
  `state_snapshots` table stays as archive but gets no new writes.
- **One-time legacy migration** (`migrate_legacy_state` in lemon_state):
  if a fresh install finds the old 6-field state but no new shape, it
  converts mood/energy → PAD coordinates, disposition → relational_stance,
  emotional_thread → first concern. recent_activity is dropped on
  migration.
- **Prompt blocks renamed** — `<internal_state>` → `<lemon_state>`. The
  separate `<user_emotion>` and `<theory_of_mind>` blocks fold into a
  single `<reading>` block (per-turn phasic layer), paired with
  `<user_state>` (per-session tonic layer). Tonic-then-phasic ordering
  for both agents.
- **`format_internal_state` removed**, replaced by `format_lemon_state`
  with the same lemon-voice framing as the legacy block (mood, traits,
  goals, concerns, stance) but rendering the new schema.

The asymmetry kept: lemon's traits are hardcoded from persona constants
and never drift; the user's traits are inferable but with `_TRAIT_NUDGE_CAP
= 0.02` per turn they're effectively frozen too. Lemon's PAD damping
(`±0.10`) is tighter than the user's (`±0.15`). Same schema, asymmetric
dynamics — psychologically honest (friends are stabler than the people
they support).

---

## 10. What this changes in user-visible behaviour

For the user (lemon's user, the human):

- **Better cross-session continuity** — lemon will remember not just
  facts and last session's loose ends but a sense of "where you've
  been" emotionally over recent sessions.
- **Better-calibrated empathy on session start** — currently lemon
  resets to upbeat-warm at session start regardless of the user's
  state. With persistent user_state, lemon can come in matched to the
  user's actual carried-in mood.
- **`/why` becomes more useful** — visible state trajectories, not
  just per-turn reads.
- **Slightly slower or pricier per turn** — depending on how the
  round-trip consolidation lands.

For the developer (you):

- **Cleaner separation of concerns** — state is state; response logic
  is response logic.
- **Easier to evaluate** — you can ablate parts of the dyadic state
  and measure the effect.
- **Foundation for multi-user, if that ever happens** — the
  representation is per-user from day one.

---

## 11. Future work (post stages 1+2+3)

With the three-stage migration complete, the obvious next moves:

1. **Trait inference** — currently essentially frozen for the user
   (`_TRAIT_NUDGE_CAP = 0.02` per turn). As session count grows,
   periodic trait re-estimation from accumulated message history
   becomes worthwhile, probably monthly-cadence or every-N-sessions
   in a separate offline pass.
2. **HEXACO upgrade** — adding Honesty-Humility as a sixth trait
   dimension. Cheap; mostly a constant swap.
3. **Schwartz-tagged values** — replace free-form `values` strings
   with tagged entries from Schwartz's 10-value taxonomy for better
   retrievability.
4. **ToM ↔ user_state divergence signal** — when lemon's per-turn
   ToM read in the `<reading>` block differs from the inferred
   `<user_state>`, treat it as a signal lemon misread; feed back
   into the next-turn prompt.
5. **Phasic-event-to-PAD mapping** — currently the LLM emits PAD
   nudges directly. A hardcoded affect→PAD mapping table per emotion
   label could be a faster / cheaper path with comparable signal.
6. **Lemon-side phasic events** — lemon currently has tonic state
   updated each turn but no explicit phasic event. A per-turn lemon
   reaction object (analogous to the user's emotion classification)
   could let `<reading>` show lemon's reaction alongside the user's.
7. **Drop the legacy `state_snapshots` table** — once the migration
   has been live long enough, the archive is no longer interesting.

---

## Sources

### Primary research

- Park, O'Brien, Cai, Morris, Liang, Bernstein. *Generative Agents:
  Interactive Simulacra of Human Behavior.* UIST 2023.
  https://arxiv.org/abs/2304.03442
- Bodie, Jones, Brinberg, Joyer, Solomon, Ram. *A Dynamic Dyadic
  Systems Approach to Interpersonal Communication.* Journal of
  Communication, 2021.
  https://academic.oup.com/joc/article/71/6/1001/6375398
- Sels, Reis, Ackermann, Boker, Bringmann, Butler, Kuppens et al.
  *From dyadic coping to emotional sharing and multimodal interpersonal
  synchrony: Protocol for a laboratory experiment.* PLOS One, 2025.
  https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0323526
- Butler. *Emotions are temporal interpersonal systems.* Current
  Opinion in Psychology, 2017.
  https://www.sciencedirect.com/science/article/abs/pii/S2352250X16302032
- Egges, Kshirsagar, Magnenat-Thalmann. *Generic personality and
  emotion simulation for conversational agents.* Computer Animation
  and Virtual Worlds, 2004.
  https://dl.acm.org/doi/abs/10.5555/1071195.1071196
- Tay et al. *A hierarchical trait and state model for decoding
  dyadic social interactions.* Scientific Reports, 2025.
  https://www.nature.com/articles/s41598-025-95916-9
- Zhong, Guo, Gao, Ye, Wang. *MemoryBank: Enhancing Large Language
  Models with Long-Term Memory.* AAAI 2024.
  https://arxiv.org/abs/2305.10250

### Surveys / context

- Ma, Nguyen, Xing, Cambria. *A survey on empathetic dialogue
  systems.* https://sentic.net/empathetic-dialogue-systems.pdf
- Spitale et al. *Socio-conversational systems: Three challenges at the
  crossroads of fields.*
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9797522/
- *Empathic Conversational Agent Platform Designs and Their Evaluation
  in the Context of Mental Health: Systematic Review.* JMIR Mental
  Health, 2024. https://mental.jmir.org/2024/1/e58974

### Personality models (trait layer)

- McAdams & Pals. *A new Big Five: Fundamental principles for an
  integrative science of personality.* American Psychologist, 2006.
  (The three-level integration: traits, characteristic adaptations,
  narrative identity.)
- Carvalho. *Understanding Personality Stability and Change From
  McAdams's Perspective.* Social and Personality Psychology Compass,
  2025. https://compass.onlinelibrary.wiley.com/doi/abs/10.1111/spc3.70114
- Costa & McCrae. NEO-PI-R / Five-Factor Model.
- Soto & John. *The next Big Five Inventory (BFI-2).* JPSP, 2017.
- Ashton & Lee. HEXACO model (Big 5 + Honesty-Humility).
- Serapio-García et al. *Personality Traits in Large Language Models.*
  2023.
- *LLMs Simulate Big Five Personality Traits: Further Evidence.*
  ACL personalize workshop, 2024. https://arxiv.org/abs/2402.01765
- *The Power of Personality: A Human Simulation Perspective to
  Investigate Large Language Model Agents.* 2025.
  https://arxiv.org/html/2502.20859

### State / mood representation

- Russell. *Core affect and the psychological construction of emotion.*
  Psychological Review, 2003.
- Mehrabian & Russell. *An approach to environmental psychology.*
  1974. (PAD model: Pleasure-Arousal-Dominance.)
- Mehrabian. *Pleasure-arousal-dominance: A general framework for
  describing and measuring individual differences in temperament.*
  Current Psychology, 1996.
  https://link.springer.com/article/10.1007/BF02686918
- Watson, Clark, Tellegen. *Development and validation of brief
  measures of positive and negative affect: the PANAS scales.* JPSP,
  1988.
- Gehm & Scherer. PAD vs PANAS comparison for differentiating affect.
  https://link.springer.com/article/10.1007/BF02229025
- Gebhard. *ALMA: A Layered Model of Affect.* AAMAS 2005.
  https://www.researchgate.net/publication/221455945_ALMA_a_layered_model_of_affect
- *From Affect Theoretical Foundations to Computational Models of
  Intelligent Affective Agents.* MDPI Applied Sciences, 2021.
  https://www.mdpi.com/2076-3417/11/22/10874

### State-trait integration

- Fleeson. *Toward a structure- and process-integrated view of
  personality: traits as density distributions of states.* JPSP, 2001.
  https://pubmed.ncbi.nlm.nih.gov/11414368/
- Fleeson & Jayawickreme. *Whole Trait Theory.* JRP, 2015.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4472377/
- Fleeson & Jayawickreme. *Getting from states to traits: Whole Trait
  Theory's explanatory and developmental engine.* European Journal of
  Personality, 2025.
  https://journals.sagepub.com/doi/10.1177/08902070251366709

### Critique of weaker personality models

- Pittenger. *Measuring the MBTI… and coming up short.* JCPE, 1993.
- Randall et al. *Validity and Reliability of the Myers-Briggs
  Personality Type Indicator: A Systematic Review.* 2017.
  https://gwern.net/doc/psychology/personality/2017-randall.pdf
- Hook et al. *The Enneagram: A systematic review of the literature
  and directions for future research.* JCP, 2021.
  https://pubmed.ncbi.nlm.nih.gov/33332604/
- *Personality Tests Aren't All the Same.* Scientific American.
  https://www.scientificamerican.com/article/personality-tests-arent-all-the-same-some-work-better-than-others/

### Discrete emotion vocabularies (phasic layer)

- Tracy & Robins. *Putting the self into self-conscious emotions: A
  theoretical model.* Psychological Inquiry, 2004. (Pride / shame /
  guilt / embarrassment cluster — see also `emotion.py` docstring.)
- Demszky et al. *GoEmotions.* 2020. (27-label vocabulary that lemon's
  23-label set was derived from.)

### Values / characteristic adaptations

- Schwartz. *Universals in the content and structure of values.* 1992.
  (10 universal values.)
- Deci & Ryan. Self-Determination Theory (autonomy / competence /
  relatedness as basic psychological needs).
- Emmons. *Personal strivings* (goal-based personality).

### Internal references

- `docs/architecture.md` — current pipeline overview
- `docs/memory_architecture.md` — composite scoring, family map (with
  the post-`pride`/`relief` taxonomy update)
- `docs/empathy_research.md` — APTNESS / appraisal-theory background
- `src/storage/state.py` — current `internal_state` definition
- `src/empathy/user_read.py` — current pre-reply read
- `src/empathy/post_exchange.py` — current post-reply state nudge
