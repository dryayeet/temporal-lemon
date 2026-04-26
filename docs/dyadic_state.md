# Dyadic state architecture

> **Status: proposal under discussion. NOT implemented.**
> Captures the rationale, research grounding, and proposed shape of an
> architectural shift from the current asymmetric internal-state design
> (lemon has tonic mood, user has only per-message phasic reads) to a
> symmetric dyadic-state design where both agents are modelled as the same
> kind of object and response generation is downstream of state.
>
> Author: lemon, in discussion with the project assistant, 2026-04-26.

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

## 6. Concrete schema sketch

### 6.1 Unified tonic state (applies to both lemon and user)

```python
TONIC_STATE = {
    "mood":            <vocab>,        # categorical, see §6.2
    "energy":          "low" | "medium" | "high",
    "engagement":      "low" | "normal" | "deep",
    "current_focus":   str | None,     # what's on their mind right now
    "carried_in":      str | None,     # cross-session continuity
    "stance":          <stance_vocab>, # how they relate to the other agent
}
```

Same shape for both agents. Field semantics may shift slightly per
agent (e.g. `stance` for lemon = `disposition` toward the user; for the
user = how they're showing up to lemon today). Not all fields will be
populated for the user from day one — `mood` and `current_focus` are
the most tractable to infer, the rest can fill in over sessions.

### 6.2 Mood vocabulary

Two viable options, to be picked during design:

**Option A: keep the small mood vocabulary, apply to both.** The
existing 8-value mood set (`neutral | good | low | happy | anxious |
restless | tired | content`) preserves the "subtle nudge" dynamics. The
EMOTION_LABELS set stays separate for *phasic* reads.

**Option B: Russell-circumplex inspired.** Replace mood with a
two-axis low-cardinality scheme: valence × arousal × small label set
(e.g. four quadrants plus neutral: `pleasant_calm`, `pleasant_active`,
`unpleasant_calm`, `unpleasant_active`, `neutral`). More principled,
but loses the folksy readability.

Recommendation when this gets decided: **Option A**. Russell-derived
schemes look clean on paper but in practice the categorical labels are
what the LLM reasons about; quadrant labels lose the prompt-engineering
affordance.

### 6.3 Phasic event (per turn, per agent)

```python
PHASIC = {
    "event_emotion": <EMOTION_LABEL>,   # from existing 23-label set
    "intensity":     float,              # [0, 1]
    "appraisal":     str | None,         # what about the message triggered this
    "trigger":       "self" | "other",   # whose utterance caused it
}
```

For the user: today's `user_read` already produces most of this. For
lemon: a new small read pass, OR an extension of `post_exchange`.

### 6.4 Theory of mind (separate object)

```python
TOM = {
    "feeling":      str | None,    # lemon's read of user's feeling
    "avoid":        str | None,    # what NOT to do
    "what_helps":   str | None,    # what to do
    "confidence":   float,          # NEW: how confident lemon is (for divergence tracking)
}
```

Stays as **lemon's model of user_state**, not the user_state itself.
When ToM and user_state diverge, that's signal: lemon misread, and the
divergence can drive a self-correction loop in future turns.

### 6.5 Where each lives in storage

| object | scope | persistence |
| --- | --- | --- |
| `lemon_tonic_state` | global | `state_snapshots`, today's table |
| `user_tonic_state` | global (per-installation; lemon is single-user) | NEW table or new state-snapshots stream |
| `lemon_phasic` | per turn | NEW column on messages, or separate per-turn log |
| `user_phasic` | per turn | already on `messages` (emotion, intensity, salience) |
| `tom` | per turn | NEW or store with phasic |

For the single-user single-process design lemon currently has, "global"
just means "the one user." If lemon were ever multi-user, all of this
becomes per-user.

---

## 7. Open design questions

These are the decisions that need explicit answers before any code is
written. Some have a default lean; all need confirmation.

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

### 7.3 Where do the LLM round trips go?

Today: 1 pre-reply read (`user_read`), 1 reply call, 1 post-reply call
(`post_exchange` for facts + state).

Naive expansion: add user_state update + lemon pre-reply state nudge =
two more round trips. That's 50% more latency and cost. Untenable.

**Mitigation paths to evaluate:**

- **Extend `user_read`** to also emit a user-tonic-state nudge. One
  call covers phasic + tonic for the user.
- **Move lemon state update into `user_read`** as well. Same call
  emits: user phasic, user tonic nudge, lemon's read of the situation.
  Then reply generation uses all of it. `post_exchange` shrinks to
  just facts + lemon-state confirmation.
- Net result: same number of LLM calls per turn, with more output
  per call. Costs go up modestly (longer JSON output) but not
  proportionally.

### 7.4 How do we infer user tonic state from messages?

Cold-start problem. First session has no prior user state to nudge
from. Defaults from facts? From the very first user_read? From a
lightweight "scan recent messages" pass on session start?

**Lean:** initialise user tonic state from `DEFAULT_USER_STATE` (TBD)
on first contact, then nudge on every read. After ~3-5 turns the noise
should settle. Cross-session, the previous session's end-state is the
new session's start-state.

### 7.5 Folding `<user_emotion>` and `<theory_of_mind>` into one block?

In the new framing, the user's *actual* phasic emotion is a state
object. ToM is *lemon's model* of user state. These could fold into
one ToM-shaped block in the prompt: "what I think they're feeling,
what's been on their mind, what helps." Cleaner prompt surface, fewer
moving parts.

**Lean: yes, fold them**, but keep the underlying objects distinct in
storage. The prompt is a presentation layer.

### 7.6 What gets exposed to lemon in the system prompt?

Today lemon sees `<internal_state>`, `<user_emotion>`, `<theory_of_
mind>`, `<facts>`, `<emotional_memory>`. In the new world the natural
arrangement is:

- `<lemon_state>` — lemon's own tonic state. (renamed from
  `<internal_state>` to make the symmetry obvious)
- `<user_state>` — lemon's model of the user's tonic state (the
  ToM-merged block from §7.5).
- `<event>` — what just happened, with phasic reads. Optional —
  the user's phasic-read may be folded into `<user_state>`.
- `<emotional_memory>`, `<facts>`, `<time_context>` — unchanged.

### 7.7 Schema unification vs minor renaming

Should the user_state schema be *literally identical* to lemon's, with
the same field names, or be a parallel-but-renamed schema (e.g.
`disposition` for lemon, `stance_toward_lemon` for the user)?

**Lean: literally identical.** Renaming feels safer but it makes the
symmetry illegible to the LLM and to future readers. Same field names,
documented clearly.

### 7.8 What does `/why` show?

Currently the `/why` introspection shows the last reply's pipeline
trace. In the new architecture it should show:

- Lemon's tonic state going in
- User's tonic state going in
- Both phasic reads
- ToM (if separate from user_state)
- What changed in each tonic state across the turn
- Memories used
- Empathy check result

This is a much richer debugging surface than today's trace and is one
of the concrete wins from the refactor.

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

## 9. Possible migration order (when this gets approved)

Stage gates so the work is interruptible at each step.

### Stage 1: persistent user_state object

Add `user_state` table or field. Extend `user_read` to emit a tonic-
state nudge alongside the existing phasic emotion + ToM. Persist user
tonic state across turns and sessions. **No reordering of the existing
pipeline yet** — this stage just adds the missing user-side object.

Most of the simulation value comes from this stage alone, because it
finally gives lemon a persistent representation of "where the user
is."

### Stage 2: move lemon's state update pre-reply

Currently lemon's tonic state updates *after* the reply, in
`post_exchange`. Move it *before* the reply, so the response generator
reads from a freshly-updated state. `post_exchange` shrinks to just
facts extraction + optional state confirmation.

This is the "state first, response second" half of the proposal.

### Stage 3: schema unification + prompt-block restructure

Rename `internal_state` → `lemon_state`, introduce `user_state` block
with the same shape, fold `<user_emotion>` + `<theory_of_mind>` into a
single ToM-shaped block. Update `/why`.

This is the cosmetic-but-clarifying stage. Could happen earlier; left
last because it's the one most likely to break tests and prompts that
reference specific block names.

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

## 11. Open thread for follow-up discussion

Decisions that still need your input before any work starts:

1. Stage 1 only, or full three stages?
2. Mood vocabulary — keep small (Option A) or Russell-derived (Option B)?
3. Field name unification — literally identical, or rename for agent
   clarity?
4. Damping levels — same for both agents, or asymmetric (lemon damped
   harder)?
5. Cold-start handling for user_state — defaults from `facts`,
   defaults from the first read, or hardcoded `DEFAULT_USER_STATE`?
6. Should lemon's phasic reaction be modelled at all in stage 1, or
   only added at stage 2?
7. Where do the round trips land (§7.3 has options)?

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

### Foundational psychology referenced

- Russell. *Core affect and the psychological construction of emotion.*
  Psychological Review, 2003.
- Mehrabian & Russell. *An approach to environmental psychology.*
  1974. (PAD model.)
- Watson, Clark, Tellegen. *Development and validation of brief
  measures of positive and negative affect: the PANAS scales.* JPSP,
  1988.
- Tracy & Robins. *Putting the self into self-conscious emotions: A
  theoretical model.* Psychological Inquiry, 2004. (Pride / shame /
  guilt / embarrassment cluster — see also `emotion.py` docstring.)

### Internal references

- `docs/architecture.md` — current pipeline overview
- `docs/memory_architecture.md` — composite scoring, family map (with
  the post-`pride`/`relief` taxonomy update)
- `docs/empathy_research.md` — APTNESS / appraisal-theory background
- `src/storage/state.py` — current `internal_state` definition
- `src/empathy/user_read.py` — current pre-reply read
- `src/empathy/post_exchange.py` — current post-reply state nudge
