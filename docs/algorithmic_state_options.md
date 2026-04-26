# Algorithmic alternatives to LLM-inferred state

> **Status: research / planning. Nothing implemented yet.**
> This doc is a survey of which fields in the dyadic-state schema can be
> derived without an LLM call, what the state-of-the-art algorithmic options
> are, and what the recommended migration shape would look like. Read together
> with `docs/dyadic_state.md` (the schema) and `src/empathy/user_read.py` (the
> single merged STATE_MODEL call this doc proposes shrinking).

---

## 0. TL;DR

Of the eight fields the merged read currently produces (mood, PAD, traits,
goals, values, concerns, stance, facts), roughly half can move to a fast
deterministic path with no LLM, and most of the rest can become *hybrid*
(algorithmic produces the signal, LLM only confirms ambiguous cases). The
biggest wins, in order:

| Field | Today | Realistic target | Why |
|---|---|---|---|
| `mood_label` | LLM picks from 10 labels | **Pure algorithm** (PAD coordinate to nearest centroid) | Already a derived view of PAD. Geometric mapping is exact and free. |
| `pad` (pleasure/arousal/dominance) | LLM emits delta | **Lexicon-first, LLM as fallback** | NRC VAD v2 has 55k words rated on the same three axes. Direct match. |
| `facts` | Separate LLM bookkeep call | **Hybrid: spaCy NER + dependency patterns first, LLM only for residue** | Most facts are entity-shaped ("I have a dog named X"). NER catches them deterministically. |
| `goals` / `concerns` | LLM emits add/remove | **Pattern-based extractor first** | "I want to / trying to / worried about / scared of" are high-precision triggers. |
| `traits` (user side) | LLM emits trait_nudges (already capped tiny) | **Lexicon-based feature aggregation over a rolling window** | Big 5 from text is one of the most studied NLP tasks. Per-turn nudge is noise; aggregate window is signal. |
| `values` (Schwartz) | LLM may add per turn | **Personal Values Dictionary (Ponizovskiy et al. 2020) lookup** | Theory-driven dictionary, freely available, validates against self-report. |
| `relational_stance` | LLM picks free-form string | **Algorithmic from PAD + politeness/formality features**, with LLM only for edge cases | Stance is a small categorical space; PAD plus a politeness classifier covers most of it. |
| `tom` (feeling / avoid / what_helps) | LLM | **Stays LLM** (genuinely generative, not classifiable) | No good algorithmic substitute. |
| `phasic emotion` (primary + intensity + undertones) | LLM classifier | **Stays LLM, or NRC EmoLex lookup as a cheaper baseline** | EmoLex covers the categorical space but loses intensity nuance. |

End state: the merged `read_user` shrinks from "produce all four sub-objects"
to "produce only `tom` and possibly `emotion`," with everything else computed
in microseconds from lexicons and patterns. Auxiliary cost per turn drops
from two STATE_MODEL calls to roughly one shorter one (or zero on cache hits
for unchanged user/lemon deltas).

---

## 1. The fields, one by one

### 1.1 `mood_label` — pure algorithm

`mood_label` is currently LLM-picked from a 10-element whitelist (`neutral`,
`calm`, `content`, `happy`, `excited`, `anxious`, `low`, `tense`, `tired`,
`frustrated`). It is a *derived view* of the PAD coordinate the LLM already
emits. There is no information in the label that is not in PAD.

**Algorithm:** for each label, predefine a centroid `(p, a, d)` in PAD space,
then assign the user's current PAD to the nearest centroid by Euclidean
distance. The Mehrabian PAD model conventionally divides the cube into eight
octants by sign of each axis (Hostile, Exuberant, Bored, Anxious, Relaxed,
etc.); the existing 10 labels are a folksier expansion of that grid and map
cleanly to centroids:

```
neutral     ( 0.00,  0.00,  0.00)
calm        ( 0.30, -0.30,  0.10)
content     ( 0.30, -0.10,  0.05)
happy       ( 0.55,  0.20,  0.15)
excited     ( 0.55,  0.55,  0.30)
anxious     (-0.10,  0.40, -0.30)
low         (-0.30, -0.10, -0.10)
tense       (-0.05,  0.40, -0.10)
tired       (-0.20, -0.40, -0.10)
frustrated  (-0.30,  0.30, -0.20)
```

These match the legacy migration map already in `src/storage/lemon_state.py:108`
(`_LEGACY_MOOD_TO_PAD`). The mapping is invertible and stable.

**Action:** replace `mood_label` in the LLM delta schema with a derived
property computed in `apply_delta`. No LLM tokens, no validator branch.

**Risk:** none. The LLM was already picking from a bounded set; the geometry
is just made explicit.

---

### 1.2 PAD core affect — lexicon-first, LLM fallback

The single largest LLM saving is on the per-turn PAD delta, because
**valence / arousal / dominance is exactly what the NRC VAD lexicon scores
words on**, and the same three axes the schema already uses.

#### 1.2.1 NRC VAD lexicon

- v1.0: ~20,000 English words, each rated on V/A/D in [0, 1] by best-worst
  scaling (Mohammad 2018, ACL).
- v2.0 (March 2025): ~55,000 words plus ~10,000 multiword expressions
  (`arxiv.org/abs/2503.23547`). Free for research, simple TSV.

Three columns, one row per term. Lookup is O(1).

#### 1.2.2 Per-message PAD nudge algorithm

```
1. Tokenize the user message (spaCy or regex).
2. For each content word (drop stopwords, optionally lemmatize):
     v_i = NRC_VAD[word].valence    in [0, 1]
     a_i = NRC_VAD[word].arousal    in [0, 1]
     d_i = NRC_VAD[word].dominance  in [0, 1]
3. Aggregate (mean of matched words; ignore unmatched).
4. Re-center to [-1, +1]: x' = 2x - 1.
5. Apply VADER-style heuristic adjusters:
     - negation flip ("not happy" -> negate valence on the next 1-3 tokens)
     - intensifier scaling ("very", "really", "so" -> *1.293 per VADER)
     - downtoner scaling ("kind of", "barely" -> *0.747)
     - punctuation amplifier ("!" -> +0.292 per character on |valence|)
     - emoji table (VADER ships one)
6. Compute a desired-PAD vector for the user.
7. Subtract from current state to get the delta.
8. Clamp to the existing per-turn caps (_PAD_NUDGE_CAP = 0.15).
```

This is essentially "VADER, but on three axes, using NRC VAD scores instead
of VADER's hand-tuned valence-only sentiment lexicon." VADER's heuristics
generalize because they are about *how a sentence's structure modulates the
intensity of an affect signal*; that logic does not care whether the signal
is one-axis valence or three-axis VAD.

#### 1.2.3 Lemon-side PAD nudge

The same pipeline, but on lemon's *draft reply* tokens, with the existing
±0.10 lemon clamp applied. A nice property: PAD becomes the bridge between
"what the user said" and "what lemon wrote" without needing a separate LLM
read.

#### 1.2.4 What this doesn't capture

- **Pragmatic / sarcastic content.** "Oh, *great.*" with NRC VAD scores as
  positive. VADER's punctuation/capitalization rules help but are not
  sarcasm detectors. (LLM still wins on this.)
- **Hinglish / Hindi tokens.** NRC VAD has translations available but quality
  is uneven; for a multilingual chat, expect partial coverage and let the
  LLM fallback fill in.
- **Topic-conditional valence.** "Gym" is positive for some users, negative
  for others. The LLM's context awareness can't be replicated lexically.

**Action:** replace `user_state_delta.pad` with a lexicon computation as the
primary path. Keep the LLM as a fallback when (a) the message has very few
matched words (< 3), (b) the magnitude is below a confidence threshold, or
(c) sarcasm markers are detected (caps + positive words + negation context).

#### 1.2.5 Python landings

- **`vaderSentiment`** (PyPI, MIT). Use as a reference implementation for
  the heuristic layer. Re-use its negation list, intensifier table, emoji
  table verbatim.
- **NRC VAD v2** (free for research; check license for commercial). One
  TSV file, ~3 MB, load once at startup.
- **`NRCLex`** (PyPI) wraps the older NRC EmoLex and is convenient for
  categorical emotion as a side product, but doesn't ship VAD.

---

### 1.3 `facts` — hybrid: NER + patterns, LLM only for residue

The `bookkeep()` call (`src/empathy/post_exchange.py:27`) is its own STATE_MODEL
round-trip dedicated to extracting `{key: value}` facts from the user's
message and lemon's reply. Most facts the LLM extracts in this corpus are
entity-shaped:

- "I live in Bangalore" → `location: Bangalore`
- "my dog is called Pepper" → `pet: dog (Pepper)`
- "I work at Acme" → `employer: Acme`
- "I'm 27" → `age: 27`

These are textbook NER + dependency-relation extraction.

#### 1.3.1 Pipeline

```
1. spaCy en_core_web_sm runs NER + dep-parse. (~10 ms, free.)
2. A small set of rule-based Matcher patterns map (entity, dep-relation,
   trigger verb) tuples to fact keys:
     - PERSON + "I" + verbs {marry, date, see} -> partner
     - GPE  + "I" + verbs {live, stay, moved}  -> location
     - ORG  + "I" + verbs {work, intern}        -> employer
     - TIME / NUM + "I" + age phrases           -> age
     - NORP / LANGUAGE + speaks/from           -> background
     - PRODUCT, lemma "named/called" -> pet/object name
3. Custom Matcher patterns for non-entity facts: hobbies, dietary,
   relationship status, etc. (Lemma + POS-tag patterns, ~20-30 rules.)
4. Whatever is left over (low-coverage residue) goes to the LLM bookkeep
   call -- which is now a much smaller call because most facts were already
   handled.
```

#### 1.3.2 Why this is a real win

- spaCy is local, deterministic, and fast (single-digit ms per message).
- The existing fact validator (`src/empathy/fact_extractor.py`) already
  enforces key/value hygiene; it doesn't care whether the source was an LLM
  or a Matcher rule.
- The bookkeep call can either disappear entirely (small loss in coverage)
  or shrink to a "did we miss anything?" prompt with the candidate facts
  pre-filled (much shorter; benefits from prompt caching).

#### 1.3.3 Python landings

- **`spaCy` 3.x** (`en_core_web_sm`, MIT). NER + dep-parse + Matcher.
- **`extractacy`** (spaCy pipeline component for entity-value extraction;
  optional, useful for "named X" patterns).
- The actual rule set lives next to the existing fact-extractor module.

**Action:** introduce `src/empathy/facts_rules.py` that runs first; the LLM
path becomes a residue cleaner gated on a "did rules find < N facts" check.

---

### 1.4 `current_goals` — pattern-based first

Goals are surface-level intentional statements. They have high-precision
linguistic triggers:

- `I want / I wanna / I'm trying to / I'd like to / I need to / my goal is /
  I'm working on / I plan to / hoping to / aiming to`

Each trigger is a verb cue followed by a verb phrase. Capture the VP
following the cue with the dependency parse, normalize to imperative form,
truncate to the existing 80-char cap, and add to `goal_add`.

```
"I want to get back into running"
  -> trigger "I want to"
  -> VP "get back into running"
  -> goal: "get back into running"
```

Removal is harder algorithmically — you'd need "I quit / I stopped / I gave
up on X" patterns and to fuzzy-match X against the existing goals list.
This is doable but lower-precision; the LLM is still better here. Reasonable
compromise: **pattern-based for `goal_add`, LLM-only for `goal_remove`**.

#### 1.4.1 Coverage caveat

Rule-based extractors break on phrasing variation: "I really should be
getting back into running" matches no trigger. Document this as a known gap;
it just means lemon's tracked goals lag the truth by a few turns until the
user phrases it canonically. The same gap exists today (the LLM is also
imperfect).

#### 1.4.2 Library

Pure spaCy Matcher / DependencyMatcher rules. No new dependency.

---

### 1.5 `concerns` — pattern-based first, same shape as goals

Same approach as goals; different trigger lexicon:

- `worried about / stressed about / scared of / afraid of / anxious about /
  nervous about / can't stop thinking about / on my mind / weighing on me`

Plus the negative-affect emotion classifier output: if the phasic primary is
{anxiety, fear, stress, sadness} above an intensity threshold and the user
mentions a noun phrase, that NP is a concern candidate.

The combined "phrase trigger OR (high-arousal-negative-affect + named NP)"
covers most of what bookkeep currently catches as concerns.

---

### 1.6 `relational_stance` — derived from PAD + politeness/formality

`relational_stance` is currently a free-form short string ("warm and present",
"polite but a little reserved", etc.). It changes infrequently and it is
really a categorical variable in disguise.

#### 1.6.1 Categorize first

Define a small fixed vocabulary, e.g.:

```
warm_present, casual_familiar, neutral_polite, guarded_formal,
distant_cold, playful_teasing, vulnerable_open, irritated_curt
```

#### 1.6.2 Compute from features

Drive the choice from features that *are* algorithmic:

- **PAD coordinate** (already derived).
- **Politeness score** from the `politeness` package (Yeomans et al.;
  `cran.r-project.org/.../politeness`, ports exist in Python). Markers:
  hedges, gratitude, please, indirection.
- **Formality score** (e.g. Pavlick & Tetreault 2016 features: avg word
  length, contraction count, capitalization, presence of slang).
- **First-person pronoun ratio** (vulnerability / openness signal, well
  documented in LIWC research).
- **Question rate, exclamation rate, emoji rate**.

A small decision tree or shallow classifier maps the feature vector to one
of the categorical labels. Train on a few hundred existing
turn-and-LLM-stance pairs from the existing DB; this is small enough to
hand-label or regress against the LLM's outputs.

#### 1.6.3 Why not stay free-form

The free-form string is rarely useful in the prompt anyway — lemon's
prompt only reads it as a soft hint. A category is just as expressive in
that role and is enormously cheaper to compute.

#### 1.6.4 Library

- `politeness` (Python port available; otherwise reimplement a few markers).
- spaCy for tokenization / POS for formality features.

**Action:** swap `relational_stance` from "string the LLM picks" to
"category the rule classifier picks." Keep the underlying schema field as
a string so we don't break stored snapshots; just constrain the value set.

---

### 1.7 `traits` — lexicon aggregation over a rolling window

Stage 1's `_TRAIT_NUDGE_CAP = 0.02` already says traits should barely move
per turn. The LLM is generating a delta inside ±0.02 every turn, which is
genuinely noise-level. Better algorithmic shape:

#### 1.7.1 Method

- Compute a Big 5 *feature vector* from a rolling window of the user's last
  N messages (say N = 50 or "this session + last 7 days"), not from a single
  turn.
- Use the validated linguistic markers from the LIWC personality literature
  (Mairesse et al. 2007; Pennebaker & King 1999; subsequent reviews):

| Trait | Positive markers | Negative markers |
|---|---|---|
| Openness | longer words, tentativity (`perhaps`, `maybe`), insight verbs (`think`, `know`, `consider`) | small concrete words, present-tense verbs |
| Conscientiousness | future tense, achievement words, no negations | discrepancies (`should`, `would`), filler |
| Extraversion | social words (`we`, `friend`, `talk`), positive emotion words, !, exclamations | tentativity, hedges, 1st-person singular |
| Agreeableness | positive emotion, family/affiliation words, fewer articles | swears, anger words, negations |
| Neuroticism | 1st-person singular, negative emotion, anxiety/sadness words | positive emotion, social words |

- Aggregate counts over the window, normalize by token count, z-score
  against population means (LIWC publishes these), pass through `tanh` or
  similar to land in [-1, +1]. Cap movement per session at the existing
  ±0.02 if you want to keep the slow-drift property.

#### 1.7.2 Why not LIWC itself

LIWC is paid software. Open alternatives:

- **`empath`** (Stanford, MIT-licensed) — generates LIWC-style category
  counts; covers many of the same categories.
- **`nrclex`** for affect-related categories.
- Hand-curated lists for the markers above (each is small).
- **Receptiviti** / commercial APIs are overkill and would re-introduce a
  network call.

#### 1.7.3 What stays LLM

Lemon's traits are hardcoded persona constants and should stay that way.
Only the *user's* traits move; this section applies to the user side only.

**Action:** drop `trait_nudges` from the LLM delta. Run a `compute_user_traits()`
job at session end (or every K turns) that recomputes the trait vector from
the rolling window and writes it. Same persistence, different source.

---

### 1.8 `values` — Personal Values Dictionary lookup

The schema already tags values with a Schwartz universal-value category
(`src/prompts/schwartz.py`). The LLM is essentially doing dictionary
lookup-with-context. The literature has the dictionary.

#### 1.8.1 Personal Values Dictionary (Ponizovskiy et al. 2020)

- Theory-driven, validated against self-report on Facebook and essay corpora.
- Maps words and phrases to all 10 Schwartz value categories
  (self-direction, stimulation, hedonism, achievement, power, security,
  conformity, tradition, benevolence, universalism).
- Free, published as supplementary material to the European Journal of
  Personality paper.

#### 1.8.2 Pipeline

```
1. From the user message, find all matched value words (PVD lookup).
2. Group by Schwartz category; each category gets a count.
3. Over a rolling window (multi-turn), tally per-category counts.
4. When a category crosses a threshold AND a specific value-noun phrase
   has appeared (e.g., "honesty", "family", "freedom"), emit a `value_add`
   with `{label: <noun>, schwartz: <category>}`.
```

The current schema's `value_add` slot accepts exactly this shape; the
existing `normalize_value_entry` validator continues to work unchanged.

#### 1.8.3 Tagging

The Schwartz tagging step (`coerce_schwartz`) becomes trivial because
the PVD already provides the category for every matched word.

**Action:** introduce `src/prompts/values_dict.py` with the PVD loaded into
a `dict[word -> schwartz]`. The user-state delta path computes `value_add`
deterministically; the LLM path is dropped or kept as a fallback for
ambiguous noun phrases.

---

### 1.9 What stays LLM

#### 1.9.1 `tom` (theory of mind)

`tom` produces three free-text fields (`feeling`, `avoid`, `what_helps`).
These are genuinely generative — they are not category lookups, they are
tailored sentences. No good algorithmic substitute. **Keep LLM.**

#### 1.9.2 `emotion.primary` + intensity + undertones (phasic event)

Two options:

- **Stay LLM.** Most accurate; today's STATE_MODEL handles it cheaply.
- **NRC EmoLex baseline.** Categorical lookup over the 8-emotion lexicon
  (`anger`, `fear`, `anticipation`, `trust`, `surprise`, `sadness`, `joy`,
  `disgust`). Loses the intensity scalar (you'd have to derive it from
  match-density and arousal), and loses sarcasm awareness. Reasonable as
  a baseline, but the LLM is doing real work here.

Recommendation: **keep LLM for emotion** until the rest of the pipeline is
swapped and we can measure the regression cleanly.

#### 1.9.3 `goal_remove` (and `concern_remove`)

Removal needs reference resolution against the existing list ("I gave that
up" → which goal?). Better as a residual LLM task than as fuzzy patterns.

---

## 2. Recommended end-state architecture

After all the swaps above, the per-turn pipeline looks like:

```
user_msg arrives
  |
  +-- algorithmic pre-read (synchronous, ~10-30 ms, free):
  |     - tokenize / spaCy parse
  |     - NRC VAD lookup -> user pad nudge candidate
  |     - VADER-style heuristic adjusters
  |     - mood_label = nearest centroid in PAD space
  |     - pattern matcher -> goal_add candidates
  |     - pattern matcher -> concern_add candidates
  |     - PVD lookup -> value_add candidates
  |     - politeness/formality features -> stance category
  |     - NER + dep patterns -> facts candidates
  |     (everything tagged with confidence)
  |
  +-- LLM read (single STATE_MODEL call, much smaller):
  |     - input includes the algorithmic candidates
  |     - LLM produces only:
  |         - tom (always)
  |         - emotion (always, until we measure regression)
  |         - residue facts (only what NER missed)
  |         - residue goal_remove / concern_remove
  |         - corrections to algorithmic candidates (low temperature)
  |
  +-- main reply (MAIN_MODEL) using the merged state
  |
  +-- algorithmic post-write:
  |     - same lexicon path on lemon's draft -> lemon pad nudge
  |     - PAD-derived mood_label for lemon
  |
  +-- traits / long-window features (lazy, every K turns or on session end)
```

**Cost shape today (rough):**

- 1 STATE_MODEL call (`read_user`, ~900 max_tokens output)
- 1 STATE_MODEL call (`bookkeep`, ~350 max_tokens output)
- 1 MAIN_MODEL call (the actual reply)

**Cost shape after migration:**

- 1 STATE_MODEL call, much smaller (only `tom` + `emotion` + residue;
  ~250-400 max_tokens output, well within prompt cache reuse)
- 1 MAIN_MODEL call (unchanged)
- (No bookkeep call. The residue-facts piece folds into the read call's
  output budget.)

That's a ~50% drop in auxiliary cost per turn, plus deterministic latency
on the algorithmic fields (no network jitter, no rate-limit risk).

---

## 3. Reliability & failure modes

### 3.1 Lexicon coverage gaps

**Symptom:** user uses words not in NRC VAD / PVD / EmoLex (slang, neologism,
non-English).

**Mitigation:** confidence threshold = `(matched_content_words / total_content_words)`.
Below 0.3, the algorithmic delta is "low confidence" and the LLM call gets
asked to do its own read for that turn. Above 0.3, trust the lexicon.

### 3.2 Sarcasm / pragmatic inversion

**Symptom:** "oh great, another Monday." NRC VAD scores positive on `great`,
the user feels negative.

**Mitigation:** explicit sarcasm markers (`oh`, `great` in caps with
exclamation, smiley + negative context) trigger LLM fallback for that turn.
Imperfect but matches what VADER already does.

### 3.3 Trait drift staleness

**Symptom:** user's life situation changes; trait window includes old data.

**Mitigation:** weight the rolling window by recency (exponential decay,
half-life ~2 weeks) when computing the aggregated trait vector. The
`recency_decay` helper already exists in `src/temporal/decay.py`.

### 3.4 Stance category misclassification

**Symptom:** the small classifier picks `neutral_polite` when the LLM would
have picked `casual_familiar`.

**Mitigation:** stance changes infrequently and is a soft hint to lemon's
prompt; misclassification produces a marginally less-tuned reply, not a
broken one. Acceptable degradation.

### 3.5 "Algorithm and LLM disagree"

**Symptom:** algorithm says `pad.pleasure = -0.4`, LLM in fallback says
`-0.1`.

**Resolution:** when both are computed for the same turn (low-confidence
fallback case), use the LLM. When only the algorithm runs, use the
algorithm. Do not blend.

---

## 4. Migration plan (no code changes here, just sequencing)

A defensible order, smallest blast radius first:

1. **`mood_label` derived from PAD.** Pure refactor, no model change. Drop
   `mood_label` from the LLM delta schema; compute in `apply_delta`.
2. **NRC VAD table loaded; PAD computed alongside the existing LLM delta.**
   *Both* values stored, *only* the LLM value used by the live system.
   Compare offline over a week of conversation history; tune the
   heuristics until algorithmic and LLM PADs correlate at r > 0.7.
3. **Switch primary PAD source to algorithmic; LLM becomes confidence-gated
   fallback.** Keep `read_user` doing emotion + tom + (residue PAD
   correction) only.
4. **NER + Matcher facts pipeline lands in shadow mode.** Same compare-and-
   tune cycle. Then promote; shrink or remove `bookkeep()`.
5. **PVD lookup for `value_add`.** Shadow, then promote.
6. **Pattern-based `goal_add` / `concern_add`.** Shadow, then promote.
7. **Stance categorical classifier.** Shadow, then promote.
8. **Trait rolling-window aggregator.** Lower priority; the per-turn nudge
   was already noise-level.

Each step is independently revertible because the LLM path stays intact
behind a feature flag until promoted.

---

## 5. Dependencies & licensing

| Library / data | License | Footprint | Notes |
|---|---|---|---|
| `vaderSentiment` | MIT | tiny | Use as reference implementation for heuristic layer; can also re-vendor the rules. |
| NRC VAD v2 lexicon | Free for research; commercial requires permission | ~3 MB TSV | Confirm license before shipping commercially. |
| `spaCy` + `en_core_web_sm` | MIT (model is Apache-2.0 compatible) | ~50 MB model | Already a candidate dep; needed for NER + Matcher. |
| `extractacy` (optional) | MIT | tiny | Helpful but not required; rules can be written directly against `Matcher`. |
| `nrclex` (NRC EmoLex wrapper) | MIT | small | Optional, useful as emotion baseline. |
| Personal Values Dictionary | Free, supplementary to Ponizovskiy 2020 | tiny | Direct dictionary file. |
| `empath` | MIT | small | LIWC-style alternative for trait features. |
| `politeness` (Python port) | varies | small | Or hand-roll a few markers from Danescu-Niculescu-Mizil 2013. |

Net add: ~50 MB of model weight (`en_core_web_sm`) plus ~5 MB of lexicons.
No network calls beyond the LLM's own.

---

## 6. Open questions

1. **Multilingual coverage.** lemon is conversational and the user
   sometimes types Hinglish/Hindi. NRC has translations but the quality is
   uneven. Decision: ship algorithmic for English-dominant turns, fall back
   to LLM whenever a non-trivial fraction of tokens are out-of-vocab.
2. **Schwartz values from a single message.** The PVD scores per-word, but
   single-message value detection is noisy. Need to decide on the rolling
   window size (per-session vs per-7-days) before promoting.
3. **Storage of confidence.** Should `user_state_snapshots` rows record the
   *source* of each delta component (LLM vs algorithm) so we can audit
   later? Probably yes; extra column in the snapshot table.
4. **Lemon-side facts.** The bookkeep call also extracts facts from lemon's
   reply. Do we want that? Today it folds in; with NER-first it stays cheap
   either way.
5. **Empirical correlation target before promoting.** What's the
   acceptable r between algorithm and LLM on PAD before we flip the
   default? r = 0.7 is a starting suggestion; this should come out of the
   shadow-mode comparison week.

---

## 7. References

- Mohammad, S. M. (2018). Obtaining Reliable Human Ratings of Valence,
  Arousal, and Dominance for 20,000 English Words. *ACL*.
- Mohammad, S. M. (2025). NRC VAD Lexicon v2: Norms for V/A/D for over 55k
  English Terms. arXiv:2503.23547.
- Mohammad, S. M., & Turney, P. D. (2013). NRC Emotion Lexicon (EmoLex).
- Hutto, C. J., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based
  Model for Sentiment Analysis of Social Media Text. *ICWSM*.
- Mehrabian, A. (1996). Pleasure-arousal-dominance: A general framework for
  describing and measuring individual differences in temperament. *Current
  Psychology*.
- Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use
  as an individual difference. *JPSP*.
- Mairesse, F., Walker, M. A., Mehl, M. R., & Moore, R. K. (2007). Using
  linguistic cues for the automatic recognition of personality in
  conversation and text. *JAIR*.
- Ponizovskiy, V., Ardag, M., Grigoryan, L., Boyd, R., Dobewall, H., &
  Holtz, P. (2020). Development and Validation of the Personal Values
  Dictionary. *European Journal of Personality*.
- Danescu-Niculescu-Mizil, C., Sudhof, M., Jurafsky, D., Leskovec, J., &
  Potts, C. (2013). A computational approach to politeness with
  application to social factors. *ACL*.
- Pavlick, E., & Tetreault, J. (2016). An Empirical Analysis of Formality
  in Online Communication. *TACL*.
- Schwartz, S. H. (1992). Universals in the content and structure of
  values: Theoretical advances and empirical tests in 20 countries.
  *Advances in Experimental Social Psychology*.
