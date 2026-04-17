# Algorithmic Empathy in LLM Chatbots

A research survey for the `lemon` project. Focus: what you can do in **code, training pipeline, decoding loop, retrieval, post-processing, and multi-agent design** to make a chatbot more empathetic *beyond* prompt engineering.

---

## Executive summary

Empathy in a chatbot context is not a single quantity. Psychology decomposes it into **cognitive empathy** (inferring what the user feels and why), **affective empathy** (responses emotionally congruent with the user's state), and **compassionate empathy** (acting in the user's interest). Sharma et al. (2020)'s EPITOME framework operationalizes this for text via three mechanisms: *Emotional Reactions*, *Interpretations*, *Explorations*. Any serious engineering approach has to target these distinct axes — "sound nicer" is not a spec.

Prompt-only approaches plateau predictably. A system prompt biases a prior but doesn't change the reward surface, doesn't give the model a separate emotional-state belief to condition on, and can't see its own failure modes turn by turn. When the base model was trained to respond in a particular register (therapist-neutral, corporate-safe), the prompt gets overridden by the weights underneath. Once you have a good prompt, the next ~30% of empathy gains come from *structural* choices: what you feed the model, how you sample, how you score, what you rewrite.

This document covers five buckets: (1) **training-time** (SFT, RLHF, DPO/KTO, RLAIF, multi-task heads); (2) **inference-time** (classifier-guided decoding, activation steering, best-of-N, two-pass); (3) **architecture/pipeline** (generator+critic, pre-gen emotion classifier, Theory-of-Mind side pass, RAG over empathic exemplars, emotional memory); (4) **post-processing**; (5) **datasets, benchmarks, evaluation**. Section 7 translates this into a concrete priority list for lemon on OpenRouter with no fine-tuning budget.

---

## 1. Training-time approaches

These change the weights. Highest effort, stickiest effect. They compose with everything else.

### 1.1 SFT on empathetic dialogue corpora

Continue-pretrain or instruction-tune on curated human-human empathetic dialogues. Format as `(context, response)` pairs; mask loss on user turns.

Examples: **SoulChat** (Chen et al., EMNLP Findings 2023) fine-tuned ChatGLM on ~1.2M counseling + multi-turn empathy conversations, measurably improving empathy/listening/comfort. **MindChat** (2026, arXiv 2601.01993) combines LoRA + federated optimization + DP on a multi-agent-generated corpus. **MeChat, PsyChat, CPsyCounX, EmoLLM** form a cluster of Chinese/English empathy fine-tunes. The **EmpatheticDialogues** paper (Rashkin et al., ACL 2019, arXiv 1811.00207) showed transformer dialog models trained on their 25k-conversation corpus are perceived as markedly more empathetic than web-scale baselines.

Tradeoffs: LoRA on ESConv runs in hours on one consumer GPU. But (a) empathy corpora are OOD for casual friendship chat — ESConv's helper-seeker structure and PsyQA's long-form QA make models sound clinical; (b) naive SFT teaches surface mimicry ("I'm so sorry you're going through this") rather than understanding.

### 1.2 RLHF with empathy-focused reward models

Train a reward model on pairwise human judgments of "which response is more empathetic", then PPO-optimize with a KL penalty to the SFT init.

Examples: Baihan Lin's *Towards Healthy AI* / SafeguardGPT (arXiv 2304.00416) frames empathy as an RLHF target. By 2025 ~70% of enterprise LLM deployments use RLHF or DPO, empathy/tone often in the reward spec. Decision-Tree-Reward-Gemma-2-27B (Jan 2025) hit SOTA on RewardBench using a modular tree of sub-rewards (helpfulness, empathy, safety).

Tradeoffs: thousands of preference pairs — empathy ratings are especially noisy between raters. PPO is finicky. Reward hacking: model pattern-matches validation phrases rather than empathizing.

### 1.3 DPO / KTO with empathy preference pairs

DPO (Rafailov et al., 2023) re-expresses the RLHF objective as a classification loss over preference pairs — no reward model, no PPO.

- **EmPO** (Sotolář et al., 2024, arXiv 2406.19071) builds preference data from EmpatheticDialogues by pairing completions under intended-emotion vs. polar-opposite-emotion conditions. Substantial gains on standard empathy metrics.
- **Empathy by Design** (2025, arXiv 2512.06097) applies DPO to healthcare dialogue with A+/A- empathic-and-accurate pairs.
- **Distilling Empathy from LLMs** (2025, arXiv 2507.08151) distills teacher-LLM preference judgments into a small student via DPO.

**KTO** (Ethayarajh et al., 2024, arXiv 2402.01306) only needs **binary** desirable/undesirable labels, not pairwise. Empathy is subjective; "was this OK?" is much easier to annotate than "is A more empathic than B?". KTO also handles contradictory rater preferences better than DPO.

Tradeoffs: DPO/KTO are ~10x cheaper and more stable than PPO-RLHF. Still need preference data. Watch over-training past the KL anchor.

### 1.4 Constitutional AI / RLAIF with empathy principles

CAI (Bai et al., Anthropic, arXiv 2212.08073) replaces the human in RLHF with an LLM judging against a written constitution. SFT on self-revised responses, then **RLAIF** — RL against a preference model trained on AI-generated preferences.

Anthropic's original constitution focused on helpful/honest/harmless, but you can write an **empathy constitution**: "acknowledge before advising", "name the feeling", "don't moralize", "ask one question not three", "avoid 'at least' framings". **Chain of Empathy** (Lee et al., 2023, arXiv 2311.04915) gives ready-made psychotherapy-based principles from CBT, DBT, PCT, RT; CBT-based produced the most balanced empathy profile.

Tradeoffs: no human raters, but inherits critic model's blind spots. Works best when critic ≥ generator.

### 1.5 Multi-task learning with emotion classification heads

Auxiliary heads — user-emotion classification, empathy-mechanism prediction, dialogue-strategy prediction — trained jointly with the generative loss.

**MoEL** (Lin et al., EMNLP 2019, arXiv 1908.07687): predicts an emotion distribution, then softly combines outputs of 32 emotion-specific listener decoders. **MIME** (Majumder et al., EMNLP 2020) groups emotions by polarity and mimics. **CEM / Knowledge Bridging** (Sabour et al., AAAI 2022) injects COMET commonsense alongside emotion reasoning.

These are encoder-decoder-era; awkward on decoder-only LLMs. But the idea reappears as the pre-generation classifier pattern (§3.2).

---

## 2. Inference-time approaches

Modify generation without changing weights. Compose with prompting. Often the highest leverage for small teams.

### 2.1 Classifier-guided decoding (PPLM-style)

At each decoding step, nudge logits toward tokens an attribute classifier scores as more empathetic. **PPLM** (Dathathri et al., ICLR 2020) does gradient ascent on hidden states using the classifier loss. **Affective Decoding** (Zheng et al., INLG 2021) applies this to empathy with a dual emotion encoder. Newer: **Controlled Decoding** (Mudgal et al., 2023, arXiv 2310.17022) trains a value function over partial generations. **Diffusion-LM** (Li et al., 2022) supports classifier guidance natively in continuous latent space. **GuidedEmpathy** (Yue et al., 2024) classifies the user's situation into categories before decoding and uses category-specific guidance.

Tradeoffs: requires decoding-loop control — NOT compatible with closed APIs (OpenRouter/Claude). Needs a locally-hosted open-weights model. Grammatical degradation at high guidance strength is the PPLM failure mode.

### 2.2 Activation / representation steering (control vectors)

Find a direction in the residual stream corresponding to "empathy"; add `α · v` to activations at layer L. Build `v` from contrastive pairs by mean activation difference.

**"Detecting and Steering LLMs' Empathy in Action"** (2025, arXiv 2511.16699) operationalizes empathy-in-action (willingness to sacrifice task efficiency for human needs) as a linear direction. AUROC 0.996–1.00 across Phi-3-mini, Qwen2.5-7B, Dolphin-Llama-3.1-8B; steering success 61–94%. **Conditional Activation Steering** (2024, arXiv 2409.05907) adds a trigger classifier so steering only fires on emotionally-loaded turns — important for avoiding a mawkish default register. Foundational: Turner et al., *Steering Language Models With Activation Engineering* (arXiv 2308.10248).

Tradeoffs: requires weight access and mech-interp tooling (nnsight, TransformerLens). But arguably the most surgical lever — vector-add at inference, no retraining.

### 2.3 Best-of-N with an empathy reward model

Generate N completions, score each, return the top. Typical N = 4–16. Scorers: an empathy classifier (Sharma's EPITOME model, or fine-tuned RoBERTa on ESConv strategies), an LLM-as-judge with a decomposed rubric, or a trained reward model.

**Regularized Best-of-N** (Jinnai et al., 2024, arXiv 2404.01054) adds a KL proximity term against reward hacking. **Best of mini-N in-loop** (Hu et al., 2025, arXiv 2510.04087) uses an early-exit threshold so you typically pay for only 2–3 samples.

Tradeoffs: **works with closed APIs** — the #1 method for OpenRouter/Claude setups. Cost scales linearly in N. A naive judge learns to love "I hear you, that's so valid" — decompose the rubric.

### 2.4 Two-pass generation / speculative re-ranking

- **Generate → Critique → Revise.** Draft, then critic answers "is this empathic? what's missing?", then revise.
- **Generate → Rerank.** Produce N responses under different persona conditions (supportive / curious / grounded friend); rerank with a judge.
- **Draft → Polish.** Cheap model drafts; strong model polishes for tone.

2–3x latency. Fine for a CLI where human read-time dominates. Composes with multi-agent (§3.1).

---

## 3. Architecture / pipeline approaches

Highest leverage for teams that can't modify weights. You reshape *what the model is asked to do*.

### 3.1 Multi-agent: generator + empathy-critic + reviser

Three specialized roles: **Generator** drafts; **Empathy critic** scores on EPITOME dimensions and flags failures ("minimizes", "pivots to advice too early", "doesn't acknowledge"); **Reviser** rewrites.

Precedents: **SafeguardGPT** (Lin, arXiv 2304.00416) uses Chatbot / User-simulator / Therapist / Critic. **CRITIC** (Gou et al., ICLR 2024) generalizes tool-interactive self-critique. Table-Critic (2025) reports 9.6% error correction with only 0.7% degradation — critic-reviser is relatively safe.

Design rules: critic ≥ generator capability; critic must have a narrow **objective behavioral rubric** ("did it name a specific feeling?") not gestalt "is this empathic?" — judges are much more reliable on discrete behaviors (§6).

### 3.2 Pre-generation user-emotion classifier

A classifier tags the user's message; its output becomes explicit side-channel context.

Classifier options:
- Small fine-tuned: `j-hartmann/emotion-english-distilroberta-base`, `SamLowe/roberta-base-go_emotions` (27 GoEmotions labels), `mrm8488/t5-base-finetuned-emotion`.
- Cheap LLM call (Haiku, GPT-4o-mini) with structured output.
- **EmoBERTa-X** (2025) for multi-label fine-grained.

Inject into prompt as `<user_state>{primary: "sadness", intensity: 0.78, undertones: ["resentment"]}</user_state>`. Without it, the model misses low-signal emotional cues in factual-sounding messages ("anyway, the interview didn't go great").

Precedents: MoEL (§1.5) did this in-model. **APTNESS** (Yang et al., CIKM 2024, ACM 10.1145/3627673.3679687) tags seven **appraisal-theory** dimensions (novelty, pleasantness, goal-relevance, agency, norm-compatibility, control, certainty) — psychologically richer than raw emotion labels and maps onto Scherer / Ortony-Clore-Collins computational emotion models.

### 3.3 Theory-of-Mind scaffolding

Intermediate LLM pass modeling the user's mental state before the main response. Call: *"What is the user actually feeling? What are they NOT saying? What would make them feel understood vs. dismissed?"* — pass output as context.

Precedents: **ToMAgent** (Chen et al., 2025, arXiv 2509.22887) pairs ToM with dialogue lookahead. **DynToM** (Xu et al., 2025, arXiv 2505.17663) shows SOTA LLMs underperform humans by 44.7% on dynamic mental-state tracking — explicit scaffolding closes a real gap. **TimeToM** reports +19–44% gains by building per-character belief timelines. **Chain-of-Empathy** (Lee et al., 2023, arXiv 2311.04915) is ToM scaffolding via psychotherapy reasoning chains. See also the 2025 ToM survey (arXiv 2502.06470).

ToM output should be **structured JSON** (`primary_emotion`, `underlying_need`, `what_would_help`, `what_to_avoid`), not free text, to prevent drift into the response.

### 3.4 Retrieval-augmented generation from empathic exemplars

At runtime, retrieve K (context, high-empathy response) pairs and include as few-shot. Follow Majumder's recipe: retrieve from a **different** conversation with **matching emotion category** so the model gets stylistic cues without copying.

Precedents: **Exemplars-guided Empathetic Response Generation** (Majumder et al., 2021, arXiv 2106.11791). **K-ESConv** (Deng et al., 2023, arXiv 2312.10371) retrieves from PsyQA. **E3RG** (2025, arXiv 2508.12854) separates emotion-retrieval from response-retrieval. **TOOL-ED** (2024, arXiv 2412.03096) treats retrieval as a tool call.

Why: the default empathic voice is shaped by pretraining, which overrepresents therapist-neutral and corporate-PR registers. Retrieved peer-support exemplars pull it toward the real register.

### 3.5 Memory architectures tracking emotional history

Persistent store of the user's emotional trajectory across sessions. Each turn retrieves relevant memories; each session ends with a summary.

Schema: `(session_id, turn_id, ts, user_msg, bot_msg, detected_emotion, intensity, topic, salience)`. Retrieval: last K turns verbatim, plus any past turn with matching emotion/topic in a recent window.

Precedents: **LoCoMo** (Maharana et al., 2024, arXiv 2402.17753) benchmarks very-long-term memory. **REMT** (Frontiers 2026) organizes memory as a graph of emotionally-valenced nodes. **DABench / TheraMind** (2025, arXiv 2510.25758) evaluates long-term affective memory. **"Dynamic Affective Memory Management"** (2025, arXiv 2510.27418) weights memories by emotional significance.

For lemon: SQLite + FTS5 + embeddings is more than enough. Salience heuristic = strong affect + novel topic + explicit emphasis markers ("I never told anyone this").

---

## 4. Post-processing approaches

Lightest-touch. Safety net applied after main generation.

**Empathy-classifier filter + retry.** Score the draft with an EPITOME classifier; if below threshold, regenerate with a note on what was missing. Catches clear failures like "That's rough, anyway here's the info."

**Response rewriting models.** A small dedicated model (fine-tuned Flan-T5 or Mistral-7B) rewrites any response in empathic style while preserving content. The Distilling Empathy work (2025, arXiv 2507.08151) is effectively a recipe for this.

**Sentiment-mirror checks.** Deterministic patterns:
- Polarity mismatch (user sad → response cheerful).
- Minimizing phrases ("at least", "could be worse", "everyone goes through this").
- Advice-pivot before acknowledgment.
- Factual-answer-after-disclosure (user shares something painful, response ignores it).

Implement in layers by cost: regex → small sentiment model → LLM judge.

---

## 5. Datasets & benchmarks

| Dataset | Size | Language | Domain | Link |
|---|---|---|---|---|
| **EmpatheticDialogues** (Rashkin et al., 2019) | 25k convs | English | open-domain emotional situations | [arXiv 1811.00207](https://arxiv.org/abs/1811.00207), [HF](https://huggingface.co/datasets/facebook/empathetic_dialogues) |
| **ESConv** (Liu et al., 2021) | 1,300 convs, 8 strategies | English | emotional support | [arXiv 2106.01144](https://arxiv.org/abs/2106.01144), [GitHub](https://github.com/thu-coai/Emotional-Support-Conversation) |
| **PsyQA** (Sun et al., 2021) | 22k QA pairs | Chinese | long-form mental health | thu-coai |
| **EPITOME corpus** (Sharma et al., 2020) | 10k pairs, 3-dim empathy | English | Reddit/TalkLife | [arXiv 2009.08441](https://arxiv.org/abs/2009.08441), [GitHub](https://github.com/behavioral-data/Empathy-Mental-Health) |
| **ESCoT** (2024) | interpretable ESConv + CoT | English | emotional support | [arXiv 2406.10960](https://arxiv.org/html/2406.10960) |
| **CPsDD** (2025) | Chinese Psy Support | Chinese | real-world | [arXiv 2507.07509](https://arxiv.org/html/2507.07509) |
| **LoCoMo** (2024) | very long-term memory eval | English | long conversations | [arXiv 2402.17753](https://arxiv.org/abs/2402.17753) |
| **GoEmotions** (Demszky et al., 2020) | 58k Reddit, 27 labels | English | emotion classification | Google |
| **DailyDialog** | 13k convs, 7 emotions | English | daily chit-chat | classic |

For preference data for DPO/KTO on empathy: **EmPO** (2024) and empathy-tagged subsets of UltraFeedback are usable.

---

## 6. Evaluation

### Automated metrics
- **EmotionACC** — does the generated response match the expected emotion (needs a classifier).
- **Empathy Identification** — Sharma's EPITOME classifier scores 0/1/2 on each of Emotional Reaction, Interpretation, Exploration.
- **BLEU / ROUGE / BERTScore** — generic overlap; weak for empathy but commonly reported.
- **Distinct-n** — catches the mode-collapse failure ("I'm sorry, that sounds hard" as every answer).
- **Strategy accuracy** for ESConv-style — does the model pick the right support strategy at the right time?
- **Sentlink / Emosight / NEmpathySort** (Performance Evaluation Metrics for Empathetic LLMs, MDPI 2025) — sentiment-level, fine-emotion-level, and naturalness submodules.

### Human evaluation
Rate 1–5 on: Empathy, Relevance, Fluency, Acknowledgment (does it name/mirror the feeling?), Non-judgment (no "should"s), Advice-timing, and for lemon specifically **perceived friend-likeness / warmth**.

### LLM-as-judge
Nature Machine Intelligence (2025, doi s42256-025-01169-6) found LLMs are reliable judges of empathic communication in the same contexts where experts are reliable — i.e., on **objective behaviors** ("did the response name a feeling?") but not on abstract constructs. **Decompose first**; don't ask "rate empathy 1–5" as a one-shot.

---

## 7. Practical recommendations for `lemon`

lemon is a CLI/web chatbot on OpenRouter (closed-weight models), 1 developer, no fine-tuning budget. Opinionated prioritized list:

### Tier 1 — do now, cheap + high impact

**1. Pre-generation emotion classifier.** Call Haiku / GPT-4o-mini (or `roberta-base-go_emotions` locally for zero latency) on the user's last message. Get structured `{primary_emotion, intensity, underlying_need}`. Inject into the system prompt as side-channel context. ~$0.0001/turn, meaningfully changes what the main model attends to.

**2. Theory-of-Mind side pass.** One extra call before the main response: *"3 bullets — what is the user actually feeling, what do they not want to hear, what would make them feel understood?"* Pass as context. This is Chain-of-Empathy by another name, and DynToM (2025) data suggests it closes a lot of the cognitive-empathy gap. One extra round-trip — use a cheap model.

**3. SQLite emotional memory.** Schema `(session_id, turn_id, ts, user_msg, bot_msg, detected_emotion, intensity, salience)`. At turn start, retrieve (a) last 3 turns, (b) past-30-day turns with matching emotion or topic. Summarize at session end. Most empathy failures are just forgetting context.

**4. Sentiment-mirror post-check.** Regex for minimizing phrases, advice-pivot detectors, polarity mismatch. On trigger, silently regenerate with a note. Zero ML, catches 20% of the worst failures.

### Tier 2 — when you have real users + feedback data

**5. Best-of-N with empathy judge.** N=3–4, small-LLM judge with a decomposed EPITOME rubric (named a feeling? asked an exploring question? avoided premature advice?). Trigger only when intensity > 0.6.

**6. Multi-agent critic-reviser** for hard turns only (distress flagged, or user says "I need to talk"). Off by default for latency.

**7. RAG over empathic exemplars.** Embed EmpatheticDialogues + ESConv with `bge-small` / `nomic-embed-text`. Retrieve 2–3 matching-emotion exemplars from *different* conversations; include as few-shot.

### Tier 3 — serious / differentiation moves

**8. DPO or KTO fine-tune** of a small open model (Qwen2.5-7B, Mistral-7B, Gemma-2-9B) on EmPO-style data. ~$50 on a rented A100. Deploy as generator or as a dedicated rewriter post-processor.

**9. Activation steering** on (8). Build an empathy control vector from contrastive pairs. Ship as a `--warmth` knob with conditional steering (fires only on emotionally-loaded turns).

**10. Constitutional AI loop** for your own preference data. Write a 10-principle empathy constitution. Use Claude to generate (response, revised) pairs. Train DPO on revisions. Scalable empathy-preference pipeline, zero human labeling.

### Anti-patterns to avoid

- **Don't** fine-tune on ESConv directly if you want "friend" not "counselor" — the helper-seeker structure will clinicize the model.
- **Don't** let the empathy layer produce validation cascades ("I hear you, that's so hard, it makes sense, your feelings are valid..."). Monitor Distinct-n.
- **Don't** trust gestalt 1–5 LLM-as-judge scores. Decompose into yes/no behaviors.
- **Don't** inject the emotion classifier label as text to paraphrase — models will label-drop ("I can see you're feeling sad..."). Feed as structured side-channel context.

### End-to-end loop for lemon

```
user_msg arrives
  │
  ▼
[emotion_classifier]   ──►  {primary: "lonely", intensity: 0.7, need: "feel heard"}
  │
  ▼
[memory_retrieval]     ──►  last 3 turns + 2 relevant past moments
  │
  ▼
[tom_pass]             ──►  3 bullets: feeling / avoid / help
  │
  ▼
[main_generation]      ──►  draft
  │
  ▼
[sentiment_mirror_check] ─ pass? ─►  emit
  │                                    ▲
  └── fail ──► [regenerate with note] ─┘

optional: wrap [main_generation] in best_of_N + empathy_judge when intensity > 0.6
```

Implementable in a weekend on OpenRouter. Should reach most of the ceiling of what's possible without fine-tuning.

---

## References

- Rashkin, H., Smith, E. M., Li, M., & Boureau, Y.-L. (2019). *Towards Empathetic Open-domain Conversation Models: a New Benchmark and Dataset.* ACL. [arXiv:1811.00207](https://arxiv.org/abs/1811.00207) · [GitHub](https://github.com/facebookresearch/EmpatheticDialogues) · [HF dataset](https://huggingface.co/datasets/facebook/empathetic_dialogues)
- Sharma, A., Miner, A. S., Atkins, D. C., & Althoff, T. (2020). *A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support* (EPITOME). EMNLP. [arXiv:2009.08441](https://arxiv.org/abs/2009.08441) · [GitHub](https://github.com/behavioral-data/Empathy-Mental-Health)
- Liu, S., Zheng, C., Demasi, O., et al. (2021). *Towards Emotional Support Dialog Systems* (ESConv). ACL. [arXiv:2106.01144](https://arxiv.org/abs/2106.01144) · [GitHub](https://github.com/thu-coai/Emotional-Support-Conversation)
- Lin, Z., Madotto, A., Shin, J., Xu, P., & Fung, P. (2019). *MoEL: Mixture of Empathetic Listeners.* EMNLP. [arXiv:1908.07687](https://arxiv.org/abs/1908.07687) · [GitHub](https://github.com/HLTCHKUST/MoEL)
- Majumder, N., Hong, P., Peng, S., et al. (2020). *MIME: MIMicking Emotions for Empathetic Response Generation.* EMNLP.
- Sabour, S., Zheng, C., & Huang, M. (2022). *CEM / Knowledge Bridging for Empathetic Dialogue Generation.* AAAI. [PDF](https://cdn.aaai.org/ojs/21347/21347-13-25360-1-2-20220628.pdf)
- Majumder, N., et al. (2021). *Exemplars-guided Empathetic Response Generation.* [arXiv:2106.11791](https://arxiv.org/abs/2106.11791)
- Zheng, C., et al. (2021). *Affective Decoding for Empathetic Response Generation.* INLG. [ACL Anthology](https://aclanthology.org/2021.inlg-1.37/)
- Dathathri, S., et al. (2020). *PPLM: Plug and Play Language Models.* ICLR.
- Mudgal, S., et al. (2023). *Controlled Decoding from Language Models.* [arXiv:2310.17022](https://arxiv.org/pdf/2310.17022)
- Li, X. L., et al. (2022). *Diffusion-LM Improves Controllable Text Generation.* [PDF](https://xiangli1999.github.io/pdf/diffusion-lm.pdf)
- Yue et al. (2024). *GuidedEmpathy.* CSAI. [ACM](https://dl.acm.org/doi/10.1145/3709026.3709029)
- Bai, Y., et al. (Anthropic, 2022). *Constitutional AI: Harmlessness from AI Feedback.* [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) · [Anthropic](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback)
- Rafailov, R., et al. (2023). *Direct Preference Optimization.*
- Ethayarajh, K., et al. (2024). *KTO: Model Alignment as Prospect Theoretic Optimization.* [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)
- Sotolář, O., et al. (2024). *EmPO: Emotion Grounding for Empathetic Response Generation through Preference Optimization.* [arXiv:2406.19071](https://arxiv.org/html/2406.19071v2)
- *Empathy by Design: Aligning LLMs for Healthcare Dialogue* (2025). [arXiv:2512.06097](https://arxiv.org/html/2512.06097)
- *Distilling Empathy from LLMs* (2025). [arXiv:2507.08151](https://arxiv.org/html/2507.08151)
- *Chain of Strategy Optimization for Emotional Support* (2025). [arXiv:2503.05362](https://arxiv.org/pdf/2503.05362)
- Lee, Y., et al. (2023). *Chain of Empathy.* [arXiv:2311.04915](https://arxiv.org/abs/2311.04915)
- Chen, Y., et al. (2023). *SoulChat: Improving LLMs' Empathy, Listening, and Comfort.* EMNLP Findings. [OpenReview](https://openreview.net/forum?id=wwm55qcNdK) · [GitHub](https://github.com/scutcyr/SoulChat2.0)
- *MindChat* (2026). [arXiv:2601.01993](https://arxiv.org/abs/2601.01993)
- Lin, B. (2023). *Towards Healthy AI / SafeguardGPT.* [arXiv:2304.00416](https://arxiv.org/pdf/2304.00416)
- Gou, Z., et al. (2024). *CRITIC: LLMs Can Self-Correct with Tool-Interactive Critiquing.* ICLR.
- *Detecting and Steering LLMs' Empathy in Action* (2025). [arXiv:2511.16699](https://arxiv.org/abs/2511.16699)
- *Conditional Activation Steering in LLMs* (2024). [arXiv:2409.05907](https://www.emergentmind.com/papers/2409.05907)
- Turner, A., et al. (2023). *Steering Language Models With Activation Engineering.* [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
- Jinnai, Y., et al. (2024). *Regularized Best-of-N Sampling.* [arXiv:2404.01054](https://arxiv.org/html/2404.01054v1)
- Hu, Y., et al. (2025). *Best of mini-N in-loop Sampling.* [arXiv:2510.04087](https://arxiv.org/html/2510.04087)
- Chen, H., et al. (2025). *Infusing Theory of Mind into Socially Intelligent LLM Agents* (ToMAgent). [arXiv:2509.22887](https://arxiv.org/abs/2509.22887)
- Xu, Y., et al. (2025). *Towards Dynamic Theory of Mind* (DynToM). [arXiv:2505.17663](https://arxiv.org/html/2505.17663)
- *A Survey of Theory of Mind in LLMs* (2025). [arXiv:2502.06470](https://arxiv.org/html/2502.06470v1)
- Maharana, A., et al. (2024). *Evaluating Very Long-Term Conversational Memory* (LoCoMo). [arXiv:2402.17753](https://arxiv.org/abs/2402.17753)
- *Dynamic Affective Memory Management for Personalized LLM Agents* (2025). [arXiv:2510.27418](https://arxiv.org/html/2510.27418v1)
- *TheraMind: Strategic and Adaptive Longitudinal Psychological Counseling* (2025). [arXiv:2510.25758](https://arxiv.org/html/2510.25758)
- *REMT: Simulated Empathy to Structural Attunement.* Frontiers (2026). [Frontiers](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1749517/full)
- Yang, Z., et al. (2024). *APTNESS: Appraisal Theory + Emotion Support Strategies.* CIKM. [ACM](https://dl.acm.org/doi/10.1145/3627673.3679687)
- *Appraisal-Based Chain-of-Emotion for Affective LM Game Agents* (2024). [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11086867/)
- Deng, Y., et al. (2023). *K-ESConv.* [arXiv:2312.10371](https://arxiv.org/html/2312.10371)
- *ESCoT: Interpretable Emotional Support Dialogue Systems* (2024). [arXiv:2406.10960](https://arxiv.org/html/2406.10960)
- *E3RG: Explicit Emotion-driven Empathetic Response Generation* (2025). [arXiv:2508.12854](https://arxiv.org/pdf/2508.12854)
- *TOOL-ED* (2024). [arXiv:2412.03096](https://arxiv.org/pdf/2412.03096)
- *Emotional Support with LLM-based Empathetic Dialogue Generation* (2025). [arXiv:2507.12820](https://arxiv.org/abs/2507.12820)
- *Large Language Models and Empathy: Systematic Review.* JMIR (2024). [JMIR](https://www.jmir.org/2024/1/e52597)
- *When LLMs are reliable for judging empathic communication.* Nature Machine Intelligence (2025). [Nature](https://www.nature.com/articles/s42256-025-01169-6)
- *Performance Evaluation Metrics for Empathetic LLMs.* MDPI Information (2025). [MDPI](https://www.mdpi.com/2078-2489/16/11/977)
- *Survey on Recent Advancements in Human-Centered Dialog Systems.* ACM Computing Surveys (2025). [ACM](https://dl.acm.org/doi/10.1145/3729220)
- *CPsDD: Real-World Chinese Psychological Support Dialogues* (2025). [arXiv:2507.07509](https://arxiv.org/html/2507.07509)
- Hugging Face models: `j-hartmann/emotion-english-distilroberta-base`, `SamLowe/roberta-base-go_emotions`, `mrm8488/t5-base-finetuned-emotion`, `llm-blender/PairRM`.
