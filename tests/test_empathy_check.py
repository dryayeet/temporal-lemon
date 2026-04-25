from empathy.empathy_check import check_response


HIGH_SAD = {"primary": "sadness", "intensity": 0.8}
LOW_SAD = {"primary": "sadness", "intensity": 0.2}
NEUTRAL = {"primary": "neutral", "intensity": 0.1}


# ---------- minimizing ----------

def test_minimizing_at_least():
    r = check_response("my dog died", "at least you had time with them", HIGH_SAD)
    assert not r.passed
    assert "minimizing" in r.failures


def test_minimizing_could_be_worse():
    r = check_response("rough day", "could be worse honestly", HIGH_SAD)
    assert not r.passed
    assert "minimizing" in r.failures


def test_neutral_response_passes():
    r = check_response("how are you", "doing alright, you?", NEUTRAL)
    assert r.passed
    assert r.failures == []


# ---------- toxic positivity ----------

def test_toxic_positivity_silver_lining():
    r = check_response("everything sucks", "look for the silver lining maybe", HIGH_SAD)
    assert "toxic_positivity" in r.failures


def test_toxic_positivity_stay_strong():
    r = check_response("hard week", "stay strong yaar", HIGH_SAD)
    assert "toxic_positivity" in r.failures


# ---------- advice pivot ----------

def test_advice_pivot_when_user_distressed():
    r = check_response("I can't sleep", "have you tried melatonin?", HIGH_SAD)
    assert "advice_pivot" in r.failures


def test_advice_pivot_skipped_when_intensity_low():
    r = check_response("can't sleep tbh", "have you tried melatonin?", LOW_SAD)
    assert "advice_pivot" not in r.failures


def test_advice_pivot_skipped_when_user_neutral():
    r = check_response("any tips for sleep?", "have you tried melatonin?", NEUTRAL)
    assert "advice_pivot" not in r.failures


# ---------- polarity mismatch ----------

def test_polarity_mismatch_haha_to_sadness():
    r = check_response("my grandma passed", "haha that's wild", {"primary": "grief", "intensity": 0.9})
    assert "polarity_mismatch" in r.failures


def test_polarity_mismatch_skipped_when_neutral():
    r = check_response("tell me a joke", "haha okay so", NEUTRAL)
    assert "polarity_mismatch" not in r.failures


# ---------- validation cascade ----------

def test_validation_cascade_three_phrases_triggers():
    draft = "I hear you. that's so valid. your feelings are valid."
    r = check_response("rough night", draft, HIGH_SAD)
    assert "validation_cascade" in r.failures


def test_validation_cascade_two_short_at_top_now_triggers():
    # behavior changed: two stacked validation phrases inside the first 80
    # chars of a reply now reads as robotic, even without a third.
    draft = "I hear you. that makes sense for sure."
    r = check_response("rough night", draft, HIGH_SAD)
    assert "validation_cascade" in r.failures


# ---------- critique aggregation ----------

def test_critique_aggregates_multiple_failures():
    # opens with advice (advice_pivot), has minimizing phrase, has cliche
    r = check_response(
        "I'm devastated",
        "you should try meditation, at least things could be worse, stay strong.",
        HIGH_SAD,
    )
    assert not r.passed
    assert "minimizing" in r.failures
    assert "toxic_positivity" in r.failures
    assert "advice_pivot" in r.failures
    text = r.critique
    assert "Rewrite" in text
    assert text.count("- ") == len(r.critiques)


def test_critique_empty_when_passed():
    assert check_response("hey", "what's up", NEUTRAL).critique == ""


# ---------- minimizing: tightened "at least" ----------

def test_minimizing_at_least_only_at_sentence_start():
    # mid-sentence "at least" should NOT trigger the opener match anymore
    r = check_response("rough day", "honestly i at least tried to sleep early", HIGH_SAD)
    assert "minimizing" not in r.failures


def test_minimizing_at_least_after_punctuation_triggers():
    # post-punctuation counts as a fresh thought opener
    r = check_response("my dog died", "it sucks. at least you had time with them.", HIGH_SAD)
    assert "minimizing" in r.failures


# ---------- toxic positivity: broader silver-linings ----------

def test_toxic_positivity_bright_side():
    r = check_response("everything is hard", "on the bright side, summer's coming", HIGH_SAD)
    assert "toxic_positivity" in r.failures


def test_toxic_positivity_good_news_is():
    r = check_response("got laid off", "the good news is you're free now", HIGH_SAD)
    assert "toxic_positivity" in r.failures


# ---------- validation cascade: 2 in head triggers ----------

def test_validation_cascade_two_at_top_triggers():
    draft = "i hear you. that makes sense. so what's been going on with work and stuff lately"
    r = check_response("rough night", draft, HIGH_SAD)
    assert "validation_cascade" in r.failures


def test_validation_cascade_two_spread_across_long_reply_passes():
    # same two phrases but separated by enough non-validation content that it
    # doesn't read as stacking
    draft = (
        "i hear you. work has been brutal for you for weeks now and it's wearing you out. "
        "and yeah, that makes sense — if anyone in your shoes wouldn't be tired right now."
    )
    r = check_response("rough night", draft, HIGH_SAD)
    assert "validation_cascade" not in r.failures


# ---------- therapy-speak ----------

def test_therapy_speak_sounds_like_anxiety():
    r = check_response("can't stop thinking about it", "that sounds like anxiety", HIGH_SAD)
    assert "therapy_speak" in r.failures


def test_therapy_speak_textbook_trauma():
    r = check_response("i shut down", "that's textbook trauma response", HIGH_SAD)
    assert "therapy_speak" in r.failures


def test_therapy_speak_catastrophizing():
    r = check_response("nothing will ever work out", "you're catastrophizing", HIGH_SAD)
    assert "therapy_speak" in r.failures


def test_therapy_speak_doesnt_fire_on_neutral_phrasing():
    r = check_response("how was your day", "kinda meh, like i've been low energy", NEUTRAL)
    assert "therapy_speak" not in r.failures


# ---------- self-centering ----------

def test_self_centering_i_just_want_you_to_know():
    r = check_response("i'm a mess", "i just want you to know i'm here", HIGH_SAD)
    assert "self_centering" in r.failures


def test_self_centering_i_wish_i_could_fix():
    r = check_response("everything fell apart", "i wish i could fix this for you", HIGH_SAD)
    assert "self_centering" in r.failures


def test_self_centering_skipped_when_neutral():
    # casual "I wish I could help" without distress should NOT trigger
    r = check_response("can you debug this", "i wish i could help with that", NEUTRAL)
    assert "self_centering" not in r.failures


def test_self_centering_skipped_when_low_intensity():
    r = check_response("kinda annoyed today", "i wish i could fix it", LOW_SAD)
    assert "self_centering" not in r.failures


# ---------- sycophancy ----------

def test_sycophancy_great_question():
    r = check_response("what do you think?", "great question! lemme think", NEUTRAL)
    assert "sycophancy" in r.failures


def test_sycophancy_youre_so_right():
    r = check_response("i think i'm overreacting", "you're so right about that", NEUTRAL)
    assert "sycophancy" in r.failures


def test_sycophancy_couldnt_agree_more():
    r = check_response("life is weird sometimes", "couldn't agree more", NEUTRAL)
    assert "sycophancy" in r.failures


def test_sycophancy_doesnt_fire_on_plain_agreement():
    r = check_response("i think i'm overreacting", "yeah, that tracks honestly", NEUTRAL)
    assert "sycophancy" not in r.failures


# ---------- false equivalence ----------

def test_false_equivalence_that_happened_to_me_too():
    r = check_response("my mom yelled at me", "that happened to me too last year", HIGH_SAD)
    assert "false_equivalence" in r.failures


def test_false_equivalence_when_i_went_through_this():
    r = check_response("can't sleep", "when i went through this i tried herbal tea", HIGH_SAD)
    assert "false_equivalence" in r.failures


def test_false_equivalence_been_there_doesnt_trigger():
    # "been there" alone is too common in casual support — not flagged
    r = check_response("rough commute today", "been there honestly", NEUTRAL)
    assert "false_equivalence" not in r.failures


# ---------- lecturing ----------

def test_lecturing_what_you_need_to_realize():
    r = check_response("i feel stuck", "what you need to realize is that you're growing", HIGH_SAD)
    assert "lecturing" in r.failures


def test_lecturing_the_important_thing_is():
    r = check_response("i bombed the interview", "the important thing is you tried", HIGH_SAD)
    assert "lecturing" in r.failures


def test_lecturing_doesnt_fire_on_questions():
    r = check_response("i feel stuck", "what's actually weighing on you the most?", HIGH_SAD)
    assert "lecturing" not in r.failures


# ---------- performative empathy ----------

def test_performative_empathy_my_heart_goes_out():
    r = check_response("dad's in the hospital", "my heart goes out to you", HIGH_SAD)
    assert "performative_empathy" in r.failures


def test_performative_empathy_sending_hugs():
    r = check_response("just bad day", "sending hugs!", HIGH_SAD)
    assert "performative_empathy" in r.failures


def test_performative_empathy_holding_space():
    r = check_response("i'm overwhelmed", "i'm holding space for you", HIGH_SAD)
    assert "performative_empathy" in r.failures


def test_performative_empathy_doesnt_fire_on_normal_warmth():
    r = check_response("i'm overwhelmed", "yeah that sounds like a lot man", HIGH_SAD)
    assert "performative_empathy" not in r.failures


# ---------- question stacking ----------

def test_question_stacking_three_qs_under_distress():
    draft = "what happened? when did it start? have you talked to anyone about it yet?"
    r = check_response("i can't function", draft, HIGH_SAD)
    assert "question_stacking" in r.failures


def test_question_stacking_skipped_when_user_neutral():
    # 3 questions when the user is just chatting is fine, even welcome
    draft = "what'd you eat? how'd it taste? would you make it again?"
    r = check_response("tried that new recipe", draft, NEUTRAL)
    assert "question_stacking" not in r.failures


def test_question_stacking_one_question_passes_under_distress():
    r = check_response("can't sleep", "what's keeping you up tonight?", HIGH_SAD)
    assert "question_stacking" not in r.failures


# ---------- multi-failure aggregation, expanded ----------

def test_multi_failure_includes_new_detectors():
    # therapy-speak + sycophancy + question-stacking in one reply
    draft = (
        "great question! that sounds like classic anxiety. "
        "what triggered it? how long has it been going on? are you sleeping ok?"
    )
    r = check_response("i can't focus on anything", draft, HIGH_SAD)
    assert not r.passed
    assert "sycophancy" in r.failures
    assert "therapy_speak" in r.failures
    assert "question_stacking" in r.failures
