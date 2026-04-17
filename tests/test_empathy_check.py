from empathy_check import check_response


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


def test_validation_cascade_two_phrases_passes():
    draft = "I hear you. that makes sense for sure."
    r = check_response("rough night", draft, HIGH_SAD)
    assert "validation_cascade" not in r.failures


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
