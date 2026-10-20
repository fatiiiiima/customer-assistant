import pytest

from dealer_core.dialogue.leadin import LeadInWriter, verify

CORE = "Of course. What kind of car would you like to see instead?"
USER = "i want another car instead"


@pytest.mark.parametrize("lead", [
    "Sure, happy to look at something different.",
    "No problem at all.",
    "Of course, let's find you something else.",
    "Got it, you'd like another car.",
])
def test_ordinary_acknowledgements_pass(lead):
    assert verify(lead, USER, CORE) == []


@pytest.mark.parametrize("lead,reason", [
    ("Sure, I have a lovely 2021 Toyota Camry for you.", "number"),                      # digits
    ("Sure, we have a Bentley arriving next week.", "outside"),                          # new fact: a car and a claim
    ("Sure, a Lexus RX would suit you.", "outside"),                                     # a car not in the conversation
    ("Sure, I can get you a great discount.", "commercial"),
    ("Sure, I promise we will find one.", "commercial"),
    ("Sure, I will find you something.", "promise"),
    ("Sure, let me check that for you.", "promise"),
    ("Sure, the price is very reasonable.", "outside"),                                  # word outside the allowed set
    ("Sure, what are you looking for?", "statement"),                                    # a question
    ("Sure. Happy to help.", "statement"),                                               # two sentences
    ("Sure, we do not have that one.", "negation"),                                      # a negative claim
    ("Sure, all cars are available today at the showroom in Dubai.", "outside"),
    ("Okay.", "words"),                                                                   # too short
    ("Sure, happy to look at something else for you right now if that is what you would like.", "words"),   # too long
    ("Sure, happy to look at something else", "statement"),                              # no end mark
    ("Sure, it costs about fifty thousand.", "outside"),
])
def test_unsafe_lead_ins_are_rejected(lead, reason):
    problems = " | ".join(verify(lead, USER, CORE))
    assert reason in problems, f"{lead!r} -> {problems!r}"


def test_a_word_the_caller_said_is_allowed_but_the_same_sentence_is_not_repeated():
    assert verify("Sure, an SUV it is.", "an suv please", CORE) == []
    assert "same as the previous" in " ".join(verify("Sure, happy to help.", USER, CORE, previous="sure, happy to help."))


class FakeBackend:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def lead_in(self, system, user):
        self.calls.append(user)
        if self.error:
            raise self.error
        return self.reply[len(self.calls) - 1] if isinstance(self.reply, list) else self.reply


def test_writer_returns_verified_text_or_a_reason():
    ok = LeadInWriter(FakeBackend("Sure, happy to look at something different."), {"search_cars"}).write(USER, CORE)
    assert ok.text and not ok.why_not
    bad = LeadInWriter(FakeBackend("We have a Bentley."), {"search_cars"}).write(USER, CORE)
    assert bad.text is None and bad.why_not
    down = LeadInWriter(FakeBackend(error=TimeoutError()), {"search_cars"}).write(USER, CORE)
    assert down.text is None and "backend" in down.why_not
    junk = LeadInWriter(FakeBackend("   "), {"search_cars"}).write(USER, CORE)
    assert junk.text is None


def test_the_model_is_never_given_prices_or_policy_just_the_caller_and_the_reply():
    fb = FakeBackend("Sure, happy to help.")
    LeadInWriter(fb, {"search_cars"}).write(USER, CORE)
    assert fb.calls[0].startswith("CALLER: ") and "REPLY: " in fb.calls[0]


def test_engine_hybrid_adds_the_lead_in_only_for_eligible_intents_and_falls_back_cleanly():
    from scenarios.conftest import Call
    from dealer_core.dialogue.leadin import LeadInWriter as W

    c = Call({})
    fb = FakeBackend(["Great, nice choice.", "Sure, happy to look at something different."])
    c.app.engine.writer = W(fb, {"search_cars"})
    c.say("do you have a toyota camry")
    r = c.say("i want another car instead")
    assert r.reply.startswith("Sure, happy to look at something different. Of course.") and r.debug.lead_in
    assert not r.debug.violations                                    # three sentences are allowed only when a lead-in is present

    # a price turn is never eligible, even if the writer would happily produce text
    n = len(fb.calls)
    c2 = Call({"how much is it": "ask_lowest_price"})
    c2.app.engine.writer = W(fb, {"search_cars"})
    c2.say("do you have a toyota camry")
    c2.say("how much is it")
    assert len(fb.calls) == n + 1                                    # only the search turn called the writer

    # a rejected lead-in leaves the template untouched and says why in the trace
    c3 = Call({})
    c3.app.engine.writer = W(FakeBackend("We have a Bentley for you."), {"search_cars"})
    c3.say("do you have a toyota camry")
    r3 = c3.say("i want another car instead")
    assert r3.reply == "Of course. What kind of car would you like to see instead?" and "lead-in dropped" in r3.debug.note


@pytest.mark.parametrize("lead", ["Got it, a pickup you say—sure thing.", "Sure – happy to help.", "Sure (happy to help).", "Sure, happy to help \U0001F600."])
def test_odd_characters_are_rejected(lead):
    assert any("characters" in p for p in verify(lead, USER, CORE))
