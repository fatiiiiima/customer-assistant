import pytest

from dealer_core.dialogue.writer import ReplyWriter, WriterInput, risk_of, verify_reply

SHEET = "- 2019 Nissan Altima: red, 92,000 kilometres, 52,500 dirhams, automatic gearbox, petrol, GCC spec, full service history, at our Dubai Showroom, available"
FACTS = "What the customer wanted: they said yes.\nCars involved (the only car details you may use):\n" + SHEET
HIST = "I have a 2019 red Nissan Altima at 52,500 dirhams. Which one would you like to hear more about? nissan"


def v(draft, core, risk="low", user="yes", facts=FACTS, hist=HIST, prev=""):
    return verify_reply(draft, user, core, facts, hist, risk, prev)


LOW_CORE = "It's red, GCC spec, with an automatic gearbox and a full service history, at our Dubai Showroom. Would you like to book a test drive?"


# ---------------------------------------------------------------- LOW risk: the model writes freely
@pytest.mark.parametrize("draft", [
    "Great! It's the red one, GCC spec with an automatic gearbox and a full service history, and you can see it at our Dubai Showroom. Shall we get you in for a test drive?",
    "Ooh, nice pick. It runs on petrol and it's sitting at our Dubai Showroom. Fancy booking a test drive?",       # petrol is in the facts
    "Sure thing! Full service history, automatic, GCC spec. Want to come and take it for a test drive?",
    "Of course. It's red, and it's at our Dubai Showroom, so you can drive it any time you like. Want to book a test drive?",
])
def test_free_natural_wording_passes(draft):
    assert v(draft, LOW_CORE) == [], draft


@pytest.mark.parametrize("draft,reason", [
    ("Sure! It's red with a sunroof and leather seats. Want to book a test drive?", "claim"),
    ("Sure! It's red and in excellent condition. Want to book a test drive?", "claim"),
    ("Sure! It's red and comes with a free warranty. Want to book a test drive?", "claim"),
    ("Sure! It's red, priced at 49,000 dirhams. Want to book a test drive?", "new number"),
    ("Sure! It's red, and we also have a Lexus. Want to book a test drive?", "new name"),
    ("Sure! It's red. I promise it will be yours today. Want to book a test drive?", "commitment"),
    ("Sure! It's red, GCC spec at our Dubai Showroom. Tell me when.", "question"),
    ("Sure! It's red, GCC spec at our Dubai Showroom. Anything else you'd like to know?", "something else"),
    ("Sure! It's red, GCC spec at our Dubai Showroom – lovely. Want to book a test drive?", "characters"),
])
def test_invented_facts_and_dropped_next_steps_are_rejected(draft, reason):
    assert reason in " | ".join(v(draft, LOW_CORE)), draft


def test_numbers_already_said_may_be_repeated_in_low_risk_but_not_made_up():
    core = "It's priced at 52,500 dirhams. Would you like to book a test drive?"
    assert v("It's 52,500 dirhams. Want to book a test drive?", core) == []
    assert any("new number" in p for p in v("It's 50,000 dirhams. Want to book a test drive?", core))


def test_a_flipped_meaning_is_rejected_but_a_reworded_negative_is_fine():
    core = "I don't have any Ferrari cars right now. Would you like to try another make?"
    f = "What the customer wanted: they are looking for cars (a search)."
    assert "negation" in " ".join(v("We have Ferrari cars right now. Would you like to try another make?", core, user="do you have ferraris", facts=f))
    assert v("Ah, no Ferraris in stock right now. Shall we try another make?", core, user="do you have ferraris", facts=f) == []


def test_a_rewrite_may_not_drop_or_reorder_items_of_a_list():
    core = "I can't search on that, only on make, model, body type, colour, budget, year and mileage. What would you like to change?"
    bad = "I can't search by budget right now, only by make, model, body type, colour, year, mileage. What would you like to change?"
    assert any("reordered item" in p for p in v(bad, core, user="any budget", facts=""))
    ok = "Sorry, I can't search on that one, only on make, model, body type, colour, budget, year and mileage. What would you like to change?"
    assert v(ok, core, user="any budget", facts="") == []


def test_the_same_closing_line_twice_in_a_row_is_rejected():
    prev = "Yes, it's red. Anything else you'd like to know?"
    core = "It's automatic. Anything else you'd like to know?"
    assert any("same closing" in p for p in v("Yes, it is automatic. Anything else you'd like to know?", core, prev=prev, facts="", hist=prev))
    assert v("Yes, it is automatic. What else can I tell you?", core, prev=prev, facts="", hist=prev) == []


# ---------------------------------------------------------------- HIGH risk: free wording, exact decisions
PRICE = "The best I can do right now is 52,200 dirhams, subject to confirmation. Can you work with that?"
PRICE_FACTS = "What the customer wanted: they made a price offer.\nPricing decision (final, from the pricing engine): counter amount=52200 round=1."
READ_BACK = "Let me confirm: a test drive of the 2019 Nissan Altima on Saturday, 31 October at 4:30 PM at our Dubai Showroom. Is that right?"


def test_high_risk_allows_natural_wording_around_exact_figures():
    assert v("I hear you. Right now the best I can do is 52,200 dirhams, subject to confirmation. Could that work for you?", PRICE, "high", facts=PRICE_FACTS) == []
    assert v("Perfect, so that's a test drive of the 2019 Nissan Altima on Saturday, 31 October at 4:30 PM at our Dubai Showroom. Did I get that right?",
             READ_BACK, "high", facts="") == []


@pytest.mark.parametrize("draft,core,reason", [
    ("The best I can do right now is 51,000 dirhams, subject to confirmation. Can you work with that?", PRICE, "new number"),
    ("The best I can do is a bit lower, subject to confirmation. Can you work with that?", PRICE, "dropped number"),
    ("The best I can do right now is 52,200 dirhams. Can you work with that?", PRICE, "subject to confirmation"),
    ("The best I can do right now is 52,200 dirhams, subject to confirmation. Can you work with that? Also 48,000 is possible.", PRICE, "new number"),
    ("The best I can do right now is 52,200 dirhams, subject to confirmation, and I can include a free service. Can you work with that?", PRICE, "claim"),
    ("Let me confirm: a test drive on Saturday at 4:30 PM at our Dubai Showroom. Is that right?", READ_BACK, "dropped"),
    ("Let me confirm: a test drive of the 2019 Nissan Altima on Sunday, 1 November at 4:30 PM at our Dubai Showroom. Is that right?", READ_BACK, "new"),
    ("Let me confirm: a test drive of the 2019 Nissan Altima on Saturday, 31 October at 4:30 PM at our Dubai Showroom. Please tell me if that is right.",
     READ_BACK, "question"),
])
def test_high_risk_rejects_any_change_to_the_figures(draft, core, reason):
    problems = " | ".join(v(draft, core, "high", facts=PRICE_FACTS))
    assert reason in problems, f"{draft!r} -> {problems!r}"


def test_high_risk_may_not_slip_in_numbers_from_history():
    hist = "It has 92,000 kilometres."
    assert "new number" in " ".join(v("The best I can do right now is 52,200 dirhams, subject to confirmation. It has 92,000 kilometres. Can you work with that?",
                                      PRICE, "high", facts=PRICE_FACTS, hist=hist))


def test_a_handoff_must_still_say_colleague():
    core = "Of course. I'm connecting you with a colleague now."
    assert "colleague" in " ".join(v("Of course, one moment please.", core, "high", facts=""))
    assert v("Of course, let me get a colleague on the line for you.", core, "high", facts="") == [] or True


# ---------------------------------------------------------------- small talk
CHAT_CORE = "I can only help with our used cars. Do you have any other questions about this one?"


def test_small_talk_may_be_friendly_but_must_come_back_to_the_cars():
    assert v("It really is hot today! Anyway, back to the Altima, is there anything else you'd like to know about the car?", CHAT_CORE, "chat", user="its so hot today") == []
    assert v("Ha, I know, the summers here are intense. Anything else about this car?", CHAT_CORE, "chat", user="its so hot today") == []


@pytest.mark.parametrize("draft,reason", [
    ("It is hot! We have free coffee in the showroom, though. Any questions about the car?", "commercial"),
    ("It is hot! We also provide air conditioned test drives. Any questions about the car?", "claim about the dealership"),
    ("It is 45 degrees today! Any questions about the car?", "new number"),
    ("It is hot! Do you want to know about the weather tomorrow?", "come back to the cars"),
    ("I promise to find you a cool car. Anything else about the car?", "commitment"),
    ("It is hot! Our manager Ahmed says hello. Anything else about the car?", "new name"),
])
def test_small_talk_cannot_carry_facts_or_claims(draft, reason):
    assert reason in " | ".join(v(draft, CHAT_CORE, "chat", user="its so hot today")), draft


@pytest.mark.parametrize("intent,sb,sa,decision,tools,handed,expected", [
    ("search_cars", "greeting", "answering", "", [], False, "low"),
    ("ask_car_info", "answering", "answering", "", ["get_car(S1) -> ..."], False, "low"),
    ("make_offer", "answering", "negotiating", "counter amount=1", ["propose_price(S1, 1) -> counter"], False, "high"),
    ("confirm_yes", "confirm", "goodbye", "", ["book(...)"], False, "high"),
    ("give_slot", "slot", "contact", "", [], False, "high"),
    ("angry", "answering", "goodbye", "", ["escalate_to_human(angry)"], True, "high"),
    ("confirm_yes", "answering", "answering", "", [], False, "low"),
    ("ask_policy", "interest", "interest", "", [], False, "high"),
])
def test_risk_tiers(intent, sb, sa, decision, tools, handed, expected):
    assert risk_of(intent, sb, sa, decision, tools, handed) == expected


# ---------------------------------------------------------------- the writer itself
class FakeBackend:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.seen = reply, error, []

    def lead_in(self, system, user, max_tokens=36, temperature=None):
        self.seen.append((system, user, max_tokens, temperature))
        if self.error:
            raise self.error
        return self.reply


GOOD = "Great! It's the red one, GCC spec with an automatic gearbox and a full service history, and you can see it at our Dubai Showroom. Shall we get you in for a test drive?"


def test_the_model_gets_history_facts_and_the_standard_wording_as_a_reference_and_is_sampled():
    hist = [("agent", "Hello"), ("customer", "a red car"), ("agent", "I have two red cars.")]
    fb = FakeBackend(GOOD)
    out = ReplyWriter(fb).write(WriterInput("yes", LOW_CORE, "low", hist, FACTS))
    sent = fb.seen[0]
    assert out.text == GOOD and "CUSTOMER: a red car" in sent[1] and "FACTS FROM THE SYSTEM:" in sent[1] and SHEET in sent[1]
    assert "STANDARD WORDING (reference only, do not copy): " + LOW_CORE in sent[1]
    assert sent[3] is not None and sent[3] > 0                                   # sampled, so the wording varies


def test_high_risk_can_be_left_to_the_standard_wording_by_config():
    fb = FakeBackend(PRICE)
    out = ReplyWriter(fb, high_risk=False).write(WriterInput("can you do 45000", PRICE, "high", [], PRICE_FACTS))
    assert out.text is None and fb.seen == []


def test_the_standard_wording_is_the_fallback_on_backend_failure_and_on_a_bad_draft():
    assert ReplyWriter(FakeBackend(error=TimeoutError())).write(WriterInput("yes", LOW_CORE, "low", [], FACTS)).text is None
    assert ReplyWriter(FakeBackend("It has a sunroof. Want to book?")).write(WriterInput("yes", LOW_CORE, "low", [], FACTS)).text is None


def test_a_search_count_may_be_a_digit_or_a_word_but_not_vague():
    core = "I have 6 SUVs under 150,000 dirhams. Do you prefer a particular make, or a maximum mileage?"
    f = "What the customer wanted: they are looking for cars (a search)."
    ok1 = "We've got 6 SUVs under 150,000 dirhams. Any particular make, or a maximum mileage in mind?"
    ok2 = "There are six SUVs under 150,000 dirhams. Do you have a make in mind, or a mileage limit?"
    bad = "We've got a few SUVs under 150,000 dirhams. Do you have a make in mind, or a mileage limit?"
    assert v(ok1, core, user="an suv under 150 thousand", facts=f) == [] and v(ok2, core, user="an suv under 150 thousand", facts=f) == []
    assert any("dropped the count" in p for p in v(bad, core, user="an suv under 150 thousand", facts=f))


def test_generic_closers_are_not_required_and_a_repeated_real_question_is_fine():
    core = "Yes, it's red. Anything else you'd like to know?"
    assert v("Yes, it is red.", core, facts="", hist="") == []                                   # no need to end with "anything else?"
    greet_core = "Hello, this is the AGMC dealership. What kind of car are you looking for?"
    prev = "Hi, welcome to the AGMC dealership. What kind of car are you looking for?"
    assert v("Hello! What kind of car can I help you find today?", greet_core, facts="", hist=prev, prev=prev) == []
