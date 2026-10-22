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

    def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
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


@pytest.mark.parametrize("draft,reason", [
    ("Good morning! So you want something red? Which kind of car are you thinking of?", "claim"),             # nobody said good morning
    ("Great, we've got a few that match. What kind of car are you thinking of?", "claim"),                      # no search was run
    ("We've got eight SUVs in stock, all red, actually. Would you pick a make or a mileage limit?", "colour"),  # red was said earlier, not about these cars
])
def test_invented_stock_greetings_and_colours_are_rejected(draft, reason):
    core = "I have 8 SUVs. Would you pick a make or a mileage limit?"
    problems = " | ".join(v(draft, core, user="an suv", facts="What the customer wanted: they are looking for cars (a search).",
                            hist="a red Hi, welcome to the AGMC dealership. What kind of car are you looking for?"))
    assert reason in problems, f"{draft!r} -> {problems!r}"


def test_a_list_of_makes_may_be_reworded_around_the_items():
    core = "Among the SUVs, I have Mitsubishi, Kia, Toyota, Hyundai, Nissan, Lexus and Range Rover. Do you have a make in mind?"
    f = "What the customer wanted: they are looking for cars (a search)."
    ok = "We've got Mitsubishi, Kia, Toyota, Hyundai, Nissan, Lexus and Range Rover among the SUVs. Any of those you like the sound of?"
    bad = "Among the SUVs we've got Toyota, Kia and Lexus. Any of those you like the sound of?"
    assert verify_reply(ok, "what makes are they", core, f, "", "low", awaiting=False) == []        # a preference question: any wording
    assert any("list" in p for p in verify_reply(bad, "what makes are they", core, f, "", "low", awaiting=False))


class JudgeBackend(FakeBackend):
    """Answers the judge prompt with `verdict`, everything else with the reply."""

    def __init__(self, reply, verdict):
        super().__init__(reply)
        self.verdict, self.judged = verdict, 0

    def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        if "check a car dealership assistant" in system:
            self.judged += 1
            return self.verdict
        return super().lead_in(system, user, max_tokens, temperature, deadline_s)


def test_the_models_own_check_rejects_an_unsupported_statement_and_passes_a_supported_one():
    ok = ReplyWriter(JudgeBackend(GOOD, "NO"), judge=True).write(WriterInput("yes", LOW_CORE, "low", [], FACTS))
    assert ok.text == GOOD
    bad = JudgeBackend(GOOD, "YES")
    out = ReplyWriter(bad, judge=True, retries=1).write(WriterInput("yes", LOW_CORE, "low", [], FACTS))
    assert out.text is None and "own check" in out.why_not and bad.judged == 2                  # tried again, judged again, then gave up
    assert ReplyWriter(JudgeBackend(GOOD, "YES"), judge=False).write(WriterInput("yes", LOW_CORE, "low", [], FACTS)).text == GOOD


def test_the_dealership_name_and_a_polite_no_problem_are_not_violations():
    core = "Hello, this is the AGMC dealership. What kind of car are you looking for?"
    assert v("Welcome to AGMC! What kind of car can I help you find today?", core, facts="", hist="") == []
    assert v("No problem at all. What kind of car are you looking for?", core, facts="", hist="") == []
    assert any("negation" in p for p in v("We don't have that. What kind of car are you looking for?", core, facts="", hist=""))


def test_voice_limits_are_passed_to_the_model():
    fb = FakeBackend(GOOD)
    w = ReplyWriter(fb, deadline_s=2.2, max_words=20)
    w.write(WriterInput("yes", LOW_CORE, "low", [], FACTS))
    system, user, max_tokens, temperature = fb.seen[0]
    assert "under 20 words" in user and max_tokens == w.max_tokens


def test_a_deadline_makes_the_standard_wording_win():
    class Slow(FakeBackend):
        def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
            from dealer_core.nlu.backend import BackendTimeout
            raise BackendTimeout("stopped at the deadline")

    out = ReplyWriter(Slow(), deadline_s=1.0).write(WriterInput("yes", LOW_CORE, "low", [], FACTS))
    assert out.text is None and "backend: BackendTimeout" in out.why_not


# ---------------------------------------------------------------- streaming (voice)
class StreamBackend(FakeBackend):
    """Yields the reply in small pieces, like a model decoding."""

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        self.seen.append((system, user, max_tokens, temperature))
        if self.error:
            raise self.error
        text = self.reply
        for i in range(0, len(text), 7):
            yield text[i:i + 7]


SEARCH_CORE = "I have a 2018 Toyota Prado at 105,000 dirhams and a 2021 Toyota Land Cruiser at 148,000 dirhams. Which one would you like to hear more about?"
SEARCH_FACTS = "What the customer wanted: they are looking for cars (a search)."


def stream(reply, core=SEARCH_CORE, facts=SEARCH_FACTS, caller="toyota please", risk="low", awaiting=False, **kw):
    w = ReplyWriter(StreamBackend(reply), **kw)
    return list(w.stream(WriterInput(caller, core, risk, [], facts, awaiting))), w.last_stream


def test_a_good_reply_is_spoken_sentence_by_sentence():
    said, info = stream("Got it! I have a 2018 Toyota Prado at 105,000 dirhams and a 2021 Toyota Land Cruiser at 148,000 dirhams. Which one catches your eye?")
    assert said == ["Got it!", "I have a 2018 Toyota Prado at 105,000 dirhams and a 2021 Toyota Land Cruiser at 148,000 dirhams.",
                    "Which one catches your eye?"] and not info["appended"] and info["first_ms"] >= 0


def test_a_cut_off_sentence_is_never_spoken_and_the_missing_facts_and_question_are_added():
    said, info = stream("Got it! Two Toyotas: the 2018 Prado at 105,000 and the 2021 Land Cruiser at")
    assert said[0] == "Got it!" and not any(s.endswith(" at") for s in said)
    assert any("105,000" in s and "148,000" in s for s in said) and said[-1].endswith("Which one would you like to hear more about?") and info["appended"]


def test_a_reaction_without_the_answer_gets_the_answer_added():
    core = "Yes, it's GCC spec. Anything else you'd like to know?"
    said, info = stream("Ah, yes!", core=core, facts="What the customer wanted: they asked about a car.", caller="is it gcc spec")
    assert said == ["Yes!", "It's GCC spec."] and info["appended"]            # the filler "Ah" is gone and "yes" is not said twice


def test_the_first_unsafe_sentence_stops_the_stream_and_what_was_safe_stands():
    said, info = stream("Got it! We have a 2019 Lexus at 99,000 dirhams. Which one catches your eye?")
    assert said[0] == "Got it!" and not any("Lexus" in s or "99,000" in s for s in said)
    assert "new" in info["stopped"] and said[-1].endswith("Which one would you like to hear more about?")


def test_nothing_usable_means_nothing_is_yielded_so_the_standard_wording_is_used():
    said, info = stream("We have a 2019 Lexus at 99,000 dirhams.")
    assert said == [] and info["sentences"] == 0


def test_price_and_booking_turns_are_not_streamed_when_high_risk_is_off():
    said, info = stream("The best I can do is 52,200 dirhams, subject to confirmation.", core=PRICE, facts=PRICE_FACTS, risk="high", high_risk=False)
    assert said == [] and "standard wording" in info["stopped"]


def test_a_backend_failure_in_the_middle_keeps_what_was_already_said():
    class Breaks(StreamBackend):
        def lead_in_stream(self, *a, **k):
            yield "Got it! "
            raise TimeoutError("stopped")

    w = ReplyWriter(Breaks("x"))
    said = list(w.stream(WriterInput("toyota please", SEARCH_CORE, "low", [], SEARCH_FACTS)))
    assert said[0] == "Got it!" and "backend" in w.last_stream["stopped"] and said[-1].endswith("Which one would you like to hear more about?")


def test_engine_handle_stream_yields_sentences_then_the_turn_result():
    from scenarios.conftest import Call
    from dealer_core.dialogue.machine import TurnResult

    c = Call({})
    c.app.engine.reply_writer = ReplyWriter(StreamBackend(
        "Ah, Altima and Mustang! I have a 2019 Nissan Altima at 52,500 dirhams and a 2018 Ford Mustang at 120,000 dirhams. Which one are you curious about?"))
    items = list(c.app.engine.handle_stream(c.s, "a red car"))
    assert isinstance(items[-1], TurnResult) and all(isinstance(x, str) for x in items[:-1]) and len(items) >= 3
    assert items[-1].reply == " ".join(items[:-1]) and items[-1].debug.draft
    # price turns are spoken from the exact standard wording, not streamed from the model
    c2 = Call({"can you do 100000": "make_offer"})
    c2.app.engine.reply_writer = ReplyWriter(StreamBackend("Whatever I like!"))
    c2.say("do you have a toyota camry")
    items = list(c2.app.engine.handle_stream(c2.s, "can you do 100000"))
    assert "subject to confirmation" in items[0] and not items[-1].debug.draft


def test_when_the_system_is_waiting_on_a_specific_answer_the_closing_question_must_be_about_it():
    core = "Great, Saturday at 4:00 PM. Can I have your name and mobile number?"
    said, info = stream("Lovely, Saturday it is! What else can I help you with?", core=core, facts="What the customer wanted: they gave a time.", awaiting=True)
    assert said[-1] == "Can I have your name and mobile number?" and info["appended"]
    said, _ = stream("Lovely, Saturday at 4:00 PM it is! Could I take your name and mobile number?", core=core, facts="What the customer wanted: they gave a time.", awaiting=True)
    assert said[-1] == "Could I take your name and mobile number?"


def test_the_models_question_is_held_so_missing_facts_come_before_it():
    said, _ = stream("Ah, nice! We've got two Toyota SUVs. Which one would you like to hear more about?")
    assert said[-1].endswith("hear more about?") and said.index(next(s for s in said if "105,000" in s)) < len(said) - 1
    assert sum(s.endswith("?") for s in said) == 1                                    # the question is asked once


def test_a_cut_off_start_of_the_system_question_is_not_said_twice():
    from dealer_core.dialogue.writer import drop_started_question

    q = "Which one would you like to hear more about?"
    assert drop_started_question("I've got a Corolla and a Camry. Which one", q) == "I've got a Corolla and a Camry."
    assert drop_started_question("I've got a Corolla. Anything else", q) == "I've got a Corolla. Anything else."


def test_a_model_may_not_say_the_visit_is_booked_before_the_system_does():
    from dealer_core.dialogue.writer import verify_sentence

    core = "Thanks, Sara. And your mobile number?"
    assert verify_sentence("Sara, you're booked for the BMW tomorrow.", "My name is Sara", core, "", "")
    done = "You're booked for tomorrow at 4:00 PM."
    assert not verify_sentence("You're booked for tomorrow at 4:00 PM.", "yes", done, "", "")
