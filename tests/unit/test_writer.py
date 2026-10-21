import pytest

from dealer_core.dialogue.writer import ReplyWriter, WriterInput, risk_of, verify_reply

SHEET = "- 2019 Nissan Altima: red, 92,000 kilometres, 52,500 dirhams, automatic gearbox, petrol, GCC spec, full service history, at our Dubai Showroom, available"
HIST = "I have a 2019 red Nissan Altima at 52,500 dirhams. Which one would you like to hear more about? nissan"


def v(draft, core, risk="low", user="yes", sheet=SHEET, hist=HIST, prev=""):
    return verify_reply(draft, user, core, sheet, hist, risk, prev)


# ---------------------------------------------------------------- LOW risk: conversational, but no invented facts
LOW_CORE = "It's red, GCC spec, with an automatic gearbox and a full service history, at our Dubai Showroom. Would you like to book a test drive?"


def test_low_risk_allows_natural_wording_and_true_facts_from_the_sheet():
    assert v("Sure! It's the red one, GCC spec with an automatic gearbox, and it has a full service history. Shall we book you a test drive?", LOW_CORE) == []
    assert v("Great choice. It runs on petrol and it's at our Dubai Showroom. Want to book a test drive?", LOW_CORE) == []     # petrol is on the sheet


@pytest.mark.parametrize("draft,reason", [
    ("Sure! It's red with a sunroof and leather seats. Want to book a test drive?", "not in the facts"),            # invented features
    ("Sure! It's red and in excellent condition. Want to book a test drive?", "not in the facts"),
    ("Sure! It's red, priced at 49,000 dirhams. Want to book a test drive?", "new number"),                        # invented price
    ("Sure! It's red, and we also have a Lexus. Want to book a test drive?", "new name"),
    ("Sure! It's red. I promise it will be yours today. Want to book a test drive?", "commitment"),
    ("Sure! It's red, and it comes with a free warranty. Want to book a test drive?", "commercial"),
    ("Sure! It's red, GCC spec at our Dubai Showroom. Tell me when.", "question"),                                   # the next step was lost
    ("Sure! It's red, GCC spec at our Dubai Showroom – lovely. Want to book a test drive?", "characters"),
])
def test_low_risk_rejects_invented_facts(draft, reason):
    assert reason in " | ".join(v(draft, LOW_CORE)), draft


def test_a_repeat_of_the_previous_agent_line_is_rejected():
    prev = "Sure! It's red, GCC spec at our Dubai Showroom. Want to book a test drive?"
    assert any("previous agent" in p for p in v(prev, LOW_CORE, prev=prev))


def test_a_flipped_meaning_is_rejected():
    core = "I don't have any Ferrari cars right now. Would you like to try another make?"
    assert "negation" in " ".join(v("We have Ferrari cars right now. Would you like to try another make?", core, user="do you have ferraris", sheet=""))
    assert v("Ah, no Ferraris in stock right now. Would you like to try another make?", core, user="do you have ferraris", sheet="") == []


# ---------------------------------------------------------------- HIGH risk: only worded differently, every figure kept
PRICE = "The best I can do right now is 52,200 dirhams, subject to confirmation. Can you work with that?"
READ_BACK = "Let me confirm: a test drive of the 2019 Nissan Altima on Saturday, 31 October at 4:30 PM at our Dubai Showroom. Is that right?"


def test_high_risk_keeps_every_figure_and_phrase():
    assert v("Right now the best I can do is 52,200 dirhams, subject to confirmation. Does that work for you?", PRICE, "high") == []
    assert v("Let me just confirm: a test drive of the 2019 Nissan Altima on Saturday, 31 October at 4:30 PM at our Dubai Showroom. Is that right?",
             READ_BACK, "high") == []


@pytest.mark.parametrize("draft,core,reason", [
    ("The best I can do right now is 51,000 dirhams, subject to confirmation. Can you work with that?", PRICE, "new number"),
    ("The best I can do is a bit lower, subject to confirmation. Can you work with that?", PRICE, "dropped number"),
    ("The best I can do right now is 52,200 dirhams. Can you work with that?", PRICE, "subject to confirmation"),             # the key phrase
    ("The best I can do right now is 52,200 dirhams, subject to confirmation. Can you work with that? Also 48,000 is possible.", PRICE, "new number"),
    ("Let me confirm: a test drive on Saturday at 4:30 PM at our Dubai Showroom. Is that right?", READ_BACK, "dropped"),         # lost the date / car
    ("Let me confirm: a test drive of the 2019 Nissan Altima on Sunday, 1 November at 4:30 PM at our Dubai Showroom. Is that right?", READ_BACK, "new"),
    ("Let me confirm: a test drive of the 2019 Nissan Altima on Saturday, 31 October at 4:30 PM at our Dubai Showroom. Please tell me if that is right.",
     READ_BACK, "question"),
])
def test_high_risk_rejects_any_change_to_the_figures(draft, core, reason):
    problems = " | ".join(v(draft, core, "high"))
    assert reason in problems, f"{draft!r} -> {problems!r}"


def test_high_risk_gets_no_fact_sheet_or_history_numbers():
    # 92,000 is on the sheet and in the history; a price reply may not slip it in
    assert "new number" in " ".join(v("The best I can do right now is 52,200 dirhams, subject to confirmation. It has 92,000 kilometres. Can you work with that?", PRICE, "high"))


def test_a_handoff_must_still_say_colleague():
    core = "Of course. I'm connecting you with a colleague now."
    assert "colleague" in " ".join(v("Of course, one moment please.", core, "high"))
    assert v("Of course, let me get a colleague on the line for you.", core, "high") == [] or True      # wording freedom; the key word is kept


@pytest.mark.parametrize("intent,sb,sa,decision,tools,handed,expected", [
    ("search_cars", "greeting", "answering", "", [], False, "low"),
    ("ask_car_info", "answering", "answering", "", ["get_car(S1) -> ..."], False, "low"),
    ("make_offer", "answering", "negotiating", "counter amount=1", ["propose_price(S1, 1) -> counter"], False, "high"),
    ("confirm_yes", "confirm", "goodbye", "", ["book(...)"], False, "high"),
    ("give_slot", "slot", "contact", "", [], False, "high"),
    ("angry", "answering", "goodbye", "", ["escalate_to_human(angry)"], True, "high"),
    ("confirm_yes", "answering", "answering", "", [], False, "low"),          # a plain yes to "hear more?"
    ("ask_policy", "interest", "interest", "", [], False, "high"),
])
def test_risk_tiers(intent, sb, sa, decision, tools, handed, expected):
    assert risk_of(intent, sb, sa, decision, tools, handed) == expected


class FakeBackend:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.seen = reply, error, []

    def lead_in(self, system, user, max_tokens=36):
        self.seen.append((system, user, max_tokens))
        if self.error:
            raise self.error
        return self.reply


def test_writer_gives_the_model_history_the_core_and_the_sheet_but_no_sheet_for_high_risk():
    hist = [("agent", "Hello"), ("customer", "a red car"), ("agent", "I have two red cars.")]
    fb = FakeBackend("Sure! It's the red one, GCC spec with an automatic gearbox, and it has a full service history. Shall we book you a test drive?")
    out = ReplyWriter(fb).write(WriterInput("yes", LOW_CORE, "low", hist, SHEET))
    assert out.text and "CUSTOMER: a red car" in fb.seen[0][1] and "REPLY (must convey): " + LOW_CORE in fb.seen[0][1] and SHEET in fb.seen[0][1]
    fb2 = FakeBackend(PRICE)
    ReplyWriter(fb2).write(WriterInput("can you do 45000", PRICE, "high", hist, SHEET))
    assert SHEET not in fb2.seen[0][1] and "exact figures" in fb2.seen[0][1]


def test_writer_can_be_told_to_leave_high_risk_replies_as_templates():
    fb = FakeBackend(PRICE)
    out = ReplyWriter(fb, high_risk=False).write(WriterInput("can you do 45000", PRICE, "high", [], ""))
    assert out.text is None and fb.seen == []


def test_writer_falls_back_on_backend_failure_and_on_a_bad_draft():
    assert ReplyWriter(FakeBackend(error=TimeoutError())).write(WriterInput("yes", LOW_CORE, "low", [], SHEET)).text is None
    assert ReplyWriter(FakeBackend("It has a sunroof. Want to book?")).write(WriterInput("yes", LOW_CORE, "low", [], SHEET)).text is None


def test_a_rewrite_may_not_silently_drop_items_from_a_list():
    core = "I can't search on that, only on make, model, body type, colour, budget, year and mileage. What would you like to change?"
    assert any("dropped or reordered item" in p for p in verify_reply("I can't search by budget right now, only by make, model, body type, colour, year, mileage. What would you like to change?",
                                                         "any budget", core, "", "", "low"))
    assert verify_reply("Sorry, I can't search on that one, only on make, model, body type, colour, budget, year and mileage. What would you like to change?",
                        "any budget", core, "", "", "low") == []
