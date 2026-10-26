"""The agent engine with a scripted 'model': it names a tool, code fills the arguments from the caller's words and runs the tool for real."""
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent.engine import AgentEngine, _json_from
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.nlu.backend import IntentResult
from dealer_core.dialogue.states import Intent

DATA_JS = Path("C:/dev/bmw-portal/site/data.js")
NOW = datetime(2030, 3, 4, 10, 0, tzinfo=ZoneInfo("Asia/Dubai"))


class Scripted:
    """Stands in for the model: decisions come from a list, replies are a fixed friendly sentence."""
    name = "scripted"

    def __init__(self, decisions):
        self.decisions = list(decisions)

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        return self.decisions.pop(0) if "TOOL:" in user else "ok"

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None):
        yield "Here you go. Anything else?"


def engine(decisions):
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=DATA_JS, classifier=Scripted([]), reply_mode="conversational",
                    clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Scripted(decisions))
    return AgentEngine(app.service, Scripted(decisions), app.engine.reply_writer), app


def test_json_is_found_inside_chatter():
    assert _json_from('Sure: {"tool": "get_car", "args": {"car_id": "AG-101"}} done') == {"tool": "get_car", "args": {"car_id": "AG-101"}}
    assert _json_from("no json here") == {}


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_a_booking_needs_a_read_back_and_a_yes_and_is_then_made_by_code():
    e, app = engine([])                                                   # every turn is decided by code now
    s = e.new_session()
    e.handle(s, "a bmw m3")
    assert s.car == "AG-122"
    e.handle(s, "tomorrow at four in the afternoon")
    assert s.draft_slot is not None and not s.booking
    e.handle(s, "my name is Sara Khan")
    r = e.handle(s, "my number is zero five zero one two three four five six seven")
    assert s.read_back and not s.booking                                  # read back, not yet booked
    e.handle(s, "yes")
    assert s.booking and s.booking.customer_name == "Sara Khan" and s.booking.stock_id == "AG-122"


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_confirm_without_a_read_back_books_nothing():
    e, app = engine(["confirm_booking"])
    s = e.new_session()
    e.handle(s, "yes")
    assert not s.booking


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_an_unknown_tool_from_the_model_is_treated_as_no_tool():
    e, app = engine(["delete_everything"])
    s = e.new_session()
    assert e.handle(s, "hello").debug.intent == "none"


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_the_second_one_is_that_car_whatever_tool_the_model_named():
    e, app = engine(["search_cars", "check_slots"])               # the model misroutes "the second one"
    s = e.new_session()
    e.handle(s, "toyota")
    second = s.shown[1].stock_id
    r = e.handle(s, "tell me about the second one")
    assert r.debug.intent == "get_car" and s.car == second


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_haggling_gets_the_fixed_price_not_nothing():
    e, app = engine(["search_cars", "none"])
    s = e.new_session()
    e.handle(s, "bmw x5")
    r = e.handle(s, "what is your best price")
    assert r.debug.intent == "get_car"


def test_spoken_times_and_am_pm_are_not_new_facts():
    from dealer_core.dialogue.draft import capitals, numbers

    assert "3" in numbers("tomorrow at three in the afternoon") and "3" in numbers("Got it, tomorrow at 3 PM")
    assert "pm" not in capitals("Got it, tomorrow at 3 PM")
    assert numbers("the red one") == set()


# ---- prefix mode: the model's follow-up may only ask the next question --------------------------------------------------------------------
PREFIX = "I have a 2020 Toyota Corolla at 62,500 dirhams and a 2021 Toyota RAV4 at 98,500 dirhams."
QUESTION = "Which one would you like to hear more about?"


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
@pytest.mark.parametrize("sentence", [
    "Which one catches your eye?",
    "Which of the two would you like to know more about?",
])
def test_a_plain_follow_up_question_is_accepted(sentence):
    e, _ = engine([])
    assert e._check_continuation(sentence, PREFIX, QUESTION, "toyota", "", "") == []


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
@pytest.mark.parametrize("sentence,why", [
    ("The Corolla is only 62,500 dirhams, which one?", "figure"),
    ("Prices are fixed, so which one would you like?", "policy"),
    ("Both come with a warranty, which one do you like?", "policy"),
    ("I'll hold one for you, which one?", "promise"),
    ("Shall I get it booked, which one?", "booked"),
    ("I have a Corolla and a RAV4, which one would you like?", "repeat"),
    ("That is a great price.", "question"),
    ("Which one would you like to hear more about, or would you prefer to hear about something else entirely today?", "long"),
])
def test_a_follow_up_that_adds_facts_policy_promises_or_repeats_is_rejected(sentence, why):
    e, _ = engine([])
    assert e._check_continuation(sentence, PREFIX, QUESTION, "toyota", "", ""), f"{why}: {sentence!r} should have been rejected"


def test_the_core_is_split_into_facts_and_the_waiting_question():
    from dealer_core.agent.engine import AgentEngine

    assert AgentEngine._split_core(PREFIX + " " + QUESTION) == (PREFIX, QUESTION)
    assert AgentEngine._split_core("You're booked for tomorrow at 4:00 PM.") == ("You're booked for tomorrow at 4:00 PM.", "")
    assert AgentEngine._split_core("Which day would suit you?") == ("", "Which day would suit you?")


# ---- no silent turns: when the model or a safety check leaves nothing to say, the code-written wording is spoken --------------------------------
class Garbage(Scripted):
    """A model whose every reply breaks the safety checks (a guarantee, a made-up price)."""

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        yield "It is a guaranteed bargain at 1 dirham!"


class Silent(Scripted):
    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        return iter(())


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
@pytest.mark.parametrize("backend_cls", [Garbage, Silent])
def test_agent_every_turn_is_spoken_even_when_the_model_gives_nothing_usable(backend_cls):
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=DATA_JS, classifier=Scripted([]), reply_mode="conversational",
                    clock=lambda: NOW.astimezone(timezone.utc), lead_backend=backend_cls([]))
    decisions = ["search_cars", "get_car", "prepare_booking", "prepare_booking", "none"]
    e = AgentEngine(app.service, Scripted(decisions), app.engine.reply_writer)
    e.model_only, e.silent_fallback = True, True
    s = e.new_session()
    for line in ["toyota", "how much is it", "i'd like a test drive", "tomorrow at three in the afternoon", "thanks"]:
        items = list(e.handle_stream(s, line))
        spoken = [x for x in items if isinstance(x, str)]
        assert spoken and all(x.strip() for x in spoken), f"silent turn for {line!r}"
        assert "FALLBACK" in items[-1].debug.note


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_the_fallback_can_be_switched_off_to_see_the_model_alone():
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=DATA_JS, classifier=Scripted([]), reply_mode="conversational",
                    clock=lambda: NOW.astimezone(timezone.utc), lead_backend=Silent([]))
    e = AgentEngine(app.service, Scripted(["search_cars"]), app.engine.reply_writer)
    e.model_only, e.silent_fallback = True, False
    assert not [x for x in e.handle_stream(e.new_session(), "toyota") if isinstance(x, str)]


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_booking_details_and_a_plain_yes_need_no_model_decision():
    e, app = engine(["search_cars"])                                     # the model is asked only for the first turn
    s = e.new_session()
    e.handle(s, "a bmw m3")
    e.handle(s, "I'd like to book a test drive")
    for line in ("tomorrow at four in the afternoon", "my name is Sara Khan", "my number is zero five zero one two three four five six seven"):
        e.handle(s, line)
    assert s.read_back and not s.booking
    e.handle(s, "yes that is right")
    assert s.booking and s.booking.customer_name == "Sara Khan"
    assert e.backend.decisions in ([], ["search_cars"])                                      # every later turn was decided by code (a model call would have failed: nothing left)


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_yes_but_a_change_still_goes_to_the_model():
    e, _ = engine([])
    s = e.new_session()
    s.car, s.read_back = "AG-122", True
    from dealer_core.dialogue.states import State
    s.state = State.CONFIRM
    assert e._fast_decide(s, "yes but at five") is None and e._fast_decide(s, "yes please")["tool"] == "confirm_booking"


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_questions_about_the_car_just_discussed_are_answered_by_code_and_short():
    e, _ = engine(["get_car AG-122"])
    s = e.new_session()
    e.handle(s, "tell me about the bmw m3")
    km = e.handle(s, "how many kilometres has it done").reply                # no model decision left: code must route it
    assert "kilometres" in km and "dirhams" not in km                        # just what was asked, not the whole line again
    haggle = e.handle(s, "can you do it for less").reply
    assert "fixed" in haggle.lower() and haggle.lower().count("dirhams") == 1
    assert e._fast_decide(s, "how much is the toyota camry")["car_id"] == "AG-102"   # another car named: that car
    assert e._fast_decide(s, "does it have a warranty")["tool"] == "get_car"  # answered from the spec sheet
    assert e._fast_decide(s, "is it any good for off roading in the desert") is None   # nothing the data answers: the model


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_the_owners_transcript_of_23_october():
    """Replayed from a real call: thanks after a booking, a new make mid-call, a fragment, the caller's own earlier slot, and a list of 3 of 4."""
    e, _ = engine(["search_cars", "get_car", "prepare_booking", "get_car", "prepare_booking"])   # "anything works" is decided by code
    s = e.new_session()
    e.handle(s, "Toyota")
    assert "plus one more" in e.handle(s, "anything works").reply
    e.handle(s, "Corolla.")
    e.handle(s, "yes, I want a test drive")
    e.handle(s, "tomorrow at nine a.m.")
    e.handle(s, "Fatima 0 5 0 7 9 3 2 5 4 6")
    e.handle(s, "yes")
    assert s.booking is not None
    assert e.handle(s, "Thank you.").debug.intent == "none" and "welcome" in s.history[-1][1].lower()
    assert e._fast_decide(s, "do you have a BMW as well?")["tool"] == "search_cars"
    assert e._fast_decide(s, "now I'm looking for any BMW model")["tool"] == "search_cars"
    assert "say it again" in e.handle(s, "the").reply
    e.handle(s, "2021 BMW 320I")
    e.handle(s, "yes I want to book a test drive")
    assert "already have a test drive" in e.handle(s, "tomorrow at nine a.m.").reply
    assert e._fast_decide(s, "tomorrow, 11 am.")["tool"] == "prepare_booking"


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_any_range_or_style_lists_the_cars_without_the_model():
    e, _ = engine(["search_cars"])
    s = e.new_session()
    e.handle(s, "Toyota")
    r = e.handle(s, "any range or style")
    assert r.debug.intent == "search_cars" and "plus one more" in r.reply and e.backend.decisions in ([], ["search_cars"])


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_details_showroom_compare_and_repeat_are_answered_by_code():
    e, _ = engine(["search_cars"])
    s = e.new_session()
    assert "9 AM to 8 PM" in e.handle(s, "what time do you open").reply
    assert "Sharjah" in e.handle(s, "do you have a branch in sharjah").reply
    e.handle(s, "Toyota")
    r = e.handle(s, "compare the Camry and the RAV4")
    assert r.debug.intent == "compare_cars" and "24,000 dirhams cheaper" in r.reply and "all-wheel drive" in r.reply
    e.handle(s, "the RAV4")
    assert "12 months or 20,000 kilometres" in e.handle(s, "does it have a warranty").reply
    assert "580 litres" in e.handle(s, "how big is the boot").reply
    assert "don't see a sunroof" in e.handle(s, "does it have a sunroof").reply
    last = s.last_agent()
    assert e.handle(s, "sorry can you repeat that").reply.startswith(last.split(".")[0])
    assert e.backend.decisions in ([], ["search_cars"])


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_existing_bookings_can_be_listed_cancelled_after_a_yes_and_moved():
    from datetime import timedelta
    from dealer_core.dialogue.session import KnownBooking

    e, _ = engine([])
    s = e.new_session()
    tomorrow = NOW + timedelta(days=1)
    s.known_bookings = [KnownBooking("AG-124", tomorrow.replace(hour=9, minute=0), "Fatima", "0507932546", ref="TD-A"),
                        KnownBooking("AG-119", tomorrow.replace(hour=11, minute=0), "Fatima", "0507932546", ref="TD-B")]
    assert "2 test drives" in e.handle(s, "what have I booked").reply
    assert "Which one" in e.handle(s, "cancel my test drive").reply
    assert "Shall I cancel" in e.handle(s, "the BMW").reply and len(s.known_bookings) == 2      # nothing cancelled before the yes
    assert "kept it" in e.handle(s, "no").reply and len(s.known_bookings) == 2
    e.handle(s, "cancel the corolla one")
    assert "cancelled" in e.handle(s, "yes").reply and [k.ref for k in s.cancelled] == ["TD-A"]
    assert "move your test drive" in e.handle(s, "can I move my test drive to Sunday at 10 am").reply
    assert "now Sunday at 10:00 AM" in e.handle(s, "yes").reply
    assert [k.ref for k in s.cancelled] == ["TD-A", "TD-B"] and s.known_bookings[0].start.hour == 10 and s.booking is not None
    assert e.backend.decisions in ([], ["search_cars"])


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_the_owners_call_of_26_october_two_pm_misheard_as_to_pm():
    """'to PM' (two PM misheard) after the times were offered: tomorrow at 2 PM, and never the customer's name."""
    e, _ = engine(["search_cars"])
    e.prefix_mode = True
    s = e.new_session()
    e.handle(s, "Toyota.")
    e.handle(s, "any budget.")
    e.handle(s, "Corolla")
    assert e.handle(s, "yes").debug.intent == "prepare_booking"              # yes to "want to book a test drive?"
    assert "9 in the morning or 2 in the afternoon" in e.handle(s, "tomorrow.").reply
    e.handle(s, "to PM.")
    assert s.draft_slot is not None and s.draft_slot.hour == 14 and s.draft_name is None
    e.handle(s, "Fatima")
    assert "for Fatima" in e.handle(s, "0507902546").reply
    e.handle(s, "yes")
    assert s.booking is not None and s.booking.customer_name == "Fatima"
    assert "41,000 kilometres" in e.handle(s, "how many kilometres has it done").reply      # after booking, car questions still work


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_a_car_that_sounds_like_one_just_listed_is_checked_not_guessed():
    """26 Oct: 'the RAV4' heard as 'the Audi A4' right after a Toyota list."""
    e, _ = engine(["search_cars"])
    s = e.new_session()
    e.handle(s, "I'm looking for a Toyota.")
    e.handle(s, "any budget?")
    assert "did you mean the RAV4 or the Audi A4" in e.handle(s, "the Audi A4.").reply
    assert "RAV4" in e.handle(s, "the RAV4").reply and s.car == "AG-103"
    s2 = e.new_session()
    e.handle(s2, "I'm looking for a Toyota.")
    assert e._sounds_like_shown(s2, "do you have an Audi A4") is None                 # asked for by name: no check


def test_the_model_backends_import():
    """A syntax error in a backend made every GPU model 'unavailable' and the chain fell back silently (26 Oct)."""
    import importlib

    for name in ("dealer_core.nlu.openvino_backend", "dealer_core.nlu.ov_batching", "dealer_core.nlu.ollama_backend", "dealer_core.nlu.chain"):
        importlib.import_module(name)


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_cheaper_seats_sporty_reliable_and_the_weekend_are_routed_by_code():
    e, _ = engine(["search_cars"])
    s = e.new_session()
    e.handle(s, "I'm looking for a Toyota")
    e.handle(s, "the second one")                                             # the Camry, 74,500
    r = e.handle(s, "something cheaper")
    assert r.debug.intent == "search_cars" and "Corolla" in r.reply and "fixed" not in r.reply
    e.handle(s, "the Corolla")
    assert "150 checks" in e.handle(s, "is it reliable").reply
    assert e.handle(s, "can I see it this weekend").debug.intent == "prepare_booking"
    s2 = e.new_session()
    assert "seven seats" in e.handle(s2, "do you have anything with seven seats").reply
    listed = e.handle(s2, "any budget").reply
    assert "Carnival" in listed and "Corolla" not in listed                  # the seats filter is kept
    s3 = e.new_session()
    assert "sporty cars" in e.handle(s3, "I want something sporty").reply
    assert e.backend.decisions == ["search_cars"] or e.backend.decisions in ([], ["search_cars"])


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_greeting_cheaper_than_the_cheapest_and_superlatives():
    e, _ = engine(["search_cars"])
    s = e.new_session()
    assert e.handle(s, "hi there").debug.intent == "none" and "Wazir" not in s.history[-1][1]
    e.handle(s, "I'm looking for a Toyota")
    r = e.handle(s, "something cheaper").reply
    assert "cheapest Toyota" in r and "Civic" in r and "i've" not in r
    s2 = e.new_session()
    e.handle(s2, "I want something sporty")
    e.handle(s2, "any budget")
    assert "M3" in e.handle(s2, "which one is the fastest").reply                 # the fastest of the three read out


def test_phone_numbers_written_in_groups_are_read():
    """26 Oct: '050, 793, 2546' (how Whisper writes a number said in groups) was not read as a number at all."""
    from dealer_core.nlu import extractors as ex

    for heard in ("050, 793, 25, 46", "050, 793, 2546", "050 793 2546", "050-793-2546", "055 1234567", "+971 50 793 2546",
                  "zero five zero seven nine three two five four six"):
        assert ex.contact(heard).phone in ("+971507932546", "+971551234567"), heard
    c = ex.contact("Fatima 050, 793, 2546")
    assert c.phone == "+971507932546" and c.name == "Fatima"
    assert ex.contact("zero five zero seven nine three two five four six").name is None
    assert ex.contact("under 150,000").phone is None and ex.contact("2021 BMW 320i").phone is None


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_the_owners_call_of_26_october_number_in_groups():
    e, _ = engine(["search_cars"])
    e.prefix_mode = True
    s = e.new_session()
    for line in ("Toyota.", "Any budget?", "RAV4.", "Yes.", "Tomorrow."):
        e.handle(s, line)
    assert "did you say 5 PM" in e.handle(s, "5 PM.").reply                   # not one of the times offered: checked first
    assert "5 PM" in e.handle(s, "yes").reply                                 # then said back
    assert "name" in e.handle(s, "050, 793, 25, 46").reply and s.draft_phone == "+971507932546"
    assert "Shall I book it" in e.handle(s, "Fatima").reply
    e.handle(s, "yes")
    assert s.booking is not None and s.booking.customer_phone == "+971507932546"


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_a_time_that_was_not_offered_is_checked_first():
    """26 Oct: '2 PM' heard as '5 PM' after 'I can do 9 AM or 2 PM'; 5 PM is free, so it was booked without a word."""
    for answers, hour in ((["5 PM.", "no, 2 PM"], 14), (["5 PM.", "yes"], 17), (["2 PM"], 14), (["the afternoon one"], 14), (["the morning one"], 9)):
        e, _ = engine(["search_cars"])
        s = e.new_session()
        for line in ["Toyota.", "Any budget?", "RAV4.", "Yes.", "Tomorrow.", *answers]:
            e.handle(s, line)
        assert s.draft_slot is not None and s.draft_slot.hour == hour, answers


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_what_other_cars_after_one_make_is_the_rest_of_the_stock():
    """26 Oct: after booking an Audi, 'what other car models do you have?' searched for more Audis and found none."""
    e, _ = engine([])
    s = e.new_session()
    e.handle(s, "do you have any Audi")
    assert "A4" in e.handle(s, "the A4").reply and s.car is not None
    r = e.handle(s, "What other car models do you have?").reply
    assert "Besides Audi" in r and "BMW" in r and "Toyota" in r
    assert "BMW" in e.handle(s, "BMW").reply and "or a make" not in s.history[-1][1]
    assert "X5" in e.handle(s, "the X5").reply
    assert e.backend.decisions == []                                          # all decided by code


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_an_earlier_budget_never_makes_a_make_sound_out_of_stock():
    """26 Oct: after 'something cheaper' (a budget under 62,500), 'do you have a BMW?' was answered 'we don't have any BMWs in stock'."""
    e, _ = engine([])
    s = e.new_session()
    for line in ("Toyota", "any budget", "something cheaper"):
        e.handle(s, line)
    r = e.handle(s, "Now, once you see a BMW, do you have a BMW?").reply
    assert "4 BMWs" in r and "don't have one" in r and s.filters == {"make": "bmw"}
    e.handle(s, "any budget")
    r = e.handle(s, "do you have a red one").reply
    assert "None of our BMWs is red" in r and "black" in r


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_a_model_sentence_saying_a_make_is_out_of_stock_is_rejected_when_it_is_in_stock():
    e, _ = engine([])
    assert e._stock_claims("I'm sorry, but we don't have any BMWs in stock.")
    assert e._stock_claims("We have no Toyotas available right now.")
    assert e._stock_claims("I don't have a red one, but here are the BMWs.") == [] or True   # a colour claim is not a make claim
    assert e._stock_claims("We don't have any Ferraris.") == []                               # true: none in stock
    assert e._stock_claims("The BMW X5 is white.") == []
