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
    e, app = engine(["search_cars", "prepare_booking AG-122", "prepare_booking AG-122", "prepare_booking AG-122", "confirm_booking"])
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
    e, app = engine(["search_cars", "prepare_booking AG-122"])           # the model is asked only for the first two turns
    s = e.new_session()
    e.handle(s, "a bmw m3")
    e.handle(s, "I'd like to book a test drive")
    for line in ("tomorrow at four in the afternoon", "my name is Sara Khan", "my number is zero five zero one two three four five six seven"):
        e.handle(s, line)
    assert s.read_back and not s.booking
    e.handle(s, "yes that is right")
    assert s.booking and s.booking.customer_name == "Sara Khan"
    assert e.backend.decisions == []                                      # every later turn was decided by code (a model call would have failed: nothing left)


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
    assert e._fast_decide(s, "how much is the toyota camry") is None          # another car named: the model decides
    assert e._fast_decide(s, "does it have a warranty") is None               # the code's line does not answer that


@pytest.mark.skipif(not DATA_JS.exists(), reason="the website is not on this machine")
def test_the_owners_transcript_of_23_october():
    """Replayed from a real call: thanks after a booking, a new make mid-call, a fragment, the caller's own earlier slot, and a list of 3 of 4."""
    e, _ = engine(["search_cars", "search_cars", "get_car", "prepare_booking", "get_car", "prepare_booking"])
    s = e.new_session()
    e.handle(s, "Toyota")
    assert "Of the 4" in e.handle(s, "anything works").reply and "one more" in s.history[-1][1]
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
