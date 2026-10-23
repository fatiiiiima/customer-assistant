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
