"""27 Oct, Chrome: the caller wanted to see a RAV4 and the agent asked "which day?" six times in a row ("Any day would work", "What day is free to come in?",
"And today." at 22:07 after closing), with "Great choice." in front of a question that had chosen nothing. Now it offers real times."""
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent.engine import _DELEGATES, _FLEXIBLE, AgentEngine
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.dialogue.states import Intent, State
from dealer_core.nlu.backend import IntentResult
from dealer_core.shared.state import restore, snapshot
from dealer_core.voice.prompts import DAY_TIME

SITE = next((p for p in (ROOT.parent / "bmw-portal" / "site" / "data.js", Path("C:/dev/bmw-portal/site/data.js")) if p.exists()), None)
TZ = ZoneInfo("Asia/Dubai")
AFTER_CLOSING = datetime(2026, 10, 27, 22, 7, tzinfo=TZ)           # the real time of the call: a Tuesday night, the showroom shut


class Scripted:
    """The model: the tool for each decision comes from a list (anything not listed is "none"), replies are a fixed sentence."""
    name = "scripted"

    def __init__(self, decisions=()):
        self.decisions = list(decisions)

    def classify(self, *a, **k):
        return IntentResult(Intent.SEARCH_CARS, 1.0, self.name, "{}")

    def lead_in(self, system, user, *a, **k):
        return (self.decisions.pop(0) if self.decisions else "none") if "TOOL:" in user else "ok"

    def lead_in_stream(self, *a, **k):
        yield "Here you go."


def engine(now=AFTER_CLOSING, decisions=(), portal=False):
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml") if portal else load_config(environ={})
    kw = dict(inventory="portal", portal_data=SITE) if portal else dict(inventory="demo")
    app = build_app(cfg, classifier=Scripted(), reply_mode="conversational", clock=lambda: now.astimezone(timezone.utc), lead_backend=Scripted(), **kw)
    return AgentEngine(app.service, Scripted(decisions), app.engine.reply_writer)


def say(e, s, text):
    out = list(e.handle_stream(s, text))
    return out[-1].debug.intent, " ".join(x for x in out if isinstance(x, str)), out[-1]


def discussing_a_car(e):
    s = e.new_session()
    say(e, s, "I'm looking for a Toyota")
    s.car, s.state = s.shown[0].stock_id, State.ANSWERING
    return s


# ------------------------------------------------------------------------------------------------ the phrases
@pytest.mark.parametrize("text", ["Any day would work.", "Any day suits me", "Any day?", "What day is free to come in?", "whenever", "as soon as possible", "the earliest you have",
                                  "you choose", "up to you", "when can I come and see it?", "what's free tomorrow", "it doesn't matter"])
def test_these_leave_the_choice_of_day_to_the_agent(text):
    assert _FLEXIBLE.search(" ".join(text.lower().replace("?", "").replace(".", "").split()))


@pytest.mark.parametrize("text", ["tomorrow", "I'm looking for a Toyota", "what colour is it", "tell me about the RAV4", "do you have any red cars", "my number is zero five zero"])
def test_these_do_not(text):
    assert not _FLEXIBLE.search(text.lower())


def test_asking_what_is_free_is_not_leaving_the_choice_to_the_agent():
    assert not _DELEGATES.search("what day is free to come in") and not _DELEGATES.search("when can i come and see it")
    assert _DELEGATES.search("any day suits me") and _DELEGATES.search("you choose")


# ------------------------------------------------------------------------------------------------ offering instead of asking
def test_when_can_i_come_gets_times_not_a_question():
    e = engine()
    s = discussing_a_car(e)
    tool, said, _ = say(e, s, "When can I come and see it?")
    assert tool == "prepare_booking" and "What day" not in said and "Which day" not in said
    assert len(s.offered_slots) == 2 and s.state is State.SLOT and s.slot_day == s.offered_slots[0].astimezone(TZ).date()
    assert "tomorrow" in said.lower() and "Which" in said            # the showroom is shut tonight: tomorrow is the first day with room


def test_any_day_in_the_booking_state_offers_the_next_free_times():
    e = engine()
    s = discussing_a_car(e)
    s.state = State.SLOT
    tool, said, _ = say(e, s, "Any day would work.")
    assert tool == "prepare_booking" and len(s.offered_slots) == 2 and "I can do" in said or "I've got" in said or "soonest" in said


def test_as_soon_as_possible_offers_the_two_earliest_times():
    e = engine()
    s = discussing_a_car(e)
    s.state = State.SLOT
    say(e, s, "as soon as possible")
    assert s.offered_slots == sorted(s.offered_slots) and (s.offered_slots[1] - s.offered_slots[0]).total_seconds() <= 3600


def test_a_part_of_the_day_in_the_words_is_respected():
    e = engine()
    s = discussing_a_car(e)
    s.state = State.SLOT
    say(e, s, "any day in the afternoon")
    assert s.offered_slots and all(12 <= x.hour < 17 for x in s.offered_slots)


def test_any_day_after_two_times_were_offered_takes_the_earlier_one():
    e = engine()
    s = discussing_a_car(e)
    say(e, s, "When can I come and see it?")
    earlier = min(s.offered_slots)
    tool, said, _ = say(e, s, "Any day suits me.")
    assert s.draft_slot == earlier and s.state is State.CONTACT and "name" in said.lower()


def test_asking_what_is_free_after_an_offer_offers_again_and_does_not_choose():
    e = engine()
    s = discussing_a_car(e)
    say(e, s, "When can I come and see it?")
    say(e, s, "What day is free to come in?")
    assert s.draft_slot is None and len(s.offered_slots) == 2


def test_a_day_with_nothing_free_is_said_not_swallowed():
    """"And today." at 22:07: the old code dropped it and asked "Which day works for you?" with "Great choice." in front."""
    e = engine()
    s = discussing_a_car(e)
    s.state = State.SLOT
    tool, said, _ = say(e, s, "And today.")
    assert "nothing free today" in said and "tomorrow" in said.lower() and "Great choice" not in said
    assert len(s.offered_slots) == 2


def test_a_day_that_has_room_is_still_handled_as_before():
    e = engine(now=datetime(2026, 10, 27, 8, 0, tzinfo=TZ))        # early morning: today still has time
    s = discussing_a_car(e)
    s.state = State.SLOT
    _, said, _ = say(e, s, "today")
    assert "nothing free" not in said and "I can do" in said and "today" in said.lower()


# ------------------------------------------------------------------------------------------------ never the same question twice
def test_the_day_is_asked_once_and_then_times_are_offered():
    e = engine(decisions=["prepare_booking"])
    s = discussing_a_car(e)
    _, first, _ = say(e, s, "I want to book a test drive")
    assert s.day_asks == 1 and ("day" in first.lower())
    _, second, _ = say(e, s, "um okay then")                         # the model sends it to the booking tool; no day in it
    assert len(s.offered_slots) == 2 and "Great choice" not in second and second != first and s.day_asks == 0


def test_leaving_the_booking_talk_makes_the_next_request_a_first_ask_again():
    e = engine()
    s = discussing_a_car(e)
    say(e, s, "I want to book a test drive")
    assert s.day_asks == 1
    say(e, s, "what colour is it")
    assert s.day_asks == 0


def test_the_count_survives_a_worker_change():
    e = engine()
    s = discussing_a_car(e)
    s.day_asks = 1
    fresh = e.new_session()
    restore(snapshot(s), fresh, e.svc)
    assert fresh.day_asks == 1


def test_the_recognizer_is_primed_for_flexible_answers():
    assert "any day" in DAY_TIME and "as soon as possible" in DAY_TIME and "two in the afternoon" in DAY_TIME


@pytest.mark.skipif(SITE is None, reason="the website is not on this machine")
def test_the_whole_27_oct_booking_conversation_never_repeats_a_question():
    e = engine(portal=True)
    s = e.new_session()
    said = []
    for text in ["I'm looking for a Toyota", "What's the last one?", "When can I come and see it?", "What day is free to come in?", "And today.",
                 "Any day would work.", "Any day suits me.", "Any day?", "Tomorrow.", "Do in the afternoon."]:
        _, reply, _ = say(e, s, text)
        said.append(reply)
    booking = said[2:]
    assert all("What day would you like" not in r and "Which day works" not in r and "Which day would suit" not in r for r in booking)
    assert all(a != b for a, b in zip(booking, booking[1:]))          # no two replies in a row are the same
    assert s.state is State.CONTACT and s.draft_slot is not None and "2 PM" in booking[-1]
