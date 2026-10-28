"""The owner's test call of 28 Oct (website, Phase 3 flag on, the laptop's MLX model): four faults, each fixed and replayed here. docs/ARCH_REVIEW.md section 24.

1. "yes, please book it" after "Shall I book it?" was not a yes: the model was too slow, and the code's own decision answered with the car's facts and another invitation.
2. "The 2019 BMW X5 is white, Would you like to come and see it?": the writer put the closing question after a reply cut off at a comma.
3. "...340 horsepower, and here is the full picture. It has a 3.0-litre I6 turbo engine with 340 horsepower.": the repair said again what had just been said.
4. "I want to see the car." right after "Would you like to come and see it?" got more facts instead of a booking."""
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.agent.engine import FINISHED_CLAUSE, AgentEngine
from dealer_core.agent.governor import QualityGovernor
from dealer_core.app import ROOT, build_app
from dealer_core.config import load_config
from dealer_core.dialogue.writer import ReplyWriter, WriterInput
from dealer_core.nlu.backend import BackendTimeout

sys.path.insert(0, str(Path(__file__).parent))
from test_not_repeating import SITE, Scripted  # noqa: E402

EVENING = datetime(2026, 10, 28, 21, 0, tzinfo=ZoneInfo("Asia/Dubai"))      # after closing, as in the call: the next times offered are tomorrow's
CALL = ["I'm looking for a BMW.", "Any budget, any model?", "I want to hear more about the BMW X5.", "Tell me more about it.", "I want to see the car.",
        "I want to see the car, I want to go on a test drive.", "Any day works.", "9 in the morning.", "Fatima 050, 793, 2546", "yes, please book it"]


class TooSlow:
    """The laptop's model: every decision misses its deadline, so the code decides."""

    def lead_in(self, *a, **k):
        raise BackendTimeout("too slow")

    def lead_in_stream(self, *a, **k):
        raise BackendTimeout("too slow")
        yield ""


class Model:
    def __init__(self, reply: str):
        self.reply = reply

    def lead_in_stream(self, system, user, max_tokens=36, temperature=None, deadline_s=None, prefill=""):
        for i in range(0, len(self.reply), 7):
            yield self.reply[i:i + 7]

    def lead_in(self, *a, **k):
        return "none"


def engine(writer_backend, decisions=None, speaks=True) -> AgentEngine:
    if SITE is None:
        pytest.skip("the website is not on this machine")
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    app = build_app(cfg, inventory="portal", portal_data=SITE, classifier=Scripted(), reply_mode="conversational", clock=lambda: EVENING.astimezone(timezone.utc),
                    lead_backend=Scripted())
    e = AgentEngine(app.service, decisions or TooSlow(), ReplyWriter(writer_backend, temperature=None))
    e.governor = QualityGovernor("full")
    e.model_speaks = speaks
    return e


def say(e, s, text):
    out = list(e.handle_stream(s, text))
    return out[-1].debug.intent, [x for x in out if isinstance(x, str)]


# ------------------------------------------------------------------------------------------------ 1. the yes to the read-back
@pytest.mark.parametrize("words", ["yes please book it", "yes book it", "go ahead", "book it", "confirm it", "yeah book a test drive", "yes please", "yes",
                                   "okay go ahead and book it", "yes please confirm"])
def test_these_are_a_yes_to_the_read_back(words):
    assert AgentEngine._confirms(words)


@pytest.mark.parametrize("words", ["no", "book a different time", "what time is it", "can i book another car", "yes but change the time"])
def test_these_are_not(words):
    assert not AgentEngine._confirms(words)


def test_the_whole_call_books_on_the_first_yes_with_a_model_too_slow_to_decide():
    e = engine(TooSlow())
    s = e.new_session()
    turns = [say(e, s, t) for t in CALL]
    assert turns[-2][1][-1].endswith("Shall I book it?")                      # the read-back
    intent, reply = turns[-1]
    assert intent == "confirm_booking" and s.booking is not None              # booked on "yes, please book it", not on a second yes
    text = " ".join(reply).lower()
    assert "previous owner" not in text and "want to book" not in text and "booked" in text


def test_while_the_read_back_waits_the_code_never_answers_with_the_cars_facts():
    e = engine(TooSlow())
    s = e.new_session()
    for t in CALL[:-1]:
        say(e, s, t)
    assert s.read_back
    assert e._code_decision(s, "hmm what")["tool"] == "prepare_booking"        # unclear: read back again
    assert e._code_decision(s, "no the number is wrong")["tool"] == "fix_booking"
    assert e._code_decision(s, "yes please book it")["tool"] == "confirm_booking"


# ------------------------------------------------------------------------------------------------ 4. "I want to see the car"
def test_i_want_to_see_the_car_after_an_invitation_is_a_visit():
    e = engine(TooSlow())
    s = e.new_session()
    for t in CALL[:3]:
        say(e, s, t)
    assert "come and see" in s.last_agent().lower() or "test drive" in s.last_agent().lower()     # the X5 answer invited
    intent, reply = say(e, s, "I want to see the car.")
    assert intent == "prepare_booking" and reply[-1].endswith("?") and "day" in reply[-1].lower()


def test_but_not_without_an_invitation_and_not_for_the_photos():
    e = engine(TooSlow())
    s = e.new_session()
    for t in CALL[:3]:
        say(e, s, t)
    assert say(e, s, "Can I see the photos?")[0] != "prepare_booking"
    s2 = e.new_session()
    for t in CALL[:2]:
        say(e, s2, t)
    s2.car = s2.shown[0].stock_id                                              # a car in focus, but no invitation in the last two replies
    assert not e._invited_recently(s2)
    assert e._fast_decide(s2, "I want to see the car.") is None or e._fast_decide(s2, "I want to see the car.")["tool"] != "prepare_booking"


# ------------------------------------------------------------------------------------------------ 2 and 3: a reply cut off after a clause
def test_the_writer_adds_nothing_after_a_reply_cut_at_a_comma():
    w = ReplyWriter(Model("The 2019 BMW X5 is white, with 999 kilometres on it."), temperature=None)
    said = list(w.stream(WriterInput("tell me about the x5", "That's a 2019 X5, white. Would you like to come and see it?", "low", [], "- 2019 BMW X5: white", True)))
    assert said == ["The 2019 BMW X5 is white,"] and w.last_stream["dangling"]           # no "Would you like to come and see it?" after the comma


def test_the_engine_finishes_the_cut_sentence_and_then_asks():
    e = engine(Model("The 2016 Porsche Macan is black, with 999 horsepower."), decisions=Scripted())
    s = e.new_session()
    say(e, s, "Porsche.")
    _, reply = say(e, s, "Yes, talk about the Macan.")
    assert reply[0].endswith(",") and reply[1] == "and here is the full picture."           # the sentence is finished before anything else
    assert not reply[1].endswith("?")


def test_the_repair_does_not_say_again_what_was_just_said():
    e = engine(Model("It has a 3.0-litre V6 twin-turbo engine with 340 horsepower, and 999 owners."), decisions=Scripted())
    s = e.new_session()
    say(e, s, "Porsche.")
    say(e, s, "Yes, talk about the Macan.")
    _, reply = say(e, s, "Tell me more about the engine.")
    joined = " ".join(reply)
    assert joined.count("340 horsepower") == 1 and FINISHED_CLAUSE in reply


def test_while_a_booking_is_being_arranged_the_code_carries_on_with_it():
    """Found replaying the call: with the model too slow, "I want to see the car, I want to go on a test drive" said while the agent waited for a day went to the car's
    facts, and the booking was lost for the rest of the call."""
    e = engine(TooSlow())
    s = e.new_session()
    for t in CALL[:5]:
        say(e, s, t)
    assert s.state.value == "slot"
    intent, reply = say(e, s, CALL[5])
    assert intent == "prepare_booking" and "horsepower" not in " ".join(reply) and "litre" not in " ".join(reply)


# ------------------------------------------------------------------------------------------------ found replaying the call with the real model (11:00 clock)
@pytest.mark.parametrize("words, hour", [("9 in the morning", 9), ("9 in the morning.", 9), ("2 in the afternoon", 14), ("5 in the evening", 17),
                                         ("at 9 in the morning", 9), ("nine in the morning", 9), ("10 at night", 22)])
def test_a_time_said_the_way_the_agent_offers_times_is_read(words, hour):
    from zoneinfo import ZoneInfo

    from dealer_core.nlu.datetime_parse import parse_when

    tz = ZoneInfo("Asia/Dubai")
    w = parse_when(words, datetime(2026, 10, 27, 11, 0, tzinfo=tz), tz)
    assert w.time is not None and w.time.hour == hour


def test_an_hour_that_was_not_offered_is_never_taken_as_one_that_was():
    """With 12 and 4 PM offered, "9 in the morning" booked 12 PM: the hour was not read, and "morning" was taken as "the earlier one"."""
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Asia/Dubai")
    s = type("S", (), {})()
    s.offered_slots = [datetime(2026, 10, 27, 12, 0, tzinfo=tz), datetime(2026, 10, 27, 16, 0, tzinfo=tz)]
    assert AgentEngine._offered_pick(s, "9 in the morning") is None
    assert AgentEngine._offered_pick(s, "the morning one").hour == 12                  # no hour named: still the earlier one
    assert AgentEngine._offered_pick(s, "4 please").hour == 16


def test_the_call_at_11_does_not_book_a_time_the_caller_did_not_ask_for():
    global EVENING
    saved, EVENING = EVENING, datetime(2026, 10, 27, 11, 0, tzinfo=ZoneInfo("Asia/Dubai"))
    try:
        e = engine(TooSlow())
        s = e.new_session()
        for t in CALL[:7]:
            say(e, s, t)                                                           # ... "Any day works." -> "Then shall we say today at 12:00 PM?"
        _, reply = say(e, s, "9 in the morning.")
        text = " ".join(reply)
        assert s.draft_slot is None or s.draft_slot.hour == 9                      # never 12
        assert "12 PM" not in text.replace("12 in the afternoon", "") or "?" in text
    finally:
        EVENING = saved


def test_a_sentence_with_a_degree_sign_from_the_facts_is_not_refused():
    from dealer_core.dialogue.writer import verify_sentence

    facts = "Highlights include Panoramic Sky Lounge, heated seats, 360° camera and gesture control."
    assert verify_sentence("It has a 360° camera and heated seats.", "tell me more", "", facts, "") == []
