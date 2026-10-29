"""29 Oct, the owner's call (Corolla, Friday 2 PM): in the name step, "Thank you." was answered "You're welcome! Anything else you'd like to know?" (the question for the name
dropped), and "No but the booking under Fatima." was not read as a name although the agent had asked "Who should I put the booking under?". docs/ARCH_REVIEW.md section 26a."""
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from dealer_core.nlu import extractors as ex

sys.path.insert(0, str(Path(__file__).parent))
import test_test_call_28oct as T  # noqa: E402

CALL = ["I'm looking for a Toyota.", "Looking for a Corolla?", "Yes, set up a test drive.", "Tomorrow.", "Do in the afternoon.", "5-050-7932546",
        "050, 0790, 2546", "050, 793, 2546", "Thank you.", "No but the booking under Fatima.", "yes"]


@pytest.fixture
def evening(monkeypatch):
    monkeypatch.setattr(T, "EVENING", datetime(2026, 10, 29, 21, 0, tzinfo=ZoneInfo("Asia/Dubai")))     # the call's evening: tomorrow is Friday 30 Oct


@pytest.mark.parametrize("text, name", [("No but the booking under Fatima.", "Fatima"), ("put it under Fatima", "Fatima"), ("under Fatima please", "Fatima"),
                                        ("under the name Omar Hassan", "Omar Hassan"), ("the booking under Layla Ahmed", "Layla Ahmed")])
def test_under_a_name_is_the_name(text, name):
    assert ex.contact(text).name == name and ex.names_someone(text)


@pytest.mark.parametrize("text", ["anything under eighty thousand", "under 100,000", "under budget", "something under a hundred thousand dirhams", "under it",
                                  "no, not now", "no thanks"])
def test_budgets_and_refusals_are_not_names(text):
    assert not ex.names_someone(text) and ex.contact(text).name is None


@pytest.mark.parametrize("decisions", [T.TooSlow(), T.Scripted()], ids=["model too slow", "model says none"])
def test_the_call_keeps_the_booking_through_a_thank_you_and_reads_the_name(evening, decisions):
    e = T.engine(T.TooSlow(), decisions=decisions)
    s = e.new_session()
    replies = {t: T.say(e, s, t) for t in CALL}
    intent, thanks = replies["Thank you."]
    text = " ".join(thanks)
    assert text.startswith("You're welcome.") and text.endswith("?") and ("name" in text.lower() or "booking under" in text.lower())     # the question for the name, again
    assert "anything else" not in text.lower()
    intent, after = replies["No but the booking under Fatima."]
    assert intent == "prepare_booking" and after[-1].endswith("Shall I book it?") and "Fatima" in " ".join(after)     # read back with the name
    assert replies["yes"][0] == "confirm_booking" and s.booking is not None and s.booking.customer_name == "Fatima"
    assert s.booking.start.strftime("%a %d %b %H:%M") == "Fri 30 Oct 14:00"


def test_a_thank_you_after_the_booking_is_still_just_a_thank_you(evening):
    e = T.engine(T.TooSlow())
    s = e.new_session()
    for t in CALL:
        T.say(e, s, t)
    intent, reply = T.say(e, s, "Thank you.")
    assert intent == "none" and " ".join(reply).startswith("You're welcome")
