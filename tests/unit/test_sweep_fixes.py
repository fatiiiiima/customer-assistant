"""The fixes from the phrasing sweep (scripts/phrasing_sweep.py, docs/ARCH_REVIEW.md section 27), batch by batch. Each test starts from the sweep's own set-up for the
step, with a model that never answers in time (the code decides alone), on a fresh copy of the website's stock."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT.parent / "bmw-portal" / "site" / "data.js"


def sweep():
    spec = importlib.util.spec_from_file_location("phrasing_sweep_script", ROOT / "scripts" / "phrasing_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SW = sweep()


def at(step: str):
    """(engine, session) at a step of the call, as the sweep reaches it."""
    if not SITE.exists():
        pytest.skip("the website is not on this machine")
    from dealer_core.config import load_config

    e = SW.make_engine(load_config(environ={}, overlay=ROOT / "config" / "portal.yaml"), SITE, SW.TooSlow())
    s = e.new_session()
    lines, reached = SW.SETUP[step]
    for line in lines:
        list(e.handle_stream(s, line))
    assert reached(s), f"set-up for {step} went elsewhere"
    return e, s


def say(e, s, text):
    out = list(e.handle_stream(s, text))
    return out[-1].debug.intent, " ".join(x for x in out if isinstance(x, str))


# ------------------------------------------------------------------------------------------------ A: never lose a booking by mistake
@pytest.mark.parametrize("text", ["hmm let me think", "one sec", "hold on", "give me a second"])
def test_thinking_while_choosing_a_time_keeps_the_offer(text):
    e, s = at("time_q")
    intent, reply = say(e, s, text)
    assert intent == "booking_aside" and s.state.value == "slot" and len(s.offered_slots) == 2
    assert reply.startswith("No problem, take your time.") and "9:00 AM" in reply and "2:00 PM" in reply


def test_never_mind_still_cancels_the_draft():
    e, s = at("time_q")
    assert say(e, s, "never mind")[0] == "cancel_draft"


@pytest.mark.parametrize("text, answer", [("does it have a warranty?", "warranty"), ("how many kilometres has it done?", "41,000"),
                                          ("is the price negotiable?", "fixed"), ("is it automatic?", "automatic"), ("what are your opening hours?", "9 AM")])
def test_a_question_while_choosing_a_time_is_answered_and_the_times_offered_again(text, answer):
    e, s = at("time_q")
    intent, reply = say(e, s, text)
    assert intent == "booking_aside" and answer in reply and reply.endswith("Which suits you better?") and len(s.offered_slots) == 2


def test_a_question_in_the_contact_step_is_answered_and_the_details_asked_again():
    e, s = at("contact_q")
    intent, reply = say(e, s, "is it automatic?")
    assert intent == "booking_aside" and "automatic" in reply and "name" in reply.lower() and s.state.value == "contact"


def test_a_question_at_the_read_back_is_answered_and_the_read_back_question_asked_again():
    e, s = at("readback")
    intent, reply = say(e, s, "does it have a warranty?")
    assert intent == "booking_aside" and "warranty" in reply and reply.endswith("Shall I book it?") and s.read_back and s.booking is None


def test_either_is_fine_proposes_one_of_the_times_offered():
    e, s = at("time_q")
    intent, reply = say(e, s, "either is fine")
    assert "shall we say" in reply.lower() and s.draft_slot is None


def test_an_hour_not_offered_is_checked_not_forgotten():
    e, s = at("time_q")
    intent, reply = say(e, s, "can we do 3 instead")
    assert "3 PM" in reply and "?" in reply and s.state.value == "slot"


@pytest.mark.parametrize("text", ["do you need my surname?", "why do you need my number?", "can I give it later?"])
def test_a_question_is_never_taken_for_a_name(text):
    from dealer_core.nlu import extractors as ex

    assert ex.contact(text).name is None
    assert ex.contact("Fatima, and what should I bring?").name == "Fatima"
