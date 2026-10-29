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


# ------------------------------------------------------------------------------------------------ B: read what was said
@pytest.mark.parametrize("text, name", [("it's for my brother Ahmed", "Ahmed"), ("it's for my wife, Sara Khan", "Sara Khan"), ("for my friend called Omar", "Omar"),
                                        ("it's Fatima", "Fatima"), ("it's for me", None), ("for my brother", None)])
def test_a_booking_for_someone_else_takes_their_name(text, name):
    from dealer_core.nlu import extractors as ex

    assert ex.contact(text).name == name


def test_for_my_brother_ahmed_in_the_contact_step():
    e, s = at("contact_q")
    say(e, s, "it's for my brother Ahmed")
    assert s.draft_name == "Ahmed"


@pytest.mark.parametrize("text, hour", [("actually can we make it 9 instead", 9), ("can we do 9 instead", 9), ("make it 9 in the morning", 9)])
def test_a_new_time_at_the_read_back_changes_the_time_and_reads_back_again(text, hour):
    e, s = at("readback")
    intent, reply = say(e, s, text)
    assert s.draft_slot.hour == hour and s.booking is None and reply.endswith("Shall I book it?") and "9:00 AM" in reply


@pytest.mark.parametrize("today, weekend", [((2026, 10, 29), "2026-10-31"), ((2026, 10, 31), "2026-10-31"), ((2026, 11, 1), "2026-11-01")])
def test_the_weekend_is_the_next_saturday_or_sunday(today, weekend):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from dealer_core.nlu.datetime_parse import parse_when

    tz = ZoneInfo("Asia/Dubai")
    assert str(parse_when("this weekend", datetime(*today, 10, tzinfo=tz), tz).day) == weekend


def test_this_weekend_offers_the_weekend():
    e, s = at("day_q")
    intent, reply = say(e, s, "this weekend")
    assert "Saturday" in reply or "Sunday" in reply


# ------------------------------------------------------------------------------------------------ C: a yes is a yes
@pytest.mark.parametrize("text", ["that's correct", "sounds good", "please do", "absolutely", "sure, go for it", "lock it in", "that works for me",
                                  "all correct, thanks", "yes that's all correct", "of course"])
def test_the_ordinary_ways_to_agree_book_at_the_read_back(text):
    e, s = at("readback")
    intent, _ = say(e, s, text)
    assert intent == "confirm_booking" and s.booking is not None


@pytest.mark.parametrize("text", ["that's not correct", "sounds good but at 5", "no that's wrong", "not yet"])
def test_but_not_a_disagreement_or_a_condition(text):
    e, s = at("readback")
    say(e, s, text)
    assert s.booking is None


def test_yes_and_a_question_books_and_answers():
    e, s = at("readback")
    intent, reply = say(e, s, "yes and does it have a warranty?")
    assert intent == "confirm_booking" and s.booking is not None and "warranty" in reply


@pytest.mark.parametrize("text", ["sure, why not", "sounds good", "let's do it", "absolutely"])
def test_agreeing_to_an_invitation_starts_the_booking(text):
    e, s = at("car")
    assert say(e, s, text)[0] == "prepare_booking"


# ------------------------------------------------------------------------------------------------ D: the search
@pytest.mark.parametrize("text", ["show me everything", "whatever you have", "show me all of them", "everything you have"])
def test_asking_to_see_everything_reads_out_the_search_under_way(text):
    e, s = at("search")
    intent, reply = say(e, s, text)
    assert intent == "search_cars" and "Corolla" in reply and "other makes" not in reply


def test_the_middle_one_of_three_and_not_of_four():
    from dealer_core.nlu import extractors as ex

    e, s = at("list")
    four = list(s.shown)                                                    # the Toyota list now names all four (no "plus one more")
    assert len(four) == 4 and ex.selection("the middle one", four) is None
    assert ex.selection("the middle one", four[:3]) == four[1].stock_id


# ------------------------------------------------------------------------------------------------ the whole sweep, as a guard
def test_the_whole_sweep_passes_with_the_code_alone():
    """Every phrasing in tests/data/phrasing_sweep.yaml, with a model that never answers in time. A new failure here is a regression (or a new case to fix)."""
    import yaml

    from dealer_core.config import load_config

    if not SITE.exists():
        pytest.skip("the website is not on this machine")
    cfg = load_config(environ={}, overlay=ROOT / "config" / "portal.yaml")
    failed = []
    for case in yaml.safe_load((ROOT / "tests" / "data" / "phrasing_sweep.yaml").read_text(encoding="utf-8")):
        e = SW.make_engine(cfg, SITE, SW.TooSlow())
        s = e.new_session()
        lines, reached = SW.SETUP[case["at"]]
        for line in lines:
            list(e.handle_stream(s, line))
        assert reached(s), case
        out = list(e.handle_stream(s, case["say"]))
        why = SW.check(case["expect"], out[-1].debug.intent, " ".join(x for x in out if isinstance(x, str)), s, e)
        if why:
            failed.append((case["at"], case["say"], why))
    assert not failed, failed
