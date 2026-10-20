from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from dealer_core.nlu.datetime_parse import combine, parse_when, part_of_day

TZ = ZoneInfo("Asia/Dubai")
NOW = datetime(2030, 3, 4, 10, 0, tzinfo=TZ)          # Monday


def w(text):
    return parse_when(text, NOW, TZ)


@pytest.mark.parametrize("text,day", [
    ("tomorrow", date(2030, 3, 5)), ("today", date(2030, 3, 4)), ("the day after tomorrow", date(2030, 3, 6)),
    ("Saturday", date(2030, 3, 9)), ("next saturday please", date(2030, 3, 9)), ("monday", date(2030, 3, 4)),
    ("next monday", date(2030, 3, 11)), ("thursday afternoon", date(2030, 3, 7)),
    ("the 25th of March", date(2030, 3, 25)), ("March 25", date(2030, 3, 25)), ("25 march", date(2030, 3, 25)),
    ("the twenty fifth of march", date(2030, 3, 25)), ("the first of april", date(2030, 4, 1)),
    ("the 2nd of february", date(2031, 2, 2)),                                   # already past this year -> next year
])
def test_days(text, day):
    assert w(text).day == day


@pytest.mark.parametrize("text,hh,mm,assumed", [
    ("4:30 pm", 16, 30, False), ("4 pm", 16, 0, False), ("4pm", 16, 0, False), ("10 am", 10, 0, False), ("10:15 a.m.", 10, 15, False),
    ("four thirty in the afternoon", 16, 30, False), ("ten in the morning", 10, 0, False), ("two pm", 14, 0, False),
    ("at ten", 10, 0, True), ("at four", 16, 0, True), ("at 4", 16, 0, True), ("at 11", 11, 0, True), ("14:00", 14, 0, False),
    ("half past four", 16, 30, True), ("quarter to five", 16, 45, True), ("quarter past ten in the morning", 10, 15, False),
    ("noon", 12, 0, False), ("six in the evening", 18, 0, False), ("twelve pm", 12, 0, False), ("12 am", 0, 0, False),
])
def test_times(text, hh, mm, assumed):
    r = w(text)
    assert (r.time, r.assumed_period) == (time(hh, mm), assumed)


def test_full_phrases_from_the_scenarios():
    r = w("Saturday at four thirty in the afternoon")
    assert combine(r.day, r.time, TZ) == datetime(2030, 3, 9, 16, 30, tzinfo=TZ)
    r = w("tomorrow at ten in the morning")
    assert (r.day, r.time) == (date(2030, 3, 5), time(10, 0))
    r = w("the twenty fifth of March at two pm")
    assert (r.day, r.time) == (date(2030, 3, 25), time(14, 0))


@pytest.mark.parametrize("text", ["the first one", "I want the second one please", "my budget is fifty thousand", "hello there",
                                  "what colour is it", "number one", "I have two kids"])
def test_no_false_positives_on_ordinary_speech(text):
    assert not w(text).found


def test_part_of_day_only_gives_no_concrete_time():
    r = w("Saturday morning")
    assert r.day == date(2030, 3, 9) and r.time is None and part_of_day("Saturday morning") == "morning"


def test_impossible_date_is_ignored():
    assert w("the 31st of february").day is None
