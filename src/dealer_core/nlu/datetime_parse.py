"""Spoken dates and times -> concrete values, relative to `now` in the branch's timezone. Pure code, no LLM.

Handles: today / tomorrow / day after tomorrow, weekday names (optionally "next"), "25th of November", "November 25",
"the twenty fifth of november"; "4:30 pm", "4 pm", "four thirty in the afternoon", "ten in the morning", "at ten",
"half past four", "quarter to five", "noon". A bare hour with no am/pm gets a business-hours reading (1-6 -> pm,
7-11 -> am) and is flagged `assumed_period` so the caller can read it back.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
            "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6}
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august", "september",
                                      "october", "november", "december"], 1)}
_HOURS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
          "eleven": 11, "twelve": 12}
_MINUTES = {"fifteen": 15, "thirty": 30, "forty five": 45, "fortyfive": 45, "oh five": 5, "ten": 10, "twenty": 20, "forty": 40,
            "fifty": 50, "twenty five": 25, "thirty five": 35, "fifty five": 55}
_UNITS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
          "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16,
          "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}
_ORD = dict(_UNITS)
for _t, _v in (("twenty", 20), ("thirty", 30)):
    for _u, _n in list(_UNITS.items())[:9]:
        _ORD[f"{_t} {_u}"] = _v + _n
_PERIOD = r"(?:in the |at )?(?:morning|afternoon|evening|night|am|pm|a m|p m|o clock)"


@dataclass(frozen=True)
class When:
    day: date | None = None
    time: time | None = None
    assumed_period: bool = False     # a bare hour was given an am/pm reading
    day_text: str = ""
    time_text: str = ""

    @property
    def found(self) -> bool:
        return self.day is not None or self.time is not None


def _clean(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"(\d)\s*(a|p)\.?\s*m\.?", r"\1 \2m", t)               # 4 p.m. -> 4 pm
    t = re.sub(r"[^a-z0-9: ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _parse_day(t: str, today: date) -> tuple[date | None, str]:
    if re.search(r"\bday after tomorrow\b", t):
        return today + timedelta(days=2), "day after tomorrow"
    if re.search(r"\b(?:tomorrow|tmrw)\b", t):
        return today + timedelta(days=1), "tomorrow"
    if re.search(r"\b(?:today|tonight)\b", t):
        return today, "today"
    mon = "|".join(MONTHS)
    ords = "|".join(sorted(_ORD, key=len, reverse=True))
    for pat, order in ((rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?: of)? ({mon})\b", "dm"), (rf"\b({mon}) (\d{{1,2}})(?:st|nd|rd|th)?\b", "md"),
                       (rf"\b(?:the )?({ords})(?: of)? ({mon})\b", "wm")):
        m = re.search(pat, t)
        if m:
            day = _ORD[m[1]] if order == "wm" else int(m[1] if order == "dm" else m[2])
            month = MONTHS[m[2] if order in ("dm", "wm") else m[1]]
            try:
                d = date(today.year, month, day)
                if d < today:
                    d = date(today.year + 1, month, day)
                return d, m[0]
            except ValueError:
                return None, ""
    m = re.search(r"\b(?:(next|this|coming) )?(" + "|".join(sorted(WEEKDAYS, key=len, reverse=True)) + r")\b", t)
    if m:
        delta = (WEEKDAYS[m[2]] - today.weekday()) % 7
        if delta == 0 and m[1] == "next":
            delta = 7
        return today + timedelta(days=delta), m[0]
    return None, ""


def _apply_period(hour: int, period: str | None) -> tuple[int, bool]:
    """-> (24h hour, assumed). With no period the business-hours reading is used."""
    if period in ("pm", "p m", "afternoon", "evening", "night"):
        return (hour % 12) + 12 if hour < 12 else hour, False
    if period in ("am", "a m", "morning"):
        return (0 if hour == 12 else hour), False
    if hour == 12:
        return 12, True
    return (hour + 12 if 1 <= hour <= 6 else hour), True


def _parse_time(t: str) -> tuple[time | None, bool, str]:
    if re.search(r"\b(?:noon|midday)\b", t):
        return time(12, 0), False, "noon"
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))? ?(am|pm|a m|p m)\b", t)
    if m and 1 <= int(m[1]) <= 12:
        h, assumed = _apply_period(int(m[1]), m[3])
        return time(h, int(m[2] or 0)), assumed, m[0]
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m and int(m[1]) < 24 and int(m[2]) < 60:
        hh = int(m[1])
        if hh >= 13 or hh == 0:
            return time(hh, int(m[2])), False, m[0]
        h, assumed = _apply_period(hh, re.search(r"\b(morning|afternoon|evening|night)\b", t) and re.search(r"\b(morning|afternoon|evening|night)\b", t)[1])
        return time(h, int(m[2])), assumed, m[0]
    hours = "|".join(_HOURS)
    mins = "|".join(sorted(_MINUTES, key=len, reverse=True))
    m = re.search(rf"\b(half past|quarter past|quarter to) ({hours})\b", t)
    if m:
        h = _HOURS[m[2]]
        mm, h2 = {"half past": (30, h), "quarter past": (15, h), "quarter to": (45, h - 1 or 12)}[m[1]]
        period = re.search(rf"\b(morning|afternoon|evening|night|am|pm)\b", t)
        hh, assumed = _apply_period(h2, period[1] if period else None)
        return time(hh, mm), assumed, m[0]
    m = re.search(rf"\b(?:(at|around|about|by|for|from) )?({hours})(?: ({mins}))?(?: {_PERIOD})?\b", t)
    if m:
        word_period = re.search(rf"\b({hours})(?: (?:{mins}))? (?:in the |at )?(morning|afternoon|evening|night|am|pm|a m|p m)\b", t)
        o_clock = re.search(rf"\b({hours}) o clock\b", t)
        if m[1] or word_period or o_clock:                              # a bare number word alone ("the first one") is not a time
            h = _HOURS[m[2]]
            hh, assumed = _apply_period(h, word_period[2] if word_period else None)
            return time(hh, _MINUTES.get(m[3] or "", 0)), assumed, m[0]
    m = re.search(r"\b(?:at|around|about|by) (\d{1,2})\b", t)
    if m and 1 <= int(m[1]) <= 12:
        hh, assumed = _apply_period(int(m[1]), None)
        return time(hh, 0), assumed, m[0]
    p = re.search(r"\b(morning|afternoon|evening)\b", t)               # a part of the day only: no concrete time
    return None, False, p[0] if p else ""


def parse_when(text: str, now: datetime, tz: ZoneInfo) -> When:
    t = _clean(text)
    today = now.astimezone(tz).date()
    day, day_text = _parse_day(t, today)
    tm, assumed, time_text = _parse_time(t)
    return When(day=day, time=tm, assumed_period=assumed and tm is not None, day_text=day_text, time_text=time_text)


def part_of_day(text: str) -> str | None:
    m = re.search(r"\b(morning|afternoon|evening)\b", _clean(text))
    return m[1] if m else None


def combine(day: date, tm: time, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, tm, tzinfo=tz)
