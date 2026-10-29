"""Written reply -> the text the voice should be given.

Piper (through espeak-ng) reads digits and codes its own way: "AED" as a word, "4:00 PM" as "four hundred", "X5" as "X five" only sometimes, "320i" as "three hundred
twenty i", a phone number as one long number. This module rewrites those into words the voice says correctly. The transcript and the logs keep the written form.

There is no phoneme override or lexicon in this Piper build (inline [[phonemes]] are read out as letters), so respelling is the only mechanism. The respellings that need
listening to get right (dirhams, letter names) are in config/pronunciations.yaml and can be edited without touching code.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from ..nlu.numbers import int_to_words

_DIGIT = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
_ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth",
        13: "thirteenth", 14: "fourteenth", 15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth",
        30: "thirtieth"}
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
# 2-4 capital letters that are said as a word, not spelled
_WORDS = {"BMW", "RAV", "AGMC", "UAE", "AMG", "GCC", "SUV", "SUVS", "AED", "PM", "AM", "OK"}


@lru_cache(maxsize=1)
def _config() -> dict:
    path = Path(__file__).resolve().parents[3] / "config" / "pronunciations.yaml"
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def reload_config() -> None:
    _config.cache_clear()


def _spell(token: str) -> str:
    return " ".join(token)


def _time(m: re.Match) -> str:
    h, mi, ap = int(m[1]), int(m[2]), m[3].upper().replace(".", "")
    hour = int_to_words(h)
    minute = "" if mi == 0 else (f" oh {int_to_words(mi)}" if mi < 10 else f" {int_to_words(mi)}")
    return f"{hour}{minute} {ap[0]} M"


def _phone(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m[0])
    if digits.startswith("971") and len(digits) >= 11:
        digits = "0" + digits[3:]                                # +971 50 123 4567 -> 050 123 4567
    if len(digits) == 10:
        groups = [digits[:3], digits[3:6], digits[6:]]           # 050, 123, 4567
    else:
        groups = [digits[i:i + 3] for i in range(0, len(digits), 3)]
    return ", ".join(" ".join(_DIGIT[c] for c in g) for g in groups)


def _date(m: re.Match) -> str:
    day = int(m[1])
    if day in _ORD:
        ordinal = _ORD[day]
    else:
        tens, ones = divmod(day, 10)
        ordinal = f"{int_to_words(tens * 10)} {_ORD[ones]}"                       # 24 -> twenty fourth
    return f"the {ordinal} of {m[2]}"


def _model_code(m: re.Match) -> str:
    letters, digits = m[1], m[2]
    head = letters if letters in _WORDS else _spell(letters)
    return f"{head} {int_to_words(int(digits))}"


def _bmw_series(m: re.Match) -> str:
    first, rest, letter = int(m[1]), int(m[2]), m[3]
    return f"{int_to_words(first)} {int_to_words(rest)} {letter}"                      # 320i -> three twenty i; 530i -> five thirty i


def speech_text(text: str, numbers_as_words: bool | None = None) -> str:
    cfg = _config()
    words = cfg.get("numbers_as_words", False) if numbers_as_words is None else numbers_as_words
    t = text
    # currency: the voice mangles "AED"; amounts are always said in dirhams
    t = re.sub(r"\bAED\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)", r"\1 dirhams", t)
    t = re.sub(r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*AED\b", r"\1 dirhams", t)
    t = re.sub(r"\bAED\b", "dirhams", t)
    # times: 4:00 PM -> four P M, 10:30 AM -> ten thirty A M
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp])\.?[Mm]\.?\b", _time, t)
    # phone numbers written as digits -> digit by digit, grouped (pauses at the commas)
    t = re.sub(r"(?<![\d,.])(?:\+?971[\s-]?)?0?5\d[\s-]?\d{3}[\s-]?\d{4}(?![\d])", _phone, t)
    # dates: 24 October -> the twenty fourth of October
    t = re.sub(rf"\b(\d{{1,2}})\s+({_MONTHS})\b", _date, t)
    # BMW series: 320i, 530i
    t = re.sub(r"\b([1-7])(\d{2})([a-z])\b", _bmw_series, t)
    # model codes: X5, M3, CX-5, RAV4, Q7, A4
    t = re.sub(r"\b([A-Z]{1,3})-?(\d{1,2})\b(?![,.]\d)", _model_code, t)
    # letter codes: GLC 300 -> G L C 300, GXR -> G X R, RX 350 -> R X 350, TFSI -> T F S I
    t = re.sub(r"\b([A-Z]{2,5})\b", lambda m: m[1] if m[1] in _WORDS else _spell(m[1]), t)
    # units
    t = re.sub(r"\bhp\b", "horsepower", t)
    t = re.sub(r"\bkm/h\b", "kilometres per hour", t, flags=re.I)
    t = re.sub(r"\bkm\b", "kilometres", t)
    # amounts and distances written with thousands separators, always as words: the voice's own reading breaks when the piece it is given ends on a comma
    # ("at 119,000 dirhams," was said "one hundred nineteen, zero zero zero"; owner's call, 29 Oct). Years and codes have no separators and are left as they are.
    t = re.sub(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?!\d|\.\d|,\d)", lambda m: int_to_words(int(m[0].replace(",", ""))), t)
    if words:                                                    # optional: every number as words ("two hundred and nine thousand")
        t = re.sub(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,9}\b", lambda m: int_to_words(int(m[0].replace(",", ""))), t)
    # editable respellings (config/pronunciations.yaml): dirhams, letter names, ...
    for pattern, spoken in (cfg.get("respell") or {}).items():
        t = re.sub(pattern, spoken, t)
    return t
