"""UAE phone-number handling. Assumption (TO CONFIRM): callers are in the UAE; numbers are stored as E.164 (+971...)."""
from __future__ import annotations

import re

_DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def normalize_phone(raw: str) -> str | None:
    """'050 123 4567' / '+971501234567' / '00971501234567' / '501234567' -> '+971501234567'. None if it is not a UAE number."""
    s = re.sub(r"[^\d+]", "", raw.strip())
    if s.startswith("00"):
        s = "+" + s[2:]
    digits = s.lstrip("+")
    if s.startswith("+") and not digits.startswith("971"):
        return None                                   # a non-UAE international number: out of scope for now
    if digits.startswith("971"):
        national = digits[3:]
    elif digits.startswith("0"):
        national = digits[1:]
    else:
        national = digits
    # UAE mobile: 5x + 7 digits (9 digits). Landlines (2-9 + 7 digits) are 8 digits; accept both.
    if re.fullmatch(r"5\d{8}", national) or re.fullmatch(r"[2-4679]\d{7}", national):
        return "+971" + national
    return None


def speak_phone(e164: str) -> str:
    """Read-back form in local style, grouped for the ear: '+971501234567' -> 'zero five zero, one two three, four five six seven'."""
    local = "0" + e164[4:] if e164.startswith("+971") else e164.lstrip("+")
    groups = [local[:3], local[3:6], local[6:]] if len(local) == 10 else [local[:2], local[2:5], local[5:]]
    return ", ".join(" ".join(_DIGIT_WORDS[int(c)] for c in g) for g in groups if g)
